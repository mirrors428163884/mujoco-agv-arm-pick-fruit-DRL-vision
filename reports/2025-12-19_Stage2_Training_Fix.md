# Stage 2 训练问题诊断与修复报告

**日期**: 2025-12-19  
**问题**: Stage 2 训练2M步后出现严重问题

---

## 📊 问题现象

| 指标 | 观察值 | 正常值 |
|------|--------|--------|
| losses/entropy | 40 → 130 (持续上升) | 应保持稳定或缓慢下降 |
| losses/critic_loss | 0-1.5振荡，偶有极端高值 | 应逐渐下降 |
| losses/actor_loss | 长期负数，最高0.02 | 应在0附近振荡 |
| episode_success_per_step | < 0.005 | 应逐渐上升 |
| episode_rewards_per_step | -6 到 -9 | 应逐渐上升至正值 |
| episode_lengths_per_step | 1.0 - 1.4 | 应接近episode时间限制 |

---

## 🔍 根本原因分析

### 1. Entropy 爆炸 (40 → 130)

**代码位置**: `gym_dcmm/algs/ppo_dcmm/stage2/ModelsStage2.py`

**原问题**:
```python
self.sigma_c = nn.Parameter(
    torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
# ...
return mu, mu * 0 + sigma, value  # log_sigma = 0, 所以 sigma = exp(0) = 1.0
```

**分析**:
- `sigma_c` 初始化为0
- PPO使用 `sigma = exp(log_sigma)`，所以初始 σ = exp(0) = 1.0
- 对于 tanh 归一化到 [-1, 1] 的动作，σ=1.0 太大
- 高斯分布 N(0, 1) 会产生大量超界动作，导致混乱探索
- Entropy = Σ log(σ) + const，σ越大entropy越高

**修复**:
```python
self.sigma_c = nn.Parameter(
    torch.full((actions_num,), -2.0, dtype=torch.float32), requires_grad=True)
self.log_sigma_min = -5.0  # exp(-5) ≈ 0.007
self.log_sigma_max = 0.0   # exp(0) = 1.0

# 在forward中:
log_sigma_c = torch.clamp(self.sigma_c, self.log_sigma_min, self.log_sigma_max)
```

**效果**: 初始 σ = exp(-2) ≈ 0.135，合理的探索范围

---

### 2. 手部动作控制不匹配

**代码位置**: `gym_dcmm/envs/control_manager.py`

**原问题**:
```python
# Stage 2: Allow hand control from policy
action_hand = action_dict["hand"]
# Update target hand positions (absolute positions, not delta)
self.env.Dcmm.target_hand_qpos[hand_joint_indices] = action_hand
```

**分析**:
- PPO 输出在 [-1, 1] 范围
- `action_catch_denorm = [1.5, 0.025, 0.15]` 将手部动作缩放到 [-0.15, 0.15]
- 但控制器期望**绝对关节位置**（如 [0, 0, 0, 1.8, ...]）
- 结果：手部始终被设置到接近0的位置，无法正确抓取

**修复**:
```python
# Stage 2: Use DELTA control for hand, same as arm
action_hand = action_dict["hand"]
# Apply delta to current target positions
self.env.Dcmm.target_hand_qpos[hand_joint_indices] += action_hand
# Clip to joint limits
```

**配套修改** (`PPO_Stage2.yaml`):
```yaml
action_catch_denorm: [0.0, 0.025, 0.05]  # 手部delta缩放
```

---

### 3. Episode 过早终止 (长度 ≈ 1.2)

**代码位置**: `gym_dcmm/envs/control_manager.py`

**原问题**:
```python
is_hand = self.env.contacts['object_contacts'] >= self.env.hand_start_id
is_bad = ~(is_hand | is_plant)
self.env.terminated = np.any(is_bad)  # 物体碰到地板也会终止!
```

**分析**:
- 物体与地板接触被视为"坏"接触，触发终止
- 在某些初始化场景下，物体可能在地板附近

**修复**:
```python
is_floor = np.isin(self.env.contacts['object_contacts'], [self.env.floor_id])
is_safe = is_hand | is_plant | is_floor  # 地板也是安全的
is_bad = ~is_safe
if len(self.env.contacts['object_contacts']) > 0:
    self.env.terminated = np.any(is_bad)
```

---

### 4. 奖励函数过于激进

**代码位置**: `gym_dcmm/envs/stage2/RewardManagerStage2.py`

**原问题**:
- 成功奖励: +50
- 碰撞惩罚: -10
- 茎干碰撞: -20
- Impact惩罚: 指数曲线，最高-30
- 总奖励范围: [-60, +60]

**分析**:
- 大范围奖励导致梯度不稳定
- 负向奖励主导（早期经常碰撞），critic难以学习

**修复**:
```python
# 简化奖励结构，降低量级
reward_reaching = 2.0 * np.exp(-2.0 * ee_dist)  # 0-2
reward_distance_shaping: 0.5, 1.0, 1.5, 2.0  # 累计5
reward_grasp = 1.0 + 0.5 * fingers  # 1-3
reward_success = 20.0  # 降低
reward_impact = -min(2.0 * (speed - 0.3), 3.0)  # 线性，上限-3
reward_collision = -2.0  # 降低
```

---

### 5. 课程学习起点过难

**代码位置**: `configs/env/DcmmCfg.py`, `gym_dcmm/envs/randomization_manager.py`

**原问题**:
```python
collision_stem_start = -0.5  # 初始惩罚就较大
ee_to_fruit_dist = np.random.uniform(0.05, 0.25)  # 90% 很近
```

**修复**:
```python
collision_stem_start = -0.1  # 降低初始惩罚
collision_stem_end = -2.0    # 降低最终惩罚

# 更渐进的距离分布
if rand_val < 0.70:
    ee_to_fruit_dist = np.random.uniform(0.10, 0.25)  # 70% 中等（更容易）
elif rand_val < 0.90:
    ee_to_fruit_dist = np.random.uniform(0.05, 0.10)  # 20% 近（较难）
else:
    ee_to_fruit_dist = np.random.uniform(0.25, 0.40)  # 10% 边缘
```

---

## 📝 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `ModelsStage2.py` | sigma初始化-2.0，添加bounds clamping |
| `control_manager.py` | 手部改用delta控制，添加joint limit clip |
| `RewardManagerStage2.py` | 简化奖励，降低量级 |
| `randomization_manager.py` | 调整spawn距离分布 |
| `PPO_Stage2.yaml` | 更新action_catch_denorm |
| `DcmmCfg.py` | 降低curriculum起点惩罚 |

---

## 🎯 预期效果

修复后的训练应该呈现：

1. **Entropy**: 保持在 20-40 范围内稳定
2. **Critic Loss**: 逐渐下降，减少oscillation
3. **Actor Loss**: 在 -0.1 到 0.1 之间正常振荡
4. **Episode Length**: 逐渐接近 5.0s 时间限制
5. **Rewards**: 从负值逐渐上升至正值
6. **Success Rate**: 在几百万步后开始上升

---

## 🔄 两阶段训练逻辑澄清

### 正确流程

```
┌─────────────────────────────────────────────────────────────┐
│                    训练流程                                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: 训练 Stage 2 (Catching)                            │
│    - 底盘固定                                                │
│    - 控制: 6-DOF机械臂 + 12-DOF灵巧手                        │
│    - 目标: 学会从近距离抓取果实                               │
│    - 输出: 训练好的 ActorCritic 模型                          │
│                                                              │
│              ↓ 复制 Critic 到 AVP 目录                        │
│                                                              │
│  Step 2: 训练 Stage 1 (Tracking)                             │
│    - 底盘可移动                                               │
│    - 控制: 底盘 + 6-DOF机械臂（手部锁定张开）                  │
│    - 目标: 导航至便于抓取的位置                               │
│    - AVP: 使用 Stage 2 Critic 评估"可抓取性"                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### AVP 工作原理

Stage 1 的奖励 = 原始奖励 + λ × Stage2_Critic(虚拟观测)

**虚拟观测**: 
- 假设机械臂已在预抓取姿态
- 使用真实的物体位置和深度图
- 让 Stage 2 Critic 评估"如果现在开始抓取，成功概率多大"

**效果**:
- 引导 Stage 1 不仅追求"接近目标"
- 还要找到"便于后续抓取"的位置和姿态

---

## ⚠️ 注意事项

1. **不要加载旧checkpoint**: 修复后的模型与旧模型不兼容（sigma形状变化）
2. **监控entropy**: 如果entropy仍然上升，检查sigma bounds是否生效
3. **调试建议**: 使用 `viewer=True num_envs=1` 观察手部是否正常移动
