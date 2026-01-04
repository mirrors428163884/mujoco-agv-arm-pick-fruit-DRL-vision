# DCMM 项目训练目标与 WandB 监控指南

> **最后更新**: 2025-01-04 (重大重构版本)

本文档详细说明了 Stage 1 (Tracking) 和 Stage 2 (Catching) 模型的训练目标、理论满分奖励计算，以及 WandB 中关键参数的解读方法。

---

## 1. 理论满分奖励计算

### 1.1 Stage 1 (Tracking) 理论满分 [2025-01-04 更新]

**[NEW] Progress + Terminal + Regularization 结构:**

**每步奖励组成：**

| 奖励组件 | 最大值 | 条件 |
|---------|--------|------|
| `r_ee_progress` | **+0.2** | 每步接近目标，clip到±0.2 |
| `r_base_progress` | **+0.2** | 底盘接近最优距离(0.8m)，clip到±0.2 |
| `r_orientation_v2` | **+0.4** | d_ee < 0.3m 且完美对齐 |
| `r_base_heading` | **+0.5** | 底盘朝向目标 |
| `r_avp` | **+0.2** | AVP势能整形，clip到±0.2 |
| `r_touch` | **+10.0** | 轻触目标（可选，非成功必需） |
| `r_alive_penalty` | **-0.01** | 每步固定惩罚 |
| `r_action_rate` | **~-0.02** | 动作平滑时接近0 |

**终态奖励：**

| 奖励组件 | 值 | 条件 |
|---------|-----|------|
| `r_pregrasp_success` | **+50** | 达到预抓取姿态（不需要接触） |
| `r_collision` | **-50** | 致命碰撞 |
| `r_timeout` | **-20** | Episode超时 |

**预抓取成功条件（不需要接触）：**
- d_ee < 0.05m
- 角度误差 < 15° (cos > 0.966)
- |v_ee| < 0.05 m/s
- 0.7m < d_base < 0.9m
- 连续保持5步

**Episode 满分计算 (125 步)：**

```
理想情况:
- 持续 progress (ee + base): 125 × 0.4 = 50.0
- 朝向奖励 (接近时): ~60 × 0.4 = 24.0
- 底盘朝向: ~125 × 0.5 = 62.5
- AVP 奖励 (接近时): ~50 × 0.15 = 7.5
- 预抓取成功: +50.0
- 存活惩罚: -125 × 0.01 = -1.25

理论上限: ≈ 200 per episode
实际优秀目标: ≈ 50-100 per episode
```

> [!IMPORTANT]
> **Stage 1 训练目标 (2025-01-04 更新):**
> - 初期目标 (0-2M steps): `mean_reward > 10`, 学习基本接近
> - 中期目标 (2-10M steps): `mean_reward > 30`, 稳定接近
> - 优秀目标 (10-25M steps): `mean_reward > 60`, 高成功率
> - 成功率目标: `> 60%` (预抓取姿态)

---

### 1.2 Stage 2 (Catching) 理论满分 [2025-01-04 更新]

**[NEW] 三阶段课程 + 抓取质量结构:**

**每步奖励组成：**

| 奖励组件 | 最大值 | 条件 |
|---------|--------|------|
| `r_ee_progress` | **+0.2** | 每步接近目标，clip到±0.2 |
| `r_milestone` | **+2.0** | d_ee < 0.05m |
| `r_orientation` | **+0.5** | d_ee < 0.3m 且完美对齐 |
| `r_grasp_quality` | **~3.0** | 多指接触 + 力范围 + 力平衡 |
| `r_finger_synergy` | **~2.5** | 拇指+其他对握 + 力均衡 |
| `r_grasp_hold` | **+6.5** | 持续稳定抓取 (5.0 + 1.5 bonus) |
| `r_perturbation` | **+10.0** | 扰动测试通过 |
| `r_alive_penalty` | **-0.01** | 每步固定惩罚 |
| `r_slip_penalty` | **~-1.0** | 滑移惩罚 (课程自适应) |
| `r_impact_penalty` | **~-5.0** | 高速首次接触惩罚 |

**终态奖励：**

| 奖励组件 | 值 | 条件 |
|---------|-----|------|
| `r_success` | **+100** | 稳定抓取1秒 |
| `r_failure` | **-50** | 非成功终止 |
| `r_collision` | **-50** | 致命碰撞 |

**成功条件：**
- ≥2个手指稳定接触 (0.1N ≤ F ≤ 2.0N)
- 稳定保持 ≥1.0秒
- 扰动测试通过 (滑移 < 1cm)

**Episode 满分计算 (125 步)：**

```
理想情况:
- 持续 progress: 125 × 0.15 = 18.75
- 里程碑: +2.0
- 朝向: ~60 × 0.5 = 30.0
- 抓取质量: ~60 × 5.0 = 300.0
- 抓取保持: ~50 × 5.5 = 275.0
- 扰动通过: ~2 × 10.0 = 20.0
- 成功奖励: +100.0
- 惩罚 (存活+滑移等): ~-30

理论上限: ≈ 700 per episode
实际优秀目标: ≈ 150-300 per episode
```

> [!IMPORTANT]
> **Stage 2 训练目标 (2025-01-04 更新):**
> - Phase 0 (0-2M): `mean_reward > 5`, 学习基础抓取
> - Phase 1 (2M-8M): `mean_reward > 30`, 学习稳定抓取
> - Phase 2 (8M+): `mean_reward > 80`, 学习扰动抵抗
> - 成功率目标: `> 70%`

---

## 2. WandB 关键参数解读

### 2.1 核心性能指标

| 参数名 | 含义 | 健康范围 | 异常信号 |
|--------|------|----------|---------|
| `metrics/episode_rewards_per_step` | 平均 episode 奖励 | **持续上升** | 停滞/下降 |
| `metrics/episode_lengths_per_step` | 平均 episode 长度 | 依任务而定 | 过短 (<20) = 早期失败 |
| `metrics/episode_success_per_step` | 成功率 | **>0.5 目标** | <0.1 持续 = 策略未学习 |

> [!TIP]
> **解读技巧:**
> - `episode_rewards` 上升 + `episode_lengths` 稳定 = 策略质量提升
> - `episode_lengths` 急剧下降 = 可能过度惩罚导致"找死"
> - `episode_success` 接近 0 持续 5M+ steps = 需检查奖励设计

---

### 2.2 损失函数监控

| 参数名 | 含义 | 健康范围 | 异常信号 |
|--------|------|----------|---------|
| `losses/actor_loss` | 策略梯度损失 | **0.01 - 0.1** | >1.0 = 不稳定 |
| `losses/critic_loss` | 价值函数损失 | **0.1 - 1.0** | >10 = 价值估计偏差大 |
| `losses/entropy` | 策略熵 | **0.5 - 3.0** | <0.1 = 过早收敛 |
| `losses/bounds_loss` | 动作边界损失 | **<0.01** | >0.1 = 动作饱和 |

> [!WARNING]
> **熵过低 (entropy < 0.2) 的危害:**
> - 策略过早收敛到次优解
> - 探索不足，无法发现更好策略
> - **解决方案:** 增加 `entropy_coef` 或降低学习率

**熵变化曲线解读:**

```
正常曲线: 高(3.0) → 缓慢下降 → 稳定(0.5-1.0)
异常曲线: 高(3.0) → 急剧下降 → 接近0 = 需调整
```

---

### 2.3 训练稳定性指标

| 参数名 | 含义 | 健康范围 | 异常信号 |
|--------|------|----------|---------|
| `info/kl` | KL 散度 | **0.005 - 0.02** | >0.05 = 策略更新过激 |
| `info/last_lr` | 当前学习率 | 依调度器而定 | 过早降为 min = 调度器问题 |
| `info/e_clip` | PPO clip 参数 | **0.2 (固定)** | N/A |

> [!CAUTION]
> **KL 散度异常高 (>0.1) 的处理:**
> 1. 检查学习率是否过高
> 2. 增加 `mini_epochs` 数量
> 3. 减小 `minibatch_size`

---

### 2.4 性能指标

| 参数名 | 含义 | 参考值 | 优化建议 |
|--------|------|--------|---------|
| `performance/RLTrainFPS` | RL 训练帧率 | **>1000** | 低 = 网络过大/minibatch 过小 |
| `performance/EnvStepFPS` | 环境采样帧率 | **>500** | 低 = 环境计算瓶颈 |

---

### 2.5 AVP 专用指标 (仅 Stage 1) [2025-01-04 更新]

| 参数名 | 含义 | 健康范围 | 异常信号 |
|--------|------|----------|---------|
| `avp/reward_mean` | 平均 AVP 奖励 | **-0.1 - 0.2** | <-0.2 = 势能下降过快 |
| `avp/critic_value_mean` | Stage 2 Critic 平均值 | **正值** | 持续负值 = 模型问题 |
| `avp/potential_diff_mean` | [NEW] 势能差均值 | **正值** | 负值 = 远离可抓取状态 |
| `avp/confidence_mean` | [NEW] MC置信度均值 | **0.5-1.0** | <0.3 = OOD检测频繁 |
| `avp/lambda` | 当前 AVP 权重 | **0→0.4→0.1** (三阶段) | 不变化 = 课程未生效 |
| `avp/distance_gate_ratio` | 距离门控跳过比例 | **30-70%** | >95% = 门控距离过严 |
| `avp/visual_gate_ratio` | [NEW] 视觉有效性跳过比例 | **<30%** | >50% = 深度图质量差 |
| `avp/uncertainty_gate_ratio` | [NEW] 不确定性跳过比例 | **<20%** | >40% = Critic不确定 |

> [!TIP]
> **AVP 调试技巧 (2025-01-04 更新):**
> - 如果 `distance_gate_ratio` 接近 100%，说明智能体从不进入 AVP 激活范围
> - 如果 `visual_gate_ratio` 过高，检查深度图质量和噪声参数
> - 如果 `uncertainty_gate_ratio` 过高，检查Stage 2 Critic是否训练充分
> - `lambda` 应该遵循 warm-up → ramp-up → full → decay 的曲线

---

## 3. 训练健康状态速查表

### 3.1 健康训练模式

```
✅ episode_rewards: 持续缓慢上升
✅ episode_success: 逐步提升到 >50%
✅ entropy: 缓慢下降但保持 >0.5
✅ kl: 稳定在 0.01-0.02
✅ actor_loss: 稳定在 0.01-0.1
✅ critic_loss: 稳定在 0.1-1.0
```

### 3.2 需要调整的信号

| 现象 | 可能原因 | 调整建议 |
|------|----------|---------|
| Reward 持平不变 | 稀疏奖励/探索不足 | 增加 `entropy_coef` |
| Reward 剧烈震荡 | 学习率过高/奖励不平衡 | 降低 `learning_rate` |
| Entropy 快速归零 | 策略过早收敛 | 增加 `entropy_coef` 到 0.05 |
| KL 持续过高 | 策略更新过激 | 降低 `learning_rate` |
| Episode length 很短 | 过度惩罚/碰撞频繁 | 检查碰撞惩罚权重 |
| Success rate = 0 | 任务过难/奖励设计问题 | 简化初始条件/检查奖励 |

---

## 4. 奖励分解监控参数 [2025-01-04 更新]

以下参数已自动记录到 WandB：

### 4.1 Stage 1 奖励分解

| 参数名 | 含义 |
|--------|------|
| `rewards/ee_progress_mean` | [NEW] 平均 EE 进度奖励 |
| `rewards/base_progress_mean` | [NEW] 平均底盘进度奖励 |
| `rewards/alive_penalty_mean` | [NEW] 平均存活惩罚 |
| `rewards/stagnation_penalty_mean` | [NEW] 平均停滞惩罚 |
| `rewards/orientation_mean` | 平均朝向奖励 |
| `rewards/collision_mean` | 平均碰撞惩罚 |
| `rewards/success_mean` | [NEW] 平均预抓取成功奖励 |
| `rewards/touch_mean` | 平均触碰奖励 |
| `rewards/reaching_mean` | (Legacy) 平均 reaching 奖励 |
| `rewards/base_approach_mean` | (Legacy) 平均底盘接近奖励 |
| `rewards/plant_collision_mean` | 平均植物碰撞惩罚 |
| `distance/ee_distance_mean` | 平均末端到目标距离 |
| `distance/base_distance_mean` | 平均底盘到目标距离 |
| `curriculum/difficulty` | 课程学习难度系数 (0→1) |
| `curriculum/w_stem` | 当前茎干碰撞惩罚权重 |
| `curriculum/orient_power` | 当前朝向严格度 |

### 4.2 Stage 2 奖励分解

| 参数名 | 含义 |
|--------|------|
| `rewards/ee_progress_mean` | [NEW] 平均 EE 进度奖励 |
| `rewards/grasp_quality_mean` | [NEW] 平均抓取质量奖励 |
| `rewards/synergy_mean` | [NEW] 平均手指协同奖励 |
| `rewards/grasp_hold_mean` | [NEW] 平均抓取保持奖励 |
| `rewards/slip_mean` | [NEW] 平均滑移惩罚 |
| `rewards/impact_mean` | [NEW] 平均冲击惩罚 |
| `rewards/perturbation_mean` | 平均扰动测试奖励 |
| `rewards/collision_mean` | 平均碰撞惩罚 |
| `rewards/success_mean` | 平均成功奖励 |



## 5. 训练监控检查清单

训练时定期检查以下内容：

- [ ] `episode_rewards` 是否持续上升？
- [ ] `episode_success` 是否逐步提升？
- [ ] `entropy` 是否维持在合理范围 (>0.5)？
- [ ] `kl` 是否稳定在 0.01-0.02？
- [ ] `actor_loss` 和 `critic_loss` 是否稳定？
- [ ] `FPS` 是否正常 (>500)？
- [ ] (Stage 1) `avp/lambda` 是否按课程衰减？
- [ ] (Stage 2) 成功率是否达到目标 (>50%)？

---

## 6. 快速参考 [2025-01-04 更新]

### Stage 1:
- **理论每步最大奖励:** ~1.5 (progress + orientation)
- **理论每 Episode 最大奖励:** ~200
- **实际优秀目标:** 50-100 per episode
- **成功率目标 (预抓取姿态):** >60%
- **关键成功条件:** d_ee < 5cm, 角度误差 < 15°, 速度 < 5cm/s

### Stage 2:
- **理论每步最大奖励:** ~15.0 (不含成功奖励)
- **理论每 Episode 最大奖励:** ~700
- **实际优秀目标:** 150-300 per episode
- **成功率目标:** >70%
- **关键成功条件:** ≥2指稳定接触1秒, 扰动测试通过

### 三阶段课程 (Stage 2):
| 阶段 | 步数 | 目标 | 配置 |
|------|------|------|------|
| Phase 0 | 0-2M | 基础抓取 | 近距离, 弱DR, 无扰动 |
| Phase 1 | 2M-8M | 稳定抓取 | 滑移惩罚渐增 |
| Phase 2 | 8M+ | 扰动抵抗 | 成功率>60%激活 |

