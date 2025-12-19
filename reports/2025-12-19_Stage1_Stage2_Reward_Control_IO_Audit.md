# Stage1/Stage2 奖励、控制、IO 网络综合审计报告

**日期**: 2025-12-19  
**范围**: Stage1 (Tracking) & Stage2 (Catching) 的奖励设计、控制接口（底盘/机械臂/灵巧手）、输入输出网络（含深度相机）与参数一致性。

---

## 总览结论
- **奖励函数**：Stage1 与 Stage2 的密集奖励形态、惩罚量级和课程步长不一致，可能导致价值尺度漂移；Stage1 触碰/碰撞分布与 Stage2 抓取评价存在偏差。
- **控制链路**：Stage2 手部改用 delta 控制后与动作去归一化仍需保持一致；Stage1 锁手逻辑可确保 AVP 虚拟手张开一致，但需确认关节限位与初始姿态匹配。
- **IO 网络**：Stage2 Critic 使用 State+Depth（7091维），Stage1 AVP 输入需保持同样噪声与归一化；速度、关节、力觉维度符合规格，但需要确认 img_size、state_dim 与 checkpoint 对齐。

---

## 详细发现与建议

### 1) 奖励函数一致性与尺度
- **Stage1 RewardManagerStage1**  
  - 触碰奖励固定 +10，碰撞惩罚 -10，Impact 惩罚基于 EE 速度；与 Stage2 的抓取力奖励区间（-10~+9）尺度不同。  
  - 建议：在 wandb 监控 reward_sum/impact/touch 与 Stage2 critic 值的相对量级，必要时按训练曲线做归一化或温度系数，避免 AVP 与本地奖励尺度差异过大。
- **Stage2 RewardManagerStage2**  
  - 奖励区间更窄（成功+20，碰撞-2，impact 上限-3），课程 stem 惩罚渐进。  
  - 建议：记录 Stage2 critic 输出均值/方差，并在 Stage1 AVP 前对 value_est 做温度或均值方差重标定，确保 λ×value 与本地奖励同量级。
- **课程步长**  
  - Stage1 使用 curriculum.stage1_steps（2e6）调整 λ 与碰撞/朝向；Stage2 使用 phase1/phase2 与 collision_stem_start/end。  
  - 建议：在 Stage1 训练日志中输出当前 difficulty 与 avp λ，确保 0~2M 步完成衰减。

### 2) 控制接口与关节限位
- **底盘**：Stage2 被锁定，Stage1 正常控制。建议在 Stage1 训练中确认 base_distance 的最优点（0.8m）是否与 RewardManager 的 base_approach 配置一致。  
- **机械臂**：两阶段均使用 delta 关节控制并裁剪到 `jnt_range[9:15]`。需确保训练前 target_arm_qpos 初始化与 DcmmCfg.arm_joints 对齐。  
- **灵巧手**：  
  - Stage1 强制张开（open_hand_qpos），符合 AVP 虚拟手假设。  
  - Stage2 使用 delta 控制，去归一化比例 `action_catch_denorm[2]=0.05` 已与控制器的增量逻辑匹配；裁剪到手部关节限位 (jnt_range offset +15)。  
  - 建议：在 wandb 记录手部动作范数与触觉力均值，检查是否存在饱和或“动而不触”的情况。

### 3) 输入/输出网络与传感
- **Stage2 Critic 输入**：State(35) + Depth(84×84) => 7091 维；DepthCNN + MLP 融合。  
- **Stage1 AVP 输入**：虚拟 state(35) + 实际深度，现已开启 `add_noise=True, add_holes=True` 与 Stage2 训练一致；使用 RunningMeanStd 归一化 state。  
  - 建议：确保 Stage2 checkpoint 的 running_mean_std 与当前 state_dim/img_size 匹配；若不匹配需重新导出 AVP 权重。  
- **深度相机**：`render_manager.get_depth_obs` 使用 meters 转换、噪声、cutout、dropout。img_size 由 config（84 或 224）决定，需与 PPO config.img_dim/Stage2 训练分辨率一致。  
  - 建议：训练前检查 config.train.ppo.img_dim 与 DcmmCfg.avp.img_size 是否一致；避免 AVP 使用 84 而主观察是 224。  
- **速度/关节/力觉维度**：Stage2 state 包含 EE pos/quaternion/vel、arm joints(6)、object pos、hand joints(12)、touch(4)，与规格一致；Stage1 vector obs (15) 与手锁定策略匹配。

### 4) 诊断与监控建议
- 在 Stage1 训练日志中增加：`avp/critic_value_mean`、`avp/lambda`、`rewards/touch_mean`、`rewards/collision_mean`，用于监测尺度与门控情况。  
- 在 Stage2 训练日志中监控：`grasp/contact_force_mean`、`grasp/fingers_touching_mean`、`rewards/impact_mean`，检查手部控制是否有效。  
- 若 AVP critic 缺失或维度不匹配，应直接 fail-fast（已实现 checkpoint 缺失报错），避免静默退化。

---

## 结论与后续动作
1. 奖励尺度需监控并可能做 value 重标定，避免 AVP 价值主导或失效。  
2. 控制链路已采用 delta+裁剪策略，需确认初始化姿态与去归一化比例一致并在日志中观测手部与臂部动作范数。  
3. IO 路径（State/Depth）需保持与 Stage2 训练分辨率和归一化一致，Stage1 AVP 已启用同噪声，但需保证 checkpoint 与 img_size/state_dim 匹配。  
4. 建议在训练脚本中增加快速配置校验（img_dim、state_dim、checkpoint 文件存在性）并在 wandb 记录关键 AVP/控制指标。
