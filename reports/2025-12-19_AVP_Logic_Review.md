# Stage 1/Stage 2 AVP 逻辑审计报告

**日期**: 2025-12-19  
**目标**: 聚焦 Stage 2 Critic 为 Stage 1 提供“可抓取性”奖励的链路，检查奖励函数与输入/输出路径的逻辑一致性及潜在问题。

---

## 核心需求回顾
1. 先训练 Stage 2（抓取），得到可信的 Critic 作为“可抓取性”评估器。  
2. Stage 1（导航追踪）训练时使用该 Critic 的 AVP 奖励，引导机器人到“易抓取”姿态/位姿。  
3. 奖励与观测分布应与 Stage 2 训练时保持一致，避免偏置或无效奖励。

---

## 发现的问题与影响

| # | 问题 | 影响 | 证据（文件/行） | 精简修改建议 |
|---|------|------|-----------------|--------------|
| 1 | **AVP 深度观测分布与 Stage 2 训练不一致**：Stage 1 在构造 AVP 输入时关闭了深度噪声与洞噪声（`add_noise=False, add_holes=False`），而 Stage 2 训练时默认开启全部噪声增强。 | Critic 在训练期学习到的分布与推理期不同，容易高估“可抓取性”，导致 Stage 1 被错误奖励牵引到并不可抓的位置。 | `RewardManagerStage1._construct_virtual_obs()` 行 503-509 vs. `ObservationManager.get_depth_obs()` 默认 `add_noise=True, add_holes=True` | AVP 路径下复用 Stage 2 的噪声策略：调用 `get_depth_obs(..., add_noise=True, add_holes=True)`，确保价值评估分布一致。 |
| 2 | **AVP 课程权重衰减绑定 `curriculum.stage2_steps`**：Stage 1 的 `curriculum_difficulty` 使用 `DcmmCfg.curriculum.stage2_steps`(1,000万步) 计算，导致 `avp_lambda` 衰减远慢于 Stage 1 设计（规范描述 0~6M 步完成衰减）。 | Stage 1 长时间维持高 AVP 权重，遮蔽自身奖励信号，违背“先用 AVP 引导，后靠自学”的课程意图，甚至可能在中后期仍被偏置。 | `DcmmVecEnvStage1.curriculum_difficulty` 行 444-472（使用 `stage2_steps`） | 将难度/λ 衰减步数改为 `curriculum.stage1_steps` 或单独的 `curriculum.avp_steps`，与 Stage 1 课程长度对齐。 |
| 3 | **缺少 AVP 依赖完整性校验**：`_load_stage2_critic()` 找不到检查点时仅打印警告并将 `grasp_critic=None`，训练仍继续，`r_avp` 恒为 0。 | 在“AVP 打开但无 Critic”场景下静默降级，违背“先训 Stage 2，再训 Stage 1”的流程，容易得到未受“可抓取性”引导的策略且难以察觉。 | `RewardManagerStage1._load_stage2_critic()` 行 103-111；`compute_avp_reward()` 行 435-449 | 当 `avp.enabled=True` 且未成功加载 Critic 时直接 `raise` 或至少将状态写入 wandb/日志并强制停训，确保流水线必须先完成 Stage 2。 |

---

## 结论与后续
- 以上问题集中在 **分布一致性**、**课程时间尺度** 与 **依赖校验** 三个环节，均直接影响 Stage 2 Critic 向 Stage 1 提供“可抓取性”信号的可靠性。  
- 建议按优先级依次修正：先保证深度观测分布一致，再校正 λ 衰减步长，最后补齐 AVP 依赖的 fail-fast/监控，以防无效训练轮次。

