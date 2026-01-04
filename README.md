# Picking_Sorting: 基于AVP的两阶段农业采摘强化学习项目

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.2.3+-green.svg)](https://mujoco.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0%2B-red.svg)](https://pytorch.org/)

## 📋 项目概述

**Picking_Sorting** 是一个基于深度强化学习的农业采摘机器人控制项目。该项目训练移动操作机器人（移动底盘 + 6自由度机械臂 + 16DOF灵巧手）在复杂农业场景中自主导航至目标果实并进行精确采摘。

### 🎯 核心特性

| 特性 | 描述 |
|------|------|
| **两阶段训练** | Stage 1 训练底盘+机械臂接近目标（预抓取姿态），Stage 2 训练灵巧手抓取 |
| **AVP技术** | Asymmetric Value Propagation，使用 Stage 2 Critic 为 Stage 1 提供"可抓取性"奖励信号（支持势能整形、OOD检测） |
| **Progress-Based奖励** | 2025-01-04重构：基于进度的奖励替代绝对距离奖励，消除"站桩刷分"行为 |
| **三阶段课程学习** | Stage 2 采用三阶段课程（基础抓取→滑移惩罚→扰动测试），渐进提升难度 |
| **GRU记忆模块** | Stage 1 支持可选的GRU网络，处理物体位置观测的帧丢失问题 |
| **关节空间控制** | 直接输出关节角度增量（Δθ），避免IK不稳定性 |
| **分离输出头** | Actor网络为底盘/手臂（Stage 1）和手臂/手部（Stage 2）使用分离输出头 |
| **域随机化** | 深度噪声、光照随机、地面纹理、物体形状/质量、YOLO检测帧丢失模拟 |
| **视觉感知** | 84×84深度图输入，支持模拟RealSense D435i噪声（Cutout、椒盐、边缘模糊） |

### 🤖 机器人平台

- **移动底盘**: Ranger Mini V2 双阿克曼转向底盘
- **机械臂**: xArm6 6自由度机械臂
- **灵巧手**: LEAP Hand 16关节灵巧手（12个可控自由度）
- **传感器**: 腕部深度相机、手指触觉传感器（4个：拇指、食指、中指、无名指）

---

## 🚀 快速开始

### 环境要求
- Linux系统（推荐 Ubuntu 20.04/22.04）
- Python 3.11（云端部署建议 3.11.x）
- NVIDIA 驱动支持 CUDA 12.1+（建议 535+），按 GPU/驱动选择匹配的 PyTorch CUDA 版本
- GPU: 云端 A10/A100/4090 级别或更高（推荐，至少具备 CUDA 加速能力）
- 内存: 16GB+ RAM

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/hwzhanng/Picking_Sorting.git
cd Picking_Sorting

# 安装图形加速库
sudo apt-get update
sudo apt-get install -y libegl1-mesa-dev libgl1-mesa-dev libosmesa6-dev libglew-dev

# 2. 创建conda环境
conda create -n dcmm python=3.11
conda activate dcmm

# 3. 安装PyTorch (根据GPU/驱动选择CUDA版本)
#   示例使用 cu121（驱动>=535）；如需新版可替换为 cu124，更多组合见 https://pytorch.org/get-started/locally/
pip install "torch>=2.4.0" "torchvision>=0.20.0" "torchaudio>=2.4.0" --index-url https://download.pytorch.org/whl/cu121

# 4. 安装项目依赖（版本未钉死，可随CUDA/驱动灵活选择）
pip install -r requirements.txt

# 5. 安装项目包
pip install -e .
```

### 验证安装

```bash
# 测试环境是否正常
python test_env.py

# 查看MuJoCo渲染
python train_stage1.py test=True num_envs=1 viewer=True
```

---

## 📊 输入输出维度规格

### Stage 1 (Tracking - 底盘+机械臂接近)

| 类型 | 维度 | 组成 |
|------|------|------|
| **观测 (State)** | 15-16 | 底盘速度(2) + 末端位置(3) + 末端四元数(4) + 末端速度(3) + 目标位置(3) + [可选: 位置有效标志(1)] |
| **观测 (Depth)** | 84×84 | 深度图像，单通道，归一化至0-255，支持域随机化噪声 |
| **动作** | 8 | 底盘速度(2) + 机械臂关节增量(6) |

**动作范围:**
- 底盘: [-4, 4] m/s (线速度)
- 机械臂: [-0.05, 0.05] rad/step (关节增量)

**新增特性 (2025-01-04):**
- 物体位置观测支持帧丢失模拟（模拟YOLO检测失败）
- 可选的`is_valid`标志帮助网络区分有效/无效观测
- GRU记忆模块支持处理观测序列

### Stage 2 (Catching - 灵巧手抓取)

| 类型 | 维度 | 组成 |
|------|------|------|
| **观测 (State)** | 35 | 末端位置(3) + 末端四元数(4) + 末端速度(3) + 机械臂关节(6) + 目标位置(3) + 手部关节(12) + 触觉(4) |
| **观测 (Depth)** | 84×84 | 深度图像，单通道，归一化至0-255 |
| **动作** | 18/20 | [底盘(2, 锁定为0)] + 机械臂关节增量(6) + 手部关节增量(12) |

**动作范围:**
- 底盘: 锁定为0 (Stage 2不控制底盘)
- 机械臂: [-0.025, 0.025] rad/step (关节增量)
- 手部: [-0.05, 0.05] rad/step (关节增量)

**网络架构特性:**
- 分离输出头：手臂(6D)和手部(12D)使用独立输出层
- 分离log_std：手臂σ=-2.5(更稳定)，手部σ=-1.5(更多探索)

---

## 🎯 奖励函数详解

### Stage 1 奖励函数 (`RewardManagerStage1.py`)

**[2025-01-04 重大重构]** 采用 "Progress + Terminal + Regularization" 三阶段奖励结构，消除奖励作弊和"站桩刷分"行为。

#### 新奖励结构

| 类别 | 奖励分量 | 公式 | 权重 | 说明 |
|------|----------|------|------|------|
| **Progress** | EE进度奖励 | `k_ee × (d_prev - d_curr)`, clip到±0.2 | 3.0 | 接近目标时获得正奖励 |
| **Progress** | 底盘进度奖励 | `k_base × (err_prev - err_curr)`, clip到±0.2 | 2.0 | 接近最优距离(0.8m)时获得正奖励 |
| **Regularization** | 存活惩罚 | `-0.01 per step` | -0.01 | 防止原地不动 |
| **Regularization** | 停滞惩罚 | `-0.1 if stagnating N steps` | -0.1 | EE距离N步无进展时惩罚 |
| **Conditional** | 朝向奖励 | `k_ori × max(0, cos(θ))`, 仅d_ee<0.3m时激活 | 0.4 | 基于余弦的朝向奖励 |
| **Terminal** | 预抓取成功 | `+50` | 50.0 | **不需要接触**，基于姿态判定成功 |
| **Terminal** | 碰撞惩罚 | `-50` | -50.0 | 严重碰撞 |
| **Terminal** | 超时惩罚 | `-20` | -20.0 | Episode超时 |
| **Optional** | AVP奖励 | `λ × (γ·Φ(s') - Φ(s))`, clip到±0.2 | 自适应 | 势能整形AVP |

#### 预抓取成功条件（不需要接触）
- d_ee < 0.05m
- 角度误差 < 15° (cos > 0.966)
- |v_ee| < 0.05 m/s
- 0.7m < d_base < 0.9m
- 连续保持5步（滞后稳定）

### Stage 2 奖励函数 (`RewardManagerStage2.py`)

**[2025-01-04 重大重构]** 采用三阶段课程学习，强调"多指稳定抓取 + 低冲击 + 扰动抵抗"。

#### 新奖励结构

| 类别 | 奖励分量 | 公式 | 权重 | 说明 |
|------|----------|------|------|------|
| **Progress** | EE进度奖励 | `k × (d_prev - d_curr)`, clip到±0.2 | 3.0 | 基于进度的距离奖励 |
| **Progress** | 里程碑奖励 | `+2.0 if d_ee < 0.05m` | 2.0 | 仅保留一个关键里程碑 |
| **Grasp Quality** | 多指计数 | `k_cnt × (n_contact / 4)` | 1.5 | 鼓励更多手指接触 |
| **Grasp Quality** | 力范围奖励 | 力在[f_low, f_high]范围内得分 | 1.0 | 鼓励适当抓取力 |
| **Grasp Quality** | 力平衡奖励 | `-k_bal × variance(forces)` | 0.5 | 惩罚力分布不均 |
| **Grasp Quality** | 手指协同 | 拇指+其他对握奖励 | 1.0 | 鼓励对握姿态 |
| **Penalty** | 滑移惩罚 | `-k_s × |v_rel|` | 课程自适应 | 惩罚物体-手相对速度 |
| **Penalty** | 冲击惩罚 | `-penalty if v_ee > 0.3 at first contact` | -5.0 | 惩罚高速首次接触 |
| **Terminal** | 成功奖励 | `+100` | 100.0 | 稳定抓取1秒 |
| **Terminal** | 失败惩罚 | `-50` | -50.0 | 成功/失败差距大 |
| **Perturbation** | 扰动测试 | `+10 / -5` | ±10.0 | 抵抗0.5-1.5N随机力 |

#### Stage 2 三阶段课程学习

| 阶段 | 步数范围 | 主要目标 | 配置 |
|------|----------|----------|------|
| **Phase 0** | 0-2M | 学习基础抓取 | 弱DR、近距离初始化(3-6cm)、无扰动 |
| **Phase 1** | 2M-8M | 学习稳定抓取 | 滑移惩罚渐增(0.2→1.0)、更大初始化范围 |
| **Phase 2** | 8M+ | 学习扰动抵抗 | 成功率>60%时激活扰动测试 |

---

## 🎓 训练指南

> ⚠️ **重要**: 正确的训练顺序是 **先训练 Stage 2，再训练 Stage 1**。Stage 1 会加载预训练的 Stage 2 Critic 进行 AVP 奖励计算。

### Step 1: Stage 2 训练 (灵巧手抓取)

**独立训练**灵巧手在近距离完成抓取任务，训练完成后导出 Critic 用于 AVP。

```bash
# 基础训练 (推荐配置)
python train_stage2.py num_envs=8 device_id=0

# 使用更多并行环境加速 (需要更大显存)
python train_stage2.py num_envs=16 device_id=0

# 从检查点恢复训练
python train_stage2.py checkpoint_catching="outputs/Dcmm_Catch/2025-12-19/nn/best_reward_XXX.pth"

# 指定随机种子进行可复现训练
python train_stage2.py seed=42 output_name=Dcmm_Catch_seed42
```

**训练完成后**，复制最佳模型到 AVP 目录：
```bash
cp outputs/Dcmm_Catch/.../nn/best_reward_XXX.pth assets/checkpoints/avp/stage2_critic.pth
```

### Step 2: Stage 1 训练 (底盘+机械臂接近)

使用预训练的 **Stage 2 Critic** (通过 AVP) 引导底盘和机械臂找到便于抓取的位置。

```bash
# 基础训练 (AVP 自动加载 Stage 2 Critic)
python train_stage1.py num_envs=8 device_id=0

# 关闭 AVP (消融实验基线)
python train_stage1.py avp_enabled=False output_name=Dcmm_NoAVP

# 从检查点恢复训练
python train_stage1.py checkpoint_tracking="outputs/Dcmm/2025-12-19/nn/best_reward_XXX.pth"
```

### 关键配置参数

**Stage 1 配置** (`configs/config_stage1.yaml`):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_envs` | 32 | 并行环境数（建议不超过CPU核心数） |
| `seed` | -1 | 随机种子（-1=自动随机） |
| `device_id` | 0 | GPU设备ID |
| `avp_enabled` | True | AVP开关 |
| `train.ppo.max_agent_steps` | 25M | 最大训练步数 |
| `train.ppo.learning_rate` | 3e-4 | 学习率 |
| `train.ppo.horizon_length` | 512 | 采样长度 |
| `train.ppo.minibatch_size` | 512 | Mini-batch大小 |
| `train.ppo.mini_epochs` | 6 | PPO更新轮数 |
| `train.ppo.entropy_coef` | 0.01 | 熵正则系数 |
| `train.ppo.gamma` | 0.99 | 折扣因子 |
| `train.ppo.tau` | 0.95 | GAE参数 |

**Stage 2 配置** (`configs/config_stage2.yaml`):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_envs` | 32 | 并行环境数 |
| `train.ppo.img_dim` | [84, 84] | 深度图尺寸 |
| `train.ppo.action_catch_denorm` | [0.0, 0.025, 0.05] | 动作反归一化 [底盘, 手臂, 手部] |

---

## 🧪 实验与测试

### 可视化验证 (单窗口)

> ⚠️ **重要**: 可视化时请使用 `num_envs=1` 避免多窗口导致系统卡死

```bash
# Stage 1 可视化训练 (AVP 开启)
python train_stage1.py num_envs=1 viewer=True

# Stage 1 可视化测试 (加载checkpoint)
python train_stage1.py test=True num_envs=1 viewer=True \
    checkpoint_tracking="outputs/Dcmm/2025-12-19/nn/best_reward_XXX.pth"

# Stage 2 可视化测试
python train_stage2.py test=True num_envs=1 viewer=True \
    checkpoint_catching="outputs/Dcmm_Catch/2025-12-19/nn/best_reward_XXX.pth"
```

### 消融实验

```bash
# AVP消融 - 基线 (无AVP)
python train_stage1.py avp_enabled=False output_name=Ablation_NoAVP seed=42

# AVP消融 - 完整方法 (有AVP)
python train_stage1.py avp_enabled=True output_name=Ablation_WithAVP seed=42

# 不同AVP权重实验
python train_stage1.py avp_enabled=True output_name=AVP_lambda_0.3
# 修改DcmmCfg.py中 avp.lambda_weight_start 和 avp.lambda_weight_end
```

### 多种子实验

```bash
# 批量训练不同种子
for seed in 42 123 456 789 1000; do
    python train_stage2.py seed=$seed output_name=Dcmm_Catch_seed$seed &
done

# 等待所有训练完成
wait
```

### WandB 日志配置

```bash
# 禁用 WandB (本地调试)
python train_stage2.py wandb_mode=disabled

# 离线模式 (稍后同步)
python train_stage2.py wandb_mode=offline

# 在线模式 (默认) - 设置项目名
python train_stage2.py wandb_project=Picking_Sorting_Exp wandb_entity=your_team output_name=Exp001
```

### 评估模式

```bash
# Stage 1 评估 (100个episodes)
python train_stage1.py test=True num_envs=1 \
    checkpoint_tracking="outputs/Dcmm/.../best_reward_XXX.pth"

# Stage 2 评估
python train_stage2.py test=True num_envs=1 \
    checkpoint_catching="outputs/Dcmm_Catch/.../best_reward_XXX.pth"
```

---

## 🔧 AVP 配置详解

**AVP (Asymmetric Value Propagation)** 使用预训练的 Stage 2 Critic 为 Stage 1 提供"可抓取性"奖励信号。

### [2025-01-04 重大更新] AVP改进

1. **势能整形 (Potential-Based Shaping)**: `r_avp = λ × (γ·Φ(s') - Φ(s))`
   - 奖励"向可抓取状态移动"，而非"处于可抓取状态"
   - 避免在已高值状态停留刷分

2. **OOD/置信度门控**:
   - 视觉有效性检测：深度图有效像素比例低于60%时跳过
   - MC Dropout不确定性估计：高不确定性时降低权重

3. **自适应Lambda调度**:
   - Warm-up阶段 (0-0.5M): λ=0，让进度奖励先主导
   - Ramp-up阶段 (0.5M-0.6M): λ从0增加到λ_max(0.4)
   - Full阶段 (0.6M-1.4M): λ=λ_max
   - Decay阶段 (1.4M-2M): λ逐渐减小到λ_min(0.1)

### 配置位置 (`configs/env/DcmmCfg.py`)

```python
class avp:
    enabled = False                   # 主开关 (False=关闭AVP进行消融实验)
    
    # Lambda调度参数
    warmup_steps = 500000             # 0.5M步: λ=0 (让进度奖励先主导)
    lambda_max = 0.4                  # 训练中期最大λ
    lambda_min = 0.1                  # 训练后期最小λ
    
    # 距离门控
    gate_distance = 1.5               # 距离门限 (仅在此距离内计算AVP)
    
    # OOD/置信度门控
    depth_valid_threshold = 0.6       # 最小有效深度像素比例
    mc_dropout_samples = 5            # MC采样数 (K=5)
    uncertainty_alpha = 2.0           # exp(-α·σ) 置信度缩放
    min_confidence = 0.3              # 最小置信度阈值
    
    # 网络配置
    checkpoint_path = "assets/checkpoints/avp/stage2_critic.pth"
    ready_pose = np.array([0.0, 0.0, 0.0, 1.8, 0.0, -0.785])  # 虚拟就绪姿态
    state_dim = 35                    # Stage 2 状态维度
    img_size = 84                     # 深度图尺寸
```

### AVP 工作原理

```
Stage 1 当前状态
      │
      ▼
构造虚拟观测 (使用真实手臂状态):
┌─────────────────────────────────┐
│ • 真实手臂关节/EE姿态           │
│ • 真实EE速度 (clip到±0.5)       │
│ • 真实物体位置                   │
│ • 真实深度图 (带噪声)            │
│ • 虚拟手部近抓取姿态             │
└─────────────────────────────────┘
      │
      ▼
[OOD检测] 视觉有效性 + MC Dropout置信度
      │ (低置信度时跳过)
      ▼
Stage 2 Critic(虚拟观测) → Φ(s')
      │
      ▼
势能整形: r_avp = λ(t) × clip(γ·Φ(s') - Φ(s), -0.2, 0.2)
      │
      ▼
总奖励 = 原始Stage1奖励 + AVP奖励
```

### 更新 AVP 权重

当训练出更好的 Stage 2 模型时：
```bash
cp outputs/Dcmm_Catch/.../nn/best_reward_XXX.pth assets/checkpoints/avp/stage2_critic.pth
```

---

## 🏗️ 神经网络架构

### Stage 1 网络 (`ModelsStage1.py`)

```
┌─────────────────────────────────────────────────────────────┐
│                      Stage 1 ActorCritic                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  State Input (15-16)       Depth Input (1×84×84)            │
│       │                        │                            │
│       │                   ┌────┴────┐                       │
│       │                   │ CNNBase │                       │
│       │                   │Conv(1→32,k3)+ReLU+Pool          │
│       │                   │Conv(32→64,k3)+ReLU+Pool         │
│       │                   │Conv(64→64,k3)+ReLU+Pool         │
│       │                   │AdaptiveAvgPool(4×4)             │
│       │                   │Flatten→Linear→256+ReLU          │
│       │                   └────┬────┘                       │
│       │                        │                            │
│       └────────┬───────────────┘                            │
│                │                                            │
│           Concatenate (15+256=271)                          │
│                │                                            │
│       ┌────────┼────────┐ (可选: GRU层)                     │
│       │        ▼        │                                   │
│       │   GRU(hidden=128)                                   │
│       │        │        │                                   │
│       └────────┼────────┘                                   │
│                │                                            │
│       ┌────────┴────────┐                                   │
│       │                 │                                   │
│  Actor MLP         Critic MLP                               │
│  [256, 128]        [256, 128]                               │
│       │                 │                                   │
│   ┌───┴───┐            │                                    │
│   │       │            │                                    │
│ [分离输出头]        Value(1)                                 │
│ base(2) arm(6)                                              │
│ σ_base  σ_arm                                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**[2025-01-04 更新]:**
- CNN增加AdaptiveAvgPool确保不同输入分辨率输出固定大小
- 支持可选GRU层处理观测序列（应对帧丢失）
- 分离输出头：底盘(2D)和手臂(6D)独立输出层和log_std

**初始化细节:**
- Hidden layers: `orthogonal_(gain=√2)`
- Policy output (μ): `orthogonal_(gain=0.01)`
- Value output: `orthogonal_(gain=1.0)`
- σ: `constant_(-1.0)` → exp(-1.0) ≈ 0.37

### Stage 2 网络 (`ModelsStage2.py`)

```
┌─────────────────────────────────────────────────────────────┐
│                      Stage 2 ActorCritic                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Flattened Input: [State(35) | Depth(7056)] = 7091          │
│       │                                                     │
│       ├── Split ──┬────────────────┐                        │
│       │           │                │                        │
│   State(35)   Depth(7056)          │                        │
│       │           │                │                        │
│       │      Reshape→(1,84,84)     │                        │
│       │           │                │                        │
│       │      ┌────┴────┐           │                        │
│       │      │DepthCNN │           │                        │
│       │      │Conv(1→32,k8,s4)+ReLU│                        │
│       │      │Conv(32→64,k4,s2)+ReLU                        │
│       │      │Conv(64→32,k3,s1)+ReLU                        │
│       │      │AdaptiveAvgPool(4×4) │                        │
│       │      │Linear(512→256)+ReLU │                        │
│       │      └────┬────┘           │                        │
│       │           │                │                        │
│  ┌────┴────┐  ┌───┴───────────────┴────┐                   │
│  │Actor MLP│  │      Critic Path        │                   │
│  │[256,128]│  │                         │                   │
│  │         │  │State→Value MLP(256,128) │                   │
│  │         │  │         │               │                   │
│  │         │  │   Concat(128+256=384)   │                   │
│  │         │  │         │               │                   │
│  │         │  │   Value Head→1          │                   │
│  └────┬────┘  └─────────┬───────────────┘                   │
│       │                 │                                   │
│  [分离输出头]            │                                    │
│  arm(6)  hand(12)       │                                    │
│  σ=-2.5   σ=-1.5        │                                    │
│       │                 │                                   │
│      μ(18)         Value(1)                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**[2025-01-04 更新]:**
- CNN增加AdaptiveAvgPool确保不同输入分辨率输出固定大小
- 分离输出头：手臂(6D)和手部(12D)独立输出层
- 分离log_std：手臂σ=-2.5(更稳定)，手部σ=-1.5(更多探索)
- log_sigma限制范围[-5, 0]防止熵爆炸/塌缩

**关键设计:**
- Actor: 仅使用State (35维) - 避免视觉过拟合
- Critic: State + Depth (35+7056维) - 更好的值估计

---

## 📁 项目结构

```
Picking_Sorting/
├── train_stage1.py                    # Stage 1 训练入口
├── train_stage2.py                    # Stage 2 训练入口
├── test_env.py                        # 环境测试脚本
├── setup.py                           # 包安装配置
├── requirements.txt                   # Python依赖
├── environment.yml                    # Conda环境配置
│
├── configs/
│   ├── config_stage1.yaml             # Stage 1 主配置 (Hydra)
│   ├── config_stage2.yaml             # Stage 2 主配置 (Hydra)
│   ├── env/
│   │   └── DcmmCfg.py                 # 环境参数 + AVP配置 + 课程学习 + 域随机化 + GRU配置
│   └── train/
│       ├── stage1/PPO_Stage1.yaml     # Stage 1 PPO超参数
│       └── stage2/PPO_Stage2.yaml     # Stage 2 PPO超参数
│
├── gym_dcmm/
│   ├── __init__.py                    # 注册Gymnasium环境
│   │                                   # - gym_dcmm/DcmmVecWorld-v0 (Stage 1)
│   │                                   # - gym_dcmm/DcmmVecWorldCatch-v0 (Stage 2)
│   ├── agents/
│   │   └── MujocoDcmm.py              # 机器人MuJoCo模型封装
│   │
│   ├── envs/
│   │   ├── stage1/
│   │   │   ├── DcmmVecEnvStage1.py    # Stage 1 环境 (预抓取姿态成功判定)
│   │   │   └── RewardManagerStage1.py # Stage 1 奖励管理 (Progress-based + AVP)
│   │   ├── stage2/
│   │   │   ├── DcmmVecEnvStage2.py    # Stage 2 环境 (三阶段课程)
│   │   │   ├── RewardManagerStage2.py # Stage 2 奖励管理 (抓取质量 + 滑移/冲击惩罚)
│   │   │   └── test_stage2_optimizations.py  # Stage 2 优化测试
│   │   │
│   │   ├── observation_manager.py     # 观测处理 (含帧丢失模拟)
│   │   │                               # - get_obs(): Stage 1 观测
│   │   │                               # - get_state_obs_stage2(): Stage 2 观测
│   │   │                               # - get_relative_object_pos3d_with_noise(): 带噪声物体位置
│   │   │
│   │   ├── control_manager.py         # 控制管理器
│   │   ├── randomization_manager.py   # 域随机化管理器 (含Stage 2 AVP场景)
│   │   ├── render_manager.py          # 渲染管理器 (深度噪声处理)
│   │   └── constants.py               # 环境常量
│   │
│   ├── algs/ppo_dcmm/
│   │   ├── stage1/
│   │   │   ├── PPO_Stage1.py          # Stage 1 PPO算法 (含GRU支持)
│   │   │   └── ModelsStage1.py        # Stage 1 网络 (分离头 + 可选GRU)
│   │   ├── stage2/
│   │   │   ├── PPO_Stage2.py          # Stage 2 PPO算法
│   │   │   └── ModelsStage2.py        # Stage 2 网络 (分离arm/hand头)
│   │   ├── experience.py              # 经验缓冲
│   │   └── utils.py                   # 工具函数 (RunningMeanStd等)
│   │
│   └── utils/
│       ├── quat_utils.py              # 四元数工具
│       ├── pid.py                     # PID控制器
│       ├── ik_pkg/                    # 逆运动学包
│       └── util.py                    # 通用工具
│
├── assets/
│   ├── checkpoints/avp/               # AVP预训练权重
│   │   └── stage2_critic.pth
│   ├── urdf/                          # MuJoCo机器人/场景模型
│   ├── meshes/                        # 3D网格模型
│   ├── textures/                      # 纹理贴图
│   └── objects/                       # 物体模型
│
├── outputs/                           # 训练输出目录
│   ├── Dcmm/                          # Stage 1 输出
│   └── Dcmm_Catch/                    # Stage 2 输出
│
└── wandb/                             # WandB日志目录
```

---

## 📈 课程学习配置

课程学习参数位于 `configs/env/DcmmCfg.py`:

```python
class curriculum:
    # ========================================
    # Stage 1 课程学习
    # ========================================
    stage1_steps = 2e6      # 2M步完成基础课程
    
    # [NEW 2025-01-04] 距离初始化课程
    stage1_init_dist_start = (0.8, 1.2)   # 初期：较近目标
    stage1_init_dist_mid = (0.6, 1.8)     # 中期：扩大范围
    stage1_init_dist_full = (0.4, 2.5)    # 全难度范围
    stage1_dist_expand_step1 = 1e6        # 1M步扩展到中期
    stage1_dist_expand_step2 = 3e6        # 3M步扩展到全难度
    
    # ========================================
    # Stage 2 三阶段课程学习 [NEW 2025-01-04]
    # ========================================
    stage2_steps = 10e6     # 扩展课程周期
    
    # Phase 0: 学习基础抓取 (弱DR、近距离初始化)
    stage2_phase0_steps = 2e6             # 前2M步
    stage2_phase0_init_dist = (0.03, 0.06)  # 非常近的初始化
    stage2_phase0_angle_err = 10            # 最大初始角度误差(度)
    stage2_phase0_dr_scale = 0.3            # 弱域随机化
    
    # Phase 1: 学习稳定抓取 (增加滑移惩罚)
    stage2_phase1_steps = 6e6             # 2M-8M步
    stage2_phase1_init_dist = (0.03, 0.12)
    stage2_phase1_slip_weight_start = 0.2   # 滑移惩罚渐增
    stage2_phase1_slip_weight_end = 1.0
    
    # Phase 2: 学习扰动抵抗 (成功率>60%时激活扰动)
    stage2_phase2_steps = 4e6             # 8M-12M步
    stage2_phase2_perturbation_start_success = 0.60
    
    # ========================================
    # 碰撞和朝向课程
    # ========================================
    collision_stem_start = -0.1   # 初始轻微惩罚
    collision_stem_end = -2.0     # 最终严厉惩罚
    
    orient_power_start = 1.0      # 初始线性
    orient_power_end = 1.5        # 最终1.5次方
```

---

## 🌍 域随机化配置

### 深度噪声 (`DcmmCfg.depth_noise`)

模拟RealSense D435i在户外农业场景的噪声特性：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cutout_num` | (1, 3) | 矩形遮挡数量 |
| `cutout_size` | (0.05, 0.15) | 遮挡大小比例 |
| `salt_ratio` | (0.01, 0.03) | 盐噪声比例 |
| `pepper_ratio` | (0.02, 0.05) | 胡椒噪声比例 |
| `dropout_rate` | (0.05, 0.10) | 像素丢失率 |
| `base_std` | 0.01 | 基础高斯噪声 |

### 光照随机化 (`DcmmCfg.lighting_dr`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ambient_range` | (0.1, 0.5) | 环境光强度 |
| `diffuse_range` | (0.3, 0.8) | 漫反射强度 |
| `dir_noise` | 0.3 | 光源方向噪声 |

### [NEW] 物体位置噪声 (`DcmmCfg.obj_pos_noise`)

模拟YOLO检测噪声和帧丢失：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gaussian_std` | 0.02 | 高斯噪声标准差(米) |
| `drop_probability` | 0.1 | 单帧丢失概率(10%) |
| `consecutive_drop_prob` | 0.05 | 连续丢失起始概率(5%) |
| `consecutive_drop_length` | (2, 5) | 连续丢失帧数范围 |
| `add_validity_flag` | True | 添加is_valid标志帮助网络 |

### [NEW] GRU配置 (`DcmmCfg.gru_config`)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `enabled` | True | 启用GRU记忆模块 |
| `hidden_size` | 128 | GRU隐藏层大小 |
| `num_layers` | 1 | GRU层数 |

---

## ❓ 常见问题

### Q: 训练时多个窗口弹出导致系统卡死?
**A**: 使用可视化时确保 `num_envs=1 viewer=True`，代码会自动强制设置。

### Q: AVP 如何开关?
**A**: 命令行传参 `avp_enabled=False` 或修改 `configs/env/DcmmCfg.py` 中 `avp.enabled = False`。

### Q: 训练速度太慢?
**A**: 
1. 增大 `num_envs`（建议不超过CPU核心数，最大32）
2. 确保使用GPU (`device_id=0`)
3. 关闭WandB (`wandb_mode=disabled`)
4. 关闭GRU (`gru_config.enabled=False` 在 DcmmCfg.py 中)

### Q: 显存不足 (OOM)?
**A**: 减少 `num_envs` 或 `minibatch_size`。Stage 2 因为深度图处理可能需要更多显存。

### Q: Stage 2 训练不稳定?
**A**: 
1. 检查Stage 2 Critic是否已训练收敛后再用于AVP
2. 确保三阶段课程学习正确配置
3. 观察滑移惩罚和冲击惩罚是否过大

### Q: Stage 1 预抓取成功率低?
**A**: 
1. 检查预抓取条件是否过严（默认d_ee<0.05m, 角度<15°）
2. 可以调整 `r_pregrasp_success` 权重
3. 确保进度奖励权重足够大

### Q: 如何查看训练曲线?
**A**: 
1. WandB: 访问 wandb.ai 查看在线日志
2. TensorBoard: `tensorboard --logdir outputs/Dcmm/YYYY-MM-DD/tb`

### Q: 如何添加新的物体?
**A**: 
1. 在 `assets/objects/` 添加物体mesh
2. 修改 `assets/urdf/` 中的MJCF文件
3. 在 `DcmmCfg.py` 的 `object_shape` 和 `object_size` 中添加配置

### Q: 帧丢失模拟如何关闭?
**A**: 在 `DcmmCfg.py` 中设置 `obj_pos_noise.enabled = False`

---

## 📚 参考文献

- [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [MuJoCo: Advanced Physics Simulation](https://mujoco.org/)
- [LEAP Hand: Low-cost Efficient Adaptable Programmable Hand](https://leaphand.com/)

---

## 📄 许可证

MIT License

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request
