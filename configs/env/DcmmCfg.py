import os
import numpy as np
from pathlib import Path

## Define the model path
path = os.path.realpath(__file__)
root = str(Path(path).parent)
ASSET_PATH = os.path.join(root, "../../assets")
# print("ASSET_PATH: ", ASSET_PATH)
# Use Leap Hand
XML_DCMM_LEAP_OBJECT_PATH = "urdf/x1_xarm6_leap_right_object.xml"
XML_DCMM_LEAP_UNSEEN_OBJECT_PATH = "urdf/x1_xarm6_leap_right_unseen_object.xml"
XML_ARM_PATH = "urdf/xarm6_right.xml"
## Weight Saved Path
WEIGHT_PATH = os.path.join(ASSET_PATH, "weights")

## The distance threshold to change the stage from 'tracking' to 'grasping'
## [Fix] Reduced from 0.25m to 0.10m for stricter tracking→grasping transition
## 0.25m was too early, causing Stage2 to learn both approach + grasp simultaneously
distance_thresh = 0.10

## [NEW 2025-01-04] Default control frequency setting
## Reduced from 20 to 10 for higher control frequency and lower latency
## This improves responsiveness of the base controller
default_steps_per_policy = 10

## Define the initial joint positions of the arm and the hand
# Pre-grasp pose for maximum flexibility (Stage 2 optimized)
# [Fix 2025-12-09] Updated to be within joint limits:
# Joint 4 (idx 3): [1.8, 4.141], Joint 5 (idx 4): [0.0, 2.65], Joint 6 (idx 5): [-2.35, -0.785]
arm_joints = np.array([
   0.0, 0.0, 0.0, 1.8, 0.0, -0.785
])

hand_joints = np.array([
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
])

## Define the reward weights
## [REFACTORED 2025-01-04] New "Progress + Terminal + Regularization" structure
## Based on issue requirements to eliminate reward hacking and "站桩刷分" behavior
reward_weights = {
    # ========================================
    # Stage 1: Progress-Based Rewards (NEW)
    # ========================================
    # ===== 关键修改4: 增强进度奖励 =====
    "r_ee_progress": 5.0,           # 从3.0提升到5.0
    "r_base_progress": 3.0,         # 从2.0提升到3.0
    
    # ========================================
    # Stage 1: Regularization Penalties (NEW)
    # ========================================
    # ===== 关键修改5: 增强存活惩罚 =====
    "r_alive_penalty": -0.05,       # 从-0.01提升到-0.05
    "r_stagnation_penalty": -0.2,   # 从-0.1提升到-0.2
    
    # ========================================
    # Stage 1: Conditional Rewards (NEW)
    # ========================================
    "r_orientation_gate": 0.30,     # Only apply orientation reward when d_ee < this (meters)
    "r_orientation_v2": 0.4,        # Cosine-based orientation reward (recommended: 0.2-0.6)
    
    # ========================================
    # Stage 1: Milestone Rewards (NEW 2025-01-04)
    # One-time bonuses for reaching distance thresholds, fills gap between
    # dense progress rewards and sparse success bonus
    # ========================================
    # ===== 关键修改6: 增强里程碑奖励 =====
    "r_milestone_1m": 10.0,         # 从5.0提升到10.0
    "r_milestone_05m": 20.0,        # 从10.0提升到20.0
    "r_milestone_02m": 30.0,        # 从15.0提升到30.0
    
    # ========================================
    # Stage 1: Terminal Rewards (UPDATED)
    # ========================================
    "r_pregrasp_success": 50.0,     # Pre-grasp pose achieved (recommended: 30-80)
    "r_collision": -50.0,           # Collision penalty (recommended: -30 to -80)
    "r_timeout": -20.0,             # Timeout penalty (recommended: -10 to -30)
    
    # ========================================
    # Stage 2: Progress-Based Rewards (NEW)
    # ========================================
    "r_stage2_ee_progress": 3.0,    # EE progress reward
    "r_stage2_alive": -0.01,        # Per-step time penalty
    
    # ========================================
    # Stage 2: Grasp Quality Rewards (NEW)
    # ========================================
    "r_grasp_count": 1.5,           # Multi-finger count reward weight
    "r_grasp_force_range": 1.0,     # Force in optimal range reward
    "r_grasp_balance": 0.5,         # Force balance (variance penalty) weight
    
    # ========================================
    # Stage 2: Slip/Impact Penalties (NEW)
    # ========================================
    "r_slip_penalty": 1.0,          # Slip penalty weight (recommended: 0.5-2.0)
    "r_impact_vel_threshold": 0.3,  # Velocity threshold for impact penalty
    "r_impact_first_contact": -5.0, # One-time penalty for high-speed first contact
    "r_impact_continuous": 2.0,     # Continuous impact penalty weight
    
    # ========================================
    # Stage 2: Terminal Rewards (NEW - Large Gap)
    # ========================================
    "r_stage2_success": 100.0,      # Success bonus (recommended: 50-150)
    "r_stage2_failure": -50.0,      # Failure penalty (recommended: -30 to -80)
    "r_stage2_collision": -50.0,    # Collision penalty
    
    # ========================================
    # Legacy Reward Weights (kept for backward compatibility)
    # ========================================
    "r_base_pos": 0.0,
    "r_ee_pos": 1.0,
    "r_precision": 10.0,
    "r_orient": 1.0,
    "r_touch": {
        'Tracking': 10,
        'Catching': 0.1
    },
    "r_constraint": 1.0,
    "r_stability": 20.0,
    "r_ctrl": {
        'base': 0.2,
        'arm': 1.0,
        'hand': 0.2,
    },
    "r_finger_approach": 1.0,
    "r_force_closure": 5.0,
    "r_regularization": 0.05,
    # [DEPRECATED] Old arm-specific rewards - kept for backward compatibility
    "r_arm_reaching": 1.0,          # Reduced, prefer progress-based
    "r_global_reaching": 0.3,       # Reduced
    # r_arm_motion and r_arm_action are deprecated and disabled (conflicts with smoothness).
    # They are no longer used in the reward function but retained here for config compatibility.
    "r_base_ctrl_scale": 0.005,
    "r_arm_ctrl_scale": 0.001,
    "r_action_rate": 0.02,
    # [KEPT] Success bonus still useful but secondary to pregrasp_success
    "r_success": 50.0,
    # Joint and heading rewards
    "r_joint_limit": -2.0,
    "r_base_heading": 0.5,
    # [NEW 2025-01-04] Distance gate for base heading reward (prevent "站桩刷分")
    # Only reward orientation when base is closer than this distance
    "r_base_heading_gate": 1.5,
    "r_contact_persistence": 2.0,
}

## Define the camera params for the MujocoRenderer.
cam_config = {
    "name": "top",
    "width": 640,
    "height": 480,
}

## Define the params of the Double Ackerman model.
RangerMiniV2Params = { 
  'wheel_radius': 0.1,                  # in meter //ranger-mini 0.1
  'steer_track': 0.364,                 # in meter (left & right wheel distance) //ranger-mini 0.364
  'wheel_base': 0.494,                   # in meter (front & rear wheel distance) //ranger-mini 0.494
  'max_linear_speed': 1.5,              # in m/s
  'max_angular_speed': 4.8,             # in rad/s
  'max_speed_cmd': 10.0,                # in rad/s
  'max_steer_angle_ackermann': 0.6981,  # 40 degree
  'max_steer_angle_parallel': 1.570,    # 180 degree
  'max_round_angle': 0.935671,
  'min_turn_radius': 0.47644,
}

## Define IK
ik_config = {
    "solver_type": "QP", 
    "ps": 0.001, 
    "λΣ": 12.5, 
    "ilimit": 100, 
    "ee_tol": 1e-4
}

# Define the Randomization Params
## Wheel Drive
k_drive = np.array([0.75, 1.25])
## Wheel Steer
k_steer = np.array([0.75, 1.25])
## Arm Joints
k_arm = np.array([0.75, 1.25])
## Hand Joints
k_hand = np.array([0.75, 1.25])
## Object Shape and Size
object_shape = ["box", "cylinder", "sphere", "ellipsoid", "capsule"]
object_mesh = ["bottle_mesh", "bread_mesh", "bowl_mesh", "cup_mesh", "winnercup_mesh"]
object_size = {
    "sphere": np.array([[0.035, 0.045]]),
    "capsule": np.array([[0.025, 0.035], [0.025, 0.04]]),
    "cylinder": np.array([[0.025, 0.035], [0.025, 0.035]]),
    "box": np.array([[0.025, 0.035], [0.025, 0.035], [0.025, 0.035]]),
    "ellipsoid": np.array([[0.03, 0.03], [0.045, 0.045], [0.045, 0.045]]),
}
object_mass = np.array([0.035, 0.075])
object_damping = np.array([5e-3, 2e-2])
object_static = np.array([0.5, 0.75])
## Observation Noise
k_obs_base = 0.01
k_obs_arm = 0.001
k_obs_object = 0.025
k_obs_hand = 0.01
## Actions Noise
k_act = 0.025
## Action Delay
act_delay = {
    'base': [1, 2, 3],
    'arm': [1, 2, 3],
    'hand': [1, 2, 3],
}

## Define PID params for wheel drive and steering. 
# driving
Kp_drive = 5
Ki_drive = 1e-3
Kd_drive = 1e-1
llim_drive = -200
ulim_drive = 200
# steering
# [UPDATED 2025-01-04] Reduced Kp_steer from 50.0 to 35.0 to prevent oscillation
# when using higher control frequency (steps_per_policy=10 instead of 20)
# Increased Kd_steer from 7.5 to 10.0 for better damping
Kp_steer = 35.0
Ki_steer = 2.5
Kd_steer = 10.0
llim_steer = -50
ulim_steer = 50

## Define PID params for the arm and hand. 
Kp_arm = np.array([300.0, 400.0, 400.0, 50.0, 200.0, 20.0])
Ki_arm = np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3])
Kd_arm = np.array([40.0, 40.0, 40.0, 5.0, 10.0, 1])
llim_arm = np.array([-300.0, -300.0, -300.0, -50.0, -50.0, -20.0])
ulim_arm = np.array([300.0, 300.0, 300.0, 50.0, 50.0, 20.0])

Kp_hand = np.array([4e-1, 1e-2, 2e-1, 2e-1,
                      4e-1, 1e-2, 2e-1, 2e-1,
                      4e-1, 1e-2, 2e-1, 2e-1,
                      1e-1, 1e-1, 1e-1, 1e-2,])
Ki_hand = 1e-2
Kd_hand = np.array([3e-2, 1e-3, 2e-3, 1e-3,
                      3e-2, 1e-3, 2e-3, 1e-3,
                      3e-2, 1e-3, 2e-3, 1e-3,
                      1e-2, 1e-2, 2e-2, 1e-3,])
llim_hand = -5.0
ulim_hand = 5.0
hand_mask = np.array([1, 0, 1, 1,
                      1, 0, 1, 1,
                      1, 0, 1, 1,
                      0, 1, 1, 1])

class curriculum:
    # ========================================
    # Stage 1 Curriculum (Tracking/Approach)
    # [FIX 2025-01-04] Aligned stage1_steps with curriculum expansion schedule
    # Previously: stage1_steps=2M but dist_expand_step2=3M caused curriculum to
    # never reach full difficulty. Now both are aligned at 3M.
    # ========================================
    stage1_steps = 3e6  # Curriculum runs for 3M steps to reach full difficulty
    
    # [NEW 2025-01-04] Stage 1 distance-based initialization curriculum
    # Start with closer targets, gradually increase range
    stage1_init_dist_start = (0.8, 1.2)   # Initial target distance range (meters)
    stage1_init_dist_mid = (0.6, 1.8)     # Mid-training distance range
    stage1_init_dist_full = (0.4, 2.5)    # Full difficulty distance range
    stage1_dist_expand_step1 = 1e6        # Step to expand to mid range (1M)
    stage1_dist_expand_step2 = 3e6        # Step to expand to full range (3M)
    
    # ========================================
    # Stage 2 Curriculum (Grasping)
    # [NEW 2025-01-04] Three-phase curriculum for grasp learning
    # ========================================
    stage2_steps = 10e6  # Extended curriculum period
    
    # Phase 0: Learn basic grasp (weak DR, close init)
    stage2_phase0_steps = 2e6             # First 2M steps
    stage2_phase0_init_dist = (0.03, 0.06)  # Very close initialization
    stage2_phase0_angle_err = 10            # Max initial angle error (degrees)
    stage2_phase0_dr_scale = 0.3            # Weak domain randomization
    
    # Phase 1: Learn stable grasp (add slip penalty, increase randomization)
    stage2_phase1_steps = 6e6             # Duration: 6M steps (runs from 2M to 8M)
    stage2_phase1_init_dist = (0.03, 0.12)  # Wider initialization
    stage2_phase1_angle_err = 20            # More angle variation
    stage2_phase1_slip_weight_start = 0.2   # Gradually increase slip penalty
    stage2_phase1_slip_weight_end = 1.0
    
    # Phase 2: Learn perturbation resistance (perturbation after grasp)
    stage2_phase2_steps = 4e6             # Duration: 4M steps (runs from 8M to 12M)
    stage2_phase2_perturbation_start_success = 0.60  # Only perturb if success > 60%
    
    # ========================================
    # Two-phase training configuration (Actor/Critic)
    # ========================================
    # [Modified 2025-12-09] Phase 1 extended from 5M to 15M steps
    phase1_steps = 15e6  # Phase 1: Learn grasping (Actor + Critic)
    phase2_steps = 10e6  # Phase 2: Learn value discrimination (Critic only)

    # [New] Success rate threshold for phase switching
    phase_switch_success_threshold = 0.30  # 30% success rate required

    # ========================================
    # Collision and Orientation Curriculum
    # ========================================
    # [Fix 2025-12-19] Much gentler stem collision penalty for early training
    # Start almost zero, gradually increase
    collision_stem_start = -0.1
    collision_stem_end = -2.0
    
    # [Fix 2025-12-19] Gentler orientation strictness
    orient_power_start = 1.0
    orient_power_end = 1.5  # Reduced from 2.0
    
    # Adaptive curriculum parameters
    success_rate_threshold = 0.3
    difficulty_decay_rate = 0.1

## AVP (Asymmetric Value Propagation) Configuration
## Toggle for ablation studies
## [REFACTORED 2025-01-04] Added potential-based shaping, OOD gating, adaptive lambda
class avp:
    # Master switch - set False for baseline experiments
    enabled = False

    # ========================================
    # Lambda Scheduling (Curriculum-Based)
    # [UPDATED 2025-01-04] Changed to smooth cosine scheduling
    # ========================================
    # Legacy weights (now used as fallback bounds)
    lambda_weight_start = 0.8   # Not used directly anymore
    lambda_weight_end = 0.2     # Not used directly anymore
    
    # [NEW 2025-01-04] Adaptive lambda scheduling with cosine transition
    # Note: Uses curriculum difficulty (step-based) as proxy for progress
    warmup_steps = 500000       # 0.5M steps: λ=0 (let progress reward dominate first)
    lambda_max = 0.4            # Max λ during mid-training phase
    lambda_min = 0.1            # Min λ during late-training decay phase
    
    # [NEW 2025-01-04] Smooth cosine scheduling parameters
    # If True, uses cosine instead of piecewise linear for smoother transitions
    use_cosine_schedule = True
    cosine_rampup_end = 0.3     # difficulty threshold for ramp-up completion
    cosine_decay_start = 0.7    # difficulty threshold for decay start
    
    # ========================================
    # Distance Gate
    # [UPDATED 2025-01-04] Relaxed from 1.5m to 2.0m based on issue feedback
    # If avp/distance_gate_ratio > 95%, the gate is too strict
    # ========================================
    # Only compute AVP when EE is closer than this (meters)
    gate_distance = 2.0  # Relaxed from 1.5m to 2.0m
    
    # ========================================
    # OOD/Confidence Gating
    # ========================================
    # [NEW 2025-01-04] Visual validity gate
    depth_valid_threshold = 0.6  # Minimum valid depth pixel ratio (60%)
    
    # [NEW 2025-01-04] MC Dropout uncertainty gating
    mc_dropout_samples = 5       # Number of MC samples (K=5)
    uncertainty_alpha = 2.0      # exp(-α·σ) confidence scale
    min_confidence = 0.3         # Minimum confidence threshold for gating
    
    # ========================================
    # Checkpoint and Network Config
    # ========================================
    # Checkpoint path for Stage 2 Critic (relative to project root)
    checkpoint_path = "assets/checkpoints/avp/stage2_critic.pth"
    
    # Virtual observation configuration (pre-grasp posture, matches Stage 2)
    # [Fix 2025-12-09] Updated to be within joint limits
    ready_pose = np.array([0.0, 0.0, 0.0, 1.8, 0.0, -0.785])

    # Network dimensions (must match Stage 2 training)
    state_dim = 35
    img_size = 84

## Depth Noise Configuration (for sim-to-real transfer)
## Simulates RealSense D435i artifacts in outdoor/agricultural scenes
class depth_noise:
    # Master switch
    enabled = True
    
    # Cutout noise (rectangular block dropout - simulates depth camera holes)
    cutout_enabled = True
    cutout_num = (1, 3)          # Range of number of cutout blocks
    cutout_size = (0.05, 0.15)   # Ratio of image dimensions
    
    # Salt-Pepper noise (extreme depth values - simulates specular reflections)
    salt_pepper_enabled = True
    salt_ratio = (0.01, 0.03)    # Pixels set to max depth
    pepper_ratio = (0.02, 0.05)  # Pixels set to 0
    
    # Pixel dropout (scattered missing pixels)
    dropout_rate = (0.05, 0.10)
    
    # Depth-dependent Gaussian noise (far objects noisier)
    depth_dependent = True
    base_std = 0.01              # Base noise for near objects
    scale_factor = 0.03          # Additional noise scaling with depth
    
    # Edge blur (depth discontinuity artifacts)
    edge_blur_enabled = True
    edge_noise_std = 0.1

## Lighting Domain Randomization Configuration
class lighting_dr:
    # Master switch
    enabled = True
    
    # Ambient light intensity range (simulates overcast vs clear sky)
    ambient_range = (0.1, 0.5)
    
    # Diffuse light intensity range (simulates sun intensity)
    diffuse_range = (0.3, 0.8)
    
    # Light direction noise (simulates different sun positions)
    dir_noise = 0.3

## Ground Color Domain Randomization
class ground_dr:
    # Master switch
    enabled = True
    
    # Ground types: soil_dry, soil_wet, grass, gravel
    ground_types = ['soil_dry', 'soil_wet', 'grass', 'gravel']


## Object Position Noise Configuration (Domain Randomization for obj_pos)
## Simulates YOLO detection noise and frame drops
## [UPDATED 2025-01-04] Added curriculum-based progressive randomization
class obj_pos_noise:
    # Master switch
    enabled = True
    
    # [NEW 2025-01-04] Curriculum-based progressive randomization
    # Phase 0 (0-1M steps): Disabled or very low probability
    # Phase 1 (1M-2M steps): Gradually increase to full strength
    # Phase 2 (2M+ steps): Full randomization
    curriculum_enabled = True
    curriculum_phase0_steps = 1e6   # 1M steps with minimal noise
    curriculum_phase1_steps = 2e6   # 2M steps to full strength
    
    # Gaussian noise (simulates sensor random measurement errors)
    gaussian_enabled = True
    gaussian_std = 0.02  # Standard deviation in meters (2cm noise)
    # Curriculum: std scales from 0.005 (phase 0) to full std (phase 2)
    gaussian_std_start = 0.005
    
    # Frame drop simulation (simulates YOLO detection failures)
    frame_drop_enabled = True
    drop_probability = 0.1  # 10% chance of frame drop per step
    drop_use_zero = False   # If True, set to 0; if False, use previous frame value
    # Curriculum: probability scales from 0.02 (phase 0) to full probability (phase 2)
    drop_probability_start = 0.02
    
    # Consecutive frame drop simulation (simulates prolonged detection loss)
    consecutive_drop_enabled = True
    consecutive_drop_prob = 0.05  # 5% chance to start consecutive drop sequence
    consecutive_drop_length = (2, 5)  # Range of consecutive frames to drop
    # Curriculum: disabled in phase 0, gradually enabled in phase 1
    consecutive_drop_prob_start = 0.0
    
    # Observation validity flag (helps network distinguish dropped vs valid observations)
    add_validity_flag = True  # If True, adds 1-bit is_valid flag to observation


## GRU (Recurrent Neural Network) Configuration for memory
## [UPDATED 2025-01-04] Added stage-specific controls and conditional disabling
class gru_config:
    # Master switch for RNN
    enabled = True
    
    # GRU hidden size
    hidden_size = 128
    
    # Number of GRU layers
    num_layers = 1
    
    # [NEW 2025-01-04] Stage-specific GRU control
    # GRU adds training instability when frame drop rate is low
    # Consider disabling for Stage 1 if YOLO drop simulation is minimal
    stage1_enabled = True     # Set to False to disable GRU in Stage 1
    stage2_enabled = True     # Stage 2 usually benefits from GRU
    
    # [NEW 2025-01-04] Conditional disabling based on frame drop rate
    # If frame drop probability < threshold, automatically disable GRU
    auto_disable_threshold = 0.05  # Disable GRU if drop_probability < 5%


## Network Architecture Configuration
## [NEW 2025-01-04] Configurable network capacity
class network_config:
    # Actor/Critic MLP hidden layer sizes
    # Default: [256, 128] - original configuration
    # Larger: [512, 256] - increased capacity for complex tasks
    mlp_units = [256, 128]
    
    # [NEW 2025-01-04] Option for increased capacity
    # Set to True to use [512, 256] for both Actor and Critic
    use_large_network = False
    large_mlp_units = [512, 256]
    
    # Separate value MLP (Critic uses separate network from Actor)
    separate_value_mlp = True