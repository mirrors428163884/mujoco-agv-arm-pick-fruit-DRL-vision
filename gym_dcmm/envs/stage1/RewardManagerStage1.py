"""
Reward Manager for Stage 1 (Tracking Task).

This module handles all reward computation for the mobile manipulation tracking task.
The agent learns to move the mobile base and arm to approach a target fruit and achieve
a "pre-grasp" pose suitable for Stage 2 handoff.

Key Design Decisions (2025-01-04 Major Refactor):
Based on the issue requirements, this implements a "Progress + Terminal + Regularization"
three-stage reward structure to eliminate reward hacking and "站桩刷分" behavior.

NEW Reward Structure:
1. Progress-based Rewards (replace absolute distance rewards):
   - EE Progress: k_ee * (d_ee_prev - d_ee_curr), clipped to [-0.5, 0.5]
   - Base Progress: k_base * (d_base_prev - d_base_curr), clipped to [-0.5, 0.5]
   
2. Regularization Penalties:
   - Alive Penalty: -0.01 per step to prevent stalling
   - Stagnation Penalty: -0.1 if EE distance doesn't decrease for N steps
   - Action Smoothness: Penalize rapid action changes
   
3. Terminal Rewards:
   - Success: +50 for achieving pre-grasp pose (NOT requiring contact)
   - Collision: -50 (strong penalty)
   - Timeout: -20
   
4. Conditional Rewards:
   - Orientation: Cosine-based, only active when d_ee < 0.30m
   - Distance gating for arm vs base emphasis

Removed/Modified:
- Removed arm motion magnitude rewards (conflicts with smoothness)
- Removed absolute distance rewards (replaced with progress)
- Contact is NO LONGER required for success (prevents hard collision incentive)
"""

import numpy as np
import torch
import os
import configs.env.DcmmCfg as DcmmCfg
from gym_dcmm.utils.quat_utils import quat_rotate_vector, quat_to_euler, angle_diff

# AVP Imports (only used when AVP is enabled)
from gym_dcmm.algs.ppo_dcmm.stage2.ModelsStage2 import ActorCritic as CriticStage2
from gym_dcmm.algs.ppo_dcmm.utils import RunningMeanStd

# Constants for success criteria
ORIENTATION_THRESHOLD_COS = 0.966  # cos(15°) - orientation alignment threshold


class RewardManagerStage1:
    """
    Manages reward computation for Stage 1 (Tracking) environment.
    
    Attributes:
        env: Reference to parent DcmmVecEnvStage1 instance
        use_avp: Whether AVP (Asymmetric Value Propagation) is enabled
        reward_stats: Dictionary tracking reward components for logging
    """

    # ========================================
    # Initialization
    # ========================================
    
    def __init__(self, env):
        """
        Initialize reward manager.

        Args:
            env: Reference to the parent DcmmVecEnv instance
        """
        self.env = env
        self.prev_action_reward = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # [NEW 2025-01-04] Progress-based reward tracking
        self.prev_ee_distance = None
        self.prev_base_distance = None
        
        # [NEW 2025-01-04] Stagnation detection
        self.stagnation_counter = 0
        self.stagnation_threshold = 10  # N steps without progress triggers penalty
        self.min_progress_threshold = 0.001  # Minimum distance change to count as progress
        
        # [NEW 2025-01-04] Milestone tracking for one-time bonuses
        # Tracks which milestone thresholds have been reached in current episode
        self.milestone_1m_reached = False   # Reached < 1.0m
        self.milestone_05m_reached = False  # Reached < 0.5m
        self.milestone_02m_reached = False  # Reached < 0.2m
        
        # Initialize reward statistics
        self._init_reward_stats()
        
        # Initialize AVP if enabled
        self._init_avp()
    
    def _init_avp(self):
        """Initialize AVP (Asymmetric Value Propagation) components.
        
        [REFACTORED 2025-01-04] Major improvements based on issue requirements:
        1. Potential-based reward shaping (r_avp = λ·(γ·Φ(s_{t+1}) - Φ(s_t)))
        2. OOD/Confidence gating (visual validity + MC Dropout uncertainty)
        3. Improved virtual observation using real arm state
        4. Success-rate based adaptive lambda scheduling
        """
        self.use_avp = getattr(DcmmCfg, 'avp', None) is not None and DcmmCfg.avp.enabled
        
        if not self.use_avp:
            print(">>> AVP: Disabled by config (DcmmCfg.avp.enabled = False)")
            self.grasp_critic = None
            self.running_mean_std = None
            self.avp_stats = None
            return
        
        # Load AVP configuration
        self.avp_lambda_start = DcmmCfg.avp.lambda_weight_start
        self.avp_lambda_end = DcmmCfg.avp.lambda_weight_end
        self.avp_lambda = 0.0  # Start at 0 for warm-up period
        self.avp_gate_distance = DcmmCfg.avp.gate_distance
        self.avp_checkpoint_path = DcmmCfg.avp.checkpoint_path
        self.avp_ready_pose = DcmmCfg.avp.ready_pose
        self.avp_state_dim = DcmmCfg.avp.state_dim
        self.avp_img_size = DcmmCfg.avp.img_size
        
        # [NEW 2025-01-04] Potential-based shaping parameters
        self.avp_gamma = 0.99  # Discount factor for potential shaping
        self.avp_prev_potential = None  # Previous timestep's potential Φ(s_t)
        self.avp_reward_clip = 0.2  # Clip AVP reward to [-clip, clip]
        
        # [NEW 2025-01-04] Lambda scheduling parameters
        self.avp_warmup_steps = getattr(DcmmCfg.avp, 'warmup_steps', 500000)  # 0.5M steps
        self.avp_lambda_max = getattr(DcmmCfg.avp, 'lambda_max', 0.4)  # Max lambda during ramp-up
        self.avp_lambda_min = getattr(DcmmCfg.avp, 'lambda_min', 0.1)  # Min lambda during decay
        
        # [NEW 2025-01-04] OOD/Confidence gating parameters
        self.avp_depth_valid_threshold = getattr(DcmmCfg.avp, 'depth_valid_threshold', 0.6)  # 60%
        self.avp_mc_dropout_samples = getattr(DcmmCfg.avp, 'mc_dropout_samples', 5)  # K samples
        self.avp_uncertainty_alpha = getattr(DcmmCfg.avp, 'uncertainty_alpha', 2.0)  # exp(-α·σ)
        self.avp_min_confidence = getattr(DcmmCfg.avp, 'min_confidence', 0.3)  # Min confidence threshold

        # Sanity checks to avoid silent AVP mismatch
        if hasattr(self.env, "img_size"):
            if tuple(self.env.img_size) != (self.avp_img_size, self.avp_img_size):
                raise ValueError(
                    f"AVP depth size mismatch: env img_size={self.env.img_size}, "
                    f"avp_img_size={self.avp_img_size}. Please align config.train.ppo.img_dim "
                    f"with DcmmCfg.avp.img_size."
                )
        
        # Initialize AVP statistics
        self.avp_stats = {
            'reward_sum': 0.0,
            'critic_value_sum': 0.0,
            'potential_diff_sum': 0.0,
            'confidence_sum': 0.0,
            'count': 0,
            'gated_count': 0,
            'warmup_gated_count': 0,
            'visual_gated_count': 0,
            'uncertainty_gated_count': 0,
        }
        
        # Load Stage 2 Critic model
        self._load_stage2_critic()
    
    def _load_stage2_critic(self):
        """Load Stage 2 Critic model for AVP reward computation."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        ckpt_path = os.path.join(project_root, self.avp_checkpoint_path)
        
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"AVP enabled but Stage 2 Critic checkpoint is missing: {ckpt_path}"
            )
        
        try:
            # Network configuration (must match Stage 2 training)
            net_config = {
                'actor_units': [256, 128],
                'actions_num': 20,
                'input_shape': (self.avp_state_dim + self.avp_img_size * self.avp_img_size,),
                'state_dim': self.avp_state_dim,
                'depth_pixels': self.avp_img_size * self.avp_img_size,
                'img_size': self.avp_img_size,
                'separate_value_mlp': True,
            }
            
            # Load model
            self.grasp_critic = CriticStage2(net_config).to(self.device)
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.grasp_critic.load_state_dict(checkpoint['model'])
            
            # Load normalization
            self.running_mean_std = RunningMeanStd((self.avp_state_dim,)).to(self.device)
            if 'running_mean_std' in checkpoint:
                self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            else:
                print(">>> AVP Warning: No running_mean_std in checkpoint")

            # Validate loaded stats dimension
            if self.running_mean_std.running_mean.numel() != self.avp_state_dim:
                raise ValueError(
                    f"AVP checkpoint state_dim mismatch: expected {self.avp_state_dim}, "
                    f"got {self.running_mean_std.running_mean.numel()}"
                )
            
            # Freeze model
            self.grasp_critic.eval()
            for param in self.grasp_critic.parameters():
                param.requires_grad = False
            self.running_mean_std.eval()
            
            print(f">>> AVP: Stage 2 Critic loaded from {ckpt_path}")
            print(f">>> AVP Config: lambda={self.avp_lambda}, gate_dist={self.avp_gate_distance}m")
            
        except Exception as e:
            print(f">>> AVP Error: {e}")
            self.grasp_critic = None
            self.running_mean_std = None

    # ========================================
    # Main Reward Computation
    # ========================================
    
    def compute_reward(self, obs, info, ctrl):
        """
        Compute total reward for the current step.
        
        [REFACTORED 2025-01-04] New "Progress + Terminal + Regularization" structure
        to eliminate reward hacking and "站桩刷分" behavior.

        Args:
            obs: Current observation dict
            info: Environment info dict containing distances
            ctrl: Control action dict with 'base', 'arm', 'hand' keys

        Returns:
            float: Total reward value
        """
        # ========================================
        # 1. PROGRESS-BASED REWARDS (replace absolute distance)
        # ========================================
        r_ee_progress = self._compute_ee_progress_reward(info)
        r_base_progress = self._compute_base_progress_reward(info)
        
        # ========================================
        # 2. REGULARIZATION / PENALTIES
        # ========================================
        r_alive = self._compute_alive_penalty()  # Per-step time penalty
        r_stagnation = self._compute_stagnation_penalty(info)  # Penalize no progress
        r_action_rate = self._compute_action_rate_penalty(ctrl)
        r_regularization = self._compute_regularization_penalty(ctrl)
        r_joint_limit = self._compute_joint_limit_penalty()
        r_plant_collision = self._compute_plant_collision_penalty()
        
        # ========================================
        # 3. CONDITIONAL REWARDS (gated by distance)
        # ========================================
        r_orientation = self._compute_orientation_reward_v2(info)  # Only when d_ee < 0.30m
        r_base_heading = self._compute_base_heading_reward(info)
        
        # ========================================
        # 4. MILESTONE REWARDS (one-time bonuses)
        # [NEW 2025-01-04] Fill gap between dense progress and sparse success
        # ========================================
        r_milestone = self._compute_milestone_reward(info)
        
        # ========================================
        # 5. TERMINAL REWARDS
        # ========================================
        r_collision = self._compute_collision_penalty()  # -50 for collision
        r_timeout = self._compute_timeout_penalty()  # -20 for timeout
        r_success = self._compute_pregrasp_success_bonus(info)  # +50 for pre-grasp achieved
        
        # ========================================
        # 6. OPTIONAL: AVP (if enabled)
        # ========================================
        r_avp = self.compute_avp_reward(obs, info)
        
        # ========================================
        # Legacy rewards (kept for transition, reduced weight)
        # ========================================
        # Touch reward: still useful but not required for success
        r_touch, r_impact = self._compute_touch_reward()
        
        # Sum all rewards
        total_reward = (
            # Progress rewards (main learning signal)
            r_ee_progress + r_base_progress +
            # Regularization (behavior shaping)
            r_alive + r_stagnation + r_action_rate + r_regularization + r_joint_limit +
            # Conditional rewards
            r_orientation + r_base_heading +
            # Milestone bonuses
            r_milestone +
            # Collision penalties
            r_collision + r_plant_collision + r_timeout +
            # Success/failure
            r_success +
            # Optional
            r_avp + r_touch
        )

        # Update statistics for logging
        self._update_reward_stats_v2(
            info, r_ee_progress, r_base_progress, r_alive, r_stagnation,
            r_orientation, r_collision, r_success, r_touch, r_milestone
        )

        # Debug output
        if self.env.print_reward:
            self._print_reward_breakdown_v2(
                r_ee_progress, r_base_progress, r_alive, r_stagnation,
                r_orientation, r_base_heading, r_touch, r_collision,
                r_plant_collision, r_action_rate, r_avp, r_success,
                r_timeout, r_milestone, info, total_reward
            )

        return total_reward
    
    def _compute_success_bonus(self):
        """
        Compute success bonus when task is successfully completed.
        
        [DEPRECATED 2025-01-04] This old version required contact.
        Use _compute_pregrasp_success_bonus instead.
        
        Returns:
            float: Success bonus (r_success) if contact_count >= 10, else 0
        """
        # Check if agent has maintained contact for 10+ steps (task success)
        # contact_count is initialized in env.reset() and updated in env.step()
        if self.env.contact_count >= 10:
            return DcmmCfg.reward_weights.get("r_success", 50.0)
        return 0.0

    # ========================================
    # NEW Progress-Based Reward Components (2025-01-04)
    # ========================================
    
    def _compute_ee_progress_reward(self, info):
        """
        Compute EE progress reward: reward for getting closer to target.
        
        Formula: r_ee = k_ee * (d_ee_prev - d_ee_curr), clipped to [-0.2, 0.2]
        
        Where:
        - d_ee_prev: EE-to-target distance at previous timestep
        - d_ee_curr: EE-to-target distance at current timestep
        - Positive reward when getting closer (d_ee_prev > d_ee_curr)
        
        This replaces absolute distance rewards with progress-based rewards
        to prevent "站桩刷分" (standing still to farm rewards).
        
        Args:
            info: Environment info containing ee_distance
            
        Returns:
            float: Progress reward value
        """
        current_dist = info["ee_distance"]
        
        # Initialize previous distance on first call
        if self.prev_ee_distance is None:
            self.prev_ee_distance = current_dist
            return 0.0
        
        # Compute progress (positive if getting closer)
        progress = self.prev_ee_distance - current_dist
        
        # Apply weight and clip
        # ===== 关键修改7: 放宽clip范围 =====
        k_ee = DcmmCfg.reward_weights.get("r_ee_progress", 5.0)
        reward = k_ee * progress
        reward = np.clip(reward, -0.5, 0.5)  # 从±0.2改到±0.5
        
        # Update previous distance
        self.prev_ee_distance = current_dist
        
        return reward
    
    def _compute_base_progress_reward(self, info):
        """
        Compute base progress reward: reward for base approaching optimal distance.
        
        The base should approach an optimal distance (0.7-0.9m) from target,
        not necessarily get as close as possible.
        
        Args:
            info: Environment info containing base_distance
            
        Returns:
            float: Progress reward value
        """
        current_dist = info["base_distance"]
        optimal_dist = 0.8  # Optimal base distance
        
        # Initialize previous distance on first call
        if self.prev_base_distance is None:
            self.prev_base_distance = current_dist
            return 0.0
        
        # Compute progress toward optimal distance
        prev_error = abs(self.prev_base_distance - optimal_dist)
        curr_error = abs(current_dist - optimal_dist)
        progress = prev_error - curr_error  # Positive if error decreased
        
        # Apply weight and clip
        # ===== 关键修改7: 放宽clip范围 =====
        k_base = DcmmCfg.reward_weights.get("r_base_progress", 3.0)
        reward = k_base * progress
        reward = np.clip(reward, -0.5, 0.5)  # 从±0.2改到±0.5
        
        # Update previous distance
        self.prev_base_distance = current_dist
        
        return reward
    
    def _compute_alive_penalty(self):
        """
        Compute per-step time penalty to discourage stalling.
        
        This prevents the agent from staying still to avoid negative rewards.
        
        Returns:
            float: Negative penalty value (-0.01 to -0.02 per step)
        """
        return DcmmCfg.reward_weights.get("r_alive_penalty", -0.01)
    
    def _compute_stagnation_penalty(self, info):
        """
        Compute stagnation penalty: penalize if EE hasn't made progress for N steps.
        
        If the agent is stuck (EE distance not decreasing), apply additional penalty.
        
        Args:
            info: Environment info containing ee_distance
            
        Returns:
            float: Stagnation penalty (-0.1 if stagnating, 0 otherwise)
        """
        current_dist = info["ee_distance"]
        
        if self.prev_ee_distance is not None:
            # Check if making progress
            progress = self.prev_ee_distance - current_dist
            if progress < self.min_progress_threshold:
                self.stagnation_counter += 1
            else:
                self.stagnation_counter = 0
        
        # Apply penalty if stagnating for too long
        if self.stagnation_counter >= self.stagnation_threshold:
            return DcmmCfg.reward_weights.get("r_stagnation_penalty", -0.1)
        
        return 0.0
    
    def _compute_orientation_reward_v2(self, info):
        """
        Compute orientation reward using cosine similarity, ONLY when close.
        
        [NEW 2025-01-04] Key changes:
        1. Only active when d_ee < 0.30m (prevents spinning at long range)
        2. Uses continuous cosine function instead of power scaling
        3. r_ori = k_ori * max(0, cos(theta))
        
        Args:
            info: Environment info containing ee_distance
            
        Returns:
            float: Orientation reward value
        """
        # Gate: only compute when EE is close
        gate_distance = DcmmCfg.reward_weights.get("r_orientation_gate", 0.30)
        if info["ee_distance"] >= gate_distance:
            return 0.0
        
        ee_pos = self.env.Dcmm.data.body("link6").xpos
        obj_pos = self.env.Dcmm.data.body(self.env.object_name).xpos
        
        # Direction from EE to object
        ee_to_obj = obj_pos - ee_pos
        ee_to_obj_norm = ee_to_obj / (np.linalg.norm(ee_to_obj) + 1e-6)
        
        # Palm forward direction (negative Z-axis of EE frame)
        ee_quat = self.env.Dcmm.data.body("link6").xquat
        palm_forward = quat_rotate_vector(ee_quat, np.array([0, 0, -1]))
        
        # Cosine alignment: 1.0 = perfect, -1.0 = backwards
        cos_theta = np.dot(palm_forward, ee_to_obj_norm)
        
        # Apply continuous reward: k_ori * max(0, cos_theta)
        k_ori = DcmmCfg.reward_weights.get("r_orientation_v2", 0.4)
        return k_ori * max(0, cos_theta)
    
    def check_pregrasp_pose(self, info):
        """
        Check if robot has achieved pre-grasp pose suitable for Stage 2 handoff.
        
        [NEW 2025-01-04] Pre-grasp pose criteria (does NOT require contact):
        - d_ee < 0.05m
        - angle_err < 15° (cos > 0.966)
        - |v_ee| < 0.05 m/s
        - 0.7m < d_base < 0.9m
        
        Returns:
            bool: True if pre-grasp pose achieved
        """
        # Check EE distance
        if info["ee_distance"] >= 0.05:
            return False
        
        # Check base distance (should be in optimal window)
        if not (0.7 < info["base_distance"] < 0.9):
            return False
        
        # Check orientation (alignment within 15°)
        ee_pos = self.env.Dcmm.data.body("link6").xpos
        obj_pos = self.env.Dcmm.data.body(self.env.object_name).xpos
        ee_to_obj = obj_pos - ee_pos
        ee_to_obj_norm = ee_to_obj / (np.linalg.norm(ee_to_obj) + 1e-6)
        ee_quat = self.env.Dcmm.data.body("link6").xquat
        palm_forward = quat_rotate_vector(ee_quat, np.array([0, 0, -1]))
        cos_theta = np.dot(palm_forward, ee_to_obj_norm)
        if cos_theta < ORIENTATION_THRESHOLD_COS:
            return False
        
        # Check EE velocity (should be low for stable handoff)
        ee_vel = self.env.Dcmm.data.body("link6").cvel[3:6]
        ee_speed = np.linalg.norm(ee_vel)
        if ee_speed >= 0.05:
            return False
        
        # All conditions met!
        return True
    
    def _compute_pregrasp_success_bonus(self, info):
        """
        Compute success bonus for achieving pre-grasp pose.
        
        [NEW 2025-01-04] Success is based on POSE, NOT contact.
        This prevents the agent from incentivizing hard collisions.
        
        Pre-grasp conditions:
        - d_ee < 0.05m
        - angle_err < 15° (cos > 0.966)
        - |v_ee| < 0.05 m/s
        - 0.7m < d_base < 0.9m
        
        Returns:
            float: Success bonus (+50) if pre-grasp achieved, else 0
        """
        if self.check_pregrasp_pose(info):
            return DcmmCfg.reward_weights.get("r_pregrasp_success", 50.0)
        return 0.0
    
    def _compute_timeout_penalty(self):
        """
        Compute timeout penalty when episode ends due to time limit.
        
        The penalty should only be applied once when the episode actually 
        terminates due to timeout, not on every step after time limit.
        
        Returns:
            float: Timeout penalty (-20) if timed out on this step, else 0
        """
        # Check if this is a timeout termination on this specific step
        # Only apply penalty if episode is terminating due to timeout
        env_time = self.env.Dcmm.data.time - self.env.start_time
        
        # Check if we've exceeded time limit and episode is actually terminating
        is_timeout = (env_time >= self.env.env_time and 
                      not self.env.step_touch and 
                      not getattr(self.env, 'terminated', False))
        
        if is_timeout:
            return DcmmCfg.reward_weights.get("r_timeout", -20.0)
        return 0.0
    
    def _compute_milestone_reward(self, info):
        """
        Compute one-time milestone bonus rewards for reaching distance thresholds.
        
        [NEW 2025-01-04] Provides intermediate rewards between dense rewards and
        final success bonus, filling the gap in reward signal when agent is 
        making good progress.
        
        Milestones:
        - 1.0m: +5.0 (first approach)
        - 0.5m: +10.0 (getting close)
        - 0.2m: +15.0 (final approach)
        
        Each milestone can only be triggered once per episode.
        
        Args:
            info: Environment info containing ee_distance
            
        Returns:
            float: Milestone bonus (0 if no new milestone reached)
        """
        ee_dist = info["ee_distance"]
        reward = 0.0
        
        # Check milestones in order (furthest to closest)
        # NOTE: If agent jumps from >1.0m to <0.2m in one step, they receive
        # ALL milestone rewards since they effectively passed through all thresholds.
        # This is intentional to provide proper credit for rapid progress.
        if not self.milestone_1m_reached and ee_dist < 1.0:
            self.milestone_1m_reached = True
            reward += DcmmCfg.reward_weights.get("r_milestone_1m", 5.0)
        
        if not self.milestone_05m_reached and ee_dist < 0.5:
            self.milestone_05m_reached = True
            reward += DcmmCfg.reward_weights.get("r_milestone_05m", 10.0)
        
        if not self.milestone_02m_reached and ee_dist < 0.2:
            self.milestone_02m_reached = True
            reward += DcmmCfg.reward_weights.get("r_milestone_02m", 15.0)
        
        return reward
    
    def reset_progress_tracking(self):
        """Reset progress tracking variables for new episode.
        
        [UPDATED 2025-01-04] Also resets AVP potential-based shaping state and milestones.
        """
        self.prev_ee_distance = None
        self.prev_base_distance = None
        self.stagnation_counter = 0
        
        # [NEW 2025-01-04] Reset milestone tracking
        self.milestone_1m_reached = False
        self.milestone_05m_reached = False
        self.milestone_02m_reached = False
        
        # [NEW 2025-01-04] Reset AVP potential for new episode
        if self.use_avp:
            self.avp_prev_potential = None
        
        # [NEW 2025-01-13] Reset episode minimum EE distance tracking
        self.current_episode_min_ee_distance = float('inf')
    
    def record_episode_end(self, termination_reason, initial_ee_distance=None, initial_base_distance=None):
        """
        Record episode termination statistics.
        
        [NEW 2025-01-13] 监控数值: Episode终止原因统计
        
        Args:
            termination_reason: str - 'timeout', 'collision', or 'success'
            initial_ee_distance: float - Initial EE-to-target distance at episode start
            initial_base_distance: float - Initial base-to-target distance at episode start
        """
        # 更新Episode终止原因统计
        self.reward_stats['episode_total_count'] += 1
        
        if termination_reason == 'timeout':
            self.reward_stats['episode_timeout_count'] += 1
        elif termination_reason == 'collision':
            self.reward_stats['episode_collision_count'] += 1
        elif termination_reason == 'success':
            self.reward_stats['episode_success_count'] += 1
        
        # 记录本Episode最小EE距离
        if self.current_episode_min_ee_distance < float('inf'):
            self.reward_stats['min_ee_distance_sum'] += self.current_episode_min_ee_distance
            self.reward_stats['min_ee_distance_list'].append(self.current_episode_min_ee_distance)
            # 限制列表长度,防止内存泄漏
            if len(self.reward_stats['min_ee_distance_list']) > 1000:
                self.reward_stats['min_ee_distance_list'] = self.reward_stats['min_ee_distance_list'][-500:]
        
        # 记录初始距离
        if initial_ee_distance is not None:
            self.reward_stats['initial_ee_distance_sum'] += initial_ee_distance
        if initial_base_distance is not None:
            self.reward_stats['initial_base_distance_sum'] += initial_base_distance

    # ========================================
    # Individual Reward Components
    # ========================================
    
    def _compute_arm_reaching_reward(self):
        """
        Compute arm reaching reward in base frame.
        
        This isolates the arm's contribution to reaching by computing
        distance in the robot's local frame.
        
        Returns:
            tuple: (reward, distance)
        """
        ee_pos_rel = self.env.obs_manager.get_relative_ee_pos3d()
        obj_pos_rel = self.env.obs_manager.get_relative_object_pos3d()
        arm_reach_distance = np.linalg.norm(ee_pos_rel - obj_pos_rel)
        
        # Use configured weight (default: 2.0)
        weight = DcmmCfg.reward_weights.get("r_arm_reaching", 2.0)
        reward = weight * (1.0 - np.tanh(3.0 * arm_reach_distance))
        return reward, arm_reach_distance
    
    def _compute_global_reaching_reward(self, info):
        """
        Compute global EE reaching reward.
        
        Args:
            info: Environment info containing ee_distance
            
        Returns:
            float: Reward value
        """
        weight = DcmmCfg.reward_weights.get("r_global_reaching", 0.5)
        return weight * (1.0 - np.tanh(2.0 * info["ee_distance"]))
    
    def _compute_base_approach_reward(self, info):
        """
        Compute base approach reward with optimal distance sweet spot.
        
        Args:
            info: Environment info containing base_distance
            
        Returns:
            float: Reward value (1.0 at optimal distance ~0.8m)
        """
        optimal_dist = 0.8
        dist_error = abs(info["base_distance"] - optimal_dist)
        return np.exp(-5.0 * dist_error**2)
    
    def _compute_arm_motion_reward(self):
        """
        Compute reward for arm joint deviation from initial pose.
        
        Encourages the agent to explore arm movements rather than
        keeping the arm in its initial configuration.
        
        Returns:
            tuple: (reward, joint_deviation)
        """
        # Arm joint indices: 15-21 in qpos
        current_joints = self.env.Dcmm.data.qpos[15:21]
        initial_joints = DcmmCfg.arm_joints
        joint_deviation = np.linalg.norm(current_joints - initial_joints)
        
        weight = DcmmCfg.reward_weights.get("r_arm_motion", 0.5)
        reward = weight * np.tanh(3.0 * joint_deviation)
        return reward, joint_deviation
    
    def _compute_arm_action_reward(self, ctrl):
        """
        Compute reward for arm action magnitude.
        
        Encourages the agent to actually use arm controls.
        
        Args:
            ctrl: Control dict with 'arm' key
            
        Returns:
            float: Reward proportional to arm action magnitude
        """
        arm_action = ctrl.get('arm', np.zeros(6))
        weight = DcmmCfg.reward_weights.get("r_arm_action", 0.2)
        return weight * np.linalg.norm(arm_action)
    
    def _compute_orientation_reward(self, info):
        """
        Compute reward for palm facing the target.
        
        Only computed when EE is within 2m of target.
        
        Args:
            info: Environment info containing ee_distance
            
        Returns:
            float: Orientation alignment reward
        """
        if info["ee_distance"] >= 2.0:
            return 0.0
        
        ee_pos = self.env.Dcmm.data.body("link6").xpos
        obj_pos = self.env.Dcmm.data.body(self.env.object_name).xpos
        
        # Direction from EE to object
        ee_to_obj = obj_pos - ee_pos
        ee_to_obj_norm = ee_to_obj / (np.linalg.norm(ee_to_obj) + 1e-6)
        
        # Palm forward direction (negative Z-axis of EE frame)
        ee_quat = self.env.Dcmm.data.body("link6").xquat
        palm_forward = quat_rotate_vector(ee_quat, np.array([0, 0, -1]))
        
        # Alignment: 1.0 = perfect, -1.0 = backwards
        alignment = np.dot(palm_forward, ee_to_obj_norm)
        
        # Apply power function for stricter alignment
        return max(0, alignment) ** self.env.current_orient_power * 2.0
    
    def _compute_touch_reward(self):
        """
        Compute reward for touching the target.
        
        Includes penalty for high-speed impacts to encourage gentle contact.
        
        Returns:
            tuple: (total_touch_reward, impact_penalty)
        """
        if not self.env.step_touch:
            return 0.0, 0.0
        
        # Get EE velocity for impact penalty
        ee_vel = self.env.Dcmm.data.body("link6").cvel[3:6]
        impact_speed = np.linalg.norm(ee_vel)
        
        base_reward = 10.0
        impact_penalty = -4.0 * impact_speed
        
        return base_reward + impact_penalty, impact_penalty
    
    def _compute_regularization_penalty(self, ctrl):
        """
        Compute control regularization penalty.
        
        Penalizes base control more than arm to encourage arm usage.
        
        Args:
            ctrl: Control dict
            
        Returns:
            float: Negative penalty value
        """
        base_scale = DcmmCfg.reward_weights.get('r_base_ctrl_scale', 0.005)
        arm_scale = DcmmCfg.reward_weights.get('r_arm_ctrl_scale', 0.001)
        
        base_penalty = -np.linalg.norm(ctrl['base']) * DcmmCfg.reward_weights['r_ctrl']['base'] * base_scale
        arm_penalty = -np.linalg.norm(ctrl['arm']) * DcmmCfg.reward_weights['r_ctrl']['arm'] * arm_scale
        return base_penalty + arm_penalty
    
    def _compute_collision_penalty(self):
        """
        Compute catastrophic collision penalty.
        
        Returns:
            float: Large negative penalty if terminated without touch
        """
        if self.env.terminated and not self.env.step_touch:
            return DcmmCfg.reward_weights["r_collision"]  # -10.0
        return 0.0
    
    def _compute_plant_collision_penalty(self):
        """
        Compute plant (stem/leaf) collision penalties.
        
        Stem collisions are penalized more severely than leaf collisions.
        Leaf penalties are velocity-dependent.
        
        Returns:
            float: Negative penalty value
        """
        penalty = 0.0
        
        # Stem collision (rigid, severe penalty)
        if self.env.contacts['plant_contacts'].size != 0:
            penalty += self.env.current_w_stem
        
        # Leaf collision (soft, velocity-dependent penalty)
        if self.env.contacts['leaf_contacts'].size != 0:
            ee_vel = np.linalg.norm(self.env.Dcmm.data.body("link6").cvel[3:6])
            penalty += -0.5 * (1.0 + ee_vel)
        
        return penalty
    
    def _compute_action_rate_penalty(self, ctrl):
        """
        Compute penalty for rapid action changes.
        
        Args:
            ctrl: Control dict
            
        Returns:
            float: Negative penalty for action changes
        """
        current_action = np.concatenate([ctrl['base'], ctrl['arm'], ctrl['hand']])
        
        if self.prev_action_reward is None:
            self.prev_action_reward = np.zeros_like(current_action)
        
        action_diff = current_action - self.prev_action_reward
        action_rate_scale = DcmmCfg.reward_weights.get('r_action_rate', 0.02)
        penalty = -np.linalg.norm(action_diff) * action_rate_scale
        
        self.prev_action_reward = current_action.copy()
        return penalty

    def _compute_joint_limit_penalty(self):
        """
        Penalize when joints approach limits (soft constraint).
        
        Uses 10% margin from joint limits to start applying penalty.
        This encourages the agent to stay away from joint limits for safety.
        
        Returns:
            float: Negative penalty value when joints are near limits
        """
        arm_qpos = self.env.Dcmm.data.qpos[15:21]
        
        # Get joint limits from MuJoCo model (indices 9-15 for arm joints)
        # jnt_range indices 9-15 correspond to qpos indices 15-21
        lower_limits = self.env.Dcmm.model.jnt_range[9:15, 0]
        upper_limits = self.env.Dcmm.model.jnt_range[9:15, 1]
        
        # 10% edge margin for soft constraint
        margin = 0.1 * (upper_limits - lower_limits)
        
        # Compute violations (positive when in margin zone)
        lower_violation = np.maximum(0, lower_limits + margin - arm_qpos)
        upper_violation = np.maximum(0, arm_qpos - (upper_limits - margin))
        
        penalty_weight = DcmmCfg.reward_weights.get('r_joint_limit', -2.0)
        return penalty_weight * (np.sum(lower_violation) + np.sum(upper_violation))

    def _compute_base_heading_reward(self, info):
        """
        Reward for base facing toward target.
        
        [UPDATED 2025-01-04] Added distance gate to prevent "站桩刷分" behavior
        (agent spinning in place to farm orientation reward without approaching).
        
        Only gives orientation reward when:
        - Base distance > 1.5m: No orientation reward (focus on approaching first)
        - Base distance <= 1.5m: Full orientation reward
        
        Args:
            info: Environment info containing distances
            
        Returns:
            float: Reward value (0.5 when perfectly aligned, 0 if gated)
        """
        # [NEW 2025-01-04] Distance gate to prevent spinning in place
        gate_distance = DcmmCfg.reward_weights.get('r_base_heading_gate', 1.5)
        if info["base_distance"] > gate_distance:
            return 0.0
        
        # [FIX 2025-01-04] Use 'arm_base' consistently with rest of codebase
        base_pos = self.env.Dcmm.data.body("arm_base").xpos[:2]
        obj_pos = self.env.Dcmm.data.body(self.env.object_name).xpos[:2]
        
        # Target direction
        to_target = obj_pos - base_pos
        target_heading = np.arctan2(to_target[1], to_target[0])
        
        # Current base heading (yaw from quaternion)
        # [FIX 2025-01-04] Use 'arm_base' consistently with rest of codebase
        base_quat = self.env.Dcmm.data.body("arm_base").xquat
        base_heading = quat_to_euler(base_quat)[2]  # yaw
        
        # [FIX 2025-01-04] Use consistent argument order: angle_diff(target, base)
        # This matches observation_manager.py for consistency
        heading_error = np.abs(angle_diff(target_heading, base_heading))
        
        weight = DcmmCfg.reward_weights.get('r_base_heading', 0.5)
        return weight * np.exp(-2.0 * heading_error)

    def _compute_precision_reward(self, info):
        """
        Precision reward when very close to target (Catching-inspired).
        
        Uses Gaussian decay to provide high reward for very precise positioning.
        Only active when EE is within 0.3m of target.
        
        Args:
            info: Environment info containing ee_distance
            
        Returns:
            float: Precision reward (up to 10.0 at very close range)
        """
        if info["ee_distance"] > 0.3:
            return 0.0
        
        # Gaussian decay: 0.05m -> ~7.8 points, 0.1m -> ~6.1 points
        weight = DcmmCfg.reward_weights.get('r_precision', 10.0)
        return weight * np.exp(-50.0 * info["ee_distance"]**2)

    def _compute_contact_persistence_reward(self):
        """
        Reward for sustained contact (inspired by Catching's stability reward).
        
        Provides incremental reward for maintaining contact over multiple steps.
        This transitions from binary touch reward to continuous contact reward.
        
        Returns:
            float: Contact persistence reward (0-2.0 based on contact duration)
        """
        if not self.env.step_touch:
            return 0.0
        
        # Contact steps capped at 10 for reward calculation
        contact_steps = min(self.env.contact_count, 10)
        weight = DcmmCfg.reward_weights.get('r_contact_persistence', 2.0)
        return weight * (contact_steps / 10.0)

    # ========================================
    # AVP (Asymmetric Value Propagation)
    # [REFACTORED 2025-01-04] Major improvements:
    # 1. Potential-based reward shaping: r_avp = λ·(γ·Φ(s_{t+1}) - Φ(s_t))
    # 2. OOD/Confidence gating: visual validity + MC Dropout uncertainty
    # 3. Improved virtual obs using real arm state
    # 4. Success-rate based adaptive lambda scheduling
    # ========================================
    
    def compute_avp_reward(self, obs, info):
        """
        Compute AVP reward using Stage 2 Critic with potential-based shaping.
        
        [REFACTORED 2025-01-04] Key changes:
        - Uses potential difference shaping: r_avp = λ·(γ·Φ(s') - Φ(s))
        - Rewards "moving toward graspable states" not "being in graspable states"
        - OOD gating: visual validity check + MC Dropout uncertainty
        - Adaptive lambda based on success rate

        Args:
            obs: Current observation dict
            info: Environment info dict

        Returns:
            float: AVP reward (0 if disabled or gated)
        """
        if not self.use_avp or self.grasp_critic is None:
            return 0.0
        
        # ========================================
        # 1. WARM-UP GATING: λ=0 for first N steps
        # ========================================
        global_step = getattr(self.env, 'global_step', 0)
        if global_step < self.avp_warmup_steps:
            if self.avp_stats is not None:
                self.avp_stats['warmup_gated_count'] += 1
            return 0.0
        
        # ========================================
        # 2. UPDATE LAMBDA (cosine or piecewise linear)
        # [UPDATED 2025-01-04] Use cosine schedule for smoother transitions
        # ========================================
        if getattr(DcmmCfg.avp, 'use_cosine_schedule', False):
            self._update_avp_lambda_cosine()
        else:
            self._update_avp_lambda()
        
        # ========================================
        # 3. DISTANCE GATING
        # ========================================
        if info["ee_distance"] > self.avp_gate_distance:
            if self.avp_stats is not None:
                self.avp_stats['gated_count'] += 1
            return 0.0
        
        try:
            # ========================================
            # 4. VISUAL VALIDITY GATING
            # ========================================
            depth_valid_ratio = self._compute_depth_valid_ratio()
            if depth_valid_ratio < self.avp_depth_valid_threshold:
                if self.avp_stats is not None:
                    self.avp_stats['visual_gated_count'] += 1
                return 0.0
            
            # ========================================
            # 5. CONSTRUCT VIRTUAL OBS (improved)
            # ========================================
            input_dict, depth_obs = self._construct_virtual_obs_v2(obs)
            
            # ========================================
            # 6. COMPUTE POTENTIAL Φ(s) WITH MC DROPOUT UNCERTAINTY
            # ========================================
            current_potential, confidence = self._compute_potential_with_uncertainty(input_dict)
            
            # Low confidence → gate out (threshold is configurable via avp.min_confidence)
            if confidence < self.avp_min_confidence:
                if self.avp_stats is not None:
                    self.avp_stats['uncertainty_gated_count'] += 1
                # Still update prev_potential for next step
                self.avp_prev_potential = current_potential
                return 0.0
            
            # ========================================
            # 7. POTENTIAL-BASED SHAPING
            # r_avp = λ · (γ · Φ(s') - Φ(s))
            # Note: We don't multiply by confidence here because:
            # 1. We already gate on low confidence above
            # 2. Weighting by confidence would diminish meaningful signals
            #    even when the critic is reasonably accurate
            # ========================================
            if self.avp_prev_potential is None:
                # First step of episode, no reward yet
                self.avp_prev_potential = current_potential
                return 0.0
            
            # Potential difference: positive if moving toward higher-value states
            potential_diff = self.avp_gamma * current_potential - self.avp_prev_potential
            
            # Apply lambda scaling (standard potential-based shaping)
            # Note: We don't multiply by confidence - already gated above
            avp_reward = self.avp_lambda * potential_diff
            
            # Clip to prevent extreme values
            avp_reward = np.clip(avp_reward, -self.avp_reward_clip, self.avp_reward_clip)
            
            # Update previous potential for next step
            self.avp_prev_potential = current_potential
            
            # ========================================
            # 8. UPDATE STATISTICS
            # ========================================
            if self.avp_stats is not None:
                self.avp_stats['reward_sum'] += avp_reward
                self.avp_stats['critic_value_sum'] += current_potential
                self.avp_stats['potential_diff_sum'] += potential_diff
                self.avp_stats['confidence_sum'] += confidence
                self.avp_stats['count'] += 1
            
            return avp_reward
            
        except Exception as e:
            if self.env.print_reward:
                print(f">>> AVP Error: {e}")
            return 0.0
    
    def _update_avp_lambda(self):
        """
        Update AVP lambda based on curriculum difficulty (step-based scheduling).
        
        [NEW 2025-01-04] Three-phase lambda scheduling:
        - Warm-up: λ=0 for first warmup_steps (handled in compute_avp_reward)
        - Ramp-up: difficulty < 0.3 → λ ramps from 0 to lambda_max
        - Full: 0.3 <= difficulty < 0.7 → λ = lambda_max
        - Decay: difficulty >= 0.7 → λ decays to lambda_min
        
        Note: Uses curriculum_difficulty (step-based, 0-1) as the scheduling signal.
        This provides smooth scheduling without complex success rate tracking.
        """
        # Use curriculum difficulty as scheduling signal
        # difficulty = global_step / max_steps (0.0 to 1.0)
        if hasattr(self.env, 'curriculum_difficulty'):
            difficulty = self.env.curriculum_difficulty
            # Early training (difficulty < 0.3): ramp up lambda
            if difficulty < 0.3:
                self.avp_lambda = self.avp_lambda_max * (difficulty / 0.3)
            # Mid training (0.3 <= difficulty < 0.7): full lambda
            elif difficulty < 0.7:
                self.avp_lambda = self.avp_lambda_max
            # Late training (difficulty >= 0.7): decay lambda
            else:
                decay_progress = (difficulty - 0.7) / 0.3
                self.avp_lambda = self.avp_lambda_max - (self.avp_lambda_max - self.avp_lambda_min) * decay_progress
    
    def _update_avp_lambda_cosine(self):
        """
        Update AVP lambda using smooth cosine scheduling.
        
        [NEW 2025-01-04] Cosine scheduling provides smoother transitions than
        piecewise linear, preventing sudden λ changes that cause policy oscillation.
        
        Schedule:
        - Warm-up (difficulty < rampup_end): Cosine ramp from 0 to lambda_max
        - Full (rampup_end <= difficulty < decay_start): lambda_max
        - Decay (difficulty >= decay_start): Cosine decay to lambda_min
        """
        if not hasattr(self.env, 'curriculum_difficulty'):
            return
        
        difficulty = self.env.curriculum_difficulty
        
        # Get cosine schedule parameters
        rampup_end = getattr(DcmmCfg.avp, 'cosine_rampup_end', 0.3)
        decay_start = getattr(DcmmCfg.avp, 'cosine_decay_start', 0.7)
        
        if difficulty < rampup_end:
            # Cosine ramp-up: 0 -> lambda_max
            # Uses (1 - cos(x * π)) / 2 for smooth start (derivative=0 at 0)
            progress = difficulty / rampup_end
            self.avp_lambda = self.avp_lambda_max * (1 - np.cos(progress * np.pi)) / 2
        elif difficulty < decay_start:
            # Full lambda
            self.avp_lambda = self.avp_lambda_max
        else:
            # Cosine decay: lambda_max -> lambda_min
            # Uses (1 + cos(x * π)) / 2 for smooth end (derivative=0 at 1)
            progress = (difficulty - decay_start) / (1.0 - decay_start)
            decay_range = self.avp_lambda_max - self.avp_lambda_min
            self.avp_lambda = self.avp_lambda_min + decay_range * (1 + np.cos(progress * np.pi)) / 2
    
    def _compute_depth_valid_ratio(self):
        """
        Compute ratio of valid depth pixels.
        
        [NEW 2025-01-04] Visual validity gating for OOD detection.
        Depth camera can have holes/invalid pixels in difficult scenes.
        
        Returns:
            float: Ratio of valid pixels (0.0 to 1.0)
        """
        try:
            depth_obs = self.env.render_manager.get_depth_obs(
                width=self.avp_img_size,
                height=self.avp_img_size,
                add_noise=False,  # Get raw depth for validity check
                add_holes=False
            )
            
            total_pixels = depth_obs.size
            # Valid: non-zero and not max depth (10m typically)
            valid_mask = (depth_obs > 0.01) & (depth_obs < 9.9)
            valid_pixels = np.sum(valid_mask)
            
            return valid_pixels / total_pixels
            
        except (AttributeError, RuntimeError) as e:
            # Render manager not available or rendering failed
            return 1.0  # Default to valid if error
    
    def _compute_potential_with_uncertainty(self, input_dict):
        """
        Compute potential Φ(s) with MC Dropout uncertainty estimation.
        
        [NEW 2025-01-04] OOD detection via critic uncertainty.
        
        Args:
            input_dict: Normalized input for critic
            
        Returns:
            tuple: (potential_mean, confidence)
                - potential_mean: Mean value estimate
                - confidence: exp(-α·σ) where σ is std of MC samples
        """
        # Check if critic has dropout layers
        has_dropout = any(
            isinstance(m, torch.nn.Dropout) 
            for m in self.grasp_critic.modules()
        )
        
        if not has_dropout or self.avp_mc_dropout_samples <= 1:
            # No dropout or single sample mode
            with torch.no_grad():
                res = self.grasp_critic.act(input_dict)
                return res['values'].item(), 1.0
        
        # MC Dropout: multiple forward passes with dropout enabled
        values = []
        
        # Temporarily enable only dropout layers for MC sampling
        # (avoid affecting BatchNorm running statistics)
        dropout_modules = []
        for m in self.grasp_critic.modules():
            if isinstance(m, torch.nn.Dropout):
                dropout_modules.append((m, m.training))
                m.train(True)
        
        with torch.no_grad():
            for _ in range(self.avp_mc_dropout_samples):
                res = self.grasp_critic.act(input_dict)
                values.append(res['values'].item())
        
        # Restore original mode for dropout layers
        for m, was_training in dropout_modules:
            m.train(was_training)
        
        # Compute mean and std
        values_np = np.array(values)
        mean_value = np.mean(values_np)
        std_value = np.std(values_np)
        
        # Confidence: exp(-α·σ)
        confidence = np.exp(-self.avp_uncertainty_alpha * std_value)
        
        return mean_value, confidence
    
    def _construct_virtual_obs_v2(self, obs):
        """
        Construct virtual observation for Stage 2 Critic (improved version).
        
        [REFACTORED 2025-01-04] Key improvements:
        1. Uses REAL arm joints/ee pose instead of fixed ready pose
        2. Uses REAL velocities with clipping
        3. Hand at near-grasp open position (not extreme open)
        
        This makes the virtual observation closer to actual Stage 2 handoff state.
        
        Args:
            obs: Current observation
            
        Returns:
            tuple: (input_dict, depth_obs)
        """
        # ========================================
        # ARM STATE: Use REAL current state
        # ========================================
        # Real EE position (relative to base frame)
        real_ee_pos = self.env.obs_manager.get_relative_ee_pos3d().astype(np.float32)
        
        # Real EE quaternion
        real_ee_quat = self.env.Dcmm.data.body("link6").xquat.astype(np.float32)
        
        # Real EE velocity (clipped for stability)
        real_ee_vel = self.env.Dcmm.data.body("link6").cvel[3:6].astype(np.float32)
        real_ee_vel = np.clip(real_ee_vel, -0.5, 0.5)  # Clip to reasonable range
        
        # Real arm joint positions
        real_arm_joints = self.env.Dcmm.data.qpos[15:21].astype(np.float32)
        
        # ========================================
        # HAND STATE: Near-grasp open (not extreme)
        # ========================================
        # Use slightly pre-closed hand for better Stage 2 value estimate
        near_grasp_hand = np.array([
            0.1, 0.1, 0.0,   # Index finger slightly flexed
            0.1, 0.1, 0.0,   # Middle finger slightly flexed
            0.1, 0.1, 0.0,   # Ring finger slightly flexed
            0.1, 0.4, 0.1    # Thumb ready for opposition
        ], dtype=np.float32)
        
        # ========================================
        # OBJECT STATE: Real position
        # ========================================
        real_obj_pos = self.env.obs_manager.get_relative_object_pos3d().astype(np.float32)
        
        # ========================================
        # TOUCH: Zero (no contact in virtual state)
        # ========================================
        virtual_touch = np.zeros(4, dtype=np.float32)
        
        # ========================================
        # CONCATENATE STATE (35 dim)
        # ========================================
        state_vec = np.concatenate([
            real_ee_pos,       # 3
            real_ee_quat,      # 4
            real_ee_vel,       # 3
            real_arm_joints,   # 6
            real_obj_pos,      # 3
            near_grasp_hand,   # 12
            virtual_touch      # 4
        ])
        
        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_normalized = self.running_mean_std(state_tensor)
        
        # ========================================
        # DEPTH IMAGE: Real with noise
        # ========================================
        depth_obs = self.env.render_manager.get_depth_obs(
            width=self.avp_img_size,
            height=self.avp_img_size,
            add_noise=True,
            add_holes=True
        )
        depth_tensor = torch.tensor(
            depth_obs.flatten(), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        
        obs_combined = torch.cat([state_normalized, depth_tensor], dim=1)
        return {'obs': obs_combined}, depth_obs
    
    # Keep legacy method for backward compatibility
    def _construct_virtual_obs(self, obs):
        """Legacy virtual obs construction (deprecated, use _construct_virtual_obs_v2)."""
        input_dict, _ = self._construct_virtual_obs_v2(obs)
        return input_dict

    # ========================================
    # Statistics and Logging
    # ========================================
    
    def _init_reward_stats(self):
        """Initialize reward statistics for WandB logging."""
        self.reward_stats = {
            # New progress-based rewards (2025-01-04)
            'ee_progress_sum': 0.0,
            'base_progress_sum': 0.0,
            'alive_penalty_sum': 0.0,
            'stagnation_penalty_sum': 0.0,
            # Conditional rewards
            'orientation_sum': 0.0,
            'base_heading_sum': 0.0,
            # [NEW 2025-01-04] Milestone rewards
            'milestone_sum': 0.0,
            # Terminal rewards
            'collision_sum': 0.0,
            'success_sum': 0.0,
            'touch_sum': 0.0,
            # Distance tracking
            'ee_distance_sum': 0.0,
            'base_distance_sum': 0.0,
            # Legacy (kept for comparison)
            'reaching_sum': 0.0,
            'base_approach_sum': 0.0,
            'plant_collision_sum': 0.0,
            'arm_reaching_sum': 0.0,
            'arm_motion_sum': 0.0,
            'arm_reach_distance_sum': 0.0,
            'joint_limit_sum': 0.0,
            'precision_sum': 0.0,
            'contact_persistence_sum': 0.0,
            'count': 0,
            # ========================================
            # [NEW 2025-01-13] 监控数值: Episode终止原因统计
            # ========================================
            'episode_timeout_count': 0,       # 超时终止次数
            'episode_collision_count': 0,     # 碰撞终止次数
            'episode_success_count': 0,       # 成功终止次数
            'episode_total_count': 0,         # 总Episode数
            # ========================================
            # [NEW 2025-01-13] 监控数值: EE距离分布
            # ========================================
            'min_ee_distance_sum': 0.0,       # 每Episode最小EE距离之和
            'min_ee_distance_list': [],       # 每Episode最小EE距离列表（用于统计分布）
            # ========================================
            # [NEW 2025-01-13] 监控数值: 初始距离配置
            # ========================================
            'initial_ee_distance_sum': 0.0,   # 初始EE距离之和
            'initial_base_distance_sum': 0.0, # 初始底盘距离之和
        }
        # [NEW 2025-01-13] 用于追踪当前Episode最小EE距离
        self.current_episode_min_ee_distance = float('inf')
    
    def _update_reward_stats_v2(self, info, r_ee_progress, r_base_progress,
                                r_alive, r_stagnation, r_orientation,
                                r_collision, r_success, r_touch, r_milestone=0.0):
        """Update running statistics for new reward structure logging."""
        self.reward_stats['ee_progress_sum'] += r_ee_progress
        self.reward_stats['base_progress_sum'] += r_base_progress
        self.reward_stats['alive_penalty_sum'] += r_alive
        self.reward_stats['stagnation_penalty_sum'] += r_stagnation
        self.reward_stats['orientation_sum'] += r_orientation
        self.reward_stats['collision_sum'] += r_collision
        self.reward_stats['success_sum'] += r_success
        self.reward_stats['touch_sum'] += r_touch
        # [NEW 2025-01-04] Milestone tracking
        self.reward_stats['milestone_sum'] += r_milestone
        self.reward_stats['ee_distance_sum'] += info["ee_distance"]
        self.reward_stats['base_distance_sum'] += info["base_distance"]
        self.reward_stats['count'] += 1
        
        # [NEW 2025-01-13] 追踪当前Episode最小EE距离
        if info["ee_distance"] < self.current_episode_min_ee_distance:
            self.current_episode_min_ee_distance = info["ee_distance"]
    
    def _update_reward_stats(self, r_reaching, r_base_approach, r_orientation,
                             r_touch, r_collision, r_plant_collision,
                             r_arm_reaching, r_arm_motion, arm_reach_dist, info,
                             r_joint_limit=0.0, r_base_heading=0.0,
                             r_precision=0.0, r_contact_persistence=0.0):
        """Update running statistics for logging (legacy version)."""
        self.reward_stats['reaching_sum'] += r_reaching
        self.reward_stats['base_approach_sum'] += r_base_approach
        self.reward_stats['orientation_sum'] += r_orientation
        self.reward_stats['touch_sum'] += r_touch
        self.reward_stats['collision_sum'] += r_collision
        self.reward_stats['plant_collision_sum'] += r_plant_collision
        self.reward_stats['ee_distance_sum'] += info["ee_distance"]
        self.reward_stats['base_distance_sum'] += info["base_distance"]
        self.reward_stats['arm_reaching_sum'] += r_arm_reaching
        self.reward_stats['arm_motion_sum'] += r_arm_motion
        self.reward_stats['arm_reach_distance_sum'] += arm_reach_dist
        # [NEW 2025-12-20] Additional reward stats
        self.reward_stats['joint_limit_sum'] += r_joint_limit
        self.reward_stats['base_heading_sum'] += r_base_heading
        self.reward_stats['precision_sum'] += r_precision
        self.reward_stats['contact_persistence_sum'] += r_contact_persistence
        self.reward_stats['count'] += 1
    
    def get_reward_stats_and_reset(self):
        """
        Get reward statistics for WandB logging and reset counters.

        Returns:
            dict: Averaged reward component statistics, or None if no data
        """
        if self.reward_stats['count'] == 0:
            return None
        
        count = self.reward_stats['count']
        stats = {
            # New progress-based rewards
            'rewards/ee_progress_mean': self.reward_stats['ee_progress_sum'] / count,
            'rewards/base_progress_mean': self.reward_stats['base_progress_sum'] / count,
            'rewards/alive_penalty_mean': self.reward_stats['alive_penalty_sum'] / count,
            'rewards/stagnation_penalty_mean': self.reward_stats['stagnation_penalty_sum'] / count,
            # Conditional/terminal
            'rewards/orientation_mean': self.reward_stats['orientation_sum'] / count,
            'rewards/collision_mean': self.reward_stats['collision_sum'] / count,
            'rewards/success_mean': self.reward_stats['success_sum'] / count,
            'rewards/touch_mean': self.reward_stats['touch_sum'] / count,
            # [NEW 2025-01-04] Milestone rewards
            'rewards/milestone_mean': self.reward_stats['milestone_sum'] / count,
            # Legacy (for comparison)
            'rewards/reaching_mean': self.reward_stats['reaching_sum'] / count,
            'rewards/base_approach_mean': self.reward_stats['base_approach_sum'] / count,
            'rewards/plant_collision_mean': self.reward_stats['plant_collision_sum'] / count,
            'rewards/arm_reaching_mean': self.reward_stats['arm_reaching_sum'] / count,
            'rewards/arm_motion_mean': self.reward_stats['arm_motion_sum'] / count,
            'rewards/joint_limit_mean': self.reward_stats['joint_limit_sum'] / count,
            'rewards/base_heading_mean': self.reward_stats['base_heading_sum'] / count,
            'rewards/precision_mean': self.reward_stats['precision_sum'] / count,
            'rewards/contact_persistence_mean': self.reward_stats['contact_persistence_sum'] / count,
            # Distances
            'distance/ee_distance_mean': self.reward_stats['ee_distance_sum'] / count,
            'distance/base_distance_mean': self.reward_stats['base_distance_sum'] / count,
            'distance/arm_reach_distance_mean': self.reward_stats['arm_reach_distance_sum'] / count,
            # Curriculum
            'curriculum/difficulty': self.env.curriculum_difficulty,
            'curriculum/w_stem': self.env.current_w_stem,
            'curriculum/orient_power': self.env.current_orient_power,
        }
        
        # ========================================
        # [NEW 2025-01-13] 监控数值: Episode终止原因统计
        # ========================================
        episode_count = self.reward_stats['episode_total_count']
        if episode_count > 0:
            stats['episode/timeout_ratio'] = self.reward_stats['episode_timeout_count'] / episode_count
            stats['episode/collision_ratio'] = self.reward_stats['episode_collision_count'] / episode_count
            stats['episode/success_ratio'] = self.reward_stats['episode_success_count'] / episode_count
            stats['episode/total_count'] = episode_count
        
        # ========================================
        # [NEW 2025-01-13] 监控数值: EE距离分布
        # ========================================
        if episode_count > 0:
            stats['distance/min_ee_distance_mean'] = self.reward_stats['min_ee_distance_sum'] / episode_count
            # 计算分布统计（如果有足够数据）
            min_ee_list = self.reward_stats['min_ee_distance_list']
            if len(min_ee_list) >= 10:
                import numpy as np
                min_ee_arr = np.array(min_ee_list)
                stats['distance/min_ee_distance_median'] = float(np.median(min_ee_arr))
                stats['distance/min_ee_distance_p10'] = float(np.percentile(min_ee_arr, 10))
                stats['distance/min_ee_distance_p90'] = float(np.percentile(min_ee_arr, 90))
        
        # ========================================
        # [NEW 2025-01-13] 监控数值: 初始距离配置
        # ========================================
        if episode_count > 0:
            stats['distance/initial_ee_distance_mean'] = self.reward_stats['initial_ee_distance_sum'] / episode_count
            stats['distance/initial_base_distance_mean'] = self.reward_stats['initial_base_distance_sum'] / episode_count
        
        self._init_reward_stats()
        return stats
    
    def get_avp_stats_and_reset(self):
        """
        Get AVP statistics for WandB logging and reset counters.
        
        [UPDATED 2025-01-04] Now includes potential-based shaping metrics
        and confidence/gating statistics.

        Returns:
            dict: AVP statistics, or None if disabled/no data
        """
        if self.avp_stats is None or self.avp_stats['count'] == 0:
            return None
        
        count = self.avp_stats['count']
        total_calls = (count + 
                       self.avp_stats.get('gated_count', 0) + 
                       self.avp_stats.get('warmup_gated_count', 0) +
                       self.avp_stats.get('visual_gated_count', 0) +
                       self.avp_stats.get('uncertainty_gated_count', 0))
        
        stats = {
            # Core metrics
            'avp/reward_mean': self.avp_stats['reward_sum'] / count,
            'avp/critic_value_mean': self.avp_stats['critic_value_sum'] / count,
            'avp/lambda': self.avp_lambda,
            
            # [NEW 2025-01-04] Potential-based shaping metrics
            'avp/potential_diff_mean': self.avp_stats.get('potential_diff_sum', 0.0) / count,
            'avp/confidence_mean': self.avp_stats.get('confidence_sum', 0.0) / count,
            
            # [NEW 2025-01-04] Detailed gating statistics
            'avp/distance_gate_ratio': self.avp_stats.get('gated_count', 0) / total_calls if total_calls > 0 else 0,
            'avp/warmup_gate_ratio': self.avp_stats.get('warmup_gated_count', 0) / total_calls if total_calls > 0 else 0,
            'avp/visual_gate_ratio': self.avp_stats.get('visual_gated_count', 0) / total_calls if total_calls > 0 else 0,
            'avp/uncertainty_gate_ratio': self.avp_stats.get('uncertainty_gated_count', 0) / total_calls if total_calls > 0 else 0,
            'avp/active_ratio': count / total_calls if total_calls > 0 else 0,
            'avp/count': count,
        }
        
        # Reset counters
        self.avp_stats = {
            'reward_sum': 0.0,
            'critic_value_sum': 0.0,
            'potential_diff_sum': 0.0,
            'confidence_sum': 0.0,
            'count': 0,
            'gated_count': 0,
            'warmup_gated_count': 0,
            'visual_gated_count': 0,
            'uncertainty_gated_count': 0,
        }
        
        return stats
    
    def _print_reward_breakdown(self, r_reaching, r_arm_reaching, r_arm_motion,
                                r_arm_action, r_base_approach, r_orientation,
                                r_touch, r_impact, r_regularization, r_collision,
                                r_plant_collision, r_action_rate, r_avp,
                                arm_reach_dist, arm_joint_dev, total,
                                r_joint_limit=0.0, r_base_heading=0.0,
                                r_precision=0.0, r_contact_persistence=0.0):
        """Print detailed reward breakdown for debugging (legacy version)."""
        print(f"reward_reaching: {r_reaching:.3f}")
        print(f"reward_arm_reaching: {r_arm_reaching:.3f}")
        print(f"reward_arm_motion: {r_arm_motion:.3f}")
        print(f"reward_arm_action: {r_arm_action:.3f}")
        print(f"reward_base_approach: {r_base_approach:.3f}")
        print(f"reward_orientation: {r_orientation:.3f}")
        print(f"reward_touch: {r_touch:.3f} (Impact: {r_impact:.3f})")
        print(f"reward_regularization: {r_regularization:.3f}")
        print(f"reward_collision: {r_collision:.3f}")
        print(f"reward_plant_collision: {r_plant_collision:.3f}")
        print(f"reward_action_rate: {r_action_rate:.3f}")
        print(f"reward_avp: {r_avp:.3f}")
        # [NEW 2025-12-20] Additional reward components
        print(f"reward_joint_limit: {r_joint_limit:.3f}")
        print(f"reward_base_heading: {r_base_heading:.3f}")
        print(f"reward_precision: {r_precision:.3f}")
        print(f"reward_contact_persistence: {r_contact_persistence:.3f}")
        print(f"arm_reach_distance: {arm_reach_dist:.3f}")
        print(f"arm_joint_deviation: {arm_joint_dev:.3f}")
        print(f"total reward: {total:.3f}\n")

    def _print_reward_breakdown_v2(self, r_ee_progress, r_base_progress,
                                   r_alive, r_stagnation, r_orientation,
                                   r_base_heading, r_touch, r_collision,
                                   r_plant_collision, r_action_rate, r_avp,
                                   r_success, r_timeout, r_milestone, info, total):
        """Print detailed reward breakdown for debugging (new progress-based version)."""
        print("=" * 50)
        print("REWARD BREAKDOWN (Progress-Based)")
        print("=" * 50)
        print(f"[PROGRESS]")
        print(f"  EE Progress:      {r_ee_progress:>8.4f}")
        print(f"  Base Progress:    {r_base_progress:>8.4f}")
        print(f"[PENALTIES]")
        print(f"  Alive Penalty:    {r_alive:>8.4f}")
        print(f"  Stagnation:       {r_stagnation:>8.4f}")
        print(f"  Action Rate:      {r_action_rate:>8.4f}")
        print(f"  Plant Collision:  {r_plant_collision:>8.4f}")
        print(f"[CONDITIONAL]")
        print(f"  Orientation:      {r_orientation:>8.4f}")
        print(f"  Base Heading:     {r_base_heading:>8.4f}")
        print(f"[MILESTONES]")
        print(f"  Milestone:        {r_milestone:>8.4f}")
        print(f"[TERMINAL]")
        print(f"  Collision:        {r_collision:>8.4f}")
        print(f"  Timeout:          {r_timeout:>8.4f}")
        print(f"  Success:          {r_success:>8.4f}")
        print(f"[OPTIONAL]")
        print(f"  Touch:            {r_touch:>8.4f}")
        print(f"  AVP:              {r_avp:>8.4f}")
        print("-" * 50)
        print(f"[DISTANCES]")
        print(f"  EE Distance:      {info['ee_distance']:>8.4f} m")
        print(f"  Base Distance:    {info['base_distance']:>8.4f} m")
        print("=" * 50)
        print(f"TOTAL REWARD:       {total:>8.4f}")
        print("=" * 50 + "\n")

    # ========================================
    # Legacy Methods (for compatibility)
    # ========================================
    
    def norm_ctrl(self, ctrl, components):
        """
        Compute weighted norm of control actions.
        
        Args:
            ctrl: Control dict
            components: List of component names ('base', 'arm', 'hand')
            
        Returns:
            float: Weighted norm value
        """
        ctrl_array = np.concatenate([
            ctrl[comp] * DcmmCfg.reward_weights['r_ctrl'][comp]
            for comp in components
        ])
        return np.linalg.norm(ctrl_array)
