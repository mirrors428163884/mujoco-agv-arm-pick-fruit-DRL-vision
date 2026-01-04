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
   - EE Progress: k_ee * (d_ee^{t-1} - d_ee^t), clipped to [-0.2, 0.2]
   - Base Progress: k_base * (d_base^{t-1} - d_base^t), clipped to [-0.2, 0.2]
   
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
        
        # Initialize reward statistics
        self._init_reward_stats()
        
        # Initialize AVP if enabled
        self._init_avp()
    
    def _init_avp(self):
        """Initialize AVP (Asymmetric Value Propagation) components."""
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
        self.avp_lambda = self.avp_lambda_start
        self.avp_gate_distance = DcmmCfg.avp.gate_distance
        self.avp_checkpoint_path = DcmmCfg.avp.checkpoint_path
        self.avp_ready_pose = DcmmCfg.avp.ready_pose
        self.avp_state_dim = DcmmCfg.avp.state_dim
        self.avp_img_size = DcmmCfg.avp.img_size

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
            'count': 0,
            'gated_count': 0,
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
        # 4. TERMINAL REWARDS
        # ========================================
        r_collision = self._compute_collision_penalty()  # -50 for collision
        r_timeout = self._compute_timeout_penalty()  # -20 for timeout
        r_success = self._compute_pregrasp_success_bonus(info)  # +50 for pre-grasp achieved
        
        # ========================================
        # 5. OPTIONAL: AVP (if enabled)
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
            r_orientation, r_collision, r_success, r_touch
        )

        # Debug output
        if self.env.print_reward:
            self._print_reward_breakdown_v2(
                r_ee_progress, r_base_progress, r_alive, r_stagnation,
                r_orientation, r_base_heading, r_touch, r_collision,
                r_plant_collision, r_action_rate, r_avp, r_success,
                r_timeout, info, total_reward
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
        
        r_ee = k_ee * (d_ee^{t-1} - d_ee^t), clipped to [-0.2, 0.2]
        
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
        k_ee = DcmmCfg.reward_weights.get("r_ee_progress", 3.0)
        reward = k_ee * progress
        reward = np.clip(reward, -0.2, 0.2)
        
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
        k_base = DcmmCfg.reward_weights.get("r_base_progress", 2.0)
        reward = k_base * progress
        reward = np.clip(reward, -0.2, 0.2)
        
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
        # Check EE distance
        if info["ee_distance"] >= 0.05:
            return 0.0
        
        # Check base distance (should be in optimal window)
        if not (0.7 < info["base_distance"] < 0.9):
            return 0.0
        
        # Check orientation (cos > 0.966 ≈ 15°)
        ee_pos = self.env.Dcmm.data.body("link6").xpos
        obj_pos = self.env.Dcmm.data.body(self.env.object_name).xpos
        ee_to_obj = obj_pos - ee_pos
        ee_to_obj_norm = ee_to_obj / (np.linalg.norm(ee_to_obj) + 1e-6)
        ee_quat = self.env.Dcmm.data.body("link6").xquat
        palm_forward = quat_rotate_vector(ee_quat, np.array([0, 0, -1]))
        cos_theta = np.dot(palm_forward, ee_to_obj_norm)
        if cos_theta < 0.966:  # ~15 degrees
            return 0.0
        
        # Check EE velocity (should be low)
        ee_vel = self.env.Dcmm.data.body("link6").cvel[3:6]
        ee_speed = np.linalg.norm(ee_vel)
        if ee_speed >= 0.05:
            return 0.0
        
        # All conditions met - pre-grasp success!
        return DcmmCfg.reward_weights.get("r_pregrasp_success", 50.0)
    
    def _compute_timeout_penalty(self):
        """
        Compute timeout penalty when episode ends due to time limit.
        
        Returns:
            float: Timeout penalty (-20) if timed out, else 0
        """
        # Check if this is a timeout termination
        env_time = self.env.Dcmm.data.time - self.env.start_time
        if env_time >= self.env.env_time and not self.env.step_touch:
            return DcmmCfg.reward_weights.get("r_timeout", -20.0)
        return 0.0
    
    def reset_progress_tracking(self):
        """Reset progress tracking variables for new episode."""
        self.prev_ee_distance = None
        self.prev_base_distance = None
        self.stagnation_counter = 0

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
        
        Encourages the robot base to orient toward the target object,
        which helps with reaching and manipulation.
        
        Args:
            info: Environment info containing distances
            
        Returns:
            float: Reward value (0.5 when perfectly aligned)
        """
        base_pos = self.env.Dcmm.data.body("base_link").xpos[:2]
        obj_pos = self.env.Dcmm.data.body(self.env.object_name).xpos[:2]
        
        # Target direction
        to_target = obj_pos - base_pos
        target_heading = np.arctan2(to_target[1], to_target[0])
        
        # Current base heading (yaw from quaternion)
        base_quat = self.env.Dcmm.data.body("base_link").xquat
        base_heading = quat_to_euler(base_quat)[2]  # yaw
        
        heading_error = np.abs(angle_diff(base_heading, target_heading))
        
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
    # ========================================
    
    def compute_avp_reward(self, obs, info):
        """
        Compute AVP reward using Stage 2 Critic.

        Args:
            obs: Current observation dict
            info: Environment info dict

        Returns:
            float: AVP reward (0 if disabled or gated)
        """
        if not self.use_avp or self.grasp_critic is None:
            return 0.0
        
        # Update lambda based on curriculum
        if hasattr(self.env, 'curriculum_difficulty'):
            difficulty = self.env.curriculum_difficulty
            self.avp_lambda = self.avp_lambda_start + \
                (self.avp_lambda_end - self.avp_lambda_start) * difficulty
        
        # Distance gating
        if info["ee_distance"] > self.avp_gate_distance:
            if self.avp_stats is not None:
                self.avp_stats['gated_count'] += 1
            return 0.0
        
        try:
            input_dict = self._construct_virtual_obs(obs)
            
            with torch.no_grad():
                res = self.grasp_critic.act(input_dict)
                value_est = res['values'].item()
            
            avp_reward = np.clip(self.avp_lambda * value_est, -5.0, 5.0)
            
            if self.avp_stats is not None:
                self.avp_stats['reward_sum'] += avp_reward
                self.avp_stats['critic_value_sum'] += value_est
                self.avp_stats['count'] += 1
            
            return avp_reward
            
        except Exception as e:
            if self.env.print_reward:
                print(f">>> AVP Error: {e}")
            return 0.0
    
    def _construct_virtual_obs(self, obs):
        """
        Construct virtual observation for Stage 2 Critic.
        
        Uses virtual arm pose but real object position and depth.
        
        Args:
            obs: Current observation
            
        Returns:
            dict: Input dict for Stage 2 Critic
        """
        # Virtual state components
        virtual_ee_pos = np.array([0.3, 0.0, 0.2], dtype=np.float32)
        virtual_ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        virtual_ee_vel = np.zeros(3, dtype=np.float32)
        virtual_arm_joints = self.avp_ready_pose.astype(np.float32)
        virtual_hand_joints = self.env.hand_open_angles.astype(np.float32)
        virtual_touch = np.zeros(4, dtype=np.float32)
        
        # Real object position
        real_obj_pos = self.env.obs_manager.get_relative_object_pos3d().astype(np.float32)
        
        # Concatenate state (35 dim)
        state_vec = np.concatenate([
            virtual_ee_pos, virtual_ee_quat, virtual_ee_vel,
            virtual_arm_joints, real_obj_pos, virtual_hand_joints, virtual_touch
        ])
        
        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        state_normalized = self.running_mean_std(state_tensor)
        
        # Real depth image
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
        return {'obs': obs_combined}

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
        }
    
    def _update_reward_stats_v2(self, info, r_ee_progress, r_base_progress,
                                r_alive, r_stagnation, r_orientation,
                                r_collision, r_success, r_touch):
        """Update running statistics for new reward structure logging."""
        self.reward_stats['ee_progress_sum'] += r_ee_progress
        self.reward_stats['base_progress_sum'] += r_base_progress
        self.reward_stats['alive_penalty_sum'] += r_alive
        self.reward_stats['stagnation_penalty_sum'] += r_stagnation
        self.reward_stats['orientation_sum'] += r_orientation
        self.reward_stats['collision_sum'] += r_collision
        self.reward_stats['success_sum'] += r_success
        self.reward_stats['touch_sum'] += r_touch
        self.reward_stats['ee_distance_sum'] += info["ee_distance"]
        self.reward_stats['base_distance_sum'] += info["base_distance"]
        self.reward_stats['count'] += 1
    
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
        
        self._init_reward_stats()
        return stats
    
    def get_avp_stats_and_reset(self):
        """
        Get AVP statistics for WandB logging and reset counters.

        Returns:
            dict: AVP statistics, or None if disabled/no data
        """
        if self.avp_stats is None or self.avp_stats['count'] == 0:
            return None
        
        total = self.avp_stats['count'] + self.avp_stats['gated_count']
        stats = {
            'avp/reward_mean': self.avp_stats['reward_sum'] / self.avp_stats['count'],
            'avp/critic_value_mean': self.avp_stats['critic_value_sum'] / self.avp_stats['count'],
            'avp/lambda': self.avp_lambda,
            'avp/gate_ratio': self.avp_stats['gated_count'] / total if total > 0 else 0,
            'avp/count': self.avp_stats['count'],
        }
        
        self.avp_stats = {
            'reward_sum': 0.0,
            'critic_value_sum': 0.0,
            'count': 0,
            'gated_count': 0,
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
                                   r_success, r_timeout, info, total):
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
