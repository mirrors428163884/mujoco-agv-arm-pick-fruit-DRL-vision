"""
Reward Manager for Stage 1 (Tracking Task).

This module handles all reward computation for the mobile manipulation tracking task.
The agent learns to move the mobile base and arm to touch a target fruit with the palm.

Key Design Decisions (2025-12-19 Fix):
1. Decoupled reaching reward into base and arm components to prevent base-only solutions
2. Added explicit arm motion reward to encourage arm joint movement
3. Reduced control regularization to allow more aggressive arm movements

Reward Components:
- Arm Reaching: EE-to-target distance in base frame (arm contribution only)
- Global Reaching: EE-to-target distance in world frame (base + arm)
- Base Approach: Encourage optimal base distance (~0.8m)
- Arm Motion: Reward arm joint deviation from initial pose
- Arm Action: Reward arm action magnitude
- Orientation: Palm facing target
- Touch: Contact reward with gentle impact bonus
- Regularization: Control smoothness penalty
- Collision: Termination penalty for crashes
- Plant Collision: Stem and leaf collision penalties
- Action Rate: Smooth action changes
- AVP: Asymmetric Value Propagation from Stage 2 Critic (optional)
"""

import numpy as np
import torch
import os
import configs.env.DcmmCfg as DcmmCfg
from gym_dcmm.utils.quat_utils import quat_rotate_vector

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
            print(f">>> AVP Warning: Stage 2 Checkpoint not found at {ckpt_path}")
            print(">>> AVP will be disabled.")
            self.grasp_critic = None
            self.running_mean_std = None
            return
        
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

        Args:
            obs: Current observation dict
            info: Environment info dict containing distances
            ctrl: Control action dict with 'base', 'arm', 'hand' keys

        Returns:
            float: Total reward value
        """
        # Compute individual reward components
        r_arm_reaching, arm_reach_dist = self._compute_arm_reaching_reward()
        r_reaching = self._compute_global_reaching_reward(info)
        r_base_approach = self._compute_base_approach_reward(info)
        r_arm_motion, arm_joint_dev = self._compute_arm_motion_reward()
        r_arm_action = self._compute_arm_action_reward(ctrl)
        r_orientation = self._compute_orientation_reward(info)
        r_touch, r_impact = self._compute_touch_reward()
        r_regularization = self._compute_regularization_penalty(ctrl)
        r_collision = self._compute_collision_penalty()
        r_plant_collision = self._compute_plant_collision_penalty()
        r_action_rate = self._compute_action_rate_penalty(ctrl)
        r_avp = self.compute_avp_reward(obs, info)

        # Sum all rewards
        total_reward = (
            r_arm_reaching + r_reaching + r_base_approach +
            r_arm_motion + r_arm_action + r_orientation +
            r_touch + r_regularization + r_collision +
            r_plant_collision + r_action_rate + r_avp
        )

        # Update statistics
        self._update_reward_stats(
            r_reaching, r_base_approach, r_orientation, r_touch,
            r_collision, r_plant_collision, r_arm_reaching, r_arm_motion,
            arm_reach_dist, info
        )

        # Debug output
        if self.env.print_reward:
            self._print_reward_breakdown(
                r_reaching, r_arm_reaching, r_arm_motion, r_arm_action,
                r_base_approach, r_orientation, r_touch, r_impact,
                r_regularization, r_collision, r_plant_collision,
                r_action_rate, r_avp, arm_reach_dist, arm_joint_dev, total_reward
            )

        return total_reward

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
            add_noise=False,
            add_holes=False
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
            'reaching_sum': 0.0,
            'base_approach_sum': 0.0,
            'orientation_sum': 0.0,
            'touch_sum': 0.0,
            'collision_sum': 0.0,
            'plant_collision_sum': 0.0,
            'ee_distance_sum': 0.0,
            'base_distance_sum': 0.0,
            'arm_reaching_sum': 0.0,
            'arm_motion_sum': 0.0,
            'arm_reach_distance_sum': 0.0,
            'count': 0,
        }
    
    def _update_reward_stats(self, r_reaching, r_base_approach, r_orientation,
                             r_touch, r_collision, r_plant_collision,
                             r_arm_reaching, r_arm_motion, arm_reach_dist, info):
        """Update running statistics for logging."""
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
            'rewards/reaching_mean': self.reward_stats['reaching_sum'] / count,
            'rewards/base_approach_mean': self.reward_stats['base_approach_sum'] / count,
            'rewards/orientation_mean': self.reward_stats['orientation_sum'] / count,
            'rewards/touch_mean': self.reward_stats['touch_sum'] / count,
            'rewards/collision_mean': self.reward_stats['collision_sum'] / count,
            'rewards/plant_collision_mean': self.reward_stats['plant_collision_sum'] / count,
            'rewards/arm_reaching_mean': self.reward_stats['arm_reaching_sum'] / count,
            'rewards/arm_motion_mean': self.reward_stats['arm_motion_sum'] / count,
            'distance/ee_distance_mean': self.reward_stats['ee_distance_sum'] / count,
            'distance/base_distance_mean': self.reward_stats['base_distance_sum'] / count,
            'distance/arm_reach_distance_mean': self.reward_stats['arm_reach_distance_sum'] / count,
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
                                arm_reach_dist, arm_joint_dev, total):
        """Print detailed reward breakdown for debugging."""
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
        print(f"arm_reach_distance: {arm_reach_dist:.3f}")
        print(f"arm_joint_deviation: {arm_joint_dev:.3f}")
        print(f"total reward: {total:.3f}\n")

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
