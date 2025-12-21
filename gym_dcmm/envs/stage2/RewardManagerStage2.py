"""
Reward Manager Stage 2 - Fixed & Improved Version
Fixed by Gemini:
1. Added missing perturbation methods and logging interface.
2. Corrected observation indices for hand joints and touch sensors.
"""

import numpy as np
import configs.env.DcmmCfg as DcmmCfg
from gym_dcmm.utils.quat_utils import quat_rotate_vector


class RewardManagerStage2:
    def __init__(self, env):
        self.env = env
        self.prev_action_reward = None

        # Perturbation test state
        self.perturbation_active = False
        self.initial_grasp_pos = None
        self.perturbation_timer = 0.0
        self.perturbation_force_mag = 0.0
        self.perturbation_direction = np.zeros(3)

        # [NEW] Grasp holding timer for progressive reward
        self.grasp_start_time = None

    # ========================================
    # [RESTORED] Missing Helper Methods
    # ========================================

    def apply_perturbation_force(self):
        """
        Apply random external force to the object to test grasp stability.
        (Restored from old version)
        """
        # Generate random force direction (uniformly on sphere)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        self.perturbation_direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])

        # Random force magnitude
        self.perturbation_force_mag = np.random.uniform(0.5, 1.5)

        # Apply force to object
        object_body_id = self.env.Dcmm.data.body(self.env.object_name).id
        force_vector = self.perturbation_direction * self.perturbation_force_mag
        self.env.Dcmm.data.xfrc_applied[object_body_id, :3] = force_vector

    def compute_slippage(self):
        """
        Measure object displacement from initial grasp position.
        (Restored from old version)
        """
        current_obj_pos = self.env.Dcmm.data.body(self.env.object_name).xpos
        if self.initial_grasp_pos is None:
            return 0.0
        slippage = np.linalg.norm(current_obj_pos - self.initial_grasp_pos)
        return slippage

    # ========================================
    # [NEW] Finger Synergy Reward
    # ========================================

    def _compute_finger_synergy_reward(self, touch_sensors):
        """
        Reward for balanced multi-finger contact (force closure principle).
        Encourages all fingers to share load evenly.

        Args:
            touch_sensors: Array of 4 touch values [thumb, index, middle, ring]
                           (Order based on ObservationManager.get_state_obs_stage2)
        """
        active_threshold = 0.1
        active_fingers = touch_sensors > active_threshold
        num_active = np.count_nonzero(active_fingers)

        if num_active < 2:
            return 0.0

        # 1. Finger count reward (more fingers = better)
        finger_count_reward = 0.5 * (num_active / 4.0)

        # 2. Force balance reward (penalize uneven distribution)
        # Low standard deviation means forces are balanced
        active_forces = touch_sensors[active_fingers]
        force_std = np.std(active_forces)
        balance_reward = 1.0 * np.exp(-3.0 * force_std)

        # 3. Opposition reward (thumb vs others)
        # ObservationManager defines order as: [thumb, index, middle, ring]
        thumb_active = touch_sensors[0] > active_threshold
        others_active = np.any(touch_sensors[1:] > active_threshold)

        opposition_reward = 1.0 if (thumb_active and others_active) else 0.0

        return finger_count_reward + balance_reward + opposition_reward

    # ========================================
    # [NEW] Progressive Grasp Holding Reward
    # ========================================

    def _compute_grasp_hold_reward(self, touch_sensors):
        """
        Progressive reward for maintaining stable grasp (Catching-inspired).
        Provides dense signal instead of sparse +20 success bonus.
        """
        min_force = 0.1
        max_force = 2.0
        stable_fingers = sum(1 for f in touch_sensors if min_force <= f <= max_force)

        if stable_fingers >= 2:
            # Initialize grasp timer
            if self.grasp_start_time is None:
                self.grasp_start_time = self.env.Dcmm.data.time

            # Time-based progressive reward (5 points per second)
            hold_time = self.env.Dcmm.data.time - self.grasp_start_time
            time_reward = 5.0 * min(hold_time / 1.0, 1.0)

            # Bonus for extra fingers
            finger_bonus = 0.5 * max(0, stable_fingers - 2)

            return time_reward + finger_bonus
        else:
            # Lost grasp, reset timer
            self.grasp_start_time = None
            return 0.0

    # ========================================
    # [IMPROVED] Perturbation Test with Curriculum
    # ========================================

    def _should_apply_perturbation(self):
        """
        Curriculum-based perturbation scheduling:
        - 0-2M:  Disabled (learn basic grasp first)
        - 2M-10M: 10% -> 50% gradual ramp-up
        - 10M+: 50% episodes (full robustness test)
        """
        step = self.env.global_step

        if step < 2e6:
            return False
        elif step < 10e6:
            # Linear ramp from 10% to 50%
            prob = 0.1 + 0.4 * ((step - 2e6) / 8e6)
            return np.random.random() < prob
        else:
            return np.random.random() < 0.5

    def evaluate_grasp_stability(self, total_contact_force):
        """Modified with curriculum."""
        if not self._should_apply_perturbation():
            return 0.0

        reward_perturbation = 0.0
        dt = self.env.Dcmm.model.opt.timestep * self.env.steps_per_policy

        if not self.perturbation_active:
            if total_contact_force >= 1.0:
                self.perturbation_active = True
                self.initial_grasp_pos = self.env.Dcmm.data.body(self.env.object_name).xpos.copy()
                self.perturbation_timer = 0.0
                self.apply_perturbation_force()
        else:
            self.perturbation_timer += dt
            if self.perturbation_timer < 0.5:
                # Refresh force
                object_body_id = self.env.Dcmm.data.body(self.env.object_name).id
                force_vector = self.perturbation_direction * self.perturbation_force_mag
                self.env.Dcmm.data.xfrc_applied[object_body_id, :3] = force_vector
            else:
                # Evaluate slippage
                slippage = self.compute_slippage()
                if slippage < 0.01:
                    reward_perturbation = 10.0
                else:
                    reward_perturbation = -5.0

                # Reset
                self.perturbation_active = False
                self.initial_grasp_pos = None
                object_body_id = self.env.Dcmm.data.body(self.env.object_name).id
                self.env.Dcmm.data.xfrc_applied[object_body_id, :] = 0.0

        return reward_perturbation

    # ========================================
    # [FIX] Regularization (Remove Base)
    # ========================================

    def _compute_regularization_penalty(self, ctrl):
        """
        Stage 2: Only penalize arm + hand (base is locked).
        """
        arm_penalty = -np.linalg.norm(ctrl['arm']) * 0.002
        hand_penalty = -np.linalg.norm(ctrl['hand']) * 0.001
        return arm_penalty + hand_penalty

    # ========================================
    # [NEW] Grasp Intent Reward (Early Training)
    # ========================================

    def _compute_grasp_intent_reward(self, obs, info):
        """
        Encourage 'reaching while closing hand' in early training.
        Prevents arm-only or hand-only failures.
        """
        if info["ee_distance"] > 0.5:
            return 0.0

        # 1. Reaching component
        reach_reward = 2.0 * np.exp(-2.0 * info["ee_distance"])

        # 2. Hand closure component (only when close)
        if info["ee_distance"] < 0.2:
            # FIX: Index correction based on DcmmVecEnvStage2 obs_state definition:
            # ee_pos(3) + ee_quat(4) + ee_vel(3) + arm_joints(6) + obj_pos(3) = 19
            # Hand joints are obs['state'][19:31] (12 DOF)
            hand_joints = obs['state'][19:31]
            initial_open = self.env.hand_open_angles

            # Measure closure progress
            finger_closure = np.mean(np.abs(hand_joints - initial_open))
            closure_reward = 1.0 * np.tanh(5.0 * finger_closure)

            return reach_reward + closure_reward
        else:
            return reach_reward

    # ========================================
    # [IMPROVED] Main Reward Function
    # ========================================

    def compute_reward(self, obs, info, ctrl):
        """
        Improved Stage 2 reward with:
        - Finger synergy
        - Progressive grasp holding
        - Curriculum perturbation
        - Fixed regularization (no base)
        - Grasp intent guidance
        """
        ee_dist = info["ee_distance"]

        # 1. EE Reaching (unchanged)
        reward_reaching = 2.0 * np.exp(-2.0 * ee_dist)

        # 2. Distance milestones (unchanged)
        reward_distance_shaping = 0.0
        if ee_dist < 0.30:  reward_distance_shaping += 0.5
        if ee_dist < 0.15: reward_distance_shaping += 1.0
        if ee_dist < 0.08: reward_distance_shaping += 1.5
        if ee_dist < 0.05: reward_distance_shaping += 2.0
        if ee_dist < 0.03: reward_distance_shaping += 2.5  # [NEW] Ultra-close bonus

        # 3. Orientation (simplified, Catching-style)
        reward_orientation = 0.0
        if ee_dist < 0.3:
            ee_pos = self.env.Dcmm.data.body("link6").xpos
            obj_pos = self.env.Dcmm.data.body(self.env.object_name).xpos
            ee_to_obj = obj_pos - ee_pos
            ee_to_obj_norm = ee_to_obj / (np.linalg.norm(ee_to_obj) + 1e-6)
            ee_quat = self.env.Dcmm.data.body("link6").xquat
            palm_forward = quat_rotate_vector(ee_quat, np.array([0, 0, -1]))
            alignment = np.dot(palm_forward, ee_to_obj_norm)
            reward_orientation = max(0, alignment) * 1.0

        # 4. [NEW] Grasp intent (early guidance)
        reward_grasp_intent = self._compute_grasp_intent_reward(obs, info)

        # 5. [IMPROVED] Grasp reward with synergy
        reward_grasp = 0.0
        total_contact_force = np.sum(obs['touch'])

        if total_contact_force > 0.01:
            fingers_touching = np.count_nonzero(obs['touch'] > 0.05)
            reward_grasp = 1.0 + 0.5 * fingers_touching

            # Force range bonus
            if 0.5 <= total_contact_force <= 3.0:
                reward_grasp += 2.0
            elif total_contact_force > 3.0:
                reward_grasp -= 0.5 * min(total_contact_force - 3.0, 2.0)

        # 6. [NEW] Finger synergy
        reward_synergy = self._compute_finger_synergy_reward(obs['touch'])

        # 7. [NEW] Progressive grasp holding
        reward_grasp_hold = self._compute_grasp_hold_reward(obs['touch'])

        # 8. [IMPROVED] Perturbation with curriculum
        reward_perturbation = self.evaluate_grasp_stability(total_contact_force)

        # 9. Impact penalty (unchanged)
        reward_impact = 0.0
        if total_contact_force > 0.01 or self.env.step_touch:
            ee_vel = np.linalg.norm(self.env.Dcmm.data.body("link6").cvel[3:6])
            if ee_vel > 0.3:
                reward_impact = -min(2.0 * (ee_vel - 0.3), 3.0)

        # 10. [FIX] Regularization (no base)
        reward_regularization = self._compute_regularization_penalty(ctrl)

        # 11. Collision (unchanged)
        reward_collision = 0.0
        if self.env.terminated and not info.get('is_success', False):
            reward_collision = -2.0

        # 12. Plant collision (unchanged)
        reward_plant_collision = 0.0
        if self.env.contacts['plant_contacts'].size != 0:
            reward_plant_collision += self.env.current_w_stem
        if self.env.contacts['leaf_contacts'].size != 0:
            reward_plant_collision += -0.05

        # 13. Action rate (unchanged)
        current_action_vec = np.concatenate([ctrl['arm'], ctrl['hand']])  # [FIX] No base
        if self.prev_action_reward is None:
            self.prev_action_reward = np.zeros_like(current_action_vec)
        action_diff = current_action_vec - self.prev_action_reward
        reward_action_rate = -np.linalg.norm(action_diff) * 0.02
        self.prev_action_reward = current_action_vec.copy()

        # 14. Success reward (reduced, now compensated by progressive rewards)
        reward_success = 0.0
        if info.get('is_success', False):
            reward_success = 15.0  # Reduced from 20 (progressive rewards compensate)

        # Total reward
        total = (
                reward_reaching + reward_distance_shaping + reward_orientation +
                reward_grasp_intent + reward_grasp + reward_synergy +
                reward_grasp_hold + reward_perturbation + reward_impact +
                reward_regularization + reward_collision + reward_plant_collision +
                reward_action_rate + reward_success
        )

        # Logging (update stats)
        if not hasattr(self, 'reward_stats'):
            self._init_reward_stats()
        self.reward_stats['reaching_sum'] += reward_reaching
        self.reward_stats['grasp_sum'] += reward_grasp
        self.reward_stats['synergy_sum'] += reward_synergy  # [NEW]
        self.reward_stats['grasp_hold_sum'] += reward_grasp_hold  # [NEW]
        self.reward_stats['perturbation_sum'] += reward_perturbation
        self.reward_stats['collision_sum'] += reward_collision
        self.reward_stats['success_sum'] += reward_success
        self.reward_stats['count'] += 1

        if self.env.print_reward:
            print(f"reward_reaching: {reward_reaching:.3f}")
            print(f"reward_grasp: {reward_grasp:.3f}")
            print(f"reward_synergy: {reward_synergy:.3f}")  # [NEW]
            print(f"reward_grasp_hold: {reward_grasp_hold:.3f}")  # [NEW]
            print(f"reward_perturbation: {reward_perturbation:.3f}")
            print(f"total:  {total:.3f}\n")

        return total

    # [UPDATE] Stats tracking
    def _init_reward_stats(self):
        self.reward_stats = {
            'reaching_sum': 0.0,
            'grasp_sum': 0.0,
            'synergy_sum': 0.0,  # [NEW]
            'grasp_hold_sum': 0.0,  # [NEW]
            'perturbation_sum': 0.0,
            'collision_sum': 0.0,
            'success_sum': 0.0,
            'count': 0,
        }

    # [RESTORED] Missing Interface Method
    def get_reward_stats_and_reset(self):
        """
        Get reward statistics for WandB logging and reset counters.

        Returns:
            dict: Reward component averages
        """
        if not hasattr(self, 'reward_stats') or self.reward_stats['count'] == 0:
            return None

        count = self.reward_stats['count']
        stats = {
            'rewards/reaching_mean': self.reward_stats['reaching_sum'] / count,
            'rewards/grasp_mean': self.reward_stats['grasp_sum'] / count,
            'rewards/synergy_mean': self.reward_stats['synergy_sum'] / count,
            'rewards/grasp_hold_mean': self.reward_stats['grasp_hold_sum'] / count,
            'rewards/perturbation_mean': self.reward_stats['perturbation_sum'] / count,
            'rewards/collision_mean': self.reward_stats['collision_sum'] / count,
            'rewards/success_mean': self.reward_stats['success_sum'] / count,
        }

        # Reset counters
        self._init_reward_stats()

        return stats
