"""
Reward Manager Stage 2 - Refactored Version (2025-01-04)

Major improvements based on issue requirements:
1. Progress-based distance rewards (replace milestones)
2. Improved grasp quality: multi-finger balanced contact, force range rewards
3. Slip penalty based on relative velocity
4. Impact penalty for high-speed contacts  
5. Larger success/failure reward gap (+100/-50)
6. Perturbation curriculum learning

Key Design Principles:
- Agent must learn "multi-finger stable grasp + low impact + perturbation resistance"
- NOT "hit once / clamp once" to farm points
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
        
        # [NEW 2025-01-04] Progress tracking
        self.prev_ee_distance = None
        
        # [NEW 2025-01-04] First contact tracking for impact penalty
        self.first_contact_occurred = False
        
        # [NEW 2025-01-04] Recent success rate tracking for perturbation curriculum
        self.success_history = []
        self.max_history_length = 100

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
        
        [UPDATED 2025-01-04] Three-phase curriculum:
        - Phase 0 (0-2M): Disabled (learn basic grasp first)
        - Phase 1 (2M-8M): 10% -> 30% gradual ramp-up
        - Phase 2 (8M+): Only if success_rate > 60%
        """
        step = self.env.global_step
        
        # Phase 0: No perturbation
        phase0_end = DcmmCfg.curriculum.stage2_phase0_steps  # 2M
        if step < phase0_end:
            return False
        
        # Phase 1: Gradual ramp-up (duration is stage2_phase1_steps, so end = phase0_end + duration)
        phase1_end = phase0_end + DcmmCfg.curriculum.stage2_phase1_steps  # 2M + 6M = 8M
        if step < phase1_end:
            # Linear ramp from 10% to 30%
            phase1_duration = phase1_end - phase0_end
            prob = 0.1 + 0.2 * ((step - phase0_end) / phase1_duration)
            return np.random.random() < prob
        
        # Phase 2: Only apply if success rate is high enough
        success_threshold = DcmmCfg.curriculum.stage2_phase2_perturbation_start_success
        try:
            success_rate = self.env.get_recent_success_rate()
            if success_rate < success_threshold:
                return False
        except (AttributeError, TypeError) as e:
            # Method doesn't exist or returned invalid type - skip success check
            pass
        
        # Phase 2: 50% perturbation probability
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
    # [REFACTORED 2025-01-04] Main Reward Function
    # ========================================

    def compute_reward(self, obs, info, ctrl):
        """
        Refactored Stage 2 reward based on issue requirements:
        
        1. Progress-based distance rewards (replace milestones)
        2. Improved grasp quality: multi-finger balanced, force range
        3. Slip penalty for unstable grasp
        4. Impact penalty for high-speed contacts
        5. Larger success/failure gap (+100/-50)
        6. Perturbation curriculum learning
        """
        ee_dist = info["ee_distance"]
        touch_sensors = obs['touch']
        total_contact_force = np.sum(touch_sensors)

        # ========================================
        # 1. PROGRESS-BASED DISTANCE REWARD (replace milestones)
        # ========================================
        reward_ee_progress = self._compute_ee_progress_reward(info)
        
        # Keep only ONE milestone (d_ee < 0.05) for transition signal
        reward_milestone = 2.0 if ee_dist < 0.05 else 0.0

        # ========================================
        # 2. ORIENTATION (only when close, cosine-based)
        # ========================================
        reward_orientation = 0.0
        if ee_dist < 0.3:
            ee_pos = self.env.Dcmm.data.body("link6").xpos
            obj_pos = self.env.Dcmm.data.body(self.env.object_name).xpos
            ee_to_obj = obj_pos - ee_pos
            ee_to_obj_norm = ee_to_obj / (np.linalg.norm(ee_to_obj) + 1e-6)
            ee_quat = self.env.Dcmm.data.body("link6").xquat
            palm_forward = quat_rotate_vector(ee_quat, np.array([0, 0, -1]))
            alignment = np.dot(palm_forward, ee_to_obj_norm)
            reward_orientation = max(0, alignment) * 0.5

        # ========================================
        # 3. IMPROVED GRASP QUALITY REWARD
        # ========================================
        reward_grasp_quality = self._compute_grasp_quality_reward(touch_sensors)
        
        # ========================================
        # 4. SLIP PENALTY (key for stable grasp)
        # ========================================
        reward_slip = self._compute_slip_penalty(touch_sensors)
        
        # ========================================
        # 5. IMPACT PENALTY (prevent high-speed contact)
        # ========================================
        reward_impact = self._compute_impact_penalty(touch_sensors)

        # ========================================
        # 6. FINGER SYNERGY (balanced multi-finger contact)
        # ========================================
        reward_synergy = self._compute_finger_synergy_reward(touch_sensors)

        # ========================================
        # 7. PROGRESSIVE GRASP HOLDING
        # ========================================
        reward_grasp_hold = self._compute_grasp_hold_reward(touch_sensors)

        # ========================================
        # 8. PERTURBATION (with curriculum)
        # ========================================
        reward_perturbation = self.evaluate_grasp_stability(total_contact_force)

        # ========================================
        # 9. REGULARIZATION PENALTIES
        # ========================================
        reward_regularization = self._compute_regularization_penalty(ctrl)
        reward_action_rate = self._compute_action_rate_penalty(ctrl)

        # ========================================
        # 10. COLLISION PENALTIES
        # ========================================
        reward_collision = 0.0
        if self.env.terminated and not info.get('is_success', False):
            reward_collision = DcmmCfg.reward_weights.get("r_stage2_collision", -50.0)

        reward_plant_collision = 0.0
        if self.env.contacts['plant_contacts'].size != 0:
            reward_plant_collision += self.env.current_w_stem
        if self.env.contacts['leaf_contacts'].size != 0:
            reward_plant_collision += -0.05

        # ========================================
        # 11. SUCCESS/FAILURE REWARDS (large gap)
        # ========================================
        reward_success = 0.0
        reward_failure = 0.0
        if info.get('is_success', False):
            reward_success = DcmmCfg.reward_weights.get("r_stage2_success", 100.0)
        elif self.env.terminated:
            reward_failure = DcmmCfg.reward_weights.get("r_stage2_failure", -50.0)

        # ========================================
        # 12. ALIVE PENALTY (prevent stalling)
        # ========================================
        reward_alive = DcmmCfg.reward_weights.get("r_stage2_alive", -0.01)

        # ========================================
        # TOTAL REWARD
        # ========================================
        total = (
            # Progress
            reward_ee_progress + reward_milestone +
            # Conditional
            reward_orientation +
            # Grasp quality
            reward_grasp_quality + reward_synergy + reward_grasp_hold +
            # Penalties
            reward_slip + reward_impact + reward_regularization + reward_action_rate +
            reward_collision + reward_plant_collision + reward_alive +
            # Perturbation
            reward_perturbation +
            # Terminal
            reward_success + reward_failure
        )

        # ========================================
        # LOGGING
        # ========================================
        if not hasattr(self, 'reward_stats'):
            self._init_reward_stats()
        self.reward_stats['ee_progress_sum'] += reward_ee_progress
        self.reward_stats['grasp_quality_sum'] += reward_grasp_quality
        self.reward_stats['synergy_sum'] += reward_synergy
        self.reward_stats['grasp_hold_sum'] += reward_grasp_hold
        self.reward_stats['slip_sum'] += reward_slip
        self.reward_stats['impact_sum'] += reward_impact
        self.reward_stats['perturbation_sum'] += reward_perturbation
        self.reward_stats['collision_sum'] += reward_collision
        self.reward_stats['success_sum'] += reward_success
        self.reward_stats['count'] += 1

        if self.env.print_reward:
            print(f"[Stage2] ee_prog={reward_ee_progress:.3f}, grasp={reward_grasp_quality:.3f}, "
                  f"synergy={reward_synergy:.3f}, slip={reward_slip:.3f}, impact={reward_impact:.3f}, "
                  f"perturb={reward_perturbation:.3f}, success={reward_success:.3f}, total={total:.3f}")

        return total

    # ========================================
    # NEW Reward Components (2025-01-04)
    # ========================================
    
    def _compute_ee_progress_reward(self, info):
        """
        Progress-based EE distance reward.
        
        Formula: r_ee = k * (d_ee_prev - d_ee_curr), clipped to [-0.2, 0.2]
        
        Where:
        - d_ee_prev: EE-to-target distance at previous timestep
        - d_ee_curr: EE-to-target distance at current timestep
        - Positive reward when getting closer (d_ee_prev > d_ee_curr)
        """
        current_dist = info["ee_distance"]
        
        if self.prev_ee_distance is None:
            self.prev_ee_distance = current_dist
            return 0.0
        
        progress = self.prev_ee_distance - current_dist
        k = DcmmCfg.reward_weights.get("r_stage2_ee_progress", 3.0)
        reward = np.clip(k * progress, -0.2, 0.2)
        
        self.prev_ee_distance = current_dist
        return reward
    
    def _compute_grasp_quality_reward(self, touch_sensors):
        """
        Improved grasp quality reward:
        1. Multi-finger contact count reward
        2. Force in optimal range [f_low, f_high] reward
        3. Force balance (penalize variance) reward
        
        Args:
            touch_sensors: Array of 4 touch values [thumb, index, middle, ring]
        """
        # Force thresholds (in Newtons).
        # Try to adapt thresholds to the simulated touch sensor configuration
        # if the environment exposes a maximum touch force; otherwise, fall back
        # to conservative defaults suitable for relatively sensitive sensors.
        sensor_force_max = getattr(self.env, "touch_sensor_force_max_n", None)
        if isinstance(sensor_force_max, (int, float)) and sensor_force_max > 0:
            # Thresholds expressed as fractions of sensor max force.
            f_min = 0.02 * sensor_force_max   # Minimum effective contact (~2% of max)
            f_low = 0.05 * sensor_force_max   # Lower bound of optimal range (~5% of max)
            f_high = 0.5 * sensor_force_max   # Upper bound of optimal range (~50% of max)
        else:
            # Fallback defaults; tune as needed to match actual sensor scaling.
            f_min = 0.02    # Minimum force for "effective contact" (N)
            f_low = 0.05    # Lower bound of optimal range (N)
            f_high = 1.0    # Upper bound of optimal range (N)
        f_mid = (f_low + f_high) / 2
        f_band = (f_high - f_low) / 2
        
        n_finger = len(touch_sensors)
        
        # Count effective contacts
        active_mask = touch_sensors > f_min
        n_contact = np.sum(active_mask)
        
        # Grasp count weight (used for both single- and multi-finger cases)
        k_cnt = DcmmCfg.reward_weights.get("r_grasp_count", 1.5)
        
        # Provide a small positive reward for establishing initial single-finger contact,
        # while keeping zero reward when there is no contact at all.
        if n_contact < 2:
            if n_contact == 1:
                # Shaping reward for single-finger contact progress toward a full grasp
                return 0.3 * k_cnt / n_finger
            return 0.0
        
        # 1. Multi-finger count reward: k_cnt * (n_contact / n_finger)
        r_cnt = k_cnt * (n_contact / n_finger)
        
        # 2. Force range reward: encourage forces in [f_low, f_high]
        k_f = DcmmCfg.reward_weights.get("r_grasp_force_range", 1.0)
        force_scores = []
        for f in touch_sensors:
            if f > f_min:
                # Score is 1.0 when f = f_mid, decays as f moves away
                score = max(0, 1 - abs(f - f_mid) / f_band)
                force_scores.append(score)
        r_force = k_f * np.mean(force_scores) if force_scores else 0.0
        
        # 3. Force balance reward: penalize variance
        k_b = DcmmCfg.reward_weights.get("r_grasp_balance", 0.5)
        active_forces = touch_sensors[active_mask]
        if len(active_forces) >= 2:
            variance = np.var(active_forces)
            r_bal = -k_b * variance
        else:
            r_bal = 0.0
        
        return r_cnt + r_force + r_bal
    
    def _compute_slip_penalty(self, touch_sensors=None):
        """
        Slip penalty: penalize relative velocity between object and hand.
        r_slip = -k_s * |v_rel|
        
        [UPDATED 2025-01-04] Curriculum-based slip penalty weight:
        - Phase 0 (0-2M): k_s = 0.2 (weak, focus on basic grasp)
        - Phase 1 (2M-8M): k_s ramps from 0.2 to 1.0
        - Phase 2 (8M+): k_s = 1.0 (full penalty)
        
        Args:
            touch_sensors: Optional touch sensor readings to verify contact
        """
        # Get curriculum-adjusted slip weight
        step = self.env.global_step
        phase0_end = DcmmCfg.curriculum.stage2_phase0_steps  # 2M
        # phase1_end = phase0_end + duration of phase 1
        phase1_end = phase0_end + DcmmCfg.curriculum.stage2_phase1_steps  # 2M + 6M = 8M
        
        if step < phase0_end:
            # Phase 0: Weak slip penalty
            k_s = DcmmCfg.curriculum.stage2_phase1_slip_weight_start
        elif step < phase1_end:
            # Phase 1: Ramp up slip penalty
            phase1_duration = phase1_end - phase0_end
            progress = (step - phase0_end) / phase1_duration
            slip_start = DcmmCfg.curriculum.stage2_phase1_slip_weight_start
            slip_end = DcmmCfg.curriculum.stage2_phase1_slip_weight_end
            k_s = slip_start + (slip_end - slip_start) * progress
        else:
            # Phase 2: Full slip penalty
            k_s = DcmmCfg.reward_weights.get("r_slip_penalty", 1.0)
        
        # Check for meaningful contact using force threshold
        min_contact_force = 0.02  # Minimum force for meaningful contact
        has_contact = False
        
        if touch_sensors is not None:
            total_force = np.sum(touch_sensors)
            has_contact = total_force > min_contact_force
        elif hasattr(self.env, 'step_touch'):
            has_contact = self.env.step_touch
        
        # Only apply penalty if there's meaningful contact
        if not has_contact:
            return 0.0
        
        # Get object velocity
        obj_vel = self.env.Dcmm.data.body(self.env.object_name).cvel[3:6]
        
        # Get EE velocity
        ee_vel = self.env.Dcmm.data.body("link6").cvel[3:6]
        
        # Relative velocity
        v_rel = np.linalg.norm(obj_vel - ee_vel)
        
        return -k_s * v_rel
    
    def _compute_impact_penalty(self, touch_sensors):
        """
        Impact penalty: penalize high-speed first contact.
        - If first contact with |v_ee| > v_thr, give large penalty
        - Also continuous penalty proportional to speed
        """
        total_force = np.sum(touch_sensors)
        
        # Check if this is first contact
        if total_force > 0.01 and not self.first_contact_occurred:
            self.first_contact_occurred = True
            
            # Check EE velocity at first contact
            ee_vel = np.linalg.norm(self.env.Dcmm.data.body("link6").cvel[3:6])
            v_thr = DcmmCfg.reward_weights.get("r_impact_vel_threshold", 0.3)
            
            if ee_vel > v_thr:
                # Large one-time penalty for high-speed first contact
                penalty = DcmmCfg.reward_weights.get("r_impact_first_contact", -5.0)
                return penalty
        
        # Continuous penalty for high-speed contact
        if total_force > 0.01 or (hasattr(self.env, 'step_touch') and self.env.step_touch):
            ee_vel = np.linalg.norm(self.env.Dcmm.data.body("link6").cvel[3:6])
            if ee_vel > 0.3:
                k_impact = DcmmCfg.reward_weights.get("r_impact_continuous", 2.0)
                return -min(k_impact * (ee_vel - 0.3), 3.0)
        
        return 0.0
    
    def _compute_action_rate_penalty(self, ctrl):
        """Action rate penalty (arm + hand, no base)."""
        current_action = np.concatenate([ctrl['arm'], ctrl['hand']])
        
        if self.prev_action_reward is None:
            self.prev_action_reward = np.zeros_like(current_action)
        
        action_diff = current_action - self.prev_action_reward
        penalty = -np.linalg.norm(action_diff) * 0.02
        
        self.prev_action_reward = current_action.copy()
        return penalty
    
    def reset_progress_tracking(self):
        """Reset progress tracking for new episode."""
        self.prev_ee_distance = None
        self.first_contact_occurred = False
        self.grasp_start_time = None
        self.perturbation_active = False
        self.initial_grasp_pos = None

    # [UPDATE] Stats tracking
    def _init_reward_stats(self):
        self.reward_stats = {
            'ee_progress_sum': 0.0,
            'grasp_quality_sum': 0.0,
            'synergy_sum': 0.0,
            'grasp_hold_sum': 0.0,
            'slip_sum': 0.0,
            'impact_sum': 0.0,
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
            'rewards/ee_progress_mean': self.reward_stats['ee_progress_sum'] / count,
            'rewards/grasp_quality_mean': self.reward_stats['grasp_quality_sum'] / count,
            'rewards/synergy_mean': self.reward_stats['synergy_sum'] / count,
            'rewards/grasp_hold_mean': self.reward_stats['grasp_hold_sum'] / count,
            'rewards/slip_mean': self.reward_stats['slip_sum'] / count,
            'rewards/impact_mean': self.reward_stats['impact_sum'] / count,
            'rewards/perturbation_mean': self.reward_stats['perturbation_sum'] / count,
            'rewards/collision_mean': self.reward_stats['collision_sum'] / count,
            'rewards/success_mean': self.reward_stats['success_sum'] / count,
        }

        # Reset counters
        self._init_reward_stats()

        return stats
