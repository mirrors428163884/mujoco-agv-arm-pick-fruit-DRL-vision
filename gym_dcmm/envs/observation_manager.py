"""
Observation management for DcmmVecEnv.
Handles all observation collection and processing.
"""

import numpy as np
from gym_dcmm.utils.quat_utils import quat_rotate_vector
import cv2 as cv
import configs.env.DcmmCfg as DcmmCfg


import mujoco

class ObservationManager:
    """Manages observation collection for the environment."""

    def __init__(self, env):
        """
        Initialize observation manager.

        Args:
            env: Reference to the parent DcmmVecEnv instance
        """
        self.env = env
        
        # Object position noise state (for frame drop simulation)
        self._prev_obj_pos = None
        self._consecutive_drop_counter = 0  # Remaining consecutive frames to drop

    def get_base_vel(self):
        """Get the velocity of the mobile base in base frame (2D)."""
        base_pos_world = self.env.Dcmm.data.body("arm_base").xpos[0:2]
        base_quat = self.env.Dcmm.data.body("arm_base").xquat
        base_vel_world = self.env.Dcmm.data.body("arm_base").cvel[3:5]

        # Transform to base frame
        base_vel_world_3d = np.array([base_vel_world[0], base_vel_world[1], 0.0])
        base_quat_inv = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]])
        base_vel_base = quat_rotate_vector(base_quat_inv, base_vel_world_3d)

        return base_vel_base[0:2]

    def get_relative_ee_pos3d(self):
        """Get end-effector position relative to base in base frame."""
        ee_pos_world = self.env.Dcmm.data.body("link6").xpos
        base_pos_world = self.env.Dcmm.data.body("arm_base").xpos
        base_quat = self.env.Dcmm.data.body("arm_base").xquat

        # Vector from base to EE in world frame
        ee_rel_world = ee_pos_world - base_pos_world

        # Transform to base frame
        base_quat_inv = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]])
        ee_rel_base = quat_rotate_vector(base_quat_inv, ee_rel_world)

        return ee_rel_base

    def get_relative_ee_quat(self):
        """Get end-effector orientation relative to base."""
        ee_quat = self.env.Dcmm.data.body("link6").xquat
        return ee_quat

    def get_relative_ee_v_lin_3d(self):
        """Get end-effector linear velocity in base frame."""
        ee_vel_world = self.env.Dcmm.data.body("link6").cvel[3:6]
        base_quat = self.env.Dcmm.data.body("arm_base").xquat

        # Transform to base frame
        base_quat_inv = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]])
        ee_vel_base = quat_rotate_vector(base_quat_inv, ee_vel_world)

        return ee_vel_base

    def get_relative_object_pos3d(self):
        """Get object position relative to base in base frame."""
        obj_pos_world = self.env.Dcmm.data.body(self.env.object_name).xpos
        base_pos_world = self.env.Dcmm.data.body("arm_base").xpos
        base_quat = self.env.Dcmm.data.body("arm_base").xquat

        # Vector from base to object in world frame
        obj_rel_world = obj_pos_world - base_pos_world

        # Transform to base frame
        base_quat_inv = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]])
        obj_rel_base = quat_rotate_vector(base_quat_inv, obj_rel_world)

        return obj_rel_base
    
    def get_relative_object_pos3d_with_noise(self):
        """
        Get object position relative to base in base frame with domain randomization noise.
        
        Applies:
        1. Gaussian noise: Simulates sensor random measurement errors
        2. Frame drop: Simulates YOLO detection failures (returns 0 or previous value)
        3. Consecutive frame drop: Simulates prolonged detection loss
        
        Returns:
            tuple: (noisy_obj_pos, is_valid) where:
                - noisy_obj_pos: Noisy object position in base frame (3D, np.float32)
                - is_valid: 1.0 if observation is valid, 0.0 if dropped (np.float32)
        """
        # Get true object position
        true_obj_pos = self.get_relative_object_pos3d()
        
        # Check if noise is enabled
        if not DcmmCfg.obj_pos_noise.enabled:
            return true_obj_pos.astype(np.float32), np.float32(1.0)
        
        # Initialize previous position if not set
        if self._prev_obj_pos is None:
            self._prev_obj_pos = true_obj_pos.copy()
        
        noisy_obj_pos = true_obj_pos.copy()
        is_valid = np.float32(1.0)  # Default: valid observation
        
        # Check for consecutive drop sequence
        if self._consecutive_drop_counter > 0:
            # Still in consecutive drop mode
            self._consecutive_drop_counter -= 1
            is_valid = np.float32(0.0)  # Mark as invalid
            if DcmmCfg.obj_pos_noise.drop_use_zero:
                noisy_obj_pos = np.zeros(3, dtype=np.float32)
            else:
                noisy_obj_pos = self._prev_obj_pos.copy()
            return noisy_obj_pos.astype(np.float32), is_valid
        
        # Check for frame drop
        frame_dropped = False
        
        # Start consecutive drop sequence with small probability
        if DcmmCfg.obj_pos_noise.consecutive_drop_enabled:
            if np.random.random() < DcmmCfg.obj_pos_noise.consecutive_drop_prob:
                drop_len_range = DcmmCfg.obj_pos_noise.consecutive_drop_length
                # Subtract 1 because this frame is already being dropped
                self._consecutive_drop_counter = np.random.randint(
                    drop_len_range[0], drop_len_range[1] + 1) - 1
                frame_dropped = True
        
        # Single frame drop
        if not frame_dropped and DcmmCfg.obj_pos_noise.frame_drop_enabled:
            if np.random.random() < DcmmCfg.obj_pos_noise.drop_probability:
                frame_dropped = True
        
        if frame_dropped:
            is_valid = np.float32(0.0)  # Mark as invalid
            if DcmmCfg.obj_pos_noise.drop_use_zero:
                noisy_obj_pos = np.zeros(3, dtype=np.float32)
            else:
                noisy_obj_pos = self._prev_obj_pos.copy()
        else:
            # Apply Gaussian noise to valid observations
            if DcmmCfg.obj_pos_noise.gaussian_enabled:
                gaussian_noise = np.random.normal(
                    0, DcmmCfg.obj_pos_noise.gaussian_std, size=3)
                noisy_obj_pos = true_obj_pos + gaussian_noise
            
            # Update previous position with true (noisy but detected) position
            self._prev_obj_pos = noisy_obj_pos.copy()
        
        return noisy_obj_pos.astype(np.float32), is_valid
    
    def reset_noise_state(self):
        """Reset noise state variables. Call this on environment reset."""
        self._prev_obj_pos = None
        self._consecutive_drop_counter = 0

    def get_relative_object_v_lin_3d(self):
        """Get object linear velocity in base frame."""
        obj_vel_world = self.env.Dcmm.data.body(self.env.object_name).cvel[3:6]
        base_quat = self.env.Dcmm.data.body("arm_base").xquat

        # Transform to base frame
        base_quat_inv = np.array([base_quat[0], -base_quat[1], -base_quat[2], -base_quat[3]])
        obj_vel_base = quat_rotate_vector(base_quat_inv, obj_vel_world)

        return obj_vel_base

    def get_obs(self):
        """
        Collect all observations for the current state.

        Returns:
            dict: Observation dictionary with base, arm, object, and depth info
        """
        # Get noisy object position and validity flag
        obj_pos, obj_valid = self.get_relative_object_pos3d_with_noise()
        
        obs = {
            "base": {
                "v_lin_2d": self.get_base_vel().astype(np.float32),
            },
            "arm": {
                "ee_pos3d": self.get_relative_ee_pos3d().astype(np.float32),
                "ee_quat": self.get_relative_ee_quat().astype(np.float32),
                "ee_v_lin_3d": self.get_relative_ee_v_lin_3d().astype(np.float32),
                "joint_pos": self.env.Dcmm.data.qpos[15:21].astype(np.float32),
            },
            "object": {
                "pos3d": obj_pos,
            },
        }
        
        # Add validity flag if configured (helps network distinguish dropped vs valid observations)
        if DcmmCfg.obj_pos_noise.enabled and DcmmCfg.obj_pos_noise.add_validity_flag:
            obs["object"]["is_valid"] = obj_valid

        # Add depth image if render mode is set
        # Use RenderManager.get_depth_obs() for consistent noise processing across stages
        if self.env.render_mode is not None:
            # Get depth with full noise augmentation (Cutout, Salt-Pepper, Edge Blur, Depth-Dependent)
            obs["depth"] = self.env.render_manager.get_depth_obs(
                width=self.env.img_size[1],  # Note: img_size is (height, width)
                height=self.env.img_size[0],
                camera_name=self.env.camera_name[0] if self.env.camera_name else "wrist",
                add_noise=True,
                add_holes=True
            )
        else:
            obs["depth"] = np.zeros((1, self.env.img_size[0], self.env.img_size[1]), dtype=np.uint8)

        if self.env.print_obs:
            print(f"##### Observation #####")
            print(f"base_vel: {obs['base']['v_lin_2d']}")
            print(f"ee_pos3d: {obs['arm']['ee_pos3d']}")
            print(f"ee_quat: {obs['arm']['ee_quat']}")
            print(f"ee_v_lin_3d: {obs['arm']['ee_v_lin_3d']}")
            print(f"joint_pos: {obs['arm']['joint_pos']}")
            print(f"object_pos3d: {obs['object']['pos3d']}")
            if 'is_valid' in obs['object']:
                print(f"object_is_valid: {obs['object']['is_valid']}")
            print(f"depth shape: {obs['depth'].shape}\n")

        return obs

    def get_hand_obs(self):
        """
        Get hand joint positions.

        Returns:
            np.ndarray: Hand joint positions
        """
        import configs.env.DcmmCfg as DcmmCfg
        hand_joint_indices = np.where(DcmmCfg.hand_mask == 1)[0] + 15
        hand_obs = self.env.Dcmm.data.qpos[hand_joint_indices].astype(np.float32)
        return hand_obs

    def get_state_obs_stage2(self):
        """
        Get the flattened state observation for Stage 2 (Catching/Tracking).
        Includes: Arm (Pos, Quat, Vel, Joints), Object (Pos), Hand (Joints), Touch.
        Excludes: Base, Depth.
        
        Returns:
            np.ndarray: Flattened state vector
        """
        # 1. Arm Data
        ee_pos = self.get_relative_ee_pos3d().astype(np.float32)
        ee_quat = self.get_relative_ee_quat().astype(np.float32)
        ee_vel = self.get_relative_ee_v_lin_3d().astype(np.float32)
        arm_joints = self.env.Dcmm.data.qpos[15:21].astype(np.float32)
        
        # 2. Object Data
        obj_pos = self.get_relative_object_pos3d().astype(np.float32)
        # Note: Object velocity is not in the original list for Stage 2 in DcmmVecEnvStage2.py's observation_space
        # But let's check DcmmVecEnvStage2.py line 170: "object": spaces.Dict({"pos3d": ...})
        # It seems only pos3d is used for object in Stage 2 observation space definition.
        
        # 3. Hand Data
        hand_joints = self.get_hand_obs()
        
        # 4. Touch Data
        touch_force = np.zeros(4, dtype=np.float32)
        for i, name in enumerate(["sensor_touch_thumb", "sensor_touch_index", "sensor_touch_middle", "sensor_touch_ring"]):
            sensor_id = mujoco.mj_name2id(self.env.Dcmm.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id != -1:
                touch_force[i] = self.env.Dcmm.data.sensordata[sensor_id]
        
        # Flatten and Concatenate
        state_obs = np.concatenate([
            ee_pos, ee_quat, ee_vel, arm_joints,
            obj_pos,
            hand_joints,
            touch_force
        ])
        
        return state_obs
