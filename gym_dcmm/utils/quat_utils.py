"""
Quaternion utility functions for orientation calculations.
"""
import numpy as np


def quat_rotate_vector(quat, vec):
    """
    Rotate a vector by a quaternion using MuJoCo convention.
    
    Args:
        quat: quaternion in MuJoCo format [w, x, y, z]
        vec: 3D vector [x, y, z]
    
    Returns:
        Rotated 3D vector
    """
    w, x, y, z = quat
    vx, vy, vz = vec
    
    # Quaternion rotation: v' = q * v * q^-1
    # Expanded form (Hamilton product)
    t = 2 * np.cross([x, y, z], vec)
    rotated = vec + w * t + np.cross([x, y, z], t)
    
    return rotated


def quat_rotate_inverse(quat, vec):
    """
    Rotate a vector by the inverse of a quaternion.
    Equivalent to transforming from world frame to local frame.
    
    Args:
        quat: quaternion in MuJoCo format [w, x, y, z]
        vec: 3D vector [x, y, z]
    
    Returns:
        Rotated 3D vector (in local frame)
    """
    # Inverse quaternion [w, -x, -y, -z]
    quat_inv = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
    return quat_rotate_vector(quat_inv, vec)


def quat_to_euler(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quat: quaternion in MuJoCo format [w, x, y, z]
    
    Returns:
        numpy array [roll, pitch, yaw] in radians
    """
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def angle_diff(a, b):
    """
    Compute the shortest angular difference between two angles.
    
    Args:
        a: First angle in radians
        b: Second angle in radians
    
    Returns:
        Shortest angular difference in radians (range: [-pi, pi])
    """
    diff = a - b
    # Normalize to [-pi, pi]
    while diff > np.pi:
        diff -= 2 * np.pi
    while diff < -np.pi:
        diff += 2 * np.pi
    return diff
