"""Terminal Engagement Analysis - Utility Functions

This module contains helper functions for TG strike analysis including:
- Quaternion to Euler conversions
- Data loading and extraction functions
- Terminal engagement detection algorithms
- Miss distance calculations
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union
try:
    import pyulog
    from pyulog import ULog
    pyulog_available = True
except ImportError:
    pyulog_available = False
    print("Warning: pyulog not available, log loading will not work.")


def quaternion_to_euler(q0: np.ndarray, q1: np.ndarray, q2: np.ndarray, q3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert quaternions to Euler angles (roll, pitch, yaw) in radians.
    
    Args:
        q0, q1, q2, q3: Quaternion components (w, x, y, z)
        
    Returns:
        tuple: (roll, pitch, yaw) in radians
    """
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1 * q1 + q2 * q2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (q0 * q2 - q3 * q1)
    pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2 * q2 + q3 * q3)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return roll, pitch, yaw


def quaternion_to_euler_degrees(q0: np.ndarray, q1: np.ndarray, q2: np.ndarray, q3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert quaternions to Euler angles (roll, pitch, yaw) in degrees.
    
    Args:
        q0, q1, q2, q3: Quaternion components (w, x, y, z)
        
    Returns:
        tuple: (roll, pitch, yaw) in degrees
    """
    roll_rad, pitch_rad, yaw_rad = quaternion_to_euler(q0, q1, q2, q3)
    return np.degrees(roll_rad), np.degrees(pitch_rad), np.degrees(yaw_rad)


def calculate_impact_angle(roll_deg: float, pitch_deg: float, yaw_deg: float) -> float:
    """
    Calculate the angle between the aircraft's forward axis and the horizontal ground plane.
    
    The impact angle represents how vertical the aircraft is at impact:
    - 0° = horizontal flight (forward axis parallel to ground)
    - 90° = vertical dive (forward axis perpendicular to ground, pointing down)
    - -90° = vertical climb (forward axis perpendicular to ground, pointing up)
    
    Args:
        roll_deg: Roll angle in degrees
        pitch_deg: Pitch angle in degrees (negative is nose down)
        yaw_deg: Yaw angle in degrees
        
    Returns:
        Impact angle in degrees (0-90 for dive angles)
    """
    # Convert to radians
    roll_rad = np.radians(roll_deg)
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)
    
    # Aircraft forward axis in body frame is [1, 0, 0]
    # Transform to NED frame using rotation matrix
    # Note: PX4 uses NED (North-East-Down) frame
    
    # Rotation matrix from body to NED
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    cos_roll = np.cos(roll_rad)
    sin_roll = np.sin(roll_rad)
    cos_pitch = np.cos(pitch_rad)
    sin_pitch = np.sin(pitch_rad)
    cos_yaw = np.cos(yaw_rad)
    sin_yaw = np.sin(yaw_rad)
    
    # Forward vector in NED frame (first column of rotation matrix)
    forward_x = cos_pitch * cos_yaw
    forward_y = cos_pitch * sin_yaw
    forward_z = sin_pitch
    
    # Calculate angle from horizontal plane
    # The down component (forward_z) determines the dive angle
    # Positive pitch in NED means nose down, so positive forward_z means diving
    impact_angle = np.degrees(np.arcsin(forward_z))
    
    return impact_angle

