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

