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


def load_log(path: Path):
    """Load ULog file and return ULog object."""
    if not pyulog_available:
        raise ImportError("pyulog not available, cannot load log files.")
        
    try:
        ulog = ULog(str(path))
        return ulog
    except Exception as e:
        print(f"Error loading log file {path}: {e}")
        raise


def extract_position_attitude(ulog) -> Dict[str, np.ndarray]:
    """
    Extract position and attitude data from ULog.
    
    Args:
        ulog: ULog object
        
    Returns:
        Dictionary containing position, attitude, and derived data arrays
    """
    # Get position data (vehicle_local_position)
    try:
        pos_data = ulog.get_dataset("vehicle_local_position").data
        timestamps = pos_data['timestamp']
        x = pos_data['x']
        y = pos_data['y'] 
        z = pos_data['z']
        vx = pos_data['vx']
        vy = pos_data['vy']
        vz = pos_data['vz']
        
        # Calculate altitude AGL (Above Ground Level)
        # Use negative z since NED coordinate system (z positive down)
        altitude_agl = -z
        
    except Exception as e:
        print(f"Error extracting position data: {e}")
        raise
    
    # Get attitude data (vehicle_attitude)
    try:
        att_data = ulog.get_dataset("vehicle_attitude").data
        att_timestamps = att_data['timestamp']
        q0 = att_data['q[0]']  # w component
        q1 = att_data['q[1]']  # x component
        q2 = att_data['q[2]']  # y component
        q3 = att_data['q[3]']  # z component
        
        # Convert quaternions to Euler angles
        roll_rad, pitch_rad, yaw_rad = quaternion_to_euler(q0, q1, q2, q3)
        roll_deg, pitch_deg, yaw_deg = quaternion_to_euler_degrees(q0, q1, q2, q3)
        
    except Exception as e:
        print(f"Error extracting attitude data: {e}")
        raise
    
    # Interpolate attitude data to position timestamps
    try:
        from scipy.interpolate import interp1d
    except ImportError:
        print("Warning: scipy not available, using simple interpolation")
        # Simple linear interpolation fallback
        roll_rad_interp = np.interp(timestamps, att_timestamps, roll_rad)
        pitch_rad_interp = np.interp(timestamps, att_timestamps, pitch_rad)
        yaw_rad_interp = np.interp(timestamps, att_timestamps, yaw_rad)
        roll_deg_interp = np.interp(timestamps, att_timestamps, roll_deg)
        pitch_deg_interp = np.interp(timestamps, att_timestamps, pitch_deg)
        yaw_deg_interp = np.interp(timestamps, att_timestamps, yaw_deg)
    else:
        # Use scipy for better interpolation
        interp_roll_rad = interp1d(att_timestamps, roll_rad, bounds_error=False, fill_value='extrapolate')
        interp_pitch_rad = interp1d(att_timestamps, pitch_rad, bounds_error=False, fill_value='extrapolate')
        interp_yaw_rad = interp1d(att_timestamps, yaw_rad, bounds_error=False, fill_value='extrapolate')
        interp_roll_deg = interp1d(att_timestamps, roll_deg, bounds_error=False, fill_value='extrapolate')
        interp_pitch_deg = interp1d(att_timestamps, pitch_deg, bounds_error=False, fill_value='extrapolate')
        interp_yaw_deg = interp1d(att_timestamps, yaw_deg, bounds_error=False, fill_value='extrapolate')
        
        roll_rad_interp = interp_roll_rad(timestamps)
        pitch_rad_interp = interp_pitch_rad(timestamps)
        yaw_rad_interp = interp_yaw_rad(timestamps)
        roll_deg_interp = interp_roll_deg(timestamps)
        pitch_deg_interp = interp_pitch_deg(timestamps)
        yaw_deg_interp = interp_yaw_deg(timestamps)
    
    # Get accelerometer data for impact detection
    try:
        accel_data = ulog.get_dataset("sensor_combined").data
        accel_timestamps = accel_data['timestamp']
        
        # Calculate accelerometer magnitude
        accel_x = accel_data['accelerometer_m_s2[0]']
        accel_y = accel_data['accelerometer_m_s2[1]']
        accel_z = accel_data['accelerometer_m_s2[2]']
        accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        
        print(f"Accelerometer data interpolated. Magnitude range: [{accel_magnitude.min():.2f}, {accel_magnitude.max():.2f}] m/s²")
        
        # Interpolate accelerometer data to position timestamps
        try:
            from scipy.interpolate import interp1d
            interp_accel_mag = interp1d(accel_timestamps, accel_magnitude, bounds_error=False, fill_value='extrapolate')
            accel_magnitude_interp = interp_accel_mag(timestamps)
        except ImportError:
            accel_magnitude_interp = np.interp(timestamps, accel_timestamps, accel_magnitude)
            
    except Exception as e:
        print(f"Warning: Could not extract accelerometer data: {e}")
        # Create dummy accelerometer data
        accel_magnitude_interp = np.ones_like(timestamps) * 9.81
    
    # Check for offboard mode (if available)
    try:
        offboard_data = ulog.get_dataset("vehicle_status").data
        offboard_timestamps = offboard_data['timestamp']
        nav_state = offboard_data['nav_state']
        
        # Find offboard mode periods (nav_state == 14 for offboard)
        offboard_mask = nav_state == 14
        if np.any(offboard_mask):
            offboard_times = (offboard_timestamps[offboard_mask] - timestamps[0]) * 1e-6
            print(f"Offboard mode detected at times: {[f'{t:.1f}s' for t in offboard_times[:5]]}{'...' if len(offboard_times) > 5 else ''}")
        
    except Exception as e:
        print(f"Note: Could not extract offboard mode data: {e}")
    
    return {
        "timestamp_us": timestamps,
        "position_x": x,
        "position_y": y,
        "position_z": z,  # Keep original z (NED, positive down)
        "altitude_agl": altitude_agl,  # AGL altitude (positive up)
        "velocity_x": vx,
        "velocity_y": vy,
        "velocity_z": vz,
        "roll_rad": roll_rad_interp,
        "pitch_rad": pitch_rad_interp,
        "yaw_rad": yaw_rad_interp,
        "roll_deg": roll_deg_interp,
        "pitch_deg": pitch_deg_interp,
        "yaw_deg": yaw_deg_interp,
        "accel_magnitude": accel_magnitude_interp
    }


def compute_terminal_engagement_mask(data: Dict[str, np.ndarray], pitch_threshold: float = -4.0,
                                   accel_threshold: float = 15.0, alt_threshold: float = 5.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute boolean mask for terminal engagement (dive-to-impact) segments.
    
    Args:
        data: Dictionary containing flight data
        pitch_threshold: Pitch angle threshold for dive detection (degrees, negative)
        accel_threshold: Acceleration threshold for impact detection (m/s²)
        alt_threshold: Altitude threshold for impact (meters AGL)
        
    Returns:
        tuple: (mask, engagement_info) where mask is boolean array and engagement_info contains metadata
    """
    # Initialize mask
    mask = np.zeros(len(data["timestamp_us"]), dtype=bool)
    
    # Dive detection: pitch < threshold (negative pitch is nose down)
    dive_mask = data["pitch_deg"] < pitch_threshold
    
    # Impact detection: high acceleration AND low altitude
    impact_mask = (data["accel_magnitude"] > accel_threshold) & (data["altitude_agl"] < alt_threshold)
    
    # Find dive periods
    dive_start_indices = np.where(np.diff(dive_mask.astype(int)) == 1)[0] + 1
    dive_end_indices = np.where(np.diff(dive_mask.astype(int)) == -1)[0] + 1
    
    # Handle edge cases
    if len(dive_start_indices) == 0 and len(dive_end_indices) == 0:
        if np.any(dive_mask):
            # Entire period is a dive
            dive_start_indices = np.array([0])
            dive_end_indices = np.array([len(dive_mask) - 1])
    elif len(dive_start_indices) == 0:
        # Starts in dive mode
        dive_start_indices = np.array([0])
    elif len(dive_end_indices) == 0:
        # Ends in dive mode
        dive_end_indices = np.array([len(dive_mask) - 1])
    elif len(dive_start_indices) > len(dive_end_indices):
        # Ends in dive mode
        dive_end_indices = np.append(dive_end_indices, len(dive_mask) - 1)
    elif len(dive_end_indices) > len(dive_start_indices):
        # Starts in dive mode
        dive_start_indices = np.insert(dive_start_indices, 0, 0)
    
    engagement_masks = {}
    terminal_segments = []
    
    # For each dive period, look for impact
    for i, (dive_start, dive_end) in enumerate(zip(dive_start_indices, dive_end_indices)):
        dive_duration = (data["timestamp_us"][dive_end] - data["timestamp_us"][dive_start]) * 1e-6
        
        # Look for impact during or shortly after dive
        search_end = min(dive_end + int(5.0 / ((data["timestamp_us"][1] - data["timestamp_us"][0]) * 1e-6)), len(impact_mask))
        impact_indices = np.where(impact_mask[dive_start:search_end])[0] + dive_start
        
        if len(impact_indices) > 0:
            # Use first impact after dive start
            impact_idx = impact_indices[0]
            terminal_end = min(impact_idx + int(2.0 / ((data["timestamp_us"][1] - data["timestamp_us"][0]) * 1e-6)), len(mask))
            
            # Mark terminal engagement segment
            mask[dive_start:terminal_end] = True
            
            segment_duration = (data["timestamp_us"][terminal_end-1] - data["timestamp_us"][dive_start]) * 1e-6
            terminal_segments.append({
                'start_idx': dive_start,
                'end_idx': terminal_end-1,
                'impact_idx': impact_idx,
                'start_time': (data["timestamp_us"][dive_start] - data["timestamp_us"][0]) * 1e-6,
                'end_time': (data["timestamp_us"][terminal_end-1] - data["timestamp_us"][0]) * 1e-6,
                'duration': segment_duration,
                'dive_duration': dive_duration
            })
            
            # Store first engagement info for compatibility
            if i == 0:
                engagement_masks['first_dive_idx'] = dive_start
                engagement_masks['first_impact_idx'] = impact_idx
    
    # Print detection summary
    print(f"Terminal engagement detection: Found {len(terminal_segments)} engagement segments")
    print(f"  - Dive threshold: pitch < {pitch_threshold}°")
    print(f"  - Impact threshold: accel > {accel_threshold} m/s², alt < {alt_threshold} m")
    
    for i, segment in enumerate(terminal_segments):
        print(f"  - Segment {i+1}: {segment['start_time']:.1f}s → {segment['end_time']:.1f}s (duration: {segment['duration']:.1f}s)")
    
    engagement_masks['segments'] = terminal_segments
    engagement_masks['num_segments'] = len(terminal_segments)
    
    return mask, engagement_masks


def calculate_miss_distances(ulog, data: Dict[str, np.ndarray], mask: np.ndarray, 
                           target_selection: str = "Container") -> Dict[str, float]:
    """
    Calculate miss distances for terminal engagement segments.
    
    Args:
        ulog: ULog object
        data: Dictionary containing flight data
        mask: Boolean mask for terminal engagement segments
        target_selection: Target selection ("Container" or "Van")
        
    Returns:
        Dictionary with miss distance information
    """
    if not np.any(mask):
        return {}
    
    # Define target locations in GPS coordinates
    all_targets = {
        'Container': {'lat': 43.2222722, 'lon': -75.3903593, 'alt_agl': 0.0},
        'Van': {'lat': 43.2221788, 'lon': -75.3905151, 'alt_agl': 0.0}
    }
    
    # Get GPS reference for coordinate conversion
    try:
        gps_data = ulog.get_dataset("vehicle_gps_position").data
        valid_fix = gps_data['fix_type'] > 2
        
        if not np.any(valid_fix):
            print("Warning: No valid GPS fix found")
            return {}
        
        # Use first valid GPS position as reference
        ref_lat = gps_data['lat'][valid_fix][0] / 1e7
        ref_lon = gps_data['lon'][valid_fix][0] / 1e7
        ref_alt = gps_data['alt'][valid_fix][0] / 1e3
        
    except Exception as e:
        print(f"Warning: Could not get GPS reference: {e}")
        return {}
    
    # Convert target GPS to local coordinates
    def gps_to_local(target_lat, target_lon, ref_lat, ref_lon):
        """Convert GPS coordinates to local NED coordinates."""
        # Approximate conversion (good for small distances)
        R_earth = 6378137.0  # Earth radius in meters
        
        dlat = np.radians(target_lat - ref_lat)
        dlon = np.radians(target_lon - ref_lon)
        
        # North and East distances
        north = dlat * R_earth
        east = dlon * R_earth * np.cos(np.radians(ref_lat))
        
        return north, east
    
    target_info = all_targets[target_selection]
    target_north, target_east = gps_to_local(target_info['lat'], target_info['lon'], ref_lat, ref_lon)
    
    # Get impact point (last point in terminal engagement)
    impact_idx = np.where(mask)[0][-1]
    impact_x = data["position_x"][impact_idx]
    impact_y = data["position_y"][impact_idx]
    impact_alt = data["altitude_agl"][impact_idx]
    
    # Calculate miss distance
    # Convert local position to same reference frame as target
    # Assume first position corresponds to GPS reference
    pos_offset_x = data["position_x"][0]
    pos_offset_y = data["position_y"][0]
    
    # Impact position relative to GPS reference
    impact_north = impact_x - pos_offset_x
    impact_east = impact_y - pos_offset_y
    
    # Miss distance calculation
    miss_north = impact_north - target_north
    miss_east = impact_east - target_east
    ground_distance = np.sqrt(miss_north**2 + miss_east**2)
    
    # 3D distance (including altitude)
    miss_alt = impact_alt - target_info['alt_agl']
    miss_distance_3d = np.sqrt(miss_north**2 + miss_east**2 + miss_alt**2)
    
    return {
        'target': target_selection,
        'ground_distance': ground_distance,
        'miss_distance_3d': miss_distance_3d,
        'miss_north': miss_north,
        'miss_east': miss_east,
        'miss_alt': miss_alt,
        'impact_position': {'x': impact_x, 'y': impact_y, 'alt': impact_alt},
        'target_position': {'north': target_north, 'east': target_east, 'alt': target_info['alt_agl']}
    }
