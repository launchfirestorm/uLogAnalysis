

from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np



# Import our custom modules
from tg_plotting import (
    plot_3d_terminal_engagement,
    plot_roll_pitch, 
    gps_trajectory_plot,
    plot_accelerometer_impacts,
    plot_miss_distance_histograms
)
from tg_utils import (
    quaternion_to_euler as quaternion_to_euler, 
    quaternion_to_euler_degrees as quaternion_to_euler_degrees, 
)

def set_matplotlib_backend(interactive: bool = False):
    """Set the appropriate matplotlib backend based on interactive mode."""
    try:
        import matplotlib
        if interactive:
            try:
                matplotlib.use('TkAgg', force=True)
                backend = 'TkAgg'
            except ImportError:
                try:
                    matplotlib.use('Qt5Agg', force=True)
                    backend = 'Qt5Agg'
                except ImportError:
                    matplotlib.use('Agg', force=True)
                    backend = 'Agg'
                    print("Warning: No interactive backend available, using Agg")
        else:
            matplotlib.use('Agg', force=True)
            backend = 'Agg'
        
        print(f"Using {'interactive' if interactive else 'static'} matplotlib backend: {backend}")
    except Exception as e:
        print(f"Backend setup error: {e}")

# Ensure we can import FlightReviewLib even though directory has a space.
LIB_PATH = Path(__file__).parent / "FlightReviewLib"
if str(LIB_PATH) not in sys.path:
    sys.path.append(str(LIB_PATH))

try:
    from helper import load_ulog_file  # pyulog wrapper with caching & msg filter
except ImportError as e:  # pragma: no cover
    raise RuntimeError(
        "Failed to import FlightReviewLib.helper. Likely missing dependency 'pyulog' or config file. "
        "Ensure the correct environment is active (conda activate python38_flight_review_env) and install pyulog via 'pip install pyulog'. "
        "Also verify 'config_default.ini' exists at the project root. Original error: " + str(e)
    ) from e



def print_ulog_debug_info(ulog):
    """Print debug information about available ULog datasets and fields.
    
    Parameters
    ----------
    ulog : ULog
        Loaded ULog object
    """
    print("\n=== ULog Debug Information ===")
    print(f"Log contains {len(ulog.data_list)} datasets:")
    
    for i, dataset in enumerate(ulog.data_list):
        print(f"\n{i+1:2d}. Dataset: '{dataset.name}' (instance {dataset.multi_id})")
        print(f"    Samples: {len(dataset.data['timestamp'])}")
        print(f"    Fields: {list(dataset.data.keys())}")
        
        # Show sample ranges for numeric fields
        for field_name, field_data in dataset.data.items():
            if field_name != 'timestamp' and hasattr(field_data, 'dtype') and np.issubdtype(field_data.dtype, np.number):
                if len(field_data) > 0:
                    print(f"      {field_name}: [{np.min(field_data):.3f}, {np.max(field_data):.3f}]")
    
    print("\n=== End Debug Information ===\n")


def load_log(path: Path):
    """Load the ULog file using FlightReviewLib helper.

    Parameters
    ----------
    path : Path
        Path to .ulg file.

    Returns
    -------
    ULog
        Parsed ULog object.
    """
    # Try to use the utils version first, fallback to FlightReviewLib
    try:
        return utils_load_log(path)
    except Exception:
        # Fallback to original FlightReviewLib version
        return load_ulog_file(str(path))


def extract_position_attitude(ulog) -> Dict[str, np.ndarray]:
    """Extract timestamp, local position (x,y,z) and attitude (roll,pitch,yaw).

    Returns dict with keys: timestamp_us, x, y, z, roll_deg, pitch_deg, yaw_deg
    Missing topics raise RuntimeError for clarity.
    """
    try:
        pos = ulog.get_dataset("vehicle_local_position").data
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ULog missing 'vehicle_local_position' dataset") from e

    try:
        att = ulog.get_dataset("vehicle_attitude").data
    except Exception as e:  # pragma: no cover
        raise RuntimeError("ULog missing 'vehicle_attitude' dataset") from e


    
    # Check for quaternion data and convert to Euler angles if needed
    q0, q1, q2, q3 = att['q[0]'], att['q[1]'], att['q[2]'], att['q[3]']
    roll, pitch, yaw = quaternion_to_euler_degrees(q0, q1, q2, q3)
        

    # Align attitude time to position time via interpolation (timestamp in microseconds)
    t_pos = pos["timestamp"].astype(np.int64)
    t_att = att["timestamp"].astype(np.int64)

    # Guard against empty arrays
    if len(t_pos) == 0 or len(t_att) == 0:  # pragma: no cover
        raise RuntimeError("Empty timestamps in position or attitude datasets.")

    # Defensive trimming if attitude field lengths differ from timestamp length
    n_att = len(t_att)
    if len(roll) != n_att:
        m = min(len(roll), n_att)
        roll = roll[:m]
        pitch = pitch[:m]
        yaw = yaw[:m]
        t_att = t_att[:m]
    if len(pitch) != len(t_att):
        m = min(len(pitch), len(t_att))
        pitch = pitch[:m]
        t_att = t_att[:m]
    if len(yaw) != len(t_att):
        m = min(len(yaw), len(t_att))
        yaw = yaw[:m]
        t_att = t_att[:m]

    roll_i = np.interp(t_pos, t_att, roll)
    pitch_i = np.interp(t_pos, t_att, pitch)
    yaw_i = np.interp(t_pos, t_att, yaw)

    # Debug: Print attitude ranges before and after interpolation
    # print(f"Raw attitude ranges - Roll: [{np.min(roll):.2f}, {np.max(roll):.2f}], "
    #       f"Pitch: [{np.min(pitch):.2f}, {np.max(pitch):.2f}], "
    #       f"Yaw: [{np.min(yaw):.2f}, {np.max(yaw):.2f}]")
    # print(f"Interpolated attitude ranges - Roll: [{np.min(roll_i):.2f}, {np.max(roll_i):.2f}], "
    #       f"Pitch: [{np.min(pitch_i):.2f}, {np.max(pitch_i):.2f}], "
    #       f"Yaw: [{np.min(yaw_i):.2f}, {np.max(yaw_i):.2f}]")

    # Extract attitude setpoints if available
    roll_sp_i = np.zeros_like(roll_i)
    pitch_sp_i = np.zeros_like(pitch_i)
    yaw_sp_i = np.zeros_like(yaw_i)
    
    try:
        att_sp = ulog.get_dataset("vehicle_attitude_setpoint").data
        # print("Found attitude setpoint data...")
        
        # Check for quaternion setpoints and convert to Euler angles if needed
        if 'q_d[0]' in att_sp and 'q_d[1]' in att_sp and 'q_d[2]' in att_sp and 'q_d[3]' in att_sp:
            # print("Converting quaternion setpoints to Euler angles...")
            q0_sp, q1_sp, q2_sp, q3_sp = att_sp['q_d[0]'], att_sp['q_d[1]'], att_sp['q_d[2]'], att_sp['q_d[3]']
            roll_sp, pitch_sp, yaw_sp = quaternion_to_euler_degrees(q0_sp, q1_sp, q2_sp, q3_sp)
            
            # Interpolate setpoints to position timestamps
            t_att_sp = att_sp["timestamp"].astype(np.int64)
            roll_sp_i = np.interp(t_pos, t_att_sp, roll_sp)
            pitch_sp_i = np.interp(t_pos, t_att_sp, pitch_sp)
            yaw_sp_i = np.interp(t_pos, t_att_sp, yaw_sp)
            
            
    except Exception as e:
        print(f"Attitude setpoint data not available: {e}")
    

    # Extract accelerometer data for impact detection
    accel_magnitude_i = np.zeros(len(t_pos))
    accel_x_i = np.zeros(len(t_pos))
    accel_y_i = np.zeros(len(t_pos))
    accel_z_i = np.zeros(len(t_pos))
    vertical_velocity_i = pos.get("vz", np.zeros(len(t_pos)))
    altitude_agl_i = -pos["z"]  # NED z is negative up
    landed_i = np.zeros(len(t_pos))
    ground_contact_i = np.zeros(len(t_pos))
    
    # Extract accelerometer data if available
    try:
        accel_data = ulog.get_dataset('sensor_combined').data
        accel_timestamps = accel_data['timestamp'].astype(np.int64)
        
        # Interpolate accelerometer data to match position timestamps
        accel_x_i = np.interp(t_pos, accel_timestamps, accel_data['accelerometer_m_s2[0]'])
        accel_y_i = np.interp(t_pos, accel_timestamps, accel_data['accelerometer_m_s2[1]'])
        accel_z_i = np.interp(t_pos, accel_timestamps, accel_data['accelerometer_m_s2[2]'])
        
        # Calculate total acceleration magnitude
        accel_magnitude_i = np.sqrt(accel_x_i**2 + accel_y_i**2 + accel_z_i**2)
    except Exception as e:
        print(f"Accelerometer data not available: {e}")

    
    # Extract flight mode data if available
    flight_mode_i = np.zeros(len(t_pos))
    offboard_mode_i = np.zeros(len(t_pos), dtype=bool)
    
    try:
        # Try vehicle_status first (contains nav_state which includes flight modes)
        status_data = ulog.get_dataset('vehicle_status').data
        status_timestamps = status_data['timestamp'].astype(np.int64)
        
        # nav_state values: 0=MANUAL, 1=ALTCTL, 2=POSCTL, 3=AUTO_MISSION, 4=AUTO_LOITER, 
        # 5=AUTO_RTL, 6=AUTO_LAND, 14=OFFBOARD, etc.
        if 'nav_state' in status_data:
            flight_mode_i = np.interp(t_pos, status_timestamps, status_data['nav_state'])
            # Mark offboard mode (nav_state == 14)
            offboard_mode_i = np.round(flight_mode_i).astype(int) == 14
            # print(f"Flight mode data interpolated. Nav state range: [{np.min(flight_mode_i):.1f}, {np.max(flight_mode_i):.1f}]")
            if np.any(offboard_mode_i):
                offboard_indices = np.where(offboard_mode_i)[0]
                offboard_times = [(t_pos[idx] - t_pos[0]) * 1e-6 for idx in offboard_indices[:5]]  # Show first 5
                print(f"Offboard mode detected at times: {[f'{t:.1f}s' for t in offboard_times]}{'...' if len(offboard_indices) > 5 else ''}")
        else:
            print("nav_state field not found in vehicle_status")
            
    except Exception as e:
        print(f"Flight mode data not available: {e}")

    return {
        "timestamp_us": t_pos,
        "x": pos["x"],
        "y": pos["y"],
        "z": pos["z"],
        "roll_deg": roll_i,
        "pitch_deg": pitch_i,
        "yaw_deg": yaw_i,
        "roll_sp_deg": roll_sp_i,
        "pitch_sp_deg": pitch_sp_i,
        "yaw_sp_deg": yaw_sp_i,
        "accel_magnitude": accel_magnitude_i,
        "accel_x": accel_x_i,
        "accel_y": accel_y_i,
        "accel_z": accel_z_i,
        "vertical_velocity": vertical_velocity_i,
        "altitude_agl": altitude_agl_i,
        "landed": landed_i,
        "ground_contact": ground_contact_i,
        "flight_mode": flight_mode_i,
        "offboard_mode": offboard_mode_i,
    }


def detect_ground_impact(data: Dict[str, np.ndarray], dive_start_idx: int = None, accel_threshold: float = 15.0, deriv_threshold: float = 2.0, alt_threshold: float = 10.0) -> np.ndarray:
    """
    Detect ground impact using accelerometer magnitude and derivative with altitude data.
    Only detects impacts after the dive starts.
    
    Args:
        data: Data dictionary containing accelerometer and altitude data
        dive_start_idx: Index where the first dive starts (impacts before this are ignored)
        accel_threshold: Acceleration magnitude threshold for impact (m/s²)
        deriv_threshold: Acceleration derivative magnitude threshold for impact (m/s²)
        alt_threshold: Altitude threshold - only consider impacts below this height (m)
    
    Returns:
        Boolean array indicating impact points
    """
    impact_mask = np.zeros(len(data['timestamp_us']), dtype=bool)
    
    # If no dive detected, return empty mask
    if dive_start_idx is None:
        print("No dive detected - cannot detect impacts")
        data['method1_impact_indices'] = np.array([], dtype=int)
        data['method2_impact_indices'] = np.array([], dtype=int)
        data['fallback_impact_indices'] = np.array([], dtype=int)
        return impact_mask
    
    low_altitude = data['altitude_agl'] < alt_threshold
    
    # Method 1: High acceleration magnitude at low altitude
    high_accel = data['accel_magnitude'] > accel_threshold
    method1_candidates = high_accel & low_altitude
    
    # Method 2: Large change in smoothed acceleration derivative at low altitude
    # Calculate acceleration derivative (change in acceleration)
    accel_deriv = np.diff(data['accel_magnitude'], prepend=data['accel_magnitude'][0])
    
    # Calculate smoothed acceleration derivative (10-sample moving average)
    window_size = 5
    if len(accel_deriv) >= window_size:
        accel_deriv_smooth = np.convolve(accel_deriv, np.ones(window_size)/window_size, mode='same')
    else:
        accel_deriv_smooth = accel_deriv
    
    # Use magnitude of smoothed derivative
    accel_deriv_smooth_mag = np.abs(accel_deriv_smooth)
    
    # Store derivatives in data for plotting
    data['accel_derivative'] = accel_deriv
    data['accel_derivative_smooth'] = accel_deriv_smooth_mag  # Store magnitude
    
    # Large change in smoothed derivative magnitude
    large_deriv_change = accel_deriv_smooth_mag > deriv_threshold
    method2_candidates = large_deriv_change & low_altitude
    
    # Store method 1 and method 2 impact indices separately for plotting
    method1_impact_indices = np.where(method1_candidates)[0]
    method2_impact_indices = np.where(method2_candidates)[0]
    data['method1_impact_indices'] = method1_impact_indices
    data['method2_impact_indices'] = method2_impact_indices
    
    # Combine both methods
    impact_candidates = method1_candidates | method2_candidates
    
    # Remove spurious detections by requiring impacts to be separated by at least 1 second
    impact_indices = np.where(impact_candidates)[0]
    if len(impact_indices) > 0:
        # Group nearby impacts
        time_diff_threshold = 1e6  # 1 second in microseconds
        timestamps = data['timestamp_us']
        
        filtered_impacts = [impact_indices[0]]  # Always keep first impact
        for i in range(1, len(impact_indices)):
            current_idx = impact_indices[i]
            last_impact_idx = filtered_impacts[-1]
            
            if timestamps[current_idx] - timestamps[last_impact_idx] > time_diff_threshold:
                filtered_impacts.append(current_idx)
        
        # Filter to only keep impacts after the dive
        filtered_impacts_after_dive = [idx for idx in filtered_impacts if idx > dive_start_idx]
        impact_mask[filtered_impacts_after_dive] = True
        
        # print(f"Impact detection: Found {len(filtered_impacts_after_dive)} impact events after dive")
        # print(f"  - Acceleration threshold: {accel_threshold} m/s²")
        # print(f"  - Altitude threshold: {alt_threshold} m") 
        # impact_times = [(data['timestamp_us'][idx] - data['timestamp_us'][0]) * 1e-6 for idx in filtered_impacts_after_dive]
        # print(f"  - Impact times (relative): {[f'{t:.1f}s' for t in impact_times]}")
    
    # Method 3: Fallback to <0m AGL AND >0.5 accel deriv magnitude if no acceleration-based impacts detected after dive
    if not np.any(impact_mask):
        print("No acceleration-based impacts detected after dive")
        
        # Find first point after dive where altitude drops below 0m AGL AND accel deriv > 0.5
        below_0m = data['altitude_agl'] < 0.0
        high_deriv = accel_deriv_smooth_mag > 0.5
        fallback_candidates = below_0m & high_deriv
        fallback_candidates[:dive_start_idx] = False  # Only consider points after dive
        
        if np.any(fallback_candidates):
            fallback_idx = np.where(fallback_candidates)[0][0]
            impact_mask[fallback_idx] = True
            data['fallback_impact_indices'] = np.array([fallback_idx])
            print(f"Using Method 3 fallback: <0m AGL + >0.5 accel deriv at index {fallback_idx}")
        else:
            data['fallback_impact_indices'] = np.array([], dtype=int)
            print("Warning: No point meeting Method 3 criteria (<0m AGL + >0.5 accel deriv) found after dive")
    else:
        data['fallback_impact_indices'] = np.array([], dtype=int)
    
    return impact_mask


def compute_engagement_masks(data: Dict[str, np.ndarray], pitch_threshold: float = -4.0,
                           accel_threshold: float = 15.0, deriv_threshold: float = 2.0, alt_threshold: float = 5.0) -> Dict[str, np.ndarray]:
    """
    Compute all engagement-related masks once for efficiency.
    
    Returns:
        Dictionary containing:
        - 'dive_mask': Boolean mask for all dive events (pitch < threshold)
        - 'impact_mask': Boolean mask for all impact events
        - 'dive_start_indices': Indices where dives start
        - 'impact_indices': Indices of impact events
        - 'terminal_engagement_mask': Boolean mask for terminal engagement segments
        - 'first_dive_idx': Index of first dive start (or None)
        - 'first_impact_idx': Index of first impact after first dive (or None)
    """
    n_samples = len(data['timestamp_us'])
    
    # Step 1: Compute dive mask and transitions
    dive_mask = data['pitch_deg'] < pitch_threshold
    dive_transitions = np.diff(dive_mask.astype(int), prepend=0)
    dive_start_indices = np.where(dive_transitions == 1)[0]
    
    # Get first dive index (or None if no dive detected)
    first_dive_idx = dive_start_indices[0] if len(dive_start_indices) > 0 else None
    
    # Step 2: Compute impact mask (only after dive starts)
    impact_mask = detect_ground_impact(data, first_dive_idx, accel_threshold, deriv_threshold, alt_threshold)
    impact_indices = np.where(impact_mask)[0]
    
    # Step 3: Find first terminal engagement (dive to impact)
    terminal_engagement_mask = np.zeros(n_samples, dtype=bool)
    first_impact_idx = None
    
    if first_dive_idx is not None and len(impact_indices) > 0:
        # By design, all impacts in impact_indices are after the dive
        first_impact_idx = impact_indices[0]
        
        # Validate timing (< 30 seconds between dive and impact)
        timestamps = data['timestamp_us']
        time_diff = (timestamps[first_impact_idx] - timestamps[first_dive_idx]) * 1e-6
        if time_diff < 30.0:
            terminal_engagement_mask[first_dive_idx:first_impact_idx+1] = True
    
    # Print detection summary
    engagements = 1 if np.any(terminal_engagement_mask) else 0
    start_time = (data['timestamp_us'][first_dive_idx] - data['timestamp_us'][0]) * 1e-6 if first_dive_idx is not None else 0
    end_time = (data['timestamp_us'][first_impact_idx] - data['timestamp_us'][0]) * 1e-6 if first_impact_idx is not None else 0
    duration = end_time - start_time if engagements > 0 else 0
    
    print(f"Terminal engagement detection: Found {engagements} engagement segments")
    if engagements > 0:
        print(f"  - Segment 1: {start_time:.1f}s → {end_time:.1f}s (duration: {duration:.1f}s)")
    
    return {
        'dive_mask': dive_mask,
        'impact_mask': impact_mask,
        'dive_start_indices': dive_start_indices,
        'impact_indices': impact_indices,
        'terminal_engagement_mask': terminal_engagement_mask,
        'first_dive_idx': first_dive_idx,
        'first_impact_idx': first_impact_idx
    }


def detect_offboard_transitions(data: Dict[str, np.ndarray], mask: np.ndarray) -> tuple[list, list]:
    """
    Detect offboard mode transitions within the filtered data segment.
    
    Args:
        data: Data dictionary containing flight mode information
        mask: Boolean mask for the segment of interest
        
    Returns:
        Tuple of (offboard_start_indices, offboard_end_indices) within the masked segment
    """
    if 'offboard_mode' not in data or not np.any(data['offboard_mode']):
        return [], []
    
    # Get offboard mode data for the masked segment
    masked_indices = np.where(mask)[0]
    if len(masked_indices) == 0:
        return [], []
    
    offboard_in_segment = data['offboard_mode'][mask]
    
    # Find transitions within the segment
    offboard_transitions = np.diff(offboard_in_segment.astype(int), prepend=0)
    
    # Find start and end indices (relative to original data, not masked)
    start_transitions = np.where(offboard_transitions == 1)[0]  # 0 -> 1 transitions
    end_transitions = np.where(offboard_transitions == -1)[0]   # 1 -> 0 transitions
    
    # Convert back to original data indices
    offboard_start_indices = [masked_indices[i] for i in start_transitions]
    offboard_end_indices = [masked_indices[i] for i in end_transitions]
    
    return offboard_start_indices, offboard_end_indices


def save_filtered_csv(data: Dict[str, np.ndarray], mask: np.ndarray, out_path: Path):
    """Write filtered samples to CSV for downstream analysis or debugging."""
    header = ["timestamp_us", "x", "y", "z", "roll_deg", "pitch_deg", "yaw_deg", "flight_mode", "offboard_mode"]
    # Only include headers for data that exists
    available_data = []
    available_headers = []
    for k in header:
        if k in data:
            available_data.append(data[k][mask])
            available_headers.append(k)
    
    if available_data:
        stacked = np.column_stack(available_data)
        np.savetxt(out_path, stacked, delimiter=",", header=",".join(available_headers), comments="")
        print(f"Saved filtered data CSV: {out_path}")
    else:
        print("No data available for CSV export")


def calculate_miss_distances(ulog, data: Dict[str, np.ndarray], mask: np.ndarray, 
                           engagement_masks: Dict = None, pitch_threshold: float = -4.0, 
                           accel_threshold: float = 15.0, deriv_threshold: float = 2.0, alt_threshold: float = 5.0,
                           target_selection: str = "both") -> Dict[str, Dict[str, float]]:
    """Calculate miss distances for targets including impact point and closest point of approach (CPA).
    
    Returns:
        Dictionary with target names as keys and miss distance info as values.
        Each target includes:
        - 'impact_ground': ground distance at impact point
        - 'impact_total': 3D distance at impact point
        - 'cpa_ground': ground distance at closest point of approach
        - 'cpa_total': 3D distance at closest point of approach (true miss distance)
    """
    import math
    
    # Define target locations (hardcoded) - altitudes are AGL (ground level)
    all_targets = {
        'Container': {'lat': 43.2222722, 'lon': -75.3903593, 'alt_agl': 2.0},
        'Van': {'lat': 43.2221788, 'lon': -75.3905151, 'alt_agl': 1.0}
    }
    
    # Filter targets based on selection
    targets = {target_selection: all_targets[target_selection]}
    
    miss_distances = {}
    
    if not np.any(mask):
        return miss_distances
    
    # Use pre-computed masks if available, otherwise compute them
    if engagement_masks is not None:
        first_dive_idx = engagement_masks.get('first_dive_idx')
        first_impact_idx = engagement_masks.get('first_impact_idx')
    else:
        # Fallback: compute masks if not provided
        masks = compute_engagement_masks(data, pitch_threshold, accel_threshold, deriv_threshold, alt_threshold)
        first_dive_idx = masks['first_dive_idx']
        first_impact_idx = masks['first_impact_idx']
    
    # Calculate miss distances if we have valid engagement
    if first_dive_idx is not None and first_impact_idx is not None:
            
            # Get GPS data for impact point
            try:
                gps_data = ulog.get_dataset("vehicle_gps_position").data
                valid_fix = gps_data['fix_type'] > 2
                
                if np.any(valid_fix):
                    gps_timestamps = gps_data['timestamp'][valid_fix]
                    impact_gps_idx = np.searchsorted(gps_timestamps, data["timestamp_us"][first_impact_idx])
                    impact_gps_idx = np.clip(impact_gps_idx, 0, len(gps_timestamps) - 1)
                    
                    impact_lat = gps_data['lat'][valid_fix][impact_gps_idx] / 1e7
                    impact_lon = gps_data['lon'][valid_fix][impact_gps_idx] / 1e7
                    impact_alt_agl = data["altitude_agl"][first_impact_idx]  # Use AGL from local position
                    impact_gps_alt_msl = gps_data['alt'][valid_fix][impact_gps_idx] / 1000.0  # GPS altitude MSL in meters
                    
                    # Haversine distance calculation helper
                    def haversine_distance(lat1, lon1, lat2, lon2):
                        R = 6371000  # Earth's radius in meters
                        
                        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
                        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
                        
                        dlat = lat2_rad - lat1_rad
                        dlon = lon2_rad - lon1_rad
                        
                        a = (math.sin(dlat/2)**2 + 
                             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
                        c = 2 * math.asin(math.sqrt(a))
                        
                        return R * c
                    
                    # Get GPS data for entire trajectory during terminal engagement
                    mask_indices = np.where(mask)[0]
                    trajectory_timestamps = data["timestamp_us"][mask]
                    
                    # Interpolate GPS coordinates for entire trajectory
                    trajectory_gps_lat = np.interp(trajectory_timestamps, gps_timestamps, gps_data['lat'][valid_fix]) / 1e7
                    trajectory_gps_lon = np.interp(trajectory_timestamps, gps_timestamps, gps_data['lon'][valid_fix]) / 1e7
                    trajectory_alt_agl = data["altitude_agl"][mask]  # Local AGL (from NED z)
                    trajectory_gps_alt_msl = np.interp(trajectory_timestamps, gps_timestamps, gps_data['alt'][valid_fix]) / 1000.0  # GPS altitude MSL in meters
                    
                    # For simulated flat world: offset GPS altitudes to AGL using first trajectory point as reference
                    # This matches the approach used in gps_trajectory_plot
                    first_trajectory_timestamp = data["timestamp_us"][0]
                    first_gps_idx = np.searchsorted(gps_timestamps, first_trajectory_timestamp)
                    first_gps_idx = np.clip(first_gps_idx, 0, len(gps_timestamps) - 1)
                    gps_altitude_offset = gps_data['alt'][valid_fix][first_gps_idx] / 1000.0
                    
                    # Convert trajectory GPS altitudes to AGL by subtracting initial altitude
                    trajectory_gps_alt_agl = trajectory_gps_alt_msl - gps_altitude_offset
                    
                    for target_name, target_info in targets.items():
                        # For simulated flat world: target altitude is simply its AGL value (no offset needed since world is flat)
                        target_alt_agl = target_info['alt_agl']
                        
                        # Impact point miss distance (original metric)
                        # Uses local AGL which assumes same ground elevation - works well for flat simulated world
                        impact_ground_distance = haversine_distance(impact_lat, impact_lon, 
                                                           target_info['lat'], target_info['lon'])
                        impact_alt_diff = impact_alt_agl - target_alt_agl
                        impact_total_distance = math.sqrt(impact_ground_distance**2 + impact_alt_diff**2)
                        
                        # Closest Point of Approach (CPA) - true miss distance
                        # Use GPS altitudes converted to AGL (with same offset as trajectory plot)
                        ground_distances = np.array([
                            haversine_distance(lat, lon, target_info['lat'], target_info['lon'])
                            for lat, lon in zip(trajectory_gps_lat, trajectory_gps_lon)
                        ])
                        # Calculate altitude differences using GPS AGL altitudes (flat world assumption)
                        alt_diffs_agl = trajectory_gps_alt_agl - target_alt_agl
                        total_distances = np.sqrt(ground_distances**2 + alt_diffs_agl**2)
                        
                        # Find closest approach
                        cpa_idx = np.argmin(total_distances)
                        cpa_ground = ground_distances[cpa_idx]
                        cpa_total = total_distances[cpa_idx]
                        cpa_alt_diff = alt_diffs_agl[cpa_idx]
                        cpa_timestamp_us = trajectory_timestamps[cpa_idx]
                        cpa_time_s = (cpa_timestamp_us - data["timestamp_us"][0]) * 1e-6
                        
                        miss_distances[target_name] = {
                            # Legacy impact point metrics (for backwards compatibility)
                            'ground': impact_ground_distance,
                            'total': impact_total_distance,
                            'altitude_diff': impact_alt_diff,
                            'impact_lat': impact_lat,
                            'impact_lon': impact_lon,
                            'impact_alt_agl': impact_alt_agl,
                            # New metrics with explicit names
                            'impact_ground': impact_ground_distance,
                            'impact_total': impact_total_distance,
                            'impact_alt_diff': impact_alt_diff,
                            # CPA metrics (true miss distance)
                            'cpa_ground': cpa_ground,
                            'cpa_total': cpa_total,
                            'cpa_alt_diff': cpa_alt_diff,
                            'cpa_time_s': cpa_time_s,
                            'cpa_lat': trajectory_gps_lat[cpa_idx],
                            'cpa_lon': trajectory_gps_lon[cpa_idx],
                            'cpa_alt_agl': trajectory_gps_alt_agl[cpa_idx]  # Use GPS-based AGL altitude
                        }
                        
            except Exception as e:
                print(f"Could not calculate miss distances: {e}")
    
    return miss_distances


def process_multiple_logs(logs_dir: Path, output_dir: Path, 
                         pitch_threshold: float = -4.0, accel_threshold: float = 15.0, 
                         deriv_threshold: float = 2.0, alt_threshold: float = 5.0,
                         target_selection: str = "both", interactive_3d: bool = False) -> None:
    """Process multiple log files and calculate miss distance statistics."""
    # Find all .ulg files in the logs directory and subdirectories
    log_files = list(logs_dir.rglob("*.ulg"))
    
    if not log_files:
        print(f"No .ulg files found in {logs_dir}")
        return
    
    print(f"\n=== Processing {len(log_files)} log files ===")
    
    # Initialize miss distances based on target selection
    all_miss_distances = {target_selection: []}
    
    log_results = []
    trajectory_data = []  # For GPS trajectory plotting
    
    for i, log_file in enumerate(log_files, 1):
        print(f"\n[{i}/{len(log_files)}] Processing: {log_file.name}")
        
        try:
            # Load and process log
            ulog = load_log(log_file)
            data = extract_position_attitude(ulog)
            
            # Calculate terminal engagement mask
            engagement_masks = compute_engagement_masks(data, pitch_threshold, 
                                                      accel_threshold, deriv_threshold, alt_threshold)
            mask = engagement_masks['terminal_engagement_mask']
            
            # Collect trajectory data for GPS plotting (even if no engagement)
            trajectory_data.append({
                'filename': log_file.name,
                'ulog': ulog,
                'data': data,
                'mask': mask,
                'engagement_masks': engagement_masks
            })
            
            if not np.any(mask):
                print(f"  No terminal engagement found in {log_file.name}")
                log_results.append({
                    'file': log_file.name,
                    'status': 'No engagement',
                    'miss_distances': {}
                })
                continue
            
            # Calculate miss distances using pre-computed masks
            miss_distances = calculate_miss_distances(ulog, data, mask, engagement_masks, 
                                                    pitch_threshold, accel_threshold, deriv_threshold, alt_threshold,
                                                    target_selection)
            
            if miss_distances:
                print(f"  Terminal engagement found:")
                for target_name, distances in miss_distances.items():
                    impact_dist = distances['impact_ground']
                    cpa_dist = distances['cpa_total']
                    print(f"    {target_name}: Impact={impact_dist:.1f}m, CPA={cpa_dist:.1f}m (true miss)")
                    all_miss_distances[target_name].append(distances)
                
                log_results.append({
                    'file': log_file.name,
                    'status': 'Success',
                    'miss_distances': miss_distances
                })
            else:
                print(f"  Could not calculate miss distances for {log_file.name}")
                log_results.append({
                    'file': log_file.name,
                    'status': 'Calculation failed',
                    'miss_distances': {}
                })
                
        except Exception as e:
            print(f"  Error processing {log_file.name}: {e}")
            log_results.append({
                'file': log_file.name,
                'status': f'Error: {str(e)}',
                'miss_distances': {}
            })
    
    # Calculate and display statistics
    print(f"\n=== Miss Distance Statistics (Target: {target_selection}) ===")
    
    stats_output = []
    for target_name, distance_dicts in all_miss_distances.items():
        if distance_dicts:
            # Extract arrays for each metric
            impact_ground_array = np.array([d['impact_ground'] for d in distance_dicts])
            impact_total_array = np.array([d['impact_total'] for d in distance_dicts])
            cpa_ground_array = np.array([d['cpa_ground'] for d in distance_dicts])
            cpa_total_array = np.array([d['cpa_total'] for d in distance_dicts])
            
            print(f"\n{target_name} Target:")
            print(f"  Successful engagements: {len(distance_dicts)}")
            print(f"\n  Impact Point Distance (ground):")
            print(f"    Mean: {np.mean(impact_ground_array):.1f}m, Std: {np.std(impact_ground_array):.1f}m")
            print(f"    Range: {np.min(impact_ground_array):.1f}m - {np.max(impact_ground_array):.1f}m")
            print(f"\n  Closest Point of Approach (CPA) - True Miss Distance (3D):")
            print(f"    Mean: {np.mean(cpa_total_array):.1f}m, Std: {np.std(cpa_total_array):.1f}m")
            print(f"    Range: {np.min(cpa_total_array):.1f}m - {np.max(cpa_total_array):.1f}m")
            
            stats_output.append({
                'target': target_name,
                'count': len(distance_dicts),
                'impact_ground_mean': np.mean(impact_ground_array),
                'impact_ground_std': np.std(impact_ground_array),
                'impact_ground_min': np.min(impact_ground_array),
                'impact_ground_max': np.max(impact_ground_array),
                'cpa_total_mean': np.mean(cpa_total_array),
                'cpa_total_std': np.std(cpa_total_array),
                'cpa_total_min': np.min(cpa_total_array),
                'cpa_total_max': np.max(cpa_total_array),
                'distance_dicts': distance_dicts
            })
    
    # Save detailed results to CSV
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Individual log results
    # Include target selection in results filename
    target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
    results_file = output_dir / f"log_analysis_results{target_suffix}.csv"
    with open(results_file, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        
        # Dynamic header based on target selection - now includes both impact and CPA metrics
        header = ['File', 'Status']
        if target_selection == "Container":
            header.extend(['Container_Impact_Ground_m', 'Container_CPA_Total_m'])
        else:  # Van
            header.extend(['Van_Impact_Ground_m', 'Van_CPA_Total_m'])
        header.extend(['Impact_Lat', 'Impact_Lon', 'Impact_Alt_AGL_m', 'CPA_Lat', 'CPA_Lon', 'CPA_Alt_AGL_m', 'CPA_Time_s'])
        writer.writerow(header)
        
        for result in log_results:
            row = [result['file'], result['status']]
            if result['miss_distances']:
                if target_selection == "both":
                    container_impact = result['miss_distances'].get('Container', {}).get('impact_ground', '')
                    container_cpa = result['miss_distances'].get('Container', {}).get('cpa_total', '')
                    greenvan_impact = result['miss_distances'].get('Van', {}).get('impact_ground', '')
                    greenvan_cpa = result['miss_distances'].get('Van', {}).get('cpa_total', '')
                    row.extend([container_impact, container_cpa, greenvan_impact, greenvan_cpa])
                elif target_selection == "Container":
                    container_impact = result['miss_distances'].get('Container', {}).get('impact_ground', '')
                    container_cpa = result['miss_distances'].get('Container', {}).get('cpa_total', '')
                    row.extend([container_impact, container_cpa])
                else:  # Van
                    greenvan_impact = result['miss_distances'].get('Van', {}).get('impact_ground', '')
                    greenvan_cpa = result['miss_distances'].get('Van', {}).get('cpa_total', '')
                    row.extend([greenvan_impact, greenvan_cpa])
                
                # Use first available target for coordinates
                first_target = next(iter(result['miss_distances'].values()), {})
                impact_lat = first_target.get('impact_lat', '')
                impact_lon = first_target.get('impact_lon', '')
                impact_alt_agl = first_target.get('impact_alt_agl', '')
                cpa_lat = first_target.get('cpa_lat', '')
                cpa_lon = first_target.get('cpa_lon', '')
                cpa_alt_agl = first_target.get('cpa_alt_agl', '')
                cpa_time = first_target.get('cpa_time_s', '')
                row.extend([impact_lat, impact_lon, impact_alt_agl, cpa_lat, cpa_lon, cpa_alt_agl, cpa_time])
            else:
                # Fill empty columns based on target selection
                metric_count = 2 if target_selection != "both" else 4  # impact + cpa for each target
                row.extend([''] * (metric_count + 7))  # metrics + coordinates + time
            writer.writerow(row)
    
    print(f"\nDetailed results saved to: {results_file} (Target: {target_selection})")
    
    # Summary statistics
    stats_file = output_dir / "miss_distance_statistics.csv"
    with open(stats_file, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Target', 'Count', 'Metric', 'Mean_m', 'StdDev_m', 'Min_m', 'Max_m'])
        
        for stat in stats_output:
            writer.writerow([stat['target'], stat['count'], 'Impact_Ground', 
                           f"{stat['impact_ground_mean']:.1f}", f"{stat['impact_ground_std']:.1f}", 
                           f"{stat['impact_ground_min']:.1f}", f"{stat['impact_ground_max']:.1f}"])
            writer.writerow([stat['target'], stat['count'], 'CPA_Total', 
                           f"{stat['cpa_total_mean']:.1f}", f"{stat['cpa_total_std']:.1f}", 
                           f"{stat['cpa_total_min']:.1f}", f"{stat['cpa_total_max']:.1f}"])
    
    print(f"Summary statistics saved to: {stats_file}")
    print(f"\nTotal logs processed: {len(log_files)}")
    print(f"Successful engagements: {len([r for r in log_results if r['status'] == 'Success'])}")
    
    # Create 3D terminal engagement plot for batch (shows all trajectories with mean miss distance)
    mean_miss = stats_output[0]['cpa_total_mean'] if stats_output else None
    plot_3d_terminal_engagement(trajectory_data, output_dir, target_selection, interactive_3d, mean_miss)
    
    # Create GPS trajectory plot for visual debugging (shows entire trajectories)
    gps_trajectory_plot(trajectory_data, output_dir, target_selection, interactive_3d)
    
    # Create accelerometer impact plot to validate impact detection
    plot_accelerometer_impacts(trajectory_data, output_dir, target_selection, interactive_3d)
    
    # Create roll/pitch timeseries plot for all trajectories with offboard mode
    plot_roll_pitch(trajectory_data, output_dir, interactive_3d, target_selection)
    
    # Create miss distance histogram plots
    plot_miss_distance_histograms(stats_output, output_dir, target_selection, interactive_3d)
    
    # Combine all PNG plots into a single PDF
    combine_plots_to_pdf(output_dir, target_selection, len(trajectory_data) > 1)


def combine_plots_to_pdf(output_dir: Path, target_selection: str, is_batch: bool = True):
    """Combine all PNG plots into a single PDF file.
    
    Args:
        output_dir: Directory containing PNG files
        target_selection: Target name for filename
        is_batch: Whether this is batch processing (affects which files to include)
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL/Pillow not available. Install with: pip install Pillow")
        print("Skipping PDF generation.")
        return
    
    target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
    
    # Define the order of plots to include in PDF
    if is_batch:
        plot_files = [
            f"combined_3d_terminal_engagement{target_suffix}.png",
            f"combined_gps_trajectories{target_suffix}.png",
            f"combined_roll_pitch{target_suffix}.png",
            f"combined_accelerometer_method1{target_suffix}.png",
            f"combined_accelerometer_method2{target_suffix}.png",
            f"miss_distance_histograms{target_suffix}.png"
        ]
    else:
        plot_files = [
            f"3d_terminal_engagement{target_suffix}.png",
            f"gps_trajectories{target_suffix}.png",
            f"roll_pitch_timeseries{target_suffix}.png",
            f"accelerometer_method1{target_suffix}.png",
            f"accelerometer_method2{target_suffix}.png"
        ]
    
    # Collect existing plot files
    images = []
    for plot_file in plot_files:
        plot_path = output_dir / plot_file
        if plot_path.exists():
            try:
                img = Image.open(plot_path)
                # Convert to RGB if necessary (PDF requires RGB)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not open {plot_file}: {e}")
    
    if not images:
        print("No PNG files found to combine into PDF.")
        return
    
    # Save as PDF
    pdf_filename = f"combined_analysis{target_suffix}.pdf" if is_batch else f"analysis{target_suffix}.pdf"
    pdf_path = output_dir / pdf_filename
    
    try:
        # Save first image and append the rest
        images[0].save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=images[1:])
        print(f"\nCombined all plots into PDF: {pdf_path}")
    except Exception as e:
        print(f"Error creating PDF: {e}")
    finally:
        # Close all images
        for img in images:
            img.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze terminal engagement (dive-to-impact) segments from PX4 flight logs.")
    parser.add_argument("--ulog", type=Path, help="Path to PX4 .ulg log file (required for single file mode)")
    parser.add_argument("--logs-dir", type=Path, help="Path to directory containing multiple .ulg files for batch processing")
    parser.add_argument("--output", type=Path, default=Path("./tg_analysis_output"), help="Output directory for plots & data")
    parser.add_argument("--pitch-threshold", type=float, default=-4.0, help="Pitch threshold (deg) for dive detection. Default -4.0")
    parser.add_argument("--accel-threshold", type=float, default=15.0, help="Acceleration magnitude threshold for impact detection (m/s²). Default 15.0")
    parser.add_argument("--deriv-threshold", type=float, default=2.0, help="Acceleration derivative threshold for impact detection (m/s²). Default 2.0")
    parser.add_argument("--alt-threshold", type=float, default=10.0, help="Altitude threshold for impact detection (m AGL). Default 10.0")
    parser.add_argument("--save-csv", action="store_true", help="Always save filtered CSV alongside plots")
    parser.add_argument("--debug", action="store_true", help="Print debug information about available ULog datasets and fields")
    parser.add_argument("--interactive-3d", action="store_true", help="Show interactive 3D plots that can be rotated and zoomed (requires display)")
    parser.add_argument("--target", choices=["Container", "Van"], required=True, help="Target selection: 'Container' or 'Van' (required)")
    
    args = parser.parse_args()
    
    # Set matplotlib backend based on interactive mode
    set_matplotlib_backend(args.interactive_3d)
    
    # Check for required arguments based on mode
    if args.logs_dir and args.ulog:
        raise SystemExit("Error: Cannot specify both --ulog and --logs-dir. Choose single file or batch processing mode.")
    
    if not args.logs_dir and not args.ulog:
        raise SystemExit("Error: Must specify either --ulog for single file or --logs-dir for batch processing.")
    
    # Batch processing mode
    if args.logs_dir:
        if not args.logs_dir.exists():
            raise SystemExit(f"Logs directory not found: {args.logs_dir}")
        
        print(f"Batch processing mode: analyzing all .ulg files in {args.logs_dir}")
        process_multiple_logs(args.logs_dir, args.output, 
                            args.pitch_threshold, args.accel_threshold, args.deriv_threshold, args.alt_threshold,
                            args.target, args.interactive_3d)
        return

    # Single file processing mode (original behavior)
    if not args.ulog.exists():
        raise SystemExit(f"ULog file not found: {args.ulog}")

    args.output.mkdir(parents=True, exist_ok=True)
    ulog = load_log(args.ulog)
    
    if args.debug:
        print_ulog_debug_info(ulog)

    data = extract_position_attitude(ulog)

        
    # Compute terminal engagement mask
    engagement_masks = compute_engagement_masks(data, args.pitch_threshold, args.accel_threshold, args.deriv_threshold, args.alt_threshold)
    mask = engagement_masks['terminal_engagement_mask']

    # Generate plots and analysis
    trajectory_data = [{
        'ulog': ulog,
        'data': data,
        'mask': mask,
        'engagement_masks': engagement_masks,
        'filename': args.ulog.name
    }]
    plot_3d_terminal_engagement(trajectory_data, args.output, args.target, args.interactive_3d)
    plot_roll_pitch(trajectory_data, args.output, args.interactive_3d, args.target)
    
    # Create GPS trajectory plot for single file (shows entire trajectory)
    gps_trajectory_plot(trajectory_data, args.output, args.target, args.interactive_3d)
    
    # Create accelerometer impact plot to validate impact detection
    plot_accelerometer_impacts(trajectory_data, args.output, args.target, args.interactive_3d)

    if args.save_csv:
        save_filtered_csv(data, mask, args.output / "position_attitude_filtered.csv")
    
    # Combine all PNG plots into a single PDF
    combine_plots_to_pdf(args.output, args.target, is_batch=False)



if __name__ == "__main__":  # pragma: no cover
    main()
