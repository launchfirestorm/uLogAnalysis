

from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Import our custom modules
from tg_plotting import (
    set_matplotlib_backend as plotting_set_backend, plot_3d_terminal_engagement, plot_3d_position,
    plot_rpy_timeseries, plot_pitch_trajectory, plot_position_trajectories,
    create_combined_gps_trajectory_plot, create_combined_3d_plot
)
from tg_utils import (
    quaternion_to_euler as utils_quaternion_to_euler, 
    quaternion_to_euler_degrees as utils_quaternion_to_euler_degrees, 
    load_log as utils_load_log,
    extract_position_attitude as utils_extract_position_attitude, 
    compute_terminal_engagement_mask as utils_compute_terminal_engagement_mask,
    calculate_miss_distances as utils_calculate_miss_distances
)

# Global flag for interactive mode - will be set when parsing arguments
_INTERACTIVE_MODE = False

def set_matplotlib_backend(interactive: bool = False):
    """Set matplotlib backend based on interactive mode."""
    global _INTERACTIVE_MODE
    _INTERACTIVE_MODE = interactive
    # Use the function from our plotting module
    plotting_set_backend(interactive)

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


# # Use the function from tg_utils module
quaternion_to_euler = utils_quaternion_to_euler


# # Use the function from tg_utils module
quaternion_to_euler_degrees = utils_quaternion_to_euler_degrees


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
    
    # Extract position setpoint index and coordinates if available
    pos_sp_index_i = np.zeros(len(t_pos), dtype=int)
    pos_sp_lat_i = np.zeros(len(t_pos))
    pos_sp_lon_i = np.zeros(len(t_pos))
    pos_sp_alt_i = np.zeros(len(t_pos))
    pos_sp_x_i = np.zeros(len(t_pos))
    pos_sp_y_i = np.zeros(len(t_pos))
    pos_sp_z_i = np.zeros(len(t_pos))

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
        print(f"Accelerometer data interpolated. Magnitude range: [{np.min(accel_magnitude_i):.2f}, {np.max(accel_magnitude_i):.2f}] m/s²")
    except Exception as e:
        print(f"Accelerometer data not available: {e}")
    
    # # Extract landing detection status if available
    # try:
    #     land_data = ulog.get_dataset('vehicle_land_detected').data
    #     land_timestamps = land_data['timestamp'].astype(np.int64)
        
    #     # Interpolate landing detection data
    #     landed_i = np.interp(t_pos, land_timestamps, land_data['landed'])
    #     ground_contact_i = np.interp(t_pos, land_timestamps, land_data['ground_contact']) 
    #     print(f"Landing detection data interpolated. Landed range: [{np.min(landed_i):.1f}, {np.max(landed_i):.1f}]")
    # except Exception as e:
    #     print(f"Landing detection data not available: {e}")
    
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
        "pos_sp_index": pos_sp_index_i,
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


def compute_terminal_engagement_mask(data: Dict[str, np.ndarray], pitch_threshold: float = -4.0,
                                    accel_threshold: float = 15.0, alt_threshold: float = 5.0) -> tuple[np.ndarray, Dict]:
    """Compute boolean mask for terminal engagement segments (dive to impact).

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Extracted data dictionary.
    pitch_threshold : float
        Pitch angle threshold (deg) for dive detection.
    accel_threshold : float
        Acceleration threshold for impact detection (m/s²).
    alt_threshold : float  
        Altitude threshold for impact detection (m above ground).

    Returns
    -------
    tuple[np.ndarray, Dict]
        Boolean mask for terminal engagement and engagement masks dictionary
    """
    engagement_masks = compute_engagement_masks(data, pitch_threshold, accel_threshold, alt_threshold)
    return engagement_masks['terminal_engagement_mask'], engagement_masks


def detect_ground_impact(data: Dict[str, np.ndarray], accel_threshold: float = 15.0, alt_threshold: float = 5.0) -> np.ndarray:
    """
    Detect ground impact using accelerometer and altitude data.
    
    Args:
        data: Data dictionary containing accelerometer and altitude data
        accel_threshold: Acceleration magnitude threshold for impact (m/s²)
        alt_threshold: Altitude threshold - only consider impacts below this height (m)
    
    Returns:
        Boolean array indicating impact points
    """
    impact_mask = np.zeros(len(data['timestamp_us']), dtype=bool)
    
    # Method 1: High acceleration spikes at low altitude
    high_accel = data['accel_magnitude'] > accel_threshold
    low_altitude = data['altitude_agl'] < alt_threshold
    
    # Method 2: Sudden velocity changes (deceleration) 
    if 'vertical_velocity' in data:
        # Detect sudden changes in vertical velocity (impact deceleration)
        vz_diff = np.diff(data['vertical_velocity'], prepend=data['vertical_velocity'][0])
        sudden_decel = np.abs(vz_diff) > 5.0  # m/s velocity change threshold
        
        # Combine criteria: high acceleration OR sudden deceleration, both at low altitude
        impact_candidates = (high_accel | sudden_decel) & low_altitude
    else:
        impact_candidates = high_accel & low_altitude
    
    # Method 3: Use landing detection if available
    if 'landed' in data and np.any(data['landed'] > 0.5):
        # Find transitions from not-landed to landed
        landed_transitions = np.diff(data['landed'] > 0.5, prepend=False)
        impact_by_landing = landed_transitions & (data['altitude_agl'] < alt_threshold * 2)
        impact_candidates = impact_candidates | impact_by_landing
    
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
        
        impact_mask[filtered_impacts] = True
        
        # print(f"Impact detection: Found {len(filtered_impacts)} impact events")
        # print(f"  - Acceleration threshold: {accel_threshold} m/s²")
        # print(f"  - Altitude threshold: {alt_threshold} m") 
        # impact_times = [(data['timestamp_us'][idx] - data['timestamp_us'][0]) * 1e-6 for idx in filtered_impacts]
        # print(f"  - Impact times (relative): {[f'{t:.1f}s' for t in impact_times]}")
    else:
        print("Impact detection: No impact events detected")
    
    return impact_mask


def compute_engagement_masks(data: Dict[str, np.ndarray], pitch_threshold: float = -4.0,
                           accel_threshold: float = 15.0, alt_threshold: float = 5.0) -> Dict[str, np.ndarray]:
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
    
    # Compute dive mask and transitions
    dive_mask = data['pitch_deg'] < pitch_threshold
    dive_transitions = np.diff(dive_mask.astype(int), prepend=0)
    dive_start_indices = np.where(dive_transitions == 1)[0]
    
    # Compute impact mask
    impact_mask = detect_ground_impact(data, accel_threshold, alt_threshold)
    impact_indices = np.where(impact_mask)[0]
    
    # Find first terminal engagement
    terminal_engagement_mask = np.zeros(n_samples, dtype=bool)
    first_dive_idx = None
    first_impact_idx = None
    
    if len(dive_start_indices) > 0 and len(impact_indices) > 0:
        first_dive_idx = dive_start_indices[0]
        future_impacts = impact_indices[impact_indices > first_dive_idx]
        
        if len(future_impacts) > 0:
            first_impact_idx = future_impacts[0]
            
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
    print(f"  - Dive threshold: pitch < {pitch_threshold}°")
    print(f"  - Impact threshold: accel > {accel_threshold} m/s², alt < {alt_threshold} m")
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
                           accel_threshold: float = 15.0, alt_threshold: float = 5.0,
                           target_selection: str = "both") -> Dict[str, Dict[str, float]]:
    """Calculate miss distances for targets without plotting.
    
    Returns:
        Dictionary with target names as keys and miss distance info as values
    """
    # Define target locations (hardcoded) - altitudes are AGL (ground level)
    all_targets = {
        'Container': {'lat': 43.2222722, 'lon': -75.3903593, 'alt_agl': 0.0},
        'Van': {'lat': 43.2221788, 'lon': -75.3905151, 'alt_agl': 0.0}
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
        masks = compute_engagement_masks(data, pitch_threshold, accel_threshold, alt_threshold)
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
                    
                    # Calculate miss distances using Haversine formula
                    def haversine_distance(lat1, lon1, lat2, lon2):
                        import math
                        R = 6371000  # Earth's radius in meters
                        
                        lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
                        lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
                        
                        dlat = lat2_rad - lat1_rad
                        dlon = lon2_rad - lon1_rad
                        
                        a = (math.sin(dlat/2)**2 + 
                             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2)
                        c = 2 * math.asin(math.sqrt(a))
                        
                        return R * c
                    
                    for target_name, target_info in targets.items():
                        ground_distance = haversine_distance(impact_lat, impact_lon, 
                                                           target_info['lat'], target_info['lon'])
                        alt_diff = impact_alt_agl - target_info['alt_agl']
                        total_distance = math.sqrt(ground_distance**2 + alt_diff**2)
                        
                        miss_distances[target_name] = {
                            'ground': ground_distance,
                            'total': total_distance,
                            'altitude_diff': alt_diff,
                            'impact_lat': impact_lat,
                            'impact_lon': impact_lon,
                            'impact_alt_agl': impact_alt_agl
                        }
                        
            except Exception as e:
                print(f"Could not calculate miss distances: {e}")
    
    return miss_distances


def process_multiple_logs(logs_dir: Path, output_dir: Path, 
                         pitch_threshold: float = -4.0, accel_threshold: float = 15.0, 
                         alt_threshold: float = 5.0, plot_combined_3d: bool = False,
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
    trajectory_data = []  # For combined 3D plotting
    
    for i, log_file in enumerate(log_files, 1):
        print(f"\n[{i}/{len(log_files)}] Processing: {log_file.name}")
        
        try:
            # Load and process log
            ulog = load_log(log_file)
            data = extract_position_attitude(ulog)
            
            # Calculate terminal engagement mask
            mask, engagement_masks = compute_terminal_engagement_mask(data, pitch_threshold, 
                                                                    accel_threshold, alt_threshold)
            
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
                                                    pitch_threshold, accel_threshold, alt_threshold,
                                                    target_selection)
            
            # Collect trajectory data for combined plotting (regardless of miss distance calculation success)
            if plot_combined_3d and np.any(mask):
                trajectory_data.append({
                    'filename': log_file.name,
                    'data': {
                        'x': data['x'][mask],
                        'y': data['y'][mask], 
                        'z': data['altitude_agl'][mask]  # Use AGL altitude for consistency
                    },
                    'miss_distances': miss_distances if miss_distances else {}
                })
                print(f"  - Added trajectory data for {log_file.name} ({np.sum(mask)} points)")
            
            if miss_distances:
                print(f"  Terminal engagement found:")
                for target_name, distances in miss_distances.items():
                    ground_dist = distances['ground']
                    print(f"    {target_name}: {ground_dist:.1f}m miss")
                    all_miss_distances[target_name].append(ground_dist)
                
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
    for target_name, distances in all_miss_distances.items():
        if distances:
            distances_array = np.array(distances)
            mean_dist = np.mean(distances_array)
            std_dist = np.std(distances_array)
            min_dist = np.min(distances_array)
            max_dist = np.max(distances_array)
            
            print(f"\n{target_name} Target:")
            print(f"  Successful engagements: {len(distances)}")
            print(f"  Mean miss distance: {mean_dist:.1f}m")
            print(f"  Standard deviation: {std_dist:.1f}m")
            print(f"  Minimum miss: {min_dist:.1f}m")
            print(f"  Maximum miss: {max_dist:.1f}m")
            
            stats_output.append({
                'target': target_name,
                'count': len(distances),
                'mean': mean_dist,
                'std': std_dist,
                'min': min_dist,
                'max': max_dist,
                'distances': distances
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
        
        # Dynamic header based on target selection
        header = ['File', 'Status']
        if target_selection == "Container":
            header.append('Container_Miss_m')
        else:  # Van
            header.append('Van_Miss_m')
        header.extend(['Impact_Lat', 'Impact_Lon', 'Impact_Alt_AGL_m'])
        writer.writerow(header)
        
        for result in log_results:
            row = [result['file'], result['status']]
            if result['miss_distances']:
                if target_selection == "both":
                    container_miss = result['miss_distances'].get('Container', {}).get('ground', '')
                    greenvan_miss = result['miss_distances'].get('Van', {}).get('ground', '')
                    row.extend([container_miss, greenvan_miss])
                elif target_selection == "Container":
                    container_miss = result['miss_distances'].get('Container', {}).get('ground', '')
                    row.append(container_miss)
                else:  # Van
                    greenvan_miss = result['miss_distances'].get('Van', {}).get('ground', '')
                    row.append(greenvan_miss)
                
                # Use first available target for impact coordinates
                first_target = next(iter(result['miss_distances'].values()), {})
                impact_lat = first_target.get('impact_lat', '')
                impact_lon = first_target.get('impact_lon', '')
                impact_alt_agl = first_target.get('impact_alt_agl', '')
                row.extend([impact_lat, impact_lon, impact_alt_agl])
            else:
                # Fill empty columns based on target selection
                empty_count = 1 if target_selection != "both" else 2
                row.extend([''] * (empty_count + 3))  # miss distances + impact coordinates
            writer.writerow(row)
    
    print(f"\nDetailed results saved to: {results_file} (Target: {target_selection})")
    
    # Summary statistics
    stats_file = output_dir / "miss_distance_statistics.csv"
    with open(stats_file, 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Target', 'Count', 'Mean_m', 'StdDev_m', 'Min_m', 'Max_m'])
        
        for stat in stats_output:
            writer.writerow([stat['target'], stat['count'], f"{stat['mean']:.1f}", 
                           f"{stat['std']:.1f}", f"{stat['min']:.1f}", f"{stat['max']:.1f}"])
    
    print(f"Summary statistics saved to: {stats_file}")
    print(f"\nTotal logs processed: {len(log_files)}")
    print(f"Successful engagements: {len([r for r in log_results if r['status'] == 'Success'])}")
    
    # Create combined GPS trajectory plot for visual debugging
    create_combined_gps_trajectory_plot(log_files, output_dir, target_selection, 
                                      pitch_threshold, accel_threshold, alt_threshold, interactive_3d)
    
    # Create combined 3D plot if requested
    if plot_combined_3d and trajectory_data:
        create_combined_3d_plot(trajectory_data, all_miss_distances, output_dir, target_selection, interactive_3d)


def main():
    parser = argparse.ArgumentParser(description="Analyze terminal engagement (dive-to-impact) segments from PX4 flight logs.")
    parser.add_argument("--ulog", type=Path, help="Path to PX4 .ulg log file (required for single file mode)")
    parser.add_argument("--logs-dir", type=Path, help="Path to directory containing multiple .ulg files for batch processing")
    parser.add_argument("--output", type=Path, default=Path("./tg_analysis_output"), help="Output directory for plots & data")
    parser.add_argument("--pitch-threshold", type=float, default=-4.0, help="Pitch threshold (deg) for dive detection. Default -4.0")
    parser.add_argument("--accel-threshold", type=float, default=15.0, help="Acceleration threshold for impact detection (m/s²). Default 15.0")
    parser.add_argument("--alt-threshold", type=float, default=5.0, help="Altitude threshold for impact detection (m above ground). Default 5.0")
    parser.add_argument("--save-csv", action="store_true", help="Always save filtered CSV alongside plots")
    parser.add_argument("--debug", action="store_true", help="Print debug information about available ULog datasets and fields")
    parser.add_argument("--show-setpoints", action="store_true", help="Overlay position setpoints on trajectory plots")
    parser.add_argument("--plot-combined-3d", action="store_true", help="For batch processing, create combined 3D plot of all trajectories with mean miss distance circles")
    parser.add_argument("--interactive-3d", action="store_true", help="Show interactive 3D plots that can be rotated and zoomed (requires display)")
    parser.add_argument("--target", choices=["Container", "Van"], required=True, help="Target selection: 'Container' or 'Van' (required)")
    
    args = parser.parse_args()
    
    # Set matplotlib backend based on interactive mode BEFORE any matplotlib imports
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
                            args.pitch_threshold, args.accel_threshold, args.alt_threshold,
                            args.plot_combined_3d, args.target, args.interactive_3d)
        return

    # Single file processing mode (original behavior)
    if not args.ulog.exists():
        raise SystemExit(f"ULog file not found: {args.ulog}")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Loading ULog: {args.ulog}")
    ulog = load_log(args.ulog)
    
    if args.debug:
        print_ulog_debug_info(ulog)

    data = extract_position_attitude(ulog)

    print(f"Computing terminal engagement segments (dive to impact)...")
    threshold_desc = f"terminal engagement (pitch < {args.pitch_threshold}° → accel > {args.accel_threshold} m/s²)"
        
    # Compute terminal engagement mask
    mask, engagement_masks = compute_terminal_engagement_mask(data, args.pitch_threshold, 
                                                            args.accel_threshold, args.alt_threshold)
    selected = int(np.count_nonzero(mask))
    print(f"Selected {selected} samples ({threshold_desc}).")

    # Generate plots and analysis
    plot_position_trajectories(ulog, data, args.output, args.show_setpoints, args.target)
    plot_3d_terminal_engagement(ulog, data, mask, args.output, args.target, args.show_setpoints, args.interactive_3d)
    
    # plot_3d_position(data, mask, args.output)
    plot_rpy_timeseries(data, mask, args.output)
    
    # Create GPS trajectory plot for single file
    create_combined_gps_trajectory_plot([args.ulog], args.output, args.target, 
                                       args.pitch_threshold, args.accel_threshold, 
                                       args.alt_threshold, args.interactive_3d)

    if args.save_csv:
        save_filtered_csv(data, mask, args.output / "position_attitude_filtered.csv")



if __name__ == "__main__":  # pragma: no cover
    main()
