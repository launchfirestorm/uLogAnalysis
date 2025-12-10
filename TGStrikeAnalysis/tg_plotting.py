"""Terminal Engagement Analysis - Plotting Functions

This module contains all matplotlib-based plotting functions for TG strike analysis.
Functions include 3D trajectory plots, combined plots, GPS trajectories, and timeseries plots.
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("Warning: matplotlib not available, plotting functions will not work.")





def plot_3d_terminal_engagement(trajectory_data: list, out_dir: Path, 
                                target_selection: str = "Container", 
                                interactive: bool = False, mean_miss_distance: float = None):
    """Create 3D trajectory plot for terminal engagement segments using GPS coordinates.
    
    For single file: Shows one trajectory with dive/impact markers.
    For batch: Shows all trajectories with mean miss distance circle around target.
    
    Args:
        trajectory_data: List of dicts with 'ulog', 'data', 'mask', 'engagement_masks', 'filename'
        out_dir: Output directory
        target_selection: Target name ('Container' or 'Van')
        interactive: Whether to show interactive plot
        mean_miss_distance: Mean miss distance for batch plots (meters)
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping 3D trajectory plot.")
        return
        
    try:
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("matplotlib not available, skipping 3D trajectory plot.")
        return
    
    # Ensure trajectory_data is a list
    if not isinstance(trajectory_data, list):
        trajectory_data = [trajectory_data]
    
    # Define target locations in GPS coordinates
    all_targets = {
        'Container': {'lat': 43.2222722, 'lon': -75.3903593, 'alt_agl': 2.0, 'color': 'red', 'marker': 's'},
        'Van': {'lat': 43.2221788, 'lon': -75.3905151, 'alt_agl': 1.0, 'color': 'green', 'marker': 'o'},
        'Conex': {'lat': 33.7578867, 'lon': -115.3099750, 'alt_agl': 2.583, 'color': 'orange', 'marker': '^'}
    }
    
    target_display = {target_selection: all_targets[target_selection]}
    
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectory_data)))
    is_batch = len(trajectory_data) > 1
    
    # Get reference origin from first trajectory's GPS data
    first_ulog = trajectory_data[0]['ulog']
    first_mask = trajectory_data[0]['mask']
    first_data = trajectory_data[0]['data']
    
    try:
        # Get GPS data from first trajectory
        gps_data = first_ulog.get_dataset("vehicle_gps_position").data
        valid_fix = gps_data['fix_type'] > 2
        
        if not np.any(valid_fix):
            print("Warning: No valid GPS fix found in first trajectory")
            return
        
        # Get GPS timestamps and find first point in masked region
        gps_timestamps = gps_data['timestamp'][valid_fix]
        first_mask_idx = np.where(first_mask)[0][0]
        first_timestamp = first_data['timestamp_us'][first_mask_idx]
        
        # Find corresponding GPS index
        ref_gps_idx = np.searchsorted(gps_timestamps, first_timestamp)
        ref_gps_idx = np.clip(ref_gps_idx, 0, len(gps_timestamps) - 1)
        
        # Use this as reference origin
        ref_lat = gps_data['lat'][valid_fix][ref_gps_idx] / 1e7
        ref_lon = gps_data['lon'][valid_fix][ref_gps_idx] / 1e7
        
    except Exception as e:
        print(f"Warning: Could not get GPS reference: {e}")
        return
    
    # Convert GPS to relative meters from reference point
    def gps_to_meters(lat, lon, ref_lat, ref_lon):
        """Convert GPS coordinates to meters relative to reference point."""
        import math
        R = 6371000  # Earth radius in meters
        
        # Simple flat-earth approximation (good for small distances)
        dlat = lat - ref_lat
        dlon = lon - ref_lon
        
        y = dlat * (math.pi / 180.0) * R  # North (meters)
        x = dlon * (math.pi / 180.0) * R * math.cos(math.radians(ref_lat))  # East (meters)
        
        return x, y
    
    # Convert target to relative coordinates
    target_x, target_y = gps_to_meters(
        all_targets[target_selection]['lat'],
        all_targets[target_selection]['lon'],
        ref_lat, ref_lon
    )
    
    # Process each trajectory
    for i, traj_info in enumerate(trajectory_data):
        ulog = traj_info['ulog']
        data = traj_info['data']
        mask = traj_info['mask']
        engagement_masks = traj_info['engagement_masks']
        filename = traj_info.get('filename', 'trajectory')
        
        if not np.any(mask):
            continue
        
        try:
            # Get GPS data for this trajectory
            gps_data = ulog.get_dataset("vehicle_gps_position").data
            valid_fix = gps_data['fix_type'] > 2
            
            if not np.any(valid_fix):
                print(f"  Warning: No valid GPS fix in {filename}")
                continue
            
            gps_lat = gps_data['lat'][valid_fix] / 1e7
            gps_lon = gps_data['lon'][valid_fix] / 1e7
            gps_timestamps = gps_data['timestamp'][valid_fix]
            
            # Get masked indices
            mask_indices = np.where(mask)[0]
            first_dive_idx = engagement_masks.get('first_dive_idx', mask_indices[0])
            first_impact_idx = engagement_masks.get('first_impact_idx', mask_indices[-1])
            
            # Get altitude offset from very first point in trajectory (ground level reference)
            altitude_offset = data["altitude_agl"][0]
            
            # Convert GPS coordinates to relative meters for masked segment
            trajectory_x = []
            trajectory_y = []
            trajectory_z = []
            
            for idx in mask_indices:
                timestamp = data['timestamp_us'][idx]
                gps_idx = np.searchsorted(gps_timestamps, timestamp)
                gps_idx = np.clip(gps_idx, 0, len(gps_timestamps) - 1)
                
                lat = gps_lat[gps_idx]
                lon = gps_lon[gps_idx]
                x, y = gps_to_meters(lat, lon, ref_lat, ref_lon)
                
                trajectory_x.append(x)
                trajectory_y.append(y)
                trajectory_z.append(data["altitude_agl"][idx] - altitude_offset)
            
            trajectory_x = np.array(trajectory_x)
            trajectory_y = np.array(trajectory_y)
            trajectory_z = np.array(trajectory_z)
            
            label = filename if is_batch else 'Terminal Engagement'
            color = colors[i] if is_batch else 'b'
            linewidth = 1.5 if is_batch else 2
            
            ax.plot(trajectory_x, trajectory_y, trajectory_z, color=color, linewidth=linewidth, 
                   alpha=0.7 if is_batch else 1.0, label=label)
            
            # Find positions within masked data
            dive_pos_in_mask = np.where(mask_indices == first_dive_idx)[0]
            impact_pos_in_mask = np.where(mask_indices == first_impact_idx)[0]
            
            # Mark dive start
            if len(dive_pos_in_mask) > 0:
                dive_idx = dive_pos_in_mask[0]
                ax.scatter(trajectory_x[dive_idx], trajectory_y[dive_idx], trajectory_z[dive_idx], 
                          color=color, s=100, marker='^', edgecolors='orange', linewidth=2, zorder=6)
            
            # Mark impact point
            if len(impact_pos_in_mask) > 0:
                impact_idx = impact_pos_in_mask[0]
                ax.scatter(trajectory_x[impact_idx], trajectory_y[impact_idx], trajectory_z[impact_idx], 
                          color=color, s=120, marker='X', edgecolors='red', linewidth=2, zorder=6)
                
        except Exception as e:
            print(f"  Error plotting trajectory for {filename}: {e}")
            continue
    
    # Plot target (altitude is relative to ground, so z=0)
    for target_name, target_info in target_display.items():
        ax.scatter(target_x, target_y, 0, 
                  color=target_info['color'], s=250, marker=target_info['marker'], 
                  label=f'{target_name} Target', zorder=10, edgecolors='black', linewidth=2)
    
    # Add mean miss distance circle for batch plots
    if is_batch and mean_miss_distance is not None:
        # Create circle at ground level around target
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = target_x + mean_miss_distance * np.cos(theta)
        circle_y = target_y + mean_miss_distance * np.sin(theta)
        circle_z = np.zeros_like(theta)
        ax.plot(circle_x, circle_y, circle_z, 'gray', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Mean Miss: {mean_miss_distance:.1f}m')
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Altitude AGL [m]')
    
    title = f'3D Terminal Engagement Trajectories (Target: {target_selection})' if is_batch else f'3D Terminal Engagement Trajectory (Target: {target_selection})'
    ax.set_title(title)
    
    # Adjust legend for batch mode
    if is_batch:
        ax.legend(loc='best', fontsize=8, ncol=2)
    else:
        ax.legend(loc='best')
    
    ax.grid(True, alpha=0.3)
    
    # Set view to show trajectories better
    ax.view_init(elev=20, azim=45)
    
    if interactive:
        plt.show()
        print("Showing interactive 3D terminal engagement plot (close window to continue)")
    else:
        # Save plot
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        prefix = "combined_3d_terminal_engagement" if is_batch else "3d_terminal_engagement"
        plot_path = out_dir / f"{prefix}{target_suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_roll_pitch(trajectory_data: list, out_dir: Path, 
                   interactive: bool = False, target_selection: str = "Van"):
    """Create roll, pitch, and yaw timeseries plots with setpoints during terminal engagement for all trajectories.
    
    Creates subplots showing roll, pitch, and yaw for each trajectory with offboard mode regions marked.
    
    Args:
        trajectory_data: List of dicts containing trajectory info
        out_dir: Output directory
        interactive: Whether to show interactive plot
        target_selection: Target name for filename
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping roll/pitch timeseries plot.")
        return
        
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping roll/pitch timeseries plot.")
        return
    
    # Ensure trajectory_data is a list
    if not isinstance(trajectory_data, list):
        trajectory_data = [trajectory_data]
    
    # Filter to only trajectories with terminal engagement
    valid_trajectories = [t for t in trajectory_data if np.any(t['mask'])]
    
    if len(valid_trajectories) == 0:
        print("No terminal engagement data to plot for roll/pitch/yaw timeseries.")
        return
    
    n_trajectories = len(valid_trajectories)
    
    # Create subplots - 3 columns (roll, pitch, yaw) per trajectory
    fig, axes = plt.subplots(n_trajectories, 3, figsize=(22, 4 * n_trajectories), sharex='col')
    if n_trajectories == 1:
        axes = axes.reshape(1, -1)
    
    for idx, traj_info in enumerate(valid_trajectories):
        data = traj_info['data']
        mask = traj_info['mask']
        engagement_masks = traj_info['engagement_masks']
        filename = traj_info.get('filename', f'trajectory_{idx+1}')
        
        # Get indices within the mask for marking dive and impact
        mask_indices = np.where(mask)[0]
        first_dive_idx = engagement_masks.get('first_dive_idx', mask_indices[0])
        first_impact_idx = engagement_masks.get('first_impact_idx', mask_indices[-1])
        
        # Find positions within masked data
        dive_pos_in_mask = np.where(mask_indices == first_dive_idx)[0]
        impact_pos_in_mask = np.where(mask_indices == first_impact_idx)[0]
        
        # Create time array relative to start of engagement
        time_engagement = (data["timestamp_us"][mask] - data["timestamp_us"][mask][0]) * 1e-6
        
        # Roll subplot
        ax_roll = axes[idx, 0]
        ax_roll.plot(time_engagement, data["roll_deg"][mask], 'b-', linewidth=2, label='Roll')
        if 'roll_sp_deg' in data and np.any(data["roll_sp_deg"][mask] != 0):
            ax_roll.plot(time_engagement, data["roll_sp_deg"][mask], 'b--', linewidth=1.5, alpha=0.7, label='Roll Setpoint')
        
        # Mark offboard mode regions on roll plot
        if 'offboard_mode' in data and np.any(data['offboard_mode'][mask]):
            offboard_in_mask = data['offboard_mode'][mask]
            # Find continuous offboard regions
            offboard_transitions = np.diff(offboard_in_mask.astype(int), prepend=0, append=0)
            offboard_starts = np.where(offboard_transitions == 1)[0]
            offboard_ends = np.where(offboard_transitions == -1)[0]
            
            for start_idx, end_idx in zip(offboard_starts, offboard_ends):
                ax_roll.axvspan(time_engagement[start_idx], time_engagement[end_idx-1], 
                               alpha=0.2, color='cyan', label='Offboard Mode' if start_idx == offboard_starts[0] else '')
        
        # Mark dive and impact on roll plot
        if len(dive_pos_in_mask) > 0:
            dive_idx = dive_pos_in_mask[0]
            ax_roll.axvline(x=time_engagement[dive_idx], color='orange', linestyle=':', alpha=0.7, linewidth=2)
            ax_roll.scatter(time_engagement[dive_idx], data["roll_deg"][mask][dive_idx], 
                           color='orange', s=100, marker='^', edgecolors='black', linewidth=1.5, zorder=5)
        if len(impact_pos_in_mask) > 0:
            impact_idx = impact_pos_in_mask[0]
            ax_roll.axvline(x=time_engagement[impact_idx], color='red', linestyle=':', alpha=0.7, linewidth=2)
            ax_roll.scatter(time_engagement[impact_idx], data["roll_deg"][mask][impact_idx], 
                           color='red', s=120, marker='X', edgecolors='black', linewidth=1.5, zorder=5)
        
        ax_roll.set_ylabel('Roll [deg]')
        ax_roll.set_title(f'{filename} - Roll', fontsize=10)
        ax_roll.grid(True, alpha=0.3)
        ax_roll.legend(loc='best', fontsize=8)
        if idx == n_trajectories - 1:
            ax_roll.set_xlabel('Time [s]')
        
        # Pitch subplot
        ax_pitch = axes[idx, 1]
        ax_pitch.plot(time_engagement, data["pitch_deg"][mask], 'g-', linewidth=2, label='Pitch')
        if 'pitch_sp_deg' in data and np.any(data["pitch_sp_deg"][mask] != 0):
            ax_pitch.plot(time_engagement, data["pitch_sp_deg"][mask], 'g--', linewidth=1.5, alpha=0.7, label='Pitch Setpoint')
        
        # Mark offboard mode regions on pitch plot
        if 'offboard_mode' in data and np.any(data['offboard_mode'][mask]):
            offboard_in_mask = data['offboard_mode'][mask]
            offboard_transitions = np.diff(offboard_in_mask.astype(int), prepend=0, append=0)
            offboard_starts = np.where(offboard_transitions == 1)[0]
            offboard_ends = np.where(offboard_transitions == -1)[0]
            
            for start_idx, end_idx in zip(offboard_starts, offboard_ends):
                ax_pitch.axvspan(time_engagement[start_idx], time_engagement[end_idx-1], 
                                alpha=0.2, color='cyan', label='Offboard Mode' if start_idx == offboard_starts[0] else '')
        
        # Mark dive and impact on pitch plot
        if len(dive_pos_in_mask) > 0:
            dive_idx = dive_pos_in_mask[0]
            ax_pitch.axvline(x=time_engagement[dive_idx], color='orange', linestyle=':', alpha=0.7, linewidth=2, label='Dive Start')
            ax_pitch.scatter(time_engagement[dive_idx], data["pitch_deg"][mask][dive_idx], 
                            color='orange', s=100, marker='^', edgecolors='black', linewidth=1.5, zorder=5)
        if len(impact_pos_in_mask) > 0:
            impact_idx = impact_pos_in_mask[0]
            ax_pitch.axvline(x=time_engagement[impact_idx], color='red', linestyle=':', alpha=0.7, linewidth=2, label='Impact')
            ax_pitch.scatter(time_engagement[impact_idx], data["pitch_deg"][mask][impact_idx], 
                            color='red', s=120, marker='X', edgecolors='black', linewidth=1.5, zorder=5)
        
        ax_pitch.set_ylabel('Pitch [deg]')
        ax_pitch.set_title(f'{filename} - Pitch', fontsize=10)
        ax_pitch.grid(True, alpha=0.3)
        ax_pitch.axhline(y=-4.0, color='purple', linestyle='--', alpha=0.5, linewidth=1.5, label='Dive threshold')
        ax_pitch.legend(loc='best', fontsize=8)
        if idx == n_trajectories - 1:
            ax_pitch.set_xlabel('Time [s]')
        
        # Yaw subplot
        ax_yaw = axes[idx, 2]
        ax_yaw.plot(time_engagement, data["yaw_deg"][mask], 'r-', linewidth=2, label='Yaw')
        if 'yaw_sp_deg' in data and np.any(data["yaw_sp_deg"][mask] != 0):
            ax_yaw.plot(time_engagement, data["yaw_sp_deg"][mask], 'r--', linewidth=1.5, alpha=0.7, label='Yaw Setpoint')
        
        # Mark offboard mode regions on yaw plot
        if 'offboard_mode' in data and np.any(data['offboard_mode'][mask]):
            offboard_in_mask = data['offboard_mode'][mask]
            offboard_transitions = np.diff(offboard_in_mask.astype(int), prepend=0, append=0)
            offboard_starts = np.where(offboard_transitions == 1)[0]
            offboard_ends = np.where(offboard_transitions == -1)[0]
            
            for start_idx, end_idx in zip(offboard_starts, offboard_ends):
                ax_yaw.axvspan(time_engagement[start_idx], time_engagement[end_idx-1], 
                               alpha=0.2, color='cyan', label='Offboard Mode' if start_idx == offboard_starts[0] else '')
        
        # Mark dive and impact on yaw plot
        if len(dive_pos_in_mask) > 0:
            dive_idx = dive_pos_in_mask[0]
            ax_yaw.axvline(x=time_engagement[dive_idx], color='orange', linestyle=':', alpha=0.7, linewidth=2)
            ax_yaw.scatter(time_engagement[dive_idx], data["yaw_deg"][mask][dive_idx], 
                          color='orange', s=100, marker='^', edgecolors='black', linewidth=1.5, zorder=5)
        if len(impact_pos_in_mask) > 0:
            impact_idx = impact_pos_in_mask[0]
            ax_yaw.axvline(x=time_engagement[impact_idx], color='red', linestyle=':', alpha=0.7, linewidth=2)
            ax_yaw.scatter(time_engagement[impact_idx], data["yaw_deg"][mask][impact_idx], 
                          color='red', s=120, marker='X', edgecolors='black', linewidth=1.5, zorder=5)
        
        ax_yaw.set_ylabel('Yaw [deg]')
        ax_yaw.set_title(f'{filename} - Yaw', fontsize=10)
        ax_yaw.grid(True, alpha=0.3)
        ax_yaw.legend(loc='best', fontsize=8)
        if idx == n_trajectories - 1:
            ax_yaw.set_xlabel('Time [s]')
    
    plt.tight_layout()
    
    if interactive:
        plt.show()
        print("Showing interactive roll/pitch/yaw timeseries plot (close window to continue)")
    else:
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        prefix = "combined_roll_pitch_yaw" if n_trajectories > 1 else "roll_pitch_yaw_timeseries"
        plot_path = out_dir / f"{prefix}{target_suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()


def gps_trajectory_plot(trajectory_data: list, output_dir: Path, target_selection: str,
                        interactive: bool = False) -> None:
    """Create GPS trajectory plots showing entire flight with dive/impact markers and zoomed views.
    
    Creates 4 subplots:
    - Top left: Full lat/lon trajectory
    - Top right: Zoomed lat/lon around impact point(s)
    - Bottom left: Full altitude profile
    - Bottom right: Zoomed altitude around impact (terminal phase)
    
    Works for both single file and batch processing.
    
    Args:
        trajectory_data: List of dicts containing:
            - 'filename': Name of the log file
            - 'ulog': Loaded ulog object
            - 'data': Extracted data dictionary
            - 'mask': Terminal engagement mask
            - 'engagement_masks': Dictionary with dive/impact indices
        output_dir: Directory to save plots
        target_selection: Target name ('Container' or 'Van')
        interactive: Whether to show interactive plots
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping GPS trajectory plot.")
        return
        
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping GPS trajectory plot.")
        return
    
    # Ensure trajectory_data is a list
    if not isinstance(trajectory_data, list):
        trajectory_data = [trajectory_data]
    
    # Define target locations for reference
    all_targets = {
        'Container': {'lat': 43.2222722, 'lon': -75.3903593, 'alt_agl': 0.0},
        'Van': {'lat': 43.2221788, 'lon': -75.3905151, 'alt_agl': 0.0},
        'Conex': {'lat': 33.7578867, 'lon': -115.3099750, 'alt_agl': 0.0}
    }
    target_info = all_targets[target_selection]
    
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    ax_latlon = axes[0, 0]
    ax_latlon_zoom = axes[0, 1]
    ax_alt = axes[1, 0]
    ax_alt_zoom = axes[1, 1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectory_data)))
    
    file_word = "file" if len(trajectory_data) == 1 else "files"
    # print(f"\n=== Creating GPS Trajectory Plot for {len(trajectory_data)} {file_word} ===")
    
    # Track all impact points for zoom calculation
    all_impact_lats = []
    all_impact_lons = []
    all_impact_times = []
    all_impact_alts = []
    
    for i, traj_info in enumerate(trajectory_data):
        try:
            # Extract pre-computed data
            filename = traj_info['filename']
            ulog = traj_info['ulog']
            data = traj_info['data']
            mask = traj_info['mask']
            engagement_masks = traj_info['engagement_masks']
            
            if not np.any(mask):
                print(f"  Skipping {filename}: No terminal engagement")
                continue
                
            # Get GPS data (entire trajectory)
            try:
                gps_data = ulog.get_dataset("vehicle_gps_position").data
                valid_fix = gps_data['fix_type'] > 2
                
                if not np.any(valid_fix):
                    print(f"  Warning: No valid GPS fix in {filename}")
                    continue
                    
                gps_lat = gps_data['lat'][valid_fix] / 1e7
                gps_lon = gps_data['lon'][valid_fix] / 1e7
                gps_alt_msl = gps_data['alt'][valid_fix] / 1e3
                gps_timestamps = gps_data['timestamp'][valid_fix]
                
                # Convert to AGL using very first altitude point as ground reference
                # Find GPS altitude at very first timestamp in data
                first_timestamp = data['timestamp_us'][0]
                first_gps_idx = np.searchsorted(gps_timestamps, first_timestamp)
                first_gps_idx = np.clip(first_gps_idx, 0, len(gps_timestamps) - 1)
                altitude_offset = gps_alt_msl[first_gps_idx]
                
                gps_alt = gps_alt_msl - altitude_offset
                
                # Get time array for altitude plot
                gps_t = (gps_timestamps - gps_timestamps[0]) * 1e-6
            except Exception as e:
                print(f"  Warning: Could not extract GPS data from {filename}: {e}")
                continue
            
            # Plot entire trajectory on main plots
            label = filename if len(trajectory_data) > 1 else 'Trajectory'
            ax_latlon.plot(gps_lon, gps_lat, color=colors[i], linewidth=1.5, alpha=0.7, label=label)
            ax_alt.plot(gps_t, gps_alt, color=colors[i], linewidth=1.5, alpha=0.7, label=label)
            
            # Also plot on zoomed views
            ax_latlon_zoom.plot(gps_lon, gps_lat, color=colors[i], linewidth=2, alpha=0.8, label=label)
            ax_alt_zoom.plot(gps_t, gps_alt, color=colors[i], linewidth=2, alpha=0.8, label=label)
            
            # Mark dive start and impact if available
            first_dive_idx = engagement_masks.get('first_dive_idx')
            first_impact_idx = engagement_masks.get('first_impact_idx')
            
            dive_lat, dive_lon = None, None
            impact_lat, impact_lon = None, None
            
            if first_dive_idx is not None:
                # Find corresponding GPS index
                dive_gps_idx = np.searchsorted(gps_timestamps, data["timestamp_us"][first_dive_idx])
                dive_gps_idx = np.clip(dive_gps_idx, 0, len(gps_timestamps) - 1)
                
                dive_lat = gps_lat[dive_gps_idx]
                dive_lon = gps_lon[dive_gps_idx]
                dive_time = gps_t[dive_gps_idx]
                dive_alt = gps_alt[dive_gps_idx]
                
                # Mark on all plots
                for ax in [ax_latlon, ax_latlon_zoom]:
                    ax.scatter(dive_lon, dive_lat, color=colors[i], s=120, marker='^', 
                             edgecolors='orange', linewidth=2, zorder=6)
                
                for ax in [ax_alt, ax_alt_zoom]:
                    ax.scatter(dive_time, dive_alt, color=colors[i], s=120, marker='^',
                             edgecolors='orange', linewidth=2, zorder=6)
            
            if first_impact_idx is not None:
                # Find corresponding GPS index
                impact_gps_idx = np.searchsorted(gps_timestamps, data["timestamp_us"][first_impact_idx])
                impact_gps_idx = np.clip(impact_gps_idx, 0, len(gps_timestamps) - 1)
                
                impact_lat = gps_lat[impact_gps_idx]
                impact_lon = gps_lon[impact_gps_idx]
                impact_time = gps_t[impact_gps_idx]
                impact_alt = gps_alt[impact_gps_idx]
                
                # Store for zoom calculation
                all_impact_lats.append(impact_lat)
                all_impact_lons.append(impact_lon)
                all_impact_times.append(impact_time)
                all_impact_alts.append(impact_alt)
                
                # Mark on all plots
                for ax in [ax_latlon, ax_latlon_zoom]:
                    ax.scatter(impact_lon, impact_lat, color=colors[i], s=150, marker='X',
                             edgecolors='red', linewidth=2, zorder=6)
                
                for ax in [ax_alt, ax_alt_zoom]:
                    ax.scatter(impact_time, impact_alt, color=colors[i], s=150, marker='X',
                             edgecolors='red', linewidth=2, zorder=6)
            
            # Print summary
            # if dive_lat is not None and impact_lat is not None:
            #     print(f"  Processed {filename}: Dive at ({dive_lat:.7f}, {dive_lon:.7f}), Impact at ({impact_lat:.7f}, {impact_lon:.7f})")
            # elif dive_lat is not None or impact_lat is not None:
            #     print(f"  Processed {filename}: Partial terminal engagement detected")
            # else:
            #     print(f"  Processed {filename}: No terminal engagement detected")
                
        except Exception as e:
            print(f"  Error processing {filename} for GPS plot: {e}")
            continue
    
    # Plot target on all lat/lon plots
    for ax in [ax_latlon, ax_latlon_zoom]:
        ax.scatter(target_info['lon'], target_info['lat'], 
                  color='green', s=250, marker='s', 
                  label=f'{target_selection} Target', zorder=10, edgecolors='black', linewidth=2)
    
    # Configure full lat/lon subplot
    ax_latlon.set_xlabel('Longitude [deg]')
    ax_latlon.set_ylabel('Latitude [deg]')
    ax_latlon.set_title(f'GPS Trajectories - Lat/Lon (Target: {target_selection})')
    ax_latlon.grid(True, alpha=0.3)
    ax_latlon.legend(loc='best', fontsize=8 if len(trajectory_data) > 1 else 10)
    
    # Configure zoomed lat/lon subplot
    if all_impact_lats:
        # Calculate zoom bounds around impact points
        impact_lats = np.array(all_impact_lats)
        impact_lons = np.array(all_impact_lons)
        
        # Add target to zoom calculation
        lat_min = min(impact_lats.min(), target_info['lat']) - 0.0001
        lat_max = max(impact_lats.max(), target_info['lat']) + 0.0001
        lon_min = min(impact_lons.min(), target_info['lon']) - 0.0001
        lon_max = max(impact_lons.max(), target_info['lon']) + 0.0001
        
        # Add some padding
        lat_padding = (lat_max - lat_min) * 0.3
        lon_padding = (lon_max - lon_min) * 0.3
        
        ax_latlon_zoom.set_xlim(lon_min - lon_padding, lon_max + lon_padding)
        ax_latlon_zoom.set_ylim(lat_min - lat_padding, lat_max + lat_padding)
    
    ax_latlon_zoom.set_xlabel('Longitude [deg]')
    ax_latlon_zoom.set_ylabel('Latitude [deg]')
    ax_latlon_zoom.set_title('Zoomed View - Impact Area')
    ax_latlon_zoom.grid(True, alpha=0.3)
    
    # Configure full altitude subplot
    ax_alt.set_xlabel('Time [s]')
    ax_alt.set_ylabel('Altitude [m AGL]')
    ax_alt.set_title('Altitude Profiles')
    ax_alt.grid(True, alpha=0.3)
    ax_alt.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Ground Level')
    ax_alt.legend(loc='best', fontsize=8 if len(trajectory_data) > 1 else 10)
    
    # Configure zoomed altitude subplot
    if all_impact_times:
        impact_times = np.array(all_impact_times)
        impact_alts = np.array(all_impact_alts)
        
        # Zoom to last 3 seconds before impact and 1 second after
        time_min = impact_times.min() - 1.0
        time_max = impact_times.max() + 1.0
        alt_min = min(-15.0, impact_alts.min()) - 10.0  # Show slightly below ground
        alt_max = max(15.0, impact_alts.max() + 10.0)  # Show up to 15m or impact alt + 10m
        
        ax_alt_zoom.set_xlim(time_min, time_max)
        ax_alt_zoom.set_ylim(alt_min, alt_max)
    
    ax_alt_zoom.set_xlabel('Time [s]')
    ax_alt_zoom.set_ylabel('Altitude [m AGL]')
    ax_alt_zoom.set_title('Zoomed View - Terminal Phase')
    ax_alt_zoom.grid(True, alpha=0.3)
    ax_alt_zoom.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, label='Ground Level')
    
    # Add legend entries for markers (only on one plot to avoid clutter)
    ax_latlon.scatter([], [], color='gray', s=120, marker='^', edgecolors='orange', linewidth=2, label='Dive Start')
    ax_latlon.scatter([], [], color='gray', s=150, marker='X', edgecolors='red', linewidth=2, label='Impact')
    ax_latlon.legend(loc='best', fontsize=8 if len(trajectory_data) > 1 else 10)
    
    plt.tight_layout()
    
    if interactive:
        plt.show()  # Show interactive plot
        print("Showing interactive GPS trajectory plot (close window to continue)")
    else:
        # Save plot with appropriate filename based on number of files
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        prefix = "gps_trajectories" if len(trajectory_data) == 1 else "combined_gps_trajectories"
        gps_plot_path = output_dir / f"{prefix}{target_suffix}.png"
        plt.savefig(gps_plot_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_accelerometer_impacts(trajectory_data: list, output_dir: Path, 
                               target_selection: str, interactive: bool = False) -> None:
    """Create accelerometer magnitude plots showing detected impact points.
    
    Creates one subplot per trajectory showing acceleration magnitude over time
    with detected impact points marked. This helps validate impact detection.
    
    Args:
        trajectory_data: List of dicts containing:
            - 'filename': Name of the log file
            - 'data': Extracted data dictionary with accelerometer data
            - 'engagement_masks': Dictionary with impact_mask and impact_indices
            - 'miss_distances': Optional dict with CPA information
        output_dir: Directory to save plots
        target_selection: Target name ('Container' or 'Van')
        interactive: Whether to show interactive plots
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping accelerometer impact plot.")
        return
        
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping accelerometer impact plot.")
        return
    
    # Ensure trajectory_data is a list
    if not isinstance(trajectory_data, list):
        trajectory_data = [trajectory_data]
    
    n_trajectories = len(trajectory_data)
    
    # Create subplots - arrange in a grid
    if n_trajectories == 1:
        fig, axes = plt.subplots(1, 1, figsize=(14, 6))
        axes = [axes]
    else:
        # Arrange in 2 columns
        n_cols = 2
        n_rows = (n_trajectories + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_trajectories > 1 else [axes]
    
    file_word = "file" if n_trajectories == 1 else "files"
    # print(f"\n=== Creating Accelerometer Impact Plot for {n_trajectories} {file_word} ===")
    
    plot_idx = 0  # Track actual plot index separately from loop index
    for i, traj_info in enumerate(trajectory_data):
        try:
            filename = traj_info['filename']
            data = traj_info['data']
            mask = traj_info['mask']
            engagement_masks = traj_info['engagement_masks']
            
            # Get first dive index - use it if available, otherwise plot from start
            first_dive_idx = engagement_masks.get('first_dive_idx')
            if first_dive_idx is None:
                # No dive detected, check if there's any data to plot
                if len(data['timestamp_us']) == 0:
                    print(f"  Skipping {filename}: No data available")
                    continue
                first_dive_idx = 0  # Start from beginning if no dive detected
            
            # Filter data from dive point onward
            dive_mask = np.zeros(len(data['timestamp_us']), dtype=bool)
            dive_mask[first_dive_idx:] = True
            
            # Get time array (relative to dive start)
            time_s = (data['timestamp_us'][dive_mask] - data['timestamp_us'][first_dive_idx]) * 1e-6
            
            # Plot acceleration magnitude from dive point (Method 1)
            ax = axes[plot_idx]
            ax.plot(time_s, data['accel_magnitude'][dive_mask], 'b-', linewidth=1, alpha=0.7, label='Accel Magnitude')
            
            # Mark first impact from Method 1 (high acceleration magnitude)
            method1_impacts = data.get('method1_impact_indices', [])
            method2_impacts = data.get('method2_impact_indices', [])
            valid_method1 = [idx for idx in method1_impacts if idx >= first_dive_idx]
            valid_method2 = [idx for idx in method2_impacts if idx >= first_dive_idx]
            
            if len(valid_method1) > 0:
                first_method1_idx = valid_method1[0]
                impact_time = (data['timestamp_us'][first_method1_idx] - data['timestamp_us'][first_dive_idx]) * 1e-6
                impact_accel = data['accel_magnitude'][first_method1_idx]
                ax.scatter(impact_time, impact_accel, color='red', s=100, marker='X',
                          edgecolors='black', linewidth=2, zorder=5, label='Method 1 Impact')
                ax.axvline(x=impact_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Mark first impact from Method 2 (high acceleration derivative)
            if len(valid_method2) > 0:
                first_method2_idx = valid_method2[0]
                impact_time = (data['timestamp_us'][first_method2_idx] - data['timestamp_us'][first_dive_idx]) * 1e-6
                impact_accel = data['accel_magnitude'][first_method2_idx]
                ax.scatter(impact_time, impact_accel, color='orange', s=100, marker='o',
                          edgecolors='black', linewidth=2, zorder=5, label='Method 2 Impact')
                ax.axvline(x=impact_time, color='orange', linestyle=':', alpha=0.5, linewidth=1)
            
            # Mark fallback impacts (Method 3: 1m AGL)
            if 'fallback_impact_indices' in data and len(data['fallback_impact_indices']) > 0:
                fallback_impacts = data['fallback_impact_indices']
                valid_fallback = [idx for idx in fallback_impacts if idx >= first_dive_idx]
                for idx in valid_fallback:
                    impact_time = time_s[np.where(dive_mask)[0] == idx][0] if np.any(np.where(dive_mask)[0] == idx) else None
                    if impact_time is not None:
                        ax.scatter(impact_time, data['accel_magnitude'][idx], color='purple', s=150, marker='D',
                                  edgecolors='black', linewidth=2, zorder=5, label='Method 3: Fallback (1m AGL)')
                        ax.axvline(x=impact_time, color='purple', linestyle=':', alpha=0.5, linewidth=1)
            
            # Mark Closest Point of Approach (CPA) if available
            miss_distances = traj_info.get('miss_distances', {})
            if miss_distances and target_selection in miss_distances:
                cpa_time_s = miss_distances[target_selection].get('cpa_time_s')
                if cpa_time_s is not None:
                    # Convert to time relative to dive start
                    cpa_time_relative = cpa_time_s - (data['timestamp_us'][first_dive_idx] - data['timestamp_us'][0]) * 1e-6
                    if 0 <= cpa_time_relative <= time_s[-1]:
                        # Find corresponding acceleration value
                        cpa_idx_in_dive = np.argmin(np.abs(time_s - cpa_time_relative))
                        cpa_accel = data['accel_magnitude'][dive_mask][cpa_idx_in_dive]
                        ax.scatter(cpa_time_relative, cpa_accel, color='cyan', s=120, marker='*',
                                  edgecolors='darkblue', linewidth=2, zorder=6, label='Closest Approach (CPA)')
                        ax.axvline(x=cpa_time_relative, color='cyan', linestyle='-.', alpha=0.6, linewidth=1.5)
            
            # Mark acceleration threshold
            ax.axhline(y=15.0, color='orange', linestyle=':', alpha=0.5, linewidth=1.5, 
                      label='Accel Threshold (15 m/s²)')
            
            # Add altitude as secondary y-axis
            ax2 = ax.twinx()
            ax2.plot(time_s, data['altitude_agl'][dive_mask], 'g-', linewidth=1, alpha=0.4, label='Altitude AGL')
            ax2.axhline(y=10.0, color='purple', linestyle=':', alpha=0.5, linewidth=1.5,
                       label='Alt Threshold (10m)')
            ax2.set_ylabel('Altitude [m AGL]', color='g')
            ax2.tick_params(axis='y', labelcolor='g')
            
            # Configure primary axis
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('Acceleration Magnitude [m/s²]', color='b')
            ax.set_title(f'{filename}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelcolor='b')
            
            # Combine legends
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
            
            impact_count = len(valid_method1) + len(valid_method2)
            # print(f"  Processed {filename}: {impact_count} impact(s) detected (M1: {len(valid_method1)}, M2: {len(valid_method2)})")
            
            plot_idx += 1  # Increment only on successful plot
            
        except Exception as e:
            print(f"  Error plotting accelerometer data for {filename}: {e}")
            continue
    
    # Hide unused subplots
    for j in range(plot_idx, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if interactive:
        plt.show()
        print("Showing interactive accelerometer impact plot (Method 1) (close window to continue)")
    else:
        # Save plot
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        prefix = "accelerometer_method1" if n_trajectories == 1 else "combined_accelerometer_method1"
        plot_path = output_dir / f"{prefix}{target_suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    # ===========================
    # Create Method 2 Plot (Acceleration Derivative)
    # ===========================
    
    # Create figure with same layout as Method 1 plot
    if n_trajectories == 1:
        fig, axes = plt.subplots(1, 1, figsize=(14, 6))
        axes = [axes]
    else:
        n_cols = 2
        n_rows = (n_trajectories + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes = axes.flatten() if n_trajectories > 1 else [axes]
    
    # print(f"\n=== Creating Method 2 (Derivative) Plot for {n_trajectories} {file_word} ===")
    
    plot_idx = 0  # Track actual plot index separately from loop index
    for i, traj_info in enumerate(trajectory_data):
        try:
            filename = traj_info['filename']
            data = traj_info['data']
            mask = traj_info['mask']
            engagement_masks = traj_info['engagement_masks']
            
            # Get first dive index - use it if available, otherwise plot from start
            first_dive_idx = engagement_masks.get('first_dive_idx')
            if first_dive_idx is None:
                # No dive detected, check if there's any data to plot
                if len(data['timestamp_us']) == 0:
                    continue
                first_dive_idx = 0  # Start from beginning if no dive detected
            
            # Filter data from dive point onward
            dive_mask = np.zeros(len(data['timestamp_us']), dtype=bool)
            dive_mask[first_dive_idx:] = True
            
            # Get time array (relative to dive start)
            time_s = (data['timestamp_us'][dive_mask] - data['timestamp_us'][first_dive_idx]) * 1e-6
            
            # Plot smoothed acceleration derivative magnitude (Method 2)
            ax = axes[plot_idx]
            if 'accel_derivative_smooth' in data:
                ax.plot(time_s, data['accel_derivative_smooth'][dive_mask], 'c-', linewidth=1.5, alpha=0.7, 
                       label='|Accel Deriv| (10-sample avg)')
            
            # Get impact indices for both methods
            method1_impacts = data.get('method1_impact_indices', [])
            method2_impacts = data.get('method2_impact_indices', [])
            valid_method1 = [idx for idx in method1_impacts if idx >= first_dive_idx]
            valid_method2 = [idx for idx in method2_impacts if idx >= first_dive_idx]
            
            # Mark first impact from Method 1 (high acceleration magnitude)
            if len(valid_method1) > 0 and 'accel_derivative_smooth' in data:
                first_method1_idx = valid_method1[0]
                impact_time = (data['timestamp_us'][first_method1_idx] - data['timestamp_us'][first_dive_idx]) * 1e-6
                impact_deriv = data['accel_derivative_smooth'][first_method1_idx]
                ax.scatter(impact_time, impact_deriv, color='red', s=100, marker='X',
                          edgecolors='black', linewidth=2, zorder=5, label='Method 1 Impact')
                ax.axvline(x=impact_time, color='red', linestyle='--', alpha=0.5, linewidth=1)
            
            # Mark first impact from Method 2 (high acceleration derivative)
            if len(valid_method2) > 0 and 'accel_derivative_smooth' in data:
                first_method2_idx = valid_method2[0]
                impact_time = (data['timestamp_us'][first_method2_idx] - data['timestamp_us'][first_dive_idx]) * 1e-6
                impact_deriv = data['accel_derivative_smooth'][first_method2_idx]
                ax.scatter(impact_time, impact_deriv, color='orange', s=100, marker='o',
                          edgecolors='black', linewidth=2, zorder=5, label='Method 2 Impact')
                ax.axvline(x=impact_time, color='orange', linestyle=':', alpha=0.5, linewidth=1)
            
            # Mark fallback impacts (Method 3: 1m AGL)
            if 'fallback_impact_indices' in data and len(data['fallback_impact_indices']) > 0:
                fallback_impacts = data['fallback_impact_indices']
                valid_fallback = [idx for idx in fallback_impacts if idx >= first_dive_idx]
                for idx in valid_fallback:
                    impact_time = time_s[np.where(dive_mask)[0] == idx][0] if np.any(np.where(dive_mask)[0] == idx) else None
                    if impact_time is not None and 'accel_derivative_smooth' in data:
                        ax.scatter(impact_time, data['accel_derivative_smooth'][idx], color='purple', s=150, marker='D',
                                  edgecolors='black', linewidth=2, zorder=5, label='Method 3: Fallback (1m AGL)')
                        ax.axvline(x=impact_time, color='purple', linestyle=':', alpha=0.5, linewidth=1)
            
            # Mark Closest Point of Approach (CPA) if available
            miss_distances = traj_info.get('miss_distances', {})
            if miss_distances and target_selection in miss_distances and 'accel_derivative_smooth' in data:
                cpa_time_s = miss_distances[target_selection].get('cpa_time_s')
                if cpa_time_s is not None:
                    # Convert to time relative to dive start
                    cpa_time_relative = cpa_time_s - (data['timestamp_us'][first_dive_idx] - data['timestamp_us'][0]) * 1e-6
                    if 0 <= cpa_time_relative <= time_s[-1]:
                        # Find corresponding derivative value
                        cpa_idx_in_dive = np.argmin(np.abs(time_s - cpa_time_relative))
                        cpa_deriv = data['accel_derivative_smooth'][dive_mask][cpa_idx_in_dive]
                        ax.scatter(cpa_time_relative, cpa_deriv, color='cyan', s=120, marker='*',
                                  edgecolors='darkblue', linewidth=2, zorder=6, label='Closest Approach (CPA)')
                        ax.axvline(x=cpa_time_relative, color='cyan', linestyle='-.', alpha=0.6, linewidth=1.5)
            
            # Mark derivative threshold (only positive since we're using magnitude)
            ax.axhline(y=2.0, color='orange', linestyle=':', alpha=0.5, linewidth=1.5, 
                      label='Deriv Threshold (2.0 m/s²)')
            
            # Configure primary axis
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('|Acceleration Derivative| [m/s²]', color='c')
            ax.set_title(f'{filename}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='y', labelcolor='c')
            
            # Add legend
            ax.legend(loc='upper right', fontsize=8)
            
            plot_idx += 1  # Increment only on successful plot
            
        except Exception as e:
            print(f"  Error plotting Method 2 data for {filename}: {e}")
            continue
    
    # Hide unused subplots
    for j in range(plot_idx, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if interactive:
        plt.show()
        print("Showing interactive Method 2 (Derivative) plot (close window to continue)")
    else:
        # Save plot
        prefix = "accelerometer_method2" if n_trajectories == 1 else "combined_accelerometer_method2"
        plot_path = output_dir / f"{prefix}{target_suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_miss_distance_histograms(stats_output: list, output_dir: Path, target_selection: str, interactive: bool = False, cpa_hit_threshold: float = 3.0, dive_angle: int = None, num_successful_hits: int = 0):
    """Create histogram plots for miss distance distributions.
    
    Creates two histogram subplots:
    1. Impact Point Distance (ground) - Original metric
    2. Closest Point of Approach (CPA) Total (3D) - True miss distance
    
    Args:
        stats_output: List of statistics dictionaries from process_multiple_logs
        output_dir: Output directory for plots
        target_selection: Target name for filename
        interactive: Whether to show interactive plot
        cpa_hit_threshold: CPA threshold for hit detection (m)
        dive_angle: Dive angle for title annotation
        num_successful_hits: Number of hits that met both CPA and time criteria
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping miss distance histogram plot.")
        return
        
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping miss distance histogram plot.")
        return
    
    if not stats_output or len(stats_output) == 0:
        print("No statistics available for histogram plotting.")
        return
    
    # Extract data from first target (since we typically analyze one target at a time)
    stat = stats_output[0]
    distance_dicts = stat['distance_dicts']
    
    if len(distance_dicts) < 2:
        print("Need at least 2 data points for histogram plotting.")
        return
    
    # Extract arrays for each metric
    impact_ground_array = np.array([d['impact_ground'] for d in distance_dicts])
    impact_total_array = np.array([d['impact_total'] for d in distance_dicts])
    cpa_ground_array = np.array([d['cpa_ground'] for d in distance_dicts])
    cpa_total_array = np.array([d['cpa_total'] for d in distance_dicts])
    cpa_alt_diff_array = np.array([d['cpa_alt_diff'] for d in distance_dicts])
    
    # Create figure with 3 subplots (3 rows)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Define bin edges for histograms (0-20m range with 2m bins)
    bin_edges = np.arange(0, 21, 1)
    
    # Subplot 1: Impact Point Distance (ground)
    ax1 = axes[0]
    counts1, bins1, patches1 = ax1.hist(impact_ground_array, bins=bin_edges, 
                                         edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(stat['impact_ground_mean'], color='red', linestyle='--', linewidth=2, 
                label=f"Mean = {stat['impact_ground_mean']:.1f}m")
    ax1.axvline(stat['impact_ground_mean'] + stat['impact_ground_std'], 
                color='orange', linestyle=':', linewidth=2, label=f"±1σ = {stat['impact_ground_std']:.1f}m")
    ax1.axvline(stat['impact_ground_mean'] - stat['impact_ground_std'], 
                color='orange', linestyle=':', linewidth=2)
    
    ax1.set_xlabel('Miss Distance [m]', fontsize=12)
    ax1.set_ylabel('Number of Cases', fontsize=12)
    title_text = 'Impact Point Miss Distance Distribution'
    if dive_angle is not None:
        title_text += f' | Dive Angle: {dive_angle}°'
    ax1.set_title(title_text, fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 20])
    ax1.set_xticks(bin_edges)

    # Set x-axis ticks to show every bin edge
    ax1.set_xticks(bin_edges)
    
    # Add count labels on bars
    for i, (count, patch) in enumerate(zip(counts1, patches1)):
        if count > 0:
            height = patch.get_height()
            ax1.text(patch.get_x() + patch.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Define bin edges for histograms (0-20m range with 2m bins)
    bin_edges = np.arange(0, 11, 1)
    # Subplot 2: Closest Point of Approach (CPA) Total (3D) - True Miss Distance
    ax2 = axes[1]
    counts2, bins2, patches2 = ax2.hist(cpa_total_array, bins=bin_edges, 
                                         edgecolor='black', alpha=0.7, color='darkred')
    
    # Color the 0-1m bin (strong hit) and 1-2m bin (hit)
    for i, patch in enumerate(patches2):
        bin_start = bins2[i]
        bin_end = bins2[i+1]
        if bin_end <= 1.0:
            # 0-1m range: Strong Hit (dark green)
            patch.set_facecolor('darkgreen')
            patch.set_alpha(0.8)
        elif bin_start < 2.0 and bin_end <= 2.0:
            # 1-2m range: Hit (medium green)
            patch.set_facecolor('darkgreen')
            patch.set_alpha(0.8)
        elif bin_start < 3.0 and bin_end <= 3.0:
            # 2-3m range: Hit (medium green)
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.8)
    
    ax2.axvline(stat['cpa_total_mean'], color='red', linestyle='--', linewidth=2, 
                label=f"Mean = {stat['cpa_total_mean']:.1f}m")
    ax2.axvline(stat['cpa_total_mean'] + stat['cpa_total_std'], 
                color='orange', linestyle=':', linewidth=2, label=f"±1σ = {stat['cpa_total_std']:.1f}m")
    ax2.axvline(stat['cpa_total_mean'] - stat['cpa_total_std'], 
                color='orange', linestyle=':', linewidth=2)
    
    ax2.set_xlabel('True Miss Distance [m]', fontsize=12)
    ax2.set_ylabel('Number of Cases', fontsize=12)
    ax2.set_title(f'Closest Point of Approach (CPA) Miss Distance Distribution', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 10])
    
    # Set x-axis ticks to show every bin edge
    ax2.set_xticks(bin_edges)
    
    # Add count labels on bars
    for i, (count, patch) in enumerate(zip(counts2, patches2)):
        if count > 0:
            height = patch.get_height()
            ax2.text(patch.get_x() + patch.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Subplot 3: Altitude Miss Distance (Vertical Component of CPA)
    ax3 = axes[2]
    
    # Calculate statistics for altitude miss
    cpa_alt_diff_mean = np.mean(cpa_alt_diff_array)
    cpa_alt_diff_std = np.std(cpa_alt_diff_array)
    
    # Define bin edges for altitude (centered around 0, symmetric range)
    max_alt_range = max(abs(np.min(cpa_alt_diff_array)), abs(np.max(cpa_alt_diff_array)))
    max_alt_range = max(5, max_alt_range)  # At least ±5m range
    alt_bin_edges = np.arange(-max_alt_range, max_alt_range + 1, 1)
    
    counts3, bins3, patches3 = ax3.hist(cpa_alt_diff_array, bins=alt_bin_edges, 
                                         edgecolor='black', alpha=0.7, color='purple')
    
    # Color bins within ±1m (target height range)
    for i, patch in enumerate(patches3):
        bin_start = bins3[i]
        bin_end = bins3[i+1]
        if bin_start >= -1.0 and bin_end <= 1.0:
            # Within ±1m of target center
            patch.set_facecolor('darkgreen')
            patch.set_alpha(0.8)
        elif (bin_start >= 2.0 and bin_end <= 3.0) :
            # Within 2-3m of target center
            patch.set_facecolor('lightgreen')
            patch.set_alpha(0.8)
        else:
            patch.set_facecolor('darkred')
            patch.set_alpha(0.7)
    
    ax3.axvline(cpa_alt_diff_mean, color='red', linestyle='--', linewidth=2, 
                label=f"Mean = {cpa_alt_diff_mean:.2f}m")
    ax3.axvline(cpa_alt_diff_mean + cpa_alt_diff_std, 
                color='orange', linestyle=':', linewidth=2, label=f"±1σ = {cpa_alt_diff_std:.2f}m")
    ax3.axvline(cpa_alt_diff_mean - cpa_alt_diff_std, 
                color='orange', linestyle=':', linewidth=2)
    ax3.axvline(0, color='green', linestyle='-', linewidth=1.5, alpha=0.5, label='Target Center')
    
    ax3.set_xlabel('Altitude Miss Distance (CPA) [m]', fontsize=12)
    ax3.set_ylabel('Number of Cases', fontsize=12)
    ax3.set_title('Altitude Miss Distance Distribution (Vertical Component)', 
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(fontsize=10)
    
    # Add count labels on bars
    for i, (count, patch) in enumerate(zip(counts3, patches3)):
        if count > 0:
            height = patch.get_height()
            ax3.text(patch.get_x() + patch.get_width()/2., height,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Calculate hit rate (successful hits that met both CPA and time criteria)
    hits = num_successful_hits
    total_cases = len(cpa_total_array)
    hit_rate = (hits / total_cases * 100) if total_cases > 0 else 0
    
    # Add text box with simulation requirements (on third subplot)
    mu_plus_1sigma = stat['cpa_total_mean'] +  stat['cpa_total_std']
    
    # Check if requirements are met
    mean_passed = stat["cpa_total_mean"] < 2.0
    sigma_passed = mu_plus_1sigma < 3.0
    
    # Color coding: green for pass, red for fail
    mean_color = 'green' if mean_passed else 'red'
    sigma_color = 'green' if sigma_passed else 'red'
    hit_rate_color  = 'green' if hit_rate > 90.0 else 'red'    

    req_text = (
        'Simulation Requirements:\n'
        f'0-1m: Strong Hit\n'
        f'1-2m: Hit\n'
        f'Hit Rate > 90%\n'
        f'• Mean CPA Miss < 3.0 m\n'
        f'• μ + 1σ < 3.0 m\n\n'
        f'Current Results:\n\n\n\n'
       
    )
    
    # Position text box on the right side of third subplot
    ax3.text(1.02, 0.5, req_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=2))
    
    # Add colored result text lines (positioned to align with "Current Results:" section)
    ax3.text(1.035, 0.37, f'• Mean = {stat["cpa_total_mean"]:.2f} m', transform=ax3.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='left',
            color=mean_color, fontweight='bold')
    ax3.text(1.035, 0.34, f'• μ + 1σ = {mu_plus_1sigma:.2f} m', transform=ax3.transAxes,
            fontsize=10, verticalalignment='center', horizontalalignment='left',
            color=sigma_color, fontweight='bold')
    ax3.text(1.035, 0.31, f'• Hit Rate = {hit_rate:.2f} %', transform=ax3.transAxes,
        fontsize=10, verticalalignment='center', horizontalalignment='left',
        color=hit_rate_color, fontweight='bold')
    
    # Add bold hit rate display on the right side above requirements
    hit_rate_text = f'HIT RATE: {hit_rate:.1f}% ({hits}/{total_cases})'
    ax3.text(1.02, 0.85, hit_rate_text, transform=ax3.transAxes,
            fontsize=14, fontweight='bold', verticalalignment='center', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=3))
    
    plt.tight_layout()
    
    if interactive:
        plt.show()
        print("Showing interactive miss distance histogram (close window to continue)")
    else:
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        plot_path = output_dir / f"miss_distance_histograms{target_suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_impact_angle_histogram(impact_angles: list, relative_yaws: list, airspeeds: list, output_dir: Path, target_selection: str, interactive: bool = False, cpa_hit_threshold: float = 3.0, dive_angle: int = None, time_diff_threshold: float = 0.1):
    """Create triple histogram plot for impact angle, relative yaw, and airspeed distribution.
    
    Shows the distribution of aircraft orientation angles and airspeed at impact for successful hits
    (cases where 3D miss distance < threshold and impact detected within time_diff_threshold of CPA).
    
    Args:
        impact_angles: List of impact pitch angles in degrees
        relative_yaws: List of relative yaw angles in degrees (difference from mean dive heading)
        airspeeds: List of airspeed values at impact in m/s
        output_dir: Output directory for plots
        target_selection: Target name for filename
        interactive: Whether to show interactive plot
        dive_angle: Dive angle in degrees (for title)
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping impact angle histogram plot.")
        return
        
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping impact angle histogram plot.")
        return
    
    if not impact_angles or len(impact_angles) == 0:
        print("No impact angle data available for histogram plotting.")
        return
    
    if len(impact_angles) < 2:
        print("Need at least 2 data points for impact angle histogram plotting.")
        return
    
    impact_angles_array = np.array(impact_angles)
    
    # Create figure with 3 rows of subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 16))
    
    # ========== TOP SUBPLOT: PITCH ANGLE ==========
    # Define bin edges for histogram (-90 to 90 degrees with 5 degree bins)
    # Negative angles = nose up, positive angles = nose down
    bin_edges = np.arange(-55, 55, 5)
    
    # Create histogram
    counts, bins, patches = ax1.hist(impact_angles_array, bins=bin_edges, 
                                    edgecolor='black', alpha=0.7, color='steelblue')
    
    # Calculate statistics
    mean_angle = np.mean(impact_angles_array)
    std_angle = np.std(impact_angles_array)
    median_angle = np.median(impact_angles_array)
    
    # Add mean and median lines
    ax1.axvline(mean_angle, color='red', linestyle='--', linewidth=2, 
               label=f"Mean = {mean_angle:.1f}°")
    ax1.axvline(median_angle, color='green', linestyle='--', linewidth=2,
               label=f"Median = {median_angle:.1f}°")
    ax1.axvline(mean_angle + std_angle, color='orange', linestyle=':', linewidth=2,
               label=f"±1σ = {std_angle:.1f}°")
    ax1.axvline(mean_angle - std_angle, color='orange', linestyle=':', linewidth=2)
    
    # Add reference line at 0° (horizontal)
    ax1.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='0° (Horizontal)')
    
    # Add count labels on bars
    for i, (count, patch) in enumerate(zip(counts, patches)):
        if count > 0:
            height = patch.get_height()
            ax1.text(patch.get_x() + patch.get_width()/2., height,
                   f'{int(count)}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Impact Pitch Angle [degrees]', fontsize=12)
    ax1.set_ylabel('Number of Cases', fontsize=12)
    title_text = f'Impact Pitch Angle Distribution (Successful Hits: CPA < {cpa_hit_threshold}m, Δt < {time_diff_threshold}s)\n' + f'Target: {target_selection}'
    if dive_angle is not None:
        title_text += f' | Dive Angle: {dive_angle}°'
    ax1.set_title(title_text, fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.set_xlim([-55, 55])
    ax1.set_xticks(np.arange(-55, 55, 5))  # Major ticks every 10 degrees
    
    # Add text box with statistics and explanation - positioned below legend
    stats_text = (
        f'Pitch Angle Statistics:\n'
        f'(n = {len(impact_angles)} hits)\n\n'
        f'Mean: {mean_angle:.1f}°\n'
        f'Median: {median_angle:.1f}°\n'
        f'Std Dev: {std_angle:.1f}°\n'
        f'Min: {np.min(impact_angles_array):.1f}°\n'
        f'Max: {np.max(impact_angles_array):.1f}°\n\n'
        f'Angle Convention:\n'
        f'  -90° = Vertical climb\n'
        f'     0° = Horizontal\n'
        f'  +90° = Vertical dive'
    )
    
    # Position text box on the right side
    ax1.text(1.02, 0.5, stats_text, transform=ax1.transAxes,
           fontsize=9, verticalalignment='center', horizontalalignment='left',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, 
                    edgecolor='black', linewidth=2))
    
    # ========== BOTTOM SUBPLOT: RELATIVE YAW ==========
    if relative_yaws and len(relative_yaws) > 0:
        relative_yaws_array = np.array(relative_yaws)
        
        # Define bin edges for yaw histogram (-180 to 180 degrees with 10 degree bins)
        yaw_bin_edges = np.arange(-35, 35, 5)
        
        # Create histogram
        yaw_counts, yaw_bins, yaw_patches = ax2.hist(relative_yaws_array, bins=yaw_bin_edges, 
                                        edgecolor='black', alpha=0.7, color='coral')
        
        # Calculate statistics
        mean_yaw = np.mean(relative_yaws_array)
        std_yaw = np.std(relative_yaws_array)
        median_yaw = np.median(relative_yaws_array)
        
        # Add mean and median lines
        ax2.axvline(mean_yaw, color='red', linestyle='--', linewidth=2, 
                   label=f"Mean = {mean_yaw:.1f}°")
        ax2.axvline(median_yaw, color='green', linestyle='--', linewidth=2,
                   label=f"Median = {median_yaw:.1f}°")
        ax2.axvline(mean_yaw + std_yaw, color='orange', linestyle=':', linewidth=2,
                   label=f"±1σ = {std_yaw:.1f}°")
        ax2.axvline(mean_yaw - std_yaw, color='orange', linestyle=':', linewidth=2)
        
        # Add reference line at 0° (on heading)
        ax2.axvline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='0° (On Heading)')
        
        # Add count labels on bars
        for i, (count, patch) in enumerate(zip(yaw_counts, yaw_patches)):
            if count > 0:
                height = patch.get_height()
                ax2.text(patch.get_x() + patch.get_width()/2., height,
                       f'{int(count)}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax2.set_xlabel('Relative Yaw Angle [degrees]', fontsize=12)
        ax2.set_ylabel('Number of Cases', fontsize=12)
        ax2.set_title(f'Relative Yaw Angle Distribution (vs Mean Dive Heading)', 
                    fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.set_xlim([-30, 30])
        ax2.set_xticks(np.arange(-30, 30, 5))  # Major ticks every 30 degrees
        
        # Add text box with statistics - positioned below legend
        yaw_stats_text = (
            f'Relative Yaw Statistics:\n'
            f'(n = {len(relative_yaws)} hits)\n\n'
            f'Mean: {mean_yaw:.1f}°\n'
            f'Median: {median_yaw:.1f}°\n'
            f'Std Dev: {std_yaw:.1f}°\n'
            f'Min: {np.min(relative_yaws_array):.1f}°\n'
            f'Max: {np.max(relative_yaws_array):.1f}°\n\n'
            f'Reference:\n'
            f'  Mean dive heading\n'
            f'  from all trajectories'
        )
        
        # Position text box on the right side
        ax2.text(1.02, 0.5, yaw_stats_text, transform=ax2.transAxes,
               fontsize=9, verticalalignment='center', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, 
                        edgecolor='black', linewidth=2))
    else:
        # No relative yaw data available
        ax2.text(0.5, 0.5, 'No relative yaw data available', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=14)
        ax2.set_xlabel('Relative Yaw Angle [degrees]', fontsize=12)
        ax2.set_ylabel('Number of Cases', fontsize=12)
        ax2.set_title(f'Relative Yaw Angle Distribution', fontsize=13, fontweight='bold')
    
    # ========== THIRD SUBPLOT: AIRSPEED ==========
    if airspeeds and len(airspeeds) > 0:
        airspeeds_array = np.array(airspeeds)
        
        # Convert airspeed from m/s to mph (1 m/s = 2.237 mph)
        airspeeds_mph = airspeeds_array * 2.237
        
        # Calculate mean to center the 30 mph range
        mean_airspeed_mph = np.mean(airspeeds_mph)
        
        # Define 30 mph range centered on mean with 5 mph bins
        range_center = round(mean_airspeed_mph / 5) * 5  # Round to nearest 5 mph
        range_min = range_center - 15
        range_max = range_center + 15
        airspeed_bin_edges = np.arange(range_min, range_max + 5, 5)
        
        # Create histogram
        airspeed_counts, airspeed_bins, airspeed_patches = ax3.hist(airspeeds_mph, bins=airspeed_bin_edges, 
                                        edgecolor='black', alpha=0.7, color='mediumseagreen')
        
        # Calculate statistics in mph
        std_airspeed_mph = np.std(airspeeds_mph)
        median_airspeed_mph = np.median(airspeeds_mph)
        
        # Add mean and median lines
        ax3.axvline(mean_airspeed_mph, color='red', linestyle='--', linewidth=2, 
                   label=f"Mean = {mean_airspeed_mph:.1f} mph")
        ax3.axvline(median_airspeed_mph, color='green', linestyle='--', linewidth=2,
                   label=f"Median = {median_airspeed_mph:.1f} mph")
        ax3.axvline(mean_airspeed_mph + std_airspeed_mph, color='orange', linestyle=':', linewidth=2,
                   label=f"±1σ = {std_airspeed_mph:.1f} mph")
        ax3.axvline(mean_airspeed_mph - std_airspeed_mph, color='orange', linestyle=':', linewidth=2)
        
        # Add count labels on bars
        for i, (count, patch) in enumerate(zip(airspeed_counts, airspeed_patches)):
            if count > 0:
                height = patch.get_height()
                ax3.text(patch.get_x() + patch.get_width()/2., height,
                       f'{int(count)}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax3.set_xlabel('Airspeed [mph]', fontsize=12)
        ax3.set_ylabel('Number of Cases', fontsize=12)
        ax3.set_title(f'Airspeed at Impact Distribution', 
                    fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend(fontsize=10, loc='upper right')
        ax3.set_xlim([range_min, range_max])
        ax3.set_xticks(np.arange(range_min, range_max + 5, 5))
        
        # Add text box with statistics - positioned below legend
        airspeed_stats_text = (
            f'Airspeed Statistics:\n'
            f'(n = {len(airspeeds)} hits)\n\n'
            f'Mean: {mean_airspeed_mph:.1f} mph\n'
            f'Median: {median_airspeed_mph:.1f} mph\n'
            f'Std Dev: {std_airspeed_mph:.1f} mph\n'
            f'Min: {np.min(airspeeds_mph):.1f} mph\n'
            f'Max: {np.max(airspeeds_mph):.1f} mph\n\n'
            f'Conversion:\n'
            f'  {mean_airspeed_mph/2.237:.1f} m/s\n'
            f'  {mean_airspeed_mph*1.609:.1f} km/h'
        )
        
        # Position text box on the right side
        ax3.text(1.02, 0.5, airspeed_stats_text, transform=ax3.transAxes,
               fontsize=9, verticalalignment='center', horizontalalignment='left',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, 
                        edgecolor='black', linewidth=2))
    else:
        # No airspeed data available
        ax3.text(0.5, 0.5, 'No airspeed data available', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=14)
        ax3.set_xlabel('Airspeed [mph]', fontsize=12)
        ax3.set_ylabel('Number of Cases', fontsize=12)
        ax3.set_title(f'Airspeed at Impact Distribution', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if interactive:
        plt.show()
        print("Showing interactive impact angle histogram (close window to continue)")
    else:
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        plot_path = output_dir / f"impact_angle_histogram{target_suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()


def plot_altitude_debug(trajectory_data: list, output_dir: Path, 
                       target_selection: str = "Van", interactive: bool = False):
    """Debug plot showing altitude trajectories with dive and impact detection markers.
    
    Creates one subplot per trajectory showing:
    - Altitude AGL over time
    - Dive start markers (pitch threshold crossed)
    - Impact detection markers (all three methods)
    - Terminal engagement segment highlighting
    
    Args:
        trajectory_data: List of dicts with 'ulog', 'data', 'mask', 'engagement_masks', 'filename'
        output_dir: Output directory for the plot
        target_selection: Target name for filename suffix
        interactive: Whether to show interactive plot
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping altitude debug plot.")
        return
    
    n_trajectories = len(trajectory_data)
    if n_trajectories == 0:
        print("No trajectory data available for altitude debug plot.")
        return
    
    # Calculate grid dimensions for subplots
    if n_trajectories == 1:
        nrows, ncols = 1, 1
        figsize = (14, 6)
    elif n_trajectories <= 4:
        nrows, ncols = 2, 2
        figsize = (16, 10)
    elif n_trajectories <= 6:
        nrows, ncols = 2, 3
        figsize = (18, 10)
    elif n_trajectories <= 9:
        nrows, ncols = 3, 3
        figsize = (18, 14)
    elif n_trajectories <= 12:
        nrows, ncols = 3, 4
        figsize = (20, 14)
    elif n_trajectories <= 16:
        nrows, ncols = 4, 4
        figsize = (20, 16)
    elif n_trajectories <= 20:
        nrows, ncols = 4, 5
        figsize = (22, 16)
    elif n_trajectories <= 25:
        nrows, ncols = 5, 5
        figsize = (22, 18)
    else:
        nrows, ncols = 6, 5
        figsize = (22, 20)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_trajectories == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle(f'Altitude Trajectory Debug - Dive & Impact Detection\nTarget: {target_selection}', 
                 fontsize=16, fontweight='bold')
    
    for idx, traj in enumerate(trajectory_data):
        ax = axes[idx]
        data = traj['data']
        mask = traj['mask']
        engagement_masks = traj['engagement_masks']
        filename = traj['filename']
        
        # Get time array in seconds (relative to start)
        time_s = (data['timestamp_us'] - data['timestamp_us'][0]) * 1e-6
        altitude_agl = data['altitude_agl']
        pitch_deg = data['pitch_deg']
        
        # Plot altitude trajectory
        ax.plot(time_s, altitude_agl, 'b-', linewidth=1.5, label='Altitude AGL', alpha=0.7)
        
        # Highlight terminal engagement segment if it exists
        if np.any(mask):
            engagement_indices = np.where(mask)[0]
            ax.fill_between(time_s[engagement_indices], 0, altitude_agl[engagement_indices],
                           color='yellow', alpha=0.2, label='Terminal Engagement')
        
        # Mark dive start
        first_dive_idx = engagement_masks.get('first_dive_idx')
        if first_dive_idx is not None:
            ax.axvline(time_s[first_dive_idx], color='green', linestyle='--', 
                      linewidth=2, label=f'Dive Start (pitch < thresh)')
            ax.plot(time_s[first_dive_idx], altitude_agl[first_dive_idx], 
                   'g^', markersize=12, markeredgewidth=2, markeredgecolor='darkgreen')
        
        # Mark impact (first impact)
        first_impact_idx = engagement_masks.get('first_impact_idx')
        if first_impact_idx is not None:
            ax.axvline(time_s[first_impact_idx], color='red', linestyle='--', 
                      linewidth=2, label='Impact (used)')
            ax.plot(time_s[first_impact_idx], altitude_agl[first_impact_idx], 
                   'rv', markersize=12, markeredgewidth=2, markeredgecolor='darkred')
        
        # Mark all detected impacts (Method 1: acceleration magnitude)
        if 'method1_impact_indices' in data:
            method1_indices = data['method1_impact_indices']
            if len(method1_indices) > 0:
                ax.plot(time_s[method1_indices], altitude_agl[method1_indices], 
                       'o', color='orange', markersize=6, label='Method 1 (accel mag)', alpha=0.6)
        
        # Mark all detected impacts (Method 2: acceleration derivative)
        if 'method2_impact_indices' in data:
            method2_indices = data['method2_impact_indices']
            if len(method2_indices) > 0:
                ax.plot(time_s[method2_indices], altitude_agl[method2_indices], 
                       's', color='purple', markersize=6, label='Method 2 (accel deriv)', alpha=0.6)
        
        # Mark fallback impact if used
        if 'fallback_impact_indices' in data:
            fallback_indices = data['fallback_impact_indices']
            if len(fallback_indices) > 0:
                ax.plot(time_s[fallback_indices], altitude_agl[fallback_indices], 
                       'd', color='cyan', markersize=8, label='Method 3 (fallback)', alpha=0.8)
        
        # Add horizontal line at z=0 (ground level)
        ax.axhline(0, color='brown', linestyle='-', linewidth=2, alpha=0.5, label='Ground (0m AGL)')
        
        # Calculate engagement status
        has_engagement = np.any(mask)
        engagement_status = "✓ Engaged" if has_engagement else "✗ No Engagement"
        status_color = 'green' if has_engagement else 'red'
        
        # Set labels and title
        ax.set_xlabel('Time [s]', fontsize=9)
        ax.set_ylabel('Altitude AGL [m]', fontsize=9)
        ax.set_title(f'{filename}\n{engagement_status}', 
                    fontsize=9, fontweight='bold', color=status_color)
        ax.grid(True, alpha=0.3)
        
        # Add legend (only for first subplot to avoid clutter)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper right')
        
        # Set reasonable y-axis limits
        y_min = min(-5, np.min(altitude_agl) - 5)
        y_max = max(100, np.max(altitude_agl) + 10)
        ax.set_ylim([y_min, y_max])
    
    # Hide unused subplots
    for idx in range(n_trajectories, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if interactive:
        plt.show()
        print("Showing interactive altitude debug plot (close window to continue)")
    else:
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        plot_path = output_dir / f"altitude_debug{target_suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved altitude debug plot: {plot_path}")


def plot_pitch_debug(trajectory_data: list, output_dir: Path, 
                     target_selection: str = "Van", pitch_threshold: float = -4.0, interactive: bool = False):
    """Debug plot showing pitch trajectories with dive and impact detection markers.
    
    Creates one subplot per trajectory showing:
    - Pitch angle over time
    - Dive start markers (pitch threshold crossed)
    - Impact detection markers
    - Terminal engagement segment highlighting
    - Pitch threshold line
    
    Args:
        trajectory_data: List of dicts with 'ulog', 'data', 'mask', 'engagement_masks', 'filename'
        output_dir: Output directory for the plot
        target_selection: Target name for filename suffix
        pitch_threshold: Pitch threshold for dive detection
        interactive: Whether to show interactive plot
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping pitch debug plot.")
        return
    
    n_trajectories = len(trajectory_data)
    if n_trajectories == 0:
        print("No trajectory data available for pitch debug plot.")
        return
    
    # Calculate grid dimensions for subplots
    if n_trajectories == 1:
        nrows, ncols = 1, 1
        figsize = (14, 6)
    elif n_trajectories <= 4:
        nrows, ncols = 2, 2
        figsize = (16, 10)
    elif n_trajectories <= 6:
        nrows, ncols = 2, 3
        figsize = (18, 10)
    elif n_trajectories <= 9:
        nrows, ncols = 3, 3
        figsize = (18, 14)
    elif n_trajectories <= 12:
        nrows, ncols = 3, 4
        figsize = (20, 14)
    elif n_trajectories <= 16:
        nrows, ncols = 4, 4
        figsize = (20, 16)
    elif n_trajectories <= 20:
        nrows, ncols = 4, 5
        figsize = (22, 16)
    elif n_trajectories <= 25:
        nrows, ncols = 5, 5
        figsize = (22, 18)
    else:
        nrows, ncols = 6, 5
        figsize = (22, 20)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_trajectories == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle(f'Pitch Trajectory Debug - Dive & Impact Detection\nTarget: {target_selection}', 
                 fontsize=16, fontweight='bold')
    
    for idx, traj in enumerate(trajectory_data):
        ax = axes[idx]
        data = traj['data']
        mask = traj['mask']
        engagement_masks = traj['engagement_masks']
        filename = traj['filename']
        
        # Get time array in seconds (relative to start)
        time_s = (data['timestamp_us'] - data['timestamp_us'][0]) * 1e-6
        pitch_deg = data['pitch_deg']
        
        # Plot pitch trajectory
        ax.plot(time_s, pitch_deg, 'g-', linewidth=1.5, label='Pitch Angle', alpha=0.7)
        
        # Highlight terminal engagement segment if it exists
        if np.any(mask):
            engagement_indices = np.where(mask)[0]
            ax.fill_between(time_s[engagement_indices], 
                           min(-90, np.min(pitch_deg) - 10), 
                           max(90, np.max(pitch_deg) + 10),
                           color='yellow', alpha=0.2, label='Terminal Engagement')
        
        # Mark dive start
        first_dive_idx = engagement_masks.get('first_dive_idx')
        if first_dive_idx is not None:
            ax.axvline(time_s[first_dive_idx], color='orange', linestyle='--', 
                      linewidth=2, label=f'Dive Start')
            ax.plot(time_s[first_dive_idx], pitch_deg[first_dive_idx], 
                   '^', color='orange', markersize=12, markeredgewidth=2, markeredgecolor='darkorange')
        
        # Mark impact (first impact)
        first_impact_idx = engagement_masks.get('first_impact_idx')
        if first_impact_idx is not None:
            ax.axvline(time_s[first_impact_idx], color='red', linestyle='--', 
                      linewidth=2, label='Impact')
            ax.plot(time_s[first_impact_idx], pitch_deg[first_impact_idx], 
                   'v', color='red', markersize=12, markeredgewidth=2, markeredgecolor='darkred')
        
        # Add horizontal line at pitch threshold
        ax.axhline(pitch_threshold, color='purple', linestyle='--', linewidth=2, 
                  alpha=0.7, label=f'Dive Threshold ({pitch_threshold}°)')
        
        # Add horizontal line at 0 degrees
        ax.axhline(0, color='gray', linestyle='-', linewidth=1, alpha=0.5)
        
        # Calculate engagement status
        has_engagement = np.any(mask)
        engagement_status = "✓ Engaged" if has_engagement else "✗ No Engagement"
        status_color = 'green' if has_engagement else 'red'
        
        # Set labels and title
        ax.set_xlabel('Time [s]', fontsize=9)
        ax.set_ylabel('Pitch [deg]', fontsize=9)
        ax.set_title(f'{filename}\n{engagement_status}', 
                    fontsize=9, fontweight='bold', color=status_color)
        ax.grid(True, alpha=0.3)
        
        # Add legend (only for first subplot to avoid clutter)
        if idx == 0:
            ax.legend(fontsize=7, loc='best')
        
        # Set reasonable y-axis limits
        y_min = min(pitch_threshold - 10, np.min(pitch_deg) - 5)
        y_max = max(20, np.max(pitch_deg) + 5)
        ax.set_ylim([y_min, y_max])
    
    # Hide unused subplots
    for idx in range(n_trajectories, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if interactive:
        plt.show()
        print("Showing interactive pitch debug plot (close window to continue)")
    else:
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        plot_path = output_dir / f"pitch_debug{target_suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved pitch debug plot: {plot_path}")


def plot_dive_angle_summary(angle_summary_data: list, output_dir: Path, target_selection: str, 
                            interactive: bool = False, cpa_hit_threshold: float = 3.0, time_diff_threshold: float = 0.1):
    """Create summary plots showing hit rate and mean CPA as a function of dive angle.
    
    Args:
        angle_summary_data: List of dicts containing angle, count, mean_cpa, std_cpa, etc.
        output_dir: Output directory for plots
        target_selection: Target name for filename
        interactive: Whether to show interactive plot
        cpa_hit_threshold: CPA threshold for hit detection (m)
        time_diff_threshold: Time difference threshold for hit detection (s)
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping dive angle summary plot.")
        return
    
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping dive angle summary plot.")
        return
    
    if len(angle_summary_data) == 0:
        print("No angle summary data available.")
        return
    
    # Sort by angle
    angle_summary_data = sorted(angle_summary_data, key=lambda x: x['angle'])
    
    # Extract data
    angles = [d['angle'] for d in angle_summary_data]
    counts = [d['count'] for d in angle_summary_data]
    mean_cpas = [d['mean_cpa'] for d in angle_summary_data]
    std_cpas = [d['std_cpa'] for d in angle_summary_data]
    
    # Calculate hit rates using successful hits (engagements that met both CPA and time criteria)
    hit_rates = []
    for data in angle_summary_data:
        num_hits = data.get('num_hits', 0)
        total_count = data['count']
        
        if total_count > 0:
            hit_rate = (num_hits / total_count) * 100
        else:
            hit_rate = 0
        hit_rates.append(hit_rate)
    
    # Extract altitude data if available
    mean_altitudes = [d.get('mean_altitude', None) for d in angle_summary_data]
    std_altitudes = [d.get('std_altitude', None) for d in angle_summary_data]
    has_altitude_data = any(a is not None for a in mean_altitudes)
    
    # Create figure with 3 subplots if altitude data available, otherwise 2
    if has_altitude_data:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Subplot 1: Hit Rate vs Dive Angle
    ax1.plot(angles, hit_rates, 'o-', color='steelblue', linewidth=2, markersize=8, label='Hit Rate')
    ax1.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% Line')
    
    # Add data labels with hit rate and sample count
    for angle, hit_rate, count in zip(angles, hit_rates, counts):
        ax1.text(angle, hit_rate + 2, f'{hit_rate:.1f}%', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Add sample count below the data point
        ax1.text(angle, hit_rate - 5, f'n={count}', 
                ha='center', va='top', fontsize=9, style='italic', color='gray')
    
    ax1.set_xlabel('Dive Angle [degrees]', fontsize=12)
    ax1.set_ylabel('Hit Rate [%]', fontsize=12)
    ax1.set_title(f'Hit Rate vs Dive Angle (CPA < {cpa_hit_threshold}m & Impact within {time_diff_threshold}s)\nTarget: {target_selection}', 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 100])
    ax1.set_xticks(angles)
    
    # Subplot 2: Mean CPA Miss Distance vs Dive Angle
    ax2.errorbar(angles, mean_cpas, yerr=std_cpas, fmt='o-', color='coral', 
                linewidth=2, markersize=8, capsize=5, label='Mean CPA ± 1σ')
    ax2.axhline(y=cpa_hit_threshold, color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Hit Threshold ({cpa_hit_threshold}m)')
    
    # Add data labels with mean CPA and sample count
    for angle, mean_cpa, count in zip(angles, mean_cpas, counts):
        ax2.text(angle, mean_cpa + 0.5, f'{mean_cpa:.1f}m', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
        # Add sample count below the data point
        ax2.text(angle, -0.5, f'n={count}', 
                ha='center', va='top', fontsize=9, style='italic', color='gray')
    
    ax2.set_xlabel('Dive Angle [degrees]', fontsize=12)
    ax2.set_ylabel('Mean CPA Miss Distance [m]', fontsize=12)
    ax2.set_title(f'Mean CPA Miss Distance vs Dive Angle\nTarget: {target_selection}', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xticks(angles)
    ax2.set_ylim(bottom=0)
    
    # Subplot 3: Mean Altitude Miss Distance vs Dive Angle (if data available)
    if has_altitude_data:
        ax3.errorbar(angles, mean_altitudes, yerr=std_altitudes, fmt='o-', color='purple', 
                    linewidth=2, markersize=8, capsize=5, label='Mean Altitude Miss ± 1σ')
        ax3.axhline(y=0, color='green', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Target Center')
        
        # Add data labels with mean altitude and sample count
        for angle, mean_alt, count in zip(angles, mean_altitudes, counts):
            if mean_alt is not None:
                ax3.text(angle, mean_alt + 0.3, f'{mean_alt:.1f}m', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax3.set_xlabel('Dive Angle [degrees]', fontsize=12)
        ax3.set_ylabel('Mean Altitude Miss Distance [m]', fontsize=12)
        ax3.set_title(f'Mean Altitude Miss Distance vs Dive Angle\nTarget: {target_selection}', 
                     fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=10)
        ax3.set_xticks(angles)
    
    plt.tight_layout()
    
    if interactive:
        plt.show()
        print("Showing interactive dive angle summary plot (close window to continue)")
    else:
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        plot_path = output_dir / f"dive_angle_summary{target_suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved dive angle summary plot: {plot_path}")


def plot_impact_locations_topdown(output_base_dir: Path, target_selection: str, angle_dirs: list, 
                                   interactive: bool = False):
    """Create top-down scatter plot of impact locations relative to target for each dive angle.
    
    Args:
        output_base_dir: Base output directory containing angle subdirectories
        target_selection: Target name
        angle_dirs: List of tuples (angle_value, angle_dir_path)
        interactive: Whether to show interactive plot
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping impact location plot.")
        return
        
    try:
        from matplotlib import pyplot as plt
        from matplotlib.patches import Circle
    except ImportError:
        print("matplotlib not available, skipping impact location plot.")
        return
    
    # Define target location
    all_targets = {
        'Container': {'lat': 43.2222722, 'lon': -75.3903593},
        'Van': {'lat': 43.2221788, 'lon': -75.3905151},
        'Conex': {'lat': 33.7578867, 'lon': -115.3099750}
    }
    target_info = all_targets[target_selection]
    target_lat = target_info['lat']
    target_lon = target_info['lon']
    
    # Helper function to convert lat/lon to meters from target
    def latlon_to_meters(lat, lon, target_lat, target_lon):
        """Convert lat/lon to x,y meters relative to target using local approximation."""
        # Approximate meters per degree at this latitude
        lat_to_m = 111132.92  # meters per degree latitude
        lon_to_m = 111132.92 * np.cos(np.radians(target_lat))  # meters per degree longitude
        
        x = (lon - target_lon) * lon_to_m
        y = (lat - target_lat) * lat_to_m
        return x, y
    
    # Collect data for each dive angle
    angle_data = []
    for angle_value, _ in angle_dirs:
        angle_output_dir = output_base_dir / f"{angle_value}Deg"
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        results_file = angle_output_dir / f"log_analysis_results{target_suffix}.csv"
        hits_file = angle_output_dir / f"successful_hits{target_suffix}.txt"
        
        if not results_file.exists():
            continue
        
        # Read successful hits count (files that met both CPA and time criteria)
        num_successful_hits = 0
        if hits_file.exists():
            try:
                with open(hits_file, 'r') as f:
                    num_successful_hits = int(f.read().strip())
            except:
                num_successful_hits = 0
        
        # Read impact locations from CSV
        impact_x = []
        impact_y = []
        cpa_distances = []
        filenames = []
        
        try:
            import csv
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Status'] == 'Success' and row.get('Impact_Lat') and row.get('Impact_Lon'):
                        try:
                            impact_lat = float(row['Impact_Lat'])
                            impact_lon = float(row['Impact_Lon'])
                            x, y = latlon_to_meters(impact_lat, impact_lon, target_lat, target_lon)
                            impact_x.append(x)
                            impact_y.append(y)
                            filenames.append(row.get('File', ''))
                            
                            # Get CPA distance if available
                            cpa_col = f'{target_selection}_CPA_Total_m'
                            if cpa_col in row and row[cpa_col]:
                                cpa_distances.append(float(row[cpa_col]))
                            else:
                                cpa_distances.append(float('inf'))
                        except (ValueError, KeyError):
                            pass
            
            # Classify hits: the N files with smallest CPA distances are hits (where N = num_successful_hits)
            is_hit = []
            if len(cpa_distances) > 0:
                # Get indices sorted by CPA distance
                sorted_indices = np.argsort(cpa_distances)
                hit_indices = set(sorted_indices[:num_successful_hits])
                is_hit = [i in hit_indices for i in range(len(cpa_distances))]
            else:
                is_hit = []
            
            if impact_x:
                angle_data.append({
                    'angle': angle_value,
                    'x': np.array(impact_x),
                    'y': np.array(impact_y),
                    'cpa': np.array(cpa_distances),
                    'filenames': filenames,
                    'is_hit': is_hit
                })
        except Exception as e:
            print(f"Warning: Could not read impact locations for {angle_value}°: {e}")
    
    if not angle_data:
        print("No impact location data found for plotting.")
        return
    
    # Create subplots - arrange in grid
    n_angles = len(angle_data)
    n_cols = min(3, n_angles)
    n_rows = int(np.ceil(n_angles / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_angles == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each dive angle
    for idx, data in enumerate(angle_data):
        ax = axes[idx]
        angle = data['angle']
        x = data['x']
        y = data['y']
        cpa = data['cpa']
        filenames = data['filenames']
        hits = data['is_hit']
        
        # Draw target rectangle with dynamic dimensions based on target
        from matplotlib.patches import Rectangle
        # Target dimensions: Van/Container = 2m wide x 5m long, Conex = 2.438m wide x 6.096m long
        if target_selection == 'Conex':
            target_width = 2.438
            target_length = 6.096
        else:
            target_width = 2.0
            target_length = 5.0
        
        target_rect = Rectangle((-target_width/2, -target_length/2), target_width, target_length, 
                               fill=True, facecolor='red', edgecolor='darkred', linewidth=2, alpha=0.3, 
                               label=f'Target ({target_width:.1f}m×{target_length:.1f}m)', zorder=6)
        ax.add_patch(target_rect)
        
        # Plot target center at origin
        ax.plot(0, 0, 'r*', markersize=15, zorder=10)
        
        # Draw circles for hit thresholds
        circle_2m = Circle((0, 0), 2, fill=False, edgecolor='green', linewidth=2, 
                          linestyle='--', label='2m (Hit)', zorder=5)
        circle_3m = Circle((0, 0), 3, fill=False, edgecolor='orange', linewidth=2, 
                          linestyle='--', label='3m (Weak Hit)', zorder=5)
        ax.add_patch(circle_2m)
        ax.add_patch(circle_3m)
        
        # Color code impacts by hit classification (based on CPA and time criteria)
        colors = ['green' if h else 'red' for h in hits]
        
        # Plot impacts
        ax.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors='black', linewidths=1, zorder=8)
        
        # Set equal aspect and limits
        ax.set_aspect('equal', adjustable='box')
        max_range = max(10, np.max(np.abs([x, y])) * 1.2) if len(x) > 0 else 10
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        ax.set_xlabel('East-West Distance from Target [m]', fontsize=10)
        ax.set_ylabel('North-South Distance from Target [m]', fontsize=10)
        ax.set_title(f'Dive Angle: {angle}° (n={len(x)})', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_angles, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Impact Locations (Top-Down View) - Target: {target_selection}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if interactive:
        plt.show()
    else:
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        plot_path = output_base_dir / f"impact_locations_topdown{target_suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved impact locations top-down plot: {plot_path}")


def plot_cpa_miss_components(output_base_dir: Path, target_selection: str, angle_dirs: list, 
                             cpa_hit_threshold: float = 3.0, time_diff_threshold: float = 0.1,
                             interactive: bool = False):
    """Create scatter plot of CPA miss distance broken into horizontal and altitude components.
    
    Args:
        output_base_dir: Base output directory containing angle subdirectories
        target_selection: Target name
        angle_dirs: List of tuples (angle_value, angle_dir_path)
        cpa_hit_threshold: CPA threshold for hit classification
        time_diff_threshold: Time threshold for hit classification
        interactive: Whether to show interactive plot
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping CPA components plot.")
        return
        
    try:
        from matplotlib import pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib not available, skipping CPA components plot.")
        return
    
    # Define target location and dimensions
    all_targets = {
        'Container': {'lat': 43.2222722, 'lon': -75.3903593, 'alt_agl': 2.0},
        'Van': {'lat': 43.2221788, 'lon': -75.3905151, 'alt_agl': 1.0},
        'Conex': {'lat': 33.7578867, 'lon': -115.3099750, 'alt_agl': 2.583}
    }
    target_info = all_targets[target_selection]
    target_lat = target_info['lat']
    target_lon = target_info['lon']
    target_alt = target_info['alt_agl']
    
    # Helper function to convert lat/lon to meters from target
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate ground distance in meters."""
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
    
    # Collect data for each dive angle
    angle_data = []
    for angle_value, _ in angle_dirs:
        angle_output_dir = output_base_dir / f"{angle_value}Deg"
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        results_file = angle_output_dir / f"log_analysis_results{target_suffix}.csv"
        hits_file = angle_output_dir / f"successful_hits{target_suffix}.txt"
        
        if not results_file.exists():
            continue
        
        # Read successful hits count (files that met both CPA and time criteria)
        num_successful_hits = 0
        if hits_file.exists():
            try:
                with open(hits_file, 'r') as f:
                    num_successful_hits = int(f.read().strip())
            except:
                num_successful_hits = 0
        
        # Read CPA locations from CSV
        horizontal_distances = []
        altitude_differences = []
        cpa_total_distances = []
        
        try:
            import csv
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Status'] == 'Success' and row.get('CPA_Lat') and row.get('CPA_Lon'):
                        try:
                            cpa_lat = float(row['CPA_Lat'])
                            cpa_lon = float(row['CPA_Lon'])
                            cpa_alt_agl = float(row['CPA_Alt_AGL_m'])
                            
                            # Calculate horizontal distance at CPA
                            horiz_dist = haversine_distance(cpa_lat, cpa_lon, target_lat, target_lon)
                            
                            # Calculate altitude difference at CPA
                            alt_diff = cpa_alt_agl - target_alt
                            
                            horizontal_distances.append(horiz_dist)
                            altitude_differences.append(alt_diff)
                            
                            # Store CPA total distance for sorting
                            cpa_col = f'{target_selection}_CPA_Total_m'
                            if cpa_col in row and row[cpa_col]:
                                cpa_total_distances.append(float(row[cpa_col]))
                            else:
                                cpa_total_distances.append(float('inf'))
                        except (ValueError, KeyError):
                            pass
            
            # Classify hits: the N files with smallest CPA distances are hits (where N = num_successful_hits)
            # These are the files that met both CPA < threshold AND time < threshold
            is_hit = []
            if len(cpa_total_distances) > 0:
                # Get indices sorted by CPA distance
                sorted_indices = np.argsort(cpa_total_distances)
                hit_indices = set(sorted_indices[:num_successful_hits])
                is_hit = [i in hit_indices for i in range(len(cpa_total_distances))]
            else:
                is_hit = []
            
            if horizontal_distances:
                angle_data.append({
                    'angle': angle_value,
                    'horizontal': np.array(horizontal_distances),
                    'altitude': np.array(altitude_differences),
                    'is_hit': np.array(is_hit)
                })
        except Exception as e:
            print(f"Warning: Could not read CPA components for {angle_value}°: {e}")
    
    if not angle_data:
        print("No CPA component data found for plotting.")
        return
    
    # Create subplots - arrange in grid
    n_angles = len(angle_data)
    n_cols = min(3, n_angles)
    n_rows = int(np.ceil(n_angles / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_angles == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each dive angle
    for idx, data in enumerate(angle_data):
        ax = axes[idx]
        angle = data['angle']
        horiz = data['horizontal']
        alt = data['altitude']
        hits = data['is_hit']
        
        # Draw target rectangle with dynamic dimensions and altitude based on target
        # Target dimensions: Van/Container = 5m wide x 2m tall, Conex = 6.096m wide x 2.591m tall
        target_center_alt = target_alt
        if target_selection == 'Conex':
            target_width = 6.096
            target_height = 2.591
        else:
            target_width = 5.0
            target_height = 2.0
        
        target_rect = Rectangle((0, target_center_alt - target_height/2), target_width, target_height, 
                               fill=True, facecolor='blue', edgecolor='darkblue', linewidth=2, alpha=0.3, 
                               label=f'Target ({target_width:.1f}m×{target_height:.1f}m)', zorder=6)
        ax.add_patch(target_rect)
        
        # Plot target center at (0, 1m altitude)
        ax.plot(0, target_center_alt, 'b*', markersize=15, zorder=10)
        
        # Color code by hit classification
        colors = ['green' if h else 'red' for h in hits]
        
        # Plot CPA miss components
        ax.scatter(horiz, alt, c=colors, s=100, alpha=0.7, edgecolors='black', linewidths=1, zorder=8)
        
        # Add reference lines at target center
        ax.axhline(target_center_alt, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
        
        # Set limits with some padding
        max_horiz = max(10, np.max(np.abs(horiz)) * 1.2) if len(horiz) > 0 else 10
        max_alt = max(10, np.max(np.abs(alt)) * 1.2) if len(alt) > 0 else 10
        ax.set_xlim(-2, max_horiz)
        ax.set_ylim(-max_alt, max_alt)
        
        # Add grid and labels
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Horizontal Miss Distance at CPA [m]', fontsize=10)
        ax.set_ylabel('Altitude Miss Distance at CPA [m]', fontsize=10)
        ax.set_title(f'Dive Angle: {angle}° (n={len(horiz)})', fontsize=12, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            target_rect,
            Patch(facecolor='green', edgecolor='black', label=f'Hit (CPA<{cpa_hit_threshold}m)'),
            Patch(facecolor='red', edgecolor='black', label='Miss')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    # Hide unused subplots
    for idx in range(n_angles, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'CPA Miss Components (Horizontal vs Altitude) - Target: {target_selection}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if interactive:
        plt.show()
    else:
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        plot_path = output_base_dir / f"cpa_miss_components{target_suffix}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved CPA miss components plot: {plot_path}")


def create_comprehensive_pdf(output_base_dir: Path, target_selection: str, angle_dirs: list):
    """Create a comprehensive PDF report with summary plots and all histogram plots.
    
    Args:
        output_base_dir: Base output directory containing angle subdirectories
        target_selection: Target name
        angle_dirs: List of tuples (angle_value, angle_dir_path)
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL/Pillow not available. Install with: pip install Pillow")
        print("Skipping comprehensive PDF generation.")
        return
    
    target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
    
    # List of plot files to include in order
    plot_files = []
    
    # Add summary plot first
    summary_plot = output_base_dir / f"dive_angle_summary{target_suffix}.png"
    if summary_plot.exists():
        plot_files.append(summary_plot)
    
    # Add impact locations top-down plot
    topdown_plot = output_base_dir / f"impact_locations_topdown{target_suffix}.png"
    if topdown_plot.exists():
        plot_files.append(topdown_plot)
    
    # Add CPA miss components plot
    components_plot = output_base_dir / f"cpa_miss_components{target_suffix}.png"
    if components_plot.exists():
        plot_files.append(components_plot)
    
    # Add setpoint tracking analysis plots
    pitch_tracking = output_base_dir / f"setpoint_tracking_pitch{target_suffix}.png"
    if pitch_tracking.exists():
        plot_files.append(pitch_tracking)
    
    roll_tracking = output_base_dir / f"setpoint_tracking_roll{target_suffix}.png"
    if roll_tracking.exists():
        plot_files.append(roll_tracking)
    
    # Add histogram plots for each dive angle
    for angle_value, _ in sorted(angle_dirs, key=lambda x: x[0]):
        angle_output_dir = output_base_dir / f"{angle_value}Deg"
        
        # Miss distance histograms
        miss_hist = angle_output_dir / f"miss_distance_histograms{target_suffix}.png"
        if miss_hist.exists():
            plot_files.append(miss_hist)
        
        # Impact angle histograms
        impact_hist = angle_output_dir / f"impact_angle_histogram{target_suffix}.png"
        if impact_hist.exists():
            plot_files.append(impact_hist)
    
    if not plot_files:
        print("No plot files found for comprehensive PDF.")
        return
    
    # Load all images
    images = []
    for plot_file in plot_files:
        try:
            img = Image.open(plot_file)
            # Convert to RGB if necessary (PDF requires RGB)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not open {plot_file.name}: {e}")
    
    if not images:
        print("No images loaded for comprehensive PDF.")
        return
    
    # Save as PDF
    pdf_path = output_base_dir / f"comprehensive_report_{target_selection.lower()}.pdf"
    
    try:
        # Save first image and append the rest
        images[0].save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=images[1:])
        print(f"Created comprehensive PDF: {pdf_path}")
    except Exception as e:
        print(f"Error creating comprehensive PDF: {e}")


def plot_setpoint_tracking_analysis(output_base_dir: Path, target_selection: str, angle_dirs: list,
                                    interactive: bool = False):
    """Create detailed setpoint tracking plots for pitch and roll during offboard mode.
    
    Analyzes the 10° and 20° dive cases, showing the best hit and 5 worst misses for each.
    
    Args:
        output_base_dir: Base output directory containing angle subdirectories
        target_selection: Target name
        angle_dirs: List of tuples (angle_value, angle_dir_path)
        interactive: Whether to show interactive plot
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping setpoint tracking analysis.")
        return
        
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping setpoint tracking analysis.")
        return
    
    # Filter to only 10° and 20° cases
    target_angles = [10, 20]
    angle_dirs_filtered = [(angle, path) for angle, path in angle_dirs if angle in target_angles]
    
    if len(angle_dirs_filtered) == 0:
        print("No 10° or 20° dive angle data found for setpoint tracking analysis.")
        return
    
    # Collect data for each angle
    tracking_data = {}
    
    for angle_value, angle_dir in angle_dirs_filtered:
        angle_output_dir = output_base_dir / f"{angle_value}Deg"
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        results_file = angle_output_dir / f"log_analysis_results{target_suffix}.csv"
        
        if not results_file.exists():
            continue
        
        # Read CPA values and filenames
        log_data = []
        try:
            import csv
            with open(results_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['Status'] == 'Success':
                        cpa_col = f'{target_selection}_CPA_Total_m'
                        if cpa_col in row and row[cpa_col]:
                            try:
                                cpa_value = float(row[cpa_col])
                                filename = row['File']
                                log_data.append({
                                    'filename': filename,
                                    'cpa': cpa_value,
                                    'path': angle_dir / filename
                                })
                            except (ValueError, KeyError):
                                pass
        except Exception as e:
            print(f"Warning: Could not read log data for {angle_value}°: {e}")
            continue
        
        if len(log_data) == 0:
            continue
        
        # Sort by CPA (best to worst)
        log_data.sort(key=lambda x: x['cpa'])
        
        # Select best hit (minimum CPA) and 5 worst misses (maximum CPA)
        selected_logs = []
        if len(log_data) > 0:
            selected_logs.append(('Best Hit', log_data[0]))
        if len(log_data) >= 6:
            # Get 5 worst misses
            for i, log in enumerate(log_data[-5:], 1):
                selected_logs.append((f'Worst Miss #{i}', log))
        elif len(log_data) > 1:
            # If less than 6 logs, get all remaining as worst misses
            for i, log in enumerate(log_data[1:], 1):
                selected_logs.append((f'Miss #{i}', log))
        
        tracking_data[angle_value] = selected_logs
    
    if len(tracking_data) == 0:
        print("No tracking data collected for setpoint analysis.")
        return
    
    # Create pitch tracking plot (12 subplots: 6 for 10°, 6 for 20°)
    fig_pitch, axes_pitch = plt.subplots(6, 2, figsize=(16, 24), sharey=True)
    fig_pitch.suptitle(f'Pitch Setpoint Tracking Analysis - Target: {target_selection}', 
                       fontsize=16, fontweight='bold')
    
    # Create roll tracking plot (12 subplots: 6 for 10°, 6 for 20°)
    fig_roll, axes_roll = plt.subplots(6, 2, figsize=(16, 24), sharey=True)
    fig_roll.suptitle(f'Roll Setpoint Tracking Analysis - Target: {target_selection}',
                      fontsize=16, fontweight='bold')
    
    # Process each angle
    for col_idx, angle_value in enumerate(target_angles):
        if angle_value not in tracking_data:
            # Hide unused subplots
            for row_idx in range(6):
                axes_pitch[row_idx, col_idx].set_visible(False)
                axes_roll[row_idx, col_idx].set_visible(False)
            continue
        
        selected_logs = tracking_data[angle_value]
        
        # Plot each selected log
        for row_idx in range(6):
            ax_pitch = axes_pitch[row_idx, col_idx]
            ax_roll = axes_roll[row_idx, col_idx]
            
            if row_idx < len(selected_logs):
                label, log_info = selected_logs[row_idx]
                filename = log_info['filename']
                cpa = log_info['cpa']
                log_path = log_info['path']
                
                try:
                    # Load and process the log
                    from TG_strike_analysis import load_log, extract_position_attitude
                    ulog = load_log(log_path)
                    data = extract_position_attitude(ulog)
                    
                    # Find offboard mode segments
                    offboard_mask = data.get('offboard_mode', np.zeros(len(data['timestamp_us']), dtype=bool))
                    
                    if np.any(offboard_mask):
                        # Get time array (relative to start of offboard)
                        offboard_indices = np.where(offboard_mask)[0]
                        first_offboard = offboard_indices[0]
                        time_array = (data['timestamp_us'] - data['timestamp_us'][first_offboard]) * 1e-6
                        
                        # Extract pitch data
                        pitch_actual = data['pitch_deg']
                        pitch_sp = data['pitch_sp_deg']
                        
                        # Extract roll data
                        roll_actual = data['roll_deg']
                        roll_sp = data['roll_sp_deg']
                        
                        # Calculate L1 errors (only during offboard)
                        pitch_error = pitch_actual[offboard_mask] - pitch_sp[offboard_mask]
                        roll_error = roll_actual[offboard_mask] - roll_sp[offboard_mask]
                        
                        # Remove NaN values before calculating L1 error
                        pitch_error_valid = pitch_error[~np.isnan(pitch_error)]
                        roll_error_valid = roll_error[~np.isnan(roll_error)]
                        pitch_l1_error = np.mean(np.abs(pitch_error_valid)) if len(pitch_error_valid) > 0 else 0.0
                        roll_l1_error = np.mean(np.abs(roll_error_valid)) if len(roll_error_valid) > 0 else 0.0
                        
                        # Plot pitch tracking (green to match combined RPY plot)
                        ax_pitch.plot(time_array[offboard_mask], pitch_actual[offboard_mask], 
                                     'g-', linewidth=2, label='Actual', alpha=0.8)
                        ax_pitch.plot(time_array[offboard_mask], pitch_sp[offboard_mask], 
                                     'g--', linewidth=2, label='Setpoint', alpha=0.8)
                        ax_pitch.fill_between(time_array[offboard_mask], 
                                             pitch_actual[offboard_mask], 
                                             pitch_sp[offboard_mask],
                                             alpha=0.2, color='green')
                        ax_pitch.set_xlabel('Time [s]', fontsize=10)
                        ax_pitch.set_ylabel('Pitch [deg]', fontsize=10)
                        ax_pitch.set_title(f'{angle_value}° - {label}\nCPA: {cpa:.2f}m | L1 Error: {pitch_l1_error:.2f}°',
                                          fontsize=11, fontweight='bold')
                        ax_pitch.grid(True, alpha=0.3)
                        ax_pitch.legend(fontsize=9, loc='best')
                        
                        # Plot roll tracking (blue to match combined RPY plot)
                        ax_roll.plot(time_array[offboard_mask], roll_actual[offboard_mask], 
                                    'b-', linewidth=2, label='Actual', alpha=0.8)
                        ax_roll.plot(time_array[offboard_mask], roll_sp[offboard_mask], 
                                    'b--', linewidth=2, label='Setpoint', alpha=0.8)
                        ax_roll.fill_between(time_array[offboard_mask], 
                                            roll_actual[offboard_mask], 
                                            roll_sp[offboard_mask],
                                            alpha=0.2, color='blue')
                        ax_roll.set_xlabel('Time [s]', fontsize=10)
                        ax_roll.set_ylabel('Roll [deg]', fontsize=10)
                        ax_roll.set_title(f'{angle_value}° - {label}\nCPA: {cpa:.2f}m | L1 Error: {roll_l1_error:.2f}°',
                                         fontsize=11, fontweight='bold')
                        ax_roll.grid(True, alpha=0.3)
                        ax_roll.legend(fontsize=9, loc='best')
                    else:
                        ax_pitch.text(0.5, 0.5, 'No Offboard Mode', transform=ax_pitch.transAxes,
                                     ha='center', va='center', fontsize=12)
                        ax_pitch.set_title(f'{angle_value}° - {label}\nCPA: {cpa:.2f}m', fontsize=11)
                        ax_roll.text(0.5, 0.5, 'No Offboard Mode', transform=ax_roll.transAxes,
                                    ha='center', va='center', fontsize=12)
                        ax_roll.set_title(f'{angle_value}° - {label}\nCPA: {cpa:.2f}m', fontsize=11)
                        
                except Exception as e:
                    print(f"Warning: Could not process {filename}: {e}")
                    ax_pitch.text(0.5, 0.5, f'Error loading\n{filename}', 
                                 transform=ax_pitch.transAxes, ha='center', va='center', fontsize=10)
                    ax_pitch.set_title(f'{angle_value}° - {label}', fontsize=11)
                    ax_roll.text(0.5, 0.5, f'Error loading\n{filename}', 
                                transform=ax_roll.transAxes, ha='center', va='center', fontsize=10)
                    ax_roll.set_title(f'{angle_value}° - {label}', fontsize=11)
            else:
                # Hide unused subplot
                ax_pitch.set_visible(False)
                ax_roll.set_visible(False)
    
    # Save plots
    plt.figure(fig_pitch.number)
    plt.tight_layout()
    if interactive:
        plt.show()
    else:
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        pitch_path = output_base_dir / f"setpoint_tracking_pitch{target_suffix}.png"
        plt.savefig(pitch_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved pitch tracking analysis: {pitch_path}")
    
    plt.figure(fig_roll.number)
    plt.tight_layout()
    if interactive:
        plt.show()
    else:
        roll_path = output_base_dir / f"setpoint_tracking_roll{target_suffix}.png"
        plt.savefig(roll_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved roll tracking analysis: {roll_path}")


