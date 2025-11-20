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
        'Van': {'lat': 43.2221788, 'lon': -75.3905151, 'alt_agl': 1.0, 'color': 'green', 'marker': 'o'}
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
        print(f"Saved 3D trajectory plot: {plot_path}")


def plot_roll_pitch(data: Dict[str, np.ndarray], mask: np.ndarray, out_dir: Path, 
                   interactive: bool = False, engagement_masks: Dict = None):
    """Create roll and pitch timeseries plots with setpoints during terminal engagement only.
    
    This plot shows only the masked terminal engagement portion (dive to impact).
    """
    if not matplotlib_available:
        print("matplotlib not available, skipping roll/pitch timeseries plot.")
        return
        
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping roll/pitch timeseries plot.")
        return
    
    # Get indices within the mask for marking dive and impact
    mask_indices = np.where(mask)[0]
    first_dive_idx = engagement_masks.get('first_dive_idx') if engagement_masks else mask_indices[0]
    first_impact_idx = engagement_masks.get('first_impact_idx') if engagement_masks else mask_indices[-1]
    
    # Find positions within masked data
    dive_pos_in_mask = np.where(mask_indices == first_dive_idx)[0]
    impact_pos_in_mask = np.where(mask_indices == first_impact_idx)[0]
    
    # Create time array relative to start of engagement
    time_engagement = (data["timestamp_us"][mask] - data["timestamp_us"][mask][0]) * 1e-6
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Roll
    axes[0].plot(time_engagement, data["roll_deg"][mask], 'b-', linewidth=2, label='Roll')
    if 'roll_sp_deg' in data and np.any(data["roll_sp_deg"][mask] != 0):
        axes[0].plot(time_engagement, data["roll_sp_deg"][mask], 'b--', linewidth=1.5, alpha=0.7, label='Roll Setpoint')
    
    # Mark dive and impact on roll plot
    if len(dive_pos_in_mask) > 0:
        dive_idx = dive_pos_in_mask[0]
        axes[0].axvline(x=time_engagement[dive_idx], color='orange', linestyle=':', alpha=0.7, linewidth=2)
        axes[0].scatter(time_engagement[dive_idx], data["roll_deg"][mask][dive_idx], 
                       color='orange', s=100, marker='^', edgecolors='black', linewidth=1.5, zorder=5)
    if len(impact_pos_in_mask) > 0:
        impact_idx = impact_pos_in_mask[0]
        axes[0].axvline(x=time_engagement[impact_idx], color='red', linestyle=':', alpha=0.7, linewidth=2)
        axes[0].scatter(time_engagement[impact_idx], data["roll_deg"][mask][impact_idx], 
                       color='red', s=120, marker='X', edgecolors='black', linewidth=1.5, zorder=5)
    
    axes[0].set_ylabel('Roll [deg]')
    axes[0].set_title('Roll and Pitch During Terminal Engagement')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Pitch
    axes[1].plot(time_engagement, data["pitch_deg"][mask], 'g-', linewidth=2, label='Pitch')
    if 'pitch_sp_deg' in data and np.any(data["pitch_sp_deg"][mask] != 0):
        axes[1].plot(time_engagement, data["pitch_sp_deg"][mask], 'g--', linewidth=1.5, alpha=0.7, label='Pitch Setpoint')
    
    # Mark dive and impact on pitch plot
    if len(dive_pos_in_mask) > 0:
        dive_idx = dive_pos_in_mask[0]
        axes[1].axvline(x=time_engagement[dive_idx], color='orange', linestyle=':', alpha=0.7, linewidth=2, label='Dive Start')
        axes[1].scatter(time_engagement[dive_idx], data["pitch_deg"][mask][dive_idx], 
                       color='orange', s=100, marker='^', edgecolors='black', linewidth=1.5, zorder=5)
    if len(impact_pos_in_mask) > 0:
        impact_idx = impact_pos_in_mask[0]
        axes[1].axvline(x=time_engagement[impact_idx], color='red', linestyle=':', alpha=0.7, linewidth=2, label='Impact')
        axes[1].scatter(time_engagement[impact_idx], data["pitch_deg"][mask][impact_idx], 
                       color='red', s=120, marker='X', edgecolors='black', linewidth=1.5, zorder=5)
    
    axes[1].set_ylabel('Pitch [deg]')
    axes[1].set_xlabel('Time [s]')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=-4.0, color='purple', linestyle='--', alpha=0.5, linewidth=1.5, label='Dive threshold')
    axes[1].legend()
    
    plt.tight_layout()
    
    if interactive:
        plt.show()
        print("Showing interactive roll/pitch timeseries plot (close window to continue)")
    else:
        plot_path = out_dir / "roll_pitch_timeseries.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved roll/pitch timeseries plot: {plot_path}")


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
        'Van': {'lat': 43.2221788, 'lon': -75.3905151, 'alt_agl': 0.0}
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
    print(f"\n=== Creating GPS Trajectory Plot for {len(trajectory_data)} {file_word} ===")
    
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
            if dive_lat is not None and impact_lat is not None:
                print(f"  Processed {filename}: Dive at ({dive_lat:.7f}, {dive_lon:.7f}), Impact at ({impact_lat:.7f}, {impact_lon:.7f})")
            elif dive_lat is not None or impact_lat is not None:
                print(f"  Processed {filename}: Partial terminal engagement detected")
            else:
                print(f"  Processed {filename}: No terminal engagement detected")
                
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
        print(f"Saved GPS trajectory plot: {gps_plot_path}")


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
    print(f"\n=== Creating Accelerometer Impact Plot for {n_trajectories} {file_word} ===")
    
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
            print(f"  Processed {filename}: {impact_count} impact(s) detected (M1: {len(valid_method1)}, M2: {len(valid_method2)})")
            
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
        print(f"Saved Method 1 accelerometer plot: {plot_path}")
    
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
    
    print(f"\n=== Creating Method 2 (Derivative) Plot for {n_trajectories} {file_word} ===")
    
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
        print(f"Saved Method 2 derivative plot: {plot_path}")


