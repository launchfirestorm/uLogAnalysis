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


def set_matplotlib_backend(interactive: bool = False):
    """Set the appropriate matplotlib backend based on interactive mode."""
    if not matplotlib_available:
        return
        
    try:
        import matplotlib
        # Force backend selection before importing pyplot
        if interactive:
            # Try TkAgg first, fallback to Qt5Agg
            try:
                matplotlib.use('TkAgg', force=True)
                backend = 'TkAgg'
            except ImportError:
                try:
                    matplotlib.use('Qt5Agg', force=True)
                    backend = 'Qt5Agg'
                except ImportError:
                    matplotlib.use('Agg', force=True)  # Fallback to non-interactive
                    backend = 'Agg'
                    print("Warning: No interactive backend available, using Agg")
        else:
            matplotlib.use('Agg', force=True)
            backend = 'Agg'
        
        print(f"Using {'interactive' if interactive else 'static'} matplotlib backend: {backend}")
    except Exception as e:
        print(f"Backend setup error: {e}")


def plot_3d_terminal_engagement(ulog, data: Dict[str, np.ndarray], mask: np.ndarray, out_dir: Path, 
                                target_selection: str = "Container", 
                                interactive: bool = False, engagement_masks: Dict = None):
    """Create 3D trajectory plot for terminal engagement segments only.
    
    This plot shows only the masked terminal engagement portion (dive to impact).
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

    # Define target locations
    all_targets = {
        'Container': {'x': 0, 'y': 0, 'z': 0, 'color': 'red', 'marker': 's'},
        'Van': {'x': -19.5, 'y': -14.4, 'z': 0, 'color': 'green', 'marker': 'o'}
    }
    
    # Filter target display based on selection
    target_display = {target_selection: all_targets[target_selection]}
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get indices within the mask for marking dive and impact
    mask_indices = np.where(mask)[0]
    first_dive_idx = engagement_masks.get('first_dive_idx') if engagement_masks else mask_indices[0]
    first_impact_idx = engagement_masks.get('first_impact_idx') if engagement_masks else mask_indices[-1]
    
    # Find positions within masked data
    dive_pos_in_mask = np.where(mask_indices == first_dive_idx)[0]
    impact_pos_in_mask = np.where(mask_indices == first_impact_idx)[0]
    
    # Plot trajectory (only terminal engagement)
    trajectory_x = data["x"][mask] - data["x"][mask][0]
    trajectory_y = data["y"][mask] - data["y"][mask][0]
    trajectory_z = data["altitude_agl"][mask]
    
    ax.plot(trajectory_x, trajectory_y, trajectory_z, 'b-', linewidth=2, label='Terminal Engagement')
    
    # Mark dive start (not "start" but "dive start")
    if len(dive_pos_in_mask) > 0:
        dive_idx = dive_pos_in_mask[0]
        ax.scatter(trajectory_x[dive_idx], trajectory_y[dive_idx], trajectory_z[dive_idx], 
                  color='orange', s=150, marker='^', edgecolors='black', linewidth=2, 
                  label='Dive Start', zorder=6)
    
    # Mark impact point
    if len(impact_pos_in_mask) > 0:
        impact_idx = impact_pos_in_mask[0]
        ax.scatter(trajectory_x[impact_idx], trajectory_y[impact_idx], trajectory_z[impact_idx], 
                  color='red', s=200, marker='X', edgecolors='black', linewidth=2, 
                  label='Impact', zorder=6)
    
    # Plot targets
    for target_name, target_info in target_display.items():
        ax.scatter(target_info['x'], target_info['y'], target_info['z'], 
                  color=target_info['color'], s=200, marker=target_info['marker'], 
                  label=f'{target_name} Target', zorder=10, edgecolors='black', linewidth=2)
    
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Altitude AGL [m]')
    ax.set_title(f'3D Terminal Engagement Trajectory (Target: {target_selection})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set equal aspect ratio
    max_range = np.array([trajectory_x.max()-trajectory_x.min(), 
                         trajectory_y.max()-trajectory_y.min(),
                         trajectory_z.max()-trajectory_z.min()]).max() / 2.0
    mid_x = (trajectory_x.max()+trajectory_x.min()) * 0.5
    mid_y = (trajectory_y.max()+trajectory_y.min()) * 0.5
    mid_z = (trajectory_z.max()+trajectory_z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if interactive:
        plt.show()  # Show interactive plot
        print("Showing interactive 3D terminal engagement plot (close window to continue)")
    else:
        # Save plot
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        plot_path = out_dir / f"3d_terminal_engagement{target_suffix}.png"
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
                
                # Convert to AGL using local position reference
                gps_alt = gps_alt_msl - np.mean(gps_alt_msl) + np.mean(data["altitude_agl"])
                
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
        time_min = max(0, impact_times.min() - 3.0)
        time_max = impact_times.max() + 1.0
        alt_min = -2.0  # Show slightly below ground
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


