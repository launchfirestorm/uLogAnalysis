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
                                target_selection: str = "Container", show_setpoints: bool = False, 
                                interactive: bool = False):
    """Create 3D trajectory plot for terminal engagement segments."""
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
    
    # Plot trajectory
    trajectory_x = data["x"][mask] - data["x"][mask][0]
    trajectory_y = data["y"][mask] - data["y"][mask][0]
    trajectory_z = data["altitude_agl"][mask]
    
    ax.plot(trajectory_x, trajectory_y, trajectory_z, 'b-', linewidth=2, label='Trajectory')
    
    # Mark start and end points
    ax.scatter(trajectory_x[0], trajectory_y[0], trajectory_z[0], 
              color='green', s=100, marker='o', label='Start', zorder=5)
    ax.scatter(trajectory_x[-1], trajectory_y[-1], trajectory_z[-1], 
              color='red', s=100, marker='x', label='End', zorder=5)
    
    # Plot targets
    for target_name, target_info in target_display.items():
        ax.scatter(target_info['x'], target_info['y'], target_info['z'], 
                  color=target_info['color'], s=200, marker=target_info['marker'], 
                  label=f'{target_name} Target', zorder=10, edgecolors='black', linewidth=2)
    
    # Plot setpoints if requested
    if show_setpoints:
        try:
            pos_sp_data = ulog.get_dataset("vehicle_local_position_setpoint").data
            sp_timestamps = pos_sp_data['timestamp']
            sp_x = pos_sp_data['x'] - data["x"][mask][0]
            sp_y = pos_sp_data['y'] - data["y"][mask][0]
            sp_z = pos_sp_data['z']
            
            # Interpolate setpoints to trajectory timestamps
            from scipy.interpolate import interp1d
            interp_sp_x = interp1d(sp_timestamps, sp_x, bounds_error=False, fill_value=np.nan)
            interp_sp_y = interp1d(sp_timestamps, sp_y, bounds_error=False, fill_value=np.nan)
            interp_sp_z = interp1d(sp_timestamps, sp_z, bounds_error=False, fill_value=np.nan)
            
            traj_timestamps = data["timestamp_us"][mask]
            sp_x_interp = interp_sp_x(traj_timestamps)
            sp_y_interp = interp_sp_y(traj_timestamps)
            sp_z_interp = interp_sp_z(traj_timestamps)
            
            valid_sp = ~(np.isnan(sp_x_interp) | np.isnan(sp_y_interp) | np.isnan(sp_z_interp))
            if np.any(valid_sp):
                ax.plot(sp_x_interp[valid_sp], sp_y_interp[valid_sp], sp_z_interp[valid_sp], 
                       'r--', linewidth=1, alpha=0.7, label='Position Setpoint')
        except Exception as e:
            print(f"Warning: Could not plot setpoints: {e}")
    
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


def plot_roll_pitch(data: Dict[str, np.ndarray], mask: np.ndarray, out_dir: Path):
    """Create roll and pitch timeseries plots with setpoints during terminal engagement."""
    if not matplotlib_available:
        print("matplotlib not available, skipping RPY timeseries plot.")
        return
        
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping RPY timeseries plot.")
        return
    
    # Create time array relative to start of engagement
    time_engagement = (data["timestamp_us"][mask] - data["timestamp_us"][mask][0]) * 1e-6
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Roll
    axes[0].plot(time_engagement, data["roll_deg"][mask], 'b-', linewidth=2, label='Roll')
    if 'roll_sp_deg' in data and np.any(data["roll_sp_deg"][mask] != 0):
        axes[0].plot(time_engagement, data["roll_sp_deg"][mask], 'b--', linewidth=1.5, alpha=0.7, label='Roll Setpoint')
    axes[0].set_ylabel('Roll [deg]')
    axes[0].set_title('Roll and Pitch During Terminal Engagement')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Pitch
    axes[1].plot(time_engagement, data["pitch_deg"][mask], 'g-', linewidth=2, label='Pitch')
    if 'pitch_sp_deg' in data and np.any(data["pitch_sp_deg"][mask] != 0):
        axes[1].plot(time_engagement, data["pitch_sp_deg"][mask], 'g--', linewidth=1.5, alpha=0.7, label='Pitch Setpoint')
    axes[1].set_ylabel('Pitch [deg]')
    axes[1].set_xlabel('Time [s]')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=-4.0, color='red', linestyle='--', alpha=0.7, label='Dive threshold')
    axes[1].legend()
    
    plt.tight_layout()
    
    plot_path = out_dir / "rpy_timeseries.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved roll/pitch timeseries plot: {plot_path}")


def gps_trajectory_plot(log_files: list, output_dir: Path, target_selection: str,
                                        pitch_threshold: float = -4.0, accel_threshold: float = 15.0, 
                                        alt_threshold: float = 5.0, interactive: bool = False) -> None:
    """Create combined GPS trajectory plots with lat/lon and altitude subplots for all logs."""
    if not matplotlib_available:
        print("matplotlib not available, skipping combined GPS trajectory plot.")
        return
        
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping combined GPS trajectory plot.")
        return
    
    # Import here to avoid circular imports
    from tg_utils import load_log, extract_position_attitude, compute_terminal_engagement_mask
    
    # Define target locations for reference
    all_targets = {
        'Container': {'lat': 43.2222722, 'lon': -75.3903593, 'alt_agl': 0.0},
        'Van': {'lat': 43.2221788, 'lon': -75.3905151, 'alt_agl': 0.0}
    }
    target_info = all_targets[target_selection]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
    
    file_word = "file" if len(log_files) == 1 else "files"
    print(f"\n=== Creating GPS Trajectory Plot for {len(log_files)} {file_word} ===")
    
    for i, log_file in enumerate(log_files):
        try:
            # Load and process log
            ulog = load_log(log_file)
            data = extract_position_attitude(ulog)
            
            # Calculate terminal engagement mask
            mask, engagement_masks = compute_terminal_engagement_mask(data, pitch_threshold, 
                                                                    accel_threshold, alt_threshold)
            
            if not np.any(mask):
                continue
                
            # Get GPS data
            gps_data = ulog.get_dataset("vehicle_gps_position").data
            valid_fix = gps_data['fix_type'] > 2
            
            if not np.any(valid_fix):
                continue
                
            gps_lat = gps_data['lat'][valid_fix] / 1e7
            gps_lon = gps_data['lon'][valid_fix] / 1e7
            gps_alt_msl = gps_data['alt'][valid_fix] / 1e3
            gps_timestamps = gps_data['timestamp'][valid_fix]
            
            # Convert to AGL using local position reference
            gps_alt = gps_alt_msl - np.mean(gps_alt_msl) + np.mean(data["altitude_agl"])
            
            # Get time array for altitude plot
            gps_t = (gps_timestamps - gps_timestamps[0]) * 1e-6
            
            # Plot lat/lon trajectory
            ax1.plot(gps_lon, gps_lat, color=colors[i], linewidth=1.5, alpha=0.8, 
                    label=f'{log_file.name}', marker='o', markersize=2)
            
            # Plot altitude over time
            ax2.plot(gps_t, gps_alt, color=colors[i], linewidth=1.5, alpha=0.8, 
                    label=f'{log_file.name}')
            
            # Mark dive start and impact
            first_dive_idx = engagement_masks.get('first_dive_idx')
            first_impact_idx = engagement_masks.get('first_impact_idx')
            
            if first_dive_idx is not None:
                dive_gps_idx = np.searchsorted(gps_timestamps, data["timestamp_us"][first_dive_idx])
                dive_gps_idx = np.clip(dive_gps_idx, 0, len(gps_timestamps) - 1)
                
                # Mark on lat/lon plot
                ax1.scatter(gps_lon[dive_gps_idx], gps_lat[dive_gps_idx], 
                           color=colors[i], s=80, marker='^', edgecolors='orange', linewidth=2, zorder=5)
                
                # Mark on altitude plot
                ax2.scatter(gps_t[dive_gps_idx], gps_alt[dive_gps_idx], 
                           color=colors[i], s=80, marker='^', edgecolors='orange', linewidth=2, zorder=5)
            
            if first_impact_idx is not None:
                impact_gps_idx = np.searchsorted(gps_timestamps, data["timestamp_us"][first_impact_idx])
                impact_gps_idx = np.clip(impact_gps_idx, 0, len(gps_timestamps) - 1)
                
                # Mark on lat/lon plot
                ax1.scatter(gps_lon[impact_gps_idx], gps_lat[impact_gps_idx], 
                           color=colors[i], s=100, marker='X', edgecolors='red', linewidth=2, zorder=5)
                
                # Mark on altitude plot
                ax2.scatter(gps_t[impact_gps_idx], gps_alt[impact_gps_idx], 
                           color=colors[i], s=100, marker='X', edgecolors='red', linewidth=2, zorder=5)
                
        except Exception as e:
            print(f"  Error processing {log_file.name} for GPS plot: {e}")
            continue
    
    # Plot target on lat/lon subplot
    ax1.scatter(target_info['lon'], target_info['lat'], 
               color='green', s=200, marker='s', 
               label=f'{target_selection} Target', zorder=10, edgecolors='black', linewidth=2)
    
    # Configure lat/lon subplot
    ax1.set_xlabel('Longitude [deg]')
    ax1.set_ylabel('Latitude [deg]')
    ax1.set_title(f'GPS Trajectories - Lat/Lon (Target: {target_selection})')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Configure altitude subplot
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Altitude [m AGL]')
    ax2.set_title('GPS Altitude Profiles')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Ground Level')
    
    # Add legend elements for markers
    ax1.scatter([], [], color='gray', s=80, marker='^', edgecolors='orange', linewidth=2, 
               label='Dive Start')
    ax1.scatter([], [], color='gray', s=100, marker='X', edgecolors='red', linewidth=2, 
               label='Impact')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if interactive:
        plt.show()  # Show interactive plot
        print("Showing interactive GPS trajectory plot (close window to continue)")
    else:
        # Save plot with appropriate filename based on number of files
        target_suffix = f"_{target_selection.replace(' ', '_').lower()}"
        prefix = "gps_trajectories" if len(log_files) == 1 else "combined_gps_trajectories"
        gps_plot_path = output_dir / f"{prefix}{target_suffix}.png"
        plt.savefig(gps_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved GPS trajectory plot: {gps_plot_path}")


