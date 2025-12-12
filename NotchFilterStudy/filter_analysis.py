#!/usr/bin/env python3
"""
Internal Filter Analysis Script

This script investigates internal filters present in PX4 ulogs by:
1. Listing all available signals in the ulog
2. Analyzing angular velocity FFT
3. Analyzing engine RPM FFT
4. Plotting frequency domain characteristics

Author: Generated for filter investigation
Date: December 8, 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import scipy.fftpack
from pathlib import Path
from pyulog import ULog

# Add FlightReviewLib to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'FlightReviewLib'))

try:
    import pyfftw
    PYFFTW_AVAILABLE = True
except ImportError:
    PYFFTW_AVAILABLE = False
    print("Warning: pyfftw not available, falling back to scipy.fft")

class FilterAnalyzer:
    """Class to analyze internal filters in PX4 ulogs"""
    
    def __init__(self, ulg_filename):
        """Initialize analyzer with ULG file"""
        self.ulg_filename = ulg_filename
        self.ulog = None
        self.signals_info = {}
        
    def load_ulog(self):
        """Load ULG file"""
        try:
            print(f"Loading ULG file: {self.ulg_filename}")
            self.ulog = ULog(str(self.ulg_filename))
            print(f"Successfully loaded ULG file with {len(self.ulog.data_list)} datasets")
            return True
        except Exception as e:
            print(f"Error loading ULG file: {e}")
            return False
    
    def print_all_signals(self, output_file=None):
        """Print all available signals in the ulog and optionally save to file"""
        if not self.ulog:
            print("ULG file not loaded")
            return False
        
        # Build the output text
        output_lines = []
        output_lines.append("="*80)
        output_lines.append("AVAILABLE SIGNALS IN ULOG")
        output_lines.append("="*80)
        output_lines.append("")
        
        for i, dataset in enumerate(self.ulog.data_list):
            output_lines.append(f"\n{i+1}. Dataset: '{dataset.name}' (instance {dataset.multi_id})")
            output_lines.append(f"   Number of samples: {len(dataset.data['timestamp'])}")
            
            # Calculate sample rate
            if len(dataset.data['timestamp']) > 1:
                timestamps_sec = dataset.data['timestamp'] / 1e6
                duration = timestamps_sec[-1] - timestamps_sec[0]
                sample_rate = len(timestamps_sec) / duration if duration > 0 else 0
                output_lines.append(f"   Sample rate: {sample_rate:.2f} Hz")
            
            output_lines.append(f"   Available fields:")
            for field_name in dataset.data.keys():
                field_data = dataset.data[field_name]
                if hasattr(field_data, 'dtype'):
                    if np.issubdtype(field_data.dtype, np.number) and len(field_data) > 0:
                        if field_name != 'timestamp':
                            output_lines.append(f"      - {field_name}: [{np.min(field_data):.4f}, {np.max(field_data):.4f}]")
                        else:
                            output_lines.append(f"      - {field_name}: [timestamp]")
                    else:
                        output_lines.append(f"      - {field_name}: [{field_data.dtype}]")
        
        output_lines.append("")
        output_lines.append("="*80)
        output_lines.append("")
        
        # Print to console
        output_text = "\n".join(output_lines)
        print(output_text)
        
        # Save to file if specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(output_text)
            print(f"\nSignal list saved to: {output_path}")
        
        return True
    
    def extract_angular_velocity(self):
        """Extract angular velocity data"""
        try:
            # Try vehicle_angular_velocity first (filtered)
            dataset = self.ulog.get_dataset('vehicle_angular_velocity')
            timestamp = dataset.data['timestamp'] / 1e6
            
            angular_vel = {
                'timestamp': timestamp,
                'x': dataset.data['xyz[0]'],
                'y': dataset.data['xyz[1]'],
                'z': dataset.data['xyz[2]']
            }
            
            # Calculate sample rate
            duration = timestamp[-1] - timestamp[0]
            sample_rate = len(timestamp) / duration if duration > 0 else 0
            
            print(f"Extracted vehicle_angular_velocity: {len(timestamp)} samples at {sample_rate:.2f} Hz")
            return angular_vel, sample_rate
            
        except Exception as e:
            print(f"Could not extract vehicle_angular_velocity: {e}")
            
            # Fallback to sensor_combined gyro data
            try:
                dataset = self.ulog.get_dataset('sensor_combined')
                timestamp = dataset.data['timestamp'] / 1e6
                
                angular_vel = {
                    'timestamp': timestamp,
                    'x': dataset.data['gyro_rad[0]'],
                    'y': dataset.data['gyro_rad[1]'],
                    'z': dataset.data['gyro_rad[2]']
                }
                
                duration = timestamp[-1] - timestamp[0]
                sample_rate = len(timestamp) / duration if duration > 0 else 0
                
                print(f"Extracted gyro from sensor_combined: {len(timestamp)} samples at {sample_rate:.2f} Hz")
                return angular_vel, sample_rate
                
            except Exception as e2:
                print(f"Could not extract gyro from sensor_combined: {e2}")
                return None, None
    
    def extract_engine_data(self):
        """Extract engine RPM and throttle position data"""
        try:
            # Try internal_combustion_engine_status (field 67)
            dataset = self.ulog.get_dataset('internal_combustion_engine_status')
            timestamp = dataset.data['timestamp'] / 1e6
            
            # Extract available fields
            engine_data = {
                'timestamp': timestamp
            }
            
            # Check for RPM
            if 'engine_speed_rpm' in dataset.data:
                engine_data['rpm'] = dataset.data['engine_speed_rpm']
            elif 'rpm' in dataset.data:
                engine_data['rpm'] = dataset.data['rpm']
            elif 'engine_rpm' in dataset.data:
                engine_data['rpm'] = dataset.data['engine_rpm']
            
            # Check for throttle position
            if 'throttle_position_percent' in dataset.data:
                engine_data['throttle'] = dataset.data['throttle_position_percent']
            elif 'throttle_position' in dataset.data:
                engine_data['throttle'] = dataset.data['throttle_position']
            elif 'throttle' in dataset.data:
                engine_data['throttle'] = dataset.data['throttle']
            
            # Calculate sample rate
            duration = timestamp[-1] - timestamp[0]
            sample_rate = len(timestamp) / duration if duration > 0 else 0
            
            print(f"Extracted engine data from internal_combustion_engine_status: {len(timestamp)} samples at {sample_rate:.2f} Hz")
            print(f"  Available fields: {[k for k in engine_data.keys() if k != 'timestamp']}")
            
            if len(engine_data) > 1:  # Has more than just timestamp
                return engine_data, sample_rate
            else:
                print(f"  No RPM or throttle data found. Available fields: {list(dataset.data.keys())}")
                return None, None
            
        except Exception as e:
            print(f"Could not extract engine data from internal_combustion_engine_status: {e}")
            
            # Try icu_status as fallback
            try:
                dataset = self.ulog.get_dataset('icu_status')
                timestamp = dataset.data['timestamp'] / 1e6
                
                engine_data = {
                    'timestamp': timestamp
                }
                
                # Check for RPM
                if 'rpm' in dataset.data:
                    engine_data['rpm'] = dataset.data['rpm']
                elif 'engine_rpm' in dataset.data:
                    engine_data['rpm'] = dataset.data['engine_rpm']
                
                # Check for throttle
                if 'throttle_position' in dataset.data:
                    engine_data['throttle'] = dataset.data['throttle_position']
                elif 'throttle' in dataset.data:
                    engine_data['throttle'] = dataset.data['throttle']
                
                duration = timestamp[-1] - timestamp[0]
                sample_rate = len(timestamp) / duration if duration > 0 else 0
                
                print(f"Extracted engine data from icu_status: {len(timestamp)} samples at {sample_rate:.2f} Hz")
                print(f"  Available fields: {[k for k in engine_data.keys() if k != 'timestamp']}")
                
                if len(engine_data) > 1:
                    return engine_data, sample_rate
                else:
                    return None, None
                
            except Exception as e2:
                print(f"Could not extract engine data from icu_status: {e2}")
                return None, None
    
    def extract_gyro_fft_data(self):
        """Extract sensor_gyro_fft data (onboard FFT analysis)"""
        try:
            dataset = self.ulog.get_dataset('sensor_gyro_fft')
            timestamp = dataset.data['timestamp'] / 1e6
            
            fft_data = {
                'timestamp': timestamp,
                'peak_freq_x': [dataset.data[f'peak_frequencies_x[{i}]'] for i in range(3)],
                'peak_freq_y': [dataset.data[f'peak_frequencies_y[{i}]'] for i in range(3)],
                'peak_freq_z': [dataset.data[f'peak_frequencies_z[{i}]'] for i in range(3)],
                'peak_snr_x': [dataset.data[f'peak_snr_x[{i}]'] for i in range(3)],
                'peak_snr_y': [dataset.data[f'peak_snr_y[{i}]'] for i in range(3)],
                'peak_snr_z': [dataset.data[f'peak_snr_z[{i}]'] for i in range(3)],
            }
            
            # Calculate sample rate
            duration = timestamp[-1] - timestamp[0]
            sample_rate = len(timestamp) / duration if duration > 0 else 0
            
            print(f"Extracted sensor_gyro_fft: {len(timestamp)} samples at {sample_rate:.2f} Hz")
            return fft_data, sample_rate
            
        except Exception as e:
            print(f"Could not extract sensor_gyro_fft: {e}")
            return None, None
    
    def extract_raw_sensor_gyro(self, instance=0):
        """Extract raw sensor_gyro data
        
        Args:
            instance: Sensor instance to extract (0, 1, or 2 for multi-IMU setups)
        
        Returns:
            Tuple of (gyro_data dict, actual_sample_rate)
            Note: The sample rate returned is the ACTUAL sensor rate from sensor_gyro_fft,
                  not the decimated logging rate
        """
        try:
            # Get all sensor_gyro datasets
            datasets = [d for d in self.ulog.data_list if d.name == 'sensor_gyro']
            
            if not datasets:
                print("No sensor_gyro datasets found")
                return None, None
            
            print(f"Found {len(datasets)} sensor_gyro instances")
            
            # Select the requested instance
            if instance >= len(datasets):
                print(f"Warning: instance {instance} not found, using instance 0")
                instance = 0
            
            dataset = datasets[instance]
            timestamp = dataset.data['timestamp'] / 1e6
            
            gyro_data = {
                'timestamp': timestamp,
                'x': dataset.data['x'],
                'y': dataset.data['y'],
                'z': dataset.data['z']
            }
            
            # Calculate LOGGED sample rate (this is decimated for log storage)
            duration = timestamp[-1] - timestamp[0]
            logged_sample_rate = len(timestamp) / duration if duration > 0 else 0
            
            # Get ACTUAL sensor sample rate from sensor_gyro_fft if available
            try:
                fft_dataset = self.ulog.get_dataset('sensor_gyro_fft')
                if 'sensor_sample_rate_hz' in fft_dataset.data:
                    actual_sample_rate = np.median(fft_dataset.data['sensor_sample_rate_hz'])
                    print(f"Extracted sensor_gyro[{instance}]: {len(timestamp)} logged samples at {logged_sample_rate:.2f} Hz")
                    print(f"  Actual sensor sample rate: {actual_sample_rate:.2f} Hz (from sensor_gyro_fft)")
                    return gyro_data, actual_sample_rate
            except:
                pass
            
            print(f"Extracted sensor_gyro[{instance}]: {len(timestamp)} samples at {logged_sample_rate:.2f} Hz")
            print(f"  Warning: Using logged sample rate (actual sensor rate may be higher)")
            return gyro_data, logged_sample_rate
            
        except Exception as e:
            print(f"Could not extract sensor_gyro: {e}")
            return None, None
    
    def compute_fft(self, data, sample_rate):
        """Compute FFT of a signal"""
        # Remove DC component
        data_centered = data - np.mean(data)
        
        # Apply Hanning window
        window = np.hanning(len(data_centered))
        data_windowed = data_centered * window
        
        # Compute FFT
        n = len(data_windowed)
        fft_vals = fft(data_windowed)
        fft_freq = fftfreq(n, 1/sample_rate)
        
        # Only keep positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_freq = fft_freq[positive_freq_idx]
        fft_magnitude = np.abs(fft_vals[positive_freq_idx])
        
        # Convert to dB
        fft_db = 20 * np.log10(fft_magnitude + 1e-10)
        
        return fft_freq, fft_db, fft_magnitude
    
    def compute_fft_flight_review(self, data, sample_rate):
        """Compute FFT using Flight Review library implementation
        
        This matches the implementation in FlightReviewLib/plotting.py
        Uses pyfftw if available, otherwise falls back to scipy.fft
        
        Args:
            data: Input signal data
            sample_rate: Sampling frequency in Hz
        
        Returns:
            Tuple of (frequencies, fft_magnitude)
        """
        data_len = len(data)
        delta_t = 1.0 / sample_rate
        
        # Calculate frequencies using scipy.fftpack (as in Flight Review)
        freqs = scipy.fftpack.fftfreq(data_len, delta_t)
        
        if PYFFTW_AVAILABLE:
            # Use pyfftw (Flight Review implementation)
            pyfftw.interfaces.cache.enable()
            fft_values = 2/data_len * np.abs(
                pyfftw.interfaces.numpy_fft.fft(data, planner_effort='FFTW_ESTIMATE')
            )
        else:
            # Fallback to scipy.fft
            fft_values = 2/data_len * np.abs(scipy.fft.fft(data))
        
        # Take only positive frequencies (as in Flight Review)
        fft_plot_values = fft_values[:len(freqs)//2]
        freqs_plot = freqs[:len(freqs)//2]
        
        return freqs_plot, fft_plot_values
    
    def plot_angular_velocity_and_engine_rpm_fft(self, output_dir=None, dnf_bandwidth=10.0, dnf_min=0.0, dnf_max=1000.0, dnf_harmonics=0, dnf_snr_min=1.0):
        """Plot FFT of angular velocities as subplots and engine RPM/throttle overlaid on same plot
        
        Args:
            output_dir: Output directory for plots
            dnf_bandwidth: Dynamic notch filter bandwidth in Hz
            dnf_min: Minimum frequency to consider for peak detection (Hz)
            dnf_max: Maximum frequency to consider for peak detection (Hz)
            dnf_harmonics: Number of harmonics to display (0 = fundamental only)
            dnf_snr_min: Minimum SNR in dB for peak detection (default: 1.0)
        """
        
        # Extract data
        print("\nExtracting angular velocity data...")
        angular_vel, gyro_sample_rate = self.extract_angular_velocity()
        
        print("\nExtracting engine data...")
        engine_data, engine_sample_rate = self.extract_engine_data()
        
        if angular_vel is None and engine_data is None:
            print("Error: Could not extract any data for analysis")
            return False
        
        # Prepare output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(".")
        
        saved_files = []
        
        # Plot angular velocity FFT as subplots
        if angular_vel is not None:
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            for idx, (axis, axis_name, color) in enumerate(zip(['x', 'y', 'z'], 
                                                                 ['Roll Rate', 'Pitch Rate', 'Yaw Rate'],
                                                                 ['blue', 'green', 'red'])):
                data = angular_vel[axis]
                
                # Compute FFT
                freq, fft_db, fft_magnitude = self.compute_fft(data, gyro_sample_rate)
                
                # Plot FFT
                axes[idx].plot(freq, fft_magnitude, color=color, linewidth=0.8)
                axes[idx].set_ylabel('Magnitude', fontsize=11)
                axes[idx].set_title(f'Angular Velocity FFT - {axis_name}', fontsize=12, fontweight='bold')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].set_xlim([0, gyro_sample_rate / 2])
                
                # Set x-axis ticks every 5 Hz
                from matplotlib.ticker import MultipleLocator
                axes[idx].xaxis.set_major_locator(MultipleLocator(5))
                axes[idx].xaxis.set_minor_locator(MultipleLocator(1))
                
                # Add annotation with stats
                textstr = f'Sample Rate: {gyro_sample_rate:.1f} Hz\n'
                textstr += f'FFT Bins: {len(freq)}\n'
                textstr += f'Nyquist: {gyro_sample_rate/2:.1f} Hz'
                axes[idx].text(0.98, 0.98, textstr, transform=axes[idx].transAxes,
                              verticalalignment='top', horizontalalignment='right',
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                              fontsize=9)
            
            axes[2].set_xlabel('Frequency (Hz)', fontsize=11)
            
            plt.tight_layout()
            save_file = output_path / "angular_velocity_fft.png"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(save_file)
            print(f"  Saved: {save_file}")
        
        # Plot engine RPM and throttle overlaid on same plot
        if engine_data is not None and ('rpm' in engine_data or 'throttle' in engine_data):
            fig, ax1 = plt.subplots(figsize=(14, 6))
            
            time = engine_data['timestamp']
            
            # Plot RPM on left y-axis
            if 'rpm' in engine_data:
                rpm_data = engine_data['rpm']
                color = 'tab:blue'
                ax1.set_xlabel('Time (s)', fontsize=12)
                ax1.set_ylabel('Engine Speed (RPM)', fontsize=12, color=color)
                ax1.plot(time, rpm_data, color=color, linewidth=1.2, label='Engine RPM')
                ax1.tick_params(axis='y', labelcolor=color)
                ax1.grid(True, alpha=0.3)
                
                # Add RPM stats
                valid_rpm = rpm_data[rpm_data > 0]
                if len(valid_rpm) > 0:
                    textstr = f'RPM Stats:\n'
                    textstr += f'Mean: {np.mean(valid_rpm):.0f} RPM\n'
                    textstr += f'Max: {np.max(valid_rpm):.0f} RPM\n'
                    textstr += f'Std: {np.std(valid_rpm):.0f} RPM'
                    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
                            verticalalignment='top', horizontalalignment='left',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                            fontsize=10)
            
            # Plot throttle on right y-axis
            if 'throttle' in engine_data:
                throttle_data = engine_data['throttle']
                ax2 = ax1.twinx()
                color = 'tab:orange'
                ax2.set_ylabel('Throttle Position (%)', fontsize=12, color=color)
                ax2.plot(time, throttle_data, color=color, linewidth=1.2, label='Throttle Position')
                ax2.tick_params(axis='y', labelcolor=color)
                ax2.set_ylim([0, 105])
                
                # Add throttle stats
                valid_throttle = throttle_data[np.isfinite(throttle_data)]
                if len(valid_throttle) > 0:
                    textstr = f'Throttle Stats:\n'
                    textstr += f'Mean: {np.mean(valid_throttle):.1f}%\n'
                    textstr += f'Max: {np.max(valid_throttle):.1f}%\n'
                    textstr += f'Std: {np.std(valid_throttle):.1f}%'
                    ax2.text(0.98, 0.98, textstr, transform=ax2.transAxes,
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                            fontsize=10)
            
            ax1.set_title('Engine RPM and Throttle Position', fontsize=14, fontweight='bold')
            
            # Add legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            if 'throttle' in engine_data:
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
            else:
                ax1.legend(lines1, labels1, loc='upper left', fontsize=10)
            
            plt.tight_layout()
            save_file = output_path / "engine_rpm_throttle.png"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(save_file)
            print(f"  Saved: {save_file}")
            
            # Plot engine frequency (RPM/60) and throttle overlaid with FFT subplots
            fig = plt.figure(figsize=(14, 16))
            gs = fig.add_gridspec(1, 1, top=0.95, bottom=0.70, hspace=0.3)
            
            # Convert RPM to Hz (frequency)
            rpm_hz = rpm_data / 60.0
            
            # Subplot 1: Engine frequency and throttle overlaid
            ax1 = fig.add_subplot(gs[0])
            
            color = 'tab:blue'
            ax1.set_xlabel('Time (s)', fontsize=12)
            ax1.set_ylabel('Engine Frequency (Hz)', fontsize=12, color=color)
            ax1.plot(time, rpm_hz, color=color, linewidth=1.2, label='Engine Frequency')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.grid(True, alpha=0.3)
            
            # Add frequency stats
            valid_hz = rpm_hz[rpm_data > 0]
            if len(valid_hz) > 0:
                textstr = f'Frequency Stats:\n'
                textstr += f'Mean: {np.mean(valid_hz):.1f} Hz\n'
                textstr += f'Max: {np.max(valid_hz):.1f} Hz\n'
                textstr += f'Std: {np.std(valid_hz):.1f} Hz'
                ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                        fontsize=10)
            
            # Plot throttle on right y-axis
            if 'throttle' in engine_data:
                ax1_twin = ax1.twinx()
                color = 'tab:orange'
                ax1_twin.set_ylabel('Throttle Position (%)', fontsize=12, color=color)
                ax1_twin.plot(time, throttle_data, color=color, linewidth=1.2, label='Throttle Position')
                ax1_twin.tick_params(axis='y', labelcolor=color)
                ax1_twin.set_ylim([0, 105])
                
                # Add throttle stats
                valid_throttle = throttle_data[np.isfinite(throttle_data)]
                if len(valid_throttle) > 0:
                    textstr = f'Throttle Stats:\n'
                    textstr += f'Mean: {np.mean(valid_throttle):.1f}%\n'
                    textstr += f'Max: {np.max(valid_throttle):.1f}%\n'
                    textstr += f'Std: {np.std(valid_throttle):.1f}%'
                    ax1_twin.text(0.98, 0.98, textstr, transform=ax1_twin.transAxes,
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                            fontsize=10)
            
            ax1.set_title('Engine Frequency (Hz) and Throttle Position', fontsize=14, fontweight='bold')
            
            # Add legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            if 'throttle' in engine_data:
                lines2, labels2 = ax1_twin.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
            else:
                ax1.legend(lines1, labels1, loc='upper left', fontsize=10)
            
            # Subplots 2-4: FFT of Roll, Pitch, Yaw Rates with notch filter bandwidth
            if angular_vel is not None:
                gs_angular = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], 
                                              top=0.65, bottom=0.05, hspace=0.3)
                
                for idx, (axis, axis_name, color) in enumerate(zip(['x', 'y', 'z'], 
                                                                     ['Roll Rate', 'Pitch Rate', 'Yaw Rate'],
                                                                     ['blue', 'green', 'red'])):
                    ax = fig.add_subplot(gs_angular[idx])
                    data = angular_vel[axis]
                    freq, fft_db, fft_mag = self.compute_fft(data, gyro_sample_rate)
                    
                    # Plot FFT
                    ax.plot(freq, fft_mag, color=color, linewidth=0.8)
                    ax.set_ylabel('Magnitude', fontsize=11)
                    ax.set_title(f'FFT of {axis_name}', fontsize=12, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.set_xlim([0, gyro_sample_rate / 2])
                    
                    # Set x-axis ticks every 5 Hz
                    from matplotlib.ticker import MultipleLocator
                    ax.xaxis.set_major_locator(MultipleLocator(5))
                    ax.xaxis.set_minor_locator(MultipleLocator(1))
                    
                    # Find top 3 peaks within specified range (mimicking PX4 DNF algorithm)
                    freq_mask = (freq >= dnf_min) & (freq <= dnf_max)
                    freq_range = freq[freq_mask]
                    fft_mag_range = fft_mag[freq_mask]
                    
                    # Compute SNR for all bins (PX4-style)
                    bin_mag_sum = np.sum(fft_mag)
                    MIN_SNR = dnf_snr_min  # dB, minimum SNR threshold
                    
                    # Find up to 3 peaks
                    MAX_NUM_PEAKS = 3
                    peak_colors = ['red', 'orange', 'purple']
                    peaks_found = []
                    
                    if len(freq_range) > 0:
                        fft_mag_work = fft_mag.copy()
                        
                        for peak_idx in range(MAX_NUM_PEAKS):
                            # Find largest peak in range
                            peak_mask = (freq >= dnf_min) & (freq <= dnf_max)
                            masked_mag = np.where(peak_mask, fft_mag_work, 0)
                            
                            if np.max(masked_mag) == 0:
                                break
                            
                            peak_bin_idx = np.argmax(masked_mag)
                            peak_freq = freq[peak_bin_idx]
                            peak_mag = fft_mag_work[peak_bin_idx]
                            
                            # Compute SNR (PX4 formula)
                            snr_db = 10.0 * np.log10((len(fft_mag) - 1) * peak_mag / (bin_mag_sum - peak_mag + 1e-10))
                            
                            # Debug output
                            if idx == 1:  # Pitch axis
                                print(f"  {axis_name} - Peak {peak_idx+1}: {peak_freq:.1f} Hz, Mag={peak_mag:.2e}, SNR={snr_db:.1f} dB")
                            
                            # Only accept if SNR > MIN_SNR
                            if snr_db > MIN_SNR:
                                peaks_found.append({
                                    'freq': peak_freq,
                                    'mag': peak_mag,
                                    'snr': snr_db,
                                    'color': peak_colors[peak_idx]
                                })
                            
                            # Zero out peak ± 1 bin to avoid selecting same peak
                            if peak_bin_idx > 0:
                                fft_mag_work[peak_bin_idx - 1] = 0
                            fft_mag_work[peak_bin_idx] = 0
                            if peak_bin_idx < len(fft_mag_work) - 1:
                                fft_mag_work[peak_bin_idx + 1] = 0
                        
                        # Plot SNR threshold line
                        if len(peaks_found) > 0:
                            # Calculate SNR curve for visualization
                            snr_curve = 10.0 * np.log10((len(fft_mag) - 1) * fft_mag / (bin_mag_sum - fft_mag + 1e-10))
                            ax_snr = ax.twinx()
                            ax_snr.plot(freq, snr_curve, 'gray', linewidth=0.5, alpha=0.3, label='SNR (dB)')
                            ax_snr.axhline(MIN_SNR, color='gray', linestyle=':', linewidth=1, alpha=0.5, label=f'Min SNR ({MIN_SNR} dB)')
                            ax_snr.set_ylabel('SNR (dB)', fontsize=9, color='gray')
                            ax_snr.tick_params(axis='y', labelcolor='gray', labelsize=8)
                            ax_snr.set_ylim([0, max(20, np.max(snr_curve[freq_mask]) * 1.1)])
                        
                        # Draw notch filters for each peak and its harmonics
                        for peak_info in peaks_found:
                            peak_freq = peak_info['freq']
                            color = peak_info['color']
                            snr = peak_info['snr']
                            
                            for harmonic in range(dnf_harmonics + 1):
                                if harmonic == 0:
                                    harmonic_freq = peak_freq
                                    label = f"Peak {peaks_found.index(peak_info)+1} ({peak_freq:.1f}Hz, SNR={snr:.1f}dB)"
                                else:
                                    harmonic_freq = peak_freq * (harmonic + 1)
                                    label = None
                                
                                # Skip if harmonic exceeds Nyquist
                                if harmonic_freq > gyro_sample_rate / 2:
                                    break
                                
                                notch_lower = harmonic_freq - dnf_bandwidth / 2
                                notch_upper = harmonic_freq + dnf_bandwidth / 2
                                
                                ax.axvline(notch_lower, color=color, linestyle='--', linewidth=1.5, alpha=0.7, label=label)
                                ax.axvline(notch_upper, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
                                ax.axvline(harmonic_freq, color=color, linestyle=':', linewidth=1, alpha=0.5)
                        
                        # Add annotation with stats
                        textstr = f'Notch BW: {dnf_bandwidth:.1f} Hz\n'
                        textstr += f'Range: [{dnf_min:.0f}-{dnf_max:.0f}] Hz\n'
                        textstr += f'Peaks found: {len(peaks_found)}\n'
                        if dnf_harmonics > 0:
                            textstr += f'Harmonics: {dnf_harmonics}'
                    else:
                        textstr = f'No peak in range\n[{dnf_min:.0f}-{dnf_max:.0f}] Hz'
                    ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                            verticalalignment='top', horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                            fontsize=9)
                    
                    if idx == 0:
                        ax.legend(loc='upper left', fontsize=9)
                    if idx == 2:
                        ax.set_xlabel('Frequency (Hz)', fontsize=11)
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plt.tight_layout()
            save_file = output_path / "engine_frequency_throttle.png"
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(save_file)
            print(f"  Saved: {save_file}")
        
        print(f"\nTotal plots saved: {len(saved_files)}")
        return True
    
    def plot_gyro_fft_peaks_over_time(self, output_dir=None):
        """Plot sensor_gyro_fft peak frequencies and SNRs over time
        
        Creates a 3x3 grid with each peak on a separate subplot
        - Rows: Roll, Pitch, Yaw
        - Columns: Peak 1, Peak 2, Peak 3
        - Each subplot shows frequency (solid) and SNR (dashed) with dual y-axes
        - Exports all peak data to text file
        """
        
        # Extract FFT data
        print("\nExtracting sensor_gyro_fft data...")
        fft_data, sample_rate = self.extract_gyro_fft_data()
        
        if fft_data is None:
            print("Error: Could not extract sensor_gyro_fft data")
            return False
        
        # Prepare output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(".")
        
        # Export peak data to text file
        data_file = output_path / "gyro_fft_peak_data.txt"
        with open(data_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("GYRO FFT PEAK FREQUENCIES AND SNR DATA\n")
            f.write("=" * 100 + "\n\n")
            
            timestamp = fft_data['timestamp']
            axis_names = ['Roll Rate (X)', 'Pitch Rate (Y)', 'Yaw Rate (Z)']
            axis_keys = ['x', 'y', 'z']
            
            for axis_name, axis_key in zip(axis_names, axis_keys):
                f.write(f"\n{axis_name}\n")
                f.write("-" * 100 + "\n")
                f.write(f"{'Time (s)':<12} {'Peak1 Hz':<12} {'Peak1 SNR':<12} {'Peak2 Hz':<12} {'Peak2 SNR':<12} {'Peak3 Hz':<12} {'Peak3 SNR':<12}\n")
                f.write("-" * 100 + "\n")
                
                for i, t in enumerate(timestamp):
                    peak1_freq = fft_data[f'peak_freq_{axis_key}'][0][i]
                    peak1_snr = fft_data[f'peak_snr_{axis_key}'][0][i]
                    peak2_freq = fft_data[f'peak_freq_{axis_key}'][1][i]
                    peak2_snr = fft_data[f'peak_snr_{axis_key}'][1][i]
                    peak3_freq = fft_data[f'peak_freq_{axis_key}'][2][i]
                    peak3_snr = fft_data[f'peak_snr_{axis_key}'][2][i]
                    
                    f.write(f"{t:<12.2f} {peak1_freq:<12.2f} {peak1_snr:<12.2f} {peak2_freq:<12.2f} {peak2_snr:<12.2f} {peak3_freq:<12.2f} {peak3_snr:<12.2f}\n")
                
                f.write("\n")
        
        print(f"  Peak data exported to: {data_file}")
        
        # Create figure with 3x3 subplots (non-interactive)
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        
        # Define colors for each axis with transparency gradient
        # Darker shade for peak 1, lighter shades for peaks 2 and 3
        axis_colors = {
            'x': ['#FF0000', '#FF6666', '#FFAAAA'],  # Red shades for Roll
            'y': ['#00AA00', '#66CC66', '#AADDAA'],  # Green shades for Pitch
            'z': ['#0000FF', '#6666FF', '#AAAAFF']   # Blue shades for Yaw
        }
        
        timestamp = fft_data['timestamp']
        axis_names = ['Roll Rate (X)', 'Pitch Rate (Y)', 'Yaw Rate (Z)']
        axis_keys = ['x', 'y', 'z']
        
        for row_idx, (axis_name, axis_key) in enumerate(zip(axis_names, axis_keys)):
            for col_idx in range(3):
                ax = axes[row_idx, col_idx]
                
                # Create twin axis for SNR
                ax_snr = ax.twinx()
                
                # Get color for this axis and peak
                color = axis_colors[axis_key][col_idx]
                
                # Get data for this peak
                peak_freq = fft_data[f'peak_freq_{axis_key}'][col_idx]
                peak_snr = fft_data[f'peak_snr_{axis_key}'][col_idx]
                
                # Filter out NaN values
                valid_snr = ~np.isnan(peak_snr)
                valid_freq = ~np.isnan(peak_freq)
                
                # Additional filter: only show frequency when SNR > 12
                valid_freq_snr_filtered = valid_freq & (peak_snr > 12)
                
                # Plot SNR first (background, light grey, dashed line, right y-axis)
                if np.any(valid_snr):
                    ax_snr.plot(timestamp[valid_snr], peak_snr[valid_snr], 
                               color='lightgrey', linewidth=2, linestyle='--', 
                               label='SNR', alpha=0.6, zorder=1)
                
                # Plot frequency (foreground, solid line, left y-axis, only when SNR > 12)
                if np.any(valid_freq_snr_filtered):
                    ax.plot(timestamp[valid_freq_snr_filtered], peak_freq[valid_freq_snr_filtered], 
                           color=color, linewidth=2, linestyle='-', 
                           label='Frequency', alpha=0.9, zorder=2)
                
                # Configure left y-axis (frequency)
                ax.set_ylabel('Frequency (Hz)', fontsize=10, fontweight='bold')
                if row_idx == 2:  # Bottom row
                    ax.set_xlabel('Time (s)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=9)
                
                # Configure right y-axis (SNR)
                ax_snr.set_ylabel('SNR (dB)', fontsize=10, fontweight='bold', color='gray')
                ax_snr.tick_params(axis='y', labelcolor='gray', labelsize=9)
                
                # Title for each subplot
                if row_idx == 0:  # Top row
                    ax.set_title(f'Peak {col_idx+1}', fontsize=11, fontweight='bold')
                
                # Add axis label on the left
                if col_idx == 0:  # Left column
                    ax.text(-0.35, 0.5, axis_name, transform=ax.transAxes,
                           fontsize=12, fontweight='bold', rotation=90,
                           verticalalignment='center', horizontalalignment='center')
                
                # Add legend only on top-right subplot
                if row_idx == 0 and col_idx == 2:
                    lines1, labels1 = ax.get_legend_handles_labels()
                    lines2, labels2 = ax_snr.get_legend_handles_labels()
                    ax.legend(lines1 + lines2, labels1 + labels2, 
                             loc='upper right', fontsize=9)
        
        plt.suptitle('Gyro FFT Peaks and SNR Over Time', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save static version
        save_file = output_path / "gyro_fft_peaks_over_time.png"
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_file}")
        
        return True
    
    def plot_dynamic_notch_evolution(self, output_dir=None, filter_params=None, num_snapshots=20):
        """Plot evolution of dynamic notch filters over time
        
        Creates a 3x3 tiled plot showing how DNF frequencies change over time
        - Rows: Roll (X), Pitch (Y), Yaw (Z)  
        - Columns: DNF Peak 1, 2, 3
        - Each subplot shows filter responses at different time snapshots with color shading
        
        Args:
            output_dir: Output directory for plots
            filter_params: Dictionary with PX4 filter parameters
            num_snapshots: Number of time snapshots to plot (default: 20)
        """
        
        # Default filter parameters
        if filter_params is None:
            filter_params = {
                'IMU_GYRO_DNF_BW': 5.0,
                'IMU_GYRO_DNF_MIN': 10.0,
            }
        
        # Extract FFT data
        print("\nExtracting dynamic notch peak data for evolution plot...")
        fft_data, sample_rate = self.extract_gyro_fft_data()
        
        if fft_data is None:
            print("Error: Could not extract sensor_gyro_fft data")
            return False
        
        # Prepare output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(".")
        
        # Create figure with 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        
        axis_names = ['Roll Rate (X)', 'Pitch Rate (Y)', 'Yaw Rate (Z)']
        axis_keys = ['x', 'y', 'z']
        
        timestamp = fft_data['timestamp']
        total_time = timestamp[-1] - timestamp[0]
        
        # Select evenly spaced snapshots
        num_samples = len(timestamp)
        snapshot_indices = np.linspace(0, num_samples - 1, num_snapshots, dtype=int)
        
        # Create colormap for time evolution
        cmap = plt.cm.viridis
        colors = [cmap(i / (num_snapshots - 1)) for i in range(num_snapshots)]
        
        # Frequency range for plotting filter responses
        freq_range = np.linspace(0, 200, 1000)  # 0 to 200 Hz
        dnf_bw = filter_params['IMU_GYRO_DNF_BW']
        
        for row_idx, (axis_name, axis_key) in enumerate(zip(axis_names, axis_keys)):
            for col_idx in range(3):  # 3 peaks
                ax = axes[row_idx, col_idx]
                
                # Get peak data
                peak_freq = fft_data[f'peak_freq_{axis_key}'][col_idx]
                peak_snr = fft_data[f'peak_snr_{axis_key}'][col_idx]
                
                # Track which frequencies were plotted for legend
                plotted_freqs = []
                
                # Plot filter responses at each snapshot
                for snap_idx, sample_idx in enumerate(snapshot_indices):
                    freq = peak_freq[sample_idx]
                    snr = peak_snr[sample_idx]
                    
                    # Only plot if SNR > 12 and frequency is valid
                    if snr > 12 and not np.isnan(freq) and freq > 0:
                        # Compute notch filter response
                        notch_depth = 1.0 / (1 + (freq / dnf_bw)**2)
                        response = 1 - (1 - notch_depth) * np.exp(-((freq_range - freq) / dnf_bw)**2)
                        
                        # Plot with time-based color
                        time_rel = (timestamp[sample_idx] - timestamp[0]) / total_time
                        ax.plot(freq_range, response, color=colors[snap_idx], 
                               linewidth=1.5, alpha=0.6, zorder=snap_idx)
                        plotted_freqs.append(freq)
                
                # Configure subplot
                ax.set_ylabel('Filter Response', fontsize=10)
                ax.set_ylim([0, 1.05])
                ax.set_xlim([0, 200])
                ax.grid(True, alpha=0.3)
                
                # Add tick marks
                from matplotlib.ticker import MultipleLocator
                ax.xaxis.set_major_locator(MultipleLocator(50))
                ax.xaxis.set_minor_locator(MultipleLocator(10))
                ax.tick_params(labelsize=9)
                
                # Title with axis and peak number
                if len(plotted_freqs) > 0:
                    freq_range_str = f"{np.min(plotted_freqs):.0f}-{np.max(plotted_freqs):.0f} Hz"
                    ax.set_title(f'{axis_name.split()[0]} - Peak {col_idx+1}\n({freq_range_str})', 
                               fontsize=11, fontweight='bold')
                else:
                    ax.set_title(f'{axis_name.split()[0]} - Peak {col_idx+1}\n(No valid data)', 
                               fontsize=11, fontweight='bold')
                
                # Add xlabel only on bottom row
                if row_idx == 2:
                    ax.set_xlabel('Frequency (Hz)', fontsize=10)
                
                # Add stats annotation
                valid_mask = peak_snr > 12
                if np.any(valid_mask):
                    valid_freqs = peak_freq[valid_mask]
                    textstr = f'Samples: {np.sum(valid_mask)}\n'
                    textstr += f'Mean: {np.mean(valid_freqs):.1f} Hz\n'
                    textstr += f'Std: {np.std(valid_freqs):.1f} Hz'
                    ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                           fontsize=8)
        
        # Add colorbar to show time progression
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=total_time))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, orientation='vertical', 
                           pad=0.02, aspect=30, shrink=0.8)
        cbar.set_label('Time (s)', fontsize=11)
        
        plt.suptitle(f'Dynamic Notch Filter Evolution Over Time ({num_snapshots} snapshots, SNR > 12 dB)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        save_file = output_path / "dynamic_notch_evolution.png"
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_file}")
        
        return True
    
    def compute_filter_responses(self, frequencies, sample_rate, filter_params):
        """Compute frequency responses for PX4 gyro filters
        
        Args:
            frequencies: Array of frequencies to evaluate (Hz)
            sample_rate: Sensor sample rate (Hz)
            filter_params: Dictionary with filter parameters
        
        Returns:
            Tuple of (mag_responses, phase_responses) dictionaries
        """
        mag_responses = {}
        phase_responses = {}
        
        # 2nd-order Butterworth low-pass filter (Direct Form II)
        # Transfer function (continuous): H(s) = ω_c² / (s² + √2 × ω_c × s + ω_c²)
        # Discretized using bilinear transform
        cutoff = filter_params.get('IMU_GYRO_CUTOFF', 30.0)
        T = 1.0 / sample_rate  # Sampling period
        wc = 2 * np.pi * cutoff  # Cutoff angular frequency
        
        # Bilinear transform: s = (2/T) × (1 - z⁻¹) / (1 + z⁻¹)
        # Pre-warp the cutoff frequency
        wc_digital = 2 / T * np.tan(wc * T / 2)
        
        # Compute analog prototype coefficients (Butterworth)
        # H(s) = 1 / (s² + √2 × s + 1) normalized to ω_c = 1
        # Scale to actual cutoff: H(s) = ω_c² / (s² + √2 × ω_c × s + ω_c²)
        
        # Apply bilinear transform to get digital filter coefficients
        k = wc_digital * T / 2
        k2 = k * k
        sqrt2 = np.sqrt(2)
        
        # Denominator coefficients
        a0 = 1 + sqrt2 * k + k2
        a1 = 2 * (k2 - 1) / a0
        a2 = (1 - sqrt2 * k + k2) / a0
        
        # Numerator coefficients
        b0 = k2 / a0
        b1 = 2 * k2 / a0
        b2 = k2 / a0
        
        # Compute frequency response: H(e^jω) = B(e^jω) / A(e^jω)
        # where ω = 2πf/f_sample
        omega = 2 * np.pi * frequencies / sample_rate
        z = np.exp(1j * omega)
        
        # Evaluate transfer function
        numerator = b0 + b1 / z + b2 / (z**2)
        denominator = 1 + a1 / z + a2 / (z**2)
        H = numerator / denominator
        mag_responses['lowpass'] = np.abs(H)
        phase_responses['lowpass'] = np.angle(H, deg=True)
        
        # Static notch filter 0
        nf0_freq = filter_params.get('IMU_GYRO_NF0_FRQ', 0)
        nf0_bw = filter_params.get('IMU_GYRO_NF0_BW', 0)
        if nf0_freq > 0:
            # Notch filter: H(w) = (1 + (w0/bw)^2) / ((w0/w)^2 + (w0/bw)^2 + (w/w0)^2)
            # Approximation: magnitude ≈ 1 - exp(-((f-f0)/bw)^2)
            mag_responses['notch0'] = np.ones_like(frequencies)
            notch_depth = 1.0 / (1 + (nf0_freq / nf0_bw)**2)
            mag_responses['notch0'] = 1 - (1 - notch_depth) * np.exp(-((frequencies - nf0_freq) / nf0_bw)**2)
            # Phase approximation: -180° at notch frequency
            phase_responses['notch0'] = -180 * np.exp(-((frequencies - nf0_freq) / (nf0_bw * 2))**2)
        else:
            mag_responses['notch0'] = np.ones_like(frequencies)
            phase_responses['notch0'] = np.zeros_like(frequencies)
        
        # Static notch filter 1
        nf1_freq = filter_params.get('IMU_GYRO_NF1_FRQ', 0)
        nf1_bw = filter_params.get('IMU_GYRO_NF1_BW', 0)
        if nf1_freq > 0:
            mag_responses['notch1'] = np.ones_like(frequencies)
            notch_depth = 1.0 / (1 + (nf1_freq / nf1_bw)**2)
            mag_responses['notch1'] = 1 - (1 - notch_depth) * np.exp(-((frequencies - nf1_freq) / nf1_bw)**2)
            phase_responses['notch1'] = -180 * np.exp(-((frequencies - nf1_freq) / (nf1_bw * 2))**2)
        else:
            mag_responses['notch1'] = np.ones_like(frequencies)
            phase_responses['notch1'] = np.zeros_like(frequencies)
        
        # Combined filter (product of magnitudes, sum of phases)
        mag_responses['combined'] = mag_responses['lowpass'] * mag_responses['notch0'] * mag_responses['notch1']
        phase_responses['combined'] = phase_responses['lowpass'] + phase_responses['notch0'] + phase_responses['notch1']
        
        return mag_responses, phase_responses
    
    def plot_raw_vs_filtered_fft(self, output_dir=None, sensor_instance=0, filter_params=None):
        """Plot comparison of raw sensor_gyro FFT vs filtered vehicle_angular_velocity FFT
        
        Creates a 4-column plot for each axis:
        - Column 1: Overlay of raw and filtered FFT
        - Column 2: Filtered FFT with combined filter model
        - Column 3: All filter models (magnitude)
        - Column 4: All filter models (phase)
        
        Args:
            output_dir: Output directory for plots
            sensor_instance: Which sensor_gyro instance to use (default: 0)
            filter_params: Dictionary with PX4 filter parameters (if None, uses defaults)
        """
        
        # Default filter parameters from PX4
        if filter_params is None:
            filter_params = {
                'IMU_GYRO_CUTOFF': 25.0,
                'IMU_GYRO_NF0_FRQ': 7.0,
                'IMU_GYRO_NF0_BW': 3.0,
                'IMU_GYRO_NF1_FRQ': 23.5,
                'IMU_GYRO_NF1_BW': 10.0,
                'IMU_GYRO_DNF_EN': 2,
                'IMU_GYRO_DNF_MIN': 10.0,
                'IMU_GYRO_DNF_BW': 5.0,
                'IMU_GYRO_DNF_HMC': 3,
            }
        
        # Extract data
        print("\nExtracting raw sensor_gyro data...")
        raw_gyro, raw_sample_rate = self.extract_raw_sensor_gyro(instance=sensor_instance)
        
        print("\nExtracting filtered vehicle_angular_velocity data...")
        filtered_gyro, filtered_sample_rate = self.extract_angular_velocity()
        
        # Extract dynamic notch peak data
        print("\nExtracting dynamic notch peak data...")
        fft_data, fft_sample_rate = self.extract_gyro_fft_data()
        
        if raw_gyro is None or filtered_gyro is None:
            print("Error: Could not extract both raw and filtered data")
            return False
        
        # Prepare output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = Path(".")
        
        # Create figure with 3 rows x 4 columns
        fig = plt.figure(figsize=(32, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.25)
        
        axis_names = ['Roll Rate (X)', 'Pitch Rate (Y)', 'Yaw Rate (Z)']
        axis_keys = ['x', 'y', 'z']
        colors_raw = ['#FF6B6B', '#4ECDC4', '#95E1D3']  # Red, Teal, Mint
        colors_filtered = ['#0000AA', '#006600', '#AA0000']  # Dark Blue, Dark Green, Dark Red
        
        print("\nComputing FFTs using Flight Review implementation...")
        
        # Get dynamic notch frequencies (all 3 peaks with SNR > 12 dB for each axis)
        dnf_frequencies = {}
        if fft_data is not None:
            for axis_idx, axis_key in enumerate(axis_keys):
                dnf_frequencies[axis_key] = []
                
                # Get all 3 peaks
                for peak_idx in range(3):
                    peak_freq = fft_data[f'peak_freq_{axis_key}'][peak_idx]
                    peak_snr = fft_data[f'peak_snr_{axis_key}'][peak_idx]
                    
                    # Find valid samples where SNR > 12
                    valid_mask = peak_snr > 12
                    if np.any(valid_mask):
                        median_freq = np.median(peak_freq[valid_mask])
                        dnf_frequencies[axis_key].append(median_freq)
                        print(f"  DNF {axis_key.upper()} Peak {peak_idx+1}: {median_freq:.1f} Hz")
                
                if len(dnf_frequencies[axis_key]) == 0:
                    print(f"  DNF {axis_key.upper()}: No peaks > 12 dB")
        
        for row_idx, (axis_name, axis_key, color_raw, color_filtered) in enumerate(
            zip(axis_names, axis_keys, colors_raw, colors_filtered)):
            
            # Extract axis data
            raw_data = raw_gyro[axis_key]
            filtered_data = filtered_gyro[axis_key]
            
            # Compute FFTs using Flight Review method
            print(f"  Computing {axis_name} FFT...")
            raw_freq, raw_fft = self.compute_fft_flight_review(raw_data, raw_sample_rate)
            filtered_freq, filtered_fft = self.compute_fft_flight_review(filtered_data, filtered_sample_rate)
            
            # === Column 1: Raw vs Filtered overlay ===
            ax_overlay = fig.add_subplot(gs[row_idx, 0])
            ax_overlay.plot(raw_freq, raw_fft, color=color_raw, linewidth=1.2, 
                          label=f'Raw sensor_gyro ({raw_sample_rate:.0f} Hz)', alpha=0.7)
            ax_overlay.plot(filtered_freq, filtered_fft, color=color_filtered, linewidth=1.2,
                          label=f'Filtered vehicle_angular_velocity ({filtered_sample_rate:.1f} Hz)', alpha=0.8)
            
            ax_overlay.set_ylabel('FFT Magnitude', fontsize=11)
            ax_overlay.set_title(f'{axis_name} - Raw vs Filtered FFT', fontsize=12, fontweight='bold')
            ax_overlay.grid(True, alpha=0.3)
            ax_overlay.legend(loc='upper right', fontsize=8)
            ax_overlay.set_xlim([0, min(raw_sample_rate, filtered_sample_rate) / 2])
            
            # Add tick marks - major every 100 Hz, minor every 20 Hz
            from matplotlib.ticker import MultipleLocator
            ax_overlay.xaxis.set_major_locator(MultipleLocator(100))
            ax_overlay.xaxis.set_minor_locator(MultipleLocator(20))
            ax_overlay.tick_params(axis='x', which='major', labelsize=9)
            ax_overlay.tick_params(axis='x', which='minor', labelsize=0)
            
            if row_idx == 2:
                ax_overlay.set_xlabel('Frequency (Hz)', fontsize=11)
            
            # Compute filter responses (dense frequency grid for smooth curves)
            filter_freq = np.linspace(0, min(raw_sample_rate, filtered_sample_rate) / 2, 2000)
            mag_responses, phase_responses = self.compute_filter_responses(filter_freq, raw_sample_rate, filter_params)
            
            # Add dynamic notch filter responses for this axis (all 3 peaks)
            dnf_freqs = dnf_frequencies.get(axis_key, [])
            dnf_responses = []
            mag_responses['combined_with_dnf'] = mag_responses['combined'].copy()
            phase_responses['combined_with_dnf'] = phase_responses['combined'].copy()
            
            if len(dnf_freqs) > 0 and filter_params['IMU_GYRO_DNF_EN'] > 0:
                dnf_bw = filter_params['IMU_GYRO_DNF_BW']
                for dnf_idx, dnf_freq in enumerate(dnf_freqs):
                    notch_depth = 1.0 / (1 + (dnf_freq / dnf_bw)**2)
                    dnf_mag = 1 - (1 - notch_depth) * np.exp(-((filter_freq - dnf_freq) / dnf_bw)**2)
                    dnf_phase = -180 * np.exp(-((filter_freq - dnf_freq) / (dnf_bw * 2))**2)
                    mag_responses[f'dnf_{dnf_idx}'] = dnf_mag
                    phase_responses[f'dnf_{dnf_idx}'] = dnf_phase
                    dnf_responses.append(dnf_mag)
                    # Multiply magnitude, add phase
                    mag_responses['combined_with_dnf'] *= dnf_mag
                    phase_responses['combined_with_dnf'] += dnf_phase
            
            # Scale factor should be 1.0 for filter models (they represent magnitude response from 0 to 1)
            # No scaling needed in column 3
            
            # For column 2, scale to match filtered FFT magnitude
            passband_mask = (filtered_freq > 1) & (filtered_freq < filter_params['IMU_GYRO_CUTOFF'] * 0.8)
            if np.any(passband_mask):
                scale_factor = np.mean(filtered_fft[passband_mask])
            else:
                scale_factor = np.max(filtered_fft) * 0.8
            
            # === Column 2: Filtered FFT with combined filter model ===
            ax_middle = fig.add_subplot(gs[row_idx, 1])
            
            # Plot filtered FFT on left y-axis
            ax_middle.plot(filtered_freq, filtered_fft, color=color_filtered, linewidth=1.5,
                          label=f'Filtered (actual)', alpha=0.8, zorder=2)
            ax_middle.set_ylabel('FFT Magnitude', fontsize=11, color=color_filtered)
            ax_middle.tick_params(axis='y', labelcolor=color_filtered)
            
            # Create second y-axis for filter model (normalized 0 to 1)
            ax_middle_filter = ax_middle.twinx()
            ax_middle_filter.plot(filter_freq, mag_responses['combined_with_dnf'],
                          color='black', linewidth=2.5, linestyle='-',
                          label='Combined filter model', alpha=0.9, zorder=3)
            ax_middle_filter.set_ylabel('Filter Response (normalized)', fontsize=11, color='black')
            ax_middle_filter.tick_params(axis='y', labelcolor='black')
            ax_middle_filter.set_ylim([0, 1.1])
            
            ax_middle.set_title(f'{axis_name} - Filtered vs Model', fontsize=12, fontweight='bold')
            ax_middle.grid(True, alpha=0.3)
            ax_middle.set_xlim([0, min(raw_sample_rate, filtered_sample_rate) / 2])
            
            # Combine legends from both axes
            lines1, labels1 = ax_middle.get_legend_handles_labels()
            lines2, labels2 = ax_middle_filter.get_legend_handles_labels()
            ax_middle.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
            
            ax_middle.xaxis.set_major_locator(MultipleLocator(100))
            ax_middle.xaxis.set_minor_locator(MultipleLocator(20))
            ax_middle.tick_params(axis='x', which='major', labelsize=9)
            ax_middle.tick_params(axis='x', which='minor', labelsize=0)
            
            if row_idx == 2:
                ax_middle.set_xlabel('Frequency (Hz)', fontsize=11)
            
            # === Column 3: Filter magnitude responses ===
            ax_filters = fig.add_subplot(gs[row_idx, 2])
            
            # Plot all individual filter responses (normalized 0 to 1)
            ax_filters.plot(filter_freq, mag_responses['lowpass'], 
                          color='orange', linewidth=2.0, linestyle='--', 
                          label=f'Low-pass ({filter_params["IMU_GYRO_CUTOFF"]:.0f} Hz)', alpha=0.8)
            
            if filter_params['IMU_GYRO_NF0_FRQ'] > 0:
                ax_filters.plot(filter_freq, mag_responses['notch0'],
                              color='red', linewidth=2.0, linestyle='--',
                              label=f'Notch 0 ({filter_params["IMU_GYRO_NF0_FRQ"]:.0f} Hz)', alpha=0.8)
            
            if filter_params['IMU_GYRO_NF1_FRQ'] > 0:
                ax_filters.plot(filter_freq, mag_responses['notch1'],
                              color='purple', linewidth=2.0, linestyle='--',
                              label=f'Notch 1 ({filter_params["IMU_GYRO_NF1_FRQ"]:.1f} Hz)', alpha=0.8)
            
            # Plot all dynamic notches if available
            dnf_colors = ['magenta', 'deeppink', 'hotpink']
            for dnf_idx, dnf_freq in enumerate(dnf_freqs):
                if f'dnf_{dnf_idx}' in mag_responses:
                    ax_filters.plot(filter_freq, mag_responses[f'dnf_{dnf_idx}'],
                                  color=dnf_colors[dnf_idx % len(dnf_colors)], linewidth=2.0, linestyle='-.',
                                  label=f'DNF {dnf_idx+1} ({dnf_freq:.1f} Hz)', alpha=0.8)
            
            # Plot combined filter (static filters only, for comparison)
            ax_filters.plot(filter_freq, mag_responses['combined'],
                          color='gray', linewidth=2.0, linestyle=':',
                          label='Combined (static)', alpha=0.7)
            
            # Plot combined filter with DNF
            ax_filters.plot(filter_freq, mag_responses['combined_with_dnf'],
                          color='black', linewidth=2.5, linestyle='-',
                          label='Combined (all filters)', alpha=0.95)
            
            ax_filters.set_ylabel('Filter Magnitude Response', fontsize=11)
            ax_filters.set_title(f'{axis_name} - Filter Magnitude', fontsize=12, fontweight='bold')
            ax_filters.grid(True, alpha=0.3)
            ax_filters.legend(loc='upper right', fontsize=7, ncol=1)
            ax_filters.set_xlim([0, min(raw_sample_rate, filtered_sample_rate) / 2])
            ax_filters.set_ylim([0, 1.1])
            
            ax_filters.xaxis.set_major_locator(MultipleLocator(100))
            ax_filters.xaxis.set_minor_locator(MultipleLocator(20))
            ax_filters.tick_params(axis='x', which='major', labelsize=9)
            ax_filters.tick_params(axis='x', which='minor', labelsize=0)
            
            if row_idx == 2:
                ax_filters.set_xlabel('Frequency (Hz)', fontsize=11)
            
            # Add filter parameters annotation
            textstr = f'LP: {filter_params["IMU_GYRO_CUTOFF"]:.0f} Hz\n'
            textstr += f'NF0: {filter_params["IMU_GYRO_NF0_FRQ"]:.0f} Hz\n'
            textstr += f'NF1: {filter_params["IMU_GYRO_NF1_FRQ"]:.1f} Hz\n'
            if len(dnf_freqs) > 0:
                for idx, freq in enumerate(dnf_freqs):
                    textstr += f'DNF{idx+1}: {freq:.1f} Hz\n'
            ax_filters.text(0.98, 0.02, textstr.rstrip('\n'), transform=ax_filters.transAxes,
                        verticalalignment='bottom', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                        fontsize=7)
            
            # === Column 4: Filter phase responses ===
            ax_phase = fig.add_subplot(gs[row_idx, 3])
            
            # Plot all individual filter phase responses
            ax_phase.plot(filter_freq, phase_responses['lowpass'], 
                        color='orange', linewidth=2.0, linestyle='--', 
                        label=f'Low-pass ({filter_params["IMU_GYRO_CUTOFF"]:.0f} Hz)', alpha=0.8)
            
            if filter_params['IMU_GYRO_NF0_FRQ'] > 0:
                ax_phase.plot(filter_freq, phase_responses['notch0'],
                            color='red', linewidth=2.0, linestyle='--',
                            label=f'Notch 0 ({filter_params["IMU_GYRO_NF0_FRQ"]:.0f} Hz)', alpha=0.8)
            
            if filter_params['IMU_GYRO_NF1_FRQ'] > 0:
                ax_phase.plot(filter_freq, phase_responses['notch1'],
                            color='purple', linewidth=2.0, linestyle='--',
                            label=f'Notch 1 ({filter_params["IMU_GYRO_NF1_FRQ"]:.1f} Hz)', alpha=0.8)
            
            # Plot dynamic notch phase responses
            for dnf_idx, dnf_freq in enumerate(dnf_freqs):
                if f'dnf_{dnf_idx}' in phase_responses:
                    ax_phase.plot(filter_freq, phase_responses[f'dnf_{dnf_idx}'],
                                color=dnf_colors[dnf_idx % len(dnf_colors)], linewidth=2.0, linestyle='-.',
                                label=f'DNF {dnf_idx+1} ({dnf_freq:.1f} Hz)', alpha=0.8)
            
            # Plot combined phase (static filters)
            ax_phase.plot(filter_freq, phase_responses['combined'],
                        color='gray', linewidth=2.0, linestyle=':',
                        label='Combined (static)', alpha=0.7)
            
            # Plot combined phase with DNF
            ax_phase.plot(filter_freq, phase_responses['combined_with_dnf'],
                        color='black', linewidth=2.5, linestyle='-',
                        label='Combined (all filters)', alpha=0.95)
            
            ax_phase.set_ylabel('Phase Response (degrees)', fontsize=11)
            ax_phase.set_title(f'{axis_name} - Filter Phase', fontsize=12, fontweight='bold')
            ax_phase.grid(True, alpha=0.3)
            ax_phase.legend(loc='lower left', fontsize=7, ncol=1)
            ax_phase.set_xlim([0, min(raw_sample_rate, filtered_sample_rate) / 2])
            ax_phase.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
            
            ax_phase.xaxis.set_major_locator(MultipleLocator(100))
            ax_phase.xaxis.set_minor_locator(MultipleLocator(20))
            ax_phase.tick_params(axis='x', which='major', labelsize=9)
            ax_phase.tick_params(axis='x', which='minor', labelsize=0)
            
            if row_idx == 2:
                ax_phase.set_xlabel('Frequency (Hz)', fontsize=11)
        
        fig_title = f'Raw vs Filtered Gyro FFT with Filter Models'
        if PYFFTW_AVAILABLE:
            fig_title += ' (FFTW)'
        else:
            fig_title += ' (SciPy FFT)'
        
        plt.suptitle(fig_title, fontsize=14, fontweight='bold', y=0.995)
        
        save_file = output_path / "raw_vs_filtered_fft_comparison.png"
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"\n  Saved: {save_file}")
        
        # Print filter configuration summary
        print("\nFilter Configuration:")
        print(f"  Low-pass cutoff: {filter_params['IMU_GYRO_CUTOFF']:.1f} Hz")
        print(f"  Notch 0: {filter_params['IMU_GYRO_NF0_FRQ']:.1f} Hz (BW: {filter_params['IMU_GYRO_NF0_BW']:.1f} Hz)")
        print(f"  Notch 1: {filter_params['IMU_GYRO_NF1_FRQ']:.1f} Hz (BW: {filter_params['IMU_GYRO_NF1_BW']:.1f} Hz)")
        print(f"  DNF enabled: {filter_params['IMU_GYRO_DNF_EN']}")
        print(f"  DNF min freq: {filter_params['IMU_GYRO_DNF_MIN']:.1f} Hz")
        print(f"  DNF bandwidth: {filter_params['IMU_GYRO_DNF_BW']:.1f} Hz")
        print(f"  DNF harmonics: {filter_params['IMU_GYRO_DNF_HMC']}")
        
        print("\nNote on Filter Model Display:")
        print("  - Column 2 shows filtered FFT (left axis) and normalized filter model (right axis)")
        print("  - Column 3 shows normalized filter response (0 to 1) - DC gain is always 1.0")
        print("  - Filters are linear systems and don't change DC gain across axes")
        print("  - Filter model uses consistent 0-1 scale across all axes")
        
        return True


def create_tiled_comparison_plot(configs, output_dir, dnf_max, dnf_harmonics, dnf_snr_min):
    """Create a 4x6 tiled comparison plot for multiple ulog files"""
    from matplotlib.ticker import MultipleLocator
    
    # Prepare output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create 4 rows x 6 columns subplot (4 rows: engine freq, roll, pitch, yaw)
    fig = plt.figure(figsize=(36, 20))
    gs = fig.add_gridspec(4, 6, hspace=0.35, wspace=0.3)
    
    axis_names = ['Roll Rate', 'Pitch Rate', 'Yaw Rate']
    axis_keys = ['x', 'y', 'z']
    
    # Process each configuration
    for col_idx, config in enumerate(configs[:6]):  # Limit to 6 columns
        print(f"\nProcessing {config['name']}...")
        
        # Create analyzer
        analyzer = FilterAnalyzer(config['file'])
        if not analyzer.load_ulog():
            print(f"  Failed to load {config['file']}")
            continue
        
        # Extract data
        angular_vel, gyro_sample_rate = analyzer.extract_angular_velocity()
        engine_data, engine_sample_rate = analyzer.extract_engine_data()
        
        if angular_vel is None:
            print(f"  No angular velocity data")
            continue
        
        dnf_min = config['dnf_min']
        dnf_bandwidth = config['dnf_bandwidth']
        
        # Row 0: Engine frequency (if available)
        if engine_data is not None and 'rpm' in engine_data:
            ax = fig.add_subplot(gs[0, col_idx])
            rpm_data = engine_data['rpm']
            rpm_hz = rpm_data / 60.0
            time = engine_data['timestamp']
            
            ax.plot(time, rpm_hz, 'b-', linewidth=0.8)
            ax.set_ylabel('Engine Hz', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)
            
            if col_idx == 0:
                ax.set_title(f"{config['name']}\nMIN={dnf_min} Hz, BW={dnf_bandwidth} Hz", 
                           fontsize=10, fontweight='bold')
            else:
                ax.set_title(f"MIN={dnf_min} Hz, BW={dnf_bandwidth} Hz", 
                           fontsize=10, fontweight='bold')
        
        # Rows 1-3: Roll, Pitch, Yaw FFTs
        for row_idx, (axis_name, axis_key) in enumerate(zip(axis_names, axis_keys), start=1):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            
            data = angular_vel[axis_key]
            freq, fft_db, fft_mag = analyzer.compute_fft(data, gyro_sample_rate)
            
            # Plot FFT
            ax.plot(freq, fft_mag, linewidth=0.8)
            ax.set_ylabel('Magnitude', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, gyro_sample_rate / 2])
            ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.tick_params(labelsize=8)
            
            if row_idx == 3:  # Last row
                ax.set_xlabel('Frequency (Hz)', fontsize=9)
            
            if col_idx == 0:
                ax.text(-0.25, 0.5, axis_name, transform=ax.transAxes,
                       fontsize=11, fontweight='bold', rotation=90,
                       verticalalignment='center', horizontalalignment='right')
            
            # Find peaks and plot notch filters
            freq_mask = (freq >= dnf_min) & (freq <= dnf_max)
            bin_mag_sum = np.sum(fft_mag)
            MIN_SNR = dnf_snr_min
            
            MAX_NUM_PEAKS = 3
            peak_colors = ['red', 'orange', 'purple']
            peaks_found = []
            
            if len(freq[freq_mask]) > 0:
                fft_mag_work = fft_mag.copy()
                
                for peak_idx in range(MAX_NUM_PEAKS):
                    peak_mask = (freq >= dnf_min) & (freq <= dnf_max)
                    masked_mag = np.where(peak_mask, fft_mag_work, 0)
                    
                    if np.max(masked_mag) == 0:
                        break
                    
                    peak_bin_idx = np.argmax(masked_mag)
                    peak_freq = freq[peak_bin_idx]
                    peak_mag = fft_mag_work[peak_bin_idx]
                    
                    snr_db = 10.0 * np.log10((len(fft_mag) - 1) * peak_mag / (bin_mag_sum - peak_mag + 1e-10))
                    
                    if snr_db > MIN_SNR:
                        peaks_found.append({
                            'freq': peak_freq,
                            'mag': peak_mag,
                            'snr': snr_db,
                            'color': peak_colors[peak_idx]
                        })
                    
                    if peak_bin_idx > 0:
                        fft_mag_work[peak_bin_idx - 1] = 0
                    fft_mag_work[peak_bin_idx] = 0
                    if peak_bin_idx < len(fft_mag_work) - 1:
                        fft_mag_work[peak_bin_idx + 1] = 0
                
                # Draw notch filters
                for peak_info in peaks_found:
                    peak_freq = peak_info['freq']
                    color = peak_info['color']
                    
                    for harmonic in range(dnf_harmonics + 1):
                        if harmonic == 0:
                            harmonic_freq = peak_freq
                        else:
                            harmonic_freq = peak_freq * (harmonic + 1)
                        
                        if harmonic_freq > gyro_sample_rate / 2:
                            break
                        
                        notch_lower = harmonic_freq - dnf_bandwidth / 2
                        notch_upper = harmonic_freq + dnf_bandwidth / 2
                        
                        ax.axvline(notch_lower, color=color, linestyle='--', linewidth=1, alpha=0.6)
                        ax.axvline(notch_upper, color=color, linestyle='--', linewidth=1, alpha=0.6)
                        ax.axvline(harmonic_freq, color=color, linestyle=':', linewidth=0.8, alpha=0.5)
    
    plt.suptitle('Dynamic Notch Filter Comparison - Angular Velocity FFT', 
                fontsize=16, fontweight='bold', y=0.995)
    
    save_file = output_path / "dnf_comparison_tiled.png"
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved: {save_file}")


def main():
    """Main function"""
    import argparse
    import re
    import glob
    
    parser = argparse.ArgumentParser(description="Analyze internal filters in PX4 ulogs")
    parser.add_argument("--ulog", type=Path, help="Path to .ulg file")
    parser.add_argument("--ulog-dir", type=Path, help="Directory containing multiple .ulg files to process in batch")
    parser.add_argument("--output", type=Path, default=Path("./Outputs/FilterAnalysis"), 
                       help="Output directory for plots")
    parser.add_argument("--dnf-bandwidth", type=float, default=10.0,
                       help="Dynamic notch filter bandwidth in Hz (default: 10.0)")
    parser.add_argument("--dnf-min", type=float, default=0.0,
                       help="Minimum frequency for peak detection in Hz (default: 0.0)")
    parser.add_argument("--dnf-max", type=float, default=1000.0,
                       help="Maximum frequency for peak detection in Hz (default: 1000.0)")
    parser.add_argument("--dnf-harmonics", type=int, default=0,
                       help="Number of harmonics to display (default: 0)")
    parser.add_argument("--dnf-snr-min", type=float, default=1.0,
                       help="Minimum SNR in dB for peak detection (default: 1.0)")
    parser.add_argument("--batch-plot", action='store_true',
                       help="Create a tiled comparison plot from all ulogs in directory")
    
    args = parser.parse_args()
    
    # Batch processing mode
    if args.batch_plot and args.ulog_dir:
        if not args.ulog_dir.exists():
            print(f"Error: Directory not found: {args.ulog_dir}")
            return 1
        
        # Find all .ulg files
        ulog_files = sorted(args.ulog_dir.glob("*.ulg"))
        if not ulog_files:
            print(f"Error: No .ulg files found in {args.ulog_dir}")
            return 1
        
        print(f"Found {len(ulog_files)} .ulg files for batch processing")
        
        # Parse DNF parameters from filenames
        configs = []
        for ulog_file in ulog_files:
            # Extract DNF_MIN and BW from filename
            match = re.search(r'DNF_MIN_(\d+)_BW_(\d+)', ulog_file.name)
            if match:
                dnf_min = float(match.group(1))
                dnf_bandwidth = float(match.group(2))
                configs.append({
                    'file': ulog_file,
                    'dnf_min': dnf_min,
                    'dnf_bandwidth': dnf_bandwidth,
                    'name': ulog_file.stem
                })
                print(f"  {ulog_file.name}: MIN={dnf_min} Hz, BW={dnf_bandwidth} Hz")
        
        if not configs:
            print("Error: No files with DNF parameters found in filenames")
            return 1
        
        # Create tiled comparison plot
        create_tiled_comparison_plot(configs, args.output, args.dnf_max, args.dnf_harmonics, args.dnf_snr_min)
        
        print("\nBatch analysis complete!")
        return 0
    
    # Single file mode
    if not args.ulog:
        print("Error: Either --ulog or --ulog-dir with --batch-plot must be specified")
        return 1
    
    if not args.ulog.exists():
        print(f"Error: ULog file not found: {args.ulog}")
        return 1
    
    # Create analyzer
    analyzer = FilterAnalyzer(args.ulog)
    
    # Load ulog
    if not analyzer.load_ulog():
        return 1
    
    # Print all available signals and save to file
    signals_file = args.output / "ulog_signals_list.txt"
    analyzer.print_all_signals(output_file=signals_file)
    
    # Plot FFT analysis
    print("\nGenerating FFT plots...")
    analyzer.plot_angular_velocity_and_engine_rpm_fft(args.output, 
                                                       dnf_bandwidth=args.dnf_bandwidth,
                                                       dnf_min=args.dnf_min,
                                                       dnf_max=args.dnf_max,
                                                       dnf_harmonics=args.dnf_harmonics,
                                                       dnf_snr_min=args.dnf_snr_min)
    
    # Plot gyro FFT peaks over time
    print("\nGenerating gyro FFT peaks over time plot...")
    analyzer.plot_gyro_fft_peaks_over_time(args.output)
    
    # Plot raw vs filtered FFT comparison
    print("\nGenerating raw vs filtered FFT comparison...")
    analyzer.plot_raw_vs_filtered_fft(args.output)
    
    # Plot dynamic notch evolution
    print("\nGenerating dynamic notch evolution plot...")
    filter_params = {
        'IMU_GYRO_DNF_BW': args.dnf_bandwidth,
        'IMU_GYRO_DNF_MIN': args.dnf_min,
    }
    analyzer.plot_dynamic_notch_evolution(args.output, filter_params=filter_params, num_snapshots=20)
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
