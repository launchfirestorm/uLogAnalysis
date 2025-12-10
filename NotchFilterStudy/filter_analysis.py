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
from pathlib import Path
from pyulog import ULog

# Add FlightReviewLib to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'FlightReviewLib'))

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
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
