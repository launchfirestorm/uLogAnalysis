#!/usr/bin/env python3
"""
Sensor Noise Variance Analysis Script

This script extracts raw sensor data from PX4 ULG files and performs sophisticated noise analysis:
1. Basic method: Stationary segment analysis with gravity compensation
2. Allan variance method: Proper noise density and bias instability characterization

Author: Generated for flight log analysis
Date: November 19, 2025
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.spatial.transform import Rotation as R
from pyulog import ULog
import warnings
warnings.filterwarnings('ignore')

# Add FlightReviewLib to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'FlightReviewLib'))

class SensorNoiseAnalyzer:
    """Class to analyze sensor noise characteristics from ULG files"""
    
    def __init__(self, ulg_filename):
        """Initialize analyzer with ULG file"""
        self.ulg_filename = ulg_filename
        self.ulog = None
        self.sensor_data = {}
        self.noise_analysis = {}
        self.stationary_segments = []
        
    def load_ulog(self):
        """Load ULG file"""
        try:
            print(f"Loading ULG file: {self.ulg_filename}")
            self.ulog = ULog(self.ulg_filename)
            print(f"Successfully loaded ULG file with {len(self.ulog.data_list)} datasets")
            return True
        except Exception as e:
            print(f"Error loading ULG file: {e}")
            return False
    
    def extract_sensor_data(self):
        """Extract raw sensor data and attitude data from ULG file"""
        if not self.ulog:
            print("ULG file not loaded")
            return False
            
        print("Extracting sensor data...")
        
        # Extract high-rate sensor data from sensor_combined (most important)
        try:
            combined_dataset = self.ulog.get_dataset('sensor_combined')
            timestamp = combined_dataset.data['timestamp'] / 1e6  # Convert to seconds
            
            self.sensor_data['high_rate'] = {
                'timestamp': timestamp,
                'sample_rate': len(timestamp) / (timestamp[-1] - timestamp[0]),
                'accelerometer': {
                    'x': combined_dataset.data['accelerometer_m_s2[0]'],
                    'y': combined_dataset.data['accelerometer_m_s2[1]'],
                    'z': combined_dataset.data['accelerometer_m_s2[2]']
                },
                'gyroscope': {
                    'x': combined_dataset.data['gyro_rad[0]'],
                    'y': combined_dataset.data['gyro_rad[1]'],
                    'z': combined_dataset.data['gyro_rad[2]']
                }
            }
            print(f"  Extracted high-rate sensor data: {len(timestamp)} samples at {self.sensor_data['high_rate']['sample_rate']:.1f} Hz")
        except Exception as e:
            print(f"  Error extracting sensor_combined: {e}")
            return False
        
        # Extract attitude data for gravity compensation
        try:
            attitude_dataset = self.ulog.get_dataset('vehicle_attitude')
            att_timestamp = attitude_dataset.data['timestamp'] / 1e6
            
            # Extract quaternions
            self.sensor_data['attitude'] = {
                'timestamp': att_timestamp,
                'q': np.column_stack([
                    attitude_dataset.data['q[0]'],  # w
                    attitude_dataset.data['q[1]'],  # x
                    attitude_dataset.data['q[2]'],  # y
                    attitude_dataset.data['q[3]']   # z
                ])
            }
            print(f"  Extracted attitude data: {len(att_timestamp)} samples")
        except Exception as e:
            print(f"  Warning: Could not extract attitude data: {e}")
            self.sensor_data['attitude'] = None
        
        # Extract vehicle status for stationary detection
        try:
            status_dataset = self.ulog.get_dataset('vehicle_status')
            status_timestamp = status_dataset.data['timestamp'] / 1e6
            
            self.sensor_data['status'] = {
                'timestamp': status_timestamp,
                'arming_state': status_dataset.data['arming_state'],
                'nav_state': status_dataset.data['nav_state']
            }
            print(f"  Extracted vehicle status: {len(status_timestamp)} samples")
        except Exception as e:
            print(f"  Warning: Could not extract vehicle status: {e}")
            self.sensor_data['status'] = None
        
        return True
    
    def find_stationary_segments(self, min_duration=30.0, max_accel_std=2.0, max_gyro_std=0.1):
        """
        Find segments where the vehicle is stationary for noise analysis
        
        Args:
            min_duration: Minimum duration in seconds for a stationary segment
            max_accel_std: Maximum acceleration standard deviation (m/s²) for stationary
            max_gyro_std: Maximum gyro standard deviation (rad/s) for stationary
        """
        print(f"Finding stationary segments (min duration: {min_duration}s)...")
        
        if 'high_rate' not in self.sensor_data:
            print("  No high-rate sensor data available")
            return
        
        timestamp = self.sensor_data['high_rate']['timestamp']
        accel_data = self.sensor_data['high_rate']['accelerometer']
        gyro_data = self.sensor_data['high_rate']['gyroscope']
        
        # Window size for stationarity check (5 seconds)
        window_duration = 5.0
        sample_rate = self.sensor_data['high_rate']['sample_rate']
        window_size = int(window_duration * sample_rate)
        
        # Compute acceleration magnitude
        accel_mag = np.sqrt(accel_data['x']**2 + accel_data['y']**2 + accel_data['z']**2)
        gyro_mag = np.sqrt(gyro_data['x']**2 + gyro_data['y']**2 + gyro_data['z']**2)
        
        stationary_windows = []
        
        # Check each window for stationarity
        for i in range(0, len(timestamp) - window_size, window_size // 2):
            end_idx = min(i + window_size, len(timestamp))
            
            # Check acceleration variability
            accel_std = np.std(accel_mag[i:end_idx])
            gyro_std = np.std(gyro_mag[i:end_idx])
            
            if accel_std < max_accel_std and gyro_std < max_gyro_std:
                stationary_windows.append((i, end_idx, timestamp[i], timestamp[end_idx-1]))
        
        # Merge adjacent stationary windows
        merged_segments = []
        if stationary_windows:
            current_start_idx, current_end_idx, current_start_time, _ = stationary_windows[0]
            
            for start_idx, end_idx, start_time, end_time in stationary_windows[1:]:
                if start_idx <= current_end_idx:  # Adjacent or overlapping
                    current_end_idx = end_idx
                else:
                    # Check if segment is long enough
                    if (timestamp[current_end_idx-1] - timestamp[current_start_idx]) >= min_duration:
                        merged_segments.append((current_start_idx, current_end_idx, 
                                              timestamp[current_start_idx], timestamp[current_end_idx-1]))
                    current_start_idx, current_end_idx = start_idx, end_idx
            
            # Add the last segment
            if (timestamp[current_end_idx-1] - timestamp[current_start_idx]) >= min_duration:
                merged_segments.append((current_start_idx, current_end_idx,
                                      timestamp[current_start_idx], timestamp[current_end_idx-1]))
        
        self.stationary_segments = merged_segments
        print(f"  Found {len(self.stationary_segments)} stationary segments:")
        for i, (start_idx, end_idx, start_time, end_time) in enumerate(self.stationary_segments):
            duration = end_time - start_time
            print(f"    Segment {i+1}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s duration)")
    
    def compensate_gravity(self, accel_data, timestamps):
        """
        Compensate gravity from accelerometer data using attitude information
        
        Args:
            accel_data: Dictionary with 'x', 'y', 'z' acceleration data
            timestamps: Timestamp array for acceleration data
            
        Returns:
            Dictionary with gravity-compensated acceleration data
        """
        if self.sensor_data['attitude'] is None:
            print("  No attitude data available, using simple gravity compensation")
            # Simple approach: subtract mean from Z-axis (assuming mostly level)
            return {
                'x': accel_data['x'] - np.mean(accel_data['x']),
                'y': accel_data['y'] - np.mean(accel_data['y']),
                'z': accel_data['z'] - np.mean(accel_data['z'])
            }
        
        # Interpolate quaternions to sensor timestamps
        att_timestamps = self.sensor_data['attitude']['timestamp']
        att_q = self.sensor_data['attitude']['q']
        
        # Interpolate each quaternion component
        q_interp = np.zeros((len(timestamps), 4))
        for i in range(4):
            q_interp[:, i] = np.interp(timestamps, att_timestamps, att_q[:, i])
        
        # Normalize quaternions
        q_norms = np.linalg.norm(q_interp, axis=1)
        q_interp = q_interp / q_norms[:, np.newaxis]
        
        # Create rotation objects
        rotations = R.from_quat(q_interp[:, [1, 2, 3, 0]])  # scipy uses [x,y,z,w] format
        
        # Expected gravity vector in NED frame (0, 0, 9.81)
        gravity_ned = np.array([0, 0, 9.81])
        
        # Rotate gravity to body frame for each sample
        gravity_body = rotations.apply(gravity_ned)
        
        # Subtract expected gravity from measured acceleration
        accel_compensated = {
            'x': accel_data['x'] - gravity_body[:, 0],
            'y': accel_data['y'] - gravity_body[:, 1],
            'z': accel_data['z'] - gravity_body[:, 2]
        }
        
        return accel_compensated
    
    def compute_allan_variance(self, data, sample_rate, max_tau_factor=0.1):
        """
        Compute Allan variance for noise characterization
        
        Args:
            data: 1D array of sensor data
            sample_rate: Sample rate in Hz
            max_tau_factor: Maximum tau as fraction of total time
            
        Returns:
            tau: Array of averaging times
            allan_var: Array of Allan variance values
        """
        N = len(data)
        dt = 1.0 / sample_rate
        max_tau = min(int(N * max_tau_factor), N // 2)
        
        # Logarithmically spaced tau values
        tau_samples = np.logspace(0, np.log10(max_tau), 50).astype(int)
        tau_samples = np.unique(tau_samples)
        tau_samples = tau_samples[tau_samples > 0]
        
        tau_values = tau_samples * dt
        allan_var = np.zeros(len(tau_samples))
        
        for i, m in enumerate(tau_samples):
            if m >= N // 2:
                break
                
            # Compute overlapping Allan deviation
            n_clusters = N // m
            if n_clusters < 2:
                break
                
            cluster_means = np.zeros(n_clusters)
            for j in range(n_clusters):
                start_idx = j * m
                end_idx = start_idx + m
                if end_idx <= N:
                    cluster_means[j] = np.mean(data[start_idx:end_idx])
            
            # Allan variance is half the variance of successive differences
            if len(cluster_means) > 1:
                diffs = np.diff(cluster_means)
                allan_var[i] = 0.5 * np.mean(diffs**2)
        
        # Remove zero entries
        valid_idx = allan_var > 0
        return tau_values[valid_idx], allan_var[valid_idx]
    
    def analyze_basic_noise(self):
        """Perform basic noise analysis on stationary segments"""
        print("Performing basic noise analysis...")
        
        if not self.stationary_segments:
            print("  No stationary segments found!")
            return
        
        self.noise_analysis['basic'] = {}
        
        # Analyze each stationary segment
        for seg_idx, (start_idx, end_idx, start_time, end_time) in enumerate(self.stationary_segments):
            print(f"  Analyzing segment {seg_idx + 1} ({start_time:.1f}s - {end_time:.1f}s)")
            
            # Extract segment data
            timestamp_seg = self.sensor_data['high_rate']['timestamp'][start_idx:end_idx]
            accel_seg = {
                'x': self.sensor_data['high_rate']['accelerometer']['x'][start_idx:end_idx],
                'y': self.sensor_data['high_rate']['accelerometer']['y'][start_idx:end_idx],
                'z': self.sensor_data['high_rate']['accelerometer']['z'][start_idx:end_idx]
            }
            gyro_seg = {
                'x': self.sensor_data['high_rate']['gyroscope']['x'][start_idx:end_idx],
                'y': self.sensor_data['high_rate']['gyroscope']['y'][start_idx:end_idx],
                'z': self.sensor_data['high_rate']['gyroscope']['z'][start_idx:end_idx]
            }
            
            # Compensate gravity for accelerometer
            accel_compensated = self.compensate_gravity(accel_seg, timestamp_seg)
            
            # Remove mean (detrend) for both sensors
            accel_detrended = {
                axis: data - np.mean(data) for axis, data in accel_compensated.items()
            }
            gyro_detrended = {
                axis: data - np.mean(data) for axis, data in gyro_seg.items()
            }
            
            # Compute variance and standard deviation
            segment_analysis = {
                'segment_info': {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'samples': end_idx - start_idx
                },
                'accelerometer': {},
                'gyroscope': {}
            }
            
            for axis in ['x', 'y', 'z']:
                # Accelerometer analysis
                accel_var = np.var(accel_detrended[axis])
                accel_std = np.sqrt(accel_var)
                segment_analysis['accelerometer'][axis] = {
                    'variance': accel_var,
                    'std_dev': accel_std,
                    'rms': np.sqrt(np.mean(accel_detrended[axis]**2))
                }
                
                # Gyroscope analysis
                gyro_var = np.var(gyro_detrended[axis])
                gyro_std = np.sqrt(gyro_var)
                segment_analysis['gyroscope'][axis] = {
                    'variance': gyro_var,
                    'std_dev': gyro_std,
                    'rms': np.sqrt(np.mean(gyro_detrended[axis]**2))
                }
            
            self.noise_analysis['basic'][f'segment_{seg_idx}'] = segment_analysis
    
    def analyze_allan_variance(self):
        """Perform Allan variance analysis on stationary segments"""
        print("Performing Allan variance analysis...")
        
        if not self.stationary_segments:
            print("  No stationary segments found!")
            return
        
        self.noise_analysis['allan'] = {}
        sample_rate = self.sensor_data['high_rate']['sample_rate']
        
        # Use the longest stationary segment for Allan analysis
        longest_segment = max(self.stationary_segments, key=lambda x: x[3] - x[2])
        start_idx, end_idx, start_time, end_time = longest_segment
        
        print(f"  Using longest segment ({start_time:.1f}s - {end_time:.1f}s, {end_time-start_time:.1f}s duration)")
        
        # Extract segment data
        timestamp_seg = self.sensor_data['high_rate']['timestamp'][start_idx:end_idx]
        accel_seg = {
            'x': self.sensor_data['high_rate']['accelerometer']['x'][start_idx:end_idx],
            'y': self.sensor_data['high_rate']['accelerometer']['y'][start_idx:end_idx],
            'z': self.sensor_data['high_rate']['accelerometer']['z'][start_idx:end_idx]
        }
        gyro_seg = {
            'x': self.sensor_data['high_rate']['gyroscope']['x'][start_idx:end_idx],
            'y': self.sensor_data['high_rate']['gyroscope']['y'][start_idx:end_idx],
            'z': self.sensor_data['high_rate']['gyroscope']['z'][start_idx:end_idx]
        }
        
        # Compensate gravity for accelerometer
        accel_compensated = self.compensate_gravity(accel_seg, timestamp_seg)
        
        # Remove mean
        accel_detrended = {
            axis: data - np.mean(data) for axis, data in accel_compensated.items()
        }
        gyro_detrended = {
            axis: data - np.mean(data) for axis, data in gyro_seg.items()
        }
        
        allan_results = {
            'segment_info': {
                'start_time': start_time,
                'end_time': end_time,
                'duration': end_time - start_time,
                'sample_rate': sample_rate
            },
            'accelerometer': {},
            'gyroscope': {}
        }
        
        # Compute Allan variance for each axis
        for axis in ['x', 'y', 'z']:
            print(f"    Computing Allan variance for accelerometer {axis}-axis")
            tau_accel, avar_accel = self.compute_allan_variance(
                accel_detrended[axis], sample_rate
            )
            
            print(f"    Computing Allan variance for gyroscope {axis}-axis")
            tau_gyro, avar_gyro = self.compute_allan_variance(
                gyro_detrended[axis], sample_rate
            )
            
            # Estimate noise parameters
            # White noise region is typically at tau = 1/sample_rate
            # Find the minimum Allan deviation (approximately white noise level)
            if len(avar_accel) > 0:
                min_avar_accel = np.min(avar_accel)
                white_noise_density_accel = np.sqrt(min_avar_accel) * np.sqrt(sample_rate / 2)
                
                allan_results['accelerometer'][axis] = {
                    'tau': tau_accel,
                    'allan_variance': avar_accel,
                    'allan_deviation': np.sqrt(avar_accel),
                    'noise_density': white_noise_density_accel,
                    'per_sample_sigma': white_noise_density_accel / np.sqrt(sample_rate / 2)
                }
            
            if len(avar_gyro) > 0:
                min_avar_gyro = np.min(avar_gyro)
                white_noise_density_gyro = np.sqrt(min_avar_gyro) * np.sqrt(sample_rate / 2)
                
                allan_results['gyroscope'][axis] = {
                    'tau': tau_gyro,
                    'allan_variance': avar_gyro,
                    'allan_deviation': np.sqrt(avar_gyro),
                    'noise_density': white_noise_density_gyro,
                    'per_sample_sigma': white_noise_density_gyro / np.sqrt(sample_rate / 2)
                }
        
        self.noise_analysis['allan'] = allan_results
    
    def plot_noise_analysis(self, save_plots=True, show_plots=True):
        """Plot noise analysis results"""
        print("Generating plots...")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join(os.path.dirname(self.ulg_filename), 'Outputs', 'sensor_noise')
        os.makedirs(output_dir, exist_ok=True)
        
        colors = {'x': 'red', 'y': 'green', 'z': 'blue'}
        
        # Plot 1: Stationary segments identification
        if self.stationary_segments:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            
            timestamp = self.sensor_data['high_rate']['timestamp']
            accel_data = self.sensor_data['high_rate']['accelerometer']
            gyro_data = self.sensor_data['high_rate']['gyroscope']
            
            # Acceleration magnitude
            accel_mag = np.sqrt(accel_data['x']**2 + accel_data['y']**2 + accel_data['z']**2)
            gyro_mag = np.sqrt(gyro_data['x']**2 + gyro_data['y']**2 + gyro_data['z']**2)
            
            # Plot acceleration
            ax1.plot(timestamp, accel_mag, 'b-', alpha=0.7, linewidth=0.5)
            ax1.set_ylabel('Acceleration Magnitude [m/s²]')
            ax1.set_title('Stationary Segments Identification')
            ax1.grid(True, alpha=0.3)
            
            # Plot gyroscope
            ax2.plot(timestamp, gyro_mag, 'r-', alpha=0.7, linewidth=0.5)
            ax2.set_ylabel('Angular Rate Magnitude [rad/s]')
            ax2.set_xlabel('Time [s]')
            ax2.grid(True, alpha=0.3)
            
            # Highlight stationary segments
            for start_idx, end_idx, start_time, end_time in self.stationary_segments:
                ax1.axvspan(start_time, end_time, alpha=0.3, color='green', label='Stationary')
                ax2.axvspan(start_time, end_time, alpha=0.3, color='green')
            
            # Remove duplicate labels
            handles, labels = ax1.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax1.legend(by_label.values(), by_label.keys())
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(os.path.join(output_dir, 'stationary_segments.png'), dpi=300, bbox_inches='tight')
                print("  Saved stationary segments plot")
            
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
        
        # Plot 2: Allan variance plots
        if 'allan' in self.noise_analysis:
            allan_data = self.noise_analysis['allan']
            
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accelerometer Allan deviation
            ax1.set_title('Accelerometer Allan Deviation')
            ax1.set_xlabel('Averaging Time τ [s]')
            ax1.set_ylabel('Allan Deviation [m/s²]')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            ax1.grid(True, alpha=0.3)
            
            for axis in ['x', 'y', 'z']:
                if axis in allan_data['accelerometer']:
                    tau = allan_data['accelerometer'][axis]['tau']
                    adev = allan_data['accelerometer'][axis]['allan_deviation']
                    ax1.plot(tau, adev, color=colors[axis], marker='o', markersize=3, 
                            label=f'{axis}-axis', linewidth=1.5)
            ax1.legend()
            
            # Gyroscope Allan deviation
            ax2.set_title('Gyroscope Allan Deviation')
            ax2.set_xlabel('Averaging Time τ [s]')
            ax2.set_ylabel('Allan Deviation [rad/s]')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
            
            for axis in ['x', 'y', 'z']:
                if axis in allan_data['gyroscope']:
                    tau = allan_data['gyroscope'][axis]['tau']
                    adev = allan_data['gyroscope'][axis]['allan_deviation']
                    ax2.plot(tau, adev, color=colors[axis], marker='o', markersize=3,
                            label=f'{axis}-axis', linewidth=1.5)
            ax2.legend()
            
            # Time series of longest stationary segment (accelerometer)
            ax3.set_title('Longest Stationary Segment - Accelerometer')
            ax3.set_xlabel('Time [s]')
            ax3.set_ylabel('Acceleration [m/s²]')
            ax3.grid(True, alpha=0.3)
            
            start_time = allan_data['segment_info']['start_time']
            end_time = allan_data['segment_info']['end_time']
            
            # Find the segment indices
            timestamp = self.sensor_data['high_rate']['timestamp']
            start_idx = np.argmin(np.abs(timestamp - start_time))
            end_idx = np.argmin(np.abs(timestamp - end_time))
            
            seg_time = timestamp[start_idx:end_idx] - start_time
            
            for axis in ['x', 'y', 'z']:
                accel_data = self.sensor_data['high_rate']['accelerometer'][axis][start_idx:end_idx]
                ax3.plot(seg_time, accel_data, color=colors[axis], alpha=0.7, 
                        linewidth=0.5, label=f'{axis}-axis')
            ax3.legend()
            
            # Time series of longest stationary segment (gyroscope)
            ax4.set_title('Longest Stationary Segment - Gyroscope')
            ax4.set_xlabel('Time [s]')
            ax4.set_ylabel('Angular Rate [rad/s]')
            ax4.grid(True, alpha=0.3)
            
            for axis in ['x', 'y', 'z']:
                gyro_data = self.sensor_data['high_rate']['gyroscope'][axis][start_idx:end_idx]
                ax4.plot(seg_time, gyro_data, color=colors[axis], alpha=0.7,
                        linewidth=0.5, label=f'{axis}-axis')
            ax4.legend()
            
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(os.path.join(output_dir, 'allan_variance_analysis.png'), dpi=300, bbox_inches='tight')
                print("  Saved Allan variance plot")
            
            if show_plots:
                plt.show()
            else:
                plt.close(fig)
    
    def generate_summary_report(self):
        """Generate a summary report of sensor noise analysis"""
        print("Generating summary report...")
        
        output_dir = os.path.join(os.path.dirname(self.ulg_filename), 'Outputs', 'sensor_noise')
        report_filename = os.path.join(output_dir, 'sensor_noise_summary.txt')
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("SENSOR NOISE ANALYSIS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"ULG File: {os.path.basename(self.ulg_filename)}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Stationary segments summary
            f.write("STATIONARY SEGMENTS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Number of segments found: {len(self.stationary_segments)}\n")
            for i, (_, _, start_time, end_time) in enumerate(self.stationary_segments):
                duration = end_time - start_time
                f.write(f"  Segment {i+1}: {start_time:.1f}s - {end_time:.1f}s ({duration:.1f}s)\n")
            
            # Basic noise analysis
            if 'basic' in self.noise_analysis:
                f.write(f"\nBASIC NOISE ANALYSIS\n")
                f.write("-" * 30 + "\n")
                f.write("Per-segment noise characteristics (sigma = sqrt(variance)):\n\n")
                
                for seg_name, seg_data in self.noise_analysis['basic'].items():
                    f.write(f"{seg_name.upper()}\n")
                    duration = seg_data['segment_info']['duration']
                    samples = seg_data['segment_info']['samples']
                    f.write(f"  Duration: {duration:.1f}s, Samples: {samples}\n")
                    
                    f.write("  ACCELEROMETER (gravity compensated):\n")
                    for axis in ['x', 'y', 'z']:
                        std_dev = seg_data['accelerometer'][axis]['std_dev']
                        f.write(f"    {axis}-axis sigma: {std_dev:.6f} m/s^2\n")
                    
                    f.write("  GYROSCOPE (bias removed):\n")
                    for axis in ['x', 'y', 'z']:
                        std_dev = seg_data['gyroscope'][axis]['std_dev']
                        f.write(f"    {axis}-axis sigma: {std_dev:.6f} rad/s\n")
                    f.write("\n")
            
            # Allan variance analysis
            if 'allan' in self.noise_analysis:
                f.write("ALLAN VARIANCE ANALYSIS\n")
                f.write("-" * 30 + "\n")
                allan_data = self.noise_analysis['allan']
                duration = allan_data['segment_info']['duration']
                sample_rate = allan_data['segment_info']['sample_rate']
                f.write(f"Analysis segment duration: {duration:.1f}s\n")
                f.write(f"Sample rate: {sample_rate:.1f} Hz\n\n")
                
                f.write("NOISE PARAMETERS FOR np.random.normal():\n")
                f.write("(sigma_sample = per-sample standard deviation)\n\n")
                
                f.write("ACCELEROMETER:\n")
                for axis in ['x', 'y', 'z']:
                    if axis in allan_data['accelerometer']:
                        sigma = allan_data['accelerometer'][axis]['per_sample_sigma']
                        noise_density = allan_data['accelerometer'][axis]['noise_density']
                        f.write(f"  {axis}-axis:\n")
                        f.write(f"    sigma_sample = {sigma:.6f} m/s^2 (use in np.random.normal(0, {sigma:.6f}))\n")
                        f.write(f"    Noise density = {noise_density:.6f} m/s^2/sqrt(Hz)\n")
                
                f.write("\nGYROSCOPE:\n")
                for axis in ['x', 'y', 'z']:
                    if axis in allan_data['gyroscope']:
                        sigma = allan_data['gyroscope'][axis]['per_sample_sigma']
                        noise_density = allan_data['gyroscope'][axis]['noise_density']
                        f.write(f"  {axis}-axis:\n")
                        f.write(f"    sigma_sample = {sigma:.6f} rad/s (use in np.random.normal(0, {sigma:.6f}))\n")
                        f.write(f"    Noise density = {noise_density:.6f} rad/s/sqrt(Hz)\n")
                
                f.write(f"\nExample usage in Python:\n")
                f.write(f"# For accelerometer noise simulation at {sample_rate:.0f} Hz\n")
                if 'x' in allan_data['accelerometer']:
                    sigma_x = allan_data['accelerometer']['x']['per_sample_sigma']
                    f.write(f"accel_noise_x = np.random.normal(0, {sigma_x:.6f}, num_samples)\n")
                if 'x' in allan_data['gyroscope']:
                    sigma_x = allan_data['gyroscope']['x']['per_sample_sigma']
                    f.write(f"gyro_noise_x = np.random.normal(0, {sigma_x:.6f}, num_samples)\n")
        
        print(f"  Saved summary report: {report_filename}")
    
    def run_analysis(self, save_plots=True, show_plots=False):
        """Run complete sensor noise analysis"""
        print("Starting sensor noise analysis...")
        print("=" * 60)
        
        # Load ULG file
        if not self.load_ulog():
            return False
        
        # Extract sensor data
        if not self.extract_sensor_data():
            return False
        
        # Find stationary segments (relaxed criteria for flight data)
        self.find_stationary_segments(min_duration=10.0, max_accel_std=5.0, max_gyro_std=0.5)
        
        if not self.stationary_segments:
            print("ERROR: No stationary segments found!")
            print("Try adjusting the stationarity criteria or check if the vehicle was stationary during the flight.")
            return False
        
        # Perform basic noise analysis
        self.analyze_basic_noise()
        
        # Perform Allan variance analysis
        self.analyze_allan_variance()
        
        # Generate plots
        self.plot_noise_analysis(save_plots, show_plots)
        
        # Generate summary report
        self.generate_summary_report()
        
        print("\n" + "=" * 60)
        print("Analysis complete!")
        return True


def main():
    """Main function"""
    # Configuration
    ulg_filename = 'B2-017_flight with Trillium gimbal and Arbitrator-15_29_48.ulg'
    
    # Check if file exists
    if not os.path.exists(ulg_filename):
        print(f"Error: ULG file '{ulg_filename}' not found!")
        print("Make sure the file is in the current directory.")
        return
    
    # Create analyzer and run analysis
    analyzer = SensorNoiseAnalyzer(ulg_filename)
    success = analyzer.run_analysis(
        save_plots=True,
        show_plots=False  # Set to True if you want to display plots interactively
    )
    
    if success:
        print("\nSensor noise analysis completed successfully!")
        print("Check the 'Outputs/sensor_noise' directory for results.")
        print("\nKey outputs:")
        print("  - sensor_noise_summary.txt: Noise parameters for simulation")
        print("  - stationary_segments.png: Identified stationary periods")  
        print("  - allan_variance_analysis.png: Allan variance plots")
        print("Use the sigma_sample values from the summary for np.random.normal() in simulations.")
    else:
        print("\nAnalysis failed. Check error messages above.")


if __name__ == "__main__":
    main()
