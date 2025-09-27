"""
Fourier Power Spectra Analysis for Raw Voltage Recordings

This module provides functions for loading and analyzing raw voltage recordings
using FFT to compute power spectra. Designed for use with electrophysiology data.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import struct
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')


class VoltageRecordingAnalyzer:
    """
    Class for analyzing raw voltage recordings with Fourier methods.
    """
    
    def __init__(self, sampling_rate=30000):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        sampling_rate : float
            Sampling rate in Hz (default: 30000 Hz for typical ephys recordings)
        """
        self.sampling_rate = sampling_rate
        self.data = None
        self.time = None
        self.n_channels = None
        
    def convert_windows_path_to_wsl(self, windows_path):
        """
        Convert Windows path to WSL mount path.
        
        Parameters:
        -----------
        windows_path : str
            Windows path (e.g., "C:\\Users\\...")
            
        Returns:
        --------
        str : WSL mount path (e.g., "/mnt/c/Users/...")
        """
        if windows_path.startswith('C:'):
            # Remove C: and replace backslashes with forward slashes
            wsl_path = windows_path[2:].replace('\\', '/')
            wsl_path = '/mnt/c' + wsl_path
        elif windows_path.startswith('/mnt/'):
            # Already a WSL path
            wsl_path = windows_path
        else:
            # Assume it's already a Unix-style path
            wsl_path = windows_path
            
        return wsl_path
    
    def load_amplifier_data(self, file_path, n_channels=64, dtype=np.int16, 
                           duration_sec=None, start_sec=0):
        """
        Load raw amplifier data from .dat file.
        
        Parameters:
        -----------
        file_path : str
            Path to the .dat file
        n_channels : int
            Number of channels in the recording
        dtype : numpy dtype
            Data type of the recorded samples (typically int16)
        duration_sec : float or None
            Duration to load in seconds (None for entire file)
        start_sec : float
            Start time in seconds for loading data
            
        Returns:
        --------
        tuple : (data, time) where data is (n_samples, n_channels) and time is (n_samples,)
        """
        # Convert Windows path if needed
        file_path = self.convert_windows_path_to_wsl(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Calculate file size and number of samples
        file_size = os.path.getsize(file_path)
        bytes_per_sample = np.dtype(dtype).itemsize
        total_samples = file_size // (n_channels * bytes_per_sample)
        
        print(f"File size: {file_size / (1024**3):.2f} GB")
        print(f"Total samples: {total_samples:,}")
        print(f"Recording duration: {total_samples / self.sampling_rate:.2f} seconds")
        
        # Calculate start and end samples
        start_sample = int(start_sec * self.sampling_rate)
        if duration_sec is None:
            end_sample = total_samples
        else:
            end_sample = min(start_sample + int(duration_sec * self.sampling_rate), total_samples)
        
        n_samples_to_load = end_sample - start_sample
        print(f"Loading {n_samples_to_load:,} samples ({n_samples_to_load/self.sampling_rate:.2f} sec)")
        
        # Load data
        with open(file_path, 'rb') as f:
            # Skip to start position
            f.seek(start_sample * n_channels * bytes_per_sample)
            
            # Read data
            data_1d = np.frombuffer(
                f.read(n_samples_to_load * n_channels * bytes_per_sample),
                dtype=dtype
            )
        
        # Reshape to (n_samples, n_channels)
        data = data_1d.reshape(-1, n_channels)
        
        # Create time vector
        time = np.arange(n_samples_to_load) / self.sampling_rate + start_sec
        
        # Store in class
        self.data = data
        self.time = time
        self.n_channels = n_channels
        
        return data, time
    
    def compute_power_spectrum(self, channel_data, nperseg=None, method='welch'):
        """
        Compute power spectrum of a single channel.
        
        Parameters:
        -----------
        channel_data : array
            1D array of voltage data for one channel
        nperseg : int or None
            Length of each segment for Welch's method
        method : str
            Method to use ('welch', 'fft')
            
        Returns:
        --------
        tuple : (frequencies, power_spectrum)
        """
        if method == 'welch':
            if nperseg is None:
                nperseg = min(len(channel_data) // 8, 2**14)  # Default segment size
            
            frequencies, psd = signal.welch(
                channel_data, 
                fs=self.sampling_rate, 
                nperseg=nperseg,
                scaling='density'
            )
        elif method == 'fft':
            # Simple FFT approach
            fft_data = fft(channel_data)
            power_spectrum = np.abs(fft_data)**2 / len(channel_data)
            frequencies = fftfreq(len(channel_data), 1/self.sampling_rate)
            
            # Take only positive frequencies
            positive_freq_idx = frequencies >= 0
            frequencies = frequencies[positive_freq_idx]
            psd = power_spectrum[positive_freq_idx]
        else:
            raise ValueError("Method must be 'welch' or 'fft'")
            
        return frequencies, psd
    
    def analyze_channel(self, channel_idx, freq_bands=None, plot=True):
        """
        Analyze a specific channel.
        
        Parameters:
        -----------
        channel_idx : int
            Channel index to analyze
        freq_bands : dict or None
            Dictionary of frequency bands to analyze (e.g., {'delta': (0.5, 4), 'theta': (4, 8)})
        plot : bool
            Whether to create plots
            
        Returns:
        --------
        dict : Analysis results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_amplifier_data first.")
        
        if channel_idx >= self.n_channels:
            raise ValueError(f"Channel {channel_idx} not available. Max channel: {self.n_channels-1}")
        
        channel_data = self.data[:, channel_idx]
        
        # Compute power spectrum
        frequencies, psd = self.compute_power_spectrum(channel_data)
        
        # Analyze frequency bands if provided
        band_powers = {}
        if freq_bands:
            for band_name, (low_freq, high_freq) in freq_bands.items():
                band_idx = (frequencies >= low_freq) & (frequencies <= high_freq)
                band_power = np.trapz(psd[band_idx], frequencies[band_idx])
                band_powers[band_name] = band_power
        
        # Create plots
        if plot:
            self.plot_channel_analysis(channel_idx, channel_data, frequencies, psd, band_powers)
        
        results = {
            'channel': channel_idx,
            'frequencies': frequencies,
            'psd': psd,
            'band_powers': band_powers,
            'peak_frequency': frequencies[np.argmax(psd[frequencies > 1])],  # Exclude DC
            'total_power': np.trapz(psd, frequencies)
        }
        
        return results
    
    def plot_channel_analysis(self, channel_idx, channel_data, frequencies, psd, band_powers=None):
        """
        Create plots for channel analysis.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Channel {channel_idx} Analysis', fontsize=16)
        
        # Time series (first 1 second)
        time_subset = self.time[:min(len(self.time), int(self.sampling_rate))]
        data_subset = channel_data[:len(time_subset)]
        
        axes[0, 0].plot(time_subset, data_subset)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Voltage (µV)')
        axes[0, 0].set_title('Raw Signal (First 1s)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Power spectrum (log scale)
        axes[0, 1].loglog(frequencies[1:], psd[1:])  # Skip DC component
        axes[0, 1].set_xlabel('Frequency (Hz)')
        axes[0, 1].set_ylabel('Power Spectral Density')
        axes[0, 1].set_title('Power Spectrum (Log-Log)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Power spectrum (linear, up to 100 Hz)
        freq_mask = frequencies <= 100
        axes[1, 0].semilogy(frequencies[freq_mask], psd[freq_mask])
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Power Spectral Density')
        axes[1, 0].set_title('Power Spectrum (0-100 Hz)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Frequency bands analysis
        if band_powers:
            bands = list(band_powers.keys())
            powers = list(band_powers.values())
            
            axes[1, 1].bar(bands, powers)
            axes[1, 1].set_ylabel('Band Power')
            axes[1, 1].set_title('Frequency Band Powers')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            # Histogram of voltage values
            axes[1, 1].hist(channel_data, bins=50, alpha=0.7)
            axes[1, 1].set_xlabel('Voltage (µV)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Voltage Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_all_channels(self, max_channels=None, freq_bands=None):
        """
        Analyze all channels and create summary plots.
        
        Parameters:
        -----------
        max_channels : int or None
            Maximum number of channels to analyze (for speed)
        freq_bands : dict or None
            Frequency bands to analyze
            
        Returns:
        --------
        dict : Results for all channels
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_amplifier_data first.")
        
        n_channels_to_analyze = min(max_channels or self.n_channels, self.n_channels)
        
        print(f"Analyzing {n_channels_to_analyze} channels...")
        
        all_results = {}
        peak_frequencies = []
        total_powers = []
        
        for ch in range(n_channels_to_analyze):
            if ch % 10 == 0:
                print(f"Processing channel {ch}/{n_channels_to_analyze}")
            
            results = self.analyze_channel(ch, freq_bands=freq_bands, plot=False)
            all_results[ch] = results
            peak_frequencies.append(results['peak_frequency'])
            total_powers.append(results['total_power'])
        
        # Create summary plots
        self.plot_channel_summary(peak_frequencies, total_powers, all_results, freq_bands)
        
        return all_results
    
    def plot_channel_summary(self, peak_frequencies, total_powers, all_results, freq_bands=None):
        """
        Create summary plots across all channels.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Channel Analysis Summary', fontsize=16)
        
        # Peak frequencies across channels
        axes[0, 0].plot(peak_frequencies, 'o-', markersize=4)
        axes[0, 0].set_xlabel('Channel')
        axes[0, 0].set_ylabel('Peak Frequency (Hz)')
        axes[0, 0].set_title('Peak Frequency by Channel')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Total power across channels
        axes[0, 1].plot(total_powers, 'o-', markersize=4)
        axes[0, 1].set_xlabel('Channel')
        axes[0, 1].set_ylabel('Total Power')
        axes[0, 1].set_title('Total Power by Channel')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average power spectrum across all channels
        if all_results:
            # Get frequency array from first channel
            frequencies = all_results[0]['frequencies']
            
            # Average PSD across channels
            all_psds = np.array([all_results[ch]['psd'] for ch in all_results.keys()])
            mean_psd = np.mean(all_psds, axis=0)
            std_psd = np.std(all_psds, axis=0)
            
            axes[1, 0].loglog(frequencies[1:], mean_psd[1:], label='Mean')
            axes[1, 0].fill_between(frequencies[1:], 
                                   (mean_psd - std_psd)[1:], 
                                   (mean_psd + std_psd)[1:], 
                                   alpha=0.3, label='±1 STD')
            axes[1, 0].set_xlabel('Frequency (Hz)')
            axes[1, 0].set_ylabel('Power Spectral Density')
            axes[1, 0].set_title('Average Power Spectrum Across Channels')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Frequency band analysis across channels
        if freq_bands and all_results:
            band_names = list(freq_bands.keys())
            n_bands = len(band_names)
            n_channels = len(all_results)
            
            band_matrix = np.zeros((n_channels, n_bands))
            for ch, results in all_results.items():
                for i, band in enumerate(band_names):
                    band_matrix[ch, i] = results['band_powers'].get(band, 0)
            
            im = axes[1, 1].imshow(band_matrix.T, aspect='auto', cmap='viridis')
            axes[1, 1].set_xlabel('Channel')
            axes[1, 1].set_ylabel('Frequency Band')
            axes[1, 1].set_yticks(range(n_bands))
            axes[1, 1].set_yticklabels(band_names)
            axes[1, 1].set_title('Band Power Across Channels')
            plt.colorbar(im, ax=axes[1, 1], label='Power')
        else:
            axes[1, 1].text(0.5, 0.5, 'No frequency band\nanalysis available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Frequency Bands')
        
        plt.tight_layout()
        plt.show()


def get_default_frequency_bands():
    """
    Get default frequency bands for analysis.
    
    Returns:
    --------
    dict : Default frequency bands
    """
    return {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100),
        'high_gamma': (100, 300),
        'ripples': (150, 250)
    }


def quick_analysis(file_path, duration_sec=10, channel=0, sampling_rate=30000):
    """
    Quick analysis function for immediate results.
    
    Parameters:
    -----------
    file_path : str
        Path to the .dat file
    duration_sec : float
        Duration to analyze in seconds
    channel : int
        Channel to analyze
    sampling_rate : float
        Sampling rate in Hz
        
    Returns:
    --------
    dict : Analysis results
    """
    analyzer = VoltageRecordingAnalyzer(sampling_rate=sampling_rate)
    
    # Load data
    data, time = analyzer.load_amplifier_data(
        file_path, 
        duration_sec=duration_sec
    )
    
    # Analyze single channel
    freq_bands = get_default_frequency_bands()
    results = analyzer.analyze_channel(channel, freq_bands=freq_bands)
    
    return results


if __name__ == "__main__":
    # Example usage
    file_path = "/mnt/c/Users/wanglab/Desktop/Club Like Endings/040425_1/amplifier.dat"
    
    # Quick analysis
    print("Running quick analysis...")
    results = quick_analysis(file_path, duration_sec=5, channel=0)
    print(f"Peak frequency: {results['peak_frequency']:.2f} Hz")
    print(f"Total power: {results['total_power']:.2e}")