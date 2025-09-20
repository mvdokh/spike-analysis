#!/usr/bin/env python3
"""
Spike Pattern Analysis Utilities

This module provides functions for analyzing neural spike patterns including:
- Burst detection
- Inter-spike interval analysis
- Firing rate calculations
- Spike train correlations
- Pattern classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats, signal
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class SpikePatternAnalyzer:
    """
    Main class for spike pattern analysis
    """
    
    def __init__(self, spike_data: pd.DataFrame, electrode_config: pd.DataFrame = None):
        """
        Initialize the analyzer with spike data
        
        Parameters:
        -----------
        spike_data : pd.DataFrame
            DataFrame with columns ['time', 'unit', 'electrode']
        electrode_config : pd.DataFrame, optional
            DataFrame with electrode configuration
        """
        self.spike_data = spike_data.copy()
        self.electrode_config = electrode_config
        self.unit_patterns = {}
        
    def get_unit_spike_times(self, unit_id: int) -> np.ndarray:
        """Get spike times for a specific unit"""
        unit_spikes = self.spike_data[self.spike_data['unit'] == unit_id]
        return np.sort(unit_spikes['time'].values)
    
    def calculate_isi(self, unit_id: int) -> np.ndarray:
        """
        Calculate inter-spike intervals for a unit
        
        Parameters:
        -----------
        unit_id : int
            Unit identifier
            
        Returns:
        --------
        np.ndarray
            Array of inter-spike intervals in seconds
        """
        spike_times = self.get_unit_spike_times(unit_id)
        if len(spike_times) < 2:
            return np.array([])
        return np.diff(spike_times)
    
    def calculate_firing_rate(self, unit_id: int, window_size: float = 1.0, 
                            step_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate instantaneous firing rate using sliding window
        
        Parameters:
        -----------
        unit_id : int
            Unit identifier
        window_size : float
            Window size in seconds
        step_size : float
            Step size in seconds
            
        Returns:
        --------
        Tuple[np.ndarray, np.ndarray]
            Time bins and corresponding firing rates
        """
        spike_times = self.get_unit_spike_times(unit_id)
        if len(spike_times) == 0:
            return np.array([]), np.array([])
        
        start_time = spike_times[0]
        end_time = spike_times[-1]
        time_bins = np.arange(start_time, end_time, step_size)
        firing_rates = []
        
        for t in time_bins:
            window_start = t - window_size / 2
            window_end = t + window_size / 2
            spikes_in_window = np.sum((spike_times >= window_start) & 
                                    (spike_times <= window_end))
            firing_rates.append(spikes_in_window / window_size)
        
        return time_bins, np.array(firing_rates)
    
    def detect_bursts(self, unit_id: int, max_isi: float = 0.01, 
                     min_spikes: int = 3) -> List[Dict]:
        """
        Detect burst patterns in spike trains
        
        Parameters:
        -----------
        unit_id : int
            Unit identifier
        max_isi : float
            Maximum inter-spike interval for burst (seconds)
        min_spikes : int
            Minimum number of spikes in a burst
            
        Returns:
        --------
        List[Dict]
            List of burst dictionaries with start_time, end_time, duration, spike_count
        """
        spike_times = self.get_unit_spike_times(unit_id)
        if len(spike_times) < min_spikes:
            return []
        
        isis = self.calculate_isi(unit_id)
        bursts = []
        
        # Find burst boundaries
        in_burst = False
        burst_start_idx = 0
        
        for i, isi in enumerate(isis):
            if isi <= max_isi and not in_burst:
                # Start of burst
                in_burst = True
                burst_start_idx = i
            elif isi > max_isi and in_burst:
                # End of burst
                in_burst = False
                burst_end_idx = i
                
                # Check if burst meets minimum spike requirement
                burst_spikes = burst_end_idx - burst_start_idx + 1
                if burst_spikes >= min_spikes:
                    burst_info = {
                        'start_time': spike_times[burst_start_idx],
                        'end_time': spike_times[burst_end_idx],
                        'duration': spike_times[burst_end_idx] - spike_times[burst_start_idx],
                        'spike_count': burst_spikes,
                        'start_idx': burst_start_idx,
                        'end_idx': burst_end_idx
                    }
                    bursts.append(burst_info)
        
        # Handle case where recording ends during a burst
        if in_burst:
            burst_end_idx = len(spike_times) - 1
            burst_spikes = burst_end_idx - burst_start_idx + 1
            if burst_spikes >= min_spikes:
                burst_info = {
                    'start_time': spike_times[burst_start_idx],
                    'end_time': spike_times[burst_end_idx],
                    'duration': spike_times[burst_end_idx] - spike_times[burst_start_idx],
                    'spike_count': burst_spikes,
                    'start_idx': burst_start_idx,
                    'end_idx': burst_end_idx
                }
                bursts.append(burst_info)
        
        return bursts
    
    def analyze_rhythmicity(self, unit_id: int, max_freq: float = 100.0) -> Dict:
        """
        Analyze rhythmic patterns using autocorrelation and spectral analysis
        
        Parameters:
        -----------
        unit_id : int
            Unit identifier
        max_freq : float
            Maximum frequency to analyze (Hz)
            
        Returns:
        --------
        Dict
            Dictionary with rhythmicity metrics
        """
        spike_times = self.get_unit_spike_times(unit_id)
        if len(spike_times) < 10:
            return {'dominant_frequency': None, 'power_spectrum': None, 'autocorr': None}
        
        # Create binary spike train
        dt = 0.001  # 1ms resolution
        total_time = spike_times[-1] - spike_times[0]
        n_bins = int(total_time / dt)
        spike_train = np.zeros(n_bins)
        
        for t in spike_times:
            bin_idx = int((t - spike_times[0]) / dt)
            if 0 <= bin_idx < n_bins:
                spike_train[bin_idx] = 1
        
        # Autocorrelation
        autocorr = np.correlate(spike_train, spike_train, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Power spectrum
        freqs, power = signal.welch(spike_train, fs=1/dt, nperseg=min(2048, len(spike_train)//4))
        freqs = freqs[freqs <= max_freq]
        power = power[:len(freqs)]
        
        # Find dominant frequency
        dominant_freq_idx = np.argmax(power[1:]) + 1  # Exclude DC component
        dominant_frequency = freqs[dominant_freq_idx] if len(freqs) > 1 else None
        
        return {
            'dominant_frequency': dominant_frequency,
            'power_spectrum': {'frequencies': freqs, 'power': power},
            'autocorr': {'lags': np.arange(len(autocorr)) * dt, 'values': autocorr[:1000]}
        }
    
    def calculate_spike_train_correlation(self, unit1: int, unit2: int, 
                                        max_lag: float = 0.1, bin_size: float = 0.001) -> Dict:
        """
        Calculate cross-correlation between two spike trains
        
        Parameters:
        -----------
        unit1, unit2 : int
            Unit identifiers
        max_lag : float
            Maximum lag to compute (seconds)
        bin_size : float
            Bin size for correlation (seconds)
            
        Returns:
        --------
        Dict
            Cross-correlation results
        """
        spikes1 = self.get_unit_spike_times(unit1)
        spikes2 = self.get_unit_spike_times(unit2)
        
        if len(spikes1) < 2 or len(spikes2) < 2:
            return {'correlation': None, 'lags': None, 'peak_lag': None}
        
        # Find common time range
        start_time = max(spikes1[0], spikes2[0])
        end_time = min(spikes1[-1], spikes2[-1])
        
        if end_time <= start_time:
            return {'correlation': None, 'lags': None, 'peak_lag': None}
        
        # Create binary spike trains
        n_bins = int((end_time - start_time) / bin_size)
        train1 = np.zeros(n_bins)
        train2 = np.zeros(n_bins)
        
        for t in spikes1:
            if start_time <= t <= end_time:
                bin_idx = int((t - start_time) / bin_size)
                if 0 <= bin_idx < n_bins:
                    train1[bin_idx] = 1
        
        for t in spikes2:
            if start_time <= t <= end_time:
                bin_idx = int((t - start_time) / bin_size)
                if 0 <= bin_idx < n_bins:
                    train2[bin_idx] = 1
        
        # Cross-correlation
        correlation = np.correlate(train1, train2, mode='full')
        n_center = len(correlation) // 2
        max_lag_bins = int(max_lag / bin_size)
        
        # Extract relevant lags
        start_idx = max(0, n_center - max_lag_bins)
        end_idx = min(len(correlation), n_center + max_lag_bins + 1)
        
        correlation = correlation[start_idx:end_idx]
        lags = (np.arange(len(correlation)) - (len(correlation) // 2)) * bin_size
        
        # Find peak
        peak_idx = np.argmax(correlation)
        peak_lag = lags[peak_idx]
        
        return {
            'correlation': correlation,
            'lags': lags,
            'peak_lag': peak_lag,
            'peak_correlation': correlation[peak_idx]
        }
    
    def classify_firing_pattern(self, unit_id: int) -> Dict:
        """
        Classify the firing pattern of a unit
        
        Parameters:
        -----------
        unit_id : int
            Unit identifier
            
        Returns:
        --------
        Dict
            Classification results
        """
        spike_times = self.get_unit_spike_times(unit_id)
        if len(spike_times) < 10:
            return {'pattern_type': 'insufficient_data', 'confidence': 0.0}
        
        # Calculate features
        isis = self.calculate_isi(unit_id)
        mean_isi = np.mean(isis)
        cv_isi = np.std(isis) / mean_isi if mean_isi > 0 else float('inf')
        mean_rate = len(spike_times) / (spike_times[-1] - spike_times[0])
        
        # Detect bursts
        bursts = self.detect_bursts(unit_id)
        burst_ratio = len(bursts) / len(spike_times) if len(spike_times) > 0 else 0
        
        # Analyze rhythmicity
        rhythm_analysis = self.analyze_rhythmicity(unit_id)
        has_rhythm = (rhythm_analysis['dominant_frequency'] is not None and 
                     rhythm_analysis['dominant_frequency'] > 1.0)
        
        # Simple classification rules
        if burst_ratio > 0.1:
            pattern_type = 'bursting'
            confidence = min(1.0, burst_ratio * 2)
        elif has_rhythm and rhythm_analysis['dominant_frequency'] > 10:
            pattern_type = 'rhythmic'
            confidence = 0.8
        elif cv_isi < 0.5:
            pattern_type = 'regular'
            confidence = max(0.5, 1.0 - cv_isi)
        elif cv_isi > 1.5:
            pattern_type = 'irregular'
            confidence = min(1.0, cv_isi / 2)
        else:
            pattern_type = 'tonic'
            confidence = 0.6
        
        return {
            'pattern_type': pattern_type,
            'confidence': confidence,
            'features': {
                'mean_isi': mean_isi,
                'cv_isi': cv_isi,
                'mean_rate': mean_rate,
                'burst_ratio': burst_ratio,
                'dominant_frequency': rhythm_analysis['dominant_frequency']
            }
        }
    
    def run_pattern_analysis(self, unit_id: int, params: Dict = None) -> Dict:
        """
        Run comprehensive pattern analysis for a unit
        
        Parameters:
        -----------
        unit_id : int
            Unit identifier
        params : Dict, optional
            Analysis parameters
            
        Returns:
        --------
        Dict
            Complete analysis results
        """
        # Default parameters
        default_params = {
            'burst_max_isi': 0.01,
            'burst_min_spikes': 3,
            'firing_rate_window': 1.0,
            'firing_rate_step': 0.1,
            'rhythm_max_freq': 100.0,
            'correlation_max_lag': 0.1
        }
        
        if params is None:
            params = default_params
        else:
            # Merge with defaults
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        
        spike_times = self.get_unit_spike_times(unit_id)
        
        if len(spike_times) < 2:
            return {
                'unit_id': unit_id,
                'spike_count': len(spike_times),
                'error': 'Insufficient spikes for analysis'
            }
        
        # Run all analyses
        results = {
            'unit_id': unit_id,
            'spike_count': len(spike_times),
            'recording_duration': spike_times[-1] - spike_times[0],
            'spike_times': spike_times,
            'parameters': params
        }
        
        # ISI analysis
        isis = self.calculate_isi(unit_id)
        results['isi_analysis'] = {
            'isis': isis,
            'mean': np.mean(isis),
            'std': np.std(isis),
            'cv': np.std(isis) / np.mean(isis) if np.mean(isis) > 0 else float('inf'),
            'median': np.median(isis),
            'percentiles': np.percentile(isis, [25, 75])
        }
        
        # Firing rate analysis
        time_bins, firing_rates = self.calculate_firing_rate(
            unit_id, params['firing_rate_window'], params['firing_rate_step']
        )
        results['firing_rate_analysis'] = {
            'time_bins': time_bins,
            'firing_rates': firing_rates,
            'mean_rate': np.mean(firing_rates),
            'max_rate': np.max(firing_rates) if len(firing_rates) > 0 else 0,
            'min_rate': np.min(firing_rates) if len(firing_rates) > 0 else 0
        }
        
        # Burst analysis
        bursts = self.detect_bursts(unit_id, params['burst_max_isi'], params['burst_min_spikes'])
        results['burst_analysis'] = {
            'bursts': bursts,
            'burst_count': len(bursts),
            'total_spikes_in_bursts': sum([b['spike_count'] for b in bursts]),
            'burst_ratio': len(bursts) / len(spike_times) if len(spike_times) > 0 else 0
        }
        
        # Rhythmicity analysis
        results['rhythm_analysis'] = self.analyze_rhythmicity(unit_id, params['rhythm_max_freq'])
        
        # Pattern classification
        results['pattern_classification'] = self.classify_firing_pattern(unit_id)
        
        # Store results
        self.unit_patterns[unit_id] = results
        
        return results
    
    def print_analysis_results(self, results: Dict, unit_id: int = None):
        """
        Print formatted analysis results
        
        Parameters:
        -----------
        results : Dict
            Analysis results from run_pattern_analysis
        unit_id : int, optional
            Unit ID (will use from results if not provided)
        """
        if unit_id is None:
            unit_id = results.get('unit_id', 'Unknown')
            
        if 'error' in results:
            print(f"Error analyzing Unit {unit_id}: {results['error']}")
            return
            
        print("\n" + "="*60)
        print(f"PATTERN ANALYSIS RESULTS - UNIT {unit_id}")
        print("="*60)
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"  Spike Count: {results['spike_count']:,}")
        print(f"  Recording Duration: {results['recording_duration']:.2f} seconds")
        print(f"  Mean Firing Rate: {results['firing_rate_analysis']['mean_rate']:.2f} Hz")
        
        # ISI analysis
        isi = results['isi_analysis']
        print(f"\nInter-Spike Interval Analysis:")
        print(f"  Mean ISI: {isi['mean']:.4f} seconds")
        print(f"  CV of ISI: {isi['cv']:.3f}")
        print(f"  Median ISI: {isi['median']:.4f} seconds")
        
        # Burst analysis
        burst = results['burst_analysis']
        print(f"\nBurst Analysis:")
        print(f"  Number of Bursts: {burst['burst_count']}")
        print(f"  Spikes in Bursts: {burst['total_spikes_in_bursts']}")
        print(f"  Burst Ratio: {burst['burst_ratio']:.3f}")
        
        if burst['bursts']:
            print(f"  Average Burst Duration: {np.mean([b['duration'] for b in burst['bursts']]):.3f} seconds")
            print(f"  Average Spikes per Burst: {np.mean([b['spike_count'] for b in burst['bursts']]):.1f}")
        
        # Rhythmicity
        rhythm = results['rhythm_analysis']
        print(f"\nRhythmicity Analysis:")
        if rhythm['dominant_frequency']:
            print(f"  Dominant Frequency: {rhythm['dominant_frequency']:.2f} Hz")
        else:
            print(f"  No dominant frequency detected")
        
        # Pattern classification
        pattern = results['pattern_classification']
        print(f"\nPattern Classification:")
        print(f"  Pattern Type: {pattern['pattern_type'].upper()}")
        print(f"  Confidence: {pattern['confidence']:.2%}")
        
        print(f"\nAnalysis completed successfully!")
    
    def print_batch_summary(self, batch_results: Dict):
        """
        Print summary of batch analysis results
        
        Parameters:
        -----------
        batch_results : Dict
            Dictionary of unit_id -> results from batch analysis
        """
        valid_results = {k: v for k, v in batch_results.items() if 'error' not in v}
        
        if not valid_results:
            print("No valid results to summarize.")
            return
            
        print(f"\n" + "="*80)
        print(f"BATCH ANALYSIS SUMMARY - {len(valid_results)} UNITS")
        print("="*80)
        
        # Create summary table
        print(f"\n{'Unit':<6} {'Spikes':<8} {'Rate(Hz)':<10} {'CV ISI':<8} {'Bursts':<8} {'Pattern':<12} {'Conf':<6}")
        print("-" * 70)
        
        for unit_id, results in valid_results.items():
            spike_count = results['spike_count']
            rate = results['firing_rate_analysis']['mean_rate']
            cv_isi = results['isi_analysis']['cv']
            bursts = results['burst_analysis']['burst_count']
            pattern = results['pattern_classification']['pattern_type']
            confidence = results['pattern_classification']['confidence']
            
            print(f"{unit_id:<6} {spike_count:<8,} {rate:<10.2f} {cv_isi:<8.3f} {bursts:<8} {pattern:<12} {confidence:<6.2f}")
        
        # Pattern type summary
        pattern_counts = {}
        for results in valid_results.values():
            pattern = results['pattern_classification']['pattern_type']
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
            
        print(f"\nPattern Type Distribution:")
        for pattern, count in pattern_counts.items():
            percentage = (count / len(valid_results)) * 100
            print(f"  {pattern.title()}: {count} units ({percentage:.1f}%)")
        
        # Overall statistics
        all_rates = [r['firing_rate_analysis']['mean_rate'] for r in valid_results.values()]
        all_cv_isi = [r['isi_analysis']['cv'] for r in valid_results.values()]
        all_bursts = [r['burst_analysis']['burst_count'] for r in valid_results.values()]
        
        print(f"\nOverall Statistics:")
        print(f"  Mean Firing Rate: {np.mean(all_rates):.2f} ± {np.std(all_rates):.2f} Hz")
        print(f"  Mean CV ISI: {np.mean(all_cv_isi):.3f} ± {np.std(all_cv_isi):.3f}")
        print(f"  Total Bursts: {sum(all_bursts)}")
        
        print(f"\nBatch analysis completed for {len(valid_results)} units!")

def load_spike_data(spike_file: str) -> pd.DataFrame:
    """Load spike data from CSV file"""
    return pd.read_csv(spike_file, header=None, names=['time', 'unit', 'electrode'])

def load_electrode_config(config_file: str) -> pd.DataFrame:
    """Load electrode configuration from .cfg file"""
    electrodes = []
    with open(config_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines[1:]:  # Skip header
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4:
                electrode_num = int(parts[0])
                x_pos = float(parts[2])
                depth = float(parts[3])
                electrodes.append({
                    'electrode': electrode_num,
                    'x_pos': x_pos,
                    'depth': depth
                })
    
    return pd.DataFrame(electrodes)

def plot_unit_analysis(analyzer: SpikePatternAnalyzer, unit_id: int, 
                      results: Dict = None, figsize: Tuple = (15, 12)) -> plt.Figure:
    """
    Create comprehensive visualization of unit analysis
    
    Parameters:
    -----------
    analyzer : SpikePatternAnalyzer
        Analyzer instance
    unit_id : int
        Unit identifier
    results : Dict, optional
        Pre-computed results
    figsize : Tuple
        Figure size
        
    Returns:
    --------
    plt.Figure
        Matplotlib figure
    """
    if results is None:
        results = analyzer.run_pattern_analysis(unit_id)
    
    if 'error' in results:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Unit {unit_id}: {results['error']}", 
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Spike raster
    ax1 = fig.add_subplot(gs[0, :])
    spike_times = results['spike_times']
    ax1.eventplot([spike_times], lineoffsets=0, linelengths=0.8, colors='black')
    ax1.set_xlim(spike_times[0], spike_times[-1])
    ax1.set_title(f'Unit {unit_id} - Spike Raster', fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_yticks([])
    
    # Add burst markers
    if results['burst_analysis']['bursts']:
        for burst in results['burst_analysis']['bursts']:
            ax1.axvspan(burst['start_time'], burst['end_time'], 
                       alpha=0.3, color='red', label='Burst' if burst == results['burst_analysis']['bursts'][0] else "")
        ax1.legend()
    
    # ISI histogram
    ax2 = fig.add_subplot(gs[1, 0])
    isis = results['isi_analysis']['isis']
    if len(isis) > 0:
        ax2.hist(isis, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(results['isi_analysis']['mean'], color='red', 
                   linestyle='--', label=f"Mean: {results['isi_analysis']['mean']:.4f}s")
        ax2.legend()
    ax2.set_xlabel('Inter-Spike Interval (s)')
    ax2.set_ylabel('Count')
    ax2.set_title('ISI Distribution')
    
    # Firing rate over time
    ax3 = fig.add_subplot(gs[1, 1])
    if len(results['firing_rate_analysis']['time_bins']) > 0:
        ax3.plot(results['firing_rate_analysis']['time_bins'], 
                results['firing_rate_analysis']['firing_rates'], 'b-', alpha=0.7)
        ax3.axhline(results['firing_rate_analysis']['mean_rate'], 
                   color='red', linestyle='--', 
                   label=f"Mean: {results['firing_rate_analysis']['mean_rate']:.2f} Hz")
        ax3.legend()
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Firing Rate (Hz)')
    ax3.set_title('Instantaneous Firing Rate')
    
    # Autocorrelation
    ax4 = fig.add_subplot(gs[1, 2])
    if results['rhythm_analysis']['autocorr'] is not None:
        autocorr = results['rhythm_analysis']['autocorr']
        ax4.plot(autocorr['lags'], autocorr['values'], 'g-', alpha=0.7)
        ax4.set_xlabel('Lag (s)')
        ax4.set_ylabel('Autocorrelation')
        ax4.set_title('Autocorrelation')
        ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    # Power spectrum
    ax5 = fig.add_subplot(gs[2, 0])
    if results['rhythm_analysis']['power_spectrum'] is not None:
        ps = results['rhythm_analysis']['power_spectrum']
        ax5.semilogy(ps['frequencies'], ps['power'], 'purple', alpha=0.7)
        if results['rhythm_analysis']['dominant_frequency']:
            ax5.axvline(results['rhythm_analysis']['dominant_frequency'], 
                       color='red', linestyle='--', 
                       label=f"Peak: {results['rhythm_analysis']['dominant_frequency']:.1f} Hz")
            ax5.legend()
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Power')
        ax5.set_title('Power Spectrum')
    
    # Statistics summary
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.axis('off')
    
    stats_text = f"""
ANALYSIS SUMMARY - UNIT {unit_id}
{'='*40}
Spike Statistics:
  • Total Spikes: {results['spike_count']:,}
  • Recording Duration: {results['recording_duration']:.2f} s
  • Mean Firing Rate: {results['firing_rate_analysis']['mean_rate']:.2f} Hz
  
ISI Statistics:
  • Mean ISI: {results['isi_analysis']['mean']:.4f} s
  • CV of ISI: {results['isi_analysis']['cv']:.3f}
  • Median ISI: {results['isi_analysis']['median']:.4f} s
  
Burst Analysis:
  • Burst Count: {results['burst_analysis']['burst_count']}
  • Spikes in Bursts: {results['burst_analysis']['total_spikes_in_bursts']}
  • Burst Ratio: {results['burst_analysis']['burst_ratio']:.3f}
  
Pattern Classification:
  • Type: {results['pattern_classification']['pattern_type'].upper()}
  • Confidence: {results['pattern_classification']['confidence']:.2f}
    """
    
    if results['rhythm_analysis']['dominant_frequency']:
        stats_text += f"  • Dominant Frequency: {results['rhythm_analysis']['dominant_frequency']:.1f} Hz"
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.suptitle(f'Comprehensive Pattern Analysis - Unit {unit_id}', 
                fontsize=16, fontweight='bold')
    
    return fig

def print_data_summary(spike_data: pd.DataFrame, electrode_config: pd.DataFrame = None):
    """
    Print summary of loaded data
    
    Parameters:
    -----------
    spike_data : pd.DataFrame
        Loaded spike data
    electrode_config : pd.DataFrame, optional
        Electrode configuration data
    """
    unique_units = sorted(spike_data['unit'].unique())
    unique_electrodes = sorted(spike_data['electrode'].unique())
    recording_duration = spike_data['time'].max() - spike_data['time'].min()

    print(f"Dataset Summary:")
    print(f"  • Recording duration: {recording_duration:.2f} seconds")
    print(f"  • Number of units: {len(unique_units)}")
    print(f"  • Number of electrodes: {len(unique_electrodes)}")
    print(f"  • Total spikes: {len(spike_data):,}")
    
    if electrode_config is not None:
        print(f"  • Electrode configuration loaded: {len(electrode_config)} electrodes")
