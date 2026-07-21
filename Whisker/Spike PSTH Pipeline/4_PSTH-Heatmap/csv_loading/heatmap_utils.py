"""
PSTH Heatmap Utilities

This module provides utilities for creating PSTH heatmaps that combine multiple units.
Each unit is plotted on the Y-axis, and the X-axis shows firing rate over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.ndimage import uniform_filter1d
from typing import List, Tuple, Optional, Dict, Any

def load_data(spikes_file: str, intervals_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load spike and interval data from CSV files.
    
    Args:
        spikes_file: Path to spikes.csv file
        intervals_file: Path to pico_time_adjust.csv file
        
    Returns:
        Tuple of (spikes_df, intervals_df)
    """
    # Convert to absolute paths
    spikes_file = os.path.abspath(spikes_file)
    intervals_file = os.path.abspath(intervals_file)
    
    print(f"Loading spike data from: {spikes_file}")
    spikes_df = pd.read_csv(spikes_file)
    spikes_df.columns = spikes_df.columns.str.strip().str.lower()  # Clean column names
    
    print(f"Loading interval data from: {intervals_file}")
    intervals_df = pd.read_csv(intervals_file)
    intervals_df.columns = intervals_df.columns.str.strip()  # Clean column names
    
    print(f"Loaded {len(spikes_df)} spikes and {len(intervals_df)} intervals")
    print(f"Available units: {sorted(spikes_df['unit'].unique())}")
    
    durations = sorted(intervals_df['pico_Interval Duration'].unique())
    print(f"Available interval durations (first 5): {[f'{d*1000:.2f}ms' for d in durations[:5]]}")
    
    return spikes_df, intervals_df

def filter_intervals_by_duration(intervals_df: pd.DataFrame, target_duration_ms: float) -> pd.DataFrame:
    """
    Filter intervals by duration with tolerance.
    
    Args:
        intervals_df: DataFrame with interval data
        target_duration_ms: Target duration in milliseconds
        
    Returns:
        Filtered DataFrame
    """
    target_duration_s = target_duration_ms / 1000.0  # Convert ms to seconds
    tolerance_s = 0.001  # ±1ms tolerance in seconds
    
    filtered = intervals_df[
        (intervals_df['pico_Interval Duration'] >= target_duration_s - tolerance_s) &
        (intervals_df['pico_Interval Duration'] <= target_duration_s + tolerance_s)
    ].copy()
    
    print(f"Found {len(filtered)} intervals with duration {target_duration_ms}ms (±1ms)")
    return filtered

def compute_psth_for_unit(spikes_df: pd.DataFrame, intervals_df: pd.DataFrame, 
                         unit: int, bin_size_ms: float, pre_interval_ms: float, 
                         post_interval_ms: float, smooth_window: Optional[int] = None) -> np.ndarray:
    """
    Compute PSTH for a single unit.
    
    Args:
        spikes_df: DataFrame with spike data
        intervals_df: DataFrame with interval data
        unit: Unit number to analyze
        bin_size_ms: Bin size in milliseconds
        pre_interval_ms: Time before interval in ms
        post_interval_ms: Time after interval in ms
        smooth_window: Number of bins for smoothing (optional)
        
    Returns:
        PSTH array (firing rate in Hz)
    """
    # Filter spikes for this unit
    unit_spikes = spikes_df[spikes_df['unit'] == unit]['time'].values
    
    if len(unit_spikes) == 0:
        print(f"No spikes found for unit {unit}")
        return np.array([])
    
    # Convert times to seconds
    bin_size_s = bin_size_ms / 1000.0
    pre_interval_s = pre_interval_ms / 1000.0
    post_interval_s = post_interval_ms / 1000.0
    
    # Create time bins
    total_time_s = pre_interval_s + post_interval_s
    n_bins = int(total_time_s / bin_size_s)
    time_bins = np.linspace(-pre_interval_s, post_interval_s, n_bins + 1)
    
    # Collect spike counts for each interval
    spike_counts_all_trials = []
    
    for _, interval in intervals_df.iterrows():
        interval_start = interval['pico_Interval Start']
        
        # Get spikes relative to interval start
        trial_start = interval_start - pre_interval_s
        trial_end = interval_start + post_interval_s
        
        # Find spikes in this trial window
        trial_spikes = unit_spikes[(unit_spikes >= trial_start) & (unit_spikes < trial_end)]
        
        # Convert to relative times (relative to interval start)
        relative_spike_times = trial_spikes - interval_start
        
        # Bin the spikes
        spike_counts, _ = np.histogram(relative_spike_times, bins=time_bins)
        spike_counts_all_trials.append(spike_counts)
    
    if len(spike_counts_all_trials) == 0:
        print(f"No trials found for unit {unit}")
        return np.array([])
    
    # Convert to array and compute average
    spike_counts_array = np.array(spike_counts_all_trials)
    mean_spike_counts = np.mean(spike_counts_array, axis=0)
    
    # Convert to firing rate (Hz)
    firing_rate = mean_spike_counts / bin_size_s
    
    # Apply smoothing if requested
    if smooth_window and smooth_window > 1:
        firing_rate = uniform_filter1d(firing_rate, size=smooth_window)
    
    return firing_rate

def create_psth_heatmap(spikes_file: str, intervals_file: str, 
                       duration_ms: float, units: Optional[List[int]] = None,
                       bin_size_ms: float = 0.6, pre_interval_ms: float = 5, 
                       post_interval_ms: float = 10, smooth_window: Optional[int] = 5,
                       save_path: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray, List[int], np.ndarray]:
    """
    Create a PSTH heatmap for multiple units.
    
    Args:
        spikes_file: Path to spikes.csv file
        intervals_file: Path to pico_time_adjust.csv file
        duration_ms: Interval duration to analyze in milliseconds
        units: List of units to include (if None, use all available units)
        bin_size_ms: Bin size in milliseconds
        pre_interval_ms: Time before interval in ms
        post_interval_ms: Time after interval in ms
        smooth_window: Number of bins for smoothing (optional)
        save_path: Path to save the figure (optional)
        
    Returns:
        Tuple of (figure, heatmap_data, unit_labels, time_bins)
    """
    # Load data
    spikes_df, intervals_df = load_data(spikes_file, intervals_file)
    
    # Filter intervals by duration
    filtered_intervals = filter_intervals_by_duration(intervals_df, duration_ms)
    
    if len(filtered_intervals) == 0:
        print(f"No intervals found for duration {duration_ms}ms")
        return None, None, None, None
    
    # Get units to analyze
    if units is None:
        units = sorted(spikes_df['unit'].unique())
    
    print(f"Computing PSTH for {len(units)} units...")
    
    # Compute PSTH for each unit
    psth_data = []
    valid_units = []
    
    for unit in units:
        psth = compute_psth_for_unit(spikes_df, filtered_intervals, unit, 
                                   bin_size_ms, pre_interval_ms, post_interval_ms, 
                                   smooth_window)
        if len(psth) > 0:
            psth_data.append(psth)
            valid_units.append(unit)
    
    if len(psth_data) == 0:
        print("No valid PSTH data found")
        return None, None, None, None
    
    # Convert to array for heatmap
    heatmap_data = np.array(psth_data)
    
    # Create time axis
    bin_size_s = bin_size_ms / 1000.0
    pre_interval_s = pre_interval_ms / 1000.0
    post_interval_s = post_interval_ms / 1000.0
    total_time_s = pre_interval_s + post_interval_s
    n_bins = int(total_time_s / bin_size_s)
    time_bins = np.linspace(-pre_interval_ms, post_interval_ms, n_bins)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, aspect='auto', origin='lower', 
                   extent=[-pre_interval_ms, post_interval_ms, 0, len(valid_units)],
                   cmap='hot', interpolation='nearest')
    
    # Customize the plot
    ax.set_xlabel('Time relative to interval start (ms)', fontsize=12)
    ax.set_ylabel('Unit', fontsize=12)
    ax.set_title(f'PSTH Heatmap - Duration: {duration_ms}ms\n'
                f'Bin size: {bin_size_ms}ms, Smoothing: {smooth_window} bins', 
                fontsize=14, fontweight='bold')
    
    # Add vertical line at interval start
    ax.axvline(x=0, color='white', linestyle='--', alpha=0.8, linewidth=2)
    
    # Set Y-axis ticks to show unit numbers centered on each unit's row
    tick_positions = [i + 0.5 for i in range(len(valid_units))]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f'Unit {u}' for u in valid_units])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Firing Rate (Hz)', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved heatmap to: {save_path}")
    
    return fig, heatmap_data, valid_units, time_bins

def create_multiple_duration_heatmaps(spikes_file: str, intervals_file: str,
                                    durations_ms: List[float], units: Optional[List[int]] = None,
                                    bin_size_ms: float = 0.6, pre_interval_ms: float = 5,
                                    post_interval_ms: float = 10, smooth_window: Optional[int] = 5,
                                    save_dir: Optional[str] = None) -> Dict[float, Tuple]:
    """
    Create PSTH heatmaps for multiple interval durations.
    
    Args:
        spikes_file: Path to spikes.csv file
        intervals_file: Path to pico_time_adjust.csv file
        durations_ms: List of interval durations to analyze in milliseconds
        units: List of units to include (if None, use all available units)
        bin_size_ms: Bin size in milliseconds
        pre_interval_ms: Time before interval in ms
        post_interval_ms: Time after interval in ms
        smooth_window: Number of bins for smoothing (optional)
        save_dir: Directory to save figures (optional)
        
    Returns:
        Dictionary mapping duration to (figure, heatmap_data, unit_labels, time_bins)
    """
    results = {}
    
    for duration in durations_ms:
        print(f"\n--- Processing duration: {duration}ms ---")
        
        # Create save path if directory provided
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'psth_heatmap_{duration}ms.png')
        
        # Create heatmap
        result = create_psth_heatmap(
            spikes_file, intervals_file, duration, units,
            bin_size_ms, pre_interval_ms, post_interval_ms, 
            smooth_window, save_path
        )
        
        results[duration] = result
    
    return results

def plot_summary_statistics(results: Dict[float, Tuple], save_dir: Optional[str] = None):
    """
    Plot summary statistics across different durations.
    
    Args:
        results: Dictionary from create_multiple_duration_heatmaps
        save_dir: Directory to save figure (optional)
    """
    # Extract data for summary
    durations = []
    max_firing_rates = []
    mean_firing_rates = []
    
    for duration, (fig, heatmap_data, units, time_bins) in results.items():
        if heatmap_data is not None:
            durations.append(duration)
            max_firing_rates.append(np.max(heatmap_data))
            mean_firing_rates.append(np.mean(heatmap_data))
    
    if not durations:
        print("No valid data for summary statistics")
        return
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Max firing rates
    ax1.bar(durations, max_firing_rates, alpha=0.7, color='red')
    ax1.set_xlabel('Interval Duration (ms)')
    ax1.set_ylabel('Maximum Firing Rate (Hz)')
    ax1.set_title('Maximum Firing Rate by Duration')
    ax1.grid(True, alpha=0.3)
    
    # Mean firing rates
    ax2.bar(durations, mean_firing_rates, alpha=0.7, color='blue')
    ax2.set_xlabel('Interval Duration (ms)')
    ax2.set_ylabel('Mean Firing Rate (Hz)')
    ax2.set_title('Mean Firing Rate by Duration')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'psth_heatmap_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary plot to: {save_path}")
    
    return fig