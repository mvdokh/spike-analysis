"""
Min-Max Normalized PSTH Heatmap Utilities (Digital Input Version)

This module provides utilities for creating Min-Max normalized PSTH heatmaps using
digital input (.dat) files instead of CSV files for interval data.

Min-Max normalization: N = (F - Min) / (Max - Min)
where:
- N = normalized value (between 0 and 1)
- F = firing rate of an individual bin
- Min = minimum firing rate of a unit (considers all bins)
- Max = maximum firing rate of a unit (considers all bins)

This normalization scales each unit's firing rates to a 0-1 range, allowing for better comparison across units with different baseline firing rates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from scipy.ndimage import uniform_filter1d
from typing import List, Tuple, Optional, Dict, Any

# Add paths to import digital input loading modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(parent_dir)  # Add Full_Pipeline to path

# Import digital input loading utilities
from binary_data import get_digital
from ttls import get_ttl_timestamps_16bit

def load_digitalin_data(digitalin_filepath: str, sampling_rate: int = 30000) -> np.ndarray:
    """
    Load digital input data from .dat file
    
    Args:
        digitalin_filepath: Path to the digitalin.dat file
        sampling_rate: Sampling rate in Hz (default: 30000)
        
    Returns:
        Array of 16-bit digital input values
    """
    print(f"Loading digital input data from: {digitalin_filepath}")
    
    # Load binary data using the existing utility
    digital_inputs = get_digital(
        filepath=digitalin_filepath,
        header_offset_in_bytes=0,  # Assume no header for .dat files
        single_sample_size_in_bytes=2  # 16-bit data
    )
    
    print(f"Loaded {len(digital_inputs)} samples at {sampling_rate} Hz")
    print(f"Recording duration: {len(digital_inputs) / sampling_rate:.2f} seconds")
    
    return digital_inputs


def extract_channel_ttl_events(digital_inputs: np.ndarray, channel: int, sampling_rate: int = 30000, 
                              isTransitionLowToHigh: Optional[bool] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract TTL events from a specific channel
    
    Args:
        digital_inputs: 16-bit digital input data
        channel: Channel number (0-15)
        sampling_rate: Sampling rate in Hz
        isTransitionLowToHigh: TTL transition direction (None for auto-detection)
        
    Returns:
        Tuple of (onsets, offsets) timestamps in seconds
    """
    print(f"Extracting TTL events from channel {channel}")
    
    # Get TTL timestamps in samples
    ttl_onsets, ttl_offsets = get_ttl_timestamps_16bit(
        digital_inputs=digital_inputs,
        ttl_index=channel,
        isTransitionLowToHigh=isTransitionLowToHigh
    )
    
    # Convert sample indices to time in seconds
    onsets_seconds = ttl_onsets / sampling_rate
    offsets_seconds = ttl_offsets / sampling_rate
    
    print(f"Found {len(onsets_seconds)} TTL events on channel {channel}")
    
    if len(onsets_seconds) > 0:
        durations = offsets_seconds - onsets_seconds
        print(f"Event durations: {np.mean(durations)*1000:.2f} ± {np.std(durations)*1000:.2f} ms")
        
        # Show some examples
        print("First 5 events:")
        for i in range(min(5, len(onsets_seconds))):
            print(f"  Event {i+1}: {onsets_seconds[i]:.6f}s - {offsets_seconds[i]:.6f}s ({durations[i]*1000:.2f}ms)")
    
    return onsets_seconds, offsets_seconds


def create_intervals_dataframe(pico_onsets: np.ndarray, pico_offsets: np.ndarray) -> pd.DataFrame:
    """
    Create a DataFrame in the same format as pico_time_adjust.csv
    
    Args:
        pico_onsets: Pico interval start times in seconds
        pico_offsets: Pico interval end times in seconds  
        
    Returns:
        DataFrame with columns matching pico_time_adjust.csv format
    """
    print(f"Creating intervals DataFrame with {len(pico_onsets)} intervals")
    
    # Calculate durations
    durations = pico_offsets - pico_onsets
    
    # Create DataFrame with the expected column names
    intervals_df = pd.DataFrame({
        'pico_Interval Start': pico_onsets,
        'pico_Interval End': pico_offsets,
        'pico_Interval Duration': durations
    })
    
    # Print duration statistics
    print(f"Duration statistics:")
    print(f"  Mean: {np.mean(durations)*1000:.2f} ms")
    print(f"  Std:  {np.std(durations)*1000:.2f} ms")
    print(f"  Min:  {np.min(durations)*1000:.2f} ms")
    print(f"  Max:  {np.max(durations)*1000:.2f} ms")
    
    # Show unique durations (rounded to nearest ms for grouping)
    unique_durations_ms = np.unique(np.round(durations * 1000))
    print(f"Unique durations (ms): {unique_durations_ms[:10]}")  # Show first 10
    
    if len(unique_durations_ms) > 10:
        print(f"... and {len(unique_durations_ms) - 10} more")
    
    return intervals_df


def load_data(spikes_file: str, digitalin_file: str, sampling_rate: int = 30000, 
              pico_channel: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load spike data from CSV and interval data from digitalin.dat file.
    
    Args:
        spikes_file: Path to spikes.csv file
        digitalin_file: Path to digitalin.dat file
        sampling_rate: Sampling rate in Hz (default: 30000)
        pico_channel: Channel for pico intervals (default: 0)
        
    Returns:
        Tuple of (spikes_df, intervals_df)
    """
    # Convert to absolute paths
    spikes_file = os.path.abspath(spikes_file)
    digitalin_file = os.path.abspath(digitalin_file)
    
    print(f"Loading spike data from: {spikes_file}")
    spikes_df = pd.read_csv(spikes_file)
    spikes_df.columns = spikes_df.columns.str.strip().str.lower()  # Clean column names
    
    print(f"Loading interval data from digitalin.dat: {digitalin_file}")
    
    # Load digital input data
    digital_inputs = load_digitalin_data(digitalin_file, sampling_rate)
    
    # Extract pico interval events from specified channel
    pico_onsets, pico_offsets = extract_channel_ttl_events(
        digital_inputs=digital_inputs,
        channel=pico_channel,
        sampling_rate=sampling_rate
    )
    
    # Create intervals DataFrame
    intervals_df = create_intervals_dataframe(pico_onsets, pico_offsets)
    
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
                         post_interval_ms: float, duration_ms: float, smooth_window: Optional[int] = None) -> np.ndarray:
    """
    Compute PSTH for a single unit.
    
    Args:
        spikes_df: DataFrame with spike data
        intervals_df: DataFrame with interval data
        unit: Unit number to analyze
        bin_size_ms: Bin size in milliseconds
        pre_interval_ms: Time before interval in ms
        post_interval_ms: Time after interval END in ms
        duration_ms: Duration of the interval in ms
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
    duration_s = duration_ms / 1000.0
    
    # Create time bins - now total time includes pre + duration + post
    total_time_s = pre_interval_s + duration_s + post_interval_s
    n_bins = int(total_time_s / bin_size_s)
    time_bins = np.linspace(-pre_interval_s, duration_s + post_interval_s, n_bins + 1)
    
    # Collect spike counts for each interval
    spike_counts_all_trials = []
    
    for _, interval in intervals_df.iterrows():
        interval_start = interval['pico_Interval Start']
        
        # Get spikes relative to interval start - now extends to post_interval_ms after interval END
        trial_start = interval_start - pre_interval_s
        trial_end = interval_start + duration_s + post_interval_s
        
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

def min_max_normalize_psth(psth_data: np.ndarray) -> np.ndarray:
    """
    Apply Min-Max normalization to PSTH data.
    
    Min-Max formula: N = (F - Min) / (Max - Min)
    where:
    - N = normalized value (between 0 and 1)
    - F = firing rate of an individual bin
    - Min = minimum firing rate of a unit (considers all bins)
    - Max = maximum firing rate of a unit (considers all bins)
    
    Args:
        psth_data: 2D array with shape (n_units, n_bins)
        
    Returns:
        Min-Max normalized array with same shape (values between 0 and 1)
    """
    normalized_data = np.zeros_like(psth_data)
    
    for i, unit_psth in enumerate(psth_data):
        min_firing_rate = np.min(unit_psth)
        max_firing_rate = np.max(unit_psth)
        
        # Avoid division by zero
        if max_firing_rate == min_firing_rate:
            print(f"Warning: Unit {i} has zero range (min=max={min_firing_rate:.3f}). Setting normalized values to 0.5.")
            normalized_data[i] = np.full_like(unit_psth, 0.5)
        else:
            normalized_data[i] = (unit_psth - min_firing_rate) / (max_firing_rate - min_firing_rate)
    
    return normalized_data

def create_normalized_psth_heatmap(spikes_file: str, digitalin_file: str, 
                                  duration_ms: float, units: Optional[List[int]] = None,
                                  bin_size_ms: float = 0.6, pre_interval_ms: float = 5, 
                                  post_interval_ms: float = 10, smooth_window: Optional[int] = 5,
                                  sort_by_firing_rate: bool = False, save_path: Optional[str] = None,
                                  sampling_rate: int = 30000, pico_channel: int = 0) -> Tuple[plt.Figure, np.ndarray, List[int], np.ndarray]:
    """
    Create a Min-Max normalized PSTH heatmap for multiple units using digitalin.dat file.
    
    Args:
        spikes_file: Path to spikes.csv file
        digitalin_file: Path to digitalin.dat file
        duration_ms: Interval duration to analyze in milliseconds
        units: List of units to include (if None, use all available units)
        bin_size_ms: Bin size in milliseconds
        pre_interval_ms: Time before interval in ms
        post_interval_ms: Time after interval in ms
        smooth_window: Number of bins for smoothing (optional)
        sort_by_firing_rate: If True, sort units by total firing rate (descending)
        save_path: Path to save the figure (optional)
        sampling_rate: Sampling rate in Hz (default: 30000)
        pico_channel: Channel for pico intervals (default: 0)
        
    Returns:
        Tuple of (figure, normalized_heatmap_data, unit_labels, time_bins)
    """
    # Load data
    spikes_df, intervals_df = load_data(spikes_file, digitalin_file, sampling_rate, pico_channel)
    
    # Filter intervals by duration
    filtered_intervals = filter_intervals_by_duration(intervals_df, duration_ms)
    
    if len(filtered_intervals) == 0:
        print(f"No intervals found for duration {duration_ms}ms")
        return None, None, None, None
    
    # Get units to analyze
    if units is None:
        units = sorted(spikes_df['unit'].unique())
    
    print(f"Computing normalized PSTH for {len(units)} units...")
    
    # Compute PSTH for each unit
    psth_data = []
    valid_units = []
    
    for unit in units:
        psth = compute_psth_for_unit(spikes_df, filtered_intervals, unit, 
                                   bin_size_ms, pre_interval_ms, post_interval_ms, 
                                   duration_ms, smooth_window)
        if len(psth) > 0:
            psth_data.append(psth)
            valid_units.append(unit)
    
    if len(psth_data) == 0:
        print("No valid PSTH data found")
        return None, None, None, None
    
    # Convert to array for normalization
    raw_heatmap_data = np.array(psth_data)
    
    # Sort units by total firing rate if requested
    if sort_by_firing_rate:
        print("Sorting units by total firing rate (descending order)...")
        # Calculate total firing rate for each unit (sum of all bins)
        total_firing_rates = np.sum(raw_heatmap_data, axis=1)
        
        # Get sorting indices (descending order)
        sort_indices = np.argsort(total_firing_rates)[::-1]
        
        # Sort the data and unit labels
        raw_heatmap_data = raw_heatmap_data[sort_indices]
        valid_units = [valid_units[i] for i in sort_indices]
        
        print(f"Units sorted by total firing rate: {valid_units}")
    
    # Apply Min-Max normalization
    print("Applying Min-Max normalization...")
    normalized_heatmap_data = min_max_normalize_psth(raw_heatmap_data)
    
    # Create time axis - now post_interval_ms is time after interval END
    bin_size_s = bin_size_ms / 1000.0
    pre_interval_s = pre_interval_ms / 1000.0
    post_interval_s = post_interval_ms / 1000.0
    
    # Total time includes: pre_interval + duration + post_interval
    total_time_s = pre_interval_s + (duration_ms / 1000.0) + post_interval_s
    n_bins = int(total_time_s / bin_size_s)
    
    # Time axis runs from -pre_interval_ms to (duration_ms + post_interval_ms)
    time_end = duration_ms + post_interval_ms
    time_bins = np.linspace(-pre_interval_ms, time_end, n_bins)
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot normalized heatmap with sequential colormap (0-1 range)
    im = ax.imshow(normalized_heatmap_data, aspect='auto', origin='lower', 
                   extent=[-pre_interval_ms, time_end, 0, len(valid_units)],
                   cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    
    # Add light grey highlight for the interval period
    ax.axvspan(0, duration_ms, alpha=0.2, color='lightgrey', zorder=1)
    
    # Customize the plot
    ax.set_xlabel('Time relative to interval start (ms)', fontsize=12)
    ax.set_ylabel('Unit', fontsize=12)
    ax.set_title(f'Min-Max Normalized PSTH Heatmap - Duration: {duration_ms}ms\n'
                f'Bin size: {bin_size_ms}ms, Smoothing: {smooth_window} bins', 
                fontsize=14, fontweight='bold')
    
    # Add vertical lines at interval start and end
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.8, linewidth=2)
    ax.axvline(x=duration_ms, color='black', linestyle='--', alpha=0.8, linewidth=2)
    
    # Set Y-axis ticks to show unit numbers centered on each unit's row
    tick_positions = [i + 0.5 for i in range(len(valid_units))]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f'Unit {u}' for u in valid_units])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Firing Rate (0-1)', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved normalized heatmap to: {save_path}")
    
    return fig, normalized_heatmap_data, valid_units, time_bins

def create_multiple_duration_normalized_heatmaps(spikes_file: str, digitalin_file: str,
                                               durations_ms: List[float], units: Optional[List[int]] = None,
                                               bin_size_ms: float = 0.6, pre_interval_ms: float = 5,
                                               post_interval_ms: float = 10, smooth_window: Optional[int] = 5,
                                               sort_by_firing_rate: bool = False, save_dir: Optional[str] = None,
                                               sampling_rate: int = 30000, pico_channel: int = 0) -> Dict[float, Tuple]:
    """
    Create Min-Max normalized PSTH heatmaps for multiple interval durations using digitalin.dat file.
    
    Args:
        spikes_file: Path to spikes.csv file
        digitalin_file: Path to digitalin.dat file
        durations_ms: List of interval durations to analyze in milliseconds
        units: List of units to include (if None, use all available units)
        bin_size_ms: Bin size in milliseconds
        pre_interval_ms: Time before interval in ms
        post_interval_ms: Time after interval in ms
        smooth_window: Number of bins for smoothing (optional)
        sort_by_firing_rate: If True, sort units by total firing rate (descending)
        save_dir: Directory to save figures (optional)
        sampling_rate: Sampling rate in Hz (default: 30000)
        pico_channel: Channel for pico intervals (default: 0)
        
    Returns:
        Dictionary mapping duration to (figure, normalized_heatmap_data, unit_labels, time_bins)
    """
    results = {}
    
    for duration in durations_ms:
        print(f"\n--- Processing normalized heatmap for duration: {duration}ms ---")
        
        # Create save path if directory provided
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'normalized_psth_heatmap_{duration}ms.png')
        
        # Create normalized heatmap
        result = create_normalized_psth_heatmap(
            spikes_file, digitalin_file, duration, units,
            bin_size_ms, pre_interval_ms, post_interval_ms, 
            smooth_window, sort_by_firing_rate, save_path,
            sampling_rate, pico_channel
        )
        
        results[duration] = result
    
    return results

def plot_normalized_summary_statistics(results: Dict[float, Tuple], save_dir: Optional[str] = None):
    """
    Plot summary statistics for normalized data across different durations.
    
    Args:
        results: Dictionary from create_multiple_duration_normalized_heatmaps
        save_dir: Directory to save figure (optional)
    """
    # Extract data for summary
    durations = []
    max_normalized_values = []
    min_normalized_values = []
    mean_normalized_values = []
    
    for duration, (fig, normalized_data, units, time_bins) in results.items():
        if normalized_data is not None:
            durations.append(duration)
            max_normalized_values.append(np.max(normalized_data))
            min_normalized_values.append(np.min(normalized_data))
            mean_normalized_values.append(np.mean(normalized_data))
    
    if not durations:
        print("No valid data for summary statistics")
        return None
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Max/Min Normalized Values
    x_pos = range(len(durations))
    ax1.bar([x - 0.2 for x in x_pos], max_normalized_values, width=0.4, alpha=0.7, 
            color='red', label='Max Normalized')
    ax1.bar([x + 0.2 for x in x_pos], min_normalized_values, width=0.4, alpha=0.7, 
            color='blue', label='Min Normalized')
    ax1.set_xlabel('Interval Duration (ms)')
    ax1.set_ylabel('Normalized Value (0-1)')
    ax1.set_title('Max/Min Normalized Values by Duration')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{d}ms' for d in durations])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Mean normalized values
    ax2.bar(durations, mean_normalized_values, alpha=0.7, color='green')
    ax2.set_xlabel('Interval Duration (ms)')
    ax2.set_ylabel('Mean Normalized Value')
    ax2.set_title('Mean Normalized Value by Duration')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'normalized_psth_heatmap_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved normalized summary plot to: {save_path}")
    
    return fig

def compare_raw_vs_normalized(spikes_file: str, digitalin_file: str, duration_ms: float,
                             units: Optional[List[int]] = None, bin_size_ms: float = 0.6,
                             pre_interval_ms: float = 5, post_interval_ms: float = 10,
                             smooth_window: Optional[int] = 5, sort_by_firing_rate: bool = False, 
                             save_path: Optional[str] = None, sampling_rate: int = 30000, 
                             pico_channel: int = 0):
    """
    Create a side-by-side comparison of raw and normalized PSTH heatmaps using digitalin.dat file.
    
    Args:
        spikes_file: Path to spikes.csv file
        digitalin_file: Path to digitalin.dat file
        duration_ms: Interval duration to analyze in milliseconds
        units: List of units to include (if None, use all available units)
        bin_size_ms: Bin size in milliseconds
        pre_interval_ms: Time before interval in ms
        post_interval_ms: Time after interval in ms
        smooth_window: Number of bins for smoothing (optional)
        sort_by_firing_rate: If True, sort units by total firing rate (descending)
        save_path: Path to save the figure (optional)
        sampling_rate: Sampling rate in Hz (default: 30000)
        pico_channel: Channel for pico intervals (default: 0)
        
    Returns:
        Figure object
    """
    # Load data and compute raw PSTH
    spikes_df, intervals_df = load_data(spikes_file, digitalin_file, sampling_rate, pico_channel)
    filtered_intervals = filter_intervals_by_duration(intervals_df, duration_ms)
    
    if len(filtered_intervals) == 0:
        print(f"No intervals found for duration {duration_ms}ms")
        return None
    
    # Get units to analyze
    if units is None:
        units = sorted(spikes_df['unit'].unique())
    
    # Compute PSTH for each unit
    psth_data = []
    valid_units = []
    
    for unit in units:
        psth = compute_psth_for_unit(spikes_df, filtered_intervals, unit, 
                                   bin_size_ms, pre_interval_ms, post_interval_ms, 
                                   duration_ms, smooth_window)
        if len(psth) > 0:
            psth_data.append(psth)
            valid_units.append(unit)
    
    if len(psth_data) == 0:
        print("No valid PSTH data found")
        return None
    
    # Convert to arrays
    raw_heatmap_data = np.array(psth_data)
    
    # Sort units by total firing rate if requested
    if sort_by_firing_rate:
        # Calculate total firing rate for each unit (sum of all bins)
        total_firing_rates = np.sum(raw_heatmap_data, axis=1)
        
        # Get sorting indices (descending order)
        sort_indices = np.argsort(total_firing_rates)[::-1]
        
        # Sort the data and unit labels
        raw_heatmap_data = raw_heatmap_data[sort_indices]
        valid_units = [valid_units[i] for i in sort_indices]
    
    normalized_heatmap_data = min_max_normalize_psth(raw_heatmap_data)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Calculate time range (post_interval_ms is now time after interval END)
    time_end = duration_ms + post_interval_ms
    
    # Raw heatmap
    im1 = ax1.imshow(raw_heatmap_data, aspect='auto', origin='lower', 
                     extent=[-pre_interval_ms, time_end, 0, len(valid_units)],
                     cmap='hot', interpolation='nearest')
    ax1.set_xlabel('Time relative to interval start (ms)', fontsize=12)
    ax1.set_ylabel('Unit', fontsize=12)
    ax1.set_title(f'Raw PSTH Heatmap\nDuration: {duration_ms}ms', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='white', linestyle='--', alpha=0.8, linewidth=2)
    ax1.axvline(x=duration_ms, color='white', linestyle='--', alpha=0.8, linewidth=2)
    
    # Add light grey highlight for the interval period
    ax1.axvspan(0, duration_ms, alpha=0.2, color='lightgrey', zorder=1)
    
    tick_positions = [i + 0.5 for i in range(len(valid_units))]
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels([f'Unit {u}' for u in valid_units])
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Firing Rate (Hz)', fontsize=12)
    
    # Normalized heatmap (min-max: 0-1 range)
    im2 = ax2.imshow(normalized_heatmap_data, aspect='auto', origin='lower', 
                     extent=[-pre_interval_ms, time_end, 0, len(valid_units)],
                     cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_xlabel('Time relative to interval start (ms)', fontsize=12)
    ax2.set_ylabel('Unit', fontsize=12) 
    ax2.set_title(f'Min-Max Normalized PSTH Heatmap\nDuration: {duration_ms}ms', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.8, linewidth=2)
    ax2.axvline(x=duration_ms, color='black', linestyle='--', alpha=0.8, linewidth=2)
    
    # Add light grey highlight for the interval period
    ax2.axvspan(0, duration_ms, alpha=0.2, color='lightgrey', zorder=1)
    
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels([f'Unit {u}' for u in valid_units])
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Normalized Firing Rate (0-1)', fontsize=12)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    
    return fig

def analyze_units_by_max_normalized_value(results: Dict[float, Tuple], display_results: bool = True) -> Dict[float, List[Tuple[int, float]]]:
    """
    Analyze units ordered by their mean normalized value (since max will always be 1.0 for min-max normalization).
    
    Args:
        results: Dictionary from create_multiple_duration_normalized_heatmaps
        display_results: Whether to print the results to console
        
    Returns:
        Dictionary mapping duration to list of (unit_id, mean_normalized_value) tuples, ordered by mean value descending
    """
    duration_unit_rankings = {}
    
    for duration, (fig, normalized_data, valid_units, time_bins) in results.items():
        if normalized_data is not None and valid_units is not None:
            # Calculate mean normalized value for each unit (more meaningful than max for min-max normalization)
            unit_mean_values = []
            for i, unit_id in enumerate(valid_units):
                mean_normalized_value = np.mean(normalized_data[i])  # Mean normalized value for this unit
                unit_mean_values.append((unit_id, mean_normalized_value))
            
            # Sort by mean normalized value in descending order (highest first)
            unit_mean_values.sort(key=lambda x: x[1], reverse=True)
            duration_unit_rankings[duration] = unit_mean_values
            
            if display_results:
                print(f"\n=== Duration: {duration}ms ===")
                print("Units ordered by highest mean normalized value:")
                print("Rank\tUnit\tMean Normalized")
                print("-" * 35)
                for rank, (unit_id, mean_value) in enumerate(unit_mean_values, 1):
                    print(f"{rank:2d}\tUnit {unit_id:2d}\t{mean_value:8.3f}")
        else:
            if display_results:
                print(f"\nNo valid data for duration {duration}ms")
            duration_unit_rankings[duration] = []
    
    return duration_unit_rankings
