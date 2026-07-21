"""
Z-Score Normalized PSTH Heatmap Utilities

This module provides utilities for creating Z-score normalized PSTH heatmaps.
Z-score normalization: z = (x - u) / o
where:
- z = z-score 
- x = one bin's value (frequency)
- u = mean of all bins for the same unit
- o = standard deviation of all bins for the same unit

This normalization allows for better comparison across units with different baseline firing rates.
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

def load_whisking_intervals(whisking_file: str) -> pd.DataFrame:
    """
    Load whisking interval data from CSV file.
    
    Args:
        whisking_file: Path to whisking.csv file containing intervals to exclude
        
    Returns:
        DataFrame with whisking intervals
    """
    whisking_file = os.path.abspath(whisking_file)
    print(f"Loading whisking intervals from: {whisking_file}")
    
    whisking_df = pd.read_csv(whisking_file)
    whisking_df.columns = whisking_df.columns.str.strip()  # Clean column names
    
    print(f"Loaded {len(whisking_df)} whisking intervals to exclude")
    return whisking_df

def filter_intervals_exclude_whisking(intervals_df: pd.DataFrame, whisking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out pico intervals that overlap with whisking intervals.
    
    Args:
        intervals_df: DataFrame with pico interval data (times in seconds)
        whisking_df: DataFrame with whisking intervals to exclude (times assumed to be in same units)
        
    Returns:
        Filtered DataFrame with whisking intervals excluded
    """
    if whisking_df is None or len(whisking_df) == 0:
        print("No whisking intervals to exclude")
        return intervals_df
    
    # Convert whisking intervals to seconds by dividing by 30,000
    print("Converting whisking intervals to seconds (dividing by 30,000)...")
    whisking_starts = whisking_df['Start'].values / 30000.0
    whisking_ends = whisking_df['End'].values / 30000.0
    print(f"Converted {len(whisking_starts)} whisking intervals to seconds")
    
    # Filter out intervals that overlap with any whisking period
    original_count = len(intervals_df)
    filtered_intervals = []
    
    for idx, interval in intervals_df.iterrows():
        pico_start = interval['pico_Interval Start']
        pico_end = interval['pico_Interval End']
        
        # Check if this pico interval overlaps with any whisking interval
        overlaps = False
        for w_start, w_end in zip(whisking_starts, whisking_ends):
            # Check for overlap: intervals overlap if one starts before the other ends
            if pico_start < w_end and pico_end > w_start:
                overlaps = True
                break
        
        if not overlaps:
            filtered_intervals.append(interval)
    
    # Convert back to DataFrame
    if filtered_intervals:
        result_df = pd.DataFrame(filtered_intervals)
    else:
        result_df = intervals_df.iloc[0:0].copy()  # Empty DataFrame with same structure
    
    excluded_count = original_count - len(result_df)
    print(f"Excluded {excluded_count} intervals that overlapped with whisking periods")
    print(f"Remaining intervals: {len(result_df)}")
    
    return result_df

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

def z_score_normalize_psth(psth_data: np.ndarray) -> np.ndarray:
    """
    Apply Z-score normalization to PSTH data.
    
    Z-score formula: z = (x - u) / o
    where:
    - z = z-score
    - x = one bin's value (frequency)
    - u = mean of all bins for the same unit
    - o = standard deviation of all bins for the same unit
    
    Args:
        psth_data: 2D array with shape (n_units, n_bins)
        
    Returns:
        Z-score normalized array with same shape
    """
    normalized_data = np.zeros_like(psth_data)
    
    for i, unit_psth in enumerate(psth_data):
        mean_firing_rate = np.mean(unit_psth)
        std_firing_rate = np.std(unit_psth)
        
        # Avoid division by zero
        if std_firing_rate == 0:
            print(f"Warning: Unit {i} has zero standard deviation. Setting z-scores to 0.")
            normalized_data[i] = np.zeros_like(unit_psth)
        else:
            normalized_data[i] = (unit_psth - mean_firing_rate) / std_firing_rate
    
    return normalized_data

def create_normalized_psth_heatmap(spikes_file: str, intervals_file: str, 
                                  duration_ms: float, units: Optional[List[int]] = None,
                                  bin_size_ms: float = 0.6, pre_interval_ms: float = 5, 
                                  post_interval_ms: float = 10, smooth_window: Optional[int] = 5,
                                  exclude_whisking: bool = False, whisking_file: Optional[str] = None,
                                  save_path: Optional[str] = None) -> Tuple[plt.Figure, np.ndarray, List[int], np.ndarray]:
    """
    Create a Z-score normalized PSTH heatmap for multiple units.
    
    Args:
        spikes_file: Path to spikes.csv file
        intervals_file: Path to pico_time_adjust.csv file
        duration_ms: Interval duration to analyze in milliseconds
        units: List of units to include (if None, use all available units)
        bin_size_ms: Bin size in milliseconds
        pre_interval_ms: Time before interval in ms
        post_interval_ms: Time after interval in ms
        smooth_window: Number of bins for smoothing (optional)
        exclude_whisking: Whether to exclude intervals that overlap with whisking periods
        whisking_file: Path to whisking.csv file (required if exclude_whisking=True)
        save_path: Path to save the figure (optional)
        
    Returns:
        Tuple of (figure, normalized_heatmap_data, unit_labels, time_bins)
    """
    # Load data
    spikes_df, intervals_df = load_data(spikes_file, intervals_file)
    
    # Filter intervals by duration
    filtered_intervals = filter_intervals_by_duration(intervals_df, duration_ms)
    
    # Exclude whisking intervals if requested
    if exclude_whisking:
        if whisking_file is None:
            raise ValueError("whisking_file must be provided when exclude_whisking=True")
        
        whisking_df = load_whisking_intervals(whisking_file)
        filtered_intervals = filter_intervals_exclude_whisking(filtered_intervals, whisking_df)
    
    if len(filtered_intervals) == 0:
        print(f"No intervals found for duration {duration_ms}ms after filtering")
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
    
    # Apply Z-score normalization
    print("Applying Z-score normalization...")
    normalized_heatmap_data = z_score_normalize_psth(raw_heatmap_data)
    
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
    
    # Plot normalized heatmap with diverging colormap
    vmax = np.max(np.abs(normalized_heatmap_data))
    im = ax.imshow(normalized_heatmap_data, aspect='auto', origin='lower', 
                   extent=[-pre_interval_ms, time_end, 0, len(valid_units)],
                   cmap='RdBu_r', interpolation='nearest', vmin=-vmax, vmax=vmax)
    
    # Add light grey highlight for the interval period
    ax.axvspan(0, duration_ms, alpha=0.2, color='lightgrey', zorder=1)
    
    # Customize the plot
    ax.set_xlabel('Time relative to interval start (ms)', fontsize=12)
    ax.set_ylabel('Unit', fontsize=12)
    ax.set_title(f'Z-Score Normalized PSTH Heatmap - Duration: {duration_ms}ms\n'
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
    cbar.set_label('Z-Score (Standard Deviations)', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved normalized heatmap to: {save_path}")
    
    return fig, normalized_heatmap_data, valid_units, time_bins

def create_multiple_duration_normalized_heatmaps(spikes_file: str, intervals_file: str,
                                               durations_ms: List[float], units: Optional[List[int]] = None,
                                               bin_size_ms: float = 0.6, pre_interval_ms: float = 5,
                                               post_interval_ms: float = 10, smooth_window: Optional[int] = 5,
                                               exclude_whisking: bool = False, whisking_file: Optional[str] = None,
                                               save_dir: Optional[str] = None) -> Dict[float, Tuple]:
    """
    Create Z-score normalized PSTH heatmaps for multiple interval durations.
    
    Args:
        spikes_file: Path to spikes.csv file
        intervals_file: Path to pico_time_adjust.csv file
        durations_ms: List of interval durations to analyze in milliseconds
        units: List of units to include (if None, use all available units)
        bin_size_ms: Bin size in milliseconds
        pre_interval_ms: Time before interval in ms
        post_interval_ms: Time after interval in ms
        smooth_window: Number of bins for smoothing (optional)
        exclude_whisking: Whether to exclude intervals that overlap with whisking periods
        whisking_file: Path to whisking.csv file (required if exclude_whisking=True)
        save_dir: Directory to save figures (optional)
        
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
            spikes_file, intervals_file, duration, units,
            bin_size_ms, pre_interval_ms, post_interval_ms, 
            smooth_window, exclude_whisking, whisking_file, save_path
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
    max_z_scores = []
    min_z_scores = []
    mean_abs_z_scores = []
    
    for duration, (fig, normalized_data, units, time_bins) in results.items():
        if normalized_data is not None:
            durations.append(duration)
            max_z_scores.append(np.max(normalized_data))
            min_z_scores.append(np.min(normalized_data))
            mean_abs_z_scores.append(np.mean(np.abs(normalized_data)))
    
    if not durations:
        print("No valid data for summary statistics")
        return None
    
    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Max/Min Z-scores
    x_pos = range(len(durations))
    ax1.bar([x - 0.2 for x in x_pos], max_z_scores, width=0.4, alpha=0.7, 
            color='red', label='Max Z-score')
    ax1.bar([x + 0.2 for x in x_pos], min_z_scores, width=0.4, alpha=0.7, 
            color='blue', label='Min Z-score')
    ax1.set_xlabel('Interval Duration (ms)')
    ax1.set_ylabel('Z-Score')
    ax1.set_title('Max/Min Z-Scores by Duration')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{d}ms' for d in durations])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Mean absolute Z-scores
    ax2.bar(durations, mean_abs_z_scores, alpha=0.7, color='green')
    ax2.set_xlabel('Interval Duration (ms)')
    ax2.set_ylabel('Mean Absolute Z-Score')
    ax2.set_title('Mean Absolute Z-Score by Duration')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'normalized_psth_heatmap_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved normalized summary plot to: {save_path}")
    
    return fig

def compare_raw_vs_normalized(spikes_file: str, intervals_file: str, duration_ms: float,
                             units: Optional[List[int]] = None, bin_size_ms: float = 0.6,
                             pre_interval_ms: float = 5, post_interval_ms: float = 10,
                             smooth_window: Optional[int] = 5, save_path: Optional[str] = None):
    """
    Create a side-by-side comparison of raw and normalized PSTH heatmaps.
    
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
        Figure object
    """
    # Load data and compute raw PSTH
    spikes_df, intervals_df = load_data(spikes_file, intervals_file)
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
    normalized_heatmap_data = z_score_normalize_psth(raw_heatmap_data)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Raw heatmap
    im1 = ax1.imshow(raw_heatmap_data, aspect='auto', origin='lower', 
                     extent=[-pre_interval_ms, post_interval_ms, 0, len(valid_units)],
                     cmap='hot', interpolation='nearest')
    ax1.set_xlabel('Time relative to interval start (ms)', fontsize=12)
    ax1.set_ylabel('Unit', fontsize=12)
    ax1.set_title(f'Raw PSTH Heatmap\nDuration: {duration_ms}ms', fontsize=14, fontweight='bold')
    ax1.axvline(x=0, color='white', linestyle='--', alpha=0.8, linewidth=2)
    
    tick_positions = [i + 0.5 for i in range(len(valid_units))]
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels([f'Unit {u}' for u in valid_units])
    
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Firing Rate (Hz)', fontsize=12)
    
    # Normalized heatmap
    vmax = np.max(np.abs(normalized_heatmap_data))
    im2 = ax2.imshow(normalized_heatmap_data, aspect='auto', origin='lower', 
                     extent=[-pre_interval_ms, post_interval_ms, 0, len(valid_units)],
                     cmap='RdBu_r', interpolation='nearest', vmin=-vmax, vmax=vmax)
    ax2.set_xlabel('Time relative to interval start (ms)', fontsize=12)
    ax2.set_ylabel('Unit', fontsize=12) 
    ax2.set_title(f'Z-Score Normalized PSTH Heatmap\nDuration: {duration_ms}ms', fontsize=14, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.8, linewidth=2)
    
    ax2.set_yticks(tick_positions)
    ax2.set_yticklabels([f'Unit {u}' for u in valid_units])
    
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Z-Score (Standard Deviations)', fontsize=12)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to: {save_path}")
    
    return fig

def analyze_units_by_max_zscore(results: Dict[float, Tuple], display_results: bool = True) -> Dict[float, List[Tuple[int, float]]]:
    """
    Analyze units ordered by their highest maximum z-score value in the positive direction.
    
    Args:
        results: Dictionary from create_multiple_duration_normalized_heatmaps
        display_results: Whether to print the results to console
        
    Returns:
        Dictionary mapping duration to list of (unit_id, max_zscore) tuples, ordered by max z-score descending
    """
    duration_unit_rankings = {}
    
    for duration, (fig, normalized_data, valid_units, time_bins) in results.items():
        if normalized_data is not None and valid_units is not None:
            # Calculate max z-score for each unit
            unit_max_zscores = []
            for i, unit_id in enumerate(valid_units):
                max_zscore = np.max(normalized_data[i])  # Maximum z-score for this unit
                unit_max_zscores.append((unit_id, max_zscore))
            
            # Sort by max z-score in descending order (highest first)
            unit_max_zscores.sort(key=lambda x: x[1], reverse=True)
            duration_unit_rankings[duration] = unit_max_zscores
            
            if display_results:
                print(f"\n=== Duration: {duration}ms ===")
                print("Units ordered by highest maximum z-score (positive direction):")
                print("Rank\tUnit\tMax Z-Score")
                print("-" * 30)
                for rank, (unit_id, max_zscore) in enumerate(unit_max_zscores, 1):
                    print(f"{rank:2d}\tUnit {unit_id:2d}\t{max_zscore:8.3f}")
        else:
            if display_results:
                print(f"\nNo valid data for duration {duration}ms")
            duration_unit_rankings[duration] = []
    
    return duration_unit_rankings
