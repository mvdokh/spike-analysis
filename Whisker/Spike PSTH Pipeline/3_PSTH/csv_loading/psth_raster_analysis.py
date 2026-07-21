import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import uniform_filter1d

def load_data(spikes_file=None, intervals_file=None):
    """Load spike and interval data"""
    # Set default paths if not provided
    if spikes_file is None:
        spikes_file = os.path.join(os.path.dirname(__file__), '../../Data/spikes.csv')
    if intervals_file is None:
        intervals_file = os.path.join(os.path.dirname(__file__), '../../Data/pico_time_adjust.csv')
    
    # Convert to absolute paths
    spikes_file = os.path.abspath(spikes_file)
    intervals_file = os.path.abspath(intervals_file)
    
    print(f"Loading spike data from: {spikes_file}")
    spikes_df = pd.read_csv(spikes_file)
    spikes_df.columns = spikes_df.columns.str.strip().str.lower()  # Remove whitespace and convert to lowercase
    
    print(f"Loading interval data from: {intervals_file}")
    intervals_df = pd.read_csv(intervals_file)
    intervals_df.columns = intervals_df.columns.str.strip()  # Remove any whitespace
    
    print(f"Loaded {len(spikes_df)} spikes and {len(intervals_df)} intervals")
    print(f"Available units: {sorted(spikes_df['unit'].unique())}")
    durations = sorted(intervals_df['pico_Interval Duration'].unique())
    print(f"Available interval durations (seconds): {durations[:10]}")  # Show first 10 as examples
    print("Duration conversions:")
    for dur in durations[:5]:  # Show first 5 as examples
        ms_val = dur * 1000  # Convert seconds to ms
        hz_val = 1 / dur if dur > 0 else 0  # Convert seconds to Hz
        print(f"  {dur:.6f}s = {ms_val:.2f}ms = {hz_val:.1f}Hz")
    
    return spikes_df, intervals_df

def filter_intervals_by_duration(intervals_df, target_duration_ms):
    """Filter intervals by duration with tolerance (input in ms, data in seconds)"""
    target_duration_s = target_duration_ms / 1000.0  # Convert ms to seconds
    tolerance_s = 0.001  # ±1ms tolerance in seconds (more appropriate for time-based data)
    
    filtered = intervals_df[
        (intervals_df['pico_Interval Duration'] >= target_duration_s - tolerance_s) &
        (intervals_df['pico_Interval Duration'] <= target_duration_s + tolerance_s)
    ].copy()
    
    print(f"Found {len(filtered)} intervals with duration {target_duration_ms}ms (±1ms)")
    return filtered

def filter_intervals_by_timeframe(intervals_df, start_time=None, end_time=None):
    """Filter intervals by a single time frame (in seconds)."""
    if start_time is None and end_time is None:
        return intervals_df

    filtered = intervals_df.copy()
    if start_time is not None:
        filtered = filtered[filtered['pico_Interval Start'] >= start_time]
    if end_time is not None:
        filtered = filtered[filtered['pico_Interval End'] <= end_time]

    print(f"Filtered to {len(filtered)} intervals within timeframe [{start_time}, {end_time}]")
    return filtered

def filter_intervals_by_timeframes(intervals_df, start_times=None, end_times=None):
    """Filter intervals by multiple time frames (in seconds), preserving the order of windows.

    Parameters:
    - start_times: list of start times (seconds)
    - end_times: list of end times (seconds)

    Returns a concatenated DataFrame of intervals from each window in the provided order.
    """
    if not start_times and not end_times:
        return intervals_df

    if start_times is None or end_times is None or len(start_times) != len(end_times):
        raise ValueError("start_times and end_times must be provided with equal lengths")

    windowed_parts = []
    total_count = 0
    for i, (s, e) in enumerate(zip(start_times, end_times)):
        part = filter_intervals_by_timeframe(intervals_df, s, e)
        # Ensure sorted by start for deterministic order within window
        part = part.sort_values(by=['pico_Interval Start', 'pico_Interval End']).copy()
        windowed_parts.append(part)
        total_count += len(part)
        print(f"Window {i+1}: [{s}, {e}] -> {len(part)} intervals")

    if windowed_parts:
        concatenated = pd.concat(windowed_parts, ignore_index=True)
    else:
        concatenated = intervals_df.iloc[0:0].copy()

    print(f"Total intervals after multi-window filtering: {total_count}")
    return concatenated

def get_spikes_for_intervals(spikes_df, intervals_df, unit, pre_interval_ms=0, post_interval_ms=0):
    """Get spikes for a specific unit within intervals with optional pre/post padding"""
    unit_spikes = spikes_df[spikes_df['unit'] == unit].copy()
    print(f"Unit {unit} has {len(unit_spikes)} total spikes")
    
    # Convert ms to seconds (data is now in seconds)
    pre_interval_s = pre_interval_ms / 1000.0  # Convert ms to seconds
    post_interval_s = post_interval_ms / 1000.0  # Convert ms to seconds
    
    if pre_interval_ms > 0 or post_interval_ms > 0:
        print(f"Including {pre_interval_ms}ms before and {post_interval_ms}ms after each interval")
        print(f"This equals {pre_interval_s:.3f}s and {post_interval_s:.3f}s respectively")
    
    # Find spikes within each interval (with padding)
    trial_spikes = []
    for idx, interval in intervals_df.iterrows():
        start_time = interval['pico_Interval Start'] - pre_interval_s
        end_time = interval['pico_Interval End'] + post_interval_s
        interval_start = interval['pico_Interval Start']  # Original interval start for reference
        
        # Get spikes within this extended interval
        interval_spikes = unit_spikes[
            (unit_spikes['time'] >= start_time) & 
            (unit_spikes['time'] <= end_time)
        ]
        
        # Convert to relative time (ms from original interval start)
        relative_times = (interval_spikes['time'] - interval_start) * 1000  # Convert seconds to ms
        
        trial_spikes.append({
            'trial': idx,
            'spike_times': relative_times.tolist(),
            'start_time': start_time,
            'end_time': end_time,
            'interval_start': interval_start,
            'duration': interval['pico_Interval End'] - interval['pico_Interval Start'],  # Duration in seconds
            'total_duration': end_time - start_time,  # Total duration in seconds
            'pre_ms': pre_interval_ms,
            'post_ms': post_interval_ms
        })
    
    print(f"Processed {len(trial_spikes)} trials for unit {unit}")
    return trial_spikes

def create_psth_raster_plot(trial_spikes, unit, duration, bin_size_ms=1, 
                           frequency_hz=None, max_trials=None, smooth_window=None, trial_ranges=None):
    """Create PSTH and raster plot"""
    
    if not trial_spikes:
        print("No trials found for plotting")
        return None, None
    
    # Convert duration from ms to seconds for filtering, then set specific frequency for header
    duration_s = duration / 1000.0  # Convert target duration from ms to seconds
    
    # Set specific frequencies for header display based on duration
    if frequency_hz is None:
        freq_mapping = {25: 20, 10: 50, 5: 100}  # ms -> Hz mapping for header
        frequency_hz = freq_mapping.get(duration, 1.0 / duration_s)  # Use mapping or calculate
    
    # Filter trials based on trial_ranges or max_trials
    if trial_ranges:
        # Select specific trial ranges
        selected_trials = []
        trial_mapping = []  # Keep track of original trial indices for raster plot
        compact_mapping = []  # Compact y-axis positions (0, 1, 2, ...)
        range_info = []  # Store range boundaries for visual breaks
        
        compact_y = 0
        for range_idx, (start_idx, end_idx) in enumerate(trial_ranges):
            # Adjust for 0-based indexing and bounds checking
            start_idx = max(0, start_idx)
            end_idx = min(len(trial_spikes), end_idx + 1)  # +1 because range is inclusive
            
            if start_idx < len(trial_spikes):
                range_trials = trial_spikes[start_idx:end_idx]
                selected_trials.extend(range_trials)
                
                # Store range info for breaks
                range_start_y = compact_y
                range_end_y = compact_y + len(range_trials) - 1
                range_info.append({
                    'original_start': start_idx,
                    'original_end': min(len(trial_spikes) - 1, end_idx - 1),
                    'compact_start': range_start_y,
                    'compact_end': range_end_y
                })
                
                # Create mapping for raster plot y-axis
                for i, trial in enumerate(range_trials):
                    trial_mapping.append(start_idx + i)  # Original trial number
                    compact_mapping.append(compact_y)    # Compact y position
                    compact_y += 1
        
        trial_spikes = selected_trials
        print(f"Selected {len(trial_spikes)} trials from specified ranges")
        
    elif max_trials and len(trial_spikes) > max_trials:
        trial_spikes = trial_spikes[:max_trials]
        trial_mapping = list(range(len(trial_spikes)))  # Sequential mapping
        compact_mapping = list(range(len(trial_spikes)))  # Sequential mapping
        range_info = None
        print(f"Limited to first {max_trials} trials")
    else:
        trial_mapping = list(range(len(trial_spikes)))  # Sequential mapping
        compact_mapping = list(range(len(trial_spikes)))  # Sequential mapping
        range_info = None
    
    # Determine time range based on pre/post interval padding
    pre_ms = trial_spikes[0].get('pre_ms', 0) if trial_spikes else 0
    post_ms = trial_spikes[0].get('post_ms', 0) if trial_spikes else 0
    # Convert from seconds to ms (data is now in seconds)
    avg_duration_ms = np.mean([trial['duration'] for trial in trial_spikes]) * 1000
    
    # Set time range
    min_time = -pre_ms
    max_time = avg_duration_ms + post_ms
    
    # Create bins for PSTH
    bins = np.arange(min_time, max_time + bin_size_ms, bin_size_ms)
    
    # Collect all spike times for PSTH
    all_spike_times = []
    for trial in trial_spikes:
        all_spike_times.extend(trial['spike_times'])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                   gridspec_kw={'height_ratios': [1, 2]})
    
    # PSTH (top plot)
    if all_spike_times:
        counts, bin_edges = np.histogram(all_spike_times, bins=bins)
        # Convert to firing rate (spikes/sec)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        firing_rate = counts / (len(trial_spikes) * bin_size_ms / 1000.0)  # spikes/sec
        
        # Apply smoothing if requested
        if smooth_window and smooth_window > 1:
            firing_rate_smooth = uniform_filter1d(firing_rate, size=smooth_window)
            print(f"Applied smoothing with window size: {smooth_window}")
        else:
            firing_rate_smooth = firing_rate
        
        # Use both bar and line plot for better visibility
        ax1.bar(bin_centers, firing_rate, width=bin_size_ms*0.8, 
                alpha=0.6, color='lightblue', edgecolor='blue', linewidth=0.5)
        
        # Plot smoothed line if smoothing was applied
        if smooth_window and smooth_window > 1:
            ax1.plot(bin_centers, firing_rate_smooth, color='red', linewidth=2, alpha=0.9, label='Smoothed')
            ax1.plot(bin_centers, firing_rate, color='darkblue', linewidth=1, alpha=0.6, label='Raw')
            ax1.legend()
        else:
            ax1.plot(bin_centers, firing_rate, color='darkblue', linewidth=2, alpha=0.8)
        
        print(f"PSTH: {len(all_spike_times)} spikes across {len(trial_spikes)} trials")
        print(f"Max firing rate: {np.max(firing_rate):.2f} Hz")
    else:
        print("No spikes found for PSTH")
    
    # Add vertical lines to mark interval boundaries
    if pre_ms > 0:
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Interval Start')
    if post_ms > 0:
        ax1.axvline(x=avg_duration_ms, color='red', linestyle='--', alpha=0.7, label='Interval End')
    
    ax1.set_ylabel('Firing Rate (Hz)')
    title_text = f'PSTH - Unit {unit} | {frequency_hz:.1f} Hz | Duration: {duration:.1f}ms | Trials: {len(trial_spikes)}'
    if pre_ms > 0 or post_ms > 0:
        title_text += f' | Pre: {pre_ms}ms, Post: {post_ms}ms'
    if smooth_window and smooth_window > 1:
        title_text += f' | Smooth: {smooth_window}'
    ax1.set_title(title_text)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min_time, max_time)
    
    # Raster plot (bottom plot)
    for plot_idx, trial in enumerate(trial_spikes):
        spike_times = trial['spike_times']
        if spike_times:
            # Use sequential y position for clean appearance
            y_pos = plot_idx
            ax2.scatter(spike_times, [y_pos] * len(spike_times), 
                       s=1, color='black', alpha=0.8)
    
    # Add vertical lines to raster plot as well
    if pre_ms > 0:
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    if post_ms > 0:
        ax2.axvline(x=avg_duration_ms, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Trial Number')
    ax2.set_title(f'Raster Plot - Unit {unit}')
    ax2.set_xlim(min_time, max_time)
    
    # Set y-axis limits
    ax2.set_ylim(-0.5, len(trial_spikes) - 0.5)
    
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print(f"Created plot with {len(trial_spikes)} trials")
    print(f"Total spikes: {len(all_spike_times)}")
    if len(all_spike_times) > 0 and len(trial_spikes) > 0:
        total_time_s = (max_time - min_time) / 1000.0
        print(f"Average firing rate: {len(all_spike_times) / (len(trial_spikes) * total_time_s):.2f} Hz")
    
    return fig, (ax1, ax2)

def run_psth_analysis(unit, duration, bin_size_ms=1, start_time=None, end_time=None, 
                     start_times=None, end_times=None,
                     max_trials=None, pre_interval_ms=0, post_interval_ms=0, 
                     smooth_window=None, trial_ranges=None, spikes_file=None, intervals_file=None,
                     save=False, output_path=None):
    """
    Main function to run PSTH analysis
    
    Parameters:
    - unit: Which unit(s) to analyze - can be single int or list of ints [1, 2, 3]
    - duration: Interval duration to filter in milliseconds (25, 10, 5)
    - bin_size_ms: Bin size in milliseconds for PSTH
    - start_time: Start time for trial filtering in seconds (optional)
    - end_time: End time for trial filtering in seconds (optional)
    - start_times: List of start times for multiple windows in seconds (optional)
    - end_times: List of end times for multiple windows in seconds (optional)
    - max_trials: Maximum number of trials to plot (optional)
    - pre_interval_ms: Milliseconds before interval to include (optional)
    - post_interval_ms: Milliseconds after interval to include (optional)
    - smooth_window: Number of bins for smoothing PSTH (optional, e.g. 3, 5, 10)
    - trial_ranges: List of trial ranges to plot [(start, end), ...] (optional, overrides max_trials)
    - spikes_file: Path to spikes file (optional, uses default spikes.csv)
    - intervals_file: Path to intervals file (optional, uses default pico_time_adjust.csv)
    
    Returns:
    - If single unit: (fig, trial_data)
    - If multiple units: list of [(fig, trial_data), ...]
    """
    
    # Handle multiple units
    if isinstance(unit, (list, tuple)):
        print("="*60)
        print(f"PSTH & RASTER ANALYSIS - MULTIPLE UNITS: {unit}")
        print("="*60)
        
        results = []
        for single_unit in unit:
            print(f"\n--- Processing Unit {single_unit} ---")
            result = run_psth_analysis(
                unit=single_unit, 
                duration=duration, 
                bin_size_ms=bin_size_ms,
                start_time=start_time, 
                end_time=end_time,
                start_times=start_times,
                end_times=end_times,
                max_trials=max_trials, 
                pre_interval_ms=pre_interval_ms, 
                post_interval_ms=post_interval_ms,
                smooth_window=smooth_window, 
                trial_ranges=trial_ranges, 
                spikes_file=spikes_file, 
                intervals_file=intervals_file,
                save=save,
                output_path=output_path
            )
            results.append(result)
        
        print(f"\n{'='*60}")
        print(f"COMPLETED ANALYSIS FOR {len(unit)} UNITS")
        print("="*60)
        return results
    
    # Single unit analysis (original code)
    print("="*60)
    print("PSTH & RASTER ANALYSIS")
    print("="*60)
    print(f"Unit: {unit}")
    print(f"Duration: {duration}ms")
    print(f"Bin size: {bin_size_ms}ms")
    if start_times and end_times:
        print("Time windows:")
        for i, (s, e) in enumerate(zip(start_times, end_times), start=1):
            print(f"  {i}. [{s}, {e}]")
    else:
        if start_time is not None: print(f"Start time: {start_time}")
        if end_time is not None: print(f"End time: {end_time}")
    if trial_ranges:
        print(f"Trial ranges: {trial_ranges}")
        max_trials = None  # Override max_trials when trial_ranges is specified
        print("Max trials set to None (using trial ranges)")
    elif max_trials: 
        print(f"Max trials: {max_trials}")
    if pre_interval_ms > 0: print(f"Pre-interval: {pre_interval_ms}ms")
    if post_interval_ms > 0: print(f"Post-interval: {post_interval_ms}ms")
    if smooth_window: print(f"Smoothing window: {smooth_window} bins")
    print("-"*60)
    
    # Load data
    spikes_df, intervals_df = load_data(spikes_file, intervals_file)
    
    # Filter intervals by duration (duration parameter is in ms)
    filtered_intervals = filter_intervals_by_duration(intervals_df, duration)
    
    # Filter by timeframe if specified (multiple windows take precedence if provided)
    if start_times and end_times:
        filtered_intervals = filter_intervals_by_timeframes(filtered_intervals, start_times, end_times)
    elif start_time is not None or end_time is not None:
        filtered_intervals = filter_intervals_by_timeframe(filtered_intervals, start_time, end_time)
    
    if len(filtered_intervals) == 0:
        print("No intervals found with specified criteria!")
        return None
    
    # Get spikes for the specified unit within intervals
    trial_spikes = get_spikes_for_intervals(spikes_df, filtered_intervals, unit, 
                                          pre_interval_ms, post_interval_ms)
    
    # Create plot
    fig, axes = create_psth_raster_plot(trial_spikes, unit, duration, 
                                       bin_size_ms, max_trials=max_trials, 
                                       smooth_window=smooth_window, trial_ranges=trial_ranges)
    
    # Save plot if requested
    if save and fig and output_path:
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Create filename with parameters
        filename = f"psth_raster_unit_{unit}_dur_{duration}ms_bin_{bin_size_ms}ms"
        if pre_interval_ms > 0 or post_interval_ms > 0:
            filename += f"_pre{pre_interval_ms}_post{post_interval_ms}"
        if smooth_window:
            filename += f"_smooth{smooth_window}"
        if trial_ranges:
            filename += f"_ranges{len(trial_ranges)}"
        elif max_trials:
            filename += f"_max{max_trials}"
        filename += ".png"
        
        filepath = os.path.join(output_path, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
    
    print("="*60)
    
    return fig, trial_spikes

if __name__ == "__main__":
    # Example usage
    fig, trial_data = run_psth_analysis(
        unit=28,
        duration=750,
        bin_size_ms=5,
        max_trials=100
    )
    
    if fig:
        plt.show()