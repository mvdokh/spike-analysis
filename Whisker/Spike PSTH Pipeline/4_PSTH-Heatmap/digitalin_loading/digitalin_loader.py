"""
Digital input (.dat) file loader for PSTH analysis

This module provides functionality to load digitalin.dat files and extract
TTL events from specific channels to replace CSV-based interval loading.

Channel mapping:
- Channel 0: Pico intervals 
- Channel 1: Time markers

Usage:
    from digitalin_loader import load_digitalin_intervals
    intervals_df = load_digitalin_intervals('path/to/digitalin.dat', sampling_rate=30000)
"""

import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path to import binary_data, ttls, and intan modules
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from binary_data import get_digital
from ttls import get_ttl_timestamps_16bit, find_ttls_on_single_channel_16bit


def load_digitalin_data(filepath, sampling_rate=30000):
    """
    Load digital input data from .dat file
    
    Parameters
    ----------
    filepath : str
        Path to the digitalin.dat file
    sampling_rate : int
        Sampling rate in Hz (default: 30000)
        
    Returns
    -------
    digital_inputs : np.ndarray
        Array of 16-bit digital input values
    """
    print(f"Loading digital input data from: {filepath}")
    
    # Load binary data using the existing utility
    digital_inputs = get_digital(
        filepath=filepath,
        header_offset_in_bytes=0,  # Assume no header for .dat files
        single_sample_size_in_bytes=2  # 16-bit data
    )
    
    print(f"Loaded {len(digital_inputs)} samples at {sampling_rate} Hz")
    print(f"Recording duration: {len(digital_inputs) / sampling_rate:.2f} seconds")
    
    return digital_inputs


def extract_channel_ttl_events(digital_inputs, channel, sampling_rate=30000, isTransitionLowToHigh=None):
    """
    Extract TTL events from a specific channel
    
    Parameters
    ----------
    digital_inputs : np.ndarray
        16-bit digital input data
    channel : int
        Channel number (0-15)
    sampling_rate : int
        Sampling rate in Hz
    isTransitionLowToHigh : bool or None
        TTL transition direction (None for auto-detection)
        
    Returns
    -------
    onsets : np.ndarray
        TTL onset timestamps in seconds
    offsets : np.ndarray
        TTL offset timestamps in seconds
    """
    print(f"\nExtracting TTL events from channel {channel}")
    
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


def create_intervals_dataframe(pico_onsets, pico_offsets, time_onsets=None):
    """
    Create a DataFrame in the same format as pico_time_adjust.csv
    
    Parameters
    ----------
    pico_onsets : np.ndarray
        Pico interval start times in seconds
    pico_offsets : np.ndarray
        Pico interval end times in seconds  
    time_onsets : np.ndarray, optional
        Time marker events (for validation/synchronization)
        
    Returns
    -------
    intervals_df : pd.DataFrame
        DataFrame with columns matching pico_time_adjust.csv format:
        - pico_Interval Start
        - pico_Interval End  
        - pico_Interval Duration
    """
    print(f"\nCreating intervals DataFrame with {len(pico_onsets)} intervals")
    
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


def load_digitalin_intervals(digitalin_filepath, sampling_rate=30000, 
                           pico_channel=0, time_channel=1, 
                           pico_transition=None, time_transition=None):
    """
    Main function to load digitalin.dat and create intervals DataFrame
    
    Parameters
    ----------
    digitalin_filepath : str
        Path to digitalin.dat file
    sampling_rate : int
        Sampling rate in Hz (default: 30000)
    pico_channel : int
        Channel for pico intervals (default: 0)
    time_channel : int  
        Channel for time markers (default: 1)
    pico_transition : bool or None
        TTL transition direction for pico channel
    time_transition : bool or None
        TTL transition direction for time channel
        
    Returns
    -------
    intervals_df : pd.DataFrame
        DataFrame compatible with existing PSTH analysis functions
    """
    print("="*60)
    print("DIGITALIN.DAT INTERVAL LOADER")
    print("="*60)
    
    # Load digital input data
    digital_inputs = load_digitalin_data(digitalin_filepath, sampling_rate)
    
    # Extract pico interval events (channel 0)
    pico_onsets, pico_offsets = extract_channel_ttl_events(
        digital_inputs=digital_inputs,
        channel=pico_channel,
        sampling_rate=sampling_rate,
        isTransitionLowToHigh=pico_transition
    )
    
    # Extract time marker events (channel 1) - optional for validation
    if time_channel is not None:
        try:
            time_onsets, time_offsets = extract_channel_ttl_events(
                digital_inputs=digital_inputs,
                channel=time_channel,
                sampling_rate=sampling_rate,
                isTransitionLowToHigh=time_transition
            )
            print(f"Loaded {len(time_onsets)} time marker events from channel {time_channel}")
        except Exception as e:
            print(f"Warning: Could not extract time markers from channel {time_channel}: {e}")
            time_onsets = None
    else:
        time_onsets = None
    
    # Create intervals DataFrame
    intervals_df = create_intervals_dataframe(pico_onsets, pico_offsets, time_onsets)
    
    print(f"\nSuccessfully created intervals DataFrame with {len(intervals_df)} intervals")
    print("="*60)
    
    return intervals_df


def validate_intervals_compatibility(intervals_df):
    """
    Validate that the intervals DataFrame is compatible with existing PSTH analysis
    
    Parameters
    ----------
    intervals_df : pd.DataFrame
        Intervals DataFrame to validate
        
    Returns
    -------
    bool
        True if compatible, False otherwise
    """
    required_columns = ['pico_Interval Start', 'pico_Interval End', 'pico_Interval Duration']
    
    print("Validating intervals DataFrame compatibility...")
    
    # Check required columns
    missing_columns = [col for col in required_columns if col not in intervals_df.columns]
    if missing_columns:
        print(f"❌ Missing required columns: {missing_columns}")
        return False
    
    # Check data types
    for col in required_columns:
        if not np.issubdtype(intervals_df[col].dtype, np.number):
            print(f"❌ Column '{col}' should be numeric, got {intervals_df[col].dtype}")
            return False
    
    # Check for valid time ordering
    invalid_durations = intervals_df['pico_Interval Duration'] <= 0
    if invalid_durations.any():
        print(f"❌ Found {invalid_durations.sum()} intervals with invalid durations")
        return False
    
    # Check for reasonable time values (not negative, not too large)
    if (intervals_df['pico_Interval Start'] < 0).any():
        print("❌ Found negative start times")
        return False
    
    print("✅ Intervals DataFrame is compatible with existing PSTH analysis")
    return True


if __name__ == "__main__":
    # Test the loader with example file path
    test_filepath = "/home/wanglab/spike-analysis/Data/040425/digitalin.dat"
    
    if os.path.exists(test_filepath):
        intervals_df = load_digitalin_intervals(test_filepath)
        validate_intervals_compatibility(intervals_df)
        print(f"\nLoaded intervals shape: {intervals_df.shape}")
        print(intervals_df.head())
    else:
        print(f"Test file not found: {test_filepath}")
        print("Please provide a valid digitalin.dat file path for testing")