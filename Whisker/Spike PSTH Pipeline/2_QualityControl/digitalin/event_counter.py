#!/usr/bin/env python3
"""
Event Counter Script
====================

This script analyzes two types of data files:
1. digitalin.dat - binary file containing digital input signals
2. pico_time_adjust.csv - CSV file containing pico interval and spike data

Functions:
- count_digitalin_events: Counts events in channel 0 from digitalin.dat
- count_pico_events: Counts total events from pico_time_adjust.csv
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the Full_Pipeline directory to the path to import the required modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from Full_Pipeline.binary_data import get_digital
from Full_Pipeline.ttls import get_ttl_timestamps_16bit


def count_digitalin_events(digitalin_filepath, channel=0):
    """
    Count the number of events in a specific channel from digitalin.dat file.
    
    Parameters
    ----------
    digitalin_filepath : str
        Path to the digitalin.dat file
    channel : int, optional
        Channel number to analyze (default: 0)
        
    Returns
    -------
    int
        Number of events (transitions) in the specified channel
    """
    try:
        # Load digital inputs from the binary file
        digital_inputs = get_digital(digitalin_filepath, header_offset_in_bytes=0, single_sample_size_in_bytes=2)
        
        # Get TTL timestamps for the specified channel
        ttl_onsets, ttl_offsets = get_ttl_timestamps_16bit(digital_inputs, channel)
        
        # Count the number of events (onsets)
        num_events = len(ttl_onsets)
        
        print(f"Channel {channel} from digitalin.dat:")
        print(f"  Number of onset events: {len(ttl_onsets)}")
        print(f"  Number of offset events: {len(ttl_offsets)}")
        print(f"  Total transitions: {num_events}")
        
        return num_events
        
    except Exception as e:
        print(f"Error processing digitalin.dat: {e}")
        return 0


def count_pico_events(pico_csv_filepath):
    """
    Count the total number of events from pico_time_adjust.csv file.
    
    Parameters
    ----------
    pico_csv_filepath : str
        Path to the pico_time_adjust.csv file
        
    Returns
    -------
    int
        Total number of pico interval events (rows in the CSV)
    """
    try:
        # Load the CSV file
        df = pd.read_csv(pico_csv_filepath)
        
        # Count the number of rows (events)
        num_events = len(df)
        
        print(f"pico_time_adjust.csv:")
        print(f"  Total number of pico intervals: {num_events}")
        print(f"  Columns: {list(df.columns[:5])}...")  # Show first 5 columns
        
        # Additional statistics
        if 'pico_Interval Duration' in df.columns:
            mean_duration = df['pico_Interval Duration'].mean()
            print(f"  Mean interval duration: {mean_duration:.6f} seconds")
        
        return num_events
        
    except Exception as e:
        print(f"Error processing pico_time_adjust.csv: {e}")
        return 0


def main():
    """
    Main function to analyze both data files.
    """
    # Define file paths
    data_dir = "/home/wanglab/spike-analysis/Data/040425"
    digitalin_file = os.path.join(data_dir, "digitalin.dat")
    pico_file = os.path.join(data_dir, "pico_time_adjust.csv")
    
    print("Event Counter Analysis")
    print("=" * 50)
    print()
    
    # Check if files exist
    if not os.path.exists(digitalin_file):
        print(f"Error: digitalin.dat not found at {digitalin_file}")
        return
    
    if not os.path.exists(pico_file):
        print(f"Error: pico_time_adjust.csv not found at {pico_file}")
        return
    
    # Count events in digitalin.dat (channel 0)
    print("1. Analyzing digitalin.dat (Channel 0)")
    print("-" * 40)
    digitalin_events = count_digitalin_events(digitalin_file, channel=0)
    print()
    
    # Count events in pico_time_adjust.csv
    print("2. Analyzing pico_time_adjust.csv")
    print("-" * 40)
    pico_events = count_pico_events(pico_file)
    print()
    
    # Summary
    print("Summary")
    print("-" * 40)
    print(f"Digital input channel 0 events: {digitalin_events}")
    print(f"Pico interval events: {pico_events}")


if __name__ == "__main__":
    main()