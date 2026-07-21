"""
Filter Single-Lick Bouts Script
For each folder in the specified directory, creates a filtered copy of behavior.csv
that excludes bouts containing only a single lick.
"""

import os
import pandas as pd
from pathlib import Path


def filter_single_lick_bouts(input_file, output_file):
    """
    Remove bouts that contain only one lick from the behavior data.
    
    Parameters:
    -----------
    input_file : Path
        Path to the original behavior.csv file
    output_file : Path
        Path to save the filtered behavior.csv file
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Get the bout ID column name
    bout_col = 'Tongue_area_interval_detection_group_intervals_Interval Overlap Assign ID'
    
    # Count how many licks per bout
    bout_counts = df[bout_col].value_counts()
    
    # Keep only bouts with 2 or more licks
    valid_bouts = bout_counts[bout_counts > 1].index
    
    # Filter the dataframe
    df_filtered = df[df[bout_col].isin(valid_bouts)].copy()
    
    # Save to the output file
    df_filtered.to_csv(output_file, index=False)
    
    removed_licks = len(df) - len(df_filtered)
    removed_bouts = len(bout_counts[bout_counts == 1])
    
    return removed_licks, removed_bouts


def process_all_folders(base_dir):
    """
    Process all folders in the base directory.
    
    Parameters:
    -----------
    base_dir : str
        Path to the directory containing session folders
    """
    base_path = Path(base_dir)
    
    print(f"Processing folders in: {base_dir}")
    print("=" * 70)
    
    total_folders_processed = 0
    total_licks_removed = 0
    total_bouts_removed = 0
    
    # Iterate through all subdirectories
    for folder in sorted(base_path.iterdir()):
        if folder.is_dir():
            input_file = folder / 'behavior.csv'
            output_file = folder / 'behavior_filtered.csv'
            
            if input_file.exists():
                try:
                    removed_licks, removed_bouts = filter_single_lick_bouts(input_file, output_file)
                    
                    print(f"✓ {folder.name:12s} | Removed {removed_bouts:2d} single-lick bouts ({removed_licks:3d} licks)")
                    
                    total_folders_processed += 1
                    total_licks_removed += removed_licks
                    total_bouts_removed += removed_bouts
                    
                except Exception as e:
                    print(f"✗ {folder.name:12s} | Error: {e}")
            else:
                print(f"⊘ {folder.name:12s} | No behavior.csv found")
    
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Folders processed: {total_folders_processed}")
    print(f"  Total single-lick bouts removed: {total_bouts_removed}")
    print(f"  Total licks removed: {total_licks_removed}")
    print(f"\nFiltered files saved as 'behavior_filtered.csv' in each folder")


if __name__ == "__main__":
    import sys
    
    # Check if directory path is provided as command line argument
    if len(sys.argv) > 1:
        base_directory = sys.argv[1]
    else:
        # Default to the Phox2B#8SFC directory
        base_directory = r"C:\Users\wanglab\Desktop\Phox2B#8SFC"
    
    # Process all folders
    process_all_folders(base_directory)
