"""
Combine traces from multiple CSV files.
This script combines all lines from multiple whisker trace files into a single output file,
where each frame will contain all lines from all input files.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def combine_traces(input_files, output_file):
    """
    Combine traces from multiple CSV files.
    For each frame, this combines all lines from all input files.
    
    Parameters:
    -----------
    input_files : list of str
        List of paths to input CSV files
    output_file : str
        Path to save the combined output CSV
    """
    
    # Read all input files
    dataframes = []
    for i, file_path in enumerate(input_files):
        print(f"Reading {file_path}...")
        df = pd.read_csv(file_path)
        
        # Add tracking columns
        df['source_file'] = Path(file_path).name
        df['source_index'] = i
        
        dataframes.append(df)
        print(f"  - Loaded {len(df)} rows from {df['frame'].nunique()} frames")
    
    # Combine all dataframes - merges all lines from all files for each frame
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Sort by frame first, then by source to keep organized
    if 'frame' in combined_df.columns:
        combined_df = combined_df.sort_values(['frame', 'source_index']).reset_index(drop=True)
    
    # Remove tracking columns for output
    output_df = combined_df.drop(columns=['source_file', 'source_index'])
    
    # Save combined data
    output_df.to_csv(output_file, index=False)
    print(f"\nCombined {len(output_df)} total rows")
    print(f"Saved to: {output_file}")
    
    # Print summary statistics
    if 'frame' in combined_df.columns:
        frames = combined_df['frame'].unique()
        lines_per_frame = combined_df.groupby('frame').size()
        print("\nSummary:")
        print(f"  - Total frames: {len(frames)}")
        print(f"  - Lines per frame (mean): {lines_per_frame.mean():.1f}")
        print(f"  - Lines per frame (min): {lines_per_frame.min()}")
        print(f"  - Lines per frame (max): {lines_per_frame.max()}")
        
        # Show example frame breakdown
        first_frame = frames[0]
        print(f"\n  - Example: Frame {first_frame} has lines from:")
        frame_data = combined_df[combined_df['frame'] == first_frame]
        for source in frame_data['source_file'].unique():
            count = len(frame_data[frame_data['source_file'] == source])
            print(f"    • {source}: {count} lines")


if __name__ == "__main__":
    # Define input files
    base_path = r"C:\Users\wanglab\Desktop\Club Like Endings\102725_1\102725_1_shortened"
    
    input_files = [
        Path(base_path) / "mask_lines.csv",
        Path(base_path) / "mask_lines+media_mod.csv",
        Path(base_path) / "lines+media_mod.csv",
        Path(base_path) / "lines.csv"
    ]
    
    # Convert to strings and verify files exist
    input_files_str = []
    for file in input_files:
        if file.exists():
            input_files_str.append(str(file))
            print(f"✓ Found: {file.name}")
        else:
            print(f"✗ Missing: {file}")
    
    if len(input_files_str) != len(input_files):
        print("\nError: Not all input files were found!")
        exit(1)
    
    # Define output file
    output_file = Path(base_path) / "combined_traces.csv"
    
    print("\n" + "="*60)
    print("Starting trace combination...")
    print("="*60 + "\n")
    
    # Combine the traces
    combine_traces(input_files_str, str(output_file))
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)
