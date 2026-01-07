import pandas as pd
import numpy as np
import os

def combine_side_bottom_csvs(directory_path):
    """
    Combine side.csv and bottom.csv with normalization and frame handling.
    
    Steps:
    1. Read both CSV files
    2. Normalize by subtracting the median baseline from each file
    3. For frames in both CSVs: sum the normalized values
    4. For frames in only one CSV: multiply the normalized value by 2
    5. Save to side+bottom.csv
    """
    
    # Construct file paths
    side_path = os.path.join(directory_path, "side.csv")
    bottom_path = os.path.join(directory_path, "bottom.csv")
    output_path = os.path.join(directory_path, "side+bottom.csv")
    
    # Read CSV files
    print(f"Reading {side_path}...")
    side_df = pd.read_csv(side_path)
    
    print(f"Reading {bottom_path}...")
    bottom_df = pd.read_csv(bottom_path)
    
    # Calculate baseline (minimum) for normalization
    # This ensures all values remain positive after normalization
    side_baseline = side_df['Data'].min()
    bottom_baseline = bottom_df['Data'].min()
    
    print(f"Side baseline (minimum): {side_baseline:.2f}")
    print(f"Bottom baseline (minimum): {bottom_baseline:.2f}")
    
    # Normalize the data by subtracting baseline (shifts minimum to 0)
    side_df['Data_normalized'] = side_df['Data'] - side_baseline
    bottom_df['Data_normalized'] = bottom_df['Data'] - bottom_baseline
    
    # Create a set of all unique time points
    all_times = sorted(set(side_df['Time'].tolist() + bottom_df['Time'].tolist()))
    
    # Create dictionaries for quick lookup
    side_dict = dict(zip(side_df['Time'], side_df['Data_normalized']))
    bottom_dict = dict(zip(bottom_df['Time'], bottom_df['Data_normalized']))
    
    # Combine the data
    combined_data = []
    frames_in_both = 0
    frames_in_side_only = 0
    frames_in_bottom_only = 0
    
    for time in all_times:
        in_side = time in side_dict
        in_bottom = time in bottom_dict
        
        if in_side and in_bottom:
            # Both frames exist: sum the normalized values
            combined_value = side_dict[time] + bottom_dict[time]
            frames_in_both += 1
        elif in_side:
            # Only in side: multiply by 2
            combined_value = side_dict[time] * 2
            frames_in_side_only += 1
        else:
            # Only in bottom: multiply by 2
            combined_value = bottom_dict[time] * 2
            frames_in_bottom_only += 1
        
        combined_data.append({'Time': time, 'Data': combined_value})
    
    # Create output dataframe
    output_df = pd.DataFrame(combined_data)
    
    # Save to CSV
    output_df.to_csv(output_path, index=False)
    
    print(f"\nCombined CSV saved to: {output_path}")
    print(f"Total frames: {len(all_times)}")
    print(f"Frames in both CSVs: {frames_in_both}")
    print(f"Frames only in side.csv: {frames_in_side_only}")
    print(f"Frames only in bottom.csv: {frames_in_bottom_only}")
    print(f"\nFirst few rows of combined data:")
    print(output_df.head(10))
    print(f"\nLast few rows of combined data:")
    print(output_df.tail(10))


if __name__ == "__main__":
    # Directory containing the CSV files
    directory = r"C:\Users\wanglab\Desktop\IRt.PCRt-1125\IRt_114\IRt_02_2025_04251_side_and_bottom"
    
    combine_side_bottom_csvs(directory)
