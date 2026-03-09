"""
Whisker Counter Script
Analyzes lines.csv to count the number of frames with 1, 2, 3, 4, 5, 6, and 7 whiskers.
Also calculates whisker lengths from coordinate data.
"""

import pandas as pd
from collections import Counter
import os
import numpy as np


def count_whiskers_per_frame(csv_path):
    """
    Count the number of whiskers (lines) per frame.
    
    Parameters:
    -----------
    csv_path : str
        Path to the lines.csv file
        
    Returns:
    --------
    dict : Dictionary with frame numbers as keys and whisker counts as values
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Count whiskers per frame
    # Each row represents one whisker, so we group by Frame and count
    whisker_counts = df.groupby('Frame').size().to_dict()
    
    return whisker_counts


def analyze_whisker_distribution(csv_path, max_whiskers=7, verbose=True):
    """
    Analyze the distribution of whiskers across frames.
    
    Parameters:
    -----------
    csv_path : str
        Path to the lines.csv file
    max_whiskers : int
        Maximum number of whiskers to count (default: 7)
    verbose : bool
        If True, print detailed output (default: True)
        
    Returns:
    --------
    dict : Dictionary with whisker counts (1-7) as keys and number of frames as values
    """
    # Get whisker counts per frame
    whisker_counts = count_whiskers_per_frame(csv_path)
    
    # Count how many frames have each number of whiskers
    distribution = Counter(whisker_counts.values())
    
    # Create a complete dictionary with all whisker counts from 1 to max_whiskers
    result = {i: distribution.get(i, 0) for i in range(1, max_whiskers + 1)}
    
    if verbose:
        print(f"Analysis of: {os.path.basename(csv_path)}")
        print(f"Total frames analyzed: {len(whisker_counts)}")
        print("\nWhisker Distribution:")
        print("-" * 40)
        for whisker_count in range(1, max_whiskers + 1):
            frame_count = result[whisker_count]
            percentage = (frame_count / len(whisker_counts) * 100) if len(whisker_counts) > 0 else 0
            print(f"Frames with {whisker_count} whisker(s): {frame_count:6d} ({percentage:5.2f}%)")
        print("-" * 40)
    
    return result


def get_frames_by_whisker_count(csv_path, whisker_count):
    """
    Get a list of frame numbers that have a specific number of whiskers.
    
    Parameters:
    -----------
    csv_path : str
        Path to the lines.csv file
    whisker_count : int
        Number of whiskers to filter by
        
    Returns:
    --------
    list : List of frame numbers with the specified whisker count
    """
    whisker_counts = count_whiskers_per_frame(csv_path)
    frames = [frame for frame, count in whisker_counts.items() if count == whisker_count]
    return sorted(frames)


def generate_summary_report(csv_path, max_whiskers=7):
    """
    Generate a comprehensive summary report.
    
    Parameters:
    -----------
    csv_path : str
        Path to the lines.csv file
    max_whiskers : int
        Maximum number of whiskers to analyze (default: 7)
        
    Returns:
    --------
    pd.DataFrame : DataFrame with summary statistics
    """
    result = analyze_whisker_distribution(csv_path, max_whiskers, verbose=False)
    whisker_counts = count_whiskers_per_frame(csv_path)
    total_frames = len(whisker_counts)
    
    summary_data = []
    for whisker_count in range(1, max_whiskers + 1):
        frame_count = result[whisker_count]
        percentage = (frame_count / total_frames * 100) if total_frames > 0 else 0
        summary_data.append({
            'Whisker Count': whisker_count,
            'Number of Frames': frame_count,
            'Percentage': f"{percentage:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    return summary_df


def calculate_whisker_length(x_coords, y_coords):
    """
    Calculate the total length of a whisker from its coordinate points.
    
    The length is calculated as the sum of Euclidean distances between
    consecutive points along the whisker.
    
    Parameters:
    -----------
    x_coords : str or list
        String of comma-separated x coordinates or list of x values
    y_coords : str or list
        String of comma-separated y coordinates or list of y values
        
    Returns:
    --------
    float : Total length of the whisker in pixels
    """
    # Parse coordinates if they are strings
    if isinstance(x_coords, str):
        x_values = [float(x.strip()) for x in x_coords.split(',')]
    else:
        x_values = list(x_coords)
    
    if isinstance(y_coords, str):
        y_values = [float(y.strip()) for y in y_coords.split(',')]
    else:
        y_values = list(y_coords)
    
    # Check that we have the same number of x and y coordinates
    if len(x_values) != len(y_values):
        raise ValueError("Number of x and y coordinates must match")
    
    if len(x_values) < 2:
        return 0.0
    
    # Calculate total length as sum of distances between consecutive points
    total_length = 0.0
    for i in range(len(x_values) - 1):
        dx = x_values[i+1] - x_values[i]
        dy = y_values[i+1] - y_values[i]
        segment_length = np.sqrt(dx**2 + dy**2)
        total_length += segment_length
    
    return total_length


def get_whisker_from_row(csv_path, row_index):
    """
    Get whisker data from a specific row in the CSV file.
    
    Parameters:
    -----------
    csv_path : str
        Path to the lines.csv file
    row_index : int
        Row index (0-based) to retrieve
        
    Returns:
    --------
    dict : Dictionary containing frame, x_coords, y_coords, and length
    """
    df = pd.read_csv(csv_path)
    
    if row_index < 0 or row_index >= len(df):
        raise ValueError(f"Row index {row_index} out of range. CSV has {len(df)} rows.")
    
    row = df.iloc[row_index]
    length = calculate_whisker_length(row['X'], row['Y'])
    
    return {
        'row_index': row_index,
        'frame': row['Frame'],
        'x_coords': row['X'],
        'y_coords': row['Y'],
        'length': length,
        'num_points': len(row['X'].split(','))
    }


def analyze_whisker_lengths_by_frame(csv_path, frame_number):
    """
    Analyze all whiskers in a specific frame and calculate their lengths.
    
    Parameters:
    -----------
    csv_path : str
        Path to the lines.csv file
    frame_number : int
        Frame number to analyze
        
    Returns:
    --------
    pd.DataFrame : DataFrame with whisker details for the frame
    """
    df = pd.read_csv(csv_path)
    frame_data = df[df['Frame'] == frame_number].copy()
    
    if len(frame_data) == 0:
        print(f"No whiskers found in frame {frame_number}")
        return pd.DataFrame()
    
    # Calculate length for each whisker
    frame_data['Length'] = frame_data.apply(
        lambda row: calculate_whisker_length(row['X'], row['Y']), 
        axis=1
    )
    
    # Count points per whisker
    frame_data['Num_Points'] = frame_data['X'].apply(lambda x: len(x.split(',')))
    
    # Add whisker number within frame
    frame_data['Whisker_Num'] = range(1, len(frame_data) + 1)
    
    # Reorder columns
    result_df = frame_data[['Whisker_Num', 'Frame', 'Length', 'Num_Points', 'X', 'Y']]
    
    return result_df


def find_shortest_whiskers_in_frames(csv_path, min_whisker_count, verbose=True):
    """
    Find the shortest whisker in frames that have more than min_whisker_count whiskers.
    
    Parameters:
    -----------
    csv_path : str
        Path to the lines.csv file
    min_whisker_count : int
        Minimum number of whiskers a frame must have to be included
    verbose : bool
        If True, print detailed output (default: True)
        
    Returns:
    --------
    tuple : (results_df, shortest_lengths)
        - results_df: DataFrame with frame number, whisker count, and shortest whisker length
        - shortest_lengths: List of shortest whisker lengths for plotting
    """
    df = pd.read_csv(csv_path)
    
    # Calculate length for each whisker
    df['Length'] = df.apply(
        lambda row: calculate_whisker_length(row['X'], row['Y']), 
        axis=1
    )
    
    # Group by frame and get frame statistics
    frame_stats = df.groupby('Frame').agg({
        'Length': ['count', 'min', 'mean', 'max']
    }).reset_index()
    
    # Flatten column names
    frame_stats.columns = ['Frame', 'Whisker_Count', 'Shortest_Length', 'Mean_Length', 'Longest_Length']
    
    # Filter frames with more than min_whisker_count whiskers
    filtered_frames = frame_stats[frame_stats['Whisker_Count'] > min_whisker_count].copy()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analysis: Frames with MORE than {min_whisker_count} whiskers")
        print(f"{'='*60}")
        print(f"Total frames in dataset: {df['Frame'].nunique()}")
        print(f"Frames with >{min_whisker_count} whiskers: {len(filtered_frames)}")
        
        if len(filtered_frames) > 0:
            print(f"\nShortest Whisker Statistics:")
            print(f"  Mean shortest length: {filtered_frames['Shortest_Length'].mean():.2f} pixels")
            print(f"  Median shortest length: {filtered_frames['Shortest_Length'].median():.2f} pixels")
            print(f"  Min shortest length: {filtered_frames['Shortest_Length'].min():.2f} pixels")
            print(f"  Max shortest length: {filtered_frames['Shortest_Length'].max():.2f} pixels")
            print(f"  Std Dev: {filtered_frames['Shortest_Length'].std():.2f} pixels")
            print(f"\n{'='*60}\n")
    
    return filtered_frames, filtered_frames['Shortest_Length'].tolist()


def plot_shortest_whisker_distribution(csv_path, min_whisker_count, save_path=None):
    """
    Create visualization for shortest whiskers in frames with >min_whisker_count whiskers.
    
    Parameters:
    -----------
    csv_path : str
        Path to the lines.csv file
    min_whisker_count : int
        Minimum number of whiskers a frame must have to be included
    save_path : str, optional
        Path to save the figure (default: None, just display)
        
    Returns:
    --------
    tuple : (fig, filtered_frames_df)
    """
    import matplotlib.pyplot as plt
    
    # Get the data
    filtered_frames, shortest_lengths = find_shortest_whiskers_in_frames(
        csv_path, min_whisker_count, verbose=True
    )
    
    if len(filtered_frames) == 0:
        print(f"No frames found with more than {min_whisker_count} whiskers.")
        return None, filtered_frames
    
    # Create single histogram plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of shortest whisker lengths
    ax.hist(shortest_lengths, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Shortest Whisker Length (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribution of Shortest Whisker Lengths\n(Frames with >{min_whisker_count} whiskers)', 
                  fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    
    return fig, filtered_frames


if __name__ == "__main__":
    # Default path - can be changed
    csv_path = "lines.csv"
    
    if os.path.exists(csv_path):
        # Run the analysis
        result = analyze_whisker_distribution(csv_path, max_whiskers=7, verbose=True)
        
        # Optional: Show summary table
        print("\nSummary Table:")
        summary_df = generate_summary_report(csv_path, max_whiskers=7)
        print(summary_df.to_string(index=False))
    else:
        print(f"Error: File '{csv_path}' not found.")
        print("Please provide the correct path to lines.csv")
