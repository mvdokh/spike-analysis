"""
Whisker Heatmap Visualization Script
Creates density heatmaps of whisker positions over frame windows.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def parse_coordinates(coord_string):
    """Parse coordinate string into list of floats."""
    if pd.isna(coord_string) or coord_string == '':
        return []
    return [float(x.strip()) for x in str(coord_string).split(',')]


def create_whisker_heatmap(df, start_frame, num_frames=100, bins=100):
    """
    Create a 2D density heatmap of whisker positions over a frame range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with whisker data
    start_frame : int
        Starting frame number
    num_frames : int
        Number of frames to include in heatmap
    bins : int
        Resolution of the heatmap grid
        
    Returns:
    --------
    tuple : (heatmap, x_edges, y_edges, extent, frame_range, all_x, all_y)
    """
    # Get all frames in the dataset
    all_frames = sorted(df['Frame'].unique())
    
    # Find the actual frame range
    if start_frame not in all_frames:
        # Find closest frame
        start_frame = min(all_frames, key=lambda x: abs(x - start_frame))
    
    start_idx = all_frames.index(start_frame)
    end_idx = min(start_idx + num_frames, len(all_frames))
    frame_range = all_frames[start_idx:end_idx]
    
    # Filter data for this frame range
    frame_data = df[df['Frame'].isin(frame_range)]
    
    # Collect all whisker coordinates
    all_x = []
    all_y = []
    
    for _, row in frame_data.iterrows():
        x_coords = parse_coordinates(row['X'])
        y_coords = parse_coordinates(row['Y'])
        all_x.extend(x_coords)
        all_y.extend(y_coords)
    
    if len(all_x) == 0:
        return None, None, None, None, frame_range, [], []
    
    # Create 2D histogram
    heatmap, x_edges, y_edges = np.histogram2d(all_x, all_y, bins=bins)
    
    # Calculate extent for imshow
    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    
    return heatmap, x_edges, y_edges, extent, frame_range, all_x, all_y


def plot_whisker_heatmap(df, start_frame, num_frames=100, bins=100, 
                         cmap='hot', figsize=(12, 10), show_whiskers=True):
    """
    Plot whisker density heatmap with optional whisker overlays.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with whisker data
    start_frame : int
        Starting frame number
    num_frames : int
        Number of frames to include
    bins : int
        Heatmap resolution
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    show_whiskers : bool
        Whether to overlay individual whiskers
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    heatmap, x_edges, y_edges, extent, frame_range, all_x, all_y = create_whisker_heatmap(
        df, start_frame, num_frames, bins
    )
    
    if heatmap is None:
        print(f"No data found for frames starting at {start_frame}")
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', 
                   cmap=cmap, aspect='auto', interpolation='gaussian', alpha=0.8)
    
    # Optionally overlay whiskers from sample frames
    if show_whiskers and len(frame_range) > 0:
        # Show every Nth frame to avoid overcrowding
        sample_interval = max(1, len(frame_range) // 10)
        sample_frames = frame_range[::sample_interval]
        
        for frame_num in sample_frames:
            frame_data = df[df['Frame'] == frame_num]
            for _, row in frame_data.iterrows():
                x_coords = parse_coordinates(row['X'])
                y_coords = parse_coordinates(row['Y'])
                if len(x_coords) > 0:
                    ax.plot(x_coords, y_coords, 'cyan', alpha=0.1, linewidth=0.5)
    
    ax.set_xlabel('X (pixels)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y (pixels)', fontsize=12, fontweight='bold')
    ax.set_title(f'Whisker Density Heatmap\nFrames {frame_range[0]} - {frame_range[-1]} ({len(frame_range)} frames)', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    return fig, ax


def get_frame_groups(df, num_frames=100):
    """
    Get all possible frame groups of specified size.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with whisker data
    num_frames : int
        Size of each frame group
        
    Returns:
    --------
    list : List of (start_frame, end_frame) tuples
    """
    all_frames = sorted(df['Frame'].unique())
    
    groups = []
    for i in range(0, len(all_frames), num_frames):
        start_frame = all_frames[i]
        end_idx = min(i + num_frames, len(all_frames))
        end_frame = all_frames[end_idx - 1]
        groups.append((start_frame, end_frame))
    
    return groups


if __name__ == "__main__":
    import os
    
    csv_path = "lines_output.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        
        # Get frame groups
        groups = get_frame_groups(df, num_frames=100)
        print(f"Total frame groups: {len(groups)}")
        
        # Plot first group
        if len(groups) > 0:
            start_frame = groups[0][0]
            fig, ax = plot_whisker_heatmap(df, start_frame, num_frames=100)
            plt.show()
    else:
        print(f"Error: File '{csv_path}' not found.")
