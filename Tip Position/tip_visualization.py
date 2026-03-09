"""
Tongue Tip Keypoint Visualization
Visualizes tongue tip position across frames with heatmap generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def load_keypoints(filepath):
    """Load keypoint data from CSV file"""
    df = pd.read_csv(filepath)
    return df


def detect_licks(df, max_gap=5):
    """
    Detect individual licks as consecutive frame sequences
    
    Parameters:
    -----------
    df : DataFrame with columns [Frame, X, Y, Probability]
    max_gap : int, maximum frame gap to still consider part of same lick
    
    Returns:
    --------
    list of DataFrames, each containing frames from one lick
    """
    df_sorted = df.sort_values('Frame').reset_index(drop=True)
    
    licks = []
    current_lick = [df_sorted.iloc[0]]
    
    for i in range(1, len(df_sorted)):
        frame_gap = df_sorted.iloc[i]['Frame'] - df_sorted.iloc[i-1]['Frame']
        
        if frame_gap <= max_gap:
            current_lick.append(df_sorted.iloc[i])
        else:
            if len(current_lick) > 0:
                licks.append(pd.DataFrame(current_lick))
            current_lick = [df_sorted.iloc[i]]
    
    # Add the last lick
    if len(current_lick) > 0:
        licks.append(pd.DataFrame(current_lick))
    
    return licks


def find_lick_peaks(df, max_gap=5, peak_metric='y_max'):
    """
    Find peak position for each detected lick
    
    Parameters:
    -----------
    df : DataFrame with columns [Frame, X, Y, Probability]
    max_gap : int, maximum frame gap to still consider part of same lick
    peak_metric : str, how to define peak
        'y_max' - maximum Y position (furthest extension)
        'y_min' - minimum Y position
        'distance' - furthest from initial position
    
    Returns:
    --------
    DataFrame with peak positions for each lick
    """
    licks = detect_licks(df, max_gap)
    
    peak_data = []
    
    for lick_id, lick_df in enumerate(licks):
        if len(lick_df) == 0:
            continue
            
        if peak_metric == 'y_max':
            peak_idx = lick_df['Y'].idxmax()
        elif peak_metric == 'y_min':
            peak_idx = lick_df['Y'].idxmin()
        elif peak_metric == 'distance':
            # Distance from first position in lick
            first_x, first_y = lick_df.iloc[0]['X'], lick_df.iloc[0]['Y']
            distances = np.sqrt((lick_df['X'] - first_x)**2 + (lick_df['Y'] - first_y)**2)
            peak_idx = distances.idxmax()
        else:
            peak_idx = lick_df['Y'].idxmax()
        
        peak_row = lick_df.loc[peak_idx].copy()
        peak_data.append({
            'lick_id': lick_id,
            'frame': peak_row['Frame'],
            'x': peak_row['X'],
            'y': peak_row['Y'],
            'probability': peak_row['Probability'],
            'lick_duration': len(lick_df),
            'lick_start_frame': lick_df['Frame'].min(),
            'lick_end_frame': lick_df['Frame'].max()
        })
    
    return pd.DataFrame(peak_data)


def plot_lick_peaks(df, frame_range=None, max_gap=5, peak_metric='y_max', figsize=(12, 8)):
    """
    Plot only the peak positions for each detected lick
    
    Parameters:
    -----------
    df : DataFrame with columns [Frame, X, Y, Probability]
    frame_range : tuple (start_frame, end_frame) or None for all frames
    max_gap : int, maximum frame gap to still consider part of same lick
    peak_metric : str, how to define peak ('y_max', 'y_min', 'distance')
    figsize : tuple for figure size
    """
    if frame_range:
        df_subset = df[(df['Frame'] >= frame_range[0]) & (df['Frame'] <= frame_range[1])].copy()
    else:
        df_subset = df.copy()
    
    peaks = find_lick_peaks(df_subset, max_gap, peak_metric)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all trajectory in light gray
    ax.plot(df_subset['X'], df_subset['Y'], 'o', color='lightgray', 
            alpha=0.3, markersize=2, label='All positions')
    
    # Plot peak positions
    scatter = ax.scatter(peaks['x'], peaks['y'], 
                        c=peaks['lick_id'], 
                        cmap='rainbow', 
                        s=100, 
                        edgecolors='black',
                        linewidths=1.5,
                        alpha=0.8,
                        label='Lick peaks')
    
    # Annotate with lick numbers
    for idx, row in peaks.iterrows():
        ax.annotate(f"{int(row['lick_id'])}", 
                   (row['x'], row['y']),
                   xytext=(5, 5), 
                   textcoords='offset points',
                   fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Lick ID', rotation=270, labelpad=20)
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(f'Lick Peak Positions ({len(peaks)} licks detected, metric: {peak_metric})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    print(f"\nDetected {len(peaks)} licks")
    print(f"Average lick duration: {peaks['lick_duration'].mean():.1f} frames")
    
    return fig, peaks


def plot_trajectory(df, frame_range=None, figsize=(10, 8)):
    """
    Plot trajectory of tongue tip across frames
    
    Parameters:
    -----------
    df : DataFrame with columns [Frame, X, Y, Probability]
    frame_range : tuple (start_frame, end_frame) or None for all frames
    figsize : tuple for figure size
    """
    if frame_range:
        df_subset = df[(df['Frame'] >= frame_range[0]) & (df['Frame'] <= frame_range[1])].copy()
    else:
        df_subset = df.copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(df_subset['X'], df_subset['Y'], 
                        c=df_subset['Frame'], 
                        cmap='viridis', 
                        alpha=0.6, 
                        s=30)
    
    ax.plot(df_subset['X'], df_subset['Y'], 'k-', alpha=0.2, linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Frame', rotation=270, labelpad=20)
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(f'Tongue Tip Trajectory (Frames {df_subset["Frame"].min()}-{df_subset["Frame"].max()})')
    ax.invert_yaxis()  # Image coordinates: Y increases downward
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_heatmap(df, frame_range=None, bins=50, figsize=(10, 8)):
    """
    Create 2D heatmap showing density of tongue tip positions
    
    Parameters:
    -----------
    df : DataFrame with columns [Frame, X, Y, Probability]
    frame_range : tuple (start_frame, end_frame) or None for all frames
    bins : number of bins for histogram (int or tuple)
    figsize : tuple for figure size
    """
    if frame_range:
        df_subset = df[(df['Frame'] >= frame_range[0]) & (df['Frame'] <= frame_range[1])].copy()
    else:
        df_subset = df.copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create 2D histogram
    hist, xedges, yedges = np.histogram2d(
        df_subset['X'], df_subset['Y'], 
        bins=bins
    )
    
    # Plot heatmap
    im = ax.imshow(hist.T, origin='lower', 
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   cmap='hot', aspect='auto', interpolation='gaussian')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density', rotation=270, labelpad=20)
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(f'Tongue Tip Position Heatmap (Frames {df_subset["Frame"].min()}-{df_subset["Frame"].max()})')
    
    plt.tight_layout()
    return fig


def create_temporal_heatmap(df, frame_range=None, bins=50, figsize=(12, 8)):
    """
    Create heatmap showing earliest to latest frame positions with temporal coloring
    
    Parameters:
    -----------
    df : DataFrame with columns [Frame, X, Y, Probability]
    frame_range : tuple (start_frame, end_frame) or None for all frames
    bins : number of bins for spatial binning
    figsize : tuple for figure size
    """
    if frame_range:
        df_subset = df[(df['Frame'] >= frame_range[0]) & (df['Frame'] <= frame_range[1])].copy()
    else:
        df_subset = df.copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create spatial bins
    x_bins = np.linspace(df_subset['X'].min(), df_subset['X'].max(), bins)
    y_bins = np.linspace(df_subset['Y'].min(), df_subset['Y'].max(), bins)
    
    # Create 2D array for earliest frame at each position
    earliest_frame = np.full((bins-1, bins-1), np.nan)
    
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            mask = ((df_subset['X'] >= x_bins[i]) & (df_subset['X'] < x_bins[i+1]) &
                   (df_subset['Y'] >= y_bins[j]) & (df_subset['Y'] < y_bins[j+1]))
            if mask.any():
                earliest_frame[j, i] = df_subset.loc[mask, 'Frame'].min()
    
    # Plot heatmap
    im = ax.imshow(earliest_frame, origin='lower',
                   extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                   cmap='viridis', aspect='auto', interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Earliest Frame', rotation=270, labelpad=20)
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(f'Earliest Frame at Each Position (Frames {df_subset["Frame"].min()}-{df_subset["Frame"].max()})')
    
    plt.tight_layout()
    return fig


def create_latest_heatmap(df, frame_range=None, bins=50, figsize=(12, 8)):
    """
    Create heatmap showing latest frame positions with temporal coloring
    
    Parameters:
    -----------
    df : DataFrame with columns [Frame, X, Y, Probability]
    frame_range : tuple (start_frame, end_frame) or None for all frames
    bins : number of bins for spatial binning
    figsize : tuple for figure size
    """
    if frame_range:
        df_subset = df[(df['Frame'] >= frame_range[0]) & (df['Frame'] <= frame_range[1])].copy()
    else:
        df_subset = df.copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create spatial bins
    x_bins = np.linspace(df_subset['X'].min(), df_subset['X'].max(), bins)
    y_bins = np.linspace(df_subset['Y'].min(), df_subset['Y'].max(), bins)
    
    # Create 2D array for latest frame at each position
    latest_frame = np.full((bins-1, bins-1), np.nan)
    
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            mask = ((df_subset['X'] >= x_bins[i]) & (df_subset['X'] < x_bins[i+1]) &
                   (df_subset['Y'] >= y_bins[j]) & (df_subset['Y'] < y_bins[j+1]))
            if mask.any():
                latest_frame[j, i] = df_subset.loc[mask, 'Frame'].max()
    
    # Plot heatmap
    im = ax.imshow(latest_frame, origin='lower',
                   extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
                   cmap='plasma', aspect='auto', interpolation='nearest')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Latest Frame', rotation=270, labelpad=20)
    
    ax.set_xlabel('X Position (pixels)')
    ax.set_ylabel('Y Position (pixels)')
    ax.set_title(f'Latest Frame at Each Position (Frames {df_subset["Frame"].min()}-{df_subset["Frame"].max()})')
    
    plt.tight_layout()
    return fig


def plot_probability_distribution(df, frame_range=None, figsize=(10, 6)):
    """Plot probability distribution across frames"""
    if frame_range:
        df_subset = df[(df['Frame'] >= frame_range[0]) & (df['Frame'] <= frame_range[1])].copy()
    else:
        df_subset = df.copy()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Probability over frames
    ax1.plot(df_subset['Frame'], df_subset['Probability'], alpha=0.6)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Probability')
    ax1.set_title('Detection Probability Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Probability histogram
    ax2.hist(df_subset['Probability'], bins=30, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Probability')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Probability Distribution')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Example usage
    df = load_keypoints('tip.csv')
    
    print(f"Loaded {len(df)} frames")
    print(f"Frame range: {df['Frame'].min()} - {df['Frame'].max()}")
    print(f"X range: {df['X'].min():.1f} - {df['X'].max():.1f}")
    print(f"Y range: {df['Y'].min():.1f} - {df['Y'].max():.1f}")
    print(f"Mean probability: {df['Probability'].mean():.3f}")
    
    # Create visualizations
    fig1 = plot_trajectory(df, frame_range=(1030, 2000))
    fig2 = create_heatmap(df, frame_range=(1030, 2000))
    fig3 = create_temporal_heatmap(df, frame_range=(1030, 2000))
    fig4 = create_latest_heatmap(df, frame_range=(1030, 2000))
    
    plt.show()
