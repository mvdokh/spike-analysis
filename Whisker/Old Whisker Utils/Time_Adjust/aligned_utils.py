#!/usr/bin/env python3
"""
Comprehensive Spike Analysis Script
Creates raster plots, electrode schematics, and firing rate heatmaps
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load all data files"""
    print("Loading data files...")
    
    # Determine the correct path to data files
    import os
    current_dir = os.getcwd()
    
    # Check if we're in the Interval_Aligned_Spikes directory
    if os.path.basename(current_dir) == 'Interval_Aligned_Spikes':
        data_path = '.'
    else:
        # Assume we're in the parent directory
        data_path = 'Interval_Aligned_Spikes'
    
    # Load spikes data (time, unit, electrode)
    spikes_df = pd.read_csv(os.path.join(data_path, 'spikes_time_adjusted.csv'), 
                           names=['time', 'unit', 'electrode'])
    
    # Load pico intervals (Start, End)
    pico_df = pd.read_csv(os.path.join(data_path, 'pico.csv'))
    
    # Load electrode configuration (skip the first header line)
    electrode_df = pd.read_csv(os.path.join(data_path, 'electrode.cfg'), 
                              sep=' ', 
                              names=['poly', 'electrode', 'x', 'y'],
                              skiprows=1)
    # Remove any rows with NaN values
    electrode_df = electrode_df.dropna()
    
    # Convert timestamps to seconds (divide by 30,000)
    spikes_df['time_sec'] = spikes_df['time'] / 30000
    pico_df['Start_sec'] = pico_df['Start'] / 30000
    pico_df['End_sec'] = pico_df['End'] / 30000
    
    print(f"Loaded {len(spikes_df)} spikes across {len(spikes_df['electrode'].unique())} electrodes")
    print(f"Loaded {len(pico_df)} pico intervals")
    print(f"Time range: {spikes_df['time_sec'].min():.1f}s to {spikes_df['time_sec'].max():.1f}s")
    
    return spikes_df, pico_df, electrode_df

def create_raster_plot(spikes_df, pico_df, time_window=None, electrode_subset=None):
    """Create comprehensive raster plot with pico intervals"""
    print("Creating raster plot...")
    
    # Filter data if time window specified
    if time_window:
        spikes_plot = spikes_df[
            (spikes_df['time_sec'] >= time_window[0]) & 
            (spikes_df['time_sec'] <= time_window[1])
        ].copy()
        pico_plot = pico_df[
            (pico_df['Start_sec'] <= time_window[1]) & 
            (pico_df['End_sec'] >= time_window[0])
        ].copy()
    else:
        spikes_plot = spikes_df.copy()
        pico_plot = pico_df.copy()
    
    # Filter electrodes if specified
    if electrode_subset:
        spikes_plot = spikes_plot[spikes_plot['electrode'].isin(electrode_subset)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot pico intervals as shaded regions
    for _, interval in pico_plot.iterrows():
        ax.axvspan(interval['Start_sec'], interval['End_sec'], 
                  alpha=0.2, color='red', label='Pico Activation' if _ == 0 else "")
    
    # Get unique electrodes and sort them
    electrodes = sorted(spikes_plot['electrode'].unique())
    electrode_colors = plt.cm.tab20(np.linspace(0, 1, len(electrodes)))
    
    # Plot spikes for each electrode
    for i, electrode in enumerate(electrodes):
        electrode_spikes = spikes_plot[spikes_plot['electrode'] == electrode]
        y_positions = np.full(len(electrode_spikes), electrode)
        
        ax.scatter(electrode_spikes['time_sec'], y_positions, 
                  s=1, alpha=0.7, color=electrode_colors[i % len(electrode_colors)])
    
    # Formatting
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Electrode Number', fontsize=12)
    ax.set_title('Spike Raster Plot with Pico Activation Intervals', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Set y-axis to show all electrodes
    if electrodes:
        ax.set_ylim(min(electrodes) - 1, max(electrodes) + 1)
    
    # Add legend for pico intervals
    if len(pico_plot) > 0:
        ax.legend(['Pico Activation'], loc='upper right')
    
    plt.tight_layout()
    return fig

def create_electrode_schematic(electrode_df):
    """Create electrode schematic with labels"""
    print("Creating electrode schematic...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot electrode positions
    for _, row in electrode_df.iterrows():
        x, y = row['x'], row['y']
        electrode_num = row['electrode']
        
        # Plot electrode as circle
        circle = plt.Circle((x, y), 15, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        
        # Add electrode number label
        ax.text(x, y, str(int(electrode_num)), ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    margin = 50
    x_min, x_max = electrode_df['x'].min() - margin, electrode_df['x'].max() + margin
    y_min, y_max = electrode_df['y'].min() - margin, electrode_df['y'].max() + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Labels and title
    ax.set_xlabel('X Position (μm)', fontsize=12)
    ax.set_ylabel('Y Position (μm)', fontsize=12)
    ax.set_title('Electrode Array Schematic', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add hemisphere labels
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(-15, y_max-20, 'Left', fontsize=12, fontweight='bold', color='red')
    ax.text(5, y_max-20, 'Right', fontsize=12, fontweight='bold', color='red')
    
    plt.tight_layout()
    return fig

def create_firing_heatmap(spikes_df, electrode_df):
    """Create heatmap showing firing rates for each electrode"""
    print("Creating firing rate heatmap...")
    
    # Calculate firing rates
    total_time = (spikes_df['time_sec'].max() - spikes_df['time_sec'].min())
    firing_rates = spikes_df.groupby('electrode').size() / total_time
    
    # Merge with electrode positions
    electrode_with_rates = electrode_df.merge(
        firing_rates.reset_index().rename(columns={0: 'firing_rate'}),
        on='electrode', how='left'
    )
    electrode_with_rates['firing_rate'] = electrode_with_rates['firing_rate'].fillna(0)
    
    # Create two visualizations: bar plot and spatial heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot of firing rates
    sorted_data = electrode_with_rates.sort_values('firing_rate', ascending=False)
    bars = ax1.bar(range(len(sorted_data)), sorted_data['firing_rate'], 
                   color=plt.cm.plasma(sorted_data['firing_rate'] / sorted_data['firing_rate'].max()))
    ax1.set_xlabel('Electrode (sorted by firing rate)', fontsize=12)
    ax1.set_ylabel('Firing Rate (spikes/second)', fontsize=12)
    ax1.set_title('Firing Rates by Electrode', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add electrode numbers as labels on bars
    for i, (bar, electrode) in enumerate(zip(bars, sorted_data['electrode'])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(electrode)}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Spatial heatmap
    scatter = ax2.scatter(electrode_with_rates['x'], electrode_with_rates['y'], 
                         c=electrode_with_rates['firing_rate'], s=300, 
                         cmap='plasma', alpha=0.8, edgecolors='black', linewidth=2)
    
    # Add electrode labels
    for _, row in electrode_with_rates.iterrows():
        ax2.text(row['x'], row['y'], str(int(row['electrode'])), 
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax2.set_xlabel('X Position (μm)', fontsize=12)
    ax2.set_ylabel('Y Position (μm)', fontsize=12)
    ax2.set_title('Spatial Distribution of Firing Rates', fontsize=14, fontweight='bold')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Add hemisphere divider
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Firing Rate (spikes/second)', fontsize=12)
    
    plt.tight_layout()
    
    # Print statistics
    print(f"\nFiring Rate Statistics:")
    print(f"Mean firing rate: {firing_rates.mean():.2f} spikes/second")
    print(f"Max firing rate: {firing_rates.max():.2f} spikes/second (Electrode {firing_rates.idxmax()})")
    print(f"Min firing rate: {firing_rates.min():.2f} spikes/second (Electrode {firing_rates.idxmin()})")
    
    return fig, electrode_with_rates

def create_summary_statistics(spikes_df, pico_df):
    """Create summary statistics and temporal analysis"""
    print("Generating summary statistics...")
    
    total_time = (spikes_df['time_sec'].max() - spikes_df['time_sec'].min())
    total_spikes = len(spikes_df)
    n_electrodes = len(spikes_df['electrode'].unique())
    n_units = len(spikes_df['unit'].unique())
    
    print(f"\n{'='*50}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*50}")
    print(f"Recording duration: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Total spikes: {total_spikes:,}")
    print(f"Number of electrodes: {n_electrodes}")
    print(f"Number of units: {n_units}")
    print(f"Overall firing rate: {total_spikes/total_time:.2f} spikes/second")
    print(f"Pico intervals: {len(pico_df)}")
    
    if len(pico_df) > 0:
        pico_duration = (pico_df['End_sec'] - pico_df['Start_sec']).mean()
        pico_interval = pico_df['Start_sec'].diff().mean()
        print(f"Average pico duration: {pico_duration:.3f} seconds")
        print(f"Average pico interval: {pico_interval:.1f} seconds")

def create_raster_plot_custom(spikes_df, pico_df, time_start=None, time_end=None, electrode_subset=None, save_path=None):
    """
    Create raster plot with custom time range
    
    Parameters:
    -----------
    spikes_df : pd.DataFrame
        Spike data with time_sec column
    pico_df : pd.DataFrame 
        Pico intervals with Start_sec and End_sec columns
    time_start : float, optional
        Start time in seconds. If None, uses minimum time in data
    time_end : float, optional
        End time in seconds. If None, uses maximum time in data
    electrode_subset : list, optional
        List of electrode numbers to include. If None, includes all
    save_path : str, optional
        Path to save the plot. If None, doesn't save
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    print(f"Creating raster plot for time range: {time_start}s to {time_end}s")
    
    # Use full time range if not specified
    if time_start is None:
        time_start = spikes_df['time_sec'].min()
    if time_end is None:
        time_end = spikes_df['time_sec'].max()
    
    # Filter data by time window
    spikes_plot = spikes_df[
        (spikes_df['time_sec'] >= time_start) & 
        (spikes_df['time_sec'] <= time_end)
    ].copy()
    
    pico_plot = pico_df[
        (pico_df['Start_sec'] <= time_end) & 
        (pico_df['End_sec'] >= time_start)
    ].copy()
    
    # Filter electrodes if specified
    if electrode_subset:
        spikes_plot = spikes_plot[spikes_plot['electrode'].isin(electrode_subset)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot pico intervals as shaded regions
    for i, interval in pico_plot.iterrows():
        # Clip intervals to the visible time window
        start_clipped = max(interval['Start_sec'], time_start)
        end_clipped = min(interval['End_sec'], time_end)
        
        ax.axvspan(start_clipped, end_clipped, 
                  alpha=0.2, color='red', label='Pico Activation' if i == 0 else "")
    
    # Get unique electrodes and sort them
    electrodes = sorted(spikes_plot['electrode'].unique())
    electrode_colors = plt.cm.tab20(np.linspace(0, 1, len(electrodes)))
    
    # Plot spikes for each electrode
    for i, electrode in enumerate(electrodes):
        electrode_spikes = spikes_plot[spikes_plot['electrode'] == electrode]
        y_positions = np.full(len(electrode_spikes), electrode)
        
        ax.scatter(electrode_spikes['time_sec'], y_positions, 
                  s=1, alpha=0.7, color=electrode_colors[i % len(electrode_colors)])
    
    # Formatting
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Electrode Number', fontsize=12)
    ax.set_title(f'Spike Raster Plot ({time_start:.1f}s - {time_end:.1f}s)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(time_start, time_end)
    
    # Set y-axis to show all electrodes
    if electrodes:
        ax.set_ylim(min(electrodes) - 1, max(electrodes) + 1)
    
    # Add legend for pico intervals
    if len(pico_plot) > 0:
        ax.legend(['Pico Activation'], loc='upper right')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def run_full_analysis(time_start=None, time_end=None, save_plots=True):
    """
    Run complete analysis with optional time window
    
    Parameters:
    -----------
    time_start : float, optional
        Start time in seconds for raster plot
    time_end : float, optional  
        End time in seconds for raster plot
    save_plots : bool
        Whether to save plots to files
        
    Returns:
    --------
    dict : Dictionary containing all generated figures and data
    """
    print("Starting comprehensive spike analysis...")
    
    # Load data
    spikes_df, pico_df, electrode_df = load_data()
    
    # Create summary statistics
    create_summary_statistics(spikes_df, pico_df)
    
    # Create custom raster plot
    time_label = ""
    if time_start is not None and time_end is not None:
        time_label = f"_{time_start}s_{time_end}s"
    elif time_start is not None:
        time_label = f"_from_{time_start}s"
    elif time_end is not None:
        time_label = f"_to_{time_end}s"
    
    raster_filename = f'raster_plot{time_label}.png' if save_plots else None
    fig_raster = create_raster_plot_custom(spikes_df, pico_df, time_start, time_end, save_path=raster_filename)
    
    # Create electrode schematic
    fig_schematic = create_electrode_schematic(electrode_df)
    if save_plots:
        fig_schematic.savefig('electrode_schematic.png', dpi=300, bbox_inches='tight')
        print("Saved: electrode_schematic.png")
    
    # Create firing rate heatmap
    fig_heatmap, electrode_rates = create_firing_heatmap(spikes_df, electrode_df)
    if save_plots:
        fig_heatmap.savefig('firing_rate_heatmap.png', dpi=300, bbox_inches='tight')
        print("Saved: firing_rate_heatmap.png")
        electrode_rates.to_csv('electrode_firing_rates.csv', index=False)
        print("Saved: electrode_firing_rates.csv")
    
    results = {
        'spikes_df': spikes_df,
        'pico_df': pico_df,
        'electrode_df': electrode_df,
        'electrode_rates': electrode_rates,
        'fig_raster': fig_raster,
        'fig_schematic': fig_schematic,
        'fig_heatmap': fig_heatmap
    }
    
    if save_plots:
        print(f"\n{'='*50}")
        print("Analysis complete! Generated files:")
        if raster_filename:
            print(f"- {raster_filename}")
        print("- electrode_schematic.png")
        print("- firing_rate_heatmap.png")
        print("- electrode_firing_rates.csv")
        print(f"{'='*50}")
    
    return results


# Example usage functions for interactive use
def quick_raster(time_start, time_end, electrodes=None):
    """Quick function to create a raster plot for a specific time range"""
    spikes_df, pico_df, _ = load_data()
    return create_raster_plot_custom(spikes_df, pico_df, time_start, time_end, electrodes)

def get_time_range_info(spikes_df=None):
    """Get information about available time ranges"""
    if spikes_df is None:
        spikes_df, _, _ = load_data()
    
    print(f"Available time range: {spikes_df['time_sec'].min():.1f}s to {spikes_df['time_sec'].max():.1f}s")
    print(f"Total duration: {spikes_df['time_sec'].max() - spikes_df['time_sec'].min():.1f}s")
    return spikes_df['time_sec'].min(), spikes_df['time_sec'].max()


if __name__ == "__main__":
    # Only run if called directly, not when imported
    run_full_analysis()