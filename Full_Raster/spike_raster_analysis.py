#!/usr/bin/env python3
"""
Spike Raster Plot Analysis
Creates a raster plot of neural spike data sorted by electrode depth,
along with electrode configuration visualization and spike count heatmap.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def load_electrode_config(config_file):
    """Load electrode configuration from .cfg file"""
    electrodes = []
    with open(config_file, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines and parse electrode data
    for line in lines[1:]:  # Skip 'poly2' header
        line = line.strip()
        if line and not line.startswith('#'):
            parts = line.split()
            if len(parts) >= 4:
                electrode_num = int(parts[0])
                electrode_num2 = int(parts[1])  # Same as first column
                x_pos = float(parts[2])  # Left/right position
                depth = float(parts[3])  # Depth
                electrodes.append({
                    'electrode': electrode_num,
                    'x_pos': x_pos,
                    'depth': depth
                })
    
    return pd.DataFrame(electrodes)

def load_spike_data(spike_file):
    """Load spike data from CSV file"""
    # Load data with proper column names
    spike_data = pd.read_csv(spike_file, header=None, 
                           names=['time', 'unit', 'electrode'])
    
    # Convert time from samples to seconds (sampling rate = 30kHz)
    # The time column appears to already be in seconds based on the values
    # If it were in samples, we would divide by 30000
    
    return spike_data

def create_raster_plot(spike_data, electrode_config, time_window=None):
    """Create raster plot sorted by electrode depth"""
    
    # Merge spike data with electrode configuration
    merged_data = spike_data.merge(electrode_config, on='electrode', how='left')
    
    # Sort by depth (shallow to deep)
    merged_data = merged_data.sort_values('depth')
    
    # Apply time window if specified
    if time_window:
        merged_data = merged_data[
            (merged_data['time'] >= time_window[0]) & 
            (merged_data['time'] <= time_window[1])
        ]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Create gridspec for layout
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[4, 1, 1])
    
    # Main raster plot
    ax_raster = fig.add_subplot(gs[0, 0])
    
    # Get unique depths and create y-axis mapping
    unique_depths = sorted(merged_data['depth'].unique())
    depth_to_y = {depth: i for i, depth in enumerate(unique_depths)}
    
    # Plot spikes for each electrode/depth
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_depths)))
    
    for i, depth in enumerate(unique_depths):
        depth_data = merged_data[merged_data['depth'] == depth]
        if not depth_data.empty:
            # Get all units at this depth
            units_at_depth = depth_data['unit'].unique()
            
            for j, unit in enumerate(units_at_depth):
                unit_data = depth_data[depth_data['unit'] == unit]
                y_pos = i + j * 0.1  # Slight offset for multiple units at same depth
                
                ax_raster.scatter(unit_data['time'], [y_pos] * len(unit_data), 
                                s=1, c=[colors[i]], alpha=0.7, marker='|')
    
    ax_raster.set_xlabel('Time (seconds)', fontsize=12)
    ax_raster.set_ylabel('Electrode Depth (μm)', fontsize=12)
    ax_raster.set_title('Neural Spike Raster Plot (Sorted by Electrode Depth)', 
                       fontsize=14, fontweight='bold')
    
    # Set y-axis labels to show actual depths
    ax_raster.set_yticks(range(len(unique_depths)))
    ax_raster.set_yticklabels([f'{int(d)}' for d in unique_depths])
    ax_raster.grid(True, alpha=0.3)
    
    # Electrode schematic
    ax_electrode = fig.add_subplot(gs[0, 1])
    
    # Plot electrode positions
    for _, electrode in electrode_config.iterrows():
        x = electrode['x_pos']
        y = electrode['depth']
        
        # Color based on x position (left vs right)
        color = 'red' if x > 0 else 'blue'
        ax_electrode.scatter(x, y, s=100, c=color, alpha=0.7, edgecolors='black')
        ax_electrode.text(x + 2, y, str(int(electrode['electrode'])), 
                         fontsize=8, ha='left', va='center')
    
    ax_electrode.set_xlabel('X Position (μm)', fontsize=10)
    ax_electrode.set_ylabel('Depth (μm)', fontsize=10)
    ax_electrode.set_title('Electrode Layout', fontsize=12, fontweight='bold')
    ax_electrode.grid(True, alpha=0.3)
    ax_electrode.invert_yaxis()  # Depth increases downward
    
    # Spike count heatmap
    ax_heatmap = fig.add_subplot(gs[0, 2])
    
    # Calculate spike counts per electrode
    spike_counts = merged_data.groupby(['electrode', 'depth']).size().reset_index(name='spike_count')
    
    # Create heatmap data
    heatmap_data = []
    for _, electrode in electrode_config.iterrows():
        electrode_num = electrode['electrode']
        spike_count = spike_counts[spike_counts['electrode'] == electrode_num]['spike_count']
        count = spike_count.iloc[0] if len(spike_count) > 0 else 0
        heatmap_data.append(count)
    
    # Sort by depth for consistent ordering
    electrode_config_sorted = electrode_config.sort_values('depth')
    heatmap_data_sorted = []
    for _, electrode in electrode_config_sorted.iterrows():
        electrode_num = electrode['electrode']
        spike_count = spike_counts[spike_counts['electrode'] == electrode_num]['spike_count']
        count = spike_count.iloc[0] if len(spike_count) > 0 else 0
        heatmap_data_sorted.append(count)
    
    # Create heatmap
    heatmap_array = np.array(heatmap_data_sorted).reshape(-1, 1)
    im = ax_heatmap.imshow(heatmap_array, cmap='YlOrRd', aspect='auto')
    
    ax_heatmap.set_title('Spike Count\nHeatmap', fontsize=12, fontweight='bold')
    ax_heatmap.set_ylabel('Electrode (by depth)', fontsize=10)
    ax_heatmap.set_yticks(range(len(electrode_config_sorted)))
    ax_heatmap.set_yticklabels([f"E{int(e)}" for e in electrode_config_sorted['electrode']])
    ax_heatmap.set_xticks([])
    
    # Add colorbar
    plt.colorbar(im, ax=ax_heatmap, shrink=0.8)
    
    # Summary statistics in bottom subplot
    ax_stats = fig.add_subplot(gs[1, :])
    ax_stats.axis('off')
    
    # Calculate statistics
    total_spikes = len(merged_data)
    unique_units = len(merged_data['unit'].unique())
    unique_electrodes = len(merged_data['electrode'].unique())
    time_range = merged_data['time'].max() - merged_data['time'].min()
    avg_firing_rate = total_spikes / time_range if time_range > 0 else 0
    
    # Top 5 most active electrodes
    top_electrodes = spike_counts.nlargest(5, 'spike_count')
    
    stats_text = f"""
    ANALYSIS SUMMARY:
    • Total Spikes: {total_spikes:,}
    • Unique Units: {unique_units}
    • Active Electrodes: {unique_electrodes}
    • Recording Duration: {time_range:.2f} seconds
    • Average Firing Rate: {avg_firing_rate:.2f} spikes/second
    
    TOP 5 MOST ACTIVE ELECTRODES:
    """
    
    for i, (_, row) in enumerate(top_electrodes.iterrows(), 1):
        electrode_num = int(row['electrode'])
        depth = row['depth']
        count = row['spike_count']
        stats_text += f"    {i}. Electrode {electrode_num} (depth: {depth:.0f}μm): {count} spikes\n"
    
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    return fig, merged_data

def main():
    """Main function to run the analysis"""
    print("Loading spike data and electrode configuration...")
    
    # Load data
    electrode_config = load_electrode_config('electrode.cfg')
    spike_data = load_spike_data('spikes.csv')
    
    print(f"Loaded {len(spike_data)} spikes from {len(spike_data['unit'].unique())} units")
    print(f"Electrode configuration: {len(electrode_config)} electrodes")
    
    # Create raster plot for first 10 seconds (adjust as needed)
    print("Creating raster plot...")
    fig, merged_data = create_raster_plot(spike_data, electrode_config, 
                                        time_window=(0, 10))
    
    # Save the plot
    output_file = 'spike_raster_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Raster plot saved as: {output_file}")
    
    # Create additional analysis for full dataset
    print("\nCreating full dataset analysis...")
    fig_full, _ = create_raster_plot(spike_data, electrode_config)
    
    output_file_full = 'spike_raster_analysis_full.png'
    plt.savefig(output_file_full, dpi=300, bbox_inches='tight')
    print(f"Full dataset analysis saved as: {output_file_full}")
    
    # Show the plots
    plt.show()
    
    # Additional analysis
    print("\nDETAILED ANALYSIS:")
    print("="*50)
    
    # Electrode depth analysis
    merged_data = spike_data.merge(electrode_config, on='electrode', how='left')
    depth_analysis = merged_data.groupby('depth').agg({
        'time': 'count',
        'unit': 'nunique',
        'electrode': 'nunique'
    }).rename(columns={'time': 'spike_count', 'unit': 'unique_units', 'electrode': 'electrodes'})
    
    print("\nSPIKE ACTIVITY BY DEPTH:")
    for depth, row in depth_analysis.iterrows():
        print(f"Depth {depth:6.0f}μm: {row['spike_count']:6d} spikes, "
              f"{row['unique_units']:2d} units, {row['electrodes']:2d} electrodes")
    
    # Unit analysis
    unit_analysis = merged_data.groupby('unit').agg({
        'time': 'count',
        'electrode': 'nunique',
        'depth': lambda x: x.iloc[0]
    }).rename(columns={'time': 'spike_count', 'electrode': 'electrode_count'})
    
    print(f"\nTOP 10 MOST ACTIVE UNITS:")
    top_units = unit_analysis.nlargest(10, 'spike_count')
    for unit, row in top_units.iterrows():
        print(f"Unit {unit:3d}: {row['spike_count']:6d} spikes, "
              f"depth {row['depth']:6.0f}μm, {row['electrode_count']} electrode(s)")

if __name__ == "__main__":
    main()
