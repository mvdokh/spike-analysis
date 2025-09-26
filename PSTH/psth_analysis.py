"""
PSTH (Peri-Stimulus Time Histogram) Analysis Script
=================================================

This script provides comprehensive PSTH analysis for neural spike data around stimulus intervals.
It supports configurable time windows, unit selection, and extensive visualization options.

Author: Generated for spike-analysis project
Date: September 26, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Tuple, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class PSTHAnalyzer:
    """
    A comprehensive class for PSTH analysis of neural spike data.
    """
    
    def __init__(self, spikes_file: str = 'spikes_time_adjusted.csv', 
                 intervals_file: str = 'pico.csv'):
        """
        Initialize the PSTH analyzer.
        
        Parameters:
        -----------
        spikes_file : str
            Path to CSV file with spike data (time, unit, channel)
        intervals_file : str
            Path to CSV file with stimulus intervals (Start, End)
        """
        self.spikes_file = spikes_file
        self.intervals_file = intervals_file
        self.spikes_df = None
        self.intervals_df = None
        self.psth_data = {}
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load spike and interval data from CSV files."""
        print("Loading data...")
        
        # Load spikes data
        self.spikes_df = pd.read_csv(self.spikes_file, names=['time', 'unit', 'channel'])
        print(f"Loaded {len(self.spikes_df)} spikes")
        
        # Load intervals data
        self.intervals_df = pd.read_csv(self.intervals_file)
        print(f"Loaded {len(self.intervals_df)} intervals")
        
        # Get data summary
        self.n_units = self.spikes_df['unit'].nunique()
        self.n_channels = self.spikes_df['channel'].nunique()
        self.units = sorted(self.spikes_df['unit'].unique())
        self.channels = sorted(self.spikes_df['channel'].unique())
        
        print(f"Data summary: {self.n_units} units, {self.n_channels} channels")
        print(f"Time range: {self.spikes_df['time'].min()} - {self.spikes_df['time'].max()}")
        
    def compute_psth(self, pre_time: float = 1000, post_time: float = 2000, 
                     bin_size: float = 50, units: Optional[List[int]] = None,
                     smoothing_window: Optional[int] = None) -> Dict:
        """
        Compute PSTH for specified units around stimulus intervals.
        
        Parameters:
        -----------
        pre_time : float
            Time before stimulus onset (ms)
        post_time : float
            Time after stimulus onset (ms)
        bin_size : float
            Bin size for histogram (ms)
        units : List[int], optional
            Specific units to analyze. If None, analyze all units.
        smoothing_window : int, optional
            Window size for smoothing (number of bins)
            
        Returns:
        --------
        Dict with PSTH data for each unit
        """
        if units is None:
            units = self.units
            
        print(f"Computing PSTH for {len(units)} units...")
        print(f"Window: -{pre_time}ms to +{post_time}ms, bin size: {bin_size}ms")
        
        # Create time bins
        time_bins = np.arange(-pre_time, post_time + bin_size, bin_size)
        bin_centers = time_bins[:-1] + bin_size/2
        
        psth_results = {}
        
        for unit in units:
            unit_spikes = self.spikes_df[self.spikes_df['unit'] == unit]['time'].values
            
            # Initialize arrays to store aligned spike times for all intervals
            all_aligned_spikes = []
            valid_intervals = 0
            
            # For each interval, find spikes within the window
            for _, interval in self.intervals_df.iterrows():
                interval_start = interval['Start']
                
                # Find spikes within the analysis window around this interval
                window_spikes = unit_spikes[
                    (unit_spikes >= interval_start - pre_time) &
                    (unit_spikes <= interval_start + post_time)
                ]
                
                if len(window_spikes) > 0:
                    # Align spikes to interval start (time 0)
                    aligned_spikes = window_spikes - interval_start
                    all_aligned_spikes.extend(aligned_spikes)
                    valid_intervals += 1
            
            # Create histogram
            if len(all_aligned_spikes) > 0:
                hist, _ = np.histogram(all_aligned_spikes, bins=time_bins)
                
                # Convert to firing rate (spikes/sec)
                firing_rate = hist / (bin_size / 1000.0) / valid_intervals
                
                # Apply smoothing if requested
                if smoothing_window and smoothing_window > 1:
                    from scipy.ndimage import uniform_filter1d
                    firing_rate = uniform_filter1d(firing_rate, size=smoothing_window)
                
            else:
                firing_rate = np.zeros(len(bin_centers))
            
            psth_results[unit] = {
                'firing_rate': firing_rate,
                'bin_centers': bin_centers,
                'total_spikes': len(all_aligned_spikes),
                'valid_intervals': valid_intervals,
                'raw_spikes': all_aligned_spikes
            }
            
        self.psth_data = psth_results
        return psth_results
    
    def plot_psth_single_unit(self, unit: int, ax: Optional[plt.Axes] = None, 
                              show_raster: bool = True) -> plt.Axes:
        """
        Plot PSTH for a single unit.
        
        Parameters:
        -----------
        unit : int
            Unit to plot
        ax : matplotlib.Axes, optional
            Axes to plot on
        show_raster : bool
            Whether to show raster plot above PSTH
            
        Returns:
        --------
        matplotlib.Axes
        """
        if unit not in self.psth_data:
            raise ValueError(f"Unit {unit} not found in PSTH data. Run compute_psth first.")
        
        data = self.psth_data[unit]
        
        # Calculate average stimulus duration from intervals
        interval_durations = self.intervals_df['End'] - self.intervals_df['Start']
        avg_stimulus_duration = interval_durations.mean()
        
        if show_raster and len(data['raw_spikes']) > 0:
            # Create figure with subplots for raster and PSTH
            fig, (ax_raster, ax_psth) = plt.subplots(2, 1, figsize=(12, 8), 
                                                    height_ratios=[1, 2], 
                                                    sharex=True)
            
            # Sample a subset of intervals for raster (to avoid overcrowding)
            max_trials = 50
            intervals_to_plot = min(max_trials, data['valid_intervals'])
            
            current_trial = 0
            
            for _, interval in self.intervals_df.iterrows():
                if current_trial >= intervals_to_plot:
                    break
                    
                interval_start = interval['Start']
                interval_duration = interval['End'] - interval['Start']
                unit_spikes = self.spikes_df[self.spikes_df['unit'] == unit]['time'].values
                
                # Use the actual analysis window from bin_centers
                window_start = data['bin_centers'][0]
                window_end = data['bin_centers'][-1]
                
                window_spikes = unit_spikes[
                    (unit_spikes >= interval_start + window_start) &
                    (unit_spikes <= interval_start + window_end)
                ]
                
                if len(window_spikes) > 0:
                    aligned_spikes = window_spikes - interval_start
                    ax_raster.scatter(aligned_spikes, [current_trial] * len(aligned_spikes), 
                                    s=1, c='black', alpha=0.7)
                    current_trial += 1
            
            # Add stimulus period highlighting to raster
            ax_raster.axvspan(0, avg_stimulus_duration, alpha=0.2, color='red', 
                            label=f'Stimulus ({avg_stimulus_duration:.0f}ms)')
            ax_raster.set_xlim(data['bin_centers'][0], data['bin_centers'][-1])
            ax_raster.set_ylabel('Trial #')
            ax_raster.set_title(f'Unit {unit} - Raster Plot ({current_trial} trials)')
            ax_raster.grid(True, alpha=0.3)
            ax_raster.legend()
            
            # Use the PSTH subplot
            ax = ax_psth
        else:
            # Create single plot if no raster
            if ax is None:
                fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot PSTH
        ax.plot(data['bin_centers'], data['firing_rate'], 'b-', linewidth=2)
        ax.fill_between(data['bin_centers'], data['firing_rate'], alpha=0.3)
        
        # Add stimulus period as highlighted region instead of just onset line
        ax.axvspan(0, avg_stimulus_duration, alpha=0.2, color='red', 
                  label=f'Stimulus period ({avg_stimulus_duration:.0f}ms)')
        
        # Add onset and offset lines for clarity
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=avg_stimulus_duration, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Time relative to stimulus onset (ms)')
        ax.set_ylabel('Firing rate (spikes/s)')
        ax.set_title(f'Unit {unit} - PSTH ({data["valid_intervals"]} trials, {data["total_spikes"]} spikes)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def plot_psth_grid(self, units: Optional[List[int]] = None, 
                       ncols: int = 4, figsize: Tuple[int, int] = (16, 12),
                       show_statistics: bool = True) -> plt.Figure:
        """
        Plot PSTH for multiple units in a grid layout.
        
        Parameters:
        -----------
        units : List[int], optional
            Units to plot. If None, plot all units.
        ncols : int
            Number of columns in the grid
        figsize : Tuple[int, int]
            Figure size
        show_statistics : bool
            Whether to show statistical information
            
        Returns:
        --------
        matplotlib.Figure
        """
        if units is None:
            units = list(self.psth_data.keys())
            
        nrows = (len(units) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        
        if nrows == 1:
            axes = axes.reshape(1, -1)
        
        # Calculate average stimulus duration
        interval_durations = self.intervals_df['End'] - self.intervals_df['Start']
        avg_stimulus_duration = interval_durations.mean()
        
        for i, unit in enumerate(units):
            row, col = i // ncols, i % ncols
            ax = axes[row, col]
            
            if unit in self.psth_data:
                data = self.psth_data[unit]
                ax.plot(data['bin_centers'], data['firing_rate'], 'b-', linewidth=1.5)
                ax.fill_between(data['bin_centers'], data['firing_rate'], alpha=0.3)
                
                # Add stimulus period highlighting
                ax.axvspan(0, avg_stimulus_duration, alpha=0.2, color='red')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
                ax.axvline(x=avg_stimulus_duration, color='red', linestyle='--', alpha=0.7, linewidth=0.8)
                
                title = f'Unit {unit}'
                if show_statistics:
                    max_rate = np.max(data['firing_rate'])
                    title += f'\nMax: {max_rate:.1f} Hz'
                
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Rate (Hz)')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(units), nrows * ncols):
            row, col = i // ncols, i % ncols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_population_psth(self, units: Optional[List[int]] = None,
                           method: str = 'mean') -> plt.Figure:
        """
        Plot population PSTH (average across units).
        
        Parameters:
        -----------
        units : List[int], optional
            Units to include. If None, use all units.
        method : str
            Method for combining units ('mean', 'sum')
            
        Returns:
        --------
        matplotlib.Figure
        """
        if units is None:
            units = list(self.psth_data.keys())
            
        # Collect firing rates from all units
        all_rates = []
        bin_centers = None
        
        for unit in units:
            if unit in self.psth_data:
                data = self.psth_data[unit]
                all_rates.append(data['firing_rate'])
                if bin_centers is None:
                    bin_centers = data['bin_centers']
        
        all_rates = np.array(all_rates)
        
        if method == 'mean':
            pop_rate = np.mean(all_rates, axis=0)
            sem = stats.sem(all_rates, axis=0)
        elif method == 'sum':
            pop_rate = np.sum(all_rates, axis=0)
            sem = None
        
        # Calculate average stimulus duration
        interval_durations = self.intervals_df['End'] - self.intervals_df['Start']
        avg_stimulus_duration = interval_durations.mean()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(bin_centers, pop_rate, 'b-', linewidth=2, label=f'Population {method}')
        
        if sem is not None:
            ax.fill_between(bin_centers, pop_rate - sem, pop_rate + sem, 
                          alpha=0.3, label='SEM')
        
        # Add stimulus period as highlighted region
        ax.axvspan(0, avg_stimulus_duration, alpha=0.2, color='red', 
                  label=f'Stimulus period ({avg_stimulus_duration:.0f}ms)')
        
        # Add onset and offset lines
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.axvline(x=avg_stimulus_duration, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Time relative to stimulus onset (ms)')
        ax.set_ylabel('Firing rate (spikes/s)')
        ax.set_title(f'Population PSTH ({len(units)} units)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    def analyze_channels(self, pre_time: float = 1000, post_time: float = 2000,
                        bin_size: float = 50) -> Dict:
        """
        Analyze spike activity by channel.
        
        Parameters:
        -----------
        pre_time : float
            Time before stimulus onset (ms)
        post_time : float
            Time after stimulus onset (ms)
        bin_size : float
            Bin size for histogram (ms)
            
        Returns:
        --------
        Dict with channel analysis results
        """
        print("Analyzing channels...")
        
        time_bins = np.arange(-pre_time, post_time + bin_size, bin_size)
        bin_centers = time_bins[:-1] + bin_size/2
        
        channel_results = {}
        
        for channel in self.channels:
            channel_spikes = self.spikes_df[self.spikes_df['channel'] == channel]['time'].values
            
            all_aligned_spikes = []
            valid_intervals = 0
            
            for _, interval in self.intervals_df.iterrows():
                interval_start = interval['Start']
                
                window_spikes = channel_spikes[
                    (channel_spikes >= interval_start - pre_time) &
                    (channel_spikes <= interval_start + post_time)
                ]
                
                if len(window_spikes) > 0:
                    aligned_spikes = window_spikes - interval_start
                    all_aligned_spikes.extend(aligned_spikes)
                    valid_intervals += 1
            
            if len(all_aligned_spikes) > 0:
                hist, _ = np.histogram(all_aligned_spikes, bins=time_bins)
                firing_rate = hist / (bin_size / 1000.0) / valid_intervals
            else:
                firing_rate = np.zeros(len(bin_centers))
            
            # Get units on this channel
            units_on_channel = sorted(self.spikes_df[self.spikes_df['channel'] == channel]['unit'].unique())
            
            channel_results[channel] = {
                'firing_rate': firing_rate,
                'bin_centers': bin_centers,
                'total_spikes': len(all_aligned_spikes),
                'valid_intervals': valid_intervals,
                'units': units_on_channel,
                'n_units': len(units_on_channel)
            }
        
        return channel_results
    
    def plot_channel_analysis(self, channel_data: Dict) -> plt.Figure:
        """
        Plot channel-based analysis.
        
        Parameters:
        -----------
        channel_data : Dict
            Channel analysis results from analyze_channels()
            
        Returns:
        --------
        matplotlib.Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Calculate average stimulus duration
        interval_durations = self.intervals_df['End'] - self.intervals_df['Start']
        avg_stimulus_duration = interval_durations.mean()
        
        # 1. Individual channel PSTHs
        ax1 = axes[0, 0]
        for channel, data in channel_data.items():
            ax1.plot(data['bin_centers'], data['firing_rate'], 
                    label=f'Ch {channel} ({data["n_units"]} units)', alpha=0.7)
        
        # Add stimulus period highlighting
        ax1.axvspan(0, avg_stimulus_duration, alpha=0.2, color='red', 
                   label=f'Stimulus ({avg_stimulus_duration:.0f}ms)')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.axvline(x=avg_stimulus_duration, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Firing rate (spikes/s)')
        ax1.set_title('PSTH by Channel')
        ax1.grid(True, alpha=0.3)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Channel activity heatmap
        ax2 = axes[0, 1]
        channels = sorted(channel_data.keys())
        firing_rates = np.array([channel_data[ch]['firing_rate'] for ch in channels])
        
        im = ax2.imshow(firing_rates, aspect='auto', cmap='viridis', interpolation='nearest')
        ax2.set_xlabel('Time bin')
        ax2.set_ylabel('Channel')
        ax2.set_title('Channel Activity Heatmap')
        ax2.set_yticks(range(len(channels)))
        ax2.set_yticklabels(channels)
        
        # Add time labels
        n_bins = len(channel_data[channels[0]]['bin_centers'])
        time_ticks = np.linspace(0, n_bins-1, 5, dtype=int)
        time_labels = [f"{channel_data[channels[0]]['bin_centers'][i]:.0f}" for i in time_ticks]
        ax2.set_xticks(time_ticks)
        ax2.set_xticklabels(time_labels)
        
        plt.colorbar(im, ax=ax2, label='Firing rate (Hz)')
        
        # 3. Units per channel
        ax3 = axes[1, 0]
        channels = sorted(channel_data.keys())
        n_units = [channel_data[ch]['n_units'] for ch in channels]
        
        bars = ax3.bar(channels, n_units, alpha=0.7)
        ax3.set_xlabel('Channel')
        ax3.set_ylabel('Number of units')
        ax3.set_title('Units per Channel')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, n in zip(bars, n_units):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{n}', ha='center', va='bottom')
        
        # 4. Total spikes per channel
        ax4 = axes[1, 1]
        total_spikes = [channel_data[ch]['total_spikes'] for ch in channels]
        
        bars = ax4.bar(channels, total_spikes, alpha=0.7, color='orange')
        ax4.set_xlabel('Channel')
        ax4.set_ylabel('Total spikes')
        ax4.set_title('Total Spikes per Channel')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics for the PSTH analysis.
        
        Returns:
        --------
        Dict with summary statistics
        """
        if not self.psth_data:
            return {"error": "No PSTH data available. Run compute_psth first."}
        
        stats = {
            'n_units_analyzed': len(self.psth_data),
            'total_intervals': len(self.intervals_df),
            'data_time_range': {
                'min': float(self.spikes_df['time'].min()),
                'max': float(self.spikes_df['time'].max())
            }
        }
        
        # Per-unit statistics
        unit_stats = {}
        for unit, data in self.psth_data.items():
            baseline_rate = np.mean(data['firing_rate'][:len(data['firing_rate'])//4])  # First quarter
            peak_rate = np.max(data['firing_rate'])
            
            unit_stats[unit] = {
                'total_spikes': data['total_spikes'],
                'valid_intervals': data['valid_intervals'],
                'baseline_rate': float(baseline_rate),
                'peak_rate': float(peak_rate),
                'modulation_ratio': float(peak_rate / baseline_rate) if baseline_rate > 0 else float('inf')
            }
        
        stats['unit_statistics'] = unit_stats
        
        return stats

def main():
    """
    Example usage of the PSTH analyzer.
    """
    # Initialize analyzer
    analyzer = PSTHAnalyzer()
    
    # Compute PSTH with default parameters
    psth_data = analyzer.compute_psth(pre_time=1000, post_time=2000, bin_size=50)
    
    # Plot examples
    print("Creating visualizations...")
    
    # Plot first few units individually
    for i, unit in enumerate(list(psth_data.keys())[:3]):
        fig, ax = plt.subplots(figsize=(10, 6))
        analyzer.plot_psth_single_unit(unit, ax)
        plt.savefig(f'psth_unit_{unit}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot population PSTH
    pop_fig = analyzer.plot_population_psth()
    plt.savefig('population_psth.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Analyze channels
    channel_data = analyzer.analyze_channels()
    channel_fig = analyzer.plot_channel_analysis(channel_data)
    plt.savefig('channel_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print summary statistics
    summary = analyzer.get_summary_statistics()
    print("\nSummary Statistics:")
    print(f"Units analyzed: {summary['n_units_analyzed']}")
    print(f"Total intervals: {summary['total_intervals']}")
    
    print("\nAnalysis complete! Check the generated PNG files for visualizations.")

if __name__ == "__main__":
    main()