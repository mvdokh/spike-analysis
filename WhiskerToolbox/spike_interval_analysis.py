#!/usr/bin/env python3
"""
Spike Interval Analysis Script
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add the current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (
    load_and_prepare_data,
    categorize_intervals,
    get_spike_columns,
    analyze_unit_activity,
    analyze_activity_by_duration,
    analyze_spike_counts,
    create_activity_heatmap_data,
    print_summary_statistics
)


def create_visualizations(df, activity_df, count_df, heatmap_df, categorized_intervals, save_plots=True):
    """
    Create comprehensive visualizations for spike data analysis.
    
    Args:
        df (pd.DataFrame): Original dataframe
        activity_df (pd.DataFrame): Activity analysis results
        count_df (pd.DataFrame): Spike count analysis results
        heatmap_df (pd.DataFrame): Heatmap data
        categorized_intervals (dict): Categorized interval dataframes
        save_plots (bool): Whether to save plots to files
    """
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Unit Activity Overview (Top 15 most active units)
    ax1 = plt.subplot(3, 4, 1)
    top_15_units = activity_df.head(15)
    bars1 = ax1.bar(range(len(top_15_units)), top_15_units['Active_Intervals'], 
                    color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_xlabel('Unit Rank')
    ax1.set_ylabel('Active Intervals')
    ax1.set_title('Top 15 Most Active Units\n(Total Active Intervals)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(top_15_units)))
    ax1.set_xticklabels([f"Unit {int(unit)}" for unit in top_15_units['Unit']], rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 2. Activity Rate Distribution
    ax2 = plt.subplot(3, 4, 2)
    ax2.hist(activity_df['Activity_Rate_Percent'], bins=20, color='lightcoral', 
             edgecolor='darkred', alpha=0.7)
    ax2.axvline(activity_df['Activity_Rate_Percent'].mean(), color='red', 
                linestyle='--', linewidth=2, label=f'Mean: {activity_df["Activity_Rate_Percent"].mean():.1f}%')
    ax2.set_xlabel('Activity Rate (%)')
    ax2.set_ylabel('Number of Units')
    ax2.set_title('Distribution of Unit Activity Rates', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Spike Count vs Activity Rate Scatter
    ax3 = plt.subplot(3, 4, 3)
    merged_df = pd.merge(activity_df, count_df, on='Unit')
    scatter = ax3.scatter(merged_df['Activity_Rate_Percent'], merged_df['Total_Spikes'], 
                         alpha=0.6, c=merged_df['Unit'], cmap='viridis', s=60)
    ax3.set_xlabel('Activity Rate (%)')
    ax3.set_ylabel('Total Spikes')
    ax3.set_title('Activity Rate vs Total Spike Count', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, ax=ax3, label='Unit Number')
    ax3.grid(True, alpha=0.3)
    
    # 4. Interval Duration Distribution
    ax4 = plt.subplot(3, 4, 4)
    duration_counts = df['pico_Interval Duration'].value_counts().sort_index()
    ax4.bar(duration_counts.index, duration_counts.values, color='gold', 
            edgecolor='orange', alpha=0.7, width=2)
    ax4.set_xlabel('Interval Duration (ms)')
    ax4.set_ylabel('Number of Intervals')
    ax4.set_title('Distribution of Interval Durations', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Activity Heatmap by Duration
    ax5 = plt.subplot(3, 4, (5, 8))
    sns.heatmap(heatmap_df, annot=False, cmap='YlOrRd', cbar_kws={'label': 'Activity Rate (%)'}, ax=ax5)
    ax5.set_title('Unit Activity Rates by Interval Duration', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Interval Duration Category')
    ax5.set_ylabel('Unit Number')
    
    # 6. Top 10 Spike Producers
    ax6 = plt.subplot(3, 4, 9)
    top_10_spike = count_df.head(10)
    bars6 = ax6.bar(range(len(top_10_spike)), top_10_spike['Total_Spikes'], 
                    color='lightgreen', edgecolor='darkgreen', alpha=0.7)
    ax6.set_xlabel('Unit Rank')
    ax6.set_ylabel('Total Spikes')
    ax6.set_title('Top 10 Spike Producing Units', fontsize=12, fontweight='bold')
    ax6.set_xticks(range(len(top_10_spike)))
    ax6.set_xticklabels([f"Unit {int(unit)}" for unit in top_10_spike['Unit']], rotation=45)
    
    # Add value labels
    for i, bar in enumerate(bars6):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{int(height)}', ha='center', va='bottom', fontsize=8)
    
    # 7. Activity by Duration Category (Bar Chart)
    ax7 = plt.subplot(3, 4, 10)
    duration_activity_summary = []
    for duration_name, duration_df_cat in categorized_intervals.items():
        event_presence_cols = [col for col in df.columns if 'Event Presence' in col]
        total_activity = sum(duration_df_cat[col].sum() for col in event_presence_cols)
        duration_activity_summary.append({'Duration': duration_name, 'Total_Activity': total_activity})
    
    duration_summary_df = pd.DataFrame(duration_activity_summary)
    bars7 = ax7.bar(duration_summary_df['Duration'], duration_summary_df['Total_Activity'],
                    color=['#ff9999', '#66b3ff', '#99ff99'], edgecolor='black', alpha=0.8)
    ax7.set_xlabel('Interval Duration Category')
    ax7.set_ylabel('Total Activity Events')
    ax7.set_title('Total Activity by Duration Category', fontsize=12, fontweight='bold')
    
    # Add value labels
    for bar in bars7:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 8. Mean Spikes per Interval Distribution
    ax8 = plt.subplot(3, 4, 11)
    ax8.hist(count_df['Mean_Spikes_Per_Interval'], bins=15, color='plum', 
             edgecolor='purple', alpha=0.7)
    ax8.axvline(count_df['Mean_Spikes_Per_Interval'].mean(), color='purple', 
                linestyle='--', linewidth=2, label=f'Mean: {count_df["Mean_Spikes_Per_Interval"].mean():.2f}')
    ax8.set_xlabel('Mean Spikes per Interval')
    ax8.set_ylabel('Number of Units')
    ax8.set_title('Distribution of Mean Spikes per Interval', fontsize=12, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Unit Activity Correlation Matrix (sample of top units)
    ax9 = plt.subplot(3, 4, 12)
    # Select top 10 most active units for correlation analysis
    top_units = activity_df.head(10)['Unit'].tolist()
    presence_cols_top = [f'spikes_{unit}_Event Presence' for unit in top_units]
    
    correlation_matrix = df[presence_cols_top].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax9,
                fmt='.2f', square=True, cbar_kws={'label': 'Correlation'})
    ax9.set_title('Activity Correlation Matrix\n(Top 10 Most Active Units)', fontsize=12, fontweight='bold')
    ax9.set_xticklabels([f'U{unit}' for unit in top_units], rotation=45)
    ax9.set_yticklabels([f'U{unit}' for unit in top_units], rotation=0)
    
    plt.tight_layout(pad=3.0)
    
    if save_plots:
        output_dir = Path('../Output')
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'comprehensive_spike_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Comprehensive analysis plot saved to {output_dir / 'comprehensive_spike_analysis.png'}")
    
    plt.show()


def create_summary_tables(activity_df, count_df, categorized_intervals):
    """
    Create and display summary tables.
    
    Args:
        activity_df (pd.DataFrame): Activity analysis results
        count_df (pd.DataFrame): Spike count analysis results
        categorized_intervals (dict): Categorized interval dataframes
    """
    print("\n" + "="*80)
    print("SUMMARY TABLES")
    print("="*80)
    
    # Table 1: Top 15 Most Active Units
    print("\n TOP 15 MOST ACTIVE UNITS (by interval count)")
    print("-" * 70)
    top_active = activity_df.head(15)[['Unit', 'Active_Intervals', 'Activity_Rate_Percent']]
    top_active.index = range(1, len(top_active) + 1)
    print(top_active.to_string(formatters={
        'Unit': lambda x: f'Unit {int(x):2d}',
        'Active_Intervals': lambda x: f'{int(x):,}',
        'Activity_Rate_Percent': lambda x: f'{x:5.1f}%'
    }))
    
    # Table 2: Bottom 15 Least Active Units
    print("\n BOTTOM 15 LEAST ACTIVE UNITS (by interval count)")
    print("-" * 70)
    bottom_active = activity_df.tail(15)[['Unit', 'Active_Intervals', 'Activity_Rate_Percent']].iloc[::-1]
    bottom_active.index = range(1, len(bottom_active) + 1)
    print(bottom_active.to_string(formatters={
        'Unit': lambda x: f'Unit {int(x):2d}',
        'Active_Intervals': lambda x: f'{int(x):,}',
        'Activity_Rate_Percent': lambda x: f'{x:5.1f}%'
    }))
    
    # Table 3: Top Spike Producers
    print("\n TOP 15 SPIKE PRODUCING UNITS (by total spike count)")
    print("-" * 85)
    top_spikes = count_df.head(15)[['Unit', 'Total_Spikes', 'Mean_Spikes_Per_Interval', 'Max_Spikes_Per_Interval']]
    top_spikes.index = range(1, len(top_spikes) + 1)
    print(top_spikes.to_string(formatters={
        'Unit': lambda x: f'Unit {int(x):2d}',
        'Total_Spikes': lambda x: f'{int(x):,}',
        'Mean_Spikes_Per_Interval': lambda x: f'{x:.2f}',
        'Max_Spikes_Per_Interval': lambda x: f'{int(x):2d}'
    }))
    
    # Table 4: Activity by Duration Summary
    print("\n  ACTIVITY SUMMARY BY INTERVAL DURATION")
    print("-" * 70)
    duration_summary = []
    event_presence_cols = [col for col in activity_df.columns if col != 'Unit' and 'Event Presence' in str(col)]
    
    for duration_name, duration_df in categorized_intervals.items():
        total_intervals = len(duration_df)
        if total_intervals > 0:
            total_activity_events = sum(duration_df[col].sum() for col in duration_df.columns if 'Event Presence' in col)
            avg_activity_per_interval = total_activity_events / total_intervals
            active_units = sum((duration_df[col].sum() > 0) for col in duration_df.columns if 'Event Presence' in col)
        else:
            total_activity_events = 0
            avg_activity_per_interval = 0
            active_units = 0
            
        duration_summary.append({
            'Duration Category': duration_name,
            'Total Intervals': f'{total_intervals:,}',
            'Total Activity Events': f'{total_activity_events:,}',
            'Avg Events/Interval': f'{avg_activity_per_interval:.2f}',
            'Active Units': f'{active_units}'
        })
    
    duration_summary_df = pd.DataFrame(duration_summary)
    print(duration_summary_df.to_string(index=False))


def main():
    """
    Main function to execute the complete spike analysis pipeline.
    """
    # File paths
    csv_path = '../Data/semi_from_table_designer.csv'
    
    print(" SPIKE INTERVAL ANALYSIS")
    print("="*50)
    
    try:
        # Step 1: Load and prepare data
        df = load_and_prepare_data(csv_path)
        
        # Step 2: Categorize intervals
        categorized_intervals = categorize_intervals(df)
        
        # Step 3: Get spike column information
        event_presence_cols, event_count_cols, unit_numbers = get_spike_columns(df)
        
        # Step 4: Analyze unit activity
        activity_df = analyze_unit_activity(df, event_presence_cols)
        
        # Step 5: Analyze activity by duration
        duration_activity = analyze_activity_by_duration(df, categorized_intervals, event_presence_cols)
        
        # Step 6: Analyze spike counts
        count_df = analyze_spike_counts(df, event_count_cols)
        
        # Step 7: Create heatmap data
        heatmap_df = create_activity_heatmap_data(categorized_intervals, event_presence_cols)
        
        # Step 8: Print comprehensive summary
        print_summary_statistics(df, activity_df, count_df)
        
        # Step 9: Create summary tables
        create_summary_tables(activity_df, count_df, categorized_intervals)
        
        # Step 10: Create visualizations
        print(f"\n Generating comprehensive visualizations...")
        create_visualizations(df, activity_df, count_df, heatmap_df, categorized_intervals)
        
        print(f"\n Analysis complete! Check the Output folder for saved plots.")
        
        return {
            'original_data': df,
            'activity_analysis': activity_df,
            'count_analysis': count_df,
            'heatmap_data': heatmap_df,
            'categorized_intervals': categorized_intervals,
            'duration_activity': duration_activity
        }
        
    except FileNotFoundError:
        print(f" Error: Could not find the CSV file at {csv_path}")
        print("Please make sure the file exists and the path is correct.")
        return None
    except Exception as e:
        print(f" Error during analysis: {str(e)}")
        return None


if __name__ == "__main__":
    results = main()
    
    if results:
        print(f"\n Analysis results are available in the 'results' variable.")
        print(f"   - results['original_data']: Original dataframe")
        print(f"   - results['activity_analysis']: Unit activity statistics")
        print(f"   - results['count_analysis']: Spike count statistics")
        print(f"   - results['heatmap_data']: Activity heatmap data")
        print(f"   - results['categorized_intervals']: Intervals by duration")
        print(f"   - results['duration_activity']: Activity by duration analysis")