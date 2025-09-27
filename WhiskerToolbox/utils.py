"""
Utility functions for spike interval analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List


def load_and_prepare_data(csv_path: str) -> pd.DataFrame:
    """
    Load the semi_from_table_designer.csv file and prepare it for analysis.
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded and prepared dataframe
    """
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    return df


def categorize_intervals(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Categorize intervals into three main duration groups: 150±4ms, 300±4ms, 750±4ms
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with categorized dataframes
    """
    duration_150 = df[(df['pico_Interval Duration'] >= 146) & (df['pico_Interval Duration'] <= 154)]
    duration_300 = df[(df['pico_Interval Duration'] >= 296) & (df['pico_Interval Duration'] <= 304)]
    duration_750 = df[(df['pico_Interval Duration'] >= 746) & (df['pico_Interval Duration'] <= 754)]
    
    print("\n=== INTERVAL DURATION CATEGORIZATION ===")
    print(f"150±4 ms intervals: {len(duration_150)} ({len(duration_150)/len(df)*100:.1f}%)")
    print(f"300±4 ms intervals: {len(duration_300)} ({len(duration_300)/len(df)*100:.1f}%)")
    print(f"750±4 ms intervals: {len(duration_750)} ({len(duration_750)/len(df)*100:.1f}%)")
    print(f"Other durations: {len(df) - len(duration_150) - len(duration_300) - len(duration_750)} intervals")
    
    return {
        '150ms': duration_150,
        '300ms': duration_300, 
        '750ms': duration_750
    }


def get_spike_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[int]]:
    """
    Extract spike-related column names and unit numbers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Tuple[List[str], List[str], List[int]]: Event presence cols, event count cols, unit numbers
    """
    event_presence_cols = [col for col in df.columns if 'Event Presence' in col]
    event_count_cols = [col for col in df.columns if 'Event Count' in col]
    
    # Extract unit numbers
    unit_numbers = []
    for col in event_presence_cols:
        unit_num = col.split('_')[1]
        unit_numbers.append(int(unit_num))
    
    unit_numbers = sorted(unit_numbers)
    
    print(f"\n=== SPIKE UNIT INFORMATION ===")
    print(f"Number of spike units: {len(event_presence_cols)}")
    print(f"Unit numbers: {min(unit_numbers)} to {max(unit_numbers)}")
    print(f"Event presence columns: {len(event_presence_cols)}")
    print(f"Event count columns: {len(event_count_cols)}")
    
    return event_presence_cols, event_count_cols, unit_numbers


def analyze_unit_activity(df: pd.DataFrame, event_presence_cols: List[str]) -> pd.DataFrame:
    """
    Analyze which units are most/least active across all intervals.
    
    Args:
        df (pd.DataFrame): Input dataframe
        event_presence_cols (List[str]): List of event presence column names
        
    Returns:
        pd.DataFrame: Summary statistics for each unit
    """
    print("\n=== UNIT ACTIVITY ANALYSIS ===")
    
    activity_stats = []
    
    for col in event_presence_cols:
        unit_num = int(col.split('_')[1])
        
        # Calculate activity metrics
        total_intervals = len(df)
        active_intervals = df[col].sum()
        activity_rate = active_intervals / total_intervals * 100
        
        activity_stats.append({
            'Unit': unit_num,
            'Active_Intervals': active_intervals,
            'Total_Intervals': total_intervals,
            'Activity_Rate_Percent': activity_rate
        })
    
    activity_df = pd.DataFrame(activity_stats).sort_values('Active_Intervals', ascending=False)
    
    # Print top and bottom units
    print(f"Most active units (top 5):")
    for i, row in activity_df.head().iterrows():
        print(f"  Unit {int(row['Unit']):2d}: {int(row['Active_Intervals']):5d} intervals ({row['Activity_Rate_Percent']:5.1f}%)")
    
    print(f"\nLeast active units (bottom 5):")
    for i, row in activity_df.tail().iterrows():
        print(f"  Unit {int(row['Unit']):2d}: {int(row['Active_Intervals']):5d} intervals ({row['Activity_Rate_Percent']:5.1f}%)")
    
    return activity_df


def analyze_activity_by_duration(df: pd.DataFrame, categorized_intervals: Dict[str, pd.DataFrame], 
                                event_presence_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Analyze unit activity patterns for each interval duration category.
    
    Args:
        df (pd.DataFrame): Input dataframe
        categorized_intervals (Dict[str, pd.DataFrame]): Categorized interval dataframes
        event_presence_cols (List[str]): List of event presence column names
        
    Returns:
        Dict[str, pd.DataFrame]: Activity analysis for each duration category
    """
    print("\n=== ACTIVITY BY INTERVAL DURATION ===")
    
    duration_activity = {}
    
    for duration_name, duration_df in categorized_intervals.items():
        print(f"\n{duration_name} intervals analysis:")
        
        activity_stats = []
        
        for col in event_presence_cols:
            unit_num = int(col.split('_')[1])
            
            total_intervals = len(duration_df)
            active_intervals = duration_df[col].sum()
            activity_rate = active_intervals / total_intervals * 100 if total_intervals > 0 else 0
            
            activity_stats.append({
                'Unit': unit_num,
                'Active_Intervals': active_intervals,
                'Total_Intervals': total_intervals,
                'Activity_Rate_Percent': activity_rate
            })
        
        activity_df = pd.DataFrame(activity_stats).sort_values('Active_Intervals', ascending=False)
        duration_activity[duration_name] = activity_df
        
        # Print top 3 for each duration
        print(f"  Top 3 most active units:")
        for i, row in activity_df.head(3).iterrows():
            print(f"    Unit {int(row['Unit']):2d}: {int(row['Active_Intervals']):4d}/{int(row['Total_Intervals']):4d} intervals ({row['Activity_Rate_Percent']:5.1f}%)")
    
    return duration_activity


def analyze_spike_counts(df: pd.DataFrame, event_count_cols: List[str]) -> pd.DataFrame:
    """
    Analyze spike count statistics for each unit.
    
    Args:
        df (pd.DataFrame): Input dataframe
        event_count_cols (List[str]): List of event count column names
        
    Returns:
        pd.DataFrame: Spike count statistics for each unit
    """
    print("\n=== SPIKE COUNT ANALYSIS ===")
    
    count_stats = []
    
    for col in event_count_cols:
        unit_num = int(col.split('_')[1])
        
        # Calculate spike count metrics
        total_spikes = df[col].sum()
        max_spikes_per_interval = df[col].max()
        mean_spikes_per_interval = df[col].mean()
        intervals_with_spikes = (df[col] > 0).sum()
        
        count_stats.append({
            'Unit': unit_num,
            'Total_Spikes': total_spikes,
            'Max_Spikes_Per_Interval': max_spikes_per_interval,
            'Mean_Spikes_Per_Interval': mean_spikes_per_interval,
            'Intervals_With_Spikes': intervals_with_spikes
        })
    
    count_df = pd.DataFrame(count_stats).sort_values('Total_Spikes', ascending=False)
    
    # Print top spike producing units
    print(f"Units with most total spikes (top 5):")
    for i, row in count_df.head().iterrows():
        print(f"  Unit {int(row['Unit']):2d}: {int(row['Total_Spikes']):6d} total spikes, max {int(row['Max_Spikes_Per_Interval']):2d} per interval, mean {row['Mean_Spikes_Per_Interval']:.2f}")
    
    return count_df


def create_activity_heatmap_data(categorized_intervals: Dict[str, pd.DataFrame], 
                                event_presence_cols: List[str]) -> pd.DataFrame:
    """
    Create data for activity heatmap showing unit activity across different interval durations.
    
    Args:
        categorized_intervals (Dict[str, pd.DataFrame]): Categorized interval dataframes
        event_presence_cols (List[str]): List of event presence column names
        
    Returns:
        pd.DataFrame: Heatmap data with units as rows and durations as columns
    """
    print("\n=== CREATING HEATMAP DATA ===")
    
    heatmap_data = []
    
    for col in event_presence_cols:
        unit_num = int(col.split('_')[1])
        
        row_data = {'Unit': unit_num}
        
        for duration_name, duration_df in categorized_intervals.items():
            if len(duration_df) > 0:
                activity_rate = duration_df[col].sum() / len(duration_df) * 100
            else:
                activity_rate = 0
            row_data[duration_name] = activity_rate
        
        heatmap_data.append(row_data)
    
    heatmap_df = pd.DataFrame(heatmap_data).set_index('Unit').sort_index()
    
    print(f"Heatmap data created: {len(heatmap_df)} units x {len(heatmap_df.columns)} duration categories")
    
    return heatmap_df


def print_summary_statistics(df: pd.DataFrame, activity_df: pd.DataFrame, count_df: pd.DataFrame):
    """
    Print comprehensive summary statistics.
    
    Args:
        df (pd.DataFrame): Original dataframe
        activity_df (pd.DataFrame): Activity analysis results
        count_df (pd.DataFrame): Spike count analysis results
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"  Total intervals analyzed: {len(df):,}")
    print(f"  Total spike units: {len(activity_df)}")
    print(f"  Date range: {df['pico_Interval Start'].min():.0f} to {df['pico_Interval End'].max():.0f}")
    
    print(f"\nActivity Summary:")
    print(f"  Most active unit: Unit {int(activity_df.iloc[0]['Unit'])} ({int(activity_df.iloc[0]['Active_Intervals']):,} intervals, {activity_df.iloc[0]['Activity_Rate_Percent']:.1f}%)")
    print(f"  Least active unit: Unit {int(activity_df.iloc[-1]['Unit'])} ({int(activity_df.iloc[-1]['Active_Intervals']):,} intervals, {activity_df.iloc[-1]['Activity_Rate_Percent']:.1f}%)")
    print(f"  Average activity rate: {activity_df['Activity_Rate_Percent'].mean():.1f}%")
    
    print(f"\nSpike Count Summary:")
    print(f"  Highest spike count unit: Unit {int(count_df.iloc[0]['Unit'])} ({int(count_df.iloc[0]['Total_Spikes']):,} total spikes)")
    print(f"  Average spikes per unit: {count_df['Total_Spikes'].mean():.1f}")
    print(f"  Total spikes across all units: {count_df['Total_Spikes'].sum():,}")
    
    print(f"\nInterval Duration Distribution:")
    duration_counts = df['pico_Interval Duration'].value_counts().sort_index()
    for duration, count in duration_counts.head(10).items():
        print(f"  {duration:.0f}ms: {count:,} intervals ({count/len(df)*100:.1f}%)")
    
    print("="*60)


def print_notebook_welcome():
    """
    Print welcome message for the notebook
    """
    print(" SPIKE INTERVAL ANALYSIS - Interactive Notebook")
    print("=" * 55)


def print_data_overview(df: pd.DataFrame):
    """
    Print data overview information
    """
    print(f"\n Data Shape: {df.shape}")
    print(f" Time range: {df['pico_Interval Start'].min():.0f} to {df['pico_Interval End'].max():.0f}")
    print(f" Duration range: {df['pico_Interval Duration'].min():.0f}ms to {df['pico_Interval Duration'].max():.0f}ms")


def print_unit_count_info(unit_numbers: List[int]):
    """
    Print information about spike units found
    """
    print(f"\nFound {len(unit_numbers)} spike units (Unit {min(unit_numbers)} to Unit {max(unit_numbers)})")


def print_table_headers():
    """
    Print formatted table headers for different analyses
    """
    headers = {
        'top_active': (" TOP 15 MOST ACTIVE UNITS (by interval count)", 80),
        'top_spikes': (" TOP 15 SPIKE PRODUCING UNITS (by total spike count)", 80),
        'bottom_active': (" BOTTOM 15 LEAST ACTIVE UNITS (by interval count)", 80),
        'duration_summary': (" ACTIVITY SUMMARY BY INTERVAL DURATION", 80)
    }
    
    return headers


def print_unit_explorer_header():
    """
    Print header for interactive unit explorer section
    """
    print(" INTERACTIVE UNIT EXPLORER")
    print("=" * 40)


def print_activity_patterns_header():
    """
    Print header for activity patterns analysis
    """
    print("\n ANALYZING DIFFERENT ACTIVITY PATTERNS")
    print("=" * 45)


def print_special_units_info(most_active_unit, least_active_unit, highest_spike_unit, moderate_activity_unit):
    """
    Print information about special units being analyzed
    """
    print(f"Analyzing special units:")
    print(f"  Most Active: Unit {most_active_unit}")
    print(f"  Least Active: Unit {least_active_unit}")
    print(f"  Highest Spike Count: Unit {highest_spike_unit}")
    print(f"  Moderate Activity: Unit {moderate_activity_unit}")


def print_export_info(df: pd.DataFrame, activity_df: pd.DataFrame, count_df: pd.DataFrame, 
                     heatmap_df: pd.DataFrame, categorized_intervals: Dict):
    """
    Print data export information and save files
    """
    print("\n" + "="*60)
    print(" DATA EXPORT CAPABILITIES")
    print("="*60)
    print("\nThe following dataframes are available for further analysis:")
    print(f"  • df: Original dataset ({len(df):,} rows)")
    print(f"  • activity_df: Unit activity analysis ({len(activity_df)} units)")
    print(f"  • count_df: Spike count analysis ({len(count_df)} units)")
    print(f"  • heatmap_df: Activity heatmap data ({len(heatmap_df)} units x {len(heatmap_df.columns)} durations)")
    print(f"  • categorized_intervals: Intervals by duration ({sum(len(v) for v in categorized_intervals.values())} total intervals)")
    
    # Save key results
    from pathlib import Path
    output_dir = Path('../Output')
    output_dir.mkdir(exist_ok=True)
    
    # Save analysis results
    activity_df.to_csv(output_dir / 'unit_activity_analysis.csv', index=False)
    count_df.to_csv(output_dir / 'spike_count_analysis.csv', index=False)
    heatmap_df.to_csv(output_dir / 'activity_heatmap_data.csv')
    
    print(f"\n Analysis results exported to {output_dir}:")
    print(f"  • unit_activity_analysis.csv")
    print(f"  • spike_count_analysis.csv")
    print(f"  • activity_heatmap_data.csv")
    
    print("\n Analysis Complete! Use the variables above for further exploration.")