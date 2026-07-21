"""
Licking Behavior Analysis Script
Analyzes behavior.csv files from multiple session folders to compute:
- Number of licks per bout
- Max tongue area average across sessions/days
- Average interlick interval (within bouts)
- Total licks normalized by total frames
- Total bouts normalized by total frames

Groups: Before (719, 720), After (808, 810)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def parse_session_name(folder_name):
    """
    Parse folder name to extract day and session number.
    e.g., '719_1' -> day='719', session=1
    """
    parts = folder_name.split('_')
    if len(parts) == 2:
        day = parts[0]
        session = int(parts[1])
        return day, session
    elif len(parts) == 1 and parts[0].isdigit():
        # Handle cases without underscore
        return parts[0], 1
    return None, None


def assign_group(day):
    """
    Assign group based on day number.
    Before: 719, 720 (and 721 for adjacent day)
    After: 808, 810
    """
    if day in ['719', '720', '721']:
        return 'Before'
    elif day in ['808', '810']:
        return 'After'
    return 'Unknown'


def compute_session_metrics(df, folder_name):
    """
    Compute all metrics for a single session.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Behavior data with columns:
        - Tongue_area_Interval Max
        - Tongue_area_interval_detection_Interval Start
        - Tongue_area_interval_detection_Interval End
        - Tongue_area_interval_detection_Interval Duration
        - Tongue_area_interval_detection_group_intervals_Interval Overlap Assign ID
    
    Returns:
    --------
    dict : Dictionary containing computed metrics
    """
    # Rename columns for easier access
    df = df.rename(columns={
        'Tongue_area_Interval Max': 'max_area',
        'Tongue_area_interval_detection_Interval Start': 'start',
        'Tongue_area_interval_detection_Interval End': 'end',
        'Tongue_area_interval_detection_Interval Duration': 'duration',
        'Tongue_area_interval_detection_group_intervals_Interval Overlap Assign ID': 'bout_id'
    })
    
    # Store all lick durations for this session
    lick_durations = df['duration'].tolist()
    
    # Total frames (max end frame in the session)
    total_frames = df['end'].max()
    
    # Total licks and bouts
    total_licks = len(df)
    total_bouts = df['bout_id'].nunique()
    
    # Licks per bout (distribution)
    licks_per_bout = df.groupby('bout_id').size()
    
    # Mean max tongue area across all licks
    mean_max_tongue_area = df['max_area'].mean()
    
    # Mean lick duration
    mean_lick_duration = df['duration'].mean()
    
    # Compute interlick intervals within each bout
    # Interval = frames between end of previous lick and start of next lick
    interlick_intervals = []
    for bout_id in df['bout_id'].unique():
        bout_data = df[df['bout_id'] == bout_id].sort_values('start')
        if len(bout_data) > 1:
            # Calculate intervals from end of lick i to start of lick i+1
            for i in range(len(bout_data) - 1):
                end_current = bout_data.iloc[i]['end']
                start_next = bout_data.iloc[i + 1]['start']
                interval = start_next - end_current
                interlick_intervals.append(interval)
    
    mean_interlick_interval = np.mean(interlick_intervals) if interlick_intervals else np.nan
    
    # Normalized metrics
    licks_per_frame = total_licks / total_frames if total_frames > 0 else 0
    bouts_per_frame = total_bouts / total_frames if total_frames > 0 else 0
    
    day, session = parse_session_name(folder_name)
    group = assign_group(day)
    
    return {
        'folder': folder_name,
        'day': day,
        'session': session,
        'group': group,
        'total_licks': total_licks,
        'total_bouts': total_bouts,
        'total_frames': total_frames,
        'licks_per_frame': licks_per_frame,
        'bouts_per_frame': bouts_per_frame,
        'mean_licks_per_bout': licks_per_bout.mean(),
        'std_licks_per_bout': licks_per_bout.std(),
        'mean_max_tongue_area': mean_max_tongue_area,
        'mean_lick_duration': mean_lick_duration,
        'mean_interlick_interval': mean_interlick_interval,
        'licks_per_bout_list': licks_per_bout.tolist(),
        'lick_durations': lick_durations
    }


def load_all_sessions(base_dir, suffix='100_3_behavior'):
    """
    Load all CSV files with the specified suffix from session folders.
    
    Parameters:
    -----------
    base_dir : str
        Path to the directory containing session folders
    suffix : str
        The suffix pattern to match (e.g., '100_3_behavior' will match *100_3_behavior.csv)
    
    Returns:
    --------
    list : List of dictionaries containing metrics for each session
    """
    base_path = Path(base_dir)
    all_metrics = []
    
    # Iterate through all subdirectories
    for folder in sorted(base_path.iterdir()):
        if folder.is_dir():
            # Find all CSV files with the specified suffix
            matching_files = list(folder.glob(f'*{suffix}.csv'))
            
            if matching_files:
                behavior_file = matching_files[0]  # Use the first match
                try:
                    df = pd.read_csv(behavior_file)
                    metrics = compute_session_metrics(df, folder.name)
                    all_metrics.append(metrics)
                    print(f"Processed: {folder.name} ({behavior_file.name})")
                except Exception as e:
                    print(f"Error processing {folder.name}: {e}")
            else:
                print(f"No file matching '*{suffix}.csv' found in {folder.name}")
    
    return all_metrics


def create_plots(metrics_df, output_dir):
    """
    Create all required plots and save as PNG and SVG.
    
    Parameters:
    -----------
    metrics_df : pd.DataFrame
        DataFrame containing all metrics
    output_dir : Path
        Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define color palette for groups with fallback
    colors = {'Before': '#1f77b4', 'After': '#ff7f0e', 'Unknown': '#808080'}
    
    # Filter out Unknown groups if any exist
    metrics_df = metrics_df[metrics_df['group'] != 'Unknown'].copy()
    
    if len(metrics_df) == 0:
        print("No valid sessions with recognized groups!")
        return
    
    # 1. Licks per bout (box plot)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for box plot
    licks_per_bout_data = []
    for _, row in metrics_df.iterrows():
        session_label = f"{row['day']}_{row['session']}"
        for licks in row['licks_per_bout_list']:
            licks_per_bout_data.append({
                'session': session_label,
                'group': row['group'],
                'licks': licks
            })
    
    lpb_df = pd.DataFrame(licks_per_bout_data)
    
    # Create box plot grouped by group
    if not lpb_df.empty:
        sns.boxplot(data=lpb_df, x='session', y='licks', hue='group', 
                   palette=colors, ax=ax)
        ax.set_xlabel('Session', fontsize=12, fontweight='bold')
        ax.set_ylabel('Licks per Bout', fontsize=12, fontweight='bold')
        ax.set_title('Number of Licks per Bout Across Sessions', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        fig.savefig(output_dir / 'licks_per_bout.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'licks_per_bout.svg', format='svg', bbox_inches='tight')
        plt.close()
        print("Saved: licks_per_bout plots")
    
    # 2. Mean max tongue area across sessions
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for group in metrics_df['group'].unique():
        group_data = metrics_df[metrics_df['group'] == group].sort_values(['day', 'session'])
        sessions = [f"{row['day']}_{row['session']}" for _, row in group_data.iterrows()]
        values = group_data['mean_max_tongue_area'].values
        
        ax.plot(sessions, values, marker='o', linewidth=2, markersize=8, 
               label=group, color=colors.get(group, 'gray'))
    
    ax.set_xlabel('Session', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Max Tongue Area', fontsize=12, fontweight='bold')
    ax.set_title('Average Max Tongue Area Across Sessions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    fig.savefig(output_dir / 'mean_max_tongue_area.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'mean_max_tongue_area.svg', format='svg', bbox_inches='tight')
    plt.close()
    print("Saved: mean_max_tongue_area plots")
    
    # 3. Mean interlick interval
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for group in metrics_df['group'].unique():
        group_data = metrics_df[metrics_df['group'] == group].sort_values(['day', 'session'])
        sessions = [f"{row['day']}_{row['session']}" for _, row in group_data.iterrows()]
        values = group_data['mean_interlick_interval'].values
        
        ax.plot(sessions, values, marker='o', linewidth=2, markersize=8, 
               label=group, color=colors.get(group, 'gray'))
    
    ax.set_xlabel('Session', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Interlick Interval (frames)', fontsize=12, fontweight='bold')
    ax.set_title('Average Interlick Interval Within Bouts', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    fig.savefig(output_dir / 'mean_interlick_interval.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'mean_interlick_interval.svg', format='svg', bbox_inches='tight')
    plt.close()
    print("Saved: mean_interlick_interval plots")
    
    # 3b. Lick duration with violin plot and individual dots
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for violin plot
    lick_duration_data = []
    for _, row in metrics_df.iterrows():
        session_label = f"{row['day']}_{row['session']}"
        for duration in row['lick_durations']:
            lick_duration_data.append({
                'session': session_label,
                'group': row['group'],
                'duration': duration
            })
    
    ld_df = pd.DataFrame(lick_duration_data)
    
    if not ld_df.empty:
        # Create violin plot
        sns.violinplot(data=ld_df, x='session', y='duration', hue='group',
                      palette=colors, ax=ax, inner=None, alpha=0.6)
        
        # Overlay individual points with jitter
        for group in ld_df['group'].unique():
            group_data = ld_df[ld_df['group'] == group]
            sessions = group_data['session'].unique()
            
            for session in sessions:
                session_data = group_data[group_data['session'] == session]
                x_pos = list(ld_df['session'].unique()).index(session)
                
                # Add jitter to x position
                x_jitter = np.random.normal(x_pos, 0.04, size=len(session_data))
                ax.scatter(x_jitter, session_data['duration'], 
                          color=colors.get(group, 'gray'), 
                          alpha=0.3, s=8, edgecolors='none')
        
        ax.set_xlabel('Session', fontsize=12, fontweight='bold')
        ax.set_ylabel('Lick Duration (frames)', fontsize=12, fontweight='bold')
        ax.set_title('Lick Duration Distribution Across Sessions', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        fig.savefig(output_dir / 'lick_duration_violin.png', dpi=300, bbox_inches='tight')
        fig.savefig(output_dir / 'lick_duration_violin.svg', format='svg', bbox_inches='tight')
        plt.close()
        print("Saved: lick_duration_violin plots")
    
    # 4. Total licks per frame (normalized)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for grouped bar plot
    sessions_sorted = metrics_df.sort_values(['day', 'session'])
    session_labels = [f"{row['day']}_{row['session']}" for _, row in sessions_sorted.iterrows()]
    
    # Get unique groups and sessions
    groups = metrics_df['group'].unique()
    x = np.arange(len(session_labels))
    width = 0.35
    
    # Plot bars for each group
    for i, group in enumerate(groups):
        group_values = []
        for _, row in sessions_sorted.iterrows():
            if row['group'] == group:
                group_values.append(row['licks_per_frame'])
            else:
                group_values.append(0)  # Placeholder for sessions not in this group
        
        offset = width * (i - len(groups)/2 + 0.5)
        ax.bar(x + offset, group_values, width, label=group, 
              color=colors.get(group, 'gray'), alpha=0.8)
    
    ax.set_xlabel('Session', fontsize=12, fontweight='bold')
    ax.set_ylabel('Licks per Frame', fontsize=12, fontweight='bold')
    ax.set_title('Total Licks Normalized by Total Frames', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(session_labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    fig.savefig(output_dir / 'licks_per_frame_normalized.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'licks_per_frame_normalized.svg', format='svg', bbox_inches='tight')
    plt.close()
    print("Saved: licks_per_frame_normalized plots")
    
    # 5. Total bouts per frame (normalized)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for grouped bar plot
    sessions_sorted = metrics_df.sort_values(['day', 'session'])
    session_labels = [f"{row['day']}_{row['session']}" for _, row in sessions_sorted.iterrows()]
    
    # Get unique groups and sessions
    groups = metrics_df['group'].unique()
    x = np.arange(len(session_labels))
    width = 0.35
    
    # Plot bars for each group
    for i, group in enumerate(groups):
        group_values = []
        for _, row in sessions_sorted.iterrows():
            if row['group'] == group:
                group_values.append(row['bouts_per_frame'])
            else:
                group_values.append(0)  # Placeholder for sessions not in this group
        
        offset = width * (i - len(groups)/2 + 0.5)
        ax.bar(x + offset, group_values, width, label=group, 
              color=colors.get(group, 'gray'), alpha=0.8)
    
    ax.set_xlabel('Session', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bouts per Frame', fontsize=12, fontweight='bold')
    ax.set_title('Total Bouts Normalized by Total Frames', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(session_labels, rotation=45, ha='right')
    ax.legend(fontsize=11)
    plt.tight_layout()
    
    fig.savefig(output_dir / 'bouts_per_frame_normalized.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'bouts_per_frame_normalized.svg', format='svg', bbox_inches='tight')
    plt.close()
    print("Saved: bouts_per_frame_normalized plots")
    
    # 6. Combined summary plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Mean licks per bout
    ax = axes[0, 0]
    for group in metrics_df['group'].unique():
        group_data = metrics_df[metrics_df['group'] == group].sort_values(['day', 'session'])
        sessions = [f"{row['day']}_{row['session']}" for _, row in group_data.iterrows()]
        values = group_data['mean_licks_per_bout'].values
        ax.plot(sessions, values, marker='o', linewidth=2, markersize=8, 
               label=group, color=colors.get(group, 'gray'))
    ax.set_xlabel('Session', fontweight='bold')
    ax.set_ylabel('Mean Licks per Bout', fontweight='bold')
    ax.set_title('Mean Licks per Bout', fontweight='bold')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    # Mean max tongue area
    ax = axes[0, 1]
    for group in metrics_df['group'].unique():
        group_data = metrics_df[metrics_df['group'] == group].sort_values(['day', 'session'])
        sessions = [f"{row['day']}_{row['session']}" for _, row in group_data.iterrows()]
        values = group_data['mean_max_tongue_area'].values
        ax.plot(sessions, values, marker='o', linewidth=2, markersize=8, 
               label=group, color=colors.get(group, 'gray'))
    ax.set_xlabel('Session', fontweight='bold')
    ax.set_ylabel('Mean Max Tongue Area', fontweight='bold')
    ax.set_title('Mean Max Tongue Area', fontweight='bold')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    # Mean interlick interval
    ax = axes[1, 0]
    for group in metrics_df['group'].unique():
        group_data = metrics_df[metrics_df['group'] == group].sort_values(['day', 'session'])
        sessions = [f"{row['day']}_{row['session']}" for _, row in group_data.iterrows()]
        values = group_data['mean_interlick_interval'].values
        ax.plot(sessions, values, marker='o', linewidth=2, markersize=8, 
               label=group, color=colors.get(group, 'gray'))
    ax.set_xlabel('Session', fontweight='bold')
    ax.set_ylabel('Mean Interlick Interval (frames)', fontweight='bold')
    ax.set_title('Mean Interlick Interval', fontweight='bold')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    # Normalized licks per frame
    ax = axes[1, 1]
    for group in metrics_df['group'].unique():
        group_data = metrics_df[metrics_df['group'] == group].sort_values(['day', 'session'])
        sessions = [f"{row['day']}_{row['session']}" for _, row in group_data.iterrows()]
        values = group_data['licks_per_frame'].values
        ax.plot(sessions, values, marker='o', linewidth=2, markersize=8, 
               label=group, color=colors.get(group, 'gray'))
    ax.set_xlabel('Session', fontweight='bold')
    ax.set_ylabel('Licks per Frame', fontweight='bold')
    ax.set_title('Normalized Licks per Frame', fontweight='bold')
    ax.legend()
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'summary_all_metrics.png', dpi=300, bbox_inches='tight')
    fig.savefig(output_dir / 'summary_all_metrics.svg', format='svg', bbox_inches='tight')
    plt.close()
    print("Saved: summary_all_metrics plots")


def main(base_dir, output_dir=None, suffix='100_3_behavior'):
    """
    Main analysis pipeline.
    
    Parameters:
    -----------
    base_dir : str
        Path to directory containing session folders
    output_dir : str, optional
        Path to save results. If None, creates 'results' in current directory
    suffix : str
        The suffix pattern to match in CSV filenames (e.g., '100_3_behavior')
    """
    # Load all sessions
    print(f"\nScanning directory: {base_dir}")
    print(f"Looking for files matching: *{suffix}.csv")
    print("=" * 60)
    
    all_metrics = load_all_sessions(base_dir, suffix)
    
    if not all_metrics:
        print("No sessions found!")
        return
    
    print(f"\n{len(all_metrics)} sessions processed successfully")
    print("=" * 60)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Remove the list columns for CSV export
    metrics_csv = metrics_df.drop(columns=['licks_per_bout_list', 'lick_durations'])
    
    # Set up output directory
    if output_dir is None:
        output_dir = Path(__file__).parent / 'results'
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary CSV
    csv_path = output_dir / 'session_metrics_summary.csv'
    metrics_csv.to_csv(csv_path, index=False)
    print(f"\nSaved summary CSV: {csv_path}")
    
    # Create all plots
    print("\nGenerating plots...")
    print("=" * 60)
    create_plots(metrics_df, output_dir)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)
    
    # Print summary statistics by group
    print("\nSummary Statistics by Group:")
    print("-" * 60)
    for group in metrics_csv['group'].unique():
        group_data = metrics_csv[metrics_csv['group'] == group]
        print(f"\n{group} Group (n={len(group_data)} sessions):")
        print(f"  Mean licks per bout: {group_data['mean_licks_per_bout'].mean():.2f} ± {group_data['mean_licks_per_bout'].std():.2f}")
        print(f"  Mean max tongue area: {group_data['mean_max_tongue_area'].mean():.2f} ± {group_data['mean_max_tongue_area'].std():.2f}")
        print(f"  Mean interlick interval: {group_data['mean_interlick_interval'].mean():.2f} ± {group_data['mean_interlick_interval'].std():.2f} frames")
        print(f"  Mean licks per frame: {group_data['licks_per_frame'].mean():.6f} ± {group_data['licks_per_frame'].std():.6f}")
        print(f"  Mean bouts per frame: {group_data['bouts_per_frame'].mean():.6f} ± {group_data['bouts_per_frame'].std():.6f}")


if __name__ == "__main__":
    import sys
    
    # Get directory path and optional suffix
    args = sys.argv[1:]
    
    if len(args) > 0:
        base_directory = args[0]
    else:
        # Default 
        base_directory = r"C:\Users\wanglab\Desktop\IRt.PCRt-1125\TeNT_1110\Phox2B#9_side_all"
    
    # Optional: specify custom suffix as second argument
    suffix = args[1] if len(args) > 1 else '100_3_behavior'
    
    # Run analysis
    main(base_directory, suffix=suffix)
