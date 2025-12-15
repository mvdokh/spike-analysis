"""
Behavior Analysis Script for Licking Data
Analyzes CSV files with suffix '100_3_behavior' from multiple recording sessions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import os
import cv2

# Set style for better-looking plots
sns.set_style("whitegrid")
sns.set_palette("husl")

class BehaviorAnalyzer:
    def __init__(self, base_dir):
        """
        Initialize the analyzer with the base directory containing session folders
        
        Parameters:
        -----------
        base_dir : str
            Path to the directory containing session folders (e.g., 720_1, 725_1, etc.)
        """
        self.base_dir = Path(base_dir)
        self.sessions_data = []
        self.output_dir = Path("/home/wanglab/spike-analysis/LICKING/OUTPUT")
        self.output_dir.mkdir(exist_ok=True)
        self.frame_counts = {}  # Store frame counts for each session
        
    def parse_session_name(self, folder_name):
        """
        Parse session folder name to extract date and trial number
        
        Parameters:
        -----------
        folder_name : str
            Folder name like '720_1', '725_2', etc.
            
        Returns:
        --------
        tuple : (date_str, trial_num, datetime_obj)
        """
        parts = folder_name.split('_')
        if len(parts) == 2:
            date_code = parts[0]  # e.g., '720'
            trial_num = int(parts[1])
            
            # Parse date (assuming format MMD or MMDD)
            if len(date_code) == 3:
                month = int(date_code[0])
                day = int(date_code[1:])
            elif len(date_code) == 4:
                month = int(date_code[:2])
                day = int(date_code[2:])
            else:
                return None
            
            # Assuming year 2025 (adjust if needed)
            year = 2025
            date_obj = datetime(year, month, day)
            date_str = date_obj.strftime('%m/%d')
            
            return date_str, trial_num, date_obj
        return None
    
    def get_video_frame_count(self, folder):
        """
        Get the frame count from the video file in the session folder
        
        Parameters:
        -----------
        folder : Path
            Path to the session folder
            
        Returns:
        --------
        int or None : Number of frames in the video, or None if not found
        """
        try:
            # Look for .mp4 files in the folder
            video_files = list(folder.glob("*.mp4"))
            
            if not video_files:
                return None
            
            # Use the first video file found
            video_path = str(video_files[0])
            
            # Open video and get frame count
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            return frame_count if frame_count > 0 else None
            
        except Exception as e:
            print(f"    Warning: Could not read video frame count: {e}")
            return None
    
    def load_all_sessions(self):
        """
        Load all CSV files with suffix '100_3_behavior' from session folders
        """
        print("Loading session data...")
        
        # Find all directories that match the pattern
        session_folders = [f for f in self.base_dir.iterdir() 
                          if f.is_dir() and '_' in f.name]
        
        for folder in session_folders:
            # Find any CSV file ending with '100_3_behavior.csv'
            behavior_files = list(folder.glob("*100_3_behavior.csv"))
            
            if behavior_files:
                behavior_file = behavior_files[0]  # Use the first match
                print(f"  Found: {behavior_file.name}")
                try:
                    # Parse session info
                    session_info = self.parse_session_name(folder.name)
                    if session_info is None:
                        print(f"Skipping {folder.name} - invalid format")
                        continue
                    
                    date_str, trial_num, date_obj = session_info
                    
                    # Get video frame count
                    frame_count = self.get_video_frame_count(folder)
                    if frame_count:
                        self.frame_counts[folder.name] = frame_count
                        print(f"    Video: {frame_count} frames")
                    
                    # Load CSV
                    df = pd.read_csv(behavior_file)
                    
                    # Add metadata
                    df['session_name'] = folder.name
                    df['date_str'] = date_str
                    df['trial_num'] = trial_num
                    df['date_obj'] = date_obj
                    df['frame_count'] = frame_count if frame_count else np.nan
                    
                    self.sessions_data.append(df)
                    print(f"Loaded {folder.name}: {len(df)} licks")
                    
                except Exception as e:
                    print(f"Error loading {folder.name}: {e}")
            else:
                print(f"Warning: No file matching '*100_3_behavior.csv' found in {folder.name}")
        
        if not self.sessions_data:
            raise ValueError("No session data loaded!")
        
        # Combine all data
        self.all_data = pd.concat(self.sessions_data, ignore_index=True)
        
        # Sort by date and trial
        self.all_data = self.all_data.sort_values(['date_obj', 'trial_num'])
        
        # Create a combined session identifier for plotting
        self.all_data['session_label'] = (
            self.all_data['date_str'] + '_T' + self.all_data['trial_num'].astype(str)
        )
        
        # Get unique sessions in chronological order
        self.unique_sessions = self.all_data.groupby(
            ['date_obj', 'date_str', 'trial_num', 'session_label', 'session_name']
        ).size().reset_index(name='n_licks').sort_values(['date_obj', 'trial_num'])
        
        print(f"\nTotal sessions loaded: {len(self.unique_sessions)}")
        print(f"Total licks: {len(self.all_data)}")
        
        return self.all_data
    
    def calculate_bout_metrics(self):
        """
        Calculate metrics for each bout
        """
        # Rename columns for easier access (handling the long column names)
        col_mapping = {}
        for col in self.all_data.columns:
            if 'Max' in col:
                col_mapping[col] = 'max_area'
            elif 'Start' in col:
                col_mapping[col] = 'start_frame'
            elif 'End' in col:
                col_mapping[col] = 'end_frame'
            elif 'Duration' in col:
                col_mapping[col] = 'duration'
            elif 'Assign ID' in col or 'ID' in col:
                col_mapping[col] = 'bout_id'
        
        self.all_data = self.all_data.rename(columns=col_mapping)
        
        # Calculate bout-level metrics
        bout_metrics = self.all_data.groupby(['session_name', 'session_label', 'date_obj', 'trial_num', 'bout_id']).agg({
            'max_area': ['mean', 'std', 'max', 'min'],
            'duration': ['mean', 'sum', 'count'],
            'start_frame': 'first',
            'end_frame': 'last'
        }).reset_index()
        
        # Flatten column names
        bout_metrics.columns = ['_'.join(col).strip('_') for col in bout_metrics.columns.values]
        
        # Rename for clarity
        bout_metrics = bout_metrics.rename(columns={
            'duration_count': 'licks_per_bout',
            'max_area_mean': 'mean_area_per_bout',
            'duration_sum': 'total_bout_duration',
            'start_frame_first': 'bout_start',
            'end_frame_last': 'bout_end'
        })
        
        self.bout_metrics = bout_metrics
        return bout_metrics
    
    def plot_licks_per_bout(self):
        """
        Plot licks per bout across all sessions
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Box plot of licks per bout for each session
        ax1 = axes[0]
        session_order = self.unique_sessions['session_label'].tolist()
        
        sns.boxplot(data=self.bout_metrics, x='session_label', y='licks_per_bout', 
                    order=session_order, ax=ax1)
        ax1.set_xlabel('Session (Chronological Order)', fontsize=12)
        ax1.set_ylabel('Licks per Bout', fontsize=12)
        ax1.set_title('Distribution of Licks per Bout Across Sessions', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Mean licks per bout with error bars
        ax2 = axes[1]
        session_stats = self.bout_metrics.groupby('session_label')['licks_per_bout'].agg(['mean', 'sem']).reset_index()
        session_stats = session_stats.set_index('session_label').reindex(session_order).reset_index()
        
        x_pos = np.arange(len(session_stats))
        ax2.bar(x_pos, session_stats['mean'], yerr=session_stats['sem'], 
                capsize=5, alpha=0.7, edgecolor='black')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(session_stats['session_label'], rotation=45, ha='right')
        ax2.set_xlabel('Session (Chronological Order)', fontsize=12)
        ax2.set_ylabel('Mean Licks per Bout', fontsize=12)
        ax2.set_title('Mean Licks per Bout Across Sessions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'licks_per_bout.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'licks_per_bout.png'}")
        plt.close()
    
    def plot_max_area_per_lick(self):
        """
        Plot maximum tongue area per lick across sessions
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Box plot of max area per lick for each session
        ax1 = axes[0]
        session_order = self.unique_sessions['session_label'].tolist()
        
        sns.violinplot(data=self.all_data, x='session_label', y='max_area', 
                       order=session_order, ax=ax1, cut=0)
        ax1.set_xlabel('Session (Chronological Order)', fontsize=12)
        ax1.set_ylabel('Max Tongue Area per Lick', fontsize=12)
        ax1.set_title('Distribution of Max Tongue Area per Lick Across Sessions', 
                      fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Mean max area per lick with error bars
        ax2 = axes[1]
        session_stats = self.all_data.groupby('session_label')['max_area'].agg(['mean', 'sem']).reset_index()
        session_stats = session_stats.set_index('session_label').reindex(session_order).reset_index()
        
        x_pos = np.arange(len(session_stats))
        ax2.bar(x_pos, session_stats['mean'], yerr=session_stats['sem'], 
                capsize=5, alpha=0.7, edgecolor='black', color='coral')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(session_stats['session_label'], rotation=45, ha='right')
        ax2.set_xlabel('Session (Chronological Order)', fontsize=12)
        ax2.set_ylabel('Mean Max Tongue Area', fontsize=12)
        ax2.set_title('Mean Max Tongue Area per Lick Across Sessions', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'max_area_per_lick.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'max_area_per_lick.png'}")
        plt.close()
    
    def plot_temporal_trends(self):
        """
        Plot temporal trends across days (combining trials from same day)
        """
        # Get bout stats per session first, then add date info
        daily_bout_stats = self.bout_metrics.groupby('session_name').agg({
            'licks_per_bout': ['mean', 'std'],
            'mean_area_per_bout': ['mean', 'std']
        }).reset_index()
        
        # Flatten the multi-level columns
        daily_bout_stats.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                                     for col in daily_bout_stats.columns.values]
        
        # Merge with date info (now both have single-level columns)
        daily_bout_stats = daily_bout_stats.merge(
            self.unique_sessions[['session_name', 'date_str', 'date_obj']].drop_duplicates(),
            on='session_name'
        )
        
        # Group by date
        daily_bout_stats = daily_bout_stats.groupby('date_str').agg({
            'licks_per_bout_mean': 'mean',
            'mean_area_per_bout_mean': 'mean'
        }).reset_index()
        
        # Calculate normalized licks per session (licks per frame)
        session_lick_stats = []
        for _, session in self.unique_sessions.iterrows():
            session_name = session['session_name']
            session_licks = self.all_data[self.all_data['session_name'] == session_name]
            
            n_licks = len(session_licks)
            frame_count = self.frame_counts.get(session_name, None)
            
            session_lick_stats.append({
                'session_label': session['session_label'],
                'session_name': session_name,
                'date_str': session['date_str'],
                'date_obj': session['date_obj'],
                'n_licks': n_licks,
                'frame_count': frame_count,
                'licks_per_frame': n_licks / frame_count if frame_count else np.nan
            })
        
        session_lick_df = pd.DataFrame(session_lick_stats)
        
        # Group by date for normalized licks
        daily_lick_stats = session_lick_df.groupby('date_str').agg({
            'licks_per_frame': 'mean',
            'n_licks': 'sum'
        }).reset_index()
        
        # Also get mean area and duration per date
        daily_area_duration = self.all_data.groupby('date_str').agg({
            'max_area': 'mean',
            'duration': 'mean'
        }).reset_index()
        
        daily_lick_stats = daily_lick_stats.merge(daily_area_duration, on='date_str')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Get dates in order
        date_order = sorted(self.all_data['date_str'].unique(), 
                           key=lambda x: self.all_data[self.all_data['date_str']==x]['date_obj'].iloc[0])
        
        # Plot 1: Normalized licks per frame per day
        ax1 = axes[0, 0]
        daily_lick_stats = daily_lick_stats.set_index('date_str').reindex(date_order).reset_index()
        
        # Check if we have frame count data
        has_frame_data = daily_lick_stats['licks_per_frame'].notna().any()
        
        if has_frame_data:
            ax1.plot(daily_lick_stats['date_str'], daily_lick_stats['licks_per_frame'], 
                    marker='o', linewidth=2, markersize=8)
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Total Licks (Normalized)', fontsize=12)
            ax1.set_title('Total Licks (Normalized) per Day', fontsize=12, fontweight='bold')
        else:
            # Fallback to total licks if no frame data
            ax1.plot(daily_lick_stats['date_str'], daily_lick_stats['n_licks'], 
                    marker='o', linewidth=2, markersize=8)
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Total Licks', fontsize=12)
            ax1.set_title('Total Licks per Day (Not Normalized)', fontsize=12, fontweight='bold')
            ax1.text(0.5, 0.95, 'Frame count data not available', 
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean licks per bout over time
        ax2 = axes[0, 1]
        daily_bout_stats = daily_bout_stats.set_index('date_str').reindex(date_order).reset_index()
        ax2.plot(daily_bout_stats['date_str'], daily_bout_stats['licks_per_bout_mean'], 
                marker='s', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Mean Licks per Bout', fontsize=12)
        ax2.set_title('Mean Licks per Bout Over Time', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mean max area over time
        ax3 = axes[1, 0]
        ax3.plot(daily_lick_stats['date_str'], daily_lick_stats['max_area'], 
                marker='^', linewidth=2, markersize=8, color='coral')
        ax3.set_xlabel('Date', fontsize=12)
        ax3.set_ylabel('Mean Max Tongue Area', fontsize=12)
        ax3.set_title('Mean Max Tongue Area Over Time', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Mean lick duration over time
        ax4 = axes[1, 1]
        ax4.plot(daily_lick_stats['date_str'], daily_lick_stats['duration'], 
                marker='D', linewidth=2, markersize=8, color='purple')
        ax4.set_xlabel('Date', fontsize=12)
        ax4.set_ylabel('Mean Lick Duration (frames)', fontsize=12)
        ax4.set_title('Mean Lick Duration Over Time', fontsize=12, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_trends.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'temporal_trends.png'}")
        plt.close()
    
    def plot_bout_characteristics(self):
        """
        Plot interesting bout characteristics
        """
        fig, ax = plt.subplots(1, 1, figsize=(24, 10))
        
        # Number of bouts per session (normalized by frames)
        bouts_per_session = self.bout_metrics.groupby(['session_label', 'session_name']).size().reset_index(name='n_bouts')
        session_order = self.unique_sessions['session_label'].tolist()
        
        # Add frame count information
        bouts_per_session['frame_count'] = bouts_per_session['session_name'].map(self.frame_counts)
        bouts_per_session['normalized_bouts'] = bouts_per_session['n_bouts'] / bouts_per_session['frame_count']
        
        # Reindex to match session order
        bouts_per_session = bouts_per_session.set_index('session_label').reindex(session_order).reset_index()
        
        # Check if we have frame data
        has_frame_data = bouts_per_session['normalized_bouts'].notna().any()
        
        if has_frame_data:
            # Plot normalized bouts (total bouts / frames)
            ax.bar(range(len(bouts_per_session)), bouts_per_session['normalized_bouts'], 
                   alpha=0.7, edgecolor='black', color='steelblue')
            ax.set_xlabel('Session', fontsize=16, labelpad=10)
            ax.set_ylabel('Bouts / Frames', fontsize=16)
            ax.set_title('Normalized Bouts per Session', fontsize=18, fontweight='bold', pad=20)
        else:
            # Fallback to raw counts if no frame data
            ax.bar(range(len(bouts_per_session)), bouts_per_session['n_bouts'], 
                   alpha=0.7, edgecolor='black')
            ax.set_xlabel('Session', fontsize=16, labelpad=10)
            ax.set_ylabel('Number of Bouts', fontsize=16)
            ax.set_title('Number of Bouts per Session (Not Normalized)', fontsize=18, fontweight='bold', pad=20)
            ax.text(0.5, 0.95, 'Frame count data not available', 
                    transform=ax.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        ax.set_xticks(range(len(bouts_per_session)))
        ax.set_xticklabels(bouts_per_session['session_label'], rotation=45, ha='right', fontsize=12)
        ax.tick_params(axis='x', pad=8)
        
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(self.output_dir / 'bout_characteristics.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'bout_characteristics.png'}")
        plt.close()
    
    def plot_heatmap_analysis(self):
        """
        Create heatmap visualizations of licking patterns
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap 1: Mean max area per session
        ax1 = axes[0]
        session_area = self.all_data.groupby('session_label')['max_area'].mean().reset_index()
        session_order = self.unique_sessions['session_label'].tolist()
        session_area = session_area.set_index('session_label').reindex(session_order).reset_index()
        
        # Create matrix for heatmap
        matrix1 = session_area['max_area'].values.reshape(-1, 1)
        sns.heatmap(matrix1.T, ax=ax1, cmap='YlOrRd', annot=True, fmt='.1f',
                   xticklabels=session_area['session_label'], yticklabels=['Mean Max Area'],
                   cbar_kws={'label': 'Tongue Area'})
        ax1.set_title('Mean Max Tongue Area Across Sessions', fontsize=12, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Heatmap 2: Mean licks per bout per session
        ax2 = axes[1]
        session_licks = self.bout_metrics.groupby('session_label')['licks_per_bout'].mean().reset_index()
        session_licks = session_licks.set_index('session_label').reindex(session_order).reset_index()
        
        matrix2 = session_licks['licks_per_bout'].values.reshape(-1, 1)
        sns.heatmap(matrix2.T, ax=ax2, cmap='Blues', annot=True, fmt='.1f',
                   xticklabels=session_licks['session_label'], yticklabels=['Mean Licks/Bout'],
                   cbar_kws={'label': 'Licks per Bout'})
        ax2.set_title('Mean Licks per Bout Across Sessions', fontsize=12, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'heatmap_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'heatmap_analysis.png'}")
        plt.close()
    
    def plot_bout_length_comparison(self):
        """
        Compare bout length distributions between early and late sessions
        """
        # Sort sessions by date
        session_order = self.unique_sessions.sort_values(['date_obj', 'trial_num'])
        
        # Split into early (first ~1/3) and late (last ~1/3) sessions
        n_sessions = len(session_order)
        split_point = n_sessions // 3
        
        early_sessions = session_order.iloc[:split_point]['session_name'].tolist()
        late_sessions = session_order.iloc[-split_point:]['session_name'].tolist()
        
        # Get bout data for early and late periods
        early_bouts = self.bout_metrics[self.bout_metrics['session_name'].isin(early_sessions)]
        late_bouts = self.bout_metrics[self.bout_metrics['session_name'].isin(late_sessions)]
        
        # Create figure with multiple comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Distribution comparison - Licks per bout
        ax1 = axes[0, 0]
        
        # Create bins for histogram
        max_licks = max(early_bouts['licks_per_bout'].max(), late_bouts['licks_per_bout'].max())
        bins = np.arange(0, max_licks + 2, 1)
        
        ax1.hist(early_bouts['licks_per_bout'], bins=bins, alpha=0.6, 
                label=f'Early Sessions (n={len(early_bouts)})', color='steelblue', edgecolor='black')
        ax1.hist(late_bouts['licks_per_bout'], bins=bins, alpha=0.6, 
                label=f'Late Sessions (n={len(late_bouts)})', color='coral', edgecolor='black')
        ax1.set_xlabel('Licks per Bout', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Bout Length Distribution: Early vs Late Sessions', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Box plot comparison
        ax2 = axes[0, 1]
        bout_comparison_data = pd.DataFrame({
            'Licks per Bout': list(early_bouts['licks_per_bout']) + list(late_bouts['licks_per_bout']),
            'Period': ['Early'] * len(early_bouts) + ['Late'] * len(late_bouts)
        })
        
        sns.boxplot(data=bout_comparison_data, x='Period', y='Licks per Bout', ax=ax2, 
                   palette={'Early': 'steelblue', 'Late': 'coral'})
        sns.swarmplot(data=bout_comparison_data, x='Period', y='Licks per Bout', ax=ax2, 
                     color='black', alpha=0.3, size=3)
        ax2.set_ylabel('Licks per Bout', fontsize=12)
        ax2.set_xlabel('Session Period', fontsize=12)
        ax2.set_title('Bout Length Comparison', fontsize=12, fontweight='bold')
        
        # Add statistics
        early_mean = early_bouts['licks_per_bout'].mean()
        late_mean = late_bouts['licks_per_bout'].mean()
        early_median = early_bouts['licks_per_bout'].median()
        late_median = late_bouts['licks_per_bout'].median()
        
        stats_text = f'Early: μ={early_mean:.1f}, Med={early_median:.0f}\n'
        stats_text += f'Late: μ={late_mean:.1f}, Med={late_median:.0f}'
        ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 3: CDF (Cumulative Distribution Function)
        ax3 = axes[1, 0]
        
        early_sorted = np.sort(early_bouts['licks_per_bout'])
        late_sorted = np.sort(late_bouts['licks_per_bout'])
        
        early_cdf = np.arange(1, len(early_sorted) + 1) / len(early_sorted)
        late_cdf = np.arange(1, len(late_sorted) + 1) / len(late_sorted)
        
        ax3.plot(early_sorted, early_cdf, label='Early Sessions', linewidth=2, color='steelblue')
        ax3.plot(late_sorted, late_cdf, label='Late Sessions', linewidth=2, color='coral')
        ax3.set_xlabel('Licks per Bout', fontsize=12)
        ax3.set_ylabel('Cumulative Probability', fontsize=12)
        ax3.set_title('Cumulative Distribution: Bout Lengths', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Proportion of long bouts over time
        ax4 = axes[1, 1]
        
        # Define "long bout" as >= 10 licks (you can adjust this threshold)
        long_bout_threshold = 10
        
        session_bout_proportions = []
        for _, session in session_order.iterrows():
            session_name = session['session_name']
            session_bouts = self.bout_metrics[self.bout_metrics['session_name'] == session_name]
            
            if len(session_bouts) > 0:
                prop_long = (session_bouts['licks_per_bout'] >= long_bout_threshold).sum() / len(session_bouts)
            else:
                prop_long = 0
            
            session_bout_proportions.append({
                'session_label': session['session_label'],
                'date_obj': session['date_obj'],
                'proportion_long_bouts': prop_long
            })
        
        prop_df = pd.DataFrame(session_bout_proportions)
        
        ax4.plot(range(len(prop_df)), prop_df['proportion_long_bouts'], 
                marker='o', linewidth=2, markersize=6, color='darkgreen')
        ax4.axhline(y=prop_df['proportion_long_bouts'].mean(), 
                   color='red', linestyle='--', alpha=0.5, label='Mean')
        ax4.set_xlabel('Session (Chronological Order)', fontsize=12)
        ax4.set_ylabel(f'Proportion of Bouts ≥ {long_bout_threshold} Licks', fontsize=12)
        ax4.set_title(f'Trend in Long Bouts Over Time', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(0, len(prop_df), max(1, len(prop_df)//10)))
        ax4.set_xticklabels([prop_df['session_label'].iloc[i] for i in range(0, len(prop_df), max(1, len(prop_df)//10))], 
                           rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'bout_length_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'bout_length_comparison.png'}")
        plt.close()
    
    def plot_max_bout_length_over_time(self):
        """
        Plot maximum bout length and bout length statistics across sessions over time
        """
        # Get bout statistics per session
        session_order = self.unique_sessions['session_label'].tolist()
        
        bout_stats_per_session = []
        for session_label in session_order:
            session_bouts = self.bout_metrics[self.bout_metrics['session_label'] == session_label]
            
            if len(session_bouts) > 0:
                bout_stats_per_session.append({
                    'session_label': session_label,
                    'max_bout_length': session_bouts['licks_per_bout'].max(),
                    'mean_bout_length': session_bouts['licks_per_bout'].mean(),
                    'median_bout_length': session_bouts['licks_per_bout'].median(),
                    'percentile_90': session_bouts['licks_per_bout'].quantile(0.9),
                    'percentile_75': session_bouts['licks_per_bout'].quantile(0.75),
                    'min_bout_length': session_bouts['licks_per_bout'].min(),
                    'n_bouts': len(session_bouts)
                })
            else:
                bout_stats_per_session.append({
                    'session_label': session_label,
                    'max_bout_length': 0,
                    'mean_bout_length': 0,
                    'median_bout_length': 0,
                    'percentile_90': 0,
                    'percentile_75': 0,
                    'min_bout_length': 0,
                    'n_bouts': 0
                })
        
        stats_df = pd.DataFrame(bout_stats_per_session)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        x_pos = np.arange(len(stats_df))
        
        # Plot 1: Maximum bout length per session
        ax1 = axes[0]
        colors = plt.cm.RdYlGn_r(stats_df['max_bout_length'] / stats_df['max_bout_length'].max())
        bars = ax1.bar(x_pos, stats_df['max_bout_length'], color=colors, edgecolor='black', alpha=0.8)
        ax1.set_xlabel('Session (Chronological Order)', fontsize=12)
        ax1.set_ylabel('Maximum Bout Length (licks)', fontsize=12)
        ax1.set_title('Maximum Bout Length Across Sessions', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(stats_df['session_label'], rotation=45, ha='right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(stats_df.iterrows()):
            if row['max_bout_length'] > 0:
                ax1.text(i, row['max_bout_length'] + 0.5, f"{int(row['max_bout_length'])}", 
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Plot 2: Range of bout lengths (min to max with quartiles)
        ax2 = axes[1]
        
        # Plot range as error bars from min to max
        ax2.errorbar(x_pos, stats_df['median_bout_length'], 
                    yerr=[stats_df['median_bout_length'] - stats_df['min_bout_length'],
                          stats_df['max_bout_length'] - stats_df['median_bout_length']],
                    fmt='o', color='steelblue', ecolor='lightgray', elinewidth=3, 
                    capsize=4, capthick=2, markersize=6, label='Min-Max Range')
        
        # Add 75th and 90th percentile lines
        ax2.plot(x_pos, stats_df['percentile_75'], 'g--', alpha=0.7, linewidth=2, label='75th Percentile')
        ax2.plot(x_pos, stats_df['percentile_90'], 'r--', alpha=0.7, linewidth=2, label='90th Percentile')
        
        ax2.set_xlabel('Session (Chronological Order)', fontsize=12)
        ax2.set_ylabel('Bout Length (licks)', fontsize=12)
        ax2.set_title('Bout Length Range Across Sessions', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(stats_df['session_label'], rotation=45, ha='right', fontsize=9)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Line plot showing decline in max bout length
        ax3 = axes[2]
        ax3.plot(x_pos, stats_df['max_bout_length'], marker='o', linewidth=2.5, 
                markersize=8, color='darkred', label='Max Bout Length', alpha=0.6)
        ax3.fill_between(x_pos, stats_df['max_bout_length'], alpha=0.3, color='red')
        
        # Add smoothed line using rolling average
        from scipy.ndimage import gaussian_filter1d
        smoothed = gaussian_filter1d(stats_df['max_bout_length'], sigma=2)
        ax3.plot(x_pos, smoothed, linewidth=3, alpha=0.9, color='navy', label='Smoothed Trend')
        
        ax3.set_xlabel('Session (Chronological Order)', fontsize=12)
        ax3.set_ylabel('Maximum Bout Length (licks)', fontsize=12)
        ax3.set_title('Maximum Bout Length Over Time', fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos[::2])  # Show every other label
        ax3.set_xticklabels(stats_df['session_label'].iloc[::2], rotation=45, ha='right', fontsize=9)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'max_bout_length_over_time.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir / 'max_bout_length_over_time.png'}")
        plt.close()
    
    def generate_summary_stats(self):
        """
        Generate and save summary statistics
        """
        summary = {}
        
        # Overall statistics
        summary['total_sessions'] = len(self.unique_sessions)
        summary['total_licks'] = len(self.all_data)
        summary['total_bouts'] = len(self.bout_metrics)
        summary['mean_licks_per_session'] = len(self.all_data) / len(self.unique_sessions)
        summary['mean_bouts_per_session'] = len(self.bout_metrics) / len(self.unique_sessions)
        summary['mean_licks_per_bout'] = self.bout_metrics['licks_per_bout'].mean()
        summary['median_licks_per_bout'] = self.bout_metrics['licks_per_bout'].median()
        summary['mean_max_area'] = self.all_data['max_area'].mean()
        summary['median_max_area'] = self.all_data['max_area'].median()
        summary['mean_lick_duration'] = self.all_data['duration'].mean()
        summary['mean_bout_duration'] = self.bout_metrics['total_bout_duration'].mean()
        
        # Per session statistics
        session_stats = []
        for _, session in self.unique_sessions.iterrows():
            session_name = session['session_name']
            session_label = session['session_label']
            
            session_licks = self.all_data[self.all_data['session_name'] == session_name]
            session_bouts = self.bout_metrics[self.bout_metrics['session_name'] == session_name]
            
            stats = {
                'session': session_label,
                'date': session['date_str'],
                'trial': session['trial_num'],
                'n_licks': len(session_licks),
                'n_bouts': len(session_bouts),
                'mean_licks_per_bout': session_bouts['licks_per_bout'].mean(),
                'mean_max_area': session_licks['max_area'].mean(),
                'mean_duration': session_licks['duration'].mean()
            }
            session_stats.append(stats)
        
        session_df = pd.DataFrame(session_stats)
        
        # Save to CSV
        session_df.to_csv(self.output_dir / 'session_summary.csv', index=False)
        print(f"Saved: {self.output_dir / 'session_summary.csv'}")
        
        # Save overall summary
        with open(self.output_dir / 'overall_summary.txt', 'w') as f:
            f.write("=== BEHAVIOR ANALYSIS SUMMARY ===\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value:.2f}\n")
        
        print(f"Saved: {self.output_dir / 'overall_summary.txt'}")
        
        return summary, session_df
    
    def run_full_analysis(self):
        """
        Run the complete analysis pipeline
        """
        print("\n" + "="*60)
        print("STARTING BEHAVIOR ANALYSIS")
        print("="*60 + "\n")
        
        # Load data
        self.load_all_sessions()
        
        # Calculate bout metrics
        print("\nCalculating bout metrics...")
        self.calculate_bout_metrics()
        
        # Generate all plots
        print("\nGenerating plots...")
        self.plot_licks_per_bout()
        self.plot_max_area_per_lick()
        self.plot_temporal_trends()
        self.plot_bout_characteristics()
        self.plot_heatmap_analysis()
        self.plot_bout_length_comparison()
        self.plot_max_bout_length_over_time()
        
        # Generate summary statistics
        print("\nGenerating summary statistics...")
        summary, session_df = self.generate_summary_stats()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated files:")
        print("  - licks_per_bout.png")
        print("  - max_area_per_lick.png")
        print("  - temporal_trends.png (with normalized licks per frame)")
        print("  - bout_characteristics.png")
        print("  - heatmap_analysis.png")
        print("  - bout_length_comparison.png (Early vs Late sessions)")
        print("  - max_bout_length_over_time.png (NEW: Maximum bout length decline)")
        print("  - session_summary.csv")
        print("  - overall_summary.txt")
        print("\n")
        
        return summary, session_df


def main():
    """
    Main function to run the analysis
    """
    # Set the base directory (WSL path)
    base_dir = "/mnt/c/Users/wanglab/Desktop/IRt.PCRt-1125/TeNT_1110/Phox2B#8_side_all"
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"ERROR: Directory not found: {base_dir}")
        print("Please check the path and try again.")
        return
    
    # Initialize analyzer
    analyzer = BehaviorAnalyzer(base_dir)
    
    # Run analysis
    try:
        summary, session_df = analyzer.run_full_analysis()
        
        # Print summary
        print("SUMMARY STATISTICS:")
        print("-" * 60)
        for key, value in summary.items():
            print(f"  {key}: {value:.2f}")
        print("-" * 60)
        
    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
