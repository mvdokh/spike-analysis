#!/usr/bin/env python3
"""
Script to analyze behavior CSV rows to video frames ratios across multiple folders.
"""

import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

def count_csv_rows(csv_path):
    """Count rows in a CSV file (excluding header)."""
    try:
        df = pd.read_csv(csv_path)
        return len(df)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def count_video_frames(video_path):
    """Count frames in a video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return frame_count
    except Exception as e:
        print(f"Error counting frames in {video_path}: {e}")
        return None

def main():
    # Define the folders and their expected file patterns
    base_path = r"C:\Users\wanglab\Desktop\IRt_Bipoles_1-20-26"
    
    folders = [
        "IRt_TeLC05",
        "IRt_TeLC05_1", 
        "IRt_TeLC06",
        "IRt_TeLC06_1",
        "IRt_TeLC07",
        "IRt_TeLC07_1"
    ]
    
    results = {}
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        print(f"\nProcessing folder: {folder}")
        
        # Find video file (.mp4)
        video_file = None
        behavior_csv = None
        
        for file in os.listdir(folder_path):
            if file.endswith('.mp4'):
                video_file = os.path.join(folder_path, file)
            elif 'behavior' in file.lower() and file.endswith('.csv'):
                behavior_csv = os.path.join(folder_path, file)
        
        if not video_file:
            print(f"  No video file found in {folder}")
            continue
        if not behavior_csv:
            print(f"  No behavior CSV file found in {folder}")
            continue
            
        print(f"  Video: {os.path.basename(video_file)}")
        print(f"  Behavior CSV: {os.path.basename(behavior_csv)}")
        
        # Count rows and frames
        csv_rows = count_csv_rows(behavior_csv)
        video_frames = count_video_frames(video_file)
        
        if csv_rows is not None and video_frames is not None:
            ratio = csv_rows / video_frames if video_frames > 0 else 0
            results[folder] = {
                'csv_rows': csv_rows,
                'video_frames': video_frames,
                'ratio': ratio
            }
            print(f"  CSV rows: {csv_rows}")
            print(f"  Video frames: {video_frames}")
            print(f"  Ratio: {ratio:.4f}")
        else:
            print(f"  Failed to process {folder}")
    
    # Create bar plot
    if results:
        folders = list(results.keys())
        ratios = [results[folder]['ratio'] for folder in folders]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(folders, ratios, color='skyblue', edgecolor='darkblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.4f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Folder', fontsize=12)
        plt.ylabel('CSV Rows / Video Frames Ratio', fontsize=12)
        plt.title('Behavior CSV Rows to Video Frames Ratio by Folder', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(os.path.dirname(__file__), 'behavior_video_ratios.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
        plt.show()
        
        # Print summary table
        print("\n" + "="*60)
        print("SUMMARY TABLE")
        print("="*60)
        print(f"{'Folder':<20} {'CSV Rows':<10} {'Video Frames':<12} {'Ratio':<8}")
        print("-" * 60)
        for folder in folders:
            data = results[folder]
            print(f"{folder:<20} {data['csv_rows']:<10} {data['video_frames']:<12} {data['ratio']:<8.4f}")
    else:
        print("No valid data found to plot.")

if __name__ == "__main__":
    main()