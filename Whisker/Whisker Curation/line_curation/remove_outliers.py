#!/usr/bin/env python3
"""
Outlier Removal Script for Line Curvature Data
This script identifies statistical outliers in the analog_output.csv file
and removes corresponding rows from both analog_output.csv and lines_output.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

def detect_outliers_iqr(data, multiplier=1.5):
    """
    Detect outliers using the Interquartile Range (IQR) method
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers, lower_bound, upper_bound

def detect_outliers_modified_zscore(data, threshold=3.5):
    """
    Detect outliers using Modified Z-score with Median Absolute Deviation (MAD)
    """
    median_val = data.median()
    mad = np.median(np.abs(data - median_val))
    modified_z_scores = 0.6745 * (data - median_val) / mad
    outliers = np.abs(modified_z_scores) > threshold
    return outliers

def detect_outliers_zscore(data, threshold=3):
    """
    Detect outliers using standard Z-score method
    """
    z_scores = np.abs((data - data.mean()) / data.std())
    outliers = z_scores > threshold
    return outliers

def main():
    # Set working directory
    os.chdir("c:/Users/wanglab/Desktop/REPO/licking-and-spike-analysis/line_curation")
    
    print("Reading data files...")
    # Read the data files
    try:
        analog_data = pd.read_csv("analog_output.csv")
        lines_data = pd.read_csv("lines_output.csv")
    except FileNotFoundError as e:
        print(f"Error reading files: {e}")
        return
    
    print(f"Original analog_output.csv: {len(analog_data)} rows")
    print(f"Original lines_output.csv: {len(lines_data)} rows")
    
    # Display basic statistics of the Data column
    print("\nOriginal Data Statistics:")
    print(f"Min: {analog_data['Data'].min():.4f}")
    print(f"Max: {analog_data['Data'].max():.4f}")
    print(f"Mean: {analog_data['Data'].mean():.4f}")
    print(f"Median: {analog_data['Data'].median():.4f}")
    print(f"Standard Deviation: {analog_data['Data'].std():.4f}")
    print(f"Range: {analog_data['Data'].max() - analog_data['Data'].min():.4f}")
    
    # Method 1: IQR-based outlier detection (more conservative)
    iqr_outliers, lower_bound_iqr, upper_bound_iqr = detect_outliers_iqr(analog_data['Data'], 1.5)
    
    print(f"\nIQR Method (1.5 * IQR):")
    print(f"Lower bound: {lower_bound_iqr:.4f}")
    print(f"Upper bound: {upper_bound_iqr:.4f}")
    print(f"Number of outliers detected: {iqr_outliers.sum()}")
    
    # Method 2: Modified Z-score method
    z_outliers = detect_outliers_modified_zscore(analog_data['Data'], 3.5)
    print(f"\nModified Z-score Method (threshold = 3.5):")
    print(f"Number of outliers detected: {z_outliers.sum()}")
    
    # Method 3: Standard Z-score method
    standard_z_outliers = detect_outliers_zscore(analog_data['Data'], 3)
    print(f"\nStandard Z-score Method (threshold = 3):")
    print(f"Number of outliers detected: {standard_z_outliers.sum()}")
    
    # Use IQR method as primary (you can change this)
    outlier_mask = iqr_outliers
    outlier_indices = analog_data[outlier_mask].index
    
    if len(outlier_indices) > 0:
        print(f"\n=== Using IQR Method ===")
        print(f"Removing {len(outlier_indices)} outlier rows...")
        
        # Get the Time/Frame values that will be removed
        outlier_times = analog_data.loc[outlier_indices, 'Time'].values
        
        print("\nOutlier values being removed:")
        outlier_summary = analog_data.loc[outlier_indices, ['Time', 'Data']]
        print(outlier_summary.to_string(index=False))
        
        # Remove outliers from analog_data
        analog_clean = analog_data[~outlier_mask].copy()
        
        # Remove corresponding rows from lines_data based on matching Frame values
        lines_clean = lines_data[~lines_data['Frame'].isin(outlier_times)].copy()
        
        print(f"\nCleaned analog_output.csv: {len(analog_clean)} rows (removed {len(analog_data) - len(analog_clean)})")
        print(f"Cleaned lines_output.csv: {len(lines_clean)} rows (removed {len(lines_data) - len(lines_clean)})")
        
        # Display cleaned statistics
        print("\nCleaned Data Statistics:")
        print(f"Min: {analog_clean['Data'].min():.4f}")
        print(f"Max: {analog_clean['Data'].max():.4f}")
        print(f"Mean: {analog_clean['Data'].mean():.4f}")
        print(f"Median: {analog_clean['Data'].median():.4f}")
        print(f"Standard Deviation: {analog_clean['Data'].std():.4f}")
        print(f"Range: {analog_clean['Data'].max() - analog_clean['Data'].min():.4f}")
        
        # Save the cleaned datasets with fixed decimal formatting
        # Format the Data column to 10 decimal places without scientific notation
        analog_clean_formatted = analog_clean.copy()
        analog_clean_formatted['Data'] = analog_clean_formatted['Data'].apply(lambda x: f"{x:.10f}")
        analog_clean_formatted.to_csv("analog_output_clean.csv", index=False)
        lines_clean.to_csv("lines_output_clean.csv", index=False)
        
        print("\nCleaned files saved as:")
        print("- analog_output_clean.csv")
        print("- lines_output_clean.csv")
        
        # Create visualizations
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Histogram of original data
            ax1.hist(analog_data['Data'], bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax1.axvline(lower_bound_iqr, color='red', linestyle='--', linewidth=2, label='IQR Lower Bound')
            ax1.axvline(upper_bound_iqr, color='red', linestyle='--', linewidth=2, label='IQR Upper Bound')
            ax1.set_title('Distribution of Original Line Curvature Data')
            ax1.set_xlabel('Line Curvature')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Histogram of cleaned data
            ax2.hist(analog_clean['Data'], bins=50, alpha=0.7, color='green', edgecolor='black')
            ax2.set_title('Distribution of Cleaned Line Curvature Data')
            ax2.set_xlabel('Line Curvature')
            ax2.set_ylabel('Frequency')
            ax2.grid(True, alpha=0.3)
            
            # Box plot comparison
            box_data = [analog_data['Data'], analog_clean['Data']]
            ax3.boxplot(box_data, labels=['Original', 'Cleaned'])
            ax3.set_title('Boxplot Comparison: Original vs Cleaned Data')
            ax3.set_ylabel('Line Curvature')
            ax3.grid(True, alpha=0.3)
            
            # Time series plot showing outliers
            ax4.plot(analog_data['Time'], analog_data['Data'], 'b-', alpha=0.6, linewidth=0.5, label='Original Data')
            ax4.scatter(analog_data.loc[outlier_indices, 'Time'], 
                       analog_data.loc[outlier_indices, 'Data'], 
                       color='red', s=20, label='Outliers', zorder=5)
            ax4.set_title('Time Series with Outliers Highlighted')
            ax4.set_xlabel('Time/Frame')
            ax4.set_ylabel('Line Curvature')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('outlier_analysis.png', dpi=300, bbox_inches='tight')
            print("\nOutlier analysis plot saved as 'outlier_analysis.png'")
            
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
        
        # Save outlier summary with proper formatting
        outlier_report = pd.DataFrame({
            'Time': outlier_times,
            'Data': [f"{x:.10f}" for x in analog_data.loc[outlier_indices, 'Data'].values],
            'Method': 'IQR_1.5',
            'Lower_Bound': f"{lower_bound_iqr:.10f}",
            'Upper_Bound': f"{upper_bound_iqr:.10f}"
        })
        outlier_report.to_csv("outliers_removed.csv", index=False)
        print("- outliers_removed.csv (summary of removed data points)")
        
    else:
        print("\nNo outliers detected with the current method.")
        print("You may want to adjust the outlier detection parameters.")
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    main()