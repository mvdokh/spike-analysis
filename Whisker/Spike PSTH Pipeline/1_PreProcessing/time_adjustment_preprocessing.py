#!/usr/bin/env python3
"""
Time Adjustment Preprocessing Script

This script divides the first three columns of pico.csv by 30,000 to convert 
from sample units to time units (seconds), and saves the result as pico_time_adjust.csv.

The division by 30,000 converts from sampling frequency units to seconds.
"""

import pandas as pd
import os

def process_pico_csv(input_file, output_file, divisor=30000):
    """
    Process pico.csv by dividing the first three columns by the specified divisor.
    
    Parameters:
    - input_file: Path to the input pico.csv file
    - output_file: Path to save the adjusted CSV file
    - divisor: Value to divide by (default: 30000)
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        return False
    
    try:
        # Read the CSV file
        print(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        
        # Display original data info
        print(f"Original data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows (original):")
        print(df.head())
        
        # Make a copy to avoid modifying original
        df_adjusted = df.copy()
        
        # Get the first three columns (regardless of their names)
        first_three_cols = df.columns[:3]
        
        # Divide the first three columns by the divisor
        for col in first_three_cols:
            df_adjusted[col] = df[col] / divisor
            print(f"Divided column '{col}' by {divisor}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        
        # Save the adjusted data
        df_adjusted.to_csv(output_file, index=False)
        print(f"\nSaved adjusted data to: {output_file}")
        
        # Display adjusted data info
        print(f"Adjusted data shape: {df_adjusted.shape}")
        print("\nFirst few rows (adjusted):")
        print(df_adjusted.head())
        
        # Show conversion example
        print(f"\nConversion example (first row):")
        for col in first_three_cols:
            original_val = df[col].iloc[0]
            adjusted_val = df_adjusted[col].iloc[0]
            print(f"  {col}: {original_val:,.0f} → {adjusted_val:.6f} seconds")
        
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return False

def main():
    """Main function to process the pico.csv file"""
    
    # Define file paths
    input_file = "/home/wanglab/spike-analysis/Data/052725_1/pico.csv"
    output_file = "/home/wanglab/spike-analysis/Data/052725_1/pico_time_adjust.csv"
    
    print("="*60)
    print("TIME ADJUSTMENT PREPROCESSING")
    print("="*60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Divisor: 30,000 (converts sampling units to seconds)")
    print("-"*60)
    
    # Process the file
    success = process_pico_csv(input_file, output_file, divisor=30000)
    
    if success:
        print("\n" + "="*60)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("PROCESSING FAILED!")
        print("="*60)

if __name__ == "__main__":
    main()
