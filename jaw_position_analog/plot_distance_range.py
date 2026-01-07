import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def gh_filter(data, g=0.00005, h=0.01):
    """
    Apply g-h (alpha-beta) smoothing filter to data.
    
    Parameters:
    -----------
    data : array-like
        Input data to smooth
    g : float
        Position gain (alpha), range [0, 1], default 0.5
    h : float
        Velocity gain (beta), range [0, 1], default 0.1
    
    Returns:
    --------
    smoothed : numpy array
        Smoothed data
    """
    smoothed = np.zeros(len(data))
    x_pred = data[0]  # Initial position prediction
    v_pred = 0  # Initial velocity prediction
    
    for i in range(len(data)):
        # Update step
        residual = data[i] - x_pred
        x_est = x_pred + g * residual
        v_est = v_pred + h * residual
        
        smoothed[i] = x_est
        
        # Prediction step
        x_pred = x_est + v_est
        v_pred = v_est
    
    return smoothed

def plot_distance_range(csv_path, start_frame, end_frame, g=0.5, h=0.1):
    """
    Plot distance values from side+bottom.csv for a specific frame range.
    Applies g-h smoothing and saves both raw and smoothed data to CSV.
    
    Parameters:
    -----------
    csv_path : str
        Path to the side+bottom.csv file
    start_frame : int
        Starting frame number
    end_frame : int
        Ending frame number
    g : float
        Position gain for g-h filter (default: 0.5)
    h : float
        Velocity gain for g-h filter (default: 0.1)
    """
    
    # Read the CSV
    print(f"Reading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter for the specified frame range
    filtered_df = df[(df['Time'] >= start_frame) & (df['Time'] <= end_frame)].copy()
    
    print(f"Plotting frames {start_frame} to {end_frame}")
    print(f"Total frames in range: {len(filtered_df)}")
    print(f"Distance range: {filtered_df['Data'].min():.2f} to {filtered_df['Data'].max():.2f}")
    
    # Apply g-h smoothing
    print(f"Applying g-h smoothing (g={g}, h={h})...")
    filtered_df['Data_Smoothed'] = gh_filter(filtered_df['Data'].values, g=g, h=h)
    
    # Save to CSV
    output_dir = os.path.dirname(csv_path)
    csv_output_path = os.path.join(output_dir, f'distance_frames_{start_frame}_{end_frame}.csv')
    filtered_df.to_csv(csv_output_path, index=False)
    print(f"Data saved to: {csv_output_path}")
    
    # Create the plot
    plt.figure(figsize=(14, 6))
    plt.plot(filtered_df['Time'], filtered_df['Data'], linewidth=1, color='blue', alpha=0.5, label='Raw')
    plt.plot(filtered_df['Time'], filtered_df['Data_Smoothed'], linewidth=1.5, color='red', label=f'g-h Smoothed (g={g}, h={h})')
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.title(f'Distance vs Frame (Frames {start_frame}-{end_frame})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plot_output_path = os.path.join(output_dir, f'distance_plot_frames_{start_frame}_{end_frame}.png')
    plt.savefig(plot_output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_output_path}")
    
    # Show the plot
    plt.show()
    
    return filtered_df


if __name__ == "__main__":
    # Path to the side+bottom.csv file
    csv_path = r"C:\Users\wanglab\Desktop\IRt.PCRt-1125\IRt_114\IRt_02_2025_04251_side_and_bottom\side+bottom.csv"
    
    # Plot frames 61292 to 63250
    # Adjust g and h for more/less smoothing: higher g = more responsive, lower g = more smoothed
    df = plot_distance_range(csv_path, start_frame=61292, end_frame=63250, g=0.05, h=0.01)
