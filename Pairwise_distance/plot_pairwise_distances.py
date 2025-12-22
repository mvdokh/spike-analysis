"""
Time Series Plot of Pairwise Distances
Visualizes all pairwise distances across epochs
Creates separate plots for each class
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")

# Read the data
csv_path = Path(__file__).parent / "pair_distances.csv"
df = pd.read_csv(csv_path)

# Get all column names except 'epoch'
distance_columns = [col for col in df.columns if col != 'epoch']

# Define classes
classes = ['Background', 'Whisker 1', 'Whisker 2', 'Whisker 3', 'Whisker 4', 'Whisker 5']

# Create diverse color palette and line styles
colors = list(plt.cm.tab20c.colors) + list(plt.cm.Set3.colors) + list(plt.cm.Paired.colors)
line_styles = ['-', '--', '-.', ':', (0, (5, 2)), (0, (3, 1, 1, 1)), (0, (5, 1)), (0, (3, 5, 1, 5))]

# Create a plot for each class
for class_name in classes:
    # Filter columns that contain this class name
    class_columns = [col for col in distance_columns if class_name in col]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(24, 10))
    
    # Plot each pairwise distance for this class
    for i, col in enumerate(class_columns):
        color = colors[i % len(colors)]
        linestyle = line_styles[i % len(line_styles)]
        ax.plot(df['epoch'], df[col], label=col, alpha=0.85, linewidth=2.5, 
                color=color, linestyle=linestyle)
    
    # Customize the plot
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Pairwise Distance', fontsize=14, fontweight='bold')
    ax.set_title(f'{class_name} - Pairwise Distances Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, 
              framealpha=0.9, edgecolor='black')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the figure as SVG
    safe_name = class_name.replace(' ', '_')
    output_path = Path(__file__).parent / f"pairwise_distances_{safe_name}.svg"
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.close()

print("\nAll plots generated successfully!")
