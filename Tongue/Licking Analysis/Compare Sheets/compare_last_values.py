import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the base directories
base_dir1 = r"C:\Users\wanglab\Desktop\PCRt_TeLC\Phox2b#38"
base_dir2 = r"C:\Users\wanglab\Desktop\PCRt_TeLC\Phox2b#38_1230_weights"

# Define the subdirectories and their corresponding CSV files
sessions = {
    '0307_1': 'phox2b38_20240307_side_100_3_behavior',
    '0318_1': 'phox2b38_20240318_side_100_3_behavior',
    '0319_1': 'phox2b38_20240319_side_100_3_behavior',
    '0320_1': 'phox2b38_20240320_side_100_3_behavior',
    '0321_1': 'phox2b38_20240321_side_100_3_behavior',
    '0322_1': 'phox2b38_20240322_side_100_3_behavior'
}

# Function to get the last value of the last column in a CSV file
def get_last_column_last_value(filepath):
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print(f"Warning: {filepath} is empty")
            return None
        # Get the last column name
        last_column = df.columns[-1]
        # Get the last value in that column
        last_value = df[last_column].iloc[-1]
        return last_value
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

# Collect last values
last_values = {}
column_name = None

print("=== Last Column Last Values ===\n")

for session_id, csv_name in sessions.items():
    # Build paths for both directories
    csv_path1 = os.path.join(base_dir1, session_id, f"{csv_name}.csv")
    csv_path2 = os.path.join(base_dir2, session_id, f"{csv_name}.csv")
    
    # Get last values for both directories
    value1 = get_last_column_last_value(csv_path1)
    value2 = get_last_column_last_value(csv_path2)
    
    if value1 is not None:
        last_values[session_id] = value1
        print(f"{session_id}: {value1}")
    
    if value2 is not None:
        last_values[f"{session_id}-1230"] = value2
        print(f"{session_id}-1230: {value2}")
    
    # Get column name from first file for display
    if column_name is None and value1 is not None:
        try:
            df = pd.read_csv(csv_path1)
            column_name = df.columns[-1]
        except:
            pass

print(f"\nColumn name: {column_name}")

# Create comparison plot
fig, ax = plt.subplots(figsize=(14, 7))

# Prepare data for plotting
labels = []
values = []

for session_id in sorted(sessions.keys()):
    if session_id in last_values:
        labels.append(session_id)
        values.append(last_values[session_id])
    
    key_1230 = f"{session_id}-1230"
    if key_1230 in last_values:
        labels.append(key_1230)
        values.append(last_values[key_1230])

# Create bar plot with colors
colors = []
for label in labels:
    if '-1230' in label:
        colors.append('coral')
    else:
        colors.append('steelblue')

bars = ax.bar(range(len(labels)), values, color=colors, edgecolor='black', linewidth=1.2)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('Last Value in Last Column', fontsize=12)
ax.set_xlabel('Session', fontsize=12)

# Create a more concise title
title = 'Last Column Last Value Comparison:\nPhox2b#38 vs Phox2b#38_1230_weights'
if column_name and len(column_name) < 50:
    title = f'Last Value Comparison - {column_name}\nPhox2b#38 vs Phox2b#38_1230_weights'
ax.set_title(title, fontsize=11, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars, values)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(value)}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', edgecolor='black', label='Phox2b#38'),
    Patch(facecolor='coral', edgecolor='black', label='Phox2b#38_1230_weights')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()

# Save the plot
output_path = os.path.join(os.path.dirname(__file__), 'last_value_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Print summary comparison
print("\n=== Summary Comparison ===")
for session_id in sorted(sessions.keys()):
    if session_id in last_values and f"{session_id}-1230" in last_values:
        value1 = last_values[session_id]
        value2 = last_values[f"{session_id}-1230"]
        diff = value2 - value1
        print(f"{session_id}: {value1} vs {value2} (diff: {diff:+d})")

plt.show()
