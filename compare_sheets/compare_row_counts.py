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

# Function to count rows in a CSV file
def count_csv_rows(filepath):
    try:
        df = pd.read_csv(filepath)
        return len(df)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

# Collect row counts
row_counts = {}

for session_id, csv_name in sessions.items():
    # Find CSV files in subdirectories
    # Check for .csv extension
    csv_path1 = os.path.join(base_dir1, session_id, f"{csv_name}.csv")
    csv_path2 = os.path.join(base_dir2, session_id, f"{csv_name}.csv")
    
    # Count rows for both directories
    count1 = count_csv_rows(csv_path1)
    count2 = count_csv_rows(csv_path2)
    
    if count1 is not None:
        row_counts[session_id] = count1
        print(f"{session_id}: {count1} rows")
    
    if count2 is not None:
        row_counts[f"{session_id}-1230"] = count2
        print(f"{session_id}-1230: {count2} rows")

# Create comparison plot
fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data for plotting
labels = []
counts = []

for session_id in sorted(sessions.keys()):
    if session_id in row_counts:
        labels.append(session_id)
        counts.append(row_counts[session_id])
    
    key_1230 = f"{session_id}-1230"
    if key_1230 in row_counts:
        labels.append(key_1230)
        counts.append(row_counts[key_1230])

# Create bar plot
colors = []
for label in labels:
    if '-1230' in label:
        colors.append('coral')
    else:
        colors.append('steelblue')

bars = ax.bar(range(len(labels)), counts, color=colors)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_ylabel('Number of Rows')
ax.set_xlabel('Session')
ax.set_title('CSV Row Count Comparison: Phox2b#38 vs Phox2b#38_1230_weights')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, count) in enumerate(zip(bars, counts)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count)}',
            ha='center', va='bottom', fontsize=9)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', label='Phox2b#38'),
    Patch(facecolor='coral', label='Phox2b#38_1230_weights')
]
ax.legend(handles=legend_elements)

plt.tight_layout()

# Save the plot
output_path = os.path.join(os.path.dirname(__file__), 'row_count_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")

# Print summary comparison
print("\n=== Summary Comparison ===")
for session_id in sorted(sessions.keys()):
    if session_id in row_counts and f"{session_id}-1230" in row_counts:
        count1 = row_counts[session_id]
        count2 = row_counts[f"{session_id}-1230"]
        diff = count2 - count1
        diff_pct = (diff / count1 * 100) if count1 > 0 else 0
        print(f"{session_id}: {count1} vs {count2} (diff: {diff:+d}, {diff_pct:+.1f}%)")

plt.show()
