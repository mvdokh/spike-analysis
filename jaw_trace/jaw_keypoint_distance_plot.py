import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SVG
import matplotlib.pyplot as plt
import os

# g-h filter implementation
def gh_filter(z, x0, dx, g, h):
    x_est = x0
    dx_est = dx
    result = []
    for measurement in z:
        # Prediction step
        x_pred = x_est + dx_est
        dx_pred = dx_est
        # Update step
        residual = measurement - x_pred
        x_est = x_pred + g * residual
        dx_est = dx_pred + h * residual
        result.append(x_est)
    return np.array(result)

# Set paths
video_dir = r"C:\Users\wanglab\Desktop\IRt.PCRt-1125\IRt_114\IRt_01_2025_04251"
csv_path = os.path.join(video_dir, "IRt_01_2025_04251_jaw.csv")

# Load jaw keypoints CSV (handle space or comma delimited)
try:
    jaw_df = pd.read_csv(csv_path)
    if len(jaw_df.columns) == 1:
        # Try space-delimited
        jaw_df = pd.read_csv(csv_path, delim_whitespace=True)
except Exception as e:
    # Fallback to space-delimited if any error
    jaw_df = pd.read_csv(csv_path, delim_whitespace=True)

cols = {col.lower(): col for col in jaw_df.columns}
if 'x' in cols and 'y' in cols:
    x = jaw_df[cols['x']]
    y = jaw_df[cols['y']]
else:
    raise ValueError("CSV must contain 'x' and 'y' columns (case-insensitive). Found columns: {}".format(jaw_df.columns.tolist()))


# Calculate distance of each keypoint from the mean (using provided reference)
ref_x = 117.35
ref_y = 178.89
distances = np.sqrt((x - ref_x) ** 2 + (y - ref_y) ** 2)

# Compute frame-to-frame movement distances
frame_movements = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
move_mean = frame_movements.mean()
move_std = frame_movements.std()
outlier_thresh = move_mean + 3 * move_std
outlier_mask = np.insert(frame_movements > outlier_thresh, 0, False)  # Insert False for first frame

# Mask outliers in the distance plot
distances_filtered = distances.copy()
distances_filtered[outlier_mask] = np.nan

# Plot distance vs frame for only the first 10,000 frames and save as SVG
# Set frame range for plotting

start_frame = 44400    # inclusive
end_frame = 46250  # exclusive

# Only include non-outlier frames in the plot
frame_range = np.arange(start_frame, end_frame)
non_outlier_mask = ~outlier_mask[start_frame:end_frame]
plot_indices = jaw_df.index[start_frame:end_frame][non_outlier_mask]
plot_distances = distances[start_frame:end_frame][non_outlier_mask]


g = 0.05  # gain for value (lower for smoother)
h = 0.005 # gain for derivative (lower for smoother)
x0 = plot_distances.iloc[0] if len(plot_distances) > 0 else 0
dx0 = 0
smoothed = gh_filter(plot_distances, x0, dx0, g, h)
x0 = plot_distances.iloc[0] if len(plot_distances) > 0 else 0
dx0 = 0
smoothed = gh_filter(plot_distances, x0, dx0, g, h)

plt.figure(figsize=(8, 4))
plt.plot(plot_indices, plot_distances, marker='o', linestyle='-', color='blue', alpha=0.5, label='Original')
plt.plot(plot_indices, smoothed, color='red', linewidth=2, label='g-h Smoothed')
plt.xlabel('Frame')
plt.ylabel('Distance from Reference Keypoint')
plt.title('Distance of Jaw Keypoint from Reference')
plt.legend()
plt.tight_layout()
svg_output_path = os.path.join(os.path.dirname(__file__), 'jaw_keypoint_distances.svg')
plt.savefig(svg_output_path, format='svg')
print(f"Saved distance plot to {svg_output_path}")
