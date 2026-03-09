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
video_dir = r"C:\Users\wanglab\Desktop\IRt.PCRt-1125\IRt_114\IRt_02_2025_04251"
csv_path = os.path.join(video_dir, "IRt_02_2025_04251_jaw.csv")

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
ref_x = 191.96
ref_y = 111.26
distances = np.sqrt((x - ref_x) ** 2 + (y - ref_y) ** 2)

# Compute frame-to-frame movement distances
frame_movements = np.sqrt(np.diff(x)**2 + np.diff(y)**2)


# Exclude frames where frame-to-frame movement exceeds 10 pixels
outlier_thresh = 10
outlier_mask = np.insert(frame_movements > outlier_thresh, 0, False)  # Insert False for first frame

# Mask outliers in the distance plot
distances_filtered = distances.copy()
distances_filtered[outlier_mask] = np.nan

# Plot distance vs frame for only the first 10,000 frames and save as SVG
# Set frame range for plotting

start_frame = 50750    # inclusive
end_frame = 52250  # exclusive



# Build cleaned sequence: only include frames with <=10 pixel movement from last included frame
selected_indices = jaw_df.index[start_frame:end_frame]
selected_x = x[start_frame:end_frame].to_numpy()
selected_y = y[start_frame:end_frame].to_numpy()
selected_distances = distances[start_frame:end_frame]

clean_indices = []
clean_distances = []
last_x = None
last_y = None
for idx, xi, yi, di in zip(selected_indices, selected_x, selected_y, selected_distances):
    if last_x is None:
        clean_indices.append(idx)
        clean_distances.append(di)
        last_x, last_y = xi, yi
    else:
        move = np.sqrt((xi - last_x)**2 + (yi - last_y)**2)
        if move <= 10:
            clean_indices.append(idx)
            clean_distances.append(di)
            last_x, last_y = xi, yi

plot_indices = pd.Index(clean_indices)
plot_distances = pd.Series(clean_distances)


g = 0.05  # gain for value (lower for smoother)
h = 0.005 # gain for derivative (lower for smoother)
x0 = plot_distances.iloc[0] if len(plot_distances) > 0 else 0
dx0 = 0
smoothed = gh_filter(plot_distances, x0, dx0, g, h)
x0 = plot_distances.iloc[0] if len(plot_distances) > 0 else 0
dx0 = 0
smoothed = gh_filter(plot_distances, x0, dx0, g, h)


# Save original plot
plt.figure(figsize=(8, 4))
plt.plot(plot_indices, plot_distances, marker='o', linestyle='-', color='blue', alpha=0.7)
plt.xlabel('Frame')
plt.ylabel('Distance from Reference Keypoint')
plt.title('Original Distance of Jaw Keypoint from Reference')
plt.tight_layout()
svg_output_path_orig = os.path.join(os.path.dirname(__file__), 'jaw_keypoint_distances_original.svg')
plt.savefig(svg_output_path_orig, format='svg')
print(f"Saved original distance plot to {svg_output_path_orig}")

# Save smoothed plot
plt.figure(figsize=(8, 4))
plt.plot(plot_indices, smoothed, color='red', linewidth=2)
plt.xlabel('Frame')
plt.ylabel('Distance from Reference Keypoint')
plt.title('Smoothed (g-h) Distance of Jaw Keypoint from Reference')
plt.tight_layout()
svg_output_path_smooth = os.path.join(os.path.dirname(__file__), 'jaw_keypoint_distances_smoothed.svg')
plt.savefig(svg_output_path_smooth, format='svg')
print(f"Saved smoothed distance plot to {svg_output_path_smooth}")
