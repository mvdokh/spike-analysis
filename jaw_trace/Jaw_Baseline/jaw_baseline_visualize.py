import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set paths
video_dir = r"C:\Users\wanglab\Desktop\IRt.PCRt-1125\IRt_114\IRt_01_2025_04251"
video_path = os.path.join(video_dir, "IRt_01_2025_04251.mp4")
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


# Try to find x and y columns (case-insensitive)
cols = {col.lower(): col for col in jaw_df.columns}
if 'x' in cols and 'y' in cols:
    x = jaw_df[cols['x']]
    y = jaw_df[cols['y']]
else:
    raise ValueError("CSV must contain 'x' and 'y' columns (case-insensitive). Found columns: {}".format(jaw_df.columns.tolist()))


# Load video and grab a sample frame (first frame)
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read frame from video.")

# Resize frame to 256x256
frame_resized = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_AREA)

# Calculate scaling factors for coordinates
orig_h, orig_w = frame.shape[:2]
scale_x = 256 / orig_w
scale_y = 256 / orig_h



# Debug: print frame size and keypoints
print(f"Original frame size: width={orig_w}, height={orig_h}")
print("First 5 original keypoints (x, y):", list(zip(x[:5], y[:5])))



# Calculate baseline (mean) position in original coordinates
baseline_x_orig = x.mean()
baseline_y_orig = y.mean()
print(f"Baseline (mean) in original coords: x={baseline_x_orig:.2f}, y={baseline_y_orig:.2f}")


# Do not rescale; use mean keypoint directly
baseline_x = baseline_x_orig
baseline_y = baseline_y_orig
print(f"Overlay baseline (no rescale): x={baseline_x:.2f}, y={baseline_y:.2f}")


# Load video and grab a sample frame (first frame)
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Failed to read frame from video.")

# Convert BGR (OpenCV) to RGB (matplotlib)
frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)


# Plot and save (overlay mean keypoint without scaling)
plt.figure(figsize=(6, 6))
plt.imshow(frame_rgb)
plt.scatter([baseline_y], [baseline_x], c='red', s=80, label='Baseline')
plt.legend()
plt.title('Baseline Jaw Position on Sample Frame')
plt.axis('off')

# Save the image
output_path = os.path.join(os.path.dirname(__file__), 'jaw_baseline_overlay.png')
plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
print(f"Saved overlay image to {output_path}")
