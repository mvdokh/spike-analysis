from whiskertoolbox_python import Interval, DigitalIntervalSeries, Point2D

# Names of your mask data
mask_keys = ["0", "1", "2", "3", "4"]

# Load MaskData
masks = [dm.getData(key) for key in mask_keys]

# Load pole position (PointData)
pole = dm.getData("keypoint_1")

# Load existing contact intervals
all_intervals = dm.getData("interval_1")

# Prepare new IntervalSeries for each mask
mask_intervals = [DigitalIntervalSeries() for _ in masks]

# Function to compute minimum distance from mask to point
def min_distance_mask_point(mask_obj, point):
    return min(((pt.x - point.x)**2 + (pt.y - point.y)**2)**0.5 
               for pt in mask_obj.points())

# Iterate over all intervals
for interval in all_intervals.toList():
    start_frame = interval.start
    end_frame = interval.end

    # Get pole position at start_frame
    pole_pts = pole.getAtTime(start_frame)
    if not pole_pts:
        continue
    pole_pt = pole_pts[0]  # assuming single pole point

    # Determine which mask is closest at start_frame
    min_dist = float('inf')
    closest_idx = None

    for i, mask_data in enumerate(masks):
        mask_at_time = mask_data.getAtTime(start_frame)
        if not mask_at_time:
            continue

        # Assuming one Mask2D per frame
        mask_obj = mask_at_time[0]

        if mask_obj.size() == 0:
            continue

        dist = min_distance_mask_point(mask_obj, pole_pt)

        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    # Assign interval to closest mask
    if closest_idx is not None:
        mask_intervals[closest_idx].addInterval(start_frame, end_frame)

# Register new mask-specific interval series
time_key = dm.getTimeKey("interval_1")

for i in range(len(mask_keys)):
    dm.setData(f"interval_{i}_mask_contact", mask_intervals[i], time_key)

print("Successfully segregated contact intervals by mask.")