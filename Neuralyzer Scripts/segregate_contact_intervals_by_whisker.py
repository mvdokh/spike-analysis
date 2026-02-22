from whiskertoolbox_python import Interval, DigitalIntervalSeries, Point2D, Line2D

# Names of your whisker lines
whisker_keys = ["0_mask_to_line", "1_mask_to_line", "2_mask_to_line", "3_mask_to_line", "4_mask_to_line"]

# Load whisker LineData
whiskers = [dm.getData(key) for key in whisker_keys]

# Load pole position (PointData)
pole = dm.getData("keypoint_1")

# Load existing contact intervals
all_intervals = dm.getData("interval_1")

# Prepare new IntervalSeries for each whisker
whisker_intervals = [DigitalIntervalSeries() for _ in whiskers]

# Function to compute minimum distance from a line to a point
def min_distance_line_point(line_pts, point):
    return min(((pt.x - point.x)**2 + (pt.y - point.y)**2)**0.5 for pt in line_pts)

# Iterate over all intervals
for interval in all_intervals.toList():
    start_frame = interval.start
    end_frame = interval.end

    # Get pole position at start_frame
    pole_pts = pole.getAtTime(start_frame)
    if not pole_pts:
        continue
    pole_pt = pole_pts[0]  # Assuming only 1 pole point

    # Determine which whisker is closest at start_frame
    min_dist = float('inf')
    closest_idx = None
    for i, whisker in enumerate(whiskers):
        line_pts = whisker.getAtTime(start_frame)
        if not line_pts:
            continue
        # Assuming one Line2D per frame
        line_pts_list = line_pts[0].toList()  
        dist = min_distance_line_point(line_pts_list, pole_pt)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i

    # Add interval to the corresponding whisker series
    if closest_idx is not None:
        whisker_intervals[closest_idx].addInterval(interval.start, interval.end)

# Register new whisker-specific interval series in DataManager
for i, key in enumerate(whisker_keys):
    dm.setData(f"interval_{i}_whisker_contact", whisker_intervals[i], dm.getTimeKey("interval_1"))

print("Successfully segregated contact intervals by whisker.")