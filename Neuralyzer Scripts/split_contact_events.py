from whiskertoolbox_python import DigitalIntervalSeries
import math

# Keys
CONTACT_KEY = "Contact_Events"
POLE_KEY = "keypoint_1"
WHISKER_KEY = "whisker_0"

LEFT_OUTPUT_KEY = "Contact_Left"
RIGHT_OUTPUT_KEY = "Contact_Right"

contact_data = dm.getData(CONTACT_KEY)
pole_data = dm.getData(POLE_KEY)
whisker_data = dm.getData(WHISKER_KEY)

if not contact_data:
    print(f"Error: '{CONTACT_KEY}' not found.")
elif not pole_data:
    print(f"Error: '{POLE_KEY}' not found.")
elif not whisker_data:
    print(f"Error: '{WHISKER_KEY}' not found.")
else:
    print("Processing contact intervals...")

    left_intervals = DigitalIntervalSeries()
    right_intervals = DigitalIntervalSeries()

    intervals = contact_data.toList()  # list of Interval objects

    for interval in intervals:

        start_t = interval.start
        end_t = interval.end

        pole_points = pole_data.getAtTime(start_t)
        whisker_lines = whisker_data.getAtTime(start_t)

        # Skip if missing geometry (should not happen, but safe)
        if not pole_points or not whisker_lines:
            continue

        pole_x = pole_points[0].x
        pole_y = pole_points[0].y

        whisker_line = whisker_lines[0]

        # Find closest whisker point to pole
        min_dist = float("inf")
        closest_x = None

        for pt in whisker_line:
            dx = pt.x - pole_x
            dy = pt.y - pole_y
            dist = dx * dx + dy * dy  # squared distance (faster)

            if dist < min_dist:
                min_dist = dist
                closest_x = pt.x

        # Classify left or right
        if closest_x is not None:
            if closest_x < pole_x:
                left_intervals.addInterval(start_t, end_t)
            else:
                right_intervals.addInterval(start_t, end_t)

    # Register results using same time base as Contact_Events
    time_key = dm.getTimeKey(CONTACT_KEY)

    dm.setData(LEFT_OUTPUT_KEY, left_intervals, time_key)
    dm.setData(RIGHT_OUTPUT_KEY, right_intervals, time_key)

    print("Done.")
    print(f"Original intervals: {len(intervals)}")
    print(f"Left intervals: {left_intervals.size()}")
    print(f"Right intervals: {right_intervals.size()}")
    print(f"Total split: {left_intervals.size() + right_intervals.size()}")