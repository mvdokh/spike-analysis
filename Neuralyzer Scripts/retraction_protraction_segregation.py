from whiskertoolbox_python import DigitalIntervalSeries
import math

def sq_dist(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2


for i in range(5):

    interval_key = f"interval_{i}_shrunk"
    line_key = f"line_{i}"
    
    interval_data = dm.getData(interval_key)
    line_data = dm.getData(line_key)
    pole_data = dm.getData("keypoint_1")
    
    if not interval_data or not line_data or not pole_data:
        print(f"Missing data for whisker {i}")
        continue

    # Output interval series
    protraction_series = DigitalIntervalSeries()
    retraction_series = DigitalIntervalSeries()

    # Iterate through intervals
    for interval in interval_data.toList():
        
        contact_time = interval.start  # classify at contact onset
        
        pole_points = pole_data.getAtTime(contact_time)
        line_objects = line_data.getAtTime(contact_time)
        
        if not pole_points or not line_objects:
            continue
        
        pole = pole_points[0]
        line = line_objects[0]
        
        # Find closest point on whisker line to pole
        min_dist = float("inf")
        closest_point = None
        
        for pt in line:
            d = sq_dist(pt.x, pt.y, pole.x, pole.y)
            if d < min_dist:
                min_dist = d
                closest_point = pt
        
        if closest_point is None:
            continue
        
        # ---- Corrected classification (image Y increases downward) ----
        # larger Y = lower in image
        if closest_point.y > pole.y:
            # PROTRACTION
            protraction_series.addInterval(interval.start, interval.end)
        elif closest_point.y < pole.y:
            # RETRACTION
            retraction_series.addInterval(interval.start, interval.end)

    # Register results
    time_key = dm.getTimeKey(interval_key)

    dm.setData(f"interval_{i}_shrunk_protraction",
               protraction_series,
               time_key)

    dm.setData(f"interval_{i}_shrunk_retraction",
               retraction_series,
               time_key)

    print(f"Finished whisker {i}")

print("All whiskers processed.")