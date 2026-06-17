from whiskertoolbox_python import AnalogTimeSeries, Point2D

def extract_y_coordinate(dm, source_key, output_key):
    """Extract Y coordinates from a PointData object and store as AnalogTimeSeries."""
    point_data = dm.getData(source_key)
    
    if not point_data:
        print(f"Error: '{source_key}' data not found.")
        return False
    
    y_values = []
    times = []
    
    for t in point_data.getTimesWithData():
        points_at_t = point_data.getAtTime(t)
        if points_at_t:
            pt = points_at_t[0]
            y_values.append(pt.y)
            times.append(t)
    
    new_series = AnalogTimeSeries(y_values, times)
    original_time_key = dm.getTimeKey(source_key)
    dm.setData(output_key, new_series, original_time_key)
    
    print(f"Created '{output_key}' with {len(y_values)} samples from '{source_key}'")
    return True


extract_y_coordinate(dm, "jaw_line_side",       "jaw_line_side_y")
extract_y_coordinate(dm, "jaw_tip_side_clean",  "jaw_tip_side_clean_y")