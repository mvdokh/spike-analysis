from whiskertoolbox_python import AnalogTimeSeries

# Retrieve your two PointData objects
jaw_data = dm.getData("jaw")
spout_data = dm.getData("spout")

if not jaw_data or not spout_data:
    print("Error: Could not find 'jaw' or 'spout' data.")
else:
    x_distances = []
    y_distances = []
    times = []

    # Find common time points
    jaw_times = set(jaw_data.getTimesWithData())
    spout_times = set(spout_data.getTimesWithData())
    common_times = sorted(jaw_times & spout_times)

    for t in common_times:
        jaw_pts = jaw_data.getAtTime(t)
        spout_pts = spout_data.getAtTime(t)

        if jaw_pts and spout_pts:
            jaw_x = jaw_pts[0].x
            jaw_y = jaw_pts[0].y

            spout_x = spout_pts[0].x
            spout_y = spout_pts[0].y

            # Absolute distances
            x_dist = abs(spout_x - jaw_x)
            y_dist = abs(spout_y - jaw_y)

            x_distances.append(x_dist)
            y_distances.append(y_dist)
            times.append(t)

    # Create AnalogTimeSeries objects
    x_series = AnalogTimeSeries(x_distances, times)
    y_series = AnalogTimeSeries(y_distances, times)

    # Register both
    time_key = dm.getTimeKey("jaw")
    dm.setData("jaw_spout_x_distance", x_series, time_key)
    dm.setData("jaw_spout_y_distance", y_series, time_key)

    print("Created 'jaw_spout_x_distance' and 'jaw_spout_y_distance'")

    # ---- Console output ----
    print("\nSample values (first 10):")
    for i in range(min(10, len(times))):
        print(f"t={times[i]} | x_dist={x_distances[i]:.3f}, y_dist={y_distances[i]:.3f}")

    print(f"\nTotal samples: {len(times)}")