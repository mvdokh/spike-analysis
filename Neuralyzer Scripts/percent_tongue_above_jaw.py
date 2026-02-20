from whiskertoolbox_python import AnalogTimeSeries

# Keys
JAW_KEY = "Jaw"
TONGUE_MASK_KEY = "Tongue"
OUTPUT_KEY = "Tongue_percent_above_jaw"

jaw_data = dm.getData(JAW_KEY)
tongue_mask_data = dm.getData(TONGUE_MASK_KEY)

if not jaw_data:
    print(f"Error: '{JAW_KEY}' not found.")
elif not tongue_mask_data:
    print(f"Error: '{TONGUE_MASK_KEY}' not found.")
else:
    print("Processing...")

    percentages = []
    times = []

    mask_times = tongue_mask_data.getTimesWithData()
    jaw_times = jaw_data.getTimesWithData()

    all_sparse_times = sorted(set(mask_times + jaw_times))

    if not all_sparse_times:
        print("No time data found.")
    else:
        start_t = min(all_sparse_times)
        end_t = max(all_sparse_times)

        for t in range(start_t, end_t + 1):
            percent_value = 0.0  # default = 0

            masks = tongue_mask_data.getAtTime(t)
            jaw_points = jaw_data.getAtTime(t)

            if masks and jaw_points:
                jaw_y = jaw_points[0].y

                total_pixels = 0
                pixels_above = 0

                for mask in masks:
                    for pt in mask:
                        total_pixels += 1
                        if pt.y < jaw_y:   # ABOVE in image coordinates
                            pixels_above += 1

                if total_pixels > 0:
                    percent_value = (pixels_above / total_pixels) * 100.0

            percent_value = max(0.0, min(100.0, percent_value))

            percentages.append(percent_value)
            times.append(t)

        time_key = dm.getTimeKey(JAW_KEY)
        new_series = AnalogTimeSeries(percentages, times)
        dm.setData(OUTPUT_KEY, new_series, time_key)

        print(f"Successfully created '{OUTPUT_KEY}'")
        print(f"Frames processed: {len(percentages)}")