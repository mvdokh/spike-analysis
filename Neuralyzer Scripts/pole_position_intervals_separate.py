from whiskertoolbox_python import DigitalIntervalSeries

pole_data = dm.getData("pole")

if not pole_data:
    print("Error: 'pole' data not found.")
else:
    times = pole_data.getTimesWithData()

    if not times:
        print("No keypoint data found in 'pole'.")
    else:
        JITTER_THRESHOLD = 5.0   # pixels — adjust as needed
        MIN_INTERVAL_LEN = 5000  # frames
        FRAME_RANGE_START = 0      # first frame to consider (inclusive)
        FRAME_RANGE_END   = 241750 # last frame to consider (inclusive)

        intervals = []
        interval_start = None
        anchor_x = None
        anchor_y = None
        prev_t = None

        times = [t for t in times if FRAME_RANGE_START <= t <= FRAME_RANGE_END]

        for t in times:
            points = pole_data.getAtTime(t)

            if not points:
                # Gap — close interval if long enough
                if interval_start is not None:
                    if (prev_t - interval_start) >= MIN_INTERVAL_LEN:
                        intervals.append((interval_start, prev_t))
                    interval_start = None
                    anchor_x = None
                    anchor_y = None
                    prev_t = None
                continue

            curr_x = points[0].x
            curr_y = points[0].y

            if anchor_x is None:
                # First valid keypoint
                interval_start = t
                anchor_x = curr_x
                anchor_y = curr_y
                prev_t = t
            else:
                dist = ((curr_x - anchor_x) ** 2 + (curr_y - anchor_y) ** 2) ** 0.5

                if dist <= JITTER_THRESHOLD:
                    # Within jitter — extend interval, keep original anchor
                    prev_t = t
                else:
                    # Pole moved — close interval if long enough, start new one
                    if (prev_t - interval_start) >= MIN_INTERVAL_LEN:
                        intervals.append((interval_start, prev_t))
                    interval_start = t
                    anchor_x = curr_x
                    anchor_y = curr_y
                    prev_t = t

        # Close any open interval at the end
        if interval_start is not None and (prev_t - interval_start) >= MIN_INTERVAL_LEN:
            intervals.append((interval_start, prev_t))

        ordinals = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
        original_time_key = dm.getTimeKey("pole")

        for i, (start, end) in enumerate(intervals):
            label = ordinals[i] if i < len(ordinals) else f"{i + 1}th"
            interval_series = DigitalIntervalSeries()
            interval_series.addInterval(start, end)
            key = f"pole_static_interval_{label}"
            dm.setData(key, interval_series, original_time_key)
            print(f"  Created '{key}' -> frames {start} to {end}")

        print(f"Done. Created {len(intervals)} static interval(s).")