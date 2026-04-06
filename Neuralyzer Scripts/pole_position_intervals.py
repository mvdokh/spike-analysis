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
        MIN_INTERVAL_LEN = 1000  # frames

        intervals = []
        interval_start = None
        anchor_x = None
        anchor_y = None
        prev_t = None

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

        result = DigitalIntervalSeries()
        for start, end in intervals:
            result.addInterval(start, end)

        original_time_key = dm.getTimeKey("pole")
        dm.setData("pole_static_intervals", result, original_time_key)

        print(f"Done. Created {result.size()} static interval(s) -> 'pole_static_intervals'")