from whiskertoolbox_python import DigitalIntervalSeries

# Load data
intervals = dm.getData("c1_angle_outliers_removed_filter_hilbert_phase_interval_detection")
phase = dm.getData("c1_angle_outliers_removed_filter_hilbert_phase")

if not intervals or not phase:
    print("Error: Could not load one or more data objects.")
else:
    phase_times = [t.getValue() for t in phase.getTimeSeries()]
    phase_list = phase.toList()

    all_intervals = intervals.toList()
    print(f"Starting interval count: {len(all_intervals)}")

    removed_phase = 0
    removed_short = 0
    kept = 0

    new_intervals = DigitalIntervalSeries()

    for interval in all_intervals:
        start = interval.start
        end = interval.end

        # 1. First check length - virtually free to compute
        if (end - start) < 50:
            removed_short += 1
            continue

        # 2. Only run expensive phase check on intervals that pass length filter
        bad_phase = False
        for i, t in enumerate(phase_times):
            if t < start:
                continue
            if t >= end:
                break
            if phase_list[i] > 8:
                bad_phase = True
                break

        if bad_phase:
            removed_phase += 1
            continue

        new_intervals.addInterval(start, end)
        kept += 1

    time_key = dm.getTimeKey("c1_angle_outliers_removed_filter_hilbert_phase_interval_detection")
    dm.setData("c1_angle_outliers_removed_filter_hilbert_phase_interval_detection_filtered", new_intervals, time_key)

    print(f"Removed {removed_short} intervals under 50 frames")
    print(f"Removed {removed_phase} intervals with Hilbert phase > 6")
    print(f"Kept {kept} intervals")
    print("Saved as 'c1_angle_outliers_removed_filter_hilbert_phase_interval_detection_filtered'")