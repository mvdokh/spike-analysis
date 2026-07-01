from whiskertoolbox_python import DigitalIntervalSeries

# --- Configuration ---
SOURCE_KEY = "whisker_0"
OUTPUT_KEY = "whisker_0_gaps"
MAX_GAP_SIZE = 10000  # gaps (in missing frames) larger than this are excluded

# --- Retrieve source LineData ---
line_data = dm.getData(SOURCE_KEY)

if line_data is None:
    print(f"Error: '{SOURCE_KEY}' not found in Data Manager.")
else:
    # Convert TimeFrameIndex objects to plain ints, then sort
    times = sorted(int(t) for t in line_data.getTimesWithData())

    if len(times) < 2:
        print(f"'{SOURCE_KEY}' has fewer than 2 timestamps; no gaps to compute.")
    else:
        gap_series = DigitalIntervalSeries()

        total_gap_frames = 0
        num_gaps_counted = 0
        num_gaps_excluded = 0

        for prev_t, next_t in zip(times[:-1], times[1:]):
            missing = next_t - prev_t - 1  # number of frames strictly between prev_t and next_t

            if missing <= 0:
                continue  # no gap, consecutive frames

            if missing > MAX_GAP_SIZE:
                num_gaps_excluded += 1
                continue  # skip huge gaps

            gap_start = prev_t + 1
            gap_end = next_t - 1
            gap_series.addInterval(gap_start, gap_end)

            total_gap_frames += missing
            num_gaps_counted += 1

        time_key = dm.getTimeKey(SOURCE_KEY)
        dm.setData(OUTPUT_KEY, gap_series, time_key)

        print(f"Source: '{SOURCE_KEY}' ({len(times)} frames with data)")
        print(f"Gaps found (excluding gaps > {MAX_GAP_SIZE} frames): {num_gaps_counted}")
        print(f"Total gap frames (excluding big gaps): {total_gap_frames}")
        print(f"Large gaps excluded (> {MAX_GAP_SIZE} frames): {num_gaps_excluded}")
        print(f"Created interval data '{OUTPUT_KEY}' with {gap_series.size()} intervals.")