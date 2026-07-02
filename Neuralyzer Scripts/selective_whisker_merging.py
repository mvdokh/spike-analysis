from whiskertoolbox_python import LineData

# Merge whisker_0 and whisker_0_new into a NEW feature: whisker_0_merged
# Rule: if a frame has lines in BOTH, keep only whisker_0_new's lines.
# Original whisker_0 and whisker_0_new are left untouched.

old_data = dm.getData("whisker_0")
new_data = dm.getData("whisker_0_new")

if old_data is None:
    print("Error: 'whisker_0' not found.")
elif new_data is None:
    print("Error: 'whisker_0_new' not found.")
else:
    # Sanity check: confirm LineData is default-constructible before doing any work
    try:
        merged = LineData()
    except TypeError as e:
        print("Could not construct an empty LineData with LineData(). "
              "Run `help(LineData)` or `dir(LineData)` in the console to find "
              "the correct constructor, then let me know what it expects.")
        raise

    old_times = set(old_data.getTimesWithData())
    new_times = set(new_data.getTimesWithData())
    all_times = old_times | new_times

    from_new = 0
    from_old = 0

    for t in sorted(all_times):
        if t in new_times:
            # whisker_0_new wins whenever it has data on this frame
            lines_at_t = new_data.getAtTime(t)
            from_new += 1
        else:
            # only whisker_0 has data on this frame
            lines_at_t = old_data.getAtTime(t)
            from_old += 1

        for line in lines_at_t:
            merged.addAtTime(t, line)

    print(f"Frames from whisker_0_new: {from_new}")
    print(f"Frames from whisker_0 only: {from_old}")
    print(f"Total merged frames: {len(all_times)}")

    time_key = dm.getTimeKey("whisker_0_new")
    dm.setData("whisker_0_merged", merged, time_key)

    print("Created new feature 'whisker_0_merged'.")