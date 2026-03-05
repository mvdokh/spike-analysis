from whiskertoolbox_python import DigitalIntervalSeries

def intervals_overlap(a_start, a_end, b_start, b_end):
    return not (a_end < b_start or b_end < a_start)

collision = dm.getData("collision_all")
all_data = dm.getData("all")

if collision is None:
    print("Error: 'collision_all' not found.")
elif all_data is None:
    print("Error: 'all' not found.")
else:
    collision_ranges = [(iv.start, iv.end) for iv in collision.toList()]
    print(f"Loaded {len(collision_ranges)} collision intervals.")

    cleaned_series = DigitalIntervalSeries()
    kept = 0
    removed = 0

    for iv in all_data.toList():
        overlaps = False
        for c_start, c_end in collision_ranges:
            if intervals_overlap(iv.start, iv.end, c_start, c_end):
                overlaps = True
                break

        if not overlaps:
            cleaned_series.addInterval(iv.start, iv.end)
            kept += 1
        else:
            removed += 1

    time_key = dm.getTimeKey("all")
    dm.setData("all_no_collision", cleaned_series, time_key)

    print(f"Kept {kept}, removed {removed} → saved as 'all_no_collision'")
