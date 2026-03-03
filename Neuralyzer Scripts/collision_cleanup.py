from whiskertoolbox_python import DigitalIntervalSeries

def intervals_overlap(a_start, a_end, b_start, b_end):
    """
    Returns True if intervals [a_start, a_end] and [b_start, b_end] overlap.
    """
    return not (a_end < b_start or b_end < a_start)


# ----------------------------
# Load collision_shrunk
# ----------------------------
collision = dm.getData("collision_shrunk")

if collision is None:
    print("Error: 'collision_shrunk' not found.")
else:
    collision_intervals = collision.toList()

    # Convert to simple (start, end) tuples
    collision_ranges = [(iv.start, iv.end) for iv in collision_intervals]

    print(f"Loaded {len(collision_ranges)} collision intervals.")

    # ----------------------------
    # Build list of features to clean
    # ----------------------------
    whiskers = ["0", "1", "2", "3", "4"]
    feature_names = []

    for w in whiskers:
        feature_names.append(w)
        feature_names.append(f"{w}_protraction")
        feature_names.append(f"{w}_retraction")

    # ----------------------------
    # Process each feature
    # ----------------------------
    for feature in feature_names:

        data = dm.getData(feature)

        if data is None:
            print(f"Skipping '{feature}' (not found).")
            continue

        original_intervals = data.toList()
        cleaned_series = DigitalIntervalSeries()

        kept_count = 0
        removed_count = 0

        for iv in original_intervals:
            start, end = iv.start, iv.end

            # Check for overlap with ANY collision interval
            overlaps = False
            for c_start, c_end in collision_ranges:
                if intervals_overlap(start, end, c_start, c_end):
                    overlaps = True
                    break

            if not overlaps:
                cleaned_series.addInterval(start, end)
                kept_count += 1
            else:
                removed_count += 1

        # Register cleaned feature
        cleaned_name = f"{feature}_cleaned"
        time_key = dm.getTimeKey(feature)
        dm.setData(cleaned_name, cleaned_series, time_key)

        print(f"{feature}: kept {kept_count}, removed {removed_count} → saved as '{cleaned_name}'")

    print("Done.")