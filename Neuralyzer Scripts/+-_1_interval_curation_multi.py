from whiskertoolbox_python import DigitalIntervalSeries

# Process interval_0 through interval_4
for i in range(5):

    interval_key = f"interval_{i}"
    interval_data = dm.getData(interval_key)

    if not interval_data:
        print(f"Error: '{interval_key}' not found.")
        continue

    # Create new interval series
    new_intervals = DigitalIntervalSeries()

    # Iterate through all intervals
    for interval in interval_data.toList():

        start = interval.start
        end = interval.end

        # Modify interval
        new_start = start + 1
        new_end = end - 1

        # Keep only valid intervals
        if new_start < new_end:
            new_intervals.addInterval(new_start, new_end)

    # Register new feature
    time_key = dm.getTimeKey(interval_key)
    new_key = f"{interval_key}_shrunk"
    dm.setData(new_key, new_intervals, time_key)

    print(f"Successfully created '{new_key}'")

print("All intervals processed.")