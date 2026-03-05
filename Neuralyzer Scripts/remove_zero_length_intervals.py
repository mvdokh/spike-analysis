from whiskertoolbox_python import DigitalIntervalSeries

# 1. Load original interval data
interval_data = dm.getData("all")

if interval_data:

    # 2. Create a new interval series
    new_intervals = DigitalIntervalSeries()

    # 3. Keep only intervals where start != end
    for interval in interval_data.toList():
        if interval.start != interval.end:
            new_intervals.addInterval(interval.start, interval.end)

    # 4. Register cleaned feature using same time base
    time_key = dm.getTimeKey("all")
    dm.setData("all_cleaned", new_intervals, time_key)

    print("Successfully created 'all_cleaned'")

else:
    print("Error: 'all' not found.")
