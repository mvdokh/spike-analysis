from whiskertoolbox_python import DigitalIntervalSeries

# 1. Load original interval data
interval_data = dm.getData("interval_0")

if interval_data:
    
    # 2. Create a new interval series
    new_intervals = DigitalIntervalSeries()
    
    # 3. Iterate through all intervals
    for interval in interval_data.toList():
        
        start = interval.start
        end = interval.end
        
        # Modify interval
        new_start = start + 1
        new_end = end - 1
        
        # 4. Keep only valid intervals (removes length 1 and invalid ones)
        if new_start < new_end:
            new_intervals.addInterval(new_start, new_end)
    
    # 5. Register new feature using same time base
    time_key = dm.getTimeKey("interval_0")
    dm.setData("interval_0_shrunk", new_intervals, time_key)
    
    print("Successfully created 'interval_0_shrunk'")

else:
    print("Error: 'interval_0' not found.")