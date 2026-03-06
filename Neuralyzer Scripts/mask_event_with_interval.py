# Retrieve your events and intervals
events = dm.getData("2_line_angle_IQR_event_detection")  # DigitalEventSeries
intervals = dm.getData("whisk_in_air")  # DigitalIntervalSeries

if events and intervals:

    # Get the list of all events
    all_events = events.toList()  # list of time indices

    # Get the list of all intervals
    valid_intervals = intervals.toList()  # list of Interval objects with start and end

    # Filter events to keep only those inside any interval
    filtered_events = []
    for e in all_events:
        for interval in valid_intervals:
            if interval.start <= e <= interval.end:
                filtered_events.append(e)
                break  # stop checking intervals once matched

    # Create new DigitalEventSeries
    from whiskertoolbox_python import DigitalEventSeries
    new_events = DigitalEventSeries()
    for e in filtered_events:
        new_events.addEvent(e)

    # Register the new event series
    original_time_key = dm.getTimeKey("2_line_angle_IQR_event_detection")
    dm.setData("2_line_angle_IQR_events_in_air", new_events, original_time_key)

    print(f"Original events: {len(all_events)}")
    print(f"Filtered events: {len(filtered_events)}")
    print("Created new event series: 2_line_angle_IQR_events_in_air")

else:
    if not events:
        print("Error: '2_line_angle_IQR_event_detection' not found.")
    if not intervals:
        print("Error: 'whisk_in_air' intervals not found.")