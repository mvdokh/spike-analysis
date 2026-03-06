from whiskertoolbox_python import DigitalIntervalSeries

PHASE_KEY = "whisker_phase"
WRAP_THRESHOLD = -3.0
MIN_CYCLE_LENGTH = 3

phase_data = dm.getData(PHASE_KEY)

if not phase_data:
    print("Error: phase data not found")
else:
    values = phase_data.toList()
    n = phase_data.getNumSamples()

    wraps = []

    # detect wrap points
    for i in range(1, n):
        diff = values[i] - values[i-1]

        if diff < WRAP_THRESHOLD:
            wraps.append(i)

    intervals = DigitalIntervalSeries()

    # convert wrap indices into intervals
    for i in range(1, len(wraps)):
        start = wraps[i-1]
        end = wraps[i]

        if (end - start) >= MIN_CYCLE_LENGTH:
            intervals.addInterval(start, end)

    time_key = dm.getTimeKey(PHASE_KEY)
    dm.setData("sawtooth", intervals, time_key)

    print("Created", intervals.size(), "sawtooth intervals")