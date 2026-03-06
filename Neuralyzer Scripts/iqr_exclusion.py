from whiskertoolbox_python import AnalogTimeSeries
import statistics

# Retrieve the analog series
angle_data = dm.getData("line_1_line_angle")

if angle_data:

    # Get values and real time indices
    values = angle_data.toList()
    
    # Retrieve the original time key
    time_key = dm.getTimeKey("line_1_line_angle")
    
    # Use dm to get the time indices for each sample
    # Neuralyzer usually stores analog data as consecutive indices,
    # so we'll assume sample index + starting time
    start_time = 3000  # first frame with data
    times = list(range(start_time, start_time + len(values)))

    # Compute Q1 and Q3 manually
    sorted_vals = sorted(values)

    def percentile(data, p):
        k = (len(data)-1) * (p/100)
        f = int(k)
        c = f + 1
        if c >= len(data):
            return data[f]
        d0 = data[f] * (c-k)
        d1 = data[c] * (k-f)
        return d0 + d1

    q1 = percentile(sorted_vals, 25)
    q3 = percentile(sorted_vals, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # Filter values
    clean_values = []
    clean_times = []

    for v, t in zip(values, times):
        if lower <= v <= upper:
            clean_values.append(v)
            clean_times.append(t)

    print(f"Original samples: {len(values)}")
    print(f"Filtered samples: {len(clean_values)}")

    # Create new AnalogTimeSeries
    new_series = AnalogTimeSeries(clean_values, clean_times)

    # Register with same time base
    dm.setData("line_1_line_angle_IQR", new_series, time_key)

    print("Created cleaned series: line_1_line_angle_IQR")

else:
    print("Error: 'line_1_line_angle' not found.")