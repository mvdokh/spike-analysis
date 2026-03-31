from whiskertoolbox_python import AnalogTimeSeries
import math

# Retrieve the analog series
angle_data = dm.getData("c2_angle")

if angle_data:
    raw = angle_data.toList()
    if not raw:
        print("Error: 'c2_angle' has no samples.")
        raise SystemExit

    # Neuralyzer AnalogTimeSeries commonly returns values-only from toList().
    # If so, use implicit sample indices to keep full length and ordering.
    if isinstance(raw[0], (tuple, list)) and len(raw[0]) >= 2:
        times = [row[0] for row in raw]
        values = [row[1] for row in raw]
    else:
        values = raw
        times = list(range(len(values)))
    
    # Retrieve the original time key
    time_key = dm.getTimeKey("c2_angle")
    
    # Compute IQR from finite values only.
    finite_vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(finite_vals) < 4:
        print("Error: not enough finite samples in 'c2_angle' to compute IQR.")
        raise SystemExit
    sorted_vals = sorted(finite_vals)

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

    # Keep every frame and set only outliers to NaN (discontinuous series).
    clean_values = []
    outlier_count = 0
    for v in values:
        if isinstance(v, (int, float)) and math.isfinite(v) and (lower <= v <= upper):
            clean_values.append(v)
        elif isinstance(v, (int, float)) and not math.isfinite(v):
            clean_values.append(float("nan"))
        else:
            clean_values.append(float("nan"))
            outlier_count += 1

    print(f"Original samples: {len(values)}")
    print(f"Outliers replaced with NaN: {outlier_count}")
    print(f"Output samples (same as input): {len(clean_values)}")

    # Create new AnalogTimeSeries
    new_series = AnalogTimeSeries(clean_values, times)

    # Register with same time base
    dm.setData("c2_angle_IQR", new_series, time_key)

    print("Created cleaned series: c2_angle_IQR")

else:
    print("Error: 'c2_angle' not found.")