import math

from whiskertoolbox_python import AnalogTimeSeries


def read_analog_series(analog_data):
    """Return (times, values) from an AnalogTimeSeries."""
    raw = analog_data.toList()
    if not raw:
        return [], []

    if isinstance(raw[0], (tuple, list)) and len(raw[0]) >= 2:
        times = [row[0] for row in raw]
        values = [row[1] for row in raw]
    else:
        values = raw
        times = list(range(len(values)))

    return times, values


def moving_average(values, window):
    """Centered moving average; NaNs are skipped within each window."""
    if window < 1:
        raise ValueError("window must be >= 1")

    n = len(values)
    radius = (window - 1) // 2
    smoothed = []

    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        finite = [
            v
            for v in values[lo:hi]
            if isinstance(v, (int, float)) and math.isfinite(v)
        ]
        if finite:
            smoothed.append(sum(finite) / len(finite))
        else:
            smoothed.append(float("nan"))

    return smoothed


def smooth_analog_trace(dm, input_key, output_key, window=5):
    """
    Smooth an AnalogTimeSeries with a centered moving average and store the
    result as a new series on the same time base.

    Example: smooth the jaw angle trace from two_point_angle.py.
    """
    analog_data = dm.getData(input_key)
    if not analog_data:
        print(f"Error: '{input_key}' not found.")
        return False

    times, values = read_analog_series(analog_data)
    if not values:
        print(f"Error: '{input_key}' has no samples.")
        return False

    smoothed = moving_average(values, window)
    new_series = AnalogTimeSeries(smoothed, times)

    time_key = dm.getTimeKey(input_key)
    dm.setData(output_key, new_series, time_key)

    print(
        f"Created '{output_key}' with {len(smoothed)} samples "
        f"from '{input_key}' (window={window})"
    )
    return True


smooth_analog_trace(dm, "jaw_angle", "jaw_angle_smooth", window=11)
