import math
from whiskertoolbox_python import AnalogTimeSeries


def tip_line_opening_angle_deg(line_x, line_y, tip_x, tip_y):
    """Tip-line opening angle with horizontal reference at 180 degrees.

    Measures elevation of the jaw tip relative to the jaw line in side view.
    Co-located horizontal positions (matching y) yield 180 degrees. When the
    tip is higher on the image (smaller y, since image y increases
    downward), the angle exceeds 180 degrees; when the jaw line is higher
    (tip lower), the angle is below 180 degrees.

    Taking abs() of the horizontal separation means the result doesn't
    depend on which side of the jaw line point the tip happens to be on.

    Returns float('nan') if the line and tip points are identical (the
    angle is undefined in that case).
    """
    vertical_sep = line_y - tip_y
    horizontal_sep = abs(tip_x - line_x)

    if vertical_sep == 0.0 and horizontal_sep == 0.0:
        return float("nan")

    elevation = math.degrees(math.atan2(vertical_sep, horizontal_sep))
    return 180.0 + elevation


def calculate_jaw_angle(dm, jaw_line_key, jaw_tip_key, output_key):
    """
    Calculate the tip-line opening angle between two tracked points (e.g. a
    jaw hinge/line point and a jaw tip point) and store the result as an
    AnalogTimeSeries.

    See `tip_line_opening_angle_deg` for the angle convention.
    """
    jaw_line_data = dm.getData(jaw_line_key)
    jaw_tip_data = dm.getData(jaw_tip_key)

    if not jaw_line_data:
        print(f"Error: '{jaw_line_key}' data not found.")
        return False
    if not jaw_tip_data:
        print(f"Error: '{jaw_tip_key}' data not found.")
        return False

    # Only compute the angle at times where both points have data
    line_times = set(jaw_line_data.getTimesWithData())
    tip_times = set(jaw_tip_data.getTimesWithData())
    common_times = sorted(line_times & tip_times)

    if not common_times:
        print(f"Error: no overlapping time points between '{jaw_line_key}' and '{jaw_tip_key}'.")
        return False

    angles = []
    times = []

    for t in common_times:
        line_points = jaw_line_data.getAtTime(t)
        tip_points = jaw_tip_data.getAtTime(t)

        if not line_points or not tip_points:
            continue

        line_pt = line_points[0]
        tip_pt = tip_points[0]

        angle = tip_line_opening_angle_deg(line_pt.x, line_pt.y, tip_pt.x, tip_pt.y)

        angles.append(angle)
        times.append(t)

    new_series = AnalogTimeSeries(angles, times)

    # Use the jaw line point's time base for the new series
    time_key = dm.getTimeKey(jaw_line_key)
    dm.setData(output_key, new_series, time_key)

    print(f"Created '{output_key}' with {len(angles)} samples "
          f"from '{jaw_line_key}' and '{jaw_tip_key}'")
    return True


calculate_jaw_angle(dm, "jaw_line_side", "jaw_tip_side_clean", "jaw_angle")