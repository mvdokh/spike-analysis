import math
from whiskertoolbox_python import AnalogTimeSeries


def calculate_jaw_angle(dm, jaw_line_key, jaw_tip_key, output_key):
    """
    Calculate the angle between two tracked points (e.g. a jaw hinge/line
    point and a jaw tip point) and store the result as an AnalogTimeSeries.

    Angle convention (image coordinates, y increases downward):
      - 180 degrees when both points are on the same y-plane (same height)
      - > 180 degrees when the jaw tip point is higher than the jaw line point
      - < 180 degrees when the jaw tip point is lower than the jaw line point

    Note: this assumes the jaw tip is generally on the same side of the jaw
    line point along the x-axis (i.e. dx doesn't change sign). If the tip
    ever crosses to the other side of the line point in x, the angle will
    wrap around past 0/360 rather than continuing past 180 in a single
    direction.
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

        dx = tip_pt.x - line_pt.x
        dy = tip_pt.y - line_pt.y

        # atan2(dy, dx) is 0 when dy == 0 (same y-plane), positive when the
        # tip is lower (larger y, image coords), negative when the tip is
        # higher. Subtracting from 180 gives: 180 when same plane, > 180
        # when the tip is higher, < 180 when the tip is lower.
        angle = 180.0 - math.degrees(math.atan2(dy, dx))

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