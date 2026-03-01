from whiskertoolbox_python import DigitalIntervalSeries
import math

def sq_dist(x1, y1, x2, y2):
    return (x1 - x2)**2 + (y1 - y2)**2


mask_keys = ["0", "1", "2", "3", "4"]

for i, mask_key in enumerate(mask_keys):

    interval_key = f"interval_{i}_mask_contact"

    mask_data = dm.getData(mask_key)
    interval_data = dm.getData(interval_key)
    pole_data = dm.getData("pole")

    if not mask_data or not interval_data or not pole_data:
        print(f"Missing data for mask {i}")
        continue

    protraction_series = DigitalIntervalSeries()
    retraction_series = DigitalIntervalSeries()

    for interval in interval_data.toList():

        contact_time = interval.start

        pole_pts = pole_data.getAtTime(contact_time)
        mask_at_time = mask_data.getAtTime(contact_time)

        if not pole_pts or not mask_at_time:
            continue

        pole = pole_pts[0]
        mask_obj = mask_at_time[0]

        if mask_obj.size() == 0:
            continue

        # Find closest mask point to pole
        min_dist = float("inf")
        closest_point = None

        for pt in mask_obj:   # Mask2D is iterable
            d = sq_dist(pt.x, pt.y, pole.x, pole.y)
            if d < min_dist:
                min_dist = d
                closest_point = pt

        if closest_point is None:
            continue

        # ---- Image coordinate system (Y increases downward) ----
        if closest_point.y > pole.y:
            protraction_series.addInterval(interval.start, interval.end)
        elif closest_point.y < pole.y:
            retraction_series.addInterval(interval.start, interval.end)

    # Register results
    time_key = dm.getTimeKey(interval_key)

    dm.setData(f"interval_{i}_mask_contact_protraction",
               protraction_series,
               time_key)

    dm.setData(f"interval_{i}_mask_contact_retraction",
               retraction_series,
               time_key)

    print(f"Finished mask {i}")

print("All masks processed.")