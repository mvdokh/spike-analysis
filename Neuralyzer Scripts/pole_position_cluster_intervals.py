from whiskertoolbox_python import DigitalIntervalSeries

# Run after pole_position_intervals_separate.py. Groups separate static-interval
# layers when their mean pole positions are within CLUSTER_DISTANCE_PX (same
# physical location visited at different times).

SOURCE_KEY_PREFIX = "pole_static_interval_"
OUTPUT_KEY_PREFIX = "pole_position_cluster_"
CLUSTER_DISTANCE_PX = 35.0  # increase if distinct positions are merged; decrease to split


def _mean_pole_xy(pole_data, start, end):
    xs, ys = [], []
    for t in pole_data.getTimesWithData():
        if start <= t <= end:
            pts = pole_data.getAtTime(t)
            if pts:
                xs.append(pts[0].x)
                ys.append(pts[0].y)
    if not xs:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _cluster_indices(centroids, threshold_px):
    n = len(centroids)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    for i in range(n):
        xi, yi = centroids[i]
        for j in range(i + 1, n):
            xj, yj = centroids[j]
            dist = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
            if dist <= threshold_px:
                union(i, j)

    roots = {}
    for i in range(n):
        r = find(i)
        roots.setdefault(r, []).append(i)
    return list(roots.values())


pole_data = dm.getData("pole")
if not pole_data:
    print("Error: 'pole' data not found.")
else:
    keys = sorted(k for k in dm.getAllKeys() if k.startswith(SOURCE_KEY_PREFIX))
    if not keys:
        print(
            f"Error: no keys starting with '{SOURCE_KEY_PREFIX}'. "
            "Run pole_position_intervals_separate.py first."
        )
    else:
        records = []
        for key in keys:
            series = dm.getData(key)
            if not series:
                continue
            for interval in series.toList():
                start, end = interval.start, interval.end
                xy = _mean_pole_xy(pole_data, start, end)
                if xy is None:
                    print(
                        f"  Warning: no pole samples in [{start}, {end}] for '{key}' — skipped."
                    )
                    continue
                records.append(
                    {
                        "source_key": key,
                        "start": start,
                        "end": end,
                        "cx": xy[0],
                        "cy": xy[1],
                    }
                )

        if not records:
            print("No usable intervals to cluster.")
        else:
            records.sort(key=lambda r: (r["start"], r["end"]))
            centroids = [(r["cx"], r["cy"]) for r in records]
            groups = _cluster_indices(centroids, CLUSTER_DISTANCE_PX)
            groups.sort(key=lambda idxs: min(records[i]["start"] for i in idxs))

            time_key = dm.getTimeKey("pole")

            for ci, idxs in enumerate(groups):
                idxs.sort(key=lambda i: records[i]["start"])
                out = DigitalIntervalSeries()
                for i in idxs:
                    r = records[i]
                    out.addInterval(r["start"], r["end"])

                mx = sum(records[i]["cx"] for i in idxs) / len(idxs)
                my = sum(records[i]["cy"] for i in idxs) / len(idxs)
                out_key = f"{OUTPUT_KEY_PREFIX}{ci + 1}"
                dm.setData(out_key, out, time_key)

                members = ", ".join(records[i]["source_key"] for i in idxs)
                print(
                    f"  '{out_key}' : {out.size()} interval(s), "
                    f"mean pole ({mx:.2f}, {my:.2f}) px — {members}"
                )

            print(
                f"Done. {len(records)} separate interval(s) → {len(groups)} cluster(s) "
                f"(threshold {CLUSTER_DISTANCE_PX} px)."
            )
