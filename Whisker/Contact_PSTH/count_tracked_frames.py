"""
Count frames covered by tracked_x intervals.

Each row is an inclusive [Start, End] interval (column 1 = start, column 2 = end).
Reports per-interval lengths and the total frame count.
"""

from __future__ import annotations

import csv
from pathlib import Path

TRACKED_CSV = Path(
    r"H:\.shortcut-targets-by-id\1kmoeHgEh2zEhzXzXT2Xs5j6gJvKJWo7A"
    r"\Chodl-TG-PT\B064\091323_1\contact\tracked_gamma.csv"
)


def load_intervals(path: Path) -> list[tuple[int, int]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Allow headered (Start,End) or bare two-column CSVs.
        rows = list(reader)
        if header is not None:
            try:
                int(header[0])
                int(header[1])
                rows = [header] + rows
            except (ValueError, IndexError):
                pass
        return [(int(r[0]), int(r[1])) for r in rows if len(r) >= 2 and r[0].strip()]


def main() -> None:
    intervals = load_intervals(TRACKED_CSV)
    lengths = [end - start + 1 for start, end in intervals]
    total = sum(lengths)

    out_dir = TRACKED_CSV.parent
    stem = TRACKED_CSV.stem
    detail_path = out_dir / f"{stem}_frame_counts.csv"
    summary_path = out_dir / f"{stem}_frame_counts_summary.csv"

    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Start", "End", "Frames"])
        writer.writeheader()
        for (start, end), n in zip(intervals, lengths):
            writer.writerow({"Start": start, "End": end, "Frames": n})

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Metric", "Value"])
        writer.writeheader()
        writer.writerows(
            [
                {"Metric": "n_intervals", "Value": len(intervals)},
                {"Metric": "total_frames", "Value": total},
            ]
        )

    print(f"Tracked: {TRACKED_CSV}")
    print(f"n intervals = {len(intervals)}")
    print(f"total frames = {total}")
    print(f"\nWrote:\n  {detail_path}\n  {summary_path}")


if __name__ == "__main__":
    main()
