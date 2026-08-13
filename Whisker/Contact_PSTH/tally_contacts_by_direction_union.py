"""
Tally contact intervals as Protraction / Retraction by union (overlap) analysis.

For each contact [Start, End] (inclusive), compute frame overlap with the union of
Protraction direction intervals and the union of Retraction direction intervals.
Classify by which side has more overlapping frames (ties / both > 0 reported as Mixed;
no overlap reported as Unclassified).
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

CONTACT_CSV = Path(
    r"H:\.shortcut-targets-by-id\1kmoeHgEh2zEhzXzXT2Xs5j6gJvKJWo7A"
    r"\Chodl-TG-PT\B065\0929_1\contact\0\contact_gamma.csv"

    
)
DIRECTION_CSV = CONTACT_CSV.parent / "direction.csv"


def _blocks_from_frame_labels(path: Path) -> list[tuple[int, int]]:
    """Convert frame-by-frame Contact/Nocontact labels into inclusive [start, end] blocks.

    Line index i is frame i (0-based), matching contact_BLOCKS.csv.
    """
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    n = 0
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            n = i + 1
            is_contact = line.strip().lower() == "contact"
            if is_contact:
                if start is None:
                    start = i
            elif start is not None:
                blocks.append((start, i - 1))
                start = None
    if start is not None and n > 0:
        blocks.append((start, n - 1))
    return blocks


def load_contacts(path: Path) -> list[tuple[int, int]]:
    """Load Start/End interval CSV, or frame-by-frame Contact/Nocontact CSV."""
    with path.open(encoding="utf-8") as f:
        first = f.readline().strip()

    if first.lower() in {"contact", "nocontact"} or (
        "start" not in first.lower() and "," not in first
    ):
        return _blocks_from_frame_labels(path)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [(int(row["Start"]), int(row["End"])) for row in reader]


def load_directions(path: Path) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    protraction: list[tuple[int, int]] = []
    retraction: list[tuple[int, int]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = int(row["Contact Start"])
            end = int(row["Contact End"])
            direction = row["Direction"].strip().lower()
            if direction.startswith("pro"):
                protraction.append((start, end))
            elif direction.startswith("ret"):
                retraction.append((start, end))
            else:
                raise ValueError(f"Unknown direction label: {row['Direction']!r}")
    return protraction, retraction


def inclusive_overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    """Number of inclusive integer frames shared by [a0, a1] and [b0, b1]."""
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0, hi - lo + 1)


def overlap_with_union(start: int, end: int, intervals: list[tuple[int, int]]) -> int:
    """Total inclusive overlap of [start, end] with the union of intervals.

    Assumes direction intervals are non-overlapping (as in direction.csv).
    If they ever overlap, overlapping frames would be double-counted — merge first.
    """
    return sum(inclusive_overlap(start, end, s, e) for s, e in intervals)


def classify_contact(
    start: int,
    end: int,
    protraction: list[tuple[int, int]],
    retraction: list[tuple[int, int]],
) -> tuple[str, int, int]:
    n_pro = overlap_with_union(start, end, protraction)
    n_ret = overlap_with_union(start, end, retraction)

    if n_pro == 0 and n_ret == 0:
        label = "Unclassified"
    elif n_pro > n_ret:
        label = "Protraction"
    elif n_ret > n_pro:
        label = "Retraction"
    else:
        label = "Mixed"  # both > 0 and equal overlap

    return label, n_pro, n_ret


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    contacts = load_contacts(CONTACT_CSV)
    protraction, retraction = load_directions(DIRECTION_CSV)

    detail_rows: list[dict] = []
    counts: Counter[str] = Counter()

    for start, end in contacts:
        label, n_pro, n_ret = classify_contact(start, end, protraction, retraction)
        counts[label] += 1
        detail_rows.append(
            {
                "Start": start,
                "End": end,
                "Length": end - start + 1,
                "ProtractionOverlap": n_pro,
                "RetractionOverlap": n_ret,
                "Direction": label,
            }
        )

    out_dir = CONTACT_CSV.parent
    stem = CONTACT_CSV.stem
    detail_path = out_dir / f"{stem}_direction_union.csv"
    summary_path = out_dir / f"{stem}_direction_union_summary.csv"

    write_csv(
        detail_path,
        detail_rows,
        [
            "Start",
            "End",
            "Length",
            "ProtractionOverlap",
            "RetractionOverlap",
            "Direction",
        ],
    )

    summary_rows = [
        {"Direction": "Protraction", "Count": counts.get("Protraction", 0)},
        {"Direction": "Retraction", "Count": counts.get("Retraction", 0)},
        {"Direction": "Mixed", "Count": counts.get("Mixed", 0)},
        {"Direction": "Unclassified", "Count": counts.get("Unclassified", 0)},
        {"Direction": "Total", "Count": len(contacts)},
    ]
    write_csv(summary_path, summary_rows, ["Direction", "Count"])

    print(f"Contacts: {CONTACT_CSV}")
    print(f"Direction: {DIRECTION_CSV}")
    print(f"n contacts = {len(contacts)}")
    print(
        f"  Protraction   = {counts.get('Protraction', 0)}\n"
        f"  Retraction    = {counts.get('Retraction', 0)}\n"
        f"  Mixed         = {counts.get('Mixed', 0)}\n"
        f"  Unclassified  = {counts.get('Unclassified', 0)}"
    )
    print(f"\nWrote:\n  {detail_path}\n  {summary_path}")


if __name__ == "__main__":
    main()
