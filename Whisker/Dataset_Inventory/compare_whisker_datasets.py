"""
Inventory whisker datasets and write summary CSVs.

Produces:
  - whisker_dataset_2_inventory.csv
  - whiskers_inventory.csv
  - dataset_differences.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

WHISKER_DATASET_2 = Path(r"\\wsl.localhost\Ubuntu\home\wanglab\Whisker_Dataset_2")
WHISKERS = Path(r"H:\.shortcut-targets-by-id\1cbnW0YmMoBX0JElXZKoE_HzQXuaTF7nz\Whiskers")

OUTPUT_DIR = Path(__file__).resolve().parent
LABEL_IDS = list(range(9))  # 0..8

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

COLUMNS = (
    ["Origin", "Folder", "Images"]
    + [f"{i} labels" for i in LABEL_IDS]
)


def count_images(images_dir: Path) -> int:
    if not images_dir.is_dir():
        return 0
    return sum(
        1
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_csv_labels(label_dir: Path) -> int:
    if not label_dir.is_dir():
        return 0
    return sum(1 for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".csv")


def is_session_folder(folder: Path) -> bool:
    """Session folders contain images/ and labels/ subdirectories."""
    return (folder / "images").is_dir() or (folder / "labels").is_dir()


def inventory_dataset(root: Path, origin: str) -> list[dict]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    rows: list[dict] = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        if not is_session_folder(folder):
            continue

        row: dict = {
            "Origin": origin,
            "Folder": folder.name,
            "Images": count_images(folder / "images"),
        }
        labels_root = folder / "labels"
        for i in LABEL_IDS:
            row[f"{i} labels"] = count_csv_labels(labels_root / str(i))
        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_differences(
    whiskers_rows: list[dict],
    dataset2_rows: list[dict],
) -> tuple[list[str], list[dict]]:
    """
    Compare folders by name.

    Status:
      - only_in_whiskers
      - only_in_whisker_dataset_2
      - identical
      - differs
    """
    w_map = {r["Folder"]: r for r in whiskers_rows}
    d_map = {r["Folder"]: r for r in dataset2_rows}
    all_folders = sorted(set(w_map) | set(d_map))

    count_cols = ["Images"] + [f"{i} labels" for i in LABEL_IDS]
    fieldnames = (
        ["Folder", "Status"]
        + [f"whiskers_{c}" for c in count_cols]
        + [f"whisker_dataset_2_{c}" for c in count_cols]
        + [f"diff_{c}" for c in count_cols]
    )

    rows: list[dict] = []
    for folder in all_folders:
        w = w_map.get(folder)
        d = d_map.get(folder)

        if w is None:
            status = "only_in_whisker_dataset_2"
        elif d is None:
            status = "only_in_whiskers"
        else:
            same = all(w[c] == d[c] for c in count_cols)
            status = "identical" if same else "differs"

        row: dict = {"Folder": folder, "Status": status}
        for c in count_cols:
            row[f"whiskers_{c}"] = w[c] if w else ""
            row[f"whisker_dataset_2_{c}"] = d[c] if d else ""
            row[f"diff_{c}"] = (d[c] - w[c]) if (w is not None and d is not None) else ""
        rows.append(row)

    return fieldnames, rows


def main() -> None:
    print(f"Scanning Whisker_Dataset_2: {WHISKER_DATASET_2}")
    dataset2_rows = inventory_dataset(WHISKER_DATASET_2, "whisker_dataset_2")
    print(f"  {len(dataset2_rows)} session folder(s)")

    print(f"Scanning Whiskers: {WHISKERS}")
    whiskers_rows = inventory_dataset(WHISKERS, "whiskers")
    print(f"  {len(whiskers_rows)} session folder(s)")

    out_d2 = OUTPUT_DIR / "whisker_dataset_2_inventory.csv"
    out_w = OUTPUT_DIR / "whiskers_inventory.csv"
    out_diff = OUTPUT_DIR / "dataset_differences.csv"

    write_csv(out_d2, dataset2_rows, list(COLUMNS))
    write_csv(out_w, whiskers_rows, list(COLUMNS))

    diff_fields, diff_rows = build_differences(whiskers_rows, dataset2_rows)
    write_csv(out_diff, diff_rows, diff_fields)

    n_only_w = sum(1 for r in diff_rows if r["Status"] == "only_in_whiskers")
    n_only_d = sum(1 for r in diff_rows if r["Status"] == "only_in_whisker_dataset_2")
    n_diff = sum(1 for r in diff_rows if r["Status"] == "differs")
    n_same = sum(1 for r in diff_rows if r["Status"] == "identical")

    print(f"\nWrote:\n  {out_d2}\n  {out_w}\n  {out_diff}")
    print(
        f"Differences: identical={n_same}, differs={n_diff}, "
        f"only_in_whiskers={n_only_w}, only_in_whisker_dataset_2={n_only_d}"
    )


if __name__ == "__main__":
    main()
