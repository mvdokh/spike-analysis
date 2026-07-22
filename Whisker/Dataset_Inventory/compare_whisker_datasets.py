"""
Inventory whisker datasets and deeply compare shared label CSVs.

Produces:
  - whisker_dataset_2_inventory.csv
  - whiskers_inventory.csv
  - dataset_differences.csv          (folder-level counts + content status)
  - csv_file_differences.csv         (per-file mismatches / missing pairs)
"""

from __future__ import annotations

import csv
import filecmp
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

# Chunk size for streaming byte compares when sizes match but we want a hard check.
# filecmp.cmp(..., shallow=False) already does content compare; used below.


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


def list_label_csvs(session_dir: Path) -> dict[str, Path]:
    """
    Map relative path (posix) -> absolute path for every labels/**/*.csv.
    Example key: 'labels/0/0000000.csv'
    """
    labels_root = session_dir / "labels"
    if not labels_root.is_dir():
        return {}

    out: dict[str, Path] = {}
    for p in labels_root.rglob("*.csv"):
        if p.is_file():
            rel = p.relative_to(session_dir).as_posix()
            out[rel] = p
    return out


def compare_csv_pair(path_w: Path, path_d: Path) -> str:
    """
    Compare one CSV pair.

    Returns one of:
      identical_size_and_content
      size_mismatch
      content_mismatch
    """
    size_w = path_w.stat().st_size
    size_d = path_d.stat().st_size
    if size_w != size_d:
        return "size_mismatch"

    # Sizes match — verify bytes (not just mtime/size shallow compare).
    if filecmp.cmp(path_w, path_d, shallow=False):
        return "identical_size_and_content"
    return "content_mismatch"


def deep_compare_session(folder: str, whiskers_root: Path, dataset2_root: Path) -> dict:
    """
    Compare all label CSVs that share the same relative path under both sessions.
    """
    w_dir = whiskers_root / folder
    d_dir = dataset2_root / folder

    w_csvs = list_label_csvs(w_dir)
    d_csvs = list_label_csvs(d_dir)

    w_keys = set(w_csvs)
    d_keys = set(d_csvs)
    shared = sorted(w_keys & d_keys)
    only_w = sorted(w_keys - d_keys)
    only_d = sorted(d_keys - w_keys)

    n_identical = 0
    n_size_mismatch = 0
    n_content_mismatch = 0
    file_rows: list[dict] = []

    for rel in shared:
        status = compare_csv_pair(w_csvs[rel], d_csvs[rel])
        if status == "identical_size_and_content":
            n_identical += 1
            continue

        if status == "size_mismatch":
            n_size_mismatch += 1
        else:
            n_content_mismatch += 1

        file_rows.append(
            {
                "Folder": folder,
                "RelativePath": rel,
                "Status": status,
                "whiskers_bytes": w_csvs[rel].stat().st_size,
                "whisker_dataset_2_bytes": d_csvs[rel].stat().st_size,
            }
        )

    for rel in only_w:
        file_rows.append(
            {
                "Folder": folder,
                "RelativePath": rel,
                "Status": "only_in_whiskers",
                "whiskers_bytes": w_csvs[rel].stat().st_size,
                "whisker_dataset_2_bytes": "",
            }
        )

    for rel in only_d:
        file_rows.append(
            {
                "Folder": folder,
                "RelativePath": rel,
                "Status": "only_in_whisker_dataset_2",
                "whiskers_bytes": "",
                "whisker_dataset_2_bytes": d_csvs[rel].stat().st_size,
            }
        )

    if only_w or only_d or n_size_mismatch or n_content_mismatch:
        content_status = "csv_differs"
    elif shared:
        content_status = "csv_identical"
    else:
        content_status = "no_csvs"

    summary = {
        "Folder": folder,
        "ContentStatus": content_status,
        "shared_csvs": len(shared),
        "identical_csvs": n_identical,
        "size_mismatch_csvs": n_size_mismatch,
        "content_mismatch_csvs": n_content_mismatch,
        "only_in_whiskers_csvs": len(only_w),
        "only_in_whisker_dataset_2_csvs": len(only_d),
    }
    return {"summary": summary, "file_rows": file_rows}


def build_differences(
    whiskers_rows: list[dict],
    dataset2_rows: list[dict],
    content_by_folder: dict[str, dict],
) -> tuple[list[str], list[dict]]:
    """
    Compare folders by name (counts) and attach deep CSV content status.

    Status:
      - only_in_whiskers
      - only_in_whisker_dataset_2
      - identical_counts_and_csv_content
      - identical_counts_but_csv_differs
      - count_differs
    """
    w_map = {r["Folder"]: r for r in whiskers_rows}
    d_map = {r["Folder"]: r for r in dataset2_rows}
    all_folders = sorted(set(w_map) | set(d_map))

    count_cols = ["Images"] + [f"{i} labels" for i in LABEL_IDS]
    content_cols = [
        "ContentStatus",
        "shared_csvs",
        "identical_csvs",
        "size_mismatch_csvs",
        "content_mismatch_csvs",
        "only_in_whiskers_csvs",
        "only_in_whisker_dataset_2_csvs",
    ]
    fieldnames = (
        ["Folder", "Status"]
        + content_cols
        + [f"whiskers_{c}" for c in count_cols]
        + [f"whisker_dataset_2_{c}" for c in count_cols]
        + [f"diff_{c}" for c in count_cols]
    )

    rows: list[dict] = []
    for folder in all_folders:
        w = w_map.get(folder)
        d = d_map.get(folder)
        content = content_by_folder.get(folder, {})

        if w is None:
            status = "only_in_whisker_dataset_2"
        elif d is None:
            status = "only_in_whiskers"
        else:
            same_counts = all(w[c] == d[c] for c in count_cols)
            csv_ok = content.get("ContentStatus") == "csv_identical"
            if same_counts and csv_ok:
                status = "identical_counts_and_csv_content"
            elif same_counts:
                status = "identical_counts_but_csv_differs"
            else:
                status = "count_differs"

        row: dict = {"Folder": folder, "Status": status}
        for c in content_cols:
            row[c] = content.get(c, "")
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

    shared_folders = sorted(
        {r["Folder"] for r in whiskers_rows} & {r["Folder"] for r in dataset2_rows}
    )
    print(f"\nDeep-comparing label CSVs in {len(shared_folders)} shared folder(s)...")

    content_by_folder: dict[str, dict] = {}
    all_file_rows: list[dict] = []

    for i, folder in enumerate(shared_folders, start=1):
        result = deep_compare_session(folder, WHISKERS, WHISKER_DATASET_2)
        content_by_folder[folder] = result["summary"]
        all_file_rows.extend(result["file_rows"])
        s = result["summary"]
        print(
            f"  [{i}/{len(shared_folders)}] {folder}: "
            f"{s['ContentStatus']} "
            f"(shared={s['shared_csvs']}, identical={s['identical_csvs']}, "
            f"size_mismatch={s['size_mismatch_csvs']}, "
            f"content_mismatch={s['content_mismatch_csvs']}, "
            f"only_w={s['only_in_whiskers_csvs']}, "
            f"only_d2={s['only_in_whisker_dataset_2_csvs']})"
        )

    out_d2 = OUTPUT_DIR / "whisker_dataset_2_inventory.csv"
    out_w = OUTPUT_DIR / "whiskers_inventory.csv"
    out_diff = OUTPUT_DIR / "dataset_differences.csv"
    out_files = OUTPUT_DIR / "csv_file_differences.csv"

    write_csv(out_d2, dataset2_rows, list(COLUMNS))
    write_csv(out_w, whiskers_rows, list(COLUMNS))

    diff_fields, diff_rows = build_differences(
        whiskers_rows, dataset2_rows, content_by_folder
    )
    write_csv(out_diff, diff_rows, diff_fields)

    file_fields = [
        "Folder",
        "RelativePath",
        "Status",
        "whiskers_bytes",
        "whisker_dataset_2_bytes",
    ]
    write_csv(out_files, all_file_rows, file_fields)

    n_only_w = sum(1 for r in diff_rows if r["Status"] == "only_in_whiskers")
    n_only_d = sum(1 for r in diff_rows if r["Status"] == "only_in_whisker_dataset_2")
    n_ok = sum(
        1 for r in diff_rows if r["Status"] == "identical_counts_and_csv_content"
    )
    n_count_diff = sum(1 for r in diff_rows if r["Status"] == "count_differs")
    n_csv_diff = sum(
        1 for r in diff_rows if r["Status"] == "identical_counts_but_csv_differs"
    )

    print(f"\nWrote:\n  {out_d2}\n  {out_w}\n  {out_diff}\n  {out_files}")
    print(
        f"Folders: identical_counts_and_csv_content={n_ok}, "
        f"identical_counts_but_csv_differs={n_csv_diff}, "
        f"count_differs={n_count_diff}, "
        f"only_in_whiskers={n_only_w}, only_in_whisker_dataset_2={n_only_d}"
    )
    print(f"File-level mismatch/missing rows written: {len(all_file_rows)}")


if __name__ == "__main__":
    main()
