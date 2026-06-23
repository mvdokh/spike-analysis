import csv
import os

ROOT_DIR = r"H:\My Drive\Ina\PCRt_BiPoles"
OUTPUT_CSV = os.path.join(ROOT_DIR, "behavior_csv_row_comparison.csv")


def count_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def find_behavior_csv(folder_path, view):
    pattern = f"{view}_view_behavior"
    matches = [
        os.path.join(folder_path, filename)
        for filename in os.listdir(folder_path)
        if pattern in filename.lower() and filename.lower().endswith(".csv")
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    return sorted(matches)[0]


def find_session_folders(root_dir):
    session_folders = []
    for dirpath, _, filenames in os.walk(root_dir):
        bottom_csv = find_behavior_csv(dirpath, "bottom")
        side_csv = find_behavior_csv(dirpath, "side")
        if bottom_csv or side_csv:
            session_folders.append(dirpath)
    return sorted(session_folders)


def parse_experiment_labels(folder_path):
    rel_path = os.path.relpath(folder_path, ROOT_DIR)
    parts = rel_path.split(os.sep)
    animal = parts[0] if parts else ""
    session = parts[-1] if parts else ""
    return rel_path, animal, session


def main():
    session_folders = find_session_folders(ROOT_DIR)
    if not session_folders:
        print(f"No behavior CSV folders found in {ROOT_DIR}")
        return

    results = []

    for folder_path in session_folders:
        experiment, animal, session = parse_experiment_labels(folder_path)

        bottom_csv = find_behavior_csv(folder_path, "bottom")
        side_csv = find_behavior_csv(folder_path, "side")

        bottom_rows = count_rows(bottom_csv) if bottom_csv else None
        side_rows = count_rows(side_csv) if side_csv else None

        if bottom_rows is not None and side_rows is not None:
            row_difference = side_rows - bottom_rows
        else:
            row_difference = ""

        print(
            f"{experiment}: "
            f"bottom={bottom_rows if bottom_rows is not None else 'MISSING'}, "
            f"side={side_rows if side_rows is not None else 'MISSING'}, "
            f"diff={row_difference if row_difference != '' else 'N/A'}"
        )

        results.append(
            (
                experiment,
                animal,
                session,
                bottom_rows if bottom_rows is not None else "MISSING",
                side_rows if side_rows is not None else "MISSING",
                row_difference if row_difference != "" else "N/A",
            )
        )

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "experiment",
                "animal",
                "session",
                "bottom_rows",
                "side_rows",
                "side_minus_bottom",
            ]
        )
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
