import csv
import os

ROOT_DIR = r"C:\Users\wanglab\Desktop\Ina\PCRt_TeLC"
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


def get_pre_post(folder_name):
    folder_name = folder_name.lower()
    if folder_name.endswith("_pre"):
        return "pre"
    if folder_name.endswith("_post"):
        return "post"
    return "unknown"


def get_experiment_folders(root_dir):
    folders = []
    for name in os.listdir(root_dir):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path):
            folders.append(path)
    return sorted(folders)


def main():
    results = []

    for folder_path in get_experiment_folders(ROOT_DIR):
        folder_name = os.path.basename(folder_path)
        pre_post = get_pre_post(folder_name)

        bottom_csv = find_behavior_csv(folder_path, "bottom")
        side_csv = find_behavior_csv(folder_path, "side")

        bottom_rows = count_rows(bottom_csv) if bottom_csv else None
        side_rows = count_rows(side_csv) if side_csv else None

        if bottom_rows is not None and side_rows is not None:
            row_difference = side_rows - bottom_rows
        else:
            row_difference = ""

        print(
            f"{folder_name} ({pre_post}): "
            f"bottom={bottom_rows if bottom_rows is not None else 'MISSING'}, "
            f"side={side_rows if side_rows is not None else 'MISSING'}, "
            f"diff={row_difference if row_difference != '' else 'N/A'}"
        )

        results.append(
            (
                folder_name,
                pre_post,
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
                "pre_post",
                "bottom_rows",
                "side_rows",
                "side_minus_bottom",
            ]
        )
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
