from __future__ import annotations

from pathlib import Path

import pandas as pd


# Edit these paths in one place before running the script.
INPUT_FOLDER = Path(r"H:\.shortcut-targets-by-id\17tfqA28bUKBm67DoeFcz3RtG8jqNNu4B\NTNG1-PrV-Piezo\P003\052725_1\contact\4")
PROTRACTION_CSV = INPUT_FOLDER / "4_protraction.csv"
RETRACTION_CSV = INPUT_FOLDER / "4_retraction.csv"
OUTPUT_CSV = INPUT_FOLDER / "direction.csv"

OUTPUT_COLUMNS = ["Contact Start", "Contact End", "Direction"]


def load_intervals(csv_path: Path, direction: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]

    if len(df.columns) < 2:
        raise ValueError(f"Expected at least two columns in {csv_path}")

    start_col, end_col = df.columns[:2]
    result = df.loc[:, [start_col, end_col]].copy()
    result.columns = ["Contact Start", "Contact End"]
    result["Contact Start"] = result["Contact Start"].astype(int)
    result["Contact End"] = result["Contact End"].astype(int)
    result["Direction"] = direction
    return result


def combine_interval_files(protraction_csv: Path, retraction_csv: Path, output_csv: Path) -> Path:
    protraction_df = load_intervals(protraction_csv, "Protraction")
    frames = [protraction_df]

    if retraction_csv.exists():
        frames.append(load_intervals(retraction_csv, "Retraction"))
    else:
        print(f"No retraction CSV found at {retraction_csv}; writing protraction rows only.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["Contact Start", "Contact End", "Direction"], kind="stable")
    combined = combined.loc[:, OUTPUT_COLUMNS]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    return output_csv


def main() -> None:
    saved_path = combine_interval_files(PROTRACTION_CSV, RETRACTION_CSV, OUTPUT_CSV)
    print(f"Saved combined intervals to {saved_path}")


if __name__ == "__main__":
    main()