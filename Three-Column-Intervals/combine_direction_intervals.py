from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


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
    retraction_df = load_intervals(retraction_csv, "Retraction")

    combined = pd.concat([protraction_df, retraction_df], ignore_index=True)
    combined = combined.sort_values(["Contact Start", "Contact End", "Direction"], kind="stable")
    combined = combined.loc[:, OUTPUT_COLUMNS]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    return output_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine protraction and retraction interval CSVs into a three-column CSV."
    )
    parser.add_argument("protraction_csv", type=Path, help="Path to the protraction interval CSV")
    parser.add_argument("retraction_csv", type=Path, help="Path to the retraction interval CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to direction.csv in the input folder.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.output is None:
        output_csv = args.protraction_csv.parent / "direction.csv"
    else:
        output_csv = args.output

    saved_path = combine_interval_files(args.protraction_csv, args.retraction_csv, output_csv)
    print(f"Saved combined intervals to {saved_path}")


if __name__ == "__main__":
    main()