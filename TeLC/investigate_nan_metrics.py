#!/usr/bin/env python3
"""
Investigate NaN metrics in TeLC licking summary CSV.

Usage:
    python investigate_nan_metrics.py
    python investigate_nan_metrics.py --csv "C:\\path\\to\\TeLC_licking_metrics.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


NAN_METRIC_COLS = ["lick_dur", "ILI", "licks_per_bout", "tongue_area"]
ROW_FOCUS = [
    ("09", "Post", "2026_04_08", "side"),
    ("09", "Post", "2026_04_09", "bottom"),
    ("09", "Post", "2026_04_09", "side"),
]


def explain_row(row: pd.Series) -> list[str]:
    reasons: list[str] = []
    n_licks = int(row["n_licks"])
    n_bouts = row["bout_rate"] * row["n_frames"] if pd.notna(row["bout_rate"]) else float("nan")

    if n_licks == 0:
        reasons.append("n_licks = 0, so lick-level means (duration/area/ILI) are undefined")
    if n_licks <= 1:
        reasons.append("n_licks <= 1, so ILI is undefined (needs at least two licks)")
    if pd.notna(n_bouts) and n_bouts == 0:
        reasons.append("n_bouts = 0, so licks_per_bout is undefined (mean over empty set)")

    nan_cols = [c for c in NAN_METRIC_COLS if pd.isna(row[c])]
    if nan_cols:
        reasons.append(f"NaN columns present: {', '.join(nan_cols)}")

    if not reasons:
        reasons.append("No obvious NaN cause detected from summary CSV alone")
    return reasons


def row_selector(df: pd.DataFrame, animal: str, condition: str, sess_lbl: str, view: str) -> pd.Series:
    mask = (
        (df["animal"] == animal)
        & (df["condition"] == condition)
        & (df["sess_lbl"] == sess_lbl)
        & (df["view"] == view)
    )
    out = df[mask]
    if out.empty:
        raise ValueError(f"Row not found: animal={animal}, condition={condition}, sess_lbl={sess_lbl}, view={view}")
    if len(out) > 1:
        raise ValueError(f"Expected one row, found {len(out)} rows for {animal}/{condition}/{sess_lbl}/{view}")
    return out.iloc[0]


def print_focus_report(df: pd.DataFrame) -> None:
    print("\n=== Focus rows (23-25 equivalent IDs) ===")
    for animal, condition, sess_lbl, view in ROW_FOCUS:
        row = row_selector(df, animal, condition, sess_lbl, view)
        print(f"\nRow: animal={animal}, condition={condition}, sess_lbl={sess_lbl}, view={view}")
        print(
            f"  n_frames={int(row['n_frames'])}, n_licks={int(row['n_licks'])}, "
            f"lick_rate={row['lick_rate']}, bout_rate={row['bout_rate']}"
        )
        for c in NAN_METRIC_COLS:
            print(f"  {c}={row[c]}")
        for reason in explain_row(row):
            print(f"  -> {reason}")


def print_global_nan_summary(df: pd.DataFrame) -> None:
    nan_mask = df[NAN_METRIC_COLS].isna().any(axis=1)
    nan_df = df[nan_mask].copy()

    print("\n=== Global NaN summary ===")
    print(f"Rows with at least one NaN among {NAN_METRIC_COLS}: {len(nan_df)}/{len(df)}")

    if nan_df.empty:
        print("No NaNs found.")
        return

    zero_lick_nan = (nan_df["n_licks"] == 0).sum()
    one_or_less_lick = (nan_df["n_licks"] <= 1).sum()
    print(f"  Rows with NaN and n_licks == 0 : {zero_lick_nan}")
    print(f"  Rows with NaN and n_licks <= 1 : {one_or_less_lick}")

    grouped = (
        nan_df.groupby(["animal", "view"], dropna=False)
        .size()
        .reset_index(name="nan_rows")
        .sort_values(["animal", "view"])
    )
    print("\nNaN rows by animal/view:")
    print(grouped.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain NaN rows in TeLC licking metrics.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "TeLC_licking_metrics.csv",
        help="Path to TeLC_licking_metrics.csv",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv, dtype={"animal": str, "condition": str, "sess_lbl": str, "view": str})
    for col in ["n_frames", "n_licks", "lick_rate", "bout_rate"] + NAN_METRIC_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing expected column: {col}")

    print(f"Loaded: {args.csv}")
    print_focus_report(df)
    print_global_nan_summary(df)

    print("\nInterpretation:")
    print("- If n_licks = 0, NaN for lick_dur/tongue_area/ILI is expected.")
    print("- If n_licks <= 1, NaN for ILI is expected.")
    print("- If n_bouts = 0 (or no bout IDs), NaN for licks_per_bout is expected.")


if __name__ == "__main__":
    main()
