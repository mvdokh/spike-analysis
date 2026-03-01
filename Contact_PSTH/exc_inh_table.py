"""
Excitation / Inhibition Classification Table

Produces a CSV with one row per unit and columns for each whisker × direction
(W0 Ret, W0 Pro, W1 Ret, …).  Each cell is labelled "excited", "inhibited",
or left blank based on whether the evoked FR deviates by ≥ 20 Hz from baseline.

Usage
-----
    python exc_inh_table.py --data_dir <session>
    python exc_inh_table.py --csv <path/to/contact_psth_firing_rates.csv>
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


def _parse_interval(name: str):
    """Return (whisker int | None, direction str)."""
    m = re.match(
        r"interval_(\d+)_mask_contact(?:_(protraction|retraction))?$", name
    )
    if m:
        whisker = int(m.group(1))
        direction = m.group(2) if m.group(2) else "all"
        return whisker, direction
    return None, "unknown"


def build_table(csv_path, threshold=20.0,
                baseline_window=(-50, 0), response_window=(0, 50)):
    df = pd.read_csv(csv_path)
    units = sorted(df["unit"].unique())
    intervals = sorted(df["interval"].unique())

    # Find available whiskers from retraction/protraction intervals
    whiskers = set()
    for name in intervals:
        w, d = _parse_interval(name)
        if w is not None and d in ("retraction", "protraction"):
            whiskers.add(w)
    whiskers = sorted(whiskers)

    # Build column order: W0 Ret, W0 Pro, W1 Ret, W1 Pro, …
    columns = []
    for w in whiskers:
        columns.append(f"W{w} Ret")
        columns.append(f"W{w} Pro")

    rows = []
    for unit in units:
        row = {"Unit": unit}
        for w in whiskers:
            for direction, col_suffix in [("retraction", "Ret"),
                                          ("protraction", "Pro")]:
                interval = f"interval_{w}_mask_contact_{direction}"
                sub = df[(df["unit"] == unit) & (df["interval"] == interval)]
                if len(sub) == 0:
                    row[f"W{w} {col_suffix}"] = ""
                    continue

                bins = sub["bin_ms"].values
                fr = sub["firing_rate_hz"].values

                bl_mask = (bins >= baseline_window[0]) & (bins < baseline_window[1])
                resp_mask = (bins >= response_window[0]) & (bins < response_window[1])

                baseline = fr[bl_mask].mean() if bl_mask.sum() > 0 else 0.0
                response = fr[resp_mask].mean() if resp_mask.sum() > 0 else 0.0
                evoked = response - baseline

                if evoked >= threshold:
                    label = "excited"
                elif evoked <= -threshold:
                    label = "inhibited"
                else:
                    label = ""

                row[f"W{w} {col_suffix}"] = label
        rows.append(row)

    result = pd.DataFrame(rows, columns=["Unit"] + columns)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate excited/inhibited classification table.")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--data_dir", help="Session directory")
    g.add_argument("--csv", help="Path to contact_psth_firing_rates.csv")
    parser.add_argument("--threshold", type=float, default=20.0,
                        help="Evoked FR threshold (Hz) for classification "
                             "(default: 20)")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(args.data_dir, "contact_psth_csv_output",
                                "contact_psth_firing_rates.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = args.output_dir or os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    table = build_table(csv_path, threshold=args.threshold)
    out_path = os.path.join(out_dir, "exc_inh_table.csv")
    table.to_csv(out_path, index=False)

    print(table.to_string(index=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
