"""
Z-Score Normalised PSTH Heatmap
================================

For each interval in the contact PSTH firing-rate CSV, produces a heatmap
where:
  - Each row is a unit
  - Each column is a 1-ms time bin
  - Values are z-scored per unit (baseline = [−50, −10) ms)
  - Units are sorted by peak response latency
  - A vertical dashed line marks contact onset (t = 0 ms)

Outputs one PNG per interval into ``<session>/contact_psth_csv_output/heatmaps/``.

Usage
-----
    python psth_heatmap.py --csv "path/to/contact_psth_firing_rates.csv"
    python psth_heatmap.py --data_dir "C:\\path\\to\\session"
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def make_heatmap(df, interval, out_dir, vmin=-3, vmax=5):
    """
    Build and save a z-score normalised heatmap for one interval.

    Z-score per unit:
        z(t) = (FR(t) − μ_baseline) / σ_baseline
    where baseline = [−50, −10) ms.
    """
    sub = df[df["interval"] == interval].copy()
    if sub.empty:
        return

    units = sorted(sub["unit"].unique())
    bins = sorted(sub["bin_ms"].unique())

    # Build (n_units × n_bins) matrix
    matrix = np.full((len(units), len(bins)), np.nan)
    for ui, unit in enumerate(units):
        usub = sub[sub["unit"] == unit].sort_values("bin_ms")
        fr = usub["firing_rate_hz"].values
        b = usub["bin_ms"].values

        # Baseline stats from [−50, −10) ms
        bl_mask = (b >= -50) & (b < -10)
        if bl_mask.sum() > 0:
            bl_mean = fr[bl_mask].mean()
            bl_std = fr[bl_mask].std(ddof=1)
        else:
            bl_mean = fr.mean()
            bl_std = fr.std(ddof=1)

        if bl_std < 1e-6:
            bl_std = 1.0  # avoid division by zero for silent units

        z = (fr - bl_mean) / bl_std
        matrix[ui, :len(z)] = z

    # Sort rows by peak response latency (0–50 ms window)
    bins_arr = np.array(bins)
    post_mask = (bins_arr >= 0) & (bins_arr < 50)
    if post_mask.sum() > 0:
        peak_latencies = bins_arr[post_mask][np.argmax(matrix[:, post_mask], axis=1)]
    else:
        peak_latencies = np.zeros(len(units))
    sort_order = np.argsort(peak_latencies)
    matrix = matrix[sort_order]
    sorted_units = [units[i] for i in sort_order]

    # ── Plot ──────────────────────────────────────────────────────────
    fig_h = max(4, len(units) * 0.35)
    fig, ax = plt.subplots(figsize=(12, fig_h))

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm,
                   interpolation="nearest",
                   extent=[bins_arr.min(), bins_arr.max(), len(units) - 0.5, -0.5])

    # Contact onset line
    ax.axvline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.8)

    ax.set_xlabel("Time from contact onset (ms)", fontsize=11)
    ax.set_ylabel("Unit", fontsize=11)
    ax.set_yticks(range(len(sorted_units)))
    ax.set_yticklabels([f"U{u}" for u in sorted_units], fontsize=7)

    # Interval label for title
    title = interval.replace("_", " ").replace("mask ", "").title()
    n_trials = sub["n_trials"].iloc[0]
    ax.set_title(f"{title}  (n = {n_trials} events)\nZ-score normalised firing rate",
                 fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Z-score (vs baseline)", fontsize=9)

    fig.tight_layout()
    fname = f"heatmap_{interval}.png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def run(csv_path, out_dir=None):
    print(f"Loading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {df['unit'].nunique()} units, "
          f"{df['interval'].nunique()} intervals, "
          f"{df['bin_ms'].nunique()} bins")

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    intervals = df["interval"].unique()
    print(f"\nGenerating {len(intervals)} heatmaps …")
    for interval in intervals:
        make_heatmap(df, interval, out_dir)

    print(f"\nDone.  All heatmaps → {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Z-score normalised PSTH heatmap per interval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--data_dir",
                   help="Session directory (looks for contact_psth_csv_output/)")
    g.add_argument("--csv",
                   help="Direct path to contact_psth_firing_rates.csv")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: <csv_dir>/heatmaps)")
    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(args.data_dir, "contact_psth_csv_output",
                                "contact_psth_firing_rates.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    run(csv_path, out_dir=args.output_dir)


if __name__ == "__main__":
    main()
