"""
Z-Score Normalised PSTH Heatmap
================================

For each contact-interval CSV in per_whisker_contact/, computes PSTH firing
rates directly from digitalin.dat + spikes.csv, then produces a heatmap where:
  - Each row is a unit
  - Each column is a 1-ms time bin
  - Values are z-scored per unit (baseline = [−50, −10) ms)
  - Units are sorted by peak response latency
  - A vertical dashed line marks contact onset (t = 0 ms)

Usage
-----
    python psth_heatmap.py --data_dir "C:\\path\\to\\session"
    python psth_heatmap.py --data_dir "..." --contact_dir "..."
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_psth import (
    load_frame_sync,
    frame_to_seconds,
    load_contact_intervals,
    load_spikes,
    align_spikes_to_events,
)


def make_heatmap(matrix, bins_arr, units, n_trials, interval, out_dir,
                 session_name="", vmin=-3, vmax=5):
    """
    Build and save a z-score normalised heatmap for one interval.

    Parameters
    ----------
    matrix : (n_units, n_bins) array of firing rates (Hz)
    bins_arr : 1-D array of bin centres (ms)
    units : list of unit IDs matching row order
    n_trials : number of trials for this interval
    interval : string label
    out_dir : output directory
    """
    # Z-score per unit:  z(t) = (FR(t) − μ_baseline) / σ_baseline
    z_matrix = np.full_like(matrix, np.nan)
    for ui in range(matrix.shape[0]):
        fr = matrix[ui]
        bl_mask = (bins_arr >= -50) & (bins_arr < -10)
        if bl_mask.sum() > 0:
            bl_mean = fr[bl_mask].mean()
            bl_std = fr[bl_mask].std(ddof=1)
        else:
            bl_mean = fr.mean()
            bl_std = fr.std(ddof=1)
        if bl_std < 1e-6:
            bl_std = 1.0
        z_matrix[ui] = (fr - bl_mean) / bl_std

    # Sort rows by peak response latency (0–50 ms window)
    post_mask = (bins_arr >= 0) & (bins_arr < 50)
    if post_mask.sum() > 0:
        peak_latencies = bins_arr[post_mask][np.argmax(z_matrix[:, post_mask], axis=1)]
    else:
        peak_latencies = np.zeros(len(units))
    sort_order = np.argsort(peak_latencies)
    z_matrix = z_matrix[sort_order]
    sorted_units = [units[i] for i in sort_order]

    # ── Plot ──────────────────────────────────────────────────────────
    fig_size = max(5, len(units) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    im = ax.imshow(z_matrix, aspect="auto", cmap="RdBu_r", norm=norm,
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
    title_line = f"{title}  (n = {n_trials} events)"
    if session_name:
        title_line = f"{session_name} — {title_line}"
    ax.set_title(f"{title_line}\nZ-score normalised firing rate",
                 fontsize=12, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Z-score (vs baseline)", fontsize=9)

    # Remove spines without scales
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    stem = f"heatmap_{interval}"
    for ext in (".png", ".svg"):
        path = os.path.join(out_dir, stem + ext)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {stem}.png / .svg")


def run(data_dir, contact_dir=None, output_dir=None,
        pre_ms=50.0, post_ms=100.0, bin_ms=1.0,
        sampling_rate=30_000, sync_channel=1, units=None):

    digitalin_path = os.path.join(data_dir, "digitalin.dat")
    spikes_path = os.path.join(data_dir, "spikes.csv")
    if contact_dir is None:
        contact_dir = os.path.join(data_dir, "per_whisker_contact")
    if output_dir is None:
        output_dir = os.path.join(data_dir, "heatmap_no_collision")
    os.makedirs(output_dir, exist_ok=True)

    for p, lab in [(digitalin_path, "digitalin.dat"),
                   (spikes_path, "spikes.csv")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    if not os.path.isdir(contact_dir):
        raise FileNotFoundError(f"Contact directory not found: {contact_dir}")

    frame_samples = load_frame_sync(digitalin_path, channel=sync_channel,
                                    sampling_rate=sampling_rate)
    spikes_df = load_spikes(spikes_path)

    available_units = sorted(spikes_df["unit"].unique())
    if units is not None:
        process_units = [u for u in units if u in available_units]
    else:
        process_units = available_units
    print(f"Units to process: {process_units}")

    # Discover contact CSVs
    all_csv_files = sorted(glob.glob(os.path.join(contact_dir, "*.csv")))
    if not all_csv_files:
        raise FileNotFoundError(f"No CSV files found in {contact_dir}")
    print(f"Found {len(all_csv_files)} contact interval file(s)")

    pre_s = pre_ms / 1000.0
    post_s = post_ms / 1000.0
    t_min = -pre_ms
    t_max = post_ms
    bins = np.arange(t_min, t_max + bin_ms, bin_ms)
    centres = (bins[:-1] + bins[1:]) / 2

    session_name = os.path.basename(os.path.normpath(data_dir))

    # Process each interval CSV
    for csv_path in all_csv_files:
        interval = os.path.splitext(os.path.basename(csv_path))[0]
        intervals_df = load_contact_intervals(csv_path)
        if len(intervals_df) == 0:
            print(f"  {interval}: no events, skipping")
            continue

        starts_s = frame_to_seconds(intervals_df["Start"].values,
                                    frame_samples, sampling_rate)
        ends_s = frame_to_seconds(intervals_df["End"].values,
                                  frame_samples, sampling_rate)

        # Build firing-rate matrix (n_units x n_bins)
        matrix = np.zeros((len(process_units), len(centres)))
        n_trials = 0
        for ui, unit in enumerate(process_units):
            unit_spikes = spikes_df.loc[spikes_df["unit"] == unit, "time"].values
            trials = align_spikes_to_events(unit_spikes, starts_s, ends_s,
                                            pre_s, post_s)
            n_trials = max(n_trials, len(trials))
            if not trials:
                continue
            all_spike_times = []
            for t in trials:
                all_spike_times.extend(t["spike_times_ms"])
            all_spike_times = np.array(all_spike_times)
            if len(all_spike_times) > 0:
                counts, _ = np.histogram(all_spike_times, bins=bins)
            else:
                counts = np.zeros(len(centres))
            matrix[ui] = counts / (len(trials) * bin_ms / 1000.0)

        print(f"  {interval}: {n_trials} trials, {len(process_units)} units")
        make_heatmap(matrix, centres, process_units, n_trials, interval,
                     output_dir, session_name=session_name)

    print(f"\nDone.  All heatmaps → {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Z-score normalised PSTH heatmap per interval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", required=True,
                        help="Session directory containing digitalin.dat, "
                             "spikes.csv, and per_whisker_contact/")
    parser.add_argument("--contact_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pre_ms", type=float, default=50)
    parser.add_argument("--post_ms", type=float, default=100)
    parser.add_argument("--bin_ms", type=float, default=1)
    parser.add_argument("--sampling_rate", type=int, default=30000)
    parser.add_argument("--sync_channel", type=int, default=1)
    parser.add_argument("--units", type=int, nargs="*", default=None)
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        contact_dir=args.contact_dir,
        output_dir=args.output_dir,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        bin_ms=args.bin_ms,
        sampling_rate=args.sampling_rate,
        sync_channel=args.sync_channel,
        units=args.units,
    )


if __name__ == "__main__":
    main()
