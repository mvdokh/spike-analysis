"""
Unit × Interval Response Grid
==============================

Produces a single heatmap where:
  - Each row is a unit
  - Each column is a contact-interval file
  - Cell value = mean z-scored firing rate in [0, 50) ms post-contact

Z-score per unit/interval:
    baseline = [−50, −10) ms
    z(t) = (FR(t) − μ_baseline) / σ_baseline
    cell  = mean z over [0, 50) ms

Usage
-----
    python unit_interval_response_grid.py --data_dir "C:\\path\\to\\session"
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import linkage, dendrogram

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_psth import (
    load_frame_sync,
    frame_to_seconds,
    load_contact_intervals,
    load_spikes,
    align_spikes_to_events,
)


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

    for p in (digitalin_path, spikes_path):
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

    all_csv_files = sorted(glob.glob(os.path.join(contact_dir, "*.csv")))
    if not all_csv_files:
        raise FileNotFoundError(f"No CSV files found in {contact_dir}")

    pre_s = pre_ms / 1000.0
    post_s = post_ms / 1000.0
    bins = np.arange(-pre_ms, post_ms + bin_ms, bin_ms)
    centres = (bins[:-1] + bins[1:]) / 2

    session_name = os.path.basename(os.path.normpath(data_dir))

    # ── Compute mean z-scored response for each unit × interval ─────
    interval_names = []
    n_trials_list = []
    # response_matrix[unit_idx, interval_idx] = mean z in [0, 50) ms
    response_matrix = []

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

        col_values = np.zeros(len(process_units))
        n_trials = 0

        for ui, unit in enumerate(process_units):
            unit_spikes = spikes_df.loc[spikes_df["unit"] == unit, "time"].values
            trials = align_spikes_to_events(unit_spikes, starts_s, ends_s,
                                            pre_s, post_s)
            n_trials = max(n_trials, len(trials))
            if not trials:
                col_values[ui] = 0.0
                continue

            all_spike_times = []
            for t in trials:
                all_spike_times.extend(t["spike_times_ms"])
            all_spike_times = np.array(all_spike_times)

            if len(all_spike_times) > 0:
                counts, _ = np.histogram(all_spike_times, bins=bins)
            else:
                counts = np.zeros(len(centres))
            fr = counts / (len(trials) * bin_ms / 1000.0)

            # Z-score using baseline [−50, −10) ms
            bl_mask = (centres >= -50) & (centres < -10)
            if bl_mask.sum() > 0:
                bl_mean = fr[bl_mask].mean()
                bl_std = fr[bl_mask].std(ddof=1)
            else:
                bl_mean = fr.mean()
                bl_std = fr.std(ddof=1)
            if bl_std < 1e-6:
                bl_std = 1.0

            z = (fr - bl_mean) / bl_std

            # Mean z in response window [0, 50) ms
            resp_mask = (centres >= 0) & (centres < 50)
            col_values[ui] = z[resp_mask].mean() if resp_mask.sum() > 0 else 0.0

        interval_names.append(interval)
        n_trials_list.append(n_trials)
        response_matrix.append(col_values)
        print(f"  {interval}: {n_trials} trials")

    if not response_matrix:
        print("No intervals with events found.")
        return

    # (n_units × n_intervals)
    matrix = np.column_stack(response_matrix)

    # ── Pretty labels ───────────────────────────────────────────────
    x_labels = []
    for name, nt in zip(interval_names, n_trials_list):
        short = name.replace("_mask_contact", "").replace("_no_collision", "")
        short = short.replace("_", " ").title()
        x_labels.append(f"{short}\n(n={nt})")

    y_labels = [f"U{u}" for u in process_units]

    n_units = len(process_units)
    n_intervals = len(interval_names)

    # ── Hierarchical clustering on units ────────────────────────────
    if n_units > 1:
        Z = linkage(matrix, method="ward", metric="euclidean")
        dn = dendrogram(Z, no_plot=True)
        row_order = dn["leaves"]
    else:
        row_order = list(range(n_units))
    matrix = matrix[row_order]
    y_labels = [y_labels[i] for i in row_order]

    # ── Plot with dendrogram ────────────────────────────────────────
    cell_size = 0.7
    fig_w = max(6, n_intervals * cell_size + 4)
    fig_h = max(5, n_units * cell_size + 2)
    side = max(fig_w, fig_h)

    fig = plt.figure(figsize=(side, side))
    # Layout: [dendrogram | heatmap | colorbar]
    gs = fig.add_gridspec(1, 2, width_ratios=[0.15, 1], wspace=0.02)

    # Dendrogram axis
    ax_dend = fig.add_subplot(gs[0, 0])
    if n_units > 1:
        dendrogram(Z, orientation="left", ax=ax_dend,
                   leaf_rotation=0, no_labels=True,
                   color_threshold=0, above_threshold_color="black")
    ax_dend.set_axis_off()

    # Heatmap axis
    ax = fig.add_subplot(gs[0, 1])

    vmax = max(3, np.nanpercentile(np.abs(matrix), 95))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm,
                   interpolation="nearest")

    ax.set_xticks(range(n_intervals))
    ax.set_xticklabels(x_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(n_units))
    ax.set_yticklabels(y_labels, fontsize=8)

    ax.set_xlabel("Interval", fontsize=11)
    ax.set_ylabel("Unit", fontsize=11)
    ax.set_title(f"{session_name}\nMean z-scored response [0–50 ms]",
                 fontsize=13, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean z-score (vs baseline)", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    stem = "unit_interval_response_grid"
    for ext in (".png", ".svg"):
        path = os.path.join(output_dir, stem + ext)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {stem}.png / .svg → {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unit × Interval z-scored response grid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", required=True)
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
