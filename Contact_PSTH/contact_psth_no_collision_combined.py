"""
Contact PSTH Combined — No-Collision Layout

Top row:  Contact Intervals (left) + Contact Intervals No Collision (center) + Collision (right).
Rows 1–5: Each whisker on its own row:
    [W# No Collision,  W# Protraction No Collision,  W# Retraction No Collision]

Usage
-----
    python contact_psth_no_collision_combined.py --data_dir <session_dir>
                                                 [--contact_dir <dir>]
                                                 [--output_dir <dir>]
                                                 [--pre_ms 50] [--post_ms 100]
                                                 [--bin_ms 1]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_psth import (
    load_frame_sync,
    frame_to_seconds,
    load_contact_intervals,
    load_spikes,
    align_spikes_to_events,
)


def draw_psth_raster_on_axes(ax_psth, ax_raster, trials, label,
                              pre_ms, post_ms, bin_ms):
    if not trials:
        ax_psth.set_title(f"{label}  (no trials)")
        return

    avg_dur = np.mean([t["duration_ms"] for t in trials])
    t_min = -pre_ms
    t_max = post_ms
    bins = np.arange(t_min, t_max + bin_ms, bin_ms)

    all_spikes = []
    for t in trials:
        all_spikes.extend(t["spike_times_ms"])
    all_spikes = np.array(all_spikes)

    if len(all_spikes) > 0:
        counts, edges = np.histogram(all_spikes, bins=bins)
        centres = (edges[:-1] + edges[1:]) / 2
        firing_rate = counts / (len(trials) * bin_ms / 1000.0)
        ax_psth.bar(centres, firing_rate, width=bin_ms,
                    color="black", edgecolor="black", linewidth=0.3)

    ax_psth.axvline(0, color="black", ls="--", alpha=0.7)
    ax_psth.set_ylabel("FR (Hz)", fontsize=7)
    ax_psth.set_title(f"{label}  |  {len(trials)} trials  |  "
                      f"avg dur {avg_dur:.1f} ms", fontsize=8)
    ax_psth.set_xlim(t_min, t_max)
    ax_psth.tick_params(labelsize=6)

    sorted_trials = sorted(trials, key=lambda t: t["duration_ms"])
    for idx, trial in enumerate(sorted_trials):
        st = trial["spike_times_ms"]
        if st:
            ax_raster.scatter(st, [idx] * len(st),
                              s=3, color="black", alpha=0.8,
                              marker=(4, 0, 45), linewidths=0)

    offset_times = [t["duration_ms"] for t in sorted_trials]
    ax_raster.plot(offset_times, range(len(sorted_trials)),
                   color="red", linewidth=0.8, alpha=0.7)

    ax_raster.axvline(0, color="black", ls="--", alpha=0.7)
    ax_raster.set_ylabel("Trial", fontsize=7)
    ax_raster.set_ylim(-0.5, len(trials) - 0.5)
    ax_raster.set_xlim(t_min, t_max)
    ax_raster.tick_params(labelsize=6)

    # Remove spines on sides without scales
    ax_psth.spines["top"].set_visible(False)
    ax_psth.spines["right"].set_visible(False)
    ax_raster.spines["top"].set_visible(False)
    ax_raster.spines["right"].set_visible(False)


# ── File-to-label mapping ──────────────────────────────────────────
# Top row
TOP_FILES = [
    ("contact_intervals.csv",  "Contact Intervals"),
    ("all_no_collision.csv",   "Contact Intervals No Collision"),
    ("collision_all.csv",      "Collision"),
]

# Per-whisker rows (interval_0 → W0, … interval_4 → W4)
WHISKER_ROWS = []
for idx in range(5):
    prefix = f"interval_{idx}_mask_contact_no_collision"
    WHISKER_ROWS.append([
        (f"{prefix}.csv",              f"W{idx} No Collision"),
        (f"{prefix}_protraction.csv",  f"W{idx} Protraction No Collision"),
        (f"{prefix}_retraction.csv",   f"W{idx} Retraction No Collision"),
    ])


def run_combined(data_dir, contact_dir=None, output_dir=None,
                 pre_ms=50.0, post_ms=100.0, bin_ms=1.0,
                 sampling_rate=30_000, sync_channel=1, units=None):

    digitalin_path = os.path.join(data_dir, "digitalin.dat")
    spikes_path = os.path.join(data_dir, "spikes.csv")
    if contact_dir is None:
        contact_dir = os.path.join(data_dir, "per_whisker_contact")
    if output_dir is None:
        output_dir = os.path.join(data_dir, "contact_psth_no_collision_output")
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

    pre_s = pre_ms / 1000.0
    post_s = post_ms / 1000.0

    # ── Build the grid of (filename, label) cells ───────────────────
    # Row 0: top items, 3 columns
    # Rows 1-5: whiskers, 3 columns each
    n_cols = 3
    n_rows = 1 + len(WHISKER_ROWS)  # 6 rows total

    # Pre-load events for every cell ─────────────────────────────────
    def _load_events(filename):
        path = os.path.join(contact_dir, filename)
        if not os.path.isfile(path):
            print(f"  Warning: {filename} not found, skipping.")
            return None
        intervals_df = load_contact_intervals(path)
        if len(intervals_df) == 0:
            return None
        starts_s = frame_to_seconds(intervals_df["Start"].values,
                                    frame_samples, sampling_rate)
        ends_s = frame_to_seconds(intervals_df["End"].values,
                                  frame_samples, sampling_rate)
        return (starts_s, ends_s)

    # grid[row][col] = (label, events | None)  or  None (empty cell)
    grid = [[None] * n_cols for _ in range(n_rows)]

    # Row 0: top items across 3 columns
    for col, (fname, label) in enumerate(TOP_FILES):
        grid[0][col] = (label, _load_events(fname))

    # Rows 1..4: whisker rows
    for w_idx, row_files in enumerate(WHISKER_ROWS):
        for col, (fname, label) in enumerate(row_files):
            grid[1 + w_idx][col] = (label, _load_events(fname))

    # ── Generate one figure per unit ────────────────────────────────
    total_plots = 0
    for unit in process_units:
        print(f"\n{'='*60}")
        print(f"Creating combined plot for Unit {unit}")
        print(f"{'='*60}")

        unit_spikes = spikes_df.loc[spikes_df["unit"] == unit, "time"].values

        cell_h = 4
        cell_w = 4
        fig = plt.figure(figsize=(cell_w * n_cols, cell_h * n_rows))
        outer_grid = fig.add_gridspec(n_rows, n_cols,
                                      hspace=0.25, wspace=0.3, top=0.97)

        for row in range(n_rows):
            for col in range(n_cols):
                cell = grid[row][col]
                if cell is None:
                    # Empty cell — create hidden axes
                    inner = outer_grid[row, col].subgridspec(2, 1)
                    for s in range(2):
                        ax = fig.add_subplot(inner[s])
                        ax.set_visible(False)
                    continue

                label, events = cell
                inner = outer_grid[row, col].subgridspec(2, 1, hspace=0.15,
                                                          height_ratios=[1, 1])
                ax_psth = fig.add_subplot(inner[0])
                ax_raster = fig.add_subplot(inner[1], sharex=ax_psth)

                if events is None:
                    ax_psth.set_title(f"{label}  (no events)", fontsize=7)
                    ax_raster.set_visible(False)
                    continue

                starts_s, ends_s = events
                trials = align_spikes_to_events(unit_spikes, starts_s,
                                                ends_s, pre_s, post_s)
                sc = sum(len(t["spike_times_ms"]) for t in trials)
                print(f"  {label}: {sc} spikes across {len(trials)} trials")

                draw_psth_raster_on_axes(ax_psth, ax_raster, trials, label,
                                         pre_ms, post_ms, bin_ms)
                ax_raster.set_xlabel("Time (ms)", fontsize=6)
                plt.setp(ax_psth.get_xticklabels(), visible=False)

        session_name = os.path.basename(os.path.normpath(data_dir))
        fig.suptitle(f"Unit {unit}, {session_name}", fontsize=14,
                     fontweight="bold", y=0.995, va="top")

        out_stem = f"unit_{unit}_no_collision_combined"
        for ext in (".png", ".svg"):
            out_path = os.path.join(output_dir, out_stem + ext)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        total_plots += 1
        print(f"  Saved → {out_stem}.png / .svg")

    print(f"\n{'='*60}")
    print(f"Done. {total_plots} combined plot(s) saved to {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined PSTH/raster — no-collision layout.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python contact_psth_no_collision_combined.py ^
      --data_dir "C:\\Users\\wanglab\\Desktop\\Club Like Endings\\102225_2"
        """,
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

    run_combined(
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
