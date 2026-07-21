"""
Contact PSTH Combined (End-Aligned) — Spikes aligned to interval END

Identical layout to contact_psth_combined.py, but every spike time is
referenced to the contact-interval *end* (offset) rather than the start
(onset).  t = 0 marks the moment the whisker breaks contact.

For each unit, produces a single tall figure where each row is a
(PSTH, Raster) pair for one contact-interval CSV file.

Usage
-----
    python contact_psth_end_aligned.py --data_dir <session_dir>
                                       [--contact_dir <dir>]
                                       [--output_dir <dir>]
                                       [--pre_ms 100] [--post_ms 50]
                                       [--bin_ms 1]
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import shared helpers
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_psth import (
    load_frame_sync,
    frame_to_seconds,
    load_contact_intervals,
    load_spikes,
)


# ──────────────────────────────────────────────────────────────────────────────
# End-aligned spike extraction
# ──────────────────────────────────────────────────────────────────────────────

def align_spikes_to_ends(spike_times, event_starts, event_ends,
                         pre_s, post_s):
    """
    For each event, collect spike times relative to event *end* (offset).

    Parameters
    ----------
    spike_times : 1-D array  – seconds (one unit)
    event_starts, event_ends : 1-D arrays – seconds
    pre_s  : float – seconds before event end to include
    post_s : float – seconds after event end to include

    Returns
    -------
    trials : list[dict]
        'spike_times_ms' – ms relative to event end (negative = before end)
        'duration_ms'    – contact duration in ms
    """
    trials = []
    for start, end in zip(event_starts, event_ends):
        window_lo = end - pre_s
        window_hi = end + post_s
        mask = (spike_times >= window_lo) & (spike_times <= window_hi)
        relative_ms = (spike_times[mask] - end) * 1000.0  # ms from offset
        duration_ms = (end - start) * 1000.0
        trials.append({
            "spike_times_ms": relative_ms.tolist(),
            "duration_ms": duration_ms,
        })
    return trials


# ──────────────────────────────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_psth_raster_on_axes(ax_psth, ax_raster, trials, label,
                              pre_ms, post_ms, bin_ms):
    """Draw a PSTH + raster pair onto the provided axes (end-aligned)."""
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

    # PSTH
    if len(all_spikes) > 0:
        counts, edges = np.histogram(all_spikes, bins=bins)
        centres = (edges[:-1] + edges[1:]) / 2
        firing_rate = counts / (len(trials) * bin_ms / 1000.0)
        ax_psth.bar(centres, firing_rate, width=bin_ms,
                    color="black", edgecolor="black", linewidth=0.3)

    ax_psth.axvline(0, color="red", ls="--", lw=1.0, alpha=0.8)
    ax_psth.set_ylabel("FR (Hz)", fontsize=7)
    ax_psth.set_title(f"{label}  |  {len(trials)} trials  |  "
                      f"avg dur {avg_dur:.1f} ms", fontsize=8)
    ax_psth.set_xlim(t_min, t_max)
    ax_psth.tick_params(labelsize=6)

    # Raster (sorted by duration, shortest at bottom)
    sorted_trials = sorted(trials, key=lambda t: t["duration_ms"])
    for idx, trial in enumerate(sorted_trials):
        st = trial["spike_times_ms"]
        if st:
            ax_raster.scatter(st, [idx] * len(st),
                              s=0.5, color="black", alpha=0.8, marker="s")

    # Onset curve: onset is at -(duration) relative to end
    onset_times = [-t["duration_ms"] for t in sorted_trials]
    ax_raster.plot(onset_times, range(len(sorted_trials)),
                   color="blue", linewidth=0.8, alpha=0.7)

    ax_raster.axvline(0, color="red", ls="--", lw=1.0, alpha=0.8)
    ax_raster.set_ylabel("Trial", fontsize=7)
    ax_raster.set_ylim(-0.5, len(trials) - 0.5)
    ax_raster.set_xlim(t_min, t_max)
    ax_raster.tick_params(labelsize=6)


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_combined_end(data_dir, contact_dir=None, output_dir=None,
                     pre_ms=100.0, post_ms=50.0, bin_ms=1.0,
                     sampling_rate=30_000, sync_channel=1, units=None):

    digitalin_path = os.path.join(data_dir, "digitalin.dat")
    spikes_path = os.path.join(data_dir, "spikes.csv")
    if contact_dir is None:
        contact_dir = os.path.join(data_dir, "per_whisker_contact")
    if output_dir is None:
        output_dir = os.path.join(data_dir, "contact_psth_end_aligned_output")
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

    # Discover contact CSVs; contact_intervals.csv first (top center)
    all_csv_files = sorted(glob.glob(os.path.join(contact_dir, "*.csv")))
    contact_intervals_file = None
    other_files = []
    for f in all_csv_files:
        if os.path.basename(f).lower() == "contact_intervals.csv":
            contact_intervals_file = f
        else:
            other_files.append(f)
    contact_files = []
    if contact_intervals_file is not None:
        contact_files.append(contact_intervals_file)
    contact_files.extend(other_files)
    if not contact_files:
        raise FileNotFoundError(f"No CSV files found in {contact_dir}")

    n_csvs = len(contact_files)
    csv_basenames = [os.path.splitext(os.path.basename(f))[0]
                     for f in contact_files]
    print(f"Found {n_csvs} contact interval file(s):")
    for name in csv_basenames:
        print(f"  {name}")

    pre_s = pre_ms / 1000.0
    post_s = post_ms / 1000.0

    # Pre-compute events for each CSV
    csv_events = []
    for csv_path in contact_files:
        intervals_df = load_contact_intervals(csv_path)
        if len(intervals_df) == 0:
            csv_events.append(None)
            continue
        starts_s = frame_to_seconds(intervals_df["Start"].values,
                                    frame_samples, sampling_rate)
        ends_s = frame_to_seconds(intervals_df["End"].values,
                                  frame_samples, sampling_rate)
        csv_events.append((starts_s, ends_s))

    # Layout: first CSV centered alone in row 0, remaining in rows of 3
    has_top = contact_intervals_file is not None
    n_cols = 3
    n_remaining = n_csvs - (1 if has_top else 0)
    n_body_rows = int(np.ceil(n_remaining / n_cols)) if n_remaining > 0 else 0
    n_total_rows = (1 if has_top else 0) + n_body_rows
    total_plots = 0

    for unit in process_units:
        print(f"\n{'='*60}")
        print(f"Creating end-aligned combined plot for Unit {unit}")
        print(f"{'='*60}")

        unit_spikes = spikes_df.loc[spikes_df["unit"] == unit, "time"].values

        cell_h = 5
        cell_w = 5
        fig = plt.figure(figsize=(cell_w * n_cols, cell_h * n_total_rows))

        outer_grid = fig.add_gridspec(n_total_rows, n_cols,
                                      hspace=0.45, wspace=0.3,
                                      top=0.97)

        def _draw_cell(grid_slot, csv_idx):
            label = csv_basenames[csv_idx]
            inner = grid_slot.subgridspec(2, 1, hspace=0.15,
                                          height_ratios=[1, 1])
            ax_psth = fig.add_subplot(inner[0])
            ax_raster = fig.add_subplot(inner[1], sharex=ax_psth)

            if csv_events[csv_idx] is None:
                ax_psth.set_title(f"{label}  (no events)", fontsize=7)
                ax_raster.set_visible(False)
                return

            starts_s, ends_s = csv_events[csv_idx]
            trials = align_spikes_to_ends(unit_spikes, starts_s, ends_s,
                                          pre_s, post_s)
            sc = sum(len(t["spike_times_ms"]) for t in trials)
            print(f"  {label}: {sc} spikes across {len(trials)} trials")

            draw_psth_raster_on_axes(ax_psth, ax_raster, trials, label,
                                     pre_ms, post_ms, bin_ms)
            ax_raster.set_xlabel("Time from contact end (ms)", fontsize=6)
            plt.setp(ax_psth.get_xticklabels(), visible=False)

        # Row 0: contact_intervals centered (column 1 of 0,1,2)
        if has_top:
            _draw_cell(outer_grid[0, 1], 0)
            for hide_col in [0, 2]:
                inner = outer_grid[0, hide_col].subgridspec(2, 1)
                for sub in range(2):
                    ax = fig.add_subplot(inner[sub])
                    ax.set_visible(False)

        # Remaining CSVs in body rows
        for i in range(n_remaining):
            actual_csv_idx = (1 if has_top else 0) + i
            row = (1 if has_top else 0) + i // n_cols
            col = i % n_cols
            _draw_cell(outer_grid[row, col], actual_csv_idx)

        # Hide unused trailing cells
        used_in_last_body = n_remaining % n_cols
        if used_in_last_body != 0:
            for col in range(used_in_last_body, n_cols):
                row = n_total_rows - 1
                inner = outer_grid[row, col].subgridspec(2, 1)
                for sub in range(2):
                    ax = fig.add_subplot(inner[sub])
                    ax.set_visible(False)

        fig.suptitle(f"Unit {unit}  (end-aligned)", fontsize=14,
                     fontweight="bold", y=0.995, va="top")

        out_name = f"unit_{unit}_end_aligned.png"
        out_path = os.path.join(output_dir, out_name)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        total_plots += 1
        print(f"  Saved → {out_name}")

    print(f"\n{'='*60}")
    print(f"Done. {total_plots} end-aligned plot(s) saved to {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate end-aligned combined PSTH/raster images — "
                    "spikes referenced to contact offset (interval end).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python contact_psth_end_aligned.py ^
      --data_dir "C:\\Users\\wanglab\\Desktop\\Club Like Endings\\102225_2"
        """,
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--contact_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--pre_ms", type=float, default=100,
                        help="ms before contact end (default: 100)")
    parser.add_argument("--post_ms", type=float, default=50,
                        help="ms after contact end (default: 50)")
    parser.add_argument("--bin_ms", type=float, default=1)
    parser.add_argument("--sampling_rate", type=int, default=30000)
    parser.add_argument("--sync_channel", type=int, default=1)
    parser.add_argument("--units", type=int, nargs="*", default=None)

    args = parser.parse_args()

    run_combined_end(
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
