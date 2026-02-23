#!/usr/bin/env python3
"""
PSTH + raster aligned to whisker contact interval onsets, per unit.

Spikes CSV: column 0 = spike time (seconds), column 1 = unit (cluster id).
Intervals CSV: Start, End = frame numbers (kept as-is).
Time alignment: spike time (s) is multiplied by FPS (500) to get spike time in frame
units; relative time for each interval is (spike_frames - Start) / FPS in seconds.

Usage: python psth_raster_whisker_contact.py [interval_csv] [spikes_csv]
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

FPS = 500.0
BIN_S = 0.01   # 10 ms bins
T_PRE = 0.3    # seconds before interval start
T_POST = 2   # seconds after interval start


def load_intervals(path):
    """Load interval CSV. Start/End are frame numbers (kept as-is, no division)."""
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            if not row:
                continue
            start_fr = int(row[0].strip())
            end_fr = int(row[1].strip())
            rows.append({"start_fr": start_fr, "end_fr": end_fr})
    return rows


def load_spikes(path):
    """
    Load spikes CSV.
    Column 0 = spike time in seconds.
    Column 1 = unit (cluster id).
    Column 2 = ignored.
    """
    times = []
    units = []
    with open(path, newline="") as f:
        r = csv.reader(f, skipinitialspace=True)
        for row in r:
            if len(row) < 2:
                continue
            t = float(row[0].strip())
            unit = int(row[1].strip())
            times.append(t)
            units.append(unit)
    return np.array(times), np.array(units)


def psth_for_unit(spike_frames, spike_units, intervals, unit, bin_s=BIN_S, t_pre=T_PRE, t_post=T_POST):
    """PSTH aligned to each interval start. spike_frames = spike time (s) * FPS. Returns bin_centers, rate, n_trials."""
    mask = spike_units == unit
    unit_frames = spike_frames[mask]

    t_edges = np.arange(-t_pre, t_post + 1e-9, bin_s)
    n_bins = len(t_edges) - 1
    bin_centers = (t_edges[:-1] + t_edges[1:]) / 2
    n_trials = len(intervals)
    counts = np.zeros(n_bins)

    for iv in intervals:
        t0_fr = iv["start_fr"]
        for sf in unit_frames:
            rel_fr = sf - t0_fr
            rel_s = rel_fr / FPS
            if -t_pre <= rel_s < t_post:
                idx = np.searchsorted(t_edges, rel_s, side="right") - 1
                idx = np.clip(idx, 0, n_bins - 1)
                counts[idx] += 1

    rate = counts / (n_trials * bin_s)
    return bin_centers, rate, n_trials


def raster_for_unit(spike_frames, spike_units, intervals, unit, t_pre=T_PRE, t_post=T_POST):
    """Relative spike times (s) and trial index. spike_frames = spike time (s) * FPS."""
    mask = spike_units == unit
    unit_frames = spike_frames[mask]

    rel_times = []
    trial_inds = []

    for i, iv in enumerate(intervals):
        t0_fr = iv["start_fr"]
        for sf in unit_frames:
            rel_fr = sf - t0_fr
            rel_s = rel_fr / FPS
            if -t_pre <= rel_s < t_post:
                rel_times.append(rel_s)
                trial_inds.append(i + 1)  # 1-based for plot

    return np.array(rel_times), np.array(trial_inds)


def main():
    base = Path("/Users/martindokholyan/Desktop/Plots/2")
    parser = argparse.ArgumentParser(description="PSTH + raster per unit, aligned to whisker contact onset.")
    parser.add_argument("interval_csv", nargs="?", default=str(base / "interval_2_whisker_contact.csv"),
                        help="CSV with Start,End frame numbers")
    parser.add_argument("spikes_csv", nargs="?", default=str(base / "spikes.csv"),
                        help="CSV: col0=time(s), col1=unit")
    args = parser.parse_args()

    interval_path = Path(args.interval_csv)
    spikes_path = Path(args.spikes_csv)
    out_dir = interval_path.parent

    # Intervals: keep Start/End as frame numbers (no division)
    intervals = load_intervals(interval_path)
    # For raster display, sort intervals by duration (shortest to longest)
    intervals_sorted = sorted(
        intervals,
        key=lambda iv: iv["end_fr"] - iv["start_fr"],
    )
    # Spikes: col0 = time in seconds, col1 = unit. Convert to frame units: multiply by FPS
    spike_times_s, spike_units = load_spikes(spikes_path)
    spike_frames = spike_times_s * FPS
    units = np.unique(spike_units)
    units = np.sort(units)

    for unit in units:
        # PSTH is invariant to trial order, so we can keep original interval order here
        bin_centers, rate, n_trials = psth_for_unit(spike_frames, spike_units, intervals, unit)
        # Raster rows ordered by interval duration via intervals_sorted
        rel_times, trial_inds = raster_for_unit(spike_frames, spike_units, intervals_sorted, unit)

        fig, (ax_psth, ax_raster) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 5.6),
                                                 gridspec_kw=dict(height_ratios=[1, 1.2]))

        # PSTH
        ax_psth.plot(bin_centers, rate, color="steelblue", lw=2)
        ax_psth.axvline(0, color="black", ls="--")
        ax_psth.set_ylabel("Rate (spikes/s)")
        ax_psth.set_title(f"PSTH — Unit {unit} (n={n_trials} intervals)")
        ax_psth.set_xlim(-T_PRE, T_POST)

        # Raster: draw a short vertical tick within each row (trial)
        if len(rel_times) > 0:
            tick_half = 0.4  # half-height of tick within each trial row
            x_seg = []
            y_seg = []
            for t, tr in zip(rel_times, trial_inds):
                x_seg.extend([t, t, np.nan])
                y_seg.extend([tr - tick_half, tr + tick_half, np.nan])
            ax_raster.plot(x_seg, y_seg, color="black", linewidth=0.8)
        ax_raster.axvline(0, color="black", ls="--")
        ax_raster.set_xlabel("Time from contact onset (s)")
        ax_raster.set_ylabel("Trial")
        ax_raster.set_title("Raster")
        ax_raster.set_ylim(0.5, n_trials + 0.5)
        ax_raster.set_xlim(-T_PRE, T_POST)

        plt.tight_layout()
        out_file = out_dir / f"psth_raster_unit_{unit}.png"
        plt.savefig(out_file, dpi=120)
        plt.close()
        print(f"Saved {out_file}")


if __name__ == "__main__":
    main()
