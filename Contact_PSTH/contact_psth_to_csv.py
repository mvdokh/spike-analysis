"""
Contact PSTH to CSV — Export per-millisecond firing rates

For each unit and each contact-interval CSV, computes the PSTH firing rate
(Hz) in 1 ms bins around the contact onset and saves the result to a single
CSV file.

Output CSV columns:
    unit, interval, bin_ms, firing_rate_hz

Usage
-----
    python contact_psth_to_csv.py --data_dir <session_dir>
                                  [--contact_dir <dir>]
                                  [--output_dir <dir>]
                                  [--pre_ms 50] [--post_ms 100]
                                  [--bin_ms 1]
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# Import shared helpers from the single-plot script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_psth import (
    load_frame_sync,
    frame_to_seconds,
    load_contact_intervals,
    load_spikes,
    align_spikes_to_events,
)


def run_csv_export(data_dir, contact_dir=None, output_dir=None,
                   pre_ms=50.0, post_ms=100.0, bin_ms=1.0,
                   sampling_rate=30_000, sync_channel=1, units=None):

    digitalin_path = os.path.join(data_dir, "digitalin.dat")
    spikes_path = os.path.join(data_dir, "spikes.csv")
    if contact_dir is None:
        contact_dir = os.path.join(data_dir, "per_whisker_contact")
    if output_dir is None:
        output_dir = os.path.join(data_dir, "contact_psth_csv_output")
    os.makedirs(output_dir, exist_ok=True)

    for p, label in [(digitalin_path, "digitalin.dat"),
                     (spikes_path, "spikes.csv")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}")
    if not os.path.isdir(contact_dir):
        raise FileNotFoundError(f"Contact directory not found: {contact_dir}")

    # Load sync and spikes
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

    csv_basenames = [os.path.splitext(os.path.basename(f))[0]
                     for f in all_csv_files]
    print(f"Found {len(all_csv_files)} contact interval file(s):")
    for name in csv_basenames:
        print(f"  {name}")

    pre_s = pre_ms / 1000.0
    post_s = post_ms / 1000.0

    # Pre-compute events for each CSV
    csv_events = []
    for csv_path in all_csv_files:
        intervals_df = load_contact_intervals(csv_path)
        if len(intervals_df) == 0:
            csv_events.append(None)
            continue
        starts_s = frame_to_seconds(intervals_df["Start"].values,
                                    frame_samples, sampling_rate)
        ends_s = frame_to_seconds(intervals_df["End"].values,
                                  frame_samples, sampling_rate)
        csv_events.append((starts_s, ends_s))

    # Bin edges
    t_min = -pre_ms
    t_max = post_ms
    bins = np.arange(t_min, t_max + bin_ms, bin_ms)
    centres = (bins[:-1] + bins[1:]) / 2

    # Collect rows
    rows = []

    for unit in process_units:
        unit_spikes = spikes_df.loc[spikes_df["unit"] == unit, "time"].values

        for csv_idx, csv_path in enumerate(all_csv_files):
            label = csv_basenames[csv_idx]

            if csv_events[csv_idx] is None:
                print(f"  Unit {unit} | {label}: no events — skipping")
                continue

            starts_s, ends_s = csv_events[csv_idx]
            trials = align_spikes_to_events(unit_spikes, starts_s, ends_s,
                                            pre_s, post_s)

            if not trials:
                print(f"  Unit {unit} | {label}: no trials — skipping")
                continue

            # Gather all spike times across trials
            all_spikes = []
            for t in trials:
                all_spikes.extend(t["spike_times_ms"])
            all_spikes = np.array(all_spikes)

            # Compute firing rate per bin
            if len(all_spikes) > 0:
                counts, _ = np.histogram(all_spikes, bins=bins)
            else:
                counts = np.zeros(len(centres), dtype=int)

            firing_rate = counts / (len(trials) * bin_ms / 1000.0)

            n_spikes = int(counts.sum())
            print(f"  Unit {unit} | {label}: {n_spikes} spikes across "
                  f"{len(trials)} trials")

            for c, fr in zip(centres, firing_rate):
                rows.append({
                    "unit": unit,
                    "interval": label,
                    "bin_ms": c,
                    "firing_rate_hz": fr,
                })

    # Build and save DataFrame
    result_df = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, "contact_psth_firing_rates.csv")
    result_df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"Done. Saved {len(result_df)} rows to {out_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Export PSTH firing rates (per ms bin) to CSV for each "
                    "unit and contact-interval file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python contact_psth_to_csv.py ^
      --data_dir "C:\\Users\\wanglab\\Desktop\\Club Like Endings\\102225_1"
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

    run_csv_export(
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
