#!/usr/bin/env python3
"""
Basic analysis of spike times in spikes.csv.

Input CSV format (no header expected):
    column 0: spike time (seconds)
    column 1: unit id (integer)
    column 2: channel id (integer)

The script computes:
- overall recording stats
- per-unit statistics (spike count, firing rate, ISI stats, channels)
- per-channel statistics
- identifies the "loudest" units (by firing rate and by spike count)

Usage (from repo root):
    python Unit_Analysis/analyze_spikes.py

You can override the input path with:
    python Unit_Analysis/analyze_spikes.py --spikes_csv path/to/spikes.csv
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class RecordingSummary:
    n_spikes: int
    n_units: int
    n_channels: int
    t_min: float
    t_max: float
    duration: float


@dataclass
class UnitStats:
    unit: int
    n_spikes: int
    firing_rate_hz: float
    mean_isi_s: float
    median_isi_s: float
    isi_cv: float
    refrac_violation_frac: float
    n_channels_seen: int
    primary_channel: int


@dataclass
class ChannelStats:
    channel: int
    n_spikes: int
    firing_rate_hz: float
    n_units: int


def load_spikes(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load spikes CSV.

    Expects three numeric columns: time (s), unit, channel.
    Lines that cannot be parsed are skipped.
    """
    times: List[float] = []
    units: List[int] = []
    channels: List[int] = []

    with path.open(newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                t = float(row[0].strip())
                u = int(row[1].strip())
                ch = int(row[2].strip())
            except ValueError:
                # Skip non-numeric or malformed rows (e.g. header)
                continue
            times.append(t)
            units.append(u)
            channels.append(ch)

    if not times:
        raise ValueError(f"No valid spikes found in {path}")

    times_arr = np.asarray(times, dtype=float)
    units_arr = np.asarray(units, dtype=int)
    channels_arr = np.asarray(channels, dtype=int)

    # Sort by time (just in case)
    order = np.argsort(times_arr)
    return times_arr[order], units_arr[order], channels_arr[order]


def summarize_recording(times: np.ndarray, units: np.ndarray, channels: np.ndarray) -> RecordingSummary:
    t_min = float(times.min())
    t_max = float(times.max())
    duration = max(t_max - t_min, 0.0)
    n_spikes = int(times.size)
    n_units = int(np.unique(units).size)
    n_channels = int(np.unique(channels).size)
    return RecordingSummary(
        n_spikes=n_spikes,
        n_units=n_units,
        n_channels=n_channels,
        t_min=t_min,
        t_max=t_max,
        duration=duration,
    )


def compute_unit_stats(
    times: np.ndarray, units: np.ndarray, channels: np.ndarray, duration: float
) -> Dict[int, UnitStats]:
    if duration <= 0:
        duration = np.nan

    unit_stats: Dict[int, UnitStats] = {}
    unique_units = np.unique(units)

    for u in unique_units:
        mask = units == u
        times_u = times[mask]
        chans_u = channels[mask]
        n_spikes = int(times_u.size)

        firing_rate_hz = float(n_spikes / duration) if duration and not np.isnan(duration) else float("nan")

        if times_u.size >= 2:
            isi = np.diff(times_u)
            mean_isi = float(np.mean(isi))
            median_isi = float(np.median(isi))
            isi_std = float(np.std(isi, ddof=1)) if isi.size > 1 else 0.0
            isi_cv = float(isi_std / mean_isi) if mean_isi > 0 else float("nan")
            # Fraction of ISIs shorter than 2 ms (0.002 s) as a simple quality metric
            refrac_violation_frac = float(np.mean(isi < 0.002))
        else:
            mean_isi = float("nan")
            median_isi = float("nan")
            isi_cv = float("nan")
            refrac_violation_frac = float("nan")

        unique_chans, counts = np.unique(chans_u, return_counts=True)
        primary_idx = int(np.argmax(counts))
        primary_channel = int(unique_chans[primary_idx])

        unit_stats[int(u)] = UnitStats(
            unit=int(u),
            n_spikes=n_spikes,
            firing_rate_hz=firing_rate_hz,
            mean_isi_s=mean_isi,
            median_isi_s=median_isi,
            isi_cv=isi_cv,
            refrac_violation_frac=refrac_violation_frac,
            n_channels_seen=int(unique_chans.size),
            primary_channel=primary_channel,
        )

    return unit_stats


def compute_channel_stats(
    times: np.ndarray, units: np.ndarray, channels: np.ndarray, duration: float
) -> Dict[int, ChannelStats]:
    if duration <= 0:
        duration = np.nan

    chan_stats: Dict[int, ChannelStats] = {}
    unique_chans = np.unique(channels)

    for ch in unique_chans:
        mask = channels == ch
        times_ch = times[mask]
        units_ch = units[mask]
        n_spikes = int(times_ch.size)
        firing_rate_hz = float(n_spikes / duration) if duration and not np.isnan(duration) else float("nan")
        n_units = int(np.unique(units_ch).size)
        chan_stats[int(ch)] = ChannelStats(
            channel=int(ch),
            n_spikes=n_spikes,
            firing_rate_hz=firing_rate_hz,
            n_units=n_units,
        )

    return chan_stats


def print_summary(
    summary: RecordingSummary,
    unit_stats: Dict[int, UnitStats],
    channel_stats: Dict[int, ChannelStats],
    top_n: int = 10,
) -> None:
    print("=" * 80)
    print("Recording summary")
    print("=" * 80)
    print(f"  Total spikes   : {summary.n_spikes}")
    print(f"  Units (clusters): {summary.n_units}")
    print(f"  Channels       : {summary.n_channels}")
    print(f"  Start time (s) : {summary.t_min:.6f}")
    print(f"  End time (s)   : {summary.t_max:.6f}")
    print(f"  Duration (s)   : {summary.duration:.3f}")
    print()

    # Convert dict to list for sorting
    units_list = list(unit_stats.values())

    # Loudest units by firing rate
    units_by_rate = sorted(
        units_list, key=lambda s: (-(s.firing_rate_hz if np.isfinite(s.firing_rate_hz) else -1.0), -s.n_spikes)
    )
    # Loudest units by spike count
    units_by_count = sorted(units_list, key=lambda s: (-s.n_spikes, -s.firing_rate_hz))

    print("=" * 80)
    print(f"Top {min(top_n, len(units_by_rate))} units by firing rate (\"loudest\" by rate)")
    print("=" * 80)
    print("  unit  spikes   rate_Hz  median_ISI_s  ISI_CV  refrac<2ms  primary_ch  n_chans")
    for s in units_by_rate[:top_n]:
        print(
            f"{s.unit:6d}  {s.n_spikes:6d}  {s.firing_rate_hz:8.3f}  "
            f"{s.median_isi_s:11.5f}  {s.isi_cv:6.3f}  {s.refrac_violation_frac:9.3f}  "
            f"{s.primary_channel:10d}  {s.n_channels_seen:7d}"
        )
    print()

    print("=" * 80)
    print(f"Top {min(top_n, len(units_by_count))} units by spike count (\"loudest\" by count)")
    print("=" * 80)
    print("  unit  spikes   rate_Hz  median_ISI_s  ISI_CV  refrac<2ms  primary_ch  n_chans")
    for s in units_by_count[:top_n]:
        print(
            f"{s.unit:6d}  {s.n_spikes:6d}  {s.firing_rate_hz:8.3f}  "
            f"{s.median_isi_s:11.5f}  {s.isi_cv:6.3f}  {s.refrac_violation_frac:9.3f}  "
            f"{s.primary_channel:10d}  {s.n_channels_seen:7d}"
        )
    print()

    # Per-channel overview
    chans_list = sorted(channel_stats.values(), key=lambda c: c.channel)
    print("=" * 80)
    print("Per-channel statistics")
    print("=" * 80)
    print("  ch   spikes   rate_Hz   n_units")
    for c in chans_list:
        print(f"{c.channel:4d}  {c.n_spikes:7d}  {c.firing_rate_hz:8.3f}  {c.n_units:7d}")
    print()

    if units_by_rate:
        loudest_by_rate = units_by_rate[0]
        print("=" * 80)
        print("Most \"loud\" unit (by firing rate)")
        print("=" * 80)
        print(
            f"  unit {loudest_by_rate.unit} with {loudest_by_rate.firing_rate_hz:.3f} spikes/s "
            f"({loudest_by_rate.n_spikes} spikes over {summary.duration:.3f} s), "
            f"primary channel {loudest_by_rate.primary_channel}"
        )
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze spikes CSV (time, unit, channel) and report per-unit and per-channel statistics. "
            "The \"loudest\" unit is defined as the one with the highest firing rate."
        )
    )
    default_spikes = Path("Unit_Analysis") / "spikes.csv"
    parser.add_argument(
        "--spikes_csv",
        type=str,
        default=str(default_spikes),
        help=f"Path to spikes CSV (default: {default_spikes})",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many top units to show in the summary tables",
    )
    args = parser.parse_args()

    spikes_path = Path(args.spikes_csv)
    if not spikes_path.is_file():
        raise FileNotFoundError(f"Spikes CSV not found: {spikes_path}")

    times, units, channels = load_spikes(spikes_path)
    summary = summarize_recording(times, units, channels)
    unit_stats = compute_unit_stats(times, units, channels, summary.duration)
    channel_stats = compute_channel_stats(times, units, channels, summary.duration)

    print_summary(summary, unit_stats, channel_stats, top_n=args.top)


if __name__ == "__main__":
    main()

