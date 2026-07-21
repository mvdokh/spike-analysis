"""
Contact PSTH Pipeline

Generates raster plots and PSTHs for each spike unit aligned to whisker contact
events. Contact events are specified as video frame intervals (Start, End) and
are converted to neural recording time via the digitalin.dat TTL sync signal.

Time alignment
--------------
- digitalin.dat channel 1 rising edges mark each video frame in the master
  30 kHz clock.  Frame N → sample index of the Nth rising edge.
- Spike times (column 0 of spikes.csv) are in seconds on the same master clock.
- Contact intervals (Start, End in frames) are mapped to seconds through the
  frame→sample lookup table derived from channel 1.

Usage
-----
    python contact_psth.py --data_dir <path_to_session>
                           [--contact_dir <path_to_contact_csvs>]
                           [--output_dir <path_to_output>]
                           [--pre_ms 50] [--post_ms 50]
                           [--bin_ms 5] [--smooth 5]
                           [--sampling_rate 30000]
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
from scipy.ndimage import uniform_filter1d

# ---------------------------------------------------------------------------
# Add the Spike PSTH Pipeline directory so we can reuse binary_data / ttls
# ---------------------------------------------------------------------------
PIPELINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Spike PSTH Pipeline",
)
sys.path.insert(0, PIPELINE_DIR)

from binary_data import get_digital
from ttls import find_ttls_on_single_channel_16bit


# ========================== Data Loading ====================================

def load_frame_sync(digitalin_path: str, channel: int = 1,
                    sampling_rate: int = 30_000) -> np.ndarray:
    """
    Load digitalin.dat and extract rising-edge sample indices on *channel*.
    These correspond 1-to-1 with video frames.

    Returns
    -------
    frame_samples : np.ndarray of int
        Sample index of each video frame in the master 30 kHz clock.
    """
    print(f"Loading digitalin.dat from: {digitalin_path}")
    digital_inputs = get_digital(
        filepath=digitalin_path,
        header_offset_in_bytes=0,
        single_sample_size_in_bytes=2,
    )

    # Extract only rising edges — we don't need paired on/off for frame sync.
    ttl_bool = find_ttls_on_single_channel_16bit(digital_inputs, channel)
    ttl_diff = np.ediff1d(ttl_bool.astype(int))
    onsets = np.where(ttl_diff > 0)[0] + 1  # low→high transitions

    print(f"  Found {len(onsets)} frame-sync rising edges on channel {channel}")
    return onsets


def frame_to_seconds(frame_indices: np.ndarray, frame_samples: np.ndarray,
                     sampling_rate: int = 30_000) -> np.ndarray:
    """
    Convert video frame numbers to time in seconds using the sync lookup.

    Parameters
    ----------
    frame_indices : array-like of int
        Frame numbers from the contact CSV.
    frame_samples : np.ndarray
        Sample indices for each frame (from load_frame_sync).
    sampling_rate : int
        Master clock rate.

    Returns
    -------
    times : np.ndarray of float
        Time in seconds for each frame index.
    """
    frame_indices = np.asarray(frame_indices, dtype=int)
    max_frame = len(frame_samples) - 1

    # Clip to valid range and warn if any are out of bounds
    oob = frame_indices > max_frame
    if oob.any():
        print(f"  WARNING: {oob.sum()} frame indices exceed max synced frame "
              f"({max_frame}). Clipping.")
        frame_indices = np.clip(frame_indices, 0, max_frame)

    return frame_samples[frame_indices] / sampling_rate


def load_contact_intervals(csv_path: str) -> pd.DataFrame:
    """
    Load a single contact-interval CSV (columns: Start, End in frames).
    """
    df = pd.read_csv(csv_path)
    # Normalise column names
    df.columns = df.columns.str.strip()
    if "Start" not in df.columns or "End" not in df.columns:
        raise ValueError(f"Contact CSV must have 'Start' and 'End' columns. "
                         f"Found: {list(df.columns)}")
    return df


def load_spikes(spikes_path: str) -> pd.DataFrame:
    """
    Load spikes.csv (3 columns: time_s, unit, <ignored>).
    """
    df = pd.read_csv(spikes_path, header=None)
    # Strip whitespace from values that might be read as strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    df.columns = ["time", "unit", "extra"]
    df["time"] = df["time"].astype(float)
    df["unit"] = df["unit"].astype(int)
    print(f"Loaded {len(df)} spikes across {df['unit'].nunique()} units: "
          f"{sorted(df['unit'].unique())}")
    return df


# ========================== PSTH Logic ======================================

def align_spikes_to_events(spike_times: np.ndarray,
                           event_starts: np.ndarray,
                           event_ends: np.ndarray,
                           pre_s: float, post_s: float):
    """
    For each event, collect spike times relative to event onset.

    Parameters
    ----------
    spike_times : 1-D array
        Spike times in seconds (for one unit).
    event_starts : 1-D array
        Event onset times in seconds.
    event_ends : 1-D array
        Event offset times in seconds.
    pre_s : float
        Seconds before event onset to include.
    post_s : float
        Seconds after event offset to include.

    Returns
    -------
    trials : list[dict]
        One entry per event with keys:
            'spike_times_ms' : list of float – spike times relative to onset (ms)
            'duration_ms'    : float – event duration in ms
    """
    trials = []
    for start, end in zip(event_starts, event_ends):
        window_lo = start - pre_s
        window_hi = start + post_s
        mask = (spike_times >= window_lo) & (spike_times <= window_hi)
        relative_ms = (spike_times[mask] - start) * 1000.0  # ms from onset
        duration_ms = (end - start) * 1000.0
        trials.append({
            "spike_times_ms": relative_ms.tolist(),
            "duration_ms": duration_ms,
        })
    return trials


def create_psth_raster(trials, unit, event_label,
                       pre_ms: float, post_ms: float,
                       bin_ms: float = 5.0, smooth: int = 0):
    """
    Create a combined PSTH + raster figure.

    Parameters
    ----------
    trials : list[dict]
        Output of align_spikes_to_events.
    unit : int
        Unit identifier for title.
    event_label : str
        Label for the event file (e.g. filename).
    pre_ms, post_ms : float
        Window around each event.
    bin_ms : float
        Histogram bin width in ms.
    smooth : int
        Width (in bins) for uniform smoothing (0 = no smoothing).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if not trials:
        return None

    avg_dur = np.mean([t["duration_ms"] for t in trials])

    # Time axis limits (relative to onset)
    t_min = -pre_ms
    t_max = post_ms

    bins = np.arange(t_min, t_max + bin_ms, bin_ms)

    # Collect all relative spike times
    all_spikes = []
    for t in trials:
        all_spikes.extend(t["spike_times_ms"])
    all_spikes = np.array(all_spikes)

    # ---- Figure -----------------------------------------------------------
    fig, (ax_psth, ax_raster) = plt.subplots(
        2, 1, figsize=(8, 8),
        gridspec_kw={"height_ratios": [1, 1]},
        sharex=True,
    )

    # PSTH -------------------------------------------------------------------
    if len(all_spikes) > 0:
        counts, edges = np.histogram(all_spikes, bins=bins)
        centres = (edges[:-1] + edges[1:]) / 2
        firing_rate = counts / (len(trials) * bin_ms / 1000.0)  # Hz

        ax_psth.bar(centres, firing_rate, width=bin_ms,
                    color="black", edgecolor="black", linewidth=0.3)

    # Mark event onset
    ax_psth.axvline(0, color="black", ls="--", alpha=0.7)
    ax_psth.set_ylabel("Firing Rate (Hz)")
    ax_psth.set_title(
        f"PSTH — Unit {unit} | {event_label} | "
        f"Trials: {len(trials)} | Avg dur: {avg_dur:.1f} ms"
    )
    # Raster (sorted by duration, shortest at bottom) -------------------------
    sorted_trials = sorted(trials, key=lambda t: t["duration_ms"])
    for trial_idx, trial in enumerate(sorted_trials):
        st = trial["spike_times_ms"]
        if st:
            ax_raster.scatter(st, [trial_idx] * len(st),
                              s=1, color="black", alpha=0.8, marker="s")

    # Offset line — each trial's duration plotted as a curve
    offset_times = [t["duration_ms"] for t in sorted_trials]
    ax_raster.plot(offset_times, range(len(sorted_trials)),
                   color="red", linewidth=1, alpha=0.7, label="offset")

    ax_raster.axvline(0, color="black", ls="--", alpha=0.7)
    ax_raster.set_xlabel("Time from contact onset (ms)")
    ax_raster.set_ylabel("Trial")
    ax_raster.set_title(f"Raster — Unit {unit}")
    ax_raster.set_ylim(-0.5, len(trials) - 0.5)
    ax_raster.set_xlim(t_min, t_max)

    plt.tight_layout()
    return fig


# ========================== Main Pipeline ===================================

def run_pipeline(data_dir: str,
                 contact_dir: str | None = None,
                 output_dir: str | None = None,
                 pre_ms: float = 50.0,
                 post_ms: float = 100.0,
                 bin_ms: float = 1.0,
                 smooth: int = 5,
                 sampling_rate: int = 30_000,
                 sync_channel: int = 1,
                 units: list[int] | None = None):
    """
    End-to-end pipeline.

    Parameters
    ----------
    data_dir : str
        Session directory containing spikes.csv and digitalin.dat.
    contact_dir : str or None
        Directory with interval_*_mask_contact.csv files.
        Defaults to <data_dir>/per_whisker_contact.
    output_dir : str or None
        Where to save figures. Defaults to <data_dir>/contact_psth_output.
    pre_ms, post_ms : float
        Window (ms) before/after each event.
    bin_ms : float
        PSTH bin width (ms).
    smooth : int
        Smoothing kernel width in bins.
    sampling_rate : int
        Master clock sample rate.
    sync_channel : int
        digitalin.dat channel carrying video-frame sync TTLs.
    units : list[int] or None
        Restrict analysis to these units (None = all).
    """

    # ---- Resolve paths ----
    digitalin_path = os.path.join(data_dir, "digitalin.dat")
    spikes_path = os.path.join(data_dir, "spikes.csv")
    if contact_dir is None:
        contact_dir = os.path.join(data_dir, "per_whisker_contact")
    if output_dir is None:
        output_dir = os.path.join(data_dir, "contact_psth_output")
    os.makedirs(output_dir, exist_ok=True)

    # Validate required files
    for p, label in [(digitalin_path, "digitalin.dat"),
                     (spikes_path, "spikes.csv")]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required file not found: {p}  ({label})")
    if not os.path.isdir(contact_dir):
        raise FileNotFoundError(f"Contact directory not found: {contact_dir}")

    # ---- Load shared data ----
    frame_samples = load_frame_sync(digitalin_path, channel=sync_channel,
                                    sampling_rate=sampling_rate)
    spikes_df = load_spikes(spikes_path)

    # Determine units to process
    available_units = sorted(spikes_df["unit"].unique())
    if units is not None:
        process_units = [u for u in units if u in available_units]
        if len(process_units) < len(units):
            missing = set(units) - set(process_units)
            print(f"WARNING: requested units {missing} not found in spikes file")
    else:
        process_units = available_units
    print(f"Units to process: {process_units}")

    # ---- Discover contact CSVs ----
    contact_files = sorted(glob.glob(os.path.join(contact_dir, "*.csv")))
    if not contact_files:
        raise FileNotFoundError(f"No CSV files found in {contact_dir}")
    print(f"Found {len(contact_files)} contact interval file(s):")
    for f in contact_files:
        print(f"  {os.path.basename(f)}")

    pre_s = pre_ms / 1000.0
    post_s = post_ms / 1000.0

    # ---- Process each contact file × unit ----
    total_plots = 0
    for csv_path in contact_files:
        basename = os.path.splitext(os.path.basename(csv_path))[0]
        print(f"\n{'='*60}")
        print(f"Processing: {basename}")
        print(f"{'='*60}")

        intervals_df = load_contact_intervals(csv_path)
        n_events = len(intervals_df)
        if n_events == 0:
            print("  No events — skipping.")
            continue

        # Convert frames → seconds
        event_starts_s = frame_to_seconds(
            intervals_df["Start"].values, frame_samples, sampling_rate
        )
        event_ends_s = frame_to_seconds(
            intervals_df["End"].values, frame_samples, sampling_rate
        )

        durations_ms = (event_ends_s - event_starts_s) * 1000
        print(f"  {n_events} events | duration: "
              f"{np.mean(durations_ms):.1f} ± {np.std(durations_ms):.1f} ms  "
              f"(range {np.min(durations_ms):.1f}–{np.max(durations_ms):.1f} ms)")

        for unit in process_units:
            unit_spikes = spikes_df.loc[spikes_df["unit"] == unit, "time"].values

            trials = align_spikes_to_events(
                unit_spikes, event_starts_s, event_ends_s, pre_s, post_s
            )

            total_spikes = sum(len(t["spike_times_ms"]) for t in trials)
            print(f"  Unit {unit}: {total_spikes} spikes across {len(trials)} trials")

            fig = create_psth_raster(
                trials, unit, basename,
                pre_ms=pre_ms, post_ms=post_ms,
                bin_ms=bin_ms, smooth=smooth,
            )
            if fig is None:
                continue

            out_name = f"{basename}_unit_{unit}.png"
            out_path = os.path.join(output_dir, out_name)
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            total_plots += 1
            print(f"    Saved → {out_name}")

    print(f"\n{'='*60}")
    print(f"Done. {total_plots} plot(s) saved to {output_dir}")
    print(f"{'='*60}")


# ========================== CLI =============================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate PSTH & raster plots for whisker-contact events "
                    "aligned to neural spike data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python contact_psth.py ^
      --data_dir "C:\\Users\\wanglab\\Desktop\\Club Like Endings\\102225_1" ^
      --pre_ms 50 --post_ms 100 --bin_ms 5 --smooth 5
        """,
    )
    parser.add_argument("--data_dir", required=True,
                        help="Session directory with spikes.csv and digitalin.dat")
    parser.add_argument("--contact_dir", default=None,
                        help="Directory with interval_*_mask_contact.csv files "
                             "(default: <data_dir>/per_whisker_contact)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for plots "
                             "(default: <data_dir>/contact_psth_output)")
    parser.add_argument("--pre_ms", type=float, default=50,
                        help="Window before event onset in ms (default: 50)")
    parser.add_argument("--post_ms", type=float, default=100,
                        help="Window after event onset in ms (default: 100)")
    parser.add_argument("--bin_ms", type=float, default=1,
                        help="PSTH bin width in ms (default: 1)")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Smoothing kernel width in bins, 0=none (default: 5)")
    parser.add_argument("--sampling_rate", type=int, default=30000,
                        help="Master clock sample rate in Hz (default: 30000)")
    parser.add_argument("--sync_channel", type=int, default=1,
                        help="digitalin.dat channel with frame-sync TTLs "
                             "(default: 1)")
    parser.add_argument("--units", type=int, nargs="*", default=None,
                        help="Restrict to specific unit IDs (default: all)")

    args = parser.parse_args()

    run_pipeline(
        data_dir=args.data_dir,
        contact_dir=args.contact_dir,
        output_dir=args.output_dir,
        pre_ms=args.pre_ms,
        post_ms=args.post_ms,
        bin_ms=args.bin_ms,
        smooth=args.smooth,
        sampling_rate=args.sampling_rate,
        sync_channel=args.sync_channel,
        units=args.units,
    )


if __name__ == "__main__":
    main()
