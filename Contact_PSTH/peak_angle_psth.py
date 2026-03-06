"""
Peak-Aligned Angle Traces + Spike PSTH
=======================================

For each peak event (a single frame number), extracts a window of angle data
and spike times, then generates one figure per unit with three panels:

    [Top]    All angle traces overlaid (min-max normalized), aligned to peak.
             Mean trace drawn in bold.  Mimics the "normalized and superimposed
             whisker position traces" panel from the literature.
    [Middle] Spike-time raster, one row per trial.
    [Bottom] Mean firing rate (PSTH) histogram.

The frame → seconds conversion uses the digitalin.dat TTL sync signal
(same approach as contact_psth.py).

Usage
-----
    python peak_angle_psth.py \\
        --session_dir  "C:/path/to/102725_1" \\
        --events_csv   "C:/path/to/2_line_angle_IQR_events_in_air.csv" \\
        --angle_csv    "C:/path/to/2_line_angle_IQR_excluded.csv" \\
        [--output_dir  "C:/path/to/output"] \\
        [--pre_frames  100] [--post_frames 100] \\
        [--bin_ms 5] [--smooth 3] \\
        [--sampling_rate 30000] [--sync_channel 1] \\
        [--units 1 5 9]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

# ── Reuse helpers from contact_psth.py ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_psth import load_frame_sync, frame_to_seconds, load_spikes


# ═══════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_peak_events(csv_path: str) -> np.ndarray:
    """Load a CSV with an 'Event' column (frame numbers). Returns int array."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    col = next((c for c in df.columns if c.lower() == "event"), None)
    if col is None:
        raise ValueError(f"No 'Event' column found in {csv_path}. "
                         f"Columns: {list(df.columns)}")
    return df[col].dropna().astype(int).values


def load_angle_series(csv_path: str) -> pd.Series:
    """
    Load the angle CSV (Time, Data).
    Returns a Series indexed by *integer* frame number.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["Time"] = pd.to_numeric(df["Time"]).astype(int)
    df["Data"] = pd.to_numeric(df["Data"]).astype(float)
    return df.set_index("Time")["Data"]


# ═══════════════════════════════════════════════════════════════════════════════
#  Alignment Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def extract_angle_traces(angle_series: pd.Series,
                         event_frames: np.ndarray,
                         pre: int, post: int) -> np.ndarray:
    """
    Extract angle windows centred on each event frame.

    Returns
    -------
    traces : ndarray, shape (n_events, pre + post + 1)
        NaN where angle data is absent.
    """
    width = pre + post + 1
    traces = np.full((len(event_frames), width), np.nan)
    for i, ev in enumerate(event_frames):
        needed = np.arange(ev - pre, ev + post + 1, dtype=int)
        traces[i] = angle_series.reindex(needed).values
    return traces


def normalize_traces(traces: np.ndarray) -> np.ndarray:
    """
    Min-max normalize each trace independently to [0, 1].
    Traces that are entirely NaN or constant are left as-is.
    """
    out = traces.copy()
    for i, trace in enumerate(traces):
        valid = trace[~np.isnan(trace)]
        if len(valid) < 2:
            continue
        lo, hi = valid.min(), valid.max()
        if hi > lo:
            out[i] = (trace - lo) / (hi - lo)
    return out


def align_spikes_to_peaks(spike_times: np.ndarray,
                           event_times_s: np.ndarray,
                           pre_s: float,
                           post_s: float) -> list:
    """
    For each peak event, collect spike times relative to the event in ms.

    Returns
    -------
    trials : list of ndarray
        One array per event containing relative spike times (ms).
    """
    trials = []
    for ev_s in event_times_s:
        mask = (spike_times >= ev_s - pre_s) & (spike_times <= ev_s + post_s)
        trials.append((spike_times[mask] - ev_s) * 1000.0)
    return trials


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_unit_figure(unit: int,
                     traces_norm: np.ndarray,
                     trials: list,
                     t_ms: np.ndarray,
                     pre_ms: float,
                     post_ms: float,
                     bin_ms: float,
                     output_dir: str,
                     smooth_bins: int = 3):
    """
    Draw three-panel figure for one unit and save to output_dir.

        Panel 1 (top)    — overlaid normalised angle traces + bold mean
        Panel 2 (middle) — spike raster
        Panel 3 (bottom) — PSTH (mean firing rate, Hz)
    """
    n_events = len(trials)

    fig = plt.figure(figsize=(9, 11))
    fig.subplots_adjust(hspace=0.08, left=0.13, right=0.96,
                        top=0.94, bottom=0.07)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 2])

    ax_traces = fig.add_subplot(gs[0])
    ax_raster  = fig.add_subplot(gs[1], sharex=ax_traces)
    ax_psth    = fig.add_subplot(gs[2], sharex=ax_traces)

    # ── Panel 1: Normalised angle traces ─────────────────────────────────────
    alpha = max(0.08, min(0.35, 15.0 / max(n_events, 1)))
    for trace in traces_norm:
        ax_traces.plot(t_ms, trace, color="steelblue", lw=0.5, alpha=alpha)

    mean_trace = np.nanmean(traces_norm, axis=0)
    ax_traces.plot(t_ms, mean_trace, color="navy", lw=1.8, zorder=5,
                   label="mean")

    ax_traces.axvline(0, color="black", ls="--", lw=0.9, alpha=0.8)
    ax_traces.set_ylabel("Angle (norm.)", fontsize=10)
    ax_traces.set_title(f"Unit {unit}  |  {n_events} events", fontsize=12,
                        fontweight="bold")
    ax_traces.set_xlim(-pre_ms, post_ms)
    ax_traces.set_ylim(-0.05, 1.15)
    ax_traces.legend(fontsize=8, loc="upper right", frameon=False)
    ax_traces.spines["top"].set_visible(False)
    ax_traces.spines["right"].set_visible(False)
    ax_traces.tick_params(labelbottom=False, labelsize=9)

    # ── Panel 2: Raster ───────────────────────────────────────────────────────
    for idx, rel_ms in enumerate(trials):
        if len(rel_ms):
            # Standard tick marker (thin vertical line via scatter diamond)
            ax_raster.scatter(
                rel_ms, [idx] * len(rel_ms),
                s=3, color="black", alpha=0.85,
                marker=(4, 0, 45), linewidths=0,
            )

    ax_raster.axvline(0, color="black", ls="--", lw=0.9, alpha=0.8)
    ax_raster.set_ylabel("Trial #", fontsize=10)
    ax_raster.set_ylim(-0.5, n_events - 0.5)
    ax_raster.invert_yaxis()         # trial 0 at top
    ax_raster.spines["top"].set_visible(False)
    ax_raster.spines["right"].set_visible(False)
    ax_raster.tick_params(labelbottom=False, labelsize=9)

    # ── Panel 3: PSTH ─────────────────────────────────────────────────────────
    bins = np.arange(-pre_ms, post_ms + bin_ms, bin_ms)
    all_spikes = np.concatenate(trials) if n_events else np.array([])

    if len(all_spikes):
        counts, edges = np.histogram(all_spikes, bins=bins)
        centres = (edges[:-1] + edges[1:]) / 2
        firing_rate = counts / (n_events * bin_ms / 1000.0)
        if smooth_bins > 1:
            firing_rate = uniform_filter1d(firing_rate, size=smooth_bins)
        ax_psth.bar(centres, firing_rate, width=bin_ms * 0.9,
                    color="black", edgecolor="none")

    ax_psth.axvline(0, color="black", ls="--", lw=0.9, alpha=0.8)
    ax_psth.set_xlabel("Time from peak (ms)", fontsize=10)
    ax_psth.set_ylabel("Firing rate (Hz)", fontsize=10)
    ax_psth.spines["top"].set_visible(False)
    ax_psth.spines["right"].set_visible(False)
    ax_psth.tick_params(labelsize=9)

    out_path = os.path.join(output_dir, f"unit_{unit:03d}_peak_psth.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run(session_dir: str,
        events_csv: str,
        angle_csv: str,
        output_dir: str | None = None,
        pre_frames: int = 100,
        post_frames: int = 100,
        bin_ms: float = 5.0,
        smooth_bins: int = 3,
        sampling_rate: int = 30_000,
        sync_channel: int = 1,
        units: list | None = None):

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(events_csv),
                                  "peak_psth_output")
    os.makedirs(output_dir, exist_ok=True)

    # ── Frame sync ────────────────────────────────────────────────────────────
    digitalin_path = os.path.join(session_dir, "digitalin.dat")
    if not os.path.isfile(digitalin_path):
        raise FileNotFoundError(f"digitalin.dat not found: {digitalin_path}")

    frame_samples = load_frame_sync(digitalin_path, channel=sync_channel,
                                    sampling_rate=sampling_rate)

    # Estimate inter-frame interval from the sync signal
    ifi_s  = float(np.median(np.diff(frame_samples))) / sampling_rate
    ifi_ms = ifi_s * 1000.0
    fps    = 1.0 / ifi_s
    pre_ms_f  = pre_frames  * ifi_ms
    post_ms_f = post_frames * ifi_ms

    print(f"\nCamera frame rate : {fps:.1f} fps  ({ifi_ms:.4f} ms/frame)")
    print(f"Alignment window  : -{pre_ms_f:.1f} ms  …  +{post_ms_f:.1f} ms")

    # ── Data ──────────────────────────────────────────────────────────────────
    events       = load_peak_events(events_csv)
    angle_series = load_angle_series(angle_csv)
    spikes_df    = load_spikes(os.path.join(session_dir, "spikes.csv"))

    print(f"Peak events       : {len(events)}")
    print(f"Angle data frames : {len(angle_series)}")

    # ── Frame → seconds for event alignment ──────────────────────────────────
    event_times_s = frame_to_seconds(events, frame_samples, sampling_rate)

    # ── Angle traces (frame domain) ───────────────────────────────────────────
    print("\nExtracting angle traces …")
    raw_traces  = extract_angle_traces(angle_series, events,
                                       pre_frames, post_frames)
    norm_traces = normalize_traces(raw_traces)

    # Drop events where the entire window is missing
    valid = ~np.all(np.isnan(norm_traces), axis=1)
    if not valid.all():
        print(f"  Dropped {(~valid).sum()} events with no angle data.")
    norm_traces   = norm_traces[valid]
    event_times_s = event_times_s[valid]
    n_valid       = int(valid.sum())
    print(f"  Valid events      : {n_valid}")

    # Shared time axis for all panels (ms from peak)
    t_ms = np.arange(-pre_frames, post_frames + 1) * ifi_ms

    # ── Per-unit figures ──────────────────────────────────────────────────────
    available = sorted(spikes_df["unit"].unique())
    process   = [u for u in units if u in available] if units else available
    print(f"\nUnits to process  : {process}\n")

    pre_s  = pre_ms_f  / 1000.0
    post_s = post_ms_f / 1000.0

    for unit in process:
        print(f"Unit {unit} …")
        spike_times = spikes_df.loc[spikes_df["unit"] == unit, "time"].values
        trials      = align_spikes_to_peaks(spike_times, event_times_s,
                                            pre_s, post_s)
        total_spikes = sum(len(t) for t in trials)
        print(f"  {total_spikes} spikes across {n_valid} events")
        plot_unit_figure(unit, norm_traces, trials, t_ms,
                         pre_ms_f, post_ms_f, bin_ms, output_dir,
                         smooth_bins=smooth_bins)

    print(f"\nAll done.  Figures saved to:\n  {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--session_dir",   required=True,
                   help="Session directory containing digitalin.dat and spikes.csv")
    p.add_argument("--events_csv",    required=True,
                   help="CSV with an 'Event' column (peak frame numbers)")
    p.add_argument("--angle_csv",     required=True,
                   help="CSV with 'Time' and 'Data' columns (angle per frame)")
    p.add_argument("--output_dir",    default=None,
                   help="Where to save figures (default: <events_csv_dir>/peak_psth_output)")
    p.add_argument("--pre_frames",    type=int,   default=10,
                   help="Frames before peak (default: 10)")
    p.add_argument("--post_frames",   type=int,   default=10,
                   help="Frames after peak (default: 10)")
    p.add_argument("--bin_ms",        type=float, default=1.0,
                   help="PSTH bin width in ms (default: 1)")
    p.add_argument("--smooth",        type=int,   default=3,
                   help="Uniform smoothing kernel width for PSTH in bins (default: 3, set 1 to disable)")
    p.add_argument("--sampling_rate", type=int,   default=30_000,
                   help="Master clock rate in Hz (default: 30000)")
    p.add_argument("--sync_channel",  type=int,   default=1,
                   help="digitalin.dat channel carrying frame-sync TTL (default: 1)")
    p.add_argument("--units",         type=int,   nargs="*", default=None,
                   help="Subset of unit IDs to process (default: all)")
    args = p.parse_args()

    run(
        session_dir   = args.session_dir,
        events_csv    = args.events_csv,
        angle_csv     = args.angle_csv,
        output_dir    = args.output_dir,
        pre_frames    = args.pre_frames,
        post_frames   = args.post_frames,
        bin_ms        = args.bin_ms,
        smooth_bins   = args.smooth,
        sampling_rate = args.sampling_rate,
        sync_channel  = args.sync_channel,
        units         = args.units,
    )


if __name__ == "__main__":
    main()
