"""
Spike Phase Distribution (Whisking-Interval Sampling)
======================================================
For each spike that falls within a valid whisking interval, look up the
instantaneous Hilbert phase of the nearest video frame and record that
phase value. The resulting histogram answers:

    "What phase was the whisker in when this neuron fired?"

A flat (uniform) distribution → no phase preference.
A peaked distribution → the neuron fires preferentially at a particular phase.

No occupancy normalisation is applied — each spike contributes exactly one
sample to the histogram, so a bin's count is purely the number of spikes
observed at that phase.

Frame → time conversion uses the digitalin.dat TTL sync method
(identical to contact_psth.py):
  - Load digitalin.dat (30 kHz, uint16)
  - Extract rising edges on channel 1 → frame_samples[]
  - time_s = frame_samples[frame_index] / SAMPLING_RATE

Inputs
------
  HILBERT_PHASE_CSV  : columns Time (frame), Data (phase radians)
  INTERVALS_CSV      : columns Start, End (frame numbers, inclusive)
  SPIKES_CSV         : no header — col0 = time_s, col1 = unit, col2 = ignored
  DIGITALIN_PATH     : <session_dir>/digitalin.dat

Outputs
-------
  <OUTPUT_DIR>/unit_<id>_spike_phase.png   — one radar plot per unit
  <OUTPUT_DIR>/spike_phase_summary.csv     — per-unit, per-bin stats
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd

# ---------------------------------------------------------------------------
# Paths — edit these to match the session
# ---------------------------------------------------------------------------
SESSION_DIR       = r"C:\Users\wanglab\Desktop\Club Like Endings\102525_1"
HILBERT_PHASE_CSV = os.path.join(SESSION_DIR,
                                 r"c1_wia\analog_output.csv")
INTERVALS_CSV     = os.path.join(SESSION_DIR,
                                 r"c1_wia\c1_whisking_in_air.csv")
SPIKES_CSV        = os.path.join(SESSION_DIR, "spikes.csv")
DIGITALIN_PATH    = os.path.join(SESSION_DIR, "digitalin.dat")
OUTPUT_DIR        = os.path.join(SESSION_DIR, "spike_phase_distribution")

# Acquisition / sync parameters
SAMPLING_RATE = 30_000   # master clock Hz (digitalin.dat)
SYNC_CHANNEL  = 1        # bit index of frame-sync TTL

# Phase binning
N_BINS = 36              # 10° per bin, spanning [−π, +π]

# Maximum time gap allowed between a spike and its nearest video frame.
# Set conservatively to ~1 frame durations at 500 fps (2 ms).
VIDEO_FPS         = 500
MAX_FRAME_GAP_S   = 1.0 / VIDEO_FPS   # 2 ms

# ---------------------------------------------------------------------------
# Add Spike PSTH Pipeline to path (for binary_data / ttls helpers)
# ---------------------------------------------------------------------------
PIPELINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Spike PSTH Pipeline",
)
sys.path.insert(0, PIPELINE_DIR)

from binary_data import get_digital
from ttls import find_ttls_on_single_channel_16bit


# ===========================================================================
# Frame-sync helpers
# ===========================================================================

def load_frame_sync(digitalin_path: str,
                    channel: int = 1,
                    sampling_rate: int = 30_000) -> np.ndarray:
    """Return rising-edge sample indices on *channel* from digitalin.dat."""
    print(f"  Loading digitalin.dat: {digitalin_path}")
    digital_inputs = get_digital(
        filepath=digitalin_path,
        header_offset_in_bytes=0,
        single_sample_size_in_bytes=2,
    )
    ttl_bool = find_ttls_on_single_channel_16bit(digital_inputs, channel)
    ttl_diff = np.ediff1d(ttl_bool.astype(int))
    onsets = np.where(ttl_diff > 0)[0] + 1
    print(f"  Found {len(onsets)} frame-sync edges on channel {channel}")
    return onsets


def frames_to_seconds(frame_indices: np.ndarray,
                      frame_samples: np.ndarray,
                      sampling_rate: int = 30_000) -> np.ndarray:
    """Convert video frame numbers to seconds via the sync lookup table."""
    frame_indices = np.asarray(frame_indices, dtype=int)
    max_frame = len(frame_samples) - 1
    oob = frame_indices > max_frame
    if oob.any():
        print(f"  WARNING: {oob.sum()} frame indices exceed max synced frame "
              f"({max_frame}). Clipping.")
        frame_indices = np.clip(frame_indices, 0, max_frame)
    return frame_samples[frame_indices] / sampling_rate


# ===========================================================================
# Data loaders
# ===========================================================================

def load_phase_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"  Phase CSV columns: {list(df.columns)}")
    for col in ("Time", "Data"):
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' in {path}. Found: {list(df.columns)}"
            )
    df = df.rename(columns={"Time": "frame", "Data": "phase"})
    df["frame"] = df["frame"].astype(int)
    df["phase"] = df["phase"].astype(float)
    return df


def load_intervals_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"  Intervals CSV columns: {list(df.columns)}")
    for col in ("Start", "End"):
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' in {path}. Found: {list(df.columns)}"
            )
    df = df.rename(columns={"Start": "start_frame", "End": "end_frame"})
    df[["start_frame", "end_frame"]] = df[["start_frame", "end_frame"]].astype(int)
    return df


def load_spikes_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None).iloc[:, :3]
    df.columns = ["time", "unit", "extra"]
    df["time"] = df["time"].astype(float)
    df["unit"] = df["unit"].astype(int)
    print(f"  Loaded {len(df)} spikes across {df['unit'].nunique()} units: "
          f"{sorted(df['unit'].unique())}")
    return df


# ===========================================================================
# Core analysis
# ===========================================================================

def convert_intervals_to_seconds(intervals_df: pd.DataFrame,
                                  frame_samples: np.ndarray,
                                  sampling_rate: int) -> pd.DataFrame:
    """Add start_s / end_s columns to intervals_df."""
    df = intervals_df.copy()
    df["start_s"] = frames_to_seconds(df["start_frame"].values,
                                      frame_samples, sampling_rate)
    df["end_s"]   = frames_to_seconds(df["end_frame"].values,
                                      frame_samples, sampling_rate)
    return df


def spikes_in_intervals(spikes_df: pd.DataFrame,
                        intervals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the rows of spikes_df whose spike time falls within
    at least one [start_s, end_s] interval.
    """
    spike_times = spikes_df["time"].values
    mask = np.zeros(len(spikes_df), dtype=bool)
    for _, row in intervals_df.iterrows():
        mask |= (spike_times >= row["start_s"]) & (spike_times <= row["end_s"])
    filtered = spikes_df[mask].copy().reset_index(drop=True)
    print(f"  Spikes in intervals: {mask.sum()} / {len(spikes_df)}")
    return filtered


def assign_spike_phases(spikes_in: pd.DataFrame,
                        phase_df: pd.DataFrame,
                        max_gap_s: float = MAX_FRAME_GAP_S) -> pd.DataFrame:
    """
    For each spike, find the nearest frame (by time) and record its phase.

    Returns a copy of spikes_in with two new columns:
        assigned_phase  — Hilbert phase at nearest frame (NaN if no frame
                          within max_gap_s)
        nearest_frame_dt_s — actual time gap to nearest frame
    """
    frame_times  = phase_df["time_s"].values   # must already be in seconds
    frame_phases = phase_df["phase"].values

    sort_idx      = np.argsort(frame_times)
    sorted_times  = frame_times[sort_idx]
    sorted_phases = frame_phases[sort_idx]

    spike_times = spikes_in["time"].values
    assigned_phases = np.full(len(spike_times), np.nan)
    nearest_dts     = np.full(len(spike_times), np.nan)

    for i, t in enumerate(spike_times):
        idx = np.searchsorted(sorted_times, t)
        best_dt    = np.inf
        best_phase = np.nan
        for cand in (idx - 1, idx):
            if 0 <= cand < len(sorted_times):
                dt = abs(sorted_times[cand] - t)
                if dt < best_dt:
                    best_dt    = dt
                    best_phase = sorted_phases[cand]
        if best_dt <= max_gap_s:
            assigned_phases[i] = best_phase
            nearest_dts[i]     = best_dt

    out = spikes_in.copy()
    out["assigned_phase"]     = assigned_phases
    out["nearest_frame_dt_s"] = nearest_dts

    n_assigned  = (~np.isnan(assigned_phases)).sum()
    n_discarded = np.isnan(assigned_phases).sum()
    print(f"  Phase assigned: {n_assigned}  |  no frame within {max_gap_s*1000:.1f} ms: "
          f"{n_discarded}")
    return out


def build_phase_histograms(spikes_with_phase: pd.DataFrame,
                           n_bins: int = N_BINS):
    """
    Build per-unit spike-count histograms over phase.

    Returns
    -------
    histograms   : dict {unit_id: np.ndarray shape (n_bins,)}
    bin_edges    : np.ndarray shape (n_bins+1,)
    bin_centers  : np.ndarray shape (n_bins,)
    """
    bin_edges   = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    units = sorted(spikes_with_phase["unit"].unique())
    histograms = {}

    for unit in units:
        phases = spikes_with_phase.loc[
            (spikes_with_phase["unit"] == unit) &
            spikes_with_phase["assigned_phase"].notna(),
            "assigned_phase"
        ].values

        phases_clipped = np.clip(phases, -np.pi, np.pi)
        counts, _ = np.histogram(phases_clipped, bins=bin_edges)
        histograms[unit] = counts

    return histograms, bin_edges, bin_centers


# ===========================================================================
# Circular statistics helpers
# ===========================================================================

def mean_resultant_length(phases: np.ndarray) -> float:
    """Compute the mean resultant length R ∈ [0, 1] for a set of phases."""
    if len(phases) == 0:
        return 0.0
    return float(np.abs(np.mean(np.exp(1j * phases))))


def rayleigh_z(phases: np.ndarray):
    """
    Rayleigh test for circular uniformity.
    Returns (Z, p-value).  Approximation valid for n ≥ 10.
    """
    n = len(phases)
    if n < 3:
        return np.nan, np.nan
    R = mean_resultant_length(phases)
    Z = n * R ** 2
    # Approximation from Zar (1999)
    p = np.exp(-Z) * (1 + (2 * Z - Z ** 2) / (4 * n) - (24 * Z - 132 * Z ** 2 +
        76 * Z ** 3 - 9 * Z ** 4) / (288 * n ** 2))
    p = float(np.clip(p, 0.0, 1.0))
    return float(Z), p


# ===========================================================================
# Radar plot
# ===========================================================================

def make_radar_plot(unit_id: int,
                    counts: np.ndarray,
                    bin_centers: np.ndarray,
                    spike_phases: np.ndarray,
                    out_path: str):
    """
    Single-panel polar histogram: spike count per phase bin.
    Overlays:
      - a dashed red circle at the uniform expectation (total/n_bins)
      - curved arrows on the perimeter indicating whisking direction:
          one from 0 → −π/2  (East → South, i.e. protracting direction)
          one from ±π → π/2  (West → North, i.e. retracting direction)
    """
    FONT         = "Arial"
    FONT_SZ_TICK = 13
    FONT_SZ_ANN  = 9
    FONT_SZ_TTL  = 15
    FONT_SZ_DIR  = 11

    n_bins    = len(bin_centers)
    bin_width = 2 * np.pi / n_bins

    # Remap [-π, π] → [0, 2π] for plotting (0 stays at East)
    theta_plot   = bin_centers % (2 * np.pi)
    sort_idx     = np.argsort(theta_plot)
    theta_sorted = theta_plot[sort_idx]
    counts_sorted = counts[sort_idx]

    total_spikes = int(counts.sum())
    uniform_level = total_spikes / n_bins   # expected count per bin if uniform

    r_max = max(counts.max(), uniform_level * 1.1, 1e-9)
    r_ylim = r_max * 1.15

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # ── Bars (black, no gaps) ─────────────────────────────────────────────────
    ax.bar(theta_sorted, counts_sorted,
           width=bin_width, color="black", alpha=0.85,
           edgecolor="black", linewidth=0, bottom=0)

    # ── Axis orientation: 0 at East, CCW ─────────────────────────────────────
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, r_ylim)

    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["0", "π/2", "±π", "−π/2"],
                       fontsize=FONT_SZ_TICK, fontfamily=FONT)

    ax.tick_params(colors="black")
    ax.grid(color="black", alpha=0.4)
    ax.spines["polar"].set_color("black")
    ax.spines["polar"].set_linewidth(1.0)

    # Radial ticks (integers)
    rticks = [t for t in ax.get_yticks() if 0 < t <= r_ylim]
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{int(t)}" for t in rticks],
                       fontsize=FONT_SZ_TICK - 3, fontfamily=FONT)

    ax.set_title(f"Unit {unit_id} — Spike Phase Distribution",
                 pad=55, fontsize=FONT_SZ_TTL + 2,
                 fontfamily=FONT, fontweight="bold")

    # ── Spike count annotation (no circular stats) ────────────────────────────
    ax.annotate(
        f"n = {total_spikes} spikes",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=FONT_SZ_ANN, color="black", fontfamily=FONT,
    )

    # ── Perimeter arrows ──────────────────────────────────────────────────────
    # Arrows are drawn in axis (data) coordinates on the polar axes.
    # We place them just outside r_ylim using ax.annotate with
    # arrowprops on the polar axes.  Because matplotlib's polar axes
    # treat (theta, r) as data coords, we trace a short arc by drawing
    # a FancyArrowPatch between two (theta, r) points at r = arrow_r.
    #
    # Arrow 1: from 0 → −π/2  (East → bottom, clockwise = CW in standard view)
    #   In plot coords (CCW convention, 0 at East):
    #     start = 0 rad,   end = 3π/2 rad  (i.e. −π/2 mapped to [0,2π])
    #   We use connectionstyle arc3,rad to curve along the perimeter.
    #
    # Arrow 2: from ±π → π/2  (West → top, clockwise)
    #   In plot coords:
    #     start = π rad,   end = π/2 rad

    # Arrows span only half the arc (quarter circle each), placed outside plot
    arrow_r = r_ylim * 1.07

    arrow_kw = dict(
        xycoords="data", textcoords="data",
        arrowprops=dict(
            arrowstyle="-|>",
            color="black",
            lw=2.0,
            mutation_scale=10,
            connectionstyle="arc3,rad=0.20",
        ),
        annotation_clip=False,
    )

    # Arrow 1 — Retraction: 0 → π/2  (East → North, CCW arc)
    ax.annotate("",
                xy=(np.pi / 2 - np.pi / 8, arrow_r),        # end ~ 3π/8
                xytext=(0 + np.pi / 8, arrow_r),             # start ~ π/8
                **arrow_kw)

    # Arrow 2 — Protraction: ±π → −π/2  (West → South, CCW arc)
    # In [0,2π] coords: from π → 3π/2
    ax.annotate("",
                xy=(3 * np.pi / 2 - np.pi / 8, arrow_r),    # end ~ 11π/8
                xytext=(np.pi + np.pi / 8, arrow_r),         # start ~ 9π/8
                **arrow_kw)

    # Direction labels: Retracting above π/2 (top), Protracting below −π/2 (bottom)
    lbl_r = r_ylim * 1.18
    ax.text(np.pi / 2,     lbl_r, "Retracting",
            ha="center", va="center", clip_on=False,
            fontsize=FONT_SZ_DIR, fontfamily=FONT, fontweight="bold")
    ax.text(3 * np.pi / 2, lbl_r, "Protracting",
            ha="center", va="center", clip_on=False,
            fontsize=FONT_SZ_DIR, fontfamily=FONT, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# Summary CSV
# ===========================================================================

def save_summary_csv(units,
                     histograms: dict,
                     bin_centers: np.ndarray,
                     spikes_with_phase: pd.DataFrame,
                     out_path: str):
    rows = []
    for unit in units:
        counts = histograms.get(unit, np.zeros(len(bin_centers), dtype=int))
        if counts.sum() == 0:
            continue
        unit_phases = spikes_with_phase.loc[
            (spikes_with_phase["unit"] == unit) &
            spikes_with_phase["assigned_phase"].notna(),
            "assigned_phase"
        ].values
        R      = mean_resultant_length(unit_phases)
        mu     = float(circmean(unit_phases)) if len(unit_phases) > 0 else np.nan
        Z, p   = rayleigh_z(unit_phases)
        total  = int(counts.sum())
        uniform = total / len(bin_centers) if total > 0 else 0.0
        for b, center in enumerate(bin_centers):
            rows.append({
                "unit_id":           unit,
                "bin_center_rad":    round(center, 6),
                "bin_center_deg":    round(np.degrees(center), 2),
                "spike_count":       int(counts[b]),
                "uniform_expected":  round(uniform, 4),
                "mean_resultant_R":  round(R, 6),
                "mean_phase_rad":    round(mu, 6) if not np.isnan(mu) else np.nan,
                "rayleigh_Z":        round(Z, 4)  if not np.isnan(Z) else np.nan,
                "rayleigh_p":        round(p, 6)  if not np.isnan(p) else np.nan,
            })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"  Summary CSV saved: {out_path}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── STEP 1: Load ─────────────────────────────────────────────────────────
    print("\n=== STEP 1: Load data ===")

    print(f"\nHilbert phase CSV:\n  {HILBERT_PHASE_CSV}")
    phase_df = load_phase_csv(HILBERT_PHASE_CSV)
    print(f"  Rows: {len(phase_df)}  |  Phase range: "
          f"{phase_df['phase'].min():.4f} to {phase_df['phase'].max():.4f} rad")

    print(f"\nWhisking intervals CSV:\n  {INTERVALS_CSV}")
    intervals_df = load_intervals_csv(INTERVALS_CSV)
    print(f"  Intervals: {len(intervals_df)}")

    print(f"\nSpikes CSV:\n  {SPIKES_CSV}")
    spikes_df = load_spikes_csv(SPIKES_CSV)

    print(f"\nFrame sync (digitalin.dat):\n  {DIGITALIN_PATH}")
    frame_samples = load_frame_sync(DIGITALIN_PATH, SYNC_CHANNEL, SAMPLING_RATE)

    # Convert phase CSV frames → seconds
    print("\nConverting phase frames to seconds via TTL sync...")
    phase_df["time_s"] = frames_to_seconds(phase_df["frame"].values,
                                           frame_samples, SAMPLING_RATE)
    print(f"  Time range: {phase_df['time_s'].min():.4f} s "
          f"to {phase_df['time_s'].max():.4f} s")

    # Convert interval frame boundaries → seconds
    print("\nConverting interval boundaries to seconds...")
    intervals_df = convert_intervals_to_seconds(intervals_df,
                                                frame_samples, SAMPLING_RATE)
    print(f"  Interval time range: {intervals_df['start_s'].min():.4f} s "
          f"to {intervals_df['end_s'].max():.4f} s")

    # ── STEP 2: Filter spikes to intervals ───────────────────────────────────
    print("\n=== STEP 2: Filter spikes to whisking intervals ===")
    spikes_in = spikes_in_intervals(spikes_df, intervals_df)

    if len(spikes_in) == 0:
        raise RuntimeError(
            "No spikes found within the whisking intervals. "
            "Check that spike times (seconds) overlap with the "
            "interval time range printed above."
        )

    # ── STEP 3: Assign phase to each spike ───────────────────────────────────
    print("\n=== STEP 3: Assign phase to each spike ===")
    spikes_with_phase = assign_spike_phases(spikes_in, phase_df,
                                            max_gap_s=MAX_FRAME_GAP_S)

    # ── STEP 4: Build histograms ──────────────────────────────────────────────
    print("\n=== STEP 4: Build phase histograms ===")
    histograms, bin_edges, bin_centers = build_phase_histograms(
        spikes_with_phase, n_bins=N_BINS
    )

    # ── STEP 5: Radar plots ───────────────────────────────────────────────────
    print("\n=== STEP 5: Save radar plots ===")
    units   = sorted(spikes_df["unit"].unique())
    skipped = []

    for unit in units:
        counts = histograms.get(unit, np.zeros(N_BINS, dtype=int))
        if counts.sum() == 0:
            if unit not in skipped:
                print(f"  WARNING: Unit {unit} — zero spikes in intervals, skipping.")
                skipped.append(unit)
            continue

        unit_phases = spikes_with_phase.loc[
            (spikes_with_phase["unit"] == unit) &
            spikes_with_phase["assigned_phase"].notna(),
            "assigned_phase"
        ].values

        out_path = os.path.join(OUTPUT_DIR, f"unit_{unit}_spike_phase.png")
        make_radar_plot(unit, counts, bin_centers, unit_phases, out_path)

    if skipped:
        print(f"\n  Skipped {len(skipped)} units with zero spikes: {skipped}")

    # ── STEP 6: Summary CSV ───────────────────────────────────────────────────
    print("\n=== STEP 6: Save summary CSV ===")
    save_summary_csv(units, histograms, bin_centers,
                     spikes_with_phase,
                     os.path.join(OUTPUT_DIR, "spike_phase_summary.csv"))

    print("\n=== Done ===")
    print(f"  Plots saved: {len(units) - len(skipped)}")
    print(f"  Output dir:  {OUTPUT_DIR}")