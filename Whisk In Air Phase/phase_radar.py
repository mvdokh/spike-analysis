"""
Whisking Phase Radar Analysis
==============================
Assigns each spike to a Hilbert phase bin during valid whisking intervals,
then plots polar "phase tuning" radar charts per unit.

Frame → time conversion uses the exact same digitalin.dat TTL sync method
as Contact_PSTH/contact_psth.py:
  - Load digitalin.dat (30 kHz master clock, uint16 samples)
  - Extract rising edges on channel 1 (bit index 1) → frame_samples[]
  - time_s = frame_samples[frame_index] / SAMPLING_RATE

Inputs
------
  HILBERT_PHASE_CSV  : columns Time (frame), Data (phase radians)
  INTERVALS_CSV      : columns Start, End (frame numbers)
  SPIKES_CSV         : no header — col0=time_s, col1=unit, col2=ignored
  DIGITALIN_PATH     : <session_dir>/digitalin.dat

Outputs
-------
  <OUTPUT_DIR>/unit_<id>_phase_radar.png  (one per unit)
  <OUTPUT_DIR>/phase_spike_summary.csv
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import circmean

# ---------------------------------------------------------------------------
# Paths — edit these to match the session
# ---------------------------------------------------------------------------
SESSION_DIR      = r"C:\Users\wanglab\Desktop\Club Like Endings\102725_1"
HILBERT_PHASE_CSV = os.path.join(SESSION_DIR,
                                 r"c1_1027_whisk_in_air\c1_hilbert_phase.csv")
INTERVALS_CSV    = os.path.join(SESSION_DIR,
                                r"c1_1027_whisk_in_air\whisk_in_air_c1.csv")
SPIKES_CSV       = os.path.join(SESSION_DIR, "spikes.csv")
DIGITALIN_PATH   = os.path.join(SESSION_DIR, "digitalin.dat")
OUTPUT_DIR       = os.path.join(SESSION_DIR, "phase_radar")

# Acquisition / sync parameters (match contact_psth.py exactly)
SAMPLING_RATE = 30_000   # master clock Hz (digitalin.dat)
SYNC_CHANNEL  = 1        # bit index of frame-sync TTL in digitalin.dat
VIDEO_FPS     = 500      # camera frame rate — used only for spike-tolerance window

# Phase binning
N_BINS = 36              # 10° each, spanning [-π, +π]; 360/36 = 10°

# ---------------------------------------------------------------------------
# Add the Spike PSTH Pipeline directory (same sys.path trick as contact_psth.py)
# ---------------------------------------------------------------------------
PIPELINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Spike PSTH Pipeline",
)
sys.path.insert(0, PIPELINE_DIR)

from binary_data import get_digital
from ttls import find_ttls_on_single_channel_16bit


# ===========================================================================
# STEP 0 — Frame sync helpers  (copied verbatim from contact_psth.py)
# ===========================================================================

def load_frame_sync(digitalin_path: str,
                    channel: int = 1,
                    sampling_rate: int = 30_000) -> np.ndarray:
    """
    Load digitalin.dat and return rising-edge sample indices on *channel*.
    These correspond 1-to-1 with video frames (frame N → frame_samples[N]).
    """
    print(f"  Loading digitalin.dat from: {digitalin_path}")
    digital_inputs = get_digital(
        filepath=digitalin_path,
        header_offset_in_bytes=0,
        single_sample_size_in_bytes=2,
    )
    ttl_bool = find_ttls_on_single_channel_16bit(digital_inputs, channel)
    ttl_diff = np.ediff1d(ttl_bool.astype(int))
    onsets = np.where(ttl_diff > 0)[0] + 1   # low→high transitions
    print(f"  Found {len(onsets)} frame-sync rising edges on channel {channel}")
    return onsets


def frame_to_seconds(frame_indices,
                     frame_samples: np.ndarray,
                     sampling_rate: int = 30_000) -> np.ndarray:
    """
    Convert video frame numbers to seconds via the sync lookup table.
    """
    frame_indices = np.asarray(frame_indices, dtype=int)
    max_frame = len(frame_samples) - 1
    oob = frame_indices > max_frame
    if oob.any():
        print(f"  WARNING: {oob.sum()} frame indices exceed max synced frame "
              f"({max_frame}). Clipping.")
        frame_indices = np.clip(frame_indices, 0, max_frame)
    return frame_samples[frame_indices] / sampling_rate


# ===========================================================================
# STEP 1 — Load data
# ===========================================================================

def load_phase_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"  Hilbert phase CSV columns: {list(df.columns)}")
    # Expect 'Time' and 'Data'; raise clearly if missing
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
    df["start_frame"] = df["start_frame"].astype(int)
    df["end_frame"]   = df["end_frame"].astype(int)
    return df


def load_spikes_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    print(f"  Spikes CSV shape: {df.shape}  (no header expected)")
    df = df.iloc[:, :3]
    df.columns = ["time", "unit", "extra"]
    df["time"] = df["time"].astype(float)
    df["unit"] = df["unit"].astype(int)
    print(f"  Loaded {len(df)} spikes across {df['unit'].nunique()} units: "
          f"{sorted(df['unit'].unique())}")
    return df


# ===========================================================================
# STEP 2 — Filter frames to whisking intervals
# ===========================================================================

def filter_to_intervals(phase_df: pd.DataFrame,
                        intervals_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows in phase_df whose frame falls in any [start, end] interval."""
    mask = np.zeros(len(phase_df), dtype=bool)
    frames = phase_df["frame"].values
    for _, row in intervals_df.iterrows():
        mask |= (frames >= row["start_frame"]) & (frames <= row["end_frame"])
    filtered = phase_df[mask].copy().reset_index(drop=True)
    print(f"  Whisking frames: {mask.sum()} / {len(phase_df)} total frames "
          f"across {len(intervals_df)} intervals")
    return filtered


# ===========================================================================
# STEP 3 — Assign spikes to phase bins
# ===========================================================================

def build_phase_spike_counts(whisking_df: pd.DataFrame,
                              spikes_df: pd.DataFrame,
                              phase_col: str = "phase",
                              n_bins: int = N_BINS,
                              fps: int = VIDEO_FPS):
    """
    For every spike, find the nearest valid whisking frame within ±1/fps seconds.
    Assign that spike to the phase bin of that frame.

    Returns
    -------
    phase_spike_counts : dict  {unit_id: np.ndarray shape (n_bins,)}
    bin_edges          : np.ndarray shape (n_bins+1,)  in radians
    bin_centers        : np.ndarray shape (n_bins,)    in radians
    bin_occupancy_s    : np.ndarray shape (n_bins,)    total seconds per bin
    assigned_count     : int  total spikes assigned
    """
    tolerance_s = 1.0 / fps   # ±1 frame duration

    bin_edges   = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Valid whisking times and phases as sorted arrays for fast lookup
    whisking_times  = whisking_df["time_s"].values
    whisking_phases = whisking_df[phase_col].values

    # Validate phase range
    raw_oob = (whisking_phases < -np.pi) | (whisking_phases > np.pi)
    if raw_oob.any():
        print(f"  WARNING: {raw_oob.sum()} phase values outside [-π, +π] — clipping.")
        whisking_phases = np.clip(whisking_phases, -np.pi, np.pi)

    # Bin occupancy: each valid whisking frame contributes 1/fps seconds
    phase_bin_indices_all = np.digitize(whisking_phases, bin_edges) - 1
    phase_bin_indices_all = np.clip(phase_bin_indices_all, 0, n_bins - 1)
    bin_occupancy_s = np.zeros(n_bins)
    for b in phase_bin_indices_all:
        bin_occupancy_s[b] += 1.0 / fps

    # Sort whisking times for searchsorted
    sort_idx     = np.argsort(whisking_times)
    sorted_times = whisking_times[sort_idx]
    sorted_phases = whisking_phases[sort_idx]

    units = sorted(spikes_df["unit"].unique())
    phase_spike_counts = {u: np.zeros(n_bins, dtype=int) for u in units}

    total_assigned = 0
    total_discarded = 0

    for unit in units:
        unit_times = spikes_df.loc[spikes_df["unit"] == unit, "time"].values
        assigned = 0
        for spike_t in unit_times:
            idx = np.searchsorted(sorted_times, spike_t)
            # Check idx and idx-1 for the closest frame
            best_dt = np.inf
            best_phase = None
            for candidate in (idx - 1, idx):
                if 0 <= candidate < len(sorted_times):
                    dt = abs(sorted_times[candidate] - spike_t)
                    if dt < best_dt:
                        best_dt = dt
                        best_phase = sorted_phases[candidate]
            if best_dt <= tolerance_s and best_phase is not None:
                b = int(np.digitize(best_phase, bin_edges) - 1)
                b = np.clip(b, 0, n_bins - 1)
                phase_spike_counts[unit][b] += 1
                assigned += 1
            else:
                total_discarded += 1
        total_assigned += assigned

    print(f"  Spikes assigned: {total_assigned}  |  discarded (no nearby frame): "
          f"{total_discarded}")
    return phase_spike_counts, bin_edges, bin_centers, bin_occupancy_s, total_assigned


# ===========================================================================
# STEP 5 — Radar plot
# ===========================================================================

def make_radar_plot(unit_id: int,
                    counts: np.ndarray,
                    bin_centers: np.ndarray,
                    bin_occupancy_s: np.ndarray,
                    n_whisking_frames: int,
                    out_path: str):
    """
    Polar bar chart for one unit.

    Orientation (set_theta_zero_location="E", CCW):
        East  (right)  = 0
        North (top)    = π/2
        West  (left)   = ±π
        South (bottom) = −π/2

    bin_centers are in [-π, π]; remapped via % (2π) to [0, 2π] so bars fill
    the correct positions without any offset shift.
    """
    FONT          = "Arial"
    FONT_SZ_TICK  = 13
    FONT_SZ_ANNOT = 9
    FONT_SZ_TITLE = 15
    FONT_SZ_ARROW = 11

    n_bins    = len(bin_centers)
    bin_width = 2 * np.pi / n_bins

    rate = counts.astype(float)

    # Remap bin centers from [-π, π] → [0, 2π] so 0 stays at East.
    # Negative phases (retraction side) wrap to [π, 2π] as expected.
    theta_plot   = bin_centers % (2 * np.pi)
    sort_idx     = np.argsort(theta_plot)
    theta_sorted = theta_plot[sort_idx]

    r_ylim = max(rate.max(), 1e-9) * 1.05   # outer ring sits just above tallest bar

    def polar_to_af(ax, theta, r):
        """Convert polar data (theta_rad, r) → axes-fraction (x, y).
        Works for r > r_ylim too (linear extrapolation outside the disc).
        """
        disp = ax.transData.transform(np.array([[theta, r]]))
        return ax.transAxes.inverted().transform(disp)[0]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)

    ax.set_ylim(0, r_ylim)   # set before any drawing so transforms are correct

    r_sorted = rate[sort_idx]
    ax.bar(theta_sorted, r_sorted, width=bin_width,
           color="black", alpha=1.0,
           edgecolor="black", linewidth=0.5, bottom=0)

    # Plot line connecting the bins
    theta_closed = np.append(theta_sorted, theta_sorted[0])
    r_closed = np.append(r_sorted, r_sorted[0])
    ax.plot(theta_closed, r_closed, color="black", linewidth=2.0)

    # Orientation: 0 at East (right), CCW increasing phase
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    # Tick positions in [0, 2π] → display positions:
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["0", "π/2", "±π", "−π/2"],
                       fontsize=FONT_SZ_TICK, fontfamily=FONT, color="black")
    
    # Tick lines and girds
    ax.tick_params(colors='black')
    ax.grid(color='black', alpha=0.5)

    # Relabel radial ticks
    rticks = [t for t in ax.get_yticks() if 0 < t <= r_ylim]
    ax.set_yticks(rticks)
    ax.set_yticklabels([f"{int(t)}" for t in rticks],
                       fontsize=FONT_SZ_TICK - 3, fontfamily=FONT, color="black")
                       
    # Style the outer spine to look like the inner grid rings
    ax.spines['polar'].set_color('black')
    ax.spines['polar'].set_linewidth(1.0)

    ax.set_title(f"Unit {unit_id} — Phase Tuning Curve", pad=28,
                 fontsize=FONT_SZ_TITLE + 2, fontfamily=FONT, fontweight="bold", color="black")

    total_spikes = int(counts.sum())
    ax.annotate(
        f"Total spikes: {total_spikes}\nWhisking frames: {n_whisking_frames}",
        xy=(0.01, 0.01), xycoords="axes fraction",
        fontsize=FONT_SZ_ANNOT, color="black", fontfamily=FONT,
    )

    # --- Directional arrows -------------------------------------------
    # Use axes fraction coordinates to draw outside the internal clip logic.
    ctr      = np.array([0.5, 0.5])
    arr_rad  = 0.525  # Tucked even closer to the perimeter ring

    # Calculate shortened arc directions (~60 degrees instead of 90 degrees)
    # Protracting: centered around 135 deg, spans 165 to 105
    p_start_dir = np.array([np.cos(np.radians(165)), np.sin(np.radians(165))])
    p_end_dir   = np.array([np.cos(np.radians(105)), np.sin(np.radians(105))])
    
    # Retracting: centered around -45 deg, spans -15 to -75
    r_start_dir = np.array([np.cos(np.radians(-15)), np.sin(np.radians(-15))])
    r_end_dir   = np.array([np.cos(np.radians(-75)), np.sin(np.radians(-75))])

    ap_start = ctr + arr_rad * p_start_dir
    ap_end   = ctr + arr_rad * p_end_dir
    
    ar_start = ctr + arr_rad * r_start_dir
    ar_end   = ctr + arr_rad * r_end_dir

    # Protracting 
    ap = mpatches.FancyArrowPatch(
        posA=ap_start, posB=ap_end,
        transform=ax.transAxes, clip_on=False, zorder=10,
        arrowstyle=mpatches.ArrowStyle("-|>", head_length=8, head_width=5),
        lw=2.0, color="black",
        connectionstyle="arc3,rad=-0.28") # Adjusted rad for the shorter 60-degree chord
    ax.add_patch(ap)
    mid_p_dir = np.array([-0.707, 0.707]) # NW unit vector for label positioning
    ax.text(*(ctr + mid_p_dir * (arr_rad + 0.085)), "Protracting",  # Pushed word out further
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONT_SZ_ARROW, fontfamily=FONT,
            color="black", fontweight="bold", clip_on=False)

    # Retracting 
    ar = mpatches.FancyArrowPatch(
        posA=ar_start, posB=ar_end,
        transform=ax.transAxes, clip_on=False, zorder=10,
        arrowstyle=mpatches.ArrowStyle("-|>", head_length=8, head_width=5),
        lw=2.0, color="black",
        connectionstyle="arc3,rad=-0.28") # Adjusted rad for the shorter 60-degree chord
    ax.add_patch(ar)
    mid_r_dir = np.array([0.707, -0.707]) # SE unit vector for label positioning
    ax.text(*(ctr + mid_r_dir * (arr_rad + 0.085)), "Retracting",  # Pushed word out further
            transform=ax.transAxes, ha="center", va="center",
            fontsize=FONT_SZ_ARROW, fontfamily=FONT,
            color="black", fontweight="bold", clip_on=False)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ===========================================================================
# STEP 6 — Summary CSV
# ===========================================================================

def save_summary_csv(units,
                     raw_counts_dict,
                     bin_centers, bin_occupancy_s,
                     out_path: str):
    rows = []
    for unit in units:
        raw_c  = raw_counts_dict[unit]
        for b, center in enumerate(bin_centers):
            occ = bin_occupancy_s[b]
            raw_rate  = raw_c[b]  / occ if occ > 0 else 0.0
            rows.append({
                "unit_id":                  unit,
                "bin_center_rad":           round(center, 6),
                "raw_spike_count":          int(raw_c[b]),
                "raw_spike_rate_hz":        round(raw_rate, 6),
            })
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_path, index=False)
    print(f"  Summary CSV saved: {out_path}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- STEP 1: LOAD -------------------------------------------------------
    print("\n=== STEP 1: Load data ===")

    print(f"\nLoading Hilbert phase CSV:\n  {HILBERT_PHASE_CSV}")
    phase_df = load_phase_csv(HILBERT_PHASE_CSV)
    print(f"  Rows: {len(phase_df)}  |  Phase range: "
          f"{phase_df['phase'].min():.4f} to {phase_df['phase'].max():.4f} rad")

    print(f"\nLoading whisking intervals CSV:\n  {INTERVALS_CSV}")
    intervals_df = load_intervals_csv(INTERVALS_CSV)
    print(f"  Intervals: {len(intervals_df)}")

    print(f"\nLoading spikes CSV:\n  {SPIKES_CSV}")
    spikes_df = load_spikes_csv(SPIKES_CSV)

    print(f"\nLoading frame sync (digitalin.dat):\n  {DIGITALIN_PATH}")
    frame_samples = load_frame_sync(DIGITALIN_PATH,
                                    channel=SYNC_CHANNEL,
                                    sampling_rate=SAMPLING_RATE)

    # Convert every phase-CSV frame → seconds using the TTL sync lookup
    print("\nConverting phase-CSV frames to seconds via TTL sync...")
    phase_df["time_s"] = frame_to_seconds(phase_df["frame"].values,
                                           frame_samples,
                                           SAMPLING_RATE)
    print(f"  Time range: {phase_df['time_s'].min():.4f} s "
          f"to {phase_df['time_s'].max():.4f} s")

    # --- STEP 2: FILTER TO WHISKING INTERVALS ------------------------------
    print("\n=== STEP 2: Filter to whisking intervals ===")
    whisking_df = filter_to_intervals(phase_df, intervals_df)
    n_whisking_frames = len(whisking_df)

    if n_whisking_frames == 0:
        raise RuntimeError("No whisking frames found after filtering — "
                           "check that frame numbers in the intervals CSV "
                           "match those in the phase CSV.")

    # --- STEP 3: Build phase → spike counts --------------------------------
    print("\n=== STEP 3: Build phase spike counts ===")
    raw_counts_dict, bin_edges, bin_centers, bin_occupancy_s, _ = \
        build_phase_spike_counts(whisking_df, spikes_df,
                                 phase_col="phase",
                                 n_bins=N_BINS,
                                 fps=VIDEO_FPS)

    # --- STEP 5: Radar plots ------------------------------------------------
    print("\n=== STEP 5: Saving radar plots ===")
    units = sorted(spikes_df["unit"].unique())
    print(f"  {len(units)} units to plot: {units}")

    skipped = []
    for unit in units:
        raw_c  = raw_counts_dict[unit]
        if raw_c.sum() == 0:
            print(f"  WARNING: Unit {unit} has zero spikes in whisking intervals — "
                  f"skipping radar plot.")
            skipped.append(unit)
            continue
        out_path = os.path.join(OUTPUT_DIR, f"unit_{unit}_phase_radar.png")
        make_radar_plot(unit, raw_c, bin_centers, bin_occupancy_s,
                        n_whisking_frames, out_path)

    if skipped:
        print(f"\n  Skipped {len(skipped)} units with zero spikes: {skipped}")

    # --- STEP 6: Summary CSV ------------------------------------------------
    print("\n=== STEP 6: Saving summary CSV ===")
    summary_path = os.path.join(OUTPUT_DIR, "phase_spike_summary.csv")
    save_summary_csv(units,
                     raw_counts_dict,
                     bin_centers, bin_occupancy_s,
                     summary_path)

    print("\n=== Done ===")
    plotted = len(units) - len(skipped)
    print(f"  Radar plots saved: {plotted}")
    print(f"  Summary CSV:       {summary_path}")
    print(f"  Output directory:  {OUTPUT_DIR}")
