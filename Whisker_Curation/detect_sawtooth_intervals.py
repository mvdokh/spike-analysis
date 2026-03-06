"""
Detect Sawtooth Whisking Intervals from Phase Data
====================================================

Reads whisker phase data (−π to π) and detects intervals of smooth
sawtooth-shaped whisking (monotonic ramp from −π → π, then wrap).

Each detected whisking bout (consecutive smooth cycles) is output as
a Start,End frame pair in CSV format.

Usage
-----
    python detect_sawtooth_intervals.py --phase_dir "C:\\...\\phase"
    python detect_sawtooth_intervals.py --phase_file "C:\\...\\1_phase.csv"

    Options:
        --output_dir    Directory for output CSVs (default: alongside phase)
        --min_cycles    Minimum consecutive good cycles to keep a bout (default: 3)
        --max_jitter    Maximum allowed derivative std / mean ratio (default: 0.6)
        --min_amplitude Minimum phase range within a cycle (default: 4.0, ~2/3 of 2π)
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def detect_sawtooth_intervals(phase_file, output_path,
                               min_cycles=3, max_jitter=0.6,
                               min_amplitude=4.0, gap_merge=2):
    """
    Detect smooth sawtooth whisking bouts from a phase time-series.

    Parameters
    ----------
    phase_file : str
        CSV with columns Time, Data (phase in radians, −π to π).
    output_path : str
        Where to write the Start,End interval CSV.
    min_cycles : int
        Minimum number of consecutive good cycles to keep a bout.
    max_jitter : float
        Maximum coefficient of variation of the positive phase derivative
        within a cycle. Lower = stricter smoothness.
    min_amplitude : float
        Minimum peak-to-trough range within a cycle (radians).
        Full cycle = 2π ≈ 6.28; default 4.0 keeps partial cycles.
    gap_merge : int
        Merge bouts separated by ≤ this many frames.
    """
    df = pd.read_csv(phase_file)
    frames = df["Time"].values
    phase = df["Data"].values

    # ── 1. Find wrap-around points (π → −π jumps) ──────────────────
    # A wrap occurs when phase drops by more than 2.5 radians in one step
    dphase = np.diff(phase)
    wrap_indices = np.where(dphase < -2.5)[0]

    if len(wrap_indices) < 2:
        print(f"  Only {len(wrap_indices)} wraps found — not enough cycles")
        pd.DataFrame(columns=["Start", "End"]).to_csv(output_path, index=False)
        return 0

    # ── 2. Score each cycle (segment between consecutive wraps) ─────
    # A good cycle: phase ramps up smoothly from near −π to near π
    cycle_good = []
    cycle_starts = []
    cycle_ends = []

    for i in range(len(wrap_indices) - 1):
        start_idx = wrap_indices[i] + 1  # frame after the wrap
        end_idx = wrap_indices[i + 1]    # frame of the next wrap

        if end_idx - start_idx < 3:
            cycle_good.append(False)
            cycle_starts.append(frames[start_idx])
            cycle_ends.append(frames[end_idx])
            continue

        seg = phase[start_idx:end_idx + 1]

        # Check amplitude (should span most of [−π, π])
        amplitude = seg.max() - seg.min()
        if amplitude < min_amplitude:
            cycle_good.append(False)
            cycle_starts.append(frames[start_idx])
            cycle_ends.append(frames[end_idx])
            continue

        # Check smoothness via derivative
        dseg = np.diff(seg)
        pos_frac = np.mean(dseg > 0)  # fraction of steps that are increasing

        # Most steps should be positive (ramping up)
        if pos_frac < 0.6:
            cycle_good.append(False)
            cycle_starts.append(frames[start_idx])
            cycle_ends.append(frames[end_idx])
            continue

        # Jitter: coefficient of variation of positive derivatives
        pos_derivs = dseg[dseg > 0]
        if len(pos_derivs) > 1:
            cv = pos_derivs.std() / (pos_derivs.mean() + 1e-10)
        else:
            cv = 0
        is_smooth = cv < max_jitter

        cycle_good.append(is_smooth)
        cycle_starts.append(frames[start_idx])
        cycle_ends.append(frames[end_idx])

    cycle_good = np.array(cycle_good)
    cycle_starts = np.array(cycle_starts)
    cycle_ends = np.array(cycle_ends)

    n_good = cycle_good.sum()
    print(f"  {len(cycle_good)} cycles found, {n_good} pass smoothness filter")

    if n_good == 0:
        pd.DataFrame(columns=["Start", "End"]).to_csv(output_path, index=False)
        return 0

    # ── 3. Group consecutive good cycles into bouts ─────────────────
    good_indices = np.where(cycle_good)[0]
    bouts = []
    bout_start = good_indices[0]
    bout_end = good_indices[0]

    for idx in good_indices[1:]:
        if idx == bout_end + 1:
            bout_end = idx
        else:
            bouts.append((bout_start, bout_end))
            bout_start = idx
            bout_end = idx
    bouts.append((bout_start, bout_end))

    # ── 4. Filter by minimum consecutive cycles ─────────────────────
    intervals = []
    for b_start, b_end in bouts:
        n_cyc = b_end - b_start + 1
        if n_cyc >= min_cycles:
            frame_start = int(cycle_starts[b_start])
            frame_end = int(cycle_ends[b_end])
            intervals.append((frame_start, frame_end))

    # ── 5. Merge intervals separated by small gaps ──────────────────
    if len(intervals) > 1:
        merged = [intervals[0]]
        for start, end in intervals[1:]:
            if start - merged[-1][1] <= gap_merge:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        intervals = merged

    # ── 6. Save ─────────────────────────────────────────────────────
    out_df = pd.DataFrame(intervals, columns=["Start", "End"])
    out_df.to_csv(output_path, index=False)
    print(f"  {len(intervals)} whisking bout(s) saved → {os.path.basename(output_path)}")
    return len(intervals)


def main():
    parser = argparse.ArgumentParser(
        description="Detect smooth sawtooth whisking intervals from phase data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python detect_sawtooth_intervals.py --phase_dir "C:\\...\\102725_1\\phase"
  python detect_sawtooth_intervals.py --phase_file "C:\\...\\1_phase.csv"
        """,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--phase_dir",
                   help="Directory containing *_phase.csv files")
    g.add_argument("--phase_file",
                   help="Single phase CSV file")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same as phase dir)")
    parser.add_argument("--min_cycles", type=int, default=3,
                        help="Minimum consecutive good cycles per bout (default: 3)")
    parser.add_argument("--max_jitter", type=float, default=0.6,
                        help="Max derivative CV for smoothness (default: 0.6)")
    parser.add_argument("--min_amplitude", type=float, default=4.0,
                        help="Min phase range per cycle in radians (default: 4.0)")
    parser.add_argument("--gap_merge", type=int, default=2,
                        help="Merge bouts separated by ≤ N frames (default: 2)")
    args = parser.parse_args()

    if args.phase_file:
        phase_files = [args.phase_file]
        out_dir = args.output_dir or os.path.dirname(args.phase_file)
    else:
        phase_files = sorted(glob.glob(os.path.join(args.phase_dir, "*_phase.csv")))
        if not phase_files:
            raise FileNotFoundError(f"No *_phase.csv files in {args.phase_dir}")
        out_dir = args.output_dir or args.phase_dir

    os.makedirs(out_dir, exist_ok=True)

    total = 0
    for pf in phase_files:
        name = os.path.splitext(os.path.basename(pf))[0]
        whisker_id = name.replace("_phase", "")
        out_name = f"{whisker_id}_sawtooth_intervals.csv"
        out_path = os.path.join(out_dir, out_name)

        print(f"\nProcessing {os.path.basename(pf)}:")
        total += detect_sawtooth_intervals(
            pf, out_path,
            min_cycles=args.min_cycles,
            max_jitter=args.max_jitter,
            min_amplitude=args.min_amplitude,
            gap_merge=args.gap_merge,
        )

    print(f"\nDone. {total} total bout(s) across {len(phase_files)} file(s).")


if __name__ == "__main__":
    main()
