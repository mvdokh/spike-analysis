# Whisk-in-Air Phase Radar

Assigns neural spikes to Hilbert phase bins during hand-picked whisking intervals and produces polar bar-chart "phase tuning curves" per unit.

## Inputs

| File | Contents |
|------|----------|
| `c2_hilbert_phase.csv` | Per-frame Hilbert phase (radians). Columns: `Time` (frame), `Data` (phase) |
| `whisk_in_air_c2.csv` | Hand-picked whisking intervals. Columns: `Start`, `End` (frame numbers) |
| `spikes.csv` | Spike times — no header, columns: time (s), unit ID, ignored |
| `digitalin.dat` | 30 kHz Intan binary; channel 1 rising edges = video frame sync TTLs |

## Frame → Time Alignment

Uses the same TTL sync method as `Contact_PSTH/contact_psth.py`:

1. Load `digitalin.dat` as uint16; extract rising edges on bit 1 → `frame_samples[]`
2. `time_s = frame_samples[frame_index] / 30000`

No FPS-based approximation — every frame is anchored to the master 30 kHz clock.

## What the Script Does

1. **Load** all inputs; print column names and shape for quick sanity checks.
2. **Filter** phase frames to whisking intervals only.
3. **Normalize phase per interval** — each interval's phase is independently min-max rescaled so its own minimum → −π and maximum → +π. This corrects for partial cycles or amplitude variation so every interval is comparable on the same [−π, +π] scale.
4. **Assign spikes to phase bins** — for each spike, find the nearest valid whisking frame within ±1 frame (±2 ms at 500 fps). Spikes with no nearby frame are discarded. Bin width: ~24° (15 bins over 360°).
5. **Compute spike rate** — divide spike counts by bin occupancy time (total seconds the animal spent in each phase bin) to get Hz.
6. **Save one PNG per unit** — side-by-side polar bar charts: left = raw phase, right = per-interval normalized phase.
7. **Save summary CSV** — spike counts and rates per unit per bin.

## Outputs

```
<session_dir>/phase_radar/
    unit_<id>_phase_radar.png   — one per unit with spikes in whisking intervals
    phase_spike_summary.csv     — unit_id, bin_center_rad, raw/norm spike count & rate
```

## Configuration (top of script)

```python
SESSION_DIR      = r"..."          # session folder containing digitalin.dat and spikes.csv
HILBERT_PHASE_CSV = ...
INTERVALS_CSV    = ...
SAMPLING_RATE    = 30_000          # master clock Hz
SYNC_CHANNEL     = 1               # bit index of frame-sync TTL
VIDEO_FPS        = 500             # used only for spike-tolerance window (±1/fps s)
N_BINS           = 15              # ~24° per bin
```

## Dependencies

`numpy`, `pandas`, `matplotlib`, `scipy`
Plus `binary_data` and `ttls` from `../Spike PSTH Pipeline/`.
