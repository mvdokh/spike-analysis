# jaw_heatmaps

MATLAB scripts for **jaw keypoint** analysis during licks: position heatmaps and phase-colored trajectories. Two experiment families are covered:

| Family | Data root | Lick definition |
|--------|-----------|-----------------|
| **IRt TeLC** | `C:\Users\wanglab\Desktop\Ina\IRt_TeLC\` | `*bottom_behavior*` / `*side_behavior*` CSV interval columns |
| **IRt / PCRt BiPoles** (opto) | `C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\`, `PCRt_BiPoles\` | `*_<view>_view_behavior_*.csv`; laser-ON licks by default |

Jaw tracking CSVs are space-delimited with columns `Frame`, `X`, `Y`, `Probability` (pixel coordinates, typically 0–256). Plots use **absolute image coordinates** (not jaw-centered), with `YDir` reversed to match video frames.

---

## Folder layout

```
jaw_heatmaps/
├── README.md                                          ← this file
├── irt_telc_jaw_lick_position_heatmap.m               ← TeLC heatmaps
├── irt_telc_jaw_lick_trajectory_phase_by_session.m      ← TeLC trajectories
├── bipoles_jaw_tip_trajectory_by_session.m            ← BiPoles per-session SVGs
├── bipoles_jaw_tip_trajectory_sideview_combined.m       ← BiPoles combined grids + outlier filter
├── irt_telc_jaw_lick_all_positions.svg                ← TeLC heatmap output (when run)
├── irt_telc_jaw_lick_trajectory_phase_by_session.svg  ← TeLC trajectory output (when run)
└── bipoles_jaw_tip_trajectories/
    ├── bipoles_jaw_tip_trajectory_sideview_combined_IRt_BiPoles.svg
    ├── bipoles_jaw_tip_trajectory_sideview_combined_PCRt_BiPoles.svg
    ├── IRt_BiPoles/
    │   └── IRt_BiPoles_<animal>_<date>_<view>_laserON_jawtip_traj.svg  (×30)
    └── PCRt_BiPoles/
        └── PCRt_BiPoles_<animal>_<date>_<view>_laserON_jawtip_traj.svg  (×20; 4 skipped)
```

---

## MATLAB scripts

### `irt_telc_jaw_lick_position_heatmap.m`

**Purpose:** Gaussian density heatmaps of jaw `(X,Y)` at frames inside behavior-defined lick intervals (all licks, not laser-filtered).

**Layout:** One SVG — rows = animal × view (TeLC08, 09, 11; bottom then side); columns = Pre, then each post session date.

**Output:** `irt_telc_jaw_lick_all_positions.svg`

**Outlier exclusion:** None.

**Run:**
```matlab
cd('...\jaw_heatmaps')
irt_telc_jaw_lick_position_heatmap
```

---

### `irt_telc_jaw_lick_trajectory_phase_by_session.m`

**Purpose:** Intra-lick jaw trajectories colored by phase (0 = lick start, 1 = lick end). Same session grid as the heatmap script.

**Output:** `irt_telc_jaw_lick_trajectory_phase_by_session.svg`

**Outlier exclusion:** None.

**Run:**
```matlab
irt_telc_jaw_lick_trajectory_phase_by_session
```

**Note:** Can be slow (many licks × scatter + line segments). Optional subsampling: `MAX_LICKS_PLOT` in CONFIG.

---

### `bipoles_jaw_tip_trajectory_by_session.m`

**Purpose:** Jaw-**tip** trajectories for **IRt_BiPoles** and **PCRt_BiPoles**, one SVG per session per camera view.

**Lick filter:** `LASER_MODE = 'on'` — only behavior rows with `laser_Interval Overlap Assign ID >= 0`.

**Axes:** Full frame `0–256` × `0–256` (not zoomed to data extent).

**Output:** `bipoles_jaw_tip_trajectories/<IRt_BiPoles|PCRt_BiPoles>/`  
Naming: `<base>_laserON_jawtip_traj.svg`  
Example: `IRt_BiPoles_01_2025_0425_side_view_laserON_jawtip_traj.svg`

**Counts (last run):** 50 SVGs written; 4 skipped (PCRt_08 `2025_0321` and `2025_0326` — jaw CSVs exist but no behavior CSV).

**Outlier exclusion:** None.

**Run:**
```matlab
bipoles_jaw_tip_trajectory_by_session
```

---

### `bipoles_jaw_tip_trajectory_sideview_combined.m`

**Purpose:** Combined **side view** only — one figure per experiment (IRt vs PCRt) as a grid:

- **Rows:** animal (`IRt_01`, `IRt_02`, …)
- **Columns:** session date (chronological); blank tiles if an animal has fewer sessions than the widest row

Same laser-ON licks and 256×256 axes as the per-session script.

**Output:**
- `bipoles_jaw_tip_trajectories/bipoles_jaw_tip_trajectory_sideview_combined_IRt_BiPoles.svg` (5×4 grid)
- `bipoles_jaw_tip_trajectories/bipoles_jaw_tip_trajectory_sideview_combined_PCRt_BiPoles.svg` (4×3 grid)

**Outlier exclusion:** Yes — see [Outlier exclusion](#outlier-exclusion) below.

**Run:**
```matlab
bipoles_jaw_tip_trajectory_sideview_combined
```

---

## Outlier exclusion

Outlier filtering is implemented **only** in `bipoles_jaw_tip_trajectory_sideview_combined.m`. It targets isolated **tracking spikes** (single frames that jump far from neighbors), not legitimate fast jaw motion.

### Algorithm

1. **Step distances**  
   For every consecutive pair of tracked points within each laser-ON lick, compute Euclidean distance in pixels:
   \[
   d_i = \sqrt{(x_{i+1}-x_i)^2 + (y_{i+1}-y_i)^2}
   \]
   Steps are pooled **per animal** (all that animal’s side-view sessions combined).

2. **Robust threshold**  
   On the pooled step distribution for that animal:
   - `median` = median(\(d\))
   - `MAD` = median(\(|d - \text{median}|\))
   - `scaledMAD` = 1.4826 × MAD (normal-consistency factor)
   - **Threshold:** \(T = \text{median} + K \times \text{scaledMAD}\)

3. **Long-tail gate**  
   Removals run **only if** at least one step satisfies \(d > T\). If the distribution has no points above \(T\) (no wide tail), **nothing is removed** for that animal.

4. **Spike point rule** (`spikeKeepMask`, up to `OUTLIER_MAX_ITERS` passes)  
   A point is removed if:
   - **Interior:** distance to **both** neighbors exceeds \(T\)
   - **Endpoint:** the single adjacent step exceeds \(T\)

5. **Post-cleaning**  
   Licks with fewer than `MIN_LICK_FRAMES` (2) tracked frames after cleaning are dropped.

### CONFIG knobs

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `OUTLIER_REMOVAL` | `true` | Master switch |
| `OUTLIER_MAD_K` | `5` | \(K\) for all animals except IRt_09 / IRt_10 |
| `OUTLIER_MAD_K_IRT_09_10` | `10` | Higher \(K\) → higher \(T\) → **fewer** points removed for IRt_09 and IRt_10 |
| `OUTLIER_MAX_ITERS` | `3` | Iterations of spike removal per lick |

### Statistics from last successful run

Values are printed to the MATLAB command window when the combined script runs. Tables below are from that run (June 2026).

#### IRt_BiPoles (side view, laser-ON)

| Animal | Step count | Median (px) | Scaled MAD (px) | Max (px) | K | Threshold T (px) | Steps > T | % above T | Points removed |
|--------|------------|-------------|-----------------|----------|---|------------------|-----------|-----------|----------------|
| IRt_01 | 19,348 | 1.00 | 1.48 | 124.34 | 5 | 8.41 | 243 | 1.26% | (included in total) |
| IRt_02 | 2,967 | 1.00 | 0.61 | 3.61 | 5 | 4.07 | 0 | 0% | 0 (no wide tail) |
| IRt_03 | 2,439 | 1.00 | 0.61 | 13.45 | 5 | 4.07 | 70 | 2.87% | (included in total) |
| IRt_09 | 26,807 | 2.24 | 3.32 | 214.11 | **10** | **35.39** | 2,605 | 9.72% | (included in total) |
| IRt_10 | 24,241 | 1.41 | 1.22 | 195.66 | **10** | **13.60** | 1,575 | 6.50% | (included in total) |
| **Total** | **75,802** | — | — | — | — | — | — | — | **1,740** |

Licks dropped after cleaning: **0**.

**IRt_09 / IRt_10 note:** These animals use `OUTLIER_MAD_K_IRT_09_10 = 10` instead of `5`. For example, IRt_09’s threshold is **35.39 px** vs **~11.9 px** if the whole IRt cohort were pooled with \(K=5\) (an earlier version of the script did that and removed **2,823** points experiment-wide).

#### PCRt_BiPoles (side view, laser-ON)

| Animal | Step count | Median (px) | Scaled MAD (px) | Max (px) | K | Threshold T (px) | Steps > T | % above T | Points removed |
|--------|------------|-------------|-----------------|----------|---|------------------|-----------|-----------|----------------|
| PCRt_02 | 6,082 | 1.00 | 1.48 | 70.88 | 5 | 8.41 | 17 | 0.28% | (included in total) |
| PCRt_07 | 7,174 | 2.00 | 1.48 | 100.66 | 5 | 9.41 | 713 | 9.94% | (included in total) |
| PCRt_08 | 3,245 | 1.00 | 0.61 | 15.56 | 5 | 4.07 | 78 | 2.40% | (included in total) |
| PCRt_09 | 13,787 | 2.00 | 1.48 | 160.03 | 5 | 9.41 | 1,408 | 10.21% | (included in total) |
| **Total** | **30,288** | — | — | — | — | — | — | — | **949** |

Licks dropped after cleaning: **1**.

#### Reference: earlier pooled IRt threshold (superseded)

Before per-animal thresholds and the IRt_09/10 relaxation, all IRt side-view steps were pooled and filtered with \(K=5\):

| Metric | Value |
|--------|-------|
| Pooled step count | 75,802 |
| Median | 1.41 px |
| Scaled MAD | 2.10 px |
| Max | 214.11 px |
| Threshold \(T\) | 11.90 px |
| Steps above \(T\) | 6,473 (8.54%) |
| Points removed | 2,823 |

---

## BiPoles behavior CSV (for lick + laser filtering)

Files: `*_<view>_view_behavior_100_3.csv` (comma-delimited).

| Column | Use |
|--------|-----|
| `Tongue_area_interval_detection_Interval Start` / `End` | Inclusive frame indices for each lick |
| `laser_Interval Overlap Assign ID` | `-1` = no laser overlap; `>= 0` = laser-ON lick |
| `laser_Interval Overlap Assign Start` / `End` | Laser on/off frames (not used for plotting by default) |

Read with `'VariableNamingRule', 'preserve'`.

---

## TeLC behavior CSV

| Pattern | View |
|---------|------|
| `*bottom_behavior*.csv` | Bottom jaw (`*__jaw.csv`) |
| `*side_behavior*.csv` | Side jaw (`*_1_jaw.csv`) |

Lick columns: `Tongue_area_interval_detection_Interval Start` / `End`.

---

## Requirements

- MATLAB R2020a+ (`exportgraphics` for SVG)
- `turbo` colormap preferred for trajectories (falls back to `jet`)

---

## Related code (not in this folder)

- `Tongue_Tip_Heatmaps/combined_keypoint_heatmaps_by_type.m` — heatmap style reference
- `Tongue_Tip_Heatmaps/lick_trajectory_phase_density_overlay_by_animal.m` — trajectory style reference

---

*Last updated: June 2026 — reflects per-animal outlier thresholds and IRt_09/10 relaxed K=10.*
