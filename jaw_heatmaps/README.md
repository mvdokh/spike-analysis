# jaw_heatmaps

MATLAB scripts for **jaw-tip** trajectories during licks in **pixel coordinates** (typically 0–256), colored by intra-lick phase (`turbo`, `YDir` reversed). Two experiment families:

| Family | Data root | Lick definition |
|--------|-----------|-----------------|
| **IRt / PCRt BiPoles** (opto) | `C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\`, `PCRt_BiPoles\` | Laser-ON; **first lick per laser pulse** (default) |
| **IRt TeLC** (spontaneous) | `C:\Users\wanglab\Desktop\Ina\IRt_TeLC\IRt_TeLC08\` (etc.) | `IRt_TeLC##_Pre` / `Post`; side view `*_1_jaw.csv` in Pre |

Jaw CSVs are space-delimited: `Frame`, `X`, `Y`, `Probability` (model confidence).

**Plot coordinates:** trajectory scripts center on the session mean jaw `(X,Y)` (resting position, same idea as tongue plots centered on jaw mean). Axes are **100×100** pixels (`PLOT_HALF = 50`, range ±50). Polylines are **clipped** at the window edge; scatter points outside the window are omitted. A small **+** marks the origin (jaw rest).

**Lick intervals:** jaw points are taken from `Start − LICK_FRAME_PAD` through `End + LICK_FRAME_PAD` (default **10** frames) to compensate for tongue-area–based behavior CSV timing. Shared helper: `extractJawLickTrajectories.m`.

---

## Folder layout

```
jaw_heatmaps/
├── README.md
├── filter_lick_trajectories.m          ← shared jump / hotspot QC
├── draw_phase_line.m                   ← phase-colored polylines (+ optional gap break)
├── telc_pre_side_jaw_paths.m           ← resolve Pre *_1_jaw.csv under IRt_TeLC##/
├── jawSessionRestXY.m                  ← session mean jaw position (rest)
├── centerLickCells.m                   ← subtract rest -> origin
├── drawJawRestMarker.m                 ← small + at (0,0)
├── extractJawLickTrajectories.m        ← lick intervals + frame pad
├── readFirstLickPerBoutIntervals.m    ← TeLC: first lick per bout (group Assign ID)
├── readSpontaneousLickIntervals.m     ← all TeLC licks (unused by default scripts)
├── clipSegmentToSquare.m               ← clip line segments to plot window
├── filterScatterToSquare.m             ← drop scatter outside window
├── draw_phase_line_frame_gaps.m        ← phase line with frame-gap breaks
├── setupCenteredJawAxes.m              ← 100x100 axes (+/-50 px)
│
├── bipoles_jaw_tip_trajectory_by_session.m
├── bipoles_jaw_tip_trajectory_sideview_combined.m
├── bipoles_jaw_tip_trajectory_sideview_combined_prob080.m
├── bipoles_jaw_tip_one_random_lick_per_animal.m
│
├── telc_spontaneous_tip_trajectory_by_session.m
├── telc_spontaneous_tip_trajectory_by_session_prob080.m
├── telc_spontaneous_one_random_lick_per_animal.m
│
├── bipoles_jaw_tip_trajectories/           ← MAD / hotspot filter (default BiPoles)
├── bipoles_jaw_tip_trajectories_prob080/   ← prob >= 0.80 only
└── telc_spontaneous_tip_trajectories/
    └── telc_spontaneous_tip_trajectories_prob080/
```

Generated `.svg` files are gitignored at repo root (`*.svg`).

---

## Shared helpers

### `filter_lick_trajectories.m`

Session-scoped quality control on trajectory point lists. Used by BiPoles scripts (and TeLC default script) unless noted.

**Step threshold** (Iglewicz–Hoaglin style, upper tail only):

\[
T = \mathrm{median}(d) + K \times 1.4826 \times \mathrm{MAD}(d)
\]

where \(d\) is the Euclidean step between consecutive frames in a lick. Optional caps:

- `stepHardMax` — also treat steps with \(d >\) `stepHardMax` as large (default BiPoles: 20 px).
- `hotspotMinCount` / `hotspotPurgeCount` — frequent rounded `(X,Y)` pile-ups (corner glitches).

**Modes:**

| `filterMode` | Behavior | Used by |
|--------------|----------|---------|
| `'points'` | Remove bad **points**; drop lick only if fewer than `MIN_LICK_FRAMES` remain | BiPoles |
| `'lick'` | Drop the **entire lick** if any point would be removed | TeLC default |

**Point rules (`'points'`):** singleton session coordinates; both endpoints of steps \(> T\); interior spikes (large step on both sides); full runs of identical rounded `(X,Y)` entered/exited by a large step; frequent-coordinate bracket/purge. Iterative passes until stable.

### `draw_phase_line.m`

Phase-colored line (per-segment `plot` colors for reliable SVG export). Optional `maxSegGap` breaks the polyline across large spatial gaps (BiPoles after point trimming). Optional `clipHalf` clips each segment to `[-clipHalf, clipHalf]`. Prob080 scripts use `draw_phase_line_frame_gaps.m` to break on **frame gaps** \(> 1\) when low-probability frames are omitted.

---

## BiPoles (opto)

Figure captions use **Jaw trajectory during laser-ON lick** (per-session, combined, and random-lick figures).

### `bipoles_jaw_tip_trajectory_by_session.m`

One SVG per session × view (side + bottom). Full **256×256** axes.

| Setting | Default |
|---------|---------|
| `LASER_MODE` | `'on'` |
| `FIRST_LICK_PER_LASER` | `true` |
| `PROB_MIN` | `0` (no probability filter) |
| `TRAJECTORY_FILTER` | `true`, mode `'points'` |
| `TRAJECTORY_STEP_MAD_K` | `5` |
| `TRAJECTORY_STEP_HARD_MAX` | `20` |
| `TRAJECTORY_HOTSPOT_MIN_COUNT` | `20` |
| `TRAJECTORY_HOTSPOT_PURGE_COUNT` | `50` |
| `TRAJECTORY_LINE_BREAK_MAX` | `20` (do not draw across trimmed gaps) |

**Output:** `bipoles_jaw_tip_trajectories/<IRt_BiPoles|PCRt_BiPoles>/`  
`IRt_BiPoles_<animal>_<date>_<view>_laserON_jawtip_traj.svg`

**Run:** `bipoles_jaw_tip_trajectory_by_session`  
**Last run:** 50 SVGs; 4 skipped (PCRt_08 `2025_0321` / `2025_0326` — no behavior CSV).

### `bipoles_jaw_tip_trajectory_sideview_combined.m`

Side view only. Grid: **rows = animal**, **columns = session** (chronological). Same lick rules and `filter_lick_trajectories` as per-session (session-pooled steps per tile, not per animal).

**Output:**

- `bipoles_jaw_tip_trajectories/bipoles_jaw_tip_trajectory_sideview_combined_IRt_BiPoles.svg`
- `bipoles_jaw_tip_trajectories/bipoles_jaw_tip_trajectory_sideview_combined_PCRt_BiPoles.svg`

**Run:** `bipoles_jaw_tip_trajectory_sideview_combined`

### `bipoles_jaw_tip_trajectory_sideview_combined_prob080.m`

Copy of the combined script with **no** `filter_lick_trajectories`. Only jaw points with **`Probability >= 0.80`** inside each lick interval. Lines break where frame index skips.

**Output:** `bipoles_jaw_tip_trajectories_prob080/bipoles_jaw_tip_trajectory_sideview_combined_prob080_<experiment>.svg`

**Run:** `bipoles_jaw_tip_trajectory_sideview_combined_prob080`

### `bipoles_jaw_tip_one_random_lick_per_animal.m`

Side view only. **`N_SIMILAR_LICKS` (default 5) most shape-similar laser-ON licks per animal** from the pool of **first lick per laser interval** (`selectMostSimilarLickIndices.m`). Overlaid per panel from all side sessions. **Animals plotted:** IRt_01–03 and PCRt_02, 07, 08 only. No trajectory or probability filter. **PCHIP + movavg** display smoothing (`SMOOTH_N_POINTS = 256`, `SMOOTH_MOVAVG_WIN = 7`); lines only (no scatter).

Per-animal panel stats: **path length (arc)** and **max distance from start** (mean ± SD over selected licks). See [`README_random_lick_figures.md`](README_random_lick_figures.md).

**Full pipeline (±10 frames, centering, metrics):** see [`README_random_lick_figures.md`](README_random_lick_figures.md).

**Output:** `bipoles_jaw_tip_trajectories/bipoles_jaw_tip_one_random_lick_per_animal_<IRt|PCRt>_BiPoles.svg`

**Run:** `bipoles_jaw_tip_one_random_lick_per_animal`

---

## TeLC (spontaneous, Pre side view)

### `telc_spontaneous_tip_trajectory_by_session.m`

Three animals (TeLC08, 09, 11), `*_1_jaw.csv` only.

1. **First lick per bout** from `*side_behavior*.csv` (`readFirstLickPerBoutIntervals.m`, column `...group_intervals_Interval Overlap Assign ID`).
2. **Trajectory filter:** `filter_lick_trajectories`, mode `'lick'` (drop whole lick on any bad jump/singleton).

Figure captions: **Jaw trajectory during spontaneous lick**.

**Output:** `telc_spontaneous_tip_trajectories/<base>_spontaneous_jawtip_traj.svg`

**Run:** `telc_spontaneous_tip_trajectory_by_session`

### `telc_spontaneous_tip_trajectory_by_session_prob080.m`

Same spontaneous lick selection; **no** jump filter. Jaw points require **`Probability >= 0.80`**. Caption: **Jaw trajectory during spontaneous lick**.

**Output:** `telc_spontaneous_tip_trajectories_prob080/<base>_spontaneous_jawtip_traj_prob080.svg`

**Run:** `telc_spontaneous_tip_trajectory_by_session_prob080`

### `telc_spontaneous_one_random_lick_per_animal.m`

Pre side view only. **`N_SIMILAR_LICKS` (default 5) most shape-similar first-in-bout spontaneous licks per animal** (TeLC08, 09, 11). PCHIP + movavg smoothing; phase-colored lines only. One figure, three panels. Details: [`README_random_lick_figures.md`](README_random_lick_figures.md).

**Full pipeline:** [`README_random_lick_figures.md`](README_random_lick_figures.md).

**Output:** `telc_spontaneous_tip_trajectories/telc_spontaneous_one_random_lick_per_animal.svg`

**Run:** `telc_spontaneous_one_random_lick_per_animal`

---

## Lick selection (BiPoles)

Behavior: `*_<view>_view_behavior_100_3.csv`.

| Column | Use |
|--------|-----|
| `Tongue_area_interval_detection_Interval Start` / `End` | Lick frame span |
| `laser_Interval Overlap Assign ID` | `>= 0` = laser-ON; group pulses |
| `laser_Interval Overlap Assign Start` / `End` | Reference (grouping uses assign ID) |

With `FIRST_LICK_PER_LASER = true`, only the row with the **minimum** `Interval Start` per assign ID is kept. Set `FIRST_LICK_PER_LASER = false` to plot all laser-ON licks.

---

## TeLC folder layout

```
IRt_TeLC/
├── IRt_TeLC08/
│   ├── IRt_TeLC08_Pre/          ← jaw + behavior CSVs for Pre
│   └── IRt_TeLC08_Post/
│       └── IRt_TeLC08_post_2026_04_05/   ← one subfolder per post day
├── IRt_TeLC09/
└── IRt_TeLC11/
```

`telc_pre_side_jaw_paths()` finds `*_1_jaw.csv` in each `IRt_TeLC##_Pre` folder.

## Lick selection (TeLC)

| Pattern | Jaw file |
|---------|----------|
| `*side_behavior*.csv` | `*_1_jaw.csv` (side), same folder as jaw CSV |

Stereotyped filters use the same MAD band helpers as in the script (duration central band + area lower fence).

---

## Choosing a quality-control strategy

| Goal | Script family | QC |
|------|---------------|-----|
| Remove tracking spikes / corner hotspots | BiPoles default, TeLC default | `filter_lick_trajectories` |
| Trust model confidence only | `*_prob080.m` | `PROB_MIN = 0.80` on `Probability` |
| Both | Not implemented | Set `PROB_MIN` and keep `TRAJECTORY_FILTER` in a custom CONFIG |

Adjust cutoffs in each script’s CONFIG block. Filter stats print to the MATLAB command window on run.

---

## Requirements

- MATLAB R2020a+ (`exportgraphics` for SVG)
- `turbo` colormap (falls back to `jet`)

---

## Related code (elsewhere in repo)

- `Tongue_Tip_Heatmaps/lick_trajectory_first_lick_per_laser_interval_by_animal.m` — tongue tip (jaw-centered), no trajectory outlier filter
- `Tongue_Tip_Heatmaps/lick_trajectory_phase_density_overlay_by_animal.m` — pooled tongue trajectories

---

*Last updated: June 2026 — `filter_lick_trajectories` (points vs lick); BiPoles hotspot/stutter rules; prob080 variants.*
