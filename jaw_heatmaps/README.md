# jaw_heatmaps

MATLAB scripts for **jaw-tip** trajectories during licks in **pixel coordinates** (typically 0–256), colored by intra-lick phase (`turbo`, `YDir` reversed). Two experiment families:

| Family | Data root | Lick definition |
|--------|-----------|-----------------|
| **IRt / PCRt BiPoles** (opto) | `C:\Users\wanglab\Desktop\Ina\IRt_BiPoles\`, `PCRt_BiPoles\` | Laser-ON; **first lick per laser pulse** (default) |
| **IRt TeLC** (spontaneous) | `C:\Users\wanglab\Desktop\Ina\IRt_TeLC\` | Side-view Pre sessions; **stereotyped** licks from behavior CSV |

Jaw CSVs are space-delimited: `Frame`, `X`, `Y`, `Probability` (model confidence).

---

## Folder layout

```
jaw_heatmaps/
├── README.md
├── filter_lick_trajectories.m          ← shared jump / hotspot QC
├── draw_phase_line.m                   ← phase-colored polylines (+ optional gap break)
│
├── bipoles_jaw_tip_trajectory_by_session.m
├── bipoles_jaw_tip_trajectory_sideview_combined.m
├── bipoles_jaw_tip_trajectory_sideview_combined_prob080.m
│
├── telc_spontaneous_tip_trajectory_by_session.m
├── telc_spontaneous_tip_trajectory_by_session_prob080.m
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

Phase-colored line via `surface(..., EdgeColor='interp')`. Optional `maxSegGap` breaks the polyline across large spatial gaps (BiPoles after point trimming). Prob080 scripts break on **frame gaps** \(> 1\) when low-probability frames are omitted.

---

## BiPoles (opto)

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

---

## TeLC (spontaneous, Pre side view)

### `telc_spontaneous_tip_trajectory_by_session.m`

Three animals (TeLC08, 09, 11), `*_1_jaw.csv` only.

1. **Stereotyped licks** from `*side_behavior*.csv` (per session):
   - Duration within median \(\pm\) `DURATION_MAD_K` × scaled MAD (`3`)
   - `Tongue_area_Interval Max` not far below median (`AREA_MAD_K = 3`)
2. **Trajectory filter:** `filter_lick_trajectories`, mode `'lick'` (drop whole lick on any bad jump/singleton).

**Output:** `telc_spontaneous_tip_trajectories/<base>_spontaneous_jawtip_traj.svg`

**Run:** `telc_spontaneous_tip_trajectory_by_session`

### `telc_spontaneous_tip_trajectory_by_session_prob080.m`

Same stereotyped lick selection; **no** jump filter. Jaw points require **`Probability >= 0.80`**.

**Output:** `telc_spontaneous_tip_trajectories_prob080/<base>_spontaneous_jawtip_traj_prob080.svg`

**Run:** `telc_spontaneous_tip_trajectory_by_session_prob080`

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

## Lick selection (TeLC)

| Pattern | Jaw file |
|---------|----------|
| `*side_behavior*.csv` | `*_1_jaw.csv` (side) |

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
