"""
Unit Analysis Suite
===================

Comprehensive per-session analyses of whisker-evoked neural responses.
Reads raw data (spikes.csv, digitalin.dat, per_whisker_contact/) and
the pre-computed PSTH CSV to produce:

    1. Trial-by-trial reliability  – Fano factor & coefficient of variation
    2. Temporal DSI                – Direction selectivity over time
    3. Adaptation                  – Evoked FR across successive trials
    4. Response classification     – Excited / inhibited / non-responsive labels
    5. Cross-session summary       – Aggregate tuning across sessions
    6. Mutual information          – Bits of whisker identity per unit
    7. Receptive field map         – Spatial heatmap of response magnitude
    8. ISI analysis                – Inter-spike interval distributions

Usage
-----
Single session (analyses 1-4, 6-8):
    python unit_analysis_suite.py --data_dir <session>

Cross-session summary (analysis 5):
    python unit_analysis_suite.py --cross_session <dir1> <dir2> ...

All outputs are saved to <session>/contact_psth_csv_output/analysis_suite/
"""

import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu, zscore

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Import shared helpers from contact_psth.py ───────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contact_psth import (
    load_frame_sync,
    frame_to_seconds,
    load_contact_intervals,
    load_spikes,
    align_spikes_to_events,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def _parse_interval_name(name: str):
    """Return (whisker: int | None, direction: str)."""
    if name == "contact_intervals":
        return (None, "all")
    m = re.match(
        r"interval_(\d+)_mask_contact(?:_(protraction|retraction))?$", name
    )
    if m:
        return (int(m.group(1)), m.group(2) if m.group(2) else "all")
    return (None, "unknown")


def _load_trial_data(data_dir, contact_dir=None, sampling_rate=30_000,
                     sync_channel=1, pre_ms=50.0, post_ms=100.0):
    """
    Load raw data and build a trial-level table.

    Returns
    -------
    trial_df : pd.DataFrame
        Columns: unit, whisker, direction, trial_idx, n_spikes,
                 spike_times_ms (list), duration_ms
    whiskers : sorted list of int
    units    : sorted list of int
    """
    digitalin_path = os.path.join(data_dir, "digitalin.dat")
    spikes_path = os.path.join(data_dir, "spikes.csv")
    if contact_dir is None:
        contact_dir = os.path.join(data_dir, "per_whisker_contact")

    frame_samples = load_frame_sync(digitalin_path, channel=sync_channel,
                                    sampling_rate=sampling_rate)
    spikes_df = load_spikes(spikes_path)
    units = sorted(spikes_df["unit"].unique())

    # Discover contact CSVs
    csv_files = sorted(glob.glob(os.path.join(contact_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {contact_dir}")

    pre_s = pre_ms / 1000.0
    post_s = post_ms / 1000.0

    # Pre-compute events per CSV
    csv_info = []  # (basename, whisker, direction, starts_s, ends_s)
    for csv_path in csv_files:
        basename = os.path.splitext(os.path.basename(csv_path))[0]
        whisker, direction = _parse_interval_name(basename)
        if whisker is None:
            continue
        intervals_df = load_contact_intervals(csv_path)
        if len(intervals_df) == 0:
            continue
        starts_s = frame_to_seconds(intervals_df["Start"].values,
                                    frame_samples, sampling_rate)
        ends_s = frame_to_seconds(intervals_df["End"].values,
                                  frame_samples, sampling_rate)
        csv_info.append((basename, whisker, direction, starts_s, ends_s))

    rows = []
    for unit in units:
        unit_spikes = spikes_df.loc[spikes_df["unit"] == unit, "time"].values
        for basename, whisker, direction, starts_s, ends_s in csv_info:
            trials = align_spikes_to_events(unit_spikes, starts_s, ends_s,
                                            pre_s, post_s)
            for ti, trial in enumerate(trials):
                spike_ms = np.array(trial["spike_times_ms"])
                # Count spikes in the response window [0, post_ms)
                resp_spikes = spike_ms[(spike_ms >= 0) & (spike_ms < post_ms)]
                bl_spikes = spike_ms[(spike_ms >= -pre_ms) & (spike_ms < 0)]
                rows.append({
                    "unit": unit,
                    "whisker": whisker,
                    "direction": direction,
                    "trial_idx": ti,
                    "n_spikes_resp": len(resp_spikes),
                    "n_spikes_bl": len(bl_spikes),
                    "spike_times_ms": trial["spike_times_ms"],
                    "duration_ms": trial["duration_ms"],
                })

    trial_df = pd.DataFrame(rows)
    whiskers = sorted(trial_df["whisker"].unique())
    return trial_df, whiskers, units


# ══════════════════════════════════════════════════════════════════════════════
#  1. Trial-by-trial reliability  (Fano factor)
# ══════════════════════════════════════════════════════════════════════════════

def compute_fano(trial_df, whiskers, units):
    """
    Compute Fano factor (var/mean of spike counts) and CV for each
    unit × whisker × direction.
    """
    rows = []
    for unit in units:
        for w in whiskers:
            for d in ["all", "retraction", "protraction"]:
                sub = trial_df[(trial_df["unit"] == unit) &
                               (trial_df["whisker"] == w) &
                               (trial_df["direction"] == d)]
                counts = sub["n_spikes_resp"].values
                if len(counts) < 3:
                    continue
                mean_c = counts.mean()
                var_c = counts.var(ddof=1)
                fano = var_c / mean_c if mean_c > 0 else np.nan
                cv = counts.std(ddof=1) / mean_c if mean_c > 0 else np.nan
                rows.append({
                    "unit": unit, "whisker": w, "direction": d,
                    "mean_count": mean_c, "var_count": var_c,
                    "fano_factor": fano, "cv": cv, "n_trials": len(counts),
                })
    return pd.DataFrame(rows)


def plot_fano(fano_df, whiskers, units, out_dir):
    """Heatmap of Fano factor (unit × whisker, direction=all)."""
    sub = fano_df[fano_df["direction"] == "all"]
    if len(sub) == 0:
        print("  No Fano data for direction=all, skipping plot.")
        return

    matrix = np.full((len(units), len(whiskers)), np.nan)
    for _, row in sub.iterrows():
        ui = units.index(row["unit"])
        wi = whiskers.index(row["whisker"])
        matrix[ui, wi] = row["fano_factor"]

    fig, ax = plt.subplots(figsize=(6, max(3, len(units) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm",
                   vmin=0, vmax=np.nanmax(matrix.clip(0, 5)),
                   interpolation="nearest")
    ax.set_xticks(range(len(whiskers)))
    ax.set_xticklabels([f"W{w}" for w in whiskers])
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels([f"U{u}" for u in units], fontsize=7)
    ax.set_xlabel("Whisker")
    ax.set_ylabel("Unit")
    ax.set_title("Fano Factor (spike count variance / mean)")
    for ui in range(len(units)):
        for wi in range(len(whiskers)):
            v = matrix[ui, wi]
            if not np.isnan(v):
                colour = "white" if v > np.nanmax(matrix) * 0.6 else "black"
                ax.text(wi, ui, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color=colour)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Fano factor")
    ax.axhline(-0.5, color="gray", lw=0.5)
    fig.tight_layout()
    path = os.path.join(out_dir, "fano_factor_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  2. Temporal DSI
# ══════════════════════════════════════════════════════════════════════════════

def compute_temporal_dsi(psth_df, whiskers, units,
                         win_ms=10, step_ms=5, post_ms=100):
    """
    Compute DSI in sliding windows after contact onset.

    Uses the averaged PSTH (retraction vs protraction) in each time window.
    """
    rows = []
    for unit in units:
        for w in whiskers:
            ret_sub = psth_df[
                (psth_df["unit"] == unit) &
                (psth_df["interval"] == f"interval_{w}_mask_contact_retraction")
            ].sort_values("bin_ms")
            pro_sub = psth_df[
                (psth_df["unit"] == unit) &
                (psth_df["interval"] == f"interval_{w}_mask_contact_protraction")
            ].sort_values("bin_ms")

            if len(ret_sub) == 0 or len(pro_sub) == 0:
                continue

            ret_bins = ret_sub["bin_ms"].values
            ret_fr = ret_sub["firing_rate_hz"].values
            pro_bins = pro_sub["bin_ms"].values
            pro_fr = pro_sub["firing_rate_hz"].values

            t_start = 0
            while t_start + win_ms <= post_ms:
                t_end = t_start + win_ms
                rmask = (ret_bins >= t_start) & (ret_bins < t_end)
                pmask = (pro_bins >= t_start) & (pro_bins < t_end)
                r_mean = ret_fr[rmask].mean() if rmask.sum() > 0 else 0
                p_mean = pro_fr[pmask].mean() if pmask.sum() > 0 else 0
                denom = r_mean + p_mean
                dsi = (r_mean - p_mean) / denom if denom > 0 else 0.0
                rows.append({
                    "unit": unit, "whisker": w,
                    "t_centre_ms": t_start + win_ms / 2,
                    "fr_ret": r_mean, "fr_pro": p_mean, "dsi": dsi,
                })
                t_start += step_ms
    return pd.DataFrame(rows)


def plot_temporal_dsi(tdsi_df, whiskers, units, out_dir):
    """One panel per unit showing DSI over time for each whisker."""
    n_units = len(units)
    ncols = min(4, n_units)
    nrows = int(np.ceil(n_units / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             squeeze=False, sharey=True)
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(whiskers), 1)))

    for idx, unit in enumerate(units):
        ax = axes[idx // ncols, idx % ncols]
        for wi, w in enumerate(whiskers):
            sub = tdsi_df[(tdsi_df["unit"] == unit) &
                          (tdsi_df["whisker"] == w)].sort_values("t_centre_ms")
            if len(sub) == 0:
                continue
            ax.plot(sub["t_centre_ms"], sub["dsi"], color=colours[wi],
                    marker=".", markersize=3, lw=1.2, label=f"W{w}")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_title(f"U{unit}", fontsize=9)
        ax.set_ylim(-1.1, 1.1)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("DSI", fontsize=9)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Time (ms)", fontsize=9)

    # Hide unused axes
    for idx in range(n_units, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7, ncol=2)
    fig.suptitle("Temporal DSI (sliding window)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    path = os.path.join(out_dir, "temporal_dsi.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  3. Adaptation across trials
# ══════════════════════════════════════════════════════════════════════════════

def compute_adaptation(trial_df, whiskers, units, n_bins=5):
    """
    Compare early vs late trial spike counts and fit a linear trend.
    Uses direction=all only.
    """
    rows = []
    for unit in units:
        for w in whiskers:
            sub = trial_df[(trial_df["unit"] == unit) &
                           (trial_df["whisker"] == w) &
                           (trial_df["direction"] == "all")]
            counts = sub.sort_values("trial_idx")["n_spikes_resp"].values
            if len(counts) < 6:
                continue

            half = len(counts) // 2
            early = counts[:half]
            late = counts[half:]

            # Linear regression (trial index vs spike count)
            x = np.arange(len(counts), dtype=float)
            slope, intercept = np.polyfit(x, counts, 1)

            # Adaptation index: (early_mean - late_mean) / (early_mean + late_mean)
            e_mean, l_mean = early.mean(), late.mean()
            denom = e_mean + l_mean
            adapt_idx = (e_mean - l_mean) / denom if denom > 0 else 0.0

            # Statistical test (early vs late)
            if len(early) > 1 and len(late) > 1:
                _, p_val = mannwhitneyu(early, late, alternative="two-sided")
            else:
                p_val = 1.0

            rows.append({
                "unit": unit, "whisker": w,
                "early_mean": e_mean, "late_mean": l_mean,
                "adaptation_index": adapt_idx,
                "slope": slope,
                "p_value": p_val,
                "n_trials": len(counts),
            })
    return pd.DataFrame(rows)


def plot_adaptation(adapt_df, trial_df, whiskers, units, out_dir):
    """Per-unit adaptation curves (spike count vs trial) for best whisker."""
    n_units = len(units)
    ncols = min(4, n_units)
    nrows = int(np.ceil(n_units / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    colours = plt.cm.tab10(np.linspace(0, 1, max(len(whiskers), 1)))

    for idx, unit in enumerate(units):
        ax = axes[idx // ncols, idx % ncols]
        for wi, w in enumerate(whiskers):
            sub = trial_df[(trial_df["unit"] == unit) &
                           (trial_df["whisker"] == w) &
                           (trial_df["direction"] == "all")]
            counts = sub.sort_values("trial_idx")["n_spikes_resp"].values
            if len(counts) < 3:
                continue
            x = np.arange(len(counts))
            ax.scatter(x, counts, s=8, alpha=0.4, color=colours[wi])
            # Running average (window=5 or fewer)
            win = min(5, len(counts))
            if win >= 2:
                running = np.convolve(counts, np.ones(win)/win, mode="valid")
                ax.plot(np.arange(win-1, len(counts)), running,
                        color=colours[wi], lw=1.5, label=f"W{w}")

        ax.set_title(f"U{unit}", fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("Spike count", fontsize=9)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Trial #", fontsize=9)

    for idx in range(n_units, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=7, ncol=2)
    fig.suptitle("Adaptation (spike count vs trial order)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    path = os.path.join(out_dir, "adaptation.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  4. Response classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_responses(trial_df, whiskers, units, alpha=0.01):
    """
    For each unit × whisker × direction, compare response-window spike count
    to baseline spike count across trials using a Wilcoxon signed-rank test.

    Labels: 'excited', 'inhibited', or 'non-responsive'.
    """
    rows = []
    for unit in units:
        for w in whiskers:
            for d in ["all", "retraction", "protraction"]:
                sub = trial_df[(trial_df["unit"] == unit) &
                               (trial_df["whisker"] == w) &
                               (trial_df["direction"] == d)]
                if len(sub) < 5:
                    rows.append({
                        "unit": unit, "whisker": w, "direction": d,
                        "label": "insufficient", "p_value": np.nan,
                        "mean_resp": np.nan, "mean_bl": np.nan,
                        "n_trials": len(sub),
                    })
                    continue

                resp = sub["n_spikes_resp"].values.astype(float)
                bl = sub["n_spikes_bl"].values.astype(float)
                diff = resp - bl

                # Wilcoxon needs at least one non-zero difference
                if np.all(diff == 0):
                    label = "non-responsive"
                    p = 1.0
                else:
                    try:
                        _, p = wilcoxon(diff, alternative="two-sided")
                    except ValueError:
                        p = 1.0
                    if p < alpha:
                        label = "excited" if diff.mean() > 0 else "inhibited"
                    else:
                        label = "non-responsive"

                rows.append({
                    "unit": unit, "whisker": w, "direction": d,
                    "label": label, "p_value": p,
                    "mean_resp": resp.mean(), "mean_bl": bl.mean(),
                    "n_trials": len(sub),
                })
    return pd.DataFrame(rows)


def plot_response_classification(cls_df, whiskers, units, out_dir):
    """Heatmap-style table of response labels (direction=all)."""
    sub = cls_df[cls_df["direction"] == "all"]
    label_to_num = {"excited": 2, "inhibited": -1, "non-responsive": 0,
                    "insufficient": -2}
    label_colors = {2: "#d73027", -1: "#4575b4", 0: "#f0f0f0", -2: "#cccccc"}

    matrix = np.zeros((len(units), len(whiskers)))
    labels_grid = [[""]*len(whiskers) for _ in range(len(units))]
    for _, row in sub.iterrows():
        try:
            ui = units.index(row["unit"])
            wi = whiskers.index(row["whisker"])
        except ValueError:
            continue
        matrix[ui, wi] = label_to_num.get(row["label"], 0)
        labels_grid[ui][wi] = row["label"]

    fig, ax = plt.subplots(figsize=(max(4, len(whiskers)*1.2),
                                    max(3, len(units) * 0.35)))
    # Custom colormap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(["#cccccc", "#4575b4", "#f0f0f0", "#d73027"])
    bounds = [-2.5, -1.5, -0.5, 0.5, 2.5]
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")
    ax.set_xticks(range(len(whiskers)))
    ax.set_xticklabels([f"W{w}" for w in whiskers])
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels([f"U{u}" for u in units], fontsize=7)
    ax.set_xlabel("Whisker")
    ax.set_ylabel("Unit")
    ax.set_title("Response Classification (Wilcoxon, p<0.01)")

    for ui in range(len(units)):
        for wi in range(len(whiskers)):
            lbl = labels_grid[ui][wi]
            short = {"excited": "E", "inhibited": "I",
                     "non-responsive": "NR", "insufficient": "?"}
            colour = "white" if lbl in ["excited", "inhibited"] else "black"
            ax.text(wi, ui, short.get(lbl, ""), ha="center", va="center",
                    fontsize=7, fontweight="bold", color=colour)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d73027", label="Excited"),
        Patch(facecolor="#4575b4", label="Inhibited"),
        Patch(facecolor="#f0f0f0", edgecolor="gray", label="Non-responsive"),
        Patch(facecolor="#cccccc", label="Insufficient trials"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              bbox_to_anchor=(1.02, 1), fontsize=7)

    fig.tight_layout()
    path = os.path.join(out_dir, "response_classification.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  5. Cross-session summary
# ══════════════════════════════════════════════════════════════════════════════

def cross_session_summary(session_dirs, out_dir):
    """
    Aggregate tuning_data.csv across sessions and produce population plots.
    """
    all_tuning = []
    all_class = []
    for sd in session_dirs:
        # Tuning data
        tuning_path = os.path.join(
            sd, "contact_psth_csv_output", "tuning_curves", "tuning_data.csv")
        if os.path.isfile(tuning_path):
            t = pd.read_csv(tuning_path)
            t["session"] = os.path.basename(sd)
            all_tuning.append(t)
        # Classification data
        cls_path = os.path.join(
            sd, "contact_psth_csv_output", "analysis_suite",
            "response_classification.csv")
        if os.path.isfile(cls_path):
            c = pd.read_csv(cls_path)
            c["session"] = os.path.basename(sd)
            all_class.append(c)

    os.makedirs(out_dir, exist_ok=True)

    if not all_tuning:
        print("  No tuning_data.csv files found across sessions. Skipping.")
        return

    tuning = pd.concat(all_tuning, ignore_index=True)
    tuning["uid"] = tuning["session"] + "_U" + tuning["unit"].astype(str)
    tuning.to_csv(os.path.join(out_dir, "cross_session_tuning.csv"),
                  index=False)

    # ── Population tuning curve (mean ± SEM across all units) ──
    sub_all = tuning[tuning["direction"] == "all"]
    whiskers = sorted(sub_all["whisker"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: mean ± SEM evoked FR across units
    ax = axes[0]
    means, sems = [], []
    for w in whiskers:
        vals = sub_all[sub_all["whisker"] == w]["evoked_fr"].values
        means.append(vals.mean())
        sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
    ax.bar(range(len(whiskers)), means, yerr=sems, capsize=4,
           color="steelblue", edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(whiskers)))
    ax.set_xticklabels([f"W{w}" for w in whiskers])
    ax.set_xlabel("Whisker")
    ax.set_ylabel("Mean Evoked FR (Hz)")
    ax.set_title(f"Population tuning ({len(sub_all['uid'].unique())} units, "
                 f"{len(all_tuning)} sessions)")
    ax.axhline(0, color="gray", lw=0.6, ls=":")

    # Right: classification pie chart (if available)
    ax2 = axes[1]
    if all_class:
        cls = pd.concat(all_class, ignore_index=True)
        cls_all = cls[cls["direction"] == "all"]
        counts = cls_all["label"].value_counts()
        colors_pie = {"excited": "#d73027", "inhibited": "#4575b4",
                      "non-responsive": "#bdbdbd", "insufficient": "#f0f0f0"}
        ax2.pie(counts.values,
                labels=counts.index,
                colors=[colors_pie.get(l, "gray") for l in counts.index],
                autopct="%1.0f%%", startangle=140, textprops={"fontsize": 9})
        ax2.set_title("Response Classification (all sessions)")
    else:
        ax2.text(0.5, 0.5, "No classification data\navailable",
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Response Classification")

    fig.suptitle("Cross-Session Summary", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "cross_session_summary.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Best-whisker distribution ──
    fig2, ax3 = plt.subplots(figsize=(6, 4))
    best_w = (sub_all.loc[sub_all.groupby("uid")["evoked_fr"].idxmax()]
              ["whisker"].value_counts().sort_index())
    ax3.bar([f"W{w}" for w in best_w.index], best_w.values,
            color="coral", edgecolor="black")
    ax3.set_xlabel("Best Whisker")
    ax3.set_ylabel("Number of Units")
    ax3.set_title("Best Whisker Distribution (all sessions)")
    fig2.tight_layout()
    path2 = os.path.join(out_dir, "best_whisker_distribution.png")
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {path2}")


# ══════════════════════════════════════════════════════════════════════════════
#  6. Mutual information
# ══════════════════════════════════════════════════════════════════════════════

def compute_mutual_info(trial_df, whiskers, units, n_bins_count=6):
    """
    Compute mutual information between whisker identity and spike count
    (plugin estimator with Panzeri–Treves bias correction).

    Uses direction=all only.
    """
    rows = []
    for unit in units:
        # Gather per-trial spike counts for each whisker
        whisker_counts = {}
        for w in whiskers:
            sub = trial_df[(trial_df["unit"] == unit) &
                           (trial_df["whisker"] == w) &
                           (trial_df["direction"] == "all")]
            whisker_counts[w] = sub["n_spikes_resp"].values

        all_counts = np.concatenate(list(whisker_counts.values()))
        if len(all_counts) == 0:
            continue

        # Discretise spike counts into bins
        max_count = int(all_counts.max())
        if max_count == 0:
            rows.append({"unit": unit, "MI_bits": 0.0, "MI_corrected": 0.0})
            continue

        # Use fixed bins: 0, 1, 2, ..., up to n_bins_count-1, then ≥n_bins_count
        # Ensure edges are strictly monotonic even when max_count < n_bins_count
        upper = max(max_count + 1, n_bins_count + 1)
        edges = list(range(min(n_bins_count, max_count + 1))) + [upper]
        # Deduplicate & sort
        edges = sorted(set(edges))
        n_r = len(edges) - 1  # number of response bins
        n_w = len(whiskers)
        N = len(all_counts)

        # Joint and marginal distributions
        joint = np.zeros((n_w, n_r))
        for wi, w in enumerate(whiskers):
            counts = whisker_counts[w]
            if len(counts) == 0:
                continue
            hist, _ = np.histogram(counts, bins=edges)
            joint[wi, :] = hist

        # Plugin MI
        p_joint = joint / N
        p_w = p_joint.sum(axis=1)      # P(whisker)
        p_r = p_joint.sum(axis=0)      # P(response)

        mi = 0.0
        for wi in range(n_w):
            for ri in range(n_r):
                if p_joint[wi, ri] > 0 and p_w[wi] > 0 and p_r[ri] > 0:
                    mi += p_joint[wi, ri] * np.log2(
                        p_joint[wi, ri] / (p_w[wi] * p_r[ri]))

        # Panzeri-Treves bias correction: bias ≈ (R_s - 1)(S - 1) / (2 N ln2)
        # R_s = number of occupied response bins per stimulus
        R_occupied = sum(1 for ri in range(n_r) if p_r[ri] > 0)
        S_occupied = sum(1 for wi in range(n_w) if p_w[wi] > 0)
        bias = (R_occupied - 1) * (S_occupied - 1) / (2 * N * np.log(2))
        mi_corr = max(mi - bias, 0.0)

        rows.append({"unit": unit, "MI_bits": mi, "MI_corrected": mi_corr})

    return pd.DataFrame(rows)


def plot_mutual_info(mi_df, units, out_dir):
    """Bar chart of mutual information per unit."""
    if len(mi_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(max(5, len(units) * 0.5), 4))
    x = range(len(mi_df))
    ax.bar(x, mi_df["MI_corrected"], color="teal", edgecolor="black",
           alpha=0.8, label="Bias-corrected")
    ax.bar(x, mi_df["MI_bits"], color="teal", edgecolor="black",
           alpha=0.25, label="Plugin")
    ax.set_xticks(x)
    ax.set_xticklabels([f"U{u}" for u in mi_df["unit"]], fontsize=7,
                       rotation=45)
    ax.set_xlabel("Unit")
    ax.set_ylabel("MI (bits)")
    ax.set_title("Mutual Information: Spike Count → Whisker Identity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "mutual_information.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  7. Receptive field map
# ══════════════════════════════════════════════════════════════════════════════


def plot_receptive_field(trial_df, whiskers, units, out_dir,
                         whisker_pos=None):
    """
    Receptive field heatmap — whiskers on y-axis, units on x-axis.

    Values = mean response spike count minus mean baseline spike count
    (baseline window = [-50, -30) ms).  Can be negative (suppression).
    Units sorted by best whisker so same-whisker units are grouped.
    """
    from matplotlib.colors import TwoSlopeNorm
    import matplotlib.gridspec as gridspec

    BL_LO, BL_HI = -50.0, -30.0   # baseline window in ms
    RESP_LO, RESP_HI = 0.0, 5.0   # response window in ms
    BL_DUR_S  = (BL_HI - BL_LO) / 1000.0    # 0.020 s
    RESP_DUR_S = (RESP_HI - RESP_LO) / 1000.0  # 0.005 s

    whiskers = sorted(whiskers)
    n_w = len(whiskers)
    n_u = len(units)

    # ── Collect baseline-subtracted firing rate [whisker × unit] ──────
    # Convert spike counts → Hz in each window, then subtract.
    mat_raw = np.zeros((n_w, n_u))
    for ci, unit in enumerate(units):
        for ri, w in enumerate(whiskers):
            sub = trial_df[(trial_df["unit"] == unit) &
                           (trial_df["whisker"] == w) &
                           (trial_df["direction"] == "all")]
            if len(sub) == 0:
                continue
            spike_lists = sub["spike_times_ms"].values
            # Firing rate (Hz) per trial in each window
            resp_hz = np.array([
                np.sum((np.array(st) >= RESP_LO) & (np.array(st) < RESP_HI))
                / RESP_DUR_S for st in spike_lists
            ])
            bl_hz = np.array([
                np.sum((np.array(st) >= BL_LO) & (np.array(st) < BL_HI))
                / BL_DUR_S for st in spike_lists
            ])
            mat_raw[ri, ci] = np.mean(resp_hz) - np.mean(bl_hz)

    # ── Filter: drop units where max |Δ| across all whiskers <= 0.05 ──
    max_abs = np.max(np.abs(mat_raw), axis=0)          # per unit
    keep_mask = max_abs > 0.06
    keep_idx = np.where(keep_mask)[0]
    if len(keep_idx) == 0:
        print("  No units exceed ±0.05 threshold — skipping RF map.")
        return
    mat_filt = mat_raw[:, keep_idx]
    units_filt = [units[i] for i in keep_idx]
    n_u = len(units_filt)

    # ── Sort kept units by best whisker (then by peak descending) ─────
    best_w_idx = np.argmax(mat_filt, axis=0)
    peak_val   = mat_filt.max(axis=0)
    sort_order = np.lexsort((-peak_val, best_w_idx))
    units_sorted = [units_filt[i] for i in sort_order]
    mat = mat_filt[:, sort_order]

    vmin, vmax = mat.min(), mat.max()
    abs_max = max(abs(vmin), abs(vmax), 1e-6)

    # ── Figure layout (heatmap + colourbar, no bar chart) ─────────────
    fig_w = max(5, 0.55 * n_u + 2.0)
    fig = plt.figure(figsize=(fig_w, n_w * 0.7 + 1.4))
    gs = gridspec.GridSpec(1, 2,
                           width_ratios=[n_u, 0.3],
                           wspace=0.08)
    ax_heat = fig.add_subplot(gs[0])
    ax_cbar = fig.add_subplot(gs[1])

    # ── Heatmap (diverging: blue = suppression, red = excitation) ─────
    cmap = plt.cm.RdBu_r
    # Normalize to [-1, 1] by dividing by the global max absolute value
    abs_max = max(np.abs(mat).max(), 1e-6)
    mat = mat / abs_max
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    ax_heat.imshow(mat, aspect="auto", cmap=cmap, norm=norm,
                   interpolation="nearest")

    # Cell annotations
    for ri in range(n_w):
        for ci in range(n_u):
            val = mat[ri, ci]
            rgba = cmap(norm(val))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "white" if lum < 0.45 else "black"
            ax_heat.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                         fontsize=6.5, fontweight="bold", color=tc)

    # Whisker-group separators
    sorted_best = best_w_idx[sort_order]
    for ci in range(1, n_u):
        if sorted_best[ci] != sorted_best[ci - 1]:
            ax_heat.axvline(ci - 0.5, color="black", linewidth=1.2,
                            zorder=4)

    ax_heat.set_xticks(range(n_u))
    ax_heat.set_xticklabels([f"U{u}" for u in units_sorted], fontsize=7,
                            rotation=45, ha="right")
    ax_heat.set_yticks(range(n_w))
    ax_heat.set_yticklabels([f"W{w}" for w in whiskers], fontsize=9,
                            fontweight="bold")
    ax_heat.set_ylabel("Whisker", fontsize=10)
    ax_heat.tick_params(length=0)
    ax_heat.set_title("Receptive Field Map  (Δ Hz: [0,5) − [−50,−30) ms)",
                      fontsize=11, fontweight="bold", pad=6)

    # ── Colour bar ────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("Δ firing rate (normalized)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    path = os.path.join(out_dir, "receptive_field_map.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  8. ISI analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_isi(trial_df, whiskers, units, burst_threshold_ms=4.0):
    """
    Compute ISI statistics from spikes in the response window.

    Returns a DataFrame with mean ISI, CV of ISI, burst percentage,
    and spike count for each unit × whisker.
    """
    rows = []
    for unit in units:
        for w in whiskers:
            sub = trial_df[(trial_df["unit"] == unit) &
                           (trial_df["whisker"] == w) &
                           (trial_df["direction"] == "all")]
            if len(sub) == 0:
                continue

            # Collect ISIs from response-window spikes across trials
            all_isis = []
            for _, row in sub.iterrows():
                spk = np.array(row["spike_times_ms"])
                resp_spk = np.sort(spk[(spk >= 0) & (spk < 100)])
                if len(resp_spk) >= 2:
                    isis = np.diff(resp_spk)
                    all_isis.extend(isis.tolist())

            all_isis = np.array(all_isis)
            if len(all_isis) == 0:
                rows.append({
                    "unit": unit, "whisker": w,
                    "mean_isi_ms": np.nan, "cv_isi": np.nan,
                    "burst_pct": 0.0, "n_isis": 0,
                })
                continue

            mean_isi = all_isis.mean()
            cv_isi = (all_isis.std() / mean_isi) if mean_isi > 0 else np.nan
            burst_pct = 100 * (all_isis < burst_threshold_ms).sum() / len(all_isis)

            rows.append({
                "unit": unit, "whisker": w,
                "mean_isi_ms": mean_isi,
                "cv_isi": cv_isi,
                "burst_pct": burst_pct,
                "n_isis": len(all_isis),
            })
    return pd.DataFrame(rows)


def plot_isi(isi_df, trial_df, whiskers, units, out_dir):
    """
    Two figures:
      1. Burst % heatmap (unit × whisker)
      2. ISI histograms for each unit (best whisker)
    """
    # ── Burst % heatmap ──────────────────────────────────────────────
    matrix = np.full((len(units), len(whiskers)), np.nan)
    for _, row in isi_df.iterrows():
        try:
            ui = units.index(row["unit"])
            wi = whiskers.index(row["whisker"])
        except ValueError:
            continue
        matrix[ui, wi] = row["burst_pct"]

    fig, ax = plt.subplots(figsize=(max(4, len(whiskers)*1.2),
                                    max(3, len(units) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks(range(len(whiskers)))
    ax.set_xticklabels([f"W{w}" for w in whiskers])
    ax.set_yticks(range(len(units)))
    ax.set_yticklabels([f"U{u}" for u in units], fontsize=7)
    ax.set_xlabel("Whisker")
    ax.set_ylabel("Unit")
    ax.set_title(f"Burst % (ISI < 4 ms)")
    for ui in range(len(units)):
        for wi in range(len(whiskers)):
            v = matrix[ui, wi]
            if not np.isnan(v):
                c = "white" if v > 60 else "black"
                ax.text(wi, ui, f"{v:.0f}", ha="center", va="center",
                        fontsize=6, color=c)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Burst %")
    fig.tight_layout()
    path = os.path.join(out_dir, "burst_percentage_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── ISI histograms (best whisker per unit) ───────────────────────
    n_units = len(units)
    ncols = min(4, n_units)
    nrows = int(np.ceil(n_units / ncols))
    fig2, axes = plt.subplots(nrows, ncols,
                              figsize=(3.5 * ncols, 2.5 * nrows),
                              squeeze=False)

    for idx, unit in enumerate(units):
        ax = axes[idx // ncols, idx % ncols]
        # Find best whisker (highest mean spike count)
        best_w = None
        best_mean = -1
        for w in whiskers:
            sub = trial_df[(trial_df["unit"] == unit) &
                           (trial_df["whisker"] == w) &
                           (trial_df["direction"] == "all")]
            m = sub["n_spikes_resp"].mean() if len(sub) else 0
            if m > best_mean:
                best_mean = m
                best_w = w

        if best_w is None:
            ax.set_visible(False)
            continue

        # Collect ISIs for best whisker
        sub = trial_df[(trial_df["unit"] == unit) &
                       (trial_df["whisker"] == best_w) &
                       (trial_df["direction"] == "all")]
        all_isis = []
        for _, row in sub.iterrows():
            spk = np.array(row["spike_times_ms"])
            resp_spk = np.sort(spk[(spk >= 0) & (spk < 100)])
            if len(resp_spk) >= 2:
                all_isis.extend(np.diff(resp_spk).tolist())

        if all_isis:
            all_isis = np.array(all_isis)
            bins_hist = np.arange(0, min(all_isis.max() + 2, 52), 1)
            ax.hist(all_isis, bins=bins_hist, color="steelblue",
                    edgecolor="black", linewidth=0.3, alpha=0.8)
            ax.axvline(4, color="red", ls="--", lw=0.8, alpha=0.7,
                       label="4 ms (burst)")
            burst = 100 * (all_isis < 4).sum() / len(all_isis)
            ax.set_title(f"U{unit} (W{best_w}, burst={burst:.0f}%)",
                         fontsize=8)
        else:
            ax.set_title(f"U{unit} (no ISIs)", fontsize=8)

        ax.tick_params(labelsize=6)
        if idx % ncols == 0:
            ax.set_ylabel("Count", fontsize=8)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("ISI (ms)", fontsize=8)

    for idx in range(n_units, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig2.suptitle("ISI Histograms (best whisker, response window)",
                  fontsize=11, fontweight="bold")
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    path2 = os.path.join(out_dir, "isi_histograms.png")
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {path2}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_single_session(data_dir, contact_dir=None, output_dir=None):
    """Run all single-session analyses."""

    csv_path = os.path.join(data_dir, "contact_psth_csv_output",
                            "contact_psth_firing_rates.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"PSTH CSV not found: {csv_path}\n"
            "Run contact_psth_to_csv.py first.")

    if output_dir is None:
        output_dir = os.path.join(data_dir, "contact_psth_csv_output",
                                  "analysis_suite")
    os.makedirs(output_dir, exist_ok=True)

    psth_df = pd.read_csv(csv_path)
    print(f"Loaded PSTH CSV: {len(psth_df)} rows")

    # ── Build trial-level data ────────────────────────────────────────
    print("\n1/8  Loading trial-level data …")
    trial_df, whiskers, units = _load_trial_data(
        data_dir, contact_dir=contact_dir)
    print(f"     {len(trial_df)} trial rows, {len(units)} units, "
          f"whiskers {whiskers}")

    # ── 1. Fano factor ────────────────────────────────────────────────
    print("\n2/8  Computing Fano factor …")
    fano_df = compute_fano(trial_df, whiskers, units)
    fano_df.to_csv(os.path.join(output_dir, "fano_factor.csv"), index=False)
    plot_fano(fano_df, whiskers, units, output_dir)

    # ── 2. Temporal DSI ───────────────────────────────────────────────
    print("\n3/8  Computing temporal DSI …")
    tdsi_df = compute_temporal_dsi(psth_df, whiskers, units)
    tdsi_df.to_csv(os.path.join(output_dir, "temporal_dsi.csv"), index=False)
    plot_temporal_dsi(tdsi_df, whiskers, units, output_dir)

    # ── 3. Adaptation ─────────────────────────────────────────────────
    print("\n4/8  Computing adaptation …")
    adapt_df = compute_adaptation(trial_df, whiskers, units)
    adapt_df.to_csv(os.path.join(output_dir, "adaptation.csv"), index=False)
    plot_adaptation(adapt_df, trial_df, whiskers, units, output_dir)

    # ── 4. Response classification ────────────────────────────────────
    print("\n5/8  Classifying responses …")
    cls_df = classify_responses(trial_df, whiskers, units)
    cls_df.to_csv(os.path.join(output_dir, "response_classification.csv"),
                  index=False)
    plot_response_classification(cls_df, whiskers, units, output_dir)

    # ── 6. Mutual information ─────────────────────────────────────────
    print("\n6/8  Computing mutual information …")
    mi_df = compute_mutual_info(trial_df, whiskers, units)
    mi_df.to_csv(os.path.join(output_dir, "mutual_information.csv"),
                 index=False)
    plot_mutual_info(mi_df, units, output_dir)

    # ── 7. Receptive field map ────────────────────────────────────────
    print("\n7/8  Plotting receptive field maps …")
    plot_receptive_field(trial_df, whiskers, units, output_dir)

    # ── 8. ISI analysis ───────────────────────────────────────────────
    print("\n8/8  Computing ISI statistics …")
    isi_df = compute_isi(trial_df, whiskers, units)
    isi_df.to_csv(os.path.join(output_dir, "isi_statistics.csv"), index=False)
    plot_isi(isi_df, trial_df, whiskers, units, output_dir)

    print(f"\n{'='*60}")
    print(f"Done.  All outputs saved to {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Unit Analysis Suite — comprehensive whisker-response analyses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single session:
    python unit_analysis_suite.py --data_dir "C:\\path\\to\\session"

  Cross-session summary:
    python unit_analysis_suite.py --cross_session "C:\\s1" "C:\\s2" "C:\\s3"
        """,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--data_dir", help="Single session directory")
    g.add_argument("--cross_session", nargs="+",
                   help="Multiple session directories for cross-session summary")
    parser.add_argument("--contact_dir", default=None)
    parser.add_argument("--output_dir", default=None)

    args = parser.parse_args()

    if args.cross_session:
        out = args.output_dir or os.path.join(
            os.path.dirname(args.cross_session[0]),
            "cross_session_summary")
        cross_session_summary(args.cross_session, out)
    else:
        run_single_session(args.data_dir,
                           contact_dir=args.contact_dir,
                           output_dir=args.output_dir)


if __name__ == "__main__":
    main()
