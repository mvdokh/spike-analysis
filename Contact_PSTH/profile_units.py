"""
Unit Profiling from Contact PSTH Firing Rates

Reads the contact_psth_firing_rates.csv produced by contact_psth_to_csv.py
and generates a comprehensive per-unit analysis:

    1. Best whisker tuning  – which whisker drives the strongest response
    2. Direction selectivity – retraction vs protraction preference (DSI)
    3. Peak latency          – time of peak firing per interval
    4. Transient vs sustained – 0-10 ms ("transient") vs 10-50 ms ("sustained")
    5. Unit clustering       – strongly directional vs non-directional
    6. Summary figures

Saves:
    unit_profiles.csv          – one row per unit with all metrics
    whisker_responses.csv      – peak FR / latency per unit × whisker × direction
    profile_summary.png        – multi-panel overview figure
    dsi_by_whisker.png         – DSI bar plot per whisker for every unit
    latency_distributions.png  – peak-latency histograms
    clustering.png             – PCA / scatter of unit sub-populations

Usage
-----
    python profile_units.py --data_dir <session_dir>
    python profile_units.py --csv <path/to/contact_psth_firing_rates.csv>
"""

import argparse
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_interval_name(name: str):
    """
    Parse an interval name into (whisker_id, direction).

    Examples
    --------
        'contact_intervals'                     → (None, 'all')
        'interval_2_mask_contact'               → (2,    'all')
        'interval_2_mask_contact_protraction'    → (2,    'protraction')
        'interval_2_mask_contact_retraction'     → (2,    'retraction')

    Returns (whisker: int | None, direction: str)
    """
    if name == "contact_intervals":
        return (None, "all")

    m = re.match(r"interval_(\d+)_mask_contact(?:_(protraction|retraction))?$",
                 name)
    if m:
        whisker = int(m.group(1))
        direction = m.group(2) if m.group(2) else "all"
        return (whisker, direction)

    return (None, "unknown")


def _peak_fr_and_latency(fr_series, bin_series, window=None):
    """
    Return (peak_fr, peak_latency_ms) within an optional (lo, hi) window.
    """
    if window is not None:
        mask = (bin_series >= window[0]) & (bin_series < window[1])
        fr_series = fr_series[mask]
        bin_series = bin_series[mask]
    if len(fr_series) == 0:
        return (0.0, np.nan)
    idx = fr_series.values.argmax()
    return (fr_series.values[idx], bin_series.values[idx])


def _mean_fr_in_window(fr_series, bin_series, lo, hi):
    """Mean firing rate in [lo, hi) ms."""
    mask = (bin_series >= lo) & (bin_series < hi)
    vals = fr_series[mask]
    return vals.mean() if len(vals) > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────────────────────────────────────

def build_whisker_response_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every (unit, whisker, direction) triplet compute:
        - n_trials       : number of contact events
        - baseline_fr    : mean FR in pre-stimulus window (< 0 ms)
        - peak_fr        : peak FR in post-onset window [0, end)
        - peak_latency   : bin of peak FR (ms after onset)
        - mean_fr_post   : mean FR in [0, end) window
        - transient_fr   : mean FR in [0, 10) ms
        - sustained_fr   : mean FR in [10, 50) ms
    """
    has_n_trials = "n_trials" in df.columns
    rows = []
    for (unit, interval), grp in df.groupby(["unit", "interval"]):
        whisker, direction = _parse_interval_name(interval)
        bins = grp["bin_ms"]
        fr = grp["firing_rate_hz"]

        # Trial count (constant within a group)
        n_trials = int(grp["n_trials"].iloc[0]) if has_n_trials else np.nan

        baseline = _mean_fr_in_window(fr, bins, bins.min(), 0)
        peak_fr, peak_lat = _peak_fr_and_latency(fr, bins, window=(0, bins.max() + 1))
        mean_post = _mean_fr_in_window(fr, bins, 0, bins.max() + 1)
        transient = _mean_fr_in_window(fr, bins, 0, 10)
        sustained = _mean_fr_in_window(fr, bins, 10, 50)

        rows.append({
            "unit": unit,
            "interval": interval,
            "whisker": whisker,
            "direction": direction,
            "n_trials": n_trials,
            "baseline_fr": baseline,
            "peak_fr": peak_fr,
            "peak_latency_ms": peak_lat,
            "mean_fr_post": mean_post,
            "transient_fr_0_10": transient,
            "sustained_fr_10_50": sustained,
        })

    return pd.DataFrame(rows)


def compute_unit_profiles(wr: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate whisker_responses into one row per unit with:
        - best_whisker, best_whisker_peak_fr
        - DSI per whisker and overall preferred direction
        - transient/sustained ratio for best whisker
        - classification: strongly_directional, weakly_directional, non_directional
    """
    units = sorted(wr["unit"].unique())
    whiskers = sorted([w for w in wr["whisker"].unique() if w is not None])
    profiles = []

    for unit in units:
        u = wr[wr["unit"] == unit]
        row = {"unit": unit}

        # ── Best whisker (by peak FR, direction = 'all') ──────────────
        all_dir = u[(u["direction"] == "all") & (u["whisker"].notna())]
        if len(all_dir) > 0:
            best_idx = all_dir["peak_fr"].idxmax()
            row["best_whisker"] = int(all_dir.loc[best_idx, "whisker"])
            row["best_whisker_peak_fr"] = all_dir.loc[best_idx, "peak_fr"]
            row["best_whisker_latency_ms"] = all_dir.loc[best_idx, "peak_latency_ms"]
        else:
            row["best_whisker"] = np.nan
            row["best_whisker_peak_fr"] = 0.0
            row["best_whisker_latency_ms"] = np.nan

        # ── DSI per whisker ───────────────────────────────────────────
        # Raw DSI = (R_ret - R_pro) / (R_ret + R_pro)
        # Adjusted DSI scales by trial-count balance factor:
        #   balance = 2*sqrt(n_ret * n_pro) / (n_ret + n_pro)
        # This equals 1 when trial counts are equal and → 0 when
        # extremely imbalanced, shrinking noisy estimates toward 0.
        dsi_vals = []
        dsi_adj_vals = []
        for w in whiskers:
            ret_row = u[(u["whisker"] == w) & (u["direction"] == "retraction")]
            pro_row = u[(u["whisker"] == w) & (u["direction"] == "protraction")]
            r_ret = ret_row["peak_fr"].values[0] if len(ret_row) else 0
            r_pro = pro_row["peak_fr"].values[0] if len(pro_row) else 0
            n_ret = ret_row["n_trials"].values[0] if len(ret_row) else 0
            n_pro = pro_row["n_trials"].values[0] if len(pro_row) else 0

            denom = r_ret + r_pro
            dsi_raw = (r_ret - r_pro) / denom if denom > 0 else 0.0

            # Trial-count balance factor (geometric / arithmetic mean)
            n_sum = n_ret + n_pro
            if n_sum > 0 and n_ret > 0 and n_pro > 0:
                balance = 2 * np.sqrt(n_ret * n_pro) / n_sum
            else:
                balance = 0.0
            dsi_adj = dsi_raw * balance

            row[f"dsi_raw_w{w}"] = dsi_raw
            row[f"dsi_adj_w{w}"] = dsi_adj
            row[f"balance_w{w}"] = balance
            row[f"n_ret_w{w}"] = int(n_ret)
            row[f"n_pro_w{w}"] = int(n_pro)
            row[f"peak_ret_w{w}"] = r_ret
            row[f"peak_pro_w{w}"] = r_pro
            dsi_vals.append(dsi_raw)
            dsi_adj_vals.append(dsi_adj)

        row["mean_abs_dsi_raw"] = np.mean(np.abs(dsi_vals)) if dsi_vals else 0.0
        row["mean_abs_dsi_adj"] = np.mean(np.abs(dsi_adj_vals)) if dsi_adj_vals else 0.0

        # ── Best-whisker DSI and preferred direction ──────────────────
        bw = row.get("best_whisker")
        if bw is not None and not np.isnan(bw):
            bw = int(bw)
            row["best_whisker_dsi_raw"] = row.get(f"dsi_raw_w{bw}", 0.0)
            row["best_whisker_dsi_adj"] = row.get(f"dsi_adj_w{bw}", 0.0)
            row["best_whisker_balance"] = row.get(f"balance_w{bw}", 0.0)
            row["best_whisker_n_ret"] = row.get(f"n_ret_w{bw}", 0)
            row["best_whisker_n_pro"] = row.get(f"n_pro_w{bw}", 0)
            row["preferred_direction"] = (
                "retraction" if row["best_whisker_dsi_adj"] > 0 else "protraction"
            )
        else:
            row["best_whisker_dsi_raw"] = 0.0
            row["best_whisker_dsi_adj"] = 0.0
            row["best_whisker_balance"] = 0.0
            row["best_whisker_n_ret"] = 0
            row["best_whisker_n_pro"] = 0
            row["preferred_direction"] = "none"

        # ── Best-whisker latencies for ret / pro ──────────────────────
        if bw is not None and not np.isnan(bw):
            bw = int(bw)
            ret_row = u[(u["whisker"] == bw) & (u["direction"] == "retraction")]
            pro_row = u[(u["whisker"] == bw) & (u["direction"] == "protraction")]
            row["ret_latency_best_w"] = (
                ret_row["peak_latency_ms"].values[0] if len(ret_row) else np.nan
            )
            row["pro_latency_best_w"] = (
                pro_row["peak_latency_ms"].values[0] if len(pro_row) else np.nan
            )
        else:
            row["ret_latency_best_w"] = np.nan
            row["pro_latency_best_w"] = np.nan

        # ── Transient / sustained ratio (best whisker, direction=all) ─
        bw_all = u[(u["whisker"] == row.get("best_whisker")) & (u["direction"] == "all")]
        if len(bw_all) > 0:
            t = bw_all["transient_fr_0_10"].values[0]
            s = bw_all["sustained_fr_10_50"].values[0]
            row["transient_fr"] = t
            row["sustained_fr"] = s
            row["transient_sustained_ratio"] = t / s if s > 0 else np.inf
        else:
            row["transient_fr"] = 0.0
            row["sustained_fr"] = 0.0
            row["transient_sustained_ratio"] = np.nan

        # ── Selectivity classification (based on adjusted DSI) ────────
        abs_dsi = abs(row["best_whisker_dsi_adj"])
        if abs_dsi >= 0.5:
            row["direction_class"] = "strongly_directional"
        elif abs_dsi >= 0.2:
            row["direction_class"] = "weakly_directional"
        else:
            row["direction_class"] = "non_directional"

        # ── Response classification ───────────────────────────────────
        tsr = row["transient_sustained_ratio"]
        if tsr is not None and not np.isnan(tsr):
            if tsr > 2.0:
                row["response_type"] = "transient"
            elif tsr < 0.5:
                row["response_type"] = "sustained"
            else:
                row["response_type"] = "mixed"
        else:
            row["response_type"] = "unclassified"

        profiles.append(row)

    return pd.DataFrame(profiles)


# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────

def plot_summary(profiles: pd.DataFrame, wr: pd.DataFrame, out_dir: str):
    """4-panel overview: best-whisker histogram, DSI distribution,
    latency distribution, transient/sustained scatter."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Unit Profile Summary", fontsize=14, fontweight="bold")

    # 1) Best whisker histogram
    ax = axes[0, 0]
    bw = profiles["best_whisker"].dropna().astype(int)
    if len(bw) > 0:
        bw.value_counts().sort_index().plot.bar(ax=ax, color="steelblue",
                                                 edgecolor="black")
    ax.set_xlabel("Whisker ID")
    ax.set_ylabel("# Units")
    ax.set_title("Best Whisker Distribution")

    # 2) DSI distribution (best whisker — raw vs adjusted)
    ax = axes[0, 1]
    dsi_raw = profiles["best_whisker_dsi_raw"].dropna()
    dsi_adj = profiles["best_whisker_dsi_adj"].dropna()
    bins_dsi = np.linspace(-1, 1, 21)
    ax.hist(dsi_raw, bins=bins_dsi, color="salmon", edgecolor="black",
            alpha=0.4, label="Raw DSI")
    ax.hist(dsi_adj, bins=bins_dsi, color="steelblue", edgecolor="black",
            alpha=0.6, label="Adjusted DSI")
    ax.axvline(0, color="black", ls="--", alpha=0.5)
    ax.set_xlabel("DSI  (>0 = retraction)")
    ax.set_ylabel("# Units")
    ax.set_title("Direction Selectivity Index (Best Whisker)")
    ax.legend(fontsize=7)

    # 3) Peak latency distribution (post-onset, direction=all, all whiskers)
    ax = axes[1, 0]
    all_lat = wr.loc[(wr["direction"] == "all") & wr["whisker"].notna(),
                     "peak_latency_ms"].dropna()
    if len(all_lat) > 0:
        ax.hist(all_lat, bins=30, color="mediumseagreen", edgecolor="black")
    ax.set_xlabel("Peak Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Peak Latency Distribution (all whiskers)")

    # 4) Transient vs sustained scatter
    ax = axes[1, 1]
    t = profiles["transient_fr"].values
    s = profiles["sustained_fr"].values
    ax.scatter(s, t, c="darkorange", edgecolors="black", s=50, alpha=0.8)
    lim = max(t.max(), s.max()) * 1.1 if len(t) > 0 else 1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="unity")
    for _, r in profiles.iterrows():
        ax.annotate(str(int(r["unit"])),
                    (r["sustained_fr"], r["transient_fr"]),
                    fontsize=6, alpha=0.7)
    ax.set_xlabel("Sustained FR (10-50 ms)")
    ax.set_ylabel("Transient FR (0-10 ms)")
    ax.set_title("Transient vs Sustained")
    ax.legend(fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "profile_summary.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_dsi_by_whisker(profiles: pd.DataFrame, out_dir: str):
    """Grouped bar chart: raw & adjusted DSI per whisker for every unit."""
    dsi_adj_cols = [c for c in profiles.columns if c.startswith("dsi_adj_w")]
    dsi_raw_cols = [c for c in profiles.columns if c.startswith("dsi_raw_w")]
    whiskers = [c.replace("dsi_adj_w", "W") for c in dsi_adj_cols]
    n_units = len(profiles)
    n_whiskers = len(dsi_adj_cols)

    fig, axes = plt.subplots(2, 1, figsize=(max(8, n_units * 0.6), 9),
                             sharex=True)

    for ax, cols, title_label in [
        (axes[0], dsi_raw_cols, "Raw DSI"),
        (axes[1], dsi_adj_cols, "Trial-Count Adjusted DSI"),
    ]:
        x = np.arange(n_units)
        w = 0.8 / n_whiskers
        colors = plt.cm.Set2(np.linspace(0, 1, n_whiskers))

        for i, col in enumerate(cols):
            wlabel = whiskers[i]
            ax.bar(x + i * w, profiles[col].values, width=w, label=wlabel,
                   color=colors[i], edgecolor="black", linewidth=0.4)

        ax.set_xticks(x + w * n_whiskers / 2)
        ax.set_xticklabels([f"U{int(u)}" for u in profiles["unit"]], fontsize=7)
        ax.axhline(0, color="black", lw=0.8)
        ax.axhline(0.5, color="red", ls="--", lw=0.6, alpha=0.5,
                   label="±0.5 threshold")
        ax.axhline(-0.5, color="red", ls="--", lw=0.6, alpha=0.5)
        ax.set_ylabel("DSI  (>0 = retraction)")
        ax.set_title(title_label)
        ax.legend(fontsize=7, ncol=n_whiskers + 1, loc="upper right")

    axes[1].set_xlabel("Unit")
    fig.suptitle("Direction Selectivity Index by Whisker", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "dsi_by_whisker.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_latency_ret_vs_pro(wr: pd.DataFrame, out_dir: str):
    """Scatter: retraction peak latency vs protraction peak latency per unit×whisker."""
    whiskers = sorted([w for w in wr["whisker"].unique() if w is not None])
    units = sorted(wr["unit"].unique())

    ret_lats, pro_lats, labels = [], [], []
    for unit in units:
        for w in whiskers:
            ret = wr[(wr["unit"] == unit) & (wr["whisker"] == w) &
                     (wr["direction"] == "retraction")]
            pro = wr[(wr["unit"] == unit) & (wr["whisker"] == w) &
                     (wr["direction"] == "protraction")]
            if len(ret) and len(pro):
                rl = ret["peak_latency_ms"].values[0]
                pl = pro["peak_latency_ms"].values[0]
                if not (np.isnan(rl) or np.isnan(pl)):
                    ret_lats.append(rl)
                    pro_lats.append(pl)
                    labels.append(f"U{unit}W{w}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pro_lats, ret_lats, c="steelblue", edgecolors="black", s=40, alpha=0.7)
    lim_max = max(max(ret_lats, default=1), max(pro_lats, default=1)) * 1.1
    ax.plot([0, lim_max], [0, lim_max], "k--", alpha=0.3, label="unity")
    for rl, pl, lab in zip(ret_lats, pro_lats, labels):
        ax.annotate(lab, (pl, rl), fontsize=5, alpha=0.6)
    ax.set_xlabel("Protraction Peak Latency (ms)")
    ax.set_ylabel("Retraction Peak Latency (ms)")
    ax.set_title("Peak Latency: Retraction vs Protraction")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "latency_ret_vs_pro.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_clustering(profiles: pd.DataFrame, out_dir: str):
    """
    Scatter of mean |DSI| vs transient/sustained ratio, coloured by
    directional classification.  Provides a simple view of unit sub-populations.
    """
    color_map = {
        "strongly_directional": "crimson",
        "weakly_directional": "orange",
        "non_directional": "steelblue",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls, color in color_map.items():
        sub = profiles[profiles["direction_class"] == cls]
        tsr = sub["transient_sustained_ratio"].replace([np.inf], np.nan).fillna(0)
        ax.scatter(sub["mean_abs_dsi_adj"], tsr, c=color, edgecolors="black",
                   s=60, alpha=0.8, label=cls.replace("_", " "))
        for _, r in sub.iterrows():
            tsr_val = r["transient_sustained_ratio"]
            if np.isinf(tsr_val) or np.isnan(tsr_val):
                tsr_val = 0
            ax.annotate(str(int(r["unit"])),
                        (r["mean_abs_dsi_adj"], tsr_val),
                        fontsize=6, alpha=0.7)

    ax.axvline(0.2, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.axvline(0.5, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(2.0, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Mean |DSI adj| across whiskers")
    ax.set_ylabel("Transient / Sustained Ratio")
    ax.set_title("Unit Clustering: Directionality × Response Type")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "clustering.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_per_unit_whisker_tuning(profiles: pd.DataFrame, wr: pd.DataFrame,
                                 out_dir: str):
    """
    One polar-style bar chart per unit showing peak FR for each whisker,
    split by retraction / protraction.
    """
    whiskers = sorted([w for w in wr["whisker"].unique() if w is not None])
    units = sorted(profiles["unit"].values)

    n_units = len(units)
    ncols = min(5, n_units)
    nrows = int(np.ceil(n_units / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3))
    if n_units == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, unit in enumerate(units):
        ax = axes[i]
        x = np.arange(len(whiskers))
        w = 0.35

        rets, pros = [], []
        for wh in whiskers:
            r = wr[(wr["unit"] == unit) & (wr["whisker"] == wh) &
                   (wr["direction"] == "retraction")]
            p = wr[(wr["unit"] == unit) & (wr["whisker"] == wh) &
                   (wr["direction"] == "protraction")]
            rets.append(r["peak_fr"].values[0] if len(r) else 0)
            pros.append(p["peak_fr"].values[0] if len(p) else 0)

        ax.bar(x - w / 2, rets, w, label="Ret", color="steelblue",
               edgecolor="black", linewidth=0.4)
        ax.bar(x + w / 2, pros, w, label="Pro", color="salmon",
               edgecolor="black", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"W{wh}" for wh in whiskers], fontsize=7)
        bw = profiles.loc[profiles["unit"] == unit, "best_whisker"].values[0]
        bw_str = f"W{int(bw)}" if not np.isnan(bw) else "?"
        dsi_r = profiles.loc[profiles["unit"] == unit, "best_whisker_dsi_raw"].values[0]
        dsi_a = profiles.loc[profiles["unit"] == unit, "best_whisker_dsi_adj"].values[0]
        ax.set_title(f"U{unit}  best={bw_str}  raw={dsi_r:.2f}  adj={dsi_a:.2f}",
                     fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Peak FR by Whisker & Direction", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "whisker_tuning_per_unit.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(csv_path: str, output_dir: str | None = None):
    print(f"Loading firing rates from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, {df['unit'].nunique()} units, "
          f"{df['interval'].nunique()} intervals")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path), "unit_profiles")
    os.makedirs(output_dir, exist_ok=True)

    # ── Build tables ──────────────────────────────────────────────────
    print("\nComputing whisker response metrics …")
    wr = build_whisker_response_table(df)

    print("Computing unit profiles …")
    profiles = compute_unit_profiles(wr)

    # ── Save CSVs ─────────────────────────────────────────────────────
    wr_path = os.path.join(output_dir, "whisker_responses.csv")
    wr.to_csv(wr_path, index=False)
    print(f"  Saved {wr_path}")

    prof_path = os.path.join(output_dir, "unit_profiles.csv")
    profiles.to_csv(prof_path, index=False)
    print(f"  Saved {prof_path}")

    # ── Print summary table ───────────────────────────────────────────
    summary_cols = ["unit", "best_whisker", "best_whisker_peak_fr",
                    "best_whisker_latency_ms",
                    "best_whisker_dsi_raw", "best_whisker_dsi_adj",
                    "best_whisker_balance",
                    "best_whisker_n_ret", "best_whisker_n_pro",
                    "preferred_direction", "direction_class",
                    "transient_sustained_ratio", "response_type"]
    print(f"\n{'='*90}")
    print("UNIT PROFILE SUMMARY")
    print(f"{'='*90}")
    print(profiles[summary_cols].to_string(index=False))
    print(f"{'='*90}")

    dc = profiles["direction_class"].value_counts()
    print(f"\nDirection classes:  {dict(dc)}")
    rt = profiles["response_type"].value_counts()
    print(f"Response types:     {dict(rt)}")

    # ── Generate figures ──────────────────────────────────────────────
    print("\nGenerating figures …")
    plot_summary(profiles, wr, output_dir)
    plot_dsi_by_whisker(profiles, output_dir)
    plot_latency_ret_vs_pro(wr, output_dir)
    plot_clustering(profiles, output_dir)
    plot_per_unit_whisker_tuning(profiles, wr, output_dir)

    print(f"\n{'='*60}")
    print(f"Done. All outputs saved to {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile units from contact PSTH firing-rate CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python profile_units.py --data_dir "C:\\path\\to\\session"
  python profile_units.py --csv "path/to/contact_psth_firing_rates.csv"
        """,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--data_dir",
                   help="Session directory (looks for contact_psth_csv_output/"
                        "contact_psth_firing_rates.csv inside it)")
    g.add_argument("--csv", help="Direct path to contact_psth_firing_rates.csv")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save results (default: unit_profiles/ "
                             "next to the input CSV)")

    args = parser.parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(args.data_dir,
                                "contact_psth_csv_output",
                                "contact_psth_firing_rates.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    run(csv_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
