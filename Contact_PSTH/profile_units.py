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
    whiskers = sorted([int(w) for w in wr["whisker"].unique()
                       if w is not None and not (isinstance(w, float) and np.isnan(w))])
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


def plot_whisker_selectivity(df: pd.DataFrame, wr: pd.DataFrame,
                              profiles: pd.DataFrame, out_dir: str):
    """
    Per-unit whisker selectivity figure with three panels per unit:

    1. PSTH overlay — mean firing-rate trace for each whisker (direction=all)
       plotted on the same axes so response magnitude, latency, and shape
       differences are immediately visible.

    2. Peak FR heatmap — whisker × direction matrix (retraction / protraction)
       with the cell value = peak FR.  Highlights which whisker × direction
       combination is dominant.

    3. Whisker Selectivity Index (WSI) bar — for each direction, compute
       WSI = (R_best - R_others_mean) / (R_best + R_others_mean)
       where R_best is the peak FR from the best whisker and R_others_mean
       is the mean peak FR from the remaining whiskers.  WSI = 1 means the
       unit responds only to one whisker; WSI ≈ 0 means equal response.

    Saves one PNG per unit.
    """
    whiskers = sorted([w for w in wr["whisker"].unique() if w is not None])
    units = sorted(profiles["unit"].values)
    colors_w = plt.cm.tab10(np.linspace(0, 1, max(len(whiskers), 1)))

    for unit in units:
        fig = plt.figure(figsize=(15, 4.5))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[2.5, 1.2, 1],
                      wspace=0.35)

        # ── Panel 1: PSTH overlay (direction = all) ───────────────────
        ax_psth = fig.add_subplot(gs[0, 0])
        for wi, w in enumerate(whiskers):
            sub = df[(df["unit"] == unit) &
                     (df["interval"] == f"interval_{w}_mask_contact")]
            if len(sub) == 0:
                continue
            sub = sub.sort_values("bin_ms")
            ax_psth.plot(sub["bin_ms"].values, sub["firing_rate_hz"].values,
                         color=colors_w[wi], linewidth=1.2,
                         label=f"W{w}", alpha=0.85)

        ax_psth.axvline(0, color="black", ls="--", lw=0.8, alpha=0.6)
        ax_psth.set_xlabel("Time from contact onset (ms)", fontsize=9)
        ax_psth.set_ylabel("Firing Rate (Hz)", fontsize=9)
        ax_psth.set_title("PSTH by Whisker", fontsize=10)
        if ax_psth.get_legend_handles_labels()[1]:
            ax_psth.legend(fontsize=7, loc="upper right")
        ax_psth.tick_params(labelsize=7)

        # ── Panel 2: Peak FR heatmap (whisker × direction) ────────────
        ax_heat = fig.add_subplot(gs[0, 1])
        directions = ["retraction", "protraction", "all"]
        matrix = np.zeros((len(whiskers), len(directions)))
        for wi, w in enumerate(whiskers):
            for di, d in enumerate(directions):
                row = wr[(wr["unit"] == unit) & (wr["whisker"] == w) &
                         (wr["direction"] == d)]
                matrix[wi, di] = row["peak_fr"].values[0] if len(row) else 0

        im = ax_heat.imshow(matrix, aspect="auto", cmap="YlOrRd",
                            interpolation="nearest")
        ax_heat.set_xticks(range(len(directions)))
        ax_heat.set_xticklabels(["Ret", "Pro", "All"], fontsize=8)
        ax_heat.set_yticks(range(len(whiskers)))
        ax_heat.set_yticklabels([f"W{w}" for w in whiskers], fontsize=8)
        ax_heat.set_title("Peak FR (Hz)", fontsize=10)
        # Annotate cells
        for wi in range(len(whiskers)):
            for di in range(len(directions)):
                val = matrix[wi, di]
                color = "white" if val > matrix.max() * 0.65 else "black"
                ax_heat.text(di, wi, f"{val:.0f}", ha="center", va="center",
                             fontsize=7, color=color)
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

        # ── Panel 3: Whisker Selectivity Index bars ───────────────────
        ax_wsi = fig.add_subplot(gs[0, 2])
        wsi_data = {}
        for d in ["retraction", "protraction", "all"]:
            frs = []
            for w in whiskers:
                row = wr[(wr["unit"] == unit) & (wr["whisker"] == w) &
                         (wr["direction"] == d)]
                frs.append(row["peak_fr"].values[0] if len(row) else 0)
            frs = np.array(frs)
            if len(frs) > 1 and frs.max() > 0:
                best_idx = frs.argmax()
                r_best = frs[best_idx]
                r_others = np.mean(np.delete(frs, best_idx))
                denom = r_best + r_others
                wsi = (r_best - r_others) / denom if denom > 0 else 0.0
            else:
                wsi = 0.0
            wsi_data[d] = wsi

        bar_labels = ["Ret", "Pro", "All"]
        bar_vals = [wsi_data["retraction"], wsi_data["protraction"],
                    wsi_data["all"]]
        bar_colors = ["steelblue", "salmon", "gray"]
        ax_wsi.bar(bar_labels, bar_vals, color=bar_colors, edgecolor="black",
                   linewidth=0.5)
        ax_wsi.set_ylim(-0.1, 1.05)
        ax_wsi.axhline(0, color="black", lw=0.6)
        ax_wsi.set_ylabel("WSI", fontsize=9)
        ax_wsi.set_title("Whisker Selectivity", fontsize=10)
        ax_wsi.tick_params(labelsize=7)

        # ── Unit title ────────────────────────────────────────────────
        bw_val = profiles.loc[profiles["unit"] == unit, "best_whisker"].values[0]
        bw_str = f"W{int(bw_val)}" if not np.isnan(bw_val) else "?"
        dsi_a = profiles.loc[profiles["unit"] == unit,
                             "best_whisker_dsi_adj"].values[0]
        pref = profiles.loc[profiles["unit"] == unit,
                            "preferred_direction"].values[0]
        fig.suptitle(f"Unit {unit}  |  Best: {bw_str}  |  "
                     f"DSI(adj): {dsi_a:.2f}  |  Pref: {pref}",
                     fontsize=11, fontweight="bold", y=1.01)

        fig.subplots_adjust(left=0.06, right=0.95, top=0.88, bottom=0.12,
                            wspace=0.35)
        path = os.path.join(out_dir, f"unit_{unit}_whisker_selectivity.png")
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
    plot_dsi_by_whisker(profiles, output_dir)
    plot_whisker_selectivity(df, wr, profiles, output_dir)

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
