"""
Whisker Tuning Curves

Reads contact_psth_firing_rates.csv and generates tuning-curve plots that
show each unit's response magnitude across whiskers.

For each unit, produces a single figure with:
    - Top row:    Tuning curves (mean evoked FR vs whisker) for All / Retraction
                  / Protraction, with baseline subtracted.  Error bars show 95%
                  bootstrap confidence intervals (1000 resamples of time bins
                  in the response window).
    - Bottom row: Overlaid PSTH traces for all whiskers (direction = all),
                  so the temporal profile can be compared alongside the
                  tuning curve.

Also produces a population summary figure with all units' normalised tuning
curves overlaid.

Usage
-----
    python tuning_curves.py --data_dir <session_dir>
    python tuning_curves.py --csv <path/to/contact_psth_firing_rates.csv>
"""

import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu
from itertools import combinations

warnings.filterwarnings("ignore", category=FutureWarning)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_interval_name(name: str):
    """Return (whisker: int | None, direction: str)."""
    if name == "contact_intervals":
        return (None, "all")
    m = re.match(r"interval_(\d+)_mask_contact(?:_(protraction|retraction))?$",
                 name)
    if m:
        whisker = int(m.group(1))
        direction = m.group(2) if m.group(2) else "all"
        return (whisker, direction)
    return (None, "unknown")


def _bootstrap_ci(values, n_boot=1000, ci=95, rng=None):
    """
    Bootstrap confidence interval for the mean of *values*.

    Returns (ci_lo, ci_hi) — absolute values, NOT offsets from the mean.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if len(values) == 0:
        return 0.0, 0.0
    boot_means = np.empty(n_boot)
    n = len(values)
    for i in range(n_boot):
        boot_means[i] = rng.choice(values, size=n, replace=True).mean()
    alpha = (100 - ci) / 2
    return float(np.percentile(boot_means, alpha)), float(np.percentile(boot_means, 100 - alpha))


def _evoked_fr(fr_vals, bin_vals, baseline_window=(-50, 0),
               response_window=(0, 50)):
    """
    Compute baseline-subtracted mean firing rate in the response window.

    Returns (evoked_mean, evoked_sem, baseline_mean, ci_lo, ci_hi).
    ci_lo / ci_hi are 95 % bootstrap confidence‐interval bounds for
    the baseline-subtracted mean.
    """
    bl_mask = (bin_vals >= baseline_window[0]) & (bin_vals < baseline_window[1])
    resp_mask = (bin_vals >= response_window[0]) & (bin_vals < response_window[1])

    baseline = fr_vals[bl_mask].mean() if bl_mask.sum() > 0 else 0.0
    resp_bins = fr_vals[resp_mask]
    if resp_bins.size > 0:
        evoked_mean = resp_bins.mean() - baseline
        evoked_sem = resp_bins.std() / np.sqrt(resp_bins.size)
        # Bootstrap on the baseline-subtracted response bins
        ci_lo, ci_hi = _bootstrap_ci(resp_bins - baseline)
    else:
        evoked_mean = 0.0
        evoked_sem = 0.0
        ci_lo, ci_hi = 0.0, 0.0
    return evoked_mean, evoked_sem, baseline, ci_lo, ci_hi


# ──────────────────────────────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────────────────────────────

def build_tuning_data(df: pd.DataFrame):
    """
    Returns a DataFrame with columns:
        unit, whisker, direction, n_trials,
        evoked_fr, evoked_sem, baseline_fr, peak_fr, peak_latency_ms
    """
    rows = []
    for (unit, interval), grp in df.groupby(["unit", "interval"]):
        whisker, direction = _parse_interval_name(interval)
        if whisker is None:
            continue
        bins = grp["bin_ms"].values
        fr = grp["firing_rate_hz"].values
        n_trials = int(grp["n_trials"].iloc[0]) if "n_trials" in grp.columns else np.nan

        evoked, sem, baseline, ci_lo, ci_hi = _evoked_fr(fr, bins)

        # Peak in response window
        resp_mask = (bins >= 0) & (bins < 100)
        resp_fr = fr[resp_mask]
        resp_bins = bins[resp_mask]
        if resp_fr.size > 0:
            peak_idx = resp_fr.argmax()
            peak_fr = resp_fr[peak_idx]
            peak_lat = resp_bins[peak_idx]
        else:
            peak_fr = 0.0
            peak_lat = np.nan

        rows.append({
            "unit": unit,
            "whisker": whisker,
            "direction": direction,
            "n_trials": n_trials,
            "evoked_fr": evoked,
            "evoked_sem": sem,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "baseline_fr": baseline,
            "peak_fr": peak_fr,
            "peak_latency_ms": peak_lat,
        })
    return pd.DataFrame(rows)


def _get_response_bins(df_raw, unit, interval_name, window=(0, 50)):
    """Extract firing-rate values in the response window for one unit×interval."""
    sub = df_raw[(df_raw["unit"] == unit) &
                 (df_raw["interval"] == interval_name)]
    if len(sub) == 0:
        return np.array([])
    bins = sub["bin_ms"].values
    fr = sub["firing_rate_hz"].values
    mask = (bins >= window[0]) & (bins < window[1])
    return fr[mask]


def _sig_stars(p):
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Per-unit tuning curve figure
# ──────────────────────────────────────────────────────────────────────────────

def plot_unit_tuning(unit, df_raw, tuning, whiskers, out_dir):
    """
    Two-row figure for one unit:
        Row 1: tuning curves (evoked FR vs whisker) for all / ret / pro
        Row 2: overlaid PSTH traces per whisker (direction=all)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel 1: Tuning curves ────────────────────────────────────────
    ax = axes[0]
    x = np.arange(len(whiskers))
    dir_styles = {
        "all":          {"color": "black",     "marker": "o", "ls": "-",  "lw": 2.0},
        "retraction":   {"color": "steelblue", "marker": "s", "ls": "--", "lw": 1.5},
        "protraction":  {"color": "salmon",    "marker": "^", "ls": "--", "lw": 1.5},
    }

    for direction, style in dir_styles.items():
        means, ci_los, ci_his, n_trials_list = [], [], [], []
        for w in whiskers:
            row = tuning[(tuning["unit"] == unit) &
                         (tuning["whisker"] == w) &
                         (tuning["direction"] == direction)]
            if len(row) > 0:
                means.append(row["evoked_fr"].values[0])
                ci_los.append(row["ci_lo"].values[0])
                ci_his.append(row["ci_hi"].values[0])
                nt = row["n_trials"].values[0]
                n_trials_list.append(int(nt) if not np.isnan(nt) else 0)
            else:
                means.append(0.0)
                ci_los.append(0.0)
                ci_his.append(0.0)
                n_trials_list.append(0)

        means = np.array(means)
        ci_los = np.array(ci_los)
        ci_his = np.array(ci_his)
        # Asymmetric error bars: distance from mean to each CI bound
        yerr_lo = np.clip(means - ci_los, 0, None)
        yerr_hi = np.clip(ci_his - means, 0, None)
        label = direction.capitalize()
        ax.errorbar(x, means, yerr=[yerr_lo, yerr_hi], label=label, capsize=3,
                    **style)

    # Annotate trial counts (from direction=all)
    for xi, w in enumerate(whiskers):
        row = tuning[(tuning["unit"] == unit) &
                     (tuning["whisker"] == w) &
                     (tuning["direction"] == "all")]
        if len(row) > 0:
            nt = row["n_trials"].values[0]
            nt_str = str(int(nt)) if not np.isnan(nt) else "?"
        else:
            nt_str = "0"
        ax.annotate(f"n={nt_str}", (xi, 0), textcoords="offset points",
                    xytext=(0, -14), fontsize=6, ha="center", color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels([f"W{w}" for w in whiskers], fontsize=9)
    ax.set_xlabel("Whisker", fontsize=10)
    ax.set_ylabel("Evoked FR (Hz, baseline-subtracted)", fontsize=10)
    ax.set_title("Tuning Curve", fontsize=11)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    # ── Significance: ret vs pro within each whisker ──────────────────
    ret_pro_stars = []
    for w in whiskers:
        ret_bins = _get_response_bins(
            df_raw, unit,
            f"interval_{w}_mask_contact_retraction")
        pro_bins = _get_response_bins(
            df_raw, unit,
            f"interval_{w}_mask_contact_protraction")
        if len(ret_bins) > 1 and len(pro_bins) > 1:
            _, p = mannwhitneyu(ret_bins, pro_bins, alternative="two-sided")
            ret_pro_stars.append(_sig_stars(p))
        else:
            ret_pro_stars.append("")

    # Get y-values for placing annotations
    all_means_arr = {}
    for d in ["retraction", "protraction"]:
        vals = []
        for w in whiskers:
            row = tuning[(tuning["unit"] == unit) &
                         (tuning["whisker"] == w) &
                         (tuning["direction"] == d)]
            vals.append(row["evoked_fr"].values[0] if len(row) else 0.0)
        all_means_arr[d] = np.array(vals)

    y_max_local = max(
        np.max(np.abs(all_means_arr.get("retraction", [0]))),
        np.max(np.abs(all_means_arr.get("protraction", [0]))),
    )
    for xi, stars in enumerate(ret_pro_stars):
        if stars:
            ret_val = all_means_arr["retraction"][xi] if xi < len(all_means_arr["retraction"]) else 0
            pro_val = all_means_arr["protraction"][xi] if xi < len(all_means_arr["protraction"]) else 0
            bar_top = max(ret_val, pro_val) + y_max_local * 0.05
            ax.annotate(stars, (xi, bar_top), ha="center", va="bottom",
                        fontsize=8, color="purple", fontweight="bold")

    # ── Significance: between whiskers (direction=all) ────────────────
    whisker_bins = {}
    for w in whiskers:
        whisker_bins[w] = _get_response_bins(
            df_raw, unit, f"interval_{w}_mask_contact")

    # Collect all means for direction=all to position brackets
    all_dir_means = []
    for w in whiskers:
        row = tuning[(tuning["unit"] == unit) &
                     (tuning["whisker"] == w) &
                     (tuning["direction"] == "all")]
        all_dir_means.append(row["evoked_fr"].values[0] if len(row) else 0.0)
    all_dir_means = np.array(all_dir_means)

    # Pairwise tests with Bonferroni correction
    n_pairs = len(list(combinations(range(len(whiskers)), 2)))
    sig_pairs = []
    for (i, j) in combinations(range(len(whiskers)), 2):
        bi = whisker_bins[whiskers[i]]
        bj = whisker_bins[whiskers[j]]
        if len(bi) > 1 and len(bj) > 1:
            _, p_raw = mannwhitneyu(bi, bj, alternative="two-sided")
            p_corr = min(p_raw * n_pairs, 1.0)  # Bonferroni
            stars = _sig_stars(p_corr)
            if stars:
                sig_pairs.append((i, j, stars, p_corr))

    # Draw brackets for significant whisker pairs
    if sig_pairs:
        y_base = max(all_dir_means.max(), y_max_local) * 1.15
        y_step = max(all_dir_means.max(), y_max_local) * 0.10
        for ki, (i, j, stars, _) in enumerate(sig_pairs):
            y_bar = y_base + ki * y_step
            ax.plot([i, i, j, j], [y_bar - y_step * 0.15, y_bar, y_bar,
                    y_bar - y_step * 0.15],
                    color="black", lw=0.8)
            ax.text((i + j) / 2, y_bar, stars, ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    # ── Panel 2: PSTH overlay ─────────────────────────────────────────
    ax2 = axes[1]
    colors_w = plt.cm.tab10(np.linspace(0, 1, max(len(whiskers), 1)))
    for wi, w in enumerate(whiskers):
        interval_name = f"interval_{w}_mask_contact"
        sub = df_raw[(df_raw["unit"] == unit) &
                     (df_raw["interval"] == interval_name)]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("bin_ms")
        smoothed = gaussian_filter1d(sub["firing_rate_hz"].values, sigma=3)
        ax2.plot(sub["bin_ms"].values, smoothed,
                 color=colors_w[wi], linewidth=1.2,
                 label=f"W{w}", alpha=0.85)

    ax2.axvline(0, color="black", ls="--", lw=0.8, alpha=0.6)
    ax2.set_xlabel("Time from contact onset (ms)", fontsize=10)
    ax2.set_ylabel("Firing Rate (Hz)", fontsize=10)
    ax2.set_title("PSTH by Whisker", fontsize=11)
    if ax2.get_legend_handles_labels()[1]:
        ax2.legend(fontsize=7, loc="upper right")
    ax2.tick_params(labelsize=8)

    fig.suptitle(f"Unit {unit}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"unit_{unit}_tuning_curve.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Population summary
# ──────────────────────────────────────────────────────────────────────────────

def plot_population_tuning(tuning, whiskers, units, out_dir):
    """
    Normalised tuning curves (direction=all) for all units on one plot.
    Each unit's curve is normalised to its own peak so shapes can be compared.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(units), 1)))
    x = np.arange(len(whiskers))

    # ── Left: normalised tuning curves ────────────────────────────────
    ax = axes[0]
    for ui, unit in enumerate(units):
        means = []
        for w in whiskers:
            row = tuning[(tuning["unit"] == unit) &
                         (tuning["whisker"] == w) &
                         (tuning["direction"] == "all")]
            means.append(row["evoked_fr"].values[0] if len(row) else 0.0)
        means = np.array(means)
        peak = means.max()
        if peak > 0:
            normed = means / peak
        else:
            normed = means
        ax.plot(x, normed, marker="o", markersize=4, linewidth=1.2,
                color=colors[ui], alpha=0.7, label=f"U{unit}")

    ax.set_xticks(x)
    ax.set_xticklabels([f"W{w}" for w in whiskers], fontsize=9)
    ax.set_xlabel("Whisker", fontsize=10)
    ax.set_ylabel("Normalised Evoked FR", fontsize=10)
    ax.set_title("Normalised Tuning Curves (All Units)", fontsize=11)
    ax.set_ylim(-0.2, 1.15)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.legend(fontsize=6, ncol=2, loc="upper right")
    ax.tick_params(labelsize=8)

    # ── Right: heatmap of evoked FR (unit × whisker) ──────────────────
    ax2 = axes[1]
    matrix = np.zeros((len(units), len(whiskers)))
    for ui, unit in enumerate(units):
        for wi, w in enumerate(whiskers):
            row = tuning[(tuning["unit"] == unit) &
                         (tuning["whisker"] == w) &
                         (tuning["direction"] == "all")]
            matrix[ui, wi] = row["evoked_fr"].values[0] if len(row) else 0.0

    im = ax2.imshow(matrix, aspect="auto", cmap="YlOrRd",
                    interpolation="nearest")
    ax2.set_xticks(range(len(whiskers)))
    ax2.set_xticklabels([f"W{w}" for w in whiskers], fontsize=9)
    ax2.set_yticks(range(len(units)))
    ax2.set_yticklabels([f"U{u}" for u in units], fontsize=8)
    ax2.set_xlabel("Whisker", fontsize=10)
    ax2.set_ylabel("Unit", fontsize=10)
    ax2.set_title("Evoked FR Heatmap (Hz)", fontsize=11)
    for ui in range(len(units)):
        for wi in range(len(whiskers)):
            val = matrix[ui, wi]
            vmax = matrix.max() if matrix.max() > 0 else 1
            color = "white" if val > vmax * 0.6 else "black"
            ax2.text(wi, ui, f"{val:.1f}", ha="center", va="center",
                     fontsize=6, color=color)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle("Population Whisker Tuning", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "population_tuning.png")
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
        output_dir = os.path.join(os.path.dirname(csv_path), "tuning_curves")
    os.makedirs(output_dir, exist_ok=True)

    # Build tuning data
    print("\nComputing tuning metrics …")
    tuning = build_tuning_data(df)

    # Save tuning CSV
    tuning_path = os.path.join(output_dir, "tuning_data.csv")
    tuning.to_csv(tuning_path, index=False)
    print(f"  Saved {tuning_path}")

    whiskers = sorted(tuning["whisker"].unique())
    units = sorted(tuning["unit"].unique())
    print(f"  {len(units)} units, {len(whiskers)} whiskers: {whiskers}")

    # Per-unit tuning curves
    print("\nGenerating per-unit tuning curves …")
    for unit in units:
        plot_unit_tuning(unit, df, tuning, whiskers, output_dir)

    # Population summary
    print("\nGenerating population summary …")
    plot_population_tuning(tuning, whiskers, units, output_dir)

    print(f"\n{'='*60}")
    print(f"Done. All outputs saved to {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate whisker tuning curve plots from PSTH firing-rate CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tuning_curves.py --data_dir "C:\\path\\to\\session"
  python tuning_curves.py --csv "path/to/contact_psth_firing_rates.csv"
        """,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--data_dir",
                   help="Session directory (looks for contact_psth_csv_output/"
                        "contact_psth_firing_rates.csv inside it)")
    g.add_argument("--csv", help="Direct path to contact_psth_firing_rates.csv")
    parser.add_argument("--output_dir", default=None,
                        help="Where to save results (default: tuning_curves/ "
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
