"""
Feature-Space Unit Classification
==================================

Extracts numerical response features from each unit's PSTH and performs
unsupervised classification via dimensionality reduction and clustering.

22 features are extracted per unit — all expressed as modulation index
    (FR_window − FR_baseline) / FR_baseline
Baseline = mean FR in [−50, −10) ms of the onset-aligned PSTH.

Clustering uses the **9 core temporal features** (onset-aligned 1–5 and
end-aligned 6–9) with a **signed-log transform** to compress skewed
modulation indices before RobustScaler standardisation.  The number of
clusters is selected automatically via the **gap statistic** (min k ≥ 3).

  Onset-aligned temporal profile (best whisker, direction = all):
     1. pre_onset      – (−10, 0) ms   anticipatory / pre-contact change
     2. early_onset    – (0, 10) ms    fast onset response
     3. late_onset     – (10, 20) ms   slower onset component
     4. sustained      – (20, 50) ms   maintained response during contact
     5. late_response  – (50, 100) ms  late / adaptation phase

  End-aligned temporal profile (best whisker, direction = all):
     6. pre_offset     – (−20, 0) ms   activity just before contact ends
     7. early_offset   – (0, 10) ms    fast off-response
     8. late_offset    – (10, 20) ms   slower off-response
     9. post_offset    – (20, 50) ms   rebound / suppression after offset

  Direction-specific onset (best whisker, onset-aligned):
    10. onset_ret      – (0, 20) ms retraction contacts
    11. onset_pro      – (0, 20) ms protraction contacts
    12. sustained_ret  – (20, 50) ms retraction contacts
    13. sustained_pro  – (20, 50) ms protraction contacts

  Direction-specific offset (best whisker, end-aligned):
    14. offset_ret     – (0, 20) ms after offset, retraction
    15. offset_pro     – (0, 20) ms after offset, protraction

  Per-whisker onset response (direction = all, 0–50 ms):
    16–20. w0_onset … w4_onset

  Summary indices:
    21. dsi  – direction selectivity index
    22. wsi  – whisker selectivity index

Pipeline:
    1. Extract 22 features → CSV
    2. Signed-log transform + RobustScaler on 9 core temporal features
    3. PCA (2-D & 3-D)
    4. t-SNE (2-D)
    5. Hierarchical clustering (Ward, k via gap statistic, min k ≥ 3)
    6. Cluster-coloured scatter plots
    7. Feature contribution radar chart per cluster
    8. Parallel coordinates plot

Usage
-----
    python feature_space_analysis.py --data_dir <session>
    python feature_space_analysis.py --csv <path/to/contact_psth_firing_rates.csv>
    python feature_space_analysis.py --csv <path> --n_clusters 6
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
from matplotlib.patches import FancyArrowPatch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore", category=FutureWarning)

# Try UMAP (optional)
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _parse_interval(name):
    if name == "contact_intervals":
        return (None, "all")
    m = re.match(
        r"interval_(\d+)_mask_contact(?:_(protraction|retraction))?$", name)
    if m:
        return (int(m.group(1)), m.group(2) if m.group(2) else "all")
    return (None, "unknown")


def _mean_fr_window(bins, fr, lo, hi):
    """Mean firing rate in [lo, hi) ms."""
    mask = (bins >= lo) & (bins < hi)
    return fr[mask].mean() if mask.sum() > 0 else 0.0


# Winsorising limits for modulation indices.  Units with very low baseline
# FR can produce extreme values (e.g. 33×); capping prevents them from
# dominating the clustering feature space.
MOD_INDEX_CLIP = (-1.0, 5.0)


def _modulation_index(response, baseline):
    """(response − baseline) / baseline, guarded and winsorised."""
    if baseline > 0:
        raw = (response - baseline) / baseline
    elif response > 0:
        raw = 1.0  # infinite increase → cap at 1
    else:
        raw = 0.0
    return float(np.clip(raw, MOD_INDEX_CLIP[0], MOD_INDEX_CLIP[1]))


def signed_log_transform(X):
    """sign(x) * log1p(|x|) — compresses skewed modulation indices.

    Units with near-zero baseline FR produce extreme modulation indices
    (e.g. 5× after clipping).  The signed-log transform preserves the
    sign but compresses the right tail so that outliers do not dominate
    the Euclidean distance used by Ward linkage.
    """
    return np.sign(X) * np.log1p(np.abs(X))


# ── Unit inclusion criteria ───────────────────────────────────────────────────
# Units that fail either criterion are excluded from clustering (but still
# appear in the full CSV with  responsive = False).
MIN_EVENTS = 100           # minimum contact events for best whisker
MIN_MAX_MOD_INDEX = 1.25   # minimum abs(modulation index) in any window


def filter_responsive_units(feat_df):
    """Mark non-responsive units and return a filtered copy for clustering.

    A unit is considered **responsive** if:
      1. ``n_events_best_whisker >= MIN_EVENTS``  (enough trials), AND
      2. at least one of the 9 core temporal features has
         ``|mod_index| >= MIN_MAX_MOD_INDEX``  (detectable FR change).

    A ``responsive`` boolean column is added to *feat_df* (in-place) so
    the full table can be saved with the flag.  The returned DataFrame
    contains only responsive units.
    """
    core = FEATURE_NAMES   # 9 core temporal features
    max_abs = feat_df[core].abs().max(axis=1)
    n_ev = feat_df["n_events_best_whisker"] if "n_events_best_whisker" in feat_df.columns else pd.Series(999, index=feat_df.index)

    responsive = (n_ev >= MIN_EVENTS) & (max_abs >= MIN_MAX_MOD_INDEX)
    feat_df["responsive"] = responsive

    n_total = len(feat_df)
    n_excl = (~responsive).sum()
    if n_excl > 0:
        print(f"   Excluded {n_excl}/{n_total} non-responsive units "
              f"(min_events={MIN_EVENTS}, min_mod={MIN_MAX_MOD_INDEX})")
        excl = feat_df.loc[~responsive]
        for _, row in excl.iterrows():
            lbl = row.get("label", f"U{row['unit']}")
            reason = []
            if n_ev.loc[row.name] < MIN_EVENTS:
                reason.append(f"events={int(n_ev.loc[row.name])}")
            if max_abs.loc[row.name] < MIN_MAX_MOD_INDEX:
                reason.append(f"max_mod={max_abs.loc[row.name]:.3f}")
            print(f"      {lbl}  ({', '.join(reason)})")
    else:
        print(f"   All {n_total} units pass responsiveness criteria.")

    return feat_df.loc[responsive].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Feature extraction
# ══════════════════════════════════════════════════════════════════════════════

# ── All 22 extracted features (kept in CSV for reference) ─────────────────
ALL_FEATURE_NAMES = [
    # Onset-aligned temporal (best whisker, all)
    "pre_onset",       # 1
    "early_onset",     # 2
    "late_onset",      # 3
    "sustained",       # 4
    "late_response",   # 5
    # End-aligned temporal (best whisker, all)
    "pre_offset",      # 6
    "early_offset",    # 7
    "late_offset",     # 8
    "post_offset",     # 9
    # Direction-specific onset (best whisker)
    "onset_ret",       # 10
    "onset_pro",       # 11
    "sustained_ret",   # 12
    "sustained_pro",   # 13
    # Direction-specific offset (best whisker)
    "offset_ret",      # 14
    "offset_pro",      # 15
    # Per-whisker onset response
    "w0_onset",        # 16
    "w1_onset",        # 17
    "w2_onset",        # 18
    "w3_onset",        # 19
    "w4_onset",        # 20
    # Summary
    "dsi",             # 21
    "wsi",             # 22
]

ALL_FEATURE_LABELS = [
    # Onset-aligned temporal
    "Pre-onset\n(−10–0 ms)",
    "Early onset\n(0–10 ms)",
    "Late onset\n(10–20 ms)",
    "Sustained\n(20–50 ms)",
    "Late resp\n(50–100 ms)",
    # End-aligned temporal
    "Pre-offset\n(−20–0 ms)",
    "Early offset\n(0–10 ms)",
    "Late offset\n(10–20 ms)",
    "Post-offset\n(20–50 ms)",
    # Direction onset
    "Onset ret",
    "Onset pro",
    "Sust. ret",
    "Sust. pro",
    # Direction offset
    "Offset ret",
    "Offset pro",
    # Per-whisker
    "W0 onset",
    "W1 onset",
    "W2 onset",
    "W3 onset",
    "W4 onset",
    # Summary
    "DSI",
    "WSI",
]

# ── 9 core temporal features used for clustering ─────────────────────────
# These capture the key response types (onset excitatory, offset excitatory,
# onset+offset, sustained, inhibited, etc.) without the correlated
# direction-specific / per-whisker features that dilute separation.
FEATURE_NAMES = [
    "pre_onset",       # anticipatory / pre-contact
    "early_onset",     # fast onset (0–10 ms)
    "late_onset",      # slower onset (10–20 ms)
    "sustained",       # maintained during contact (20–50 ms)
    "late_response",   # late adaptation (50–100 ms)
    "pre_offset",      # just before offset (−20–0 ms)
    "early_offset",    # fast off-response (0–10 ms)
    "late_offset",     # slower off-response (10–20 ms)
    "post_offset",     # post-offset rebound/suppression (20–50 ms)
]

FEATURE_LABELS = [
    "Pre-onset\n(−10–0 ms)",
    "Early onset\n(0–10 ms)",
    "Late onset\n(10–20 ms)",
    "Sustained\n(20–50 ms)",
    "Late resp\n(50–100 ms)",
    "Pre-offset\n(−20–0 ms)",
    "Early offset\n(0–10 ms)",
    "Late offset\n(10–20 ms)",
    "Post-offset\n(20–50 ms)",
]


def extract_features(df, df_end=None):
    """
    Compute 22-dimensional feature vectors for every unit.

    Parameters
    ----------
    df : pd.DataFrame
        Onset-aligned firing-rate CSV.
    df_end : pd.DataFrame, optional
        End-aligned firing-rate CSV.  If provided, offset features
        (6–9, 14–15) are computed; otherwise they default to 0.

    Returns
    -------
    feat_df : pd.DataFrame  (one row per unit)
    """
    units = sorted(df["unit"].unique())

    # Discover whiskers present in the data
    whiskers = set()
    for name in df["interval"].unique():
        w, d = _parse_interval(name)
        if w is not None:
            whiskers.add(w)
    whiskers = sorted(whiskers)

    rows = []
    for unit in units:
        # ── Per-whisker evoked magnitude (direction=all) ──────────────
        whisker_evoked = {}   # abs(response − baseline) for best-whisker
        whisker_mod = {}      # signed modulation index for per-whisker feat
        whisker_bl = {}
        for w in whiskers:
            interval = f"interval_{w}_mask_contact"
            sub = df[(df["unit"] == unit) & (df["interval"] == interval)]
            if len(sub) == 0:
                whisker_evoked[w] = 0.0
                whisker_mod[w] = 0.0
                whisker_bl[w] = 0.0
                continue
            bins = sub["bin_ms"].values
            fr = sub["firing_rate_hz"].values
            bl = _mean_fr_window(bins, fr, -50, -10)
            resp = _mean_fr_window(bins, fr, 0, 50)
            whisker_evoked[w] = abs(resp - bl)
            whisker_mod[w] = _modulation_index(resp, bl)
            whisker_bl[w] = bl

        best_w = max(whisker_evoked, key=whisker_evoked.get)

        # ── Onset-aligned features 1-5 (best whisker, all) ───────────
        interval_all = f"interval_{best_w}_mask_contact"
        sub = df[(df["unit"] == unit) & (df["interval"] == interval_all)]
        if len(sub) == 0:
            continue
        bins = sub["bin_ms"].values
        fr = sub["firing_rate_hz"].values

        # Number of contact events for the best whisker
        n_events_bw = int(sub["n_trials"].iloc[0]) if "n_trials" in sub.columns else 0

        bl = _mean_fr_window(bins, fr, -50, -10)

        pre_onset     = _modulation_index(_mean_fr_window(bins, fr, -10, 0),   bl)
        early_onset   = _modulation_index(_mean_fr_window(bins, fr,   0, 10),  bl)
        late_onset    = _modulation_index(_mean_fr_window(bins, fr,  10, 20),  bl)
        sustained     = _modulation_index(_mean_fr_window(bins, fr,  20, 50),  bl)
        late_response = _modulation_index(_mean_fr_window(bins, fr,  50, 100), bl)

        # ── End-aligned features 6-9 (best whisker, all) ─────────────
        pre_offset   = 0.0
        early_offset = 0.0
        late_offset  = 0.0
        post_offset  = 0.0
        if df_end is not None:
            sub_e = df_end[(df_end["unit"] == unit) &
                           (df_end["interval"] == interval_all)]
            if len(sub_e) > 0:
                be = sub_e["bin_ms"].values
                fe = sub_e["firing_rate_hz"].values
                pre_offset   = _modulation_index(
                    _mean_fr_window(be, fe, -20, 0), bl)
                early_offset = _modulation_index(
                    _mean_fr_window(be, fe, 0, 10), bl)
                late_offset  = _modulation_index(
                    _mean_fr_window(be, fe, 10, 20), bl)
                post_offset  = _modulation_index(
                    _mean_fr_window(be, fe, 20, 50), bl)

        # ── Direction-specific onset features 10-13 ──────────────────
        def _dir_onset(direction, df_src, lo, hi):
            intv = f"interval_{best_w}_mask_contact_{direction}"
            s = df_src[(df_src["unit"] == unit) & (df_src["interval"] == intv)]
            if len(s) == 0:
                return 0.0
            return _modulation_index(
                _mean_fr_window(s["bin_ms"].values,
                                s["firing_rate_hz"].values, lo, hi), bl)

        onset_ret     = _dir_onset("retraction",  df,  0, 20)
        onset_pro     = _dir_onset("protraction", df,  0, 20)
        sustained_ret = _dir_onset("retraction",  df, 20, 50)
        sustained_pro = _dir_onset("protraction", df, 20, 50)

        # ── Direction-specific offset features 14-15 ─────────────────
        offset_ret = 0.0
        offset_pro = 0.0
        if df_end is not None:
            offset_ret = _dir_onset("retraction",  df_end, 0, 20)
            offset_pro = _dir_onset("protraction", df_end, 0, 20)

        # ── Per-whisker onset features 16-20 ─────────────────────────
        w_onset = {}
        for w in range(5):
            w_onset[w] = whisker_mod.get(w, 0.0)

        # ── Summary features 21-22 ───────────────────────────────────
        # DSI from best whisker (onset 0-50 ms)
        fr_ret = 0.0
        fr_pro = 0.0
        ret_int = f"interval_{best_w}_mask_contact_retraction"
        pro_int = f"interval_{best_w}_mask_contact_protraction"
        sub_ret = df[(df["unit"] == unit) & (df["interval"] == ret_int)]
        sub_pro = df[(df["unit"] == unit) & (df["interval"] == pro_int)]
        if len(sub_ret) > 0:
            fr_ret = _mean_fr_window(sub_ret["bin_ms"].values,
                                     sub_ret["firing_rate_hz"].values, 0, 50)
        if len(sub_pro) > 0:
            fr_pro = _mean_fr_window(sub_pro["bin_ms"].values,
                                     sub_pro["firing_rate_hz"].values, 0, 50)
        denom = fr_pro + fr_ret
        dsi = (fr_pro - fr_ret) / denom if denom > 0 else 0.0

        # WSI
        evoked_vals = np.array([whisker_evoked.get(w, 0) for w in whiskers])
        total = evoked_vals.sum()
        wsi = evoked_vals.max() / total if total > 0 else 0.0

        rows.append({
            "unit": unit,
            "best_whisker": best_w,
            "pre_onset": pre_onset,
            "early_onset": early_onset,
            "late_onset": late_onset,
            "sustained": sustained,
            "late_response": late_response,
            "pre_offset": pre_offset,
            "early_offset": early_offset,
            "late_offset": late_offset,
            "post_offset": post_offset,
            "onset_ret": onset_ret,
            "onset_pro": onset_pro,
            "sustained_ret": sustained_ret,
            "sustained_pro": sustained_pro,
            "offset_ret": offset_ret,
            "offset_pro": offset_pro,
            "w0_onset": w_onset[0],
            "w1_onset": w_onset[1],
            "w2_onset": w_onset[2],
            "w3_onset": w_onset[3],
            "w4_onset": w_onset[4],
            "dsi": dsi,
            "wsi": wsi,
            "baseline_fr": bl,
            "n_events_best_whisker": n_events_bw,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  Clustering
# ══════════════════════════════════════════════════════════════════════════════

def cluster_units(X_scaled, n_clusters=None, method="ward",
                  min_k=3, max_k=12):
    """
    Hierarchical clustering with gap-statistic k selection.

    Parameters
    ----------
    X_scaled : ndarray  (n_units × n_features)
        Should already be signed-log transformed & RobustScaled.
    n_clusters : int or None
        Force a specific k.  If None, the gap statistic picks k.
    method : str
        Linkage method (default ``"ward"``).
    min_k, max_k : int
        Search range for automatic k selection.

    Returns
    -------
    labels, Z, n_clusters
    """
    Z = linkage(X_scaled, method=method)

    if n_clusters is None:
        from sklearn.metrics import silhouette_score
        max_k = min(max_k, len(X_scaled) - 1)
        n_ref = 25
        rng = np.random.RandomState(42)

        gaps, ses = [], []
        for k in range(1, max_k + 1):
            # Observed WCSS
            labels_k = fcluster(Z, k, criterion="maxclust") \
                if k > 1 else np.ones(len(X_scaled), dtype=int)
            wcss = 0
            for c in np.unique(labels_k):
                mask = labels_k == c
                center = X_scaled[mask].mean(axis=0)
                wcss += np.sum((X_scaled[mask] - center) ** 2)
            log_wcss = np.log(wcss)

            # Null reference WCSS
            ref_log = []
            for _ in range(n_ref):
                X_ref = rng.uniform(X_scaled.min(axis=0),
                                    X_scaled.max(axis=0), X_scaled.shape)
                Z_ref = linkage(X_ref, method=method)
                lab_ref = fcluster(Z_ref, k, criterion="maxclust") \
                    if k > 1 else np.ones(len(X_ref), dtype=int)
                w = 0
                for c in np.unique(lab_ref):
                    m = lab_ref == c
                    if m.sum() > 0:
                        ctr = X_ref[m].mean(axis=0)
                        w += np.sum((X_ref[m] - ctr) ** 2)
                ref_log.append(np.log(w))

            gap = np.mean(ref_log) - log_wcss
            se = np.std(ref_log) * np.sqrt(1 + 1.0 / n_ref)
            gaps.append(gap)
            ses.append(se)

        # Gap criterion: smallest k ≥ min_k where gap(k) ≥ gap(k+1) − se(k+1)
        n_clusters = max_k   # fallback
        for i in range(min_k - 1, len(gaps) - 1):
            if gaps[i] >= gaps[i + 1] - ses[i + 1]:
                n_clusters = i + 1
                break

        final_labels = fcluster(Z, n_clusters, criterion="maxclust")
        sil = silhouette_score(X_scaled, final_labels)
        print(f"  Gap statistic → k={n_clusters}  "
              f"(gap={gaps[n_clusters - 1]:.3f}, silhouette={sil:.3f})")

    labels = fcluster(Z, n_clusters, criterion="maxclust")
    return labels, Z, n_clusters


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ══════════════════════════════════════════════════════════════════════════════

def _cluster_colors(n):
    cmap = plt.cm.tab10
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def plot_dendrogram(Z, units, labels, out_dir):
    """Dendrogram coloured by assigned cluster."""
    fig, ax = plt.subplots(figsize=(max(6, len(units) * 0.4), 5))
    unit_labels = [f"U{u}" for u in units]
    dendrogram(Z, labels=unit_labels, ax=ax, leaf_rotation=90,
               leaf_font_size=7, color_threshold=0)
    ax.set_ylabel("Distance", fontsize=10)
    ax.set_title("Hierarchical Clustering Dendrogram", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "dendrogram.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pca(X_scaled, units, labels, feat_names, n_clusters, out_dir):
    """PCA 2-D scatter with loading arrows."""
    pca = PCA(n_components=min(3, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    colours = _cluster_colors(n_clusters)

    # ── 2-D ──
    fig, ax = plt.subplots(figsize=(8, 6))
    for ci in range(1, n_clusters + 1):
        mask = labels == ci
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=60,
                   color=colours[ci - 1], edgecolors="black", linewidths=0.5,
                   label=f"Cluster {ci}", zorder=3)
    for i, u in enumerate(units):
        ax.annotate(f"U{u}", (X_pca[i, 0], X_pca[i, 1]),
                    fontsize=6, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 4))

    # Loading arrows
    loadings = pca.components_[:2, :]  # (2 × n_feat)
    scale = np.abs(X_pca[:, :2]).max() * 0.8
    for fi in range(loadings.shape[1]):
        lx = loadings[0, fi] * scale
        ly = loadings[1, fi] * scale
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
        ax.text(lx * 1.12, ly * 1.12, FEATURE_LABELS[fi],
                fontsize=6, color="gray", ha="center", va="center")

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=10)
    ax.set_title("PCA — Feature Space (units coloured by cluster)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.axhline(0, color="lightgray", lw=0.5, zorder=0)
    ax.axvline(0, color="lightgray", lw=0.5, zorder=0)
    fig.tight_layout()
    path = os.path.join(out_dir, "pca_2d.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # ── Variance explained bar ──
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    n_comp = len(pca.explained_variance_ratio_)
    ax2.bar(range(1, n_comp + 1), pca.explained_variance_ratio_ * 100,
            color="steelblue", edgecolor="black")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Variance Explained (%)")
    ax2.set_title("PCA Scree Plot")
    ax2.set_xticks(range(1, n_comp + 1))
    fig2.tight_layout()
    path2 = os.path.join(out_dir, "pca_scree.png")
    fig2.savefig(path2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved {path2}")


def plot_tsne(X_scaled, units, labels, n_clusters, out_dir):
    """t-SNE 2-D."""
    n = len(units)
    if n < 4:
        print("  Too few units for t-SNE — skipping.")
        return
    perp = min(15, max(2, n // 4))
    try:
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                    init="pca", learning_rate="auto")
        X_tsne = tsne.fit_transform(X_scaled)
    except Exception as e:
        print(f"  t-SNE failed ({e}) — skipping.")
        return
    colours = _cluster_colors(n_clusters)

    fig, ax = plt.subplots(figsize=(7, 6))
    for ci in range(1, n_clusters + 1):
        mask = labels == ci
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=60,
                   color=colours[ci - 1], edgecolors="black", linewidths=0.5,
                   label=f"Cluster {ci}", zorder=3)
    for i, u in enumerate(units):
        ax.annotate(f"U{u}", (X_tsne[i, 0], X_tsne[i, 1]),
                    fontsize=6, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 4))
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_title("t-SNE — Feature Space", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = os.path.join(out_dir, "tsne_2d.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_umap(X_scaled, units, labels, n_clusters, out_dir):
    """UMAP 2-D (skipped if umap-learn not installed)."""
    if not HAS_UMAP:
        print("  UMAP not installed — skipping.  pip install umap-learn")
        return
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(10, len(units)-1))
    X_umap = reducer.fit_transform(X_scaled)
    colours = _cluster_colors(n_clusters)

    fig, ax = plt.subplots(figsize=(7, 6))
    for ci in range(1, n_clusters + 1):
        mask = labels == ci
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], s=60,
                   color=colours[ci - 1], edgecolors="black", linewidths=0.5,
                   label=f"Cluster {ci}", zorder=3)
    for i, u in enumerate(units):
        ax.annotate(f"U{u}", (X_umap[i, 0], X_umap[i, 1]),
                    fontsize=6, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 4))
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title("UMAP — Feature Space", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = os.path.join(out_dir, "umap_2d.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_radar(feat_df, labels, n_clusters, out_dir):
    """Radar (spider) chart of mean feature values per cluster."""
    n_feat = len(FEATURE_NAMES)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon
    colours = _cluster_colors(n_clusters)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    for ci in range(1, n_clusters + 1):
        mask = labels == ci
        means = feat_df.loc[mask, FEATURE_NAMES].mean().values.tolist()
        means += means[:1]
        ax.plot(angles, means, linewidth=2, color=colours[ci - 1],
                label=f"Cluster {ci} (n={mask.sum()})")
        ax.fill(angles, means, alpha=0.15, color=colours[ci - 1])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(FEATURE_LABELS, fontsize=5)
    ax.set_title("Cluster Feature Profiles", fontsize=12, fontweight="bold",
                 pad=20)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.15))
    fig.tight_layout()
    path = os.path.join(out_dir, "cluster_radar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_parallel_coordinates(feat_df, labels, n_clusters, out_dir):
    """Parallel coordinates plot — one line per unit, coloured by cluster."""
    colours = _cluster_colors(n_clusters)
    fig, ax = plt.subplots(figsize=(max(12, len(FEATURE_NAMES) * 0.6), 5))
    x = np.arange(len(FEATURE_NAMES))

    for ci in range(1, n_clusters + 1):
        mask = labels == ci
        sub = feat_df.loc[mask, FEATURE_NAMES].values
        for row in sub:
            ax.plot(x, row, color=colours[ci - 1], alpha=0.4, lw=1)
        # Cluster mean
        mean_vals = sub.mean(axis=0)
        ax.plot(x, mean_vals, color=colours[ci - 1], lw=2.5,
                marker="o", markersize=5, label=f"Cluster {ci}")

    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_LABELS, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Standardised feature value", fontsize=10)
    ax.set_title("Parallel Coordinates (unit feature profiles)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    fig.tight_layout()
    path = os.path.join(out_dir, "parallel_coordinates.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_feature_heatmap(feat_df_std, units, labels, n_clusters, out_dir):
    """Heatmap of standardised features, rows sorted by cluster then unit."""
    order = np.argsort(labels)
    matrix = feat_df_std.loc[order, FEATURE_NAMES].values
    sorted_units = [units[i] for i in order]
    sorted_labels = labels[order]

    fig, ax = plt.subplots(figsize=(max(10, len(FEATURE_NAMES) * 0.5),
                                     max(4, len(units) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest",
                   vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_LABELS, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(sorted_units)))
    ax.set_yticklabels([f"U{u} (C{l})" for u, l in
                        zip(sorted_units, sorted_labels)], fontsize=6)
    ax.set_ylabel("Unit (sorted by cluster)")
    ax.set_title("Standardised Feature Heatmap", fontsize=12,
                 fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Z-score")

    # Cluster boundaries
    boundaries = np.where(np.diff(sorted_labels))[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="black", lw=1.5)

    fig.tight_layout()
    path = os.path.join(out_dir, "feature_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def run(csv_path, output_dir=None, n_clusters=None, csv_end_path=None):
    print(f"Loading onset-aligned: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df)} rows, {df['unit'].nunique()} units")

    # Discover end-aligned CSV if not supplied explicitly
    df_end = None
    if csv_end_path is None:
        # Try sibling directory
        parent = os.path.dirname(os.path.dirname(csv_path))
        candidate = os.path.join(
            parent, "contact_psth_end_aligned_csv_output",
            "contact_psth_end_aligned_firing_rates.csv")
        if os.path.isfile(candidate):
            csv_end_path = candidate
    if csv_end_path and os.path.isfile(csv_end_path):
        print(f"Loading end-aligned:   {csv_end_path}")
        df_end = pd.read_csv(csv_end_path)
        print(f"  {len(df_end)} rows, {df_end['unit'].nunique()} units")
    else:
        print("  No end-aligned CSV found — offset features will be zero.")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(csv_path),
                                  "feature_space")
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Extract features ───────────────────────────────────────────
    print("\n1. Extracting features …")
    feat_df = extract_features(df, df_end=df_end)
    feat_csv = os.path.join(output_dir, "unit_features.csv")
    feat_df.to_csv(feat_csv, index=False)
    print(f"   Saved {feat_csv}")
    print(feat_df[["unit", "best_whisker"] + FEATURE_NAMES].to_string(
        index=False))

    # ── 1b. Filter non-responsive units ───────────────────────────────
    print("\n   Filtering non-responsive units …")
    feat_df_all = feat_df.copy()  # keep full table for CSV
    feat_df = filter_responsive_units(feat_df)
    feat_df_all.to_csv(feat_csv, index=False)  # update with responsive col

    units = feat_df["unit"].tolist()
    X_raw = feat_df[FEATURE_NAMES].values

    if len(units) < 3:
        print("\n  Fewer than 3 units — clustering/dim-reduction not meaningful.")
        return

    # ── 2. Standardise ────────────────────────────────────────────────
    print("\n2. Standardising features (StandardScaler) …")
    print("   Clustering on 9 core temporal features")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    feat_df_std = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)

    # ── 3. Cluster ────────────────────────────────────────────────────
    print("\n3. Hierarchical clustering …")
    labels, Z, n_clusters = cluster_units(X_scaled, n_clusters=n_clusters)
    feat_df["cluster"] = labels
    feat_df.to_csv(feat_csv, index=False)  # update with cluster col

    # Print cluster assignments
    for ci in range(1, n_clusters + 1):
        members = feat_df.loc[labels == ci, "unit"].tolist()
        print(f"   Cluster {ci}: {['U'+str(u) for u in members]}")

    # ── 4. Plots ──────────────────────────────────────────────────────
    print("\n4. Generating plots …")
    plot_dendrogram(Z, units, labels, output_dir)
    plot_pca(X_scaled, units, labels, FEATURE_NAMES, n_clusters, output_dir)
    plot_tsne(X_scaled, units, labels, n_clusters, output_dir)
    plot_umap(X_scaled, units, labels, n_clusters, output_dir)
    plot_radar(feat_df, labels, n_clusters, output_dir)
    plot_parallel_coordinates(feat_df_std, labels, n_clusters, output_dir)
    plot_feature_heatmap(feat_df_std, units, labels, n_clusters, output_dir)

    print(f"\n{'='*60}")
    print(f"Done.  All outputs in {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Feature-space unit classification via dimensionality "
                    "reduction and unsupervised clustering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python feature_space_analysis.py --data_dir "C:\\path\\to\\session"
  python feature_space_analysis.py --csv "path/to/firing_rates.csv" --n_clusters 3
        """,
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--data_dir", help="Session directory")
    g.add_argument("--csv", help="Direct path to contact_psth_firing_rates.csv")
    parser.add_argument("--csv_end", default=None,
                        help="Path to end-aligned firing-rates CSV "
                             "(auto-detected if omitted)")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Number of clusters (default: auto via gap statistic)")
    parser.add_argument("--min_events", type=int, default=None,
                        help="Minimum contact events for best whisker "
                             "(default: %(default)s)")
    parser.add_argument("--min_mod", type=float, default=None,
                        help="Minimum max |modulation index| to be "
                             "considered responsive (default: %(default)s)")
    args = parser.parse_args()

    # Override module-level thresholds if CLI flags provided
    import feature_space_analysis as _self
    if args.min_events is not None:
        _self.MIN_EVENTS = args.min_events
    if args.min_mod is not None:
        _self.MIN_MAX_MOD_INDEX = args.min_mod

    if args.csv:
        csv_path = args.csv
    else:
        csv_path = os.path.join(args.data_dir, "contact_psth_csv_output",
                                "contact_psth_firing_rates.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    csv_end_path = args.csv_end
    if csv_end_path is None and args.data_dir:
        csv_end_path = os.path.join(
            args.data_dir, "contact_psth_end_aligned_csv_output",
            "contact_psth_end_aligned_firing_rates.csv")

    run(csv_path, output_dir=args.output_dir, n_clusters=args.n_clusters,
        csv_end_path=csv_end_path)


if __name__ == "__main__":
    main()
