"""
Batch Cross-Session Feature-Space Clustering
=============================================

Pools units from multiple sessions, extracts PSTH response features, and
performs unsupervised clustering on the combined population.

Clustering uses the 9 core temporal features (onset-aligned + end-aligned)
with signed-log transform + RobustScaler.  The gap statistic selects k
(minimum k ≥ 3).

Each unit is labelled as  ``session | Uxx``  so clusters can be interpreted
across recording days.

Outputs (saved to ``<base_dir>/batch_feature_space/``):
    - batch_unit_features.csv       combined feature table with cluster labels
    - dendrogram.png                hierarchical clustering dendrogram
    - pca_2d.png                    PCA scatter (coloured by cluster)
    - pca_2d_by_session.png         PCA scatter (coloured by session)
    - pca_scree.png                 variance-explained bar chart
    - tsne_2d.png                   t-SNE scatter
    - cluster_radar.png             radar chart of mean cluster profiles
    - parallel_coordinates.png      parallel coordinates per unit
    - feature_heatmap.png           z-scored feature heatmap
    - cluster_session_table.csv     cluster × session contingency table

Usage
-----
    python batch_feature_space.py
    python batch_feature_space.py --n_clusters 5
    python batch_feature_space.py --base_dir "D:\\my_data"
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

warnings.filterwarnings("ignore", category=FutureWarning)

# Try optional t-SNE / UMAP
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# ── Import feature helpers from the single-session script ─────────────────
# Add Contact_PSTH to path so we can import from feature_space_analysis
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_space_analysis import (
    extract_features, FEATURE_NAMES, FEATURE_LABELS,
    ALL_FEATURE_NAMES, ALL_FEATURE_LABELS,
    _cluster_colors, cluster_units, signed_log_transform,
    filter_responsive_units,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Default session list
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_BASE = r"C:\Users\wanglab\Desktop\Club Like Endings"

DEFAULT_SESSIONS = [
    "101925_1",
    "101925_2",
    "102225_1",
    "102225_2",
    "102525_1",
    "102625_1",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Feature extraction across sessions
# ══════════════════════════════════════════════════════════════════════════════

def load_all_features(base_dir, sessions):
    """
    Load firing-rate CSVs (onset + end-aligned) from each session,
    extract features, and return a combined DataFrame with a ``session``
    column.
    """
    all_feats = []
    for session in sessions:
        csv_path = os.path.join(base_dir, session,
                                "contact_psth_csv_output",
                                "contact_psth_firing_rates.csv")
        if not os.path.isfile(csv_path):
            print(f"  WARNING: onset CSV not found for {session} — skipping.")
            continue

        print(f"  Loading {session} …")
        df = pd.read_csv(csv_path)

        # Try to load end-aligned CSV
        csv_end_path = os.path.join(
            base_dir, session,
            "contact_psth_end_aligned_csv_output",
            "contact_psth_end_aligned_firing_rates.csv")
        df_end = None
        if os.path.isfile(csv_end_path):
            df_end = pd.read_csv(csv_end_path)
            print(f"    (end-aligned CSV found)")
        else:
            print(f"    (no end-aligned CSV — offset features = 0)")

        feat = extract_features(df, df_end=df_end)
        if len(feat) == 0:
            print(f"    No units extracted for {session} — skipping.")
            continue
        feat["session"] = session
        all_feats.append(feat)
        print(f"    {len(feat)} units")

    if len(all_feats) == 0:
        raise RuntimeError("No features could be extracted from any session.")

    combined = pd.concat(all_feats, ignore_index=True)
    # Create a unique label for each unit across sessions
    combined["label"] = combined["session"] + " | U" + combined["unit"].astype(str)
    return combined


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers  (adapted for cross-session labels)
# ══════════════════════════════════════════════════════════════════════════════

SESSION_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _session_colors(sessions):
    cmap = plt.cm.Set2
    return {s: cmap(i / max(len(sessions) - 1, 1)) for i, s in enumerate(sessions)}


def plot_dendrogram(Z, unit_labels, labels, out_dir):
    fig, ax = plt.subplots(figsize=(max(8, len(unit_labels) * 0.35), 6))
    dendrogram(Z, labels=unit_labels, ax=ax, leaf_rotation=90,
               leaf_font_size=5, color_threshold=0)
    ax.set_ylabel("Distance", fontsize=10)
    ax.set_title("Hierarchical Clustering — All Sessions",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "dendrogram.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_pca_by_cluster(X_scaled, unit_labels, labels, n_clusters, out_dir):
    pca = PCA(n_components=min(3, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    colours = _cluster_colors(n_clusters)

    fig, ax = plt.subplots(figsize=(10, 7))
    for ci in range(1, n_clusters + 1):
        mask = labels == ci
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=50,
                   color=colours[ci - 1], edgecolors="black", linewidths=0.4,
                   label=f"Cluster {ci}", zorder=3)
    for i, lbl in enumerate(unit_labels):
        ax.annotate(lbl, (X_pca[i, 0], X_pca[i, 1]),
                    fontsize=4, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 3))

    # Loading arrows
    loadings = pca.components_[:2, :]
    scale = np.abs(X_pca[:, :2]).max() * 0.8
    for fi in range(loadings.shape[1]):
        lx = loadings[0, fi] * scale
        ly = loadings[1, fi] * scale
        ax.annotate("", xy=(lx, ly), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))
        ax.text(lx * 1.12, ly * 1.12, FEATURE_LABELS[fi],
                fontsize=5, color="gray", ha="center", va="center")

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=10)
    ax.set_title("PCA — Coloured by Cluster (all sessions pooled)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.axhline(0, color="lightgray", lw=0.5, zorder=0)
    ax.axvline(0, color="lightgray", lw=0.5, zorder=0)
    fig.tight_layout()
    path = os.path.join(out_dir, "pca_2d.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

    # Scree
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

    return pca, X_pca


def plot_pca_by_session(X_pca, pca, unit_labels, sessions_arr, out_dir):
    """Same PCA projection but coloured by session instead of cluster."""
    unique_sessions = sorted(set(sessions_arr))
    sess_colours = _session_colors(unique_sessions)

    fig, ax = plt.subplots(figsize=(10, 7))
    for si, sess in enumerate(unique_sessions):
        mask = sessions_arr == sess
        marker = SESSION_MARKERS[si % len(SESSION_MARKERS)]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=50,
                   color=sess_colours[sess], edgecolors="black",
                   linewidths=0.4, marker=marker,
                   label=sess, zorder=3)
    for i, lbl in enumerate(unit_labels):
        ax.annotate(lbl, (X_pca[i, 0], X_pca[i, 1]),
                    fontsize=4, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 3))

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=10)
    ax.set_title("PCA — Coloured by Session",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.axhline(0, color="lightgray", lw=0.5, zorder=0)
    ax.axvline(0, color="lightgray", lw=0.5, zorder=0)
    fig.tight_layout()
    path = os.path.join(out_dir, "pca_2d_by_session.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_tsne(X_scaled, unit_labels, labels, n_clusters, out_dir):
    n = len(unit_labels)
    if n < 4:
        print("  Too few units for t-SNE — skipping.")
        return
    if not HAS_TSNE:
        print("  TSNE import unavailable — skipping.")
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
    fig, ax = plt.subplots(figsize=(10, 7))
    for ci in range(1, n_clusters + 1):
        mask = labels == ci
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=50,
                   color=colours[ci - 1], edgecolors="black", linewidths=0.4,
                   label=f"Cluster {ci}", zorder=3)
    for i, lbl in enumerate(unit_labels):
        ax.annotate(lbl, (X_tsne[i, 0], X_tsne[i, 1]),
                    fontsize=4, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 3))
    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_title("t-SNE — All Sessions Pooled",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = os.path.join(out_dir, "tsne_2d.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_umap(X_scaled, unit_labels, labels, n_clusters, out_dir):
    if not HAS_UMAP:
        print("  UMAP not installed — skipping.  pip install umap-learn")
        return
    n = len(unit_labels)
    reducer = UMAP(n_components=2, random_state=42,
                   n_neighbors=min(10, n - 1))
    X_umap = reducer.fit_transform(X_scaled)
    colours = _cluster_colors(n_clusters)

    fig, ax = plt.subplots(figsize=(10, 7))
    for ci in range(1, n_clusters + 1):
        mask = labels == ci
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], s=50,
                   color=colours[ci - 1], edgecolors="black", linewidths=0.4,
                   label=f"Cluster {ci}", zorder=3)
    for i, lbl in enumerate(unit_labels):
        ax.annotate(lbl, (X_umap[i, 0], X_umap[i, 1]),
                    fontsize=4, ha="center", va="bottom",
                    textcoords="offset points", xytext=(0, 3))
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_title("UMAP — All Sessions Pooled",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = os.path.join(out_dir, "umap_2d.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_radar(feat_df, labels, n_clusters, out_dir):
    n_feat = len(FEATURE_NAMES)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]
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
    ax.set_title("Cluster Feature Profiles — All Sessions",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.35, 1.15))
    fig.tight_layout()
    path = os.path.join(out_dir, "cluster_radar.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_parallel_coordinates(feat_df_std, labels, n_clusters, out_dir):
    colours = _cluster_colors(n_clusters)
    fig, ax = plt.subplots(figsize=(max(12, len(FEATURE_NAMES) * 0.6), 5))
    x = np.arange(len(FEATURE_NAMES))

    for ci in range(1, n_clusters + 1):
        mask = labels == ci
        sub = feat_df_std.loc[mask, FEATURE_NAMES].values
        for row in sub:
            ax.plot(x, row, color=colours[ci - 1], alpha=0.25, lw=0.8)
        mean_vals = sub.mean(axis=0)
        ax.plot(x, mean_vals, color=colours[ci - 1], lw=2.5,
                marker="o", markersize=5, label=f"Cluster {ci}")

    ax.set_xticks(x)
    ax.set_xticklabels(FEATURE_LABELS, fontsize=6, rotation=45, ha="right")
    ax.set_ylabel("Standardised feature value", fontsize=10)
    ax.set_title("Parallel Coordinates — All Sessions",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="best")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    fig.tight_layout()
    path = os.path.join(out_dir, "parallel_coordinates.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_feature_heatmap(feat_df_std, unit_labels, labels, n_clusters,
                         sessions_arr, out_dir):
    order = np.argsort(labels)
    matrix = feat_df_std.loc[order, FEATURE_NAMES].values
    sorted_labels_arr = labels[order]
    sorted_unit_labels = [unit_labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(max(12, len(FEATURE_NAMES) * 0.5),
                                     max(5, len(unit_labels) * 0.25)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest", vmin=-2.5, vmax=2.5)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_LABELS, fontsize=5, rotation=45, ha="right")
    ax.set_yticks(range(len(sorted_unit_labels)))
    ax.set_yticklabels([f"{lbl} (C{c})" for lbl, c in
                        zip(sorted_unit_labels, sorted_labels_arr)],
                       fontsize=4)
    ax.set_ylabel("Unit (sorted by cluster)")
    ax.set_title("Standardised Feature Heatmap — All Sessions",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Z-score")

    boundaries = np.where(np.diff(sorted_labels_arr))[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="black", lw=1.5)

    fig.tight_layout()
    path = os.path.join(out_dir, "feature_heatmap.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cluster_session_composition(labels, n_clusters, sessions_arr,
                                     out_dir):
    """Stacked bar chart showing session composition within each cluster."""
    unique_sessions = sorted(set(sessions_arr))
    sess_colours = _session_colors(unique_sessions)

    # Build contingency table
    data = {}
    for sess in unique_sessions:
        counts = []
        for ci in range(1, n_clusters + 1):
            counts.append(np.sum((labels == ci) & (sessions_arr == sess)))
        data[sess] = counts

    ct_df = pd.DataFrame(data, index=[f"Cluster {ci}"
                                       for ci in range(1, n_clusters + 1)])
    ct_df.to_csv(os.path.join(out_dir, "cluster_session_table.csv"))
    print(f"  Saved cluster_session_table.csv")

    fig, ax = plt.subplots(figsize=(max(6, n_clusters * 1.2), 5))
    x = np.arange(n_clusters)
    bottom = np.zeros(n_clusters)
    bar_width = 0.6

    for sess in unique_sessions:
        vals = np.array(data[sess], dtype=float)
        ax.bar(x, vals, bar_width, bottom=bottom, label=sess,
               color=sess_colours[sess], edgecolor="black", linewidth=0.5)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"C{ci}" for ci in range(1, n_clusters + 1)])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of units")
    ax.set_title("Cluster Composition by Session",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, title="Session", loc="best")
    ax.set_ylim(0, bottom.max() * 1.15)
    fig.tight_layout()
    path = os.path.join(out_dir, "cluster_session_composition.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run(base_dir, sessions, output_dir, n_clusters=None):
    print("=" * 60)
    print("  Batch Feature-Space Clustering")
    print("=" * 60)

    # ── 1. Load & extract features ────────────────────────────────────
    print("\n1. Loading sessions and extracting features …")
    feat_df = load_all_features(base_dir, sessions)
    print(f"\n   Total units: {len(feat_df)} across "
          f"{feat_df['session'].nunique()} sessions")

    os.makedirs(output_dir, exist_ok=True)

    # ── 2. Save full feature table & filter non-responsive units ──────
    feat_csv = os.path.join(output_dir, "batch_unit_features.csv")

    print("\n   Filtering non-responsive units …")
    feat_df_all = feat_df.copy()
    feat_df = filter_responsive_units(feat_df)
    feat_df_all.to_csv(feat_csv, index=False)  # full table with responsive col
    print(f"   {len(feat_df)} responsive units retained for clustering")

    unit_labels = feat_df["label"].tolist()
    sessions_arr = feat_df["session"].values
    X_raw = feat_df[FEATURE_NAMES].values

    if len(feat_df) < 3:
        print("\n  Fewer than 3 total units — cannot cluster.")
        return

    # ── 3. Standardise ────────────────────────────────────────────────
    print("\n2. Standardising features (StandardScaler) …")
    print("   Clustering on 9 core temporal features")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    feat_df_std = pd.DataFrame(X_scaled, columns=FEATURE_NAMES)

    # ── 4. Cluster ────────────────────────────────────────────────────
    print("\n3. Hierarchical clustering …")
    labels, Z, n_clusters = cluster_units(X_scaled, n_clusters=n_clusters)
    feat_df["cluster"] = labels
    feat_df.to_csv(feat_csv, index=False)  # update with cluster column

    for ci in range(1, n_clusters + 1):
        members = feat_df.loc[labels == ci, "label"].tolist()
        print(f"   Cluster {ci} ({len(members)} units):")
        for m in members:
            print(f"      {m}")

    # ── 5. Plots ──────────────────────────────────────────────────────
    print("\n4. Generating plots …")
    plot_dendrogram(Z, unit_labels, labels, output_dir)
    pca_obj, X_pca = plot_pca_by_cluster(X_scaled, unit_labels, labels,
                                          n_clusters, output_dir)
    plot_pca_by_session(X_pca, pca_obj, unit_labels, sessions_arr, output_dir)
    plot_tsne(X_scaled, unit_labels, labels, n_clusters, output_dir)
    plot_umap(X_scaled, unit_labels, labels, n_clusters, output_dir)
    plot_radar(feat_df, labels, n_clusters, output_dir)
    plot_parallel_coordinates(feat_df_std, labels, n_clusters, output_dir)
    plot_feature_heatmap(feat_df_std, unit_labels, labels, n_clusters,
                         sessions_arr, output_dir)
    plot_cluster_session_composition(labels, n_clusters, sessions_arr,
                                     output_dir)

    print(f"\n{'=' * 60}")
    print(f"  Done.  All outputs → {output_dir}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch cross-session feature-space clustering.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base_dir", default=DEFAULT_BASE,
                        help="Parent directory containing session folders "
                             f"(default: {DEFAULT_BASE})")
    parser.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS,
                        help="Session folder names "
                             f"(default: {DEFAULT_SESSIONS})")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: <base_dir>/"
                             "batch_feature_space)")
    parser.add_argument("--n_clusters", type=int, default=None,
                        help="Force number of clusters "
                             "(default: auto via gap statistic)")
    parser.add_argument("--min_events", type=int, default=None,
                        help="Minimum contact events for best whisker "
                             "(default: module constant MIN_EVENTS)")
    parser.add_argument("--min_mod", type=float, default=None,
                        help="Minimum max |modulation index| to be "
                             "considered responsive (default: module "
                             "constant MIN_MAX_MOD_INDEX)")
    args = parser.parse_args()

    # Override thresholds if CLI flags provided
    import feature_space_analysis as _fsa
    if args.min_events is not None:
        _fsa.MIN_EVENTS = args.min_events
    if args.min_mod is not None:
        _fsa.MIN_MAX_MOD_INDEX = args.min_mod

    output_dir = args.output_dir or os.path.join(args.base_dir,
                                                  "batch_feature_space")
    run(args.base_dir, args.sessions, output_dir, args.n_clusters)


if __name__ == "__main__":
    main()
