"""
Cluster changepoint-defined segments by feature group.

Reads segment summary CSVs from changepoint_feature_groups.py, builds per-segment
feature vectors (summary mean + std + log10 n_frames), and runs hierarchical
clustering separately for gsa, iodine, na_water, and combined groups.

Example
-------
python scripts/cluster_changepoint_segments.py \\
    --changepoints-dir output/changepoints \\
    --output-dir output/changepoints/clusters
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.cluster_inspection import (
    state_transitions_from_dataframe,
    write_all_k_cluster_results,
    write_cluster_inspection,
)
from src.utils.metastable_states import (
    Segment,
    assign_cluster_labels,
    cluster_metastable_states,
    format_cluster_summary,
    plot_dendrogram,
    prepare_feature_clustering,
)
from src.utils.run_log import RunContext, log_event, step

# Import _SUMMARY_COLS from changepoint_feature_groups without package import issues.
_cpfg_spec = importlib.util.spec_from_file_location(
    "changepoint_feature_groups",
    ROOT / "scripts" / "changepoint_feature_groups.py",
)
_cpfg = importlib.util.module_from_spec(_cpfg_spec)
_cpfg_spec.loader.exec_module(_cpfg)
_SUMMARY_COLS: dict[str, list[str]] = _cpfg._SUMMARY_COLS

DEFAULT_GROUPS = ("gsa", "iodine", "na_water", "combined")
META_COLS = frozenset({
    "traj_id", "group", "segment_id", "start_frame", "end_frame",
    "start_ps", "end_ps", "n_frames", "n_cols_used",
})


def _summary_feature_columns(group: str) -> list[str]:
    """Expand summary base columns to mean + std CSV column names."""
    bases = _SUMMARY_COLS.get(group, [])
    cols: list[str] = []
    for base in bases:
        cols.append(f"{base}_mean")
        cols.append(f"{base}_std")
    return cols


def load_segment_stats(changepoints_dir: Path) -> pd.DataFrame:
    """Load merged all_segment_stats.csv or stack per-trajectory files."""
    merged = changepoints_dir / "all_segment_stats.csv"
    if merged.exists():
        return pd.read_csv(merged)

    parts = sorted(changepoints_dir.glob("*_segment_stats.csv"))
    if not parts:
        raise FileNotFoundError(
            f"No segment stats found in {changepoints_dir} "
            "(expected all_segment_stats.csv or *_segment_stats.csv)"
        )
    return pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)


def build_feature_matrix(
    df: pd.DataFrame,
    group: str,
) -> tuple[np.ndarray, list[str]]:
    """Build (n_segments, n_features) matrix: summary mean/std + log10(n_frames)."""
    sub = df[df["group"] == group].copy()
    if sub.empty:
        raise ValueError(f"No segments for group '{group}'")

    candidate_cols = _summary_feature_columns(group)
    available = [c for c in candidate_cols if c in sub.columns]
    # Drop all-NaN columns within this group.
    keep: list[str] = []
    for col in available:
        if sub[col].notna().any():
            keep.append(col)

    mat = sub[keep].to_numpy(dtype=np.float64) if keep else np.empty((len(sub), 0))
    if mat.size > 0:
        for j in range(mat.shape[1]):
            col_vals = mat[:, j]
            finite = col_vals[np.isfinite(col_vals)]
            fill = float(np.median(finite)) if len(finite) else 0.0
            bad = ~np.isfinite(col_vals)
            if bad.any():
                mat[bad, j] = fill

    duration = np.log10(np.maximum(sub["n_frames"].to_numpy(dtype=np.float64), 1.0))
    if mat.shape[1] == 0:
        mat = duration.reshape(-1, 1)
    else:
        mat = np.column_stack([mat, duration])

    feature_names = keep + ["log10_n_frames"]
    return mat, feature_names


def _segments_from_df(df: pd.DataFrame) -> list[Segment]:
    segments: list[Segment] = []
    for _, row in df.iterrows():
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        rep = (start + end) // 2
        segments.append(
            Segment(
                traj_id=str(row["traj_id"]),
                segment_id=int(row["segment_id"]),
                start_frame=start,
                end_frame=end,
                rep_frame=rep,
                rmsd_mean=float(row.get("assembly_rmsd_to_ref_mean", np.nan))
                if pd.notna(row.get("assembly_rmsd_to_ref_mean"))
                else float("nan"),
                rg_mean=float(row.get("assembly_rg_mean", np.nan))
                if pd.notna(row.get("assembly_rg_mean"))
                else float("nan"),
            )
        )
    return segments


def plot_pca_clusters(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    leaf_labels: list[str],
    output_path: Path,
    *,
    title: str,
) -> None:
    """2D PCA scatter colored by cluster label."""
    n = feature_matrix.shape[0]
    if n < 3:
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)
    X = np.nan_to_num(X, nan=0.0)

    n_comp = min(2, X.shape[1], n - 1)
    pca = PCA(n_components=n_comp)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(int(l) for l in labels))
    cmap = plt.cm.tab10
    for lab in unique_labels:
        mask = labels == lab
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1] if n_comp > 1 else np.zeros(mask.sum()),
            label=f"cluster {lab}",
            alpha=0.7,
            s=40,
            c=[cmap(lab % 10)],
        )

    var = pca.explained_variance_ratio_
    xlabel = f"PC1 ({var[0]:.1%} var)" if len(var) > 0 else "PC1"
    ylabel = f"PC2 ({var[1]:.1%} var)" if len(var) > 1 else "PC2"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def cluster_one_group(
    df: pd.DataFrame,
    group: str,
    out_dir: Path,
    *,
    linkage: str,
    n_clusters: int,
    silhouette_k_max: int,
    all_k_min: int,
    all_k_max: int,
    save_all_k: bool,
    distance_cutoff: Optional[float],
    auto_select_k: bool,
    k_max: int,
    skip_pca: bool,
) -> pd.DataFrame:
    """Cluster segments for one feature group; write per-group artifacts."""
    sub = df[df["group"] == group].copy().reset_index(drop=True)
    if sub.empty:
        log_event(
            "warning",
            f"No segments for group '{group}', skipping",
            component="cluster_changepoint_segments",
        )
        return pd.DataFrame()

    feature_matrix, feature_names = build_feature_matrix(df, group)
    feat_input = prepare_feature_clustering(feature_matrix)
    clustering = cluster_metastable_states(
        feat_input.distance_matrix,
        scaled_features=feat_input.scaled_features,
        method=linkage,
        n_clusters=n_clusters if distance_cutoff is None and not auto_select_k else None,
        rmsd_cutoff=distance_cutoff,
        auto_select_k=auto_select_k,
        k_max=k_max,
    )

    segments = _segments_from_df(sub)
    assign_cluster_labels(segments, clustering)

    base_segments = _segments_from_df(sub)

    leaf_labels = [f"{s.traj_id}:seg{s.segment_id}" for s in segments]
    sub["leaf_label"] = leaf_labels
    sub["cluster_label"] = [s.cluster_label for s in segments]
    sub["features_used"] = ";".join(feature_names)

    group_dir = out_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)

    sub.to_csv(group_dir / "segments_clustered.csv", index=False)

    pd.DataFrame(
        feat_input.distance_matrix, index=leaf_labels, columns=leaf_labels
    ).to_csv(
        group_dir / "distance_matrix.csv"
    )

    plot_dendrogram(
        clustering.linkage_matrix,
        leaf_labels,
        group_dir / "dendrogram.png",
        title=f"Changepoint segments — {group} (k={clustering.n_clusters})",
        rmsd_cutoff=clustering.cutoff_distance if distance_cutoff else None,
    )

    summary_path = group_dir / "cluster_summary.txt"
    summary_path.write_text(
        format_cluster_summary(segments, clustering),
        encoding="utf-8",
    )

    if not skip_pca:
        plot_pca_clusters(
            feature_matrix,
            clustering.labels,
            leaf_labels,
            group_dir / "pca_clusters.png",
            title=f"PCA — {group} segments (k={clustering.n_clusters})",
        )

    write_cluster_inspection(
        group_dir,
        segments,
        clustering.labels,
        feat_input.distance_matrix,
        clustering.linkage_matrix,
        chosen_k=clustering.n_clusters,
        feature_matrix=feature_matrix,
        feature_names=feature_names,
        leaf_labels=leaf_labels,
        silhouette_k_max=silhouette_k_max,
        group_label=group,
    )

    use_fixed_k = distance_cutoff is None and not auto_select_k
    if save_all_k and use_fixed_k:
        write_all_k_cluster_results(
            group_dir,
            base_segments,
            feat_input.distance_matrix,
            clustering.linkage_matrix,
            k_min=all_k_min,
            k_max=all_k_max,
            chosen_k=n_clusters,
            linkage_method=clustering.method,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            leaf_labels=leaf_labels,
            group_label=group,
        )

    log_event(
        "info",
        f"group={group}: {len(sub)} segments → {clustering.n_clusters} clusters "
        f"({clustering.selection_mode}, silhouette={clustering.silhouette:.3f})",
        component="cluster_changepoint_segments",
    )
    return sub


def write_cohort_outputs(
    clustered_parts: list[pd.DataFrame],
    out_dir: Path,
) -> None:
    """Write merged cohort-level cluster tables."""
    if not clustered_parts:
        return

    all_clustered = pd.concat(clustered_parts, ignore_index=True)
    all_clustered.to_csv(out_dir / "all_segments_clustered.csv", index=False)

    counts = (
        all_clustered.groupby(["group", "cluster_label"])
        .size()
        .reset_index(name="n_segments")
        .sort_values(["group", "cluster_label"])
    )
    counts.to_csv(out_dir / "cluster_counts_by_group.csv", index=False)

    trans_dir = out_dir / "transitions"
    trans_dir.mkdir(parents=True, exist_ok=True)
    for group, grp_df in all_clustered.groupby("group"):
        n_cl = int(grp_df["cluster_label"].max()) + 1
        paths, summary, count_df, prob_df = state_transitions_from_dataframe(
            grp_df,
            n_clusters=n_cl,
        )
        group_trans = trans_dir / group
        group_trans.mkdir(parents=True, exist_ok=True)
        paths.to_csv(group_trans / "trajectory_state_paths.csv", index=False)
        summary.to_csv(group_trans / "trajectory_state_summary.csv", index=False)
        count_df.to_csv(group_trans / "state_transition_matrix.csv")
        prob_df.to_csv(group_trans / "state_transition_prob.csv")

    # Wide composition: one row per trajectory, columns per group×segment cluster
    rows: list[dict] = []
    for traj_id, grp_df in all_clustered.groupby("traj_id"):
        row: dict = {"traj_id": traj_id}
        for _, seg in grp_df.iterrows():
            key = f"{seg['group']}_seg{int(seg['segment_id'])}_cluster"
            row[key] = int(seg["cluster_label"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        out_dir / "cluster_composition_by_trajectory.csv", index=False
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster changepoint segment stats by feature group."
    )
    parser.add_argument(
        "--changepoints-dir",
        default="output/changepoints",
        help="Directory with all_segment_stats.csv (default: output/changepoints)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Cluster output directory (default: {changepoints-dir}/clusters)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        choices=list(DEFAULT_GROUPS),
        help="Feature groups to cluster (default: all four)",
    )
    parser.add_argument(
        "--linkage",
        default="ward",
        help="SciPy linkage method (ward on scaled features; average/complete/weighted for precomputed distances)",
    )
    parser.add_argument(
        "--n-clusters",
        "--k",
        type=int,
        default=5,
        dest="n_clusters",
        help="Fixed number of clusters (default: 5)",
    )
    parser.add_argument(
        "--silhouette-k-max",
        type=int,
        default=10,
        help="Upper k for silhouette curve at group root (does not select k)",
    )
    parser.add_argument(
        "--all-k-min",
        type=int,
        default=2,
        help="Minimum k when saving all-k results (default: 2)",
    )
    parser.add_argument(
        "--all-k-max",
        type=int,
        default=None,
        help="Maximum k to save under by_k/ (default: --silhouette-k-max)",
    )
    parser.add_argument(
        "--no-save-all-k",
        action="store_true",
        help="Skip writing full inspection artifacts for every k under by_k/",
    )
    parser.add_argument(
        "--auto-select-k",
        action="store_true",
        help="Legacy: pick k by highest silhouette instead of --n-clusters",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=10,
        help="Max k when --auto-select-k is set",
    )
    parser.add_argument(
        "--distance-cutoff",
        type=float,
        default=None,
        help="Fixed distance cutoff for fcluster (overrides --n-clusters)",
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip PCA scatter plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    changepoints_dir = Path(args.changepoints_dir)
    out_dir = Path(args.output_dir) if args.output_dir else changepoints_dir / "clusters"
    out_dir.mkdir(parents=True, exist_ok=True)

    with RunContext.from_namespace(args, name="cluster_changepoint_segments"):
        df = load_segment_stats(changepoints_dir)
        all_k_max = args.all_k_max if args.all_k_max is not None else args.silhouette_k_max
        log_event(
            "info",
            f"Loaded {len(df)} segment rows from {changepoints_dir}",
            component="cluster_changepoint_segments",
        )

        clustered_parts: list[pd.DataFrame] = []
        for group in args.groups:
            with step(f"group={group}"):
                part = cluster_one_group(
                    df,
                    group,
                    out_dir,
                    linkage=args.linkage,
                    n_clusters=args.n_clusters,
                    silhouette_k_max=args.silhouette_k_max,
                    all_k_min=args.all_k_min,
                    all_k_max=all_k_max,
                    save_all_k=not args.no_save_all_k,
                    distance_cutoff=args.distance_cutoff,
                    auto_select_k=args.auto_select_k,
                    k_max=args.k_max,
                    skip_pca=args.skip_pca,
                )
                if not part.empty:
                    clustered_parts.append(part)

        write_cohort_outputs(clustered_parts, out_dir)

    print("\nOutput files:")
    for path in sorted(out_dir.rglob("*")):
        if path.is_file():
            print(f"  {path}")


if __name__ == "__main__":
    main()
