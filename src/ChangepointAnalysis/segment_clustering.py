"""Cluster changepoint-defined segments by feature group."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.utils.cluster_inspection import (
    plot_pca_clusters,
    plot_pca_clusters_by_k,
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
from src.utils.run_log import log_event, step

from .feature_groups import DEFAULT_GROUPS, summary_feature_columns

META_COLS = frozenset({
    "traj_id", "group", "segment_id", "start_frame", "end_frame",
    "start_ps", "end_ps", "n_frames", "n_cols_used",
})


@dataclass
class SegmentClusteringConfig:
    """Parameters for hierarchical clustering of changepoint segments."""

    groups: tuple[str, ...] = DEFAULT_GROUPS
    linkage: str = "ward"
    n_clusters: int = 5
    silhouette_k_max: int = 10
    all_k_min: int = 2
    all_k_max: Optional[int] = None
    save_all_k: bool = True
    distance_cutoff: Optional[float] = None
    auto_select_k: bool = False
    k_max: int = 10
    skip_pca: bool = False
    pca_by_k_only: bool = False


def load_segment_stats(changepoints_dir: Path) -> pd.DataFrame:
    """Load merged all_segment_stats.csv or stack per-trajectory files."""
    changepoints_dir = Path(changepoints_dir)
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


def build_segment_feature_matrix(
    df: pd.DataFrame,
    group: str,
) -> tuple[np.ndarray, list[str]]:
    """Build (n_segments, n_features) matrix: summary mean/std + log10(n_frames)."""
    sub = df[df["group"] == group].copy()
    if sub.empty:
        raise ValueError(f"No segments for group '{group}'")

    candidate_cols = summary_feature_columns(group, df=sub)
    available = [c for c in candidate_cols if c in sub.columns]
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


def build_feature_matrix_from_segments_df(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Rebuild the clustering feature matrix from a segments_clustered.csv slice."""
    if "features_used" in df.columns and df["features_used"].notna().any():
        feature_names = [
            c for c in df["features_used"].iloc[0].split(";") if c
        ]
    else:
        group = str(df["group"].iloc[0]) if "group" in df.columns else "gsa"
        bases = summary_feature_columns(group, df=df)
        keep = [c for c in bases if c in df.columns and df[c].notna().any()]
        feature_names = keep + ["log10_n_frames"]

    cols: list[np.ndarray] = []
    for name in feature_names:
        if name == "log10_n_frames":
            vals = np.log10(
                np.maximum(df["n_frames"].to_numpy(dtype=np.float64), 1.0)
            )
        elif name in df.columns:
            vals = df[name].to_numpy(dtype=np.float64)
        else:
            raise ValueError(f"Missing feature column {name!r} in segments table")
        cols.append(vals)

    mat = np.column_stack(cols) if cols else np.empty((len(df), 0))
    if mat.size > 0:
        for j in range(mat.shape[1]):
            col_vals = mat[:, j]
            finite = col_vals[np.isfinite(col_vals)]
            fill = float(np.median(finite)) if len(finite) else 0.0
            bad = ~np.isfinite(col_vals)
            if bad.any():
                mat[bad, j] = fill

    return mat, feature_names


def segments_from_dataframe(df: pd.DataFrame) -> list[Segment]:
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


def cluster_segment_group(
    df: pd.DataFrame,
    group: str,
    out_dir: Path,
    config: SegmentClusteringConfig,
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

    feature_matrix, feature_names = build_segment_feature_matrix(df, group)
    feat_input = prepare_feature_clustering(feature_matrix)
    clustering = cluster_metastable_states(
        feat_input.distance_matrix,
        scaled_features=feat_input.scaled_features,
        method=config.linkage,
        n_clusters=(
            config.n_clusters
            if config.distance_cutoff is None and not config.auto_select_k
            else None
        ),
        rmsd_cutoff=config.distance_cutoff,
        auto_select_k=config.auto_select_k,
        k_max=config.k_max,
    )

    segments = segments_from_dataframe(sub)
    assign_cluster_labels(segments, clustering)

    base_segments = segments_from_dataframe(sub)

    leaf_labels = [f"{s.traj_id}:seg{s.segment_id}" for s in segments]
    sub["leaf_label"] = leaf_labels
    sub["cluster_label"] = [s.cluster_label for s in segments]
    sub["features_used"] = ";".join(feature_names)

    group_dir = Path(out_dir) / group
    group_dir.mkdir(parents=True, exist_ok=True)

    sub.to_csv(group_dir / "segments_clustered.csv", index=False)

    pd.DataFrame(
        feat_input.distance_matrix, index=leaf_labels, columns=leaf_labels
    ).to_csv(group_dir / "distance_matrix.csv")

    plot_dendrogram(
        clustering.linkage_matrix,
        leaf_labels,
        group_dir / "dendrogram.png",
        title=f"Changepoint segments — {group} (k={clustering.n_clusters})",
        rmsd_cutoff=clustering.cutoff_distance if config.distance_cutoff else None,
    )

    summary_path = group_dir / "cluster_summary.txt"
    summary_path.write_text(
        format_cluster_summary(segments, clustering),
        encoding="utf-8",
    )

    if not config.skip_pca:
        plot_pca_clusters(
            feature_matrix,
            clustering.labels,
            leaf_labels,
            group_dir / "pca_clusters.png",
            title=f"PCA — {group} segments (k={clustering.n_clusters})",
        )

    all_k_max = (
        config.all_k_max if config.all_k_max is not None else config.silhouette_k_max
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
        silhouette_k_max=config.silhouette_k_max,
        group_label=group,
    )

    use_fixed_k = config.distance_cutoff is None and not config.auto_select_k
    if config.save_all_k and use_fixed_k:
        write_all_k_cluster_results(
            group_dir,
            base_segments,
            feat_input.distance_matrix,
            clustering.linkage_matrix,
            k_min=config.all_k_min,
            k_max=all_k_max,
            chosen_k=config.n_clusters,
            linkage_method=clustering.method,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            leaf_labels=leaf_labels,
            group_label=group,
            include_pca_by_k=not config.skip_pca,
        )

    log_event(
        "info",
        f"group={group}: {len(sub)} segments → {clustering.n_clusters} clusters "
        f"({clustering.selection_mode}, silhouette={clustering.silhouette:.3f})",
        component="cluster_changepoint_segments",
    )
    return sub


def write_cohort_cluster_outputs(
    clustered_parts: list[pd.DataFrame],
    out_dir: Path,
) -> None:
    """Write merged cohort-level cluster tables."""
    if not clustered_parts:
        return

    out_dir = Path(out_dir)
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


def cluster_all_groups(
    df: pd.DataFrame,
    out_dir: Path,
    config: SegmentClusteringConfig,
) -> pd.DataFrame:
    """Cluster all configured groups and write cohort outputs.

    Returns the concatenated clustered segments DataFrame.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.pca_by_k_only:
        for group in config.groups:
            group_dir = out_dir / group
            if not (group_dir / "by_k").is_dir():
                log_event(
                    "warning",
                    f"No by_k/ under {group_dir}, skipping",
                    component="cluster_changepoint_segments",
                )
                continue
            seg_csv = group_dir / "segments_clustered.csv"
            if not seg_csv.is_file():
                log_event(
                    "warning",
                    f"Missing {seg_csv}, skipping",
                    component="cluster_changepoint_segments",
                )
                continue
            with step(f"pca-by-k group={group}"):
                seg_df = pd.read_csv(seg_csv)
                feature_matrix, _ = build_feature_matrix_from_segments_df(seg_df)
                written = plot_pca_clusters_by_k(
                    group_dir,
                    feature_matrix,
                    group_label=group,
                )
                log_event(
                    "info",
                    f"group={group}: wrote {len(written)} PCA plots under by_k/",
                    component="cluster_changepoint_segments",
                )
        return pd.DataFrame()

    clustered_parts: list[pd.DataFrame] = []
    for group in config.groups:
        with step(f"group={group}"):
            part = cluster_segment_group(df, group, out_dir, config)
            if not part.empty:
                clustered_parts.append(part)

    write_cohort_cluster_outputs(clustered_parts, out_dir)
    if not clustered_parts:
        return pd.DataFrame()
    return pd.concat(clustered_parts, ignore_index=True)
