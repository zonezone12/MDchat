"""
Multi-trajectory metastable-state identification.

Segments each trajectory with ruptures changepoint detection, extracts
representative structures, clusters them hierarchically by pairwise RMSD,
and assigns structural-type labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.TrajectoryIterator import TrajectoryIterator
from src.TrajectoryMetrics import MetricPassSpec, StackedMetricsObserver
from src.TrajectoryMetrics.TrajectoryMetrics import rmsd_value_aligned
from src.utils.gsa_feature_observer import compute_gsa_features
from src.utils.gsa_selections import GSAFeatureSelections
from src.utils.ruptures_utils import ChangePointResult, detect_changepoints

# Columns used for multivariate changepoint detection on GSA nanocube trajectories.
DEFAULT_GSA_SIGNAL_COLUMNS: Tuple[str, ...] = (
    "assembly_rmsd_to_ref",
    "assembly_rg",
    "octahedrality_score",
    "total_inter_monomer_contacts",
    "cavity_water_count",
    "hydrophilic_minus_hydrophobic_radial_distance",
)

_METADATA_COLUMNS = frozenset({"traj_id", "frame", "time_ps"})


@dataclass
class Segment:
    """One metastable segment within a single trajectory."""

    traj_id: str
    segment_id: int
    start_frame: int
    end_frame: int
    rep_frame: int
    rmsd_mean: float
    rg_mean: float
    signal_start: int = 0
    signal_end: int = 0
    cluster_label: int = -1
    feature_means: Dict[str, float] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass
class ClusteringResult:
    """Output of hierarchical clustering on representative structures."""

    linkage_matrix: np.ndarray
    labels: np.ndarray
    n_clusters: int
    cutoff_distance: float
    silhouette: float
    method: str


def _effective_n_frames(n_traj_frames: int, stride: int) -> int:
    return len(range(0, n_traj_frames, stride))


def _signal_index_to_traj_frame(signal_idx: int, stride: int) -> int:
    return int(signal_idx * stride)


def compute_metrics_single_pass(
    universe: Any,
    selection: str,
    *,
    stride: int = 1,
    ref_frame: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute RMSD and Rg in one iterator pass.

    Returns
    -------
    rmsd, rg, frame_indices
        Arrays aligned to subsampled trajectory frames.
    """
    n_frames = _effective_n_frames(universe.trajectory.n_frames, stride)
    specs = [
        MetricPassSpec(
            result_key="rmsd",
            kind="rmsd",
            selection=selection,
            ref_frame=ref_frame,
        ),
        MetricPassSpec(result_key="rg", kind="rg", selection=selection),
    ]
    observer = StackedMetricsObserver(specs, n_frames=n_frames)
    iterator = TrajectoryIterator(universe)
    iterator.subscribe(observer)
    iterator.iterate(step=stride)

    rmsd = np.asarray(observer.results["rmsd"], dtype=np.float64)
    rg = np.asarray(observer.results["rg"], dtype=np.float64)
    frame_indices = np.arange(n_frames, dtype=np.int64) * stride
    return rmsd, rg, frame_indices


def compute_gsa_features_pass(
    universe: Any,
    *,
    selections: Optional[GSAFeatureSelections] = None,
    gsa_resname: str = "MOL",
    n_monomers: int = 6,
    stride: int = 1,
    ref_frame: int = 0,
    include_tier2: bool = True,
    include_guest: bool = True,
    auto_tooth: bool = True,
    traj_id: str = "",
) -> "Any":
    """
    Run one GSA feature extraction pass over a trajectory.

    Returns a per-frame DataFrame aligned to subsampled frames.
    """
    import pandas as pd

    df = compute_gsa_features(
        universe,
        selections=selections,
        gsa_resname=gsa_resname,
        n_monomers=n_monomers,
        include_tier1=True,
        include_tier2=include_tier2,
        include_guest=include_guest,
        auto_tooth=auto_tooth,
        ref_frame=ref_frame,
        traj_id=traj_id,
        step=stride,
        n_jobs=1,
    )
    if not isinstance(df, pd.DataFrame):
        raise TypeError("compute_gsa_features must return a DataFrame")
    return df


def _gsa_signal_matrix(
    features_df: Any,
    signal_columns: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Build a (n_frames, n_features) signal array for ruptures."""
    cols = list(signal_columns) if signal_columns else list(DEFAULT_GSA_SIGNAL_COLUMNS)
    available = [c for c in cols if c in features_df.columns]
    if not available:
        raise ValueError(
            f"No signal columns found in GSA features. Requested: {cols}, "
            f"available sample: {list(features_df.columns[:10])}"
        )
    mat = features_df[available].to_numpy(dtype=np.float64)
    # Replace inf with NaN, then column-wise nanmean fill for ruptures stability
    mat = np.where(np.isfinite(mat), mat, np.nan)
    col_means = np.nanmean(mat, axis=0)
    inds = np.where(np.isnan(mat))
    mat[inds] = np.take(col_means, inds[1])
    return mat, available


def _segment_feature_means(
    features_df: Any,
    s_start: int,
    s_end: int,
) -> Dict[str, float]:
    """Mean of numeric GSA columns over a signal-index slice."""
    import pandas as pd

    numeric_cols = [
        c for c in features_df.columns
        if c not in _METADATA_COLUMNS
        and pd.api.types.is_numeric_dtype(features_df[c])
    ]
    slice_df = features_df.iloc[s_start:s_end]
    means: Dict[str, float] = {}
    for col in numeric_cols:
        val = float(np.nanmean(slice_df[col].to_numpy(dtype=np.float64)))
        if np.isfinite(val):
            means[col] = val
    return means


def segment_trajectory_gsa(
    universe: Any,
    traj_id: str,
    *,
    selections: Optional[GSAFeatureSelections] = None,
    gsa_resname: str = "MOL",
    n_monomers: int = 6,
    stride: int = 1,
    ref_frame: int = 0,
    signal_columns: Optional[Sequence[str]] = None,
    include_tier2: bool = True,
    include_guest: bool = True,
    auto_tooth: bool = True,
    method: str = "Pelt",
    cost_model: str = "rbf",
    penalty: Optional[float] = None,
    n_bkps: Optional[int] = None,
    min_segment_frames: int = 10,
    jump: int = 5,
    window_width: int = 100,
) -> Tuple[List[Segment], Any]:
    """
    Segment a trajectory using GSA feature time series (multivariate ruptures).

    Returns segments and the full per-frame features DataFrame.
    """
    features_df = compute_gsa_features_pass(
        universe,
        selections=selections,
        gsa_resname=gsa_resname,
        n_monomers=n_monomers,
        stride=stride,
        ref_frame=ref_frame,
        include_tier2=include_tier2,
        include_guest=include_guest,
        auto_tooth=auto_tooth,
        traj_id=traj_id,
    )
    n_frames = len(features_df)
    if n_frames == 0:
        return [], features_df

    signal, _used_cols = _gsa_signal_matrix(features_df, signal_columns)
    cp_result: ChangePointResult = detect_changepoints(
        signal,
        method=method,
        cost_model=cost_model,
        penalty=penalty,
        n_bkps=n_bkps,
        jump=jump,
        window_width=window_width,
    )

    frame_col = features_df["frame"].to_numpy(dtype=np.int64) if "frame" in features_df.columns else np.arange(n_frames) * stride

    segments: List[Segment] = []
    for seg_id, (s_start, s_end) in enumerate(cp_result.segment_ranges):
        if s_end <= s_start:
            continue

        traj_start = int(frame_col[s_start])
        traj_end = int(frame_col[min(s_end - 1, n_frames - 1)]) + stride
        if traj_end <= traj_start:
            continue
        if (traj_end - traj_start) < min_segment_frames:
            continue

        rep_signal = (s_start + s_end) // 2
        rep_frame = int(frame_col[min(rep_signal, n_frames - 1)])

        feat_means = _segment_feature_means(features_df, s_start, s_end)
        rmsd_mean = feat_means.get("assembly_rmsd_to_ref", float("nan"))
        rg_mean = feat_means.get("assembly_rg", float("nan"))

        segments.append(
            Segment(
                traj_id=traj_id,
                segment_id=seg_id,
                start_frame=traj_start,
                end_frame=traj_end,
                rep_frame=rep_frame,
                rmsd_mean=rmsd_mean,
                rg_mean=rg_mean,
                signal_start=s_start,
                signal_end=s_end,
                feature_means=feat_means,
            )
        )

    if not segments and n_frames >= min_segment_frames:
        feat_means = _segment_feature_means(features_df, 0, n_frames)
        segments.append(
            Segment(
                traj_id=traj_id,
                segment_id=0,
                start_frame=int(frame_col[0]),
                end_frame=int(frame_col[-1]) + stride,
                rep_frame=int(frame_col[n_frames // 2]),
                rmsd_mean=feat_means.get("assembly_rmsd_to_ref", float("nan")),
                rg_mean=feat_means.get("assembly_rg", float("nan")),
                signal_start=0,
                signal_end=n_frames,
                feature_means=feat_means,
            )
        )

    return segments, features_df


def segment_trajectory(
    universe: Any,
    selection: str,
    traj_id: str,
    *,
    stride: int = 1,
    ref_frame: int = 0,
    signal_metric: str = "rmsd",
    method: str = "Pelt",
    cost_model: str = "rbf",
    penalty: Optional[float] = None,
    n_bkps: Optional[int] = None,
    min_segment_frames: int = 10,
    jump: int = 5,
    window_width: int = 100,
) -> List[Segment]:
    """
    Segment one trajectory into metastable regions via ruptures.

    Parameters
    ----------
    signal_metric
        ``'rmsd'`` or ``'rg'`` — time series used for changepoint detection.
    min_segment_frames
        Minimum segment length in *trajectory* frames (after stride mapping).
    """
    rmsd, rg, frame_indices = compute_metrics_single_pass(
        universe,
        selection,
        stride=stride,
        ref_frame=ref_frame,
    )

    if signal_metric.lower() == "rg":
        signal = rg
    else:
        signal = rmsd

    cp_result: ChangePointResult = detect_changepoints(
        signal,
        method=method,
        cost_model=cost_model,
        penalty=penalty,
        n_bkps=n_bkps,
        jump=jump,
        window_width=window_width,
    )

    segments: List[Segment] = []
    for seg_id, (s_start, s_end) in enumerate(cp_result.segment_ranges):
        if s_end <= s_start:
            continue

        traj_start = _signal_index_to_traj_frame(s_start, stride)
        traj_end = _signal_index_to_traj_frame(s_end, stride)
        if traj_end <= traj_start:
            continue
        if (traj_end - traj_start) < min_segment_frames:
            continue

        rep_signal = (s_start + s_end) // 2
        rep_frame = _signal_index_to_traj_frame(rep_signal, stride)
        rep_frame = min(rep_frame, int(frame_indices[-1]))

        rmsd_slice = rmsd[s_start:s_end]
        rg_slice = rg[s_start:s_end]

        segments.append(
            Segment(
                traj_id=traj_id,
                segment_id=seg_id,
                start_frame=traj_start,
                end_frame=traj_end,
                rep_frame=rep_frame,
                rmsd_mean=float(np.nanmean(rmsd_slice)),
                rg_mean=float(np.nanmean(rg_slice)),
                signal_start=s_start,
                signal_end=s_end,
            )
        )

    if not segments and len(signal) >= min_segment_frames:
        segments.append(
            Segment(
                traj_id=traj_id,
                segment_id=0,
                start_frame=int(frame_indices[0]),
                end_frame=int(frame_indices[-1]) + stride,
                rep_frame=int(frame_indices[len(frame_indices) // 2]),
                rmsd_mean=float(np.nanmean(rmsd)),
                rg_mean=float(np.nanmean(rg)),
                signal_start=0,
                signal_end=len(signal),
            )
        )

    return segments


def extract_representative_positions(
    universe: Any,
    selection: str,
    segments: Sequence[Segment],
) -> Tuple[np.ndarray, List[Segment]]:
    """
    Extract target-molecule coordinates at each segment's representative frame.

    Returns
    -------
    positions : ndarray, shape (n_segments, n_atoms, 3)
    segments : same list (order preserved)
    """
    import MDAnalysis as mda

    if not isinstance(universe, mda.Universe):
        raise TypeError("universe must be an MDAnalysis Universe")

    sel = universe.select_atoms(selection)
    if len(sel) == 0:
        raise ValueError(f"Empty selection: {selection!r}")

    n_atoms = len(sel)
    positions = np.zeros((len(segments), n_atoms, 3), dtype=np.float64)

    for i, seg in enumerate(segments):
        universe.trajectory[seg.rep_frame]
        positions[i] = sel.positions.copy()

    return positions, list(segments)


def compute_pairwise_rmsd_matrix(positions: np.ndarray) -> np.ndarray:
    """
    Symmetric pairwise minimum-RMSD matrix after optimal superposition.

    Parameters
    ----------
    positions : ndarray, shape (n_structures, n_atoms, 3)

    Returns
    -------
    dist : ndarray, shape (n_structures, n_structures)
    """
    n = positions.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dist[i, i] = 0.0
        for j in range(i + 1, n):
            r = rmsd_value_aligned(positions[i], positions[j])
            dist[i, j] = r
            dist[j, i] = r
    return dist


def cluster_metastable_states(
    distance_matrix: np.ndarray,
    *,
    method: str = "ward",
    k_min: int = 2,
    k_max: Optional[int] = None,
    rmsd_cutoff: Optional[float] = None,
) -> ClusteringResult:
    """
    Hierarchical clustering on a precomputed distance matrix.

    If *rmsd_cutoff* is given, ``fcluster`` uses that distance threshold.
    Otherwise the number of clusters is chosen by best silhouette score
    over ``k_min`` … ``k_max``.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    n = distance_matrix.shape[0]
    if n < 2:
        labels = np.zeros(n, dtype=int)
        return ClusteringResult(
            linkage_matrix=np.zeros((0, 4)),
            labels=labels,
            n_clusters=1 if n == 1 else 0,
            cutoff_distance=0.0,
            silhouette=-1.0,
            method=method,
        )

    condensed = squareform(distance_matrix, checks=False)
    Z = linkage(condensed, method=method)

    if rmsd_cutoff is not None:
        labels = fcluster(Z, t=rmsd_cutoff, criterion="distance")
        labels = labels - 1
        n_clusters = len(set(labels))
        sil = -1.0
        if n_clusters >= 2:
            try:
                sil = float(silhouette_score(distance_matrix, labels, metric="precomputed"))
            except Exception:
                sil = -1.0
        return ClusteringResult(
            linkage_matrix=Z,
            labels=labels.astype(int),
            n_clusters=n_clusters,
            cutoff_distance=float(rmsd_cutoff),
            silhouette=sil,
            method=method,
        )

    k_max_eff = k_max if k_max is not None else min(10, n - 1)
    k_max_eff = max(k_min, k_max_eff)

    best_k, best_labels, best_sil = k_min, None, -1.0
    for k in range(k_min, k_max_eff + 1):
        labels_k = fcluster(Z, t=k, criterion="maxclust") - 1
        if len(set(labels_k)) < 2:
            continue
        try:
            sil = silhouette_score(distance_matrix, labels_k, metric="precomputed")
        except Exception:
            continue
        if sil > best_sil:
            best_sil, best_k, best_labels = sil, k, labels_k

    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
        best_k = 1
        best_sil = -1.0

    cutoff = float(Z[-(best_k - 1), 2]) if best_k > 1 and Z.shape[0] >= best_k - 1 else 0.0

    return ClusteringResult(
        linkage_matrix=Z,
        labels=best_labels.astype(int),
        n_clusters=best_k,
        cutoff_distance=cutoff,
        silhouette=best_sil,
        method=method,
    )


def plot_dendrogram(
    linkage_matrix: np.ndarray,
    leaf_labels: Sequence[str],
    output_path: Union[str, Path],
    *,
    title: str = "Metastable state hierarchical clustering",
    rmsd_cutoff: Optional[float] = None,
) -> Path:
    """Save a dendrogram PNG for visual cutoff selection."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(max(10, len(leaf_labels) * 0.35), 6))
    dendrogram(
        linkage_matrix,
        labels=list(leaf_labels),
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
    )
    if rmsd_cutoff is not None:
        ax.axhline(y=rmsd_cutoff, color="red", ls="--", lw=1.2, label=f"cutoff={rmsd_cutoff:.2f} Å")
        ax.legend(loc="upper right")
    ax.set_title(title)
    ax.set_ylabel("RMSD distance")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def assign_cluster_labels(
    segments: List[Segment],
    clustering: ClusteringResult,
) -> List[Segment]:
    """Write cluster labels onto segments (in place)."""
    if len(segments) != len(clustering.labels):
        raise ValueError(
            f"Segment count ({len(segments)}) != label count ({len(clustering.labels)})"
        )
    for seg, lab in zip(segments, clustering.labels):
        seg.cluster_label = int(lab)
    return segments


def build_feature_matrix(segments: Sequence[Segment]) -> np.ndarray:
    """Feature vector per segment: GSA means when available, else [rmsd, rg, log_duration]."""
    if segments and segments[0].feature_means:
        keys = sorted(segments[0].feature_means.keys())
        rows = []
        for s in segments:
            row = [s.feature_means.get(k, np.nan) for k in keys]
            dur = max(s.n_frames, 1)
            row.append(np.log10(float(dur)))
            rows.append(row)
        return np.asarray(rows, dtype=np.float64)

    rows = []
    for s in segments:
        dur = max(s.n_frames, 1)
        rows.append([s.rmsd_mean, s.rg_mean, np.log10(float(dur))])
    return np.asarray(rows, dtype=np.float64)


def compute_pairwise_feature_distance_matrix(
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """Pairwise Euclidean distance on standardized segment feature vectors."""
    from sklearn.preprocessing import StandardScaler

    n = feature_matrix.shape[0]
    if n < 2:
        return np.zeros((n, n), dtype=np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)
    # Replace any remaining NaN with 0 after scaling
    X = np.nan_to_num(X, nan=0.0)

    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(X[i] - X[j]))
            dist[i, j] = d
            dist[j, i] = d
    return dist


def format_cluster_summary(segments: Sequence[Segment], clustering: ClusteringResult) -> str:
    """Human-readable summary of structural types."""
    lines = [
        f"Structural types identified: {clustering.n_clusters}",
        f"Clustering method: {clustering.method}",
        f"Auto silhouette score: {clustering.silhouette:.4f}",
        f"Cutoff distance: {clustering.cutoff_distance:.4f} Å",
        "",
        "Per-cluster membership:",
    ]
    by_cluster: Dict[int, List[str]] = {}
    for s in segments:
        by_cluster.setdefault(s.cluster_label, []).append(
            f"{s.traj_id}:seg{s.segment_id} (frames {s.start_frame}-{s.end_frame}, rep={s.rep_frame})"
        )
    for cl in sorted(by_cluster):
        members = by_cluster[cl]
        lines.append(f"  Type {cl}: {len(members)} segment(s)")
        for m in members:
            lines.append(f"    - {m}")
    return "\n".join(lines)


def process_trajectory_files(
    topology: Union[str, Path],
    trajectory_paths: Sequence[Union[str, Path]],
    selection: str,
    *,
    stride: int = 1,
    ref_frame: int = 0,
    signal_metric: str = "rmsd",
    ruptures_method: str = "Pelt",
    ruptures_cost: str = "rbf",
    penalty: Optional[float] = None,
    n_bkps: Optional[int] = None,
    min_segment_frames: int = 10,
    linkage_method: str = "ward",
    rmsd_cutoff: Optional[float] = None,
    k_max: Optional[int] = None,
    use_gsa_features: bool = False,
    gsa_selections: Optional[GSAFeatureSelections] = None,
    gsa_resname: str = "MOL",
    n_monomers: int = 6,
    gsa_signal_columns: Optional[Sequence[str]] = None,
    include_tier2: bool = True,
    include_guest: bool = True,
    auto_tooth: bool = True,
    cluster_mode: str = "rmsd",
) -> Dict[str, Any]:
    """
    Full pipeline: segment all trajectories, cluster representatives, return results.

    When *use_gsa_features* is True, segmentation uses multivariate GSA feature
    time series and *cluster_mode* can be ``'features'`` (segment-mean GSA
    vectors) or ``'rmsd'`` (pairwise RMSD on representative structures).

    Returns dict with keys: segments, positions, distance_matrix, clustering,
    leaf_labels, and optionally features_dfs.
    """
    import MDAnalysis as mda

    all_segments: List[Segment] = []
    all_positions: List[np.ndarray] = []
    all_features_dfs: List[Any] = []

    top = str(topology)
    for traj_path in trajectory_paths:
        traj_path = Path(traj_path)
        traj_id = traj_path.stem
        u = mda.Universe(top, str(traj_path))

        if use_gsa_features:
            segs, feat_df = segment_trajectory_gsa(
                u,
                traj_id,
                selections=gsa_selections,
                gsa_resname=gsa_resname,
                n_monomers=n_monomers,
                stride=stride,
                ref_frame=ref_frame,
                signal_columns=gsa_signal_columns,
                include_tier2=include_tier2,
                include_guest=include_guest,
                auto_tooth=auto_tooth,
                method=ruptures_method,
                cost_model=ruptures_cost,
                penalty=penalty,
                n_bkps=n_bkps,
                min_segment_frames=min_segment_frames,
            )
            all_features_dfs.append(feat_df)
        else:
            segs = segment_trajectory(
                u,
                selection,
                traj_id,
                stride=stride,
                ref_frame=ref_frame,
                signal_metric=signal_metric,
                method=ruptures_method,
                cost_model=ruptures_cost,
                penalty=penalty,
                n_bkps=n_bkps,
                min_segment_frames=min_segment_frames,
            )

        if not segs:
            continue

        assembly_sel = (
            gsa_selections.assembly_sel
            if gsa_selections and gsa_selections.assembly_sel
            else f"resname {gsa_resname}"
        ) if use_gsa_features else selection

        pos, segs = extract_representative_positions(u, assembly_sel, segs)
        all_segments.extend(segs)
        all_positions.append(pos)

    if not all_segments:
        raise RuntimeError("No segments extracted from any trajectory.")

    positions = np.vstack(all_positions)

    if use_gsa_features and cluster_mode == "features":
        feat_mat = build_feature_matrix(all_segments)
        dist_mat = compute_pairwise_feature_distance_matrix(feat_mat)
    else:
        dist_mat = compute_pairwise_rmsd_matrix(positions)

    clustering = cluster_metastable_states(
        dist_mat,
        method=linkage_method,
        rmsd_cutoff=rmsd_cutoff,
        k_max=k_max,
    )
    assign_cluster_labels(all_segments, clustering)

    leaf_labels = [
        f"{s.traj_id}:seg{s.segment_id}" for s in all_segments
    ]

    result: Dict[str, Any] = {
        "segments": all_segments,
        "positions": positions,
        "distance_matrix": dist_mat,
        "clustering": clustering,
        "leaf_labels": leaf_labels,
        "cluster_mode": cluster_mode if use_gsa_features else "rmsd",
    }
    if all_features_dfs:
        result["features_dfs"] = all_features_dfs
    return result
