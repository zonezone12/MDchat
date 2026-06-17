"""
Inspection helpers for fixed-k metastable-state clustering.

Produces population, lifetime, transition, representative-structure, and
feature-mean summaries plus silhouette curves for manual k selection.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from src.utils.metastable_states import Segment, clustering_result_at_k


def silhouette_scores_by_k(
    distance_matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    *,
    k_min: int = 2,
    k_max: Optional[int] = None,
) -> "pd.DataFrame":
    """Silhouette score for each k (inspection only; does not select k)."""
    import pandas as pd
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics import silhouette_score

    n = distance_matrix.shape[0]
    if n < 2:
        return pd.DataFrame(columns=["k", "silhouette", "n_clusters_effective"])

    k_max_eff = k_max if k_max is not None else min(10, n - 1)
    k_max_eff = max(k_min, k_max_eff)

    rows: list[dict] = []
    for k in range(k_min, k_max_eff + 1):
        labels_k = fcluster(linkage_matrix, t=k, criterion="maxclust") - 1
        n_eff = len(set(labels_k))
        sil = float("nan")
        if n_eff >= 2:
            try:
                sil = float(
                    silhouette_score(distance_matrix, labels_k, metric="precomputed")
                )
            except Exception:
                sil = float("nan")
        rows.append({"k": k, "silhouette": sil, "n_clusters_effective": n_eff})

    return pd.DataFrame(rows)


def plot_silhouette_by_k(
    scores_df: "pd.DataFrame",
    output_path: Union[str, Path],
    *,
    chosen_k: Optional[int] = None,
    title: str = "Silhouette score vs k",
) -> Path:
    """Line plot of silhouette scores across k values."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    valid = scores_df["silhouette"].notna()
    ax.plot(
        scores_df.loc[valid, "k"],
        scores_df.loc[valid, "silhouette"],
        "o-",
        lw=1.5,
        ms=6,
    )
    if chosen_k is not None:
        ax.axvline(chosen_k, color="red", ls="--", lw=1.2, label=f"chosen k={chosen_k}")
        ax.legend(loc="best")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def cluster_population_table(segments: Sequence[Segment]) -> "pd.DataFrame":
    """Segment counts and trajectory coverage per cluster."""
    import pandas as pd

    rows: list[dict] = []
    total = len(segments)
    by_cluster: Dict[int, List[Segment]] = {}
    for seg in segments:
        by_cluster.setdefault(seg.cluster_label, []).append(seg)

    for cl in sorted(by_cluster):
        members = by_cluster[cl]
        trajs = {s.traj_id for s in members}
        rows.append({
            "cluster_label": cl,
            "n_segments": len(members),
            "n_trajectories": len(trajs),
            "fraction_of_segments": len(members) / total if total else 0.0,
        })
    return pd.DataFrame(rows)


def cluster_population_by_trajectory(segments: Sequence[Segment]) -> "pd.DataFrame":
    """Segment counts per trajectory × cluster."""
    import pandas as pd

    counts: Dict[tuple[str, int], int] = {}
    for seg in segments:
        key = (seg.traj_id, seg.cluster_label)
        counts[key] = counts.get(key, 0) + 1

    rows = [
        {"traj_id": t, "cluster_label": c, "n_segments": n}
        for (t, c), n in sorted(counts.items())
    ]
    return pd.DataFrame(rows)


def cluster_lifetime_table(segments: Sequence[Segment]) -> "pd.DataFrame":
    """Residence-time statistics per cluster (frames)."""
    import pandas as pd

    by_cluster: Dict[int, List[int]] = {}
    for seg in segments:
        by_cluster.setdefault(seg.cluster_label, []).append(seg.n_frames)

    total_frames = sum(seg.n_frames for seg in segments)
    rows: list[dict] = []
    for cl in sorted(by_cluster):
        durations = by_cluster[cl]
        total_cl = int(sum(durations))
        rows.append({
            "cluster_label": cl,
            "n_segments": len(durations),
            "total_frames": total_cl,
            "mean_frames": float(np.mean(durations)),
            "median_frames": float(np.median(durations)),
            "min_frames": int(min(durations)),
            "max_frames": int(max(durations)),
            "fraction_of_time": total_cl / total_frames if total_frames else 0.0,
        })
    return pd.DataFrame(rows)


def _segments_by_trajectory(
    segments: Sequence[Segment],
) -> Dict[str, List[Segment]]:
    """Group segments and order chronologically within each trajectory."""
    by_traj: Dict[str, List[Segment]] = {}
    for seg in segments:
        by_traj.setdefault(seg.traj_id, []).append(seg)
    for traj_id in by_traj:
        by_traj[traj_id] = sorted(
            by_traj[traj_id],
            key=lambda s: (s.segment_id, s.start_frame),
        )
    return by_traj


def trajectory_state_paths_table(segments: Sequence[Segment]) -> "pd.DataFrame":
    """
    Ordered metastable-state path per trajectory after clustering.

    Each row is one segment with its cluster label and adjacent states along
    the changepoint-defined path (segment 0 → 1 → 2 → …).
    """
    import pandas as pd

    rows: list[dict] = []
    for traj_id, ordered in _segments_by_trajectory(segments).items():
        states = [int(s.cluster_label) for s in ordered]
        path_str = "→".join(str(s) for s in states)
        for i, seg in enumerate(ordered):
            rows.append({
                "traj_id": traj_id,
                "segment_id": seg.segment_id,
                "start_frame": seg.start_frame,
                "end_frame": seg.end_frame,
                "n_frames": seg.n_frames,
                "cluster_label": int(seg.cluster_label),
                "path_position": i,
                "path_length": len(ordered),
                "prev_cluster": int(ordered[i - 1].cluster_label) if i > 0 else pd.NA,
                "next_cluster": int(ordered[i + 1].cluster_label)
                if i < len(ordered) - 1
                else pd.NA,
                "state_sequence": path_str,
            })
    df = pd.DataFrame(rows)
    for col in ("prev_cluster", "next_cluster"):
        if col in df.columns:
            df[col] = df[col].astype("Int64")
    return df


def trajectory_state_summary_table(segments: Sequence[Segment]) -> "pd.DataFrame":
    """One row per trajectory with its full ordered state-label sequence."""
    import pandas as pd

    rows: list[dict] = []
    for traj_id, ordered in _segments_by_trajectory(segments).items():
        states = [int(s.cluster_label) for s in ordered]
        rows.append({
            "traj_id": traj_id,
            "n_segments": len(states),
            "state_sequence": "→".join(str(s) for s in states),
            "n_transitions": max(0, len(states) - 1),
        })
    return pd.DataFrame(rows)


def cluster_transition_table(segments: Sequence[Segment]) -> "pd.DataFrame":
    """Count consecutive segment transitions between cluster labels per trajectory."""
    import pandas as pd

    counts: Dict[tuple[int, int], int] = {}
    for ordered in _segments_by_trajectory(segments).values():
        for prev, nxt in zip(ordered, ordered[1:]):
            key = (int(prev.cluster_label), int(nxt.cluster_label))
            counts[key] = counts.get(key, 0) + 1

    rows = [
        {"from_cluster": a, "to_cluster": b, "count": n}
        for (a, b), n in sorted(counts.items())
    ]
    return pd.DataFrame(rows)


def transition_count_matrix(
    transition_df: "pd.DataFrame",
    n_clusters: int,
) -> np.ndarray:
    """Square matrix of transition counts (rows=from, cols=to)."""
    mat = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    if transition_df.empty:
        return mat
    for _, row in transition_df.iterrows():
        i, j = int(row["from_cluster"]), int(row["to_cluster"])
        if 0 <= i < n_clusters and 0 <= j < n_clusters:
            mat[i, j] = int(row["count"])
    return mat


def transition_probability_matrix(count_mat: np.ndarray) -> np.ndarray:
    """Row-normalized transition probabilities P(to | from)."""
    prob = count_mat.astype(np.float64)
    row_sums = prob.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(row_sums > 0, prob / row_sums, 0.0)
    return prob


def transition_matrix_dataframe(
    mat: np.ndarray,
    *,
    value_name: str = "count",
) -> "pd.DataFrame":
    """Labeled square transition matrix (index/columns = state labels)."""
    import pandas as pd

    n = mat.shape[0]
    labels = [f"state_{i}" for i in range(n)]
    return pd.DataFrame(mat, index=labels, columns=labels)


def build_state_transition_matrices(
    segments: Sequence[Segment],
    n_clusters: int,
) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    """
    Build count, probability, and long-form transition tables.

    Returns
    -------
    count_df, prob_df, long_df
        Square count matrix, row-normalized probability matrix, and
        from_cluster / to_cluster / count table.
    """
    long_df = cluster_transition_table(segments)
    count_mat = transition_count_matrix(long_df, n_clusters)
    prob_mat = transition_probability_matrix(count_mat)
    count_df = transition_matrix_dataframe(count_mat)
    prob_df = transition_matrix_dataframe(prob_mat, value_name="probability")
    return count_df, prob_df, long_df


def state_transitions_from_dataframe(
    df: "pd.DataFrame",
    *,
    traj_col: str = "traj_id",
    segment_col: str = "segment_id",
    label_col: str = "cluster_label",
    start_frame_col: Optional[str] = "start_frame",
    n_clusters: Optional[int] = None,
) -> tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    """
    Transition matrices from a labeled segment table (changepoint CSV).

    Returns path_table, summary_table, count_matrix, prob_matrix.
    """
    import pandas as pd

    from src.utils.metastable_states import Segment

    segments: list[Segment] = []
    sort_cols = [traj_col, segment_col]
    if start_frame_col and start_frame_col in df.columns:
        sort_cols.append(start_frame_col)
    for _, row in df.sort_values(sort_cols).iterrows():
        start = int(row.get("start_frame", 0)) if "start_frame" in row else 0
        end = int(row.get("end_frame", start + 1)) if "end_frame" in row else start + 1
        segments.append(
            Segment(
                traj_id=str(row[traj_col]),
                segment_id=int(row[segment_col]),
                start_frame=start,
                end_frame=end,
                rep_frame=(start + end) // 2,
                rmsd_mean=float("nan"),
                rg_mean=float("nan"),
                cluster_label=int(row[label_col]),
            )
        )

    if n_clusters is None:
        n_clusters = int(max(s.cluster_label for s in segments)) + 1 if segments else 0

    paths = trajectory_state_paths_table(segments)
    summary = trajectory_state_summary_table(segments)
    count_df, prob_df, _ = build_state_transition_matrices(segments, n_clusters)
    return paths, summary, count_df, prob_df


def plot_transition_heatmap(
    transition_df: "pd.DataFrame",
    n_clusters: int,
    output_path: Union[str, Path],
    *,
    title: str = "Cluster transition network",
    normalize_rows: bool = False,
) -> Path:
    """Heatmap of consecutive-segment transition counts or probabilities."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count_mat = transition_count_matrix(transition_df, n_clusters)
    mat = transition_probability_matrix(count_mat) if normalize_rows else count_mat
    fmt = ".2f" if normalize_rows else "d"
    cbar_label = "P(to|from)" if normalize_rows else "transition count"

    fig, ax = plt.subplots(figsize=(max(5, n_clusters), max(4, n_clusters - 1)))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0)
    labels = [str(i) for i in range(n_clusters)]
    ax.set_xticks(range(n_clusters))
    ax.set_yticks(range(n_clusters))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("To state (cluster_label)")
    ax.set_ylabel("From state (cluster_label)")
    ax.set_title(title)
    for i in range(n_clusters):
        for j in range(n_clusters):
            val = mat[i, j]
            if val > 0:
                text = f"{val:{fmt}}" if normalize_rows else str(int(val))
                ax.text(j, i, text, ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_transition_network_diagram(
    count_mat: np.ndarray,
    output_path: Union[str, Path],
    *,
    title: str = "State transition network",
    min_edge_count: int = 1,
) -> Path:
    """Directed network diagram: nodes = states, arrows = transitions."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = count_mat.shape[0]
    if n == 0:
        return output_path

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 1.0
    positions = {
        i: (radius * np.cos(a), radius * np.sin(a))
        for i, a in enumerate(angles)
    }

    max_count = int(count_mat.max()) if count_mat.size else 1
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")

    for i in range(n):
        for j in range(n):
            count = int(count_mat[i, j])
            if count < min_edge_count:
                continue
            x0, y0 = positions[i]
            x1, y1 = positions[j]
            lw = 0.8 + 3.0 * (count / max(max_count, 1))
            if i == j:
                # Self-loop above the node
                loop = FancyArrowPatch(
                    (x0, y0 + 0.12),
                    (x0 + 0.18, y0 + 0.12),
                    connectionstyle="arc3,rad=1.8",
                    arrowstyle="-|>",
                    mutation_scale=12,
                    lw=lw,
                    color="0.35",
                )
                ax.add_patch(loop)
                ax.text(x0, y0 + 0.35, str(count), ha="center", fontsize=8)
            else:
                dx, dy = x1 - x0, y1 - y0
                dist = np.hypot(dx, dy) or 1.0
                shrink = 0.18
                start = (x0 + shrink * dx / dist, y0 + shrink * dy / dist)
                end = (x1 - shrink * dx / dist, y1 - shrink * dy / dist)
                arrow = FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    lw=lw,
                    color="0.35",
                    connectionstyle="arc3,rad=0.12",
                )
                ax.add_patch(arrow)
                mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
                ax.text(mx, my, str(count), ha="center", fontsize=8, color="0.2")

    for i, (x, y) in positions.items():
        ax.scatter([x], [y], s=900, c="white", edgecolors="black", zorder=3)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=11, zorder=4)

    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def cluster_medoid_indices(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, int]:
    """Index of the medoid segment for each cluster label."""
    medoids: Dict[int, int] = {}
    unique = sorted(set(int(l) for l in labels))
    for cl in unique:
        idx = np.where(labels == cl)[0]
        if len(idx) == 1:
            medoids[cl] = int(idx[0])
            continue
        sub = distance_matrix[np.ix_(idx, idx)]
        medoids[cl] = int(idx[int(np.argmin(sub.sum(axis=1)))])
    return medoids


def cluster_representatives_table(
    segments: Sequence[Segment],
    medoid_indices: Dict[int, int],
    leaf_labels: Optional[Sequence[str]] = None,
) -> "pd.DataFrame":
    """Medoid segment per cluster (structural representative)."""
    import pandas as pd

    rows: list[dict] = []
    for cl in sorted(medoid_indices):
        idx = medoid_indices[cl]
        seg = segments[idx]
        row: dict = {
            "cluster_label": cl,
            "medoid_index": idx,
            "traj_id": seg.traj_id,
            "segment_id": seg.segment_id,
            "rep_frame": seg.rep_frame,
            "start_frame": seg.start_frame,
            "end_frame": seg.end_frame,
            "n_frames": seg.n_frames,
            "rmsd_mean": seg.rmsd_mean,
            "rg_mean": seg.rg_mean,
        }
        if leaf_labels is not None:
            row["leaf_label"] = leaf_labels[idx]
        rows.append(row)
    return pd.DataFrame(rows)


def cluster_feature_means_table(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
) -> "pd.DataFrame":
    """Per-cluster mean of raw (unscaled) feature vectors."""
    import pandas as pd

    rows: list[dict] = []
    for cl in sorted(set(int(l) for l in labels)):
        mask = labels == cl
        means = np.nanmean(feature_matrix[mask], axis=0)
        row: dict = {"cluster_label": cl, "n_segments": int(mask.sum())}
        for name, val in zip(feature_names, means):
            row[name] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)


def write_cluster_inspection(
    output_dir: Union[str, Path],
    segments: Sequence[Segment],
    labels: np.ndarray,
    distance_matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    *,
    chosen_k: int,
    feature_matrix: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    leaf_labels: Optional[Sequence[str]] = None,
    silhouette_k_max: int = 10,
    group_label: str = "",
    include_silhouette_curve: bool = True,
) -> Dict[str, Path]:
    """
    Write inspection tables and plots for a fixed-k clustering result.

    Returns paths of written artifacts keyed by artifact name.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    title_suffix = f" — {group_label}" if group_label else ""
    n_cl = int(max(labels)) + 1 if len(labels) else chosen_k
    written: Dict[str, Path] = {}

    pop = cluster_population_table(segments)
    p_pop = output_dir / "cluster_population.csv"
    pop.to_csv(p_pop, index=False)
    written["population"] = p_pop

    pop_traj = cluster_population_by_trajectory(segments)
    p_pop_traj = output_dir / "cluster_population_by_trajectory.csv"
    pop_traj.to_csv(p_pop_traj, index=False)
    written["population_by_trajectory"] = p_pop_traj

    lifetimes = cluster_lifetime_table(segments)
    p_life = output_dir / "cluster_lifetimes.csv"
    lifetimes.to_csv(p_life, index=False)
    written["lifetimes"] = p_life

    paths = trajectory_state_paths_table(segments)
    p_paths = output_dir / "trajectory_state_paths.csv"
    paths.to_csv(p_paths, index=False)
    written["trajectory_paths"] = p_paths

    path_summary = trajectory_state_summary_table(segments)
    p_path_sum = output_dir / "trajectory_state_summary.csv"
    path_summary.to_csv(p_path_sum, index=False)
    written["trajectory_summary"] = p_path_sum

    count_df, prob_df, transitions = build_state_transition_matrices(
        segments, n_cl
    )
    p_trans = output_dir / "cluster_transitions.csv"
    transitions.to_csv(p_trans, index=False)
    written["transitions"] = p_trans

    p_count = output_dir / "state_transition_matrix.csv"
    count_df.to_csv(p_count)
    written["transition_matrix"] = p_count

    p_prob = output_dir / "state_transition_prob.csv"
    prob_df.to_csv(p_prob)
    written["transition_prob"] = p_prob

    count_mat = count_df.to_numpy()
    p_heat = plot_transition_heatmap(
        transitions,
        n_cl,
        output_dir / "transition_matrix.png",
        title=f"State transition matrix{title_suffix} (counts)",
    )
    written["transition_heatmap"] = p_heat

    p_heat_prob = plot_transition_heatmap(
        transitions,
        n_cl,
        output_dir / "transition_matrix_prob.png",
        title=f"State transition matrix{title_suffix} (P(to|from))",
        normalize_rows=True,
    )
    written["transition_heatmap_prob"] = p_heat_prob

    p_net = plot_transition_network_diagram(
        count_mat,
        output_dir / "transition_network.png",
        title=f"State transition network{title_suffix} (k={chosen_k})",
    )
    written["transition_network"] = p_net

    medoids = cluster_medoid_indices(distance_matrix, labels)
    reps = cluster_representatives_table(segments, medoids, leaf_labels)
    p_reps = output_dir / "cluster_representatives.csv"
    reps.to_csv(p_reps, index=False)
    written["representatives"] = p_reps

    if feature_matrix is not None and feature_names is not None:
        feat_means = cluster_feature_means_table(
            feature_matrix, labels, feature_names
        )
        p_feat = output_dir / "cluster_feature_means.csv"
        feat_means.to_csv(p_feat, index=False)
        written["feature_means"] = p_feat

    if include_silhouette_curve:
        sil_df = silhouette_scores_by_k(
            distance_matrix,
            linkage_matrix,
            k_max=silhouette_k_max,
        )
        p_sil_csv = output_dir / "silhouette_by_k.csv"
        sil_df.to_csv(p_sil_csv, index=False)
        written["silhouette_csv"] = p_sil_csv

        p_sil_plot = plot_silhouette_by_k(
            sil_df,
            output_dir / "silhouette_by_k.png",
            chosen_k=chosen_k,
            title=f"Silhouette vs k{title_suffix}",
        )
        written["silhouette_plot"] = p_sil_plot

    return written


def _segments_with_labels(
    base_segments: Sequence[Segment],
    labels: np.ndarray,
) -> List[Segment]:
    """Copy segments and assign cluster labels."""
    segs = [copy.copy(s) for s in base_segments]
    for seg, lab in zip(segs, labels):
        seg.cluster_label = int(lab)
    return segs


def write_all_k_cluster_results(
    output_dir: Union[str, Path],
    base_segments: Sequence[Segment],
    distance_matrix: np.ndarray,
    linkage_matrix: np.ndarray,
    *,
    k_min: int = 2,
    k_max: int = 10,
    chosen_k: Optional[int] = None,
    linkage_method: str = "ward",
    feature_matrix: Optional[np.ndarray] = None,
    feature_names: Optional[Sequence[str]] = None,
    leaf_labels: Optional[Sequence[str]] = None,
    group_label: str = "",
) -> Dict[str, Path]:
    """
    Save full inspection artifacts for every k in ``k_min`` … ``k_max``.

    Results are written under ``by_k/k_XX/``. A summary table
    ``by_k/k_comparison.csv`` compares silhouette and population across k.
    """
    import pandas as pd

    output_dir = Path(output_dir)
    by_k_dir = output_dir / "by_k"
    by_k_dir.mkdir(parents=True, exist_ok=True)

    n = distance_matrix.shape[0]
    k_max_eff = max(k_min, min(k_max, n - 1)) if n > 1 else k_min
    written: Dict[str, Path] = {}
    comparison_rows: list[dict] = []

    for k in range(k_min, k_max_eff + 1):
        result = clustering_result_at_k(
            distance_matrix,
            linkage_matrix,
            k,
            method=linkage_method,
        )
        segs_k = _segments_with_labels(base_segments, result.labels)
        pop = cluster_population_table(segs_k)
        min_pop = int(pop["n_segments"].min()) if not pop.empty else 0
        max_pop = int(pop["n_segments"].max()) if not pop.empty else 0

        comparison_rows.append({
            "k": k,
            "n_clusters_effective": result.n_clusters,
            "silhouette": result.silhouette,
            "linkage_merge_height": result.cutoff_distance,
            "min_cluster_size": min_pop,
            "max_cluster_size": max_pop,
            "is_chosen_k": bool(chosen_k is not None and k == chosen_k),
        })

        k_dir = by_k_dir / f"k_{k:02d}"
        k_written = write_cluster_inspection(
            k_dir,
            segs_k,
            result.labels,
            distance_matrix,
            linkage_matrix,
            chosen_k=k,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            leaf_labels=leaf_labels,
            silhouette_k_max=k_max,
            group_label=group_label,
            include_silhouette_curve=False,
        )
        written.update({f"k_{k:02d}/{name}": path for name, path in k_written.items()})

        if leaf_labels is not None:
            seg_rows = []
            for seg, leaf in zip(segs_k, leaf_labels):
                seg_rows.append({
                    "traj_id": seg.traj_id,
                    "segment_id": seg.segment_id,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "n_frames": seg.n_frames,
                    "cluster_label": seg.cluster_label,
                    "leaf_label": leaf,
                })
            seg_path = k_dir / "segments_clustered.csv"
            pd.DataFrame(seg_rows).to_csv(seg_path, index=False)
            written[f"k_{k:02d}/segments_clustered"] = seg_path

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = by_k_dir / "k_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    written["k_comparison"] = comp_path

    if comparison_rows:
        plot_silhouette_by_k(
            comp_df,
            by_k_dir / "silhouette_by_k.png",
            chosen_k=chosen_k,
            title=f"Silhouette vs k — {group_label}" if group_label else "Silhouette vs k",
        )
        written["silhouette_plot"] = by_k_dir / "silhouette_by_k.png"
        comp_df[["k", "silhouette", "n_clusters_effective"]].to_csv(
            by_k_dir / "silhouette_by_k.csv", index=False
        )

    return written
