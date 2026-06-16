"""
Validate feature-based changepoint clusters with structural RMSD.

For each feature group (gsa, iodine, na_water, combined), extracts GSA assembly
coordinates at each segment's representative frame, computes pairwise aligned
RMSD, and tests whether feature cluster labels correspond to structurally
distinct metastable states.

Designed for HPC: point --topology and --trajectory-dir at cluster paths, run
after uploading changepoint cluster CSVs from local analysis.

Example (on HPC)
----------------
python scripts/validate_changepoint_cluster_structures.py \\
    --clusters-dir output/changepoints/clusters \\
    --topology /path/on/hpc/traj/BMMpM_ca.prmtop \\
    --trajectory-dir /path/on/hpc/traj \\
    --selection "resname MOL" \\
    --groups gsa iodine na_water combined \\
    --output-dir output/changepoints/structure_validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.utils.metastable_states import (
    Segment,
    compute_pairwise_rmsd_matrix,
    extract_representative_positions,
)
from src.utils.run_log import RunContext, log_event, step

DEFAULT_GROUPS = ("gsa", "iodine", "na_water", "combined")
_TRAJ_EXTENSIONS = (".trj", ".xtc", ".dcd", ".nc", ".crd")


def _resolve_trajectory(trajectory_dir: Path, traj_id: str) -> Path:
    """Map traj_id to an on-disk trajectory file."""
    for ext in _TRAJ_EXTENSIONS:
        candidate = trajectory_dir / f"{traj_id}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No trajectory for {traj_id!r} under {trajectory_dir} "
        f"(tried {', '.join(_TRAJ_EXTENSIONS)})"
    )


def _segments_from_df(df: pd.DataFrame) -> list[Segment]:
    segments: list[Segment] = []
    for _, row in df.iterrows():
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        rep = int((start + end) // 2)
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


def extract_positions_multi_trajectory(
    segments: list[Segment],
    *,
    topology: Path,
    trajectory_dir: Path,
    selection: str,
) -> np.ndarray:
    """Extract rep-frame coordinates; one universe load per unique traj_id."""
    import MDAnalysis as mda

    if not segments:
        raise ValueError("No segments to extract")

    by_traj: dict[str, list[tuple[int, Segment]]] = {}
    for idx, seg in enumerate(segments):
        by_traj.setdefault(seg.traj_id, []).append((idx, seg))

    n_atoms: Optional[int] = None
    positions = np.empty((len(segments), 0, 3), dtype=np.float64)

    for traj_id, indexed_segs in by_traj.items():
        traj_path = _resolve_trajectory(trajectory_dir, traj_id)
        u = mda.Universe(str(topology), str(traj_path))
        seg_list = [s for _, s in indexed_segs]
        pos, _ = extract_representative_positions(u, selection, seg_list)

        if n_atoms is None:
            n_atoms = pos.shape[1]
            positions = np.zeros((len(segments), n_atoms, 3), dtype=np.float64)
        elif pos.shape[1] != n_atoms:
            raise ValueError(
                f"Atom count mismatch for {traj_id}: {pos.shape[1]} vs {n_atoms}"
            )

        for (global_idx, _), coords in zip(indexed_segs, pos):
            positions[global_idx] = coords

        log_event(
            "info",
            f"Extracted {len(seg_list)} segment(s) from {traj_path.name}",
            component="validate_changepoint_cluster_structures",
        )

    return positions


def _cluster_labels_fixed_k(
    dist_mat: np.ndarray,
    k: int,
    *,
    method: str = "ward",
) -> np.ndarray:
    """Hierarchical clustering with a fixed number of clusters."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = dist_mat.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    k_eff = max(1, min(k, n))
    Z = linkage(squareform(dist_mat, checks=False), method=method)
    return (fcluster(Z, t=k_eff, criterion="maxclust") - 1).astype(int)


def _pairwise_rmsd_table(
    dist_mat: np.ndarray,
    labels: np.ndarray,
    leaf_labels: list[str],
) -> pd.DataFrame:
    rows: list[dict] = []
    n = dist_mat.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            rows.append({
                "leaf_a": leaf_labels[i],
                "leaf_b": leaf_labels[j],
                "cluster_a": int(labels[i]),
                "cluster_b": int(labels[j]),
                "same_cluster": bool(labels[i] == labels[j]),
                "rmsd": float(dist_mat[i, j]),
            })
    return pd.DataFrame(rows)


def _cluster_pair_summary(pair_df: pd.DataFrame) -> pd.DataFrame:
    return (
        pair_df.groupby(["cluster_a", "cluster_b"], as_index=False)
        .agg(
            mean_rmsd=("rmsd", "mean"),
            median_rmsd=("rmsd", "median"),
            n_pairs=("rmsd", "count"),
        )
        .sort_values(["cluster_a", "cluster_b"])
    )


def _within_between_metrics(pair_df: pd.DataFrame) -> dict[str, float]:
    within = pair_df.loc[pair_df["same_cluster"], "rmsd"]
    between = pair_df.loc[~pair_df["same_cluster"], "rmsd"]
    within_mean = float(within.mean()) if len(within) else float("nan")
    between_mean = float(between.mean()) if len(between) else float("nan")
    ratio = (
        between_mean / within_mean
        if within_mean > 0 and np.isfinite(within_mean)
        else float("nan")
    )
    return {
        "within_mean_rmsd": within_mean,
        "within_median_rmsd": float(within.median()) if len(within) else float("nan"),
        "between_mean_rmsd": between_mean,
        "between_median_rmsd": float(between.median()) if len(between) else float("nan"),
        "separation_gap": between_mean - within_mean
        if np.isfinite(within_mean) and np.isfinite(between_mean)
        else float("nan"),
        "between_over_within_ratio": ratio,
        "n_within_pairs": int(len(within)),
        "n_between_pairs": int(len(between)),
    }


def _silhouette_on_rmsd(dist_mat: np.ndarray, labels: np.ndarray) -> float:
    unique = set(int(x) for x in labels)
    if len(unique) < 2 or dist_mat.shape[0] < 3:
        return float("nan")
    try:
        return float(silhouette_score(dist_mat, labels, metric="precomputed"))
    except Exception:
        return float("nan")


def plot_rmsd_heatmap(
    dist_mat: np.ndarray,
    labels: np.ndarray,
    leaf_labels: list[str],
    output_path: Path,
    *,
    title: str,
) -> None:
    order = np.argsort(labels)
    ordered = dist_mat[np.ix_(order, order)]
    ordered_labels = [leaf_labels[i] for i in order]

    size = max(8, min(24, 0.12 * len(order)))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(ordered, cmap="viridis", aspect="auto")
    fig.colorbar(im, ax=ax, label="RMSD (Å)")
    ax.set_title(title)
    if len(ordered_labels) <= 40:
        ax.set_xticks(range(len(ordered_labels)), ordered_labels, rotation=90, fontsize=6)
        ax.set_yticks(range(len(ordered_labels)), ordered_labels, fontsize=6)
    else:
        ax.set_xlabel("Segments (ordered by cluster_label)")
        ax.set_ylabel("Segments (ordered by cluster_label)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_within_between_boxplot(
    pair_df: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> None:
    within = pair_df.loc[pair_df["same_cluster"], "rmsd"]
    between = pair_df.loc[~pair_df["same_cluster"], "rmsd"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(
        [within.to_numpy(), between.to_numpy()],
        tick_labels=["within cluster", "between cluster"],
    )
    ax.set_ylabel("Pairwise RMSD (Å)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def validate_group(
    group: str,
    *,
    clusters_dir: Path,
    topology: Path,
    trajectory_dir: Path,
    selection: str,
    output_dir: Path,
    linkage: str,
    max_segments: Optional[int],
) -> dict:
    """Run structural validation for one feature group."""
    seg_csv = clusters_dir / group / "segments_clustered.csv"
    if not seg_csv.is_file():
        raise FileNotFoundError(f"Missing {seg_csv}")

    df = pd.read_csv(seg_csv)
    if max_segments is not None and len(df) > max_segments:
        df = df.head(max_segments).copy()

    segments = _segments_from_df(df)
    labels = df["cluster_label"].to_numpy(dtype=int)
    leaf_labels = df["leaf_label"].astype(str).tolist()
    n_clusters = int(df["cluster_label"].nunique())

    positions = extract_positions_multi_trajectory(
        segments,
        topology=topology,
        trajectory_dir=trajectory_dir,
        selection=selection,
    )
    dist_mat = compute_pairwise_rmsd_matrix(positions)

    pair_df = _pairwise_rmsd_table(dist_mat, labels, leaf_labels)
    metrics = _within_between_metrics(pair_df)

    struct_labels = _cluster_labels_fixed_k(
        dist_mat, n_clusters, method=linkage
    )
    ari = float(adjusted_rand_score(labels, struct_labels))
    sil_feat = _silhouette_on_rmsd(dist_mat, labels)

    group_dir = output_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        group_dir / "positions.npz",
        positions=positions,
        leaf_labels=np.array(leaf_labels, dtype=object),
        cluster_labels=labels,
    )
    pd.DataFrame(dist_mat, index=leaf_labels, columns=leaf_labels).to_csv(
        group_dir / "rmsd_distance_matrix.csv"
    )
    pair_df.to_csv(group_dir / "within_between_rmsd.csv", index=False)
    _cluster_pair_summary(pair_df).to_csv(
        group_dir / "cluster_pair_rmsd_summary.csv", index=False
    )

    summary_lines = [
        f"Structural validation — feature group: {group}",
        f"Selection: {selection}",
        f"Segments: {len(df)}",
        f"Feature clusters (k): {n_clusters}",
        f"Within-cluster mean RMSD: {metrics['within_mean_rmsd']:.4f} Å",
        f"Between-cluster mean RMSD: {metrics['between_mean_rmsd']:.4f} Å",
        f"Separation gap (between - within): {metrics['separation_gap']:.4f} Å",
        f"Between/within ratio: {metrics['between_over_within_ratio']:.4f}",
        f"Silhouette (feature labels on RMSD): {sil_feat:.4f}",
        f"ARI (feature vs structure clustering, k={n_clusters}): {ari:.4f}",
    ]
    (group_dir / "structure_validation_summary.txt").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    plot_rmsd_heatmap(
        dist_mat,
        labels,
        leaf_labels,
        group_dir / "rmsd_heatmap_by_cluster.png",
        title=f"RMSD heatmap — {group} (ordered by feature cluster)",
    )
    plot_within_between_boxplot(
        pair_df,
        group_dir / "within_between_boxplot.png",
        title=f"Within vs between RMSD — {group}",
    )

    row = {
        "group": group,
        "n_segments": len(df),
        "n_feature_clusters": n_clusters,
        "ari_feature_vs_structure": ari,
        "silhouette_feature_on_rmsd": sil_feat,
        **metrics,
    }
    log_event(
        "info",
        f"group={group}: within={metrics['within_mean_rmsd']:.3f} Å, "
        f"between={metrics['between_mean_rmsd']:.3f} Å, ARI={ari:.3f}",
        component="validate_changepoint_cluster_structures",
    )
    return row


def write_validation_report(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write cohort VALIDATION_REPORT.md from summary metrics."""
    lines = [
        "# Structural Validation Report",
        "",
        "Feature-based changepoint clusters validated against GSA assembly RMSD "
        "(`resname MOL`) at segment representative frames.",
        "",
        "## Cohort summary",
        "",
        "| Group | Segments | k | Within mean (Å) | Between mean (Å) | Ratio | ARI | Silhouette |",
        "|-------|----------|---|-----------------|------------------|-------|-----|------------|",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {r['group']} | {int(r['n_segments'])} | {int(r['n_feature_clusters'])} | "
            f"{r['within_mean_rmsd']:.3f} | {r['between_mean_rmsd']:.3f} | "
            f"{r['between_over_within_ratio']:.3f} | {r['ari_feature_vs_structure']:.3f} | "
            f"{r['silhouette_feature_on_rmsd']:.3f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- **gsa / combined**: Expect lower within-cluster RMSD and higher between/within ratio "
        "(structural metastable types). ARI > 0.3 supports the workflow.",
        "- **iodine / na_water**: Weaker structural separation is expected — guest/solvent "
        "regime labels on similar cage structures (see ANALYSIS_SUMMARY.md timing results).",
        "",
        "## Per-group artifacts",
        "",
        "See `{group}/structure_validation_summary.txt`, `rmsd_heatmap_by_cluster.png`, "
        "and `within_between_boxplot.png` for each feature group.",
        "",
    ])
    (output_dir / "VALIDATION_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate changepoint feature clusters with structural RMSD."
    )
    parser.add_argument(
        "--clusters-dir",
        default="output/changepoints/clusters",
        help="Directory with per-group segments_clustered.csv",
    )
    parser.add_argument(
        "--topology",
        required=True,
        help="Topology file (e.g. traj/BMMpM_ca.prmtop)",
    )
    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help="Directory containing {traj_id}.trj (or .xtc, .dcd, .nc)",
    )
    parser.add_argument(
        "--selection",
        default="resname MOL",
        help="MDAnalysis selection for structural RMSD (default: resname MOL)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/changepoints/structure_validation",
        help="Validation output directory",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        choices=list(DEFAULT_GROUPS),
        help="Feature groups to validate",
    )
    parser.add_argument(
        "--linkage",
        default="ward",
        help="Linkage for structure re-clustering (ARI comparison)",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Limit segments per group (smoke test)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clusters_dir = Path(args.clusters_dir)
    topology = Path(args.topology)
    trajectory_dir = Path(args.trajectory_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not topology.is_file():
        print(f"Topology not found: {topology}", file=sys.stderr)
        sys.exit(1)
    if not trajectory_dir.is_dir():
        print(f"Trajectory directory not found: {trajectory_dir}", file=sys.stderr)
        sys.exit(1)

    with RunContext.from_namespace(args, name="validate_changepoint_cluster_structures"):
        summary_rows: list[dict] = []
        for group in args.groups:
            with step(f"group={group}"):
                try:
                    row = validate_group(
                        group,
                        clusters_dir=clusters_dir,
                        topology=topology,
                        trajectory_dir=trajectory_dir,
                        selection=args.selection,
                        output_dir=output_dir,
                        linkage=args.linkage,
                        max_segments=args.max_segments,
                    )
                    summary_rows.append(row)
                except FileNotFoundError as exc:
                    log_event(
                        "error",
                        str(exc),
                        component="validate_changepoint_cluster_structures",
                    )
                    raise

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(
                output_dir / "validation_summary_all_groups.csv", index=False
            )
            write_validation_report(summary_df, output_dir)

    print("\nOutput files:")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(f"  {path}")


if __name__ == "__main__":
    main()
