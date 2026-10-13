"""
Cluster changepoint-defined segments by feature group.

Logic lives in ``src.ChangepointAnalysis``; this file only parses arguments.

Example
-------
python scripts/cluster_changepoint_segments.py \\
    --changepoints-dir output/changepoints \\
    --output-dir output/changepoints/clusters

# With the assembly QC rule (drop replicas where the cube comes apart)
python scripts/cluster_changepoint_segments.py \\
    --changepoints-dir output/endpoint_changepoints_matched_BHHpM \\
    --groups endpoint \\
    --qc-gsa-dir output/gsa_features_by_cube/BHHpM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis import SegmentClusteringConfig
from src.ChangepointAnalysis.assembly_qc import (
    ASSEMBLY_QC_MIN_COMPONENT_SIZE,
    ASSEMBLY_QC_RG_MAX,
    assembly_qc_failures,
    scan_assembly_qc_dir,
)
from src.ChangepointAnalysis.feature_groups import ALL_GROUPS, DEFAULT_GROUPS
from src.ChangepointAnalysis.segment_clustering import (
    cluster_all_groups,
    load_segment_stats,
)
from src.utils.run_log import RunContext, log_event


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
        choices=list(ALL_GROUPS),
        help="Feature groups to cluster (default: the four GSA groups)",
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
    parser.add_argument(
        "--pca-by-k-only",
        action="store_true",
        help="Only write pca_clusters.png under existing by_k/k_XX/ folders "
        "(no re-clustering)",
    )
    parser.add_argument(
        "--qc-gsa-dir",
        default=None,
        help=(
            "One cube's folder of *_gsa_features.csv "
            "(e.g. output/gsa_features_by_cube/BHHpM). Trajectories with any frame "
            "above --qc-rg-max or below --qc-min-component-size are excluded "
            "before clustering; the scan is written to <output-dir>/assembly_qc.csv"
        ),
    )
    parser.add_argument(
        "--qc-rg-max",
        type=float,
        default=ASSEMBLY_QC_RG_MAX,
        help=f"Assembly QC: largest allowed assembly_rg in Å (default: {ASSEMBLY_QC_RG_MAX})",
    )
    parser.add_argument(
        "--qc-min-component-size",
        type=int,
        default=ASSEMBLY_QC_MIN_COMPONENT_SIZE,
        help=(
            "Assembly QC: smallest allowed largest_connected_component_size "
            f"(default: {ASSEMBLY_QC_MIN_COMPONENT_SIZE})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    changepoints_dir = Path(args.changepoints_dir)
    out_dir = Path(args.output_dir) if args.output_dir else changepoints_dir / "clusters"
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude: tuple[str, ...] = ()
    if args.qc_gsa_dir and not args.pca_by_k_only:
        qc = scan_assembly_qc_dir(
            args.qc_gsa_dir,
            rg_max=args.qc_rg_max,
            min_component_size=args.qc_min_component_size,
        )
        qc.to_csv(out_dir / "assembly_qc.csv", index=False)
        exclude = assembly_qc_failures(qc)
        print(
            f"Assembly QC: scanned {len(qc)} trajectories; "
            f"excluded: {', '.join(exclude) or 'none'}"
        )

    config = SegmentClusteringConfig(
        groups=tuple(args.groups),
        linkage=args.linkage,
        n_clusters=args.n_clusters,
        silhouette_k_max=args.silhouette_k_max,
        all_k_min=args.all_k_min,
        all_k_max=args.all_k_max,
        save_all_k=not args.no_save_all_k,
        distance_cutoff=args.distance_cutoff,
        auto_select_k=args.auto_select_k,
        k_max=args.k_max,
        skip_pca=args.skip_pca,
        pca_by_k_only=args.pca_by_k_only,
        exclude_traj_ids=exclude,
    )

    with RunContext.from_namespace(args, name="cluster_changepoint_segments"):
        df = load_segment_stats(changepoints_dir)
        log_event(
            "info",
            f"Loaded {len(df)} segment rows from {changepoints_dir}",
            component="cluster_changepoint_segments",
        )
        cluster_all_groups(df, out_dir, config)

    if args.pca_by_k_only:
        print("\nPCA plots:")
        for path in sorted(out_dir.rglob("by_k/k_*/pca_clusters.png")):
            print(f"  {path}")
    else:
        print("\nOutput files:")
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                print(f"  {path}")


if __name__ == "__main__":
    main()
