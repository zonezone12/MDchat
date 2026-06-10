"""
Find metastable structural states across multiple MD trajectories.

Segments each trajectory with ruptures changepoint detection, extracts
representative frames, clusters structures hierarchically by pairwise RMSD,
and writes labeled results.

Example
-------
python scripts/find_metastable_states.py \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectories "traj/BMMpM_*_mdcrd_v.trj" \\
    --selection "not water and not resname I and not resname Na+" \\
    --output-dir output/metastable_states
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.gsa_selections import GSAFeatureSelections
from src.utils.metastable_states import (
    format_cluster_summary,
    plot_dendrogram,
    process_trajectory_files,
)


def _expand_trajectories(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            p = Path(pat)
            if p.is_file():
                paths.append(p)
            elif p.is_dir():
                for ext in ("*.xtc", "*.trj", "*.dcd", "*.nc"):
                    paths.extend(sorted(p.glob(ext)))
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify metastable states across multiple trajectories."
    )
    parser.add_argument(
        "--topology",
        required=True,
        help="Path to shared topology (prmtop, pdb, etc.)",
    )
    parser.add_argument(
        "--trajectories",
        nargs="+",
        required=True,
        help="Trajectory file paths or glob patterns (e.g. traj/run*/WT_trj.xtc)",
    )
    parser.add_argument(
        "--selection",
        default="not water and not resname I and not resname Na+",
        help="MDAnalysis atom selection for target molecule (ignored with --use-gsa-features)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/metastable_states",
        help="Directory for CSV, plots, and summary",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for analysis")
    parser.add_argument("--ref-frame", type=int, default=0, help="RMSD reference frame")
    parser.add_argument(
        "--signal-metric",
        choices=["rmsd", "rg"],
        default="rmsd",
        help="Time series for ruptures segmentation",
    )
    parser.add_argument(
        "--method",
        default="Pelt",
        help="Ruptures search method (Pelt, Binseg, BottomUp, Window, Dynp, KernelCPD)",
    )
    parser.add_argument(
        "--cost-model",
        default="rbf",
        help="Ruptures cost model (l1, l2, rbf, normal, ...)",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=None,
        help="Ruptures penalty (omit for auto heuristic)",
    )
    parser.add_argument(
        "--n-bkps",
        type=int,
        default=None,
        help="Fixed number of breakpoints (Binseg/Dynp/Window)",
    )
    parser.add_argument(
        "--min-segment-frames",
        type=int,
        default=50,
        help="Minimum segment length in trajectory frames",
    )
    parser.add_argument(
        "--linkage",
        default="ward",
        help="SciPy hierarchical linkage method",
    )
    parser.add_argument(
        "--rmsd-cutoff",
        type=float,
        default=None,
        help="RMSD distance cutoff for fcluster (Å); omit for auto silhouette",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=10,
        help="Maximum clusters for auto silhouette selection",
    )
    parser.add_argument(
        "--use-gsa-features",
        action="store_true",
        help="Segment trajectories using GSA nanocube feature time series",
    )
    parser.add_argument(
        "--gsa-resname",
        default="MOL",
        help="GSA monomer residue name when --use-gsa-features (default: MOL)",
    )
    parser.add_argument(
        "--n-monomers",
        type=int,
        default=6,
        help="Number of GSA monomers in the nanocube",
    )
    parser.add_argument(
        "--guest-selection",
        default=None,
        help="Guest selection for GSA features (e.g. 'resname IOD')",
    )
    parser.add_argument(
        "--cluster-mode",
        choices=["rmsd", "features"],
        default="features",
        help="Cluster segments by GSA feature vectors or pairwise RMSD (GSA mode only)",
    )
    parser.add_argument(
        "--no-tier2",
        action="store_true",
        help="Skip Tier-2 GSA features during extraction",
    )
    parser.add_argument(
        "--no-guest",
        action="store_true",
        help="Skip guest GSA features during extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_paths = _expand_trajectories(args.trajectories)
    if not traj_paths:
        print("No trajectory files matched.", file=sys.stderr)
        sys.exit(1)

    print(f"Topology: {args.topology}")
    print(f"Trajectories ({len(traj_paths)}):")
    for p in traj_paths:
        print(f"  {p}")
    print(f"Selection: {args.selection}")
    if args.use_gsa_features:
        print(f"GSA mode: resname={args.gsa_resname}, n_monomers={args.n_monomers}")
        print(f"Cluster mode: {args.cluster_mode}")
        if args.guest_selection:
            print(f"Guest: {args.guest_selection}")
    print(f"Output: {out_dir}")
    print()

    gsa_selections = None
    if args.use_gsa_features:
        gsa_selections = GSAFeatureSelections(guest_sel=args.guest_selection)

    results = process_trajectory_files(
        args.topology,
        traj_paths,
        args.selection,
        stride=args.stride,
        ref_frame=args.ref_frame,
        signal_metric=args.signal_metric,
        ruptures_method=args.method,
        ruptures_cost=args.cost_model,
        penalty=args.penalty,
        n_bkps=args.n_bkps,
        min_segment_frames=args.min_segment_frames,
        linkage_method=args.linkage,
        rmsd_cutoff=args.rmsd_cutoff,
        k_max=args.k_max,
        use_gsa_features=args.use_gsa_features,
        gsa_selections=gsa_selections,
        gsa_resname=args.gsa_resname,
        n_monomers=args.n_monomers,
        include_tier2=not args.no_tier2,
        include_guest=not args.no_guest,
        cluster_mode=args.cluster_mode,
    )

    segments = results["segments"]
    dist_mat = results["distance_matrix"]
    clustering = results["clustering"]
    leaf_labels = results["leaf_labels"]

    seg_rows = []
    for s in segments:
        row = {
            "traj_file": s.traj_id,
            "segment_id": s.segment_id,
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
            "rep_frame": s.rep_frame,
            "n_frames": s.n_frames,
            "rmsd_mean": s.rmsd_mean,
            "rg_mean": s.rg_mean,
            "cluster_label": s.cluster_label,
        }
        if s.feature_means:
            row.update(s.feature_means)
        seg_rows.append(row)
    seg_df = pd.DataFrame(seg_rows)
    seg_csv = out_dir / "segments_all.csv"
    seg_df.to_csv(seg_csv, index=False)
    print(f"Wrote {seg_csv}")

    if "features_dfs" in results:
        for i, feat_df in enumerate(results["features_dfs"]):
            traj_stem = traj_paths[i].stem if i < len(traj_paths) else f"traj_{i}"
            feat_csv = out_dir / f"{traj_stem}_gsa_features.csv"
            feat_df.to_csv(feat_csv, index=False)
            print(f"Wrote {feat_csv}")

    n = dist_mat.shape[0]
    dist_df = pd.DataFrame(
        dist_mat,
        index=leaf_labels,
        columns=leaf_labels,
    )
    dist_csv = out_dir / "distance_matrix.csv"
    dist_df.to_csv(dist_csv)
    print(f"Wrote {dist_csv}")

    dendro_path = out_dir / "dendrogram.png"
    plot_dendrogram(
        clustering.linkage_matrix,
        leaf_labels,
        dendro_path,
        rmsd_cutoff=clustering.cutoff_distance if args.rmsd_cutoff else None,
    )
    print(f"Wrote {dendro_path}")

    summary_text = format_cluster_summary(segments, clustering)
    summary_path = out_dir / "cluster_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")
    print()
    print(summary_text)
    print()
    print(f"Wrote {summary_path}")
    print(f"Done — {clustering.n_clusters} structural type(s) across {n} segment(s).")


if __name__ == "__main__":
    main()
