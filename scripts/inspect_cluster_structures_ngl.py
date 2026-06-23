"""
Visual inspection of changepoint cluster structures in NGL.

For a chosen feature group and cluster label, extracts the per-segment
*average* structure (mean coordinates over each segment's frame range),
superposes all averages onto the first segment, and writes a self-contained
NGL HTML viewer with checkboxes to show/hide individual segments.

Example
-------
python scripts/inspect_cluster_structures_ngl.py \\
    --segments-csv output/changepoints/clusters/gsa/segments_clustered.csv \\
    --cluster-label 2 \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectory-dir traj/BMMpM.bak \\
    --trajectory-layout nested \\
    --selection "resname MOL" \\
    --output-dir output/changepoints/cluster_inspection/k2_gsa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.utils.cluster_structure_ngl import inspect_cluster_structures
from src.utils.metastable_states import Segment
from src.utils.run_log import RunContext, log_event, step


def _segments_from_df(df: pd.DataFrame) -> list[Segment]:
    segments: list[Segment] = []
    for _, row in df.iterrows():
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        rep = (
            int(row["rep_frame"])
            if "rep_frame" in row and pd.notna(row["rep_frame"])
            else (start + end) // 2
        )
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


def resolve_segments_csv(
    segments_csv: Optional[Path],
    *,
    clusters_dir: Optional[Path],
    group: Optional[str],
) -> Path:
    if segments_csv is not None:
        return segments_csv
    if clusters_dir is None or group is None:
        raise ValueError("Provide --segments-csv or both --clusters-dir and --group")
    path = clusters_dir / group / "segments_clustered.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return path


def inspect_changepoint_cluster(
    df: pd.DataFrame,
    cluster_label: int,
    *,
    topology: Path,
    trajectory_dir: Path,
    selection: str,
    output_dir: Path,
    group: Optional[str],
    trajectory_layout: str,
    trajectory_filename: str,
    trajectory_format: Optional[str],
    segment_stride: int,
    repr_style: str,
    radius_scale: float,
    default_visible: str,
    write_pdb: bool,
    ref_index: int,
) -> dict:
    sub = df[df["cluster_label"] == cluster_label].copy().reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No segments with cluster_label={cluster_label}")

    segments = _segments_from_df(sub)
    leaf_labels = (
        sub["leaf_label"].astype(str).tolist()
        if "leaf_label" in sub.columns
        else [f"{s.traj_id}:seg{s.segment_id}" for s in segments]
    )
    group_tag = group or "segments"

    return inspect_cluster_structures(
        segments,
        leaf_labels,
        cluster_label=cluster_label,
        topology=topology,
        selection=selection,
        output_dir=output_dir,
        group_tag=group_tag,
        trajectory_dir=trajectory_dir,
        trajectory_layout=trajectory_layout,
        trajectory_filename=trajectory_filename,
        trajectory_format=trajectory_format,
        segment_stride=segment_stride,
        repr_style=repr_style,
        radius_scale=radius_scale,
        default_visible=default_visible,
        write_pdb=write_pdb,
        ref_index=ref_index,
        log_component="inspect_cluster_structures_ngl",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract segment-average structures for a changepoint cluster, "
            "align them, and write an interactive NGL HTML viewer."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--segments-csv",
        type=Path,
        help="Path to segments_clustered.csv",
    )
    src.add_argument(
        "--clusters-dir",
        type=Path,
        help="Cluster output dir (use with --group)",
    )
    parser.add_argument(
        "--group",
        help="Feature group subdirectory under --clusters-dir (gsa, iodine, ...)",
    )
    parser.add_argument(
        "--cluster-label",
        type=int,
        required=True,
        help="Cluster label to inspect (from segments_clustered.csv)",
    )
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument(
        "--trajectory-layout",
        default="nested",
        choices=("nested", "flat", "auto"),
    )
    parser.add_argument("--trajectory-filename", default="mdcrd_v")
    parser.add_argument(
        "--trajectory-format",
        default="TRJ",
        help="MDAnalysis format for extensionless trajectories (empty to auto-detect)",
    )
    parser.add_argument("--selection", default="resname MOL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/changepoints/cluster_inspection"),
    )
    parser.add_argument(
        "--segment-stride",
        type=int,
        default=1,
        help="Subsample frames when averaging within each segment (default: 1)",
    )
    parser.add_argument(
        "--repr-style",
        default="licorice",
        choices=("licorice", "ball+stick", "cartoon", "spacefill", "line"),
    )
    parser.add_argument("--radius-scale", type=float, default=1.4)
    parser.add_argument(
        "--default-visible",
        default="first",
        choices=("first", "all", "none"),
        help="Which segments are visible on load (default: first only)",
    )
    parser.add_argument(
        "--ref-index",
        type=int,
        default=0,
        help="Index of segment used as alignment reference (default: 0)",
    )
    parser.add_argument(
        "--write-pdb",
        action="store_true",
        help="Also write one aligned PDB per segment",
    )
    parser.add_argument(
        "--all-clusters",
        action="store_true",
        help="Process every cluster label in the CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topology = Path(args.topology)
    trajectory_dir = Path(args.trajectory_dir)
    output_dir = Path(args.output_dir)

    if not topology.is_file():
        print(f"Topology not found: {topology}", file=sys.stderr)
        sys.exit(1)
    if not trajectory_dir.is_dir():
        print(f"Trajectory directory not found: {trajectory_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        segments_csv = resolve_segments_csv(
            args.segments_csv,
            clusters_dir=args.clusters_dir,
            group=args.group,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if args.clusters_dir and not args.group and args.segments_csv is None:
        print("--group is required with --clusters-dir", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(segments_csv)
    if "cluster_label" not in df.columns:
        print("CSV missing cluster_label column", file=sys.stderr)
        sys.exit(1)

    group = args.group
    if group is None and "group" in df.columns and df["group"].nunique() == 1:
        group = str(df["group"].iloc[0])

    traj_format = args.trajectory_format.strip() or None
    cluster_labels = (
        sorted(df["cluster_label"].unique())
        if args.all_clusters
        else [args.cluster_label]
    )

    with RunContext.from_namespace(args, name="inspect_cluster_structures_ngl"):
        rows: list[dict] = []
        for cl in cluster_labels:
            with step(f"cluster={cl}"):
                try:
                    row = inspect_changepoint_cluster(
                        df,
                        int(cl),
                        topology=topology,
                        trajectory_dir=trajectory_dir,
                        selection=args.selection,
                        output_dir=output_dir,
                        group=group,
                        trajectory_layout=args.trajectory_layout,
                        trajectory_filename=args.trajectory_filename,
                        trajectory_format=traj_format,
                        segment_stride=max(1, args.segment_stride),
                        repr_style=args.repr_style,
                        radius_scale=args.radius_scale,
                        default_visible=args.default_visible,
                        write_pdb=args.write_pdb,
                        ref_index=args.ref_index,
                    )
                    rows.append(row)
                except ValueError as exc:
                    log_event(
                        "warning",
                        str(exc),
                        component="inspect_cluster_structures_ngl",
                    )

        if rows:
            summary_df = pd.DataFrame(rows)
            output_dir.mkdir(parents=True, exist_ok=True)
            summary_df.to_csv(output_dir / "inspection_summary.csv", index=False)

    print("\nOutput files:")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(f"  {path}")


if __name__ == "__main__":
    main()
