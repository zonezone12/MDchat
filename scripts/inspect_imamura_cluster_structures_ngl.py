"""
Visual inspection of Imamura MSM macro/micro state structures in NGL.

Reads frame_states.csv from run_imamura_msm.py, groups consecutive frames
with the same label into dwell segments, extracts per-dwell average structures,
superposes them, and writes an interactive NGL HTML viewer with per-segment
toggles (same workflow as inspect_cluster_structures_ngl.py).

Example (explicit trajectories, as in run_imamura_msm)
------------------------------------------------------
python scripts/inspect_imamura_cluster_structures_ngl.py \\
    --imamura-dir output/imamura_msm \\
    --state-label 5 \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectories traj/BMMpM_891249_mdcrd_v.trj \\
    --format TRJ \\
    --selection "resname MOL" \\
    --output-dir output/imamura_msm/structure_inspection

Example (nested HPC .bak layout)
----------------------------------
python scripts/inspect_imamura_cluster_structures_ngl.py \\
    --frame-states-csv output/imamura_msm/frame_states.csv \\
    --state-label 5 \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectory-dir traj/BMMpM.bak \\
    --trajectory-layout nested \\
    --selection "resname MOL"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.cluster_structure_ngl import (
    build_trajectory_map,
    dwell_segments_from_frame_states,
    inspect_cluster_structures,
)
from src.utils.run_log import RunContext, log_event, step


def resolve_frame_states_csv(
    frame_states_csv: Optional[Path],
    *,
    imamura_dir: Optional[Path],
) -> Path:
    if frame_states_csv is not None:
        return frame_states_csv
    if imamura_dir is None:
        raise ValueError("Provide --frame-states-csv or --imamura-dir")
    path = imamura_dir / "frame_states.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return path


def inspect_imamura_state(
    df: pd.DataFrame,
    state_label: int,
    *,
    label_col: str,
    topology: Path,
    selection: str,
    output_dir: Path,
    trajectory_dir: Optional[Path],
    trajectory_map: Optional[dict[str, Path]],
    trajectory_layout: str,
    trajectory_filename: str,
    trajectory_format: Optional[str],
    segment_stride: int,
    repr_style: str,
    radius_scale: float,
    default_visible: str,
    write_pdb: bool,
    ref_index: int,
    min_dwell_frames: int,
    frame_step: Optional[int],
) -> dict:
    segments, leaf_labels = dwell_segments_from_frame_states(
        df,
        state_label,
        label_col=label_col,
        min_dwell_frames=min_dwell_frames,
        frame_step=frame_step,
    )
    if not segments:
        raise ValueError(
            f"No dwell segments for {label_col}={state_label} "
            f"(try lowering --min-dwell-frames)"
        )

    label_name = label_col.replace("_label", "")
    group_tag = f"imamura {label_name}"

    return inspect_cluster_structures(
        segments,
        leaf_labels,
        cluster_label=state_label,
        topology=topology,
        selection=selection,
        output_dir=output_dir,
        group_tag=group_tag,
        trajectory_dir=trajectory_dir,
        trajectory_map=trajectory_map,
        trajectory_layout=trajectory_layout,
        trajectory_filename=trajectory_filename,
        trajectory_format=trajectory_format,
        segment_stride=segment_stride,
        repr_style=repr_style,
        radius_scale=radius_scale,
        default_visible=default_visible,
        write_pdb=write_pdb,
        ref_index=ref_index,
        log_component="inspect_imamura_cluster_structures_ngl",
        cluster_prefix=label_name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Imamura MSM state structures: dwell-segment averages "
            "aligned in an interactive NGL HTML viewer."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--frame-states-csv",
        type=Path,
        help="Path to frame_states.csv from run_imamura_msm.py",
    )
    src.add_argument(
        "--imamura-dir",
        type=Path,
        help="Imamura MSM output directory (reads frame_states.csv inside)",
    )
    parser.add_argument(
        "--state-label",
        type=int,
        default=None,
        help="Macrostate or microstate label to inspect (omit with --all-states)",
    )
    parser.add_argument(
        "--label-col",
        default="macro_label",
        choices=("macro_label", "micro_label"),
        help="Cluster column in frame_states.csv (default: macro_label)",
    )
    parser.add_argument("--topology", type=Path, required=True)
    traj = parser.add_mutually_exclusive_group(required=True)
    traj.add_argument(
        "--trajectories",
        nargs="+",
        help="Trajectory paths or globs (same as run_imamura_msm.py)",
    )
    traj.add_argument(
        "--trajectory-dir",
        type=Path,
        help="Base directory for nested/flat trajectory layout",
    )
    parser.add_argument(
        "--trajectory-layout",
        default="nested",
        choices=("nested", "flat", "auto"),
    )
    parser.add_argument("--trajectory-filename", default="mdcrd_v")
    parser.add_argument(
        "--format",
        "--trajectory-format",
        dest="trajectory_format",
        default="TRJ",
        help="MDAnalysis format for extensionless trajectories (empty to auto-detect)",
    )
    parser.add_argument("--selection", default="resname MOL")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <imamura-dir>/structure_inspection)",
    )
    parser.add_argument(
        "--segment-stride",
        type=int,
        default=1,
        help="Subsample frames when averaging within each dwell (default: 1)",
    )
    parser.add_argument(
        "--min-dwell-frames",
        type=int,
        default=1,
        help="Skip dwell periods shorter than this many trajectory frames",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=None,
        help="Override frame spacing in frame_states.csv (auto-detected if omitted)",
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
    )
    parser.add_argument("--ref-index", type=int, default=0)
    parser.add_argument("--write-pdb", action="store_true")
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Generate viewers for every label in --label-col",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    topology = Path(args.topology)

    if not topology.is_file():
        print(f"Topology not found: {topology}", file=sys.stderr)
        sys.exit(1)

    try:
        frame_states_csv = resolve_frame_states_csv(
            args.frame_states_csv,
            imamura_dir=args.imamura_dir,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif args.imamura_dir is not None:
        output_dir = Path(args.imamura_dir) / "structure_inspection"
    else:
        output_dir = frame_states_csv.parent / "structure_inspection"

    trajectory_map: Optional[dict[str, Path]] = None
    trajectory_dir: Optional[Path] = None
    if args.trajectories:
        trajectory_map = build_trajectory_map(args.trajectories)
        if not trajectory_map:
            print("No trajectories matched --trajectories patterns.", file=sys.stderr)
            sys.exit(1)
    else:
        trajectory_dir = Path(args.trajectory_dir)
        if not trajectory_dir.is_dir():
            print(f"Trajectory directory not found: {trajectory_dir}", file=sys.stderr)
            sys.exit(1)

    df = pd.read_csv(frame_states_csv)
    if args.label_col not in df.columns:
        print(f"CSV missing {args.label_col!r} column", file=sys.stderr)
        sys.exit(1)

    traj_format = (args.trajectory_format or "").strip() or None
    if args.all_states:
        state_labels = sorted(int(x) for x in df[args.label_col].unique())
    elif args.state_label is not None:
        state_labels = [args.state_label]
    else:
        print("Provide --state-label or use --all-states", file=sys.stderr)
        sys.exit(1)

    with RunContext.from_namespace(args, name="inspect_imamura_cluster_structures_ngl"):
        rows: list[dict] = []
        for state in state_labels:
            with step(f"{args.label_col}={state}"):
                try:
                    row = inspect_imamura_state(
                        df,
                        state,
                        label_col=args.label_col,
                        topology=topology,
                        selection=args.selection,
                        output_dir=output_dir,
                        trajectory_dir=trajectory_dir,
                        trajectory_map=trajectory_map,
                        trajectory_layout=args.trajectory_layout,
                        trajectory_filename=args.trajectory_filename,
                        trajectory_format=traj_format,
                        segment_stride=max(1, args.segment_stride),
                        repr_style=args.repr_style,
                        radius_scale=args.radius_scale,
                        default_visible=args.default_visible,
                        write_pdb=args.write_pdb,
                        ref_index=args.ref_index,
                        min_dwell_frames=max(1, args.min_dwell_frames),
                        frame_step=args.frame_step,
                    )
                    rows.append(row)
                except ValueError as exc:
                    log_event(
                        "warning",
                        str(exc),
                        component="inspect_imamura_cluster_structures_ngl",
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
