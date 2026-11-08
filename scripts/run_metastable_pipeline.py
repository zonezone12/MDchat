"""From a trajectory folder (or existing own-elbow clusters) to the metastable catalog.

Two uses:

1. **Trajectories of one cube.** Extract GSA features, run GSA and endpoint
   changepoints (own-elbow folders, Ward k = 2…15), compute motif RMSD, then
   run the catalog on whatever cubes are already clustered under ``--output-root``.
2. **Catalog only.** Own-elbow segments already on disk; skip trajectory reads.

The catalog does not re-detect changepoints or refit Ward.

Example
-------
python scripts/run_metastable_pipeline.py `
    --cube BHHpH `
    --topology traj/BHHpH_ca.prmtop `
    --trajectories "traj/BHHpH_*_mdcrd_v.trj"

# Nested HPC layout
python scripts/run_metastable_pipeline.py `
    --cube BMMpM `
    --topology "$TOPOLOGY" `
    --trajectories "$TRAJ_DIR/*/mdcrd_v"

# Catalog on existing own-elbow dirs
python scripts/run_metastable_pipeline.py --output-root output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.metastable_pipeline import (
    _cube_from_topology,
    run_cube_from_trajectories,
    run_metastable_pipeline,
)
from src.ChangepointAnalysis.traj_paths import expand_trajectories, format_no_trajectories_error
from src.utils.run_log import RunContext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Trajectory folder → GSA/endpoint/motif, then the metastable catalog. "
            "Omit --trajectories to run the catalog only."
        )
    )
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--cube", choices=list(KNOWN_CUBES), help="Required with --trajectories")
    p.add_argument("--topology", type=Path, help="Shared topology (required with --trajectories)")
    p.add_argument(
        "--trajectories",
        nargs="+",
        help="Trajectory paths or globs. Nested HPC: '$TRAJ_DIR/*/mdcrd_v'.",
    )
    p.add_argument("--traj-format", default=None)
    p.add_argument("--gsa-resname", default="MOL")
    p.add_argument("--guest-selection", default="resname IOD")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--traj-jobs", type=int, default=-1)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--no-sweep", action="store_true", help="Skip Pelt penalty elbow")
    p.add_argument("--no-site-pairs", action="store_true", help="Do not write the 960 raw site pairs")
    p.add_argument("--k-max", type=int, default=15, help="Ward by_k upper cut (default: 15)")
    p.add_argument("--force", action="store_true", help="Re-run trajectory stages even if outputs exist")
    p.add_argument("--skip-gsa-features", action="store_true")
    p.add_argument("--skip-gsa-changepoint", action="store_true")
    p.add_argument("--skip-endpoint", action="store_true")
    p.add_argument("--skip-motif", action="store_true")
    p.add_argument("--skip-catalog", action="store_true", help="Stop after this cube's trajectory stages")
    p.add_argument("--skip-guest", action="store_true")
    p.add_argument("--skip-annotation", action="store_true")
    p.add_argument(
        "--chem-dir",
        type=Path,
        default=None,
        help="Default: {output-root}/endpoint_cluster_chemical_k_own_elbow_replica_aware",
    )
    p.add_argument(
        "--physical-k-dir",
        type=Path,
        default=None,
        help="Default: {output-root}/endpoint_cluster_physical_k_own_elbow",
    )
    p.add_argument("--max-top-replica-share", type=float, default=0.5)
    p.add_argument("--n-permutations", type=int, default=999)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    from_traj = bool(args.trajectories)
    if from_traj:
        if not args.topology:
            print("--topology is required with --trajectories", file=sys.stderr)
            return 2
        cube = args.cube or _cube_from_topology(args.topology)
        if cube not in KNOWN_CUBES:
            print(f"Unknown cube {cube!r}; pass --cube ({', '.join(KNOWN_CUBES)})", file=sys.stderr)
            return 2
        trajs = expand_trajectories(args.trajectories)
        if not trajs:
            print(format_no_trajectories_error(args.trajectories), file=sys.stderr)
            return 1
        print(f"Found {len(trajs)} trajectory file(s) for {cube}", flush=True)

    chem_dir = args.chem_dir or output_root / "endpoint_cluster_chemical_k_own_elbow_replica_aware"
    physical_k_dir = args.physical_k_dir or output_root / "endpoint_cluster_physical_k_own_elbow"
    gsa_root = output_root / "gsa_features_by_cube"

    with RunContext.from_namespace(args, name="run_metastable_pipeline"):
        written: dict = {}
        if from_traj:
            written.update(
                run_cube_from_trajectories(
                    cube=cube,
                    topology=args.topology,
                    trajectories=args.trajectories,
                    output_root=output_root,
                    gsa_resname=args.gsa_resname,
                    guest_selection=args.guest_selection,
                    traj_format=args.traj_format,
                    stride=args.stride,
                    start=args.start,
                    stop=args.stop,
                    traj_jobs=args.traj_jobs,
                    n_jobs=args.n_jobs,
                    with_sweep=not args.no_sweep,
                    include_site_pairs=not args.no_site_pairs,
                    k_max=args.k_max,
                    skip_gsa_features=args.skip_gsa_features,
                    skip_gsa_changepoint=args.skip_gsa_changepoint,
                    skip_endpoint=args.skip_endpoint,
                    skip_motif=args.skip_motif,
                    force=args.force,
                )
            )
        if not args.skip_catalog:
            print("catalog: chemical scan → physical k → structures", flush=True)
            try:
                written.update(
                    run_metastable_pipeline(
                        output_root=output_root,
                        chem_dir=chem_dir,
                        physical_k_dir=physical_k_dir,
                        gsa_features_root=gsa_root,
                        cubes=None,
                        k_max=args.k_max,
                        max_top_replica_share=args.max_top_replica_share,
                        n_permutations=args.n_permutations,
                        skip_guest=args.skip_guest,
                        skip_annotation=args.skip_annotation,
                    )
                )
            except FileNotFoundError as exc:
                print(exc, flush=True)
                if from_traj:
                    print(
                        "This cube's trajectory stages finished. "
                        "Re-run without --trajectories after the other cubes are clustered, "
                        "or pass --skip-catalog.",
                        flush=True,
                    )
                return 1
    for path in written.values():
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
