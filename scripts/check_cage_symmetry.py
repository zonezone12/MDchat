#!/usr/bin/env python3
"""Decide which monomer relabellings are physically admissible (Test D1).

Thin CLI around ``src.ChangepointAnalysis.cage_symmetry.check_cage_symmetry``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.cage_symmetry import check_cage_symmetry
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Superposition test of cage symmetry operations.")
    p.add_argument("--cube", required=True, choices=list(KNOWN_CUBES))
    p.add_argument("--topology", type=Path)
    p.add_argument("--trajectory", type=Path)
    p.add_argument("--trajectory-format", default="TRJ")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: output/endpoint_cluster_validation_physical_k/symmetry",
    )
    p.add_argument("--n-ref-frames", type=int, default=50)
    p.add_argument("--own-elbow", action="store_true", default=True)
    p.add_argument("--k", type=int)
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = check_cage_symmetry(
            cube=args.cube,
            topology=args.topology,
            trajectory=args.trajectory,
            output_root=args.output_root,
            out_dir=args.out_dir,
            k_selected=args.k_selected,
            k=args.k,
            n_ref_frames=args.n_ref_frames,
            traj_format=args.trajectory_format,
        )
    except FileNotFoundError as exc:
        print(exc, flush=True)
        return 1
    for path in written.values():
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
