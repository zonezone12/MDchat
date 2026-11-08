"""Remap stored hull-index site pairs onto Murata equatorial d1 (C2–C3).

Thin CLI around ``src.ChangepointAnalysis.murata_d1.remap_murata_d1_cohorts``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.murata_d1 import CUBES, remap_murata_d1_cohorts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Map existing endpoint_dist site-pair columns onto Murata d1 "
            "(R2/R3 equatorial C2–C3 proxies) without re-reading trajectories."
        )
    )
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "murata_d1")
    p.add_argument("--traj-dir", type=Path, default=ROOT / "traj")
    p.add_argument(
        "--cubes",
        nargs="*",
        default=list(CUBES),
        help="Cohort names (default: all six B* cubes)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = remap_murata_d1_cohorts(
            output_root=args.output_root,
            out_dir=args.out_dir,
            traj_dir=args.traj_dir,
            cubes=args.cubes,
        )
    except FileNotFoundError as exc:
        print(exc, flush=True)
        return 1
    for path in written.values():
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
