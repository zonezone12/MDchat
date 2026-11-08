"""Automatic endpoint k without reference labels: chemical gate, then held-out cage geometry.

Thin CLI around ``src.ChangepointAnalysis.cluster_k_diagnostics.write_physical_k_selection``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.cluster_k_diagnostics import write_physical_k_selection
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select endpoint k by held-out cage geometry among chemically valid k."
    )
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--chem-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_chemical_k_own_elbow",
    )
    p.add_argument(
        "--gsa-features-root",
        type=Path,
        default=ROOT / "output" / "gsa_features_by_cube",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow",
    )
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument("--k-min", type=int, default=None)
    p.add_argument("--k-max", type=int, default=None)
    p.add_argument("--stride", type=int, default=5)
    p.add_argument("--n-bootstrap", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--own-elbow", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = write_physical_k_selection(
            output_root=args.output_root,
            chem_dir=args.chem_dir,
            gsa_features_root=args.gsa_features_root,
            out_dir=args.out_dir,
            cubes=args.cubes,
            k_min=args.k_min,
            k_max=args.k_max,
            stride=args.stride,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            own_elbow=args.own_elbow,
        )
    except FileNotFoundError as exc:
        print(exc, flush=True)
        return 1
    for path in written.values():
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
