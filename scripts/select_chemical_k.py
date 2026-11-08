"""Automatic endpoint k: chemical_separation first, then Murata NMI with a one-SE rule.

Thin CLI around ``src.ChangepointAnalysis.cluster_k_diagnostics.write_chemical_k_selection``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.cluster_k_diagnostics import write_chemical_k_selection
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select endpoint k by Murata-label NMI among chemically valid k."
    )
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--chem-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_chemical_k",
        help="Directory with chemical_separation_by_k.csv",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Default: --chem-dir")
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument("--k-min", type=int, default=None, help="Default: the chemical scan's smallest k")
    p.add_argument("--k-max", type=int, default=None, help="Default: the chemical scan's largest k")
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Replica resamples for the NMI standard error (0 = plain max NMI)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sensitivity-k-max",
        type=int,
        nargs="*",
        default=[10],
        help="Scan upper bounds to re-run the selection at (the scan's own bound is always added)",
    )
    p.add_argument(
        "--own-elbow",
        action="store_true",
        help="Score the clusters of the own-elbow detections (OWN_ELBOW_ENDPOINT_DIRS)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = write_chemical_k_selection(
            output_root=args.output_root,
            chem_dir=args.chem_dir,
            out_dir=args.out_dir,
            cubes=args.cubes,
            k_min=args.k_min,
            k_max=args.k_max,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            sensitivity_k_max=args.sensitivity_k_max,
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
