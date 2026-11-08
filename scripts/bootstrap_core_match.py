#!/usr/bin/env python3
"""Replica bootstrap of core-to-core deformation-profile r.

Thin CLI around ``src.ChangepointAnalysis.core_bootstrap.bootstrap_core_match``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.core_bootstrap import bootstrap_core_match
from src.ChangepointAnalysis.metastable_catalog import COPY_R


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--qc-gsa-root", type=Path, default=ROOT / "output" / "gsa_features_by_cube")
    p.add_argument("--cube", required=True)
    p.add_argument("--k", type=int, required=True, help="Ward cut")
    p.add_argument("--clusters", type=int, nargs="+", required=True, help="Clusters whose cores are compared")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--min-core", type=int, default=3, help="Minimum core segments in a draw to use that core")
    p.add_argument("--copy-r", type=float, default=COPY_R)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_validation_physical_k" / "merge_dilution" / "core_bootstrap",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = bootstrap_core_match(
            cube=args.cube,
            k=args.k,
            clusters=args.clusters,
            output_root=args.output_root,
            qc_gsa_root=args.qc_gsa_root,
            out_dir=args.out_dir,
            n_boot=args.n_boot,
            seed=args.seed,
            min_core=args.min_core,
            copy_r=args.copy_r,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, flush=True)
        return 1
    for path in written.values():
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
