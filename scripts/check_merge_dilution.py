#!/usr/bin/env python3
"""Inspect clusters whose merge status is not settled by the catalog rule.

Thin CLI around ``src.ChangepointAnalysis.merge_dilution.check_merge_dilution``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.merge_dilution import check_merge_dilution
from src.ChangepointAnalysis.metastable_catalog import COPY_R


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--qc-gsa-root", type=Path, default=ROOT / "output" / "gsa_features_by_cube")
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument("--cube", nargs="+", default=None)
    p.add_argument("--k", type=int, default=None, help="Ward cut; default is the selected k of each cube")
    p.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=None,
        help="Compare this list instead of auto-detecting tangled pairs",
    )
    p.add_argument("--copy-r", type=float, default=COPY_R)
    p.add_argument("--r-min", type=float, default=0.70, help="Cross-family pairs at or above this r are near-misses")
    p.add_argument("--min-segments", type=int, default=5, help="Minimum segments in a part to trust its r")
    p.add_argument(
        "--belt-crosscheck",
        action="store_true",
        help="Also tabulate core / weak against the majority-belt labels (post hoc; needs motif frames)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_validation_physical_k" / "merge_dilution",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = check_merge_dilution(
            output_root=args.output_root,
            qc_gsa_root=args.qc_gsa_root,
            k_selected=args.k_selected,
            out_dir=args.out_dir,
            cubes=args.cube,
            k=args.k,
            clusters=args.clusters,
            belt_crosscheck=args.belt_crosscheck,
            copy_r=args.copy_r,
            r_min=args.r_min,
            min_segments=args.min_segments,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, flush=True)
        return 1
    for path in written.values():
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
