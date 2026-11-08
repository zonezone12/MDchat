#!/usr/bin/env python3
"""Outlier runs: replicas the workflow flags as abnormal rather than as ensemble states.

Thin CLI around ``src.ChangepointAnalysis.cluster_replicas.write_replica_outliers``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.cluster_replicas import write_replica_outliers
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flag outlier replicas: dissociation and isolation by clustering.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--gsa-root", type=Path, default=ROOT / "output" / "gsa_features_by_cube")
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=15)
    p.add_argument("--min-top-share", type=float, default=0.5)
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "replica_outliers")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = write_replica_outliers(
            output_root=args.output_root,
            gsa_root=args.gsa_root,
            k_selected=args.k_selected,
            cubes=args.cubes,
            k_min=args.k_min,
            k_max=args.k_max,
            min_top_share=args.min_top_share,
            out_dir=args.out_dir,
        )
    except FileNotFoundError as exc:
        print(exc, flush=True)
        return 1
    for path in written.values():
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
