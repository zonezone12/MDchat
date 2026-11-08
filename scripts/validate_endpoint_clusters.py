#!/usr/bin/env python3
"""Structural quality checks on endpoint clusters after clustering (Tests A–D).

Thin CLI around ``src.ChangepointAnalysis.endpoint_cluster_checks.validate_endpoint_clusters``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.cluster_validation import (
    DEFAULT_FRAME_SAMPLE_CAP,
    DEFAULT_FRAME_STRIDE_PS,
)
from src.ChangepointAnalysis.endpoint_cluster_checks import validate_endpoint_clusters
from src.ChangepointAnalysis.metastable_catalog import COPY_R


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate endpoint clusters (Tests A–D).")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--own-elbow", action="store_true")
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument("--cube", nargs="+", default=None)
    p.add_argument("--tests", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    p.add_argument(
        "--group",
        default="admissible",
        choices=("admissible", "rotations", "contact"),
        help="Monomer relabellings. 'admissible' needs D1 output.",
    )
    p.add_argument("--apo-only", action="store_true")
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=ROOT / "output" / "metastable_structures_physical_k",
    )
    p.add_argument(
        "--symmetry-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_validation_physical_k" / "symmetry",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_validation_physical_k",
    )
    p.add_argument("--no-assembly-qc", action="store_true")
    p.add_argument("--eccentric-z", type=float, default=3.5)
    p.add_argument("--min-segments", type=int, default=15)
    p.add_argument("--delta-bic", type=float, default=10.0)
    p.add_argument("--min-ashman", type=float, default=2.0)
    p.add_argument("--min-minor", type=float, default=0.15)
    p.add_argument("--min-minor-replicas", type=int, default=3)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--coverage-min", type=float, default=0.8)
    p.add_argument("--replica-r-min", type=float, default=0.8)
    p.add_argument("--sign-agree-min", type=float, default=0.9)
    p.add_argument("--merge-p95-max", type=float, default=1.2)
    p.add_argument("--copy-r", type=float, default=COPY_R)
    p.add_argument("--min-replica-ps", type=float, default=100.0)
    p.add_argument("--frame-stride-ps", type=int, default=DEFAULT_FRAME_STRIDE_PS)
    p.add_argument("--frame-cap", type=int, default=DEFAULT_FRAME_SAMPLE_CAP)
    p.add_argument("--random-state", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    kwargs = vars(args)
    cubes = kwargs.pop("cube")
    try:
        written = validate_endpoint_clusters(cubes=cubes, **kwargs)
    except FileNotFoundError as exc:
        print(exc, flush=True)
        return 1
    for path in written.values():
        print(f"Wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
