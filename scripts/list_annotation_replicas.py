#!/usr/bin/env python3
"""Replicas whose endpoint state leaves the tight family once.

Thin CLI around ``src.ChangepointAnalysis.catalog_runs.list_annotation_replicas``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.catalog_runs import MIN_BOUNDARY_PS, MIN_FLANK_PS, list_annotation_replicas


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List replicas with one departure from the tight endpoint family.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=ROOT / "output" / "metastable_structures_physical_k",
    )
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument("--min-flank-ps", type=float, default=MIN_FLANK_PS)
    p.add_argument("--min-boundary-ps", type=float, default=MIN_BOUNDARY_PS)
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "annotation_replicas")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = list_annotation_replicas(
            output_root=args.output_root,
            catalog_dir=args.catalog_dir,
            k_selected=args.k_selected,
            min_flank_ps=args.min_flank_ps,
            min_boundary_ps=args.min_boundary_ps,
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
