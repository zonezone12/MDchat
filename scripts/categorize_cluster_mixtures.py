#!/usr/bin/env python3
"""Split each endpoint cluster into contact categories that do not depend on k.

Thin CLI around ``src.ChangepointAnalysis.catalog_runs.categorize_cluster_mixtures``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.catalog_runs import MIN_SHARE, categorize_cluster_mixtures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Categorize the mixture inside each endpoint cluster.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument("--qc-gsa-root", type=Path, default=ROOT / "output" / "gsa_features_by_cube")
    p.add_argument("--min-share", type=float, default=MIN_SHARE)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "metastable_structures_physical_k",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = categorize_cluster_mixtures(
            output_root=args.output_root,
            k_selected=args.k_selected,
            qc_gsa_root=args.qc_gsa_root,
            min_share=args.min_share,
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
