#!/usr/bin/env python3
"""One row per metastable structure, with the numbers quoted in docs/metastable_structures.md §7.

Thin CLI around ``src.ChangepointAnalysis.catalog_runs.summarize_metastable_structures``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.catalog_runs import summarize_metastable_structures


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize metastable structures for the methods note.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=ROOT / "output" / "metastable_structures_physical_k",
        help="Folder with clusters_ranked.csv and families_ranked.csv",
    )
    p.add_argument(
        "--segment-belt",
        type=Path,
        default=ROOT / "output" / "metastable_structures_physical_k" / "min_share_0" / "segment_belt.csv",
        help="segment_belt.csv written with --min-share 0",
    )
    p.add_argument(
        "--top-pairs",
        type=Path,
        default=ROOT / "output" / "endpoint_pair_ranking_physical_k" / "top_pairs.csv",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Default: --catalog-dir")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = summarize_metastable_structures(
            output_root=args.output_root,
            catalog_dir=args.catalog_dir,
            segment_belt=args.segment_belt,
            top_pairs=args.top_pairs,
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
