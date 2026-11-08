#!/usr/bin/env python3
"""G4 at the quoted k: rank endpoint site pairs by η² at each cube's selected Ward cut.

Thin CLI around ``src.ChangepointAnalysis.reporting.rank_endpoint_pairs_at_k``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.reporting import rank_endpoint_pairs_at_k


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank endpoint site pairs by η² at a chosen Ward cut.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--out-dir", type=Path, default=ROOT / "output" / "endpoint_pair_ranking_selected_k"
    )
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument(
        "--own-elbow",
        action="store_true",
        help="Read endpoint clusters from the own-elbow detections (OWN_ELBOW_ENDPOINT_DIRS)",
    )
    k_choice = p.add_mutually_exclusive_group()
    k_choice.add_argument("--k", type=int, help="Rank the Ward cut at this k")
    k_choice.add_argument(
        "--k-selected",
        type=Path,
        help="chemical_k_selected.csv from select_chemical_k.py; use each cube's selected_k",
    )
    p.add_argument("--n-permutations", type=int, default=999)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--top-n", type=int, default=10)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = rank_endpoint_pairs_at_k(
            output_root=args.output_root,
            out_dir=args.out_dir,
            cubes=args.cubes,
            own_elbow=args.own_elbow,
            k=args.k,
            k_selected=args.k_selected,
            n_permutations=args.n_permutations,
            fdr_alpha=args.fdr_alpha,
            top_n=args.top_n,
        )
    except FileNotFoundError as exc:
        print(exc, flush=True)
        return 1
    for path in written.values():
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
