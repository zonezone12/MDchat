#!/usr/bin/env python3
"""G4 at the quoted k: rank endpoint site pairs by η² at each cube's selected Ward cut.

The top-level ``endpoint_pair_cluster_correlation.csv`` ranks the stored
clustering (k=5). This ranks the cut at each cube's ``selected_k``
(``--k-selected``), a fixed ``--k``, or the stored k, with the same label-shuffle
permutation null and BH-FDR, and writes

  {changepoints_dir}/clusters/endpoint/by_k/k_XX/endpoint_pair_cluster_correlation.csv
  {out_dir}/top_pairs.csv     top --top-n site pairs per cube
  {out_dir}/fdr_summary.csv   site pairs tested and passing FDR per cube

Example
-------
python scripts/rank_endpoint_pairs_at_k.py --own-elbow \\
    --k-selected output/endpoint_cluster_chemical_k_own_elbow/chemical_k_selected.csv \\
    --out-dir output/endpoint_pair_ranking_selected_k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.cluster_k_diagnostics import (
    chosen_k_from_cohort,
    load_segments_at_k,
    resolve_endpoint_features_dir,
)
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.reporting import (
    endpoint_changepoints_dir,
    rank_cluster_discriminating_endpoint_features,
)


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
    selected_k: dict[str, int] = {}
    if args.k_selected is not None:
        sel = pd.read_csv(args.k_selected)
        selected_k = dict(zip(sel["cohort"].astype(str), sel["selected_k"].astype(int)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    top_rows: list[pd.DataFrame] = []
    fdr_rows: list[dict] = []

    for cube in args.cubes:
        cp_dir = endpoint_changepoints_dir(args.output_root, cube, own_elbow=args.own_elbow)
        if not (cp_dir / "clusters" / "endpoint").is_dir():
            print(f"  {cube}: no endpoint clustering in {cp_dir}", flush=True)
            continue
        if args.k_selected is not None and cube not in selected_k:
            print(f"  {cube}: not in {args.k_selected}, using the stored clustering", flush=True)
        k = args.k or selected_k.get(cube) or chosen_k_from_cohort(cp_dir, group="endpoint")
        segs = load_segments_at_k(cp_dir, k, group="endpoint")
        if segs.empty:
            print(f"  {cube}: no endpoint clustering at k={k}", flush=True)
            continue
        print(f"ranking {cube} at k={k} ({len(segs)} segments) …", flush=True)
        ranking = rank_cluster_discriminating_endpoint_features(
            cp_dir,
            resolve_endpoint_features_dir(cp_dir),
            group="endpoint",
            clustered_df=segs,
            n_permutations=args.n_permutations,
            fdr_alpha=args.fdr_alpha,
        )
        k_dir = cp_dir / "clusters" / "endpoint" / "by_k" / f"k_{k:02d}"
        k_dir.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(k_dir / "endpoint_pair_cluster_correlation.csv", index=False)

        sizes = segs.groupby("cluster_label").size()
        site = ranking[ranking["feature_kind"] == "site_pair"]
        top = site.head(args.top_n).copy()
        top.insert(0, "k", k)
        top.insert(0, "cohort", cube)
        top["best_cluster_size"] = top["best_cluster"].map(sizes).astype("Int64")
        top_rows.append(top)
        fdr_rows.append(
            {
                "cohort": cube,
                "k": k,
                "cluster_sizes": ",".join(str(int(sizes[c])) for c in sorted(sizes.index)),
                "n_pairs_tested": len(site),
                "n_fdr_pass": int(site["significant_fdr"].astype(bool).sum()),
                "n_eta_squared_ge_0_5": int((site["eta_squared"] >= 0.5).sum()),
                "top_pair": site["endpoint_label"].iloc[0] if len(site) else "",
                "top_eta_squared": float(site["eta_squared"].iloc[0]) if len(site) else float("nan"),
            }
        )

    if not fdr_rows:
        print("No cubes ranked", flush=True)
        return 1
    pd.concat(top_rows, ignore_index=True).to_csv(args.out_dir / "top_pairs.csv", index=False)
    fdr = pd.DataFrame(fdr_rows)
    fdr.to_csv(args.out_dir / "fdr_summary.csv", index=False)
    print(fdr.to_string(index=False), flush=True)
    print(f"Wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
