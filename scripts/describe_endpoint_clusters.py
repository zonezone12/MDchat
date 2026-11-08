#!/usr/bin/env python3
"""Per-cluster motif profile, and the site pairs that separate a chosen set of clusters.

Thin CLI around ``src.ChangepointAnalysis.catalog_runs.describe_endpoint_clusters``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.catalog_runs import describe_endpoint_clusters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Describe endpoint clusters by motif state and site pairs.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--cube", required=True)
    p.add_argument(
        "--own-elbow",
        action="store_true",
        help="Read endpoint clusters from the own-elbow detections (OWN_ELBOW_ENDPOINT_DIRS)",
    )
    p.add_argument("--k", type=int, help="Ward cut (default: the stored clustering)")
    p.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        help="Clusters to contrast with each other (default: the belt-broken clusters)",
    )
    p.add_argument(
        "--qc-gsa-root",
        type=Path,
        default=ROOT / "output" / "gsa_features_by_cube",
        help="Root of per-cube *_gsa_features.csv folders used by the assembly QC rule",
    )
    p.add_argument("--no-assembly-qc", action="store_true")
    p.add_argument("--n-permutations", type=int, default=999)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument(
        "--out-dir",
        type=Path,
        help="Default: {output-root}/endpoint_cluster_profiles/{CUBE}_kXX",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = describe_endpoint_clusters(
            output_root=args.output_root,
            cube=args.cube,
            own_elbow=args.own_elbow,
            k=args.k,
            clusters=args.clusters,
            qc_gsa_root=args.qc_gsa_root,
            no_assembly_qc=args.no_assembly_qc,
            n_permutations=args.n_permutations,
            fdr_alpha=args.fdr_alpha,
            top_n=args.top_n,
            out_dir=args.out_dir,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(exc, flush=True)
        return 1
    for path in written.values():
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
