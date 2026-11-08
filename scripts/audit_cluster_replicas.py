"""Which endpoint clusters are one replica's excursion rather than a population state?

Thin CLI around ``src.ChangepointAnalysis.cluster_replicas.audit_cluster_replicas``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.cluster_replicas import audit_cluster_replicas
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flag endpoint clusters carried by a single replica.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "cluster_replica_audit")
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument(
        "--own-elbow",
        action="store_true",
        help="Audit the clusters of the own-elbow detections (OWN_ELBOW_ENDPOINT_DIRS)",
    )
    k_choice = p.add_mutually_exclusive_group()
    k_choice.add_argument("--k", type=int, help="Audit the Ward cut at this k (default: the stored clustering)")
    k_choice.add_argument(
        "--k-selected",
        type=Path,
        help="chemical_k_selected.csv from select_chemical_k.py; use each cube's selected_k",
    )
    p.add_argument(
        "--gsa-root",
        type=Path,
        default=ROOT / "output" / "gsa_features_by_cube",
        help="Root of per-cube *_gsa_features.csv folders (assembly Rg, connectivity)",
    )
    p.add_argument(
        "--min-top-share",
        type=float,
        default=0.8,
        help="Flag a cluster when one replica holds at least this share of its frames (default: 0.8)",
    )
    p.add_argument("--top-n", type=int, default=10, help="Feature margins kept per flagged cluster")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = audit_cluster_replicas(
            output_root=args.output_root,
            out_dir=args.out_dir,
            cubes=args.cubes,
            own_elbow=args.own_elbow,
            k=args.k,
            k_selected=args.k_selected,
            gsa_root=args.gsa_root,
            min_top_share=args.min_top_share,
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
