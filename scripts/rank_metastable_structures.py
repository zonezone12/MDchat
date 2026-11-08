#!/usr/bin/env python3
"""Rank each cube's endpoint clusters by apo-frame occupancy.

Thin CLI around ``src.ChangepointAnalysis.catalog_runs.rank_metastable_structures``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.catalog_runs import rank_metastable_structures
from src.ChangepointAnalysis.metastable_catalog import COPY_R


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank metastable endpoint structures at the label-free k.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument("--profiles", type=Path, default=ROOT / "output" / "endpoint_cluster_profiles")
    p.add_argument(
        "--replica-audit",
        type=Path,
        default=ROOT / "output" / "cluster_replica_audit_physical_k" / "single_replica_clusters.csv",
    )
    p.add_argument("--copy-r", type=float, default=COPY_R)
    p.add_argument(
        "--own-elbow",
        action="store_true",
        default=True,
        help="Read site-pair means from the own-elbow detections (default)",
    )
    p.add_argument("--qc-gsa-root", type=Path, default=ROOT / "output" / "gsa_features_by_cube")
    p.add_argument("--no-assembly-qc", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "metastable_structures_physical_k",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = rank_metastable_structures(
            output_root=args.output_root,
            k_selected=args.k_selected,
            profiles=args.profiles,
            replica_audit=args.replica_audit,
            copy_r=args.copy_r,
            own_elbow=args.own_elbow,
            qc_gsa_root=args.qc_gsa_root,
            no_assembly_qc=args.no_assembly_qc,
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
