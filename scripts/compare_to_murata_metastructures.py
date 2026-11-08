"""G5: Murata Table 1 occupancy from motif-RMSD frames.

Thin CLI around ``src.ChangepointAnalysis.catalog_runs.compare_to_murata_metastructures``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.catalog_runs import compare_to_murata_metastructures
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare motif-frame metastructure occupancy to Murata Table 1."
    )
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "murata_g5")
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument(
        "--qc-gsa-root",
        type=Path,
        default=ROOT / "output" / "gsa_features_by_cube",
        help="Root of per-cube *_gsa_features.csv folders used by the assembly QC rule",
    )
    p.add_argument(
        "--no-assembly-qc",
        action="store_true",
        help="Keep replicas that fail the assembly QC rule",
    )
    p.add_argument(
        "--own-elbow",
        action="store_true",
        help="Read endpoint clusters from the own-elbow detections (OWN_ELBOW_ENDPOINT_DIRS)",
    )
    k_choice = p.add_mutually_exclusive_group()
    k_choice.add_argument(
        "--k",
        type=int,
        help="Cross-tabulate the Ward cut at this k (default: the stored clustering)",
    )
    k_choice.add_argument(
        "--k-selected",
        type=Path,
        help="chemical_k_selected.csv from select_chemical_k.py; use each cube's selected_k",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = compare_to_murata_metastructures(
            output_root=args.output_root,
            out_dir=args.out_dir,
            cubes=args.cubes,
            qc_gsa_root=args.qc_gsa_root,
            no_assembly_qc=args.no_assembly_qc,
            own_elbow=args.own_elbow,
            k=args.k,
            k_selected=args.k_selected,
        )
    except FileNotFoundError as exc:
        print(exc, flush=True)
        return 1
    for path in written.values():
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
