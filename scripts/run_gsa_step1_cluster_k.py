"""
Per-cube changepoint from mixed ``gsa_features_step1`` CSVs, then k-diagnostics.

Thin CLI around ``src.ChangepointAnalysis.gsa_cohort_run.run_gsa_step1_cluster_k``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES, run_gsa_step1_cluster_k
from src.utils.run_log import RunContext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run per-cube GSA changepoint+clustering on gsa_features_step1, "
            "then silhouette / switching / geometry k-diagnostics."
        )
    )
    p.add_argument(
        "--features-dir",
        type=Path,
        default=Path("output/gsa_features_step1"),
        help="Root with mixed (and nested) *_gsa_features.csv files",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Where gsa_changepoints_{CUBE} directories are written",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--groups", nargs="+", default=["gsa"])
    p.add_argument("--cubes", nargs="*", default=None, choices=list(KNOWN_CUBES))
    p.add_argument("--n-clusters", "--k", type=int, default=5, dest="n_clusters")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip-detect", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--k-from", type=int, default=5)
    p.add_argument("--k-to", type=int, default=6)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with RunContext.from_namespace(args, name="run_gsa_step1_cluster_k"):
        try:
            written = run_gsa_step1_cluster_k(
                features_dir=args.features_dir,
                output_root=args.output_root,
                out_dir=args.out_dir,
                groups=args.groups,
                cubes=args.cubes,
                n_clusters=args.n_clusters,
                workers=args.workers,
                skip_detect=args.skip_detect,
                force=args.force,
                k_from=args.k_from,
                k_to=args.k_to,
            )
        except FileNotFoundError as exc:
            print(exc, flush=True)
            return 1
    for path in written.values():
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
