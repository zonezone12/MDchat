"""
Per-cube changepoint from mixed ``gsa_features_step1`` CSVs, then k-diagnostics.

Already-clustered groups are skipped, so iodine / na_water / combined can be
added on top of an existing ``gsa`` run without overwriting it.

Example
-------
python scripts/run_gsa_step1_cluster_k.py `
    --features-dir output/gsa_features_step1 `
    --output-root output `
    --groups gsa `
    --workers 6

python scripts/run_gsa_step1_cluster_k.py `
    --groups iodine na_water combined `
    --workers 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis import (
    compare_cluster_k_diagnostics,
    discover_cluster_k_cohort_dirs,
    run_gsa_step1_cohorts,
)
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
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
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "k-diagnostics directory. One group: this path "
            "(default output/{group}_cluster_k_diagnostics). "
            "Several groups: output/{group}_cluster_k_diagnostics each, "
            "or {out-dir}/{group} if --out-dir is set."
        ),
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=["gsa"],
        help="Feature groups to detect/cluster (default: gsa). "
        "Already-clustered groups are skipped.",
    )
    p.add_argument(
        "--cubes",
        nargs="*",
        default=None,
        choices=list(KNOWN_CUBES),
        help="Subset of cubes (default: all discovered B* prefixes)",
    )
    p.add_argument("--n-clusters", "--k", type=int, default=5, dest="n_clusters")
    p.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Parallel cube pipelines (default: 6)",
    )
    p.add_argument(
        "--skip-detect",
        action="store_true",
        help="Only run k-diagnostics on existing gsa_changepoints_* dirs",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run detect/cluster even if segments_clustered.csv exists",
    )
    p.add_argument(
        "--k-from",
        type=int,
        default=5,
        help="Lower k for Ward-split table (default: 5)",
    )
    p.add_argument(
        "--k-to",
        type=int,
        default=6,
        help="Upper k for Ward-split table (default: 6)",
    )
    return p.parse_args()


def _kdiag_out_dir(args: argparse.Namespace, group: str, n_groups: int) -> Path:
    if n_groups == 1:
        return args.out_dir or Path(f"output/{group}_cluster_k_diagnostics")
    if args.out_dir is None:
        return Path(f"output/{group}_cluster_k_diagnostics")
    return Path(args.out_dir) / group


def main() -> None:
    args = parse_args()
    groups = list(args.groups)
    output_root = Path(args.output_root)
    pattern = "gsa_changepoints_B*"

    with RunContext.from_namespace(args, name="run_gsa_step1_cluster_k"):
        if not args.skip_detect:
            result_dirs = run_gsa_step1_cohorts(
                Path(args.features_dir),
                output_root,
                cubes=args.cubes,
                groups=tuple(groups),
                n_clusters=args.n_clusters,
                skip_if_clustered=not args.force,
                force=args.force,
                workers=args.workers,
            )
            print(f"Changepoint dirs ({len(result_dirs)}):")
            for d in result_dirs:
                print(f"  {d}")

        any_written = False
        for group in groups:
            if args.cubes:
                cohort_dirs = [
                    output_root / f"gsa_changepoints_{c}" for c in args.cubes
                ]
                cohort_dirs = [
                    d
                    for d in cohort_dirs
                    if (d / "clusters" / group / "segments_clustered.csv").exists()
                ]
            else:
                cohort_dirs = discover_cluster_k_cohort_dirs(
                    output_root,
                    pattern,
                    group=group,
                    exclude_test=True,
                )

            if not cohort_dirs:
                print(
                    f"No clustered '{group}' cohort directories found.",
                    file=sys.stderr,
                )
                continue

            out_dir = _kdiag_out_dir(args, group, len(groups))
            written = compare_cluster_k_diagnostics(
                cohort_dirs,
                out_dir,
                group=group,
                k_from=args.k_from,
                k_to=args.k_to,
            )
            any_written = True
            print(f"\n[{group}] Wrote {len(written)} artifact(s) to {out_dir}:")
            for name, path in sorted(written.items()):
                print(f"  {name}: {path}")

    if not any_written:
        print(
            "No clustered cohort directories found. Run without --skip-detect first.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
