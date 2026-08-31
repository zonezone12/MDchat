"""
Compare endpoint cluster deformation proxies across B* cohort result directories.

Reads existing ``endpoint_cluster_deformation_summary.csv``,
``paper_d1_segment_states.csv``, and ``endpoint_pair_cluster_correlation.csv``
from each cohort and writes consolidated tables and plots.

Example
-------
python scripts/compare_endpoint_clusters_across_cohorts.py \\
    --output-root output \\
    --pattern "endpoint_changepoints_B*" \\
    --exclude-test \\
    --out-dir output/endpoint_cluster_cross_cohort \\
    --top-pairs 10
"""

from __future__ import annotations

import argparse
import fnmatch
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis import compare_endpoint_clusters_across_cohorts
from src.utils.run_log import RunContext


def discover_cohort_dirs(
    output_root: Path,
    pattern: str,
    *,
    exclude_test: bool,
) -> list[Path]:
    """Return changepoint dirs under *output_root* matching *pattern*."""
    if not output_root.is_dir():
        return []

    dirs: list[Path] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        if not fnmatch.fnmatch(child.name, pattern):
            continue
        if exclude_test and ("_test" in child.name or "_smoke" in child.name):
            continue
        deform = child / "endpoint_cluster_deformation_summary.csv"
        if not deform.exists():
            continue
        dirs.append(child)
    return dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare endpoint cluster deformation proxies across "
            "endpoint_changepoints_B* cohort directories."
        )
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory containing endpoint_changepoints_* folders",
    )
    p.add_argument(
        "--pattern",
        default="endpoint_changepoints_B*",
        help="Glob-style pattern for cohort directory names",
    )
    p.add_argument(
        "--exclude-test",
        action="store_true",
        default=True,
        help="Skip directories whose names contain _test or _smoke (default: on)",
    )
    p.add_argument(
        "--include-test",
        action="store_true",
        help="Include _test / _smoke directories",
    )
    p.add_argument(
        "--cohort-dirs",
        nargs="*",
        type=Path,
        default=None,
        help="Explicit cohort directories (overrides auto-discovery)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/endpoint_cluster_cross_cohort"),
        help="Directory for consolidated CSVs and plots",
    )
    p.add_argument(
        "--top-pairs",
        type=int,
        default=10,
        help="Number of top η² site pairs to include per cohort (default: 10)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exclude_test = args.exclude_test and not args.include_test

    if args.cohort_dirs:
        cohort_dirs = [Path(d) for d in args.cohort_dirs]
    else:
        cohort_dirs = discover_cohort_dirs(
            Path(args.output_root),
            args.pattern,
            exclude_test=exclude_test,
        )

    if not cohort_dirs:
        print(
            "No cohort directories found. Check --output-root / --pattern.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Cohorts ({len(cohort_dirs)}):")
    for d in cohort_dirs:
        print(f"  {d}")

    with RunContext.from_namespace(args, name="compare_endpoint_clusters_across_cohorts"):
        written = compare_endpoint_clusters_across_cohorts(
            cohort_dirs,
            args.out_dir,
            top_pairs=args.top_pairs,
        )

    if not written:
        print("No artifacts written.", file=sys.stderr)
        sys.exit(1)

    print(f"\nWrote {len(written)} artifact(s) to {args.out_dir}:")
    for name, path in sorted(written.items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
