"""
Compare endpoint-segment k diagnostics across B* cohort result directories.

Reads existing ``clusters/endpoint/`` silhouette curves, clustered segments,
paper-d1 tables, and endpoint-site maps. Does not re-detect or re-cluster.

Example
-------
python scripts/compare_endpoint_cluster_k.py \\
    --output-root output \\
    --pattern "endpoint_changepoints_B*" \\
    --exclude-test \\
    --chemical-scan \\
    --out-dir output/endpoint_cluster_chemical_k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis import (
    compare_cluster_k_diagnostics,
    compare_endpoint_cluster_k_diagnostics,
    discover_cluster_k_cohort_dirs,
    discover_endpoint_k_cohort_dirs,
)
from src.utils.run_log import RunContext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare silhouette-vs-k, Pelt switching, and paper-d1 occupancy "
            "across endpoint_changepoints_B* cohort directories."
        )
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root directory containing endpoint_changepoints_* folders",
    )
    p.add_argument(
        "--group",
        default="endpoint",
        help="Feature group whose clusters/ folder to read (default: endpoint)",
    )
    p.add_argument(
        "--pattern",
        default=None,
        help="Glob-style pattern for cohort directory names "
        "(default: endpoint_changepoints_B* or gsa_changepoints_B*)",
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
        default=None,
        help="Directory for CSVs and plots "
        "(default: output/{group}_cluster_k_diagnostics)",
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
    p.add_argument(
        "--chemical-scan",
        action="store_true",
        help=(
            "G3: rank site pairs at each k=2–10 and flag chemical separation "
            "(distinct top-η² contacts / Cohen's d signs + G4 permutation FDR). "
            "Reads endpoint feature CSVs once per cohort."
        ),
    )
    p.add_argument(
        "--chemical-k-min",
        type=int,
        default=2,
        help="Minimum k for the chemical scan (default: 2)",
    )
    p.add_argument(
        "--chemical-k-max",
        type=int,
        default=10,
        help="Maximum k for the chemical scan (default: 10)",
    )
    p.add_argument(
        "--n-permutations",
        type=int,
        default=999,
        help="Label-shuffle permutations for η² p-values (default: 999)",
    )
    p.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        help="BH-FDR alpha (default: 0.05)",
    )
    p.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="Ignore clusters smaller than this when scoring chemical splits (default: 5)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exclude_test = args.exclude_test and not args.include_test

    group = str(args.group)
    pattern = args.pattern or (
        "gsa_changepoints_B*" if group == "gsa" else f"{group}_changepoints_B*"
        if group != "endpoint"
        else "endpoint_changepoints_B*"
    )
    out_dir = args.out_dir or Path(f"output/{group}_cluster_k_diagnostics")

    if args.cohort_dirs:
        cohort_dirs = [Path(d) for d in args.cohort_dirs]
    elif group == "endpoint":
        cohort_dirs = discover_endpoint_k_cohort_dirs(
            Path(args.output_root),
            pattern,
            exclude_test=exclude_test,
        )
    else:
        cohort_dirs = discover_cluster_k_cohort_dirs(
            Path(args.output_root),
            pattern,
            group=group,
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

    with RunContext.from_namespace(args, name="compare_endpoint_cluster_k"):
        if group == "endpoint":
            written = compare_endpoint_cluster_k_diagnostics(
                cohort_dirs,
                out_dir,
                k_from=args.k_from,
                k_to=args.k_to,
                include_chemical_scan=args.chemical_scan,
                chemical_k_min=args.chemical_k_min,
                chemical_k_max=args.chemical_k_max,
                n_permutations=args.n_permutations,
                fdr_alpha=args.fdr_alpha,
                min_cluster_size=args.min_cluster_size,
            )
        else:
            written = compare_cluster_k_diagnostics(
                cohort_dirs,
                out_dir,
                group=group,
                k_from=args.k_from,
                k_to=args.k_to,
            )

    if not written:
        print("No artifacts written.", file=sys.stderr)
        sys.exit(1)

    print(f"\nWrote {len(written)} artifact(s) to {out_dir}:")
    for name, path in sorted(written.items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
