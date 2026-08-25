"""
Compare changepoint timing across already-computed result directories.

Use this after an endpoint run and a GSA-feature run exist. Trajectory IDs
are matched after stripping cube prefixes (``BMMpM_109345_mdcrd_v`` ↔
``109345_mdcrd_v``).

Example
-------
python scripts/compare_changepoint_timing.py \\
    --changepoints-dirs output/endpoint_changepoints_BMMpM output/changepoints \\
    --output-dir output/endpoint_vs_gsa_timing_BMMpM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis import compare_changepoint_timing
from src.ChangepointAnalysis.reporting import plot_jaccard_heatmap
from src.utils.run_log import RunContext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare breakpoint timing between changepoint result directories "
            "(e.g. endpoint vs GSA groups)."
        )
    )
    p.add_argument(
        "--changepoints-dirs",
        nargs="+",
        required=True,
        help=(
            "Two or more directories that contain all_breakpoints.csv "
            "(endpoint run + GSA changepoints)."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the merged comparison CSVs and Jaccard heatmap",
    )
    p.add_argument(
        "--tolerance-frames",
        type=int,
        default=50,
        help="Window for counting a breakpoint as shared (default: 50)",
    )
    p.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip writing plots/cohort_jaccard_heatmap.png",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dirs = [Path(d) for d in args.changepoints_dirs]
    missing = [d for d in dirs if not (d / "all_breakpoints.csv").exists()]
    if missing:
        print("Missing all_breakpoints.csv in:", file=sys.stderr)
        for d in missing:
            print(f"  {d}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    with RunContext.from_namespace(args, name="compare_changepoint_timing"):
        cmp = compare_changepoint_timing(
            *dirs,
            tolerance_frames=args.tolerance_frames,
            output_dir=output_dir,
        )
        if not args.skip_plot:
            plot_dir = output_dir / "plots"
            plot_jaccard_heatmap(output_dir, plot_dir)

    print(f"Wrote {output_dir / 'changepoint_timing_comparison.csv'} ({len(cmp)} rows)")
    print(f"Wrote {output_dir / 'cohort_timing_summary.csv'}")
    if not cmp.empty:
        print("\nMedian Jaccard by group pair:")
        summary = (
            cmp.groupby(["group_a", "group_b"])["jaccard"]
            .median()
            .sort_values(ascending=False)
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
