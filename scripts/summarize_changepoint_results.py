"""
Summarize changepoint outputs and generate cohort plots.

Logic lives in ``src.ChangepointAnalysis``; this file only parses arguments.

Example
-------
python scripts/summarize_changepoint_results.py \\
    --changepoints-dir output/changepoints \\
    --features-dir output/gsa_features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis import summarize_changepoint_results
from src.utils.run_log import RunContext


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize changepoint CSV outputs and generate cohort plots."
    )
    parser.add_argument(
        "--changepoints-dir",
        default="output/changepoints",
        help="Directory with changepoint CSV outputs (default: output/changepoints)",
    )
    parser.add_argument(
        "--features-dir",
        default="output/gsa_features",
        help="Directory with *_gsa_features.csv files for timeline traces",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Plot output directory (default: {changepoints-dir}/plots)",
    )
    parser.add_argument(
        "--trajectories",
        nargs="*",
        default=None,
        help="Trajectory IDs for timeline plots (default: auto-pick 3 representative)",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Only regenerate plots, not summary CSVs",
    )
    parser.add_argument(
        "--skip-individual-timelines",
        action="store_true",
        help="Skip per-trajectory timeline_*.png plots",
    )
    parser.add_argument(
        "--cluster-representatives-csv",
        nargs="*",
        default=None,
        help=(
            "cluster_representatives.csv path(s) for medoid-segment timeline plots "
            "(default: auto-discover under {changepoints-dir}/clusters/**/)"
        ),
    )
    parser.add_argument(
        "--skip-cluster-rep-timelines",
        action="store_true",
        help="Skip timeline plots derived from cluster_representatives.csv",
    )
    parser.add_argument(
        "--features-suffix",
        default=None,
        help=(
            "Feature CSV suffix (e.g. _endpoint_features.csv). "
            "Auto-detected when omitted."
        ),
    )
    parser.add_argument(
        "--timeline-top-pairs",
        type=int,
        default=5,
        help=(
            "Top-N endpoint features for timeline_cluster panels "
            "(by segment-level η² vs cluster). 0 disables (default: 5)."
        ),
    )
    parser.add_argument(
        "--ranking-n-permutations",
        type=int,
        default=999,
        help="Label-shuffle permutations for η² p-values (0 disables; default: 999).",
    )
    parser.add_argument(
        "--ranking-fdr-alpha",
        type=float,
        default=0.05,
        help="BH-FDR alpha for significant_fdr on ranked pairs (default: 0.05).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    changepoints_dir = Path(args.changepoints_dir)
    features_dir = Path(args.features_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else changepoints_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    required = [
        changepoints_dir / "all_breakpoints.csv",
        changepoints_dir / "all_segment_stats.csv",
        changepoints_dir / "changepoint_timing_comparison.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing required changepoint outputs:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    with RunContext.from_namespace(args, name="summarize_changepoint_results"):
        summarize_changepoint_results(
            changepoints_dir,
            features_dir,
            plot_dir=plot_dir,
            trajectories=args.trajectories,
            skip_tables=args.skip_tables,
            skip_individual_timelines=args.skip_individual_timelines,
            cluster_representatives_csv=args.cluster_representatives_csv,
            skip_cluster_rep_timelines=args.skip_cluster_rep_timelines,
            features_suffix=args.features_suffix,
            cluster_timeline_top_pairs=args.timeline_top_pairs,
            ranking_n_permutations=args.ranking_n_permutations,
            ranking_fdr_alpha=args.ranking_fdr_alpha,
        )

    print("\nPlots:")
    for path in sorted(plot_dir.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
