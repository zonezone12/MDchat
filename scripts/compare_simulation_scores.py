#!/usr/bin/env python3
"""
CLI: load and compare simulation scores from multiple MD trajectories.

Logic lives in ``src.FrameSelection.simulation_scores``; this file only parses arguments.

Usage:
    python scripts/compare_simulation_scores.py --input_dir output/ --output output/all_simulations_ranked.csv
    python scripts/compare_simulation_scores.py --input_dir output/ --top_n 10 --sort_by volume_dynamics_score
    python scripts/compare_simulation_scores.py --csv_files output/traj1_simulation_score.csv output/traj2_simulation_score.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.FrameSelection.simulation_scores import (
    display_summary_statistics,
    display_top_simulations,
    find_simulation_score_files,
    load_and_compare_simulation_scores,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare simulation scores from multiple MD trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/compare_simulation_scores.py --input_dir output/
  python scripts/compare_simulation_scores.py --input_dir output/ --output ranked.csv --top_n 20
  python scripts/compare_simulation_scores.py --input_dir output/ --sort_by volume_dynamics_score
  python scripts/compare_simulation_scores.py --csv_files output/a_simulation_score.csv output/b_simulation_score.csv
        """,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing *_simulation_score.csv files",
    )
    input_group.add_argument(
        "--csv_files",
        nargs="+",
        help="Explicit list of simulation score CSV paths",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*_simulation_score.csv",
        help="Glob pattern for files under --input_dir (default: *_simulation_score.csv)",
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Search only the top level of --input_dir",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV for ranked results (default: all_simulations_ranked.csv)",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="overall_score",
        help="Column to sort by (default: overall_score)",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending (default: descending)",
    )

    parser.add_argument("--top_n", type=int, default=10, help="Rows to print (default: 10)")
    parser.add_argument("--no_summary", action="store_true", help="Skip summary block")
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Columns to show in the top-N table (default: a fixed set of key columns)",
    )

    args = parser.parse_args()

    if args.input_dir:
        print(f"Searching for simulation score files in: {args.input_dir}")
        csv_files = find_simulation_score_files(
            args.input_dir,
            pattern=args.pattern,
            recursive=not args.no_recursive,
        )
        print(f"Found {len(csv_files)} simulation score file(s)")
    else:
        csv_files = args.csv_files
        print(f"Using {len(csv_files)} specified CSV file(s)")

    if not csv_files:
        print("ERROR: No simulation score CSV files found.")
        print("Generate scores first (e.g. scripts/run_volume_endpoint_guest_analysis.py).")
        return 1

    print("\nFiles to compare:")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f}")
    print()

    output_path = args.output if args.output else "all_simulations_ranked.csv"

    try:
        print("Loading and comparing simulations...")
        print(f"Sorting by: {args.sort_by}")
        print(f"Order: {'ascending' if args.ascending else 'descending'}")

        comparison_df = load_and_compare_simulation_scores(
            csv_paths=csv_files,
            output_path=output_path,
            sort_by=args.sort_by,
            ascending=args.ascending,
        )

        print(f"\nSuccessfully loaded {len(comparison_df)} simulation(s)")
        print(f"Ranked results saved to: {output_path}")

        if not args.no_summary:
            display_summary_statistics(comparison_df)

        display_top_simulations(
            comparison_df,
            top_n=args.top_n,
            columns=args.columns,
        )

        print(f"\nFull ranked results: {output_path}\n")
        return 0

    except Exception as e:
        print(f"\nERROR: Failed to compare simulation scores: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
