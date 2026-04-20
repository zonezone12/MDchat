#!/usr/bin/env python3
"""
CLI: rescore simulations from existing ``*_simulation_score.csv`` rows and sidecar CSVs.

Core logic: ``src.FrameSelection.simulation_scores``.

Usage:
    python scripts/rescore_simulations.py --input_dir output/ --output output/rescored_scores.csv
    python scripts/rescore_simulations.py --csv_files output/traj1_simulation_score.csv output/traj2_simulation_score.csv
    python scripts/rescore_simulations.py --input_dir output/ --overwrite
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.FrameSelection.simulation_scores import (
    find_simulation_score_files,
    overwrite_simulation_score_sources,
    rescore_simulation_csv_files,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rescore MD simulations from CSV artifacts using the current scoring function.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/rescore_simulations.py --input_dir output/
  python scripts/rescore_simulations.py --input_dir output/ --output output/rescored_scores.csv
  python scripts/rescore_simulations.py --csv_files output/a_simulation_score.csv output/b_simulation_score.csv
  python scripts/rescore_simulations.py --input_dir output/ --overwrite
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
        help="Glob pattern under --input_dir (default: *_simulation_score.csv)",
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Search only the top level of --input_dir",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV for rescoring summary (default: rescored_simulation_scores.csv)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original *_simulation_score.csv files with new scores",
    )

    parser.add_argument(
        "--min_volume_change_pct",
        type=float,
        default=10.0,
        help="Passed to FrameSelection.score_simulation (default: 10.0)",
    )
    parser.add_argument(
        "--min_correlation",
        type=float,
        default=0.5,
        help="Passed to FrameSelection.score_simulation (default: 0.5)",
    )
    parser.add_argument(
        "--require_guest_entry",
        action="store_true",
        help="Require guest entry for a valid simulation",
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
        return 1

    print("\nFiles to rescore:")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f}")
    print()

    all_results: list = []

    for csv_file in csv_files:
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(csv_file)}")
        print(f"{'='*80}")

        try:
            batch = rescore_simulation_csv_files(
                [csv_file],
                min_guest_entry=args.require_guest_entry,
                min_volume_change_pct=args.min_volume_change_pct,
                min_correlation=args.min_correlation,
            )

            if len(batch) == 0:
                print("  Warning: Empty CSV file, skipping")
                continue

            for _, result in batch.iterrows():
                r = result.to_dict()
                all_results.append(r)
                if r.get("success"):
                    print(f"  Trajectory: {r['trajectory_id']}")
                    print(f"    Old score: {r['old_overall_score']:.4f}")
                    print(f"    New score: {r['new_overall_score']:.4f}")
                    print(f"    Change:    {r['score_change']:+.4f}")
                else:
                    print(f"  Trajectory: {r['trajectory_id']}")
                    print(f"    Error: {r.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"  ERROR: Failed to process {csv_file}: {e}")
            import traceback

            traceback.print_exc()

    if not all_results:
        print("\nNo results to save.")
        return 1

    results_df = pd.DataFrame(all_results)

    if args.overwrite:
        print(f"\n{'='*80}")
        print("Updating original CSV files...")
        print(f"{'='*80}")
        overwrite_simulation_score_sources(results_df, csv_files)

    output_path = args.output if args.output else "rescored_simulation_scores.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print("Rescoring complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")

    successful = results_df[results_df["success"] == True]
    if len(successful) > 0:
        print("Summary:")
        print(f"  Total processed: {len(results_df)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(results_df) - len(successful)}")
        print("\nScore statistics:")
        print(f"  Old score mean: {successful['old_overall_score'].mean():.4f}")
        print(f"  New score mean: {successful['new_overall_score'].mean():.4f}")
        print(f"  Mean change: {successful['score_change'].mean():+.4f}")
        print(
            "  Score range (old): "
            f"{successful['old_overall_score'].min():.4f} - {successful['old_overall_score'].max():.4f}"
        )
        print(
            "  Score range (new): "
            f"{successful['new_overall_score'].min():.4f} - {successful['new_overall_score'].max():.4f}"
        )
        print(f"  Score std (old): {successful['old_overall_score'].std():.4f}")
        print(f"  Score std (new): {successful['new_overall_score'].std():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
