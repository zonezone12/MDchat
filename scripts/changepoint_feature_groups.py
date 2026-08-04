"""
Changepoint detection on four feature groups from pre-computed GSA feature CSVs.

Logic lives in ``src.ChangepointAnalysis``; this file only parses arguments.

Groups:
  gsa       — nanocube cage geometry and monomer structure
  iodine    — iodine (IOD) guest dynamics and contacts
  na_water  — cavity Na+/water solvent environment
  combined  — all numeric features

Example
-------
python scripts/changepoint_feature_groups.py \\
    --input-dir output/gsa_features \\
    --output-dir output/changepoints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis import ChangepointConfig, detect_cohort_changepoints
from src.ChangepointAnalysis.detection import discover_feature_csvs
from src.utils.run_log import RunContext, log_event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Changepoint detection on GSA feature groups from pre-computed CSVs."
    )
    parser.add_argument(
        "--input-dir",
        default="output/gsa_features",
        help="Directory containing *_gsa_features.csv files (default: output/gsa_features)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/changepoints",
        help="Output directory for all result CSVs (default: output/changepoints)",
    )
    parser.add_argument(
        "--method",
        default="Pelt",
        help="Ruptures search method: Pelt (default), Binseg, BottomUp, Window, Dynp",
    )
    parser.add_argument(
        "--cost-model",
        default="rbf",
        help="Ruptures cost function: rbf (default), l1, l2, normal, ar, linear, rank",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=None,
        help="Penalty value for Pelt (omit to use log(n)*var heuristic)",
    )
    parser.add_argument(
        "--n-bkps",
        type=int,
        default=None,
        help="Fixed number of breakpoints; requires --method Binseg or Dynp",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=10,
        help="Minimum segment length in frames (default: 10)",
    )
    parser.add_argument(
        "--jump",
        type=int,
        default=5,
        help="Ruptures subsample step for speed (default: 5)",
    )
    parser.add_argument(
        "--tolerance-frames",
        type=int,
        default=50,
        help="Frame tolerance for timing comparison; breakpoints within ±N are 'shared' (default: 50)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["gsa", "iodine", "na_water", "combined"],
        choices=["gsa", "iodine", "na_water", "combined"],
        help="Feature groups to analyse (default: all four)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip z-score normalisation before changepoint detection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = discover_feature_csvs(input_dir)
    if not csv_files:
        print(f"No *_gsa_features.csv files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    config = ChangepointConfig(
        method=args.method,
        cost_model=args.cost_model,
        penalty=args.penalty,
        n_bkps=args.n_bkps,
        min_size=args.min_size,
        jump=args.jump,
        tolerance_frames=args.tolerance_frames,
        normalize=not args.no_normalize,
        groups=tuple(args.groups),
    )

    with RunContext.from_namespace(args, name="changepoint_feature_groups"):
        log_event(
            "info",
            f"Found {len(csv_files)} feature CSV(s); groups={args.groups}; "
            f"method={args.method}; cost={args.cost_model}",
            component="changepoint_feature_groups",
        )
        detect_cohort_changepoints(csv_files, config, output_dir=output_dir)

    print("\nOutput files:")
    for f in sorted(output_dir.glob("*.csv")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
