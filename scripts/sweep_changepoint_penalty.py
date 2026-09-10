"""
Penalty sensitivity sweep for changepoint_feature_groups.

Logic lives in ``src.ChangepointAnalysis``; this file only parses arguments.

Chooses a penalty by structural stability, not a single heuristic value.
For each penalty in a grid, scores breakpoint-count plateaus, timing stability,
cross-group Jaccard, and segment-regime consistency, then recommends the elbow.

Example
-------
python scripts/sweep_changepoint_penalty.py \\
    --input-dir output/gsa_features \\
    --output-dir output/changepoints/penalty_sweep \\
    --n-penalties 17

python scripts/sweep_changepoint_penalty.py \\
    --input-dir output/gsa_features \\
    --output-dir output/changepoints/penalty_sweep \\
    --method Pelt --cost-model rbf --min-size 10 --jump 5 --tolerance-frames 50 \\
    --run-final --final-output-dir output/changepoints
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis import ChangepointConfig, PenaltySweepConfig
from src.ChangepointAnalysis.detection import discover_feature_csvs
from src.ChangepointAnalysis.feature_groups import DEFAULT_GROUPS
from src.ChangepointAnalysis.penalty_sweep import run_final_detection, sweep_penalties
from src.utils.run_log import RunContext

DEFAULT_METHOD = "Pelt"
DEFAULT_COST_MODEL = "rbf"
DEFAULT_MIN_SIZE = 10
DEFAULT_JUMP = 5
DEFAULT_TOLERANCE_FRAMES = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep Pelt penalty values and recommend a structurally stable range "
            "for changepoint_feature_groups."
        )
    )
    p.add_argument(
        "--input-dir",
        default="output/gsa_features",
        help="Directory with feature CSVs",
    )
    p.add_argument(
        "--suffix",
        default="_gsa_features.csv",
        help="Feature-CSV suffix (default: _gsa_features.csv; use _endpoint_features.csv for endpoint)",
    )
    p.add_argument(
        "--output-dir",
        default="output/changepoints/penalty_sweep",
        help="Sweep summaries and plots (default: output/changepoints/penalty_sweep)",
    )
    p.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help="Ruptures search method (default: Pelt)",
    )
    p.add_argument(
        "--cost-model",
        default=DEFAULT_COST_MODEL,
        help="Ruptures cost function (default: rbf)",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=DEFAULT_MIN_SIZE,
        help="Minimum segment length in frames (default: 10)",
    )
    p.add_argument(
        "--jump",
        type=int,
        default=DEFAULT_JUMP,
        help="Ruptures subsample step for speed (default: 5)",
    )
    p.add_argument(
        "--tolerance-frames",
        type=int,
        default=DEFAULT_TOLERANCE_FRAMES,
        help=(
            "Frame tolerance for timing comparison; breakpoints within ±N are "
            "'shared' (default: 50)"
        ),
    )
    p.add_argument(
        "--n-penalties",
        type=int,
        default=15,
        help="Number of log-spaced penalty values (default: 15)",
    )
    p.add_argument(
        "--penalty-min",
        type=float,
        default=None,
        help="Override sweep lower bound (default: ~0.2 × log(n))",
    )
    p.add_argument(
        "--penalty-max",
        type=float,
        default=None,
        help="Override sweep upper bound (default: ~5 × log(n))",
    )
    p.add_argument(
        "--reference-penalty",
        type=float,
        default=None,
        help="Penalty for regime-direction reference (default: log(n) auto)",
    )
    p.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Limit trajectories for a faster exploratory sweep",
    )
    p.add_argument(
        "--timing-threshold",
        type=float,
        default=0.75,
        help="Min timing Jaccard vs previous step inside a stable range (default: 0.75)",
    )
    p.add_argument(
        "--count-change-threshold",
        type=float,
        default=0.35,
        help=(
            "Max step-to-step relative change in cohort breakpoint totals "
            "inside a stable range (default: 0.35)"
        ),
    )
    p.add_argument(
        "--regime-threshold",
        type=float,
        default=0.85,
        help=(
            "Regime-direction agreement threshold (reported as passes_regime_threshold; "
            "default: 0.85)"
        ),
    )
    p.add_argument(
        "--min-plateau-steps",
        type=int,
        default=3,
        help="Minimum consecutive structurally stable penalty steps (default: 3)",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=None,
        help="Feature groups to sweep (default: gsa iodine na_water combined)",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers across trajectories per penalty (default: 1; -1 = all CPUs)",
    )
    p.add_argument(
        "--run-final",
        action="store_true",
        help="After sweep, run detection at recommended penalty (in-process)",
    )
    p.add_argument(
        "--final-output-dir",
        default="output/changepoints",
        help="Output dir for --run-final (default: output/changepoints)",
    )
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = discover_feature_csvs(input_dir, suffix=args.suffix)
    if not csv_files:
        print(f"No *{args.suffix} in {input_dir}", file=sys.stderr)
        sys.exit(1)

    detection = ChangepointConfig(
        method=args.method,
        cost_model=args.cost_model,
        min_size=args.min_size,
        jump=args.jump,
        tolerance_frames=args.tolerance_frames,
        normalize=True,
        groups=tuple(args.groups) if args.groups else DEFAULT_GROUPS,
    )
    sweep_cfg = PenaltySweepConfig(
        n_penalties=args.n_penalties,
        penalty_min=args.penalty_min,
        penalty_max=args.penalty_max,
        reference_penalty=args.reference_penalty,
        max_trajectories=args.max_trajectories,
        timing_threshold=args.timing_threshold,
        count_change_threshold=args.count_change_threshold,
        regime_threshold=args.regime_threshold,
        min_plateau_steps=args.min_plateau_steps,
        no_plots=args.no_plots,
        n_jobs=args.n_jobs,
        detection=detection,
    )

    with RunContext.from_namespace(args, name="sweep_changepoint_penalty"):
        if args.max_trajectories is not None:
            csv_files = csv_files[: args.max_trajectories]
        result = sweep_penalties(csv_files, sweep_cfg, output_dir=output_dir)

        if args.run_final and result.recommended_penalty is not None:
            run_final_detection(
                csv_files,
                penalty=result.recommended_penalty,
                detection=detection,
                output_dir=Path(args.final_output_dir),
            )

    rec_penalty = result.recommended_penalty
    rec_range = result.recommended_range
    print(f"\nRecommended penalty (elbow): {rec_penalty:.6g}")
    if rec_range is not None:
        print(f"Stable range: [{rec_range[0]:.6g}, {rec_range[1]:.6g}]")
    print(f"\nOutputs in {output_dir}:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  {f}")


if __name__ == "__main__":
    main()
