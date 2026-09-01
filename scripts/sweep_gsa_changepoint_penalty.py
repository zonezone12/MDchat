"""
Per-cube GSA penalty sweeps for ``gsa_changepoints_B*``.

Reads staged ``output/gsa_features_by_cube/{CUBE}`` CSVs (falls back to
discovering mixed ``gsa_features_step1``) and writes
``output/gsa_changepoints_{CUBE}/penalty_sweep/``. Does **not** re-run
final detection — existing breakpoint tables are left in place.

Each cube is a separate subprocess so trajectory-level ``--n-jobs`` pools
do not nest.

Example
-------
python scripts/sweep_gsa_changepoint_penalty.py --cube-jobs 6 --n-jobs 4
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.gsa_cohort_run import (
    KNOWN_CUBES,
    discover_gsa_feature_csvs_by_cube,
    gsa_changepoints_dir,
)

SWEEP_SCRIPT = ROOT / "scripts" / "sweep_changepoint_penalty.py"
LOGN_5000 = float(math.log(5000))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run per-cube GSA Pelt penalty sweeps (no final re-detect)."
    )
    p.add_argument(
        "--features-root",
        type=Path,
        default=Path("output/gsa_features_by_cube"),
        help="Per-cube feature dirs, or mixed gsa_features_step1 (default: by_cube)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Where gsa_changepoints_{CUBE}/penalty_sweep is written",
    )
    p.add_argument(
        "--cubes",
        nargs="*",
        default=None,
        choices=list(KNOWN_CUBES),
        help="Subset of cubes (default: all discovered B* prefixes)",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=["gsa", "iodine"],
        help="Feature groups to sweep (default: gsa iodine). "
        "na_water rbf Pelt is ~7× slower; add it explicitly if needed.",
    )
    p.add_argument("--n-penalties", type=int, default=15)
    p.add_argument(
        "--penalty-min",
        type=float,
        default=round(0.2 * LOGN_5000, 6),
        help="Sweep lower bound (default: 0.2 × log(5000))",
    )
    p.add_argument(
        "--penalty-max",
        type=float,
        default=round(2.5 * LOGN_5000, 6),
        help=(
            "Sweep upper bound (default: 2.5 × log(5000)). "
            "Capped below 5× because rbf Pelt on 5000-frame GSA signals "
            "slows ~40× at 5× log(n)."
        ),
    )
    p.add_argument(
        "--reference-penalty",
        type=float,
        default=round(LOGN_5000, 6),
        help="Regime-direction reference (default: log(5000))",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Trajectory workers inside each cube (default: 4)",
    )
    p.add_argument(
        "--cube-jobs",
        type=int,
        default=6,
        help="How many cube sweeps to run at once (default: 6)",
    )
    p.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Limit trajectories per cube (smoke / debug)",
    )
    return p.parse_args()


def _feature_dir_for_cube(features_root: Path, cube: str, csvs: list[Path]) -> Path:
    staged = features_root / cube
    if staged.is_dir() and any(staged.glob("*_gsa_features.csv")):
        return staged
    return csvs[0].parent


def _cmd_for_cube(
    cube: str,
    features_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    del cube
    cmd = [
        sys.executable,
        str(SWEEP_SCRIPT),
        "--input-dir",
        str(features_dir),
        "--output-dir",
        str(output_dir),
        "--n-penalties",
        str(args.n_penalties),
        "--n-jobs",
        str(args.n_jobs),
        "--groups",
        *list(args.groups),
        "--penalty-min",
        str(args.penalty_min),
        "--penalty-max",
        str(args.penalty_max),
        "--reference-penalty",
        str(args.reference_penalty),
    ]
    if args.max_trajectories is not None:
        cmd.extend(["--max-trajectories", str(args.max_trajectories)])
    return cmd


def main() -> None:
    args = parse_args()
    features_root = Path(args.features_root)
    output_root = Path(args.output_root)
    by_cube = discover_gsa_feature_csvs_by_cube(features_root, cubes=args.cubes)
    if not by_cube:
        print(
            f"No *_gsa_features.csv matching cubes in {features_root}",
            file=sys.stderr,
        )
        sys.exit(1)

    jobs: list[tuple[str, list[str], Path]] = []
    for cube, csvs in by_cube.items():
        feat_dir = _feature_dir_for_cube(features_root, cube, csvs)
        out_dir = gsa_changepoints_dir(output_root, cube) / "penalty_sweep"
        out_dir.mkdir(parents=True, exist_ok=True)
        jobs.append((cube, _cmd_for_cube(cube, feat_dir, out_dir, args), out_dir))

    cube_jobs = max(1, min(int(args.cube_jobs), len(jobs)))
    print(
        f"Launching {len(jobs)} cube sweep(s), {cube_jobs} at a time, "
        f"n_jobs={args.n_jobs}"
    )

    pending = list(jobs)
    running: list[tuple[str, subprocess.Popen, object, Path]] = []
    failures: list[str] = []

    def launch_one() -> None:
        cube, cmd, out_dir = pending.pop(0)
        log_path = out_dir / "sweep.log"
        handle = log_path.open("w", encoding="utf-8")
        print(f"{cube}: starting -> {out_dir}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        running.append((cube, proc, handle, out_dir))

    def reap_finished() -> None:
        still: list[tuple[str, subprocess.Popen, object, Path]] = []
        for cube, proc, handle, out_dir in running:
            ret = proc.poll()
            if ret is None:
                still.append((cube, proc, handle, out_dir))
                continue
            handle.close()
            if ret != 0:
                failures.append(cube)
                print(
                    f"{cube}: FAILED exit={ret}  log={out_dir / 'sweep.log'}",
                    file=sys.stderr,
                )
            else:
                print(f"{cube}: done  {out_dir}")
        running[:] = still

    while pending or running:
        while pending and len(running) < cube_jobs:
            launch_one()
        reap_finished()
        if running:
            time.sleep(5)

    if failures:
        print(f"Failed cubes: {', '.join(failures)}", file=sys.stderr)
        sys.exit(1)
    print("All GSA penalty sweeps finished.")


if __name__ == "__main__":
    main()
