"""G1a/G1b: own-elbow endpoint detections for BMHpH and BMMpM, then timing.

Their published sweeps (``penalty_sweep/``) built the penalty grid from the
first endpoint CSV, which in both cubes is the truncated ``109345_mdcrd_v``
run. The sweep now uses the cohort median length. Each cube is detected at its
own elbow (plan.md G1b) and timed against the matched GSA detections, like the
other four cubes (``recompute_matched_penalties.py``).

  output/endpoint_changepoints_{CUBE}/penalty_sweep_median_n/   (with --sweep)
  output/endpoint_changepoints_g1a_BMHpH/
  output/endpoint_changepoints_g1b_BMMpM/
  output/endpoint_vs_gsa_timing_g1a_BMHpH/
  output/endpoint_vs_gsa_timing_g1b_BMMpM_gsa_elbow/
  output/g1_own_elbow_summary.csv

Without --sweep the penalty is read from the existing penalty_sweep_median_n/.
Clustering is a separate step (scripts/cluster_changepoint_segments.py).

With --compare-to, every regenerated sweep table (with --sweep), detection
file (pooled and per trajectory) and timing table is compared with the copy
under that root and listed in ``g1_reproduction_check.csv`` as identical,
equal_within_tolerance (numbers within 1e-6 relative), differs or missing; the
exit code is 1 unless all match.

Example
-------
python scripts/recompute_own_elbow_g1.py
python scripts/recompute_own_elbow_g1.py --sweep --output-root output/_g1_check --compare-to output
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# One BLAS thread per Pelt process so the sweep workers don't oversubscribe.
_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.csv_compare import compare_csv_trees

PY = sys.executable
SWEEP = ROOT / "scripts" / "sweep_changepoint_penalty.py"
DETECT = ROOT / "scripts" / "changepoint_feature_groups.py"
COMPARE = ROOT / "scripts" / "compare_changepoint_timing.py"
SWEEP_SUBDIR = "penalty_sweep_median_n"
SWEEP_TABLES = (
    "penalty_elbow.csv",
    "penalty_sweep_summary.csv",
    "penalty_sweep_by_trajectory.csv",
    "penalty_stable_ranges.csv",
)
TIMING_TABLES = ("changepoint_timing_comparison.csv", "cohort_timing_summary.csv")

# cube, stage, detection folder, timing folder
G1_RUNS = (
    ("BMHpH", "g1a", "endpoint_changepoints_g1a_BMHpH", "endpoint_vs_gsa_timing_g1a_BMHpH"),
    (
        "BMMpM",
        "g1b",
        "endpoint_changepoints_g1b_BMMpM",
        "endpoint_vs_gsa_timing_g1b_BMMpM_gsa_elbow",
    ),
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input-root",
        type=Path,
        default=ROOT / "output",
        help="Folder holding endpoint_changepoints_{CUBE}/endpoint_features and "
        "gsa_changepoints_matched_{CUBE} (default: output)",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output",
        help="Folder for sweeps, detections, timing and the summary (default: output)",
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Re-run the median-length penalty sweep before detecting",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        help="Sweep workers per cube (default: 8)",
    )
    p.add_argument(
        "--compare-to",
        type=Path,
        help="Root holding the reported G1 outputs; check the regenerated files against it",
    )
    return p.parse_args()


def read_recommendation(sweep_dir: Path) -> tuple[float, float]:
    """Return (elbow penalty, log(n) reference) from penalty_recommendation.txt."""
    penalty = reference = None
    text = (sweep_dir / "penalty_recommendation.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("Reference penalty"):
            reference = float(line.split(":")[-1])
        elif "Recommended penalty (fallback)" in line:
            raise RuntimeError(f"No penalty elbow in {sweep_dir}")
        elif "Recommended penalty" in line:
            penalty = float(line.split(":")[-1])
    if penalty is None or reference is None:
        raise RuntimeError(f"No recommendation in {sweep_dir}")
    return penalty, reference


def run_cmd(args: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {log_path.stem}", flush=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(" ".join(args) + "\n\n")
        fh.flush()
        env = os.environ.copy()
        env.update(_THREAD_ENV)
        proc = subprocess.run(args, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)
    print(f"DONE  {log_path.stem} rc={proc.returncode} {time.perf_counter() - t0:.1f}s", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{log_path.stem} failed (rc={proc.returncode}); see {log_path}")


def summary_row(
    cube: str, stage: str, penalty: float, reference: float, det_dir: Path, timing_dir: Path
) -> dict:
    seg = pd.read_csv(det_dir / "all_segment_stats.csv", usecols=["traj_id", "group"])
    trajs = set(seg.loc[seg["group"] == "endpoint", "traj_id"].astype(str))
    bkps = pd.read_csv(det_dir / "all_breakpoints.csv", usecols=["traj_id", "group"])
    bkps = bkps[bkps["group"] == "endpoint"]
    row = {
        "cube": cube,
        "stage": stage,
        "log_median_n": reference,
        "endpoint_penalty": penalty,
        "rel_to_log_median_n": penalty / reference,
        "n_traj": len(trajs),
        "bkps": len(bkps),
        "bkps_per_traj": len(bkps) / len(trajs),
        "silent_traj": len(trajs - set(bkps["traj_id"].astype(str))),
    }
    timing = pd.read_csv(timing_dir / "cohort_timing_summary.csv")
    for other in ("gsa", "iodine"):
        hit = timing[(timing["group_a"] == "endpoint") & (timing["group_b"] == other)]
        if hit.empty:
            continue
        row[f"n_traj_endpoint_{other}"] = int(hit["n_trajectories"].iloc[0])
        row[f"median_jaccard_endpoint_{other}"] = float(hit["median_jaccard"].iloc[0])
        row[f"median_offset_ps_endpoint_{other}"] = float(hit["median_offset_ps"].iloc[0])
    return row


def run_cube(cube: str, stage: str, det_name: str, timing_name: str, args: argparse.Namespace) -> dict:
    features = args.input_root / f"endpoint_changepoints_{cube}" / "endpoint_features"
    gsa_dir = args.input_root / f"gsa_changepoints_matched_{cube}"
    sweep_root = args.output_root if args.sweep else args.input_root
    sweep_dir = sweep_root / f"endpoint_changepoints_{cube}" / SWEEP_SUBDIR
    det_dir = args.output_root / det_name
    timing_dir = args.output_root / timing_name
    log_dir = args.output_root / "g1_own_elbow_logs"
    for path in (features, gsa_dir):
        if not path.is_dir():
            raise FileNotFoundError(path)

    if args.sweep:
        run_cmd(
            [
                PY,
                str(SWEEP),
                "--input-dir",
                str(features),
                "--suffix",
                "_endpoint_features.csv",
                "--output-dir",
                str(sweep_dir),
                "--groups",
                "endpoint",
                "--n-jobs",
                str(args.n_jobs),
                "--no-plots",
            ],
            log_dir / f"sweep_{stage}_{cube}.log",
        )
    penalty, reference = read_recommendation(sweep_dir)
    print(f"  {cube}: elbow {penalty:.6g} = {penalty / reference:.3f} x log(median n)", flush=True)

    run_cmd(
        [
            PY,
            str(DETECT),
            "--input-dir",
            str(features),
            "--suffix",
            "_endpoint_features.csv",
            "--output-dir",
            str(det_dir),
            "--groups",
            "endpoint",
            "--penalty",
            f"{penalty:.6g}",
        ],
        log_dir / f"detect_{stage}_{cube}.log",
    )
    run_cmd(
        [
            PY,
            str(COMPARE),
            "--changepoints-dirs",
            str(det_dir),
            str(gsa_dir),
            "--output-dir",
            str(timing_dir),
        ],
        log_dir / f"timing_{stage}_{cube}.log",
    )
    return summary_row(cube, stage, penalty, reference, det_dir, timing_dir)


def reproduction_files(reference_root: Path, *, sweep: bool) -> list[Path]:
    """Relative paths of the G1 outputs to check, per-trajectory files listed from the reference."""
    rel: list[Path] = []
    for cube, _, det_name, timing_name in G1_RUNS:
        if sweep:
            rel += [Path(f"endpoint_changepoints_{cube}") / SWEEP_SUBDIR / name for name in SWEEP_TABLES]
        det = reference_root / det_name
        rel += [Path(det_name) / "all_breakpoints.csv", Path(det_name) / "all_segment_stats.csv"]
        for pattern in ("*_breakpoints.csv", "*_segment_stats.csv"):
            rel += [Path(det_name) / p.name for p in sorted(det.glob(pattern)) if not p.name.startswith("all_")]
        rel += [Path(timing_name) / name for name in TIMING_TABLES]
    return rel


def check_reproduction(args: argparse.Namespace) -> bool:
    check = compare_csv_trees(args.compare_to, args.output_root, reproduction_files(args.compare_to, sweep=args.sweep))
    path = args.output_root / "g1_reproduction_check.csv"
    check.to_csv(path, index=False)
    print(f"\nReproduction check vs {args.compare_to}: {path}", flush=True)
    print(check["status"].value_counts().to_string(), flush=True)
    inexact = check[check["status"] != "identical"]
    if not inexact.empty:
        print(inexact.to_string(index=False), flush=True)
    if args.sweep:
        for cube, *_ in G1_RUNS:
            for root in (args.compare_to, args.output_root):
                penalty, _ = read_recommendation(root / f"endpoint_changepoints_{cube}" / SWEEP_SUBDIR)
                print(f"  {cube} recommended penalty in {root}: {penalty:.6g}", flush=True)
    return bool(check["status"].isin(["identical", "equal_within_tolerance"]).all())


def main() -> int:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(G1_RUNS)) as pool:
        futures = [pool.submit(run_cube, *run, args) for run in G1_RUNS]
        rows = [fut.result() for fut in futures]
    summary = args.output_root / "g1_own_elbow_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    print(f"\nSummary: {summary}", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    if args.compare_to is not None and not check_reproduction(args):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
