"""G1: re-detect at sweep-elbow penalties into matched dirs, then recompute timing.

Penalties come from each cube's penalty_sweep/penalty_recommendation.txt
(authoritative vs docs/penalty_sweep_B_cohorts.md). Detection only — no
re-clustering. GSA uses the swept groups (gsa + iodine) at the per-cube
elbow; na_water / combined were not part of the GSA elbow fit.

  output/endpoint_changepoints_matched_{CUBE}/
  output/gsa_changepoints_matched_{CUBE}/
  output/endpoint_vs_gsa_timing_matched_{CUBE}/
  output/matched_penalty_recompute_summary.csv
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# One BLAS thread per Pelt process so six cubes don't oversubscribe.
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

from src.ChangepointAnalysis.penalty_sweep import find_penalty_elbow

CUBES = ("BHHpH", "BHHpM", "BMHpH", "BMHpM", "BMMpH", "BMMpM")
PY = sys.executable
DETECT = ROOT / "scripts" / "changepoint_feature_groups.py"
COMPARE = ROOT / "scripts" / "compare_changepoint_timing.py"
OUT = ROOT / "output"
LOG_DIR = OUT / "matched_penalty_recompute_logs"


def elbow_penalty(sweep_dir: Path) -> float:
    rec = sweep_dir / "penalty_recommendation.txt"
    if rec.is_file():
        for line in rec.read_text(encoding="utf-8").splitlines():
            if "Recommended penalty" in line:
                return float(line.split(":")[-1].strip())
    df = pd.read_csv(sweep_dir / "penalty_sweep_summary.csv").sort_values("penalty")
    elb = find_penalty_elbow(df)
    if not elb:
        raise RuntimeError(f"No elbow in {sweep_dir}")
    return float(elb["elbow_penalty"])


def run_cmd(args: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"START {log_path.stem}", flush=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(" ".join(args) + "\n\n")
        fh.flush()
        env = os.environ.copy()
        env.update(_THREAD_ENV)
        proc = subprocess.run(
            args,
            cwd=ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
    dt = time.perf_counter() - t0
    print(f"DONE  {log_path.stem} rc={proc.returncode} {dt:.1f}s", flush=True)
    if proc.returncode != 0:
        print(f"  LOG {log_path}", flush=True)
    return proc.returncode


def detect_jobs() -> list[tuple[list[str], Path]]:
    jobs: list[tuple[list[str], Path]] = []
    for cube in CUBES:
        ep = elbow_penalty(OUT / f"endpoint_changepoints_{cube}" / "penalty_sweep")
        jobs.append(
            (
                [
                    PY,
                    str(DETECT),
                    "--input-dir",
                    str(OUT / f"endpoint_changepoints_{cube}" / "endpoint_features"),
                    "--suffix",
                    "_endpoint_features.csv",
                    "--output-dir",
                    str(OUT / f"endpoint_changepoints_matched_{cube}"),
                    "--groups",
                    "endpoint",
                    "--penalty",
                    f"{ep:.6g}",
                ],
                LOG_DIR / f"detect_endpoint_{cube}.log",
            )
        )
        gp = elbow_penalty(OUT / f"gsa_changepoints_{cube}" / "penalty_sweep")
        jobs.append(
            (
                [
                    PY,
                    str(DETECT),
                    "--input-dir",
                    str(OUT / "gsa_features_by_cube" / cube),
                    "--suffix",
                    "_gsa_features.csv",
                    "--output-dir",
                    str(OUT / f"gsa_changepoints_matched_{cube}"),
                    "--groups",
                    "gsa",
                    "iodine",
                    "--penalty",
                    f"{gp:.6g}",
                ],
                LOG_DIR / f"detect_gsa_{cube}.log",
            )
        )
    return jobs


def timing_jobs() -> list[tuple[list[str], Path]]:
    jobs: list[tuple[list[str], Path]] = []
    for cube in CUBES:
        jobs.append(
            (
                [
                    PY,
                    str(COMPARE),
                    "--changepoints-dirs",
                    str(OUT / f"endpoint_changepoints_matched_{cube}"),
                    str(OUT / f"gsa_changepoints_matched_{cube}"),
                    "--output-dir",
                    str(OUT / f"endpoint_vs_gsa_timing_matched_{cube}"),
                ],
                LOG_DIR / f"timing_{cube}.log",
            )
        )
    return jobs


def _n_traj_from_segments(changepoints_dir: Path, group: str) -> int:
    seg = changepoints_dir / "all_segment_stats.csv"
    if not seg.is_file():
        return 0
    df = pd.read_csv(seg, usecols=["traj_id", "group"])
    sub = df[df["group"].astype(str) == group]
    return int(sub["traj_id"].nunique()) if not sub.empty else 0


def _bkps_per_traj(changepoints_dir: Path, group: str) -> tuple[int, int, float]:
    bkp_path = changepoints_dir / "all_breakpoints.csv"
    if not bkp_path.is_file():
        return 0, 0, float("nan")
    df = pd.read_csv(bkp_path)
    sub = df[df["group"].astype(str) == group]
    n_bkps = len(sub)
    n_traj = _n_traj_from_segments(changepoints_dir, group)
    if n_traj == 0:
        n_traj = int(sub["traj_id"].nunique()) if not sub.empty else 0
    if n_traj == 0:
        return n_bkps, 0, float("nan")
    return n_bkps, n_traj, n_bkps / n_traj


def write_summary() -> Path:
    rows = []
    for cube in CUBES:
        ep_pen = elbow_penalty(OUT / f"endpoint_changepoints_{cube}" / "penalty_sweep")
        gp_pen = elbow_penalty(OUT / f"gsa_changepoints_{cube}" / "penalty_sweep")
        ep_new = OUT / f"endpoint_changepoints_matched_{cube}" / "all_breakpoints.csv"
        gp_new = OUT / f"gsa_changepoints_matched_{cube}" / "all_breakpoints.csv"
        ep_old = OUT / f"endpoint_changepoints_{cube}" / "all_breakpoints.csv"
        gp_old = OUT / f"gsa_changepoints_{cube}" / "all_breakpoints.csv"
        timing = OUT / f"endpoint_vs_gsa_timing_matched_{cube}" / "cohort_timing_summary.csv"

        e_bkps, e_traj, e_rate = _bkps_per_traj(ep_new.parent, "endpoint")
        e_old_bkps, e_old_traj, e_old_rate = _bkps_per_traj(ep_old.parent, "endpoint")
        g_bkps, g_traj, g_rate = _bkps_per_traj(gp_new.parent, "gsa")
        g_old_bkps, g_old_traj, g_old_rate = _bkps_per_traj(gp_old.parent, "gsa")

        j_ep_gsa = j_comb_ep = float("nan")
        n_pairs = 0
        if timing.is_file():
            tdf = pd.read_csv(timing)
            n_pairs = int(tdf["n_trajectories"].sum()) if "n_trajectories" in tdf.columns else len(tdf)

            def _med(a: str, b: str) -> float:
                col = "median_jaccard" if "median_jaccard" in tdf.columns else "jaccard"
                m = (
                    ((tdf["group_a"] == a) & (tdf["group_b"] == b))
                    | ((tdf["group_a"] == b) & (tdf["group_b"] == a))
                )
                if not m.any():
                    return float("nan")
                return float(tdf.loc[m, col].iloc[0])

            j_ep_gsa = _med("endpoint", "gsa")
            j_comb_ep = _med("combined", "endpoint")

        rows.append(
            {
                "cube": cube,
                "endpoint_penalty": ep_pen,
                "gsa_penalty": gp_pen,
                "endpoint_n_traj_matched": e_traj,
                "endpoint_bkps_matched": e_bkps,
                "endpoint_bkps_per_traj_matched": e_rate,
                "endpoint_n_traj_published": e_old_traj,
                "endpoint_bkps_published": e_old_bkps,
                "endpoint_bkps_per_traj_published": e_old_rate,
                "gsa_n_traj_matched": g_traj,
                "gsa_bkps_matched": g_bkps,
                "gsa_bkps_per_traj_matched": g_rate,
                "gsa_n_traj_published": g_old_traj,
                "gsa_bkps_published": g_old_bkps,
                "gsa_bkps_per_traj_published": g_old_rate,
                "median_jaccard_endpoint_gsa": j_ep_gsa,
                "median_jaccard_combined_endpoint": j_comb_ep,
                "n_timing_pairs": n_pairs,
            }
        )
    out = OUT / "matched_penalty_recompute_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _run_pool(jobs: list[tuple[list[str], Path]], workers: int) -> None:
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_cmd, args, log): log for args, log in jobs}
        for fut in as_completed(futs):
            log = futs[fut]
            rc = fut.result()
            if rc != 0:
                failures.append((log, rc))
    if failures:
        for log, rc in failures:
            print(f"FAILED rc={rc} {log}", flush=True)
        raise SystemExit(1)


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print("Penalties (sweep elbow):", flush=True)
    for cube in CUBES:
        ep = elbow_penalty(OUT / f"endpoint_changepoints_{cube}" / "penalty_sweep")
        gp = elbow_penalty(OUT / f"gsa_changepoints_{cube}" / "penalty_sweep")
        print(f"  {cube}: endpoint={ep:.6g}  gsa={gp:.6g}", flush=True)

    detect = detect_jobs()
    endpoint_jobs = [
        j
        for j in detect
        if "detect_endpoint_" in j[1].name
        and not (OUT / j[1].stem.replace("detect_endpoint_", "endpoint_changepoints_matched_") / "all_breakpoints.csv").is_file()
    ]
    # Fresh GSA dirs: a previous 6-way run oversubscribed BLAS and was killed mid-cube.
    for cube in CUBES:
        gsa_dir = OUT / f"gsa_changepoints_matched_{cube}"
        if gsa_dir.is_dir() and not (gsa_dir / "all_breakpoints.csv").is_file():
            shutil.rmtree(gsa_dir)
            print(f"removed incomplete {gsa_dir.name}", flush=True)
    gsa_jobs = [
        j
        for j in detect
        if "detect_gsa_" in j[1].name
        and not (OUT / j[1].stem.replace("detect_gsa_", "gsa_changepoints_matched_") / "all_breakpoints.csv").is_file()
    ]

    if endpoint_jobs:
        print(f"\n=== endpoint re-detect ({len(endpoint_jobs)} cubes) ===", flush=True)
        _run_pool(endpoint_jobs, workers=6)
    else:
        print("\n=== endpoint re-detect skipped (all_breakpoints.csv present) ===", flush=True)

    print(f"\n=== GSA re-detect gsa+iodine ({len(gsa_jobs)} cubes) ===", flush=True)
    _run_pool(gsa_jobs, workers=6)

    print("\n=== timing comparison ===", flush=True)
    _run_pool(timing_jobs(), workers=6)

    summary = write_summary()
    print(f"\nSummary: {summary}", flush=True)
    print(pd.read_csv(summary).to_string(index=False), flush=True)
    print("\nMATCHED RECOMPUTE DONE", flush=True)


if __name__ == "__main__":
    main()
