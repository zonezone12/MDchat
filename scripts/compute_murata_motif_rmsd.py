"""Compute Murata motif RMSD: cation–π (Py+, Ph, Py+) and equator (Py+, R2, R3, Py+).

Whole-cube ``assembly_rmsd_to_ref`` smears those openings. This pass Kabsch-
aligns only the motif heavy atoms vs each replica's frame 0.

Trajectory globs match ``run_endpoint_changepoint.py`` (HPC nested
``$TRAJ_DIR/<run>/mdcrd_v``). Replicas run in parallel via ``--traj-jobs``.

Example
-------
python scripts/compute_murata_motif_rmsd.py \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectories traj/BMMpM_*_mdcrd_v.trj \\
    --output-dir output/murata_motif_rmsd \\
    --traj-jobs -1

# HPC nested layout
python scripts/compute_murata_motif_rmsd.py \\
    --topology "$TOPOLOGY" \\
    --trajectories "$TRAJ_DIR/*/mdcrd_v" \\
    --output-dir output/murata_motif_rmsd \\
    --gsa-dir output/gsa_features_step1 \\
    --traj-jobs -1
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.murata_rmsd import (
    MOTIF_XLABELS,
    plot_murata_rmsd_distributions,
    plot_murata_rmsd_overlay,
    summarize_apo_rmsd,
    write_trajectory_motif_rmsd,
)
from src.ChangepointAnalysis.pipeline import _resolve_traj_workers
from src.ChangepointAnalysis.traj_paths import (
    expand_trajectories,
    format_no_trajectories_error,
    unique_trajectory_ids,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "RMSD of Murata cation–π (Py+, Ph, Py+) and equatorial "
            "(Py+, R2, R3, Py+) atoms vs the initial NVT frame."
        )
    )
    p.add_argument("--topology", required=True, help="Shared topology file")
    p.add_argument(
        "--trajectories",
        nargs="+",
        required=True,
        help=(
            "Trajectory paths or globs. Nested HPC layout: "
            "'$TRAJ_DIR/*/mdcrd_v' (ids become <run>_mdcrd_v)."
        ),
    )
    p.add_argument(
        "--traj-format",
        default=None,
        help=(
            "MDAnalysis trajectory format (default: TRJ for extensionless "
            "mdcrd_v / .trj; auto-detect otherwise)."
        ),
    )
    p.add_argument(
        "--output-dir",
        "--out-dir",
        dest="output_dir",
        type=Path,
        default=ROOT / "output" / "murata_motif_rmsd",
        help="Output directory (default: output/murata_motif_rmsd)",
    )
    p.add_argument(
        "--gsa-dir",
        type=Path,
        default=ROOT / "output" / "gsa_features_step1",
        help="Optional GSA CSVs for iodide occupancy (apo filter).",
    )
    p.add_argument("--n-monomers", type=int, default=6)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--ref-frame", type=int, default=0)
    p.add_argument(
        "--traj-jobs",
        type=int,
        default=-1,
        help=(
            "Parallel workers across trajectories "
            "(default: -1 = all CPUs, capped by traj count)."
        ),
    )
    p.add_argument(
        "--cohort",
        default=None,
        help="Cohort label (default: inferred from topology, e.g. BMMpM_ca.prmtop).",
    )
    return p.parse_args()


def cohort_from_topology(topology: Path | str) -> str:
    name = Path(topology).name
    for suffix in ("_ca.prmtop", ".prmtop", ".parm7", ".top"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(topology).stem


def find_gsa_feature_csv(
    gsa_dir: Path,
    traj_id: str,
    *,
    cube: str,
) -> Optional[Path]:
    """Match a GSA features CSV to an HPC or flat traj id."""
    if not gsa_dir.is_dir():
        return None
    names = [
        f"{traj_id}_gsa_features.csv",
        f"{cube}_{traj_id}_gsa_features.csv",
    ]
    if traj_id.startswith(f"{cube}_"):
        names.append(f"{traj_id[len(cube) + 1 :]}_gsa_features.csv")
    for name in names:
        hits = list(gsa_dir.rglob(name))
        if hits:
            return hits[0]
    suffix = f"_{traj_id}_gsa_features.csv"
    for path in gsa_dir.rglob("*_gsa_features.csv"):
        if path.name == names[0] or path.name.endswith(suffix):
            return path
    return None


def _plot_stacked(stacked: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    plots = out_dir / "plots"
    for col, xlabel in MOTIF_XLABELS.items():
        if not any(col in df.columns for df in stacked.values()):
            continue
        for cube, df in stacked.items():
            if col not in df.columns:
                continue
            for filt in ("frames", "all"):
                if filt == "frames" and "n_guest_inside_cavity" not in df.columns:
                    continue
                summaries.append(
                    summarize_apo_rmsd(df, cohort=cube, rmsd_col=col, traj_filter=filt)
                )
        tag = col.replace("rmsd_", "").replace("_to_ref", "")
        plot_murata_rmsd_distributions(
            stacked,
            plots / f"murata_{tag}_by_cube.png",
            traj_filter="all",
            rmsd_col=col,
            xlabel=xlabel,
        )
        plot_murata_rmsd_overlay(
            stacked,
            plots / f"murata_{tag}_overlay.png",
            traj_filter="all",
            rmsd_col=col,
            xlabel=xlabel,
        )
        if any("n_guest_inside_cavity" in df.columns for df in stacked.values()):
            plot_murata_rmsd_distributions(
                stacked,
                plots / f"murata_{tag}_by_cube_apo.png",
                traj_filter="frames",
                rmsd_col=col,
                xlabel=xlabel,
            )
    return pd.DataFrame(summaries)


def main() -> int:
    args = parse_args()
    traj_paths = expand_trajectories(args.trajectories)
    if not traj_paths:
        raise SystemExit(format_no_trajectories_error(args.trajectories))
    print(f"Found {len(traj_paths)} trajectory file(s):", flush=True)
    for p in traj_paths[:20]:
        print(f"  {p}", flush=True)
    if len(traj_paths) > 20:
        print(f"  ... and {len(traj_paths) - 20} more", flush=True)

    cube = args.cohort or cohort_from_topology(args.topology)
    traj_ids = unique_trajectory_ids(traj_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cube_dir = args.output_dir / cube
    cube_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for traj, traj_id in zip(traj_paths, traj_ids):
        gsa = find_gsa_feature_csv(args.gsa_dir, traj_id, cube=cube)
        jobs.append(
            (
                str(args.topology),
                str(traj),
                traj_id,
                cube,
                str(cube_dir / f"{traj_id}_motif_rmsd.csv"),
                str(gsa) if gsa is not None else None,
                args.start,
                args.stop,
                args.step,
                args.ref_frame,
                args.n_monomers,
                args.traj_format,
            )
        )

    n_workers = _resolve_traj_workers(args.traj_jobs, len(jobs))
    results: list[dict[str, object]] = []
    print(
        f"Computing motif RMSD for {len(jobs)} trajectories "
        f"with traj_jobs={n_workers}",
        flush=True,
    )
    if n_workers == 1 or len(jobs) == 1:
        for job in jobs:
            res = write_trajectory_motif_rmsd(*job)
            print(f"  wrote {res['path']}  n={res['n']}", flush=True)
            results.append(res)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(write_trajectory_motif_rmsd, *job): job[2] for job in jobs
            }
            for fut in as_completed(futures):
                res = fut.result()
                print(f"  wrote {res['path']}  n={res['n']}", flush=True)
                results.append(res)

    frames: list[pd.DataFrame] = []
    for res in results:
        frames.append(pd.read_csv(res["path"]))
    if not frames:
        print("No motif RMSD CSVs written.", flush=True)
        return 1
    stacked = {cube: pd.concat(frames, ignore_index=True)}
    summary = _plot_stacked(stacked, args.output_dir)
    if not summary.empty:
        summary_path = args.output_dir / "murata_motif_rmsd_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"wrote {summary_path}", flush=True)
        print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
