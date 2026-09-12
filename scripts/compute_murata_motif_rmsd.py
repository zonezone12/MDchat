"""Compute Murata motif RMSD: cation–π (Py+, Ph, Py+) and equator (Py+, R2, R3, Py+).

Whole-cube ``assembly_rmsd_to_ref`` smears those openings. This pass Kabsch-
aligns only the motif heavy atoms vs each replica's frame 0.

Example
-------
python scripts/compute_murata_motif_rmsd.py
python scripts/compute_murata_motif_rmsd.py --step 5
python scripts/compute_murata_motif_rmsd.py --topology traj/BMMpM_ca.prmtop \\
    --trajectories traj/BMMpM_891249_mdcrd_v.trj
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.murata_rmsd import (
    MOTIF_XLABELS,
    attach_guest_occupancy,
    compute_trajectory_motif_rmsd,
    plot_murata_rmsd_distributions,
    plot_murata_rmsd_overlay,
    summarize_apo_rmsd,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "RMSD of Murata cation–π (Py+, Ph, Py+) and equatorial "
            "(Py+, R2, R3, Py+) atoms vs the initial NVT frame."
        )
    )
    p.add_argument("--traj-dir", type=Path, default=ROOT / "traj")
    p.add_argument("--topology", type=Path, default=None)
    p.add_argument("--trajectories", nargs="*", default=None)
    p.add_argument(
        "--cubes",
        nargs="*",
        default=list(KNOWN_CUBES),
        help="Cohorts to glob under --traj-dir when --trajectories is omitted.",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "murata_motif_rmsd")
    p.add_argument(
        "--gsa-dir",
        type=Path,
        default=ROOT / "output" / "gsa_features_step1",
        help="Optional GSA CSVs for iodide occupancy (apo filter).",
    )
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--ref-frame", type=int, default=0)
    return p.parse_args()


def _discover(traj_dir: Path, cubes: list[str]) -> list[tuple[str, Path, Path]]:
    jobs: list[tuple[str, Path, Path]] = []
    for cube in cubes:
        topo = traj_dir / f"{cube}_ca.prmtop"
        if not topo.is_file():
            continue
        for path in sorted(traj_dir.glob(f"{cube}_*_mdcrd_v.trj")):
            jobs.append((cube, topo, path))
    return jobs


def _gsa_csv(gsa_dir: Path, traj_id: str) -> Path | None:
    if not gsa_dir.is_dir():
        return None
    hits = list(gsa_dir.rglob(f"{traj_id}_gsa_features.csv"))
    return hits[0] if hits else None


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.topology and args.trajectories:
        cube = args.topology.name.replace("_ca.prmtop", "")
        jobs = [(cube, args.topology, Path(p)) for p in args.trajectories]
    else:
        jobs = _discover(args.traj_dir, args.cubes)
    if not jobs:
        print("No topology/trajectory pairs found.", flush=True)
        return 1

    frames_by_cube: dict[str, list[pd.DataFrame]] = {}
    for cube, topo, traj in jobs:
        traj_id = traj.stem
        print(f"{cube} {traj_id} …", flush=True)
        df = compute_trajectory_motif_rmsd(
            topo,
            traj,
            traj_id=traj_id,
            start=args.start,
            stop=args.stop,
            step=args.step,
            ref_frame=args.ref_frame,
        )
        df.insert(0, "cohort", cube)
        gsa = _gsa_csv(args.gsa_dir, traj_id)
        if gsa is not None:
            df = attach_guest_occupancy(df, gsa)
            print(f"  joined guest occupancy from {gsa.name}", flush=True)
        elif "n_guest_inside_cavity" not in df.columns:
            df["n_guest_inside_cavity"] = 0.0
        cube_dir = args.out_dir / cube
        cube_dir.mkdir(parents=True, exist_ok=True)
        path = cube_dir / f"{traj_id}_motif_rmsd.csv"
        df.to_csv(path, index=False)
        print(f"  wrote {path}  n={len(df)}", flush=True)
        frames_by_cube.setdefault(cube, []).append(df)

    stacked = {c: pd.concat(rows, ignore_index=True) for c, rows in frames_by_cube.items()}
    summaries = []
    plots = args.out_dir / "plots"
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

    if summaries:
        summary = pd.DataFrame(summaries)
        summary_path = args.out_dir / "murata_motif_rmsd_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"wrote {summary_path}", flush=True)
        print(summary.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
