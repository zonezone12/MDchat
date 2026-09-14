"""Plot time (ns) vs RMSD and per-unit distances from stored motif CSVs.

Does not re-read trajectories. Per-unit plots appear only after CSVs include
``cation_pi_pole_e*`` / ``cation_pi_eq_e*`` / ``cation_pi_angle_e*`` /
``equator_d1_e*`` / ``equator_d2_e*`` / ``equator_angle_e*``
columns (re-run compute_murata_motif_rmsd).

Example
-------
python scripts/plot_murata_motif_rmsd_timeseries.py
python scripts/plot_murata_motif_rmsd_timeseries.py --output-root output --cubes BMHpH BMMpM
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.murata_rmsd import (
    CUBE_COLORS,
    MOTIF_XLABELS,
    discover_motif_rmsd_csvs_by_cube,
    load_motif_rmsd_csvs,
    plot_per_unit_motif_series,
    plot_rmsd_vs_time,
    plot_rmsd_vs_time_mean_by_cube,
    plot_rmsd_vs_time_three_motifs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot RMSD vs time (ns) from stored motif RMSD CSVs."
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "output",
        help="Directory containing B*_motif_rmsd folders (default: output)",
    )
    p.add_argument(
        "--cubes",
        nargs="*",
        default=list(KNOWN_CUBES),
        help="Cohort names (default: all six B* cubes)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    by_cube = discover_motif_rmsd_csvs_by_cube(args.output_root, cubes=args.cubes)
    if not by_cube:
        print(f"No *_motif_rmsd.csv under {args.output_root}", flush=True)
        return 1

    stacked: dict[str, object] = {}
    for cube, paths in by_cube.items():
        print(f"loading {cube}: {len(paths)} CSVs …", flush=True)
        df = load_motif_rmsd_csvs(paths)
        if df.empty:
            print(f"  skip {cube}: empty", flush=True)
            continue
        stacked[cube] = df
        plots = args.output_root / f"{cube}_motif_rmsd" / "plots"
        color = CUBE_COLORS.get(cube, "0.35")
        n_traj = int(df["traj_id"].nunique()) if "traj_id" in df.columns else 1
        three = plot_rmsd_vs_time_three_motifs(
            df,
            plots / "rmsd_vs_time_three_motifs.png",
            color=color,
            title=f"{cube} RMSD vs time (n={n_traj} replicas)",
        )
        print(f"  wrote {three}", flush=True)
        for col, xlabel in MOTIF_XLABELS.items():
            if col not in df.columns:
                continue
            tag = col.replace("rmsd_", "").replace("_to_ref", "")
            path = plot_rmsd_vs_time(
                df,
                plots / f"rmsd_vs_time_{tag}.png",
                rmsd_col=col,
                ylabel=xlabel,
                title=f"{cube}  {xlabel.split(' RMSD')[0]}  n={n_traj}",
                color=color,
            )
            print(f"  wrote {path}", flush=True)
        for path in plot_per_unit_motif_series(df, plots, cube=cube):
            print(f"  wrote {path}", flush=True)

    if len(stacked) > 1:
        combined = args.output_root / "murata_motif_rmsd" / "plots"
        for col, xlabel in MOTIF_XLABELS.items():
            if not any(col in df.columns for df in stacked.values()):
                continue
            tag = col.replace("rmsd_", "").replace("_to_ref", "")
            path = plot_rmsd_vs_time_mean_by_cube(
                stacked,
                combined / f"rmsd_vs_time_{tag}_mean.png",
                rmsd_col=col,
                ylabel=xlabel,
            )
            print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
