"""Remap stored hull-index site pairs onto Murata equatorial d1 (C2–C3).

Reads ``output/endpoint_changepoints_B*/endpoint_features/*_endpoint_features.csv``
(no trajectory replay). Matches hull sites to canonical R2/R3, keeps the six
equatorial interlocking contacts, and writes occupancy vs 4.5–5.5 Å / ≥ 7.0 Å.

Example
-------
python scripts/remap_murata_d1.py \\
    --output-root output \\
    --out-dir output/murata_d1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.murata_d1 import (
    CUBES,
    plot_murata_d1_overview,
    remap_endpoint_cohort_d1,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Map existing endpoint_dist site-pair columns onto Murata d1 "
            "(R2/R3 equatorial C2–C3 proxies) without re-reading trajectories."
        )
    )
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "murata_d1")
    p.add_argument("--traj-dir", type=Path, default=ROOT / "traj")
    p.add_argument(
        "--cubes",
        nargs="*",
        default=list(CUBES),
        help="Cohort names (default: all six B* cubes)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[pd.DataFrame] = []
    frames_by_cohort: dict[str, pd.DataFrame] = {}

    for cube in args.cubes:
        cp_dir = args.output_root / f"endpoint_changepoints_{cube}"
        topo = args.traj_dir / f"{cube}_ca.prmtop"
        if not cp_dir.is_dir():
            print(f"skip {cube}: missing {cp_dir}", flush=True)
            continue
        if not topo.is_file():
            print(f"skip {cube}: missing {topo}", flush=True)
            continue
        cube_out = args.out_dir / cube
        print(f"remapping {cube} …", flush=True)
        written = remap_endpoint_cohort_d1(cp_dir, topo, cube_out)
        summaries.append(pd.read_csv(written["murata_d1_summary"]))
        frames_by_cohort[cube] = pd.read_csv(written["murata_d1_frames"])
        print(f"  edges: {written['equatorial_edges']}", flush=True)

    if not summaries:
        print("No cohorts remapped.", flush=True)
        return 1

    summary = pd.concat(summaries, ignore_index=True)
    summary_path = args.out_dir / "murata_d1_summary.csv"
    summary.to_csv(summary_path, index=False)
    maps = []
    edges = []
    for cube in frames_by_cohort:
        maps.append(pd.read_csv(args.out_dir / cube / "role_site_map.csv"))
        edges.append(pd.read_csv(args.out_dir / cube / "equatorial_edges.csv"))
    pd.concat(maps, ignore_index=True).to_csv(
        args.out_dir / "role_site_map.csv", index=False
    )
    pd.concat(edges, ignore_index=True).to_csv(
        args.out_dir / "equatorial_edges.csv", index=False
    )
    plot_path = args.out_dir / "plots" / "murata_d1_overview.png"
    plot_murata_d1_overview(summary, frames_by_cohort, plot_path)
    print(f"wrote {summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"wrote {plot_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
