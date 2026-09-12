"""Remap stored hull site pairs onto Murata cation–π units and metastructure labels.

Reads ``output/endpoint_changepoints_B*/endpoint_features/*_endpoint_features.csv``
(no trajectory replay). Matches hull sites to pole Py⁺ and Ph, locks six
intermolecular cation–π contacts (open ≥ 6.5 Å), and if remapped d1 frames are
present, emits A/B/C1/C2/other occupancy.

Example
-------
python scripts/remap_murata_criteria.py \\
    --output-root output \\
    --d1-dir output/murata_d1 \\
    --out-dir output/murata_criteria
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.murata_criteria import (
    occupancy_table,
    remap_endpoint_cohort_cation_pi,
)
from src.ChangepointAnalysis.murata_d1 import CUBES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Map existing endpoint_dist site-pair columns onto Murata cation–π "
            "units (pole Py⁺ COM to Ph COM) without re-reading trajectories."
        )
    )
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "murata_criteria")
    p.add_argument("--d1-dir", type=Path, default=ROOT / "output" / "murata_d1")
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
    occupancy_rows: list[pd.DataFrame] = []

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
        d1_frames = args.d1_dir / cube / "murata_d1_frames.csv"
        print(f"cation–π remap {cube} …", flush=True)
        written = remap_endpoint_cohort_cation_pi(
            cp_dir,
            topo,
            cube_out,
            d1_frames_csv=d1_frames if d1_frames.is_file() else None,
        )
        row = pd.read_csv(written["murata_cation_pi_summary"])
        row.insert(0, "cohort", cube)
        summaries.append(row)
        frames = pd.read_csv(written["murata_cation_pi_frames"])
        if "murata_metastructure" in frames.columns:
            occ = occupancy_table(frames["murata_metastructure"])
            occ.insert(0, "cohort", cube)
            occupancy_rows.append(occ)
        print(f"  edges: {written['cation_pi_edges']}", flush=True)

    if not summaries:
        print("No cohorts remapped.", flush=True)
        return 1

    summary = pd.concat(summaries, ignore_index=True)
    summary_path = args.out_dir / "murata_cation_pi_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)
    if occupancy_rows:
        occ_path = args.out_dir / "murata_metastructure_occupancy.csv"
        pd.concat(occupancy_rows, ignore_index=True).to_csv(occ_path, index=False)
        print(f"wrote {occ_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
