"""Plot locked Murata cation–π / equator units as an interactive NGL HTML view.

Same viewer language as ``endpoint_sites_transition_0_to_1.html``: licorice
cube, centroid spheres, cylinders, click-to-solo sidebar. Labels are

* ``Cation–π unit k: pole Py⁺ i → Ph j ← eq Py⁺ k``
* ``Equator unit k: R2 i → R3 j ← eq Py⁺ i``

Hungarian pairing is taken from ``*_motif_units.csv`` when given, otherwise
locked on the chosen frame (same as ``compute_murata_motif_rmsd``).

Example
-------
python scripts/plot_murata_units_ngl.py \\
    --topology traj/BMMpH_ca.prmtop \\
    --trajectory traj/BMMpH_109345_mdcrd_v.trj \\
    --output output/BMMpH_motif_rmsd/plots/murata_units.html

# Same mapping as a motif RMSD pass, at the endpoint-plot frame
python scripts/plot_murata_units_ngl.py \\
    --topology "$TOPOLOGY" \\
    --trajectory "$TRAJ_DIR/109345/mdcrd_v" \\
    --units-csv output/BMMpH_motif_rmsd/BMMpH/109345_mdcrd_v_motif_units.csv \\
    --frame 2525
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.murata_units_ngl import write_murata_units_ngl_html
from src.ChangepointAnalysis.pipeline import _load_universe
from src.ChangepointAnalysis.traj_paths import expand_trajectories, format_no_trajectories_error


def cohort_from_topology(topology: Path | str) -> str:
    name = Path(topology).name
    for suffix in ("_ca.prmtop", ".prmtop", ".parm7", ".top"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(topology).stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "NGL HTML of locked cation-pi (pole Py+ -> Ph) and equator "
            "(R2 -> R3) units, styled like the endpoint transition viewer."
        )
    )
    p.add_argument("--topology", required=True, help="Shared topology file")
    p.add_argument(
        "--trajectory",
        "--trajectories",
        dest="trajectories",
        nargs="+",
        required=True,
        help="Trajectory path(s). Only the first file is drawn.",
    )
    p.add_argument(
        "--traj-format",
        default=None,
        help="MDAnalysis format (default: TRJ for extensionless mdcrd_v / .trj).",
    )
    p.add_argument(
        "--output",
        "--output-path",
        dest="output",
        type=Path,
        default=None,
        help="HTML path (default: output/<cube>_motif_rmsd/plots/murata_units.html)",
    )
    p.add_argument(
        "--units-csv",
        type=Path,
        default=None,
        help="Optional *_motif_units.csv from compute_murata_motif_rmsd.",
    )
    p.add_argument("--frame", type=int, default=0, help="Frame to snapshot (default: 0)")
    p.add_argument("--n-monomers", type=int, default=6)
    p.add_argument("--selection", default="resname MOL")
    p.add_argument(
        "--title",
        default=None,
        help="Sidebar title (default: Murata cation-pi and equator units)",
    )
    p.add_argument(
        "--cohort",
        default=None,
        help="Cohort label used only for the default output path.",
    )
    return p.parse_args()


def _default_output(topology: Path, cohort: Optional[str]) -> Path:
    cube = cohort or cohort_from_topology(topology)
    return ROOT / "output" / f"{cube}_motif_rmsd" / "plots" / "murata_units.html"


def main() -> int:
    args = parse_args()
    traj_paths = expand_trajectories(args.trajectories)
    if not traj_paths:
        raise SystemExit(format_no_trajectories_error(args.trajectories))
    traj = traj_paths[0]
    if len(traj_paths) > 1:
        print(
            f"Drawing first of {len(traj_paths)} trajectories: {traj}",
            flush=True,
        )

    units_df = None
    if args.units_csv is not None:
        if not args.units_csv.is_file():
            raise SystemExit(f"units CSV not found: {args.units_csv}")
        units_df = pd.read_csv(args.units_csv)

    out = args.output or _default_output(Path(args.topology), args.cohort)
    print(f"Loading {args.topology} + {traj} (frame {args.frame})", flush=True)
    universe = _load_universe(args.topology, traj, traj_format=args.traj_format)
    path = write_murata_units_ngl_html(
        universe,
        out,
        frame=args.frame,
        selection=args.selection,
        n_monomers=args.n_monomers,
        units_df=units_df,
        title=args.title,
    )
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
