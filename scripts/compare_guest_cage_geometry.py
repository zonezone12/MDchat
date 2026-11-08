"""Cage geometry with iodide in the cavity versus the same replica empty.

Thin CLI around ``src.ChangepointAnalysis.guest_geometry.compare_guest_cage_geometry``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.guest_geometry import compare_guest_cage_geometry


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--gsa-features-root", type=Path, default=ROOT / "output" / "gsa_features_by_cube")
    p.add_argument("--dwells", type=Path, default=ROOT / "output" / "guest_event_cotiming" / "guest_dwells.csv")
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "guest_cage_geometry")
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument("--min-dwell", type=int, default=20)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        written = compare_guest_cage_geometry(
            output_root=args.output_root,
            gsa_features_root=args.gsa_features_root,
            dwells=args.dwells,
            k_selected=args.k_selected,
            out_dir=args.out_dir,
            cubes=args.cubes,
            min_dwell=args.min_dwell,
        )
    except FileNotFoundError as exc:
        print(exc, flush=True)
        return 1
    print(f"wrote {next(iter(written.values())).parent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
