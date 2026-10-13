"""G5: do finer endpoint-cluster cuts separate Murata's C2 from "other"?

For the mapped cohorts (BMMpM = 1₆, BMHpM = 2₆, BHHpM = 3₆), apo motif frames
are labelled A/B/C1/C2/other and mapped to the endpoint clusters at every Ward
cut stored under ``clusters/endpoint/by_k/``. Per cohort and k,
``murata_labels_across_k.csv`` gives:

* ``tvd_C2_vs_other_pct``: total-variation distance between the cluster
  distributions of C2 and "other" frames (0% = the cut cannot tell them apart,
  100% = they never share a cluster);
* the intact-belt cluster (most frames with no elongated d1): its frame count,
  its share of the intact-belt frames, the % of its frames that are intact, and
  the % of each Murata label it holds. A frame count that stays the same over k
  means the Ward tree never splits that cluster.

Replicas failing the assembly QC rule are dropped first unless
``--no-assembly-qc`` is given.

Example
-------
python scripts/murata_labels_across_k.py --own-elbow
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.assembly_qc import (
    assembly_qc_failures,
    drop_trajectories,
    scan_assembly_qc_dir,
)
from src.ChangepointAnalysis.cluster_k_diagnostics import chosen_k_from_cohort, murata_labels_across_k
from src.ChangepointAnalysis.murata_criteria import COHORT_TO_MURATA_SYSTEM
from src.ChangepointAnalysis.murata_rmsd import (
    discover_motif_rmsd_csvs_by_cube,
    filter_non_encapsulated,
    label_motif_metastructures,
    load_motif_rmsd_csvs,
)
from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir

_K_DIR_RE = re.compile(r"k_(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Murata label spread over endpoint clusters at each k.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "murata_labels_across_k")
    p.add_argument("--cubes", nargs="*", default=list(COHORT_TO_MURATA_SYSTEM))
    p.add_argument(
        "--own-elbow",
        action="store_true",
        help="Read endpoint clusters from the own-elbow detections (OWN_ELBOW_ENDPOINT_DIRS)",
    )
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, help="Largest k to score (default: every stored cut)")
    p.add_argument(
        "--qc-gsa-root",
        type=Path,
        default=ROOT / "output" / "gsa_features_by_cube",
        help="Root of per-cube *_gsa_features.csv folders used by the assembly QC rule",
    )
    p.add_argument(
        "--no-assembly-qc",
        action="store_true",
        help="Keep replicas that fail the assembly QC rule",
    )
    return p.parse_args()


def stored_k_values(cp_dir: Path) -> list[int]:
    by_k = cp_dir / "clusters" / "endpoint" / "by_k"
    found = {int(m.group(1)) for p in by_k.glob("k_*") if p.is_dir() and (m := _K_DIR_RE.fullmatch(p.name))}
    found.add(chosen_k_from_cohort(cp_dir, group="endpoint"))
    return sorted(found)


def main() -> int:
    args = parse_args()
    by_cube = discover_motif_rmsd_csvs_by_cube(args.output_root, cubes=args.cubes)
    if not by_cube:
        print(f"No *_motif_rmsd.csv under {args.output_root}", flush=True)
        return 1
    parts: list[pd.DataFrame] = []
    for cube, paths in by_cube.items():
        cp_dir = endpoint_changepoints_dir(args.output_root, cube, own_elbow=args.own_elbow)
        if not (cp_dir / "clusters" / "endpoint").is_dir():
            print(f"skip {cube}: no endpoint clustering in {cp_dir}", flush=True)
            continue
        k_values = [
            k for k in stored_k_values(cp_dir) if k >= args.k_min and (args.k_max is None or k <= args.k_max)
        ]
        print(f"{cube} ({COHORT_TO_MURATA_SYSTEM.get(cube, '')}): {len(paths)} CSVs, k = {k_values}", flush=True)
        df = load_motif_rmsd_csvs(paths)
        if not args.no_assembly_qc:
            gsa_dir = args.qc_gsa_root / cube
            if gsa_dir.is_dir():
                failed = assembly_qc_failures(scan_assembly_qc_dir(gsa_dir))
                if failed:
                    print(f"  assembly QC excludes {', '.join(failed)}", flush=True)
                    df = drop_trajectories(df, failed)
            else:
                print(f"  assembly QC skipped: {gsa_dir} not found", flush=True)
        apo = filter_non_encapsulated(label_motif_metastructures(df), traj_filter="frames")
        apo = apo.dropna(subset=["murata_metastructure"])
        table = murata_labels_across_k(cp_dir, apo, cube=cube, k_values=k_values)
        if not table.empty:
            table.insert(1, "murata_system", COHORT_TO_MURATA_SYSTEM.get(cube, ""))
            parts.append(table)

    if not parts:
        print("No endpoint clusterings scored.", flush=True)
        return 1
    out = pd.concat(parts, ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "murata_labels_across_k.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}", flush=True)
    show = [
        "cohort",
        "k",
        "tvd_C2_vs_other_pct",
        "intact_cluster",
        "intact_cluster_n_frames",
        "intact_cluster_share_of_intact_pct",
        "pct_A_in_intact_cluster",
        "pct_C2_in_intact_cluster",
        "pct_other_in_intact_cluster",
    ]
    print(out[[c for c in show if c in out.columns]].round(1).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
