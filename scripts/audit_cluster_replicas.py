"""Which endpoint clusters are one replica's excursion rather than a population state?

For each cube's clustering (the stored k, ``--k``, or each cube's ``--k-selected``)
this writes:

* ``cluster_replica_composition.csv``: per cluster, its segments, replicas and
  frames, and the replica contributing the most frames (``top_traj``,
  ``top_traj_frame_share``);
* ``single_replica_clusters.csv``: clusters whose top replica holds at least
  ``--min-top-share`` of the frames. For that replica: assembly Rg inside the
  cluster's segments vs the rest of its run, where its Rg maximum falls,
  whether the run ends in the cluster, how its Rg maximum ranks among every
  replica scanned (1 = largest), and its assembly QC verdict (Rg > 15 Å or a
  lost monomer);
* ``single_replica_feature_margins.csv``: for those clusters, the monomer-pair
  segment means that lie outside the range of every other cluster.

Example
-------
python scripts/audit_cluster_replicas.py --own-elbow
python scripts/audit_cluster_replicas.py --own-elbow \\
    --k-selected output/endpoint_cluster_chemical_k_own_elbow/chemical_k_selected.csv \\
    --out-dir output/cluster_replica_audit_selected_k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.assembly_qc import scan_assembly_qc_dir
from src.ChangepointAnalysis.cluster_k_diagnostics import chosen_k_from_cohort, load_segments_at_k
from src.ChangepointAnalysis.cluster_replicas import (
    cluster_feature_margins,
    cluster_replica_composition,
    replica_window_rg,
)
from src.ChangepointAnalysis.detection import canonicalize_traj_id
from src.ChangepointAnalysis.feature_groups import traj_id_from_features_stem
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flag endpoint clusters carried by a single replica.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "cluster_replica_audit")
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument(
        "--own-elbow",
        action="store_true",
        help="Audit the clusters of the own-elbow detections (OWN_ELBOW_ENDPOINT_DIRS)",
    )
    k_choice = p.add_mutually_exclusive_group()
    k_choice.add_argument("--k", type=int, help="Audit the Ward cut at this k (default: the stored clustering)")
    k_choice.add_argument(
        "--k-selected",
        type=Path,
        help="chemical_k_selected.csv from select_chemical_k.py; use each cube's selected_k",
    )
    p.add_argument(
        "--gsa-root",
        type=Path,
        default=ROOT / "output" / "gsa_features_by_cube",
        help="Root of per-cube *_gsa_features.csv folders (assembly Rg, connectivity)",
    )
    p.add_argument(
        "--min-top-share",
        type=float,
        default=0.8,
        help="Flag a cluster when one replica holds at least this share of its frames (default: 0.8)",
    )
    p.add_argument("--top-n", type=int, default=10, help="Feature margins kept per flagged cluster")
    return p.parse_args()


def gsa_paths_by_traj(gsa_dir: Path) -> dict[str, Path]:
    return {
        canonicalize_traj_id(traj_id_from_features_stem(p.stem)): p
        for p in sorted(gsa_dir.glob("*_gsa_features.csv"))
    }


def main() -> int:
    args = parse_args()
    selected_k: dict[str, int] = {}
    if args.k_selected is not None:
        sel = pd.read_csv(args.k_selected)
        selected_k = dict(zip(sel["cohort"].astype(str), sel["selected_k"].astype(int)))
    comp_parts, single_rows, margin_parts, qc_parts = [], [], [], []

    for cube in args.cubes:
        cp_dir = endpoint_changepoints_dir(args.output_root, cube, own_elbow=args.own_elbow)
        if not (cp_dir / "clusters" / "endpoint").is_dir():
            print(f"skip {cube}: no endpoint clustering in {cp_dir}", flush=True)
            continue
        k = args.k or selected_k.get(cube) or chosen_k_from_cohort(cp_dir, group="endpoint")
        segs = load_segments_at_k(cp_dir, k, group="endpoint")
        if segs.empty:
            print(f"skip {cube}: no endpoint clustering at k={k}", flush=True)
            continue
        comp = cluster_replica_composition(segs)
        comp["single_replica"] = comp["top_traj_frame_share"] >= args.min_top_share
        comp.insert(0, "k", int(k))
        comp.insert(0, "cohort", cube)
        comp_parts.append(comp)

        gsa_dir = args.gsa_root / cube
        paths = gsa_paths_by_traj(gsa_dir) if gsa_dir.is_dir() else {}
        if gsa_dir.is_dir():
            qc_parts.append(scan_assembly_qc_dir(gsa_dir).assign(cohort=cube))
        else:
            print(f"  {cube}: {gsa_dir} not found; Rg columns left empty", flush=True)
        for rec in comp[comp["single_replica"]].itertuples(index=False):
            tid = str(rec.top_traj)
            replica = segs[segs["traj_id"].astype(str) == tid]
            row = {
                "cohort": cube,
                "k": int(k),
                "cluster_label": int(rec.cluster_label),
                "cluster_n_segments": int(rec.n_segments),
                "traj_id": tid,
                "top_traj_frame_share": float(rec.top_traj_frame_share),
                "n_segments": int((replica["cluster_label"] == rec.cluster_label).sum()),
            }
            path = paths.get(canonicalize_traj_id(tid))
            if path is not None:
                gsa = pd.read_csv(path, usecols=["frame", "assembly_rg"])
                row.update(replica_window_rg(gsa, replica, int(rec.cluster_label)))
            single_rows.append(row)
            margins = cluster_feature_margins(segs, int(rec.cluster_label)).head(args.top_n)
            if not margins.empty:
                margins.insert(0, "traj_id", tid)
                margins.insert(0, "cluster_label", int(rec.cluster_label))
                margins.insert(0, "k", int(k))
                margins.insert(0, "cohort", cube)
                margin_parts.append(margins)

    if not comp_parts:
        print("No endpoint clusterings found.", flush=True)
        return 1
    single = pd.DataFrame(single_rows)
    if not single.empty and qc_parts:
        qc = pd.concat(qc_parts, ignore_index=True)
        qc["rg_max_rank_all"] = qc["assembly_rg_max"].rank(ascending=False, method="min")
        by_key = {
            (rec["cohort"], canonicalize_traj_id(rec["traj_id"])): rec for rec in qc.to_dict("records")
        }
        keys = list(zip(single["cohort"], single["traj_id"].map(canonicalize_traj_id)))
        for col, src in (
            ("replica_rg_max", "assembly_rg_max"),
            ("replica_rg_max_rank_all", "rg_max_rank_all"),
            ("replica_min_component_size", "min_component_size"),
            ("replica_qc_pass", "qc_pass"),
        ):
            single[col] = [by_key.get(key, {}).get(src) for key in keys]
        single["n_replicas_scanned"] = len(qc)
        single["rg_max_median_all"] = float(qc["assembly_rg_max"].median())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "cluster_replica_composition.csv": pd.concat(comp_parts, ignore_index=True),
        "single_replica_clusters.csv": single,
        "single_replica_feature_margins.csv": (
            pd.concat(margin_parts, ignore_index=True) if margin_parts else pd.DataFrame()
        ),
    }
    for name, table in tables.items():
        path = args.out_dir / name
        table.to_csv(path, index=False)
        print(f"wrote {path}", flush=True)
    if single.empty:
        print(f"\nNo cluster has a replica holding >= {args.min_top_share:.0%} of its frames.", flush=True)
    else:
        print("\nSingle-replica clusters:", flush=True)
        print(single.round(2).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
