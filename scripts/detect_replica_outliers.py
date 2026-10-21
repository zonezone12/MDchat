#!/usr/bin/env python3
"""Outlier runs: replicas the workflow flags as abnormal rather than as ensemble states.

Two independent signals, both automatic:

* **Dissociation (assembly QC).** A run fails when any frame has assembly
  Rg > 15 Å or a largest connected component below six monomers. Failed runs
  are removed before clustering.
* **Isolation by clustering.** At each Ward cut k = ``--k-min``..``--k-max``,
  a cluster whose largest single replica holds at least ``--min-top-share`` of
  its frames is that run's own excursion. A run isolated over many cuts, and
  from small k, is an outlier of the ensemble.

Writes, under ``--out-dir``:

  replica_outliers_by_k.csv   one row per (cube, k, isolated cluster)
  replica_outliers.csv        one row per flagged run: kind, the cuts where it is
                              isolated, its window, Rg inside vs outside the
                              window, and its Rg maximum ranked over all runs

Example
-------
python scripts/detect_replica_outliers.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.assembly_qc import scan_assembly_qc_dir
from src.ChangepointAnalysis.cluster_k_diagnostics import load_segments_at_k
from src.ChangepointAnalysis.cluster_replicas import isolated_replicas_by_k, replica_window_rg
from src.ChangepointAnalysis.detection import canonicalize_traj_id
from src.ChangepointAnalysis.feature_groups import traj_id_from_features_stem
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flag outlier replicas: dissociation and isolation by clustering.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--gsa-root", type=Path, default=ROOT / "output" / "gsa_features_by_cube")
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=15)
    p.add_argument("--min-top-share", type=float, default=0.5)
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "replica_outliers")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    selected = {}
    if args.k_selected.is_file():
        sel = pd.read_csv(args.k_selected)
        selected = dict(zip(sel["cohort"].astype(str), sel["selected_k"].astype(int)))

    qc_parts, iso_parts, rows = [], [], []
    seg_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for cube in args.cubes:
        gsa_dir = args.gsa_root / cube
        qc_parts.append(scan_assembly_qc_dir(gsa_dir).assign(cohort=cube))
        cp_dir = endpoint_changepoints_dir(args.output_root, cube, own_elbow=True)
        by_k = {}
        for k in range(args.k_min, args.k_max + 1):
            segs = load_segments_at_k(cp_dir, k, group="endpoint")
            if not segs.empty:
                by_k[k] = segs
                seg_cache[(cube, k)] = segs
        iso = isolated_replicas_by_k(by_k, min_top_share=args.min_top_share)
        if not iso.empty:
            iso.insert(0, "cohort", cube)
            iso_parts.append(iso)

    qc = pd.concat(qc_parts, ignore_index=True)
    qc["rg_max_rank_all"] = qc["assembly_rg_max"].rank(ascending=False, method="min").astype(int)
    qc["key"] = [(c, canonicalize_traj_id(t)) for c, t in zip(qc["cohort"], qc["traj_id"])]
    qc_by_key = {rec["key"]: rec for rec in qc.to_dict("records")}
    iso_all = pd.concat(iso_parts, ignore_index=True) if iso_parts else pd.DataFrame()

    def qc_fields(cube: str, tid: str) -> dict:
        rec = qc_by_key.get((cube, canonicalize_traj_id(tid)), {})
        return {
            "n_frames": rec.get("n_frames"),
            "rg_max": rec.get("assembly_rg_max"),
            "rg_max_rank_of_all": rec.get("rg_max_rank_all"),
            "min_component_size": rec.get("min_component_size"),
            "qc_pass": rec.get("qc_pass"),
            "qc_first_fail_frame": rec.get("first_fail_frame"),
        }

    for rec in qc[~qc["qc_pass"].astype(bool)].itertuples(index=False):
        rows.append(
            {
                "cohort": rec.cohort,
                "traj_id": canonicalize_traj_id(rec.traj_id),
                "kind": "dissociated (removed by QC)",
                **qc_fields(rec.cohort, rec.traj_id),
            }
        )

    if not iso_all.empty:
        for (cube, tid), grp in iso_all.groupby(["cohort", "traj_id"], sort=False):
            grp = grp.sort_values("k")
            first = grp.iloc[0]
            k_sel = selected.get(cube)
            row = {
                "cohort": cube,
                "traj_id": canonicalize_traj_id(tid),
                "kind": "isolated by clustering",
                "first_k_isolated": int(first["k"]),
                "n_cuts_isolated": int(grp["k"].nunique()),
                "cuts_isolated": ",".join(str(int(k)) for k in sorted(grp["k"].unique())),
                "selected_k": k_sel,
                "isolated_at_selected_k": bool(k_sel in set(grp["k"])) if k_sel is not None else None,
                "max_top_share": float(grp["top_traj_frame_share"].max()),
                "window_first_frame": int(first["window_first_frame"]),
                "window_last_frame": int(first["window_last_frame"]),
                "cluster_n_segments_at_first_k": int(first["cluster_n_segments"]),
                **qc_fields(cube, tid),
            }
            segs = seg_cache[(cube, int(first["k"]))]
            replica = segs[segs["traj_id"].astype(str) == str(tid)]
            row["ends_in_window"] = bool(int(first["window_last_frame"]) >= int(replica["end_frame"].max()))
            path = args.gsa_root / cube / f"{cube}_{canonicalize_traj_id(tid)}_gsa_features.csv"
            if not path.is_file():
                cands = [
                    p for p in (args.gsa_root / cube).glob("*_gsa_features.csv")
                    if canonicalize_traj_id(traj_id_from_features_stem(p.stem)) == canonicalize_traj_id(tid)
                ]
                path = cands[0] if cands else None
            if path is not None:
                gsa = pd.read_csv(path, usecols=["frame", "assembly_rg"])
                win = replica_window_rg(gsa, replica, int(first["cluster_label"]))
                row.update({key: win.get(key) for key in ("rg_mean_in", "rg_max_in", "rg_mean_out", "rg_max_out")})
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["kind", "first_k_isolated", "rg_max_rank_of_all"], na_position="first")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("replica_outliers_by_k.csv", iso_all), ("replica_outliers.csv", out)):
        table.to_csv(args.out_dir / name, index=False)
        print(f"wrote {args.out_dir / name}", flush=True)
    print(f"runs scanned: {len(qc)}; median Rg max {qc['assembly_rg_max'].median():.2f} Å", flush=True)
    show = [c for c in [
        "cohort", "traj_id", "kind", "first_k_isolated", "n_cuts_isolated", "isolated_at_selected_k",
        "max_top_share", "window_first_frame", "window_last_frame", "ends_in_window", "n_frames",
        "rg_max", "rg_max_rank_of_all", "rg_mean_in", "rg_mean_out", "qc_pass",
    ] if c in out.columns]
    pd.set_option("display.width", 250)
    print(out[show].round(2).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
