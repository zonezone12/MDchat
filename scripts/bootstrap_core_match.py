#!/usr/bin/env python3
"""Replica bootstrap of core-to-core deformation-profile r.

Companion to ``check_merge_dilution.py``. That script splits each cluster
label-free into a core and a weak part (``core_split``) and compares the
cores. This one asks how stable the core r is when the replicas are
resampled, and whether the cores are carried by a few replicas.

Each draw samples replicas with replacement, pools their segments, re-runs
``core_split`` on the listed clusters and recomputes the best-relabelled r
of every pair of cores, against the grand mean of the draw. A core with
fewer than ``--min-core`` segments in a draw is skipped for that draw.

Writes, under ``--out-dir``:

  {CUBE}_kXX_core_replicas.csv     segments, replicas and top-replica share per part
  {CUBE}_kXX_core_bootstrap.csv    median, 2.5/97.5% and P(r >= COPY_R) per core pair
  {CUBE}_kXX_core_bootstrap_draws.csv   r of every pair in every draw

Example
-------
python scripts/bootstrap_core_match.py --cube BMHpH --k 6 --clusters 2 3 4 5
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.ChangepointAnalysis.assembly_qc import assembly_qc_failures, drop_trajectories, scan_assembly_qc_dir
from src.ChangepointAnalysis.cluster_k_diagnostics import load_segments_at_k, resolve_endpoint_features_dir
from src.ChangepointAnalysis.cluster_validation import best_profile_r, core_split, mappings_from_perms
from src.ChangepointAnalysis.metastable_catalog import COPY_R, contact_automorphisms
from src.ChangepointAnalysis.reporting import collect_endpoint_segment_means, endpoint_changepoints_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--qc-gsa-root", type=Path, default=ROOT / "output" / "gsa_features_by_cube")
    p.add_argument("--cube", required=True)
    p.add_argument("--k", type=int, required=True, help="Ward cut")
    p.add_argument("--clusters", type=int, nargs="+", required=True, help="Clusters whose cores are compared")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--min-core", type=int, default=3, help="Minimum core segments in a draw to use that core")
    p.add_argument("--copy-r", type=float, default=COPY_R)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_validation_physical_k" / "merge_dilution" / "core_bootstrap",
    )
    return p.parse_args()


def load_segment_means(args: argparse.Namespace):
    cp_dir = endpoint_changepoints_dir(args.output_root, args.cube, own_elbow=True)
    segs = load_segments_at_k(cp_dir, args.k, group="endpoint")
    gsa_dir = args.qc_gsa_root / args.cube
    if gsa_dir.is_dir():
        failed = assembly_qc_failures(scan_assembly_qc_dir(gsa_dir))
        if failed:
            segs = drop_trajectories(segs, failed)
    segs = segs.reset_index(drop=True)
    X, y, keys, specs, kind, _ = collect_endpoint_segment_means(
        cp_dir, resolve_endpoint_features_dir(cp_dir), group="endpoint", clustered_df=segs, include_frame_corr=False
    )
    if kind != "site_pair" or X.size == 0:
        raise FileNotFoundError(f"{args.cube}: no site-pair segment means")
    return X, np.asarray(y), keys.reset_index(drop=True)["traj_id"].astype(str).to_numpy(), specs


def replica_table(y: np.ndarray, part: np.ndarray, rep: np.ndarray, clusters, cube: str, k: int) -> pd.DataFrame:
    rows = []
    for c in clusters:
        for name in ("core", "weak"):
            m = (y == c) & (part == name)
            counts = pd.Series(rep[m]).value_counts()
            rows.append(
                {
                    "cohort": cube,
                    "k": k,
                    "cluster_label": int(c),
                    "part": name,
                    "n_segments": int(m.sum()),
                    "n_replicas": int(counts.size),
                    "top_replica": counts.index[0] if counts.size else "",
                    "top_replica_share": float(counts.iloc[0] / m.sum()) if counts.size else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    clusters = [int(c) for c in args.clusters]
    if len(clusters) < 2:
        print("--clusters needs at least two labels", flush=True)
        return 1
    X, y, rep, specs = load_segment_means(args)
    mappings = mappings_from_perms(specs, contact_automorphisms())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.cube}_k{args.k:02d}"

    replicas = replica_table(y, core_split(X, y, clusters), rep, clusters, args.cube, args.k)
    replicas.to_csv(args.out_dir / f"{stem}_core_replicas.csv", index=False)
    print(replicas.round(2).to_string(index=False), flush=True)

    reps = np.unique(rep)
    rows_of = {r: np.flatnonzero(rep == r) for r in reps}
    rng = np.random.default_rng(args.seed)
    pairs = list(combinations(clusters, 2))
    draws = []
    for b in range(int(args.n_boot)):
        idx = np.concatenate([rows_of[r] for r in rng.choice(reps, size=reps.size, replace=True)])
        Xb, yb = X[idx], y[idx]
        part = core_split(Xb, yb, clusters)
        base = Xb.mean(axis=0)
        cores = {}
        for c in clusters:
            m = (yb == c) & (part == "core")
            if m.sum() >= int(args.min_core):
                cores[c] = Xb[m].mean(axis=0) - base
        for a, c in pairs:
            if a in cores and c in cores:
                draws.append({"draw": b, "cluster_a": a, "cluster_b": c, "r_core": best_profile_r(cores[a], cores[c], mappings)[0]})
    draws = pd.DataFrame(draws)
    draws.to_csv(args.out_dir / f"{stem}_core_bootstrap_draws.csv", index=False)

    summary = []
    for (a, c), grp in draws.groupby(["cluster_a", "cluster_b"]):
        v = grp["r_core"].to_numpy()
        summary.append(
            {
                "cohort": args.cube,
                "k": args.k,
                "cluster_a": int(a),
                "cluster_b": int(c),
                "n_draws": int(v.size),
                "median_r": float(np.median(v)),
                "r_2p5": float(np.percentile(v, 2.5)),
                "r_97p5": float(np.percentile(v, 97.5)),
                "p_at_or_above_copy_r": float((v >= float(args.copy_r)).mean()),
            }
        )
    summary = pd.DataFrame(summary)
    path = args.out_dir / f"{stem}_core_bootstrap.csv"
    summary.to_csv(path, index=False)
    print(summary.round(3).to_string(index=False), flush=True)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
