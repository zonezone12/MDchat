#!/usr/bin/env python3
"""Split each endpoint cluster into contact categories that do not depend on k.

Every apo frame is labeled from its own geometry: which equatorial d1 edges
are elongated (>= 7.0 Å) and how many pole cation–π contacts are open
(>= 6.5 Å). Those labels are then counted inside the Ward clusters. A piece
is kept when it holds at least 5% of the cluster. Its dwell says whether the
piece stays (runs of >= 20 ps) or flickers.

Writes, under ``--out-dir``:

  belt_inside_cluster.csv     frame labels inside each cluster (includes cutoff flicker)
  segment_belt.csv            each changepoint segment's majority belt, then summed inside the cluster
  cluster_mixture.csv         one row per cluster

Example
-------
python scripts/categorize_cluster_mixtures.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.ChangepointAnalysis.assembly_qc import (
    assembly_qc_failures,
    drop_trajectories,
    scan_assembly_qc_dir,
)
from src.ChangepointAnalysis.cluster_k_diagnostics import (
    _norm_traj_id,
    load_segments_at_k,
    map_frames_to_segment_clusters,
)
from src.ChangepointAnalysis.metastable_catalog import belt_label, dwell_stats, opening_label
from src.ChangepointAnalysis.murata_d1 import D1_ELONGATED_LO
from src.ChangepointAnalysis.murata_rmsd import (
    discover_motif_rmsd_csvs_by_cube,
    filter_non_encapsulated,
    label_motif_metastructures,
    load_motif_rmsd_csvs,
    unit_series_columns,
)
from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir

_EDGE_RE = re.compile(r"_e(\d+)_m(\d+)[a-z0-9]*_m(\d+)")
MIN_SHARE = 5.0
MIN_PERSIST_PS = 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Categorize the mixture inside each endpoint cluster.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument(
        "--qc-gsa-root",
        type=Path,
        default=ROOT / "output" / "gsa_features_by_cube",
    )
    p.add_argument("--min-share", type=float, default=MIN_SHARE)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "metastable_structures_physical_k",
    )
    return p.parse_args()


def _edge_name(column: str) -> tuple[int, str]:
    match = _EDGE_RE.search(column)
    if not match:
        return 0, column
    return int(match.group(1)), f"M{int(match.group(2))}-M{int(match.group(3))}"


def load_apo_frames(output_root: Path, qc_root: Path, cube: str) -> pd.DataFrame:
    paths = discover_motif_rmsd_csvs_by_cube(output_root, cubes=[cube]).get(cube, [])
    if not paths:
        raise FileNotFoundError(f"No {cube} motif CSVs under {output_root}")
    df = label_motif_metastructures(load_motif_rmsd_csvs(paths))
    gsa_dir = qc_root / cube
    if gsa_dir.is_dir():
        failed = assembly_qc_failures(scan_assembly_qc_dir(gsa_dir))
        if failed:
            df = drop_trajectories(df, failed)
    return filter_non_encapsulated(df, traj_filter="frames")


def _frame_belts(joined: pd.DataFrame, d1_cols: list[str]) -> pd.Series:
    order = [_edge_name(col) for col in d1_cols]
    elongated = joined[d1_cols].apply(pd.to_numeric, errors="coerce").to_numpy() >= D1_ELONGATED_LO
    labels = []
    for row in elongated:
        edges = [name for flag, (_idx, name) in zip(row, order) if flag]
        labels.append(belt_label(edges))
    return pd.Series(labels, index=joined.index)


def _segment_modes(joined: pd.DataFrame, segments: pd.DataFrame, cube: str) -> pd.DataFrame:
    """Majority belt of each changepoint segment, weighted later by its apo frames."""
    segs = segments.reset_index(drop=True).copy()
    segs["_tid"] = segs["traj_id"].map(lambda t: _norm_traj_id(t, cube))
    joined = joined.copy()
    joined["_tid"] = joined["traj_id"].map(lambda t: _norm_traj_id(t, cube))
    joined["segment_key"] = -1
    for key, rec in segs.iterrows():
        mask = joined["_tid"].to_numpy() == rec["_tid"]
        if not mask.any():
            continue
        fr = joined.loc[mask, "frame"].to_numpy()
        hit = (fr >= float(rec.start_frame)) & (fr <= float(rec.end_frame))
        joined.loc[joined.index[mask][hit], "segment_key"] = int(key)
    covered = joined[joined["segment_key"] >= 0]
    rows = []
    for key, grp in covered.groupby("segment_key", sort=False):
        belt_counts = grp["belt"].value_counts()
        opening_counts = grp["opening"].value_counts()
        rows.append(
            {
                "segment_key": int(key),
                "cluster_label": int(grp["cluster_label"].iloc[0]),
                "n_frames": int(len(grp)),
                "belt_mode": str(belt_counts.index[0]),
                "belt_mode_share": float(belt_counts.iloc[0] / len(grp)),
                "opening_mode": str(opening_counts.index[0]),
                "opening_mode_share": float(opening_counts.iloc[0] / len(grp)),
            }
        )
    return pd.DataFrame(rows)


def _category_table(part: pd.DataFrame, cube: str, k: int, cluster: int, cluster_n: int) -> pd.DataFrame:
    rows = []
    for belt, grp in part.groupby("belt", sort=False):
        median, persistent = dwell_stats(
            grp["traj_id"].to_numpy(),
            grp["frame"].to_numpy(),
            np.full(len(grp), belt),
            min_persist=MIN_PERSIST_PS,
        )
        openings = grp["opening"].value_counts(normalize=True)
        row = {
            "cohort": cube,
            "k": k,
            "cluster_label": cluster,
            "belt": belt,
            "n_frames": int(len(grp)),
            "pct_of_cluster": 100.0 * len(grp) / cluster_n,
            "median_dwell_ps": median,
            "pct_frames_in_dwells_ge_20ps": persistent,
            "pct_0_poles": 100.0 * float(openings.get("0 poles open", 0.0)),
            "pct_1_pole": 100.0 * float(openings.get("1 pole open", 0.0)),
            "pct_2_poles": 100.0 * float(openings.get("2 poles open", 0.0)),
            "pct_3plus_poles": 100.0 * float(openings.get("3+ poles open", 0.0)),
        }
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("pct_of_cluster", ascending=False)
    table.insert(0, "rank_in_cluster", range(1, len(table) + 1))
    return table


def main() -> int:
    args = parse_args()
    selected = pd.read_csv(args.k_selected)
    belt_parts = []
    segment_parts = []
    mixture_rows = []
    for row in selected.itertuples(index=False):
        cube = str(row.cohort)
        k = int(row.selected_k)
        print(f"{cube} k={k}: labeling apo frames …", flush=True)
        cp_dir = endpoint_changepoints_dir(args.output_root, cube, own_elbow=True)
        segs = load_segments_at_k(cp_dir, k, group="endpoint")
        joined = map_frames_to_segment_clusters(load_apo_frames(args.output_root, args.qc_gsa_root, cube), segs, cube=cube)
        joined = joined.dropna(subset=["cluster_label", "n_open_cation_pi"])
        joined["cluster_label"] = joined["cluster_label"].astype(int)
        joined["frame"] = pd.to_numeric(joined["frame"], errors="coerce")
        d1_cols = unit_series_columns(joined, "equator_d1_e")
        joined["belt"] = _frame_belts(joined, d1_cols)
        joined["opening"] = joined["n_open_cation_pi"].map(lambda n: opening_label(int(n)))
        modes = _segment_modes(joined, segs, cube)
        print(f"  {cube}: {len(joined)} apo frames in a cluster", flush=True)
        for cluster, part in joined.groupby("cluster_label"):
            table = _category_table(part, cube, k, int(cluster), len(part))
            belt_parts.append(table)
            kept = table[table["pct_of_cluster"] >= args.min_share]
            top = table.iloc[0]
            mixture_rows.append(
                {
                    "cohort": cube,
                    "k": k,
                    "cluster_label": int(cluster),
                    "n_frames": int(len(part)),
                    "n_belt_categories": int(len(table)),
                    "n_categories_ge_5pct": int(len(kept)),
                    "top_belt": top["belt"],
                    "top_belt_pct": float(top["pct_of_cluster"]),
                    "top_belt_pct_persistent": float(top["pct_frames_in_dwells_ge_20ps"]),
                }
            )
            sub = modes[modes["cluster_label"] == int(cluster)]
            total = int(sub["n_frames"].sum()) if not sub.empty else 0
            seg_rows = []
            for belt, grp in sub.groupby("belt_mode", sort=False):
                frames = int(grp["n_frames"].sum())
                share = 100.0 * frames / total if total else 0.0
                if share < args.min_share:
                    continue
                purity = float(np.average(grp["belt_mode_share"], weights=grp["n_frames"]))
                openings = grp.groupby("opening_mode")["n_frames"].sum()
                openings = openings / openings.sum()
                seg_rows.append(
                    {
                        "cohort": cube,
                        "k": k,
                        "cluster_label": int(cluster),
                        "belt": belt,
                        "n_segments": int(len(grp)),
                        "n_frames": frames,
                        "pct_of_cluster": share,
                        "mean_majority_share": purity,
                        "pct_0_poles": 100.0 * float(openings.get("0 poles open", 0.0)),
                        "pct_1_pole": 100.0 * float(openings.get("1 pole open", 0.0)),
                        "pct_2_poles": 100.0 * float(openings.get("2 poles open", 0.0)),
                        "pct_3plus_poles": 100.0 * float(openings.get("3+ poles open", 0.0)),
                    }
                )
            seg_table = pd.DataFrame(seg_rows)
            if not seg_table.empty:
                seg_table = seg_table.sort_values("pct_of_cluster", ascending=False)
                segment_parts.append(seg_table)
            brief = ", ".join(
                f"{rec.belt} {rec.pct_of_cluster:.0f}% of segments"
                f" (majority {100 * rec.mean_majority_share:.0f}%;"
                f" poles 0/1/2/3+ ="
                f" {rec.pct_0_poles:.0f}/{rec.pct_1_pole:.0f}/{rec.pct_2_poles:.0f}/{rec.pct_3plus_poles:.0f})"
                for rec in seg_table.itertuples(index=False)
            )
            print(f"  cluster {int(cluster)}: {brief}", flush=True)

    belts = pd.concat(belt_parts, ignore_index=True)
    segments_out = pd.concat(segment_parts, ignore_index=True) if segment_parts else pd.DataFrame()
    mixture = pd.DataFrame(mixture_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    belts.to_csv(args.out_dir / "belt_inside_cluster.csv", index=False)
    segments_out.to_csv(args.out_dir / "segment_belt.csv", index=False)
    mixture.to_csv(args.out_dir / "cluster_mixture.csv", index=False)
    print(f"Wrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
