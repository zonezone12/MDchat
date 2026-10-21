#!/usr/bin/env python3
"""One row per metastable structure, with the numbers quoted in docs/metastable_structures.md §7.

Combines what the two catalog scripts wrote:

* ``rank_metastable_structures.py``: structures (orientation families), their
  clusters, apo-frame shares and best-relabelled r;
* ``categorize_cluster_mixtures.py --min-share 0``: every changepoint
  segment's majority belt and pole count, with no reporting floor, so the
  shares below add up to 100% of each structure;

plus the replicas behind each structure (union over its clusters' segments)
and the top site pairs of ``rank_endpoint_pairs_at_k.py`` mapped onto
structures.

Writes, under ``--out-dir``:

  structures_summary.csv   per structure: share, orientations, segments, replicas,
                           Murata label mix (apo frames), segment-majority belt
                           shares and the pole count inside broken / intact segments
  structure_markers.csv    the top site pairs per cube, with the structure each one
                           marks and the relation of its two monomers

Site indices (``S3`` in ``M2S3-M5S3``) come from each cube's hull detection and
are not chemical roles, so markers are compared across cubes by monomer pair
and relation only.

Example
-------
python scripts/categorize_cluster_mixtures.py --min-share 0 \\
    --out-dir output/metastable_structures_physical_k/min_share_0
python scripts/summarize_metastable_structures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.ChangepointAnalysis.cluster_k_diagnostics import load_segments_at_k
from src.ChangepointAnalysis.metastable_catalog import belt_kind, contact_pairs, pair_relation
from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir

POLE_BINS = ("pct_0_poles", "pct_1_pole", "pct_2_poles", "pct_3plus_poles")
MURATA = ("pct_A", "pct_B", "pct_C1", "pct_C2", "pct_other")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize metastable structures for the methods note.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=ROOT / "output" / "metastable_structures_physical_k",
        help="Folder with clusters_ranked.csv and families_ranked.csv",
    )
    p.add_argument(
        "--segment-belt",
        type=Path,
        default=ROOT / "output" / "metastable_structures_physical_k" / "min_share_0" / "segment_belt.csv",
        help="segment_belt.csv written with --min-share 0",
    )
    p.add_argument(
        "--top-pairs",
        type=Path,
        default=ROOT / "output" / "endpoint_pair_ranking_physical_k" / "top_pairs.csv",
    )
    p.add_argument("--out-dir", type=Path, default=None, help="Default: --catalog-dir")
    return p.parse_args()


def _pole_mix(rows: pd.DataFrame) -> dict[str, float]:
    n = float(rows["n_frames"].sum())
    if n <= 0:
        return {col: np.nan for col in POLE_BINS}
    return {col: float((rows[col] * rows["n_frames"]).sum() / n) for col in POLE_BINS}


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or args.catalog_dir
    clusters = pd.read_csv(args.catalog_dir / "clusters_ranked.csv")
    families = pd.read_csv(args.catalog_dir / "families_ranked.csv")
    if not args.segment_belt.is_file():
        print(
            f"Missing {args.segment_belt}; run categorize_cluster_mixtures.py --min-share 0 first",
            flush=True,
        )
        return 1
    seg = pd.read_csv(args.segment_belt)
    family_of = {(r.cohort, int(r.cluster_label)): int(r.family_id) for r in clusters.itertuples(index=False)}
    seg["family_id"] = [family_of[(c, int(l))] for c, l in zip(seg["cohort"], seg["cluster_label"])]
    seg["kind"] = seg["belt"].map(belt_kind)
    pole_pairs = contact_pairs(clusters.columns, "pct_pi_open_")
    belt_pairs = contact_pairs(clusters.columns, "pct_d1_elongated_")

    rows = []
    replicas: dict[tuple[str, int], int] = {}
    for cube, part in clusters.groupby("cohort", sort=False):
        k = int(part["k"].iloc[0])
        segs = load_segments_at_k(endpoint_changepoints_dir(args.output_root, cube, own_elbow=True), k)
        segs["family_id"] = segs["cluster_label"].astype(int).map(
            {int(l): family_of[(cube, int(l))] for l in part["cluster_label"]}
        )
        for fam, n in segs.groupby("family_id")["traj_id"].nunique().items():
            replicas[(cube, int(fam))] = int(n)

    for fam in families.itertuples(index=False):
        members = clusters[(clusters["cohort"] == fam.cohort) & (clusters["family_id"] == fam.family_id)]
        weights = members["n_apo_frames"].to_numpy(dtype=float)
        s = seg[(seg["cohort"] == fam.cohort) & (seg["family_id"] == fam.family_id)]
        total = float(s["n_frames"].sum())
        row = {
            "cohort": fam.cohort,
            "k": int(fam.k),
            "rank_in_cube": int(fam.rank_in_cube),
            "family_id": int(fam.family_id),
            "structure": fam.structure,
            "member_clusters": fam.member_clusters,
            "pct_of_cube": float(fam.pct_of_cube),
            "n_orientations": int(fam.n_orientations),
            "min_r_in_family": float(fam.min_r_in_family),
            "n_segments": int(fam.n_segments),
            "n_replicas": replicas.get((fam.cohort, int(fam.family_id)), 0),
            "single_replica": bool(fam.single_replica),
        }
        for col in MURATA:
            row[col] = float(np.average(members[col], weights=weights))
        for kind in ("intact", "one_edge", "two_plus_edges"):
            row[f"pct_frames_majority_{kind}"] = (
                100.0 * float(s.loc[s["kind"] == kind, "n_frames"].sum()) / total if total else np.nan
            )
        broken = _pole_mix(s[s["kind"] != "intact"])
        intact = _pole_mix(s[s["kind"] == "intact"])
        row.update({f"broken_{col}": val for col, val in broken.items()})
        row.update({f"intact_{col}": val for col, val in intact.items()})
        row["majority_cover_pct"] = (
            100.0 * float(np.average(s["mean_majority_share"], weights=s["n_frames"])) if total else np.nan
        )
        rows.append(row)
    summary = pd.DataFrame(rows)

    top = pd.read_csv(args.top_pairs)
    top = top[top["cohort"].isin(clusters["cohort"].unique())].copy()
    top["family_id"] = [family_of.get((c, int(b)), -1) for c, b in zip(top["cohort"], top["best_cluster"])]
    names = {(r.cohort, int(r.family_id)): r.structure for r in families.itertuples(index=False)}
    top["structure"] = [names.get((c, f), "") for c, f in zip(top["cohort"], top["family_id"])]
    top["pair_relation"] = [
        pair_relation(i, j, pole_pairs, belt_pairs) for i, j in zip(top["monomer_i"], top["monomer_j"])
    ]
    top["direction"] = np.where(top["cohens_d_best"] > 0, "longer", "shorter")
    markers = top[
        [
            "cohort", "k", "rank", "endpoint_label", "monomer_i", "monomer_j", "pair_relation",
            "eta_squared", "eta_squared_q_value", "best_cluster", "family_id", "structure",
            "cohens_d_best", "direction",
        ]
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("structures_summary.csv", summary), ("structure_markers.csv", markers)):
        table.to_csv(out_dir / name, index=False)
        print(f"wrote {out_dir / name}", flush=True)
    pd.set_option("display.width", 250)
    show = [
        "cohort", "rank_in_cube", "structure", "pct_of_cube", "n_orientations", "min_r_in_family",
        "n_segments", "n_replicas", "pct_frames_majority_intact", "pct_frames_majority_one_edge",
        "broken_pct_2_poles", "broken_pct_3plus_poles", "majority_cover_pct",
    ]
    print(summary[show].round(1).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
