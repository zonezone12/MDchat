#!/usr/bin/env python3
"""Per-cluster motif profile, and the site pairs that separate a chosen set of clusters.

For one cube at one Ward cut, writes to ``--out-dir``:

  cluster_motif_profile.csv  per cluster, over apo frames: Murata label shares and,
                             per edge, the share of frames with d1 elongated
                             (>= 7.0 Å) and pole cation–π open (>= 6.5 Å)
  single_elongated_edge.csv  per cluster: which edge is elongated in frames with
                             exactly one elongated d1
  subset_pair_ranking.csv    η², permutation p and BH-FDR among --clusters only,
                             with each monomer pair's relation (same pole,
                             equatorial, opposite)
  subset_pair_means.csv      per-cluster mean distance (Å) of the top subset pairs
  pair_deformation.csv       per contrasted cluster and monomer pair: mean change (Å)
                             of its site pairs from the reference clusters
  symmetry_profile_correlation.csv
                             for each two contrasted clusters, Pearson r of their
                             deformation profiles as labelled and after the best
                             monomer relabelling that preserves the pole and d1
                             contact graphs

``--clusters`` defaults to the belt-broken clusters, whose apo frames mostly
have at least one elongated d1. The reference for deformations is every other
cluster.

Example
-------
python scripts/describe_endpoint_clusters.py --own-elbow --cube BHHpM --k 4
"""

from __future__ import annotations

import argparse
import re
import sys
from itertools import combinations, permutations
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
    chosen_k_from_cohort,
    load_segments_at_k,
    map_frames_to_segment_clusters,
    resolve_endpoint_features_dir,
)
from src.ChangepointAnalysis.murata_criteria import CATION_PI_OPEN_LO, METASTRUCTURES
from src.ChangepointAnalysis.murata_d1 import D1_ELONGATED_LO
from src.ChangepointAnalysis.murata_rmsd import (
    discover_motif_rmsd_csvs_by_cube,
    filter_non_encapsulated,
    label_motif_metastructures,
    load_motif_rmsd_csvs,
    unit_series_columns,
)
from src.ChangepointAnalysis.reporting import (
    collect_endpoint_segment_means,
    endpoint_changepoints_dir,
    score_cluster_discriminating_features,
)

_EDGE_RE = re.compile(r"_e(\d+)_m(\d+)[a-z0-9]*_m(\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Describe endpoint clusters by motif state and site pairs.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--cube", required=True)
    p.add_argument(
        "--own-elbow",
        action="store_true",
        help="Read endpoint clusters from the own-elbow detections (OWN_ELBOW_ENDPOINT_DIRS)",
    )
    p.add_argument("--k", type=int, help="Ward cut (default: the stored clustering)")
    p.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        help="Clusters to contrast with each other (default: the belt-broken clusters)",
    )
    p.add_argument(
        "--qc-gsa-root",
        type=Path,
        default=ROOT / "output" / "gsa_features_by_cube",
        help="Root of per-cube *_gsa_features.csv folders used by the assembly QC rule",
    )
    p.add_argument("--no-assembly-qc", action="store_true")
    p.add_argument("--n-permutations", type=int, default=999)
    p.add_argument("--fdr-alpha", type=float, default=0.05)
    p.add_argument("--top-n", type=int, default=15)
    p.add_argument(
        "--out-dir",
        type=Path,
        help="Default: {output-root}/endpoint_cluster_profiles/{CUBE}_kXX",
    )
    return p.parse_args()


def _edge_monomers(cols: list[str]) -> dict[str, tuple[int, int, int]]:
    """Motif column -> (edge, first monomer, second monomer)."""
    out: dict[str, tuple[int, int, int]] = {}
    for col in cols:
        m = _EDGE_RE.search(col)
        if m:
            out[col] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return out


def _edge_label(prefix: str, edge: tuple[int, int, int]) -> str:
    e, a, b = edge
    return f"{prefix}_e{e}_M{a}-M{b}"


def contact_automorphisms(*edge_maps: dict[str, tuple[int, int, int]]) -> list[tuple[int, ...]]:
    """Monomer permutations that map each directed contact set onto itself."""
    edge_sets = [{(a, b) for _, a, b in edges.values()} for edges in edge_maps]
    monomers = sorted({m for edges in edge_sets for pair in edges for m in pair})
    found = []
    for image in permutations(monomers):
        perm = dict(zip(monomers, image))
        if all({(perm[a], perm[b]) for a, b in edges} == edges for edges in edge_sets):
            found.append(tuple(perm[m] for m in monomers))
    return found


def _cycles(perm: tuple[int, ...]) -> str:
    seen: set[int] = set()
    parts = []
    for start in range(len(perm)):
        if start in seen or perm[start] == start:
            continue
        cycle = [start]
        seen.add(start)
        nxt = perm[start]
        while nxt != start:
            cycle.append(nxt)
            seen.add(nxt)
            nxt = perm[nxt]
        parts.append("(" + " ".join(str(m) for m in cycle) + ")")
    return "".join(parts) or "identity"


def deformation_tables(
    X: np.ndarray,
    y: np.ndarray,
    specs: list[tuple],
    clusters: list[int],
    relation,
    automorphisms: list[tuple[int, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-pair mean change from the reference clusters, and symmetry-matched profile r."""
    ref = ~np.isin(y, clusters)
    base = X[ref].mean(axis=0) if ref.any() else X[np.isin(y, clusters)].mean(axis=0)
    delta = {c: X[y == c].mean(axis=0) - base for c in clusters}
    col_of = {}
    for col, (_name, i, a, j, b) in enumerate(specs):
        col_of[(i, a, j, b)] = col
        col_of[(j, b, i, a)] = col

    pair_cols: dict[tuple[int, int], list[int]] = {}
    for col, (_name, i, _a, j, _b) in enumerate(specs):
        pair_cols.setdefault((min(i, j), max(i, j)), []).append(col)
    rows = []
    for c in clusters:
        for (i, j), cols in sorted(pair_cols.items()):
            rows.append(
                {
                    "cluster_label": c,
                    "monomer_pair": f"M{i}-M{j}",
                    "pair_relation": relation(i, j),
                    "mean_delta_A": float(delta[c][cols].mean()),
                    "mean_abs_delta_A": float(np.abs(delta[c][cols]).mean()),
                }
            )
    pairs = pd.DataFrame(rows)

    sym_rows = []
    for a, b in combinations(clusters, 2):
        scores = []
        for perm in automorphisms:
            mapped = np.array([col_of.get((perm[i], s, perm[j], t), -1) for _n, i, s, j, t in specs])
            if (mapped < 0).any():
                continue
            scores.append((float(np.corrcoef(delta[a], delta[b][mapped])[0, 1]), perm))
        identity = tuple(range(len(automorphisms[0]))) if automorphisms else ()
        r_identity = next((r for r, p in scores if p == identity), float("nan"))
        r_best, best = max(scores) if scores else (float("nan"), identity)
        sym_rows.append(
            {
                "cluster_a": a,
                "cluster_b": b,
                "n_automorphisms": len(automorphisms),
                "r_as_labelled": r_identity,
                "best_relabelling": _cycles(best),
                "r_best_relabelling": r_best,
            }
        )
    return pairs, pd.DataFrame(sym_rows)


def load_apo_frames(args: argparse.Namespace) -> pd.DataFrame:
    paths = discover_motif_rmsd_csvs_by_cube(args.output_root, cubes=[args.cube]).get(args.cube, [])
    if not paths:
        raise FileNotFoundError(f"No {args.cube} *_motif_rmsd.csv under {args.output_root}")
    df = load_motif_rmsd_csvs(paths)
    if not args.no_assembly_qc:
        gsa_dir = args.qc_gsa_root / args.cube
        if gsa_dir.is_dir():
            failed = assembly_qc_failures(scan_assembly_qc_dir(gsa_dir))
            if failed:
                print(f"  assembly QC excludes {', '.join(failed)}", flush=True)
                df = drop_trajectories(df, failed)
        else:
            print(f"  assembly QC skipped: {gsa_dir} not found", flush=True)
    return filter_non_encapsulated(label_motif_metastructures(df), traj_filter="frames")


def motif_profile(
    joined: pd.DataFrame,
    segs: pd.DataFrame,
    elongated: pd.DataFrame,
    opened: pd.DataFrame,
    d1_edges: dict[str, tuple[int, int, int]],
    pi_edges: dict[str, tuple[int, int, int]],
) -> pd.DataFrame:
    labels = joined["cluster_label"].astype(int)
    seg_labels = segs["cluster_label"].astype(int)
    top_share = segs.groupby(seg_labels)["traj_id"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    meta = pd.crosstab(labels, joined["murata_metastructure"], normalize="index")
    meta = meta.reindex(columns=list(METASTRUCTURES), fill_value=0.0)
    parts = [
        seg_labels.value_counts().sort_index().rename("n_segments"),
        segs.groupby(seg_labels)["traj_id"].nunique().rename("n_traj"),
        top_share.rename("top_traj_share_of_segments"),
        labels.value_counts().sort_index().rename("n_apo_frames"),
        (100.0 * meta).add_prefix("pct_"),
        (100.0 * elongated.any(axis=1).groupby(labels).mean()).rename("pct_any_d1_elongated"),
        (100.0 * elongated.groupby(labels).mean()).rename(
            columns={c: _edge_label("pct_d1_elongated", d1_edges[c]) for c in elongated.columns}
        ),
        (100.0 * opened.groupby(labels).mean()).rename(
            columns={c: _edge_label("pct_pi_open", pi_edges[c]) for c in opened.columns}
        ),
    ]
    return pd.concat(parts, axis=1).rename_axis("cluster_label").reset_index()


def single_elongated_edge(
    joined: pd.DataFrame, elongated: pd.DataFrame, d1_edges: dict[str, tuple[int, int, int]]
) -> pd.DataFrame:
    one = elongated.sum(axis=1) == 1
    edge = elongated[one].idxmax(axis=1).map(lambda c: _edge_label("d1", d1_edges[c]))
    labels = joined.loc[one, "cluster_label"].astype(int)
    share = 100.0 * pd.crosstab(labels, edge, normalize="index")
    share.insert(0, "n_frames_one_elongated", labels.value_counts().sort_index())
    return share.rename_axis("cluster_label").reset_index()


def main() -> int:
    args = parse_args()
    cp_dir = endpoint_changepoints_dir(args.output_root, args.cube, own_elbow=args.own_elbow)
    k = args.k or chosen_k_from_cohort(cp_dir, group="endpoint")
    segs = load_segments_at_k(cp_dir, k, group="endpoint")
    if segs.empty:
        print(f"{args.cube}: no endpoint clustering at k={k} in {cp_dir}", flush=True)
        return 1
    if "group" in segs.columns:
        segs = segs[segs["group"].astype(str) == "endpoint"]
    out_dir = args.out_dir or args.output_root / "endpoint_cluster_profiles" / f"{args.cube}_k{k:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{args.cube} k={k}: labelling apo motif frames …", flush=True)
    joined = map_frames_to_segment_clusters(load_apo_frames(args), segs, cube=args.cube)
    joined = joined.dropna(subset=["cluster_label"])
    joined = joined[joined["murata_metastructure"].notna()]
    d1_cols = unit_series_columns(joined, "equator_d1_e")
    pi_cols = unit_series_columns(joined, "cation_pi_pole_e")
    d1_edges = _edge_monomers(d1_cols)
    pi_edges = _edge_monomers(pi_cols)
    elongated = joined[d1_cols].apply(pd.to_numeric, errors="coerce") >= D1_ELONGATED_LO
    opened = joined[pi_cols].apply(pd.to_numeric, errors="coerce") >= CATION_PI_OPEN_LO

    profile = motif_profile(joined, segs, elongated, opened, d1_edges, pi_edges)
    profile.to_csv(out_dir / "cluster_motif_profile.csv", index=False)
    single = single_elongated_edge(joined, elongated, d1_edges)
    single.to_csv(out_dir / "single_elongated_edge.csv", index=False)

    clusters = args.clusters
    if not clusters:
        clusters = profile.loc[profile["pct_any_d1_elongated"] > 50.0, "cluster_label"].astype(int).tolist()
    if len(clusters) < 2:
        print(f"Need at least two clusters to contrast; got {clusters}", flush=True)
        return 1
    print(f"ranking site pairs among clusters {clusters} …", flush=True)
    X, y, _keys, specs, kind, _ = collect_endpoint_segment_means(
        cp_dir,
        resolve_endpoint_features_dir(cp_dir),
        group="endpoint",
        clustered_df=segs,
        include_frame_corr=False,
    )
    mask = np.isin(y, clusters)
    ranking = score_cluster_discriminating_features(
        X[mask],
        y[mask],
        specs,
        kind,
        n_permutations=args.n_permutations,
        fdr_alpha=args.fdr_alpha,
    )
    pole_pairs = {frozenset((a, b)) for _, a, b in pi_edges.values()}
    belt_pairs = {frozenset((a, b)) for _, a, b in d1_edges.values()}

    def relation(i: int, j: int) -> str:
        pair = frozenset((int(i), int(j)))
        if len(pair) == 1:
            return "same monomer"
        if pair in pole_pairs:
            return "same pole"
        return "equatorial" if pair in belt_pairs else "opposite"

    ranking.insert(
        ranking.columns.get_loc("endpoint_label") + 1,
        "pair_relation",
        [relation(i, j) for i, j in zip(ranking["monomer_i"], ranking["monomer_j"])],
    )
    ranking.insert(0, "contrasted_clusters", ",".join(str(c) for c in clusters))
    ranking.to_csv(out_dir / "subset_pair_ranking.csv", index=False)

    cols = [str(spec[0]) for spec in specs]
    means = pd.DataFrame(X[mask], columns=cols).groupby(y[mask]).mean()
    top = ranking.head(args.top_n)
    pair_means = means[top["feature"].tolist()].T
    pair_means.columns = [f"mean_A_cluster_{c}" for c in pair_means.columns]
    keep = ["endpoint_label", "pair_relation", "eta_squared", "eta_squared_q_value", "best_cluster", "cohens_d_best"]
    pair_means = pd.concat([top.set_index("feature")[keep], pair_means], axis=1).reset_index()
    pair_means.to_csv(out_dir / "subset_pair_means.csv", index=False)

    tables = [profile.round(1), single.round(1), pair_means.round(3)]
    if kind == "site_pair":
        automorphisms = contact_automorphisms(pi_edges, d1_edges)
        pairs, sym = deformation_tables(X, y, specs, clusters, relation, automorphisms)
        pairs.to_csv(out_dir / "pair_deformation.csv", index=False)
        sym.to_csv(out_dir / "symmetry_profile_correlation.csv", index=False)
        wide = pairs.pivot_table(
            index=["monomer_pair", "pair_relation"], columns="cluster_label", values="mean_delta_A"
        )
        tables += [wide.round(2).reset_index(), sym.round(3)]

    with pd.option_context("display.width", 220, "display.max_columns", 60):
        for table in tables:
            print(table.to_string(index=False))
    print(f"Wrote {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
