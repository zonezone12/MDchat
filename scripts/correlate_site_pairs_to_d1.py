#!/usr/bin/env python3
"""Pearson r of contact-signature site pairs with Murata d1.

The global top 10 site pairs miss some clusters. For each cluster in the
panel (b) catalog, this script keeps the three site pairs assigned to that
cluster with the largest η², breaking ties by |Cohen's d|. Each r is the
correlation, across endpoint segments at the selected k, of that site
pair's segment-mean distance with one Murata equatorial d1 segment mean.
The cluster label is the panel (b) tick from ``plot_contact_signatures.py``.

One CSV per cube is written under ``docs/figures/si/``.

Example
-------
python scripts/correlate_site_pairs_to_d1.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.cluster_k_diagnostics import (
    load_segments_at_k,
    load_selected_k,
    resolve_endpoint_features_dir,
)
from src.ChangepointAnalysis.feature_groups import resolve_features_csv
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.metastable_catalog import (
    DEFAULT_D1_EDGES,
    DEFAULT_POLE_EDGES,
    pair_relation,
)
from src.ChangepointAnalysis.murata_rmsd import discover_motif_rmsd_csvs_by_cube
from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir
from src.utils.run_log import RunContext

EDGE_TICK_PCT = 30.0
MOTIF_SUFFIX = "_motif_rmsd.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Correlate contact-signature site pairs with Murata d1."
    )
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=ROOT / "output" / "metastable_structures_physical_k",
    )
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "figures" / "si",
    )
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument(
        "--per-cluster",
        type=int,
        default=3,
        help="Site pairs kept for each cluster, sorted by η² then |Cohen's d|",
    )
    return p.parse_args()


def _state_tag(structure: str) -> str:
    if structure.startswith("compact cage"):
        return "compact cage"
    if structure.startswith("mostly compact"):
        return "mostly compact"
    if structure.startswith("intact"):
        return "intact belt"
    if structure.startswith("loosened"):
        return "one replica"
    if structure.startswith("mixed belt"):
        return "mixed belt"
    if structure.startswith(("belt broken", "one belt break", "opened", "partly")):
        return "belt break"
    return structure


def panel_b_label(clusters: pd.DataFrame, belts: pd.DataFrame, cube: str, cluster: int) -> str:
    """Tick text used on panel (b) for this cluster, without the line break."""
    row = clusters[(clusters["cohort"] == cube) & (clusters["cluster_label"] == int(cluster))].iloc[0]
    sub = belts[(belts["cohort"] == cube) & (belts["cluster_label"] == int(cluster))]
    single = sub[sub["belt"].astype(str).str.startswith("edge ")]
    edge, pct = "", 0.0
    if not single.empty:
        top = single.sort_values("n_frames", ascending=False).iloc[0]
        edge = str(top["belt"]).removeprefix("edge ").replace("-", "–")
        pct = float(top["pct_of_cluster"])
    in_family = int(row["n_orientations"]) > 1
    tag = edge if edge and (in_family or pct >= EDGE_TICK_PCT) else _state_tag(str(row["structure"]))
    return f"{int(cluster)}, {tag}"


def _traj_key(name: str, cube: str) -> str:
    text = str(name)
    prefix = f"{cube}_"
    return text[len(prefix) :] if text.startswith(prefix) else text


def motif_by_traj(output_root: Path, cube: str) -> dict[str, Path]:
    found = {}
    for path in discover_motif_rmsd_csvs_by_cube(output_root, cubes=[cube]).get(cube, []):
        found[_traj_key(path.name[: -len(MOTIF_SUFFIX)], cube)] = path
    return found


def d1_edges(units_path: Path) -> list[tuple[str, str]]:
    """Equatorial d1 columns as ``(M0–M4, equator_d1_e0_...)``, in edge order."""
    units = pd.read_csv(units_path)
    equator = units[units["kind"].astype(str) == "equator"].sort_values("edge")
    edges = []
    for rec in equator.itertuples(index=False):
        edges.append((f"M{int(rec.mon_a)}–M{int(rec.mon_b)}", str(rec.distance_column)))
    if len(edges) != 6:
        raise ValueError(f"{units_path} has {len(edges)} equatorial d1 edges")
    return edges


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def segment_table(
    segments: pd.DataFrame,
    motif_paths: dict[str, Path],
    features_dir: Path,
    edges: list[tuple[str, str]],
    site_features: list[str],
) -> tuple[pd.DataFrame, int]:
    """One row per segment: d1 means and the panel-(a) site-pair means."""
    d1_cols = [col for _, col in edges]
    rows = []
    omitted = 0
    grouped = segments.groupby(segments["traj_id"].astype(str), sort=False)
    for traj, part in grouped:
        motif_path = motif_paths.get(traj) or motif_paths.get(_traj_key(traj, ""))
        feat_path = resolve_features_csv(features_dir, traj)
        if motif_path is None or feat_path is None:
            omitted += len(part)
            continue
        motif = pd.read_csv(motif_path, usecols=["frame", *d1_cols])
        features = pd.read_csv(feat_path, usecols=["frame", *site_features])
        for rec in part.itertuples(index=False):
            mf = motif[(motif["frame"] >= rec.start_frame) & (motif["frame"] <= rec.end_frame)]
            ef = features[(features["frame"] >= rec.start_frame) & (features["frame"] <= rec.end_frame)]
            if mf.empty or ef.empty:
                omitted += 1
                continue
            row = {"cluster_label": int(rec.cluster_label)}
            for label, col in edges:
                row[label] = float(mf[col].mean())
            for col in site_features:
                row[col] = float(ef[col].mean())
            rows.append(row)
    return pd.DataFrame(rows), omitted


def pairs_for_clusters(ranking: pd.DataFrame, cluster_labels: list[int], per_cluster: int) -> pd.DataFrame:
    """Three (or ``per_cluster``) site pairs per cluster, by η² then |Cohen's d|."""
    work = ranking
    if "feature_kind" in work.columns:
        work = work[work["feature_kind"].astype(str) == "site_pair"]
    work = work.copy()
    work["abs_d"] = work["cohens_d_best"].abs()
    pole = {frozenset(pair) for pair in DEFAULT_POLE_EDGES}
    belt = {frozenset(pair) for pair in DEFAULT_D1_EDGES}
    picked = []
    for label in cluster_labels:
        part = work[work["best_cluster"].astype(int) == int(label)]
        part = part.sort_values(["eta_squared", "abs_d"], ascending=[False, False]).head(per_cluster)
        if part.empty:
            continue
        part = part.copy()
        part["rank_in_cluster"] = range(1, len(part) + 1)
        part["pair_relation"] = [
            pair_relation(i, j, pole, belt) for i, j in zip(part["monomer_i"], part["monomer_j"])
        ]
        picked.append(part)
    if not picked:
        return pd.DataFrame()
    return pd.concat(picked, ignore_index=True)


def correlate_cube(
    cube: str,
    *,
    k: int,
    output_root: Path,
    clusters: pd.DataFrame,
    belts: pd.DataFrame,
    per_cluster: int,
) -> pd.DataFrame:
    cp_dir = endpoint_changepoints_dir(output_root, cube, own_elbow=True)
    segments = load_segments_at_k(cp_dir, k, group="endpoint")
    if "group" in segments.columns:
        segments = segments[segments["group"].astype(str) == "endpoint"]
    ranking_path = cp_dir / "clusters" / "endpoint" / "by_k" / f"k_{k:02d}" / "endpoint_pair_cluster_correlation.csv"
    ranking = pd.read_csv(ranking_path)
    cube_clusters = clusters[clusters["cohort"] == cube].sort_values("rank_in_cube")
    cluster_labels = [int(label) for label in cube_clusters["cluster_label"]]
    panel = pairs_for_clusters(ranking, cluster_labels, per_cluster)
    if panel.empty:
        raise FileNotFoundError(f"{cube}: no site pairs in {ranking_path}")
    short = panel.groupby("best_cluster").size()
    for label in cluster_labels:
        n_kept = int(short.get(label, 0))
        if n_kept < per_cluster:
            print(f"  {cube} cluster {label}: {n_kept} site pairs (asked for {per_cluster})")
    site_features = panel["feature"].tolist()
    motif_paths = motif_by_traj(output_root, cube)
    if not motif_paths:
        raise FileNotFoundError(f"No {cube} motif RMSD files under {output_root}")
    units = next(iter(motif_paths.values())).with_name(
        next(iter(motif_paths.values())).name.replace(MOTIF_SUFFIX, "_motif_units.csv")
    )
    edges = d1_edges(units)
    means, omitted = segment_table(
        segments,
        motif_paths,
        resolve_endpoint_features_dir(cp_dir),
        edges,
        site_features,
    )
    if means.empty:
        raise RuntimeError(f"{cube}: no segments could be joined to motif and endpoint features")
    edge_labels = [label for label, _ in edges]
    rows = []
    for rec in panel.itertuples(index=False):
        series = means[rec.feature].to_numpy(dtype=float)
        scores = {label: _pearson(series, means[label].to_numpy(dtype=float)) for label in edge_labels}
        strongest = max(scores, key=lambda label: abs(scores[label]))
        row = {
            "cohort": cube,
            "k": int(k),
            "n_segments": int(len(means)),
            "n_segments_omitted": int(omitted),
            "best_cluster": int(rec.best_cluster),
            "panel_b": panel_b_label(clusters, belts, cube, int(rec.best_cluster)),
            "rank_in_cluster": int(rec.rank_in_cluster),
            "endpoint_label": rec.endpoint_label,
            "pair_relation": rec.pair_relation,
            "direction": "longer" if float(rec.cohens_d_best) > 0 else "shorter",
            "eta_squared": float(rec.eta_squared),
            "cohens_d_best": float(rec.cohens_d_best),
            "strongest_d1": strongest,
            "r": scores[strongest],
        }
        for label in edge_labels:
            row[f"r_{label.replace('–', '_')}"] = scores[label]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    selected = load_selected_k(args.k_selected)
    catalog = Path(args.catalog_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters = pd.read_csv(catalog / "clusters_ranked.csv")
    belts = pd.read_csv(catalog / "min_share_0" / "segment_belt.csv")
    with RunContext.from_namespace(args, name="correlate_site_pairs_to_d1"):
        for cube in args.cubes:
            if cube not in selected:
                raise KeyError(f"{cube} is absent from {args.k_selected}")
            table = correlate_cube(
                cube,
                k=int(selected[cube]),
                output_root=Path(args.output_root),
                clusters=clusters,
                belts=belts,
                per_cluster=int(args.per_cluster),
            )
            path = out_dir / f"site_pair_d1_r_{cube}.csv"
            table.to_csv(path, index=False)
            omitted = int(table["n_segments_omitted"].iloc[0])
            note = f", {omitted} segment{'s' if omitted != 1 else ''} omitted" if omitted else ""
            print(f"wrote {path} ({len(table)} pairs, {int(table['n_segments'].iloc[0])} segments{note})")


if __name__ == "__main__":
    main()
