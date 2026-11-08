"""Catalog stages: describe clusters, merge orientations, belt majority, summaries.

These are the analysis bodies previously living in ``scripts/describe_endpoint_clusters.py``,
``rank_metastable_structures.py``, ``categorize_cluster_mixtures.py``,
``summarize_metastable_structures.py``, ``compare_to_murata_metastructures.py``,
and ``list_annotation_replicas.py``.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .assembly_qc import assembly_qc_failures, drop_trajectories, scan_assembly_qc_dir
from .cluster_k_diagnostics import (
    _norm_traj_id,
    chosen_k_from_cohort,
    load_segments_at_k,
    load_selected_k,
    map_frames_to_segment_clusters,
    resolve_endpoint_features_dir,
)
from .cluster_validation import mappings_from_perms, profile_correlation_table
from .gsa_cohort_run import KNOWN_CUBES
from .metastable_catalog import (
    COPY_R,
    belt_kind,
    belt_label,
    contact_automorphisms,
    contact_pairs,
    dwell_stats,
    family_ids,
    opening_label,
    pair_relation,
    perm_cycles,
    site_pair_column_permutation,
    structure_name,
)
from .murata_criteria import CATION_PI_OPEN_LO, COHORT_TO_MURATA_SYSTEM, METASTRUCTURES, occupancy_vs_murata
from .murata_d1 import D1_ELONGATED_LO
from .murata_rmsd import (
    CUBE_COLORS,
    discover_motif_rmsd_csvs_by_cube,
    filter_non_encapsulated,
    label_motif_metastructures,
    load_motif_rmsd_csvs,
    unit_series_columns,
)
from .reporting import (
    collect_endpoint_segment_means,
    endpoint_changepoints_dir,
    score_cluster_discriminating_features,
)

_EDGE_RE = re.compile(r"_e(\d+)_m(\d+)[a-z0-9]*_m(\d+)")
MIN_SHARE = 5.0
MIN_PERSIST_PS = 20
POLE_BINS = ("pct_0_poles", "pct_1_pole", "pct_2_poles", "pct_3plus_poles")
MURATA = ("pct_A", "pct_B", "pct_C1", "pct_C2", "pct_other")
CUBE_ORDER = ("BHHpH", "BHHpM", "BMHpH", "BMHpM", "BMMpH", "BMMpM")
MIN_FLANK_PS = 400.0
MIN_BOUNDARY_PS = 200.0


def _edge_monomers(cols: list[str]) -> dict[str, tuple[int, int, int]]:
    out: dict[str, tuple[int, int, int]] = {}
    for col in cols:
        m = _EDGE_RE.search(col)
        if m:
            out[col] = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return out


def _edge_label(prefix: str, edge: tuple[int, int, int]) -> str:
    e, a, b = edge
    return f"{prefix}_e{e}_M{a}-M{b}"


def _edge_name(column: str) -> tuple[int, str]:
    match = _EDGE_RE.search(column)
    if not match:
        return 0, column
    return int(match.group(1)), f"M{int(match.group(2))}-M{int(match.group(3))}"


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
            mapped = site_pair_column_permutation(specs, perm)
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
                "best_relabelling": perm_cycles(best),
                "r_best_relabelling": r_best,
            }
        )
    return pairs, pd.DataFrame(sym_rows)


def load_apo_motif_frames(
    output_root: Path | str,
    cube: str,
    *,
    qc_gsa_root: Path | str,
    no_assembly_qc: bool = False,
) -> pd.DataFrame:
    paths = discover_motif_rmsd_csvs_by_cube(output_root, cubes=[cube]).get(cube, [])
    if not paths:
        raise FileNotFoundError(f"No {cube} *_motif_rmsd.csv under {output_root}")
    df = load_motif_rmsd_csvs(paths)
    if not no_assembly_qc:
        gsa_dir = Path(qc_gsa_root) / cube
        if gsa_dir.is_dir():
            failed = assembly_qc_failures(scan_assembly_qc_dir(gsa_dir))
            if failed:
                df = drop_trajectories(df, failed)
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


def describe_endpoint_clusters(
    *,
    output_root: Path | str = "output",
    cube: str,
    own_elbow: bool = True,
    k: Optional[int] = None,
    clusters: Optional[Sequence[int]] = None,
    qc_gsa_root: Path | str = "output/gsa_features_by_cube",
    no_assembly_qc: bool = False,
    n_permutations: int = 999,
    fdr_alpha: float = 0.05,
    top_n: int = 15,
    out_dir: Optional[Path | str] = None,
) -> dict[str, Path]:
    """Per-cluster motif profile and the site pairs that separate chosen clusters."""
    output_root = Path(output_root)
    cp_dir = endpoint_changepoints_dir(output_root, cube, own_elbow=own_elbow)
    cut = k or chosen_k_from_cohort(cp_dir, group="endpoint")
    segs = load_segments_at_k(cp_dir, cut, group="endpoint")
    if segs.empty:
        raise FileNotFoundError(f"{cube}: no endpoint clustering at k={cut} in {cp_dir}")
    if "group" in segs.columns:
        segs = segs[segs["group"].astype(str) == "endpoint"]
    out_dir = Path(out_dir) if out_dir is not None else output_root / "endpoint_cluster_profiles" / f"{cube}_k{cut:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    joined = map_frames_to_segment_clusters(
        load_apo_motif_frames(output_root, cube, qc_gsa_root=qc_gsa_root, no_assembly_qc=no_assembly_qc),
        segs,
        cube=cube,
    )
    joined = joined.dropna(subset=["cluster_label"])
    joined = joined[joined["murata_metastructure"].notna()]
    d1_cols = unit_series_columns(joined, "equator_d1_e")
    pi_cols = unit_series_columns(joined, "cation_pi_pole_e")
    d1_edges = _edge_monomers(d1_cols)
    pi_edges = _edge_monomers(pi_cols)
    elongated = joined[d1_cols].apply(pd.to_numeric, errors="coerce") >= D1_ELONGATED_LO
    opened = joined[pi_cols].apply(pd.to_numeric, errors="coerce") >= CATION_PI_OPEN_LO

    profile = motif_profile(joined, segs, elongated, opened, d1_edges, pi_edges)
    written = {"cluster_motif_profile": out_dir / "cluster_motif_profile.csv"}
    profile.to_csv(written["cluster_motif_profile"], index=False)
    single = single_elongated_edge(joined, elongated, d1_edges)
    written["single_elongated_edge"] = out_dir / "single_elongated_edge.csv"
    single.to_csv(written["single_elongated_edge"], index=False)

    contrast = list(clusters) if clusters else profile.loc[
        profile["pct_any_d1_elongated"] > 50.0, "cluster_label"
    ].astype(int).tolist()
    if len(contrast) < 2:
        raise ValueError(f"Need at least two clusters to contrast; got {contrast}")
    X, y, _keys, specs, kind, _ = collect_endpoint_segment_means(
        cp_dir,
        resolve_endpoint_features_dir(cp_dir),
        group="endpoint",
        clustered_df=segs,
        include_frame_corr=False,
    )
    mask = np.isin(y, contrast)
    ranking = score_cluster_discriminating_features(
        X[mask], y[mask], specs, kind, n_permutations=n_permutations, fdr_alpha=fdr_alpha
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
    ranking.insert(0, "contrasted_clusters", ",".join(str(c) for c in contrast))
    written["subset_pair_ranking"] = out_dir / "subset_pair_ranking.csv"
    ranking.to_csv(written["subset_pair_ranking"], index=False)

    cols = [str(spec[0]) for spec in specs]
    means = pd.DataFrame(X[mask], columns=cols).groupby(y[mask]).mean()
    top = ranking.head(top_n)
    pair_means = means[top["feature"].tolist()].T
    pair_means.columns = [f"mean_A_cluster_{c}" for c in pair_means.columns]
    keep = ["endpoint_label", "pair_relation", "eta_squared", "eta_squared_q_value", "best_cluster", "cohens_d_best"]
    pair_means = pd.concat([top.set_index("feature")[keep], pair_means], axis=1).reset_index()
    written["subset_pair_means"] = out_dir / "subset_pair_means.csv"
    pair_means.to_csv(written["subset_pair_means"], index=False)

    if kind == "site_pair":
        automorphisms = contact_automorphisms(pi_edges, d1_edges)
        all_labels = sorted({int(c) for c in y})
        pairs, _subset_sym = deformation_tables(X, y, specs, contrast, relation, automorphisms)
        _all_pairs, sym = deformation_tables(X, y, specs, all_labels, relation, automorphisms)
        written["pair_deformation"] = out_dir / "pair_deformation.csv"
        written["symmetry_profile_correlation"] = out_dir / "symmetry_profile_correlation.csv"
        pairs.to_csv(written["pair_deformation"], index=False)
        sym.to_csv(written["symmetry_profile_correlation"], index=False)
    return written


def profile_dirs(profiles: Path, cube: str, k: int) -> list[Path]:
    stem = f"{cube}_k{k:02d}"
    if not profiles.is_dir():
        return []
    return sorted(
        path
        for path in profiles.iterdir()
        if path.is_dir() and (path.name == stem or path.name.startswith(stem + "_"))
    )


def _read_motif(dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for path in dirs:
        csv = path / "cluster_motif_profile.csv"
        if csv.is_file():
            frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError("no cluster_motif_profile.csv")
    frames.sort(key=lambda frame: (-len(frame), len(frame.columns)))
    return frames[0]


def all_cluster_symmetry(
    output_root: Path,
    cube: str,
    k: int,
    *,
    own_elbow: bool,
    qc_gsa_root: Path,
    no_assembly_qc: bool,
) -> pd.DataFrame:
    """Best-relabelled profile r for every pair of clusters of this cube."""
    cp_dir = endpoint_changepoints_dir(output_root, cube, own_elbow=own_elbow)
    segs = load_segments_at_k(cp_dir, k, group="endpoint")
    if segs.empty:
        raise FileNotFoundError(f"{cube}: no endpoint clustering at k={k} in {cp_dir}")
    if not no_assembly_qc:
        gsa_dir = qc_gsa_root / cube
        if gsa_dir.is_dir():
            failed = assembly_qc_failures(scan_assembly_qc_dir(gsa_dir))
            if failed:
                segs = drop_trajectories(segs, failed)
    X, y, _keys, specs, kind, _ = collect_endpoint_segment_means(
        cp_dir,
        resolve_endpoint_features_dir(cp_dir),
        group="endpoint",
        clustered_df=segs,
        include_frame_corr=False,
    )
    if kind != "site_pair" or X.size == 0:
        raise FileNotFoundError(f"{cube}: no site-pair segment means")
    perms = contact_automorphisms()
    mappings = mappings_from_perms(specs, perms)
    return profile_correlation_table(X, y, mappings, perms)


def _replica_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.is_file():
        return set()
    audit = pd.read_csv(path)
    return {
        (str(row.cohort), int(row.k), int(row.cluster_label))
        for row in audit.itertuples(index=False)
    }


def _is_belt_break(name: str) -> bool:
    return name.startswith("belt broken") or name.startswith("partly belt broken")


def _family_name(members: pd.DataFrame) -> str:
    names = [str(name) for name in members["structure"]]
    if len(members) == 1:
        return names[0]
    if all(_is_belt_break(name) for name in names):
        return f"one belt break, {len(members)} orientations"
    elongated = float(
        (members["pct_any_d1_elongated"] * members["n_apo_frames"]).sum() / members["n_apo_frames"].sum()
    )
    if elongated >= 20.0:
        return f"opened cage, {len(members)} orientations"
    return " + ".join(dict.fromkeys(names))


def rank_cube(
    cube: str,
    k: int,
    motif: pd.DataFrame,
    symmetry: pd.DataFrame,
    replicas: set[tuple[str, int, int]],
    copy_r: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = motif.copy()
    work["cluster_label"] = work["cluster_label"].astype(int)
    total = float(work["n_apo_frames"].sum())
    work["pct_of_cube"] = 100.0 * work["n_apo_frames"] / total
    labels = work["cluster_label"].tolist()
    pairs = [
        (int(row.cluster_a), int(row.cluster_b), float(row.r_best_relabelling))
        for row in symmetry.itertuples(index=False)
    ]
    families = family_ids(labels, pairs, threshold=copy_r)
    r_of = {}
    for a, b, score in pairs:
        r_of[(a, b)] = score
        r_of[(b, a)] = score

    rows = []
    for record in work.to_dict(orient="records"):
        label = int(record["cluster_label"])
        replica = (cube, k, label) in replicas or (
            int(record["n_traj"]) == 1 and float(record["top_traj_share_of_segments"]) >= 0.5
        )
        family = families[label]
        mates = [other for other, head in families.items() if head == family and other != label]
        mate_r = [r_of[(label, other)] for other in mates if (label, other) in r_of]
        record.update(
            {
                "cohort": cube,
                "k": k,
                "single_replica": replica,
                "structure": structure_name(record, single_replica=replica),
                "family_id": family,
                "n_orientations": 1 + len(mates),
                "min_r_in_family": min(mate_r) if mate_r else float("nan"),
            }
        )
        rows.append(record)
    clusters = pd.DataFrame(rows).sort_values("pct_of_cube", ascending=False)
    clusters.insert(0, "rank_in_cube", range(1, len(clusters) + 1))

    family_rows = []
    for family, members in clusters.groupby("family_id", sort=False):
        members = members.sort_values("pct_of_cube", ascending=False)
        family_rows.append(
            {
                "cohort": cube,
                "k": int(k),
                "family_id": int(family),
                "member_clusters": " ".join(str(int(c)) for c in members["cluster_label"]),
                "n_orientations": int(len(members)),
                "n_apo_frames": int(members["n_apo_frames"].sum()),
                "pct_of_cube": float(members["pct_of_cube"].sum()),
                "n_segments": int(members["n_segments"].sum()),
                "n_traj_max": int(members["n_traj"].max()),
                "structure": _family_name(members),
                "min_r_in_family": float(members["min_r_in_family"].min(skipna=True)),
                "single_replica": bool(members["single_replica"].all() and len(members) == 1),
                "pct_A": float(
                    (members["pct_A"] * members["n_apo_frames"]).sum() / members["n_apo_frames"].sum()
                ),
                "pct_C2": float(
                    (members["pct_C2"] * members["n_apo_frames"]).sum() / members["n_apo_frames"].sum()
                ),
                "pct_other": float(
                    (members["pct_other"] * members["n_apo_frames"]).sum() / members["n_apo_frames"].sum()
                ),
                "pct_any_d1_elongated": float(
                    (members["pct_any_d1_elongated"] * members["n_apo_frames"]).sum()
                    / members["n_apo_frames"].sum()
                ),
            }
        )
    families_df = pd.DataFrame(family_rows).sort_values("pct_of_cube", ascending=False)
    families_df.insert(0, "rank_in_cube", range(1, len(families_df) + 1))
    return clusters, families_df


def rank_metastable_structures(
    *,
    output_root: Path | str = "output",
    k_selected: Path | str,
    profiles: Path | str = "output/endpoint_cluster_profiles",
    replica_audit: Path | str = "output/cluster_replica_audit_physical_k/single_replica_clusters.csv",
    copy_r: float = COPY_R,
    own_elbow: bool = True,
    qc_gsa_root: Path | str = "output/gsa_features_by_cube",
    no_assembly_qc: bool = False,
    out_dir: Path | str = "output/metastable_structures_physical_k",
) -> dict[str, Path]:
    """Rank each cube's endpoint clusters by apo-frame occupancy and merge copies."""
    output_root = Path(output_root)
    profiles = Path(profiles)
    out_dir = Path(out_dir)
    selected = pd.read_csv(k_selected)
    replicas = _replica_keys(Path(replica_audit))
    cluster_parts, family_parts, symmetry_parts = [], [], []
    for row in selected.itertuples(index=False):
        cube = str(row.cohort)
        k = int(row.selected_k)
        dirs = profile_dirs(profiles, cube, k)
        if not dirs:
            raise FileNotFoundError(f"{cube}: no profile folder at k={k}")
        symmetry = all_cluster_symmetry(
            output_root, cube, k,
            own_elbow=own_elbow, qc_gsa_root=Path(qc_gsa_root), no_assembly_qc=no_assembly_qc,
        )
        symmetry_parts.append(symmetry.assign(cohort=cube, k=k))
        clusters, families = rank_cube(cube, k, _read_motif(dirs), symmetry, replicas, copy_r)
        cluster_parts.append(clusters)
        family_parts.append(families)
    clusters = pd.concat(cluster_parts, ignore_index=True)
    families = pd.concat(family_parts, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_cube = out_dir / "by_cube"
    by_cube.mkdir(parents=True, exist_ok=True)
    written = {
        "clusters_ranked": out_dir / "clusters_ranked.csv",
        "families_ranked": out_dir / "families_ranked.csv",
        "all_cluster_symmetry": out_dir / "all_cluster_symmetry.csv",
    }
    clusters.to_csv(written["clusters_ranked"], index=False)
    families.to_csv(written["families_ranked"], index=False)
    if symmetry_parts:
        pd.concat(symmetry_parts, ignore_index=True).to_csv(written["all_cluster_symmetry"], index=False)
    show = ["rank_in_cube", "pct_of_cube", "n_orientations", "member_clusters", "structure"]
    for cube, part in families.groupby("cohort", sort=False):
        path = by_cube / f"{cube}.csv"
        part[show].round(1).to_csv(path, index=False)
        written[f"by_cube_{cube}"] = path
    return written


def _frame_belts(joined: pd.DataFrame, d1_cols: list[str]) -> pd.Series:
    order = [_edge_name(col) for col in d1_cols]
    elongated = joined[d1_cols].apply(pd.to_numeric, errors="coerce").to_numpy() >= D1_ELONGATED_LO
    labels = []
    for row in elongated:
        edges = [name for flag, (_idx, name) in zip(row, order) if flag]
        labels.append(belt_label(edges))
    return pd.Series(labels, index=joined.index)


def _segment_modes(joined: pd.DataFrame, segments: pd.DataFrame, cube: str) -> pd.DataFrame:
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
        rows.append(
            {
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
        )
    table = pd.DataFrame(rows).sort_values("pct_of_cluster", ascending=False)
    table.insert(0, "rank_in_cluster", range(1, len(table) + 1))
    return table


def categorize_cluster_mixtures(
    *,
    output_root: Path | str = "output",
    k_selected: Path | str,
    qc_gsa_root: Path | str = "output/gsa_features_by_cube",
    min_share: float = MIN_SHARE,
    out_dir: Path | str = "output/metastable_structures_physical_k",
) -> dict[str, Path]:
    """Split each endpoint cluster into contact categories independent of k."""
    output_root = Path(output_root)
    out_dir = Path(out_dir)
    selected = pd.read_csv(k_selected)
    belt_parts, segment_parts, mixture_rows = [], [], []
    for row in selected.itertuples(index=False):
        cube = str(row.cohort)
        k = int(row.selected_k)
        cp_dir = endpoint_changepoints_dir(output_root, cube, own_elbow=True)
        segs = load_segments_at_k(cp_dir, k, group="endpoint")
        joined = map_frames_to_segment_clusters(
            load_apo_motif_frames(output_root, cube, qc_gsa_root=qc_gsa_root),
            segs,
            cube=cube,
        )
        joined = joined.dropna(subset=["cluster_label", "n_open_cation_pi"])
        joined["cluster_label"] = joined["cluster_label"].astype(int)
        joined["frame"] = pd.to_numeric(joined["frame"], errors="coerce")
        d1_cols = unit_series_columns(joined, "equator_d1_e")
        joined["belt"] = _frame_belts(joined, d1_cols)
        joined["opening"] = joined["n_open_cation_pi"].map(lambda n: opening_label(int(n)))
        modes = _segment_modes(joined, segs, cube)
        for cluster, part in joined.groupby("cluster_label"):
            table = _category_table(part, cube, k, int(cluster), len(part))
            belt_parts.append(table)
            kept = table[table["pct_of_cluster"] >= min_share]
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
                if share < min_share:
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
                segment_parts.append(seg_table.sort_values("pct_of_cluster", ascending=False))
    belts = pd.concat(belt_parts, ignore_index=True)
    segments_out = pd.concat(segment_parts, ignore_index=True) if segment_parts else pd.DataFrame()
    mixture = pd.DataFrame(mixture_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "belt_inside_cluster": out_dir / "belt_inside_cluster.csv",
        "segment_belt": out_dir / "segment_belt.csv",
        "cluster_mixture": out_dir / "cluster_mixture.csv",
    }
    belts.to_csv(written["belt_inside_cluster"], index=False)
    segments_out.to_csv(written["segment_belt"], index=False)
    mixture.to_csv(written["cluster_mixture"], index=False)
    return written


def _pole_mix(rows: pd.DataFrame) -> dict[str, float]:
    n = float(rows["n_frames"].sum())
    if n <= 0:
        return {col: np.nan for col in POLE_BINS}
    return {col: float((rows[col] * rows["n_frames"]).sum() / n) for col in POLE_BINS}


def summarize_metastable_structures(
    *,
    output_root: Path | str = "output",
    catalog_dir: Path | str = "output/metastable_structures_physical_k",
    segment_belt: Optional[Path | str] = None,
    top_pairs: Path | str = "output/endpoint_pair_ranking_physical_k/top_pairs.csv",
    out_dir: Optional[Path | str] = None,
) -> dict[str, Path]:
    """One row per metastable structure, with the numbers in the methods note §7."""
    catalog_dir = Path(catalog_dir)
    out_dir = Path(out_dir) if out_dir is not None else catalog_dir
    belt_path = Path(segment_belt) if segment_belt is not None else catalog_dir / "min_share_0" / "segment_belt.csv"
    clusters = pd.read_csv(catalog_dir / "clusters_ranked.csv")
    families = pd.read_csv(catalog_dir / "families_ranked.csv")
    if not belt_path.is_file():
        raise FileNotFoundError(belt_path)
    seg = pd.read_csv(belt_path)
    family_of = {(r.cohort, int(r.cluster_label)): int(r.family_id) for r in clusters.itertuples(index=False)}
    seg["family_id"] = [family_of[(c, int(l))] for c, l in zip(seg["cohort"], seg["cluster_label"])]
    seg["kind"] = seg["belt"].map(belt_kind)
    pole_pairs = contact_pairs(clusters.columns, "pct_pi_open_")
    belt_pairs = contact_pairs(clusters.columns, "pct_d1_elongated_")

    rows = []
    replicas: dict[tuple[str, int], int] = {}
    for cube, part in clusters.groupby("cohort", sort=False):
        k = int(part["k"].iloc[0])
        segs = load_segments_at_k(endpoint_changepoints_dir(output_root, cube, own_elbow=True), k)
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

    top = pd.read_csv(top_pairs)
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
    written = {
        "structures_summary": out_dir / "structures_summary.csv",
        "structure_markers": out_dir / "structure_markers.csv",
    }
    summary.to_csv(written["structures_summary"], index=False)
    markers.to_csv(written["structure_markers"], index=False)
    return written


def assign_endpoint_clusters(frames: pd.DataFrame, segments: pd.DataFrame, *, cube: str) -> pd.DataFrame:
    out = frames.copy()
    out["_tid"] = out["traj_id"].map(lambda t: _norm_traj_id(t, cube))
    segs = segments.copy()
    if "group" in segs.columns:
        segs = segs[segs["group"].astype(str) == "endpoint"]
    segs["_tid"] = segs["traj_id"].map(lambda t: _norm_traj_id(t, cube))
    out["cluster_label"] = np.nan
    for tid, grp in segs.groupby("_tid"):
        mask = out["_tid"] == tid
        if not mask.any():
            continue
        fr = pd.to_numeric(out.loc[mask, "frame"], errors="coerce").to_numpy()
        lab = np.full(fr.shape, np.nan)
        for rec in grp.itertuples(index=False):
            start = float(rec.start_frame)
            end = float(rec.end_frame)
            m = (fr >= start) & (fr <= end)
            lab[m] = rec.cluster_label
        out.loc[mask, "cluster_label"] = lab
    return out.drop(columns=["_tid"])


def cluster_composition(joined: pd.DataFrame, *, cube: str, k: int) -> pd.DataFrame:
    joined = joined[joined["murata_metastructure"].notna()]
    labels = joined["cluster_label"].astype(int)
    intact = labels[pd.to_numeric(joined["murata_d1_n_elongated"], errors="coerce") == 0]
    parts = [
        labels.value_counts().sort_index().rename("n_frames"),
        (100.0 * intact.value_counts(normalize=True)).rename("pct_of_intact_belt"),
    ]
    for col, prefix in (("murata_d1_n_elongated", "pct_d1_elongated"), ("n_open_cation_pi", "pct_open")):
        cap = 2 if prefix == "pct_d1_elongated" else 3
        counts = pd.to_numeric(joined[col], errors="coerce").clip(upper=cap).astype(int)
        share = pd.crosstab(labels, counts, normalize="index").reindex(columns=range(cap + 1), fill_value=0.0)
        share.columns = [f"{prefix}_{c}" if c < cap else f"{prefix}_{c}plus" for c in share.columns]
        parts.append(100.0 * share)
    out = pd.concat(parts, axis=1).rename_axis("cluster_label").reset_index()
    out["pct_of_intact_belt"] = out["pct_of_intact_belt"].fillna(0.0)
    out.insert(0, "k", k)
    out.insert(0, "murata_system", COHORT_TO_MURATA_SYSTEM[cube])
    out.insert(0, "cohort", cube)
    return out


def plot_occupancy_vs_murata(cmp: pd.DataFrame, output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    mapped = cmp[cmp["murata_system"].astype(str) != ""].copy()
    mapped = mapped[mapped["guest_filter"] == "frames"]
    cohorts = [c for c in KNOWN_CUBES if c in set(mapped["cohort"])]
    if not cohorts:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.0 * len(cohorts), 3.8), squeeze=False)
    x = np.arange(len(METASTRUCTURES))
    width = 0.36
    for ax, cube in zip(axes[0], cohorts):
        sub = mapped[mapped["cohort"] == cube].set_index("metastructure")
        ours = [float(sub.loc[m, "our_percent"]) for m in METASTRUCTURES]
        refs = [float(sub.loc[m, "murata_percent"]) for m in METASTRUCTURES]
        ax.bar(x - width / 2, ours, width, color=CUBE_COLORS.get(cube, "0.4"), label="this work")
        ax.bar(x + width / 2, refs, width, color="0.65", label="Murata Table 1")
        system = str(sub["murata_system"].iloc[0])
        ax.set_xticks(x)
        ax.set_xticklabels(METASTRUCTURES)
        ax.set_ylim(0, 100)
        ax.set_ylabel("occupancy (%)")
        ax.set_title(f"{cube}  ({system})")
        ax.legend(fontsize=8)
    fig.suptitle("Non-encapsulated frames vs Murata Table 1", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def compare_to_murata_metastructures(
    *,
    output_root: Path | str = "output",
    out_dir: Path | str = "output/murata_g5",
    cubes: Optional[Sequence[str]] = None,
    qc_gsa_root: Path | str = "output/gsa_features_by_cube",
    no_assembly_qc: bool = False,
    own_elbow: bool = True,
    k: Optional[int] = None,
    k_selected: Optional[Path | str] = None,
) -> dict[str, Path]:
    """G5: occupancy vs Murata Table 1 and Murata label × endpoint cluster tables."""
    output_root = Path(output_root)
    out_dir = Path(out_dir)
    wanted = list(cubes) if cubes is not None else list(KNOWN_CUBES)
    by_cube = discover_motif_rmsd_csvs_by_cube(output_root, cubes=wanted)
    if not by_cube:
        raise FileNotFoundError(f"No *_motif_rmsd.csv under {output_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_k = load_selected_k(k_selected)
    occupancy_rows, xtab_rows, other_rows, composition_rows, qc_rows = [], [], [], [], []
    written: dict[str, Path] = {}
    for cube, paths in by_cube.items():
        df = load_motif_rmsd_csvs(paths)
        if not no_assembly_qc:
            gsa_dir = Path(qc_gsa_root) / cube
            if gsa_dir.is_dir():
                qc = scan_assembly_qc_dir(gsa_dir)
                qc_rows.append(qc)
                failed = assembly_qc_failures(qc)
                if failed:
                    df = drop_trajectories(df, failed)
        df = label_motif_metastructures(df)
        filters = [("all", "all")]
        if "n_guest_inside_cavity" in df.columns:
            filters.extend([("frames", "frames"), ("never", "never")])
        for filt, name in filters:
            labeled = filter_non_encapsulated(df, traj_filter=filt)
            occupancy_rows.append(
                occupancy_vs_murata(labeled["murata_metastructure"], cohort=cube, guest_filter=name)
            )
        cp_dir = endpoint_changepoints_dir(output_root, cube, own_elbow=own_elbow)
        if cube not in COHORT_TO_MURATA_SYSTEM or not (cp_dir / "clusters" / "endpoint").is_dir():
            continue
        cut = k or selected_k.get(cube) or chosen_k_from_cohort(cp_dir, group="endpoint")
        segs = load_segments_at_k(cp_dir, cut, group="endpoint")
        if segs.empty:
            continue
        apo = filter_non_encapsulated(df, traj_filter="frames")
        joined = assign_endpoint_clusters(apo, segs, cube=cube).dropna(subset=["cluster_label"])
        if joined.empty:
            continue
        xtab = pd.crosstab(
            joined["murata_metastructure"], joined["cluster_label"].astype(int), normalize="index"
        ) * 100.0
        xtab = xtab.reindex(METASTRUCTURES)
        xtab.insert(0, "cohort", cube)
        xtab.insert(1, "murata_system", COHORT_TO_MURATA_SYSTEM[cube])
        xtab.insert(2, "k", cut)
        xtab_rows.append(xtab.reset_index())
        composition_rows.append(cluster_composition(joined, cube=cube, k=cut))
        other = joined[joined["murata_metastructure"] == "other"]
        if not other.empty:
            counts = other["cluster_label"].astype(int).value_counts(normalize=True).sort_index()
            other_rows.append(
                pd.DataFrame(
                    {
                        "cohort": cube,
                        "murata_system": COHORT_TO_MURATA_SYSTEM[cube],
                        "k": cut,
                        "cluster_label": counts.index.astype(int),
                        "percent_of_other": 100.0 * counts.to_numpy(),
                        "n_other_frames": int(len(other)),
                    }
                )
            )
    occ = pd.concat(occupancy_rows, ignore_index=True)
    written["murata_table1_occupancy"] = out_dir / "murata_table1_occupancy.csv"
    occ.to_csv(written["murata_table1_occupancy"], index=False)
    if qc_rows:
        written["assembly_qc"] = out_dir / "assembly_qc.csv"
        pd.concat(qc_rows, ignore_index=True).to_csv(written["assembly_qc"], index=False)
    mapped = occ[(occ["guest_filter"] == "frames") & (occ["murata_system"].astype(str) != "")]
    if not mapped.empty:
        written["occupancy_vs_murata"] = plot_occupancy_vs_murata(
            occ, out_dir / "plots" / "occupancy_vs_murata_table1.png"
        )
    if xtab_rows:
        written["murata_vs_endpoint_cluster"] = out_dir / "murata_vs_endpoint_cluster.csv"
        pd.concat(xtab_rows, ignore_index=True).to_csv(written["murata_vs_endpoint_cluster"], index=False)
    if other_rows:
        written["murata_other_vs_endpoint_cluster"] = out_dir / "murata_other_vs_endpoint_cluster.csv"
        pd.concat(other_rows, ignore_index=True).to_csv(written["murata_other_vs_endpoint_cluster"], index=False)
    if composition_rows:
        written["endpoint_cluster_composition"] = out_dir / "endpoint_cluster_composition.csv"
        pd.concat(composition_rows, ignore_index=True).to_csv(written["endpoint_cluster_composition"], index=False)
    return written


def monomer_pair_columns(columns: list[str]) -> list[str]:
    cols = []
    for name in columns:
        if not (name.startswith("endpoint_dist_") and name.endswith("_mean_mean")):
            continue
        body = name.removeprefix("endpoint_dist_").removesuffix("_mean_mean")
        if body.count("_") == 1:
            cols.append(name)
    return cols


def pair_label(column: str) -> str:
    body = column.removeprefix("endpoint_dist_").removesuffix("_mean_mean")
    return "M" + body.replace("_", "–M")


def family_map(summary: pd.DataFrame, cube: str, k: int) -> tuple[dict[int, int], dict[int, str], set[int]]:
    sub = summary[(summary["cohort"] == cube) & (summary["k"] == k)]
    clusters: dict[int, int] = {}
    names: dict[int, str] = {}
    single: set[int] = set()
    for rec in sub.itertuples(index=False):
        family = int(rec.family_id)
        names[family] = str(rec.structure)
        if bool(rec.single_replica):
            single.add(family)
        for token in str(rec.member_clusters).split():
            clusters[int(token)] = family
    return clusters, names, single


def load_annotation_segments(output_root: Path, cube: str, k: int) -> tuple[pd.DataFrame, list[str]]:
    base = endpoint_changepoints_dir(output_root, cube, own_elbow=True) / "clusters" / "endpoint"
    labels = pd.read_csv(base / "by_k" / f"k_{k:02d}" / "segments_clustered.csv")
    feats = pd.read_csv(base / "segments_clustered.csv")
    cols = monomer_pair_columns(list(feats.columns))
    keep = ["traj_id", "segment_id", "start_frame", "end_frame", "start_ps", "end_ps", "n_frames", *cols]
    merged = labels.merge(
        feats[keep],
        on=["traj_id", "segment_id", "start_frame", "end_frame"],
        how="inner",
        suffixes=("_label", "_feat"),
    )
    if "n_frames_feat" in merged.columns:
        merged["n_samples"] = merged["n_frames_feat"]
    elif "n_frames" in merged.columns:
        merged["n_samples"] = merged["n_frames"]
    else:
        raise KeyError("segment tables have no n_frames column")
    return merged, cols


def weighted_mean(vectors: list[np.ndarray], weights: list[float]) -> np.ndarray:
    return np.average(np.vstack(vectors), axis=0, weights=np.asarray(weights, dtype=float))


def list_annotation_replicas(
    *,
    output_root: Path | str = "output",
    catalog_dir: Path | str = "output/metastable_structures_physical_k",
    k_selected: Path | str,
    min_flank_ps: float = MIN_FLANK_PS,
    min_boundary_ps: float = MIN_BOUNDARY_PS,
    out_dir: Path | str = "output/annotation_replicas",
) -> dict[str, Path]:
    """Replicas whose endpoint state leaves the tight family once."""
    output_root = Path(output_root)
    catalog_dir = Path(catalog_dir)
    out_dir = Path(out_dir)
    summary = pd.read_csv(catalog_dir / "structures_summary.csv")
    selected = pd.read_csv(k_selected)
    k_of = dict(zip(selected["cohort"].astype(str), selected["selected_k"].astype(int)))
    markers = pd.read_csv(catalog_dir / "structure_markers.csv")
    cut_rows, replica_rows, segment_rows = [], [], []
    for cube in CUBE_ORDER:
        if cube not in k_of:
            continue
        k = int(k_of[cube])
        clusters, names, single = family_map(summary, cube, k)
        seg, cols = load_annotation_segments(output_root, cube, k)
        seg = seg[seg["cluster_label"].isin(clusters)].copy()
        seg["family"] = seg["cluster_label"].map(clusters)
        population = seg[~seg["family"].isin(single)]
        family_level = population.groupby("family")[cols].mean().mean(axis=1)
        tight = int(family_level.idxmin())
        for tid, group in seg.groupby("traj_id", sort=False):
            group = group.sort_values(["start_frame", "segment_id"])
            runs: list[dict] = []
            for rec in group.itertuples(index=False):
                vector = np.array([getattr(rec, c) for c in cols], dtype=float)
                if runs and runs[-1]["family"] == int(rec.family):
                    run = runs[-1]
                    run["end_frame"] = int(rec.end_frame)
                    run["end_ps"] = float(rec.end_ps)
                    run["weights"].append(float(rec.n_samples))
                    run["vectors"].append(vector)
                    run["segment_ids"].append(int(rec.segment_id))
                else:
                    runs.append(
                        {
                            "family": int(rec.family),
                            "cluster": int(rec.cluster_label),
                            "start_frame": int(rec.start_frame),
                            "end_frame": int(rec.end_frame),
                            "start_ps": float(rec.start_ps),
                            "end_ps": float(rec.end_ps),
                            "single": int(rec.family) in single,
                            "weights": [float(rec.n_samples)],
                            "vectors": [vector],
                            "segment_ids": [int(rec.segment_id)],
                        }
                    )
            previous = None
            for rec in group.itertuples(index=False):
                vector = np.array([getattr(rec, c) for c in cols], dtype=float)
                if previous is None:
                    previous = (int(rec.family), vector, int(rec.family) in single, int(rec.end_frame))
                    continue
                fam, pvec, psingle, end_frame = previous
                delta_seg = vector - pvec
                if psingle or int(rec.family) in single:
                    seg_kind = "single-replica"
                elif fam == int(rec.family):
                    seg_kind = "same-family"
                elif fam == tight:
                    seg_kind = "leave-tight"
                elif int(rec.family) == tight:
                    seg_kind = "return-tight"
                else:
                    seg_kind = "between-deformed"
                cut_rows.append(
                    {
                        "cohort": cube,
                        "k": k,
                        "traj_id": tid,
                        "kind": seg_kind,
                        "boundary_frame": end_frame,
                        "shape_A": float(np.sqrt(np.mean(delta_seg**2))),
                        "peak_A": float(np.max(np.abs(delta_seg))),
                    }
                )
                previous = (int(rec.family), vector, int(rec.family) in single, int(rec.end_frame))
            events: list[tuple[str, dict, dict, np.ndarray]] = []
            n_leave = 0
            for left, right in zip(runs, runs[1:]):
                if left["single"] or right["single"]:
                    kind = "single-replica"
                elif left["family"] == tight and right["family"] != tight:
                    kind = "leave-tight"
                    n_leave += 1
                elif right["family"] == tight and left["family"] != tight:
                    kind = "return-tight"
                else:
                    kind = "between-deformed"
                delta = weighted_mean(right["vectors"], right["weights"]) - weighted_mean(
                    left["vectors"], left["weights"]
                )
                events.append((kind, left, right, delta))
            listed: list[tuple[dict, dict, np.ndarray]] = []
            for kind, left, right, delta in events:
                if kind != "leave-tight" or n_leave != 1:
                    continue
                pre_ps = left["end_ps"] - left["start_ps"]
                post_ps = right["end_ps"] - right["start_ps"]
                if pre_ps < min_flank_ps or post_ps < min_flank_ps or left["end_ps"] < min_boundary_ps:
                    continue
                listed.append((left, right, delta))
            if len(listed) != 1:
                continue
            left, right, delta = listed[0]
            order = np.argsort(-np.abs(delta))
            top = [f"{pair_label(cols[i])} {delta[i]:+.2f}" for i in order[:4]]
            mark = markers[
                (markers["cohort"] == cube) & (markers["k"] == k) & (markers["best_cluster"] == right["cluster"])
            ].sort_values("rank")
            site_pairs = [
                f"{rec.endpoint_label} {rec.cohens_d_best:+.2f}" for rec in mark.head(3).itertuples(index=False)
            ]
            run_id = str(tid).replace("_mdcrd_v", "")
            min_flank = float(min(left["end_ps"] - left["start_ps"], right["end_ps"] - right["start_ps"]))
            shape = float(np.sqrt(np.mean(delta**2)))
            replica_rows.append(
                {
                    "cohort": cube,
                    "k": k,
                    "traj_id": run_id,
                    "boundary_frame": int(left["end_frame"]),
                    "boundary_ps": float(left["end_ps"]),
                    "tight_start_frame": int(left["start_frame"]),
                    "tight_end_frame": int(left["end_frame"]),
                    "deformed_start_frame": int(right["start_frame"]),
                    "deformed_end_frame": int(right["end_frame"]),
                    "tight_ps": float(left["end_ps"] - left["start_ps"]),
                    "deformed_ps": float(right["end_ps"] - right["start_ps"]),
                    "shape_A": shape,
                    "mean_shift_A": float(delta.mean()),
                    "n_longer": int(np.sum(delta > 0.3)),
                    "n_shorter": int(np.sum(delta < -0.3)),
                    "tight_family": int(left["family"]),
                    "deformed_family": int(right["family"]),
                    "deformed_cluster": int(right["cluster"]),
                    "tight_name": names[left["family"]],
                    "deformed_name": names[right["family"]],
                    "top_monomer_pairs": "; ".join(top),
                    "marker_site_pairs": "; ".join(site_pairs),
                    "min_flank_ps": min_flank,
                    "score": shape * min(min_flank / 1000.0, 2.0),
                }
            )
            boundary_ids = set(left["segment_ids"])
            deformed_ids = set(right["segment_ids"])
            for rec in group.itertuples(index=False):
                segment_id = int(rec.segment_id)
                if segment_id in boundary_ids and int(rec.end_frame) == int(left["end_frame"]):
                    role = "tight, ends at the boundary"
                elif segment_id in boundary_ids:
                    role = "tight dwell"
                elif segment_id in deformed_ids and int(rec.start_frame) == int(right["start_frame"]):
                    role = "deformed, starts after the boundary"
                elif segment_id in deformed_ids:
                    role = "deformed dwell"
                else:
                    role = "outside the annotated transition"
                segment_rows.append(
                    {
                        "cohort": cube,
                        "k": k,
                        "traj_id": run_id,
                        "segment_id": segment_id,
                        "start_frame": int(rec.start_frame),
                        "end_frame": int(rec.end_frame),
                        "start_ps": float(rec.start_ps),
                        "end_ps": float(rec.end_ps),
                        "cluster_label": int(rec.cluster_label),
                        "family_id": int(rec.family),
                        "structure": names[int(rec.family)],
                        "role": role,
                    }
                )
    cuts = pd.DataFrame(cut_rows)
    replicas = pd.DataFrame(replica_rows)
    segments = pd.DataFrame(segment_rows)
    if replicas.empty:
        raise FileNotFoundError("no replica passed the annotation filter")
    replicas["cohort"] = pd.Categorical(replicas["cohort"], list(CUBE_ORDER), ordered=True)
    replicas = replicas.sort_values(["cohort", "score"], ascending=[True, False])
    priority_index = replicas.groupby("cohort", observed=True).head(1).index
    replicas["priority"] = replicas.index.isin(priority_index)
    replicas["cohort"] = replicas["cohort"].astype(str)
    segments = segments.merge(
        replicas[["cohort", "traj_id", "boundary_frame", "priority"]],
        on=["cohort", "traj_id"],
        how="left",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "cuts": out_dir / "cuts.csv",
        "replicas": out_dir / "replicas.csv",
        "segments": out_dir / "segments.csv",
        "priority": out_dir / "priority.csv",
    }
    cuts.to_csv(written["cuts"], index=False)
    replicas.to_csv(written["replicas"], index=False)
    segments.to_csv(written["segments"], index=False)
    replicas[replicas["priority"]].to_csv(written["priority"], index=False)
    return written
