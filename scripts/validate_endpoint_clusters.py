#!/usr/bin/env python3
"""Structural quality checks on endpoint clusters after clustering (Tests A–D).

Reads existing Ward labels, site-pair CSVs, motif distances and GSA geometry.
Does not re-segment. Ward is refit only inside Test C hold-out folds.

Until D1 has written admissible operations, pass ``--group rotations`` or
``--group contact``.

Example
-------
python scripts/validate_endpoint_clusters.py --own-elbow --group contact \\
    --k-selected output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ChangepointAnalysis.assembly_qc import (
    assembly_qc_failures,
    drop_trajectories,
    scan_assembly_qc_dir,
)
from src.ChangepointAnalysis.cluster_k_diagnostics import (
    load_segments_at_k,
    map_frames_to_segment_clusters,
    resolve_endpoint_features_dir,
)
from src.ChangepointAnalysis.cluster_validation import (
    DEFAULT_ENDPOINT_STRIDE_PS,
    DEFAULT_FRAME_SAMPLE_CAP,
    DEFAULT_FRAME_STRIDE_PS,
    NEGATIVE_CONTROL_PAIRS,
    POLE_H_FROM_POLE_M,
    balanced_accuracy_grouped,
    best_profile_r,
    cluster_diameter_rows,
    cohens_d,
    coverage_at_radius,
    deformation_deltas,
    hungarian_match,
    labelled_pairwise_distances,
    label_permutation_null,
    mappings_from_perms,
    medoid_index,
    merge_ratio,
    nearest_centroid_labels,
    overlap_coefficient,
    pairwise_symmetry_distances,
    profile_correlation_table,
    refit_ward_labels,
    reorient_rows,
    replica_block_energy_p,
    replica_holdout_folds,
    sample_frame_indices,
    sensitivity_family_table,
    sign_agreement,
    standardized_median_shift,
    symmetry_adjusted_distance,
    two_population_test,
    variance_split,
    wasserstein_1d,
    within_dwell_spread,
)
from src.ChangepointAnalysis.feature_groups import resolve_features_csv
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.metastable_catalog import (
    COPY_R,
    contact_automorphisms,
    family_ids,
    perm_cycles,
    rotation_contact_automorphisms,
)
from src.ChangepointAnalysis.murata_rmsd import (
    discover_motif_rmsd_csvs_by_cube,
    filter_non_encapsulated,
    load_motif_rmsd_csvs,
    unit_series_columns,
)
from src.ChangepointAnalysis.reporting import (
    collect_endpoint_segment_means,
    endpoint_changepoints_dir,
    list_endpoint_site_pair_columns,
    score_cluster_discriminating_features,
)
from src.ChangepointAnalysis.segment_clustering import build_feature_matrix_from_segments_df

GSA_OBS = (
    "assembly_rg",
    "assembly_relative_shape_anisotropy",
    "assembly_asphericity",
    "total_inter_monomer_contacts",
    "n_active_interfaces",
)
ADJACENT_PAIRS = (
    (0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3),
    (0, 4), (1, 5), (2, 3), (3, 1), (4, 2), (5, 0),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate endpoint clusters (Tests A–D).")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--own-elbow", action="store_true")
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument("--cube", nargs="+", default=None)
    p.add_argument("--tests", nargs="+", default=["A", "B", "C", "D"], choices=["A", "B", "C", "D"])
    p.add_argument(
        "--group",
        default="admissible",
        choices=("admissible", "rotations", "contact"),
        help="Monomer relabellings. 'admissible' needs D1 output.",
    )
    p.add_argument("--apo-only", action="store_true")
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=ROOT / "output" / "metastable_structures_physical_k",
    )
    p.add_argument(
        "--symmetry-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_validation_physical_k" / "symmetry",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_validation_physical_k",
    )
    p.add_argument("--no-assembly-qc", action="store_true")
    p.add_argument("--eccentric-z", type=float, default=3.5)
    p.add_argument("--min-segments", type=int, default=15)
    p.add_argument("--delta-bic", type=float, default=10.0)
    p.add_argument("--min-ashman", type=float, default=2.0)
    p.add_argument("--min-minor", type=float, default=0.15)
    p.add_argument("--min-minor-replicas", type=int, default=3)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=20)
    p.add_argument("--coverage-min", type=float, default=0.8)
    p.add_argument("--replica-r-min", type=float, default=0.8)
    p.add_argument("--sign-agree-min", type=float, default=0.9)
    p.add_argument("--merge-p95-max", type=float, default=1.2)
    p.add_argument("--copy-r", type=float, default=COPY_R)
    p.add_argument("--min-replica-ps", type=float, default=100.0)
    p.add_argument("--frame-stride-ps", type=int, default=DEFAULT_FRAME_STRIDE_PS)
    p.add_argument("--frame-cap", type=int, default=DEFAULT_FRAME_SAMPLE_CAP)
    p.add_argument("--random-state", type=int, default=0)
    return p.parse_args()


def _cubes(args: argparse.Namespace) -> list[tuple[str, int]]:
    table = pd.read_csv(args.k_selected)
    wanted = set(args.cube) if args.cube else set(KNOWN_CUBES)
    out = []
    for rec in table.itertuples(index=False):
        cube = str(rec.cohort)
        if cube in wanted:
            out.append((cube, int(rec.selected_k)))
    return out


def _load_group(args: argparse.Namespace, cube: str, specs: list[tuple]):
    kind = args.group
    site_maps = None
    if kind == "admissible":
        op_path = args.symmetry_dir / "symmetry_operations.csv"
        if not op_path.is_file():
            raise FileNotFoundError(
                "No D1 symmetry_operations.csv. Run check_cage_symmetry.py "
                "or pass --group rotations / --group contact."
            )
        ops = pd.read_csv(op_path)
        src = cube
        sub = ops[(ops["cube"].astype(str) == cube) & ops["admissible"].astype(bool)]
        if sub.empty:
            src = POLE_H_FROM_POLE_M.get(cube, cube)
            sub = ops[(ops["cube"].astype(str) == src) & ops["admissible"].astype(bool)]
        if sub.empty:
            raise FileNotFoundError(
                f"No admissible operations for {cube}. Run D1 or pass --group contact."
            )
        perms = []
        for rec in sub.itertuples(index=False):
            perms.append(tuple(int(x) for x in str(rec.perm).split()))
        site_path = args.symmetry_dir / "site_maps.csv"
        smap_by_cycles: dict[str, dict[int, list[int]]] = {}
        if site_path.is_file():
            sites = pd.read_csv(site_path)
            sites = sites[sites["cube"].astype(str).isin({cube, src})]
            for rec in sites.itertuples(index=False):
                smap_by_cycles.setdefault(str(rec.cycles), {})[int(rec.dest_monomer)] = [
                    int(x) for x in str(rec.site_map).split()
                ]
        site_maps = [smap_by_cycles.get(perm_cycles(p)) or None for p in perms]
        # drop empty maps so identity site mapping is used
        maps = mappings_from_perms(specs, perms, site_maps=site_maps)
        return perms, maps
    if kind == "rotations":
        perms = rotation_contact_automorphisms()
    else:
        perms = contact_automorphisms()
    return perms, mappings_from_perms(specs, perms)


def _apply_qc(df: pd.DataFrame, cube: str, args: argparse.Namespace) -> pd.DataFrame:
    if args.no_assembly_qc:
        return df
    gsa_dir = args.output_root / "gsa_features_by_cube" / cube
    if not gsa_dir.is_dir():
        return df
    failed = assembly_qc_failures(scan_assembly_qc_dir(gsa_dir))
    return drop_trajectories(df, failed) if failed else df


def _load_catalog(args: argparse.Namespace, cube: str) -> pd.DataFrame:
    path = args.catalog_dir / "clusters_ranked.csv"
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["cohort"].astype(str) == cube].copy()


def _family_map(catalog: pd.DataFrame) -> dict[int, int]:
    if catalog.empty or "family_id" not in catalog.columns:
        return {}
    return {
        int(r.cluster_label): int(r.family_id)
        for r in catalog.itertuples(index=False)
    }


def _load_observables(
    args: argparse.Namespace,
    cube: str,
    segs: pd.DataFrame,
) -> pd.DataFrame:
    motif_paths = discover_motif_rmsd_csvs_by_cube(args.output_root, cubes=[cube]).get(cube, [])
    motif = load_motif_rmsd_csvs(motif_paths) if motif_paths else pd.DataFrame()
    if not motif.empty:
        motif = _apply_qc(motif, cube, args)
        if args.apo_only:
            motif = filter_non_encapsulated(motif, traj_filter="frames")
        motif = _attach_segment_keys(motif, segs, cube)
        motif = motif.dropna(subset=["cluster_label"])
    gsa_dir = args.output_root / "gsa_features_by_cube" / cube
    gsa_parts = []
    if gsa_dir.is_dir():
        usecols = {"traj_id", "frame", "n_guest_inside_cavity", *GSA_OBS}
        for path in sorted(gsa_dir.glob("*_gsa_features.csv")):
            header = pd.read_csv(path, nrows=0)
            cols = [c for c in header.columns if c in usecols]
            gsa_parts.append(pd.read_csv(path, usecols=cols))
    gsa = pd.concat(gsa_parts, ignore_index=True) if gsa_parts else pd.DataFrame()
    if not gsa.empty:
        gsa = _apply_qc(gsa, cube, args)
        if args.apo_only and "n_guest_inside_cavity" in gsa.columns:
            gsa = filter_non_encapsulated(gsa, traj_filter="frames")
        gsa = _attach_segment_keys(gsa, segs, cube)
        gsa = gsa.dropna(subset=["cluster_label"])
    return motif, gsa


def _norm_tid(traj_id: str, cube: str) -> str:
    text = str(traj_id)
    prefix = f"{cube}_"
    return text[len(prefix):] if text.startswith(prefix) else text


def _attach_segment_keys(frames: pd.DataFrame, segs: pd.DataFrame, cube: str) -> pd.DataFrame:
    """Copy cluster_label and segment_id from covering segments onto frames."""
    if frames.empty:
        return frames
    out = frames.copy()
    out["_tid"] = out["traj_id"].map(lambda t: _norm_tid(t, cube))
    work = segs.copy()
    if "group" in work.columns:
        work = work[work["group"].astype(str) == "endpoint"]
    work["_tid"] = work["traj_id"].map(lambda t: _norm_tid(t, cube))
    out["segment_id"] = np.nan
    out["cluster_label"] = np.nan
    for tid, grp in work.groupby("_tid"):
        mask = out["_tid"] == tid
        if not mask.any():
            continue
        fr = pd.to_numeric(out.loc[mask, "frame"], errors="coerce").to_numpy()
        sid = np.full(fr.shape, np.nan)
        lab = np.full(fr.shape, np.nan)
        for rec in grp.itertuples(index=False):
            hit = (fr >= rec.start_frame) & (fr <= rec.end_frame)
            sid[hit] = rec.segment_id
            lab[hit] = rec.cluster_label
        out.loc[mask, "segment_id"] = sid
        out.loc[mask, "cluster_label"] = lab
    return out.drop(columns=["_tid"])


def _segment_medians(frames: pd.DataFrame, columns: list[str], segs: pd.DataFrame, cube: str) -> pd.DataFrame:
    if frames.empty:
        return pd.DataFrame()
    have = [c for c in columns if c in frames.columns]
    if not have:
        return pd.DataFrame()
    work = frames if "segment_id" in frames.columns else _attach_segment_keys(frames, segs, cube)
    work = work.dropna(subset=["cluster_label", "segment_id"])
    if work.empty:
        return pd.DataFrame()
    g = work.groupby(["traj_id", "segment_id", "cluster_label"], as_index=False)[have].median()
    g["traj_id"] = g["traj_id"].map(lambda t: _norm_tid(t, cube))
    return g


def _sorted_unit_medians(frames: pd.DataFrame, prefix: str, segs: pd.DataFrame, cube: str) -> pd.DataFrame:
    cols = unit_series_columns(frames, prefix) if not frames.empty else []
    if not cols:
        return pd.DataFrame()
    work = frames if "segment_id" in frames.columns else _attach_segment_keys(frames, segs, cube)
    work = work.dropna(subset=["cluster_label", "segment_id"])
    if work.empty:
        return pd.DataFrame()
    med = work.groupby(["traj_id", "segment_id", "cluster_label"], as_index=False)[cols].median()
    vals = np.sort(med[cols].to_numpy(dtype=float), axis=1)[:, ::-1]
    out = med[["traj_id", "segment_id", "cluster_label"]].copy()
    out["traj_id"] = out["traj_id"].map(lambda t: _norm_tid(t, cube))
    for i in range(min(3, vals.shape[1])):
        out[f"{prefix}rank{i + 1}"] = vals[:, i]
    return out


def _interface_breaks(
    X: np.ndarray,
    specs: list[tuple],
    y: np.ndarray,
    compact_labels: Sequence[int],
) -> np.ndarray:
    """Count of 12 adjacent interfaces whose min site-pair exceeds the compact 99th pct."""
    pair_cols: dict[tuple[int, int], list[int]] = {}
    for col, spec in enumerate(specs):
        _n, i, _a, j, _b = spec[:5]
        pair_cols.setdefault((min(int(i), int(j)), max(int(i), int(j))), []).append(col)
    mins = np.full((X.shape[0], len(ADJACENT_PAIRS)), np.nan)
    for k, pair in enumerate(ADJACENT_PAIRS):
        cols = pair_cols.get((min(pair), max(pair)), [])
        if cols:
            mins[:, k] = np.nanmin(X[:, cols], axis=1)
    compact = np.isin(y, list(compact_labels))
    if not compact.any():
        compact = np.ones(len(y), dtype=bool)
    thr = np.nanquantile(mins[compact], 0.99, axis=0)
    return np.nansum(mins > thr, axis=1)


def _plot_diameters(diam: pd.DataFrame, cube: str, path: Path) -> None:
    sub = diam[(diam["unit"] == "segment") & (diam["group_kind"] == "cluster")]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(sub))
    ax.bar(x - 0.15, sub["median"], width=0.3, label="median")
    ax.bar(x + 0.15, sub["p95"], width=0.3, label="p95")
    ax.plot(x, sub["max"], "o", color="C3", label="max")
    ax.set_xticks(x, [str(int(v)) for v in sub["cluster_label"]])
    ax.set_xlabel("cluster")
    ax.set_ylabel("distance-RMSD (Å)")
    ax.set_title(f"{cube} cluster diameters")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_observables(obs: pd.DataFrame, cube: str, path: Path) -> None:
    if obs.empty or "assembly_rg" not in obs.columns:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, part in obs.groupby("cluster_label"):
        ax.hist(part["assembly_rg"].dropna(), bins=20, alpha=0.45, label=str(int(lab)))
    ax.set_xlabel("assembly Rg (Å)")
    ax.set_title(f"{cube} segment-median Rg")
    ax.legend(title="cluster", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _ward_features(cp_dir: Path, segs: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    stored = cp_dir / "clusters" / "endpoint" / "segments_clustered.csv"
    if not stored.is_file():
        return np.empty((0, 0)), segs
    feat_df = pd.read_csv(stored)
    keys = ["traj_id", "segment_id"]
    extra = [c for c in feat_df.columns if c not in keys and c not in segs.columns]
    merged = segs.merge(feat_df[keys + extra], on=keys, how="left")
    try:
        X, _names = build_feature_matrix_from_segments_df(merged)
    except Exception:
        return np.empty((0, 0)), segs
    return X, merged


def run_cube(cube: str, k: int, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    tests = {t.upper() for t in args.tests}
    cp_dir = endpoint_changepoints_dir(args.output_root, cube, own_elbow=args.own_elbow)
    segs = load_segments_at_k(cp_dir, k, group="endpoint")
    if segs.empty:
        print(f"{cube}: no segments at k={k}", flush=True)
        return {}
    segs = _apply_qc(segs, cube, args)
    feat_dir = resolve_endpoint_features_dir(cp_dir)
    X, y, keys, specs, kind, _ = collect_endpoint_segment_means(
        cp_dir, feat_dir, group="endpoint", clustered_df=segs, include_frame_corr=False
    )
    if kind != "site_pair" or X.size == 0:
        print(f"{cube}: no site-pair segment means", flush=True)
        return {}
    keys = keys.reset_index(drop=True)
    y = keys["cluster_label"].to_numpy(dtype=int)
    perms, mappings = _load_group(args, cube, specs)
    catalog = _load_catalog(args, cube)
    families = _family_map(catalog)
    fam_labels = np.array([families.get(int(c), int(c)) for c in y])

    out: dict[str, pd.DataFrame] = {}
    print(f"{cube} k={k}: {len(keys)} segments, {len(mappings)} relabellings ({args.group})", flush=True)

    dist, _best = pairwise_symmetry_distances(X, mappings)
    labelled = labelled_pairwise_distances(X)
    dwell = float("nan")

    # thermal scale: a few long segments, one CSV read per chosen trajectory
    stride = max(1, int(args.frame_stride_ps) // DEFAULT_ENDPOINT_STRIDE_PS)
    rng = np.random.default_rng(args.random_state)
    long_segs = segs.sort_values("n_frames", ascending=False).head(12)
    frame_rows = []
    loaded: dict[str, pd.DataFrame] = {}
    for rec in long_segs.itertuples(index=False):
        tid = str(rec.traj_id)
        if tid not in loaded:
            path = resolve_features_csv(feat_dir, tid, suffix="_endpoint_features.csv")
            if path is None:
                continue
            header = pd.read_csv(path, nrows=0)
            cols = [c for c, *_ in list_endpoint_site_pair_columns(header.columns)]
            if not cols:
                continue
            loaded[tid] = pd.read_csv(path, usecols=lambda c: c in set(cols) | {"frame"})
        feat = loaded[tid]
        cols = [c for c in feat.columns if c != "frame"]
        fr = pd.to_numeric(feat["frame"], errors="coerce")
        part = feat.loc[(fr >= rec.start_frame) & (fr <= rec.end_frame), cols]
        if len(part) < 2:
            continue
        idx = sample_frame_indices(len(part), stride=stride, cap=min(40, args.frame_cap), rng=rng)
        for row in part.to_numpy(dtype=float)[idx]:
            frame_rows.append((str(rec.traj_id), int(rec.segment_id), row))
    if frame_rows:
        frame_X = np.vstack([r[2] for r in frame_rows])
        frame_seg = np.array([f"{r[0]}:{r[1]}" for r in frame_rows])
        dwell = within_dwell_spread(frame_X, frame_seg, mappings, rng=rng)

    if "A" in tests:
        diam_c, ecc_c = cluster_diameter_rows(
            dist, labelled, y, keys=keys, unit="segment", eccentric_z=args.eccentric_z
        )
        diam_c["group_kind"] = "cluster"
        diam_f, ecc_f = cluster_diameter_rows(
            dist, labelled, fam_labels, keys=keys, unit="segment", group_col="family_id",
            eccentric_z=args.eccentric_z,
        )
        diam_f["group_kind"] = "structure"
        diam_f = diam_f.rename(columns={"family_id": "cluster_label"})
        ecc_f["group_kind"] = "structure"
        ecc_c["group_kind"] = "cluster"
        diam = pd.concat([diam_c, diam_f], ignore_index=True)
        diam["cohort"] = cube
        diam["k"] = k
        diam["within_dwell_spread"] = dwell
        ecc = pd.concat([ecc_c, ecc_f], ignore_index=True)
        ecc["cohort"] = cube
        out["cluster_diameter"] = diam
        out["member_eccentricity"] = ecc
        _plot_diameters(diam, cube, args.out_dir / "plots" / f"diameter_{cube}.png")

    motif, gsa = (pd.DataFrame(), pd.DataFrame())
    seg_obs = pd.DataFrame()
    if "B" in tests or "C" in tests or "D" in tests:
        motif, gsa = _load_observables(args, cube, segs)
        parts = []
        if not gsa.empty:
            parts.append(_segment_medians(gsa, list(GSA_OBS), segs, cube))
        if not motif.empty:
            if "n_open_cation_pi" in motif.columns:
                parts.append(_segment_medians(motif, ["n_open_cation_pi"], segs, cube))
            parts.append(_sorted_unit_medians(motif, "cation_pi_pole_e", segs, cube))
            parts.append(_sorted_unit_medians(motif, "equator_d1_e", segs, cube))
            parts.append(_sorted_unit_medians(motif, "equator_d2_e", segs, cube))
        seg_obs = keys[["traj_id", "segment_id", "cluster_label"]].copy()
        for part in parts:
            if part is None or part.empty:
                continue
            seg_obs = seg_obs.merge(part, on=["traj_id", "segment_id", "cluster_label"], how="left")
        compact = []
        if not catalog.empty and "structure" in catalog.columns:
            compact = catalog.loc[
                catalog["structure"].astype(str).str.contains("compact"), "cluster_label"
            ].astype(int).tolist()
        if not compact:
            # fallback: cluster with smallest mean endpoint distance
            compact = [int(pd.Series(y).value_counts().index[0])]
        seg_obs["n_broken_adjacent"] = _interface_breaks(X, specs, y, compact)
        med_idx = {}
        for lab in sorted(set(y.tolist())):
            idx = np.where(y == lab)[0]
            med_idx[lab] = int(idx[medoid_index(dist[np.ix_(idx, idx)])]) if len(idx) else 0
        seg_obs["rmsd_to_medoid"] = [
            float(dist[i, med_idx[int(lab)]]) for i, lab in enumerate(y)
        ]
        seg_obs["family_id"] = fam_labels
        seg_obs["cohort"] = cube

    if "B" in tests and not seg_obs.empty:
        obs_cols = [
            c
            for c in (
                "assembly_rg",
                "assembly_relative_shape_anisotropy",
                "assembly_asphericity",
                "n_open_cation_pi",
                "cation_pi_pole_erank1",
                "cation_pi_pole_erank2",
                "equator_d1_erank1",
                "n_broken_adjacent",
                "total_inter_monomer_contacts",
                "n_active_interfaces",
                "rmsd_to_medoid",
            )
            if c in seg_obs.columns
        ]
        summaries = []
        two_pop = []
        # reoriented two-pop on site-pair PCA of reoriented vectors
        template = X.mean(axis=0)
        X_re, _g = reorient_rows(X, template, mappings)
        for lab in sorted(set(y.tolist())):
            part = seg_obs[seg_obs["cluster_label"] == lab]
            for col in obs_cols:
                vals = pd.to_numeric(part[col], errors="coerce").to_numpy()
                split = variance_split(vals, part["segment_id"].to_numpy(), part["traj_id"].to_numpy())
                summaries.append(
                    {
                        "cohort": cube,
                        "cluster_label": lab,
                        "observable": col,
                        "n": int(np.isfinite(vals).sum()),
                        "median": float(np.nanmedian(vals)),
                        **split,
                    }
                )
                flag = two_population_test(
                    vals,
                    part["traj_id"].to_numpy(),
                    min_segments=args.min_segments,
                    delta_bic=args.delta_bic,
                    min_ashman=args.min_ashman,
                    min_minor=args.min_minor,
                    min_minor_replicas=args.min_minor_replicas,
                )
                two_pop.append({"cohort": cube, "cluster_label": lab, "observable": col, "oriented": False, **flag})
            # joint first PCs of reoriented site pairs
            idx = np.where(y == lab)[0]
            if idx.size >= args.min_segments:
                from sklearn.decomposition import PCA

                pcs = PCA(n_components=min(2, X_re.shape[1]), random_state=0).fit_transform(X_re[idx])
                flag = two_population_test(
                    pcs,
                    keys.iloc[idx]["traj_id"].to_numpy(),
                    min_segments=args.min_segments,
                    delta_bic=args.delta_bic,
                    min_ashman=args.min_ashman,
                    min_minor=args.min_minor,
                    min_minor_replicas=args.min_minor_replicas,
                )
                two_pop.append(
                    {"cohort": cube, "cluster_label": lab, "observable": "reoriented_pca", "oriented": True, **flag}
                )
        overlap_rows = []
        labs = sorted(set(y.tolist()))
        for a, b in ((i, j) for i in labs for j in labs if i < j):
            pa = seg_obs[seg_obs["cluster_label"] == a]
            pb = seg_obs[seg_obs["cluster_label"] == b]
            for col in obs_cols:
                overlap_rows.append(
                    {
                        "cohort": cube,
                        "cluster_a": a,
                        "cluster_b": b,
                        "observable": col,
                        "overlap": overlap_coefficient(
                            pd.to_numeric(pa[col], errors="coerce").to_numpy(),
                            pd.to_numeric(pb[col], errors="coerce").to_numpy(),
                        ),
                    }
                )
        out["observable_summary"] = pd.DataFrame(summaries)
        out["two_population"] = pd.DataFrame(two_pop)
        out["observable_overlap"] = pd.DataFrame(overlap_rows)
        out["segment_observables"] = seg_obs
        _plot_observables(seg_obs, cube, args.out_dir / "plots" / f"observables_{cube}.png")

    holdout_cluster = pd.DataFrame()
    if "C" in tests:
        Xw, merged = _ward_features(cp_dir, keys)
        y_true = y
        guard_ari = float("nan")
        if Xw.shape[0] == len(y_true) and Xw.size:
            from sklearn.metrics import adjusted_rand_score

            y_refit = refit_ward_labels(Xw, k)
            guard_ari = float(adjusted_rand_score(y_true, y_refit))
        print(f"{cube}: Ward refit ARI vs stored labels = {guard_ari:.3f}", flush=True)
        folds = replica_holdout_folds(
            keys["traj_id"].astype(str), n_splits=args.n_splits, n_repeats=args.n_repeats,
            random_state=args.random_state,
        )
        ranking = score_cluster_discriminating_features(X, y, specs, "site_pair", n_permutations=0)
        top = ranking.head(10) if not ranking.empty else pd.DataFrame()
        fold_rows = []
        for fi, train_mask in enumerate(folds):
            test_mask = ~train_mask
            if train_mask.sum() < k or test_mask.sum() == 0:
                continue
            for variant, X_space in (("refit", Xw if Xw.shape[0] == len(y) else X), ("no_refit", X)):
                if X_space.size == 0 or X_space.shape[0] != len(y):
                    continue
                if variant == "refit":
                    y_tr_raw = refit_ward_labels(X_space[train_mask], k)
                    # match to reported labels on training
                    true_tr = y[train_mask]
                    labs_true = sorted(set(true_tr.tolist()))
                    labs_pred = sorted(set(y_tr_raw.tolist()))
                    table = np.zeros((len(labs_true), len(labs_pred)))
                    for i, a in enumerate(labs_true):
                        for j, b in enumerate(labs_pred):
                            table[i, j] = np.sum((true_tr == a) & (y_tr_raw == b))
                    mapped = hungarian_match(table)
                    y_tr = np.array([
                        labs_true[mapped[labs_pred.index(c)]] if c in labs_pred and mapped[labs_pred.index(c)] >= 0 else c
                        for c in y_tr_raw
                    ])
                else:
                    y_tr = y[train_mask]
                pred = nearest_centroid_labels(X_space[train_mask], y_tr, X_space[test_mask])
                from sklearn.metrics import adjusted_rand_score, accuracy_score

                fold_rows.append(
                    {
                        "cohort": cube,
                        "fold": fi,
                        "variant": variant,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "ari": float(adjusted_rand_score(y[test_mask], pred)),
                        "accuracy": float(accuracy_score(y[test_mask], pred)),
                    }
                )
                # per-cluster coverage / profile
                for lab in sorted(set(y.tolist())):
                    tr = train_mask & (y == lab)
                    te = test_mask & (y == lab)
                    if tr.sum() == 0 or te.sum() == 0:
                        continue
                    tr_i = np.where(tr)[0]
                    te_i = np.where(te)[0]
                    cov = coverage_at_radius(
                        X[tr],
                        X[te],
                        mappings,
                        train_dist=dist[np.ix_(tr_i, tr_i)],
                        test_to_train=dist[np.ix_(te_i, tr_i)],
                    )
                    base = X[train_mask].mean(axis=0)
                    d_tr = X[tr].mean(axis=0) - base
                    d_te = X[te].mean(axis=0) - base
                    r, _ = best_profile_r(d_tr, d_te, mappings)
                    train_ds = []
                    test_ds = []
                    if not top.empty:
                        for rec in top.itertuples(index=False):
                            feat = str(rec.feature)
                            j = next((i for i, s in enumerate(specs) if s[0] == feat), None)
                            if j is None:
                                continue
                            train_ds.append(cohens_d(X[tr, j], X[train_mask & (y != lab), j]))
                            test_ds.append(cohens_d(X[te, j], X[test_mask & (y != lab), j] if (test_mask & (y != lab)).any() else X[te, j]))
                    fold_rows[-1]  # keep last
                    fold_rows.append(
                        {
                            "cohort": cube,
                            "fold": fi,
                            "variant": variant,
                            "cluster_label": lab,
                            "coverage": cov,
                            "profile_r": r,
                            "sign_agreement": sign_agreement(train_ds, test_ds),
                            "abs_d_ratio": float(
                                np.nanmedian(
                                    [
                                        abs(b) / abs(a)
                                        for a, b in zip(train_ds, test_ds)
                                        if np.isfinite(a) and abs(a) > 0 and np.isfinite(b)
                                    ]
                                )
                            )
                            if train_ds
                            else float("nan"),
                            "n_train": int(tr.sum()),
                            "n_test": int(te.sum()),
                        }
                    )
        hold_df = pd.DataFrame(fold_rows)
        out["replica_holdout_folds"] = hold_df
        if not hold_df.empty and "cluster_label" in hold_df.columns:
            by = (
                hold_df.dropna(subset=["cluster_label"])
                .groupby(["cohort", "cluster_label", "variant"], as_index=False)
                .agg(
                    coverage=("coverage", "mean"),
                    profile_r=("profile_r", "mean"),
                    sign_agreement=("sign_agreement", "mean"),
                    n_folds=("fold", "nunique"),
                )
            )
            by["guard_ari"] = guard_ari
            holdout_cluster = by
            out["replica_holdout_by_cluster"] = by

        cons = []
        deltas = deformation_deltas(X, y)
        for lab in sorted(set(y.tolist())):
            idx = np.where(y == lab)[0]
            part = keys.iloc[idx]
            durations = segs.merge(part[["traj_id", "segment_id"]], on=["traj_id", "segment_id"], how="inner")
            n_ps = durations.groupby(durations["traj_id"].astype(str))["n_frames"].sum() * DEFAULT_ENDPOINT_STRIDE_PS
            for traj, ps in n_ps.items():
                if ps < args.min_replica_ps:
                    continue
                in_rep = part["traj_id"].astype(str) == str(traj)
                if in_rep.sum() == 0 or (~in_rep).sum() == 0:
                    continue
                d_rep = X[idx[in_rep.to_numpy()]].mean(axis=0) - X.mean(axis=0)
                d_rest = X[idx[(~in_rep).to_numpy()]].mean(axis=0) - X.mean(axis=0)
                r, _ = best_profile_r(d_rep, d_rest, mappings)
                d_mean, _ = symmetry_adjusted_distance(
                    X[idx[in_rep.to_numpy()]].mean(axis=0),
                    X[idx[(~in_rep).to_numpy()]].mean(axis=0),
                    mappings,
                )
                med = float(np.median(upper_from(dist[np.ix_(idx, idx)])))
                cons.append(
                    {
                        "cohort": cube,
                        "cluster_label": lab,
                        "traj_id": traj,
                        "n_ps": float(ps),
                        "profile_r": r,
                        "mean_distance": d_mean,
                        "distance_over_median": d_mean / med if med > 0 else float("nan"),
                    }
                )
        # between-cluster and split-half references
        labs = sorted(set(y.tolist()))
        for a, b in ((i, j) for i in labs for j in labs if i < j):
            r, _ = best_profile_r(deltas[a], deltas[b], mappings)
            cons.append({"cohort": cube, "cluster_label": a, "traj_id": f"cluster_{b}", "profile_r": r, "n_ps": np.nan})
        out["replica_profile_consistency"] = pd.DataFrame(cons)

    if "D" in tests:
        # D2 members
        member_rows = []
        fam_groups: dict[int, list[int]] = {}
        for lab, head in families.items():
            fam_groups.setdefault(head, []).append(lab)
        for head, members in fam_groups.items():
            if len(members) < 2:
                continue
            sizes = {m: int((y == m).sum()) for m in members}
            template_lab = max(sizes, key=sizes.get)
            tmpl = X[y == template_lab].mean(axis=0)
            idx = np.where(np.isin(y, members))[0]
            Y, choice = reorient_rows(X[idx], tmpl, mappings)
            d_fam, _ = pairwise_symmetry_distances(Y, [np.arange(Y.shape[1])])
            from src.ChangepointAnalysis.cluster_validation import pairwise_percentiles, upper_triangle

            fam_stats = pairwise_percentiles(upper_triangle(d_fam))
            mem_idx = np.where(y == template_lab)[0]
            d_mem, _ = pairwise_symmetry_distances(X[mem_idx], mappings)
            mem_stats = pairwise_percentiles(upper_triangle(d_mem))
            ratio = merge_ratio(fam_stats["p95"], mem_stats["p95"], max_ratio=args.merge_p95_max)
            y_mem = y[idx]
            share = []
            for m in members:
                local = y_mem == m
                if not local.any():
                    continue
                # cluster-level best mapping vs template
                _d, g0 = symmetry_adjusted_distance(tmpl, X[y == m].mean(axis=0), mappings)
                share.append(float(np.mean(choice[local] == g0)))
            acc = balanced_accuracy_grouped(Y, y_mem, keys.iloc[idx]["traj_id"].to_numpy())
            null = label_permutation_null(Y, y_mem, keys.iloc[idx]["traj_id"].to_numpy())
            e_dist, e_p = replica_block_energy_p(Y, y_mem, keys.iloc[idx]["traj_id"].to_numpy())
            member_rows.append(
                {
                    "cohort": cube,
                    "family_id": head,
                    "members": " ".join(str(m) for m in sorted(members)),
                    "template_cluster": template_lab,
                    "family_p95": fam_stats["p95"],
                    "member_p95": mem_stats["p95"],
                    **ratio,
                    "mean_relabel_share": float(np.mean(share)) if share else float("nan"),
                    "balanced_accuracy": acc,
                    "balanced_accuracy_null": null,
                    "energy_distance": e_dist,
                    "energy_p": e_p,
                }
            )
        out["merge_member_checks"] = pd.DataFrame(member_rows)

        neg_rows = []
        for a, b in NEGATIVE_CONTROL_PAIRS.get(cube, []):
            if a not in set(y.tolist()) or b not in set(y.tolist()):
                continue
            idx = np.where(np.isin(y, [a, b]))[0]
            tmpl = X[y == a].mean(axis=0)
            Y, _c = reorient_rows(X[idx], tmpl, mappings)
            acc = balanced_accuracy_grouped(Y, y[idx], keys.iloc[idx]["traj_id"].to_numpy())
            null = label_permutation_null(Y, y[idx], keys.iloc[idx]["traj_id"].to_numpy())
            e_dist, e_p = replica_block_energy_p(Y, y[idx], keys.iloc[idx]["traj_id"].to_numpy())
            d_a, _ = pairwise_symmetry_distances(X[y == a], mappings)
            d_ab = np.array(
                [symmetry_adjusted_distance(X[i], X[j], mappings)[0] for i in np.where(y == a)[0][:30] for j in np.where(y == b)[0][:30]]
            )
            from src.ChangepointAnalysis.cluster_validation import pairwise_percentiles, upper_triangle

            neg_rows.append(
                {
                    "cohort": cube,
                    "cluster_a": a,
                    "cluster_b": b,
                    "balanced_accuracy": acc,
                    "balanced_accuracy_null": null,
                    "energy_distance": e_dist,
                    "energy_p": e_p,
                    "within_a_p95": pairwise_percentiles(upper_triangle(d_a))["p95"],
                    "cross_median": float(np.median(d_ab)) if d_ab.size else float("nan"),
                    "should_separate": True,
                    "separates": bool(np.isfinite(acc) and acc > (null if np.isfinite(null) else 0.6) + 0.1),
                }
            )
        out["merge_negative_controls"] = pd.DataFrame(neg_rows)

        # D4 sensitivity: three groups × thresholds × linkage
        pair_tables = []
        for gname, gperms in (
            ("admissible", perms),
            ("rotations", rotation_contact_automorphisms()),
            ("contact", contact_automorphisms()),
        ):
            gmaps = mappings_from_perms(specs, gperms)
            table = profile_correlation_table(X, y, gmaps, gperms)
            table["group"] = gname
            pair_tables.append(table)
            pairs = [
                (int(r.cluster_a), int(r.cluster_b), float(r.r_best_relabelling))
                for r in table.itertuples(index=False)
            ]
            sens = sensitivity_family_table(sorted(set(y.tolist())), pairs)
            sens["cohort"] = cube
            sens["group"] = gname
            out.setdefault("merge_sensitivity", [])
            if isinstance(out["merge_sensitivity"], list):
                out["merge_sensitivity"].append(sens)
        if isinstance(out.get("merge_sensitivity"), list):
            out["merge_sensitivity"] = pd.concat(out["merge_sensitivity"], ignore_index=True)
        out["profile_correlations"] = pd.concat(pair_tables, ignore_index=True) if pair_tables else pd.DataFrame()

    # summary verdicts
    summary = []
    for lab in sorted(set(y.tolist())):
        row = {"cohort": cube, "k": k, "unit": "cluster", "cluster_label": lab, "family_id": families.get(lab, lab)}
        if "A" in tests and "cluster_diameter" in out:
            drow = out["cluster_diameter"]
            hit = drow[(drow["group_kind"] == "cluster") & (drow["cluster_label"] == lab)]
            if not hit.empty:
                row["test_a_p95"] = float(hit["p95"].iloc[0])
                row["test_a_n_eccentric"] = int(hit["n_eccentric"].iloc[0])
                row["test_a"] = (
                    "outlier_driven"
                    if int(hit["n_eccentric"].iloc[0]) > 0
                    else ("broad" if float(hit["p95"].iloc[0]) > (dwell * 4 if np.isfinite(dwell) else 2.0) else "compact")
                )
        if "B" in tests and "two_population" in out:
            flags = out["two_population"]
            flags = flags[flags["cluster_label"] == lab]
            if (flags["flag"] == "two_populations").any():
                row["test_b"] = "two_populations"
            elif (flags["flag"] == "single_outlier").any():
                row["test_b"] = "single_outlier"
            elif (flags["flag"] == "homogeneous").any():
                row["test_b"] = "homogeneous"
            else:
                row["test_b"] = "untested"
        if "C" in tests and not holdout_cluster.empty:
            hit = holdout_cluster[
                (holdout_cluster["cluster_label"] == lab) & (holdout_cluster["variant"] == "no_refit")
            ]
            if hit.empty:
                hit = holdout_cluster[holdout_cluster["cluster_label"] == lab]
            if not hit.empty:
                cov = float(hit["coverage"].mean())
                r = float(hit["profile_r"].mean())
                sg = float(hit["sign_agreement"].mean())
                row["test_c_coverage"] = cov
                row["test_c_r"] = r
                n_traj = int(keys.loc[y == lab, "traj_id"].nunique())
                if n_traj <= 1:
                    row["test_c"] = "single_replica"
                elif (
                    (not np.isfinite(cov) or cov >= args.coverage_min)
                    and (not np.isfinite(r) or r >= args.replica_r_min)
                    and (not np.isfinite(sg) or sg >= args.sign_agree_min)
                ):
                    row["test_c"] = "reproducible"
                else:
                    row["test_c"] = "not_reproducible"
        if "D" in tests and "merge_member_checks" in out and not out["merge_member_checks"].empty:
            fam = families.get(lab, lab)
            hit = out["merge_member_checks"][out["merge_member_checks"]["family_id"] == fam]
            if hit.empty:
                row["test_d"] = "not_merged"
            else:
                row["test_d"] = "merge_supported" if bool(hit["merge_supported"].iloc[0]) else "merge_weak"
        summary.append(row)
    out["validation_summary"] = pd.DataFrame(summary)
    return out


def upper_from(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    return dist[np.triu_indices(n, k=1)] if n > 1 else np.asarray([], dtype=float)


def main() -> int:
    args = parse_args()
    if args.apo_only and args.out_dir == ROOT / "output" / "endpoint_cluster_validation_physical_k":
        args.out_dir = args.out_dir / "apo_only"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plots").mkdir(parents=True, exist_ok=True)
    collected: dict[str, list[pd.DataFrame]] = {}
    for cube, k in _cubes(args):
        try:
            parts = run_cube(cube, k, args)
        except FileNotFoundError as exc:
            print(f"{cube}: {exc}", file=sys.stderr)
            return 1
        for name, df in parts.items():
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                continue
            collected.setdefault(name, []).append(df)
    for name, frames in collected.items():
        stacked = pd.concat(frames, ignore_index=True)
        stacked.to_csv(args.out_dir / f"{name}.csv", index=False)
        print(f"Wrote {args.out_dir / (name + '.csv')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
