"""Is a segment cluster a population state or one replica's excursion?

Ward clusters of changepoint segments can be carried by a single trajectory,
for example a cage that loosens late in its run. :func:`cluster_replica_composition`
counts the replicas behind each cluster, :func:`replica_window_rg` compares one
replica's assembly Rg inside the cluster's segments with the rest of its run,
and :func:`cluster_feature_margins` lists the segment features on which a
cluster lies outside the range of every other cluster.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

PAIR_MEAN_COLUMN_RE = re.compile(r"endpoint_dist_\d+_\d+_mean_mean")


def _group_rows(segments: pd.DataFrame, group: str) -> pd.DataFrame:
    if "group" in segments.columns:
        return segments[segments["group"].astype(str) == group]
    return segments


def _segment_frames(segments: pd.DataFrame) -> pd.Series:
    if "n_frames" in segments.columns:
        return pd.to_numeric(segments["n_frames"], errors="coerce").fillna(0.0)
    start = pd.to_numeric(segments["start_frame"], errors="coerce")
    end = pd.to_numeric(segments["end_frame"], errors="coerce")
    return (end - start + 1).fillna(0.0)


def cluster_replica_composition(
    segments: pd.DataFrame,
    *,
    group: str = "endpoint",
) -> pd.DataFrame:
    """Per cluster: segments, replicas, frames, and the share of its largest replica.

    ``top_traj`` is the replica contributing the most frames (``n_frames`` of
    ``segments_clustered.csv``); its frame and segment shares say whether the
    cluster is one trajectory's excursion.
    """
    segs = _group_rows(segments, group)
    if segs.empty:
        return pd.DataFrame()
    segs = segs.assign(_frames=_segment_frames(segs).to_numpy(), _tid=segs["traj_id"].astype(str))
    rows: list[dict] = []
    for label, grp in segs.groupby("cluster_label", sort=True):
        frames = grp.groupby("_tid")["_frames"].sum().sort_values(ascending=False, kind="stable")
        total = float(frames.sum())
        top = str(frames.index[0])
        rows.append(
            {
                "cluster_label": int(label),
                "n_segments": int(len(grp)),
                "n_traj": int(grp["_tid"].nunique()),
                "n_frames": total,
                "top_traj": top,
                "top_traj_frame_share": float(frames.iloc[0]) / total if total > 0 else np.nan,
                "top_traj_segment_share": float((grp["_tid"] == top).mean()),
            }
        )
    return pd.DataFrame(rows)


def _nan_stat(values: np.ndarray, fn) -> float:
    finite = values[np.isfinite(values)]
    return float(fn(finite)) if finite.size else float("nan")


def replica_window_rg(
    gsa: pd.DataFrame,
    replica_segments: pd.DataFrame,
    cluster_label: int,
    *,
    rg_col: str = "assembly_rg",
) -> dict:
    """One replica's assembly Rg inside one cluster's segments vs the rest of its run.

    ``gsa`` is that replica's per-frame GSA table (``frame`` and ``rg_col``),
    ``replica_segments`` all of its segments with ``cluster_label``. Segments
    sit on the detection stride, so each one covers the frames up to the next
    segment's start, and the last one runs to the end of the trajectory.
    ``ends_in_cluster`` is true when the replica's last segment is in the
    cluster, i.e. the run ends without leaving it.
    """
    segs = replica_segments.sort_values("start_frame", kind="stable")
    starts = pd.to_numeric(segs["start_frame"], errors="coerce").to_numpy(dtype=float)
    stops = np.append(starts[1:], np.inf)
    in_cluster = segs["cluster_label"].to_numpy() == cluster_label
    fr = pd.to_numeric(gsa["frame"], errors="coerce").to_numpy(dtype=float)
    rg = pd.to_numeric(gsa[rg_col], errors="coerce").to_numpy(dtype=float)
    inside = np.zeros(fr.shape, dtype=bool)
    for start, stop in zip(starts[in_cluster], stops[in_cluster]):
        inside |= (fr >= start) & (fr < stop)
    window = segs[in_cluster]
    out = {
        "window_first_frame": int(window["start_frame"].min()),
        "window_last_frame": int(window["end_frame"].max()),
        "ends_in_cluster": bool(in_cluster[-1]),
        "n_gsa_frames_in": int(inside.sum()),
        "n_gsa_frames_out": int((~inside).sum()),
        "rg_mean_in": _nan_stat(rg[inside], np.mean),
        "rg_max_in": _nan_stat(rg[inside], np.max),
        "rg_mean_out": _nan_stat(rg[~inside], np.mean),
        "rg_max_out": _nan_stat(rg[~inside], np.max),
    }
    if np.isfinite(rg).any():
        peak = int(np.nanargmax(rg))
        out["rg_max_frame"] = int(fr[peak])
        out["rg_max_in_window"] = bool(inside[peak])
    return out


def cluster_feature_margins(
    segments: pd.DataFrame,
    cluster_label: int,
    *,
    columns: Sequence[str] | None = None,
    group: str = "endpoint",
) -> pd.DataFrame:
    """Per feature: one cluster's segment mean against every other cluster's.

    ``margin`` is how far the cluster's mean lies outside the range of the other
    clusters' means (positive above, negative below, 0 inside the range), so a
    large |margin| marks a feature on which no other cluster comes close.
    Rows are sorted by |margin|. Default columns: the per-monomer-pair
    ``endpoint_dist_{i}_{j}_mean_mean``.
    """
    segs = _group_rows(segments, group)
    if columns is None:
        columns = [c for c in segs.columns if PAIR_MEAN_COLUMN_RE.fullmatch(c)]
    columns = list(columns)
    if not columns:
        return pd.DataFrame()
    means = segs.groupby("cluster_label")[columns].mean()
    if cluster_label not in means.index or len(means) < 2:
        return pd.DataFrame()
    own = means.loc[cluster_label]
    others = means.drop(index=cluster_label)
    lo, hi = others.min(), others.max()
    margin = np.where(own > hi, own - hi, np.where(own < lo, own - lo, 0.0))
    out = pd.DataFrame(
        {
            "feature": columns,
            "cluster_mean": own.to_numpy(dtype=float),
            "others_min": lo.to_numpy(dtype=float),
            "others_max": hi.to_numpy(dtype=float),
            "margin": margin.astype(float),
        }
    )
    order = np.argsort(-np.abs(out["margin"].to_numpy()), kind="stable")
    return out.iloc[order].reset_index(drop=True)


def isolated_replicas_by_k(
    segments_by_k: dict[int, pd.DataFrame],
    *,
    min_top_share: float = 0.5,
    group: str = "endpoint",
) -> pd.DataFrame:
    """Replicas the clustering isolates: per k, clusters whose largest replica holds ``min_top_share`` of the frames.

    One row per (k, flagged cluster), with the replica, its frame share, the
    cluster size, and the first and last frame of the replica's segments in
    that cluster. A replica that is isolated across many k is an outlier run,
    not a state shared by the ensemble.
    """
    rows: list[dict] = []
    for k in sorted(segments_by_k):
        segs = segments_by_k[k]
        comp = cluster_replica_composition(segs, group=group)
        if comp.empty:
            continue
        flagged = comp[comp["top_traj_frame_share"] >= float(min_top_share)]
        segs = _group_rows(segs, group)
        for rec in flagged.itertuples(index=False):
            own = segs[
                (segs["cluster_label"] == rec.cluster_label) & (segs["traj_id"].astype(str) == str(rec.top_traj))
            ]
            rows.append(
                {
                    "k": int(k),
                    "cluster_label": int(rec.cluster_label),
                    "traj_id": str(rec.top_traj),
                    "top_traj_frame_share": float(rec.top_traj_frame_share),
                    "cluster_n_segments": int(rec.n_segments),
                    "cluster_n_traj": int(rec.n_traj),
                    "replica_n_segments": int(len(own)),
                    "window_first_frame": int(own["start_frame"].min()) if len(own) else -1,
                    "window_last_frame": int(own["end_frame"].max()) if len(own) else -1,
                }
            )
    return pd.DataFrame(rows)


def gsa_paths_by_traj(gsa_dir: Path) -> dict[str, Path]:
    from .detection import canonicalize_traj_id
    from .feature_groups import traj_id_from_features_stem

    return {
        canonicalize_traj_id(traj_id_from_features_stem(p.stem)): p
        for p in sorted(Path(gsa_dir).glob("*_gsa_features.csv"))
    }


def audit_cluster_replicas(
    *,
    output_root: Path | str = "output",
    out_dir: Path | str,
    cubes: Optional[Sequence[str]] = None,
    own_elbow: bool = True,
    k: Optional[int] = None,
    k_selected: Optional[Path | str] = None,
    gsa_root: Path | str = "output/gsa_features_by_cube",
    min_top_share: float = 0.8,
    top_n: int = 10,
) -> dict[str, Path]:
    """Flag endpoint clusters carried by one replica; write audit CSVs."""
    from .assembly_qc import scan_assembly_qc_dir
    from .cluster_k_diagnostics import chosen_k_from_cohort, load_segments_at_k, load_selected_k
    from .detection import canonicalize_traj_id
    from .gsa_cohort_run import KNOWN_CUBES
    from .reporting import endpoint_changepoints_dir

    output_root = Path(output_root)
    out_dir = Path(out_dir)
    gsa_root = Path(gsa_root)
    selected_k = load_selected_k(k_selected)
    comp_parts, single_rows, margin_parts, qc_parts = [], [], [], []
    for cube in list(cubes) if cubes is not None else list(KNOWN_CUBES):
        cp_dir = endpoint_changepoints_dir(output_root, cube, own_elbow=own_elbow)
        if not (cp_dir / "clusters" / "endpoint").is_dir():
            continue
        cut = k or selected_k.get(cube) or chosen_k_from_cohort(cp_dir, group="endpoint")
        segs = load_segments_at_k(cp_dir, cut, group="endpoint")
        if segs.empty:
            continue
        comp = cluster_replica_composition(segs)
        comp["single_replica"] = comp["top_traj_frame_share"] >= min_top_share
        comp.insert(0, "k", int(cut))
        comp.insert(0, "cohort", cube)
        comp_parts.append(comp)
        gsa_dir = gsa_root / cube
        paths = gsa_paths_by_traj(gsa_dir) if gsa_dir.is_dir() else {}
        if gsa_dir.is_dir():
            qc_parts.append(scan_assembly_qc_dir(gsa_dir).assign(cohort=cube))
        for rec in comp[comp["single_replica"]].itertuples(index=False):
            tid = str(rec.top_traj)
            replica = segs[segs["traj_id"].astype(str) == tid]
            row = {
                "cohort": cube,
                "k": int(cut),
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
            margins = cluster_feature_margins(segs, int(rec.cluster_label)).head(top_n)
            if not margins.empty:
                margins.insert(0, "traj_id", tid)
                margins.insert(0, "cluster_label", int(rec.cluster_label))
                margins.insert(0, "k", int(cut))
                margins.insert(0, "cohort", cube)
                margin_parts.append(margins)
    if not comp_parts:
        raise FileNotFoundError("No endpoint clusterings found")
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
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "cluster_replica_composition": pd.concat(comp_parts, ignore_index=True),
        "single_replica_clusters": single,
        "single_replica_feature_margins": (
            pd.concat(margin_parts, ignore_index=True) if margin_parts else pd.DataFrame()
        ),
    }
    written: dict[str, Path] = {}
    for name, table in tables.items():
        path = out_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        written[name] = path
    return written


def write_replica_outliers(
    *,
    output_root: Path | str = "output",
    gsa_root: Path | str = "output/gsa_features_by_cube",
    k_selected: Optional[Path | str] = None,
    cubes: Optional[Sequence[str]] = None,
    k_min: int = 2,
    k_max: int = 15,
    min_top_share: float = 0.5,
    out_dir: Path | str = "output/replica_outliers",
) -> dict[str, Path]:
    """Dissociation QC plus isolation-by-clustering across Ward cuts."""
    from .assembly_qc import scan_assembly_qc_dir
    from .cluster_k_diagnostics import load_segments_at_k, load_selected_k
    from .detection import canonicalize_traj_id
    from .feature_groups import traj_id_from_features_stem
    from .gsa_cohort_run import KNOWN_CUBES
    from .reporting import endpoint_changepoints_dir

    output_root = Path(output_root)
    gsa_root = Path(gsa_root)
    out_dir = Path(out_dir)
    selected = load_selected_k(k_selected)
    qc_parts, iso_parts, rows = [], [], []
    seg_cache: dict[tuple[str, int], pd.DataFrame] = {}
    for cube in list(cubes) if cubes is not None else list(KNOWN_CUBES):
        gsa_dir = gsa_root / cube
        qc_parts.append(scan_assembly_qc_dir(gsa_dir).assign(cohort=cube))
        cp_dir = endpoint_changepoints_dir(output_root, cube, own_elbow=True)
        by_k = {}
        for cut in range(k_min, k_max + 1):
            segs = load_segments_at_k(cp_dir, cut, group="endpoint")
            if not segs.empty:
                by_k[cut] = segs
                seg_cache[(cube, cut)] = segs
        iso = isolated_replicas_by_k(by_k, min_top_share=min_top_share)
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
                "cuts_isolated": ",".join(str(int(cut)) for cut in sorted(grp["k"].unique())),
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
            path = gsa_root / cube / f"{cube}_{canonicalize_traj_id(tid)}_gsa_features.csv"
            if not path.is_file():
                cands = [
                    p for p in (gsa_root / cube).glob("*_gsa_features.csv")
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
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "replica_outliers_by_k": out_dir / "replica_outliers_by_k.csv",
        "replica_outliers": out_dir / "replica_outliers.csv",
    }
    iso_all.to_csv(written["replica_outliers_by_k"], index=False)
    out.to_csv(written["replica_outliers"], index=False)
    return written
