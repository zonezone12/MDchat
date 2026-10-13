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
from typing import Sequence

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
