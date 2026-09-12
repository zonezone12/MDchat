"""Attribute directed endpoint-cluster transitions to site-pair distance features.

For each consecutive segment transition (from_cluster → to_cluster), rank the
raw ``endpoint_dist_{i}s{a}_{j}s{b}`` columns by standardized source→destination
effect and point-biserial correlation, then map winners back to monomer/site/atom
identities from ``endpoint_sites.csv``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from src.utils.run_log import log_event

from .feature_groups import resolve_features_csv
from .endpoint_features import (
    classify_paper_d1_state,
    list_paper_d1_columns,
    summarize_paper_d1_by_segments,
)

_SITE_PAIR_RE = re.compile(
    r"^endpoint_dist_(\d+)s(\d+)_(\d+)s(\d+)$"
)
_PAPER_D1_RE = re.compile(
    r"^paper_d1_m(\d+)(s\d+|r[123])_m(\d+)(s\d+|r[123])$"
)


@dataclass
class EndpointTransitionAttributionConfig:
    """Parameters for endpoint transition → feature attribution."""

    top_n: int = 10
    min_abs_correlation: float = 0.0
    effect_weight: float = 0.5
    correlation_weight: float = 0.5
    group: str = "endpoint"
    # If set, use short windows around the boundary instead of full segments.
    window_frames: Optional[int] = None
    # Network plot: how many pair labels to show per edge.
    network_label_top_n: int = 3
    include_paper_d1: bool = True
    paper_d1_open_lo: float = 4.5
    paper_d1_open_hi: float = 5.5


def parse_site_pair_feature(name: str) -> Optional[tuple[int, int, int, int]]:
    """Parse ``endpoint_dist_{i}s{a}_{j}s{b}`` → (mon_i, site_a, mon_j, site_b)."""
    m = _SITE_PAIR_RE.fullmatch(str(name))
    if not m:
        return None
    return tuple(int(x) for x in m.groups())  # type: ignore[return-value]


def parse_paper_d1_feature(name: str) -> Optional[tuple[int, Any, int, Any]]:
    """Parse ``paper_d1_m{i}r2_m{j}r3`` or legacy ``…s3…s7``."""
    m = _PAPER_D1_RE.fullmatch(str(name))
    if not m:
        return None

    def _tok(raw: str) -> Any:
        if raw.startswith("s"):
            return int(raw[1:])
        return raw

    return (int(m.group(1)), _tok(m.group(2)), int(m.group(3)), _tok(m.group(4)))


def list_site_pair_columns(columns: Sequence[str]) -> list[str]:
    """Return columns that encode raw site–site distances."""
    return [c for c in columns if parse_site_pair_feature(c) is not None]


def list_attribution_feature_columns(
    columns: Sequence[str],
    *,
    include_paper_d1: bool = True,
) -> list[str]:
    """Site-pair columns plus optional corrected paper-d1 columns."""
    cols = list_site_pair_columns(columns)
    if include_paper_d1:
        cols = cols + list_paper_d1_columns(columns)
    return cols


def load_endpoint_sites_map(
    sites_path: Path | str,
    *,
    traj_id: Optional[str] = None,
) -> pd.DataFrame:
    """Load ``endpoint_sites.csv``, optionally filtered to one trajectory."""
    sites = pd.read_csv(sites_path)
    if traj_id is not None and "traj_id" in sites.columns:
        filtered = sites[sites["traj_id"] == traj_id]
        if not filtered.empty:
            return filtered.reset_index(drop=True)
        # Topology-defined sites may be stored under any traj_id; take the first.
        first = sites["traj_id"].iloc[0]
        return sites[sites["traj_id"] == first].reset_index(drop=True)
    return sites


def resolve_site_pair_metadata(
    feature: str,
    sites_df: pd.DataFrame,
    *,
    d1_atoms_df: Optional[pd.DataFrame] = None,
) -> dict[str, Any]:
    """Map a site-pair or paper-d1 feature name to atom/site metadata."""
    parsed = parse_site_pair_feature(feature)
    feature_block = "endpoint_site_pair"
    if parsed is None:
        parsed = parse_paper_d1_feature(feature)
        feature_block = "paper_d1"
    empty = {
        "feature": feature,
        "feature_block": feature_block if parsed is not None else "",
        "monomer_i": np.nan,
        "site_i": np.nan,
        "kind_i": "",
        "atom_ids_i": "",
        "monomer_j": np.nan,
        "site_j": np.nan,
        "kind_j": "",
        "atom_ids_j": "",
        "pair_label": feature,
        "d1_state_hint": "",
    }
    if parsed is None:
        return empty
    mon_i, site_i, mon_j, site_j = parsed

    def _lookup_site(monomer: int, site_key: Any) -> dict[str, Any]:
        if (
            isinstance(site_key, str)
            and not str(site_key).isdigit()
            and "role" in sites_df.columns
        ):
            hit = sites_df[
                (sites_df["monomer"] == monomer) & (sites_df["role"].astype(str) == str(site_key))
            ]
        else:
            hit = sites_df[
                (sites_df["monomer"] == monomer) & (sites_df["site_index"] == int(site_key))
            ]
        if hit.empty:
            return {"kind": "", "atom_ids": "", "n_atoms": 0, "d1_ring_atom_id": ""}
        row = hit.iloc[0]
        return {
            "kind": str(row.get("kind", "")),
            "atom_ids": str(row.get("atom_ids", "")),
            "n_atoms": int(row.get("n_atoms", 0)) if pd.notna(row.get("n_atoms")) else 0,
            "d1_ring_atom_id": str(row.get("d1_ring_atom_id", "") or ""),
        }

    def _lookup_d1_ring(monomer: int, site_key: Any) -> str:
        if d1_atoms_df is not None and not d1_atoms_df.empty:
            if (
                isinstance(site_key, str)
                and not str(site_key).isdigit()
                and "role" in d1_atoms_df.columns
            ):
                hit = d1_atoms_df[
                    (d1_atoms_df["monomer"] == monomer)
                    & (d1_atoms_df["role"].astype(str) == str(site_key))
                ]
            else:
                hit = d1_atoms_df[
                    (d1_atoms_df["monomer"] == monomer)
                    & (d1_atoms_df["endpoint_site"] == int(site_key))
                ]
            if not hit.empty:
                return str(int(hit.iloc[0]["d1_ring_atom_id"]))
        site = _lookup_site(monomer, site_key)
        return site.get("d1_ring_atom_id", "") or ""

    si = _lookup_site(mon_i, site_i)
    sj = _lookup_site(mon_j, site_j)
    if feature_block == "paper_d1":
        atom_i = _lookup_d1_ring(mon_i, site_i)
        atom_j = _lookup_d1_ring(mon_j, site_j)
        label = f"d1:M{mon_i}:s{site_i}_ring↔M{mon_j}:s{site_j}_ring"
        return {
            "feature": feature,
            "feature_block": feature_block,
            "monomer_i": mon_i,
            "site_i": site_i,
            "kind_i": "d1_ring",
            "atom_ids_i": atom_i,
            "monomer_j": mon_j,
            "site_j": site_j,
            "kind_j": "d1_ring",
            "atom_ids_j": atom_j,
            "pair_label": label,
            "d1_state_hint": "open≈4.5–5.5Å",
        }

    label = (
        f"M{mon_i}:{si['kind'] or 's'}{site_i}"
        f"↔M{mon_j}:{sj['kind'] or 's'}{site_j}"
    )
    return {
        "feature": feature,
        "feature_block": feature_block,
        "monomer_i": mon_i,
        "site_i": site_i,
        "kind_i": si["kind"],
        "atom_ids_i": si["atom_ids"],
        "monomer_j": mon_j,
        "site_j": site_j,
        "kind_j": sj["kind"],
        "atom_ids_j": sj["atom_ids"],
        "pair_label": label,
        "d1_state_hint": "",
    }


def build_transition_events(
    segments_df: pd.DataFrame,
    *,
    group: str = "endpoint",
) -> pd.DataFrame:
    """Build consecutive segment transitions from a clustered segment table.

    Returns one row per directed boundary where ``cluster_label`` changes.
    """
    required = {
        "traj_id",
        "segment_id",
        "cluster_label",
        "start_frame",
        "end_frame",
    }
    missing = required - set(segments_df.columns)
    if missing:
        raise ValueError(f"segments_df missing columns: {sorted(missing)}")

    df = segments_df
    if "group" in df.columns:
        df = df[df["group"] == group].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "traj_id",
                "from_cluster",
                "to_cluster",
                "from_segment_id",
                "to_segment_id",
                "from_start_frame",
                "from_end_frame",
                "to_start_frame",
                "to_end_frame",
                "boundary_frame",
                "boundary_time_ps",
                "from_n_frames",
                "to_n_frames",
            ]
        )

    sort_cols = ["traj_id", "segment_id"]
    if "start_frame" in df.columns:
        sort_cols = ["traj_id", "start_frame", "segment_id"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    event_id = 0
    for traj_id, grp in df.groupby("traj_id", sort=False):
        ordered = grp.reset_index(drop=True)
        for i in range(len(ordered) - 1):
            src = ordered.iloc[i]
            dst = ordered.iloc[i + 1]
            from_cl = int(src["cluster_label"])
            to_cl = int(dst["cluster_label"])
            if from_cl == to_cl:
                continue
            boundary_frame = int(dst["start_frame"])
            if "start_ps" in dst and pd.notna(dst["start_ps"]):
                boundary_time = float(dst["start_ps"])
            elif "end_ps" in src and pd.notna(src["end_ps"]):
                boundary_time = float(src["end_ps"])
            else:
                boundary_time = float(boundary_frame)
            rows.append(
                {
                    "event_id": event_id,
                    "traj_id": str(traj_id),
                    "from_cluster": from_cl,
                    "to_cluster": to_cl,
                    "from_segment_id": int(src["segment_id"]),
                    "to_segment_id": int(dst["segment_id"]),
                    "from_start_frame": int(src["start_frame"]),
                    "from_end_frame": int(src["end_frame"]),
                    "to_start_frame": int(dst["start_frame"]),
                    "to_end_frame": int(dst["end_frame"]),
                    "boundary_frame": boundary_frame,
                    "boundary_time_ps": boundary_time,
                    "from_n_frames": int(src.get("n_frames", src["end_frame"] - src["start_frame"] + 1)),
                    "to_n_frames": int(dst.get("n_frames", dst["end_frame"] - dst["start_frame"] + 1)),
                }
            )
            event_id += 1
    return pd.DataFrame(rows)


def _select_segment_frames(
    feat: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    *,
    window_frames: Optional[int] = None,
    side: str = "source",
    boundary_frame: Optional[int] = None,
) -> pd.DataFrame:
    """Select feature rows by actual ``frame`` values (stride-safe)."""
    if "frame" not in feat.columns:
        raise ValueError("Feature CSV missing 'frame' column")
    frames = feat["frame"].to_numpy(dtype=int)
    if window_frames is not None and boundary_frame is not None:
        if side == "source":
            lo = boundary_frame - int(window_frames)
            hi = boundary_frame - 1
            mask = (frames >= lo) & (frames <= hi) & (frames >= start_frame) & (frames <= end_frame)
        else:
            lo = boundary_frame
            hi = boundary_frame + int(window_frames) - 1
            mask = (frames >= lo) & (frames <= hi) & (frames >= start_frame) & (frames <= end_frame)
    else:
        mask = (frames >= start_frame) & (frames <= end_frame)
    return feat.loc[mask]


def _point_biserial(
    source_vals: np.ndarray,
    dest_vals: np.ndarray,
) -> float:
    """Point-biserial correlation treating destination membership as the binary label."""
    src = np.asarray(source_vals, dtype=float)
    dst = np.asarray(dest_vals, dtype=float)
    src = src[np.isfinite(src)]
    dst = dst[np.isfinite(dst)]
    if src.size < 1 or dst.size < 1:
        return float("nan")
    y = np.concatenate([np.zeros(src.size), np.ones(dst.size)])
    x = np.concatenate([src, dst])
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    # Pearson of continuous x with binary y
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = np.sqrt((x_c * x_c).sum() * (y_c * y_c).sum())
    if denom <= 0:
        return float("nan")
    return float((x_c * y_c).sum() / denom)


def score_transition_features(
    events: pd.DataFrame,
    features_dir: Path | str,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    features_suffix: str = "_endpoint_features.csv",
    window_frames: Optional[int] = None,
    effect_weight: float = 0.5,
    correlation_weight: float = 0.5,
    include_paper_d1: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score site-pair / paper-d1 features for each transition event and aggregate.

    Returns
    -------
    event_feature_df
        One row per (event, feature).
    ranking_df
        One row per (from_cluster, to_cluster, feature), ranked within transition.
    """
    features_dir = Path(features_dir)
    if events.empty:
        empty_evt = pd.DataFrame()
        empty_rank = pd.DataFrame()
        return empty_evt, empty_rank

    # Discover feature columns from the first available CSV if not provided.
    cols = list(feature_columns) if feature_columns is not None else None
    event_rows: list[dict[str, Any]] = []
    global_scales: dict[str, float] = {}

    # First pass: load scales from all involved trajectories
    traj_cache: dict[str, pd.DataFrame] = {}
    for traj_id in events["traj_id"].unique():
        path = resolve_features_csv(
            features_dir, str(traj_id), suffix=features_suffix
        )
        if path is None:
            log_event(
                "warning",
                f"No endpoint features CSV for traj_id={traj_id} under {features_dir}",
                component="endpoint_transition_attribution",
            )
            continue
        df = pd.read_csv(path)
        if cols is None:
            cols = list_attribution_feature_columns(
                df.columns, include_paper_d1=include_paper_d1
            )
        traj_cache[str(traj_id)] = df
        for c in cols or []:
            if c not in df.columns:
                continue
            vals = df[c].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                s = float(np.std(vals))
                if c not in global_scales or (
                    np.isfinite(s) and s > global_scales.get(c, 0.0)
                ):
                    # Use pooled std across trajs: accumulate then finalize below
                    global_scales.setdefault(c, 0.0)

    if not cols:
        log_event(
            "warning",
            (
                "No attribution feature columns found "
                "(need --include-site-pairs and/or paper_d1 columns)."
            ),
            component="endpoint_transition_attribution",
        )
        return pd.DataFrame(), pd.DataFrame()

    # Compute pooled std properly
    for c in cols:
        chunks: list[np.ndarray] = []
        for df in traj_cache.values():
            if c in df.columns:
                v = df[c].to_numpy(dtype=float)
                chunks.append(v[np.isfinite(v)])
        if chunks:
            all_v = np.concatenate(chunks)
            global_scales[c] = float(np.std(all_v)) if all_v.size else float("nan")
        else:
            global_scales[c] = float("nan")

    for event in events.itertuples(index=False):
        traj_id = str(event.traj_id)
        feat = traj_cache.get(traj_id)
        if feat is None:
            continue
        src = _select_segment_frames(
            feat,
            int(event.from_start_frame),
            int(event.from_end_frame),
            window_frames=window_frames,
            side="source",
            boundary_frame=int(event.boundary_frame),
        )
        dst = _select_segment_frames(
            feat,
            int(event.to_start_frame),
            int(event.to_end_frame),
            window_frames=window_frames,
            side="dest",
            boundary_frame=int(event.boundary_frame),
        )
        if src.empty or dst.empty:
            continue
        for c in cols:
            if c not in feat.columns:
                continue
            src_vals = src[c].to_numpy(dtype=float)
            dst_vals = dst[c].to_numpy(dtype=float)
            src_mean = float(np.nanmean(src_vals)) if np.isfinite(src_vals).any() else float("nan")
            dst_mean = float(np.nanmean(dst_vals)) if np.isfinite(dst_vals).any() else float("nan")
            delta = dst_mean - src_mean
            scale = global_scales.get(c, float("nan"))
            std_delta = (
                delta / scale if np.isfinite(scale) and scale > 0 and np.isfinite(delta) else float("nan")
            )
            corr = _point_biserial(src_vals, dst_vals)
            event_rows.append(
                {
                    "event_id": int(event.event_id),
                    "traj_id": traj_id,
                    "from_cluster": int(event.from_cluster),
                    "to_cluster": int(event.to_cluster),
                    "from_segment_id": int(event.from_segment_id),
                    "to_segment_id": int(event.to_segment_id),
                    "boundary_frame": int(event.boundary_frame),
                    "boundary_time_ps": float(event.boundary_time_ps),
                    "feature": c,
                    "source_mean": src_mean,
                    "dest_mean": dst_mean,
                    "delta_angstrom": delta,
                    "standardized_delta": std_delta,
                    "abs_standardized_delta": abs(std_delta) if np.isfinite(std_delta) else float("nan"),
                    "point_biserial_corr": corr,
                    "abs_point_biserial_corr": abs(corr) if np.isfinite(corr) else float("nan"),
                    "source_n_rows": int(np.isfinite(src_vals).sum()),
                    "dest_n_rows": int(np.isfinite(dst_vals).sum()),
                }
            )

    event_df = pd.DataFrame(event_rows)
    if event_df.empty:
        return event_df, pd.DataFrame()

    agg_rows: list[dict[str, Any]] = []
    for (from_cl, to_cl, feature), grp in event_df.groupby(
        ["from_cluster", "to_cluster", "feature"], sort=True
    ):
        n_events = int(grp["event_id"].nunique())
        n_traj = int(grp["traj_id"].nunique())
        mean_delta = float(grp["delta_angstrom"].mean())
        mean_std_delta = float(grp["standardized_delta"].mean())
        mean_abs_std = float(grp["abs_standardized_delta"].mean())
        mean_corr = float(grp["point_biserial_corr"].mean())
        mean_abs_corr = float(grp["abs_point_biserial_corr"].mean())
        # Direction consistency of signed deltas across events
        signs = np.sign(grp["delta_angstrom"].dropna().to_numpy(dtype=float))
        if signs.size and np.isfinite(mean_delta) and mean_delta != 0:
            consistency = float(np.mean(signs == np.sign(mean_delta)))
        else:
            consistency = float("nan")
        # Combined score in [0, ~1+] after normalizing abs effect by max within group later
        agg_rows.append(
            {
                "from_cluster": int(from_cl),
                "to_cluster": int(to_cl),
                "feature": feature,
                "n_events": n_events,
                "n_trajectories": n_traj,
                "mean_source": float(grp["source_mean"].mean()),
                "mean_dest": float(grp["dest_mean"].mean()),
                "mean_delta_angstrom": mean_delta,
                "mean_standardized_delta": mean_std_delta,
                "mean_abs_standardized_delta": mean_abs_std,
                "mean_point_biserial_corr": mean_corr,
                "mean_abs_point_biserial_corr": mean_abs_corr,
                "consistent_direction_fraction": consistency,
                "is_descriptive": n_events < 2,
            }
        )

    ranking = pd.DataFrame(agg_rows)
    if ranking.empty:
        return event_df, ranking

    # Within each directed transition, normalize abs effect to [0, 1] then combine.
    ranking["combined_score"] = np.nan
    ranking["rank"] = np.nan
    for (from_cl, to_cl), grp_idx in ranking.groupby(
        ["from_cluster", "to_cluster"]
    ).groups.items():
        sub = ranking.loc[list(grp_idx)].copy()
        max_eff = sub["mean_abs_standardized_delta"].max()
        if np.isfinite(max_eff) and max_eff > 0:
            norm_eff = sub["mean_abs_standardized_delta"] / max_eff
        else:
            norm_eff = pd.Series(0.0, index=sub.index)
        corr_term = sub["mean_abs_point_biserial_corr"].fillna(0.0)
        score = effect_weight * norm_eff + correlation_weight * corr_term
        ranking.loc[sub.index, "combined_score"] = score
        order = score.sort_values(ascending=False).index
        for rank_i, idx in enumerate(order, start=1):
            ranking.loc[idx, "rank"] = rank_i

    ranking = ranking.sort_values(
        ["from_cluster", "to_cluster", "rank"]
    ).reset_index(drop=True)
    return event_df, ranking


def attach_site_metadata(
    ranking_df: pd.DataFrame,
    sites_df: pd.DataFrame,
    *,
    d1_atoms_df: Optional[pd.DataFrame] = None,
    open_lo: float = 4.5,
    open_hi: float = 5.5,
) -> pd.DataFrame:
    """Add site/atom metadata columns to a ranking table."""
    if ranking_df.empty:
        return ranking_df
    meta_rows = [
        resolve_site_pair_metadata(str(f), sites_df, d1_atoms_df=d1_atoms_df)
        for f in ranking_df["feature"]
    ]
    meta = pd.DataFrame(meta_rows)
    keep = [c for c in meta.columns if c != "feature"]
    out = ranking_df.copy()
    for c in keep:
        out[c] = meta[c].to_numpy()
    # Classify mean destination distance for paper-d1 features.
    if "feature_block" in out.columns and "mean_dest" in out.columns:
        states = []
        for _, row in out.iterrows():
            if row.get("feature_block") == "paper_d1":
                states.append(
                    classify_paper_d1_state(
                        float(row["mean_dest"]),
                        open_lo=open_lo,
                        open_hi=open_hi,
                    )
                )
            else:
                states.append("")
        out["d1_dest_state"] = states
    return out


def top_features_per_transition(
    ranking_df: pd.DataFrame,
    *,
    top_n: int = 10,
    min_abs_correlation: float = 0.0,
) -> pd.DataFrame:
    """Filter rankings to top-N features per directed transition."""
    if ranking_df.empty:
        return ranking_df
    parts: list[pd.DataFrame] = []
    for _, grp in ranking_df.groupby(["from_cluster", "to_cluster"], sort=True):
        sub = grp
        if min_abs_correlation > 0:
            sub = sub[
                sub["mean_abs_point_biserial_corr"].fillna(0.0) >= min_abs_correlation
            ]
        parts.append(sub.nsmallest(top_n, "rank") if "rank" in sub.columns else sub.head(top_n))
    if not parts:
        return ranking_df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


# ── Visualizations ──────────────────────────────────────────────────────────


def plot_transition_feature_heatmap(
    top_df: pd.DataFrame,
    output_path: Path | str,
    *,
    title: str = "Endpoint transition drivers (standardized Δ)",
) -> Optional[Path]:
    """Heatmap of directed transitions × top endpoint pairs (signed std effect)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if top_df.empty:
        return None

    work = top_df.copy()
    work["transition"] = work.apply(
        lambda r: f"{int(r['from_cluster'])}→{int(r['to_cluster'])}", axis=1
    )
    label_col = "pair_label" if "pair_label" in work.columns else "feature"
    # Union of features ordered by mean |score|
    feat_order = (
        work.groupby(label_col)["mean_abs_standardized_delta"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    trans_order = sorted(work["transition"].unique(), key=lambda s: tuple(map(int, s.split("→"))))

    mat = pd.DataFrame(np.nan, index=trans_order, columns=feat_order)
    corr_mat = pd.DataFrame(np.nan, index=trans_order, columns=feat_order)
    for _, row in work.iterrows():
        t = row["transition"]
        f = row[label_col]
        mat.loc[t, f] = row["mean_standardized_delta"]
        corr_mat.loc[t, f] = row["mean_point_biserial_corr"]

    vmax = float(np.nanmax(np.abs(mat.to_numpy()))) if mat.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig_w = max(8.0, 0.55 * len(feat_order) + 3)
    fig_h = max(3.5, 0.7 * len(trans_order) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(feat_order)))
    ax.set_xticklabels(feat_order, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(trans_order)))
    ax.set_yticklabels(trans_order)
    ax.set_xlabel("Endpoint site-pair")
    ax.set_ylabel("Cluster transition")
    ax.set_title(title)
    for i in range(len(trans_order)):
        for j in range(len(feat_order)):
            val = mat.to_numpy(dtype=float)[i, j]
            if not np.isfinite(val):
                continue
            corr = corr_mat.to_numpy(dtype=float)[i, j]
            text = f"{val:+.2f}"
            if np.isfinite(corr):
                text += f"\nr={corr:+.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="Mean standardized Δ (dest − source)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_attributed_transition_network(
    top_df: pd.DataFrame,
    output_path: Path | str,
    *,
    n_clusters: Optional[int] = None,
    label_top_n: int = 3,
    title: str = "Attributed endpoint cluster transitions",
) -> Optional[Path]:
    """Transition network with edge labels listing top mapped endpoint pairs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if top_df.empty:
        return None

    edges = (
        top_df.groupby(["from_cluster", "to_cluster"], as_index=False)
        .agg(n_events=("n_events", "max"))
    )
    if n_clusters is None:
        n_clusters = int(
            max(edges["from_cluster"].max(), edges["to_cluster"].max())
        ) + 1

    label_col = "pair_label" if "pair_label" in top_df.columns else "feature"
    edge_labels: dict[tuple[int, int], str] = {}
    for (fc, tc), grp in top_df.groupby(["from_cluster", "to_cluster"]):
        lines = []
        for _, row in grp.nsmallest(label_top_n, "rank").iterrows():
            delta = row["mean_delta_angstrom"]
            lab = str(row[label_col])
            if np.isfinite(delta):
                lines.append(f"{lab} ({delta:+.2f} Å)")
            else:
                lines.append(lab)
        edge_labels[(int(fc), int(tc))] = "\n".join(lines)

    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    positions = {
        i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)
    }
    max_count = int(edges["n_events"].max()) if not edges.empty else 1

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_aspect("equal")
    ax.axis("off")

    for _, erow in edges.iterrows():
        i, j = int(erow["from_cluster"]), int(erow["to_cluster"])
        count = int(erow["n_events"])
        x0, y0 = positions[i]
        x1, y1 = positions[j]
        lw = 0.8 + 3.0 * (count / max(max_count, 1))
        dx, dy = x1 - x0, y1 - y0
        dist = np.hypot(dx, dy) or 1.0
        shrink = 0.2
        start = (x0 + shrink * dx / dist, y0 + shrink * dy / dist)
        end = (x1 - shrink * dx / dist, y1 - shrink * dy / dist)
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=14,
            lw=lw,
            color="0.35",
            connectionstyle="arc3,rad=0.15",
        )
        ax.add_patch(arrow)
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        # Offset label slightly outward from chord midpoint
        ox, oy = -0.08 * dy / dist, 0.08 * dx / dist
        text = edge_labels.get((i, j), str(count))
        ax.text(
            mx + ox,
            my + oy,
            text,
            ha="center",
            va="center",
            fontsize=7,
            color="0.15",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="0.8"),
        )

    for i, (x, y) in positions.items():
        ax.scatter([x], [y], s=1100, c="white", edgecolors="black", zorder=3)
        ax.text(x, y, str(i), ha="center", va="center", fontsize=12, zorder=4, fontweight="bold")

    ax.set_title(title)
    # Expand limits so labels fit
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _parse_atom_id_field(value: Any) -> list[int]:
    """Parse space- or pipe-separated MDA atom ids from ``endpoint_sites.csv``."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    return [int(tok) for tok in re.split(r"[\s|]+", text) if tok]


_RANK_COLORS = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#469990",
    "#dcbeff",
]

# Default atom-site colors requested for BMMpM-style tooth / tip sites.
DEFAULT_SITE_HIGHLIGHT_COLORS: dict[str, str] = {
    "R2": "#4363d8",
    "R3": "#f58231",
    "R1": "#e6194b",
    "s3:A": "#4363d8",
    "s6:A": "#f58231",
    "s7:A": "#e6194b",
}


def _site_label(site_index: Any, kind: Any, role: Any = "") -> str:
    """Canonical label: Murata role (``R1``) or hull ``s3:A``."""
    if role:
        from src.EndpointAnalyzer.gsa_site_map import plot_label_for_role

        return plot_label_for_role(str(role), str(kind or ""))
    try:
        idx = int(site_index)
    except (TypeError, ValueError):
        return str(role or "")
    kind_s = str(kind or "").strip().lower()
    letter = "A" if kind_s.startswith("atom") else "R" if kind_s.startswith("ring") else "?"
    return f"s{idx}:{letter}"


def _resolve_site_color(
    site_index: Any,
    kind: Any,
    site_colors: Optional[dict[str, str]] = None,
    *,
    default: str = "#f5c542",
    role: Any = "",
) -> str:
    colors = site_colors if site_colors is not None else DEFAULT_SITE_HIGHLIGHT_COLORS
    label = _site_label(site_index, kind, role=role)
    if label in colors:
        return colors[label]
    # Also allow bare keys like "s3" or "3:A"
    for key, val in colors.items():
        if key.lower() == label.lower():
            return val
    return default


_NGL_TRANSITION_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; overflow: hidden; font-family: system-ui, sans-serif; }}
  #viewport {{ width: 100vw; height: 100vh; }}
  #sidebar {{
    position: absolute; top: 12px; left: 12px; width: 320px; max-height: calc(100vh - 24px);
    overflow-y: auto; color: #e8e8e8; background: rgba(0,0,0,0.78);
    padding: 12px 14px; border-radius: 10px; backdrop-filter: blur(8px);
    font-size: 12px; line-height: 1.45;
  }}
  #sidebar h2 {{ font-size: 14px; margin-bottom: 6px; font-weight: 600; }}
  #sidebar .meta {{ color: #aaa; font-size: 11px; margin-bottom: 10px; }}
  .pair-row {{
    display: flex; gap: 8px; align-items: flex-start; padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08); cursor: pointer;
  }}
  .pair-row:hover {{ background: rgba(255,255,255,0.05); }}
  .pair-row.dimmed {{ opacity: 0.35; }}
  .swatch {{ width: 12px; height: 12px; border-radius: 2px; flex-shrink: 0; margin-top: 3px; }}
  .pair-body {{ flex: 1; }}
  .pair-title {{ font-weight: 600; }}
  .pair-stats {{ color: #b0b0b0; font-size: 11px; }}
  .legend {{ margin-top: 10px; color: #aaa; font-size: 11px; }}
  .site-legend-row {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; }}
  .btn-row {{ display: flex; gap: 6px; margin: 8px 0 4px; }}
  button {{
    flex: 1; padding: 5px 8px; border: none; border-radius: 6px;
    background: #2d2d44; color: #ddd; cursor: pointer; font-size: 11px;
  }}
  button:hover {{ background: #3d3d5c; }}
  #controls {{
    position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
    color: #bbb; font-size: 12px; background: rgba(0,0,0,0.45);
    padding: 6px 16px; border-radius: 6px; pointer-events: none;
    backdrop-filter: blur(6px);
  }}
</style>
<script src="https://unpkg.com/ngl@2.3.1/dist/ngl.js"></script>
</head>
<body>
<div id="viewport"></div>
<div id="sidebar">
  <h2>{title}</h2>
  <div class="meta">{meta_text}</div>
  <div class="btn-row">
    <button id="show-all">Show all pairs</button>
    <button id="hide-all">Hide pairs</button>
  </div>
  <div id="pair-list"></div>
  <div class="legend">
    <div style="margin-bottom:6px;"><b>Site colors</b></div>
    {site_legend_html}
    <div style="margin-top:8px;">
      Driver cylinders: red = Δ&gt;0, blue = Δ&lt;0.<br/>
      Paper d1 cylinders: green = open (4.5–5.5 Å), gray = closed (&lt;4.5), magenta = elongated (&gt;5.5).<br/>
      Spheres use site colors when available; click a pair to solo it.
    </div>
  </div>
</div>
<div id="controls">Scroll to zoom &middot; Click-drag to rotate &middot; Right-drag to pan</div>
<script>
document.addEventListener("DOMContentLoaded", function() {{
  var pairs = {pairs_json};
  var siteGroups = {site_groups_json};
  var d1Pairs = {d1_pairs_json};
  var stage = new NGL.Stage("viewport", {{
    backgroundColor: "#1a1a2e",
    ambientIntensity: 0.35,
    quality: "high"
  }});
  window.addEventListener("resize", function() {{ stage.handleResize(); }});

  var pdbBlob = new Blob([`{pdb_string}`], {{ type: "text/plain" }});
  var shapeComp = null;
  var siteReprs = [];
  var visible = pairs.map(function() {{ return true; }});

  function hexToRgb(hex) {{
    var h = String(hex).replace("#", "");
    if (h.length === 3) h = h[0]+h[0]+h[1]+h[1]+h[2]+h[2];
    return [
      parseInt(h.slice(0, 2), 16) / 255,
      parseInt(h.slice(2, 4), 16) / 255,
      parseInt(h.slice(4, 6), 16) / 255
    ];
  }}

  function d1StateColor(state) {{
    if (state === "open") return [0.15, 0.75, 0.35];
    if (state === "closed") return [0.55, 0.55, 0.55];
    if (state === "elongated") return [0.75, 0.2, 0.75];
    return [0.8, 0.8, 0.2];
  }}

  function rebuildShapes() {{
    if (shapeComp) {{
      stage.removeComponent(shapeComp);
      shapeComp = null;
    }}
    var shape = new NGL.Shape("transition-pairs");
    pairs.forEach(function(p, i) {{
      if (!visible[i]) return;
      var openColor = p.delta >= 0 ? [0.85, 0.2, 0.2] : [0.2, 0.45, 0.9];
      shape.addSphere(p.centroid_i, hexToRgb(p.color_i || p.color), 0.55);
      shape.addSphere(p.centroid_j, hexToRgb(p.color_j || p.color), 0.55);
      shape.addCylinder(p.centroid_i, p.centroid_j, openColor, 0.12);
    }});
    // Paper d1 ring-neighbor contacts (always shown).
    (d1Pairs || []).forEach(function(p) {{
      var col = d1StateColor(p.state);
      shape.addSphere(p.centroid_i, col, 0.35);
      shape.addSphere(p.centroid_j, col, 0.35);
      shape.addCylinder(p.centroid_i, p.centroid_j, col, 0.08);
      if (p.labelVisible) {{
        shape.addText(p.midpoint, col, 1.2, p.label);
      }}
    }});
    shapeComp = stage.addComponentFromObject(shape);
    shapeComp.addRepresentation("buffer");
  }}

  function rebuildSiteHighlight(comp) {{
    siteReprs.forEach(function(r) {{
      try {{ comp.removeRepresentation(r); }} catch (e) {{}}
    }});
    siteReprs = [];
    // Always show configured site groups (s3:A / s6:A / s7:A, …).
    siteGroups.forEach(function(g) {{
      if (!g.atoms || !g.atoms.length) return;
      siteReprs.push(comp.addRepresentation("ball+stick", {{
        sele: "@" + g.atoms.join(","),
        color: g.color,
        radiusScale: 2.6
      }}));
    }});
    // Also emphasize atoms from currently visible ranked pairs in soft yellow
    // when they are not already covered by a named site color.
    var colored = {{}};
    siteGroups.forEach(function(g) {{
      (g.atoms || []).forEach(function(a) {{ colored[a] = true; }});
    }});
    var extra = {{}};
    pairs.forEach(function(p, i) {{
      if (!visible[i]) return;
      p.atoms_i.concat(p.atoms_j).forEach(function(a) {{
        if (!colored[a]) extra[a] = true;
      }});
    }});
    var extraKeys = Object.keys(extra);
    if (extraKeys.length) {{
      siteReprs.push(comp.addRepresentation("ball+stick", {{
        sele: "@" + extraKeys.join(","),
        color: "#f5c542",
        radiusScale: 2.0
      }}));
    }}
  }}

  function renderList() {{
    var list = document.getElementById("pair-list");
    list.innerHTML = "";
    pairs.forEach(function(p, i) {{
      var row = document.createElement("div");
      row.className = "pair-row" + (visible[i] ? "" : " dimmed");
      var sw = document.createElement("div");
      sw.className = "swatch";
      sw.style.background = p.color;
      var body = document.createElement("div");
      body.className = "pair-body";
      var title = document.createElement("div");
      title.className = "pair-title";
      title.textContent = "#" + p.rank + " " + p.label;
      var stats = document.createElement("div");
      stats.className = "pair-stats";
      stats.textContent = "Δ=" + (p.delta >= 0 ? "+" : "") + p.delta.toFixed(2) +
        " Å · r=" + (p.corr >= 0 ? "+" : "") + p.corr.toFixed(2);
      body.appendChild(title);
      body.appendChild(stats);
      row.appendChild(sw);
      row.appendChild(body);
      row.addEventListener("click", function() {{
        visible = pairs.map(function(_, j) {{ return j === i; }});
        rebuildShapes();
        rebuildSiteHighlight(window._comp);
        renderList();
      }});
      list.appendChild(row);
    }});
  }}

  document.getElementById("show-all").addEventListener("click", function() {{
    visible = pairs.map(function() {{ return true; }});
    rebuildShapes();
    rebuildSiteHighlight(window._comp);
    renderList();
  }});
  document.getElementById("hide-all").addEventListener("click", function() {{
    visible = pairs.map(function() {{ return false; }});
    rebuildShapes();
    rebuildSiteHighlight(window._comp);
    renderList();
  }});

  stage.loadFile(pdbBlob, {{ ext: "pdb", defaultRepresentation: false }}).then(function(comp) {{
    window._comp = comp;
    comp.addRepresentation("licorice", {{ color: "element", radiusScale: 0.9, opacity: 0.55 }});
    rebuildSiteHighlight(comp);
    rebuildShapes();
    renderList();
    comp.autoView();
  }});
}});
</script>
</body>
</html>
"""


def _atom_ids_to_positions(
    universe: Any,
    atom_ids: Sequence[int],
    id_to_pos: dict[int, int],
) -> tuple[list[int], Optional[list[float]]]:
    """Map MDA atom ids → NGL @ indices and mean Cartesian position."""
    if not atom_ids:
        return [], None
    ids = [int(a) for a in atom_ids]
    try:
        atoms = universe.select_atoms("id " + " ".join(str(i) for i in ids))
    except Exception:
        atoms = universe.atoms[[]]
    if len(atoms) == 0:
        try:
            atoms = universe.atoms[ids]
        except Exception:
            return [], None
    ngl_idxs = sorted({id_to_pos[a.index] for a in atoms if a.index in id_to_pos})
    if not ngl_idxs:
        return [], None
    centroid = atoms.positions.mean(axis=0)
    return ngl_idxs, [float(centroid[0]), float(centroid[1]), float(centroid[2])]


def write_transition_sites_ngl_html(
    universe: Any,
    top_df: pd.DataFrame,
    from_cluster: int,
    to_cluster: int,
    output_path: Path | str,
    *,
    frame: Optional[int] = None,
    selection: str = "resname MOL",
    title: Optional[str] = None,
    top_n_pairs: int = 10,
    sites_df: Optional[pd.DataFrame] = None,
    site_colors: Optional[dict[str, str]] = None,
    d1_atoms_df: Optional[pd.DataFrame] = None,
    open_lo: float = 4.5,
    open_hi: float = 5.5,
    d1_max_draw: int = 12,
) -> Optional[Path]:
    """Write an interactive NGL HTML view of top transition-driving site pairs.

    Structure is shown as licorice; ranked pairs appear as centroid spheres +
    cylinders (red = distance increase / opening, blue = closing). Named atom
    sites (default ``s3:A`` blue, ``s6:A`` orange, ``s7:A`` red) are highlighted
    across all monomers when *sites_df* is provided. Corrected paper-d1
    ring-neighbor contacts are drawn with open/closed/elongated colors.
    """
    import json
    import tempfile

    colors = dict(site_colors) if site_colors is not None else dict(DEFAULT_SITE_HIGHLIGHT_COLORS)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub = top_df[
        (top_df["from_cluster"] == from_cluster)
        & (top_df["to_cluster"] == to_cluster)
    ].copy()
    if sub.empty:
        return None
    sub = sub.sort_values("rank").head(int(top_n_pairs))

    n_frames = len(universe.trajectory)
    if frame is None:
        frame = 0
    frame = int(max(0, min(int(frame), n_frames - 1)))
    universe.trajectory[frame]

    try:
        atoms = universe.select_atoms(selection)
    except Exception:
        atoms = universe.atoms
    if len(atoms) == 0:
        atoms = universe.atoms

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        atoms.write(str(tmp_path))
        pdb_string = tmp_path.read_text(encoding="utf-8", errors="replace")
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    pdb_string = (
        pdb_string.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    )
    id_to_pos = {atom.index: pos for pos, atom in enumerate(atoms)}

    # Build global site highlight groups from the site map (all monomers).
    site_groups: list[dict[str, Any]] = []
    if sites_df is not None and not sites_df.empty:
        grouped: dict[str, dict[str, Any]] = {}
        for _, srow in sites_df.iterrows():
            label = _site_label(
                srow.get("site_index"), srow.get("kind"), role=srow.get("role", "")
            )
            if label not in colors:
                continue
            ngl_idxs, _ = _atom_ids_to_positions(
                universe, _parse_atom_id_field(srow.get("atom_ids")), id_to_pos
            )
            if not ngl_idxs:
                continue
            bucket = grouped.setdefault(
                label, {"label": label, "color": colors[label], "atoms": set()}
            )
            bucket["atoms"].update(ngl_idxs)
        for label in sorted(grouped.keys()):
            g = grouped[label]
            site_groups.append(
                {
                    "label": label,
                    "color": g["color"],
                    "atoms": sorted(g["atoms"]),
                }
            )

    # Paper-d1 ring atoms as an extra highlight group.
    d1_pairs_payload: list[dict[str, Any]] = []
    if d1_atoms_df is not None and not d1_atoms_df.empty:
        ring_by_mon_site: dict[tuple[int, int], int] = {}
        d1_ring_ngl: set[int] = set()
        for _, drow in d1_atoms_df.iterrows():
            mon = int(drow["monomer"])
            site = int(drow["endpoint_site"])
            ring_id = int(drow["d1_ring_atom_id"])
            ring_by_mon_site[(mon, site)] = ring_id
            ngl_idxs, _ = _atom_ids_to_positions(universe, [ring_id], id_to_pos)
            d1_ring_ngl.update(ngl_idxs)
        if d1_ring_ngl:
            site_groups.append(
                {
                    "label": "d1_ring",
                    "color": "#2ecc71",
                    "atoms": sorted(d1_ring_ngl),
                }
            )

        # Draw shortest d1 contacts (and any currently open ones).
        candidates: list[tuple[float, str, list[float], list[float], str]] = []
        monomers = sorted({int(m) for m, _ in ring_by_mon_site})
        for i in monomers:
            for j in monomers:
                if i == j:
                    continue
                for si, sj in ((3, 7), (7, 3)):
                    if (i, si) not in ring_by_mon_site or (j, sj) not in ring_by_mon_site:
                        continue
                    _, cen_i = _atom_ids_to_positions(
                        universe, [ring_by_mon_site[(i, si)]], id_to_pos
                    )
                    _, cen_j = _atom_ids_to_positions(
                        universe, [ring_by_mon_site[(j, sj)]], id_to_pos
                    )
                    if cen_i is None or cen_j is None:
                        continue
                    dist = float(np.linalg.norm(np.asarray(cen_i) - np.asarray(cen_j)))
                    state = classify_paper_d1_state(
                        dist, open_lo=open_lo, open_hi=open_hi
                    )
                    label = f"d1 m{i}s{si}-m{j}s{sj} {dist:.2f}A ({state})"
                    candidates.append((dist, state, cen_i, cen_j, label))
        candidates.sort(key=lambda x: (0 if x[1] == "open" else 1, x[0]))
        for dist, state, cen_i, cen_j, label in candidates[: int(d1_max_draw)]:
            mid = [
                0.5 * (cen_i[0] + cen_j[0]),
                0.5 * (cen_i[1] + cen_j[1]),
                0.5 * (cen_i[2] + cen_j[2]),
            ]
            d1_pairs_payload.append(
                {
                    "distance": dist,
                    "state": state,
                    "centroid_i": cen_i,
                    "centroid_j": cen_j,
                    "midpoint": mid,
                    "label": f"{dist:.2f}",
                    "labelVisible": state == "open" or dist <= open_hi + 1.0,
                }
            )

    pairs_payload: list[dict[str, Any]] = []
    for _, row in sub.iterrows():
        rank = int(row["rank"])
        ids_i = _parse_atom_id_field(row.get("atom_ids_i"))
        ids_j = _parse_atom_id_field(row.get("atom_ids_j"))
        ngl_i, cen_i = _atom_ids_to_positions(universe, ids_i, id_to_pos)
        ngl_j, cen_j = _atom_ids_to_positions(universe, ids_j, id_to_pos)
        if cen_i is None or cen_j is None:
            continue
        label = str(row.get("pair_label") or row.get("feature") or f"rank{rank}")
        label = label.replace("↔", "<->").replace("→", "->")
        delta = float(row.get("mean_delta_angstrom", float("nan")))
        corr = float(row.get("mean_point_biserial_corr", float("nan")))
        color_i = _resolve_site_color(row.get("site_i"), row.get("kind_i"), colors)
        color_j = _resolve_site_color(row.get("site_j"), row.get("kind_j"), colors)
        if str(row.get("kind_i", "")) == "d1_ring":
            color_i = "#2ecc71"
        if str(row.get("kind_j", "")) == "d1_ring":
            color_j = "#2ecc71"
        pairs_payload.append(
            {
                "rank": rank,
                "label": label,
                "delta": delta if np.isfinite(delta) else 0.0,
                "corr": corr if np.isfinite(corr) else 0.0,
                "color": _RANK_COLORS[(rank - 1) % len(_RANK_COLORS)],
                "color_i": color_i,
                "color_j": color_j,
                "atoms_i": ngl_i,
                "atoms_j": ngl_j,
                "centroid_i": cen_i,
                "centroid_j": cen_j,
            }
        )

    if not pairs_payload:
        return None

    if title is None:
        title = f"Endpoint drivers {from_cluster}->{to_cluster}"
    n_events = int(sub["n_events"].iloc[0]) if "n_events" in sub.columns else 0
    descriptive = "descriptive (n_events&lt;2)" if n_events < 2 else f"n_events={n_events}"
    n_open = sum(1 for p in d1_pairs_payload if p["state"] == "open")
    meta = (
        f"Frame {frame} &bull; {len(atoms)} atoms &bull; "
        f"{len(pairs_payload)} pairs &bull; {descriptive}"
        f" &bull; d1 open shown={n_open}"
    )
    if site_groups:
        legend_rows = []
        for g in site_groups:
            legend_rows.append(
                '<div class="site-legend-row">'
                f'<div class="swatch" style="background:{g["color"]};"></div>'
                f'<span>{g["label"]} ({len(g["atoms"])} atoms)</span>'
                "</div>"
            )
        site_legend_html = "\n    ".join(legend_rows)
    else:
        site_legend_html = (
            '<div class="site-legend-row">'
            '<div class="swatch" style="background:#4363d8;"></div><span>s3:A blue</span></div>\n'
            '    <div class="site-legend-row">'
            '<div class="swatch" style="background:#f58231;"></div><span>s6:A orange</span></div>\n'
            '    <div class="site-legend-row">'
            '<div class="swatch" style="background:#e6194b;"></div><span>s7:A red</span></div>\n'
            '    <div class="site-legend-row">'
            '<div class="swatch" style="background:#2ecc71;"></div><span>d1 ring neighbor</span></div>'
        )

    html = _NGL_TRANSITION_HTML.format(
        title=title,
        meta_text=meta,
        pdb_string=pdb_string,
        pairs_json=json.dumps(pairs_payload),
        site_groups_json=json.dumps(site_groups),
        d1_pairs_json=json.dumps(d1_pairs_payload),
        site_legend_html=site_legend_html,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def plot_transition_site_highlights(
    sites_df: pd.DataFrame,
    top_df: pd.DataFrame,
    from_cluster: int,
    to_cluster: int,
    output_path: Path | str,
    *,
    universe: Any = None,
    monomer_selections: Optional[Sequence[str]] = None,
    stored_sites: Optional[Sequence[Sequence[Sequence[int]]]] = None,
    frame: Optional[int] = None,
    selection: str = "resname MOL",
    title: Optional[str] = None,
    d1_atoms_df: Optional[pd.DataFrame] = None,
    open_lo: float = 4.5,
    open_hi: float = 5.5,
) -> Optional[Path]:
    """Visualize sites involved in a directed transition's top pairs.

    Prefer an interactive 3D NGL HTML view when *universe* is available.
    Always also write a ``*_pair_ranks.png`` bar chart beside the main artifact.
    Falls back to the bar chart alone when no structure is available.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub = top_df[
        (top_df["from_cluster"] == from_cluster)
        & (top_df["to_cluster"] == to_cluster)
    ].copy()
    if sub.empty:
        return None

    if title is None:
        title = f"Top endpoint pairs for cluster {from_cluster} -> {to_cluster}"

    if universe is not None:
        html_path = output_path.with_suffix(".html")
        html = write_transition_sites_ngl_html(
            universe,
            top_df,
            from_cluster,
            to_cluster,
            html_path,
            frame=frame,
            selection=selection,
            title=title,
            sites_df=sites_df,
            d1_atoms_df=d1_atoms_df,
            open_lo=open_lo,
            open_hi=open_hi,
        )
        legend_path = output_path.with_name(output_path.stem + "_pair_ranks.png")
        _plot_pair_rank_bars(
            sub, legend_path, title=f"Ranked pairs {from_cluster}->{to_cluster}"
        )
        if html is not None:
            return html
        # HTML failed; try 2D RDKit, else bars as primary PNG.
        if (
            monomer_selections is not None
            and stored_sites is not None
            and len(stored_sites) == len(monomer_selections)
        ):
            from .endpoint_features import plot_endpoint_sites

            return plot_endpoint_sites(
                universe,
                monomer_selections,
                stored_sites,
                output_path.with_suffix(".png"),
                title=title + "  (2D fallback; blue=ring, orange=atom)",
            )
        return legend_path

    return _plot_pair_rank_bars(sub, output_path.with_suffix(".png"), title=title)



def _plot_pair_rank_bars(
    sub: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    label_col = "pair_label" if "pair_label" in sub.columns else "feature"
    plot_df = sub.sort_values("rank").copy()
    labels = plot_df[label_col].astype(str).tolist()
    effects = plot_df["mean_standardized_delta"].to_numpy(dtype=float)
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in effects]

    fig_h = max(3.5, 0.35 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, effects, color=colors, edgecolor="0.3")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [
            f"#{int(r)} {lab}  (Δ={d:+.2f} Å, r={c:+.2f})"
            for r, lab, d, c in zip(
                plot_df["rank"],
                labels,
                plot_df["mean_delta_angstrom"],
                plot_df["mean_point_biserial_corr"],
            )
        ],
        fontsize=8,
    )
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("Mean standardized Δ (dest − source)")
    n_events = int(plot_df["n_events"].iloc[0]) if "n_events" in plot_df.columns else 0
    note = " [descriptive: n_events<2]" if n_events < 2 else ""
    ax.set_title(title + note)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ── Orchestration ───────────────────────────────────────────────────────────


def attribute_endpoint_transitions(
    changepoints_dir: Path | str,
    features_dir: Path | str,
    *,
    config: Optional[EndpointTransitionAttributionConfig] = None,
    clustered_df: Optional[pd.DataFrame] = None,
    sites_df: Optional[pd.DataFrame] = None,
    d1_atoms_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path | str] = None,
    plot_dir: Optional[Path | str] = None,
    universe: Any = None,
    monomer_selections: Optional[Sequence[str]] = None,
    stored_sites: Optional[Sequence[Sequence[Sequence[int]]]] = None,
) -> dict[str, Path]:
    """Run full transition attribution and write CSV + plot artifacts.

    Expected layout:
    - ``{changepoints_dir}/clusters/endpoint/segments_clustered.csv``
    - ``{features_dir}/*_endpoint_features.csv`` with site-pair / paper_d1 columns
    - ``{features_dir}/endpoint_sites.csv``
    - ``{features_dir}/paper_d1_atoms.csv`` (optional)
    """
    cfg = config or EndpointTransitionAttributionConfig()
    changepoints_dir = Path(changepoints_dir)
    features_dir = Path(features_dir)
    out_dir = Path(output_dir) if output_dir else changepoints_dir
    plot_dir = Path(plot_dir) if plot_dir else out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if clustered_df is None:
        seg_path = (
            changepoints_dir / "clusters" / cfg.group / "segments_clustered.csv"
        )
        if not seg_path.exists():
            # Fall back to cohort merge
            alt = changepoints_dir / "clusters" / "all_segments_clustered.csv"
            if not alt.exists():
                log_event(
                    "warning",
                    f"No clustered segments found under {changepoints_dir}/clusters",
                    component="endpoint_transition_attribution",
                )
                return written
            clustered_df = pd.read_csv(alt)
        else:
            clustered_df = pd.read_csv(seg_path)

    events = build_transition_events(clustered_df, group=cfg.group)
    events_path = out_dir / "endpoint_transition_events.csv"
    events.to_csv(events_path, index=False)
    written["endpoint_transition_events.csv"] = events_path

    if events.empty:
        log_event(
            "info",
            "No directed cluster transitions to attribute",
            component="endpoint_transition_attribution",
        )
        return written

    event_feat, ranking = score_transition_features(
        events,
        features_dir,
        window_frames=cfg.window_frames,
        effect_weight=cfg.effect_weight,
        correlation_weight=cfg.correlation_weight,
        include_paper_d1=cfg.include_paper_d1,
    )

    if sites_df is None:
        sites_path = features_dir / "endpoint_sites.csv"
        if sites_path.exists():
            sites_df = load_endpoint_sites_map(sites_path)
        else:
            sites_df = pd.DataFrame(
                columns=["monomer", "site_index", "kind", "atom_ids"]
            )

    if d1_atoms_df is None:
        d1_path = features_dir / "paper_d1_atoms.csv"
        if d1_path.exists():
            d1_atoms_df = pd.read_csv(d1_path)

    ranking_columns = [
        "from_cluster",
        "to_cluster",
        "feature",
        "feature_block",
        "n_events",
        "n_trajectories",
        "mean_source",
        "mean_dest",
        "mean_delta_angstrom",
        "mean_standardized_delta",
        "mean_abs_standardized_delta",
        "mean_point_biserial_corr",
        "mean_abs_point_biserial_corr",
        "consistent_direction_fraction",
        "is_descriptive",
        "combined_score",
        "rank",
        "monomer_i",
        "site_i",
        "kind_i",
        "atom_ids_i",
        "monomer_j",
        "site_j",
        "kind_j",
        "atom_ids_j",
        "pair_label",
        "d1_state_hint",
        "d1_dest_state",
    ]
    if ranking.empty:
        ranking = pd.DataFrame(columns=ranking_columns)
    else:
        ranking = attach_site_metadata(
            ranking,
            sites_df,
            d1_atoms_df=d1_atoms_df,
            open_lo=cfg.paper_d1_open_lo,
            open_hi=cfg.paper_d1_open_hi,
        )

    rankings_path = out_dir / "endpoint_transition_feature_rankings.csv"
    ranking.to_csv(rankings_path, index=False)
    written["endpoint_transition_feature_rankings.csv"] = rankings_path

    top = top_features_per_transition(
        ranking,
        top_n=cfg.top_n,
        min_abs_correlation=cfg.min_abs_correlation,
    )
    if top.empty:
        top = pd.DataFrame(columns=ranking_columns)
    top_path = out_dir / "endpoint_transition_top_features.csv"
    top.to_csv(top_path, index=False)
    written["endpoint_transition_top_features.csv"] = top_path

    # Paper-d1 only ranking extract for quick reading.
    if not ranking.empty and "feature_block" in ranking.columns:
        d1_rank = ranking[ranking["feature_block"] == "paper_d1"].copy()
        if not d1_rank.empty:
            d1_top = top_features_per_transition(
                d1_rank,
                top_n=cfg.top_n,
                min_abs_correlation=cfg.min_abs_correlation,
            )
            d1_path_out = out_dir / "paper_d1_transition_top_features.csv"
            d1_top.to_csv(d1_path_out, index=False)
            written["paper_d1_transition_top_features.csv"] = d1_path_out

    # Per-segment d1 state summary.
    try:
        # Use first available features CSV as frame table; if multiple trajs,
        # concatenate.
        feat_frames: list[pd.DataFrame] = []
        for tid in clustered_df["traj_id"].astype(str).unique():
            path = resolve_features_csv(
                features_dir, tid, suffix="_endpoint_features.csv"
            )
            if path is not None:
                feat_frames.append(pd.read_csv(path))
        if feat_frames:
            all_feat = pd.concat(feat_frames, ignore_index=True)
            seg_d1 = summarize_paper_d1_by_segments(
                all_feat,
                clustered_df,
                open_lo=cfg.paper_d1_open_lo,
                open_hi=cfg.paper_d1_open_hi,
                group=cfg.group,
            )
            if not seg_d1.empty:
                seg_path = out_dir / "paper_d1_segment_states.csv"
                seg_d1.to_csv(seg_path, index=False)
                written["paper_d1_segment_states.csv"] = seg_path
    except Exception as exc:
        log_event(
            "warning",
            f"Failed to write paper_d1_segment_states.csv: {exc}",
            component="endpoint_transition_attribution",
        )

    # Also write per-event feature table for reproducibility
    if not event_feat.empty:
        evt_feat_path = out_dir / "endpoint_transition_event_features.csv"
        event_feat.to_csv(evt_feat_path, index=False)
        written["endpoint_transition_event_features.csv"] = evt_feat_path

    if top.empty or "from_cluster" not in top.columns:
        log_event(
            "info",
            "No ranked site-pair features to visualize",
            component="endpoint_transition_attribution",
        )
        return written

    heat = plot_transition_feature_heatmap(
        top,
        plot_dir / "endpoint_transition_feature_heatmap.png",
    )
    if heat is not None:
        written["endpoint_transition_feature_heatmap.png"] = heat

    net = plot_attributed_transition_network(
        top,
        plot_dir / "endpoint_transition_network_attributed.png",
        label_top_n=cfg.network_label_top_n,
    )
    if net is not None:
        written["endpoint_transition_network_attributed.png"] = net

    for (fc, tc), _ in top.groupby(["from_cluster", "to_cluster"]):
        # Prefer the first observed boundary frame for this directed transition.
        frame = None
        if not events.empty:
            match = events[
                (events["from_cluster"] == int(fc))
                & (events["to_cluster"] == int(tc))
            ]
            if not match.empty:
                frame = int(match.iloc[0]["boundary_frame"])
        stem = plot_dir / f"endpoint_sites_transition_{int(fc)}_to_{int(tc)}"
        out = plot_transition_site_highlights(
            sites_df,
            top,
            int(fc),
            int(tc),
            stem.with_suffix(".png"),
            universe=universe,
            monomer_selections=monomer_selections,
            stored_sites=stored_sites,
            frame=frame,
            d1_atoms_df=d1_atoms_df,
            open_lo=cfg.paper_d1_open_lo,
            open_hi=cfg.paper_d1_open_hi,
        )
        if out is not None:
            written[out.name] = out
            pair_png = stem.with_name(stem.name + "_pair_ranks.png")
            if pair_png.exists():
                written[pair_png.name] = pair_png

    log_event(
        "info",
        (
            f"Attributed {len(events)} transition event(s); "
            f"{top['from_cluster'].nunique()} directed types; "
            f"wrote {len(written)} artifact(s)"
        ),
        component="endpoint_transition_attribution",
    )
    return written
