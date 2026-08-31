"""Summarize changepoint outputs and generate cohort plots."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.utils.run_log import log_event

from .feature_groups import (
    DEFAULT_GROUPS,
    GROUP_COLORS,
    GROUP_MARKERS,
    REGIME_COLS,
    resolve_features_csv,
)

# See module docstring in summarize script for why assembly_rg is the structural reference.
DEFAULT_TIMELINE_PANELS: list[tuple[str, str]] = [
    ("assembly_rg", "Assembly Rg (Å) — global cage size"),
    ("n_guest_inside_cavity", "Guest inside cavity"),
    ("cavity_ion_count", "Cavity Na⁺ count"),
]

DEFAULT_ENDPOINT_TIMELINE_PANELS: list[tuple[str, str]] = [
    ("endpoint_dist_mean", "Endpoint-site dist mean (Å)"),
    ("endpoint_dist_min", "Endpoint-site dist min (Å)"),
    ("endpoint_dist_max", "Endpoint-site dist max (Å)"),
]

_ENDPOINT_PAIR_MEAN_RE = re.compile(r"^endpoint_dist_(\d+)_(\d+)_mean$")


def resolve_timeline_panels(
    df: pd.DataFrame,
    panels: list[tuple[str, str]] | None = None,
    fallback: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    """Keep requested panels that exist in *df*; fall back to endpoint aggregates."""
    requested = list(panels) if panels is not None else list(DEFAULT_TIMELINE_PANELS)
    present = [(c, lab) for c, lab in requested if c in df.columns]
    if present:
        return present
    fb = fallback if fallback is not None else DEFAULT_ENDPOINT_TIMELINE_PANELS
    return [(c, lab) for c, lab in fb if c in df.columns]


def list_endpoint_pair_mean_columns(
    columns: Sequence[str],
) -> list[tuple[str, int, int]]:
    """Return ``(col, monomer_i, monomer_j)`` for ``endpoint_dist_{i}_{j}_mean``."""
    out: list[tuple[str, int, int]] = []
    for col in columns:
        m = _ENDPOINT_PAIR_MEAN_RE.match(str(col))
        if m:
            out.append((str(col), int(m.group(1)), int(m.group(2))))
    return out


def list_endpoint_site_pair_columns(
    columns: Sequence[str],
) -> list[tuple[str, int, int, int, int]]:
    """Return ``(col, mon_i, site_a, mon_j, site_b)`` for site-pair distance columns."""
    from .transition_attribution import parse_site_pair_feature

    out: list[tuple[str, int, int, int, int]] = []
    for col in columns:
        parsed = parse_site_pair_feature(str(col))
        if parsed is not None:
            mon_i, site_a, mon_j, site_b = parsed
            out.append((str(col), mon_i, site_a, mon_j, site_b))
    return out


def format_endpoint_site_pair_label(
    mon_i: int,
    site_a: int,
    mon_j: int,
    site_b: int,
) -> str:
    """Human label like ``M0S3-M1S6``."""
    return f"M{mon_i}S{site_a}-M{mon_j}S{site_b}"


def _discover_cluster_correlation_features(
    columns: Sequence[str],
) -> tuple[list[str], str, list[tuple]]:
    """Prefer raw site-pair columns; fall back to monomer-pair means."""
    site_specs = list_endpoint_site_pair_columns(columns)
    if site_specs:
        return [c for c, *_ in site_specs], "site_pair", site_specs
    mean_specs = list_endpoint_pair_mean_columns(columns)
    if mean_specs:
        return [c for c, *_ in mean_specs], "pair_mean", mean_specs
    return [], "none", []


def _feature_metadata_row(spec: tuple, feature_kind: str) -> dict:
    """Identity fields for a ranked endpoint feature."""
    if feature_kind == "site_pair":
        col, mon_i, site_a, mon_j, site_b = spec
        return {
            "feature": str(col),
            "feature_kind": feature_kind,
            "monomer_i": mon_i,
            "site_i": site_a,
            "monomer_j": mon_j,
            "site_j": site_b,
            "endpoint_label": format_endpoint_site_pair_label(
                mon_i, site_a, mon_j, site_b
            ),
        }
    col, mon_i, mon_j = spec
    return {
        "feature": str(col),
        "feature_kind": feature_kind,
        "monomer_i": mon_i,
        "monomer_j": mon_j,
        "endpoint_label": f"M{mon_i}-M{mon_j} pair mean",
    }


def _eta_squared(x: np.ndarray, y: np.ndarray, cluster_ids: Sequence[int]) -> float:
    """Fraction of variance in *x* explained by cluster labels *y*."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    finite = np.isfinite(x)
    if finite.sum() < 2:
        return float("nan")
    xv = x[finite]
    yv = y[finite]
    gm = float(np.mean(xv))
    sst = float(np.sum((xv - gm) ** 2))
    if sst <= 0:
        return float("nan")
    ssb = 0.0
    for c in cluster_ids:
        mask = yv == c
        if not np.any(mask):
            continue
        ssb += float(mask.sum()) * (float(np.mean(xv[mask])) - gm) ** 2
    return float(ssb / sst)


def _epsilon_squared_kw(
    x: np.ndarray,
    y: np.ndarray,
    cluster_ids: Sequence[int],
) -> float:
    """Kruskal–Wallis epsilon-squared effect size."""
    from scipy import stats

    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    groups = []
    for c in cluster_ids:
        vals = x[y == c]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            groups.append(vals)
    if len(groups) < 2:
        return float("nan")
    n = int(sum(g.size for g in groups))
    k = len(groups)
    if n <= k:
        return float("nan")
    try:
        h = float(stats.kruskal(*groups).statistic)
    except ValueError:
        return float("nan")
    return float((h - k + 1) / (n - k))


def _cohens_d_one_vs_rest(x: np.ndarray, mask: np.ndarray) -> float:
    """Cohen's d for in-cluster vs out-of-cluster means."""
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    a = x[mask]
    b = x[~mask]
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = np.sqrt(((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def _point_biserial_from_sums(
    n: float,
    sum_x: float,
    sum_x2: float,
    n_c: float,
    sum_x_c: float,
) -> float:
    """Point-biserial of continuous x vs binary cluster membership from sums."""
    n_rest = n - n_c
    if n < 2 or n_c < 1 or n_rest < 1:
        return float("nan")
    mean_all = sum_x / n
    var = sum_x2 / n - mean_all * mean_all
    if var <= 0:
        return float("nan")
    s_x = np.sqrt(var)
    mean_c = sum_x_c / n_c
    mean_rest = (sum_x - sum_x_c) / n_rest
    return float((mean_c - mean_rest) / s_x * np.sqrt(n_c * n_rest / (n * n)))


def load_clustered_segments(
    changepoints_dir: Path | str,
    *,
    group: str = "endpoint",
    clustered_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Load clustered segments for *group*.

    Preference order:
    1. In-memory ``clustered_df`` (filtered to *group* when a ``group`` column exists)
    2. ``clusters/{group}/segments_clustered.csv``
    3. ``clusters/all_segments_clustered.csv`` (filtered to *group*)
    """
    changepoints_dir = Path(changepoints_dir)
    if clustered_df is not None and len(clustered_df):
        work = clustered_df
        if "group" in work.columns:
            work = work[work["group"] == group]
        return work.reset_index(drop=True)

    group_path = changepoints_dir / "clusters" / group / "segments_clustered.csv"
    if group_path.exists():
        return pd.read_csv(group_path)

    alt = changepoints_dir / "clusters" / "all_segments_clustered.csv"
    if alt.exists():
        df = pd.read_csv(alt)
        if "group" in df.columns:
            df = df[df["group"] == group]
        return df.reset_index(drop=True)

    log_event(
        "warning",
        f"No clustered segments found under {changepoints_dir}/clusters for group={group}",
        component="summarize_changepoint_results",
    )
    return pd.DataFrame()


def rank_cluster_discriminating_endpoint_features(
    changepoints_dir: Path | str,
    features_dir: Path | str,
    *,
    group: str = "endpoint",
    clustered_df: Optional[pd.DataFrame] = None,
    features_suffix: Optional[str] = None,
) -> pd.DataFrame:
    """Rank endpoint features by segment-level cluster discrimination.

    Prefers raw site-pair columns ``endpoint_dist_{i}s{a}_{j}s{b}`` when present;
    otherwise falls back to ``endpoint_dist_{i}_{j}_mean``.

    Primary score is segment-level eta-squared on raw (Å) per-segment means.
    Also reports Kruskal–Wallis epsilon-squared, Cohen's d for the best
    one-vs-rest cluster, and frame-level max |point-biserial| (per-traj
    z-scored) for comparison with older frame-level rankings.
    """
    changepoints_dir = Path(changepoints_dir)
    features_dir = Path(features_dir)
    segments = load_clustered_segments(
        changepoints_dir, group=group, clustered_df=clustered_df
    )
    required = {"traj_id", "cluster_label", "start_frame", "end_frame"}
    if segments.empty or not required.issubset(segments.columns):
        return pd.DataFrame()

    cluster_counts = segments["cluster_label"].value_counts()
    if len(cluster_counts) < 2 or int(cluster_counts.min()) < 2:
        return pd.DataFrame()

    feature_cols: list[str] = []
    feature_kind = "none"
    feature_specs: list[tuple] = []
    for tid in segments["traj_id"].astype(str).unique():
        path = resolve_features_csv(features_dir, tid, suffix=features_suffix)
        if path is None:
            continue
        header = pd.read_csv(path, nrows=0)
        feature_cols, feature_kind, feature_specs = _discover_cluster_correlation_features(
            header.columns
        )
        if feature_cols:
            break
    if not feature_cols:
        return pd.DataFrame()

    n_feat = len(feature_cols)
    usecols = set(feature_cols) | {"frame", "time_ps"}
    seg_means: list[np.ndarray] = []
    seg_labels: list[int] = []

    # Per-feature streaming frame-level sums for max |point-biserial|.
    frame_n = np.zeros(n_feat, dtype=float)
    frame_sum = np.zeros(n_feat, dtype=float)
    frame_sum2 = np.zeros(n_feat, dtype=float)
    frame_n_c: dict[int, np.ndarray] = {}
    frame_sum_c: dict[int, np.ndarray] = {}

    for traj_id, traj_segs in segments.groupby(segments["traj_id"].astype(str)):
        feat_path = resolve_features_csv(
            features_dir, str(traj_id), suffix=features_suffix
        )
        if feat_path is None:
            continue
        feat = pd.read_csv(feat_path, usecols=lambda c: c in usecols)
        missing = [c for c in feature_cols if c not in feat.columns]
        if missing or "frame" not in feat.columns:
            continue

        frames = feat["frame"].to_numpy(dtype=int)
        raw = feat[feature_cols].to_numpy(dtype=float)

        labels = np.full(len(feat), -1, dtype=int)
        for _, seg in traj_segs.iterrows():
            start_f = int(seg["start_frame"])
            end_f = int(seg["end_frame"])
            cl = int(seg["cluster_label"])
            mask = (frames >= start_f) & (frames <= end_f)
            if not np.any(mask):
                continue
            labels[mask] = cl
            # Segment means in raw Å — absolute levels carry cluster signal
            # across trajectories; per-traj z-scoring collapses that.
            seg_means.append(np.nanmean(raw[mask], axis=0))
            seg_labels.append(cl)

        labeled = labels >= 0
        if not np.any(labeled):
            continue
        # Frame-level comparison uses per-traj z-scores so baselines do not
        # dominate the pooled point-biserial.
        mu = np.nanmean(raw, axis=0)
        sigma = np.nanstd(raw, axis=0)
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
        z = (raw - mu) / sigma
        z_lab = z[labeled]
        lab = labels[labeled]
        ok = np.isfinite(z_lab)
        z_filled = np.where(ok, z_lab, 0.0)
        frame_n += ok.sum(axis=0).astype(float)
        frame_sum += z_filled.sum(axis=0)
        frame_sum2 += (z_filled * z_filled).sum(axis=0)
        for c in np.unique(lab):
            c = int(c)
            cmask = lab == c
            cok = ok[cmask]
            cvals = np.where(cok, z_lab[cmask], 0.0)
            if c not in frame_n_c:
                frame_n_c[c] = np.zeros(n_feat, dtype=float)
                frame_sum_c[c] = np.zeros(n_feat, dtype=float)
            frame_n_c[c] += cok.sum(axis=0).astype(float)
            frame_sum_c[c] += cvals.sum(axis=0)

    if not seg_means:
        return pd.DataFrame()

    X = np.vstack(seg_means)
    y = np.asarray(seg_labels, dtype=int)
    cluster_ids = sorted(set(y.tolist()))

    rows: list[dict] = []
    for j, spec in enumerate(feature_specs):
        xj = X[:, j]
        eta = _eta_squared(xj, y, cluster_ids)
        eps = _epsilon_squared_kw(xj, y, cluster_ids)

        best_c = cluster_ids[0]
        best_d = float("nan")
        best_abs = -1.0
        for c in cluster_ids:
            d = _cohens_d_one_vs_rest(xj, y == c)
            if not np.isfinite(d):
                continue
            if abs(d) > best_abs:
                best_abs = abs(d)
                best_d = d
                best_c = c

        frame_abs: list[float] = []
        for c in cluster_ids:
            if c not in frame_n_c:
                continue
            r = _point_biserial_from_sums(
                float(frame_n[j]),
                float(frame_sum[j]),
                float(frame_sum2[j]),
                float(frame_n_c[c][j]),
                float(frame_sum_c[c][j]),
            )
            if np.isfinite(r):
                frame_abs.append(abs(r))
        frame_max = float(max(frame_abs)) if frame_abs else float("nan")

        if not np.isfinite(eta):
            continue
        row = _feature_metadata_row(spec, feature_kind)
        row.update(
            {
                "eta_squared": eta,
                "epsilon_squared_kw": eps,
                "cohens_d_best": best_d,
                "best_cluster": int(best_c),
                "frame_max_abs_corr": frame_max,
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(
        ["eta_squared", "epsilon_squared_kw"], ascending=False
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def rank_cluster_correlated_pair_means(
    changepoints_dir: Path | str,
    features_dir: Path | str,
    *,
    group: str = "endpoint",
    clustered_df: Optional[pd.DataFrame] = None,
    features_suffix: Optional[str] = None,
) -> pd.DataFrame:
    """Compatibility alias for :func:`rank_cluster_discriminating_endpoint_features`."""
    return rank_cluster_discriminating_endpoint_features(
        changepoints_dir,
        features_dir,
        group=group,
        clustered_df=clustered_df,
        features_suffix=features_suffix,
    )


def cluster_correlated_timeline_panels(
    ranking: pd.DataFrame,
    *,
    top_n: int = 5,
    include_global_mean: bool = True,
) -> list[tuple[str, str]]:
    """Build timeline panels from a cluster-discrimination ranking.

    Always puts ``endpoint_dist_mean`` first (when requested) so the assembly
    aggregate remains visible next to the top discriminating endpoint pairs.
    """
    if ranking is None or ranking.empty or top_n <= 0:
        return []

    panels: list[tuple[str, str]] = []
    if include_global_mean:
        panels.append(("endpoint_dist_mean", "Endpoint-site dist mean (Å)"))

    top = ranking.head(int(top_n))
    for _, row in top.iterrows():
        if "endpoint_label" in row and pd.notna(row["endpoint_label"]):
            name = str(row["endpoint_label"])
        elif {"monomer_i", "site_i", "monomer_j", "site_j"}.issubset(row.index):
            name = format_endpoint_site_pair_label(
                int(row["monomer_i"]),
                int(row["site_i"]),
                int(row["monomer_j"]),
                int(row["site_j"]),
            )
        else:
            name = f"M{int(row['monomer_i'])}-M{int(row['monomer_j'])} pair mean"
        if "eta_squared" in row and pd.notna(row["eta_squared"]):
            label = f"{name} (Å)  η²={float(row['eta_squared']):.2f}"
        else:
            max_abs = float(row.get("frame_max_abs_corr", row.get("max_abs_corr", np.nan)))
            label = f"{name} (Å)  |r|={max_abs:.2f}"
        panels.append((str(row["feature"]), label))
    return panels


def write_cohort_tables(changepoints_dir: Path) -> dict[str, Path]:
    """Write aggregated CSV summaries next to changepoint outputs."""
    changepoints_dir = Path(changepoints_dir)
    cmp_path = changepoints_dir / "changepoint_timing_comparison.csv"
    try:
        cmp = pd.read_csv(cmp_path)
    except pd.errors.EmptyDataError:
        cmp = pd.DataFrame(
            columns=[
                "traj_id",
                "group_a",
                "group_b",
                "tolerance_frames",
                "n_bkps_a",
                "n_bkps_b",
                "n_shared",
                "jaccard",
                "mean_timing_offset_ps",
                "mean_timing_offset_frames",
                "n_shared",
            ]
        )
    seg = pd.read_csv(changepoints_dir / "all_segment_stats.csv")
    bkp = pd.read_csv(changepoints_dir / "all_breakpoints.csv")

    written: dict[str, Path] = {}

    if cmp.empty or "group_a" not in cmp.columns:
        pairs = pd.DataFrame(
            columns=[
                "group_a",
                "group_b",
                "n_trajectories",
                "median_jaccard",
                "mean_jaccard",
                "median_offset_ps",
                "mean_offset_ps",
                "median_n_shared",
                "median_n_bkps_a",
                "median_n_bkps_b",
            ]
        )
    else:
        pairs = (
            cmp.groupby(["group_a", "group_b"], as_index=False)
            .agg(
                n_trajectories=("traj_id", "count"),
                median_jaccard=("jaccard", "median"),
                mean_jaccard=("jaccard", "mean"),
                median_offset_ps=("mean_timing_offset_ps", "median"),
                mean_offset_ps=("mean_timing_offset_ps", "mean"),
                median_n_shared=("n_shared", "median"),
                median_n_bkps_a=("n_bkps_a", "median"),
                median_n_bkps_b=("n_bkps_b", "median"),
            )
            .sort_values("median_jaccard", ascending=False)
        )
    path = changepoints_dir / "cohort_timing_summary.csv"
    pairs.to_csv(path, index=False)
    written["cohort_timing_summary"] = path

    if cmp.empty or "group_a" not in cmp.columns:
        iod_gsa = pd.DataFrame(columns=list(cmp.columns) if len(cmp.columns) else ["jaccard"])
    else:
        iod_gsa = cmp[
            ((cmp["group_a"] == "iodine") & (cmp["group_b"] == "gsa"))
            | ((cmp["group_a"] == "gsa") & (cmp["group_b"] == "iodine"))
        ].copy()
    low_path = changepoints_dir / "timing_low_agreement_iodine_vs_gsa.csv"
    high_path = changepoints_dir / "timing_high_agreement_iodine_vs_gsa.csv"
    if "jaccard" in iod_gsa.columns and not iod_gsa.empty:
        iod_gsa.sort_values("jaccard").head(10).to_csv(low_path, index=False)
        iod_gsa.sort_values("jaccard", ascending=False).head(10).to_csv(high_path, index=False)
    else:
        iod_gsa.to_csv(low_path, index=False)
        iod_gsa.to_csv(high_path, index=False)
    written["timing_low_agreement_iodine_vs_gsa"] = low_path
    written["timing_high_agreement_iodine_vs_gsa"] = high_path

    rows: list[dict] = []
    for grp, cols in REGIME_COLS.items():
        sub = seg[seg["group"] == grp]
        for sid in sorted(sub["segment_id"].unique()):
            ssub = sub[sub["segment_id"] == sid]
            row: dict = {
                "group": grp,
                "segment_id": sid,
                "n_trajectories": len(ssub),
                "median_n_frames": ssub["n_frames"].median(),
            }
            for col in cols:
                if col in ssub.columns:
                    row[f"median_{col}"] = ssub[col].median()
                    row[f"mean_{col}"] = ssub[col].mean()
            rows.append(row)
    regime_path = changepoints_dir / "cohort_segment_regime_summary.csv"
    pd.DataFrame(rows).to_csv(regime_path, index=False)
    written["cohort_segment_regime_summary"] = regime_path

    bkp_path = changepoints_dir / "breakpoints_per_trajectory.csv"
    bkp.groupby(["traj_id", "group"]).size().unstack(fill_value=0).to_csv(bkp_path)
    written["breakpoints_per_trajectory"] = bkp_path
    return written


def plot_jaccard_heatmap(changepoints_dir: Path, plot_dir: Path) -> Path:
    changepoints_dir = Path(changepoints_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(changepoints_dir / "cohort_timing_summary.csv")

    # Prefer groups that actually appear in pairwise timing comparisons.
    # Single-group runs (e.g. endpoint-only) have an empty table — fall back to
    # groups present in breakpoints, not DEFAULT_GROUPS (which are GSA-path only).
    if not pairs.empty and "group_a" in pairs.columns:
        groups_in_data = sorted(
            set(pairs["group_a"].dropna().tolist() + pairs["group_b"].dropna().tolist())
        )
    else:
        groups_in_data = []

    if not groups_in_data:
        bkp_path = changepoints_dir / "all_breakpoints.csv"
        if bkp_path.exists():
            bkp = pd.read_csv(bkp_path)
            if "group" in bkp.columns and not bkp.empty:
                groups_in_data = sorted(bkp["group"].dropna().unique().tolist())
        if not groups_in_data:
            groups_in_data = list(DEFAULT_GROUPS)

    groups = groups_in_data
    mat = pd.DataFrame(np.nan, index=groups, columns=groups)
    for _, row in pairs.iterrows():
        if pd.isna(row.get("group_a")) or pd.isna(row.get("group_b")):
            continue
        mat.loc[row["group_a"], row["group_b"]] = row["median_jaccard"]
        mat.loc[row["group_b"], row["group_a"]] = row["median_jaccard"]
    np.fill_diagonal(mat.values, 1.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat.values, vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(len(groups)), groups, rotation=45, ha="right")
    ax.set_yticks(range(len(groups)), groups)
    for i in range(len(groups)):
        for j in range(len(groups)):
            val = mat.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Median Jaccard")
    n_traj = (
        int(pairs["n_trajectories"].iloc[0])
        if (not pairs.empty and "n_trajectories" in pairs.columns)
        else 0
    )
    if pairs.empty or len(groups) < 2:
        ax.set_title(
            f"Cohort timing agreement — no cross-group pairs "
            f"({', '.join(groups)}; {n_traj} trajectories)"
        )
    else:
        ax.set_title(f"Cohort changepoint timing agreement ({n_traj} trajectories)")
    fig.tight_layout()
    out = plot_dir / "cohort_jaccard_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def variant_sort_key(traj_id: str) -> tuple[int | str, str]:
    """Sort trajectories by numeric variant id (e.g. BMMpM_109234_mdcrd_v → 109234)."""
    parts = traj_id.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return (int(parts[1]), traj_id)
    return (10**12, traj_id)


def variant_label(traj_id: str) -> str:
    parts = traj_id.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[1]
    return traj_id


def sorted_trajectory_ids(bkp: pd.DataFrame) -> list[str]:
    return sorted(bkp["traj_id"].unique(), key=variant_sort_key)


def plot_cohort_breakpoints_all_trajectories(
    changepoints_dir: Path,
    plot_dir: Path,
) -> Path:
    """Single overview: every trajectory (row) × breakpoint time (x), colored by group."""
    changepoints_dir = Path(changepoints_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    bkp = pd.read_csv(changepoints_dir / "all_breakpoints.csv")
    trajs = sorted_trajectory_ids(bkp)
    traj_to_y = {t: i for i, t in enumerate(trajs)}
    labels = [variant_label(t) for t in trajs]

    fig_h = max(8.0, 0.22 * len(trajs) + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    for grp in sorted(bkp["group"].unique()) if "group" in bkp.columns else list(DEFAULT_GROUPS):
        sub = bkp[bkp["group"] == grp]
        if sub.empty:
            continue
        ax.scatter(
            sub["time_ps"],
            sub["traj_id"].map(traj_to_y),
            c=GROUP_COLORS.get(grp, "#333333"),
            marker=GROUP_MARKERS.get(grp, "o"),
            s=55 if grp == "gsa" else 40,
            alpha=0.85,
            linewidths=0.8,
            label=grp,
            zorder=3,
        )

    ax.set_yticks(range(len(trajs)), labels, fontsize=7)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Trajectory variant")
    n_groups = bkp["group"].nunique() if "group" in bkp.columns else 0
    ax.set_title(
        f"Changepoints across all trajectories ({len(trajs)} variants, {n_groups} feature groups)"
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper right", ncol=4, framealpha=0.9)
    fig.tight_layout()
    out = plot_dir / "cohort_breakpoints_all_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _normalize_trace(values: np.ndarray) -> np.ndarray:
    """Min–max scale to [0, 1] for overlay comparison."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.zeros_like(values, dtype=float)
    lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    out = (values - lo) / (hi - lo)
    return np.where(np.isfinite(out), out, np.nan)


def plot_cohort_traces_normalized_all_trajectories(
    changepoints_dir: Path,
    features_dir: Path,
    plot_dir: Path,
    panels: list[tuple[str, str]] | None = None,
) -> Optional[Path]:
    """All trajectories per feature on shared axes (min–max normalized per trajectory)."""
    changepoints_dir = Path(changepoints_dir)
    features_dir = Path(features_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    panels = panels or DEFAULT_TIMELINE_PANELS

    bkp = pd.read_csv(changepoints_dir / "all_breakpoints.csv")
    trajs = sorted_trajectory_ids(bkp)
    if not trajs:
        return None

    fig, axes = plt.subplots(
        len(panels), 1, figsize=(14, 3.2 * len(panels)), sharex=True
    )
    if len(panels) == 1:
        axes = [axes]

    n_plotted = 0
    for ax, (col, ylabel) in zip(axes, panels):
        for traj_id in trajs:
            feat_path = resolve_features_csv(features_dir, traj_id)
            if feat_path is None:
                continue
            df = pd.read_csv(feat_path, usecols=lambda c: c in ("time_ps", col))
            if col not in df.columns:
                continue
            t = df["time_ps"].to_numpy(dtype=float)
            y = _normalize_trace(df[col].to_numpy(dtype=float))
            ax.plot(t, y, color="0.35", lw=0.6, alpha=0.35)
            n_plotted += 1
        ax.set_ylabel(f"{ylabel}\n(per-traj norm.)")
        ax.grid(alpha=0.2)

    if n_plotted == 0:
        plt.close(fig)
        return None

    axes[-1].set_xlabel("Time (ps)")
    fig.suptitle(
        f"All {len(trajs)} trajectories overlaid (each trace min–max normalized)",
        y=1.01,
    )
    fig.tight_layout()
    out = plot_dir / "cohort_traces_normalized_all_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_breakpoint_histogram(changepoints_dir: Path, plot_dir: Path) -> Path:
    changepoints_dir = Path(changepoints_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    bkp = pd.read_csv(changepoints_dir / "all_breakpoints.csv")
    counts = bkp.groupby(["traj_id", "group"]).size().reset_index(name="n_bkps")

    fig, ax = plt.subplots(figsize=(7, 4))
    for grp in DEFAULT_GROUPS:
        sub = counts[counts["group"] == grp]["n_bkps"]
        ax.hist(sub, bins=range(0, 8), alpha=0.5, label=grp, align="left")
    ax.set_xlabel("Breakpoints per trajectory")
    ax.set_ylabel("Count")
    ax.set_title("Breakpoint count distribution by feature group")
    ax.legend()
    fig.tight_layout()
    out = plot_dir / "breakpoint_count_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def discover_cluster_representatives_csvs(changepoints_dir: Path) -> list[Path]:
    """Find chosen-k ``cluster_representatives.csv`` under ``clusters/``.

    Prefers ``clusters/<group>/cluster_representatives.csv`` and ignores
    ``by_k/`` inspection outputs unless no top-level file exists.
    """
    clusters_root = Path(changepoints_dir) / "clusters"
    if not clusters_root.is_dir():
        # Also accept changepoints_dir itself if it already is the clusters root.
        clusters_root = Path(changepoints_dir)
        if not clusters_root.is_dir():
            return []

    top_level = sorted(
        p
        for p in clusters_root.glob("*/cluster_representatives.csv")
        if "by_k" not in p.parts
    )
    if top_level:
        return top_level

    # Fallback: any representatives CSV, still skipping by_k when possible.
    all_found = sorted(clusters_root.glob("**/cluster_representatives.csv"))
    non_by_k = [p for p in all_found if "by_k" not in p.parts]
    return non_by_k if non_by_k else all_found


def resolve_cluster_representatives_csvs(
    changepoints_dir: Path,
    explicit_paths: Optional[Sequence[str | Path]] = None,
) -> list[Path]:
    if explicit_paths:
        return [Path(p) for p in explicit_paths]
    return discover_cluster_representatives_csvs(changepoints_dir)


def load_cluster_representatives(csv_paths: Sequence[Path]) -> pd.DataFrame:
    """Load and stack one or more cluster_representatives.csv files."""
    required = {"traj_id", "cluster_label", "segment_id", "start_frame", "end_frame"}
    frames: list[pd.DataFrame] = []
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            log_event(
                "warning",
                f"Skipping missing cluster representatives CSV: {csv_path}",
                component="summarize_changepoint_results",
            )
            continue
        df = pd.read_csv(csv_path)
        missing = required - set(df.columns)
        if missing:
            log_event(
                "warning",
                f"Skipping {csv_path}: missing columns {sorted(missing)}",
                component="summarize_changepoint_results",
            )
            continue
        tagged = df.copy()
        tagged["source_csv"] = str(csv_path)
        frames.append(tagged)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def frame_to_time_ps(df: pd.DataFrame, frame: int) -> Optional[float]:
    """Map a simulation frame index to time (ps) using feature CSV columns."""
    if "frame" in df.columns:
        match = df.loc[df["frame"] == frame, "time_ps"]
        if len(match):
            return float(match.iloc[0])
    if "time_ps" in df.columns and 0 <= frame < len(df):
        return float(df["time_ps"].iloc[frame])
    return None


def segment_time_bounds(
    df: pd.DataFrame,
    start_frame: int,
    end_frame: int,
    rep_frame: Optional[int] = None,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (start_ps, end_ps, rep_ps) for a segment frame range."""
    start_ps = frame_to_time_ps(df, int(start_frame))
    end_ps = frame_to_time_ps(df, int(end_frame))
    rep_ps = frame_to_time_ps(df, int(rep_frame)) if rep_frame is not None else None
    return start_ps, end_ps, rep_ps


def default_timeline_trajectories(changepoints_dir: Path) -> list[str]:
    changepoints_dir = Path(changepoints_dir)
    cmp = pd.read_csv(changepoints_dir / "changepoint_timing_comparison.csv")
    iod_gsa = cmp[
        ((cmp["group_a"] == "iodine") & (cmp["group_b"] == "gsa"))
        | ((cmp["group_a"] == "gsa") & (cmp["group_b"] == "iodine"))
    ]
    chosen: list[str] = []
    if (changepoints_dir / "all_breakpoints.csv").exists():
        bkp = pd.read_csv(changepoints_dir / "all_breakpoints.csv")
        if len(bkp):
            chosen.append(str(bkp["traj_id"].iloc[0]))
    if len(iod_gsa):
        chosen.append(str(iod_gsa.sort_values("jaccard").iloc[0]["traj_id"]))
        chosen.append(
            str(iod_gsa.sort_values("jaccard", ascending=False).iloc[0]["traj_id"])
        )
    return list(dict.fromkeys(chosen))


def plot_timeline(
    traj_id: str,
    *,
    changepoints_dir: Path,
    features_dir: Path,
    plot_dir: Path,
    panels: list[tuple[str, str]] | None = None,
    cluster_label: Optional[int] = None,
    segment_id: Optional[int] = None,
    segment_start_ps: Optional[float] = None,
    segment_end_ps: Optional[float] = None,
    rep_time_ps: Optional[float] = None,
    features_suffix: Optional[str] = None,
) -> Optional[Path]:
    changepoints_dir = Path(changepoints_dir)
    features_dir = Path(features_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    feat_path = resolve_features_csv(
        features_dir, traj_id, suffix=features_suffix
    )
    if feat_path is None:
        log_event(
            "warning",
            f"Skipping timeline for {traj_id}: no features CSV found",
            component="summarize_changepoint_results",
        )
        return None

    df = pd.read_csv(feat_path)
    panels = resolve_timeline_panels(df, panels)
    if not panels:
        log_event(
            "warning",
            f"Skipping timeline for {traj_id}: no usable panel columns",
            component="summarize_changepoint_results",
        )
        return None

    bkp = pd.read_csv(changepoints_dir / "all_breakpoints.csv")
    bsub = bkp[bkp["traj_id"] == traj_id]
    time = df["time_ps"] if "time_ps" in df.columns else pd.Series(df.index, name="time_ps")

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.8 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    highlight_segment = (
        segment_start_ps is not None
        and segment_end_ps is not None
        and np.isfinite(segment_start_ps)
        and np.isfinite(segment_end_ps)
    )

    groups_present = (
        sorted(bsub["group"].unique()) if "group" in bsub.columns else list(DEFAULT_GROUPS)
    )

    for ax, (col, ylabel) in zip(axes, panels):
        if col not in df.columns:
            ax.set_ylabel(ylabel)
            ax.text(0.5, 0.5, f"missing column: {col}", transform=ax.transAxes, ha="center")
            continue
        if highlight_segment:
            ax.axvspan(
                segment_start_ps,
                segment_end_ps,
                color="#ffdf80",
                alpha=0.25,
                zorder=1,
            )
        ax.plot(time, df[col], color="0.3", lw=0.8, zorder=2)
        for grp in groups_present:
            color = GROUP_COLORS.get(grp, "#333333")
            for _, row in bsub[bsub["group"] == grp].iterrows():
                ax.axvline(
                    row["time_ps"],
                    color=color,
                    alpha=0.7,
                    lw=1.2,
                    ls="--",
                    zorder=3,
                )
        if rep_time_ps is not None and np.isfinite(rep_time_ps):
            ax.axvline(rep_time_ps, color="0.15", alpha=0.9, lw=1.0, ls=":", zorder=4)
        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel("Time (ps)")
    handles = [
        Line2D([0], [0], color=GROUP_COLORS.get(g, "#333333"), ls="--", label=g)
        for g in groups_present
    ]
    if highlight_segment:
        handles.append(
            Line2D([0], [0], color="#ffdf80", alpha=0.6, lw=6, label="cluster medoid segment")
        )
    if rep_time_ps is not None and np.isfinite(rep_time_ps):
        handles.append(Line2D([0], [0], color="0.15", ls=":", label="medoid frame"))
    fig.legend(handles=handles, loc="upper right")
    if cluster_label is not None and segment_id is not None:
        title = f"Cluster {cluster_label} medoid (seg {segment_id}): {traj_id}"
        out_name = f"timeline_cluster_{cluster_label}_{traj_id}.png"
    else:
        title = f"Breakpoints overlay: {traj_id}"
        out_name = f"timeline_{traj_id}.png"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out = plot_dir / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cluster_representative_timelines(
    reps: pd.DataFrame,
    *,
    changepoints_dir: Path,
    features_dir: Path,
    plot_dir: Path,
    panels: list[tuple[str, str]] | None = None,
    features_suffix: Optional[str] = None,
) -> list[str]:
    """Plot breakpoint timelines for each row in cluster_representatives.csv."""
    written: list[str] = []
    for _, row in reps.iterrows():
        traj_id = str(row["traj_id"])
        feat_path = resolve_features_csv(
            features_dir, traj_id, suffix=features_suffix
        )
        if feat_path is None:
            log_event(
                "warning",
                f"Skipping cluster rep timeline for {traj_id}: no features CSV found",
                component="summarize_changepoint_results",
            )
            continue
        df = pd.read_csv(feat_path, usecols=lambda c: c in ("frame", "time_ps"))
        rep_frame = (
            int(row["rep_frame"])
            if "rep_frame" in row and pd.notna(row["rep_frame"])
            else None
        )
        start_ps, end_ps, rep_ps = segment_time_bounds(
            df,
            int(row["start_frame"]),
            int(row["end_frame"]),
            rep_frame,
        )
        out = plot_timeline(
            traj_id,
            changepoints_dir=changepoints_dir,
            features_dir=features_dir,
            plot_dir=plot_dir,
            panels=panels,
            cluster_label=int(row["cluster_label"]),
            segment_id=int(row["segment_id"]),
            segment_start_ps=start_ps,
            segment_end_ps=end_ps,
            rep_time_ps=rep_ps,
            features_suffix=features_suffix,
        )
        if out is not None:
            written.append(out.name)
    return written


def summarize_changepoint_results(
    changepoints_dir: Path,
    features_dir: Path,
    *,
    plot_dir: Optional[Path] = None,
    trajectories: Optional[Sequence[str]] = None,
    skip_tables: bool = False,
    skip_individual_timelines: bool = False,
    cluster_representatives_csv: Optional[Sequence[str | Path]] = None,
    skip_cluster_rep_timelines: bool = False,
    features_suffix: Optional[str] = None,
    cluster_timeline_top_pairs: int = 5,
    cluster_timeline_group: str = "endpoint",
    clustered_df: Optional[pd.DataFrame] = None,
) -> dict[str, Path]:
    """Full summarize stage: cohort tables + plots + optional timelines."""
    changepoints_dir = Path(changepoints_dir)
    features_dir = Path(features_dir)
    plot_dir = Path(plot_dir) if plot_dir else changepoints_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    required = [
        changepoints_dir / "all_breakpoints.csv",
        changepoints_dir / "all_segment_stats.csv",
        changepoints_dir / "changepoint_timing_comparison.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required changepoint outputs:\n"
            + "\n".join(f"  {p}" for p in missing)
        )

    if not skip_tables:
        written.update(write_cohort_tables(changepoints_dir))
        log_event(
            "info",
            "Wrote cohort summary CSVs",
            component="summarize_changepoint_results",
        )

    written["cohort_jaccard_heatmap"] = plot_jaccard_heatmap(changepoints_dir, plot_dir)
    written["breakpoint_count_histogram"] = plot_breakpoint_histogram(
        changepoints_dir, plot_dir
    )
    written["cohort_breakpoints_all_trajectories"] = (
        plot_cohort_breakpoints_all_trajectories(changepoints_dir, plot_dir)
    )

    # Choose timeline panels from the first available features CSV.
    sample_panels = list(DEFAULT_TIMELINE_PANELS)
    bkp = pd.read_csv(changepoints_dir / "all_breakpoints.csv")
    sample_trajs = sorted_trajectory_ids(bkp)
    for tid in sample_trajs:
        sample_path = resolve_features_csv(
            features_dir, tid, suffix=features_suffix
        )
        if sample_path is not None:
            sample_df = pd.read_csv(sample_path, nrows=5)
            sample_panels = resolve_timeline_panels(sample_df, DEFAULT_TIMELINE_PANELS)
            break

    traces = plot_cohort_traces_normalized_all_trajectories(
        changepoints_dir,
        features_dir,
        plot_dir,
        sample_panels if sample_panels else DEFAULT_TIMELINE_PANELS,
    )
    if traces is not None:
        written["cohort_traces_normalized_all_trajectories"] = traces

    timeline_names: list[str] = []
    if not skip_individual_timelines:
        traj_ids = list(trajectories) if trajectories else default_timeline_trajectories(
            changepoints_dir
        )
        for traj_id in traj_ids:
            out = plot_timeline(
                traj_id,
                changepoints_dir=changepoints_dir,
                features_dir=features_dir,
                plot_dir=plot_dir,
                panels=sample_panels if sample_panels else None,
                features_suffix=features_suffix,
            )
            if out is not None:
                written[out.name] = out
                timeline_names.append(out.name)

    cluster_written: list[str] = []
    if not skip_cluster_rep_timelines:
        rep_csvs = resolve_cluster_representatives_csvs(
            changepoints_dir,
            cluster_representatives_csv,
        )
        reps = load_cluster_representatives(rep_csvs)
        if len(reps):
            cluster_panels = sample_panels if sample_panels else None
            if cluster_timeline_top_pairs > 0:
                ranking = rank_cluster_correlated_pair_means(
                    changepoints_dir,
                    features_dir,
                    group=cluster_timeline_group,
                    clustered_df=clustered_df,
                    features_suffix=features_suffix,
                )
                if len(ranking):
                    corr_csv = (
                        changepoints_dir / "endpoint_pair_cluster_correlation.csv"
                    )
                    ranking.to_csv(corr_csv, index=False)
                    written["endpoint_pair_cluster_correlation.csv"] = corr_csv
                    correlated = cluster_correlated_timeline_panels(
                        ranking, top_n=cluster_timeline_top_pairs
                    )
                    if correlated:
                        cluster_panels = correlated
                        log_event(
                            "info",
                            (
                                f"Cluster timeline panels: top {cluster_timeline_top_pairs} "
                                f"endpoint features by segment-level η² vs cluster"
                            ),
                            component="summarize_changepoint_results",
                        )
            cluster_written = plot_cluster_representative_timelines(
                reps,
                changepoints_dir=changepoints_dir,
                features_dir=features_dir,
                plot_dir=plot_dir,
                panels=cluster_panels,
                features_suffix=features_suffix,
            )
            for name in cluster_written:
                written[name] = plot_dir / name
            timeline_names.extend(cluster_written)

    log_event(
        "info",
        (
            f"Cohort plots written; {len(timeline_names)} timeline plot(s) "
            f"({len(cluster_written)} cluster medoid)"
        ),
        component="summarize_changepoint_results",
    )
    return written


def compare_endpoint_clusters_to_deformation(
    changepoints_dir: Path | str,
    features_dir: Path | str,
    *,
    rmsd_from: Optional[Path | str] = None,
    plot_dir: Optional[Path | str] = None,
    group: str = "endpoint",
) -> dict[str, Path]:
    """Summarise endpoint clusters vs deformation proxies (distance + optional RMSD).

    Joins cluster labels from ``clusters/<group>/`` with segment stats and
    optional ``assembly_rmsd_to_ref`` from a matching GSA features directory.
    Emits ``endpoint_cluster_deformation_summary.csv`` and a bar/scatter plot.
    """
    changepoints_dir = Path(changepoints_dir)
    features_dir = Path(features_dir)
    plot_dir = Path(plot_dir) if plot_dir else changepoints_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # Prefer clustered segment table if present
    clustered_candidates = [
        changepoints_dir / "clusters" / group / "segments_clustered.csv",
        changepoints_dir / "clusters" / "all_segments_clustered.csv",
    ]
    seg_path = changepoints_dir / "all_segment_stats.csv"
    seg: Optional[pd.DataFrame] = None
    for clustered_path in clustered_candidates:
        if clustered_path.exists():
            seg = pd.read_csv(clustered_path)
            if "group" in seg.columns:
                seg = seg[seg["group"] == group].copy()
            break
    if seg is None and seg_path.exists():
        seg = pd.read_csv(seg_path)
        if "group" in seg.columns:
            seg = seg[seg["group"] == group].copy()
    if seg is None:
        log_event(
            "warning",
            "No segment stats found for endpoint deformation comparison",
            component="endpoint_deformation_compare",
        )
        return written

    if seg.empty:
        return written

    label_col = None
    for cand in ("cluster_label", "cluster", "label"):
        if cand in seg.columns:
            label_col = cand
            break

    # Optional RMSD overlay: mean assembly_rmsd_to_ref over each segment
    rmsd_means: list[float] = []
    rmsd_dir = Path(rmsd_from) if rmsd_from else None
    if rmsd_dir is not None and rmsd_dir.exists():
        for _, row in seg.iterrows():
            traj_id = str(row["traj_id"])
            gsa_path = resolve_features_csv(
                rmsd_dir, traj_id, suffix="_gsa_features.csv"
            )
            if gsa_path is None or "assembly_rmsd_to_ref" not in pd.read_csv(
                gsa_path, nrows=0
            ).columns:
                rmsd_means.append(np.nan)
                continue
            gsa = pd.read_csv(
                gsa_path, usecols=lambda c: c in ("frame", "assembly_rmsd_to_ref")
            )
            start_f = int(row["start_frame"])
            end_f = int(row["end_frame"])
            mask = (gsa["frame"] >= start_f) & (gsa["frame"] <= end_f)
            vals = gsa.loc[mask, "assembly_rmsd_to_ref"].to_numpy(dtype=float)
            rmsd_means.append(float(np.nanmean(vals)) if len(vals) else np.nan)
        seg = seg.copy()
        seg["assembly_rmsd_to_ref_mean"] = rmsd_means

    dist_col_candidates = [
        "endpoint_dist_mean_mean",
        "endpoint_dist_mean",
        "endpoint_dist_min_mean",
        "endpoint_dist_max_mean",
    ]
    dist_col = next((c for c in dist_col_candidates if c in seg.columns), None)

    summary_rows: list[dict] = []
    if label_col is not None:
        for label, sub in seg.groupby(label_col):
            row: dict = {
                "cluster_label": label,
                "n_segments": len(sub),
                "n_trajectories": sub["traj_id"].nunique() if "traj_id" in sub.columns else np.nan,
                "total_frames": int(sub["n_frames"].sum()) if "n_frames" in sub.columns else np.nan,
            }
            if dist_col is not None:
                row["endpoint_dist_mean"] = float(sub[dist_col].mean())
                row["endpoint_dist_std"] = float(sub[dist_col].std())
            if "assembly_rmsd_to_ref_mean" in sub.columns:
                row["assembly_rmsd_to_ref_mean"] = float(
                    np.nanmean(sub["assembly_rmsd_to_ref_mean"])
                )
                row["assembly_rmsd_to_ref_std"] = float(
                    np.nanstd(sub["assembly_rmsd_to_ref_mean"])
                )
            summary_rows.append(row)
    else:
        # No clusters — report whole-group stats
        row = {
            "cluster_label": "all",
            "n_segments": len(seg),
            "n_trajectories": seg["traj_id"].nunique() if "traj_id" in seg.columns else np.nan,
            "total_frames": int(seg["n_frames"].sum()) if "n_frames" in seg.columns else np.nan,
        }
        if dist_col is not None:
            row["endpoint_dist_mean"] = float(seg[dist_col].mean())
            row["endpoint_dist_std"] = float(seg[dist_col].std())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    if "endpoint_dist_mean" in summary.columns:
        summary = summary.sort_values("endpoint_dist_mean").reset_index(drop=True)

    out_csv = changepoints_dir / "endpoint_cluster_deformation_summary.csv"
    summary.to_csv(out_csv, index=False)
    written["endpoint_cluster_deformation_summary.csv"] = out_csv

    # Bar / scatter plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(summary))
    labels = [str(v) for v in summary["cluster_label"]]
    if "endpoint_dist_mean" in summary.columns:
        ax.bar(
            x,
            summary["endpoint_dist_mean"],
            yerr=summary.get("endpoint_dist_std"),
            color=GROUP_COLORS.get("endpoint", "#ff7f0e"),
            alpha=0.75,
            label="endpoint dist mean",
        )
        ax.set_ylabel("Mean endpoint-site distance (Å)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Cluster (ordered by mean endpoint distance)")
    ax.set_title("Endpoint clusters vs deformation proxy")
    ax.grid(axis="y", alpha=0.25)

    if "assembly_rmsd_to_ref_mean" in summary.columns:
        ax2 = ax.twinx()
        ax2.plot(
            x,
            summary["assembly_rmsd_to_ref_mean"],
            color="#1f77b4",
            marker="o",
            lw=1.5,
            label="mean RMSD",
        )
        ax2.set_ylabel("Mean assembly RMSD to ref (Å)")
        # Combined legend
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left")
    else:
        ax.legend(loc="upper left")

    # Annotate paper-like RMSD reference bands
    ax.annotate(
        "paper refs: A~1.0  B~1.5  C1~1.6–2.5  C2~2.7 Å RMSD",
        xy=(0.5, -0.18),
        xycoords="axes fraction",
        ha="center",
        fontsize=8,
        color="0.4",
    )
    fig.tight_layout()
    out_png = plot_dir / "endpoint_cluster_deformation.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written["endpoint_cluster_deformation.png"] = out_png

    log_event(
        "info",
        f"Wrote endpoint deformation comparison ({len(summary)} cluster rows)",
        component="endpoint_deformation_compare",
    )
    return written


_ENDPOINT_CHANGPOINTS_PREFIX = "endpoint_changepoints_"
_GSA_CHANGPOINTS_PREFIX = "gsa_changepoints_"
_LOW_CONFIDENCE_N_SEGMENTS = 5


def cohort_name_from_changepoints_dir(changepoints_dir: Path | str) -> str:
    """Extract cube cohort id (e.g. ``BMMpM``) from a changepoints path."""
    name = Path(changepoints_dir).name
    for prefix in (_ENDPOINT_CHANGPOINTS_PREFIX, _GSA_CHANGPOINTS_PREFIX):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _cohens_d_direction(d: float) -> str:
    if not np.isfinite(d) or abs(d) < 1e-12:
        return "neutral"
    return "open" if d > 0 else "closed"


def _aggregate_paper_d1_by_cluster(d1_df: pd.DataFrame) -> pd.DataFrame:
    """Cluster-level means from ``paper_d1_segment_states.csv``."""
    if d1_df.empty or "cluster_label" not in d1_df.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    for label, sub in d1_df.groupby("cluster_label"):
        row: dict = {"cluster_label": label}
        for col in (
            "paper_d1_min_mean",
            "paper_d1_n_open_mean",
            "n_open_pairs",
            "n_closed_pairs",
            "n_elongated_pairs",
        ):
            if col in sub.columns:
                row[col] = float(sub[col].mean())
        if "n_open_pairs" in sub.columns:
            row["open_pair_segment_fraction"] = float((sub["n_open_pairs"] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_endpoint_cluster_proxies(
    changepoints_dir: Path | str,
    *,
    top_pairs: int = 10,
    low_confidence_n_segments: int = _LOW_CONFIDENCE_N_SEGMENTS,
) -> pd.DataFrame:
    """Per-cluster deformation proxies for one endpoint changepoint cohort.

    Combines endpoint distance (structural), paper d1 (cation–π), site-pair η²,
    and Cohen's *d* direction from existing pipeline CSVs. Assigns
    ``deformation_rank`` 0 = most closed (smallest mean endpoint distance).
    """
    changepoints_dir = Path(changepoints_dir)
    cohort = cohort_name_from_changepoints_dir(changepoints_dir)

    deform_path = changepoints_dir / "endpoint_cluster_deformation_summary.csv"
    if not deform_path.exists():
        log_event(
            "warning",
            f"Missing {deform_path.name} in {changepoints_dir}",
            component="endpoint_cluster_proxies",
        )
        return pd.DataFrame()

    deform = pd.read_csv(deform_path)
    if deform.empty or "endpoint_dist_mean" not in deform.columns:
        return pd.DataFrame()

    deform = deform.sort_values("endpoint_dist_mean").reset_index(drop=True)
    deform["deformation_rank"] = np.arange(len(deform), dtype=int)
    deform["cohort"] = cohort

    d1_path = changepoints_dir / "paper_d1_segment_states.csv"
    if d1_path.exists():
        d1_agg = _aggregate_paper_d1_by_cluster(pd.read_csv(d1_path))
        if not d1_agg.empty:
            deform = deform.merge(d1_agg, on="cluster_label", how="left")

    pairs_path = changepoints_dir / "endpoint_pair_cluster_correlation.csv"
    top_pair_df = pd.DataFrame()
    if pairs_path.exists():
        pairs = pd.read_csv(pairs_path)
        if not pairs.empty and "rank" in pairs.columns:
            top_pair_df = pairs.sort_values("rank").head(int(top_pairs)).copy()
        elif not pairs.empty and "eta_squared" in pairs.columns:
            top_pair_df = pairs.sort_values("eta_squared", ascending=False).head(
                int(top_pairs)
            ).copy()

    cohort_top_labels = (
        ";".join(top_pair_df["endpoint_label"].astype(str).tolist())
        if not top_pair_df.empty and "endpoint_label" in top_pair_df.columns
        else ""
    )

    dominant_labels: list[str] = []
    dominant_eta2: list[float] = []
    dominant_d: list[float] = []
    dominant_direction: list[str] = []
    n_marking: list[int] = []

    for _, row in deform.iterrows():
        label = row["cluster_label"]
        if top_pair_df.empty:
            dominant_labels.append("")
            dominant_eta2.append(np.nan)
            dominant_d.append(np.nan)
            dominant_direction.append("")
            n_marking.append(0)
            continue

        marked = top_pair_df[top_pair_df["best_cluster"] == label]
        n_marking.append(len(marked))
        if marked.empty:
            dominant_labels.append("")
            dominant_eta2.append(np.nan)
            dominant_d.append(np.nan)
            dominant_direction.append("")
        else:
            best = marked.sort_values("eta_squared", ascending=False).iloc[0]
            d_val = float(best.get("cohens_d_best", np.nan))
            dominant_labels.append(str(best.get("endpoint_label", "")))
            dominant_eta2.append(float(best.get("eta_squared", np.nan)))
            dominant_d.append(d_val)
            dominant_direction.append(_cohens_d_direction(d_val))

    deform["cohort_top_pair_labels"] = cohort_top_labels
    deform["dominant_top_pair"] = dominant_labels
    deform["dominant_top_pair_eta2"] = dominant_eta2
    deform["dominant_top_pair_cohens_d"] = dominant_d
    deform["cohens_d_direction"] = dominant_direction
    deform["n_top_pairs_marking_cluster"] = n_marking
    deform["low_confidence"] = deform["n_segments"] < int(low_confidence_n_segments)

    return deform


def compare_endpoint_clusters_across_cohorts(
    cohort_dirs: Sequence[Path | str],
    out_dir: Path | str,
    *,
    top_pairs: int = 10,
) -> dict[str, Path]:
    """Aggregate endpoint cluster proxies across multiple B* cohort directories."""
    out_dir = Path(out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    frames: list[pd.DataFrame] = []
    for d in cohort_dirs:
        summary = summarize_endpoint_cluster_proxies(d, top_pairs=top_pairs)
        if not summary.empty:
            frames.append(summary)

    if not frames:
        log_event(
            "warning",
            "No cohort summaries produced for cross-cohort comparison",
            component="endpoint_cluster_cross_cohort",
        )
        return written

    all_summary = pd.concat(frames, ignore_index=True)
    summary_path = out_dir / "cluster_proxy_summary.csv"
    all_summary.to_csv(summary_path, index=False)
    written["cluster_proxy_summary.csv"] = summary_path

    confident = all_summary[~all_summary["low_confidence"]].copy()
    rank_rows: list[dict] = []
    for rank in sorted(confident["deformation_rank"].unique()):
        row: dict = {"deformation_rank": int(rank)}
        for cohort in sorted(confident["cohort"].unique()):
            sub = confident[
                (confident["cohort"] == cohort)
                & (confident["deformation_rank"] == rank)
            ]
            if sub.empty:
                continue
            s = sub.iloc[0]
            row[f"{cohort}_endpoint_dist_mean"] = s.get("endpoint_dist_mean", np.nan)
            row[f"{cohort}_paper_d1_min_mean"] = s.get("paper_d1_min_mean", np.nan)
            row[f"{cohort}_n_open_pairs"] = s.get("n_open_pairs", np.nan)
            row[f"{cohort}_dominant_top_pair"] = s.get("dominant_top_pair", "")
            row[f"{cohort}_dominant_top_pair_eta2"] = s.get(
                "dominant_top_pair_eta2", np.nan
            )
        rank_rows.append(row)
    by_rank = pd.DataFrame(rank_rows)
    by_rank_path = out_dir / "cluster_proxy_by_rank.csv"
    by_rank.to_csv(by_rank_path, index=False)
    written["cluster_proxy_by_rank.csv"] = by_rank_path

    pair_rows: list[dict] = []
    matrix_rows: list[dict] = []
    for d in cohort_dirs:
        cohort = cohort_name_from_changepoints_dir(d)
        pairs_path = Path(d) / "endpoint_pair_cluster_correlation.csv"
        if not pairs_path.exists():
            continue
        pairs = pd.read_csv(pairs_path)
        if pairs.empty:
            continue
        if "rank" in pairs.columns:
            top = pairs.sort_values("rank").head(int(top_pairs))
        else:
            top = pairs.sort_values("eta_squared", ascending=False).head(int(top_pairs))
        for _, pr in top.iterrows():
            pair_rows.append(
                {
                    "cohort": cohort,
                    "rank": int(pr.get("rank", 0)),
                    "endpoint_label": str(pr.get("endpoint_label", "")),
                    "eta_squared": float(pr.get("eta_squared", np.nan)),
                    "cohens_d_best": float(pr.get("cohens_d_best", np.nan)),
                    "best_cluster": pr.get("best_cluster", np.nan),
                    "cohens_d_direction": _cohens_d_direction(
                        float(pr.get("cohens_d_best", np.nan))
                    ),
                }
            )
        for _, pr in top.iterrows():
            matrix_rows.append(
                {
                    "cohort": cohort,
                    "endpoint_label": str(pr.get("endpoint_label", "")),
                    "eta_squared": float(pr.get("eta_squared", np.nan)),
                    "best_cluster": pr.get("best_cluster", np.nan),
                    "deformation_rank_of_best_cluster": np.nan,
                }
            )

    top_pairs_df = pd.DataFrame(pair_rows)
    if not top_pairs_df.empty:
        freq = (
            top_pairs_df.groupby("endpoint_label")
            .size()
            .reset_index(name="cohort_count")
            .sort_values("cohort_count", ascending=False)
        )
        top_pairs_df = top_pairs_df.merge(freq, on="endpoint_label", how="left")
        top_path = out_dir / "top_site_pairs_by_cohort.csv"
        top_pairs_df.to_csv(top_path, index=False)
        written["top_site_pairs_by_cohort.csv"] = top_path

    matrix_df = pd.DataFrame(matrix_rows)
    if not matrix_df.empty and not all_summary.empty:
        rank_map = all_summary.set_index(["cohort", "cluster_label"])[
            "deformation_rank"
        ]
        for i, mrow in matrix_df.iterrows():
            key = (mrow["cohort"], mrow["best_cluster"])
            if key in rank_map.index:
                matrix_df.at[i, "deformation_rank_of_best_cluster"] = float(
                    rank_map.loc[key]
                )
        matrix_path = out_dir / "pair_best_cluster_matrix.csv"
        matrix_df.to_csv(matrix_path, index=False)
        written["pair_best_cluster_matrix.csv"] = matrix_path

    _write_cross_cohort_proxy_plots(
        all_summary,
        top_pairs_df,
        plot_dir,
        written,
    )

    log_event(
        "info",
        f"Wrote cross-cohort endpoint cluster proxy comparison ({len(all_summary)} rows)",
        component="endpoint_cluster_cross_cohort",
    )
    return written


def _write_cross_cohort_proxy_plots(
    summary: pd.DataFrame,
    top_pairs: pd.DataFrame,
    plot_dir: Path,
    written: dict[str, Path],
) -> None:
    """Matplotlib figures for cross-cohort cluster proxy comparison."""
    cohorts = sorted(summary["cohort"].unique())
    confident = summary[~summary["low_confidence"]]

    if not confident.empty and "endpoint_dist_mean" in confident.columns:
        pivot_dist = confident.pivot_table(
            index="deformation_rank",
            columns="cohort",
            values="endpoint_dist_mean",
            aggfunc="first",
        )
        fig, ax = plt.subplots(figsize=(max(6, len(cohorts) * 0.9), 4.5))
        im = ax.imshow(pivot_dist.values, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(np.arange(len(pivot_dist.columns)))
        ax.set_xticklabels(pivot_dist.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot_dist.index)))
        ax.set_yticklabels([str(int(v)) for v in pivot_dist.index])
        ax.set_xlabel("Cohort")
        ax.set_ylabel("Deformation rank (0=closed)")
        ax.set_title("Mean endpoint distance (Å) by deformation rank")
        fig.colorbar(im, ax=ax, label="endpoint_dist_mean (Å)")
        fig.tight_layout()
        p = plot_dir / "deformation_rank_vs_endpoint_dist.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written["plots/deformation_rank_vs_endpoint_dist.png"] = p

    if not confident.empty and "paper_d1_min_mean" in confident.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, col, title in (
            (axes[0], "paper_d1_min_mean", "paper_d1_min mean"),
            (axes[1], "n_open_pairs", "n_open_pairs mean"),
        ):
            if col not in confident.columns:
                ax.set_visible(False)
                continue
            pivot = confident.pivot_table(
                index="deformation_rank",
                columns="cohort",
                values=col,
                aggfunc="first",
            )
            im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([str(int(v)) for v in pivot.index])
            ax.set_xlabel("Cohort")
            ax.set_ylabel("Deformation rank")
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Paper d1 proxies by deformation rank", y=1.02)
        fig.tight_layout()
        p = plot_dir / "deformation_rank_vs_paper_d1.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written["plots/deformation_rank_vs_paper_d1.png"] = p

    if not top_pairs.empty:
        top5 = (
            top_pairs.sort_values(["cohort", "rank"])
            .groupby("cohort")
            .head(5)
        )
        labels = sorted(top5["endpoint_label"].unique())
        label_to_i = {lab: i for i, lab in enumerate(labels)}
        mat = np.full((len(cohorts), len(labels)), np.nan)
        for i, cohort in enumerate(cohorts):
            sub = top5[top5["cohort"] == cohort]
            for _, row in sub.iterrows():
                j = label_to_i.get(str(row["endpoint_label"]))
                if j is not None:
                    mat[i, j] = float(row["eta_squared"])
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), max(4, len(cohorts) * 0.6)))
        im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticks(np.arange(len(cohorts)))
        ax.set_yticklabels(cohorts)
        ax.set_title("Top-5 site-pair η² by cohort")
        fig.colorbar(im, ax=ax, label="η²")
        fig.tight_layout()
        p = plot_dir / "top_pairs_eta2_heatmap.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written["plots/top_pairs_eta2_heatmap.png"] = p

    if (
        not confident.empty
        and "endpoint_dist_mean" in confident.columns
        and "paper_d1_min_mean" in confident.columns
    ):
        n_c = len(cohorts)
        ncols = min(3, n_c)
        nrows = int(np.ceil(n_c / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes_flat = np.atleast_1d(axes).ravel()
        for ax, cohort in zip(axes_flat, cohorts):
            sub = confident[confident["cohort"] == cohort]
            sc = ax.scatter(
                sub["endpoint_dist_mean"],
                sub["paper_d1_min_mean"],
                c=sub["deformation_rank"],
                cmap="coolwarm",
                s=60,
                edgecolors="k",
                linewidths=0.4,
            )
            for _, r in sub.iterrows():
                ax.annotate(
                    str(int(r["deformation_rank"])),
                    (r["endpoint_dist_mean"], r["paper_d1_min_mean"]),
                    fontsize=7,
                    ha="center",
                    va="center",
                    color="white",
                )
            ax.set_xlabel("endpoint_dist_mean (Å)")
            ax.set_ylabel("paper_d1_min_mean (Å)")
            ax.set_title(cohort)
            ax.grid(alpha=0.25)
        for ax in axes_flat[len(cohorts) :]:
            ax.set_visible(False)
        cbar = fig.colorbar(sc, ax=axes_flat[: len(cohorts)], shrink=0.6)
        cbar.set_label("deformation_rank")
        fig.suptitle("Endpoint distance vs paper d1 (annotated by rank)", y=1.01)
        fig.subplots_adjust(top=0.88, wspace=0.35)
        p = plot_dir / "proxy_consistency_scatter.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written["plots/proxy_consistency_scatter.png"] = p