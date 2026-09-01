"""Changepoint detection on GSA feature groups from pre-computed feature CSVs."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.ruptures_utils import detect_changepoints
from src.utils.run_log import log_event, step

from .feature_groups import (
    DEFAULT_GROUPS,
    group_column_map,
    summary_base_columns,
    traj_id_from_features_stem,
)


@dataclass
class ChangepointConfig:
    """Parameters for multivariate Pelt / ruptures detection per feature group."""

    method: str = "Pelt"
    cost_model: str = "rbf"
    penalty: Optional[float] = None
    n_bkps: Optional[int] = None
    min_size: int = 10
    jump: int = 5
    tolerance_frames: int = 50
    normalize: bool = True
    groups: tuple[str, ...] = DEFAULT_GROUPS
    include_site_pairs_in_detection: bool = False


@dataclass
class ChangepointTables:
    """Tidy tables produced by changepoint detection."""

    breakpoints: pd.DataFrame = field(default_factory=pd.DataFrame)
    segment_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    comparison: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_signal_matrix(
    df: pd.DataFrame,
    cols: list[str],
    normalize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build (n_frames, n_features) float64 array ready for ruptures.

    Filters to available columns, fills NaN/inf with column means, drops
    zero-variance columns, and optionally z-score normalises.

    Returns (matrix, used_column_names). Returns empty array if nothing usable.
    """
    available = [c for c in cols if c in df.columns]
    if not available:
        return np.empty((len(df), 0), dtype=np.float64), []

    mat = df[available].to_numpy(dtype=np.float64)
    mat = np.where(np.isfinite(mat), mat, np.nan)
    col_means = np.nanmean(mat, axis=0)
    for j in range(mat.shape[1]):
        fill = col_means[j] if np.isfinite(col_means[j]) else 0.0
        mat[np.isnan(mat[:, j]), j] = fill

    var = mat.var(axis=0)
    keep = var > 0
    mat = mat[:, keep]
    used_cols = [c for c, k in zip(available, keep) if k]

    if mat.shape[1] == 0:
        return np.empty((len(df), 0), dtype=np.float64), []

    if normalize:
        mat = StandardScaler().fit_transform(mat)

    return mat, used_cols


def auto_penalty(n: int, normalize: bool) -> Optional[float]:
    """BIC-inspired penalty for normalized multivariate Pelt.

    After z-score normalization each feature has unit variance, so we use
    log(n) rather than log(n)*d*var to avoid over-penalising high-dimensional
    groups.  For un-normalised signals we fall back to None (let detect_changepoints
    apply its own heuristic).
    """
    return float(np.log(n)) if normalize else None


def _match_breakpoints(
    bkps_a: Sequence[int],
    bkps_b: Sequence[int],
    tolerance: int,
) -> list[tuple[int, int, float]]:
    """Greedy 1-to-1 matches within ±tolerance frames.

    Returns ``(index_in_a, index_in_b, abs_distance)``. Each index is used at
    most once. Edges are sorted by distance (then breakpoint values) so the
    matched set does not depend on argument order when distances are unique.
    """
    if not bkps_a or not bkps_b or tolerance < 0:
        return []
    a = [int(x) for x in bkps_a]
    b = [int(x) for x in bkps_b]
    candidates: list[tuple[int, int, int]] = []
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            d = abs(x - y)
            if d <= tolerance:
                candidates.append((d, i, j))
    candidates.sort(
        key=lambda t: (t[0], min(a[t[1]], b[t[2]]), max(a[t[1]], b[t[2]]))
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for d, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j, float(d)))
    return matched


def compare_breakpoints(
    bkps_a: list[int],
    bkps_b: list[int],
    tolerance: int,
    time_arr: np.ndarray,
) -> dict[str, float]:
    """Pairwise timing similarity between two sets of signal-index breakpoints.

    Matches are greedy 1-to-1 within ±tolerance frames. ``n_shared`` is the
    number of matched pairs; Jaccard is ``n_shared / (n_a + n_b - n_shared)``;
    ``mean_timing_offset_*`` is the mean distance of those matched pairs
    (NaN if none). ``compare_breakpoints(A, B)`` and ``(B, A)`` agree on
    n_shared, Jaccard, and offset.
    """
    n_a, n_b = len(bkps_a), len(bkps_b)
    matched = _match_breakpoints(bkps_a, bkps_b, tolerance)
    n_shared = len(matched)

    if n_a == 0 and n_b == 0:
        jaccard = 1.0
    else:
        denom = n_a + n_b - n_shared
        jaccard = (n_shared / denom) if denom else 0.0

    if matched:
        mean_off_frames = float(np.mean([d for _, _, d in matched]))
        if len(time_arr) >= 2:
            frame_step_ps = float(time_arr[1] - time_arr[0])
        else:
            frame_step_ps = 1.0
        mean_off_ps = mean_off_frames * frame_step_ps
    else:
        mean_off_frames = float("nan")
        mean_off_ps = float("nan")

    return {
        "n_bkps_a": n_a,
        "n_bkps_b": n_b,
        "n_shared": n_shared,
        "jaccard": jaccard,
        "mean_timing_offset_frames": mean_off_frames,
        "mean_timing_offset_ps": mean_off_ps,
    }


_CUBE_PREFIX_RE = re.compile(
    r"^(?:BHHpH|BHHpM|BMHpH|BMHpM|BMMpH|BMMpM)_"
)
_FEATURE_SUFFIX_RE = re.compile(r"_(?:gsa|endpoint)_features$")


def canonicalize_traj_id(traj_id: str) -> str:
    """Strip cube prefix / feature-CSV suffix so GSA and endpoint IDs match.

    ``BMMpM_109345_mdcrd_v`` and ``109345_mdcrd_v`` both become ``109345_mdcrd_v``.
    """
    s = str(traj_id).strip()
    s = _CUBE_PREFIX_RE.sub("", s)
    s = _FEATURE_SUFFIX_RE.sub("", s)
    return s


def _load_breakpoints_table(source: Path | str | pd.DataFrame) -> pd.DataFrame:
    """Load ``all_breakpoints.csv`` from a path, directory, or DataFrame."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(source)
        if path.is_dir():
            path = path / "all_breakpoints.csv"
        if not path.exists():
            raise FileNotFoundError(f"No breakpoints table at {path}")
        df = pd.read_csv(path)
    if df.empty:
        return df
    if "traj_id" not in df.columns or "group" not in df.columns:
        raise ValueError(
            f"Breakpoints table must include traj_id and group columns; got {list(df.columns)}"
        )
    df["source_traj_id"] = df["traj_id"].astype(str)
    df["traj_id"] = df["traj_id"].map(canonicalize_traj_id)
    return df


def _frame_times(sub: pd.DataFrame) -> tuple[np.ndarray, Optional[np.ndarray]]:
    loc_col = "frame" if "frame" in sub.columns else "signal_index"
    frames = sub[loc_col].to_numpy(dtype=np.int64)
    times = (
        sub["time_ps"].to_numpy(dtype=np.float64)
        if "time_ps" in sub.columns
        else None
    )
    return frames, times


def _mean_time_offset_ps(
    frames_a: np.ndarray,
    frames_b: np.ndarray,
    times_a: Optional[np.ndarray],
    times_b: Optional[np.ndarray],
    mean_off_frames: float,
    tolerance_frames: int,
) -> float:
    """Mean |t_a − t_b| over the same 1-to-1 matches used for Jaccard.

    Falls back to ``mean_off_frames * dt`` when per-breakpoint times are
    missing. Returns NaN when there are no matches.
    """
    if not np.isfinite(mean_off_frames) or not len(frames_a) or not len(frames_b):
        return float("nan")
    matched = _match_breakpoints(
        [int(x) for x in frames_a],
        [int(x) for x in frames_b],
        tolerance_frames,
    )
    if (
        matched
        and times_a is not None
        and times_b is not None
        and len(times_a) == len(frames_a)
        and len(times_b) == len(frames_b)
    ):
        offsets = []
        for i, j, _d in matched:
            ta = float(times_a[i])
            tb = float(times_b[j])
            if np.isfinite(ta) and np.isfinite(tb):
                offsets.append(abs(ta - tb))
        if offsets:
            return float(np.mean(offsets))
    dt = 1.0
    for frames, times in ((frames_a, times_a), (frames_b, times_b)):
        if times is None or len(frames) < 2:
            continue
        order = np.argsort(frames)
        dts = np.diff(np.asarray(times, dtype=float)[order])
        dts = dts[np.isfinite(dts) & (dts > 0)]
        if len(dts):
            dt = float(np.median(dts))
            break
    return float(mean_off_frames * dt)


def compare_changepoint_timing(
    *sources: Path | str | pd.DataFrame,
    tolerance_frames: int = 50,
    output_dir: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Compare breakpoint timing across already-computed changepoint tables.

    Each *source* is an ``all_breakpoints.csv`` path, a changepoints directory,
    or a DataFrame. Trajectory IDs are canonicalized so endpoint IDs
    (``109345_mdcrd_v``) match GSA IDs (``BMMpM_109345_mdcrd_v``).

    A group pair is reported for a trajectory only when that trajectory is
    present in a source that contains each group (missing groups count as
    zero breakpoints). Writes ``changepoint_timing_comparison.csv`` (and a
    merged ``all_breakpoints.csv``) when *output_dir* is set.
    """
    if len(sources) < 1:
        raise ValueError("Need at least one breakpoints source")

    loaded: list[pd.DataFrame] = []
    coverage: dict[str, set[str]] = {}
    all_groups: set[str] = set()
    for src in sources:
        df = _load_breakpoints_table(src)
        if df.empty:
            continue
        groups = {str(g) for g in df["group"].dropna().unique()}
        all_groups.update(groups)
        for tid in df["traj_id"].astype(str).unique():
            coverage.setdefault(tid, set()).update(groups)
        loaded.append(df)

    if not loaded:
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
                "mean_timing_offset_frames",
                "mean_timing_offset_ps",
            ]
        )
        if output_dir is not None:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            cmp.to_csv(Path(output_dir) / "changepoint_timing_comparison.csv", index=False)
        return cmp

    merged = pd.concat(loaded, ignore_index=True)
    groups = sorted(all_groups)
    rows: list[dict] = []
    for tid, covered in sorted(coverage.items()):
        tdf = merged[merged["traj_id"] == tid]
        comparable = sorted(covered)
        if len(comparable) < 2:
            continue
        for grp_a, grp_b in itertools.combinations(comparable, 2):
            sub_a = tdf[tdf["group"] == grp_a]
            sub_b = tdf[tdf["group"] == grp_b]
            frames_a, times_a = _frame_times(sub_a)
            frames_b, times_b = _frame_times(sub_b)
            dummy_t = np.array([0.0, 1.0])
            metrics = compare_breakpoints(
                [int(x) for x in frames_a],
                [int(x) for x in frames_b],
                tolerance_frames,
                dummy_t,
            )
            metrics["mean_timing_offset_ps"] = _mean_time_offset_ps(
                frames_a,
                frames_b,
                times_a,
                times_b,
                float(metrics["mean_timing_offset_frames"]),
                tolerance_frames,
            )
            rows.append(
                {
                    "traj_id": tid,
                    "group_a": grp_a,
                    "group_b": grp_b,
                    "tolerance_frames": int(tolerance_frames),
                    **metrics,
                }
            )

    cmp = pd.DataFrame(rows)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_dir / "all_breakpoints.csv", index=False)
        cmp.to_csv(output_dir / "changepoint_timing_comparison.csv", index=False)
        if not cmp.empty:
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
        else:
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
        pairs.to_csv(output_dir / "cohort_timing_summary.csv", index=False)
        log_event(
            "info",
            (
                f"Wrote timing comparison ({len(cmp)} traj-pairs, "
                f"{len(coverage)} trajectories, groups={groups})"
            ),
            component="changepoint_timing_comparison",
        )
    return cmp


def _segment_stats(
    df: pd.DataFrame,
    s_start: int,
    s_end: int,
    summary_cols: list[str],
) -> dict[str, float]:
    """Mean and std of summary_cols over df rows [s_start, s_end)."""
    slice_df = df.iloc[s_start:s_end]
    out: dict[str, float] = {}
    for col in summary_cols:
        if col not in df.columns:
            continue
        vals = slice_df[col].to_numpy(dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue
        out[f"{col}_mean"] = float(np.mean(finite))
        out[f"{col}_std"] = float(np.std(finite))
    return out


def detect_trajectory_changepoints(
    df: pd.DataFrame,
    traj_id: str,
    config: ChangepointConfig,
) -> ChangepointTables:
    """Run changepoint detection on one features DataFrame.

    Takes a DataFrame (not a path) so callers can reuse a parsed CSV across
    multiple penalty values without re-reading disk.
    """
    n_signal = len(df)

    frame_arr = (
        df["frame"].to_numpy(dtype=np.int64)
        if "frame" in df.columns
        else np.arange(n_signal, dtype=np.int64)
    )
    time_arr = (
        df["time_ps"].to_numpy(dtype=np.float64)
        if "time_ps" in df.columns
        else np.arange(n_signal, dtype=np.float64)
    )

    col_map = group_column_map(
        df,
        config.groups,
        include_site_pairs=config.include_site_pairs_in_detection,
    )

    bkp_rows: list[dict] = []
    seg_rows: list[dict] = []
    group_bkps: dict[str, list[int]] = {}

    for grp in config.groups:
        with step(f"group={grp}"):
            cols = col_map[grp]
            signal, used_cols = build_signal_matrix(df, cols, normalize=config.normalize)

            if signal.shape[1] == 0:
                log_event(
                    "warning",
                    f"{traj_id}: group '{grp}' — no usable columns, skipping",
                    component="changepoint_feature_groups",
                )
                group_bkps[grp] = []
                continue

            log_event(
                "info",
                f"{traj_id}: group '{grp}' — {signal.shape[1]} cols, {n_signal} frames",
                component="changepoint_feature_groups",
            )

            effective_penalty = (
                config.penalty
                if config.penalty is not None or config.n_bkps is not None
                else auto_penalty(n_signal, config.normalize)
            )
            cp = detect_changepoints(
                signal,
                method=config.method,
                cost_model=config.cost_model,
                penalty=effective_penalty,
                n_bkps=config.n_bkps,
                min_size=config.min_size,
                jump=config.jump,
            )
            group_bkps[grp] = cp.breakpoints

            for idx, sig_idx in enumerate(cp.breakpoints):
                safe_idx = min(sig_idx, n_signal - 1)
                bkp_rows.append({
                    "traj_id": traj_id,
                    "group": grp,
                    "breakpoint_idx": idx,
                    "signal_index": sig_idx,
                    "frame": int(frame_arr[safe_idx]),
                    "time_ps": float(time_arr[safe_idx]),
                    "n_cols_used": len(used_cols),
                })

            summary_cols = summary_base_columns(grp, df=df)
            boundaries = [0] + cp.breakpoints + [n_signal]
            for seg_id, (s_start, s_end) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            ):
                start_frame = int(frame_arr[s_start])
                end_frame = int(frame_arr[min(s_end - 1, n_signal - 1)])
                stats = _segment_stats(df, s_start, s_end, summary_cols)
                row: dict = {
                    "traj_id": traj_id,
                    "group": grp,
                    "segment_id": seg_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_ps": float(time_arr[s_start]),
                    "end_ps": float(time_arr[min(s_end - 1, n_signal - 1)]),
                    "n_frames": s_end - s_start,
                    "n_cols_used": len(used_cols),
                }
                row.update(stats)
                seg_rows.append(row)

    cmp_rows: list[dict] = []
    for grp_a, grp_b in itertools.combinations(config.groups, 2):
        if grp_a not in group_bkps or grp_b not in group_bkps:
            continue
        metrics = compare_breakpoints(
            group_bkps[grp_a], group_bkps[grp_b], config.tolerance_frames, time_arr
        )
        cmp_rows.append({
            "traj_id": traj_id,
            "group_a": grp_a,
            "group_b": grp_b,
            "tolerance_frames": config.tolerance_frames,
            **metrics,
        })

    return ChangepointTables(
        breakpoints=pd.DataFrame(bkp_rows),
        segment_stats=pd.DataFrame(seg_rows),
        comparison=pd.DataFrame(cmp_rows),
    )


def detect_trajectory_changepoints_from_csv(
    csv_path: Path,
    config: ChangepointConfig,
) -> ChangepointTables:
    """Load a features CSV and run detect_trajectory_changepoints."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    traj_id = traj_id_from_features_stem(csv_path.stem)
    return detect_trajectory_changepoints(df, traj_id, config)


def detect_cohort_changepoints(
    csv_paths: Sequence[Path],
    config: ChangepointConfig,
    *,
    output_dir: Optional[Path] = None,
) -> ChangepointTables:
    """Run changepoint detection on all feature CSVs and merge results.

    If *output_dir* is set, also writes per-trajectory and merged CSVs.
    """
    all_bkp: list[pd.DataFrame] = []
    all_seg: list[pd.DataFrame] = []
    all_cmp: list[pd.DataFrame] = []

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        traj_label = csv_path.stem
        with step(f"trajectory={traj_label}"):
            log_event(
                "info",
                f"Processing {csv_path.name}",
                component="changepoint_feature_groups",
            )
            tables = detect_trajectory_changepoints_from_csv(csv_path, config)
            traj_id = traj_id_from_features_stem(traj_label)

            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                bkp_out = output_dir / f"{traj_id}_breakpoints.csv"
                seg_out = output_dir / f"{traj_id}_segment_stats.csv"
                tables.breakpoints.to_csv(bkp_out, index=False)
                tables.segment_stats.to_csv(seg_out, index=False)
                log_event(
                    "info",
                    f"  → {bkp_out.name} ({len(tables.breakpoints)} breakpoints), "
                    f"{seg_out.name} ({len(tables.segment_stats)} segments)",
                    component="changepoint_feature_groups",
                )

            all_bkp.append(tables.breakpoints)
            all_seg.append(tables.segment_stats)
            all_cmp.append(tables.comparison)

    merged = ChangepointTables(
        breakpoints=pd.concat(all_bkp, ignore_index=True) if all_bkp else pd.DataFrame(),
        segment_stats=pd.concat(all_seg, ignore_index=True) if all_seg else pd.DataFrame(),
        comparison=pd.concat(all_cmp, ignore_index=True) if all_cmp else pd.DataFrame(),
    )

    if output_dir is not None:
        write_changepoint_tables(merged, Path(output_dir), per_trajectory=False)

    return merged


def write_changepoint_tables(
    tables: ChangepointTables,
    output_dir: Path,
    *,
    per_trajectory: bool = True,
) -> dict[str, Path]:
    """Write merged (and optionally per-trajectory) changepoint CSVs.

    Returns a dict of label -> path for the files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if per_trajectory and not tables.breakpoints.empty:
        for traj_id, bkp_df in tables.breakpoints.groupby("traj_id"):
            path = output_dir / f"{traj_id}_breakpoints.csv"
            bkp_df.to_csv(path, index=False)
            written[f"{traj_id}_breakpoints"] = path
        if not tables.segment_stats.empty:
            for traj_id, seg_df in tables.segment_stats.groupby("traj_id"):
                path = output_dir / f"{traj_id}_segment_stats.csv"
                seg_df.to_csv(path, index=False)
                written[f"{traj_id}_segment_stats"] = path

    bkp_path = output_dir / "all_breakpoints.csv"
    seg_path = output_dir / "all_segment_stats.csv"
    cmp_path = output_dir / "changepoint_timing_comparison.csv"
    tables.breakpoints.to_csv(bkp_path, index=False)
    tables.segment_stats.to_csv(seg_path, index=False)
    if tables.comparison.empty:
        # Single-group runs have no pairwise comparisons; keep a header-only file.
        pd.DataFrame(
            columns=[
                "traj_id",
                "group_a",
                "group_b",
                "tolerance_frames",
                "n_bkps_a",
                "n_bkps_b",
                "n_shared",
                "jaccard",
                "mean_timing_offset_frames",
                "mean_timing_offset_ps",
            ]
        ).to_csv(cmp_path, index=False)
    else:
        tables.comparison.to_csv(cmp_path, index=False)
    written["all_breakpoints"] = bkp_path
    written["all_segment_stats"] = seg_path
    written["changepoint_timing_comparison"] = cmp_path

    log_event(
        "info",
        f"Wrote all_breakpoints.csv ({len(tables.breakpoints)} rows), "
        f"all_segment_stats.csv ({len(tables.segment_stats)} rows), "
        f"changepoint_timing_comparison.csv ({len(tables.comparison)} rows)",
        component="changepoint_feature_groups",
    )
    return written


def discover_feature_csvs(
    features_dir: Path,
    *,
    suffix: str = "_gsa_features.csv",
) -> list[Path]:
    """Return sorted feature-CSV paths under *features_dir* matching *suffix*."""
    features_dir = Path(features_dir)
    if not suffix.startswith("*"):
        pattern = f"*{suffix}" if suffix.startswith("_") or suffix.startswith(".") else f"*_{suffix}"
    else:
        pattern = suffix
    if not pattern.endswith(".csv"):
        pattern = f"{pattern}.csv"
    return sorted(features_dir.glob(pattern))
