"""Penalty sensitivity sweep for changepoint feature-group detection."""

from __future__ import annotations

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.run_log import log_event, step

from .detection import (
    ChangepointConfig,
    ChangepointTables,
    auto_penalty,
    compare_breakpoints,
    detect_cohort_changepoints,
    detect_trajectory_changepoints,
)
from .feature_groups import DEFAULT_GROUPS, REGIME_COLS, traj_id_from_features_stem

# Legacy GSA cross-group pairs (used only when those groups are active).
_PAIR_LABELS = (
    ("gsa", "combined"),
    ("gsa", "iodine"),
    ("gsa", "na_water"),
    ("iodine", "na_water"),
)


def _group_pair_labels(groups: Sequence[str]) -> list[tuple[str, str]]:
    """Ordered group pairs for cross-group Jaccard columns in sweep summaries."""
    group_set = set(groups)
    # Prefer stable GSA ordering when multiple default groups are present.
    if group_set >= {"gsa", "combined", "iodine", "na_water"}:
        return [(a, b) for a, b in _PAIR_LABELS if a in group_set and b in group_set]
    return list(itertools.combinations(groups, 2))


@dataclass
class PenaltySweepConfig:
    """Parameters controlling the penalty sensitivity sweep."""

    n_penalties: int = 15
    penalty_min: Optional[float] = None
    penalty_max: Optional[float] = None
    reference_penalty: Optional[float] = None
    max_trajectories: Optional[int] = None
    timing_threshold: float = 0.75
    count_change_threshold: float = 0.35
    regime_threshold: float = 0.85
    min_plateau_steps: int = 3
    no_plots: bool = False
    n_jobs: int = 1
    detection: ChangepointConfig = field(default_factory=ChangepointConfig)


@dataclass
class PenaltySweepResult:
    """Tables and recommendation produced by a penalty sweep."""

    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_trajectory: pd.DataFrame = field(default_factory=pd.DataFrame)
    plateaus: pd.DataFrame = field(default_factory=pd.DataFrame)
    elbow: dict[str, float] = field(default_factory=dict)
    recommended_penalty: Optional[float] = None
    recommended_range: Optional[tuple[float, float]] = None
    reference_penalty: Optional[float] = None
    reference_grid_penalty: Optional[float] = None
    n_trajectories: int = 0
    penalty_grid: np.ndarray = field(default_factory=lambda: np.array([]))


def default_penalty_grid(n_signal: int, normalize: bool, n_points: int) -> np.ndarray:
    """Log-spaced penalties centred on the BIC-style default log(n)."""
    ref = auto_penalty(n_signal, normalize)
    if ref is None or ref <= 0:
        ref = float(np.log(max(n_signal, 2)))
    lo = max(ref * 0.2, 0.5)
    hi = ref * 5.0
    return np.unique(np.geomspace(lo, hi, num=n_points))


def _breakpoint_jaccard_vs_ref(
    bkps_a: list[int],
    bkps_b: list[int],
    tolerance: int,
) -> float:
    return compare_breakpoints(bkps_a, bkps_b, tolerance, np.arange(1, dtype=float))[
        "jaccard"
    ]


def regime_directions(seg_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Map (group, metric_col) -> sign of late-minus-early segment mean (-1, 0, +1)."""
    out: dict[tuple[str, str], int] = {}
    if seg_df.empty:
        return out
    for grp, cols in REGIME_COLS.items():
        sub = seg_df[seg_df["group"] == grp]
        if sub.empty:
            continue
        seg_ids = sorted(sub["segment_id"].unique())
        if len(seg_ids) < 2:
            continue
        early = sub[sub["segment_id"] == seg_ids[0]]
        late = sub[sub["segment_id"] == seg_ids[-1]]
        for col in cols:
            if col not in sub.columns:
                continue
            e_vals = early[col].dropna()
            l_vals = late[col].dropna()
            if e_vals.empty or l_vals.empty:
                continue
            delta = float(l_vals.mean() - e_vals.mean())
            if abs(delta) < 1e-9:
                sign = 0
            else:
                sign = 1 if delta > 0 else -1
            out[(grp, col)] = sign
    return out


def regime_agreement(
    ref_dirs: dict[tuple[str, str], int],
    dirs: dict[tuple[str, str], int],
) -> float:
    keys = set(ref_dirs) & set(dirs)
    if not keys:
        return float("nan")
    agree = sum(1 for k in keys if ref_dirs[k] == dirs[k])
    return agree / len(keys)


def _init_sweep_worker() -> None:
    """Cap BLAS threads in Pelt workers so process pools do not oversubscribe."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:
        pass


def _resolve_sweep_workers(n_jobs: int, n_traj: int) -> int:
    """Resolve ``n_jobs`` to a concrete worker count capped by ``n_traj``."""
    if n_traj <= 0:
        return 1
    if int(n_jobs) == -1:
        requested = os.cpu_count() or 1
    else:
        requested = max(1, int(n_jobs))
    return max(1, min(requested, n_traj))


def _sweep_one_trajectory(
    traj_id: str,
    df: pd.DataFrame,
    det_cfg: ChangepointConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, str], list[int]]]:
    """Process-pool worker: detect one trajectory at a fixed penalty."""
    tables = detect_trajectory_changepoints(df, traj_id, det_cfg)
    per_traj_bkps: dict[tuple[str, str], list[int]] = {}
    if not tables.breakpoints.empty:
        for (tid, grp), sub in tables.breakpoints.groupby(["traj_id", "group"]):
            per_traj_bkps[(str(tid), str(grp))] = sub["signal_index"].astype(int).tolist()
    return tables.breakpoints, tables.segment_stats, tables.comparison, per_traj_bkps


def _run_sweep_for_penalty(
    feature_dfs: list[tuple[str, pd.DataFrame]],
    *,
    penalty: float,
    config: ChangepointConfig,
    executor: Optional[ProcessPoolExecutor] = None,
) -> tuple[
    ChangepointTables,
    dict[tuple[str, str], list[int]],
    dict[tuple[str, str], int],
]:
    """Run all trajectories at one penalty; return merged tables + regime directions."""
    det_cfg = replace(config, penalty=penalty, n_bkps=None)
    if executor is None:
        parts = [
            _sweep_one_trajectory(traj_id, df, det_cfg) for traj_id, df in feature_dfs
        ]
    else:
        futures = [
            executor.submit(_sweep_one_trajectory, traj_id, df, det_cfg)
            for traj_id, df in feature_dfs
        ]
        parts = [fut.result() for fut in futures]

    all_bkp = [p[0] for p in parts]
    all_seg = [p[1] for p in parts]
    all_cmp = [p[2] for p in parts]
    per_traj_bkps: dict[tuple[str, str], list[int]] = {}
    for part in parts:
        per_traj_bkps.update(part[3])

    merged = ChangepointTables(
        breakpoints=pd.concat(all_bkp, ignore_index=True) if all_bkp else pd.DataFrame(),
        segment_stats=pd.concat(all_seg, ignore_index=True) if all_seg else pd.DataFrame(),
        comparison=pd.concat(all_cmp, ignore_index=True) if all_cmp else pd.DataFrame(),
    )
    return merged, per_traj_bkps, regime_directions(merged.segment_stats)


def _median_pair_jaccard(cmp_df: pd.DataFrame, grp_a: str, grp_b: str) -> float:
    if (
        cmp_df.empty
        or "group_a" not in cmp_df.columns
        or "group_b" not in cmp_df.columns
        or "jaccard" not in cmp_df.columns
    ):
        return float("nan")
    sub = cmp_df[
        ((cmp_df["group_a"] == grp_a) & (cmp_df["group_b"] == grp_b))
        | ((cmp_df["group_a"] == grp_b) & (cmp_df["group_b"] == grp_a))
    ]
    if sub.empty:
        return float("nan")
    return float(sub["jaccard"].median())


def annotate_count_change(summary: pd.DataFrame) -> pd.DataFrame:
    """Add step-to-step relative change in cohort breakpoint totals."""
    out = summary.copy()
    counts = out["total_breakpoints"].astype(float)
    out["count_rel_change"] = counts.pct_change().abs()
    return out


def find_penalty_elbow(summary: pd.DataFrame) -> dict[str, float]:
    """Elbow on log10(penalty) vs cohort total breakpoints.

    Uses the point farthest from the chord joining the first and last grid
    points (standard knee heuristic on a decreasing curve).
    """
    if len(summary) < 3:
        return {}

    penalties = summary["penalty"].to_numpy(dtype=float)
    counts = summary["total_breakpoints"].to_numpy(dtype=float)
    x = np.log10(penalties)
    y = counts

    x_norm = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
    y_norm = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y

    x0, y0 = x_norm[0], y_norm[0]
    x1, y1 = x_norm[-1], y_norm[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = np.hypot(dx, dy)
    if denom == 0:
        return {}

    distances = np.array(
        [
            abs(dy * x_norm[i] - dx * y_norm[i] + x1 * y0 - y1 * x0) / denom
            for i in range(len(x_norm))
        ]
    )
    interior = np.arange(1, len(distances) - 1)
    idx = int(interior[np.argmax(distances[interior])])

    return {
        "elbow_penalty": float(penalties[idx]),
        "elbow_total_breakpoints": float(counts[idx]),
        "elbow_score": float(distances[idx]),
        "elbow_grid_index": float(idx),
    }


def find_stable_plateaus(
    summary: pd.DataFrame,
    *,
    min_plateau_steps: int,
    timing_threshold: float,
    regime_threshold: float,
    count_change_threshold: float,
) -> pd.DataFrame:
    """Optional robust band: high timing stability and slow count drift."""
    if len(summary) < min_plateau_steps:
        return pd.DataFrame()

    df = annotate_count_change(summary)
    timing_ok = df["timing_jaccard_vs_prev"].fillna(1.0) >= timing_threshold
    count_ok = df["count_rel_change"].fillna(0.0) <= count_change_threshold
    stable = timing_ok & count_ok

    rows: list[dict] = []
    n = len(df)
    i = 0
    while i < n:
        if not stable.iloc[i]:
            i += 1
            continue
        j = i + 1
        while j < n and stable.iloc[j]:
            j += 1
        length = j - i
        if length >= min_plateau_steps:
            block = df.iloc[i:j]
            jaccard_cols = [c for c in block.columns if c.startswith("median_jaccard_")]
            row = {
                "penalty_min": float(block["penalty"].iloc[0]),
                "penalty_max": float(block["penalty"].iloc[-1]),
                "penalty_mid": float(
                    np.sqrt(block["penalty"].iloc[0] * block["penalty"].iloc[-1])
                ),
                "n_penalty_steps": length,
                "total_breakpoints_min": int(block["total_breakpoints"].min()),
                "total_breakpoints_max": int(block["total_breakpoints"].max()),
                "median_timing_jaccard": float(
                    block["timing_jaccard_vs_prev"].median()
                ),
                "median_regime_agreement": float(
                    block["regime_agreement_vs_ref"].median()
                ),
                "median_count_rel_change": float(
                    block["count_rel_change"].median()
                ),
                "passes_regime_threshold": bool(
                    (block["regime_agreement_vs_ref"] >= regime_threshold).all()
                ),
            }
            if jaccard_cols:
                row["median_jaccard_gsa_combined"] = float(
                    block[jaccard_cols[0]].median()
                )
            rows.append(row)
        i = j if j > i else i + 1

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    regime_weight = out["median_regime_agreement"].fillna(0).clip(0, 1)
    out["stability_score"] = (
        out["n_penalty_steps"]
        * out["median_timing_jaccard"].fillna(0)
        * (0.5 + 0.5 * regime_weight)
    )
    return out.sort_values("stability_score", ascending=False)


def _sweep_overview_panels(summary: pd.DataFrame) -> list[tuple[str, str]]:
    """Return (column, ylabel) for the 2×2 penalty sweep overview."""
    panels: list[tuple[str, str]] = [
        ("total_breakpoints", "Total breakpoints (cohort)"),
        ("timing_jaccard_vs_prev", "Timing Jaccard vs previous penalty"),
        ("regime_agreement_vs_ref", "Regime direction agreement vs reference"),
    ]
    jaccard_cols = [c for c in summary.columns if c.startswith("median_jaccard_")]
    if jaccard_cols:
        col = jaccard_cols[0]
        pair = col.replace("median_jaccard_", "").replace("_", " ↔ ")
        panels.append((col, f"Median Jaccard ({pair})"))
    else:
        count_cols = [c for c in summary.columns if c.startswith("n_bkps_")]
        if count_cols:
            col = count_cols[0]
            grp = col.replace("n_bkps_", "")
            panels.append((col, f"Breakpoint count ({grp})"))
        else:
            panels.append(
                ("timing_jaccard_vs_prev", "Timing Jaccard vs previous penalty")
            )
    return panels


def plot_sweep(
    summary: pd.DataFrame,
    plot_dir: Path,
    *,
    elbow: Optional[dict[str, float]] = None,
) -> list[Path]:
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    x = summary["penalty"]
    for ax, (col, label) in zip(axes.ravel(), _sweep_overview_panels(summary)):
        y = summary[col]
        ax.plot(x, y, "o-", lw=1.2, ms=4)
        ax.set_xscale("log")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        if col == "total_breakpoints":
            for val in y.unique():
                ax.axhline(val, color="0.85", lw=0.8, zorder=0)
            if elbow and "elbow_penalty" in elbow:
                ep = elbow["elbow_penalty"]
                eb = elbow.get("elbow_total_breakpoints")
                ax.axvline(ep, color="#d62728", ls="--", lw=1.2, alpha=0.85, label="elbow")
                if eb is not None:
                    ax.scatter([ep], [eb], color="#d62728", s=60, zorder=5)
                ax.legend(loc="upper right", fontsize=8)
    axes[1, 0].set_xlabel("Penalty")
    axes[1, 1].set_xlabel("Penalty")
    fig.suptitle("Changepoint penalty sensitivity")
    fig.tight_layout()
    p = plot_dir / "penalty_sweep_overview.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    count_cols = [c for c in summary.columns if c.startswith("n_bkps_")]
    if count_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        for col in count_cols:
            grp = col.replace("n_bkps_", "")
            ax.plot(x, summary[col], "o-", ms=3, label=grp)
        ax.set_xscale("log")
        ax.set_xlabel("Penalty")
        ax.set_ylabel("Cohort breakpoint count")
        ax.legend(ncol=4, fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p2 = plot_dir / "penalty_sweep_counts_by_group.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        written.append(p2)

    return written


def _build_recommendation_text(result: PenaltySweepResult) -> str:
    penalties = result.penalty_grid
    lines = [
        "Changepoint penalty sweep recommendation",
        "=" * 42,
        f"Trajectories: {result.n_trajectories}",
        f"Reference penalty (log(n)): {result.reference_penalty:.6g}",
        f"Regime reference (nearest grid step): {result.reference_grid_penalty:.6g}",
        f"Penalty grid: {penalties[0]:.6g} … {penalties[-1]:.6g} ({len(penalties)} steps)",
        "",
    ]
    if result.elbow:
        lines.extend(
            [
                "Primary selection: elbow on total breakpoints vs penalty",
                f"  Recommended penalty: {result.elbow['elbow_penalty']:.6g}",
                f"  Cohort breakpoints at elbow: {int(result.elbow['elbow_total_breakpoints'])}",
                "",
                "The elbow is where further increasing the penalty yields diminishing",
                "reduction in breakpoint count (bias–variance knee on the sweep curve).",
            ]
        )
    if result.recommended_range is not None:
        rec_range = result.recommended_range
        lines.extend(
            [
                "",
                f"Optional robust band (timing-stable, slow count drift): "
                f"[{rec_range[0]:.6g}, {rec_range[1]:.6g}]",
                f"  Band midpoint: {float(np.sqrt(rec_range[0] * rec_range[1])):.6g}",
            ]
        )
    if not result.elbow and result.recommended_penalty is not None:
        lines.append(f"Recommended penalty (fallback): {result.recommended_penalty:.6g}")
    lines.extend(
        [
            "",
            "Cross-group Jaccard (gsa↔combined, gsa↔iodine, …) is reported in",
            "penalty_sweep_summary.csv for interpretation only: each feature group",
            "is segmented independently, so agreement checks whether cage, guest,",
            "and solvent changepoints coincide — it does not define the penalty.",
        ]
    )
    if not result.plateaus.empty:
        lines.extend(["", "Robust bands (ranked):", result.plateaus.to_string(index=False)])
    return "\n".join(lines) + "\n"


def write_sweep_artifacts(
    result: PenaltySweepResult,
    output_dir: Path,
    *,
    no_plots: bool = False,
) -> dict[str, Path]:
    """Write sweep CSVs, recommendation text, and optional plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    summary_path = output_dir / "penalty_sweep_summary.csv"
    detail_path = output_dir / "penalty_sweep_by_trajectory.csv"
    plateaus_path = output_dir / "penalty_stable_ranges.csv"
    elbow_path = output_dir / "penalty_elbow.csv"
    rec_path = output_dir / "penalty_recommendation.txt"

    result.summary.to_csv(summary_path, index=False)
    result.by_trajectory.to_csv(detail_path, index=False)
    result.plateaus.to_csv(plateaus_path, index=False)
    if result.elbow:
        pd.DataFrame([result.elbow]).to_csv(elbow_path, index=False)
    else:
        pd.DataFrame(columns=["elbow_penalty"]).to_csv(elbow_path, index=False)
    rec_path.write_text(_build_recommendation_text(result), encoding="utf-8")

    written["penalty_sweep_summary"] = summary_path
    written["penalty_sweep_by_trajectory"] = detail_path
    written["penalty_stable_ranges"] = plateaus_path
    written["penalty_elbow"] = elbow_path
    written["penalty_recommendation"] = rec_path

    if not no_plots and not result.summary.empty:
        for path in plot_sweep(result.summary, output_dir / "plots", elbow=result.elbow or None):
            written[path.name] = path

    log_event(
        "info",
        f"Wrote {summary_path.name}, {detail_path.name}, {plateaus_path.name}, "
        f"{elbow_path.name}, {rec_path.name}",
        component="sweep_changepoint_penalty",
    )
    return written


def sweep_penalties(
    csv_paths: Sequence[Path],
    config: PenaltySweepConfig,
    *,
    output_dir: Optional[Path] = None,
) -> PenaltySweepResult:
    """Sweep Pelt penalties and recommend a structurally stable value."""
    csv_paths = [Path(p) for p in csv_paths]
    if config.max_trajectories is not None:
        csv_paths = csv_paths[: config.max_trajectories]

    feature_dfs: list[tuple[str, pd.DataFrame]] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        traj_id = traj_id_from_features_stem(path.stem)
        feature_dfs.append((traj_id, df))

    lengths = [len(df) for _, df in feature_dfs]
    n_signal = int(np.median(lengths)) if lengths else 2
    det = config.detection
    n_jobs = _resolve_sweep_workers(config.n_jobs, len(feature_dfs))

    if config.penalty_min is not None and config.penalty_max is not None:
        penalties = np.geomspace(
            config.penalty_min, config.penalty_max, num=config.n_penalties
        )
    else:
        penalties = default_penalty_grid(
            n_signal, normalize=det.normalize, n_points=config.n_penalties
        )
        if config.penalty_min is not None:
            penalties = penalties[penalties >= config.penalty_min]
        if config.penalty_max is not None:
            penalties = penalties[penalties <= config.penalty_max]

    ref_penalty = config.reference_penalty
    if ref_penalty is None:
        ref_penalty = auto_penalty(n_signal, normalize=det.normalize)
    if ref_penalty is None:
        ref_penalty = float(np.log(max(n_signal, 2)))

    log_event(
        "info",
        f"Sweeping {len(penalties)} penalties on {len(feature_dfs)} trajectories; "
        f"reference penalty={ref_penalty:.4g}; n_jobs={n_jobs}; "
        f"median n={n_signal}",
        component="sweep_changepoint_penalty",
    )

    regime_dirs_by_penalty: dict[float, dict[tuple[str, str], int]] = {}
    summary_rows: list[dict] = []
    detail_rows: list[dict] = []
    prev_per_traj_bkps: Optional[dict[tuple[str, str], list[int]]] = None
    groups = list(det.groups) if det.groups else list(DEFAULT_GROUPS)
    pair_labels = _group_pair_labels(groups)

    executor: Optional[ProcessPoolExecutor] = (
        ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_sweep_worker)
        if n_jobs > 1
        else None
    )
    try:
        for pen in penalties:
            with step(f"penalty={pen:.4g}"):
                tables, per_traj_bkps, regime_dirs = _run_sweep_for_penalty(
                    feature_dfs,
                    penalty=float(pen),
                    config=det,
                    executor=executor,
                )
            regime_dirs_by_penalty[float(pen)] = regime_dirs

            timing_jaccards: list[float] = []
            if prev_per_traj_bkps is not None:
                keys = set(per_traj_bkps) | set(prev_per_traj_bkps)
                for key in keys:
                    a = per_traj_bkps.get(key, [])
                    b = prev_per_traj_bkps.get(key, [])
                    timing_jaccards.append(
                        _breakpoint_jaccard_vs_ref(a, b, det.tolerance_frames)
                    )
            prev_per_traj_bkps = per_traj_bkps

            bkp_df = tables.breakpoints
            cmp_df = tables.comparison
            row: dict = {
                "penalty": float(pen),
                "n_trajectories": len(feature_dfs),
                "total_breakpoints": len(bkp_df),
                "timing_jaccard_vs_prev": (
                    float(np.mean(timing_jaccards)) if timing_jaccards else float("nan")
                ),
            }
            for grp in groups:
                row[f"n_bkps_{grp}"] = int(
                    len(bkp_df[bkp_df["group"] == grp]) if not bkp_df.empty else 0
                )
            for grp_a, grp_b in pair_labels:
                row[f"median_jaccard_{grp_a}_{grp_b}"] = _median_pair_jaccard(
                    cmp_df, grp_a, grp_b
                )
            summary_rows.append(row)

            if not bkp_df.empty:
                counts = (
                    bkp_df.groupby(["traj_id", "group"])
                    .size()
                    .reset_index(name="n_bkps")
                )
                for _, r in counts.iterrows():
                    detail_rows.append(
                        {
                            "penalty": float(pen),
                            "traj_id": r["traj_id"],
                            "group": r["group"],
                            "n_bkps": int(r["n_bkps"]),
                        }
                    )

    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    summary = pd.DataFrame(summary_rows)
    ref_pen_key = float(
        summary.loc[(summary["penalty"] - ref_penalty).abs().idxmin(), "penalty"]
    )
    ref_regime_dirs = regime_dirs_by_penalty[ref_pen_key]
    summary["regime_agreement_vs_ref"] = summary["penalty"].map(
        lambda p: regime_agreement(ref_regime_dirs, regime_dirs_by_penalty[float(p)])
    )
    summary = annotate_count_change(summary)
    detail = pd.DataFrame(detail_rows)
    plateaus = find_stable_plateaus(
        summary,
        min_plateau_steps=config.min_plateau_steps,
        timing_threshold=config.timing_threshold,
        regime_threshold=config.regime_threshold,
        count_change_threshold=config.count_change_threshold,
    )
    elbow = find_penalty_elbow(summary)

    rec_penalty: Optional[float] = None
    rec_range: Optional[tuple[float, float]] = None
    if elbow:
        rec_penalty = float(elbow["elbow_penalty"])
    elif not plateaus.empty:
        best = plateaus.iloc[0]
        rec_penalty = float(best["penalty_mid"])
        rec_range = (float(best["penalty_min"]), float(best["penalty_max"]))
    else:
        sub = summary.copy()
        sub["ref_dist"] = (sub["penalty"] - ref_penalty).abs()
        sub = sub.sort_values(
            ["ref_dist", "regime_agreement_vs_ref"], ascending=[True, False]
        )
        rec_penalty = float(sub.iloc[0]["penalty"])
        log_event(
            "warning",
            "No elbow or stable band found; falling back to penalty nearest "
            f"log(n) reference ({rec_penalty:.4g})",
            component="sweep_changepoint_penalty",
        )

    if not plateaus.empty and rec_range is None:
        best = plateaus.iloc[0]
        rec_range = (float(best["penalty_min"]), float(best["penalty_max"]))

    result = PenaltySweepResult(
        summary=summary,
        by_trajectory=detail,
        plateaus=plateaus,
        elbow=elbow,
        recommended_penalty=rec_penalty,
        recommended_range=rec_range,
        reference_penalty=float(ref_penalty),
        reference_grid_penalty=ref_pen_key,
        n_trajectories=len(feature_dfs),
        penalty_grid=np.asarray(penalties, dtype=float),
    )

    if output_dir is not None:
        write_sweep_artifacts(result, Path(output_dir), no_plots=config.no_plots)

    return result


def run_final_detection(
    csv_paths: Sequence[Path],
    *,
    penalty: float,
    detection: ChangepointConfig,
    output_dir: Path,
) -> ChangepointTables:
    """Re-run cohort detection at the recommended penalty (in-process)."""
    cfg = replace(detection, penalty=penalty, n_bkps=None)
    log_event(
        "info",
        f"Running changepoint detection at penalty={penalty:.4g}",
        component="sweep_changepoint_penalty",
    )
    return detect_cohort_changepoints(csv_paths, cfg, output_dir=Path(output_dir))
