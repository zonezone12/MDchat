"""
Penalty sensitivity sweep for changepoint_feature_groups.

Chooses a penalty by structural stability, not a single heuristic value.
For each penalty in a grid, reuses ``changepoint_feature_groups.process_csv`` and
scores:

  * breakpoint-count plateaus (cohort totals and per-trajectory stability)
  * timing stability (Jaccard vs the previous penalty step)
  * cross-group timing agreement (median Jaccard gsa↔combined, gsa↔iodine, …)
  * segment-regime consistency (do headline segment trends keep the same sign?)

Writes CSV summaries and optional plots under --output-dir, then prints a
recommended penalty from the **elbow** of total breakpoints vs penalty
(max distance from the chord on log-penalty). Cross-group Jaccard and timing
metrics are recorded as diagnostics for interpreting the four parallel group
runs, not as primary selection criteria. Optional stable-range bands (slow
count drift + high timing Jaccard) are listed when found.

Example
-------
python scripts/sweep_changepoint_penalty.py \\
    --input-dir output/gsa_features \\
    --output-dir output/changepoints/penalty_sweep \\
    --n-penalties 17

python scripts/sweep_changepoint_penalty.py \\
    --input-dir output/gsa_features \\
    --output-dir output/changepoints/penalty_sweep \\
    --method Pelt --cost-model rbf --min-size 10 --jump 5 --tolerance-frames 50 \\
    --run-final --final-output-dir output/changepoints
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(_SCRIPTS))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from changepoint_feature_groups import (
    _auto_penalty,
    _compare_breakpoints,
    process_csv,
)
from src.utils.run_log import RunContext, log_event, step

# Detection settings forwarded to changepoint_feature_groups (defaults match that script).
DEFAULT_METHOD = "Pelt"
DEFAULT_COST_MODEL = "rbf"
DEFAULT_MIN_SIZE = 10
DEFAULT_JUMP = 5
DEFAULT_TOLERANCE_FRAMES = 50
DEFAULT_FEATURE_GROUPS = ["gsa", "iodine", "na_water", "combined"]

# Headline segment metrics for regime-direction checks (early vs late segments).
_REGIME_COLS: dict[str, list[str]] = {
    "gsa": [
        "assembly_rg_mean",
        "assembly_rmsd_to_ref_mean",
        "octahedrality_score_mean",
        "total_inter_monomer_contacts_mean",
    ],
    "iodine": [
        "n_guest_inside_cavity_mean",
        "n_guest_bulk_mean",
        "guest_radial_distance_mean_mean",
        "guest_contacted_monomer_count_mean",
    ],
    "na_water": [
        "cavity_water_count_mean",
        "cavity_ion_count_mean",
        "guest_water_contact_count_mean",
    ],
    "combined": [
        "assembly_rg_mean",
        "n_guest_inside_cavity_mean",
        "cavity_ion_count_mean",
    ],
}

_PAIR_LABELS = (
    ("gsa", "combined"),
    ("gsa", "iodine"),
    ("gsa", "na_water"),
    ("iodine", "na_water"),
)


def _default_penalty_grid(n_signal: int, normalize: bool, n_points: int) -> np.ndarray:
    """Log-spaced penalties centred on the BIC-style default log(n)."""
    ref = _auto_penalty(n_signal, normalize)
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
    return _compare_breakpoints(bkps_a, bkps_b, tolerance, np.arange(1, dtype=float))[
        "jaccard"
    ]


def _regime_directions(seg_df: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Map (group, metric_col) -> sign of late-minus-early segment mean (-1, 0, +1)."""
    out: dict[tuple[str, str], int] = {}
    if seg_df.empty:
        return out
    for grp, cols in _REGIME_COLS.items():
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


def _regime_agreement(
    ref_dirs: dict[tuple[str, str], int],
    dirs: dict[tuple[str, str], int],
) -> float:
    keys = set(ref_dirs) & set(dirs)
    if not keys:
        return float("nan")
    agree = sum(1 for k in keys if ref_dirs[k] == dirs[k])
    return agree / len(keys)


def _run_sweep_for_penalty(
    csv_files: list[Path],
    *,
    penalty: float,
    method: str,
    cost_model: str,
    min_size: int,
    jump: int,
    tolerance_frames: int,
    groups: list[str],
    normalize: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str], list[int]],
    dict[tuple[str, str], int],
]:
    """Run all trajectories at one penalty; return merged tables + regime directions."""
    all_bkp: list[pd.DataFrame] = []
    all_seg: list[pd.DataFrame] = []
    all_cmp: list[pd.DataFrame] = []
    per_traj_bkps: dict[tuple[str, str], list[int]] = {}

    for csv_path in csv_files:
        bkp_df, seg_df, cmp_df = process_csv(
            csv_path,
            method=method,
            cost_model=cost_model,
            penalty=penalty,
            n_bkps=None,
            min_size=min_size,
            jump=jump,
            tolerance_frames=tolerance_frames,
            groups=groups,
            normalize=normalize,
        )
        all_bkp.append(bkp_df)
        all_seg.append(seg_df)
        all_cmp.append(cmp_df)
        if not bkp_df.empty:
            for (traj_id, grp), sub in bkp_df.groupby(["traj_id", "group"]):
                per_traj_bkps[(str(traj_id), str(grp))] = sub["signal_index"].astype(
                    int
                ).tolist()

    merged_bkp = pd.concat(all_bkp, ignore_index=True) if all_bkp else pd.DataFrame()
    merged_seg = pd.concat(all_seg, ignore_index=True) if all_seg else pd.DataFrame()
    merged_cmp = pd.concat(all_cmp, ignore_index=True) if all_cmp else pd.DataFrame()
    regime_dirs = _regime_directions(merged_seg)
    return merged_bkp, merged_seg, merged_cmp, per_traj_bkps, regime_dirs  # type: ignore[return-value]


def _median_pair_jaccard(cmp_df: pd.DataFrame, grp_a: str, grp_b: str) -> float:
    sub = cmp_df[
        ((cmp_df["group_a"] == grp_a) & (cmp_df["group_b"] == grp_b))
        | ((cmp_df["group_a"] == grp_b) & (cmp_df["group_b"] == grp_a))
    ]
    if sub.empty:
        return float("nan")
    return float(sub["jaccard"].median())


def _annotate_count_change(summary: pd.DataFrame) -> pd.DataFrame:
    """Add step-to-step relative change in cohort breakpoint totals."""
    out = summary.copy()
    counts = out["total_breakpoints"].astype(float)
    out["count_rel_change"] = counts.pct_change().abs()
    return out


def _find_penalty_elbow(summary: pd.DataFrame) -> dict[str, float]:
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
    # Prefer an interior knee; endpoints are artifacts of the grid.
    interior = np.arange(1, len(distances) - 1)
    idx = int(interior[np.argmax(distances[interior])])

    return {
        "elbow_penalty": float(penalties[idx]),
        "elbow_total_breakpoints": float(counts[idx]),
        "elbow_score": float(distances[idx]),
        "elbow_grid_index": float(idx),
    }


def _find_stable_plateaus(
    summary: pd.DataFrame,
    *,
    min_plateau_steps: int,
    timing_threshold: float,
    regime_threshold: float,
    count_change_threshold: float,
) -> pd.DataFrame:
    """Optional robust band: high timing stability and slow count drift.

    Cross-group Jaccard is reported per range but not required to pass.
    Primary penalty selection uses the breakpoint-count elbow instead.
    """
    if len(summary) < min_plateau_steps:
        return pd.DataFrame()

    df = _annotate_count_change(summary)
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
            rows.append(
                {
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
                    "median_jaccard_gsa_combined": float(
                        block["median_jaccard_gsa_combined"].median()
                    ),
                    "median_count_rel_change": float(
                        block["count_rel_change"].median()
                    ),
                    "passes_regime_threshold": bool(
                        (block["regime_agreement_vs_ref"] >= regime_threshold).all()
                    ),
                }
            )
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


def _plot_sweep(
    summary: pd.DataFrame,
    plot_dir: Path,
    *,
    elbow: Optional[dict[str, float]] = None,
) -> list[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    x = summary["penalty"]
    for ax, col, label in zip(
        axes.ravel(),
        [
            "total_breakpoints",
            "timing_jaccard_vs_prev",
            "regime_agreement_vs_ref",
            "median_jaccard_gsa_combined",
        ],
        [
            "Total breakpoints (cohort)",
            "Timing Jaccard vs previous penalty",
            "Regime direction agreement vs reference",
            "Median Jaccard (gsa ↔ combined)",
        ],
    ):
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


def _invoke_changepoint_feature_groups(
    *,
    input_dir: Path,
    output_dir: Path,
    penalty: float,
    method: str = DEFAULT_METHOD,
    cost_model: str = DEFAULT_COST_MODEL,
    min_size: int = DEFAULT_MIN_SIZE,
    jump: int = DEFAULT_JUMP,
    tolerance_frames: int = DEFAULT_TOLERANCE_FRAMES,
) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "changepoint_feature_groups.py"),
        "--input-dir",
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--method",
        method,
        "--cost-model",
        cost_model,
        "--penalty",
        str(penalty),
        "--min-size",
        str(min_size),
        "--jump",
        str(jump),
        "--tolerance-frames",
        str(tolerance_frames),
    ]
    log_event(
        "info",
        f"Running changepoint_feature_groups.py at penalty={penalty:.4g}",
        component="sweep_changepoint_penalty",
    )
    return subprocess.call(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep Pelt penalty values and recommend a structurally stable range "
            "for changepoint_feature_groups."
        )
    )
    p.add_argument(
        "--input-dir",
        default="output/gsa_features",
        help="Directory with *_gsa_features.csv files",
    )
    p.add_argument(
        "--output-dir",
        default="output/changepoints/penalty_sweep",
        help="Sweep summaries and plots (default: output/changepoints/penalty_sweep)",
    )
    p.add_argument(
        "--method",
        default=DEFAULT_METHOD,
        help="Ruptures search method (default: Pelt)",
    )
    p.add_argument(
        "--cost-model",
        default=DEFAULT_COST_MODEL,
        help="Ruptures cost function (default: rbf)",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=DEFAULT_MIN_SIZE,
        help="Minimum segment length in frames (default: 10)",
    )
    p.add_argument(
        "--jump",
        type=int,
        default=DEFAULT_JUMP,
        help="Ruptures subsample step for speed (default: 5)",
    )
    p.add_argument(
        "--tolerance-frames",
        type=int,
        default=DEFAULT_TOLERANCE_FRAMES,
        help=(
            "Frame tolerance for timing comparison; breakpoints within ±N are "
            "'shared' (default: 50)"
        ),
    )
    p.add_argument(
        "--n-penalties",
        type=int,
        default=15,
        help="Number of log-spaced penalty values (default: 15)",
    )
    p.add_argument(
        "--penalty-min",
        type=float,
        default=None,
        help="Override sweep lower bound (default: ~0.2 × log(n))",
    )
    p.add_argument(
        "--penalty-max",
        type=float,
        default=None,
        help="Override sweep upper bound (default: ~5 × log(n))",
    )
    p.add_argument(
        "--reference-penalty",
        type=float,
        default=None,
        help="Penalty for regime-direction reference (default: log(n) auto)",
    )
    p.add_argument(
        "--max-trajectories",
        type=int,
        default=None,
        help="Limit trajectories for a faster exploratory sweep",
    )
    p.add_argument(
        "--timing-threshold",
        type=float,
        default=0.75,
        help="Min timing Jaccard vs previous step inside a stable range (default: 0.75)",
    )
    p.add_argument(
        "--count-change-threshold",
        type=float,
        default=0.35,
        help=(
            "Max step-to-step relative change in cohort breakpoint totals "
            "inside a stable range (default: 0.35)"
        ),
    )
    p.add_argument(
        "--regime-threshold",
        type=float,
        default=0.85,
        help=(
            "Regime-direction agreement threshold (reported as passes_regime_threshold; "
            "default: 0.85)"
        ),
    )
    p.add_argument(
        "--min-plateau-steps",
        type=int,
        default=3,
        help="Minimum consecutive structurally stable penalty steps (default: 3)",
    )
    p.add_argument(
        "--run-final",
        action="store_true",
        help="After sweep, call changepoint_feature_groups.py at recommended penalty",
    )
    p.add_argument(
        "--final-output-dir",
        default="output/changepoints",
        help="Output dir for --run-final (default: output/changepoints)",
    )
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*_gsa_features.csv"))
    if not csv_files:
        print(f"No *_gsa_features.csv in {input_dir}", file=sys.stderr)
        sys.exit(1)
    if args.max_trajectories is not None:
        csv_files = csv_files[: args.max_trajectories]

    n_signal = len(pd.read_csv(csv_files[0]))

    if args.penalty_min is not None and args.penalty_max is not None:
        penalties = np.geomspace(args.penalty_min, args.penalty_max, num=args.n_penalties)
    else:
        penalties = _default_penalty_grid(n_signal, normalize=True, n_points=args.n_penalties)
        if args.penalty_min is not None:
            penalties = penalties[penalties >= args.penalty_min]
        if args.penalty_max is not None:
            penalties = penalties[penalties <= args.penalty_max]

    ref_penalty = args.reference_penalty
    if ref_penalty is None:
        ref_penalty = _auto_penalty(n_signal, normalize=True)
    if ref_penalty is None:
        ref_penalty = float(np.log(max(n_signal, 2)))

    with RunContext.from_namespace(args, name="sweep_changepoint_penalty"):
        log_event(
            "info",
            f"Sweeping {len(penalties)} penalties on {len(csv_files)} trajectories; "
            f"reference penalty={ref_penalty:.4g}",
            component="sweep_changepoint_penalty",
        )

        regime_dirs_by_penalty: dict[float, dict[tuple[str, str], int]] = {}
        summary_rows: list[dict] = []
        detail_rows: list[dict] = []
        prev_per_traj_bkps: Optional[dict[tuple[str, str], list[int]]] = None

        for pen in penalties:
            with step(f"penalty={pen:.4g}"):
                bkp_df, seg_df, cmp_df, per_traj_bkps, regime_dirs = _run_sweep_for_penalty(
                    csv_files,
                    penalty=float(pen),
                    method=args.method,
                    cost_model=args.cost_model,
                    min_size=args.min_size,
                    jump=args.jump,
                    tolerance_frames=args.tolerance_frames,
                    groups=DEFAULT_FEATURE_GROUPS,
                    normalize=True,
                )
                regime_dirs_by_penalty[float(pen)] = regime_dirs

                timing_jaccards: list[float] = []
                if prev_per_traj_bkps is not None:
                    keys = set(per_traj_bkps) | set(prev_per_traj_bkps)
                    for key in keys:
                        a = per_traj_bkps.get(key, [])
                        b = prev_per_traj_bkps.get(key, [])
                        timing_jaccards.append(
                            _breakpoint_jaccard_vs_ref(
                                a, b, args.tolerance_frames
                            )
                        )
                prev_per_traj_bkps = per_traj_bkps

                row: dict = {
                    "penalty": float(pen),
                    "n_trajectories": len(csv_files),
                    "total_breakpoints": len(bkp_df),
                    "timing_jaccard_vs_prev": (
                        float(np.mean(timing_jaccards)) if timing_jaccards else float("nan")
                    ),
                }
                for grp in DEFAULT_FEATURE_GROUPS:
                    row[f"n_bkps_{grp}"] = int(
                        len(bkp_df[bkp_df["group"] == grp]) if not bkp_df.empty else 0
                    )
                for grp_a, grp_b in _PAIR_LABELS:
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

        summary = pd.DataFrame(summary_rows)
        ref_pen_key = float(summary.loc[(summary["penalty"] - ref_penalty).abs().idxmin(), "penalty"])
        ref_regime_dirs = regime_dirs_by_penalty[ref_pen_key]
        summary["regime_agreement_vs_ref"] = summary["penalty"].map(
            lambda p: _regime_agreement(ref_regime_dirs, regime_dirs_by_penalty[float(p)])
        )
        summary = _annotate_count_change(summary)
        detail = pd.DataFrame(detail_rows)
        plateaus = _find_stable_plateaus(
            summary,
            min_plateau_steps=args.min_plateau_steps,
            timing_threshold=args.timing_threshold,
            regime_threshold=args.regime_threshold,
            count_change_threshold=args.count_change_threshold,
        )
        elbow = _find_penalty_elbow(summary)

        summary_path = output_dir / "penalty_sweep_summary.csv"
        detail_path = output_dir / "penalty_sweep_by_trajectory.csv"
        plateaus_path = output_dir / "penalty_stable_ranges.csv"
        elbow_path = output_dir / "penalty_elbow.csv"
        summary.to_csv(summary_path, index=False)
        detail.to_csv(detail_path, index=False)
        plateaus.to_csv(plateaus_path, index=False)
        if elbow:
            pd.DataFrame([elbow]).to_csv(elbow_path, index=False)
        else:
            pd.DataFrame(columns=["elbow_penalty"]).to_csv(elbow_path, index=False)

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
            sub = sub.sort_values(["ref_dist", "regime_agreement_vs_ref"], ascending=[True, False])
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

        rec_path = output_dir / "penalty_recommendation.txt"
        lines = [
            "Changepoint penalty sweep recommendation",
            "=" * 42,
            f"Trajectories: {len(csv_files)}",
            f"Reference penalty (log(n)): {ref_penalty:.6g}",
            f"Regime reference (nearest grid step): {ref_pen_key:.6g}",
            f"Penalty grid: {penalties[0]:.6g} … {penalties[-1]:.6g} ({len(penalties)} steps)",
            "",
        ]
        if elbow:
            lines.extend(
                [
                    "Primary selection: elbow on total breakpoints vs penalty",
                    f"  Recommended penalty: {elbow['elbow_penalty']:.6g}",
                    f"  Cohort breakpoints at elbow: {int(elbow['elbow_total_breakpoints'])}",
                    "",
                    "The elbow is where further increasing the penalty yields diminishing",
                    "reduction in breakpoint count (bias–variance knee on the sweep curve).",
                ]
            )
        if rec_range is not None:
            lines.extend(
                [
                    "",
                    f"Optional robust band (timing-stable, slow count drift): "
                    f"[{rec_range[0]:.6g}, {rec_range[1]:.6g}]",
                    f"  Band midpoint: {float(np.sqrt(rec_range[0] * rec_range[1])):.6g}",
                ]
            )
        if not elbow and rec_penalty is not None:
            lines.append(f"Recommended penalty (fallback): {rec_penalty:.6g}")
        lines.extend(
            [
                "",
                "Cross-group Jaccard (gsa↔combined, gsa↔iodine, …) is reported in",
                "penalty_sweep_summary.csv for interpretation only: each feature group",
                "is segmented independently, so agreement checks whether cage, guest,",
                "and solvent changepoints coincide — it does not define the penalty.",
            ]
        )
        if not plateaus.empty:
            lines.extend(["", "Robust bands (ranked):", plateaus.to_string(index=False)])
        rec_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        if not args.no_plots:
            _plot_sweep(summary, output_dir / "plots", elbow=elbow or None)

        log_event(
            "info",
            f"Wrote {summary_path.name}, {detail_path.name}, {plateaus_path.name}, "
            f"{elbow_path.name}, {rec_path.name}",
            component="sweep_changepoint_penalty",
        )

        if args.run_final and rec_penalty is not None:
            rc = _invoke_changepoint_feature_groups(
                input_dir=input_dir,
                output_dir=Path(args.final_output_dir),
                penalty=rec_penalty,
                method=args.method,
                cost_model=args.cost_model,
                min_size=args.min_size,
                jump=args.jump,
                tolerance_frames=args.tolerance_frames,
            )
            if rc != 0:
                sys.exit(rc)

    print(f"\nRecommended penalty (elbow): {rec_penalty:.6g}")
    if rec_range is not None:
        print(f"Stable range: [{rec_range[0]:.6g}, {rec_range[1]:.6g}]")
    print(f"\nOutputs in {output_dir}:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            print(f"  {f}")


if __name__ == "__main__":
    main()
