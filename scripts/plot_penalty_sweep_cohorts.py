"""
Cross-cohort penalty-sweep figures for endpoint and GSA B* cubes.

Reads existing ``penalty_sweep/`` CSVs (does not re-run Pelt). Recreates the
comparison plots from the B* sweep canvas: breakpoints vs relative penalty,
timing Jaccard, and per-1000-frame density.

Example
-------
python scripts/plot_penalty_sweep_cohorts.py
python scripts/plot_penalty_sweep_cohorts.py --out-dir output/penalty_sweep_cohort_comparison
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.reporting import cohort_name_from_changepoints_dir
from src.utils.run_log import RunContext

_LOGN_RE = re.compile(r"Reference penalty \(log\(n\)\):\s*([0-9.eE+-]+)")

CUBE_COLORS: dict[str, str] = {
    "BHHpH": "#1f77b4",
    "BHHpM": "#5fa8d3",
    "BMHpH": "#2ca02c",
    "BMHpM": "#98df8a",
    "BMMpH": "#ff7f0e",
    "BMMpM": "#d62728",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot cross-cohort endpoint and GSA penalty-sweep comparison figures."
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root with endpoint_changepoints_B* and gsa_changepoints_B* (default: output)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/penalty_sweep_cohort_comparison"),
        help="Where CSVs and plots are written",
    )
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def _parse_logn(rec_path: Path) -> float | None:
    if not rec_path.is_file():
        return None
    text = rec_path.read_text(encoding="utf-8", errors="replace")
    m = _LOGN_RE.search(text)
    return float(m.group(1)) if m else None


def _median_n_frames(changepoints_dir: Path, group: str | None) -> tuple[float, int]:
    seg_path = changepoints_dir / "all_segment_stats.csv"
    if not seg_path.is_file():
        return float("nan"), 0
    seg = pd.read_csv(seg_path)
    if seg.empty or "n_frames" not in seg.columns:
        return float("nan"), 0
    if group and "group" in seg.columns:
        sub = seg[seg["group"].astype(str) == group]
        if not sub.empty:
            seg = sub
    per = seg.groupby("traj_id")["n_frames"].sum()
    return float(per.median()), int(per.size)


def _published_group_counts(changepoints_dir: Path) -> dict[str, int]:
    bkp_path = changepoints_dir / "all_breakpoints.csv"
    if not bkp_path.is_file():
        return {}
    bkp = pd.read_csv(bkp_path)
    if bkp.empty or "group" not in bkp.columns:
        return {}
    return {str(k): int(v) for k, v in bkp.groupby("group").size().items()}


def load_sweep_cohort(
    changepoints_dir: Path,
    *,
    family: str,
    default_group: str,
) -> dict | None:
    sweep = Path(changepoints_dir) / "penalty_sweep"
    summary_path = sweep / "penalty_sweep_summary.csv"
    if not summary_path.is_file():
        return None
    summary = pd.read_csv(summary_path)
    if summary.empty or "penalty" not in summary.columns:
        return None
    elbow_path = sweep / "penalty_elbow.csv"
    elbow = {}
    if elbow_path.is_file():
        el = pd.read_csv(elbow_path)
        if not el.empty:
            elbow = el.iloc[0].to_dict()
    logn = _parse_logn(sweep / "penalty_recommendation.txt")
    if logn is None or not np.isfinite(logn) or logn <= 0:
        n_frames, _ = _median_n_frames(changepoints_dir, default_group)
        logn = float(np.log(max(n_frames, 2.0))) if np.isfinite(n_frames) else float("nan")
    n_frames, n_traj_seg = _median_n_frames(changepoints_dir, default_group)
    n_traj = int(summary["n_trajectories"].iloc[0]) if "n_trajectories" in summary else n_traj_seg
    published = _published_group_counts(changepoints_dir)
    plateaus_path = sweep / "penalty_stable_ranges.csv"
    plateaus = (
        pd.read_csv(plateaus_path) if plateaus_path.is_file() else pd.DataFrame()
    )
    return {
        "family": family,
        "cube": cohort_name_from_changepoints_dir(changepoints_dir),
        "changepoints_dir": Path(changepoints_dir),
        "summary": summary,
        "elbow": elbow,
        "logn": float(logn),
        "n_frames_median": n_frames,
        "n_trajectories": n_traj,
        "published": published,
        "plateaus": plateaus,
    }


def discover_sweeps(output_root: Path) -> list[dict]:
    cohorts: list[dict] = []
    for cube in KNOWN_CUBES:
        ep = output_root / f"endpoint_changepoints_{cube}"
        loaded = load_sweep_cohort(ep, family="endpoint", default_group="endpoint")
        if loaded:
            cohorts.append(loaded)
        gsa = output_root / f"gsa_changepoints_{cube}"
        loaded = load_sweep_cohort(gsa, family="gsa", default_group="gsa")
        if loaded:
            cohorts.append(loaded)
    return cohorts


def _curve_frame(cohort: dict) -> pd.DataFrame:
    sm = cohort["summary"].copy()
    n_traj = max(int(cohort["n_trajectories"]), 1)
    n_frames = float(cohort["n_frames_median"])
    logn = float(cohort["logn"])
    sm["cube"] = cohort["cube"]
    sm["family"] = cohort["family"]
    sm["rel_penalty"] = sm["penalty"] / logn if logn > 0 else np.nan
    sm["bkps_per_traj"] = sm["total_breakpoints"].astype(float) / n_traj
    scale = 1000.0 / n_frames if np.isfinite(n_frames) and n_frames > 0 else np.nan
    sm["bkps_per_1000_frames"] = sm["bkps_per_traj"] * scale
    for col in [c for c in sm.columns if c.startswith("n_bkps_")]:
        grp = col.replace("n_bkps_", "")
        sm[f"bkps_per_traj_{grp}"] = sm[col].astype(float) / n_traj
        sm[f"bkps_per_1000_{grp}"] = sm[f"bkps_per_traj_{grp}"] * scale
    elbow_pen = cohort["elbow"].get("elbow_penalty")
    sm["is_elbow"] = False
    if elbow_pen is not None and np.isfinite(float(elbow_pen)):
        sm["is_elbow"] = np.isclose(sm["penalty"].astype(float), float(elbow_pen))
    sm["is_logn"] = np.isclose(sm["rel_penalty"].astype(float), 1.0, atol=0.08)
    return sm


def _cube_style(cube: str) -> dict:
    return {"color": CUBE_COLORS.get(cube, "#333333"), "marker": "o", "ms": 4, "lw": 1.3}


def _mark_elbows(ax, curves: pd.DataFrame, ycol: str) -> None:
    elbows = curves[curves["is_elbow"]]
    if elbows.empty:
        return
    ax.scatter(
        elbows["rel_penalty"],
        elbows[ycol],
        s=55,
        zorder=5,
        facecolors="none",
        edgecolors="0.15",
        linewidths=1.2,
        label="elbow",
    )


def plot_bkps_vs_rel(
    curves: pd.DataFrame,
    *,
    ycol: str,
    ylabel: str,
    title: str,
    out: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cube, sub in curves.groupby("cube", sort=False, observed=True):
        ax.plot(sub["rel_penalty"], sub[ycol], **_cube_style(str(cube)), label=str(cube))
    _mark_elbows(ax, curves, ycol)
    ax.axvline(1.0, color="0.55", ls="--", lw=1.0, label="log(n)")
    ax.set_xscale("log")
    ax.set_xlabel("Penalty / log(n)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_timing_jaccard(curves: pd.DataFrame, *, title: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5.0))
    for cube, sub in curves.groupby("cube", sort=False, observed=True):
        plot = sub.dropna(subset=["timing_jaccard_vs_prev"])
        ax.plot(
            plot["rel_penalty"],
            plot["timing_jaccard_vs_prev"],
            **_cube_style(str(cube)),
            label=str(cube),
        )
    ax.axhline(0.75, color="0.45", ls="--", lw=1.0, label="plateau threshold 0.75")
    ax.axvline(1.0, color="0.55", ls=":", lw=1.0, label="log(n)")
    ax.set_xscale("log")
    ax.set_ylim(0.6, 1.0)
    ax.set_xlabel("Penalty / log(n)")
    ax.set_ylabel("Timing Jaccard vs previous penalty")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_gsa_groups(curves: pd.DataFrame, *, out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharex=True, sharey=True)
    for ax, grp, title in (
        (axes[0], "gsa", "GSA geometry"),
        (axes[1], "iodine", "Iodine guest"),
    ):
        ycol = f"bkps_per_traj_{grp}"
        if ycol not in curves.columns:
            ax.set_visible(False)
            continue
        for cube, sub in curves.groupby("cube", sort=False, observed=True):
            ax.plot(sub["rel_penalty"], sub[ycol], **_cube_style(str(cube)), label=str(cube))
        elbows = curves[curves["is_elbow"]]
        if not elbows.empty:
            ax.scatter(
                elbows["rel_penalty"],
                elbows[ycol],
                s=55,
                zorder=5,
                facecolors="none",
                edgecolors="0.15",
                linewidths=1.2,
            )
        ax.axvline(1.0, color="0.55", ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Penalty / log(n)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Mean breakpoints / trajectory")
    axes[1].legend(ncol=2, fontsize=8, loc="upper right")
    fig.suptitle("GSA penalty sweep by feature group")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_density_bars(rows: pd.DataFrame, *, out: Path) -> Path:
    cubes = [c for c in KNOWN_CUBES if c in set(rows["cube"])]
    series = [
        ("endpoint_elbow", "Endpoint at elbow"),
        ("gsa_geom_elbow", "GSA geometry at elbow"),
        ("gsa_geom_logn", "GSA geometry at log(n)"),
        ("gsa_iodine_logn", "GSA iodine at log(n)"),
    ]
    x = np.arange(len(cubes))
    width = 0.18
    fig, ax = plt.subplots(figsize=(11, 5.5))
    offsets = np.linspace(-1.5, 1.5, len(series)) * width
    colors = ["#ff7f0e", "#1f77b4", "#5fa8d3", "#d62728"]
    for off, (col, label), color in zip(offsets, series, colors):
        vals = [
            float(rows.loc[rows["cube"] == c, col].iloc[0])
            if c in set(rows["cube"]) and col in rows.columns
            and np.isfinite(rows.loc[rows["cube"] == c, col].iloc[0])
            else 0.0
            for c in cubes
        ]
        ax.bar(x + off, vals, width=width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(cubes)
    ax.set_ylabel("Breakpoints per 1000 frames")
    ax.set_xlabel("Cube")
    ax.set_title("Event density at matched relative penalties")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overlay_per_1000(
    endpoint: pd.DataFrame,
    gsa: pd.DataFrame,
    *,
    out: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharex=True)
    panels = (
        (axes[0], endpoint, "bkps_per_1000_frames", "Endpoint (all groups)"),
        (axes[1], gsa, "bkps_per_1000_gsa", "GSA geometry"),
    )
    for ax, curves, ycol, title in panels:
        if ycol not in curves.columns:
            ycol = "bkps_per_1000_frames"
        for cube, sub in curves.groupby("cube", sort=False, observed=True):
            ax.plot(sub["rel_penalty"], sub[ycol], **_cube_style(str(cube)), label=str(cube))
        _mark_elbows(ax, curves, ycol)
        ax.axvline(1.0, color="0.55", ls="--", lw=1.0)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Penalty / log(n)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Breakpoints per 1000 frames")
    axes[1].legend(ncol=2, fontsize=8, loc="upper right")
    fig.suptitle("Endpoint vs GSA geometry on a shared relative-penalty axis")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _density_row(cohort: dict) -> dict:
    sm = _curve_frame(cohort)
    n_frames = float(cohort["n_frames_median"])
    scale = 1000.0 / n_frames if np.isfinite(n_frames) and n_frames > 0 else np.nan
    n_traj = max(int(cohort["n_trajectories"]), 1)
    elbow_pen = cohort["elbow"].get("elbow_penalty")
    elbow_row = None
    if elbow_pen is not None and not sm.empty:
        elbow_row = sm.iloc[(sm["penalty"] - float(elbow_pen)).abs().idxmin()]
    logn_row = sm.iloc[(sm["rel_penalty"] - 1.0).abs().idxmin()] if not sm.empty else None
    published = cohort["published"]
    out = {
        "family": cohort["family"],
        "cube": cohort["cube"],
        "logn": cohort["logn"],
        "elbow_penalty": float(elbow_pen) if elbow_pen is not None else np.nan,
        "elbow_vs_logn": (
            float(elbow_pen) / cohort["logn"]
            if elbow_pen is not None and cohort["logn"]
            else np.nan
        ),
        "n_trajectories": n_traj,
        "n_frames_median": n_frames,
        "published_n_bkps": int(sum(published.values())) if published else 0,
    }
    if elbow_row is not None:
        out["elbow_total_bkps"] = int(elbow_row["total_breakpoints"])
        out["elbow_bkps_per_traj"] = float(elbow_row["bkps_per_traj"])
        out["elbow_bkps_per_1000"] = float(elbow_row["bkps_per_1000_frames"])
        for col in sm.columns:
            if col.startswith("n_bkps_"):
                grp = col.replace("n_bkps_", "")
                out[f"elbow_n_bkps_{grp}"] = int(elbow_row[col])
    if logn_row is not None:
        out["logn_grid_penalty"] = float(logn_row["penalty"])
        out["logn_total_bkps"] = int(logn_row["total_breakpoints"])
        out["logn_bkps_per_traj"] = float(logn_row["bkps_per_traj"])
        out["logn_bkps_per_1000"] = float(logn_row["bkps_per_1000_frames"])
        for col in sm.columns:
            if col.startswith("n_bkps_"):
                grp = col.replace("n_bkps_", "")
                out[f"logn_n_bkps_{grp}"] = int(logn_row[col])
                out[f"logn_bkps_per_1000_{grp}"] = float(
                    logn_row.get(f"bkps_per_1000_{grp}", np.nan)
                )
    for grp, n in published.items():
        out[f"published_n_bkps_{grp}"] = n
        out[f"published_bkps_per_1000_{grp}"] = (
            (n / n_traj) * scale if np.isfinite(scale) else np.nan
        )
    plat = cohort["plateaus"]
    if plat is not None and not plat.empty:
        best = plat.iloc[0]
        out["band_penalty_min"] = float(best.get("penalty_min", np.nan))
        out["band_penalty_max"] = float(best.get("penalty_max", np.nan))
        out["band_n_steps"] = int(best.get("n_penalty_steps", 0))
        out["band_bkps_min"] = int(best.get("total_breakpoints_min", 0))
        out["band_bkps_max"] = int(best.get("total_breakpoints_max", 0))
        out["elbow_in_band"] = bool(
            elbow_pen is not None
            and float(best.get("penalty_min", np.nan))
            <= float(elbow_pen)
            <= float(best.get("penalty_max", np.nan))
        )
        out["band_passes_regime"] = bool(best.get("passes_regime_threshold", False))
        out["band_regime_agreement"] = float(best.get("median_regime_agreement", np.nan))
    return out


def _bar_table(density: pd.DataFrame) -> pd.DataFrame:
    ep = density[density["family"] == "endpoint"].set_index("cube")
    gsa = density[density["family"] == "gsa"].set_index("cube")
    rows = []
    for cube in KNOWN_CUBES:
        row = {"cube": cube}
        if cube in ep.index:
            row["endpoint_elbow"] = float(ep.loc[cube, "elbow_bkps_per_1000"])
        if cube in gsa.index:
            row["gsa_geom_elbow"] = float(
                gsa.loc[cube].get("elbow_n_bkps_gsa", np.nan)
            ) / max(float(gsa.loc[cube, "n_trajectories"]), 1) * (
                1000.0 / float(gsa.loc[cube, "n_frames_median"])
            )
            row["gsa_geom_logn"] = float(
                gsa.loc[cube].get("published_bkps_per_1000_gsa", np.nan)
            )
            row["gsa_iodine_logn"] = float(
                gsa.loc[cube].get("published_bkps_per_1000_iodine", np.nan)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_all(cohorts: list[dict], out_dir: Path) -> dict[str, Path]:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    ep_curves = pd.concat(
        [_curve_frame(c) for c in cohorts if c["family"] == "endpoint"],
        ignore_index=True,
    )
    gsa_curves = pd.concat(
        [_curve_frame(c) for c in cohorts if c["family"] == "gsa"],
        ignore_index=True,
    )
    cube_cat = list(KNOWN_CUBES)
    for frame in (ep_curves, gsa_curves):
        if not frame.empty:
            frame["cube"] = pd.Categorical(frame["cube"], categories=cube_cat, ordered=True)
            frame.sort_values(["cube", "penalty"], inplace=True)
    density = pd.DataFrame([_density_row(c) for c in cohorts])
    density_path = out_dir / "cohort_sweep_summary.csv"
    density.to_csv(density_path, index=False)
    written["cohort_sweep_summary"] = density_path
    if not ep_curves.empty:
        written["endpoint_bkps_per_traj"] = plot_bkps_vs_rel(
            ep_curves,
            ycol="bkps_per_traj",
            ylabel="Mean breakpoints / trajectory",
            title="Endpoint breakpoints vs relative penalty",
            out=plot_dir / "endpoint_bkps_per_traj_vs_rel_penalty.png",
        )
        written["endpoint_timing_jaccard"] = plot_timing_jaccard(
            ep_curves,
            title="Endpoint timing Jaccard vs relative penalty",
            out=plot_dir / "endpoint_timing_jaccard_vs_rel_penalty.png",
        )
    if not gsa_curves.empty:
        written["gsa_bkps_per_traj"] = plot_bkps_vs_rel(
            gsa_curves,
            ycol="bkps_per_traj",
            ylabel="Mean breakpoints / trajectory (gsa + iodine)",
            title="GSA breakpoints vs relative penalty",
            out=plot_dir / "gsa_bkps_per_traj_vs_rel_penalty.png",
        )
        written["gsa_timing_jaccard"] = plot_timing_jaccard(
            gsa_curves,
            title="GSA timing Jaccard vs relative penalty",
            out=plot_dir / "gsa_timing_jaccard_vs_rel_penalty.png",
        )
        if "bkps_per_traj_gsa" in gsa_curves.columns:
            written["gsa_bkps_by_group"] = plot_gsa_groups(
                gsa_curves,
                out=plot_dir / "gsa_bkps_per_traj_by_group.png",
            )
    if not ep_curves.empty and not gsa_curves.empty:
        written["overlay_per_1000"] = plot_overlay_per_1000(
            ep_curves,
            gsa_curves,
            out=plot_dir / "endpoint_vs_gsa_bkps_per_1000_vs_rel_penalty.png",
        )
        bars = _bar_table(density)
        bars_path = out_dir / "density_per_1000_frames.csv"
        bars.to_csv(bars_path, index=False)
        written["density_table"] = bars_path
        written["density_bars"] = plot_density_bars(
            bars,
            out=plot_dir / "bkps_per_1000_frames_by_cube.png",
        )
    return written


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with RunContext.from_namespace(args, name="plot_penalty_sweep_cohorts"):
        cohorts = discover_sweeps(output_root)
        if not cohorts:
            print(f"No penalty_sweep/ summaries under {output_root}", file=sys.stderr)
            sys.exit(1)
        if args.no_plots:
            density = pd.DataFrame([_density_row(c) for c in cohorts])
            path = out_dir / "cohort_sweep_summary.csv"
            density.to_csv(path, index=False)
            written = {"cohort_sweep_summary": path}
        else:
            written = plot_all(cohorts, out_dir)
        print(f"Wrote {len(written)} artifact(s) to {out_dir}:")
        for name, path in sorted(written.items()):
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
