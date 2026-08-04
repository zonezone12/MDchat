"""Summarize changepoint outputs and generate cohort plots."""

from __future__ import annotations

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
)

# See module docstring in summarize script for why assembly_rg is the structural reference.
DEFAULT_TIMELINE_PANELS: list[tuple[str, str]] = [
    ("assembly_rg", "Assembly Rg (Å) — global cage size"),
    ("n_guest_inside_cavity", "Guest inside cavity"),
    ("cavity_ion_count", "Cavity Na⁺ count"),
]


def write_cohort_tables(changepoints_dir: Path) -> dict[str, Path]:
    """Write aggregated CSV summaries next to changepoint outputs."""
    changepoints_dir = Path(changepoints_dir)
    cmp = pd.read_csv(changepoints_dir / "changepoint_timing_comparison.csv")
    seg = pd.read_csv(changepoints_dir / "all_segment_stats.csv")
    bkp = pd.read_csv(changepoints_dir / "all_breakpoints.csv")

    written: dict[str, Path] = {}

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

    iod_gsa = cmp[
        ((cmp["group_a"] == "iodine") & (cmp["group_b"] == "gsa"))
        | ((cmp["group_a"] == "gsa") & (cmp["group_b"] == "iodine"))
    ].copy()
    low_path = changepoints_dir / "timing_low_agreement_iodine_vs_gsa.csv"
    high_path = changepoints_dir / "timing_high_agreement_iodine_vs_gsa.csv"
    iod_gsa.sort_values("jaccard").head(10).to_csv(low_path, index=False)
    iod_gsa.sort_values("jaccard", ascending=False).head(10).to_csv(high_path, index=False)
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
    groups = list(DEFAULT_GROUPS)
    mat = pd.DataFrame(np.nan, index=groups, columns=groups)
    for _, row in pairs.iterrows():
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
    ax.set_title(
        f"Cohort changepoint timing agreement ({pairs['n_trajectories'].iloc[0]} trajectories)"
    )
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

    for grp in DEFAULT_GROUPS:
        sub = bkp[bkp["group"] == grp]
        if sub.empty:
            continue
        ax.scatter(
            sub["time_ps"],
            sub["traj_id"].map(traj_to_y),
            c=GROUP_COLORS[grp],
            marker=GROUP_MARKERS[grp],
            s=55 if grp == "gsa" else 40,
            alpha=0.85,
            linewidths=0.8,
            label=grp,
            zorder=3,
        )

    ax.set_yticks(range(len(trajs)), labels, fontsize=7)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Trajectory variant")
    ax.set_title(
        f"Changepoints across all trajectories ({len(trajs)} variants, 4 feature groups)"
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
            feat_path = features_dir / f"{traj_id}_gsa_features.csv"
            if not feat_path.exists():
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
    """Find cluster_representatives.csv under {changepoints_dir}/clusters/."""
    clusters_root = Path(changepoints_dir)
    if not clusters_root.is_dir():
        return []
    return sorted(clusters_root.glob("**/cluster_representatives.csv"))


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
) -> Optional[Path]:
    changepoints_dir = Path(changepoints_dir)
    features_dir = Path(features_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    panels = panels or DEFAULT_TIMELINE_PANELS

    feat_path = features_dir / f"{traj_id}_gsa_features.csv"
    if not feat_path.exists():
        log_event(
            "warning",
            f"Skipping timeline for {traj_id}: missing {feat_path.name}",
            component="summarize_changepoint_results",
        )
        return None

    df = pd.read_csv(feat_path)
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
        for grp in DEFAULT_GROUPS:
            for _, row in bsub[bsub["group"] == grp].iterrows():
                ax.axvline(
                    row["time_ps"],
                    color=GROUP_COLORS[grp],
                    alpha=0.7,
                    lw=1.2,
                    ls="--",
                    zorder=3,
                )
        if rep_time_ps is not None and np.isfinite(rep_time_ps):
            ax.axvline(rep_time_ps, color="0.15", alpha=0.9, lw=1.0, ls=":", zorder=4)
        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel("Time (ps)")
    handles = [Line2D([0], [0], color=GROUP_COLORS[g], ls="--", label=g) for g in DEFAULT_GROUPS]
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
) -> list[str]:
    """Plot breakpoint timelines for each row in cluster_representatives.csv."""
    panels = panels or DEFAULT_TIMELINE_PANELS
    written: list[str] = []
    for _, row in reps.iterrows():
        traj_id = str(row["traj_id"])
        feat_path = Path(features_dir) / f"{traj_id}_gsa_features.csv"
        if not feat_path.exists():
            log_event(
                "warning",
                f"Skipping cluster rep timeline for {traj_id}: missing {feat_path.name}",
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
    traces = plot_cohort_traces_normalized_all_trajectories(
        changepoints_dir,
        features_dir,
        plot_dir,
        DEFAULT_TIMELINE_PANELS,
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
                panels=DEFAULT_TIMELINE_PANELS,
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
            cluster_written = plot_cluster_representative_timelines(
                reps,
                changepoints_dir=changepoints_dir,
                features_dir=features_dir,
                plot_dir=plot_dir,
                panels=DEFAULT_TIMELINE_PANELS,
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
