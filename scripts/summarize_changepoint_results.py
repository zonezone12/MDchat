"""
Summarize changepoint outputs and generate cohort plots.

Reads CSVs produced by ``changepoint_feature_groups.py`` and writes:
  cohort_timing_summary.csv
  cohort_segment_regime_summary.csv
  breakpoints_per_trajectory.csv
  timing_low/high_agreement_iodine_vs_gsa.csv
  plots/cohort_jaccard_heatmap.png
  plots/breakpoint_count_histogram.png
  plots/cohort_breakpoints_all_trajectories.png
  plots/cohort_traces_normalized_all_trajectories.png
  plots/timeline_{traj_id}.png

Timeline default panels
-----------------------
The first panel uses ``assembly_rg`` because it is a single, smooth scalar for
overall nanocube size (radius of gyration of the full GSA assembly). It is
already a headline metric in ``_SUMMARY_COLS`` for the gsa and combined groups
in ``changepoint_feature_groups.py``, appears in segment summaries, and tracks
global expansion or compaction without the high dimensionality of per-monomer
endpoint distances. The other two default panels mirror the iodine and na_water
groups: ``n_guest_inside_cavity`` (guest location) and ``cavity_ion_count``
(cavity solvent).

Example
-------
python scripts/summarize_changepoint_results.py \\
    --changepoints-dir output/changepoints \\
    --features-dir output/gsa_features
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src.utils.run_log import RunContext, log_event

GROUPS = ("gsa", "iodine", "na_water", "combined")
GROUP_COLORS = {
    "gsa": "#1f77b4",
    "iodine": "#d62728",
    "na_water": "#2ca02c",
    "combined": "#9467bd",
}
GROUP_MARKERS = {
    "gsa": "|",
    "iodine": "v",
    "na_water": "s",
    "combined": "D",
}

# See module docstring for why assembly_rg is the structural reference trace.
DEFAULT_TIMELINE_PANELS: list[tuple[str, str]] = [
    ("assembly_rg", "Assembly Rg (Å) — global cage size"),
    ("n_guest_inside_cavity", "Guest inside cavity"),
    ("cavity_ion_count", "Cavity Na⁺ count"),
]

SEGMENT_KEY_COLS: dict[str, list[str]] = {
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


def write_cohort_tables(changepoints_dir: Path) -> None:
    """Write aggregated CSV summaries next to changepoint outputs."""
    cmp = pd.read_csv(changepoints_dir / "changepoint_timing_comparison.csv")
    seg = pd.read_csv(changepoints_dir / "all_segment_stats.csv")
    bkp = pd.read_csv(changepoints_dir / "all_breakpoints.csv")

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
    pairs.to_csv(changepoints_dir / "cohort_timing_summary.csv", index=False)

    iod_gsa = cmp[
        ((cmp["group_a"] == "iodine") & (cmp["group_b"] == "gsa"))
        | ((cmp["group_a"] == "gsa") & (cmp["group_b"] == "iodine"))
    ].copy()
    iod_gsa.sort_values("jaccard").head(10).to_csv(
        changepoints_dir / "timing_low_agreement_iodine_vs_gsa.csv", index=False
    )
    iod_gsa.sort_values("jaccard", ascending=False).head(10).to_csv(
        changepoints_dir / "timing_high_agreement_iodine_vs_gsa.csv", index=False
    )

    rows: list[dict] = []
    for grp, cols in SEGMENT_KEY_COLS.items():
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
    pd.DataFrame(rows).to_csv(
        changepoints_dir / "cohort_segment_regime_summary.csv", index=False
    )

    bkp.groupby(["traj_id", "group"]).size().unstack(fill_value=0).to_csv(
        changepoints_dir / "breakpoints_per_trajectory.csv"
    )


def plot_jaccard_heatmap(changepoints_dir: Path, plot_dir: Path) -> Path:
    pairs = pd.read_csv(changepoints_dir / "cohort_timing_summary.csv")
    groups = list(GROUPS)
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
    ax.set_title(f"Cohort changepoint timing agreement ({pairs['n_trajectories'].iloc[0]} trajectories)")
    fig.tight_layout()
    out = plot_dir / "cohort_jaccard_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _variant_sort_key(traj_id: str) -> tuple[int | str, str]:
    """Sort trajectories by numeric variant id (e.g. BMMpM_109234_mdcrd_v → 109234)."""
    parts = traj_id.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return (int(parts[1]), traj_id)
    return (10**12, traj_id)


def _variant_label(traj_id: str) -> str:
    parts = traj_id.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[1]
    return traj_id


def _sorted_trajectory_ids(bkp: pd.DataFrame) -> list[str]:
    return sorted(bkp["traj_id"].unique(), key=_variant_sort_key)


def plot_cohort_breakpoints_all_trajectories(
    changepoints_dir: Path,
    plot_dir: Path,
) -> Path:
    """Single overview: every trajectory (row) × breakpoint time (x), colored by group."""
    bkp = pd.read_csv(changepoints_dir / "all_breakpoints.csv")
    trajs = _sorted_trajectory_ids(bkp)
    traj_to_y = {t: i for i, t in enumerate(trajs)}
    labels = [_variant_label(t) for t in trajs]

    fig_h = max(8.0, 0.22 * len(trajs) + 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))

    for grp in GROUPS:
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
    panels: list[tuple[str, str]],
) -> Optional[Path]:
    """All trajectories per feature on shared axes (min–max normalized per trajectory)."""
    bkp = pd.read_csv(changepoints_dir / "all_breakpoints.csv")
    trajs = _sorted_trajectory_ids(bkp)
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
    bkp = pd.read_csv(changepoints_dir / "all_breakpoints.csv")
    counts = bkp.groupby(["traj_id", "group"]).size().reset_index(name="n_bkps")

    fig, ax = plt.subplots(figsize=(7, 4))
    for grp in GROUPS:
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


def _default_timeline_trajectories(changepoints_dir: Path) -> list[str]:
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
        chosen.append(str(iod_gsa.sort_values("jaccard", ascending=False).iloc[0]["traj_id"]))
    # Stable dedupe preserving order
    return list(dict.fromkeys(chosen))


def plot_timeline(
    traj_id: str,
    *,
    changepoints_dir: Path,
    features_dir: Path,
    plot_dir: Path,
    panels: list[tuple[str, str]],
) -> Optional[Path]:
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

    for ax, (col, ylabel) in zip(axes, panels):
        if col not in df.columns:
            ax.set_ylabel(ylabel)
            ax.text(0.5, 0.5, f"missing column: {col}", transform=ax.transAxes, ha="center")
            continue
        ax.plot(time, df[col], color="0.3", lw=0.8)
        for grp in GROUPS:
            for _, row in bsub[bsub["group"] == grp].iterrows():
                ax.axvline(row["time_ps"], color=GROUP_COLORS[grp], alpha=0.7, lw=1.2, ls="--")
        ax.set_ylabel(ylabel)

    axes[-1].set_xlabel("Time (ps)")
    handles = [Line2D([0], [0], color=GROUP_COLORS[g], ls="--", label=g) for g in GROUPS]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle(f"Breakpoints overlay: {traj_id}", y=1.02)
    fig.tight_layout()
    out = plot_dir / f"timeline_{traj_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize changepoint CSV outputs and generate cohort plots."
    )
    parser.add_argument(
        "--changepoints-dir",
        default="output/changepoints",
        help="Directory with changepoint CSV outputs (default: output/changepoints)",
    )
    parser.add_argument(
        "--features-dir",
        default="output/gsa_features",
        help="Directory with *_gsa_features.csv files for timeline traces",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Plot output directory (default: {changepoints-dir}/plots)",
    )
    parser.add_argument(
        "--trajectories",
        nargs="*",
        default=None,
        help="Trajectory IDs for timeline plots (default: auto-pick 3 representative)",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Only regenerate plots, not summary CSVs",
    )
    parser.add_argument(
        "--skip-individual-timelines",
        action="store_true",
        help="Skip per-trajectory timeline_*.png plots",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    changepoints_dir = Path(args.changepoints_dir)
    features_dir = Path(args.features_dir)
    plot_dir = Path(args.plot_dir) if args.plot_dir else changepoints_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    required = [
        changepoints_dir / "all_breakpoints.csv",
        changepoints_dir / "all_segment_stats.csv",
        changepoints_dir / "changepoint_timing_comparison.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("Missing required changepoint outputs:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    with RunContext.from_namespace(args, name="summarize_changepoint_results"):
        if not args.skip_tables:
            write_cohort_tables(changepoints_dir)
            log_event(
                "info",
                "Wrote cohort summary CSVs",
                component="summarize_changepoint_results",
            )

        plot_jaccard_heatmap(changepoints_dir, plot_dir)
        plot_breakpoint_histogram(changepoints_dir, plot_dir)
        plot_cohort_breakpoints_all_trajectories(changepoints_dir, plot_dir)
        plot_cohort_traces_normalized_all_trajectories(
            changepoints_dir,
            features_dir,
            plot_dir,
            DEFAULT_TIMELINE_PANELS,
        )

        written: list[str] = []
        if not args.skip_individual_timelines:
            traj_ids = args.trajectories or _default_timeline_trajectories(changepoints_dir)
            for traj_id in traj_ids:
                out = plot_timeline(
                    traj_id,
                    changepoints_dir=changepoints_dir,
                    features_dir=features_dir,
                    plot_dir=plot_dir,
                    panels=DEFAULT_TIMELINE_PANELS,
                )
                if out is not None:
                    written.append(out.name)

        log_event(
            "info",
            f"Cohort plots written; {len(written)} individual timeline plot(s)",
            component="summarize_changepoint_results",
        )

    print("\nPlots:")
    for path in sorted(plot_dir.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
