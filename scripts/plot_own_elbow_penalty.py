"""Figure 2: own-elbow PELT curves and segment totals for the six cubes.

The elbow is chosen on cohort breakpoint totals (``find_penalty_elbow``):
log10(penalty) and the count are each scaled to [0, 1], and the interior
grid point farthest from the chord between the first and last points is
kept. Every trajectory contributes one more segment than it has
breakpoints, and the trajectory count does not change across the grid, so
segments = breakpoints + n_trajectories. That constant shift leaves the
normalized curve, and therefore the selected penalty, unchanged.

Panel (a) plots that segment count against log10(penalty), with the elbow
marked. Panel (b) is the number of segments written in each own-elbow
detection folder (``all_segment_stats.csv``).

BMHpH and BMMpM are read from ``penalty_sweep_median_n`` (the median-length
grid). The other four cubes use ``penalty_sweep``. Neither panel re-runs Pelt.

Example
-------
python scripts/plot_own_elbow_penalty.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.penalty_sweep import find_penalty_elbow
from src.ChangepointAnalysis.reporting import (
    OWN_ELBOW_ENDPOINT_DIRS,
    endpoint_changepoints_dir,
)
from src.utils.run_log import RunContext

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
        description="Plot own-elbow event-count curves and segment totals."
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root that holds the endpoint changepoint folders (default: output)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/own_elbow_penalty_figure"),
        help="Where the figure and tables are written",
    )
    p.add_argument(
        "--manuscript-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Also copy the figure here (default: docs/figures). Pass '' to skip.",
    )
    return p.parse_args()


def sweep_dir(output_root: Path, cube: str) -> Path:
    """Sweep that selected this cube's own elbow.

    The median-length grid replaces the first-CSV grid for BMHpH and BMMpM.
    """
    published = endpoint_changepoints_dir(output_root, cube, own_elbow=False)
    median = published / "penalty_sweep_median_n"
    if (median / "penalty_sweep_summary.csv").is_file():
        return median
    sweep = published / "penalty_sweep"
    if not (sweep / "penalty_sweep_summary.csv").is_file():
        raise FileNotFoundError(f"No penalty sweep for {cube} under {published}")
    return sweep


def load_curve(output_root: Path, cube: str) -> pd.DataFrame:
    sweep = sweep_dir(output_root, cube)
    summary = pd.read_csv(sweep / "penalty_sweep_summary.csv").sort_values("penalty")
    if summary.empty or "penalty" not in summary.columns:
        raise ValueError(f"{sweep} has no penalty column")
    elbow = find_penalty_elbow(summary)
    if not elbow:
        raise ValueError(f"{sweep} has no elbow")
    stored_path = sweep / "penalty_elbow.csv"
    if stored_path.is_file():
        stored = float(pd.read_csv(stored_path)["elbow_penalty"].iloc[0])
        if not np.isclose(stored, elbow["elbow_penalty"], rtol=1e-5, atol=1e-5):
            raise ValueError(
                f"{cube}: recomputed elbow {elbow['elbow_penalty']} "
                f"disagrees with {stored_path} ({stored})"
            )
    out = summary[["penalty", "total_breakpoints", "n_trajectories"]].copy()
    out.insert(0, "cube", cube)
    out.insert(1, "sweep_dir", str(sweep))
    out["log10_penalty"] = np.log10(out["penalty"].astype(float))
    # One segment per trajectory beyond its breakpoints. n_trajectories is
    # fixed on the grid, so this shift does not move the elbow index.
    out["n_segments"] = (
        out["total_breakpoints"].astype(int) + out["n_trajectories"].astype(int)
    )
    out["is_elbow"] = np.isclose(
        out["penalty"].astype(float), float(elbow["elbow_penalty"])
    )
    return out


def load_segments(output_root: Path, cube: str) -> dict:
    det = endpoint_changepoints_dir(output_root, cube, own_elbow=True)
    seg_path = det / "all_segment_stats.csv"
    bkp_path = det / "all_breakpoints.csv"
    if not seg_path.is_file():
        raise FileNotFoundError(seg_path)
    seg = pd.read_csv(seg_path)
    n_bkps = int(len(pd.read_csv(bkp_path))) if bkp_path.is_file() else 0
    return {
        "cube": cube,
        "detection_dir": str(det),
        "n_segments": int(len(seg)),
        "n_breakpoints": n_bkps,
        "n_trajectories": int(seg["traj_id"].nunique()) if "traj_id" in seg.columns else 0,
    }


def plot_figure(curves: pd.DataFrame, segments: pd.DataFrame, out: Path) -> Path:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.15))

    ax = axes[0]
    for cube in KNOWN_CUBES:
        sub = curves[curves["cube"] == cube]
        ax.plot(
            sub["log10_penalty"],
            sub["n_segments"],
            color=CUBE_COLORS[cube],
            marker="o",
            ms=3.5,
            lw=1.4,
            label=cube,
        )
    elbows = curves[curves["is_elbow"]]
    ax.scatter(
        elbows["log10_penalty"],
        elbows["n_segments"],
        s=78,
        facecolors="none",
        edgecolors="0.1",
        linewidths=1.3,
        zorder=5,
        label="elbow",
    )
    ax.set_xlabel(r"$\log_{10}$ penalty")
    ax.set_ylabel("Segments")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, frameon=False, loc="upper right")
    ax.text(-0.12, 1.04, "(a)", transform=ax.transAxes, fontweight="bold", va="bottom")

    ax = axes[1]
    x = np.arange(len(KNOWN_CUBES))
    heights = [
        int(segments.loc[segments["cube"] == cube, "n_segments"].iloc[0])
        for cube in KNOWN_CUBES
    ]
    ax.bar(
        x,
        heights,
        color=[CUBE_COLORS[c] for c in KNOWN_CUBES],
        width=0.72,
        edgecolor="0.15",
        linewidth=0.4,
    )
    for i, h in enumerate(heights):
        ax.text(i, h + 8, str(h), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(list(KNOWN_CUBES), rotation=30, ha="right")
    ax.set_ylabel("Segments")
    ax.set_xlabel("System")
    ax.set_ylim(0, max(heights) * 1.16)
    ax.grid(axis="y", alpha=0.25)
    ax.text(-0.14, 1.04, "(b)", transform=ax.transAxes, fontweight="bold", va="bottom")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with RunContext.from_namespace(args, name="plot_own_elbow_penalty"):
        missing = [c for c in KNOWN_CUBES if c not in OWN_ELBOW_ENDPOINT_DIRS]
        if missing:
            raise SystemExit(f"No own-elbow folder for {missing}")
        curves = pd.concat(
            [load_curve(output_root, cube) for cube in KNOWN_CUBES],
            ignore_index=True,
        )
        segments = pd.DataFrame([load_segments(output_root, cube) for cube in KNOWN_CUBES])
        elbows = curves[curves["is_elbow"]][
            ["cube", "penalty", "total_breakpoints", "sweep_dir"]
        ]
        segments = segments.merge(
            elbows.rename(
                columns={
                    "penalty": "elbow_penalty",
                    "total_breakpoints": "elbow_breakpoints",
                }
            ),
            on="cube",
            how="left",
        )
        curves_path = out_dir / "own_elbow_curves.csv"
        segments_path = out_dir / "own_elbow_segments.csv"
        curves.to_csv(curves_path, index=False)
        segments.to_csv(segments_path, index=False)
        fig_path = plot_figure(curves, segments, out_dir / "figure2_own_elbow_penalty.png")
        manuscript_dir = str(args.manuscript_dir).strip()
        copied = None
        if manuscript_dir:
            dest_dir = Path(manuscript_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in (fig_path, fig_path.with_suffix(".pdf")):
                dest = dest_dir / src.name
                dest.write_bytes(src.read_bytes())
            copied = dest_dir / fig_path.name
        print(f"Wrote {fig_path}")
        print(f"Wrote {curves_path}")
        print(f"Wrote {segments_path}")
        if copied is not None:
            print(f"Copied figure to {copied}")
        print(segments[["cube", "n_segments", "n_breakpoints", "n_trajectories", "elbow_penalty"]].to_string(index=False))


if __name__ == "__main__":
    main()
