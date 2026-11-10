#!/usr/bin/env python3
"""Figure 4c and 4d: how exclusively a site-pair marker follows one Murata d1.

Panel (c) plots, for every leading site pair, the absolute correlation
with the preferred equatorial d1 against the absolute correlation with
the next edge. The vertical gap is the margin.

Panel (d) is the margin itself for belt-break markers only: clusters
named for a single broken edge. One row per system, ordered by the
median, with a tick at that median.

Reads ``docs/figures/si/site_pair_d1_r_*.csv``.

Example
-------
python scripts/plot_figure4c_d1_margin.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.murata_rmsd import CUBE_COLORS
from src.utils.run_log import RunContext

# Medians for clusters named after one broken edge, as reported with Figure 4c.
QUOTED_EDGE_MEDIANS: dict[str, float] = {
    "BHHpM": 0.07,
    "BHHpH": 0.17,
    "BMHpH": 0.27,
    "BMMpM": 0.36,
    "BMHpM": 0.66,
}
EDGE_TAG = re.compile(r"M\d–M\d")
MARGIN_GUIDES = (0.2, 0.4, 0.6)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot Figure 4c, preferred d1 correlation against the next edge."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "docs" / "figures" / "si",
        help="Directory of site_pair_d1_r_<cube>.csv tables",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "figure4c_d1_margin",
    )
    p.add_argument(
        "--si-dir",
        type=Path,
        default=ROOT / "docs" / "figures" / "si",
        help="Also copy the figure here. Pass '' to skip.",
    )
    return p.parse_args()


def _named_edge(panel_b: str) -> str | None:
    """Edge tag of a cluster named for one broken edge, or None."""
    tag = str(panel_b).split(",", 1)[-1].strip().replace("-", "–")
    if EDGE_TAG.fullmatch(tag):
        return tag
    return None


def _r_columns(table: pd.DataFrame) -> list[str]:
    columns = [col for col in table.columns if col.startswith("r_M")]
    if len(columns) != 6:
        raise ValueError(f"Expected six d1 correlations, found {columns}")
    return columns


def marker_table(data_dir: Path) -> pd.DataFrame:
    """One row per leading site pair, with preferred and next |r|."""
    frames = []
    for cube in KNOWN_CUBES:
        path = data_dir / f"site_pair_d1_r_{cube}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    table = pd.concat(frames, ignore_index=True)
    columns = _r_columns(table)
    rows = []
    for rec in table.itertuples(index=False):
        scores = []
        for col in columns:
            label = col[len("r_") :].replace("_", "–")
            value = float(getattr(rec, col))
            if not np.isfinite(value):
                raise ValueError(f"{rec.cohort} {rec.endpoint_label} has a non-finite {col}")
            scores.append((label, abs(value)))
        scores.sort(key=lambda item: (-item[1], item[0]))
        preferred, preferred_abs = scores[0]
        nxt, next_abs = scores[1]
        if preferred_abs + 1e-12 < next_abs:
            raise ValueError(f"{rec.cohort} {rec.endpoint_label}: preferred |r| is below the next edge")
        named = _named_edge(rec.panel_b)
        effect = float(rec.cohens_d_best)
        rows.append(
            {
                "cohort": rec.cohort,
                "k": int(rec.k),
                "best_cluster": int(rec.best_cluster),
                "panel_b": rec.panel_b,
                "rank_in_cluster": int(rec.rank_in_cluster),
                "endpoint_label": rec.endpoint_label,
                "pair_relation": rec.pair_relation,
                "cohens_d_best": effect,
                "shorter": effect < 0,
                "named_edge": named or "",
                "preferred_d1": preferred,
                "next_d1": nxt,
                "abs_r_preferred": preferred_abs,
                "abs_r_next": next_abs,
                "margin": preferred_abs - next_abs,
                "matches_named_edge": bool(named == preferred) if named else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def _check_text_numbers(points: pd.DataFrame) -> pd.Series:
    """Median margin on single-edge clusters, checked against the manuscript."""
    named = points[points["named_edge"].astype(str) != ""]
    n_match = int(named["matches_named_edge"].sum())
    if len(named) != 54 or n_match != 49:
        raise RuntimeError(
            f"Single-edge markers preferring their own edge: {n_match} of {len(named)} (expected 49 of 54)"
        )
    if (points["cohort"] == "BMMpH").sum() and (points.loc[points["cohort"] == "BMMpH", "named_edge"] != "").any():
        raise RuntimeError("BMMpH has a single-edge cluster; the text omits it from the margin list")
    medians = named.groupby("cohort")["margin"].median()
    for cube, quoted in QUOTED_EDGE_MEDIANS.items():
        got = float(medians[cube])
        if abs(got - quoted) > 0.005:
            raise RuntimeError(f"{cube} median margin is {got:.4f}, text quotes {quoted:.2f}")
    return medians


def plot_figure(points: pd.DataFrame, out: Path) -> Path:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.15, 5.85))

    ax.plot([0, 1], [0, 1], color="0.25", lw=0.9, zorder=2)
    for margin in MARGIN_GUIDES:
        y_end = 1.0 - margin
        ax.plot(
            [margin, 1.0],
            [0.0, y_end],
            color="0.75",
            lw=0.7,
            ls=(0, (2.4, 2.2)),
            zorder=1,
        )
        ax.annotate(
            f"{margin:.1f}",
            xy=(1.0, y_end),
            xytext=(11, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.5,
            color="0.35",
            annotation_clip=False,
            arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 0.6, "shrinkA": 0, "shrinkB": 1},
        )
    ax.annotate(
        "margin",
        xy=(1.0, 1.0 - MARGIN_GUIDES[0]),
        xytext=(11, 9),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="0.35",
        annotation_clip=False,
    )
    # Sit above the diagonal, on the empty side. Data stop near |r| = 0.58.
    ax.text(
        0.76,
        0.76,
        "equal $|r|$",
        rotation=45,
        rotation_mode="anchor",
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.28",
        zorder=4,
    )

    # Draw the wide-margin systems first so the diagonal cloud stays visible.
    order = sorted(KNOWN_CUBES, key=lambda cube: -float(points.loc[points["cohort"] == cube, "margin"].median()))
    for cube in order:
        part = points[points["cohort"] == cube]
        color = CUBE_COLORS[cube]
        for shorter, marker, size in ((False, "^", 46), (True, "o", 38)):
            sub = part[part["shorter"] == shorter]
            if sub.empty:
                continue
            ax.scatter(
                sub["abs_r_preferred"],
                sub["abs_r_next"],
                s=size,
                marker=marker,
                c=color,
                edgecolors="0.15",
                linewidths=0.45,
                alpha=0.92,
                zorder=3,
            )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel(r"absolute correlation with preferred $d_1$")
    ax.set_ylabel(r"absolute correlation with next $d_1$")
    ax.grid(alpha=0.22, zorder=0)
    ax.text(-0.12, 1.04, "(c)", transform=ax.transAxes, fontweight="bold", va="bottom", fontsize=12)

    system_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=CUBE_COLORS[cube],
            markeredgecolor="0.15",
            markeredgewidth=0.45,
            markersize=6.5,
            label=cube,
        )
        for cube in KNOWN_CUBES
    ]
    sign_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="0.45",
            markeredgecolor="0.15",
            markeredgewidth=0.45,
            markersize=6.5,
            label="negative $d$, shorter",
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            linestyle="none",
            markerfacecolor="0.45",
            markeredgecolor="0.15",
            markeredgewidth=0.45,
            markersize=6.5,
            label="positive $d$, longer",
        ),
    ]
    legend = ax.legend(
        handles=system_handles + sign_handles,
        loc="upper left",
        frameon=True,
        fontsize=8,
        borderpad=0.45,
        handletextpad=0.4,
        labelspacing=0.28,
        edgecolor="0.75",
    )
    legend.set_zorder(5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def _strip_offsets(values: np.ndarray, min_dx: float = 0.022, step: float = 0.14) -> np.ndarray:
    """Vertical dodge so markers that share an x stay readable."""
    offsets = np.zeros(len(values), dtype=float)
    placed: list[tuple[float, float]] = []
    for idx in np.argsort(values, kind="mergesort"):
        x = float(values[idx])
        level = 0
        y = 0.0
        while level < 16:
            if level == 0:
                y = 0.0
            else:
                sign = 1.0 if level % 2 else -1.0
                y = sign * ((level + 1) // 2) * step
            clear = True
            for px, py in placed:
                if abs(x - px) < min_dx and abs(y - py) < step * 0.85:
                    clear = False
                    break
            if clear:
                break
            level += 1
        offsets[idx] = y
        placed.append((x, y))
    return offsets


def plot_margin_strip(points: pd.DataFrame, medians: pd.Series, out: Path) -> Path:
    """Belt-break margins, one row per system, ordered by the median."""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    order = list(QUOTED_EDGE_MEDIANS)
    named = points[points["named_edge"].astype(str) != ""].copy()
    if set(named["cohort"]) != set(order):
        raise RuntimeError(f"Belt-break systems are {sorted(named['cohort'].unique())}, expected {order}")

    fig, ax = plt.subplots(figsize=(6.4, 3.35))
    y_of = {cube: len(order) - 1 - i for i, cube in enumerate(order)}
    for cube in order:
        part = named[named["cohort"] == cube]
        y = y_of[cube]
        values = part["margin"].to_numpy(dtype=float)
        offsets = _strip_offsets(values)
        ax.scatter(
            values,
            y + offsets,
            s=36,
            c=CUBE_COLORS[cube],
            edgecolors="0.15",
            linewidths=0.45,
            zorder=3,
        )
        median = float(medians[cube])
        ax.plot(
            [median, median],
            [y - 0.34, y + 0.34],
            color="0.1",
            lw=1.7,
            solid_capstyle="butt",
            zorder=4,
        )
        ax.text(
            1.03,
            y,
            f"{QUOTED_EDGE_MEDIANS[cube]:.2f}",
            ha="left",
            va="center",
            fontsize=8,
            color="0.15",
            clip_on=False,
        )

    ax.text(
        1.03,
        len(order) - 0.05,
        "median",
        ha="left",
        va="bottom",
        fontsize=8,
        color="0.35",
        clip_on=False,
    )
    ax.set_yticks([y_of[cube] for cube in order])
    ax.set_yticklabels(order)
    ax.set_ylim(-0.72, len(order) - 0.28)
    ax.set_xlim(-0.02, 1.0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("margin")
    ax.grid(axis="x", alpha=0.25, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.text(-0.08, 1.06, "(d)", transform=ax.transAxes, fontweight="bold", va="bottom", fontsize=12)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def _copy_manuscript(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    dest.write_bytes(src.read_bytes())
    return dest


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with RunContext.from_namespace(args, name="plot_figure4c_d1_margin"):
        points = marker_table(Path(args.data_dir))
        medians = _check_text_numbers(points)
        points_path = out_dir / "figure4c_points.csv"
        points.to_csv(points_path, index=False)
        fig_path = plot_figure(points, out_dir / "figure4c.png")
        strip_path = plot_margin_strip(points, medians, out_dir / "figure4d.png")
        named = points[points["named_edge"].astype(str) != ""]
        print(
            f"single-edge markers: {int(named['matches_named_edge'].sum())} of {len(named)} "
            "prefer the named edge"
        )
        quoted = ", ".join(f"{cube} {float(medians[cube]):.2f}" for cube in QUOTED_EDGE_MEDIANS)
        print(f"median margin on those clusters: {quoted}")
        print(f"wrote {points_path} ({len(points)} markers)")
        print(f"wrote {fig_path}")
        print(f"wrote {strip_path}")
        si_dir = str(args.si_dir).strip()
        if si_dir:
            for src in (
                fig_path,
                fig_path.with_suffix(".pdf"),
                strip_path,
                strip_path.with_suffix(".pdf"),
            ):
                print(f"copied figure to {_copy_manuscript(src, Path(si_dir))}")


if __name__ == "__main__":
    main()
