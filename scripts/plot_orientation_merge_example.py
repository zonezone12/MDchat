"""Worked example of the S6 orientation merge, using BMHpM.

Ward keeps the monomer numbers, so one belt break is stored as six clusters.
This figure follows clusters 1 and 2: their deformation profiles correlate at
r = 0.01 with those numbers held fixed, and at r = 0.99 after the belt step
(0 4 2 3 1 5), the best of the six contact-graph automorphisms. Single linkage
at r >= 0.95 then collects all six clusters into one state.

The bars are the share of each cluster's apo frames on which that equatorial
d1 is elongated. The quoted r is the Pearson correlation of the full
960 site-pair deformation profiles (``all_cluster_symmetry.csv``).

Example
-------
python scripts/plot_orientation_merge_example.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.metastable_catalog import BELT_SIX_CYCLE, COPY_R
from src.utils.run_log import RunContext

CUBE = "BMHpM"
EXAMPLE = (1, 2)
# Equatorial belt in cycle order, matching Figure 3.
BELT: tuple[tuple[int, int], ...] = (
    (0, 4),
    (4, 2),
    (2, 3),
    (3, 1),
    (1, 5),
    (5, 0),
)
MONOMERS: tuple[str, ...] = tuple(f"M{a}" for a, _b in BELT)
NORTH_POLE = frozenset({0, 1, 2})
FACE_COLOR = {m: ("#9ecae1" if m in NORTH_POLE else "#fdae6b") for m in range(6)}
CLUSTER_COLOR = {1: "#d94801", 2: "#2171b5"}
OPEN = "#d94801"
CLOSED = "#d9d9d9"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Illustrate the S6 orientation merge on BMHpM.")
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=Path("output/metastable_structures_physical_k"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/symmetry_merge_example"),
    )
    p.add_argument("--manuscript-dir", type=Path, default=Path("docs/figures"))
    return p.parse_args()


def edge_name(a: int, b: int) -> str:
    return f"M{a}–M{b}"


def _column_for(columns: list[str], a: int, b: int) -> str:
    for col in columns:
        if col.endswith(f"M{a}-M{b}") or col.endswith(f"M{b}-M{a}"):
            return col
    raise KeyError(f"No elongation column for M{a}-M{b}")


def edge_shares(row: pd.Series, columns: list[str]) -> np.ndarray:
    return np.array([float(row[_column_for(columns, a, b)]) for a, b in BELT], dtype=float)


def apply_belt_step(shares: np.ndarray) -> np.ndarray:
    """Reindex *shares* so axis edge (i, j) reads the value on (p(i), p(j)).

    ``BELT_SIX_CYCLE`` is the image tuple of (0 4 2 3 1 5): monomer m moves
    to ``BELT_SIX_CYCLE[m]``. Comparing a profile with the relabeled one
    lines cluster 2's long edge up with cluster 1's.
    """
    perm = {m: int(image) for m, image in enumerate(BELT_SIX_CYCLE)}
    index = {pair: i for i, pair in enumerate(BELT)}
    index.update({(b, a): i for (a, b), i in index.items()})
    out = np.zeros(len(BELT), dtype=float)
    for i, (a, b) in enumerate(BELT):
        out[i] = shares[index[(perm[a], perm[b])]]
    return out


def pair_row(symmetry: pd.DataFrame, a: int, b: int) -> pd.Series:
    hit = symmetry[
        ((symmetry["cluster_a"] == a) & (symmetry["cluster_b"] == b))
        | ((symmetry["cluster_a"] == b) & (symmetry["cluster_b"] == a))
    ]
    if len(hit) != 1:
        raise ValueError(f"{CUBE} pair {a},{b} appears {len(hit)} times")
    return hit.iloc[0]


def _hexagon(cx: float, cy: float, radius: float) -> np.ndarray:
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 6, endpoint=False)
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])


def draw_hexagon(ax, cx: float, cy: float, radius: float, open_index: int) -> None:
    vertices = _hexagon(cx, cy, radius)
    for i in range(6):
        j = (i + 1) % 6
        opened = i == open_index
        ax.plot(
            [vertices[i, 0], vertices[j, 0]],
            [vertices[i, 1], vertices[j, 1]],
            color=OPEN if opened else "#bdbdbd",
            lw=3.4 if opened else 1.5,
            solid_capstyle="round",
            zorder=2,
        )
    colors = [FACE_COLOR[int(name[1])] for name in MONOMERS]
    ax.scatter(
        vertices[:, 0],
        vertices[:, 1],
        s=58,
        c=colors,
        zorder=3,
        linewidths=0.5,
        edgecolors="0.25",
    )
    for (x, y), name in zip(vertices, MONOMERS):
        ax.text(
            cx + (x - cx) * 1.55,
            cy + (y - cy) * 1.55,
            name,
            ha="center",
            va="center",
            fontsize=7.5,
            color="0.2",
            zorder=4,
        )


def _bars(
    ax,
    left: np.ndarray,
    right: np.ndarray,
    ylabel: str | None,
    *,
    right_label: str = "cluster 2",
) -> None:
    x = np.arange(len(BELT))
    width = 0.38
    ax.bar(
        x - width / 2,
        left,
        width,
        color=CLUSTER_COLOR[1],
        label="cluster 1",
        zorder=2,
    )
    ax.bar(
        x + width / 2,
        right,
        width,
        color=CLUSTER_COLOR[2],
        label=right_label,
        zorder=2,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([edge_name(a, b) for a, b in BELT], rotation=35, ha="right")
    ax.set_ylim(0, 118)
    ax.set_xlim(-0.6, len(BELT) - 0.4)
    ax.yaxis.grid(True, lw=0.4, color="0.85", zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.tick_params(labelleft=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _family_notes(symmetry: pd.DataFrame, members: list[int]) -> str:
    pairs = []
    for i, a in enumerate(members):
        for b in members[i + 1 :]:
            rec = pair_row(symmetry, a, b)
            pairs.append((a, b, float(rec.r_best_relabelling)))
    weak_a, weak_b, weak_r = min(pairs, key=lambda item: item[2])
    weak_a, weak_b = sorted((weak_a, weak_b))
    bridges = []
    for node in (weak_a, weak_b):
        others = [(a, b, r) for a, b, r in pairs if node in (a, b) and r >= COPY_R]
        if not others:
            continue
        best_r = max(score for _a, _b, score in others)
        near = sorted(
            {
                (_b if _a == node else _a)
                for _a, _b, score in others
                if abs(score - best_r) < 0.005
            }
        )
        bridges.append((node, near, best_r))
    bridge_text = ", and ".join(
        f"{node} matches {' and '.join(str(p) for p in partners)} at {score:.2f}"
        for node, partners, score in bridges
    )
    return (
        f"Clusters {weak_a} and {weak_b} correlate at {weak_r:.2f}. "
        f"Single linkage still joins them: {bridge_text}."
    )


def plot_figure(
    clusters: pd.DataFrame,
    symmetry: pd.DataFrame,
    out: Path,
) -> Path:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    part = clusters[clusters["cohort"] == CUBE].copy()
    family = part[part["n_orientations"] == 6]
    if family.empty:
        raise ValueError(f"{CUBE} has no six-orientation family")
    columns = [col for col in part.columns if str(col).startswith("pct_d1_elongated_e")]
    rows = {int(row.cluster_label): row for row in family.itertuples(index=False)}
    by_label = {int(row["cluster_label"]): row for _, row in family.iterrows()}
    for label in EXAMPLE:
        if label not in rows:
            raise ValueError(f"Cluster {label} is not in the {CUBE} belt-break family")

    shares = {}
    open_at = {}
    for label, series in by_label.items():
        values = edge_shares(series, columns)
        shares[label] = values
        open_at[label] = int(np.argmax(values))

    relabeled = apply_belt_step(shares[EXAMPLE[1]])
    example = pair_row(symmetry, *EXAMPLE)
    best = str(example.best_relabelling).strip()
    if best != "(0 4 2 3 1 5)":
        raise ValueError(f"Expected the belt step for clusters {EXAMPLE}, found {best}")
    r_fixed = float(example.r_as_labelled)
    r_best = float(example.r_best_relabelling)
    moved_peak = int(np.argmax(relabeled))
    if moved_peak != open_at[EXAMPLE[0]]:
        raise ValueError("Belt step did not place cluster 2's long edge on cluster 1's")

    fig = plt.figure(figsize=(12.4, 9.2))
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=[1.05, 1.28, 1.05],
        left=0.07,
        right=0.985,
        top=0.88,
        bottom=0.035,
        hspace=0.42,
    )
    top = outer[0].subgridspec(1, 3, width_ratios=[1.0, 0.38, 1.0], wspace=0.02)
    ax_left = fig.add_subplot(top[0, 0])
    ax_mid = fig.add_subplot(top[0, 1])
    ax_right = fig.add_subplot(top[0, 2])
    mid_bars = outer[1].subgridspec(1, 2, wspace=0.18)
    ax_before = fig.add_subplot(mid_bars[0, 0])
    ax_after = fig.add_subplot(mid_bars[0, 1], sharey=ax_before)
    ax_d = fig.add_subplot(outer[2])

    fig.suptitle(
        "Merging equivalent orientations (S$_{6}$): one BMHpM belt break",
        fontsize=13,
        fontweight="bold",
        x=0.07,
        y=0.975,
        ha="left",
    )
    fig.text(
        0.07,
        0.915,
        "(a)  Ward keeps the monomer numbers",
        fontsize=11,
        ha="left",
        va="bottom",
    )

    hex_axes = {EXAMPLE[0]: ax_left, EXAMPLE[1]: ax_right}
    for label, ax in hex_axes.items():
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-2.15, 2.15)
        ax.set_ylim(-2.35, 2.45)
        draw_hexagon(ax, 0.0, 0.05, 1.0, open_at[label])
        rec = rows[label]
        ax.text(
            0.0,
            1.85,
            f"cluster {label}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color=CLUSTER_COLOR[label],
        )
        ax.text(
            0.0,
            -2.15,
            f"{edge_name(*BELT[open_at[label]])}    {rec.pct_of_cube:.1f}% of apo frames",
            ha="center",
            va="top",
            fontsize=8.5,
            color="0.2",
        )
    ax_mid.axis("off")
    ax_mid.text(
        0.5,
        0.62,
        "fixed labels",
        transform=ax_mid.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        color="0.35",
    )
    ax_mid.text(
        0.5,
        0.46,
        f"r = {r_fixed:+.2f}",
        transform=ax_mid.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="0.15",
    )
    ax_mid.text(
        0.5,
        0.16,
        "Blue and amber\nvertices: the two poles\nOrange: open edge",
        transform=ax_mid.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color="0.4",
        linespacing=1.3,
    )

    ylabel = "Apo frames with this\nequatorial edge elongated (%)"
    _bars(ax_before, shares[EXAMPLE[0]], shares[EXAMPLE[1]], ylabel)
    _bars(ax_after, shares[EXAMPLE[0]], relabeled, None, right_label="cluster 2, relabeled")
    ax_before.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    ax_after.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    ax_before.set_title("(b)  As stored", loc="left", fontsize=11)
    ax_after.set_title("(c)  After the belt step  (0 4 2 3 1 5)", loc="left", fontsize=11)
    ax_before.text(
        0.98,
        0.97,
        f"profile r = {r_fixed:+.2f}",
        transform=ax_before.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="0.2",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "lw": 0.6},
    )
    ax_after.text(
        0.98,
        0.97,
        f"profile r = {r_best:+.2f}",
        transform=ax_after.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="0.15",
        bbox={"boxstyle": "round,pad=0.25", "fc": "#fff5eb", "ec": OPEN, "lw": 0.8},
    )
    peak = open_at[EXAMPLE[0]]
    ax_after.axvspan(peak - 0.45, peak + 0.45, color=OPEN, alpha=0.08, zorder=0)

    ax_d.axis("off")
    ax_d.set_xlim(0, 12)
    ax_d.set_ylim(0, 10)
    ax_d.set_title("(d)  Single linkage at r ≥ 0.95", loc="left", fontsize=11, pad=8)
    ax_d.add_patch(
        FancyBboxPatch(
            (0.15, 4.55),
            11.7,
            5.15,
            boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor="#fff5eb",
            edgecolor=OPEN,
            lw=1.15,
        )
    )
    ax_d.text(
        6.0,
        9.05,
        f"one state     {family['pct_of_cube'].sum():.1f}% of apo frames",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=OPEN,
    )
    order = sorted(rows, key=lambda label: open_at[label])
    for n, label in enumerate(order):
        cx = 1.05 + n * 1.95
        cy = 6.55
        worked = label in EXAMPLE
        ax_d.add_patch(
            FancyBboxPatch(
                (cx - 0.82, cy - 1.15),
                1.64,
                2.3,
                boxstyle="round,pad=0.03,rounding_size=0.08",
                facecolor="white",
                edgecolor=CLUSTER_COLOR[label] if worked else "#bdbdbd",
                lw=1.5 if worked else 0.7,
            )
        )
        rec = rows[label]
        ax_d.text(
            cx,
            cy + 0.55,
            f"cluster {label}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=CLUSTER_COLOR.get(label, "0.15"),
        )
        ax_d.text(
            cx,
            cy - 0.05,
            edge_name(*BELT[open_at[label]]),
            ha="center",
            va="center",
            fontsize=7.5,
            color="0.15",
        )
        ax_d.text(
            cx,
            cy - 0.62,
            f"{rec.pct_of_cube:.1f}%",
            ha="center",
            va="center",
            fontsize=7.5,
            color="0.35",
        )
    members = [int(c) for c in family["cluster_label"]]
    outside = part[part["n_orientations"] == 1].sort_values("cluster_label")
    outside_r = max(
        float(pair_row(symmetry, int(a), int(b)).r_best_relabelling)
        for a in members
        for b in outside["cluster_label"]
    )
    ax_d.text(
        0.15,
        3.7,
        _family_notes(symmetry, members),
        ha="left",
        va="top",
        fontsize=8,
        color="0.15",
    )
    outside_bits = [
        f"{int(rec.cluster_label)} ({rec.structure})" for rec in outside.itertuples(index=False)
    ]
    ax_d.text(
        0.15,
        2.35,
        "Outside the family: "
        + "; ".join(outside_bits)
        + f".  Strongest link to the family is r = {outside_r:.2f}.",
        ha="left",
        va="top",
        fontsize=8,
        color="0.25",
    )
    ax_d.text(
        0.15,
        0.85,
        "Bars in (b) and (c) are apo-frame shares with that d$_{1}$ ≥ 7 Å.  "
        "Profile r correlates all 960 site-pair distances; "
        f"{r_best:.2f} is the best of the six S$_{6}$ scores.",
        ha="left",
        va="top",
        fontsize=7.5,
        color="0.4",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    catalog = Path(args.catalog_dir)
    with RunContext.from_namespace(args, name="plot_orientation_merge_example"):
        clusters = pd.read_csv(catalog / "clusters_ranked.csv")
        symmetry = pd.read_csv(catalog / "all_cluster_symmetry.csv")
        symmetry = symmetry[symmetry["cohort"] == CUBE]
        fig_path = plot_figure(clusters, symmetry, Path(args.out_dir) / "bmhpm_orientation_merge.png")
        manuscript_dir = str(args.manuscript_dir).strip()
        if manuscript_dir:
            dest_dir = Path(manuscript_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in (fig_path, fig_path.with_suffix(".pdf")):
                dest = dest_dir / src.name
                dest.write_bytes(src.read_bytes())
                print(f"Copied figure to {dest}")
        print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
