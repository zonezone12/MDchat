"""Transition-frequency diagrams in Murata's layout, and for our clusters.

The metastructure figure counts A / B / C1 / C2 / other on every 1 ps
snapshot and scales the counts so a panel sums to 1,000 (per ns of
trajectory, Murata's unit). The runs start compact and empty and last
5 ns, so both clocks are relaxation counts, not equilibrium rates.
The cluster figure uses the same counting. The segment figure counts one
step per neighboring segment at the selected k. Node text is the Figure
4b tick: cluster number, then the short state used on the contact-signature
panels.

The iodine pair keeps those counts and adds the 20 ps iodide events.
Each box gives the fraction of its frames with iodide in the cavity over
the same 5 ns window, a relaxation occupancy, and the entries (+) and
exits (-) that begin while the replica is in that state. A purple numeral on an arrow is how many of those events fall
within 50 ps of that cluster change.

Example
-------
python scripts/plot_transition_frequencies.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir
from src.utils.run_log import RunContext

META_ORDER = ("A", "B", "C1", "C2", "other")
PUBLISHED = ("BMMpM", "BMHpM", "BHHpM")
SERIES = ("BMMpM", "BMHpM", "BHHpM", "BMMpH", "BMHpH", "BHHpH")
# One-SE k. Same cut as scripts/transition_frequencies.py.
SELECTED_K = {
    "BHHpH": 4,
    "BHHpM": 4,
    "BMHpH": 6,
    "BMHpM": 10,
    "BMMpH": 3,
    "BMMpM": 4,
}
MURATA_NAME = {"BMMpM": "1₆", "BMHpM": "2₆", "BHHpM": "3₆"}
BLUE = "#2166ac"
ORANGE = "#e08214"
IODINE = "#6a3d9a"
INK = "0.15"
IODINE_MIN_DWELL = 20
IODINE_EDGE_PS = 50


def _signatures():
    path = ROOT / "scripts" / "plot_contact_signatures.py"
    spec = importlib.util.spec_from_file_location("plot_contact_signatures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _rates(table: pd.DataFrame, cube: str, kind: str) -> dict[tuple[str, str], float]:
    sub = table[
        (table["cohort"] == cube)
        & (table["label_kind"] == kind)
        & (table["guest_filter"] == "all")
    ]
    return {
        (str(rec.from_state), str(rec.to_state)): float(rec.per_ns)
        for rec in sub.itertuples(index=False)
    }


def _fmt_stay(value: float) -> str:
    return str(int(round(value)))


def _fmt_change(value: float) -> str:
    if value >= 10:
        return str(int(round(value)))
    if value >= 1:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _fmt_dwell(ns: float) -> str:
    return f"{ns:.1f} ns"


def _node(
    ax,
    xy,
    text,
    width,
    height,
    fontsize,
    caption: str | None = None,
    note: str | None = None,
) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.06",
            facecolor="white",
            edgecolor=INK,
            linewidth=1.0,
            zorder=4,
        )
    )
    n_extra = (caption is not None) + (note is not None)
    longest = max(len(line) for line in str(text).split("\n"))
    name_size = fontsize - 0.7 if note is not None and longest >= 11 else fontsize
    roomy = height > 1.5
    tall = roomy and height >= 2.4 and n_extra == 2
    name_dy = (0.62 if tall else 0.46 if n_extra == 2 else 0.14 if n_extra == 1 else 0.0) if roomy else (
        0.30 if n_extra == 2 else 0.08 if n_extra == 1 else 0.0
    )
    ax.text(
        x,
        y + name_dy,
        text,
        ha="center",
        va="center",
        fontsize=name_size,
        color=INK,
        zorder=5,
        linespacing=0.95,
    )
    detail = fontsize - 1.4 if roomy else None
    if caption and note:
        ax.text(
            x,
            y - height / 2 + (0.92 if tall else 0.62 if roomy else 0.40),
            caption,
            ha="center",
            va="center",
            fontsize=6.0 if detail is None else detail,
            color="0.35",
            zorder=5,
        )
        ax.text(
            x,
            y - height / 2 + (0.34 if tall else 0.28 if roomy else 0.16),
            note,
            ha="center",
            va="center",
            fontsize=5.8 if detail is None else detail,
            color=IODINE,
            zorder=5,
        )
    elif caption:
        ax.text(
            x,
            y - height / 2 + (0.28 if roomy else 0.13),
            caption,
            ha="center",
            va="center",
            fontsize=6.5 if detail is None else detail,
            color="0.35",
            zorder=5,
        )
    elif note:
        ax.text(
            x,
            y - height / 2 + (0.28 if roomy else 0.16),
            note,
            ha="center",
            va="center",
            fontsize=6.2 if detail is None else detail,
            color=IODINE,
            zorder=5,
        )


def _self_loop(
    ax,
    xy,
    value,
    *,
    above: bool,
    reach: float,
    side: str | None = None,
    pad: float = 0.28,
    half_width: float = 0.72,
    fontsize: float = 8.5,
) -> None:
    x, y = xy
    if side is None:
        side = "above" if above else "below"
    if side in ("right", "left"):
        sign = 1.0 if side == "right" else -1.0
        x0 = x + sign * (half_width + 0.22)
        arc = FancyArrowPatch(
            (x0, y - 0.16),
            (x0, y + 0.16),
            connectionstyle=f"arc3,rad={2.2 * sign}",
            arrowstyle="-|>",
            mutation_scale=9,
            lw=1.05,
            color=INK,
            zorder=3,
        )
        ax.add_patch(arc)
        ax.text(
            x0 + sign * 0.42,
            y,
            _fmt_stay(value),
            ha="left" if side == "right" else "right",
            va="center",
            fontsize=fontsize,
            color=INK,
            zorder=6,
        )
        return
    sign = 1.0 if side == "above" else -1.0
    y0 = y + sign * pad
    arc = FancyArrowPatch(
        (x - 0.16, y0),
        (x + 0.16, y0),
        connectionstyle=f"arc3,rad={-2.2 * sign}",
        arrowstyle="-|>",
        mutation_scale=9,
        lw=1.05,
        color=INK,
        zorder=3,
    )
    ax.add_patch(arc)
    ax.text(
        x,
        y + sign * reach,
        _fmt_stay(value),
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        zorder=6,
    )


def _arrow(ax, start, end, color, rad: float = 0.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.15,
            color=color,
            zorder=2,
        )
    )


def _ortho_arrow(ax, points, color) -> None:
    """Axis-aligned arrow. Each bend is a right angle."""
    verts = [np.asarray(point, float) for point in points]
    keep = [verts[0]]
    for point in verts[1:]:
        if float(np.hypot(*(point - keep[-1]))) > 1e-6:
            keep.append(point)
    if len(keep) < 2:
        return
    codes = [MPath.MOVETO, *([MPath.LINETO] * (len(keep) - 1))]
    ax.add_patch(
        FancyArrowPatch(
            path=MPath(keep, codes),
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.15,
            color=color,
            zorder=2,
        )
    )


def _clear_of_boxes(point, positions, width: float, height: float, margin: float = 0.06) -> bool:
    """True when *point* sits outside every state box."""
    px, py = float(point[0]), float(point[1])
    half_w = width / 2 + margin
    half_h = height / 2 + margin
    for nx, ny in positions.values():
        if abs(px - nx) < half_w and abs(py - ny) < half_h:
            return False
    return True


def _gap_along(start, end, positions, width: float, height: float) -> np.ndarray:
    """A point on the open shaft, off any state the chord passes through."""
    start = np.asarray(start, float)
    end = np.asarray(end, float)
    for t in (0.5, 0.4, 0.6, 0.3, 0.7, 0.22, 0.78, 0.15, 0.85):
        point = start + t * (end - start)
        if _clear_of_boxes(point, positions, width, height, margin=0.02):
            return point
    return start + 0.5 * (end - start)


def _gap_bezier(start, control, end, positions, width: float, height: float) -> np.ndarray:
    for t in (0.5, 0.4, 0.6, 0.3, 0.7, 0.22, 0.78):
        point = _bezier(start, control, end, t)
        if _clear_of_boxes(point, positions, width, height, margin=0.02):
            return point
    return _bezier(start, control, end, 0.5)


def _rate_text(ax, xy, text, color, size: float | None = None) -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=getattr(_rate_text, "size", 8) if size is None else size,
        color=color,
        zorder=6,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.15, "alpha": 0.92},
    )


def _horizontal_pair(ax, x_left, x_right, y, forward: float, backward: float, *, min_rate: float) -> None:
    """Blue left → right above the line. Orange right → left below it."""
    if forward >= min_rate:
        _arrow(ax, (x_left, y + 0.13), (x_right, y + 0.13), BLUE)
        _rate_text(ax, ((x_left + x_right) / 2, y + 0.34), _fmt_change(forward), BLUE)
    if backward >= min_rate:
        _arrow(ax, (x_right, y - 0.13), (x_left, y - 0.13), ORANGE)
        _rate_text(ax, ((x_left + x_right) / 2, y - 0.34), _fmt_change(backward), ORANGE)


def _arc_pair(ax, start, end, forward: float, backward: float, label_at, *, min_rate: float, rad: float) -> None:
    """Blue follows start → end. The number sits at label_at, off the arc."""
    lx, ly = label_at
    if forward >= min_rate:
        _arrow(ax, start, end, BLUE, rad + 0.06)
        _rate_text(ax, (lx, ly + 0.18), _fmt_change(forward), BLUE)
    if backward >= min_rate:
        _arrow(ax, end, start, ORANGE, rad - 0.06)
        _rate_text(ax, (lx, ly - 0.18), _fmt_change(backward), ORANGE)


def plot_metastructures(
    table: pd.DataFrame,
    out: Path,
    cubes: tuple[str, ...] = PUBLISHED,
    letters: str = "abc",
) -> Path:
    """Three panels in the published node order: A, B, C1, C2, other."""
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.4))
    xs = {"A": 0.0, "B": 2.15, "C1": 4.3, "C2": 6.45}
    y_row = 0.0
    y_other = -2.85
    width, height = 1.05, 0.50
    for ax, letter, cube in zip(axes, letters, cubes):
        rates = _rates(table, cube, "murata")
        ax.set_xlim(-1.05, 7.85)
        ax.set_ylim(-3.85, 2.15)
        ax.set_aspect("equal")
        ax.axis("off")
        for name in ("A", "B", "C1", "C2"):
            _node(ax, (xs[name], y_row), name, width, height, 12)
            _self_loop(ax, (xs[name], y_row), rates.get((name, name), 0.0), above=True, reach=0.82)
        _node(ax, (xs["C2"], y_other), "other", width, height, 11)
        _self_loop(ax, (xs["C2"], y_other), rates.get(("other", "other"), 0.0), above=False, reach=0.78)

        for src, dst in (("A", "B"), ("B", "C1"), ("C1", "C2")):
            _horizontal_pair(
                ax,
                xs[src] + width / 2,
                xs[dst] - width / 2,
                y_row,
                rates.get((src, dst), 0.0),
                rates.get((dst, src), 0.0),
                min_rate=0.5,
            )
        # When B or C1 also connects to "other", the B ↔ C2 bypass goes over the row.
        bypass_above = max(
            rates.get(("B", "other"), 0.0),
            rates.get(("other", "B"), 0.0),
            rates.get(("C1", "other"), 0.0),
            rates.get(("other", "C1"), 0.0),
        ) >= 0.5
        if bypass_above:
            _arc_pair(
                ax,
                (xs["B"], y_row + height / 2 + 0.02),
                (xs["C2"], y_row + height / 2 + 0.02),
                rates.get(("B", "C2"), 0.0),
                rates.get(("C2", "B"), 0.0),
                ((xs["B"] + xs["C2"]) / 2, 1.55),
                min_rate=0.5,
                rad=0.18,
            )
        else:
            _arc_pair(
                ax,
                (xs["B"], y_row - height / 2 - 0.02),
                (xs["C2"] - 0.05, y_row - height / 2 - 0.02),
                rates.get(("B", "C2"), 0.0),
                rates.get(("C2", "B"), 0.0),
                ((xs["B"] + xs["C2"]) / 2, -0.78),
                min_rate=0.5,
                rad=-0.12,
            )
        # C2 ↔ other runs down the right side of the pair, numbers to the right.
        _arc_pair(
            ax,
            (xs["C2"] + 0.22, y_row - height / 2),
            (xs["C2"] + 0.22, y_other + height / 2),
            rates.get(("C2", "other"), 0.0),
            rates.get(("other", "C2"), 0.0),
            (xs["C2"] + 0.78, (y_row + y_other) / 2),
            min_rate=0.5,
            rad=0.0,
        )
        # B and C1 reach "other" on separated curves, numbers beside each curve.
        for src, label_xy, rad in (
            ("B", (xs["B"] + 0.15, -1.85), 0.16),
            ("C1", (xs["C1"] - 0.15, -1.55), -0.05),
        ):
            _arc_pair(
                ax,
                (xs[src] + 0.1, y_row - height / 2),
                (xs["C2"] - 0.35, y_other + height / 2),
                rates.get((src, "other"), 0.0),
                rates.get(("other", src), 0.0),
                label_xy,
                min_rate=0.5,
                rad=rad,
            )
        system = MURATA_NAME.get(cube, "")
        title = f"{cube}   {system}" if system else cube
        ax.text(0.0, 1.01, f"({letter})", transform=ax.transAxes, fontweight="bold", fontsize=12, va="bottom")
        ax.text(0.52, 1.01, title, transform=ax.transAxes, ha="center", fontsize=11, va="bottom")
    fig.suptitle(
        "Metastructure transitions per ns  ·  every 1 ps snapshot, panel sums to 1,000",
        fontsize=11,
        y=1.01,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cluster_heatmaps(table: pd.DataFrame, out: Path) -> Path:
    """Same frequencies as a matrix, in the Figure 4b cluster order."""
    _style()
    signatures = _signatures()
    catalog = ROOT / "output" / "metastable_structures_physical_k"
    clusters = pd.read_csv(catalog / "clusters_ranked.csv")
    symmetry = pd.read_csv(catalog / "all_cluster_symmetry.csv")
    belt = pd.read_csv(catalog / "min_share_0" / "segment_belt.csv")

    fig, axes = plt.subplots(2, 3, figsize=(14.8, 11.0), constrained_layout=True)
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#f4f4f4")
    norm = Normalize(vmin=0.0, vmax=0.16)
    image = None
    for ax, letter, cube in zip(axes.ravel(), "abcdef", SERIES):
        order, _corr = signatures.panel_b_table(clusters, symmetry, belt, cube)
        labels = [int(v) for v in order["cluster_label"]]
        ticks = [str(v) for v in order["tick"]]
        rates = _rates(table, cube, "cluster")
        n = len(labels)
        matrix = np.full((n, n), np.nan)
        for i, src in enumerate(labels):
            for j, dst in enumerate(labels):
                matrix[i, j] = rates.get((str(src), str(dst)), 0.0)
        stays = np.diag(matrix).copy()
        shown = matrix.copy()
        np.fill_diagonal(shown, np.nan)
        image = ax.imshow(shown, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        size = 8.0 if n <= 4 else 6.6
        ax.set_xticklabels(ticks, fontsize=size, rotation=40, ha="right")
        ax.set_yticklabels(ticks, fontsize=size)
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", length=0)
        for i in range(n):
            for j in range(n):
                if i == j:
                    ax.text(j, i, _fmt_stay(stays[i]), ha="center", va="center", fontsize=size, color=INK)
                    continue
                value = float(matrix[i, j])
                if value < 0.01:
                    continue
                ink = "white" if norm(value) > 0.55 else INK
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=size - 0.6, color=ink)
        system = MURATA_NAME.get(cube, "")
        title = f"{cube}   {system}" if system else cube
        ax.set_title(title, fontsize=11)
        ax.text(-0.18, 1.06, f"({letter})", transform=ax.transAxes, fontweight="bold", fontsize=12, va="bottom")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        fraction=0.02,
        pad=0.02,
        label="changes per ns",
    )
    fig.suptitle(
        "Cluster transition frequency per ns  ·  diagonal is stays  ·  labels match Figure 4b",
        fontsize=12,
    )
    del image
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def _box_edge(start, end, width: float, height: float):
    """Point where the segment from start to end leaves a box centered on start."""
    delta = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    scale_x = (width / 2) / abs(delta[0]) if abs(delta[0]) > 1e-8 else np.inf
    scale_y = (height / 2) / abs(delta[1]) if abs(delta[1]) > 1e-8 else np.inf
    return np.asarray(start, dtype=float) + delta * min(scale_x, scale_y)


def _flow_positions(order: pd.DataFrame, rates: dict) -> dict[int, tuple[float, float]]:
    """Figure 4b order. A long orientation family sits on the top row."""
    family = [int(v) for v in order.loc[order["in_family"], "cluster_label"]]
    singles = [int(v) for v in order.loc[~order["in_family"], "cluster_label"]]
    positions: dict[int, tuple[float, float]] = {}
    if len(family) >= 4:
        for i, label in enumerate(family):
            positions[label] = ((i - (len(family) - 1) / 2) * 2.15, 1.85)
        for i, label in enumerate(singles):
            positions[label] = ((i - (len(singles) - 1) / 2) * 3.15, -1.7)
        return positions
    labels = [int(v) for v in order["cluster_label"]]
    outgoing = {
        label: sum(
            v for (src, dst), v in rates.items() if src == str(label) and dst != src
        )
        for label in labels
    }
    hub = max(labels, key=lambda label: (outgoing[label], -labels.index(label)))
    others = [label for label in labels if label != hub]
    positions[hub] = (0.0, 0.0)
    for i, label in enumerate(others):
        positions[label] = (3.2, ((len(others) - 1) / 2 - i) * 1.85)
    return positions


def _column_beside_row(
    positions: dict[int, tuple[float, float]],
    gap: float = 4.4,
) -> dict[int, tuple[float, float]]:
    """Put the lower group in a column whose middle lines up with the row.

    The line between the two midpoints is horizontal, so it is perpendicular
    to the column.
    """
    top = sorted((lab for lab, (_x, y) in positions.items() if y > 0), key=lambda lab: positions[lab][0])
    bottom = sorted((lab for lab, (_x, y) in positions.items() if y <= 0), key=lambda lab: positions[lab][0])
    placed: dict[int, tuple[float, float]] = {}
    for i, lab in enumerate(top):
        placed[lab] = (2.8 + i * 5.0, 0.0)
    mid = (len(bottom) - 1) / 2
    for i, lab in enumerate(bottom):
        placed[lab] = (-7.4, (mid - i) * gap)
    return placed


def _rim_point(center, side: str, width: float, height: float, index: int, count: int) -> np.ndarray:
    """A point on one side of a box, spread when several arrows share that side."""
    cx, cy = float(center[0]), float(center[1])
    frac = 0.5 if count <= 1 else 0.2 + 0.6 * index / (count - 1)
    if side == "right":
        return np.array([cx + width / 2, cy + (frac - 0.5) * height * 0.75])
    y = cy + height / 2 if side == "top" else cy - height / 2
    return np.array([cx + (frac - 0.5) * width * 0.7, y])


def _bmhpm_cross_plan(positions, rates, iodine_edges, min_count):
    """Upper column nodes meet the row on its top edge. Lower nodes meet the bottom."""
    col_x = min(xy[0] for xy in positions.values())
    column = [lab for lab, xy in positions.items() if abs(xy[0] - col_x) < 0.3]
    row = [lab for lab in positions if lab not in set(column)]
    mid = float(np.mean([positions[lab][1] for lab in row]))
    plan: dict[tuple[int, int], str] = {}
    groups: dict[tuple, list] = {}
    for src in column:
        for dst in row:
            forward = rates.get((str(src), str(dst)), 0.0)
            backward = rates.get((str(dst), str(src)), 0.0)
            iodine = int(iodine_edges.get((str(src), str(dst)), 0)) + int(
                iodine_edges.get((str(dst), str(src)), 0)
            )
            if max(forward, backward) < min_count and iodine == 0:
                continue
            side = "top" if positions[src][1] >= mid else "bottom"
            plan[(src, dst)] = side
            groups.setdefault((src, side), []).append(dst)
            groups.setdefault((dst, side), []).append(src)
    slot: dict[tuple, tuple[int, int]] = {}
    for (node, side), partners in groups.items():
        ordered = sorted(set(partners), key=lambda lab: (positions[lab][0], positions[lab][1]))
        for index, partner in enumerate(ordered):
            slot[(node, partner, side)] = (index, len(ordered))
    return plan, slot, column, mid


def _segment_counts(cube: str) -> dict[tuple[str, str], float]:
    """Selected-k segment pairs. A cell is one changepoint followed by the next segment."""
    k = SELECTED_K[cube]
    changepoints = endpoint_changepoints_dir(ROOT / "output", cube, own_elbow=True)
    path = (
        changepoints
        / "clusters"
        / "endpoint"
        / "by_k"
        / f"k_{k:02d}"
        / "state_transition_matrix.csv"
    )
    matrix = pd.read_csv(path, index_col=0)
    counts: dict[tuple[str, str], float] = {}
    for src in matrix.index:
        for dst in matrix.columns:
            a = str(src).replace("state_", "")
            b = str(dst).replace("state_", "")
            counts[(a, b)] = float(matrix.loc[src, dst])
    return counts


def _mean_dwells_ns(cube: str) -> dict[int, float]:
    """Mean observed dwell at the selected k.

    A dwell is a run of neighboring segments with the same label. The endpoint
    signal is sampled every 5 ps, and the next segment starts 5 ps after the
    previous end, so the dwell lasts through that last step.
    """
    k = SELECTED_K[cube]
    changepoints = endpoint_changepoints_dir(ROOT / "output", cube, own_elbow=True)
    path = (
        changepoints
        / "clusters"
        / "endpoint"
        / "by_k"
        / f"k_{k:02d}"
        / "segments_clustered.csv"
    )
    segments = pd.read_csv(path)
    dwells: dict[int, list[float]] = {}
    stride_ps = 5.0
    for _traj, sub in segments.groupby("traj_id", sort=False):
        sub = sub.sort_values(["start_frame", "segment_id"])
        label = None
        start = end = None
        def _close() -> None:
            if label is None:
                return
            dwells.setdefault(label, []).append(float(end - start) + stride_ps)

        for rec in sub.itertuples(index=False):
            gap = int(rec.start_frame) - int(end) if end is not None else 0
            if label is None or int(rec.cluster_label) != label or gap > stride_ps:
                _close()
                label = int(rec.cluster_label)
                start = int(rec.start_frame)
                end = int(rec.end_frame)
            else:
                end = int(rec.end_frame)
        _close()
    return {lab: float(np.mean(times)) / 1000.0 for lab, times in dwells.items()}


def _strip_traj(traj_id: str, cube: str) -> str:
    text = str(traj_id)
    prefix = f"{cube}_"
    return text[len(prefix) :] if text.startswith(prefix) else text


def _labeled_spans(segments: pd.DataFrame) -> list[tuple[int, int, int]]:
    """Inclusive frame spans. A gap of at most 5 ps keeps the previous label.

    Same rule as ``transition_frequencies._paint_clusters``: the changepoint
    is one 1 ps step, and the frames between end and the next start stay with
    the segment that just ended.
    """
    rows = segments.sort_values(["start_frame", "segment_id"])
    bounds = [
        (int(rec.start_frame), int(rec.end_frame), int(rec.cluster_label))
        for rec in rows.itertuples(index=False)
    ]
    spans: list[tuple[int, int, int]] = []
    for i, (start, end, label) in enumerate(bounds):
        hi = end
        if i + 1 < len(bounds):
            gap = bounds[i + 1][0] - end
            if 0 < gap <= 5:
                hi = bounds[i + 1][0] - 1
        spans.append((start, hi, label))
    return spans


def _cluster_changes(segments: pd.DataFrame) -> list[tuple[int, int, int]]:
    """``(boundary frame, source, destination)`` where the label actually changes."""
    rows = list(segments.sort_values(["start_frame", "segment_id"]).itertuples(index=False))
    changes = []
    for prev, nxt in zip(rows, rows[1:]):
        src = int(prev.cluster_label)
        dst = int(nxt.cluster_label)
        if src == dst:
            continue
        changes.append((int(nxt.start_frame), src, dst))
    return changes


def _fmt_iodine(occupancy: float, n_entry: int, n_exit: int) -> str:
    percent = f"{100.0 * occupancy:.0f}%"
    if n_entry == 0 and n_exit == 0:
        return percent
    return f"{percent}  +{n_entry}-{n_exit}"


def _compute_iodine() -> tuple[dict[str, dict[int, str]], dict[str, dict[tuple[str, str], int]], pd.DataFrame, pd.DataFrame]:
    """Occupancy and 20 ps iodide events on the selected-k endpoint clusters.

    An event is placed in the cluster that contains its frame. It is also
    counted on a change when that frame is within 50 ps of the boundary.
    """
    dwells = pd.read_csv(ROOT / "output" / "guest_event_cotiming" / "guest_dwells.csv")
    dwells = dwells[(dwells["min_dwell"] == IODINE_MIN_DWELL) & (dwells["event"] != "start")]
    notes: dict[str, dict[int, str]] = {}
    edges: dict[str, dict[tuple[str, str], int]] = {}
    state_rows: list[dict] = []
    edge_rows: list[dict] = []
    for cube in SERIES:
        k = SELECTED_K[cube]
        changepoints = endpoint_changepoints_dir(ROOT / "output", cube, own_elbow=True)
        segments = pd.read_csv(
            changepoints / "clusters" / "endpoint" / "by_k" / f"k_{k:02d}" / "segments_clustered.csv"
        )
        by_traj = {str(tid): sub for tid, sub in segments.groupby("traj_id", sort=False)}
        spans = {tid: _labeled_spans(sub) for tid, sub in by_traj.items()}
        changes = {tid: _cluster_changes(sub) for tid, sub in by_traj.items()}
        occupied = {label: 0 for label in segments["cluster_label"].astype(int).unique()}
        labeled = {label: 0 for label in occupied}
        feat_dir = ROOT / "output" / "gsa_features_by_cube" / cube
        for path in sorted(feat_dir.glob("*_gsa_features.csv")):
            frame = pd.read_csv(path, usecols=["traj_id", "frame", "n_guest_inside_cavity"])
            if frame.empty:
                continue
            tid = _strip_traj(str(frame["traj_id"].iloc[0]), cube)
            traj_spans = spans.get(tid)
            if not traj_spans:
                continue
            frame = frame.sort_values("frame")
            frames = frame["frame"].to_numpy(dtype=np.int64)
            guest = pd.to_numeric(frame["n_guest_inside_cavity"], errors="coerce").to_numpy() >= 1
            for start, end, label in traj_spans:
                lo = int(np.searchsorted(frames, start, side="left"))
                hi = int(np.searchsorted(frames, end, side="right"))
                if hi <= lo:
                    continue
                labeled[label] = labeled.get(label, 0) + (hi - lo)
                occupied[label] = occupied.get(label, 0) + int(guest[lo:hi].sum())
        entries = {label: 0 for label in labeled}
        exits = {label: 0 for label in labeled}
        edge_counts: dict[tuple[str, str], int] = {}
        outside = 0
        sub_events = dwells[dwells["cube"] == cube]
        for rec in sub_events.itertuples(index=False):
            tid = str(rec.traj_id)
            traj_spans = spans.get(tid)
            if not traj_spans:
                outside += 1
                continue
            frame_no = int(rec.start)
            label = next((lab for start, end, lab in traj_spans if start <= frame_no <= end), None)
            if label is None:
                outside += 1
                continue
            if rec.event == "entry":
                entries[label] = entries.get(label, 0) + 1
            elif rec.event == "exit":
                exits[label] = exits.get(label, 0) + 1
            nearest = None
            for boundary, src, dst in changes.get(tid, []):
                distance = abs(frame_no - boundary)
                if distance <= IODINE_EDGE_PS and (nearest is None or distance < nearest[0]):
                    nearest = (distance, src, dst)
            if nearest is not None:
                key = (str(nearest[1]), str(nearest[2]))
                edge_counts[key] = edge_counts.get(key, 0) + 1
        notes[cube] = {
            label: _fmt_iodine(
                occupied.get(label, 0) / labeled[label] if labeled.get(label, 0) else float("nan"),
                int(entries.get(label, 0)),
                int(exits.get(label, 0)),
            )
            for label in labeled
            if labeled.get(label, 0)
        }
        edges[cube] = edge_counts
        for label in sorted(labeled):
            n = labeled.get(label, 0)
            state_rows.append(
                {
                    "cohort": cube,
                    "cluster_label": int(label),
                    "labeled_frames": int(n),
                    "occupied_frames": int(occupied.get(label, 0)),
                    "occupancy": (occupied.get(label, 0) / n) if n else np.nan,
                    "n_entries": int(entries.get(label, 0)),
                    "n_exits": int(exits.get(label, 0)),
                    "n_events_outside_clustered_replica": int(outside),
                }
            )
        for (src, dst), count in sorted(edge_counts.items()):
            edge_rows.append(
                {
                    "cohort": cube,
                    "from_state": src,
                    "to_state": dst,
                    "n_events_within_50ps": int(count),
                }
            )
        print(
            f"iodine {cube}: entries {sum(entries.values())} exits {sum(exits.values())} "
            f"outside {outside} co-timed changes {sum(edge_counts.values())}",
            flush=True,
        )
    state_table = pd.DataFrame(state_rows)
    edge_table = pd.DataFrame(edge_rows)
    out_dir = ROOT / "output" / "transition_frequencies"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_table.to_csv(out_dir / "iodine_on_states.csv", index=False)
    edge_table.to_csv(out_dir / "iodine_on_changes.csv", index=False)
    return notes, edges, state_table, edge_table


_IODINE_CACHE: tuple | None = None


def _iodine():
    global _IODINE_CACHE
    if _IODINE_CACHE is None:
        _IODINE_CACHE = _compute_iodine()
    return _IODINE_CACHE


def _loop_side(xy, positions: dict[int, tuple[float, float]], *, two_rows: bool) -> str:
    """Keep a stay-count off the arrows that leave the node."""
    if two_rows:
        return "above" if xy[1] >= 0 else "below"
    x, y = xy
    same_column = [
        other
        for other in positions.values()
        if abs(other[0] - x) < 0.3 and abs(other[1] - y) > 0.3
    ]
    occupied_above = any(other[1] > y for other in same_column)
    occupied_below = any(other[1] < y for other in same_column)
    if occupied_above and occupied_below:
        return "right"
    if occupied_above:
        return "below"
    return "above"


def _chord_hits_node(start, end, node, width: float, height: float) -> bool:
    """True when the open chord runs through another node's box."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    node = np.asarray(node, dtype=float)
    delta = end - start
    length2 = float(np.dot(delta, delta))
    if length2 < 1e-8:
        return False
    t = float(np.dot(node - start, delta) / length2)
    if t <= 0.12 or t >= 0.88:
        return False
    closest = start + t * delta
    offset = np.abs(node - closest)
    return bool(offset[0] < width / 2 + 0.08 and offset[1] < height / 2 + 0.08)


def _arc_geometry(start, end, positions, bulge: float):
    """Quadratic arc bowed away from the other nodes. Returns rad and control point."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    length = float(np.hypot(*delta)) or 1.0
    away = (start + end) / 2 - np.mean(list(positions.values()), axis=0)
    base = np.array([delta[1], -delta[0]])
    sign = 1.0 if float(np.dot(base, away)) >= 0 else -1.0
    rad = sign * (2.0 * bulge / length)
    control = (start + end) / 2 + np.array([rad * delta[1], -rad * delta[0]])
    return rad, control


def _bezier(start, control, end, t: float) -> np.ndarray:
    start = np.asarray(start, dtype=float)
    control = np.asarray(control, dtype=float)
    end = np.asarray(end, dtype=float)
    return (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t ** 2 * end


def _flow_figure(
    counts_for,
    out: Path,
    *,
    min_count: float,
    suptitle: str,
    show_k: bool,
    arrow_fmt=_fmt_change,
    clear_crossings: bool = False,
    dwell_for=None,
        iodine_for=None,
    iodine_edges_for=None,
    only: str | None = None,
) -> Path:
    """Arrow diagram of cluster changes, with the Figure 4b names."""
    _style()
    signatures = _signatures()
    catalog = ROOT / "output" / "metastable_structures_physical_k"
    clusters = pd.read_csv(catalog / "clusters_ranked.csv")
    symmetry = pd.read_csv(catalog / "all_cluster_symmetry.csv")
    belt = pd.read_csv(catalog / "min_share_0" / "segment_belt.csv")

    has_iodine = iodine_for is not None
    single = only is not None
    if single:
        wide = only == "BMHpM"
        fig = plt.figure(figsize=(22.0, 15.5) if wide else (15.4, 12.2))
        grid = fig.add_gridspec(1, 1)
        slots = {only: grid[0, 0]}
    else:
        fig = plt.figure(figsize=(14.6, 12.4))
        grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.15], hspace=0.28, wspace=0.18)
        # 2₆ has ten clusters, so it takes the full width of the bottom row.
        slots = {
            "BMMpM": grid[0, 0],
            "BHHpM": grid[0, 1],
            "BMMpH": grid[0, 2],
            "BMHpH": grid[1, 0],
            "BHHpH": grid[1, 1],
            "BMHpM": grid[2, :],
        }
    if dwell_for and has_iodine:
        height = 1.26
    elif dwell_for:
        height = 0.78
    elif has_iodine:
        height = 0.92
    else:
        height = 0.58
    width = 1.58 if has_iodine else 1.45
    if single:
        # The segment boxes also carry a dwell line, so they are a little taller.
        width, height = ((2.6, 2.6) if dwell_for else (2.6, 2.25)) if only == "BMHpM" else (2.2, 1.85)
    letters = {
        "BMMpM": "a",
        "BMHpM": "b",
        "BHHpM": "c",
        "BMMpH": "d",
        "BMHpH": "e",
        "BHHpH": "f",
    }
    for cube in slots:
        _rate_text.size = 14.5 if single else 8
        ax = fig.add_subplot(slots[cube])
        letter = letters[cube]
        order, _corr = signatures.panel_b_table(clusters, symmetry, belt, cube)
        rates = counts_for(cube)
        positions = _flow_positions(order, rates)
        if single and cube == "BMHpM":
            # 0, 3, 5, 9 stand in a column, perpendicular to the belt-break row.
            # Taller boxes need a wider gap so the inner pair clears the row.
            positions = _column_beside_row(positions, gap=6.0 if height >= 2.4 else 4.4)
        elif single:
            positions = {lab: (x * 1.75, y * 1.75) for lab, (x, y) in positions.items()}
        col_x = min(xy[0] for xy in positions.values()) if single and cube == "BMHpM" else None
        col_width = width * 1.5

        def _nw(xy) -> float:
            if col_x is not None and abs(float(np.asarray(xy, float)[0]) - col_x) < 0.3:
                return col_width
            return width

        ticks = {int(rec.cluster_label): str(rec.tick) for rec in order.itertuples(index=False)}
        dwells = dwell_for(cube) if dwell_for else {}
        notes = iodine_for(cube) if iodine_for else {}
        iodine_edges = iodine_edges_for(cube) if iodine_edges_for else {}
        ys = [y for _x, y in positions.values()]
        # A real two-row layout has a row of states on top. A vertical stack
        # also spans positive and negative y, and its middle loop must go sideways.
        family_row = sum(y > 0.5 for y in ys) >= 3 and any(y < -0.5 for y in ys)
        if has_iodine:
            loop_pad = height / 2 + 0.08
            loop_reach = loop_pad + 0.56
        else:
            loop_reach = 1.12 if clear_crossings else 0.78
            loop_pad = height / 2 + 0.06 if dwell_for else 0.28
        for label, xy in positions.items():
            caption = _fmt_dwell(dwells[label]) if label in dwells else None
            _node(
                ax,
                xy,
                ticks[label],
                _nw(xy),
                height,
                13 if single else (6.2 if has_iodine else 7.2),
                caption=caption,
                note=notes.get(label),
            )
            if single and cube == "BMHpM":
                # Stay counts sit on the left of every box, off the arrows that
                # enter the row from above and below.
                side = "left"
            elif clear_crossings or has_iodine:
                side = _loop_side(xy, positions, two_rows=family_row)
            else:
                side = "above" if (xy[1] >= 0 or not family_row) else "below"
            pad_i = loop_pad + (0.18 if has_iodine and side == "below" else 0.0)
            reach_i = loop_reach + (0.34 if has_iodine and side == "below" else 0.0)
            _self_loop(
                ax,
                xy,
                rates.get((str(label), str(label)), 0.0),
                above=side == "above",
                reach=reach_i,
                side=side,
                pad=pad_i,
                half_width=_nw(xy) / 2,
                fontsize=14.5 if single else 8.5,
            )
        labels = list(positions)
        extra_right = 0.0
        placed: list[np.ndarray] = []
        peak_hi = [max(ys)]
        peak_lo = [min(ys)]
        if single and cube == "BMHpM":
            cross_plan, cross_slot, column_labs, row_mid = _bmhpm_cross_plan(
                positions, rates, iodine_edges, min_count
            )
        else:
            cross_plan, cross_slot, column_labs, row_mid = {}, {}, [], 0.0
        lane_of: dict[tuple[int, int, str], float] = {}
        # Every cross edge leaves the vertical middle of its column box, then
        # steps to its own horizontal lane just outside that box.
        by_col: dict[int, list[int]] = {}
        for col, row in cross_plan:
            rows = by_col.setdefault(col, [])
            if row not in rows:
                rows.append(row)
        col_order = sorted(by_col, key=lambda lab: positions[lab][1])
        for col in col_order:
            directed: list[tuple[float, int, str]] = []
            for row in by_col[col]:
                forward = rates.get((str(col), str(row)), 0.0)
                backward = rates.get((str(row), str(col)), 0.0)
                iodine_f = int(iodine_edges.get((str(col), str(row)), 0))
                iodine_b = int(iodine_edges.get((str(row), str(col)), 0))
                if forward >= min_count or iodine_f:
                    directed.append((positions[row][0], row, "blue"))
                if backward >= min_count or iodine_b:
                    directed.append((positions[row][0], row, "orange"))
            directed.sort(key=lambda item: (item[0], 0 if item[2] == "blue" else 1))
            n = len(directed)
            if n == 0:
                continue
            step = 0.58
            span = (n - 1) * step
            center = positions[col][1]
            above = [positions[lab][1] for lab in col_order if positions[lab][1] > center + 0.2]
            below = [positions[lab][1] for lab in col_order if positions[lab][1] < center - 0.2]
            y_max = (min(above) - height / 2 - 0.45) if above else center + span + 1.0
            y_min = (max(below) + height / 2 + 0.45) if below else center - span - 1.0
            if center >= 0:
                y_min = max(y_min, height / 2 + 0.45)
            else:
                y_max = min(y_max, -height / 2 - 0.45)
            bot = center - span / 2
            if bot < y_min:
                bot = y_min
            if bot + span > y_max:
                bot = y_max - span
            # Nearest target takes the lane closer to the row, so a farther
            # horizontal clears that target's vertical drop.
            ys = [bot + i * step for i in range(n)]
            if center < 0:
                ys = list(reversed(ys))
            for (_tx, row, color), y_lane in zip(directed, ys):
                lane_of[(col, row, color)] = y_lane
                peak_hi[0] = max(peak_hi[0], y_lane)
                peak_lo[0] = min(peak_lo[0], y_lane)
        row_left = 0.0
        if column_labs:
            row_left = min(xy[0] for lab, xy in positions.items() if lab not in column_labs) - width / 2
        # Each directed skip on the BMHpM column gets its own vertical lane,
        # far enough apart that the counts are not stacked on one stroke.
        skip_lane: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        if column_labs:
            ordered = sorted(column_labs, key=lambda lab: -positions[lab][1])
            skips = []
            for i, upper in enumerate(ordered):
                for lower in ordered[i + 2 :]:
                    fa = rates.get((str(upper), str(lower)), 0.0)
                    fb = rates.get((str(lower), str(upper)), 0.0)
                    iod = int(iodine_edges.get((str(upper), str(lower)), 0)) + int(
                        iodine_edges.get((str(lower), str(upper)), 0)
                    )
                    if max(fa, fb) < min_count and iod == 0:
                        continue
                    span = abs(positions[upper][1] - positions[lower][1])
                    skips.append((span, upper, lower))
            skips.sort()
            step = 0.95
            for rank, (_span, upper, lower) in enumerate(skips):
                inner = 3.2 + rank * step * 2.0
                # Both directions leave the vertical middle of each box.
                skip_lane[(upper, lower)] = (inner, inner + step, 0.0, 0.0)

        hit_w = col_width if col_x is not None else width

        def _place_rate(point, prefer) -> np.ndarray:
            point = np.asarray(point, float)
            prefer = np.asarray(prefer, float)
            norm = float(np.hypot(*prefer)) or 1.0
            step = prefer / norm
            chosen = point
            for dist in (0.0, 0.28, 0.48, 0.68, 0.9):
                cand = point + step * dist
                if _clear_of_boxes(cand, positions, hit_w, height, margin=0.22):
                    chosen = cand
                    break
            placed.append(chosen)
            return chosen

        def _iodine_callout(anchor, perp, count: int) -> None:
            """Purple count beside the shaft. No dash and no leader."""
            if count <= 0:
                return
            anchor = np.asarray(anchor, float)
            perp = np.asarray(perp, float)
            norm = float(np.hypot(*perp)) or 1.0
            perp = perp / norm
            interior = np.mean(np.asarray(list(positions.values()), float), axis=0)
            first = 1.0 if float(np.dot(perp, interior - anchor)) >= 0 else -1.0
            best = None
            for dist in (0.62, 0.84, 1.08, 1.32):
                for sign in (first, -first):
                    cand = anchor + perp * sign * dist
                    if not _clear_of_boxes(cand, positions, hit_w, height, margin=0.16):
                        continue
                    near = min((float(np.hypot(*(cand - p))) for p in placed), default=9.0)
                    if near < 0.48:
                        continue
                    best = cand
                    break
                if best is not None:
                    break
            if best is None:
                best = anchor + perp * first * 0.9
            _rate_text(ax, tuple(best), str(int(count)), IODINE)
            placed.append(np.asarray(best, float))

        def _counted_arc(a, b, bulge):
            """Bow a counted arrow toward the middle of the panel, off the self-loops."""
            rad, control = _arc_geometry(a, b, positions, bulge)
            if clear_crossings:
                return rad, control
            mid = (np.asarray(a, float) + np.asarray(b, float)) / 2
            interior = np.mean(np.asarray(list(positions.values()), float), axis=0)
            if float(np.dot(np.asarray(control) - mid, interior - mid)) < 0:
                rad = -rad
                control = mid - (np.asarray(control) - mid)
            return rad, control

        for i, src in enumerate(labels):
            for dst in labels[i + 1 :]:
                forward = rates.get((str(src), str(dst)), 0.0)
                backward = rates.get((str(dst), str(src)), 0.0)
                iodine_forward = int(iodine_edges.get((str(src), str(dst)), 0))
                iodine_backward = int(iodine_edges.get((str(dst), str(src)), 0))
                if max(forward, backward) < min_count and iodine_forward + iodine_backward == 0:
                    continue
                p = np.asarray(positions[src], dtype=float)
                q = np.asarray(positions[dst], dtype=float)
                p_lab, q_lab = src, dst
                # Blue travels toward larger x, or upward when x is tied.
                if (q[0], q[1]) < (p[0], p[1]):
                    p, q = q, p
                    p_lab, q_lab = q_lab, p_lab
                    forward, backward = backward, forward
                    iodine_forward, iodine_backward = iodine_backward, iodine_forward
                start = _box_edge(p, q, _nw(p), height)
                end = _box_edge(q, p, _nw(q), height)
                hits_node = any(
                    _chord_hits_node(start, end, node, _nw(node), height)
                    for node in positions.values()
                    if min(
                        float(np.hypot(*(np.asarray(node) - start))),
                        float(np.hypot(*(np.asarray(node) - end))),
                    )
                    > 0.45
                )
                blocked = clear_crossings and hits_node
                # On the BMHpM column, a skip must leave the stack. Otherwise the
                # count is drawn through the boxes between its ends.
                column_skip = single and cube == "BMHpM" and hits_node
                route = cross_plan.get((src, dst)) or cross_plan.get((dst, src))
                if route:
                    # Leave the vertical middle of the column box, then run
                    # horizontally on a lane of that same box.
                    col_lab, row_lab = (src, dst) if (src, dst) in cross_plan else (dst, src)
                    side = route
                    _i_c, _n_c = cross_slot[(col_lab, row_lab, side)]
                    i_r, n_r = cross_slot[(row_lab, col_lab, side)]
                    arrive = _rim_point(positions[row_lab], side, width, height, i_r, n_r)
                    mid_y = positions[col_lab][1]
                    right = positions[col_lab][0] + _nw(positions[col_lab]) / 2

                    def _elbow(color: str, x_shift: float) -> list[np.ndarray]:
                        y = lane_of[(col_lab, row_lab, color)]
                        exit_pt = np.array([right, mid_y])
                        stub = np.array([right + 0.7, mid_y])
                        rise = np.array([right + 0.7, y])
                        corner = np.array([arrive[0] + x_shift, y])
                        end_pt = np.array([arrive[0] + x_shift, arrive[1]])
                        return [exit_pt, stub, rise, corner, end_pt]

                    def _on_run(pts: list[np.ndarray], frac: float) -> np.ndarray:
                        x0 = float(pts[2][0]) + 0.4
                        x1 = min(float(pts[3][0]) - 0.35, row_left - 0.25)
                        if x1 < x0 + 0.6:
                            x1 = float(pts[3][0]) - 0.25
                        return np.array([x0 + frac * (x1 - x0), float(pts[2][1])])

                    if forward >= min_count:
                        pts = _elbow("blue", 0.0)
                        _ortho_arrow(ax, pts, BLUE)
                        _rate_text(ax, tuple(_on_run(pts, 0.32)), arrow_fmt(forward), BLUE)
                        if iodine_forward:
                            _rate_text(ax, tuple(_on_run(pts, 0.70)), str(iodine_forward), IODINE)
                    elif iodine_forward:
                        pts = _elbow("blue", 0.0)
                        _ortho_arrow(ax, pts, BLUE)
                        _rate_text(ax, tuple(_on_run(pts, 0.5)), str(iodine_forward), IODINE)
                    if backward >= min_count:
                        pts = _elbow("orange", 0.14)
                        _ortho_arrow(ax, list(reversed(pts)), ORANGE)
                        _rate_text(ax, tuple(_on_run(pts, 0.55)), arrow_fmt(backward), ORANGE)
                        if iodine_backward:
                            _rate_text(ax, tuple(_on_run(pts, 0.22)), str(iodine_backward), IODINE)
                    elif iodine_backward:
                        pts = _elbow("orange", 0.14)
                        _ortho_arrow(ax, list(reversed(pts)), ORANGE)
                        _rate_text(ax, tuple(_on_run(pts, 0.5)), str(iodine_backward), IODINE)
                    continue
                delta = end - start
                length = float(np.hypot(*delta)) or 1.0
                perp = np.array([-delta[1], delta[0]]) / length
                if column_skip:
                    # A skip would cut the boxes between its ends. Step out to a
                    # dedicated vertical lane, then turn back in at a right angle.
                    vertical_skip = abs(float(delta[0])) < abs(float(delta[1]))
                    away = np.array([-1.0, 0.0] if vertical_skip else [0.0, 1.0])
                    centroid = np.mean(np.asarray(list(positions.values()), float), axis=0)
                    if float(np.dot(away, centroid - (start + end) / 2)) > 0:
                        away = -away
                    upper, lower = (q_lab, p_lab) if positions[q_lab][1] >= positions[p_lab][1] else (p_lab, q_lab)
                    up_clear, down_clear, y_up, y_down = skip_lane.get(
                        (upper, lower), (3.2, 4.15, 0.0, 0.0)
                    )

                    def _side_rim(lab: int, y_shift: float) -> np.ndarray:
                        cx, cy = positions[lab]
                        return np.array([cx + away[0] * _nw(positions[lab]) / 2, cy + y_shift])

                    def _channel(clear: float, y_shift: float, up: bool) -> list[np.ndarray]:
                        rim_lo = _side_rim(lower, y_shift)
                        rim_hi = _side_rim(upper, y_shift)
                        bend_lo = rim_lo + away * clear
                        bend_hi = rim_hi + away * clear
                        if up:
                            return [rim_lo, bend_lo, bend_hi, rim_hi]
                        return [rim_hi, bend_hi, bend_lo, rim_lo]

                    def _on_channel(pts: list[np.ndarray], frac: float) -> np.ndarray:
                        return pts[1] + frac * (pts[2] - pts[1])

                    if forward >= min_count:
                        pts = _channel(up_clear, y_up, True)
                        _ortho_arrow(ax, pts, BLUE)
                        _rate_text(ax, tuple(_on_channel(pts, 0.22)), arrow_fmt(forward), BLUE)
                        if iodine_forward:
                            _rate_text(ax, tuple(_on_channel(pts, 0.62)), str(iodine_forward), IODINE)
                    elif iodine_forward:
                        pts = _channel(up_clear, y_up, True)
                        _ortho_arrow(ax, pts, BLUE)
                        _rate_text(ax, tuple(_on_channel(pts, 0.5)), str(iodine_forward), IODINE)
                    if backward >= min_count:
                        pts = _channel(down_clear, y_down, False)
                        _ortho_arrow(ax, pts, ORANGE)
                        _rate_text(ax, tuple(_on_channel(pts, 0.22)), arrow_fmt(backward), ORANGE)
                        if iodine_backward:
                            _rate_text(ax, tuple(_on_channel(pts, 0.62)), str(iodine_backward), IODINE)
                    elif iodine_backward:
                        pts = _channel(down_clear, y_down, False)
                        _ortho_arrow(ax, pts, ORANGE)
                        _rate_text(ax, tuple(_on_channel(pts, 0.5)), str(iodine_backward), IODINE)
                    continue
                if blocked:
                    # Vertical skips clear the side loop. Horizontal skips clear the box.
                    bulge = 1.75 if abs(delta[0]) < abs(delta[1]) else 0.85
                    if forward >= min_count:
                        rad, control = _counted_arc(start, end, bulge)
                        _arrow(ax, tuple(start), tuple(end), BLUE, rad)
                        spot = _place_rate(_bezier(start, control, end, 0.42), perp)
                        _rate_text(ax, tuple(spot), arrow_fmt(forward), BLUE)
                        if iodine_forward:
                            _iodine_callout(
                                _gap_bezier(start, control, end, positions, width, height),
                                perp,
                                iodine_forward,
                            )
                    elif iodine_forward:
                        rad, control = _counted_arc(start, end, bulge)
                        _iodine_callout(
                            _gap_bezier(start, control, end, positions, width, height),
                            perp,
                            iodine_forward,
                        )
                    if backward >= min_count:
                        rad, control = _counted_arc(end, start, bulge + 0.35)
                        _arrow(ax, tuple(end), tuple(start), ORANGE, rad)
                        spot = _place_rate(_bezier(end, control, start, 0.42), -perp)
                        _rate_text(ax, tuple(spot), arrow_fmt(backward), ORANGE)
                        if iodine_backward:
                            _iodine_callout(
                                _gap_bezier(end, control, start, positions, width, height),
                                -perp,
                                iodine_backward,
                            )
                    elif iodine_backward:
                        rad, control = _counted_arc(end, start, bulge + 0.35)
                        _iodine_callout(
                            _gap_bezier(end, control, start, positions, width, height),
                            -perp,
                            iodine_backward,
                        )
                    extra_right = max(extra_right, bulge + 0.45)
                    continue
                vertical = clear_crossings and abs(float(delta[0])) < 0.35 * abs(float(delta[1]))
                column_shaft = (
                    single
                    and cube == "BMHpM"
                    and abs(float(delta[0])) < 0.35 * abs(float(delta[1]))
                )
                pair_gap = 0.36 if column_shaft else 0.07
                if vertical:
                    hub_x = min(xy[0] for xy in positions.values())
                    toward_hub = np.array([-1.0 if start[0] > hub_x + 0.5 else 1.0, 0.0])
                if forward >= min_count:
                    shift = perp * pair_gap
                    shaft_a = start + shift
                    shaft_b = end + shift
                    _arrow(ax, tuple(shaft_a), tuple(shaft_b), BLUE)
                    on_shaft = column_shaft
                    label_at = shaft_a + 0.34 * (shaft_b - shaft_a) if on_shaft else (
                        start + 0.34 * delta + toward_hub * 0.48
                        if vertical
                        else start + 0.38 * delta + perp * 0.36
                    )
                    if on_shaft:
                        _rate_text(ax, tuple(label_at), arrow_fmt(forward), BLUE)
                    else:
                        prefer = toward_hub if vertical else perp
                        _rate_text(ax, tuple(_place_rate(label_at, prefer)), arrow_fmt(forward), BLUE)
                    if iodine_forward and on_shaft:
                        _rate_text(ax, tuple(shaft_a + 0.5 * (shaft_b - shaft_a)), str(iodine_forward), IODINE)
                    elif iodine_forward:
                        _iodine_callout(
                            _gap_along(shaft_a, shaft_b, positions, width, height),
                            perp,
                            iodine_forward,
                        )
                elif iodine_forward:
                    _iodine_callout(
                        _gap_along(start, end, positions, width, height),
                        perp,
                        iodine_forward,
                    )
                if backward >= min_count:
                    shift = -perp * pair_gap
                    shaft_a = end + shift
                    shaft_b = start + shift
                    _arrow(ax, tuple(shaft_a), tuple(shaft_b), ORANGE)
                    on_shaft = column_shaft
                    label_at = shaft_a + 0.32 * (shaft_b - shaft_a) if on_shaft else (
                        start + 0.66 * delta + toward_hub * 0.48
                        if vertical
                        else start + 0.62 * delta - perp * 0.36
                    )
                    if on_shaft:
                        _rate_text(ax, tuple(label_at), arrow_fmt(backward), ORANGE)
                    else:
                        prefer = toward_hub if vertical else -perp
                        _rate_text(ax, tuple(_place_rate(label_at, prefer)), arrow_fmt(backward), ORANGE)
                    if iodine_backward and on_shaft:
                        _rate_text(ax, tuple(shaft_a + 0.5 * (shaft_b - shaft_a)), str(iodine_backward), IODINE)
                    elif iodine_backward:
                        _iodine_callout(
                            _gap_along(shaft_a, shaft_b, positions, width, height),
                            -perp,
                            iodine_backward,
                        )
                elif iodine_backward:
                    _iodine_callout(
                        _gap_along(end, start, positions, width, height),
                        -perp,
                        iodine_backward,
                    )
        xs = [x for x, _y in positions.values()]
        y_margin = max(1.5, loop_reach + 0.85) if has_iodine else 1.35
        x_margin = 1.7 if has_iodine else 1.35
        x_left = 11.2 if single and cube == "BMHpM" else x_margin
        ax.set_xlim(min(xs) - x_left, max(xs) + x_margin + extra_right)
        if single and cube == "BMHpM":
            ax.set_ylim(peak_lo[0] - height / 2 - 0.85, peak_hi[0] + height / 2 + 0.85)
        else:
            ax.set_ylim(
                min(ys) - y_margin,
                max(ys) + (y_margin if has_iodine else (1.45 if clear_crossings else 1.15)),
            )
        ax.set_aspect("equal")
        ax.axis("off")
        system = MURATA_NAME.get(cube, "")
        title = f"{cube}   {system}" if system else cube
        if show_k:
            title = f"{title}   k={SELECTED_K[cube]}"
        ax.text(0.0, 1.02, f"({letter})", transform=ax.transAxes, fontweight="bold", fontsize=16 if single else 12, va="bottom")
        ax.text(0.5, 1.02, title, transform=ax.transAxes, ha="center", fontsize=15 if single else 11, va="bottom")
    two_lines = "\n" in suptitle
    fig.suptitle(suptitle, fontsize=(14 if single else 11) if two_lines else (15 if single else 12), y=0.995 if two_lines else 0.98)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cluster_flows(table: pd.DataFrame, out: Path) -> Path:
    """Arrow diagram of 1 ps cluster changes, with the Figure 4b names."""
    return _flow_figure(
        lambda cube: _rates(table, cube, "cluster"),
        out,
        min_count=0.01,
        suptitle="Cluster changes per ns  ·  Figure 4b labels  ·  blue points right or up, orange the return",
        show_k=False,
    )


def plot_segment_flows(out: Path) -> Path:
    """Same layout, counted once per neighboring segment at the selected k."""
    return _flow_figure(
        _segment_counts,
        out,
        min_count=1,
        suptitle="Segment-to-segment changes  ·  selected k  ·  time in each box is the mean dwell  ·  loop = segments continued",
        show_k=True,
        arrow_fmt=_fmt_stay,
        clear_crossings=True,
        dwell_for=_mean_dwells_ns,
    )


def _iodine_notes(cube: str) -> dict[int, str]:
    notes, _edges, _states, _changes = _iodine()
    return notes[cube]


def _iodine_edges(cube: str) -> dict[tuple[str, str], int]:
    _notes, edges, _states, _changes = _iodine()
    return edges[cube]


def plot_iodine_panels(table: pd.DataFrame, out_dir: Path) -> list[Path]:
    """One expanded iodide figure per cube, for the cluster rates and the segment counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    cluster_title = (
        "Cluster changes per ns  ·  blue points right or up, orange the return\n"
        "purple in the box: iodide occupancy, and 20 ps entries (+) and exits (-) begun in that state"
        "  ·  purple count: events within 50 ps of that change"
    )
    segment_title = (
        "Segment-to-segment changes  ·  selected k  ·  time in each box is the mean dwell"
        "  ·  loop = segments continued\n"
        "purple in the box: iodide occupancy, and 20 ps entries (+) and exits (-) begun in that state"
        "  ·  purple count: events within 50 ps of that change"
    )
    for cube in SERIES:
        cluster = _flow_figure(
            lambda cube_name: _rates(table, cube_name, "cluster"),
            out_dir / f"cluster_{cube}.png",
            min_count=0.01,
            suptitle=cluster_title,
            show_k=False,
            iodine_for=_iodine_notes,
            iodine_edges_for=_iodine_edges,
            only=cube,
        )
        segment = _flow_figure(
            _segment_counts,
            out_dir / f"segment_{cube}.png",
            min_count=1,
            suptitle=segment_title,
            show_k=True,
            arrow_fmt=_fmt_stay,
            clear_crossings=True,
            dwell_for=_mean_dwells_ns,
            iodine_for=_iodine_notes,
            iodine_edges_for=_iodine_edges,
            only=cube,
        )
        written.extend((cluster, segment))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=ROOT / "output" / "transition_frequencies" / "transition_frequencies.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "docs" / "figures",
    )
    args = parser.parse_args()
    table = pd.read_csv(args.table)
    with RunContext.from_namespace(args, name="plot_transition_frequencies"):
        meta = plot_metastructures(table, args.out_dir / "transition_metastructures.png")
        meta_pole = plot_metastructures(
            table,
            args.out_dir / "transition_metastructures_poleH.png",
            cubes=("BHHpH", "BMMpH", "BMHpH"),
        )
        heat = plot_cluster_heatmaps(table, args.out_dir / "transition_clusters.png")
        flows = plot_cluster_flows(table, args.out_dir / "transition_cluster_flows.png")
        segments = plot_segment_flows(args.out_dir / "transition_segment_flows.png")
        panels = plot_iodine_panels(table, args.out_dir / "iodine_flows")
        print(f"wrote {meta}")
        print(f"wrote {meta_pole}")
        print(f"wrote {heat}")
        print(f"wrote {flows}")
        print(f"wrote {segments}")
        for path in panels:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
