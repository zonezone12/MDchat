"""Figure 3: conformational landscape of the six GSA nanocubes.

Reads the physical-k catalog (does not recluster).

(a) Apo-frame occupancy. Structures are the rows of ``structures_summary.csv``.
    Compact cages, mostly-compact cages and intact belts are one class;
    belt-break and opened structures are the second; mixed belts stay a
    remainder so each cube still sums to its apo frames.
(b) A reference cube with the six monomer faces labeled, then BMHpM's
    six-orientation belt-break family. Face centers and contacts come from
    the catalog (``OCTAHEDRON_COORDS``, pole edges, equatorial edges): M0,
    M1 and M2 meet at one pole and M3, M4 and M5 at the other. Each
    member's dominant single elongated edge is taken from
    ``segment_belt.csv`` and drawn on the equatorial hexagon
    M0–M4–M2–M3–M1–M5. Vertex colors match the cube.
(c) Majority open-pole count among broken-belt segments. Same frame
    weighting as ``broken_pct_*`` in the structure summary: within each
    segment the pole count is the majority over its frames, then those
    segments are pooled by frame count. Intact-majority segments are left
    out.

Example
-------
python scripts/plot_conformational_landscape.py
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
from src.ChangepointAnalysis.metastable_catalog import (
    DEFAULT_D1_EDGES,
    DEFAULT_POLE_EDGES,
    OCTAHEDRON_COORDS,
    belt_kind,
)
from src.utils.run_log import RunContext

# Equatorial belt of the six monomers, in cycle order.
MONOMERS: tuple[str, ...] = ("M0", "M4", "M2", "M3", "M1", "M5")
EDGES: tuple[tuple[str, str], ...] = tuple(
    (MONOMERS[i], MONOMERS[(i + 1) % len(MONOMERS)]) for i in range(len(MONOMERS))
)

STATE_ORDER: tuple[str, ...] = (
    "Compact cage",
    "Intact belt",
    "Opened / belt-break",
    "Mixed",
    "Single replica",
)
STATE_COLORS: dict[str, str] = {
    "Compact cage": "#08519c",
    "Intact belt": "#6baed6",
    "Opened / belt-break": "#e6550d",
    "Mixed": "#bdbdbd",
    "Single replica": "#525252",
}
POLE_BINS: tuple[tuple[str, str], ...] = (
    ("pct_0_poles", "0"),
    ("pct_1_pole", "1"),
    ("pct_2_poles", "2"),
    ("pct_3plus_poles", "3+"),
)
POLE_COLORS: dict[str, str] = {
    "0": "#d9d9d9",
    "1": "#9ecae1",
    "2": "#2171b5",
    "3+": "#cb181d",
}
# M0, M1, M2 meet at one pole; M3, M4, M5 at the other. Opposites are i and i+3.
NORTH_POLE = frozenset({0, 1, 2})
FACE_COLOR = {
    0: "#9ecae1",
    1: "#9ecae1",
    2: "#9ecae1",
    3: "#fdae6b",
    4: "#fdae6b",
    5: "#fdae6b",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot the six-cube conformational landscape.")
    p.add_argument("--output-root", type=Path, default=Path("output"))
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=Path("output/metastable_structures_physical_k"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/conformational_landscape_figure"),
    )
    p.add_argument("--manuscript-dir", type=Path, default=Path("docs/figures"))
    return p.parse_args()


def state_class(structure: str) -> str:
    """Map a catalog structure name onto the three occupancy classes."""
    if structure.startswith("compact cage"):
        return "Compact cage"
    if structure.startswith(("mostly compact", "intact belt")):
        return "Intact belt"
    if structure.startswith("mixed belt"):
        return "Mixed"
    if structure.startswith("loosened"):
        return "Single replica"
    if structure.startswith(("belt broken", "one belt break", "opened cage", "partly belt broken")):
        return "Opened / belt-break"
    raise ValueError(f"Unclassified structure name: {structure}")


def occupancy_table(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["state_class"] = out["structure"].map(state_class)
    rows = []
    for cube in KNOWN_CUBES:
        part = out[out["cohort"] == cube]
        for state in STATE_ORDER:
            share = float(part.loc[part["state_class"] == state, "pct_of_cube"].sum())
            rows.append({"cube": cube, "state_class": state, "pct_of_cube": share})
    return pd.DataFrame(rows)


def dominant_edges(segment_belt: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """One equatorial edge per member of BMHpM's six-orientation family."""
    fam = clusters[(clusters["cohort"] == "BMHpM") & (clusters["n_orientations"] == 6)]
    if fam.empty:
        raise ValueError("BMHpM has no six-orientation family")
    rows = []
    belts = segment_belt[segment_belt["cohort"] == "BMHpM"]
    for rec in fam.itertuples(index=False):
        sub = belts[belts["cluster_label"] == int(rec.cluster_label)]
        single = sub[sub["belt"].str.startswith("edge ")]
        if single.empty:
            raise ValueError(f"BMHpM cluster {rec.cluster_label} has no single-edge belt")
        top = single.sort_values("n_frames", ascending=False).iloc[0]
        edge = str(top["belt"]).removeprefix("edge ").replace("-", "–")
        pair = tuple(edge.split("–"))
        if pair not in EDGES and tuple(reversed(pair)) not in EDGES:
            raise ValueError(f"Edge {edge} is not on the equatorial cycle")
        rows.append(
            {
                "cluster_label": int(rec.cluster_label),
                "edge": edge,
                "pct_of_cube": float(rec.pct_of_cube),
                "n_frames_on_edge": int(top["n_frames"]),
                "pct_of_cluster": float(top["pct_of_cluster"]),
            }
        )
    out = pd.DataFrame(rows)
    order = {f"{a}–{b}": i for i, (a, b) in enumerate(EDGES)}
    out["cycle_index"] = out["edge"].map(order)
    return out.sort_values("cycle_index")


def pole_table(segment_belt: pd.DataFrame) -> pd.DataFrame:
    """Frame-weighted pole-count mix of broken-belt segments, per cube.

    Matches ``_pole_mix`` in ``catalog_runs``: the pole bin is the segment
    majority, and segments are then weighted by how many frames they cover.
    """
    seg = segment_belt.copy()
    seg["kind"] = seg["belt"].map(belt_kind)
    broken = seg[seg["kind"] != "intact"]
    rows = []
    for cube in KNOWN_CUBES:
        part = broken[broken["cohort"] == cube]
        weight = part["n_frames"].to_numpy(dtype=float)
        total = float(weight.sum())
        row = {"cube": cube, "n_frames": total, "n_segments": int(part["n_segments"].sum())}
        for col, label in POLE_BINS:
            row[label] = float((part[col] * weight).sum() / total) if total else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _stacked_bars(ax, cubes: list[str], table: pd.DataFrame, key: str, order, colors, ylabel: str) -> None:
    x = np.arange(len(cubes))
    bottom = np.zeros(len(cubes))
    for label in order:
        vals = np.array(
            [
                float(table.loc[(table["cube"] == cube) & (table[key] == label), "value"].iloc[0])
                for cube in cubes
            ]
        )
        if np.nanmax(vals) < 0.05:
            continue
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=0.72,
            color=colors[label],
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        bottom += np.nan_to_num(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(cubes, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        fontsize=7.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.92),
        ncol=3,
        borderaxespad=0.0,
    )


def _project_face(point: np.ndarray) -> tuple[float, float]:
    """Oblique view from the (+x, +y, +z) side, with +z upward on the page."""
    x, y, z = (float(v) for v in point)
    return (x - y) * np.cos(np.pi / 6), z + (x + y) * 0.5


def _face_corners(monomer: int) -> np.ndarray:
    """Four corners of the cube face whose outward normal is this monomer."""
    normal = OCTAHEDRON_COORDS[monomer]
    axis = int(np.argmax(np.abs(normal)))
    sign = float(np.sign(normal[axis]))
    corners = [
        np.array([sx, sy, sz], dtype=float)
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
        if (sx, sy, sz)[axis] == sign
    ]
    center = np.mean(corners, axis=0)
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent = np.cross(normal, helper)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    angles = [
        np.arctan2(np.dot(corner - center, bitangent), np.dot(corner - center, tangent))
        for corner in corners
    ]
    return np.array([corners[i] for i in np.argsort(angles)])


def _draw_face_cube(ax, origin: tuple[float, float], scale: float) -> None:
    """Reference cage: one monomer per face, poles colored, all six faces named."""
    view = np.array([1.0, 1.0, 1.0])
    view /= np.linalg.norm(view)

    def screen(point: np.ndarray) -> tuple[float, float]:
        sx, sy = _project_face(point)
        return origin[0] + scale * sx, origin[1] + scale * sy

    visible = {
        monomer
        for monomer in range(6)
        if float(np.dot(OCTAHEDRON_COORDS[monomer], view)) > 0
    }
    for monomer in visible:
        normal = OCTAHEDRON_COORDS[monomer]
        corners = _face_corners(monomer)
        poly = np.array([screen(corner) for corner in corners])
        ax.fill(
            poly[:, 0],
            poly[:, 1],
            facecolor=FACE_COLOR[monomer],
            edgecolor="none",
            zorder=2,
        )
        label_at = screen(normal)
        ax.text(
            label_at[0],
            label_at[1],
            f"M{monomer}",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="0.15",
            zorder=4,
        )

    equatorial_pairs = {frozenset(edge) for edge in DEFAULT_D1_EDGES}
    drawn: set[frozenset[int]] = set()
    for a, b in list(DEFAULT_D1_EDGES) + list(DEFAULT_POLE_EDGES):
        pair = frozenset((int(a), int(b)))
        if pair in drawn or not (int(a) in visible or int(b) in visible):
            continue
        drawn.add(pair)
        shared = _shared_edge(int(a), int(b))
        pts = np.array([screen(corner) for corner in shared])
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color="#636363" if pair in equatorial_pairs else "#2171b5",
            lw=1.6 if pair in equatorial_pairs else 1.15,
            solid_capstyle="round",
            zorder=3,
        )

    center = np.array(screen(np.zeros(3)))
    for monomer in range(6):
        if monomer in visible:
            continue
        normal = OCTAHEDRON_COORDS[monomer]
        direction = np.array(screen(normal)) - center
        direction /= np.linalg.norm(direction)
        anchor = center + direction * scale * 1.45
        parked = center + direction * scale * 2.45
        ax.annotate(
            f"M{monomer}",
            xy=anchor,
            xytext=parked,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="#d94801",
            arrowprops={"arrowstyle": "-", "color": "#fdae6b", "lw": 0.7},
            zorder=4,
        )

    ax.text(
        origin[0],
        origin[1] - scale * 3.35,
        "Blue: one pole\n"
        "Amber: hidden pole\n"
        "M3 opposite M0\n"
        "M4 opposite M1\n"
        "M5 opposite M2\n"
        "Gray outline = belt",
        ha="center",
        va="top",
        fontsize=6.5,
        color="0.25",
        linespacing=1.2,
    )


def _shared_edge(a: int, b: int) -> np.ndarray:
    """The cube edge where faces *a* and *b* meet, as two corners."""
    corners_a = _face_corners(a)
    normals_b = OCTAHEDRON_COORDS[b]
    axis = int(np.argmax(np.abs(normals_b)))
    sign = float(np.sign(normals_b[axis]))
    shared = [corner for corner in corners_a if corner[axis] == sign]
    if len(shared) != 2:
        raise ValueError(f"Faces M{a} and M{b} do not share an edge")
    return np.array(shared)


def _draw_hexagon(ax, cx: float, cy: float, radius: float, open_index: int) -> None:
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 6, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    for i in range(6):
        j = (i + 1) % 6
        opened = i == open_index
        ax.plot(
            [xs[i], xs[j]],
            [ys[i], ys[j]],
            color="#d94801" if opened else "#bdbdbd",
            lw=2.6 if opened else 1.15,
            solid_capstyle="round",
            zorder=2,
        )
    colors = [FACE_COLOR[int(name[1])] for name in MONOMERS]
    ax.scatter(xs, ys, s=22, c=colors, zorder=3, linewidths=0.4, edgecolors="0.25")
    for x, y, name in zip(xs, ys, MONOMERS):
        ax.text(
            cx + (x - cx) * 1.48,
            cy + (y - cy) * 1.48,
            name,
            ha="center",
            va="center",
            fontsize=5.5,
            color="0.25",
        )


def plot_figure(occupancy: pd.DataFrame, edges: pd.DataFrame, poles: pd.DataFrame, out: Path) -> Path:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(14.2, 5.6))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[0.7, 1.05, 1.15],
        width_ratios=[1.0, 2.05, 1.0],
        wspace=0.32,
        hspace=0.08,
    )
    ax_a = fig.add_subplot(gs[1, 0])
    ax_b = fig.add_subplot(gs[:, 1])
    ax_c = fig.add_subplot(gs[1, 2])

    occ = occupancy.rename(columns={"pct_of_cube": "value"})
    _stacked_bars(
        ax_a,
        list(KNOWN_CUBES),
        occ,
        "state_class",
        STATE_ORDER,
        STATE_COLORS,
        "Apo frames (%)",
    )
    ax_a.set_xlabel("System")
    ax_a.text(-0.18, 1.05, "(a)", transform=ax_a.transAxes, fontweight="bold", va="bottom")

    ax_b.set_aspect("equal")
    ax_b.axis("off")
    _draw_face_cube(ax_b, origin=(1.45, 2.25), scale=0.42)
    radius = 0.30
    centers = [(col, row) for row in (1, 0) for col in range(3)]
    ordered = edges.sort_values("cycle_index")
    for (col, row), rec in zip(centers, ordered.itertuples(index=False)):
        cx, cy = 4.15 + col * 1.28, 1.15 + row * 1.75
        _draw_hexagon(ax_b, cx, cy, radius, int(rec.cycle_index))
        ax_b.text(
            cx,
            cy - radius - 0.18,
            f"{rec.edge}   {rec.pct_of_cube:.1f}%",
            ha="center",
            va="top",
            fontsize=7,
        )
    ax_b.set_xlim(-0.2, 7.85)
    ax_b.set_ylim(-0.45, 3.75)
    ax_b.text(0.0, 1.02, "(b)", transform=ax_b.transAxes, fontweight="bold", va="bottom")
    ax_b.text(0.72, 1.02, "BMHpM opened edge", transform=ax_b.transAxes, ha="center", va="bottom", fontsize=10)

    pole_long = poles.melt(
        id_vars=["cube"],
        value_vars=[label for _, label in POLE_BINS],
        var_name="poles",
        value_name="value",
    )
    _stacked_bars(
        ax_c,
        list(KNOWN_CUBES),
        pole_long,
        "poles",
        [label for _, label in POLE_BINS],
        POLE_COLORS,
        "Broken-belt segments (%)",
    )
    ax_c.set_xlabel("System")
    legend = ax_c.get_legend()
    if legend is not None:
        legend.set_title("Open poles")
        legend.get_title().set_fontsize(8)
    ax_c.text(-0.2, 1.05, "(c)", transform=ax_c.transAxes, fontweight="bold", va="bottom")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    catalog = Path(args.catalog_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with RunContext.from_namespace(args, name="plot_conformational_landscape"):
        summary = pd.read_csv(catalog / "structures_summary.csv")
        clusters = pd.read_csv(catalog / "clusters_ranked.csv")
        segment_belt = pd.read_csv(catalog / "min_share_0" / "segment_belt.csv")
        occupancy = occupancy_table(summary)
        edges = dominant_edges(segment_belt, clusters)
        poles = pole_table(segment_belt)
        occupancy.to_csv(out_dir / "occupancy_by_state.csv", index=False)
        edges.to_csv(out_dir / "bmhpm_belt_edges.csv", index=False)
        poles.to_csv(out_dir / "broken_belt_poles.csv", index=False)
        fig_path = plot_figure(occupancy, edges, poles, out_dir / "figure3_conformational_landscape.png")
        manuscript_dir = str(args.manuscript_dir).strip()
        if manuscript_dir:
            dest_dir = Path(manuscript_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in (fig_path, fig_path.with_suffix(".pdf")):
                dest = dest_dir / src.name
                try:
                    dest.write_bytes(src.read_bytes())
                except OSError:
                    dest = dest_dir / f"{src.stem}_cube{src.suffix}"
                    dest.write_bytes(src.read_bytes())
                print(f"Copied figure to {dest}")
        print(f"Wrote {fig_path}")
        print(occupancy.pivot(index="cube", columns="state_class", values="pct_of_cube").reindex(KNOWN_CUBES).to_string())
        print(edges[["cluster_label", "edge", "pct_of_cube", "pct_of_cluster"]].to_string(index=False))
        print(poles.to_string(index=False))


if __name__ == "__main__":
    main()
