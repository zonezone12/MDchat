"""
Annotated atom-index reference diagrams for the NH4NO3 proton-transfer systems.

Feature columns in the analysis (e.g. `N0-H8`, `O6-O8`) are labeled by
``<species><atom_index>``. This script renders each frame-0 geometry with every
atom labeled by its index + element, bonds drawn, and the proton-transfer /
water atoms highlighted, so distance features are easy to read in a presentation.

Outputs (into output/nh4no3_pt/):
  atom_index_map_m1n0.png
  atom_index_map_m1n1.png
  atom_index_map.png          (both systems side by side)

Example
-------
python scripts/plot_atom_index_map.py
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from src.ProtonTransfer import detect_pt_atoms, load_pimd_xyz

_ELEMENT_STYLE: Dict[str, Tuple[str, float]] = {
    "N": ("#2b6cff", 900.0),
    "O": ("#ff3b30", 820.0),
    "H": ("#e6e6e6", 430.0),
    "C": ("#444444", 820.0),
}
_COVALENT_CUTOFF = 1.35

_SYSTEMS = [
    ("m1n0", "2-m1n0/1-CLMD/Result/coor.xyz", 9),
    ("m1n1", "3-m1n1/1-CLMD/Result/coor.xyz", 12),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Atom-index reference diagrams")
    p.add_argument("--traj-root", type=Path, default=ROOT / "traj" / "NH4NO3")
    p.add_argument("--output-dir", type=Path, default=ROOT / "output" / "nh4no3_pt")
    return p.parse_args()


def _pca_project(coords: np.ndarray) -> np.ndarray:
    """Project 3D coordinates onto their 2 principal axes (max-variance plane)."""
    c = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    return c @ vt[:2].T


def _covalent_bonds(coords: np.ndarray, species: List[str]) -> List[Tuple[int, int]]:
    bonds = []
    for i, j in combinations(range(len(species)), 2):
        if species[i] == "H" and species[j] == "H":
            continue
        if float(np.linalg.norm(coords[i] - coords[j])) <= _COVALENT_CUTOFF:
            bonds.append((i, j))
    return bonds


def _roles(species: List[str], pt) -> Dict[int, str]:
    roles: Dict[int, str] = {}
    roles[pt.n_amine] = "amine N (acceptor)"
    roles[pt.h_acid] = "acid proton (transfers)"
    roles[pt.o_acid] = "acid O (donor)"
    if pt.water_o is not None:
        roles[pt.water_o] = "water O"
        for h in pt.water_h:
            roles[h] = "water H"
    return roles


def _draw_system(ax, system: str, xyz: Path, n_physical: int) -> List[str]:
    traj = load_pimd_xyz(xyz, n_physical=n_physical, max_frames=1)
    species = traj.species
    coords3d = traj.centroid[0]
    pt = detect_pt_atoms(species, coords3d)
    roles = _roles(species, pt)

    xy = _pca_project(coords3d)

    # Bonds
    for i, j in _covalent_bonds(coords3d, species):
        ax.plot(
            [xy[i, 0], xy[j, 0]],
            [xy[i, 1], xy[j, 1]],
            color="#888888",
            lw=2.2,
            zorder=1,
        )

    # Highlight rings for PT / water atoms
    for idx, role in roles.items():
        if "amine" in role:
            ring = "#2b6cff"
        elif "acid proton" in role:
            ring = "#ffb000"
        elif "acid O" in role:
            ring = "#ff3b30"
        elif "water" in role:
            ring = "#00c2ff"
        else:
            ring = None
        if ring:
            ax.scatter(
                xy[idx, 0],
                xy[idx, 1],
                s=_ELEMENT_STYLE.get(species[idx], ("", 800))[1] * 1.7,
                facecolors="none",
                edgecolors=ring,
                linewidths=2.6,
                zorder=2,
            )

    # Atoms
    for idx, sym in enumerate(species):
        color, size = _ELEMENT_STYLE.get(sym, ("#999999", 700.0))
        ax.scatter(
            xy[idx, 0],
            xy[idx, 1],
            s=size,
            c=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
        )
        txt_color = "black" if sym == "H" else "white"
        ax.text(
            xy[idx, 0],
            xy[idx, 1],
            f"{sym}{idx}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=txt_color,
            zorder=4,
        )

    ax.set_title(
        f"{system}  ({n_physical} atoms)\n"
        f"CV: δ = d({species[pt.n_amine]}{pt.n_amine}–"
        f"{species[pt.h_acid]}{pt.h_acid}) − "
        f"d({species[pt.o_acid]}{pt.o_acid}–{species[pt.h_acid]}{pt.h_acid})",
        fontsize=11,
    )
    ax.set_aspect("equal","datalim","C")
    ax.axis("off")

    # Build a legend text block for this system
    lines = [f"{system}: index -> element (role)"]
    for idx, sym in enumerate(species):
        role = roles.get(idx, "")
        lines.append(f"  {sym}{idx}" + (f"  = {role}" if role else ""))
    return lines


def _element_legend(ax) -> None:
    handles = [
        mpatches.Patch(color="#2b6cff", label="N"),
        mpatches.Patch(color="#ff3b30", label="O"),
        mpatches.Patch(color="#e6e6e6", label="H"),
    ]
    ring_handles = [
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#2b6cff",
                   mew=2.2, ms=12, label="amine N (acceptor)"),
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#ffb000",
                   mew=2.2, ms=12, label="acid proton (transfers)"),
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#ff3b30",
                   mew=2.2, ms=12, label="acid O (donor)"),
        plt.Line2D([], [], marker="o", ls="", mfc="none", mec="#00c2ff",
                   mew=2.2, ms=12, label="water"),
    ]
    ax.legend(
        handles=handles + ring_handles,
        loc="center",
        ncol=1,
        frameon=False,
        fontsize=10,
        title="Legend",
    )
    ax.axis("off")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Individual figures
    for system, rel, n_phys in _SYSTEMS:
        xyz = (args.traj_root / rel).resolve()
        if not xyz.is_file():
            print(f"SKIP {system}: missing {xyz}")
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        _draw_system(ax, system, xyz, n_phys)
        fig.tight_layout()
        p = out_dir / f"atom_index_map_{system}.png"
        fig.savefig(p, dpi=300)
        plt.close(fig)
        print(f"Saved: {p}")

    # Combined side-by-side with a shared legend column
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.5])
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_leg = fig.add_subplot(gs[0, 2])
    all_lines: List[str] = []
    for ax, (system, rel, n_phys) in zip(axes, _SYSTEMS):
        xyz = (args.traj_root / rel).resolve()
        if not xyz.is_file():
            ax.axis("off")
            continue
        all_lines += _draw_system(ax, system, xyz, n_phys) + [""]
    _element_legend(ax_leg)
    fig.suptitle(
        "NH4NO3 proton-transfer systems — atom index map "
        "(feature labels use <element><index>, e.g. N0-H8)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = out_dir / "atom_index_map.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")

    # Text mapping for quick reference / presentation notes
    txt = out_dir / "atom_index_map.txt"
    txt.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"Saved: {txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
