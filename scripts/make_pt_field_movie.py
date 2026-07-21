"""
Quantum-delocalization "field" movie of an NH3...HNO3(+H2O) proton transfer.

Instead of only the bead-averaged (centroid) geometry, this renders the full
PIMD ring-polymer: every atom's 60 beads are drawn as a translucent point cloud,
and the space they occupy is wrapped in a convex-hull surface. The result is a
"field" whose size reflects each atom's quantum delocalization -- largest for the
transferring proton, which smears out between the donor O and acceptor N during
transfer. The water molecule and its approach to the amine N are highlighted.

Right panels: delta = d(N-H) - d(O-H) and the transferring-proton ring-polymer
radius of gyration (delocalization) over the window, with a moving cursor.

Output: an animated GIF (ffmpeg not required).

Example
-------
python scripts/make_pt_field_movie.py                    # m1n1 PIMD, deepest event
python scripts/make_pt_field_movie.py --center 2651 --half 45 --fps 12
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
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import ConvexHull

try:  # scipy >= 1.8 exposes QhullError at package root
    from scipy.spatial import QhullError
except ImportError:  # older scipy
    from scipy.spatial.qhull import QhullError

from src.ProtonTransfer import (
    detect_pt_atoms,
    load_pimd_xyz,
    proton_delocalization,
    proton_transfer_cv,
)

# Element display: (color, centroid marker size, bead point color)
_ELEMENT_STYLE: Dict[str, Tuple[str, float]] = {
    "N": ("#2b6cff", 300.0),
    "O": ("#ff3b30", 280.0),
    "H": ("#d8d8d8", 110.0),
    "C": ("#444444", 280.0),
}
_COVALENT_CUTOFF = 1.35  # Angstrom, for drawing covalent bonds (on centroid)


def _job_relpath(system: str, method: str) -> str:
    folder = {"m1n0": "2-m1n0", "m1n1": "3-m1n1"}[system]
    sub = {"CLMD": "1-CLMD", "PIMD": "2-PIMD"}[method]
    return f"{folder}/{sub}/Result/coor.xyz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PIMD bead-field proton-transfer movie")
    p.add_argument("--traj-root", type=Path, default=ROOT / "traj" / "NH4NO3")
    p.add_argument("--system", default="m1n1", choices=("m1n0", "m1n1"))
    p.add_argument("--method", default="PIMD", choices=("CLMD", "PIMD"))
    p.add_argument(
        "--center",
        type=int,
        default=-1,
        help="Center frame; -1 = auto-pick deepest transfer (min delta)",
    )
    p.add_argument("--half", type=int, default=45, help="Half-window in frames")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GIF (default: output/nh4no3_pt/field_movie_<system>_<method>.gif)",
    )
    p.add_argument("--elev", type=float, default=18.0)
    p.add_argument("--azim", type=float, default=-60.0)
    p.add_argument("--spin", type=float, default=0.4)
    p.add_argument(
        "--hull-atoms",
        default="light",
        choices=("proton", "light", "all"),
        help="Which atoms get a convex-hull field surface "
        "(proton only / H+proton / all atoms)",
    )
    return p.parse_args()


def covalent_bonds(coords: np.ndarray, species: List[str]) -> List[Tuple[int, int]]:
    bonds = []
    for i, j in combinations(range(len(species)), 2):
        if species[i] == "H" and species[j] == "H":
            continue
        if float(np.linalg.norm(coords[i] - coords[j])) <= _COVALENT_CUTOFF:
            bonds.append((i, j))
    return bonds


def _hull_faces(points: np.ndarray):
    """Return list of triangle vertex arrays for a convex hull, or None."""
    if points.shape[0] < 4:
        return None
    # Degenerate (near-coplanar) clouds raise QhullError; skip gracefully.
    try:
        hull = ConvexHull(points, qhull_options="QJ")
    except (QhullError, ValueError):
        return None
    return [points[simplex] for simplex in hull.simplices]


def main() -> int:
    args = parse_args()
    relpath = _job_relpath(args.system, args.method)
    xyz = (args.traj_root / relpath).resolve()
    if not xyz.is_file():
        print(f"Missing trajectory: {xyz}", file=sys.stderr)
        return 1
    n_physical = 9 if args.system == "m1n0" else 12

    if args.center < 0:
        print("Auto-picking deepest transfer frame (scanning delta)...")
        traj = load_pimd_xyz(xyz, n_physical=n_physical)
        pt = detect_pt_atoms(traj.species, traj.centroid[0])
        cv_full = proton_transfer_cv(traj.centroid, pt.n_amine, pt.h_acid, pt.o_acid)
        center = int(np.argmin(cv_full["delta"].to_numpy()))
    else:
        center = args.center
        traj = load_pimd_xyz(
            xyz, n_physical=n_physical, max_frames=center + args.half + 2
        )
        pt = detect_pt_atoms(traj.species, traj.centroid[0])

    if traj.n_beads < 2:
        print(
            "WARNING: this trajectory has 1 bead (classical); the field will be "
            "a single point per atom. Use --method PIMD for the bead field.",
            file=sys.stderr,
        )

    start = max(0, center - args.half)
    end = min(traj.n_frames, center + args.half + 1)
    frames = list(range(start, end))

    centroid = traj.centroid  # (F, A, 3)
    beads = traj.beads  # (F, B, A, 3)
    species = traj.species
    n_amine, h_acid, o_acid = pt.n_amine, pt.h_acid, pt.o_acid
    water_o = pt.water_o

    cv = proton_transfer_cv(centroid, n_amine, h_acid, o_acid)
    delta = cv["delta"].to_numpy()
    deloc = proton_delocalization(beads, n_amine, h_acid, o_acid)
    proton_rg = deloc.proton_rg

    if water_o is not None:
        water_dist = np.linalg.norm(
            centroid[:, n_amine, :] - centroid[:, water_o, :], axis=1
        )
    else:
        water_dist = np.full(traj.n_frames, np.nan)

    # Fixed limits from all beads in the window
    win_beads = beads[start:end].reshape(-1, 3)
    cmin = win_beads.min(axis=0)
    cmax = win_beads.max(axis=0)
    pad = 0.25
    span = float(np.max(cmax - cmin)) + 2 * pad
    mid = 0.5 * (cmin + cmax)
    lims = [(mid[k] - span / 2, mid[k] + span / 2) for k in range(3)]

    # Which atoms get a hull surface
    if args.hull_atoms == "proton":
        hull_idx = {h_acid}
    elif args.hull_atoms == "light":
        hull_idx = {i for i, s in enumerate(species) if s == "H"}
    else:
        hull_idx = set(range(len(species)))

    out = args.output or (
        ROOT / "output" / "nh4no3_pt"
        / f"field_movie_{args.system}_{args.method}.gif"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1, 1])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_d = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, 1])

    fr = np.array(frames)
    ax_d.plot(fr, delta[start:end], color="#333333", lw=1.3)
    ax_d.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax_d.axhspan(-0.05, 0.05, color="gold", alpha=0.18)
    ax_d.set_ylabel(r"$\delta = d_{N-H}-d_{O-H}$ [Å]")
    ax_d.set_title("Proton-transfer CV")
    d_cursor = ax_d.axvline(frames[0], color="crimson", lw=1.5)

    ax_r.plot(fr, proton_rg[start:end], color="#b8860b", lw=1.3)
    ax_r.set_ylabel(r"proton $R_g$ [Å]")
    ax_r.set_title("Proton quantum delocalization (bead spread)")
    ax_r.set_xlabel("frame")
    r_cursor = ax_r.axvline(frames[0], color="crimson", lw=1.5)

    def draw_frame(fi: int):
        ax3d.clear()
        pos = centroid[fi]
        bead_pos = beads[fi]  # (B, A, 3)

        # Covalent bonds on centroid
        for i, j in covalent_bonds(pos, species):
            ax3d.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                [pos[i, 2], pos[j, 2]],
                color="#888888",
                lw=1.6,
                zorder=1,
            )

        # Bead point clouds + hull "field" per atom
        for idx, sym in enumerate(species):
            base_color, csize = _ELEMENT_STYLE.get(sym, ("#999999", 200.0))
            pts = bead_pos[:, idx, :]  # (B, 3)

            if idx == h_acid:
                bcolor, balpha, bsize = "#ffcc00", 0.32, 130.0
                hcolor, halpha = "#ffcc00", 0.30
            elif water_o is not None and idx == water_o:
                bcolor, balpha, bsize = "#00c2ff", 0.26, 110.0
                hcolor, halpha = "#00c2ff", 0.18
            elif water_o is not None and idx in pt.water_h:
                bcolor, balpha, bsize = "#aee7ff", 0.24, 90.0
                hcolor, halpha = "#aee7ff", 0.16
            else:
                bcolor, balpha, bsize = base_color, 0.16, 75.0
                hcolor, halpha = base_color, 0.13

            ax3d.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                s=bsize,
                c=bcolor,
                alpha=balpha,
                edgecolors="none",
                depthshade=True,
                zorder=2,
            )

            if idx in hull_idx:
                faces = _hull_faces(pts)
                if faces is not None:
                    poly = Poly3DCollection(
                        faces, alpha=halpha, facecolor=hcolor, edgecolor="none"
                    )
                    ax3d.add_collection3d(poly)

        # Centroid spheres (solid) on top
        for idx, sym in enumerate(species):
            color, csize = _ELEMENT_STYLE.get(sym, ("#999999", 200.0))
            edge = "black"
            if idx == h_acid:
                color, csize, edge = "#ffcc00", 120.0, "#b8860b"
            elif water_o is not None and idx == water_o:
                color, csize, edge = "#00c2ff", 220.0, "#0077aa"
            elif water_o is not None and idx in pt.water_h:
                color, csize, edge = "#aee7ff", 80.0, "#4a90a4"
            ax3d.scatter(
                pos[idx, 0],
                pos[idx, 1],
                pos[idx, 2],
                s=csize,
                c=color,
                edgecolors=edge,
                linewidths=0.8,
                depthshade=True,
                zorder=4,
            )

        # Water approach line: N...Ow
        if water_o is not None:
            ax3d.plot(
                [pos[n_amine, 0], pos[water_o, 0]],
                [pos[n_amine, 1], pos[water_o, 1]],
                [pos[n_amine, 2], pos[water_o, 2]],
                color="#00c2ff",
                ls=":",
                lw=2.2,
                zorder=3,
            )

        ax3d.set_xlim(*lims[0])
        ax3d.set_ylim(*lims[1])
        ax3d.set_zlim(*lims[2])
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
        ax3d.view_init(elev=args.elev, azim=args.azim + args.spin * (fi - start))

        dv = delta[fi]
        if dv > 0.05:
            state, scolor = "proton on ACID (O-H)", "#ff3b30"
        elif dv < -0.05:
            state, scolor = "TRANSFERRED to AMINE (N-H)", "#2b6cff"
        else:
            state, scolor = "SHARED proton", "#b8860b"

        wtxt = (
            f"   |   N$\\cdots$O$_w$ = {water_dist[fi]:.2f} Å"
            if water_o is not None
            else ""
        )
        ax3d.set_title(
            f"{args.system} {args.method}  frame {fi}  "
            f"({traj.n_beads} beads)\n"
            f"$\\delta$ = {dv:+.2f} Å   ({state}){wtxt}",
            color=scolor,
            fontsize=11,
        )
        ax3d.text2D(
            0.02,
            0.02,
            f"proton $R_g$ = {proton_rg[fi]:.3f} Å  (quantum spread)",
            transform=ax3d.transAxes,
            fontsize=9,
            color="#b8860b",
        )

        d_cursor.set_xdata([fi, fi])
        r_cursor.set_xdata([fi, fi])
        return ()

    print(
        f"Rendering {len(frames)} frames "
        f"(center={center}, window=[{start},{end}), beads={traj.n_beads}) -> {out}"
    )
    anim = animation.FuncAnimation(
        fig, draw_frame, frames=frames, interval=1000 / args.fps, blit=False
    )
    writer = animation.PillowWriter(fps=args.fps)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    anim.save(str(out), writer=writer, dpi=110)

    still = out.with_name(out.stem + "_peak.png")
    draw_frame(center)
    fig.savefig(still, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {still}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
