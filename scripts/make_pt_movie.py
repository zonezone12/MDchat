"""
Animate an NH3...HNO3(+H2O) proton-transfer event and highlight how the water
molecule approaches the amine nitrogen and stabilizes/induces the transfer.

Left panel: 3D molecular scene (centroid geometry) with the transferring proton
in gold, the donor O / acceptor N dashed contacts, and the water molecule with a
cyan "water approach" dashed line to the amine N.

Right panels: delta = d(N-H) - d(O-H) and the water-approach distance (N...Ow)
over the window, with a moving cursor.

Output: an animated GIF (ffmpeg not required).

Example
-------
python scripts/make_pt_movie.py                       # m1n1 PIMD, deepest event
python scripts/make_pt_movie.py --method CLMD
python scripts/make_pt_movie.py --center 2651 --half 45 --fps 12
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
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: F401

from src.ProtonTransfer import detect_pt_atoms, load_pimd_xyz, proton_transfer_cv

# Element display: (color, marker size)
_ELEMENT_STYLE: Dict[str, Tuple[str, float]] = {
    "N": ("#2b6cff", 320.0),
    "O": ("#ff3b30", 300.0),
    "H": ("#d8d8d8", 130.0),
    "C": ("#444444", 300.0),
}
_COVALENT_CUTOFF = 1.35  # Angstrom, for drawing covalent bonds


def _job_relpath(system: str, method: str) -> str:
    folder = {"m1n0": "2-m1n0", "m1n1": "3-m1n1"}[system]
    sub = {"CLMD": "1-CLMD", "PIMD": "2-PIMD"}[method]
    return f"{folder}/{sub}/Result/coor.xyz"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Proton-transfer + water-approach movie")
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
        help="Output GIF path (default: output/nh4no3_pt/movie_<system>_<method>.gif)",
    )
    p.add_argument(
        "--elev", type=float, default=18.0, help="3D elevation angle (deg)"
    )
    p.add_argument(
        "--azim", type=float, default=-60.0, help="3D azimuth angle (deg)"
    )
    p.add_argument(
        "--spin",
        type=float,
        default=0.4,
        help="Azimuth degrees added per frame (slow turntable)",
    )
    return p.parse_args()


def covalent_bonds(coords: np.ndarray, species: List[str]) -> List[Tuple[int, int]]:
    bonds = []
    for i, j in combinations(range(len(species)), 2):
        d = float(np.linalg.norm(coords[i] - coords[j]))
        # Skip H-H; only bond if within covalent cutoff
        if species[i] == "H" and species[j] == "H":
            continue
        if d <= _COVALENT_CUTOFF:
            bonds.append((i, j))
    return bonds


def main() -> int:
    args = parse_args()
    relpath = _job_relpath(args.system, args.method)
    xyz = (args.traj_root / relpath).resolve()
    if not xyz.is_file():
        print(f"Missing trajectory: {xyz}", file=sys.stderr)
        return 1
    n_physical = 9 if args.system == "m1n0" else 12

    # Determine center frame. If auto, we must scan delta first (cheap load capped).
    if args.center < 0:
        print("Auto-picking deepest transfer frame (scanning delta)...")
        traj_full = load_pimd_xyz(xyz, n_physical=n_physical)
        pt = detect_pt_atoms(traj_full.species, traj_full.centroid[0])
        cv = proton_transfer_cv(
            traj_full.centroid, pt.n_amine, pt.h_acid, pt.o_acid
        )
        center = int(np.argmin(cv["delta"].to_numpy()))
        traj = traj_full
    else:
        center = args.center
        # Load only up to the window end to save time on huge PIMD files.
        traj = load_pimd_xyz(
            xyz, n_physical=n_physical, max_frames=center + args.half + 2
        )
        pt = detect_pt_atoms(traj.species, traj.centroid[0])

    if args.system != "m1n1" or pt.water_o is None:
        print(
            "NOTE: no water in this system; the movie will show transfer only.",
            file=sys.stderr,
        )

    start = max(0, center - args.half)
    end = min(traj.n_frames, center + args.half + 1)
    frames = list(range(start, end))
    coords = traj.centroid  # (F, n_atoms, 3)
    species = traj.species

    cv = proton_transfer_cv(coords, pt.n_amine, pt.h_acid, pt.o_acid)
    delta = cv["delta"].to_numpy()

    n_amine, h_acid, o_acid = pt.n_amine, pt.h_acid, pt.o_acid
    water_o = pt.water_o
    # Water-approach distance: amine N to water O (falls at transfer)
    if water_o is not None:
        water_dist = np.linalg.norm(
            coords[:, n_amine, :] - coords[:, water_o, :], axis=1
        )
    else:
        water_dist = np.full(traj.n_frames, np.nan)

    # Fixed 3D limits from the window (with padding), centered on N0-O5 midpoint
    win = coords[start:end]
    cmin = win.reshape(-1, 3).min(axis=0)
    cmax = win.reshape(-1, 3).max(axis=0)
    pad = 0.6
    span = float(np.max(cmax - cmin)) + 2 * pad
    mid = 0.5 * (cmin + cmax)
    lims = [(mid[k] - span / 2, mid[k] + span / 2) for k in range(3)]

    out = args.output or (
        ROOT / "output" / "nh4no3_pt" / f"movie_{args.system}_{args.method}.gif"
    )
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1, 1])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_d = fig.add_subplot(gs[0, 1])
    ax_w = fig.add_subplot(gs[1, 1])

    # Static timeseries context (whole window)
    fr = np.array(frames)
    ax_d.plot(fr, delta[start:end], color="#333333", lw=1.3)
    ax_d.axhline(0.0, color="gray", ls=":", lw=1.0)
    ax_d.axhspan(-0.05, 0.05, color="gold", alpha=0.18)
    ax_d.set_ylabel(r"$\delta = d_{N-H}-d_{O-H}$ [Å]")
    ax_d.set_title("Proton-transfer CV")
    d_cursor = ax_d.axvline(frames[0], color="crimson", lw=1.5)

    if water_o is not None:
        ax_w.plot(fr, water_dist[start:end], color="#0aa0c0", lw=1.3)
        ax_w.set_ylabel(r"$d(\mathrm{N}\cdots\mathrm{O_w})$ [Å]")
        ax_w.set_title("Water approach to amine N")
    ax_w.set_xlabel("frame")
    w_cursor = ax_w.axvline(frames[0], color="crimson", lw=1.5)

    def draw_frame(fi: int):
        ax3d.clear()
        pos = coords[fi]

        # Covalent bonds
        for i, j in covalent_bonds(pos, species):
            ax3d.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                [pos[i, 2], pos[j, 2]],
                color="#777777",
                lw=2.0,
                zorder=1,
            )

        # Atoms
        for idx, sym in enumerate(species):
            color, size = _ELEMENT_STYLE.get(sym, ("#999999", 200.0))
            edge = "black"
            if idx == h_acid:
                color, size, edge = "#ffcc00", 220.0, "#b8860b"  # transferring proton
            elif water_o is not None and idx == water_o:
                color, size, edge = "#00c2ff", 320.0, "#0077aa"  # water O
            elif water_o is not None and idx in pt.water_h:
                color, size, edge = "#aee7ff", 130.0, "#4a90a4"  # water H
            ax3d.scatter(
                pos[idx, 0],
                pos[idx, 1],
                pos[idx, 2],
                s=size,
                c=color,
                edgecolors=edge,
                linewidths=0.8,
                depthshade=True,
                zorder=3,
            )

        # Transfer contacts: N...H (acceptor) and O...H (donor)
        d_nh = float(np.linalg.norm(pos[n_amine] - pos[h_acid]))
        d_oh = float(np.linalg.norm(pos[o_acid] - pos[h_acid]))
        ax3d.plot(
            [pos[n_amine, 0], pos[h_acid, 0]],
            [pos[n_amine, 1], pos[h_acid, 1]],
            [pos[n_amine, 2], pos[h_acid, 2]],
            color="#2b6cff",
            ls="--",
            lw=1.8,
            zorder=2,
        )
        ax3d.plot(
            [pos[o_acid, 0], pos[h_acid, 0]],
            [pos[o_acid, 1], pos[h_acid, 1]],
            [pos[o_acid, 2], pos[h_acid, 2]],
            color="#ff3b30",
            ls="--",
            lw=1.8,
            zorder=2,
        )

        # Water approach: N...Ow
        if water_o is not None:
            ax3d.plot(
                [pos[n_amine, 0], pos[water_o, 0]],
                [pos[n_amine, 1], pos[water_o, 1]],
                [pos[n_amine, 2], pos[water_o, 2]],
                color="#00c2ff",
                ls=":",
                lw=2.2,
                zorder=2,
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
            state = "proton on ACID (O-H)"
            scolor = "#ff3b30"
        elif dv < -0.05:
            state = "TRANSFERRED to AMINE (N-H)"
            scolor = "#2b6cff"
        else:
            state = "SHARED proton"
            scolor = "#b8860b"

        wtxt = (
            f"   |   N$\\cdots$O$_w$ = {water_dist[fi]:.2f} Å"
            if water_o is not None
            else ""
        )
        ax3d.set_title(
            f"{args.system} {args.method}  frame {fi}\n"
            f"$\\delta$ = {dv:+.2f} Å   ({state}){wtxt}",
            color=scolor,
            fontsize=11,
        )
        # small text with the two defining distances
        ax3d.text2D(
            0.02,
            0.02,
            f"d(N–H)={d_nh:.2f}  d(O–H)={d_oh:.2f} Å",
            transform=ax3d.transAxes,
            fontsize=9,
            color="#333333",
        )

        d_cursor.set_xdata([fi, fi])
        w_cursor.set_xdata([fi, fi])
        return ()

    print(
        f"Rendering {len(frames)} frames "
        f"(center={center}, window=[{start},{end})) -> {out}"
    )
    anim = animation.FuncAnimation(
        fig, draw_frame, frames=frames, interval=1000 / args.fps, blit=False
    )
    writer = animation.PillowWriter(fps=args.fps)
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    anim.save(str(out), writer=writer, dpi=110)

    # Static "peak transfer" still for the summary/report
    still = out.with_name(out.stem + "_peak.png")
    draw_frame(center)
    fig.savefig(still, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {still}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
