#!/usr/bin/env python3
"""
2D PCA scatter plot for MD trajectories.

Supports two modes:
  cartesian  – PCA on aligned Cartesian coordinates (default).
  dihedral   – PCA on sin/cos-transformed torsion angles.
               Works for any molecule (not just proteins).
               Auto-detects all proper torsions from the bond graph,
               or accepts explicit atom-name quadruplets via --dihedral-atoms.
               Rotation- and translation-invariant; no alignment needed.

Produces a PC1 vs PC2 scatter plot colored by frame index (time) or cluster label.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute and plot 2D PCA (PC1 vs PC2) from an MD trajectory.",
    )
    p.add_argument("--top", required=True, help="Topology file (PDB/PRMTOP/PSF/…)")
    p.add_argument(
        "--traj", required=True, nargs="+", help="One or more trajectory files"
    )
    p.add_argument(
        "--mode",
        choices=["cartesian", "dihedral", "angles"],
        default="cartesian",
        help="PCA feature space: 'cartesian' (positions), 'dihedral' (torsions), "
             "or 'angles' (inter-face normal angles + planarity only). Default: cartesian",
    )
    p.add_argument(
        "--selection",
        default="all",
        help="Atom selection for Cartesian PCA (default: all). Ignored in dihedral mode.",
    )
    p.add_argument(
        "--align-sel",
        default=None,
        help="Atom selection for alignment (default: same as --selection). Ignored in dihedral mode.",
    )
    p.add_argument(
        "--dihedral-sel",
        default="resid 1-6",
        help="Residue selection for dihedral PCA (default: resid 1-6)",
    )
    p.add_argument(
        "--dihedral-atoms",
        nargs="+",
        default=None,
        metavar="A1,A2,A3,A4",
        help="Explicit dihedral definitions as comma-separated atom-name quadruplets, "
             "e.g. 'C1,C2,C3,C4' 'C2,C3,C4,C5'. Applied to every residue in --dihedral-sel. "
             "If omitted, all proper torsions are auto-detected from the bond graph.",
    )
    p.add_argument(
        "--dihedral-heavy-only",
        action="store_true",
        default=True,
        help="When auto-detecting, skip torsions involving hydrogen (default: True)",
    )
    p.add_argument(
        "--no-dihedral-heavy-only",
        action="store_false",
        dest="dihedral_heavy_only",
        help="Include hydrogen torsions in auto-detection",
    )
    p.add_argument(
        "--include-angles",
        action="store_true",
        help="Include inter-face geometry as additional features. "
             "Each residue is treated as a planar molecule; a best-fit plane "
             "is computed to get the face normal. Features: pairwise normal "
             "angles (C(6,2)=15 × sin/cos = 30) + per-face planarity RMS (6). "
             "Total: 36 extra features for 6 residues. Used in dihedral mode.",
    )
    p.add_argument(
        "--n-components",
        type=int,
        default=5,
        help="Number of PCA components to compute (default: 5; first two are plotted)",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth frame (default: 1 = all frames)",
    )
    p.add_argument(
        "--cluster",
        action="store_true",
        help="Cluster frames in PC space and color by cluster label",
    )
    p.add_argument(
        "--out-prefix",
        default="pca_2d",
        help="Output prefix for CSV and PNG (default: pca_2d)",
    )
    return p.parse_args()


def _find_torsions_from_bonds(residue, heavy_only: bool = True) -> list[tuple[str, ...]]:
    """Auto-detect proper torsion quadruplets from a residue's bond graph.

    Walks every bond path of length 4 (a-b-c-d) and returns unique
    canonical quadruplets as tuples of atom names.
    """
    atoms = residue.atoms
    if heavy_only:
        atoms = atoms.select_atoms("not name H*")

    name_to_idx = {a.name: a.index for a in atoms}
    idx_set = set(a.index for a in atoms)

    adj: dict[int, list[int]] = {a.index: [] for a in atoms}
    for bond in residue.atoms.bonds:
        i, j = bond.atoms[0].index, bond.atoms[1].index
        if i in idx_set and j in idx_set:
            adj[i].append(j)
            adj[j].append(i)

    idx_to_name = {a.index: a.name for a in atoms}
    seen: set[tuple[str, ...]] = set()
    torsions: list[tuple[str, ...]] = []

    for b in idx_set:
        for a in adj[b]:
            for c in adj[b]:
                if c == a:
                    continue
                for d in adj[c]:
                    if d == b or d == a:
                        continue
                    quad = (idx_to_name[a], idx_to_name[b],
                            idx_to_name[c], idx_to_name[d])
                    canon = quad if quad < quad[::-1] else quad[::-1]
                    if canon not in seen:
                        seen.add(canon)
                        torsions.append(quad)
    return torsions


def _fit_plane_normal(positions: np.ndarray) -> tuple[float, np.ndarray]:
    """Fit a plane to positions via SVD, return (planar_rms, unit_normal)."""
    P0 = positions - positions.mean(axis=0)
    _, S, Vt = np.linalg.svd(P0, full_matrices=False)
    normal = Vt[-1]
    dist = np.abs(P0 @ normal)
    planar_rms = float(np.sqrt((dist ** 2).mean()))
    return planar_rms, normal


def _collect_inter_residue_angles(
    u,
    dihedral_sel: str,
    stride: int,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Compute inter-face angles using best-fit plane normals per frame.

    Each residue is treated as a planar molecule. For every frame:
      1. Fit a plane to each residue → face normal (oriented outward).
      2. Compute pairwise normal angles: C(N,2) = 15 for N=6.
      3. Compute per-face planarity RMS (deviation from plane): N values.

    Features per frame = C(N,2) normal angles × sin/cos + N planarities
                       = 15×2 + 6 = 36 for N=6.

    Returns (feature_matrix, frame_indices, feature_names).
    """
    from itertools import combinations

    residues = u.select_atoms(dihedral_sel).residues
    n_res = len(residues)
    resids = [r.resid for r in residues]
    pairs = list(combinations(range(n_res), 2))

    print(f"  Face-normal angles: {len(pairs)} pairs from {n_res} planar faces")
    print(f"  Planarity features: {n_res} (per-face RMS deviation from plane)")

    res_atom_groups = [res.atoms for res in residues]

    frames_used: list[int] = []
    all_normal_angles: list[np.ndarray] = []
    all_planarities: list[np.ndarray] = []

    for ts in u.trajectory[:: stride]:
        frames_used.append(ts.frame)

        centroids = np.array([ag.center_of_mass() for ag in res_atom_groups])
        cube_center = centroids.mean(axis=0)

        normals = np.empty((n_res, 3))
        planarities = np.empty(n_res)
        for r_idx, ag in enumerate(res_atom_groups):
            prms, n_vec = _fit_plane_normal(ag.positions)
            planarities[r_idx] = prms
            # orient normal outward (pointing away from cube center)
            outward = centroids[r_idx] - cube_center
            if np.dot(n_vec, outward) < 0:
                n_vec = -n_vec
            normals[r_idx] = n_vec

        frame_angles = np.empty(len(pairs))
        for idx, (i, j) in enumerate(pairs):
            cos_val = np.clip(np.dot(normals[i], normals[j]), -1.0, 1.0)
            frame_angles[idx] = np.arccos(cos_val)
        all_normal_angles.append(frame_angles)
        all_planarities.append(planarities)

    angles_rad = np.array(all_normal_angles)  # (n_frames, n_pairs)
    planarities = np.array(all_planarities)   # (n_frames, n_res)

    sin_block = np.sin(angles_rad)
    cos_block = np.cos(angles_rad)

    X = np.hstack([sin_block, cos_block, planarities])

    angle_labels = [f"n{resids[i]}-n{resids[j]}" for i, j in pairs]
    feat_names = (
        [f"sin(∠{l})" for l in angle_labels]
        + [f"cos(∠{l})" for l in angle_labels]
        + [f"planarity_r{rid}" for rid in resids]
    )
    return X, frames_used, feat_names


def _collect_dihedrals(
    u,
    dihedral_sel: str,
    dihedral_atom_defs: list[str] | None,
    heavy_only: bool,
    stride: int,
) -> tuple[np.ndarray, list[int], list[str]]:
    """Compute sin/cos-transformed dihedrals per frame for arbitrary molecules.

    If *dihedral_atom_defs* is given (e.g. ["C1,C2,C3,C4", "C2,C3,C4,C5"]),
    those atom-name quadruplets are used for every residue in the selection.
    Otherwise torsions are auto-detected from the bond graph of the first
    residue (all residues are assumed identical).

    Returns (feature_matrix, frame_indices, feature_names).
    """
    from MDAnalysis.lib.distances import calc_dihedrals

    residues = u.select_atoms(dihedral_sel).residues
    if len(residues) == 0:
        sys.exit(f"ERROR: dihedral selection '{dihedral_sel}' matched 0 residues.")

    # ── resolve quadruplets ──────────────────────────────────────────
    if dihedral_atom_defs is not None:
        quads = [tuple(d.split(",")) for d in dihedral_atom_defs]
        for q in quads:
            if len(q) != 4:
                sys.exit(f"ERROR: dihedral definition must have 4 atom names, got {q}")
    else:
        ref_res = residues[0]
        try:
            quads = _find_torsions_from_bonds(ref_res, heavy_only=heavy_only)
        except (AttributeError, Exception):
            sys.exit(
                "ERROR: topology has no bond information — cannot auto-detect "
                "torsions. Supply explicit definitions via --dihedral-atoms "
                "or use a topology format that includes bonds (e.g. PDB, MOL2, PRMTOP)."
            )
        if not quads:
            sys.exit(
                "ERROR: no torsions detected. Check your --dihedral-sel or "
                "provide explicit --dihedral-atoms."
            )

    n_torsions_per_res = len(quads)
    n_res = len(residues)
    total_torsions = n_torsions_per_res * n_res
    print(f"  {n_torsions_per_res} torsion(s)/residue × {n_res} residue(s) = {total_torsions} total")
    for i, q in enumerate(quads):
        print(f"    τ{i}: {'-'.join(q)}")

    # ── build atom-index lists for vectorised calc_dihedrals ─────────
    #    shape per group: (total_torsions,) indexing into universe.atoms
    quad_global_indices: list[list[int]] = [[] for _ in range(4)]
    labels: list[str] = []
    for res in residues:
        res_atoms = res.atoms
        name_map = {a.name: a.index for a in res_atoms}
        for q in quads:
            missing = [n for n in q if n not in name_map]
            if missing:
                sys.exit(
                    f"ERROR: atom(s) {missing} not found in resid {res.resid}. "
                    f"Available: {sorted(name_map.keys())}"
                )
            for k in range(4):
                quad_global_indices[k].append(name_map[q[k]])
            labels.append(f"r{res.resid}_{'-'.join(q)}")

    idx_arrays = [np.array(gi) for gi in quad_global_indices]

    # ── iterate frames and compute angles ────────────────────────────
    frames_used: list[int] = []
    all_angles: list[np.ndarray] = []

    for ts in u.trajectory[:: stride]:
        frames_used.append(ts.frame)
        pos = u.atoms.positions
        angles = calc_dihedrals(
            pos[idx_arrays[0]],
            pos[idx_arrays[1]],
            pos[idx_arrays[2]],
            pos[idx_arrays[3]],
        )
        all_angles.append(angles)

    angles_rad = np.array(all_angles)  # (n_frames, total_torsions), already radians

    # ── sin/cos transform ────────────────────────────────────────────
    sin_block = np.sin(angles_rad)
    cos_block = np.cos(angles_rad)
    X = np.hstack([sin_block, cos_block])  # (n_frames, 2 * total_torsions)

    feat_names = [f"sin({l})" for l in labels] + [f"cos({l})" for l in labels]
    return X, frames_used, feat_names


def main() -> None:
    args = parse_args()

    import warnings

    import matplotlib.pyplot as plt
    import MDAnalysis as mda

    from src.TrajectoryMetrics.TrajectoryMetrics import pca_from_coords

    # ── load ──────────────────────────────────────────────────────────
    print(f"Loading topology: {args.top}")
    print(f"Loading trajectory: {args.traj}")
    try:
        u = mda.Universe(args.top, *args.traj)
    except Exception as e:
        warnings.warn(f"Standard load failed ({e}); retrying with format='TRJ'")
        u = mda.Universe(args.top, *args.traj, format="TRJ")

    n_frames_total = len(u.trajectory)
    print(f"Trajectory has {n_frames_total} frames (stride={args.stride})")
    print(f"Mode: {args.mode}")

    # ── build feature matrix ──────────────────────────────────────────
    if args.mode == "angles":
        print(f"Angle selection: '{args.dihedral_sel}'")
        X, frames_used, feat_names = _collect_inter_residue_angles(
            u, args.dihedral_sel, args.stride,
        )
        n_feat = X.shape[1]
        print(f"Total feature matrix: {len(frames_used)} frames × {n_feat} features")

    elif args.mode == "dihedral":
        print(f"Dihedral selection: '{args.dihedral_sel}'")
        if args.dihedral_atoms:
            print(f"Explicit dihedral defs: {args.dihedral_atoms}")
        else:
            print("Auto-detecting torsions from bond graph …")
        X, frames_used, feat_names = _collect_dihedrals(
            u, args.dihedral_sel, args.dihedral_atoms,
            args.dihedral_heavy_only, args.stride,
        )
        n_torsion_feat = X.shape[1]
        print(f"Torsion features: {n_torsion_feat} ({n_torsion_feat // 2} torsions × sin/cos)")

        if args.include_angles:
            X_ang, _, ang_names = _collect_inter_residue_angles(
                u, args.dihedral_sel, args.stride,
            )
            if len(X_ang) != len(X):
                sys.exit("ERROR: frame count mismatch between torsions and angles.")
            X = np.hstack([X, X_ang])
            feat_names.extend(ang_names)
            print(f"Face-geometry features: {X_ang.shape[1]} (normal angles + planarity)")

        n_feat = X.shape[1]
        print(f"Total feature matrix: {len(frames_used)} frames × {n_feat} features")
    else:
        from MDAnalysis.analysis import align as mda_align

        align_sel = args.align_sel or args.selection

        print(f"Aligning on '{align_sel}' …")
        mda_align.AlignTraj(u, u, select=align_sel, in_memory=True).run()

        sel = u.select_atoms(args.selection)
        if len(sel) == 0:
            sys.exit(f"ERROR: selection '{args.selection}' matched 0 atoms.")
        print(f"PCA selection '{args.selection}' → {len(sel)} atoms")

        frames_used: list[int] = []
        coords_list: list[np.ndarray] = []
        for ts in u.trajectory[:: args.stride]:
            frames_used.append(ts.frame)
            coords_list.append(sel.positions.copy())
        X = np.array(coords_list)

    # ── PCA ───────────────────────────────────────────────────────────
    if args.mode in ("dihedral", "angles"):
        from sklearn.decomposition import PCA

        mode_tag = "angle" if args.mode == "angles" else "dihedral"
        n_comp = min(args.n_components, len(frames_used) - 1, X.shape[1])
        print(f"Running {mode_tag} PCA ({n_comp} components on {len(frames_used)} frames) …")
        Xc = X - X.mean(axis=0)
        pca_model = PCA(n_components=n_comp, svd_solver="auto")
        pcs = pca_model.fit_transform(Xc)
    else:
        n_comp = min(args.n_components, len(frames_used) - 1, X.shape[1] * 3)
        print(f"Running Cartesian PCA ({n_comp} components on {len(frames_used)} frames) …")
        pcs, pca_model = pca_from_coords(X, n_components=n_comp)

    var = pca_model.explained_variance_ratio_ * 100

    print("Explained variance:")
    for i, v in enumerate(var):
        print(f"  PC{i + 1}: {v:.2f}%")

    # ── optional clustering ───────────────────────────────────────────
    labels = None
    if args.cluster:
        from src.ClusteringAnalysis import ClusteringAnalysis

        ca = ClusteringAnalysis()
        labels, medoids = ca.cluster_frames(pcs[:, :min(5, pcs.shape[1])])
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Clustering: {n_clusters} clusters found")

    # ── save CSV ──────────────────────────────────────────────────────
    out_dir = Path(args.out_prefix).parent
    if str(out_dir) not in ("", "."):
        out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "frame": frames_used,
            **{f"PC{i + 1}": pcs[:, i] for i in range(pcs.shape[1])},
        }
    )
    if labels is not None:
        df["cluster"] = labels
    csv_path = f"{args.out_prefix}_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved PCA scores → {csv_path}")

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    if labels is not None:
        unique_labels = sorted(set(labels))
        cmap = plt.cm.get_cmap("tab10", len(unique_labels))
        for i, cl in enumerate(unique_labels):
            mask = labels == cl
            lbl = "noise" if cl == -1 else f"cluster {cl}"
            ax.scatter(
                pcs[mask, 0],
                pcs[mask, 1],
                s=8,
                alpha=0.7,
                color=cmap(i),
                label=lbl,
            )
        ax.legend(fontsize=8, markerscale=2.5)
    else:
        sc = ax.scatter(
            pcs[:, 0],
            pcs[:, 1],
            c=frames_used,
            cmap="viridis",
            s=8,
            alpha=0.7,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("Frame index", fontsize=10)

    mode_labels = {"cartesian": "Cartesian", "dihedral": "Dihedral", "angles": "Face-Normal Angle"}
    mode_label = mode_labels[args.mode]
    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=12)
    ax.set_title(f"2D {mode_label} PCA — PC1 vs PC2", fontsize=13)
    fig.tight_layout()

    png_path = f"{args.out_prefix}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot     → {png_path}")


if __name__ == "__main__":
    main()
