#!/usr/bin/env python3
"""Decide which monomer relabellings are physically admissible (Test D1).

Loads one compact reference cage from a local trajectory, tests the 48
face-adjacency automorphisms by superposition (rotations only, then allowing
a reflection), and records a site map for each contact-graph automorphism.

Pole-hydrogen cubes have no local trajectory here: they inherit the result of
the pole-methyl cube with the same equator and are flagged ``inferred``.

Example
-------
python scripts/check_cage_symmetry.py --cube BHHpM \\
    --topology traj/BHHpM_ca.prmtop --trajectory traj/BHHpM_109345_mdcrd_v.trj
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.ChangepointAnalysis.cluster_k_diagnostics import load_segments_at_k
from src.ChangepointAnalysis.cluster_validation import kabsch_rmsd
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.metastable_catalog import (
    N_MONOMERS,
    adjacency_automorphisms,
    contact_automorphisms,
    is_proper_permutation,
    perm_cycles,
)
from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir
from src.EndpointAnalyzer import EndpointAnalyzer
from src.utils.gsa_selections import resolve_selections

POLE_H_FROM_POLE_M = {
    "BHHpH": "BHHpM",
    "BMHpH": "BMHpM",
    "BMMpH": "BMMpM",
}
DEFAULT_LOCAL_TRAJ = {
    "BHHpM": ("traj/BHHpM_ca.prmtop", "traj/BHHpM_109345_mdcrd_v.trj"),
    "BMHpM": ("traj/BMHpM_ca.prmtop", "traj/BMHpM_109345_mdcrd_v.trj"),
    "BMMpM": ("traj/BMMpM_ca.prmtop", "traj/BMMpM_891249_mdcrd_v.trj"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Superposition test of cage symmetry operations.")
    p.add_argument("--cube", required=True, choices=list(KNOWN_CUBES))
    p.add_argument("--topology", type=Path)
    p.add_argument("--trajectory", type=Path)
    p.add_argument("--trajectory-format", default="TRJ")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: output/endpoint_cluster_validation_physical_k/symmetry",
    )
    p.add_argument("--n-ref-frames", type=int, default=50)
    p.add_argument("--own-elbow", action="store_true", default=True)
    p.add_argument("--k", type=int)
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    return p.parse_args()


def _selected_k(path: Path, cube: str, fallback: int | None) -> int | None:
    if fallback is not None:
        return int(fallback)
    if path.is_file():
        table = pd.read_csv(path)
        hit = table.loc[table["cohort"].astype(str) == cube]
        if not hit.empty and "selected_k" in hit.columns:
            return int(hit["selected_k"].iloc[0])
    return None


def _compact_frames(
    output_root: Path,
    cube: str,
    traj_stem: str,
    n_frames: int,
    *,
    k: int | None,
) -> list[int]:
    gsa_dir = output_root / "gsa_features_by_cube" / cube
    paths = sorted(gsa_dir.glob("*_gsa_features.csv")) if gsa_dir.is_dir() else []
    frames = None
    for path in paths:
        if traj_stem in path.name:
            df = pd.read_csv(path, usecols=lambda c: c in {"traj_id", "frame", "assembly_rg"})
            frames = df
            break
    if frames is None or frames.empty:
        return list(range(0, n_frames * 5, 5))[:n_frames]

    segs = pd.DataFrame()
    if k is not None:
        cp = endpoint_changepoints_dir(output_root, cube, own_elbow=True)
        if not cp.is_dir():
            cp = endpoint_changepoints_dir(output_root, cube, own_elbow=False)
        segs = load_segments_at_k(cp, k, group="endpoint")
        if not segs.empty:
            segs = segs[segs["traj_id"].astype(str).str.contains(traj_stem)]
    if not segs.empty and "assembly_rg" in frames.columns:
        joined = frames.copy()
        joined["_keep"] = False
        fr = pd.to_numeric(joined["frame"], errors="coerce")
        for rec in segs.itertuples(index=False):
            joined.loc[(fr >= rec.start_frame) & (fr <= rec.end_frame), "_keep"] = True
        kept = joined[joined["_keep"]]
        if not kept.empty:
            frames = kept
    frames = frames.sort_values("assembly_rg") if "assembly_rg" in frames.columns else frames
    # Prefer compact frames that appear early so a sequential TRJ read can stop.
    early = frames[pd.to_numeric(frames["frame"], errors="coerce") <= 800]
    pool = early if len(early) >= n_frames else frames
    picked = pool.head(max(n_frames * 3, n_frames))
    idx = picked["frame"].astype(int).to_numpy()
    if idx.size > n_frames:
        step = max(1, idx.size // n_frames)
        idx = np.sort(idx[::step][:n_frames])
    return [int(x) for x in idx]


def _heavy_indices(atomgroup) -> np.ndarray:
    names = np.array([a.name for a in atomgroup])
    elements = np.array([getattr(a, "element", "") for a in atomgroup])
    heavy = np.array(
        [
            (el and el.upper() != "H") or (not el and not str(name).upper().startswith("H"))
            for el, name in zip(elements, names)
        ]
    )
    return np.where(heavy)[0]


def _graph_automorphisms(atomgroup) -> list[np.ndarray]:
    """Heavy-atom index permutations that are RDKit graph automorphisms."""
    heavy = _heavy_indices(atomgroup)
    identity = [np.arange(heavy.size, dtype=int)]
    try:
        mol = atomgroup.convert_to("RDKIT")
    except Exception:
        return identity
    if mol is None:
        return identity
    matches = mol.GetSubstructMatches(mol, uniquify=False, maxMatches=32)
    if not matches:
        return identity
    heavy_set = set(int(x) for x in heavy)
    # convert_to RDKit uses the atomgroup order as RDKit indices
    perms = []
    seen = set()
    for match in matches:
        mapping = {int(src): int(dst) for src, dst in enumerate(match)}
        heavy_perm = []
        ok = True
        for h in heavy:
            dst = mapping.get(int(h))
            if dst is None or dst not in heavy_set:
                ok = False
                break
            heavy_perm.append(int(np.where(heavy == dst)[0][0]))
        if not ok:
            continue
        key = tuple(heavy_perm)
        if key not in seen:
            seen.add(key)
            perms.append(np.asarray(heavy_perm, dtype=int))
    return perms or identity


def _collect_monomer_coords(universe, sels: list[str], frames: list[int]) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """Average heavy-atom coords per monomer and the thermal frame-to-frame RMSD."""
    groups = [universe.select_atoms(sel) for sel in sels]
    heavy = [_heavy_indices(g) for g in groups]
    needed = set(int(f) for f in frames)
    last = max(needed) if needed else 0
    stored: dict[int, list[np.ndarray]] = {}
    for ts in universe.trajectory:
        fr = int(ts.frame)
        if fr in needed:
            stored[fr] = [g.positions[h].copy() for g, h in zip(groups, heavy)]
        if fr >= last and len(stored) == len(needed):
            break
    if not stored:
        raise RuntimeError("No requested frames were read from the trajectory")
    order = [f for f in frames if f in stored]
    stacked = [np.stack([stored[f][i] for f in order], axis=0) for i in range(len(sels))]
    f2f = []
    for t in range(1, len(order)):
        prev = np.concatenate([stacked[i][t - 1] for i in range(len(sels))], axis=0)
        now = np.concatenate([stacked[i][t] for i in range(len(sels))], axis=0)
        rmsd, _ = kabsch_rmsd(now, prev, allow_reflection=False)
        f2f.append(rmsd)
    # iterative mean
    means = [s.mean(axis=0) for s in stacked]
    for _ in range(4):
        aligned = []
        ref = np.concatenate(means, axis=0)
        for t in range(len(order)):
            frame = np.concatenate([stacked[i][t] for i in range(len(sels))], axis=0)
            _rmsd, rot = kabsch_rmsd(frame, ref, allow_reflection=False)
            com = frame.mean(axis=0)
            shifted = (frame - com) @ rot.T + ref.mean(axis=0)
            parts = []
            start = 0
            for i in range(len(sels)):
                n = means[i].shape[0]
                parts.append(shifted[start : start + n])
                start += n
            aligned.append(parts)
        means = [
            np.mean(np.stack([aligned[t][i] for t in range(len(aligned))], axis=0), axis=0)
            for i in range(len(sels))
        ]
    thermal = float(np.median(f2f)) if f2f else float("nan")
    return means, heavy, thermal


def _best_intra_perm(
    src: np.ndarray,
    dst: np.ndarray,
    auts: list[np.ndarray],
    rot: np.ndarray,
    src_com: np.ndarray,
    dst_com: np.ndarray,
) -> np.ndarray:
    best = auts[0]
    best_d = float("inf")
    for aut in auts:
        moved = (src[aut] - src_com) @ rot.T + dst_com
        d = float(np.mean(np.sum((moved - dst) ** 2, axis=1)))
        if d < best_d:
            best_d = d
            best = aut
    return best


def _relabelled_coords(
    means: list[np.ndarray],
    perm: Sequence[int],
    auts_per_monomer: list[list[np.ndarray]],
    *,
    allow_reflection: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dest-ordered coordinates for *perm* and superimpose on the reference."""
    ref_parts = means
    src_parts = [means[i] for i in range(len(means))]
    # initial correspondence: identity intra-monomer aut, dest j gets src perm^{-1}[j]
    inv = [0] * len(perm)
    for i, j in enumerate(perm):
        inv[int(j)] = i
    dest = [src_parts[inv[j]] for j in range(len(perm))]
    ref = np.concatenate(ref_parts, axis=0)
    mobile = np.concatenate(dest, axis=0)
    _rmsd, rot = kabsch_rmsd(mobile, ref, allow_reflection=allow_reflection)
    src_coms = [p.mean(axis=0) for p in src_parts]
    dst_coms = [p.mean(axis=0) for p in ref_parts]
    dest_refined = []
    for j in range(len(perm)):
        i = inv[j]
        aut = _best_intra_perm(src_parts[i], ref_parts[j], auts_per_monomer[i], rot, src_coms[i], dst_coms[j])
        dest_refined.append(src_parts[i][aut])
    mobile = np.concatenate(dest_refined, axis=0)
    rmsd, rot = kabsch_rmsd(mobile, ref, allow_reflection=allow_reflection)
    return rmsd, rot


def _site_centroids(universe, sels: list[str], coords_means: list[np.ndarray]) -> list[np.ndarray]:
    """Site centroids in the averaged heavy-atom frame, one array per monomer."""
    _ = coords_means
    sites = []
    for sel in sels:
        _c, groups = EndpointAnalyzer.find_residue_endpoint_sites(universe, sel)
        cents = []
        ag = universe.select_atoms(sel)
        id_to_pos = {int(a.id): a.position.copy() for a in ag}
        for site in groups:
            pts = np.array([id_to_pos[int(i)] for i in site if int(i) in id_to_pos])
            if pts.size:
                cents.append(pts.mean(axis=0))
        sites.append(np.asarray(cents, dtype=float) if cents else np.zeros((0, 3)))
    return sites


def _hungarian_site_map(
    src_sites: list[np.ndarray],
    perm: Sequence[int],
    rot: np.ndarray,
) -> list[list[int]]:
    from scipy.optimize import linear_sum_assignment

    inv = [0] * len(perm)
    for i, j in enumerate(perm):
        inv[int(j)] = i
    maps = []
    for j in range(len(perm)):
        i = inv[j]
        a = src_sites[i]
        b = src_sites[j]
        if a.size == 0 or b.size == 0 or a.shape[0] != b.shape[0]:
            maps.append(list(range(max(a.shape[0], b.shape[0], 1))))
            continue
        ac = (a - a.mean(axis=0)) @ rot.T + b.mean(axis=0)
        d2 = ((ac[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        _r, c = linear_sum_assignment(d2)
        maps.append([int(c[k]) for k in range(len(c))])
    return maps


def run_cube(
    cube: str,
    topology: Path,
    trajectory: Path,
    *,
    output_root: Path,
    n_ref_frames: int,
    k: int | None,
    traj_format: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import MDAnalysis as mda

    kwargs = {}
    if traj_format and trajectory.suffix == "":
        kwargs["format"] = traj_format
    elif trajectory.suffix.lower() in {".trj", ".crd"}:
        kwargs["format"] = traj_format or "TRJ"
    u = mda.Universe(str(topology), str(trajectory), **kwargs)
    sels = resolve_selections(u, gsa_resname="MOL", n_monomers=N_MONOMERS).monomer_selections
    if not sels or len(sels) != N_MONOMERS:
        raise RuntimeError(f"{cube}: expected {N_MONOMERS} monomers, got {sels}")
    stem = trajectory.stem.replace("_mdcrd_v", "")
    if "_" in stem:
        # BHHpM_109345_mdcrd_v → 109345
        parts = trajectory.name.split("_")
        digits = [p for p in parts if p.isdigit()]
        stem = digits[0] if digits else stem
    frames = _compact_frames(output_root, cube, stem, n_ref_frames, k=k)
    print(f"{cube}: averaging {len(frames)} compact frames from {trajectory.name}", flush=True)
    means, _heavy, thermal = _collect_monomer_coords(u, sels, frames)
    auts = [_graph_automorphisms(u.select_atoms(sel)) for sel in sels]
    u.trajectory[int(frames[0])]
    site_cents = _site_centroids(u, sels, means)

    contact = {perm_cycles(p): p for p in contact_automorphisms()}
    rows = []
    site_rows = []
    for perm in adjacency_automorphisms():
        cycles = perm_cycles(perm)
        proper = is_proper_permutation(perm)
        is_contact = cycles in contact or tuple(perm) in contact.values()
        rmsd_rot, rot_p = _relabelled_coords(means, perm, auts, allow_reflection=False)
        rmsd_ref, rot_i = _relabelled_coords(means, perm, auts, allow_reflection=True)
        # achiral if the identity (or this op) matches its mirror
        self_mirror = rmsd_ref if not proper else rmsd_rot
        admissible = bool(
            is_contact
            and np.isfinite(thermal)
            and (
                (proper and rmsd_rot <= thermal * 1.5 + 0.15)
                or ((not proper) and rmsd_ref <= thermal * 1.5 + 0.15)
            )
        )
        if not proper:
            # improper ops only if the cage is achiral: identity-vs-mirror already
            # encoded by a small reflection RMSD on this op
            admissible = admissible and rmsd_ref <= thermal * 1.5 + 0.15
        rows.append(
            {
                "cube": cube,
                "perm": " ".join(str(x) for x in perm),
                "cycles": cycles,
                "proper": bool(proper),
                "contact_automorphism": bool(is_contact),
                "self_rmsd_rotation": float(rmsd_rot),
                "self_rmsd_reflection": float(rmsd_ref),
                "thermal_rmsd": float(thermal),
                "admissible": bool(admissible),
                "inferred": False,
                "source_cube": cube,
            }
        )
        if is_contact:
            rot = rot_p if proper else rot_i
            smap = _hungarian_site_map(site_cents, perm, rot)
            for dest, mapping in enumerate(smap):
                site_rows.append(
                    {
                        "cube": cube,
                        "cycles": cycles,
                        "perm": " ".join(str(x) for x in perm),
                        "dest_monomer": dest,
                        "site_map": " ".join(str(x) for x in mapping),
                        "inferred": False,
                        "source_cube": cube,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(site_rows)


def infer_from(source_ops: pd.DataFrame, source_sites: pd.DataFrame, dest: str, src: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ops = source_ops[source_ops["cube"] == src].copy()
    sites = source_sites[source_sites["cube"] == src].copy()
    ops["cube"] = dest
    ops["inferred"] = True
    ops["source_cube"] = src
    sites["cube"] = dest
    sites["inferred"] = True
    sites["source_cube"] = src
    return ops, sites


def _write(ops: pd.DataFrame, sites: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    op_path = out_dir / "symmetry_operations.csv"
    site_path = out_dir / "site_maps.csv"
    if op_path.is_file():
        prev = pd.read_csv(op_path)
        prev = prev[~prev["cube"].isin(ops["cube"].unique())]
        ops = pd.concat([prev, ops], ignore_index=True)
    if site_path.is_file():
        prev = pd.read_csv(site_path)
        prev = prev[~prev["cube"].isin(sites["cube"].unique())]
        sites = pd.concat([prev, sites], ignore_index=True)
    ops.to_csv(op_path, index=False)
    sites.to_csv(site_path, index=False)
    print(f"Wrote {op_path} and {site_path}", flush=True)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or args.output_root / "endpoint_cluster_validation_physical_k" / "symmetry"
    if args.cube in POLE_H_FROM_POLE_M and (args.topology is None or args.trajectory is None):
        src = POLE_H_FROM_POLE_M[args.cube]
        op_path = out_dir / "symmetry_operations.csv"
        if not op_path.is_file():
            print(f"{args.cube}: run D1 on {src} first, or pass --topology and --trajectory", file=sys.stderr)
            return 1
        ops = pd.read_csv(op_path)
        sites = pd.read_csv(out_dir / "site_maps.csv") if (out_dir / "site_maps.csv").is_file() else pd.DataFrame()
        if src not in set(ops["cube"].astype(str)):
            print(f"{args.cube}: {src} is not in {op_path}", file=sys.stderr)
            return 1
        inferred_ops, inferred_sites = infer_from(ops, sites, args.cube, src)
        _write(inferred_ops, inferred_sites, out_dir)
        return 0

    topo = args.topology
    traj = args.trajectory
    if topo is None or traj is None:
        if args.cube not in DEFAULT_LOCAL_TRAJ:
            print(f"{args.cube}: pass --topology and --trajectory", file=sys.stderr)
            return 1
        topo, traj = (ROOT / p for p in DEFAULT_LOCAL_TRAJ[args.cube])
    if not Path(topo).is_file() or not Path(traj).is_file():
        print(f"Missing topology or trajectory: {topo} {traj}", file=sys.stderr)
        return 1
    k = _selected_k(args.k_selected, args.cube, args.k)
    ops, sites = run_cube(
        args.cube,
        Path(topo),
        Path(traj),
        output_root=args.output_root,
        n_ref_frames=args.n_ref_frames,
        k=k,
        traj_format=args.trajectory_format,
    )
    _write(ops, sites, out_dir)
    n_adm = int(ops["admissible"].sum())
    n_contact = int(ops["contact_automorphism"].sum())
    print(
        f"{args.cube}: {len(ops)} adjacency ops, {n_contact} contact, "
        f"{n_adm} admissible, thermal RMSD {ops['thermal_rmsd'].iloc[0]:.3f} Å",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
