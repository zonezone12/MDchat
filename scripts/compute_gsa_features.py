"""
Extract per-frame GSA nanocube features from one or more trajectories.

Example
-------
python scripts/compute_gsa_features.py \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectories traj/BMMpM_891249_mdcrd_v.trj \\
    --gsa-resname MOL \\
    --guest-selection "resname IOD" \\
    --output-dir output/gsa_features \\
    --stride 10
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import MDAnalysis as mda

from src.utils.gsa_feature_observer import compute_gsa_features
from src.utils.gsa_selections import GSAFeatureSelections


def _expand_trajectories(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            paths.extend(Path(m) for m in matches)
        else:
            p = Path(pat)
            if p.is_file():
                paths.append(p)
            elif p.is_dir():
                for ext in ("*.xtc", "*.trj", "*.dcd", "*.nc"):
                    paths.extend(sorted(p.glob(ext)))
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute GSA nanocube features over MD trajectories."
    )
    parser.add_argument(
        "--topology",
        required=True,
        help="Path to shared topology (prmtop, pdb, etc.)",
    )
    parser.add_argument(
        "--trajectories",
        nargs="+",
        required=True,
        help="Trajectory paths or glob patterns",
    )
    parser.add_argument(
        "--output-dir",
        default="output/gsa_features",
        help="Directory for per-trajectory feature CSVs",
    )
    parser.add_argument(
        "--gsa-resname",
        default="MOL",
        help="Residue name of GSA amphiphile monomers (default: MOL)",
    )
    parser.add_argument(
        "--n-monomers",
        type=int,
        default=6,
        help="Expected number of monomers in the nanocube",
    )
    parser.add_argument(
        "--guest-selection",
        default=None,
        help="MDAnalysis selection for guest (e.g. 'resname IOD')",
    )
    parser.add_argument(
        "--assembly-selection",
        default=None,
        help="Override assembly selection (default: resname GSA_RESNAME)",
    )
    parser.add_argument("--stride", type=int, default=1, help="Frame stride")
    parser.add_argument("--start", type=int, default=None, help="Start frame index")
    parser.add_argument("--stop", type=int, default=None, help="Stop frame index (exclusive)")
    parser.add_argument("--ref-frame", type=int, default=0, help="RMSD reference frame")
    parser.add_argument(
        "--no-tier2",
        action="store_true",
        help="Skip Tier-2 gear/interface chemistry features",
    )
    parser.add_argument(
        "--no-guest",
        action="store_true",
        help="Skip guest-related features",
    )
    parser.add_argument(
        "--contact-cutoff",
        type=float,
        default=4.5,
        help="Inter-monomer contact cutoff (Angstrom)",
    )
    parser.add_argument(
        "--cavity-radius",
        type=float,
        default=8.0,
        help="Cavity radius for solvent counts (Angstrom)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs (1=sequential; >1 uses TrajectoryIterator parallel batches)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_paths = _expand_trajectories(args.trajectories)
    if not traj_paths:
        print("No trajectory files matched.", file=sys.stderr)
        sys.exit(1)

    selections = GSAFeatureSelections(
        assembly_sel=args.assembly_selection,
        guest_sel=args.guest_selection,
    )

    print(f"Topology: {args.topology}")
    print(f"Trajectories ({len(traj_paths)}):")
    for p in traj_paths:
        print(f"  {p}")
    print(f"GSA resname: {args.gsa_resname}, n_monomers: {args.n_monomers}")
    if args.guest_selection:
        print(f"Guest: {args.guest_selection}")
    print(f"Output: {out_dir}")
    print()

    top = str(args.topology)
    for traj_path in traj_paths:
        traj_id = traj_path.stem
        print(f"Processing {traj_id} ...")
        u = mda.Universe(top, str(traj_path))

        out_prefix = str(out_dir / traj_id)
        df = compute_gsa_features(
            u,
            selections=selections,
            gsa_resname=args.gsa_resname,
            n_monomers=args.n_monomers,
            include_tier1=True,
            include_tier2=not args.no_tier2,
            include_guest=not args.no_guest,
            ref_frame=args.ref_frame,
            contact_cutoff=args.contact_cutoff,
            cavity_radius=args.cavity_radius,
            traj_id=traj_id,
            out_prefix=out_prefix,
            start=args.start,
            stop=args.stop,
            step=args.stride,
            n_jobs=args.n_jobs,
        )

        csv_path = out_dir / f"{traj_id}_gsa_features.csv"
        print(f"  Wrote {csv_path} ({len(df)} frames, {len(df.columns)} columns)")

    print("Done.")


if __name__ == "__main__":
    main()
