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
from src.utils.run_log import RunContext, log_event, step


_GENERIC_TRAJ_STEMS = frozenset({
    "mdcrd", "mdcrd_v", "crd", "dcd", "xtc", "trj", "nc", "prod", "equil",
})


def _trajectory_output_id(traj_path: Path) -> str:
    """Stable CSV basename; disambiguate generic Amber names like ``mdcrd_v``."""
    stem = traj_path.stem
    parent = traj_path.parent.name
    if stem.lower() in _GENERIC_TRAJ_STEMS and parent not in ("", ".", ".."):
        return f"{parent}_{stem}"
    return stem


def _infer_cube_name(topology: str) -> str | None:
    """Guess cube/system name from topology file (e.g. ``BHHpH_ca.prmtop`` → ``BHHpH``)."""
    stem = Path(topology).stem
    for suffix in ("_ca", "_box", "_wat", "_ion", "_solv"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)] or None
    return stem or None


def _resolve_cube_name(topology: str, cube: str | None) -> str | None:
    if cube:
        name = cube.strip()
        return name or None
    return _infer_cube_name(topology)


def _unique_trajectory_ids(
    traj_paths: list[Path],
    cube: str | None = None,
) -> list[str]:
    """Assign output IDs; suffix duplicates when two paths still collide."""
    ids: list[str] = []
    counts: dict[str, int] = {}
    for traj_path in traj_paths:
        base = _trajectory_output_id(traj_path)
        if cube and not base.startswith(f"{cube}_"):
            base = f"{cube}_{base}"
        n = counts.get(base, 0)
        counts[base] = n + 1
        ids.append(base if n == 0 else f"{base}_{n}")
    return ids


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
        "--cube",
        default=None,
        help="Cube/system name prefix for output CSVs (e.g. BHHpH). "
        "Default: inferred from topology (BHHpH_ca.prmtop → BHHpH).",
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
        default='resname IOD',
        help="MDAnalysis selection for guest (e.g. 'resname IOD')",
    )
    parser.add_argument(
        "--assembly-selection",
        default=None,
        help="Override assembly selection (default: resname GSA_RESNAME)",
    )
    parser.add_argument(
        "--format",
        default='TRJ',
        help="Trajectory format (e.g. 'TRJ', 'DCD', 'XTC')",
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
        "--no-auto-tooth",
        action="store_true",
        help="Disable EndpointAnalyzer auto tooth detection for Tier-2 gear features",
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
        "--n_jobs",
        type=int,
        default=1,
        help="Parallel jobs (1=sequential; >1 uses TrajectoryIterator parallel batches)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    traj_paths = _expand_trajectories(args.trajectories)
    if not traj_paths:
        print("No trajectory files matched.", file=sys.stderr)
        sys.exit(1)

    selections = GSAFeatureSelections(
        assembly_sel=args.assembly_selection,
        guest_sel=args.guest_selection,
    )

    cube_name = _resolve_cube_name(args.topology, args.cube)

    with RunContext.from_namespace(args, name="compute_gsa_features") as run:
        print(f"Run log: {run.run_dir}")
        if cube_name:
            print(f"Cube prefix: {cube_name}")
        log_event(
            "config",
            "compute_gsa_features configuration",
            component="compute_gsa_features",
            context={
                "topology": args.topology,
                "cube": cube_name,
                "n_trajectories": len(traj_paths),
                "output_dir": str(out_dir),
                "gsa_resname": args.gsa_resname,
                "n_monomers": args.n_monomers,
                "guest_selection": args.guest_selection,
            },
        )

        top = str(args.topology)
        traj_ids = _unique_trajectory_ids(traj_paths, cube=cube_name)
        for traj_path, traj_id in zip(traj_paths, traj_ids):
            with step(
                f"trajectory:{traj_id}",
                component="compute_gsa_features",
                context={"traj_path": str(traj_path), "traj_id": traj_id},
            ):
                u = mda.Universe(top, str(traj_path),format=args.format)
                out_prefix = str(out_dir / traj_id)
                df = compute_gsa_features(
                    u,
                    selections=selections,
                    gsa_resname=args.gsa_resname,
                    n_monomers=args.n_monomers,
                    include_tier1=True,
                    include_tier2=not args.no_tier2,
                    include_guest=not args.no_guest,
                    auto_tooth=not args.no_auto_tooth,
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
                log_event(
                    "artifact_written",
                    str(csv_path),
                    component="compute_gsa_features",
                    context={
                        "traj_id": traj_id,
                        "n_frames": len(df),
                        "n_columns": len(df.columns),
                        "csv_path": str(csv_path),
                    },
                )
                print(f"  Wrote {csv_path} ({len(df)} frames, {len(df.columns)} columns)")

    print("Done.")


if __name__ == "__main__":
    main()


#python scripts/compute_gsa_features.py --topology traj/BMMpM_ca.prmtop --trajectories traj/BMMpM_891249_mdcrd_v.trj --gsa-resname MOL --guest-selection "resname IOD" --output-dir output/gsa_features --stride 10