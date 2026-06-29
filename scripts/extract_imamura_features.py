"""
Extract per-frame Imamura inter-bead distance features → imamura_features.csv.

Uses :func:`src.utils.imamura_msm.compute_imamura_features` (via
``extract_imamura_features_multi``) to build sorted type-1 / type-4 distance
vectors for clustering or sweep scripts.

Example
-------
python scripts/extract_imamura_features.py \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectories traj/BMMpM_891249_mdcrd_v.trj \\
    --auto-beads --gsa-resname MOL \\
    --format TRJ \\
    --output-dir output/imamura_msm/BMMpM

Endpoint type-4 beads (no methyl filter):
python scripts/extract_imamura_features.py \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectories traj/BMMpM_891249_mdcrd_v.trj \\
    --auto-beads --type4-source endpoint \\
    --format TRJ \\
    --output-dir output/imamura_msm/endpoint_BMMpM

Reuse a saved bead spec:
python scripts/extract_imamura_features.py \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectories traj/BMMpM_891249_mdcrd_v.trj \\
    --bead-spec output/imamura_msm/endpoint_BMMpM/bead_spec.json \\
    --format TRJ \\
    --output-dir output/imamura_msm/endpoint_BMMpM
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.imamura_msm import (
    DEFAULT_N_TYPE1_BEADS,
    DEFAULT_N_TYPE4_BEADS,
    ImamuraBeadResolutionOptions,
    ImamuraBeadSpec,
    detect_formation_frames,
    extract_imamura_features_multi,
    load_imamura_bead_spec,
    plot_imamura_bead_spec,
    resolve_imamura_bead_spec,
    write_imamura_bead_spec,
)
from src.utils.gsa_selections import resolve_selections
from src.utils.run_log import RunContext, log_event, step


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


def _load_formation_frames(path: Path) -> dict[str, int]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): int(v) for k, v in data.items()}
    df = pd.read_csv(path)
    if "traj_id" not in df.columns or "formation_frame" not in df.columns:
        raise ValueError(
            f"{path}: CSV must have columns traj_id, formation_frame"
        )
    return {
        str(row["traj_id"]): int(row["formation_frame"])
        for _, row in df.iterrows()
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Imamura bead-distance features to imamura_features.csv.",
    )
    p.add_argument("--topology", required=True, help="Shared topology file")
    p.add_argument(
        "--trajectories",
        nargs="+",
        required=True,
        help="Trajectory paths or glob patterns",
    )
    p.add_argument(
        "--output-dir",
        default="output/imamura_msm",
        help="Directory for imamura_features.csv and bead_spec.json",
    )
    p.add_argument(
        "--format",
        default=None,
        help="Trajectory format (e.g. TRJ, XTC, DCD)",
    )
    p.add_argument("--stride", type=int, default=1, help="Frame stride")
    p.add_argument(
        "--n-jobs",
        type=int,
        default=8,
        dest="n_jobs",
        help="Parallel TrajectoryIterator workers per trajectory",
    )

    bead = p.add_argument_group("bead selections")
    bead.add_argument(
        "--type1-selection",
        default=None,
        help="MDAnalysis selection for type-1 (central benzene) beads",
    )
    bead.add_argument(
        "--type4-selection",
        default=None,
        help="MDAnalysis selection for type-4 (methyl/endpoint) beads",
    )
    bead.add_argument(
        "--auto-beads",
        action="store_true",
        help="Derive beads from RDKit substructure + EndpointAnalyzer",
    )
    bead.add_argument(
        "--bead-spec",
        default=None,
        help="Load bead_spec.json instead of re-resolving",
    )
    bead.add_argument(
        "--type4-source",
        choices=["methyl", "endpoint"],
        default="methyl",
        help="Type-4 source when --auto-beads (default: methyl)",
    )
    bead.add_argument(
        "--type4-rank",
        choices=["bond", "centroid3d"],
        default="bond",
        help="Rank type-4 candidates per monomer (default: bond)",
    )
    bead.add_argument(
        "--endpoint-extend-ring",
        dest="endpoint_extend_ring",
        action="store_true",
        default=True,
        help="Extend EndpointsFinder hits to full substituent rings (default)",
    )
    bead.add_argument(
        "--no-endpoint-extend-ring",
        dest="endpoint_extend_ring",
        action="store_false",
        help="Use hull endpoints only",
    )
    bead.add_argument(
        "--endpoint-exclude-center-benzene",
        dest="endpoint_exclude_center_benzene",
        action="store_true",
        default=True,
        help="Drop central benzene ring atoms from endpoint type-4 (default)",
    )
    bead.add_argument(
        "--no-endpoint-exclude-center-benzene",
        dest="endpoint_exclude_center_benzene",
        action="store_false",
        help="Keep central benzene ring atoms in endpoint type-4",
    )
    bead.add_argument(
        "--bead-mode",
        choices=["rdkit", "cg"],
        default="rdkit",
        help="Auto-bead strategy: rdkit (default) or coarse-grained bead names",
    )
    bead.add_argument("--gsa-resname", default="MOL", help="GSA residue name for --auto-beads")
    bead.add_argument("--n-monomers", type=int, default=6)
    bead.add_argument(
        "--type4-per-monomer",
        type=int,
        default=None,
        dest="type4_per_monomer",
        help="Cap type-4 beads per monomer for methyl source (default 3)",
    )
    bead.add_argument(
        "--type1-atom-name",
        default="B1",
        help="CG bead name for type-1 when --bead-mode cg",
    )
    bead.add_argument(
        "--type4-atom-name",
        default="B4",
        help="CG bead name for type-4 when --bead-mode cg",
    )
    bead.add_argument("--n-type1-beads", type=int, default=DEFAULT_N_TYPE1_BEADS)
    bead.add_argument("--n-type4-beads", type=int, default=DEFAULT_N_TYPE4_BEADS)

    trunc = p.add_argument_group("formation / truncation")
    trunc.add_argument(
        "--truncate-at-formation",
        action="store_true",
        help="Start each trajectory at detected nanocube formation frame",
    )
    trunc.add_argument(
        "--formation-frames",
        default=None,
        help="CSV/JSON with traj_id, formation_frame (skip auto-detection)",
    )
    trunc.add_argument(
        "--detect-formation-only",
        action="store_true",
        help="Only detect/write formation frames; skip feature extraction",
    )
    trunc.add_argument(
        "--formation-min-interfaces",
        type=int,
        default=15,
        help="Min active monomer interfaces for formation (default 15)",
    )
    trunc.add_argument(
        "--formation-sustain-frames",
        type=int,
        default=1,
        help="Consecutive frames required at formation criterion",
    )
    trunc.add_argument("--contact-cutoff", type=float, default=4.5)
    return p.parse_args()


def _bead_resolution_options(args: argparse.Namespace) -> ImamuraBeadResolutionOptions:
    return ImamuraBeadResolutionOptions(
        type4_source=args.type4_source,
        type4_rank=args.type4_rank,
        endpoint_extend_ring=args.endpoint_extend_ring,
        endpoint_exclude_center_benzene=args.endpoint_exclude_center_benzene,
        type4_per_monomer=args.type4_per_monomer,
    )


def _sync_resolved_type4_bead_count(
    n_type4: int,
    bead_spec: ImamuraBeadSpec,
) -> int:
    if not bead_spec.uses_ring_centroids:
        return n_type4
    resolved = len(bead_spec.type4_atom_ids or ())
    if resolved and resolved != n_type4:
        print(f"  n_type4_beads: {n_type4} -> {resolved}")
        return resolved
    return n_type4


def _write_bead_spec_artifacts(
    u0,
    monomer_sels: list[str],
    bead_spec: ImamuraBeadSpec,
    out_dir: Path,
) -> Path:
    bead_path = write_imamura_bead_spec(bead_spec, out_dir / "bead_spec.json")
    print(f"  Wrote {bead_path}")
    bead_plot = plot_imamura_bead_spec(
        u0,
        monomer_sels or [],
        bead_spec,
        out_dir / "auto_beads.png",
    )
    print(f"  Wrote {bead_plot}")
    if bead_spec.uses_ring_centroids:
        type4_desc = (
            "methyl atoms"
            if bead_spec.type4_source == "methyl"
            else "endpoint atoms"
        )
        print(
            f"  type-1: {len(bead_spec.type1_ring_groups or [])} benzene ring centroids"
        )
        print(
            f"  type-4 ({bead_spec.type4_source}, rank={bead_spec.type4_rank}): "
            f"{len(bead_spec.type4_atom_ids or ())} {type4_desc}"
        )
    else:
        print(f"  type-1: {bead_spec.type1_selection}")
        print(f"  type-4: {bead_spec.type4_selection}")
    return bead_path


def _resolve_bead_spec(
    args: argparse.Namespace,
    u0,
    monomer_sels: list[str],
    out_dir: Path,
) -> tuple[ImamuraBeadSpec, int, int]:
    n_type1 = args.n_type1_beads
    n_type4 = args.n_type4_beads
    bead_resolution = _bead_resolution_options(args)

    if args.bead_spec:
        with step("load_bead_spec"):
            bead_spec = load_imamura_bead_spec(Path(args.bead_spec))
            print(f"  Loaded {args.bead_spec}")
            n_type4 = _sync_resolved_type4_bead_count(n_type4, bead_spec)
            bead_spec.validate(n_type1=n_type1, n_type4=n_type4)
            _write_bead_spec_artifacts(u0, monomer_sels, bead_spec, out_dir)
        return bead_spec, n_type1, n_type4

    if args.auto_beads or not (args.type1_selection and args.type4_selection):
        with step("resolve_bead_selections"):
            bead_spec = resolve_imamura_bead_spec(
                u0,
                explicit_type1=args.type1_selection,
                explicit_type4=args.type4_selection,
                gsa_resname=args.gsa_resname,
                n_monomers=args.n_monomers,
                bead_mode=args.bead_mode,
                type1_atom_name=args.type1_atom_name,
                type4_atom_name=args.type4_atom_name,
                resolution=bead_resolution,
            )
            n_type4 = _sync_resolved_type4_bead_count(n_type4, bead_spec)
            bead_spec.validate(n_type1=n_type1, n_type4=n_type4)
            _write_bead_spec_artifacts(u0, monomer_sels, bead_spec, out_dir)
        return bead_spec, n_type1, n_type4

    bead_spec = resolve_imamura_bead_spec(
        u0,
        explicit_type1=args.type1_selection,
        explicit_type4=args.type4_selection,
    )
    bead_spec.validate(n_type1=n_type1, n_type4=n_type4)
    _write_bead_spec_artifacts(u0, monomer_sels, bead_spec, out_dir)
    return bead_spec, n_type1, n_type4


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_paths = _expand_trajectories(args.trajectories)
    if not traj_paths:
        print("No trajectory files matched.", file=sys.stderr)
        sys.exit(1)

    formation_frames: dict[str, int] = {}
    if args.formation_frames:
        formation_frames = _load_formation_frames(Path(args.formation_frames))

    with RunContext.from_namespace(args, name="extract_imamura_features") as run:
        print(f"Run log: {run.run_dir}")
        log_event("start", f"{len(traj_paths)} trajectories → {out_dir}")

        import MDAnalysis as mda

        u0 = mda.Universe(
            args.topology,
            str(traj_paths[0]),
            **({"format": args.format} if args.format else {}),
        )

        resolved = resolve_selections(
            u0,
            gsa_resname=args.gsa_resname,
            n_monomers=args.n_monomers,
            auto_tooth=False,
        )
        monomer_sels = resolved.monomer_selections or []

        bead_spec, n_type1, n_type4 = _resolve_bead_spec(
            args, u0, monomer_sels, out_dir
        )

        if args.truncate_at_formation and not formation_frames:
            with step("detect_formation_frames"):
                formation_frames = detect_formation_frames(
                    args.topology,
                    traj_paths,
                    monomer_sels,
                    n_monomers=args.n_monomers,
                    traj_format=args.format,
                    contact_cutoff=args.contact_cutoff,
                    min_active_interfaces=args.formation_min_interfaces,
                    sustain_frames=args.formation_sustain_frames,
                    stride=args.stride,
                )
                ff_path = out_dir / "formation_frames.csv"
                pd.DataFrame([
                    {"traj_id": k, "formation_frame": v}
                    for k, v in sorted(formation_frames.items())
                ]).to_csv(ff_path, index=False)
                print(f"  Wrote {ff_path}")

        if args.detect_formation_only:
            print("Formation detection complete (--detect-formation-only).")
            return

        with step("extract_imamura_features"):
            feature_result = extract_imamura_features_multi(
                args.topology,
                traj_paths,
                bead_spec,
                traj_format=args.format,
                formation_frames=formation_frames if args.truncate_at_formation else None,
                truncate_at_formation=args.truncate_at_formation,
                stride=args.stride,
                n_type1=n_type1,
                n_type4=n_type4,
                n_jobs=args.n_jobs,
            )
            feature_result.formation_frames = formation_frames

        feat_path = out_dir / "imamura_features.csv"
        feature_result.dataframe.to_csv(feat_path, index=False)

        n_frames = len(feature_result.dataframe)
        n_feat = len(feature_result.feature_names)
        print(f"  Wrote {feat_path}")
        print(f"  {n_frames} frames × {n_feat} features")
        log_event("done", f"imamura_features.csv → {feat_path}")


if __name__ == "__main__":
    main()
