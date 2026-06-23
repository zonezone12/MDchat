"""
Imamura et al. nanocube MSM workflow (Chem. Phys. Lett. 742, 137135, 2020).

Pipeline
--------
1. Truncate each trajectory at nanocube formation (optional).
2. Build 168-D sorted inter-bead distance vectors (type-1 + type-4 beads).
3. PCA → 5 components.
4. MiniBatchKMeans (1500 microstates) + Ward merge (22 macrostates).
5. Lagged (2 ns) transition network between macrostates.

Example
-------
python scripts/run_imamura_msm.py \\
    --topology traj/nanocube.top \\
    --trajectories "traj/run_*/prod.xtc" \\
    --type1-selection "name B1 and resname MOL" \\
    --type4-selection "name B4 and resname MOL" \\
    --output-dir output/imamura_msm

Auto-resolve beads (RDKit: central benzene + methyl endpoints):
python scripts/run_imamura_msm.py \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectories traj/BMMpM_891249_mdcrd_v.trj \\
    --auto-beads --gsa-resname MOL \\
    --format TRJ \\
    --output-dir output/imamura_msm

Use raw endpoint atoms instead of methyl filtering:
python scripts/run_imamura_msm.py ... --auto-beads --type4-source endpoint

Exclude benzene ring expansion when finding endpoints:
python scripts/run_imamura_msm.py ... --auto-beads --no-endpoint-extend-ring
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
    DEFAULT_N_MACROSTATES,
    DEFAULT_N_MICROCLUSTERS,
    DEFAULT_N_TYPE1_BEADS,
    DEFAULT_N_TYPE4_BEADS,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_TRANSITION_LAG_NS,
    ImamuraBeadResolutionOptions,
    ImamuraBeadSpec,
    ImamuraMSMConfig,
    cluster_imamura_features,
    count_lagged_transitions,
    detect_formation_frames,
    extract_imamura_features_multi,
    lag_frames_from_ns,
    resolve_imamura_bead_spec,
    plot_imamura_bead_spec,
    write_imamura_artifacts,
    write_imamura_bead_spec,
    _infer_dt_ps,
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
        description="Imamura nanocube MSM: bead distances → PCA → clustering → transitions.",
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
        help="Directory for CSVs and plots",
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
        help="MDAnalysis selection for 6 type-1 (central benzene) beads",
    )
    bead.add_argument(
        "--type4-selection",
        default=None,
        help="MDAnalysis selection for 18 type-4 (methyl) beads",
    )
    bead.add_argument(
        "--auto-beads",
        action="store_true",
        help="Derive type-1 (central benzene centroid) and type-4 beads from "
        "RDKit substructure + EndpointAnalyzer",
    )
    bead.add_argument(
        "--type4-source",
        choices=["methyl", "endpoint"],
        default="methyl",
        help="Type-4 bead source when --auto-beads: methyl (CH3 filter) "
        "or endpoint (raw EndpointsFinder output)",
    )
    bead.add_argument(
        "--type4-rank",
        choices=["bond", "centroid3d"],
        default="bond",
        help="How to pick type4_per_monomer beads from candidates: bond (graph "
        "distance from central benzene; default) or centroid3d (3D distance "
        "from central ring centroid, farthest kept)",
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
        help="Use hull endpoints only (do not expand to ring atoms)",
    )
    bead.add_argument(
        "--endpoint-exclude-center-benzene",
        dest="endpoint_exclude_center_benzene",
        action="store_true",
        default=True,
        help="When --type4-source endpoint, drop central benzene ring atoms "
        "already used for type-1 (default)",
    )
    bead.add_argument(
        "--no-endpoint-exclude-center-benzene",
        dest="endpoint_exclude_center_benzene",
        action="store_false",
        help="Keep central benzene ring atoms in endpoint type-4 selection",
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
        help="Cap type-4 beads per monomer for methyl source (default 3). "
        "For endpoint source all filtered endpoints are kept unless this is set.",
    )
    bead.add_argument(
        "--methyl-per-monomer",
        type=int,
        default=None,
        dest="type4_per_monomer",
        help=argparse.SUPPRESS,
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
        help="Truncate each trajectory at detected nanocube formation (off by "
        "default; use for assembly trajectories)",
    )
    trunc.add_argument(
        "--no-truncate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    trunc.add_argument(
        "--formation-frames",
        default=None,
        help="CSV/JSON with traj_id, formation_frame (skip auto-detection)",
    )
    trunc.add_argument(
        "--detect-formation-only",
        action="store_true",
        help="Only detect/write formation frames; do not run MSM",
    )
    trunc.add_argument(
        "--formation-min-interfaces",
        type=int,
        default=15,
        help="Min active monomer–monomer interfaces for formation (default 15/15)",
    )
    trunc.add_argument(
        "--formation-sustain-frames",
        type=int,
        default=1,
        help="Consecutive frames required at formation criterion",
    )
    trunc.add_argument("--contact-cutoff", type=float, default=4.5)

    msm = p.add_argument_group("PCA / clustering / transitions")
    msm.add_argument("--n-pca", type=int, default=DEFAULT_PCA_COMPONENTS)
    msm.add_argument("--n-micro", type=int, default=DEFAULT_N_MICROCLUSTERS)
    msm.add_argument("--n-macro", type=int, default=DEFAULT_N_MACROSTATES)
    msm.add_argument(
        "--transition-lag-ns",
        type=float,
        default=DEFAULT_TRANSITION_LAG_NS,
        help="Lag time for transition counting (default 2 ns)",
    )
    msm.add_argument(
        "--dt-ps",
        type=float,
        default=1,
        help="Override trajectory spacing in ps (auto-detected if omitted)",
    )
    msm.add_argument(
        "--features-csv",
        default=None,
        help="Skip extraction; cluster/transitions from existing imamura_features.csv",
    )
    return p.parse_args()


def _build_config(args: argparse.Namespace) -> ImamuraMSMConfig:
    return ImamuraMSMConfig(
        n_type1_beads=args.n_type1_beads,
        n_type4_beads=args.n_type4_beads,
        n_pca_components=args.n_pca,
        n_microclusters=args.n_micro,
        n_macrostates=args.n_macro,
        transition_lag_ns=args.transition_lag_ns,
        formation_min_active_interfaces=args.formation_min_interfaces,
        formation_sustain_frames=args.formation_sustain_frames,
        contact_cutoff=args.contact_cutoff,
    )


def _bead_resolution_options(args: argparse.Namespace) -> ImamuraBeadResolutionOptions:
    return ImamuraBeadResolutionOptions(
        type4_source=args.type4_source,
        type4_rank=args.type4_rank,
        endpoint_extend_ring=args.endpoint_extend_ring,
        endpoint_exclude_center_benzene=args.endpoint_exclude_center_benzene,
        type4_per_monomer=args.type4_per_monomer,
    )


def _sync_resolved_type4_bead_count(
    config: ImamuraMSMConfig,
    bead_spec: ImamuraBeadSpec,
) -> None:
    """Match feature dimensions to auto-resolved type-4 bead count."""
    if not bead_spec.uses_ring_centroids:
        return
    resolved = len(bead_spec.type4_atom_ids or ())
    if resolved and resolved != config.n_type4_beads:
        print(f"  n_type4_beads: {config.n_type4_beads} -> {resolved}")
        config.n_type4_beads = resolved


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_paths = _expand_trajectories(args.trajectories)
    if not traj_paths:
        print("No trajectory files matched.", file=sys.stderr)
        sys.exit(1)

    config = _build_config(args)
    truncate = args.truncate_at_formation and not args.no_truncate
    formation_frames: dict[str, int] = {}
    if args.formation_frames:
        formation_frames = _load_formation_frames(Path(args.formation_frames))

    with RunContext.from_namespace(args, name="run_imamura_msm") as run:
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
        monomer_sels = resolved.monomer_selections
        bead_resolution = _bead_resolution_options(args)

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
                _sync_resolved_type4_bead_count(config, bead_spec)
                bead_spec.validate(
                    n_type1=config.n_type1_beads,
                    n_type4=config.n_type4_beads,
                )
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
                        f"  type-1: {len(bead_spec.type1_ring_groups or [])} benzene "
                        "ring centroids"
                    )
                    print(
                        f"  type-4 ({bead_spec.type4_source}, rank={bead_spec.type4_rank}): "
                        f"{len(bead_spec.type4_atom_ids or ())} {type4_desc}"
                    )
                    print(
                        f"  endpoint_extend_ring={bead_spec.endpoint_extend_ring}, "
                        f"exclude_center_benzene="
                        f"{bead_spec.endpoint_exclude_center_benzene}"
                    )
                else:
                    print(f"  type-1: {bead_spec.type1_selection}")
                    print(f"  type-4: {bead_spec.type4_selection}")
        else:
            bead_spec = resolve_imamura_bead_spec(
                u0,
                explicit_type1=args.type1_selection,
                explicit_type4=args.type4_selection,
            )
            bead_spec.validate(
                n_type1=args.n_type1_beads,
                n_type4=args.n_type4_beads,
            )
            write_imamura_bead_spec(bead_spec, out_dir / "bead_spec.json")
            plot_imamura_bead_spec(
                u0,
                monomer_sels or [],
                bead_spec,
                out_dir / "auto_beads.png",
            )

        if truncate and not formation_frames:
            with step("detect_formation_frames"):
                formation_frames = detect_formation_frames(
                    args.topology,
                    traj_paths,
                    monomer_sels,
                    n_monomers=args.n_monomers,
                    traj_format=args.format,
                    contact_cutoff=config.contact_cutoff,
                    min_active_interfaces=config.formation_min_active_interfaces,
                    sustain_frames=config.formation_sustain_frames,
                    stride=args.stride,
                )
                ff_path = out_dir / "formation_frames.csv"
                pd.DataFrame([
                    {"traj_id": k, "formation_frame": v}
                    for k, v in sorted(formation_frames.items())
                ]).to_csv(ff_path, index=False)
                print(f"  Wrote {ff_path}")
                for tid, fr in sorted(formation_frames.items()):
                    print(f"    {tid}: frame {fr}")

        if args.detect_formation_only:
            print("Formation detection complete (--detect-formation-only).")
            return

        if args.features_csv:
            with step("load_features"):
                feat_df = pd.read_csv(args.features_csv)
                from src.utils.imamura_msm import ImamuraFeatureResult, imamura_feature_names

                feat_names = [
                    c for c in feat_df.columns
                    if c.startswith("v1_") or c.startswith("v4_")
                ]
                if not feat_names:
                    feat_names = imamura_feature_names(
                        config.n_type1_beads, config.n_type4_beads
                    )
                feature_result = ImamuraFeatureResult(
                    dataframe=feat_df,
                    feature_names=feat_names,
                    formation_frames=formation_frames,
                )
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
                _sync_resolved_type4_bead_count(config, bead_spec)
                feature_result.bead_spec = bead_spec
        else:
            with step("extract_imamura_features"):
                feature_result = extract_imamura_features_multi(
                    args.topology,
                    traj_paths,
                    bead_spec,
                    traj_format=args.format,
                    formation_frames=formation_frames if truncate else None,
                    truncate_at_formation=truncate,
                    stride=args.stride,
                    n_type1=config.n_type1_beads,
                    n_type4=config.n_type4_beads,
                    n_jobs=args.n_jobs,
                )
                feature_result.formation_frames = formation_frames
                print(f"  {len(feature_result.dataframe)} frames extracted")

        with step("pca_and_cluster"):
            X = feature_result.dataframe[feature_result.feature_names].to_numpy()
            clustering = cluster_imamura_features(X, config=config)
            print(
                f"  PCA variance explained (sum): "
                f"{clustering.explained_variance_ratio.sum():.3f}"
            )
            print(
                f"  Microclusters: {len(set(clustering.micro_labels))}, "
                f"Macrostates: {len(set(clustering.macro_labels))}"
            )

        dt_ps = args.dt_ps
        if dt_ps is None:
            sample = feature_result.dataframe["frame"].head(2).tolist()
            dt_ps = _infer_dt_ps(u0, sample)
        lag = lag_frames_from_ns(config.transition_lag_ns, dt_ps, stride=args.stride)

        with step("transition_network"):
            labeled = feature_result.dataframe.copy()
            labeled["macro_label"] = clustering.macro_labels
            transitions = count_lagged_transitions(
                labeled,
                config.n_macrostates,
                lag_frames=lag,
            )
            transitions.lag_ns = config.transition_lag_ns
            print(
                f"  Lag: {config.transition_lag_ns} ns → {lag} frames "
                f"(dt={dt_ps} ps, stride={args.stride})"
            )
            print(f"  Transitions counted: {int(transitions.count_matrix.sum())}")

        with step("write_artifacts"):
            artifacts = write_imamura_artifacts(
                out_dir,
                feature_result,
                clustering,
                transitions,
                config=config,
            )

        print("\nArtifacts:")
        for name, path in sorted(artifacts.items()):
            print(f"  {name}: {path}")

        log_event("done", f"Imamura MSM complete → {out_dir}")


if __name__ == "__main__":
    main()
