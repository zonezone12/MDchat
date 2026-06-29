"""
Annotate Imamura bead map with top clustering (or PCA) drivers from an MSM run.

Re-fits PCA + clustering on saved ``imamura_features.csv``, traces feature
importance back to bead pairs using the first trajectory frame as reference
geometry, and writes ranked CSVs plus 2D/3D bead maps.

Default mode ``cluster`` ranks beads whose inter-bead distances differ most
between macrostates (ANOVA). Use ``--importance pca`` for the legacy
PCA-variance attribution.

Example
-------
python scripts/annotate_imamura_bead_pca.py \\
    --output-dir output/imamura_msm/endpoint_BMMpM \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectories traj/BMMpM_891249_mdcrd_v.trj \\
    --format TRJ
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.gsa_selections import resolve_selections
from src.utils.imamura_msm import (
    DEFAULT_PCA_COMPONENTS,
    ImamuraBeadSpec,
    ImamuraFeatureResult,
    ImamuraMSMConfig,
    cluster_imamura_features,
    imamura_feature_names,
    write_imamura_bead_importance_artifacts,
)


def _load_bead_spec(path: Path) -> ImamuraBeadSpec:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("mode") == "ring_centroids":
        return ImamuraBeadSpec(
            type1_ring_groups=[tuple(g) for g in data["type1_ring_groups"]],
            type4_atom_ids=tuple(data["type4_atom_ids"]),
            type4_source=data.get("type4_source", "methyl"),
            endpoint_extend_ring=data.get("endpoint_extend_ring", True),
            endpoint_exclude_center_benzene=data.get(
                "endpoint_exclude_center_benzene", True
            ),
            type4_rank=data.get("type4_rank", "bond"),
        )
    return ImamuraBeadSpec(
        type1_selection=data.get("type1_selection"),
        type4_selection=data.get("type4_selection"),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Trace Imamura clustering (or PCA) drivers to beads.",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--topology", required=True)
    p.add_argument("--trajectories", nargs="+", required=True)
    p.add_argument("--format", default=None)
    p.add_argument("--features-csv", default=None, help="Override imamura_features.csv")
    p.add_argument("--bead-spec", default=None, help="Override bead_spec.json")
    p.add_argument("--top-n", type=int, default=6)
    p.add_argument("--n-pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    p.add_argument(
        "--importance",
        choices=("cluster", "pca"),
        default="cluster",
        help="cluster = macrostate separation (default); pca = explained variance",
    )
    p.add_argument(
        "--cluster-level",
        choices=("macro", "micro"),
        default="macro",
        help="Label column for cluster attribution (default: macro)",
    )
    p.add_argument(
        "--cluster-method",
        choices=("pca_centroid", "anova", "between_var"),
        default="pca_centroid",
        help="pca_centroid = two-stage PCA centroids (default); anova = raw-feature F-test",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    feat_path = Path(args.features_csv) if args.features_csv else out_dir / "imamura_features.csv"
    spec_path = Path(args.bead_spec) if args.bead_spec else out_dir / "bead_spec.json"

    if not feat_path.is_file():
        raise FileNotFoundError(f"Features not found: {feat_path}")
    if not spec_path.is_file():
        raise FileNotFoundError(f"Bead spec not found: {spec_path}")

    bead_spec = _load_bead_spec(spec_path)
    n_type1 = len(bead_spec.type1_ring_groups or ())
    n_type4 = len(bead_spec.type4_atom_ids or ())
    if n_type1 < 1 or n_type4 < 1:
        raise ValueError("Bead spec must define type-1 rings and type-4 atoms.")

    print(f"Loading features from {feat_path} ...")
    feat_df = pd.read_csv(feat_path)
    feat_names = [c for c in feat_df.columns if c.startswith("v1_") or c.startswith("v4_")]
    if not feat_names:
        feat_names = imamura_feature_names(n_type1, n_type4)

    feature_result = ImamuraFeatureResult(
        dataframe=feat_df,
        feature_names=feat_names,
        bead_spec=bead_spec,
    )

    config = ImamuraMSMConfig(
        n_type1_beads=n_type1,
        n_type4_beads=n_type4,
        n_pca_components=args.n_pca_components,
    )
    X = feat_df[feat_names].to_numpy(dtype="float64")
    print(f"Fitting PCA + clustering on {X.shape[0]} frames × {X.shape[1]} features ...")
    clustering = cluster_imamura_features(X, config=config)
    print(
        "Explained variance:",
        ", ".join(
            f"PC{i + 1}={v:.1%}"
            for i, v in enumerate(clustering.explained_variance_ratio)
        ),
    )
    print(
        f"Clusters: {len(set(clustering.micro_labels))} micro, "
        f"{len(set(clustering.macro_labels))} macro"
    )

    import MDAnalysis as mda

    u0 = mda.Universe(
        args.topology,
        str(args.trajectories[0]),
        **({"format": args.format} if args.format else {}),
    )
    resolved = resolve_selections(u0, gsa_resname="MOL", n_monomers=6, auto_tooth=False)
    monomer_sels = resolved.monomer_selections or []

    written = write_imamura_bead_importance_artifacts(
        out_dir,
        feature_result,
        clustering,
        u0,
        monomer_sels,
        bead_spec,
        top_n=args.top_n,
        importance=args.importance,
        cluster_level=args.cluster_level,
        cluster_method=args.cluster_method,
    )

    prefix = "cluster" if args.importance == "cluster" else "pca"
    bead_csv = written[f"bead_{prefix}_importance"]
    top = pd.read_csv(bead_csv).head(args.top_n)
    label = "cluster driver" if args.importance == "cluster" else "PCA driver"
    print(f"\nTop {args.top_n} {label} beads:")
    for _, row in top.iterrows():
        print(
            f"  #{int(row['rank'])} {row['bead_label']} "
            f"(monomer {int(row['monomer'])}, importance={row['importance']:.4g})"
        )

    print("\nArtifacts:")
    for name, path in sorted(written.items()):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
