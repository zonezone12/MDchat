"""
End-to-end GSA feature-group changepoint pipeline.

Runs detect → (optional penalty sweep) → segment clustering → summarize
via ``src.ChangepointAnalysis.ChangepointPipeline``.

Example
-------
python scripts/run_changepoint_pipeline.py \\
    --features-dir output/gsa_features \\
    --output-dir output/changepoints

python scripts/run_changepoint_pipeline.py \\
    --features-dir output/gsa_features \\
    --output-dir output/changepoints \\
    --with-sweep --n-penalties 11 --n-clusters 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis import (
    ChangepointConfig,
    ChangepointPipeline,
    PenaltySweepConfig,
    SegmentClusteringConfig,
)
from src.ChangepointAnalysis.feature_groups import DEFAULT_GROUPS
from src.utils.run_log import RunContext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the full changepoint analysis pipeline (detect → cluster → summarize)."
    )
    p.add_argument(
        "--features-dir",
        default="output/gsa_features",
        help="Directory with *_gsa_features.csv files",
    )
    p.add_argument(
        "--output-dir",
        default="output/changepoints",
        help="Pipeline output directory (default: output/changepoints)",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=list(DEFAULT_GROUPS),
        choices=list(DEFAULT_GROUPS),
        help="Feature groups to analyse (default: all four)",
    )
    p.add_argument("--method", default="Pelt")
    p.add_argument("--cost-model", default="rbf")
    p.add_argument("--penalty", type=float, default=None)
    p.add_argument("--min-size", type=int, default=10)
    p.add_argument("--jump", type=int, default=5)
    p.add_argument("--tolerance-frames", type=int, default=50)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument(
        "--with-sweep",
        action="store_true",
        help="Run penalty sweep first and use the recommended elbow penalty",
    )
    p.add_argument("--n-penalties", type=int, default=15)
    p.add_argument("--n-clusters", "--k", type=int, default=5, dest="n_clusters")
    p.add_argument("--linkage", default="ward")
    p.add_argument("--skip-clustering", action="store_true")
    p.add_argument("--skip-summarize", action="store_true")
    p.add_argument("--skip-pca", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    features_dir = Path(args.features_dir)
    output_dir = Path(args.output_dir)

    detection = ChangepointConfig(
        method=args.method,
        cost_model=args.cost_model,
        penalty=args.penalty,
        min_size=args.min_size,
        jump=args.jump,
        tolerance_frames=args.tolerance_frames,
        normalize=not args.no_normalize,
        groups=tuple(args.groups),
    )
    clustering = SegmentClusteringConfig(
        groups=tuple(args.groups),
        linkage=args.linkage,
        n_clusters=args.n_clusters,
        skip_pca=args.skip_pca,
    )
    sweep = None
    if args.with_sweep:
        sweep = PenaltySweepConfig(
            n_penalties=args.n_penalties,
            detection=detection,
        )

    pipe = ChangepointPipeline(
        features_dir,
        output_dir,
        detection=detection,
        sweep=sweep,
        clustering=clustering,
    )

    with RunContext.from_namespace(args, name="run_changepoint_pipeline"):
        artifacts = pipe.run_all(
            with_sweep=args.with_sweep,
            skip_clustering=args.skip_clustering,
            skip_summarize=args.skip_summarize,
        )

    print(f"\nPipeline complete. Artifacts under {output_dir}:")
    for label, path in sorted(artifacts.items(), key=lambda kv: str(kv[1])):
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
