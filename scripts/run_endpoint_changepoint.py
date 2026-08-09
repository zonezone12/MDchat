"""
Extract endpoint-site distances and run changepoint analysis.

Computes ring-system / atom site centroid distances per GSA monomer pair,
writes ``*_endpoint_features.csv``, then runs the changepoint pipeline on the
``endpoint`` feature group. Optionally overlays assembly RMSD from a prior GSA
features directory for comparison against the paper's A/B/C1/C2 RMSD peaks.

Example
-------
python scripts/run_endpoint_changepoint.py \\
    --topology traj/BHHpH_ca.prmtop \\
    --trajectories traj/BHHpH_*_mdcrd_v.trj \\
    --gsa-resname MOL \\
    --output-dir output/endpoint_changepoints \\
    --rmsd-from output/gsa_features
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis import (
    ChangepointConfig,
    EndpointTransitionAttributionConfig,
    PenaltySweepConfig,
    SegmentClusteringConfig,
)
from src.ChangepointAnalysis.pipeline import run_endpoint_changepoint
from src.utils.run_log import RunContext


_GENERIC_TRAJ_STEMS = frozenset({
    "mdcrd", "mdcrd_v", "crd", "dcd", "xtc", "trj", "nc", "prod", "equil",
})


def _trajectory_output_id(traj_path: Path) -> str:
    stem = traj_path.stem
    parent = traj_path.parent.name
    if stem.lower() in _GENERIC_TRAJ_STEMS and parent not in ("", ".", ".."):
        return f"{parent}_{stem}"
    return stem


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
    p = argparse.ArgumentParser(
        description=(
            "Endpoint-site (ring-centroid) distances → changepoint pipeline "
            "for A/B/C1/C2 metastructure comparison."
        )
    )
    p.add_argument("--topology", required=True, help="Shared topology file")
    p.add_argument(
        "--trajectories",
        nargs="+",
        required=True,
        help="Trajectory paths or glob patterns",
    )
    p.add_argument("--gsa-resname", default="MOL")
    p.add_argument("--n-monomers", type=int, default=6)
    p.add_argument(
        "--no-ring-centroids",
        action="store_true",
        help="Use flat atom endpoints instead of ring-system centroids",
    )
    p.add_argument("--ring-min-gap-deg", type=float, default=35.0)
    p.add_argument("--ring-max-per-ring", type=int, default=3)
    p.add_argument(
        "--include-site-pairs",
        action="store_true",
        help=(
            "Also emit per-site-pair columns (endpoint_dist_isA_jsB). "
            "Required for transition driver attribution."
        ),
    )
    p.add_argument(
        "--no-paper-d1",
        action="store_true",
        help=(
            "Disable paper-d1 features (one-step-in ring neighbors of s3/s7). "
            "Enabled by default."
        ),
    )
    p.add_argument(
        "--paper-d1-open-lo",
        type=float,
        default=4.5,
        help="Lower bound (Å) of paper open cation–π window (default: 4.5)",
    )
    p.add_argument(
        "--paper-d1-open-hi",
        type=float,
        default=5.5,
        help="Upper bound (Å) of paper open cation–π window (default: 5.5)",
    )
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--stop", type=int, default=None)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--time-per-frame-ps", type=float, default=1.0)
    p.add_argument(
        "--output-dir",
        default="output/endpoint_changepoints",
        help="Pipeline output directory",
    )
    p.add_argument(
        "--features-dir",
        default=None,
        help="Where to write *_endpoint_features.csv (default: <output-dir>/endpoint_features)",
    )
    p.add_argument(
        "--rmsd-from",
        default=None,
        help="Optional directory of *_gsa_features.csv for assembly_rmsd_to_ref overlay",
    )
    p.add_argument("--method", default="Pelt")
    p.add_argument("--cost-model", default="rbf")
    p.add_argument("--penalty", type=float, default=None)
    p.add_argument("--min-size", type=int, default=10)
    p.add_argument("--jump", type=int, default=5)
    p.add_argument("--tolerance-frames", type=int, default=50)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--with-sweep", action="store_true")
    p.add_argument("--n-penalties", type=int, default=15)
    p.add_argument("--n-clusters", "--k", type=int, default=5, dest="n_clusters")
    p.add_argument("--linkage", default="ward")
    p.add_argument("--skip-clustering", action="store_true")
    p.add_argument("--skip-summarize", action="store_true")
    p.add_argument(
        "--skip-transition-attribution",
        action="store_true",
        help="Skip attributing directed cluster transitions to site-pair features",
    )
    p.add_argument(
        "--transition-top-n",
        type=int,
        default=10,
        help="Top-N site-pair drivers to retain per directed transition (default: 10)",
    )
    p.add_argument(
        "--transition-min-abs-corr",
        type=float,
        default=0.0,
        help=(
            "Minimum |point-biserial correlation| to retain a driver "
            "(default: 0 = keep all ranked features)"
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    traj_paths = _expand_trajectories(args.trajectories)
    if not traj_paths:
        raise SystemExit("No trajectories found for the given patterns.")

    traj_ids = [_trajectory_output_id(p) for p in traj_paths]
    # Disambiguate collisions
    counts: dict[str, int] = {}
    unique_ids: list[str] = []
    for tid in traj_ids:
        n = counts.get(tid, 0)
        counts[tid] = n + 1
        unique_ids.append(tid if n == 0 else f"{tid}_{n}")

    detection = ChangepointConfig(
        method=args.method,
        cost_model=args.cost_model,
        penalty=args.penalty,
        min_size=args.min_size,
        jump=args.jump,
        tolerance_frames=args.tolerance_frames,
        normalize=not args.no_normalize,
        groups=("endpoint",),
    )
    clustering = SegmentClusteringConfig(
        groups=("endpoint",),
        linkage=args.linkage,
        n_clusters=args.n_clusters,
    )
    sweep = None
    if args.with_sweep:
        sweep = PenaltySweepConfig(detection=detection, n_penalties=args.n_penalties)

    transition_cfg = EndpointTransitionAttributionConfig(
        top_n=int(args.transition_top_n),
        min_abs_correlation=float(args.transition_min_abs_corr),
    )

    with RunContext.from_namespace(args, name="run_endpoint_changepoint"):
        artifacts = run_endpoint_changepoint(
            args.topology,
            traj_paths,
            output_dir=args.output_dir,
            features_dir=args.features_dir,
            gsa_resname=args.gsa_resname,
            n_monomers=args.n_monomers,
            use_ring_centroids=not args.no_ring_centroids,
            ring_min_gap_deg=args.ring_min_gap_deg,
            ring_max_per_ring=args.ring_max_per_ring,
            include_site_pairs=args.include_site_pairs,
            include_paper_d1=not args.no_paper_d1,
            paper_d1_open_lo=args.paper_d1_open_lo,
            paper_d1_open_hi=args.paper_d1_open_hi,
            start=args.start,
            stop=args.stop,
            step=args.step,
            time_per_frame_ps=args.time_per_frame_ps,
            traj_ids=unique_ids,
            detection=detection,
            clustering=clustering,
            with_sweep=args.with_sweep,
            sweep=sweep,
            skip_clustering=args.skip_clustering,
            skip_summarize=args.skip_summarize,
            skip_transition_attribution=args.skip_transition_attribution,
            transition_attribution=transition_cfg,
            rmsd_from=args.rmsd_from,
        )

    print("Artifacts:")
    for name, path in sorted(artifacts.items(), key=lambda kv: str(kv[0])):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
