"""
Extract endpoint-site distances and run changepoint analysis.

Computes ring-system / atom site centroid distances per GSA monomer pair,
writes ``*_endpoint_features.csv``, then runs the changepoint pipeline on the
``endpoint`` feature group. Optionally overlays assembly RMSD from a prior GSA
features directory for comparison against the paper's A/B/C1/C2 RMSD peaks.

Example
-------
# Flat layout
python scripts/run_endpoint_changepoint.py \\
    --topology traj/BHHpH_ca.prmtop \\
    --trajectories traj/BHHpH_*_mdcrd_v.trj \\
    --gsa-resname MOL \\
    --output-dir output/endpoint_changepoints \\
    --rmsd-from output/gsa_features

# HPC nested layout ($TRAJ_DIR/<run_id>/mdcrd_v)
python scripts/run_endpoint_changepoint.py \\
    --topology "$TOPOLOGY" \\
    --trajectories "$TRAJ_DIR/*/mdcrd_v" \\
    --gsa-resname MOL \\
    --output-dir output/endpoint_changepoints \\
    --include-site-pairs --traj-jobs -1
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Optional

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
_TRAJ_EXTENSIONS = (".trj", ".xtc", ".dcd", ".nc", ".crd")


def _trajectory_output_id(traj_path: Path) -> str:
    stem = traj_path.stem
    parent = traj_path.parent.name
    if stem.lower() in _GENERIC_TRAJ_STEMS and parent not in ("", ".", ".."):
        return f"{parent}_{stem}"
    return stem


def _concrete_traj_files(path: Path) -> list[Path]:
    """Resolve a glob hit to one or more readable trajectory files.

    Supports HPC nested layouts such as ``$TRAJ_DIR/<run_id>/mdcrd_v``
    (extensionless Amber mdcrd) as well as ``mdcrd_v.trj`` / directory hits.
    """
    if path.is_file():
        return [path]

    out: list[Path] = []
    if path.is_dir():
        for name in ("mdcrd_v", "mdcrd"):
            base = path / name
            if base.is_file():
                out.append(base)
                continue
            for ext in _TRAJ_EXTENSIONS:
                cand = Path(f"{base}{ext}")
                if cand.is_file():
                    out.append(cand)
        if out:
            return out
        for ext in ("*.xtc", "*.trj", "*.dcd", "*.nc", "*.crd"):
            out.extend(sorted(path.glob(ext)))
        return out

    # Path does not exist as written — try common extensions on the basename.
    for ext in _TRAJ_EXTENSIONS:
        cand = Path(f"{path}{ext}")
        if cand.is_file():
            out.append(cand)
    return out


def _expand_trajectories(patterns: list[str]) -> list[Path]:
    """Expand globs / paths into concrete trajectory files.

    Typical HPC pattern::

        --trajectories "$TRAJ_DIR/*/mdcrd_v"
    """
    paths: list[Path] = []
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        # Also accept mdcrd_v.trj etc. when the pattern ends with a bare basename.
        if not matches and not any(ch in Path(pat).name for ch in "*?[]"):
            for ext in _TRAJ_EXTENSIONS:
                matches.extend(sorted(glob.glob(f"{pat}{ext}")))
        elif not matches and Path(pat).name in _GENERIC_TRAJ_STEMS:
            matches = sorted(glob.glob(f"{pat}.*"))

        # Fallback: "$TRAJ_DIR/*" style — each match is a run folder.
        if not matches:
            alt = sorted(glob.glob(pat.rstrip("/")))
            if not alt and pat.endswith("*/mdcrd_v"):
                alt = sorted(glob.glob(pat[: -len("/mdcrd_v")]))
            if alt:
                matches = alt

        if matches:
            for m in matches:
                paths.extend(_concrete_traj_files(Path(m)))
        else:
            p = Path(pat)
            resolved = _concrete_traj_files(p)
            if resolved:
                paths.extend(resolved)
            elif p.is_dir():
                for ext in ("*.xtc", "*.trj", "*.dcd", "*.nc", "*.crd"):
                    paths.extend(sorted(p.glob(ext)))

    seen: set[Path] = set()
    unique: list[Path] = []
    for p in paths:
        if not p.is_file():
            continue
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


def _format_no_trajectories_error(patterns: list[str]) -> str:
    """Build a diagnostic message when trajectory globs match nothing."""
    lines = [
        "No trajectories found for the given patterns:",
        *[f"  {pat!r}" for pat in patterns],
        "",
        "Hints:",
        "  • Quote the glob so Python expands it: --trajectories \"$TRAJ_DIR/*/mdcrd_v\"",
        "  • Put a space before every line-continuation backslash (and no space after \\).",
        "  • Check the nested layout exists, e.g.:",
    ]
    for pat in patterns:
        norm = pat.replace("\\", "/")
        traj_dir: Optional[Path] = None
        if "/*/mdcrd_v" in norm:
            traj_dir = Path(norm.split("/*/mdcrd_v", 1)[0])
        elif norm.endswith("/*"):
            traj_dir = Path(norm[:-2])
        else:
            parent = Path(pat).parent
            if "*" not in parent.name:
                traj_dir = parent
        if traj_dir is None:
            continue
        lines.append(f"      ls \"{traj_dir}\" | head")
        if traj_dir.is_dir():
            kids = sorted(traj_dir.iterdir())[:8]
            if not kids:
                lines.append(f"    (directory exists but is empty: {traj_dir})")
            else:
                lines.append(f"    Found under {traj_dir}:")
                for kid in kids:
                    marker = ""
                    if kid.is_dir():
                        md = kid / "mdcrd_v"
                        marker = (
                            " [has mdcrd_v]"
                            if md.is_file()
                            else " [NO mdcrd_v]"
                        )
                    lines.append(f"      - {kid.name}{marker}")
                if len(list(traj_dir.iterdir())) > 8:
                    lines.append("      - ...")
        else:
            lines.append(f"    (path does not exist: {traj_dir})")
    return "\n".join(lines)


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
        help=(
            "Trajectory paths or globs. Nested HPC layout: "
            "'$TRAJ_DIR/*/mdcrd_v' (ids become <run>_mdcrd_v)."
        ),
    )
    p.add_argument(
        "--traj-format",
        default=None,
        help=(
            "MDAnalysis trajectory format (default: TRJ for extensionless "
            "mdcrd_v / .trj; auto-detect otherwise)."
        ),
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
        "--no-murata-criteria",
        action="store_true",
        help=(
            "Disable Murata cation–π / d2 / RMSD / A-B-C1-C2 labels. "
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
        "--n-jobs",
        type=int,
        default=None,
        help=(
            "Parallel workers for frames within one trajectory "
            "(default: 1 when traj_jobs>1, else platform default)."
        ),
    )
    p.add_argument(
        "--traj-jobs",
        type=int,
        default=-1,
        help=(
            "Parallel workers across trajectories "
            "(default: -1 = all CPUs, capped by traj count)."
        ),
    )
    p.add_argument(
        "--use-dask",
        action="store_true",
        help="Use Dask Distributed for the endpoint distance pass.",
    )
    p.add_argument(
        "--max-workers-for-io",
        type=int,
        default=None,
        help=(
            "Cap parallel workers for I/O-bound trajectory reads "
            "(TrajectoryIterator default limit is 16)."
        ),
    )
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
    p.add_argument(
        "--include-site-pairs-in-detection",
        action="store_true",
        help=(
            "Include raw endpoint_dist_{i}s{a}_{j}s{b} columns in changepoint "
            "detection (default: aggregates only)"
        ),
    )
    p.add_argument("--with-sweep", action="store_true")
    p.add_argument("--n-penalties", type=int, default=15)
    p.add_argument("--n-clusters", "--k", type=int, default=5, dest="n_clusters")
    p.add_argument("--linkage", default="ward")
    p.add_argument("--skip-clustering", action="store_true")
    p.add_argument("--skip-summarize", action="store_true")
    p.add_argument(
        "--timeline-top-pairs",
        type=int,
        default=5,
        help=(
            "Top-N endpoint_dist_{i}_{j}_mean features (by |point-biserial| vs "
            "cluster membership) for timeline_cluster plots. 0 disables and "
            "keeps aggregate endpoint panels (default: 5)."
        ),
    )
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
        raise SystemExit(_format_no_trajectories_error(args.trajectories))
    print(f"Found {len(traj_paths)} trajectory file(s):")
    for p in traj_paths[:20]:
        print(f"  {p}")
    if len(traj_paths) > 20:
        print(f"  ... and {len(traj_paths) - 20} more")

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
        include_site_pairs_in_detection=args.include_site_pairs_in_detection,
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
            include_murata_criteria=not args.no_murata_criteria,
            paper_d1_open_lo=args.paper_d1_open_lo,
            paper_d1_open_hi=args.paper_d1_open_hi,
            n_jobs=args.n_jobs,
            traj_jobs=args.traj_jobs,
            traj_format=args.traj_format,
            use_dask=args.use_dask,
            max_workers_for_io=args.max_workers_for_io,
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
            cluster_timeline_top_pairs=args.timeline_top_pairs,
            skip_transition_attribution=args.skip_transition_attribution,
            transition_attribution=transition_cfg,
            rmsd_from=args.rmsd_from,
        )

    print("Artifacts:")
    for name, path in sorted(artifacts.items(), key=lambda kv: str(kv[0])):
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
