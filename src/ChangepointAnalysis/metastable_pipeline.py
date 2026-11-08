"""End-to-end metastable-structure pipeline on existing own-elbow segments.

Does not re-detect changepoints or refit Ward. Stages follow
``docs/metastable_structures.md`` reproducibility order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .catalog_runs import (
    categorize_cluster_mixtures,
    compare_to_murata_metastructures,
    describe_endpoint_clusters,
    list_annotation_replicas,
    rank_metastable_structures,
    summarize_metastable_structures,
)
from .cluster_k_diagnostics import (
    discover_endpoint_k_cohort_dirs,
    scan_chemical_separation_by_k,
    write_physical_k_selection,
)
from .cluster_replicas import audit_cluster_replicas, write_replica_outliers
from .gsa_cohort_run import KNOWN_CUBES
from .reporting import OWN_ELBOW_ENDPOINT_DIRS, endpoint_changepoints_dir, rank_endpoint_pairs_at_k


def _own_elbow_dirs(output_root: Path) -> list[Path]:
    dirs = [
        endpoint_changepoints_dir(output_root, cube, own_elbow=True)
        for cube in OWN_ELBOW_ENDPOINT_DIRS
    ]
    found = [d for d in dirs if (d / "clusters" / "endpoint" / "segments_clustered.csv").exists()]
    if found:
        return found
    return discover_endpoint_k_cohort_dirs(output_root, "endpoint_changepoints_B*", exclude_test=True)


def run_metastable_pipeline(
    *,
    output_root: Path | str = "output",
    chem_dir: Path | str = "output/endpoint_cluster_chemical_k_own_elbow_replica_aware",
    physical_k_dir: Path | str = "output/endpoint_cluster_physical_k_own_elbow",
    gsa_features_root: Path | str = "output/gsa_features_by_cube",
    profiles_dir: Path | str = "output/endpoint_cluster_profiles",
    catalog_dir: Path | str = "output/metastable_structures_physical_k",
    pair_ranking_dir: Path | str = "output/endpoint_pair_ranking_physical_k",
    replica_audit_dir: Path | str = "output/cluster_replica_audit_physical_k",
    cubes: Optional[Sequence[str]] = None,
    k_min: int = 2,
    k_max: int = 15,
    max_top_replica_share: float = 0.5,
    n_permutations: int = 999,
    skip_guest: bool = False,
    skip_annotation: bool = False,
) -> dict[str, Path]:
    """Chemical gate → physical k → catalog → audit; guest/annotation when inputs exist."""
    output_root = Path(output_root)
    chem_dir = Path(chem_dir)
    physical_k_dir = Path(physical_k_dir)
    gsa_features_root = Path(gsa_features_root)
    profiles_dir = Path(profiles_dir)
    catalog_dir = Path(catalog_dir)
    pair_ranking_dir = Path(pair_ranking_dir)
    replica_audit_dir = Path(replica_audit_dir)
    wanted = list(cubes) if cubes is not None else list(KNOWN_CUBES)
    written: dict[str, Path] = {}

    dirs = _own_elbow_dirs(output_root)
    if not dirs:
        raise FileNotFoundError(
            f"No own-elbow endpoint clusterings under {output_root}. "
            "Run run_endpoint_changepoint / cluster_changepoint_segments first."
        )
    chem_written = scan_chemical_separation_by_k(
        dirs,
        chem_dir,
        k_min=k_min,
        k_max=k_max,
        n_permutations=n_permutations,
        max_top_replica_share=max_top_replica_share,
    )
    written.update({f"chem_{k}": v for k, v in chem_written.items()})

    phys = write_physical_k_selection(
        output_root=output_root,
        chem_dir=chem_dir,
        gsa_features_root=gsa_features_root,
        out_dir=physical_k_dir,
        cubes=wanted,
        own_elbow=True,
    )
    written.update({f"physical_{k}": v for k, v in phys.items()})
    k_selected = physical_k_dir / "physical_k_selected.csv"

    ranked = rank_endpoint_pairs_at_k(
        output_root=output_root,
        out_dir=pair_ranking_dir,
        cubes=wanted,
        own_elbow=True,
        k_selected=k_selected,
        n_permutations=n_permutations,
    )
    written.update({f"pairs_{k}": v for k, v in ranked.items()})

    selected = pd.read_csv(k_selected)
    for rec in selected.itertuples(index=False):
        cube = str(rec.cohort)
        if cube not in wanted:
            continue
        k = int(rec.selected_k)
        desc = describe_endpoint_clusters(
            output_root=output_root,
            cube=cube,
            own_elbow=True,
            k=k,
            qc_gsa_root=gsa_features_root,
            n_permutations=n_permutations,
            out_dir=profiles_dir / f"{cube}_k{k:02d}",
        )
        written.update({f"describe_{cube}_{name}": path for name, path in desc.items()})

    catalog = rank_metastable_structures(
        output_root=output_root,
        k_selected=k_selected,
        profiles=profiles_dir,
        replica_audit=replica_audit_dir / "single_replica_clusters.csv",
        own_elbow=True,
        qc_gsa_root=gsa_features_root,
        out_dir=catalog_dir,
    )
    written.update({f"rank_{k}": v for k, v in catalog.items()})

    mix = categorize_cluster_mixtures(
        output_root=output_root,
        k_selected=k_selected,
        qc_gsa_root=gsa_features_root,
        out_dir=catalog_dir,
    )
    written.update({f"mix_{k}": v for k, v in mix.items()})
    mix0 = categorize_cluster_mixtures(
        output_root=output_root,
        k_selected=k_selected,
        qc_gsa_root=gsa_features_root,
        min_share=0.0,
        out_dir=catalog_dir / "min_share_0",
    )
    written.update({f"mix0_{k}": v for k, v in mix0.items()})

    summary = summarize_metastable_structures(
        output_root=output_root,
        catalog_dir=catalog_dir,
        segment_belt=catalog_dir / "min_share_0" / "segment_belt.csv",
        top_pairs=pair_ranking_dir / "top_pairs.csv",
        out_dir=catalog_dir,
    )
    written.update({f"summary_{k}": v for k, v in summary.items()})

    audit = audit_cluster_replicas(
        output_root=output_root,
        out_dir=replica_audit_dir,
        cubes=wanted,
        own_elbow=True,
        k_selected=k_selected,
        gsa_root=gsa_features_root,
        min_top_share=0.5,
    )
    written.update({f"audit_{k}": v for k, v in audit.items()})

    outliers = write_replica_outliers(
        output_root=output_root,
        gsa_root=gsa_features_root,
        k_selected=k_selected,
        cubes=wanted,
        min_top_share=0.5,
        out_dir=output_root / "replica_outliers",
    )
    written.update({f"outlier_{k}": v for k, v in outliers.items()})

    try:
        murata = compare_to_murata_metastructures(
            output_root=output_root,
            out_dir=output_root / "murata_g5_own_elbow_physical_k",
            cubes=wanted,
            qc_gsa_root=gsa_features_root,
            own_elbow=True,
            k_selected=k_selected,
        )
        written.update({f"murata_{k}": v for k, v in murata.items()})
    except FileNotFoundError:
        pass

    if not skip_guest:
        dwells = output_root / "guest_event_cotiming" / "guest_dwells.csv"
        try:
            from .guest_cotiming import guest_event_cotiming
            from .guest_entry_response import guest_entry_geometry_response
            from .guest_geometry import compare_guest_cage_geometry

            cot = guest_event_cotiming(
                output_root=output_root,
                gsa_features_root=gsa_features_root,
                out_dir=output_root / "guest_event_cotiming",
                cubes=wanted,
                min_dwell=(20, 50),
            )
            written.update({f"guest_{k}": v for k, v in cot.items()})
            geom = compare_guest_cage_geometry(
                output_root=output_root,
                gsa_features_root=gsa_features_root,
                dwells=dwells if dwells.is_file() else cot["guest_dwells"],
                k_selected=k_selected,
                out_dir=output_root / "guest_cage_geometry",
                cubes=wanted,
            )
            written.update({f"guestgeom_{k}": v for k, v in geom.items()})
            entry = guest_entry_geometry_response(
                output_root=output_root,
                gsa_features_root=gsa_features_root,
                out_dir=output_root / "guest_cage_geometry",
                cubes=wanted,
            )
            written.update({f"guestentry_{k}": v for k, v in entry.items()})
        except FileNotFoundError:
            pass

    if not skip_annotation:
        try:
            ann = list_annotation_replicas(
                output_root=output_root,
                catalog_dir=catalog_dir,
                k_selected=k_selected,
                out_dir=output_root / "annotation_replicas",
            )
            written.update({f"ann_{k}": v for k, v in ann.items()})
        except FileNotFoundError:
            pass

    return written


def _cube_from_topology(topology: Path | str) -> str:
    name = Path(topology).name
    for suffix in ("_ca.prmtop", ".prmtop", ".parm7", ".top"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(topology).stem


def _prefixed_traj_ids(paths: Sequence[Path | str], cube: str) -> list[str]:
    from .traj_paths import unique_trajectory_ids

    prefix = f"{cube}_"
    ids = unique_trajectory_ids(paths)
    return [tid if tid.startswith(prefix) else prefix + tid for tid in ids]


def _gsa_changepoints_dir(output_root: Path, cube: str) -> Path:
    if cube in OWN_ELBOW_ENDPOINT_DIRS:
        return output_root / f"gsa_changepoints_matched_{cube}"
    return output_root / f"gsa_changepoints_{cube}"


def run_cube_from_trajectories(
    *,
    cube: str,
    topology: Path | str,
    trajectories: Sequence[Path | str],
    output_root: Path | str = "output",
    gsa_resname: str = "MOL",
    guest_selection: str = "resname IOD",
    traj_format: Optional[str] = None,
    stride: int = 1,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    traj_jobs: int = -1,
    n_jobs: int = 1,
    with_sweep: bool = True,
    include_site_pairs: bool = True,
    k_max: int = 15,
    skip_gsa_features: bool = False,
    skip_gsa_changepoint: bool = False,
    skip_endpoint: bool = False,
    skip_motif: bool = False,
    force: bool = False,
) -> dict[str, Path]:
    """GSA features, GSA/endpoint changepoints, and motif RMSD for one cube.

    Writes into the own-elbow folder names used by :func:`run_metastable_pipeline`.
    Does not run the catalog; call that afterward.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import MDAnalysis as mda

    from src.utils.gsa_feature_observer import compute_gsa_features
    from src.utils.gsa_selections import GSAFeatureSelections

    from .detection import ChangepointConfig
    from .murata_rmsd import write_trajectory_motif_rmsd
    from .penalty_sweep import PenaltySweepConfig
    from .pipeline import ChangepointPipeline, _resolve_traj_workers, run_endpoint_changepoint
    from .segment_clustering import SegmentClusteringConfig
    from .traj_paths import expand_trajectories, format_no_trajectories_error

    output_root = Path(output_root)
    topology = Path(topology)
    trajs = expand_trajectories([str(p) for p in trajectories])
    if not trajs:
        raise FileNotFoundError(format_no_trajectories_error(trajectories))
    ids = _prefixed_traj_ids(trajs, cube)
    written: dict[str, Path] = {}

    gsa_dir = output_root / "gsa_features_by_cube" / cube
    gsa_csvs = [gsa_dir / f"{tid}_gsa_features.csv" for tid in ids]
    if force or not (skip_gsa_features or all(p.is_file() for p in gsa_csvs)):
        print(f"{cube}: GSA features ({len(trajs)} trajectories) → {gsa_dir}", flush=True)
        gsa_dir.mkdir(parents=True, exist_ok=True)
        selections = GSAFeatureSelections(guest_sel=guest_selection)
        fmt = traj_format or "TRJ"
        for path, traj_id in zip(trajs, ids):
            u = mda.Universe(str(topology), str(path), format=fmt)
            compute_gsa_features(
                u,
                selections=selections,
                gsa_resname=gsa_resname,
                n_monomers=6,
                traj_id=traj_id,
                out_prefix=str(gsa_dir / traj_id),
                start=start,
                stop=stop,
                step=stride,
                n_jobs=n_jobs,
            )
    for csv in gsa_csvs:
        if csv.is_file():
            written[f"gsa_{csv.stem}"] = csv

    gsa_cp = _gsa_changepoints_dir(output_root, cube)
    if force or not (skip_gsa_changepoint or (gsa_cp / "all_breakpoints.csv").is_file()):
        print(f"{cube}: GSA changepoint → {gsa_cp}", flush=True)
        detection = ChangepointConfig(groups=("gsa",))
        clustering = SegmentClusteringConfig(groups=("gsa",), n_clusters=5)
        sweep = PenaltySweepConfig(detection=detection) if with_sweep else None
        ChangepointPipeline(
            gsa_dir,
            gsa_cp,
            detection=detection,
            sweep=sweep,
            clustering=clustering,
        ).run_all(with_sweep=with_sweep)
    if (gsa_cp / "all_breakpoints.csv").is_file():
        written["gsa_changepoints"] = gsa_cp

    ep_dir = endpoint_changepoints_dir(output_root, cube, own_elbow=True)
    clustered = ep_dir / "clusters" / "endpoint" / "segments_clustered.csv"
    if force or not (skip_endpoint or clustered.is_file()):
        print(f"{cube}: endpoint changepoint → {ep_dir}", flush=True)
        detection = ChangepointConfig(groups=("endpoint",))
        clustering = SegmentClusteringConfig(
            groups=("endpoint",),
            n_clusters=5,
            silhouette_k_max=k_max,
            all_k_min=2,
            all_k_max=k_max,
            save_all_k=True,
        )
        run_endpoint_changepoint(
            topology,
            trajs,
            output_dir=ep_dir,
            gsa_resname=gsa_resname,
            include_site_pairs=include_site_pairs,
            traj_ids=ids,
            traj_format=traj_format,
            start=start,
            stop=stop,
            step=stride,
            traj_jobs=traj_jobs,
            n_jobs=n_jobs if traj_jobs == 1 else 1,
            detection=detection,
            clustering=clustering,
            with_sweep=with_sweep,
            sweep=PenaltySweepConfig(detection=detection) if with_sweep else None,
            rmsd_from=gsa_dir if gsa_dir.is_dir() else None,
        )
    if clustered.is_file():
        written["endpoint_changepoints"] = ep_dir

    motif_dir = output_root / f"{cube}_motif_rmsd"
    motif_csvs = [motif_dir / f"{tid}_motif_rmsd.csv" for tid in ids]
    if force or not (skip_motif or all(p.is_file() for p in motif_csvs)):
        print(f"{cube}: motif RMSD → {motif_dir}", flush=True)
        motif_dir.mkdir(parents=True, exist_ok=True)
        workers = _resolve_traj_workers(traj_jobs, len(trajs))
        jobs = []
        for path, traj_id, csv in zip(trajs, ids, motif_csvs):
            gsa_csv = gsa_dir / f"{traj_id}_gsa_features.csv"
            jobs.append(
                (
                    str(topology),
                    str(path),
                    traj_id,
                    cube,
                    str(csv),
                    str(gsa_csv) if gsa_csv.is_file() else None,
                    start,
                    stop,
                    stride,
                    0,
                    6,
                    traj_format,
                )
            )
        if workers == 1:
            for job in jobs:
                write_trajectory_motif_rmsd(*job)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(write_trajectory_motif_rmsd, *job) for job in jobs]
                for fut in as_completed(futs):
                    fut.result()
    for csv in motif_csvs:
        if csv.is_file():
            written[f"motif_{csv.stem}"] = csv
    return written
