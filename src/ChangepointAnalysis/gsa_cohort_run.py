"""Per-cube GSA changepoint + clustering from mixed ``gsa_features_step1`` CSVs.

``output/gsa_features_step1`` holds all B* cubes in one folder (BMHpM nested).
Clustering must stay per cube, matching the endpoint ``endpoint_changepoints_B*``
layout. Groups can be added incrementally: existing ``gsa`` (or other) tables
are kept and merged rather than overwritten.
"""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src.utils.run_log import log_event, step

KNOWN_CUBES: tuple[str, ...] = (
    "BHHpH",
    "BHHpM",
    "BMHpH",
    "BMHpM",
    "BMMpH",
    "BMMpM",
)


def cube_from_features_name(name: str) -> Optional[str]:
    """Return the B* cube prefix for a ``*_gsa_features.csv`` filename."""
    stem = Path(name).name
    for cube in sorted(KNOWN_CUBES, key=len, reverse=True):
        if stem.startswith(f"{cube}_"):
            return cube
    return None


def discover_gsa_feature_csvs_by_cube(
    features_root: Path | str,
    *,
    cubes: Optional[Sequence[str]] = None,
) -> dict[str, list[Path]]:
    """Recursively find ``*_gsa_features.csv`` files grouped by cube prefix."""
    features_root = Path(features_root)
    wanted = tuple(cubes) if cubes else KNOWN_CUBES
    wanted_set = set(wanted)
    by_cube: dict[str, list[Path]] = {c: [] for c in wanted}
    if not features_root.is_dir():
        return {c: [] for c in wanted}

    for path in features_root.rglob("*_gsa_features.csv"):
        cube = cube_from_features_name(path.name)
        if cube is None or cube not in wanted_set:
            continue
        by_cube[cube].append(path)

    return {c: sorted(v) for c, v in by_cube.items() if v}


def stage_cube_feature_dir(
    csv_paths: Sequence[Path],
    dest_dir: Path | str,
) -> Path:
    """Hard-link (or copy) feature CSVs into a cube-only directory."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in csv_paths:
        src = Path(src)
        dst = dest_dir / src.name
        if dst.exists() or dst.is_symlink():
            continue
        try:
            os.link(src.resolve(), dst)
        except OSError:
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
    return dest_dir


def gsa_changepoints_dir(output_root: Path | str, cube: str) -> Path:
    return Path(output_root) / f"gsa_changepoints_{cube}"


def cube_is_clustered(changepoints_dir: Path | str, group: str = "gsa") -> bool:
    clustered = Path(changepoints_dir) / "clusters" / group / "segments_clustered.csv"
    return clustered.is_file()


def missing_groups(
    changepoints_dir: Path | str,
    groups: Sequence[str],
) -> list[str]:
    """Groups in *groups* that do not yet have clustered segments."""
    return [g for g in groups if not cube_is_clustered(changepoints_dir, g)]


def merge_tables_by_group(
    old: pd.DataFrame,
    new: pd.DataFrame,
    groups: Sequence[str],
    *,
    group_col: str = "group",
) -> pd.DataFrame:
    """Keep rows whose group is not being replaced; append *new*."""
    if new is None or new.empty:
        return old if old is not None else pd.DataFrame()
    if old is None or old.empty:
        return new
    if group_col not in old.columns:
        return pd.concat([old, new], ignore_index=True)
    wanted = {str(g) for g in groups}
    keep = old[~old[group_col].astype(str).isin(wanted)]
    return pd.concat([keep, new], ignore_index=True)


def _read_csv(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


def _run_one_cube_payload(payload: dict) -> dict:
    """Process-pool worker: detect + cluster requested groups; merge with existing."""
    from src.ChangepointAnalysis import (
        ChangepointConfig,
        SegmentClusteringConfig,
        compare_changepoint_timing,
        detect_cohort_changepoints,
        discover_feature_csvs,
        write_changepoint_tables,
    )
    from src.ChangepointAnalysis.segment_clustering import cluster_all_groups

    cube = payload["cube"]
    features_dir = Path(payload["features_dir"])
    output_dir = Path(payload["output_dir"])
    groups = tuple(payload["groups"])
    n_clusters = int(payload["n_clusters"])
    skip_pca = bool(payload["skip_pca"])
    force = bool(payload.get("force", False))

    to_run = list(groups) if force else missing_groups(output_dir, groups)
    if not to_run:
        return {
            "cube": cube,
            "ok": True,
            "n_csv": 0,
            "groups": [],
            "skipped": True,
            "output_dir": str(output_dir),
        }

    csv_paths = discover_feature_csvs(features_dir)
    detection = ChangepointConfig(groups=tuple(to_run))
    tables = detect_cohort_changepoints(csv_paths, detection, output_dir=None)

    old_bkp = _read_csv(output_dir / "all_breakpoints.csv")
    old_seg = _read_csv(output_dir / "all_segment_stats.csv")
    merged_bkp = merge_tables_by_group(old_bkp, tables.breakpoints, to_run)
    merged_seg = merge_tables_by_group(old_seg, tables.segment_stats, to_run)
    tables.breakpoints = merged_bkp
    tables.segment_stats = merged_seg
    output_dir.mkdir(parents=True, exist_ok=True)
    write_changepoint_tables(tables, output_dir, per_trajectory=True)
    if merged_bkp["group"].nunique() >= 2:
        compare_changepoint_timing(merged_bkp, output_dir=output_dir)

    clustering = SegmentClusteringConfig(
        groups=tuple(to_run),
        n_clusters=n_clusters,
        skip_pca=skip_pca,
    )
    clusters_dir = output_dir / "clusters"
    existing_all = _read_csv(clusters_dir / "all_segments_clustered.csv")
    new_clustered = cluster_all_groups(merged_seg, clusters_dir, clustering)
    if not new_clustered.empty:
        merged_cl = merge_tables_by_group(existing_all, new_clustered, to_run)
        merged_cl.to_csv(clusters_dir / "all_segments_clustered.csv", index=False)

    return {
        "cube": cube,
        "ok": True,
        "n_csv": len(csv_paths),
        "groups": list(to_run),
        "skipped": False,
        "output_dir": str(output_dir),
    }


def run_gsa_step1_cohorts(
    features_root: Path | str,
    output_root: Path | str,
    *,
    cubes: Optional[Sequence[str]] = None,
    groups: Sequence[str] = ("gsa",),
    n_clusters: int = 5,
    stage_root: Optional[Path | str] = None,
    skip_if_clustered: bool = True,
    force: bool = False,
    skip_pca: bool = True,
    skip_summarize: bool = True,
    workers: int = 1,
) -> list[Path]:
    """Stage per-cube feature dirs and run changepoint+clustering.

    Already-clustered groups are skipped unless *force* is set, so iodine /
    na_water / combined can be added on top of an existing ``gsa`` run.

    Returns the list of ``gsa_changepoints_{cube}`` directories (including
    cubes that were already clustered and skipped).
    """
    del skip_summarize  # detect+cluster only; kept for call-site compatibility
    features_root = Path(features_root)
    output_root = Path(output_root)
    stage_root = Path(stage_root) if stage_root is not None else output_root / "gsa_features_by_cube"
    by_cube = discover_gsa_feature_csvs_by_cube(features_root, cubes=cubes)
    if not by_cube:
        raise FileNotFoundError(
            f"No *_gsa_features.csv files matching requested cubes in {features_root}"
        )

    result_dirs: list[Path] = []
    jobs: list[dict] = []
    for cube, csvs in by_cube.items():
        out_dir = gsa_changepoints_dir(output_root, cube)
        result_dirs.append(out_dir)
        need = list(groups) if force else missing_groups(out_dir, groups)
        if not need and skip_if_clustered:
            log_event(
                "info",
                f"{cube}: skipping detect/cluster (already have {tuple(groups)})",
                component="gsa_cohort_run",
            )
            continue
        staged = stage_cube_feature_dir(csvs, stage_root / cube)
        jobs.append(
            {
                "cube": cube,
                "features_dir": str(staged),
                "output_dir": str(out_dir),
                "groups": list(groups),
                "n_clusters": n_clusters,
                "skip_pca": skip_pca,
                "force": force,
            }
        )
        log_event(
            "info",
            f"{cube}: staged {len(csvs)} CSVs → {staged}; "
            f"groups {need}; output {out_dir}",
            component="gsa_cohort_run",
        )

    if not jobs:
        return result_dirs

    n_workers = max(1, min(int(workers), len(jobs)))
    if n_workers == 1:
        for job in jobs:
            with step(f"gsa-changepoints cube={job['cube']}"):
                _run_one_cube_payload(job)
        return result_dirs

    log_event(
        "info",
        f"Running {len(jobs)} cube pipeline(s) with {n_workers} worker(s)",
        component="gsa_cohort_run",
    )
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_one_cube_payload, job): job["cube"] for job in jobs}
        for fut in as_completed(futures):
            cube = futures[fut]
            result = fut.result()
            log_event(
                "info",
                f"{cube}: finished n_csv={result.get('n_csv')} "
                f"groups={result.get('groups')}",
                component="gsa_cohort_run",
            )
    return result_dirs
