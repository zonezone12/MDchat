"""Orchestrator for the GSA feature-group changepoint analysis pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from .detection import (
    ChangepointConfig,
    ChangepointTables,
    detect_cohort_changepoints,
    discover_feature_csvs,
    write_changepoint_tables,
)
from .penalty_sweep import (
    PenaltySweepConfig,
    PenaltySweepResult,
    run_final_detection,
    sweep_penalties,
)
from .reporting import (
    compare_endpoint_clusters_to_deformation,
    summarize_changepoint_results,
)
from .segment_clustering import (
    SegmentClusteringConfig,
    cluster_all_groups,
    load_segment_stats,
)
from .transition_attribution import (
    EndpointTransitionAttributionConfig,
    attribute_endpoint_transitions,
    list_site_pair_columns,
)


class ChangepointPipeline:
    """Chain detect → (optional sweep) → cluster → summarize stages.

    Stages reuse in-memory tables when available and fall back to reading
    from ``output_dir`` so each CLI can still run independently.
    """

    def __init__(
        self,
        features_dir: Path | str,
        output_dir: Path | str,
        *,
        detection: Optional[ChangepointConfig] = None,
        sweep: Optional[PenaltySweepConfig] = None,
        clustering: Optional[SegmentClusteringConfig] = None,
        sweep_output_dir: Optional[Path | str] = None,
        clusters_output_dir: Optional[Path | str] = None,
        plot_dir: Optional[Path | str] = None,
        features_suffix: str = "_gsa_features.csv",
    ) -> None:
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.detection = detection or ChangepointConfig()
        self.sweep_config = sweep
        self.clustering = clustering or SegmentClusteringConfig()
        self.sweep_output_dir = (
            Path(sweep_output_dir)
            if sweep_output_dir is not None
            else self.output_dir / "penalty_sweep"
        )
        self.clusters_output_dir = (
            Path(clusters_output_dir)
            if clusters_output_dir is not None
            else self.output_dir / "clusters"
        )
        self.plot_dir = (
            Path(plot_dir) if plot_dir is not None else self.output_dir / "plots"
        )
        self.features_suffix = features_suffix

        self._tables: Optional[ChangepointTables] = None
        self._sweep_result: Optional[PenaltySweepResult] = None
        self._clustered: Optional[pd.DataFrame] = None
        self._artifacts: dict[str, Path] = {}

    def discover_feature_csvs(self) -> list[Path]:
        """Return sorted feature-CSV paths under ``features_dir``."""
        return discover_feature_csvs(self.features_dir, suffix=self.features_suffix)

    def sweep_penalty(self) -> PenaltySweepResult:
        """Optional stage: sweep penalties and set ``detection.penalty``."""
        cfg = self.sweep_config or PenaltySweepConfig(detection=self.detection)
        # Keep detection settings in sync with the sweep's nested config.
        if self.sweep_config is None:
            cfg = replace(cfg, detection=self.detection)
        else:
            # Prefer explicit detection overrides already on the pipeline.
            cfg = replace(
                cfg,
                detection=replace(
                    cfg.detection,
                    method=self.detection.method,
                    cost_model=self.detection.cost_model,
                    min_size=self.detection.min_size,
                    jump=self.detection.jump,
                    tolerance_frames=self.detection.tolerance_frames,
                    normalize=self.detection.normalize,
                    groups=self.detection.groups,
                ),
            )

        csv_paths = self.discover_feature_csvs()
        if not csv_paths:
            raise FileNotFoundError(
                f"No *{self.features_suffix} files found in {self.features_dir}"
            )

        self.sweep_output_dir.mkdir(parents=True, exist_ok=True)
        result = sweep_penalties(csv_paths, cfg, output_dir=self.sweep_output_dir)
        self._sweep_result = result
        if result.recommended_penalty is not None:
            self.detection = replace(
                self.detection, penalty=result.recommended_penalty, n_bkps=None
            )
        return result

    def detect(self) -> ChangepointTables:
        """Run cohort changepoint detection and write tables under ``output_dir``."""
        csv_paths = self.discover_feature_csvs()
        if not csv_paths:
            raise FileNotFoundError(
                f"No *{self.features_suffix} files found in {self.features_dir}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tables = detect_cohort_changepoints(
            csv_paths, self.detection, output_dir=self.output_dir
        )
        self._tables = tables
        return tables

    def cluster_segments(self) -> pd.DataFrame:
        """Cluster changepoint segments; prefer in-memory segment stats."""
        if self._tables is not None and not self._tables.segment_stats.empty:
            df = self._tables.segment_stats
        else:
            df = load_segment_stats(self.output_dir)

        self.clusters_output_dir.mkdir(parents=True, exist_ok=True)
        clustered = cluster_all_groups(df, self.clusters_output_dir, self.clustering)
        self._clustered = clustered
        return clustered

    def summarize(
        self,
        *,
        trajectories: Optional[list[str]] = None,
        skip_tables: bool = False,
        skip_individual_timelines: bool = False,
        cluster_representatives_csv: Optional[list[str | Path]] = None,
        skip_cluster_rep_timelines: bool = False,
    ) -> dict[str, Path]:
        """Write cohort summary CSVs and plots."""
        # Ensure detection outputs exist on disk for the reporting stage.
        required = [
            self.output_dir / "all_breakpoints.csv",
            self.output_dir / "all_segment_stats.csv",
            self.output_dir / "changepoint_timing_comparison.csv",
        ]
        if self._tables is not None and any(not p.exists() for p in required):
            write_changepoint_tables(self._tables, self.output_dir)

        artifacts = summarize_changepoint_results(
            self.output_dir,
            self.features_dir,
            plot_dir=self.plot_dir,
            trajectories=trajectories,
            skip_tables=skip_tables,
            skip_individual_timelines=skip_individual_timelines,
            cluster_representatives_csv=cluster_representatives_csv,
            skip_cluster_rep_timelines=skip_cluster_rep_timelines,
            features_suffix=self.features_suffix,
        )
        self._artifacts.update(artifacts)
        return artifacts

    def attribute_endpoint_transitions(
        self,
        *,
        config: Optional[EndpointTransitionAttributionConfig] = None,
        sites_df: Optional[pd.DataFrame] = None,
        d1_atoms_df: Optional[pd.DataFrame] = None,
        universe: Any = None,
        monomer_selections: Optional[Sequence[str]] = None,
        stored_sites: Optional[Sequence[Sequence[Sequence[int]]]] = None,
    ) -> dict[str, Path]:
        """Attribute directed endpoint-cluster transitions to site-pair features.

        Prefers in-memory ``self._clustered``; falls back to
        ``clusters/endpoint/segments_clustered.csv``. Skips with a warning when
        no raw site-pair or paper-d1 columns are present in the feature CSVs.
        """
        from src.utils.run_log import log_event

        from .endpoint_features import list_paper_d1_columns
        from .transition_attribution import list_attribution_feature_columns

        cfg = config or EndpointTransitionAttributionConfig()
        # Guard: attribution features must exist.
        csv_paths = self.discover_feature_csvs()
        has_feats = False
        for path in csv_paths:
            try:
                header = pd.read_csv(path, nrows=0)
            except Exception:
                continue
            cols = list_attribution_feature_columns(
                header.columns, include_paper_d1=cfg.include_paper_d1
            )
            if cols:
                has_feats = True
                break
        if not has_feats:
            log_event(
                "warning",
                (
                    "Skipping endpoint transition attribution: no "
                    "endpoint_dist_{i}s{a}_{j}s{b} or paper_d1_* columns found. "
                    "Re-run with --include-site-pairs and/or paper d1 enabled."
                ),
                component="endpoint_transition_attribution",
            )
            return {}

        written = attribute_endpoint_transitions(
            self.output_dir,
            self.features_dir,
            config=cfg,
            clustered_df=self._clustered,
            sites_df=sites_df,
            d1_atoms_df=d1_atoms_df,
            output_dir=self.output_dir,
            plot_dir=self.plot_dir,
            universe=universe,
            monomer_selections=monomer_selections,
            stored_sites=stored_sites,
        )
        self._artifacts.update(written)
        return written

    def run_all(
        self,
        *,
        with_sweep: bool = False,
        run_final_after_sweep: bool = True,
        skip_clustering: bool = False,
        skip_summarize: bool = False,
    ) -> dict[str, Path]:
        """Run the full pipeline and return a flat artifact map."""
        artifacts: dict[str, Path] = {}

        if with_sweep:
            sweep_result = self.sweep_penalty()
            if (
                run_final_after_sweep
                and sweep_result.recommended_penalty is not None
            ):
                csv_paths = self.discover_feature_csvs()
                tables = run_final_detection(
                    csv_paths,
                    penalty=sweep_result.recommended_penalty,
                    detection=self.detection,
                    output_dir=self.output_dir,
                )
                self._tables = tables
            elif not run_final_after_sweep:
                # Detect with whatever penalty is currently on self.detection.
                self.detect()
        else:
            self.detect()

        if not skip_clustering:
            self.cluster_segments()

        if not skip_summarize:
            artifacts.update(self.summarize())

        # Collect primary detection outputs.
        for name in (
            "all_breakpoints.csv",
            "all_segment_stats.csv",
            "changepoint_timing_comparison.csv",
        ):
            path = self.output_dir / name
            if path.exists():
                artifacts[name] = path

        self._artifacts.update(artifacts)
        return artifacts


def run_endpoint_changepoint(
    topology: str | Path,
    trajectories: Sequence[str | Path],
    *,
    output_dir: str | Path = "output/endpoint_changepoints",
    features_dir: Optional[str | Path] = None,
    gsa_resname: str = "MOL",
    n_monomers: int = 6,
    monomer_selections: Optional[list[str]] = None,
    use_ring_centroids: bool = True,
    ring_min_gap_deg: Optional[float] = 35.0,
    ring_max_per_ring: int = 3,
    include_site_pairs: bool = False,
    include_paper_d1: bool = True,
    paper_d1_open_lo: float = 4.5,
    paper_d1_open_hi: float = 5.5,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: int = 1,
    time_per_frame_ps: float = 1.0,
    traj_ids: Optional[Sequence[str]] = None,
    detection: Optional[ChangepointConfig] = None,
    clustering: Optional[SegmentClusteringConfig] = None,
    with_sweep: bool = False,
    sweep: Optional[PenaltySweepConfig] = None,
    skip_clustering: bool = False,
    skip_summarize: bool = False,
    skip_transition_attribution: bool = False,
    transition_attribution: Optional[EndpointTransitionAttributionConfig] = None,
    rmsd_from: Optional[str | Path] = None,
    universe_factory: Optional[Any] = None,
) -> dict[str, Path]:
    """Generate endpoint-site features then run the changepoint pipeline.

    Parameters
    ----------
    topology, trajectories
        Shared topology and one or more trajectory paths.
    rmsd_from
        Optional directory of ``*_gsa_features.csv`` used to overlay
        ``assembly_rmsd_to_ref`` on the deformation comparison.
    skip_transition_attribution
        If False (default), after clustering attribute directed cluster
        transitions to raw site-pair columns. Requires ``include_site_pairs``.
    transition_attribution
        Optional config for top-N / correlation threshold / ranking weights.
    """
    import MDAnalysis as mda

    from .endpoint_features import (
        EndpointFeatureConfig,
        generate_endpoint_features,
        write_endpoint_features_csv,
    )

    output_dir = Path(output_dir)
    features_dir = Path(features_dir) if features_dir else output_dir / "endpoint_features"
    features_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_paths = [Path(p) for p in trajectories]
    if traj_ids is None:
        ids = [p.stem for p in traj_paths]
    else:
        ids = list(traj_ids)
        if len(ids) != len(traj_paths):
            raise ValueError("traj_ids length must match trajectories")

    feat_cfg = EndpointFeatureConfig(
        gsa_resname=gsa_resname,
        n_monomers=n_monomers,
        monomer_selections=monomer_selections,
        use_ring_centroids=use_ring_centroids,
        ring_min_gap_deg=ring_min_gap_deg,
        ring_max_per_ring=ring_max_per_ring,
        include_site_pairs=include_site_pairs,
        include_paper_d1=include_paper_d1,
        paper_d1_open_lo=paper_d1_open_lo,
        paper_d1_open_hi=paper_d1_open_hi,
        start=start,
        stop=stop,
        step=step,
        time_per_frame_ps=time_per_frame_ps,
    )

    last_sites_df = None
    last_d1_atoms_df = None
    last_universe = None
    last_monomer_sels: Optional[list[str]] = None
    last_stored_sites = None

    for i, (traj_path, traj_id) in enumerate(zip(traj_paths, ids)):
        if universe_factory is not None:
            u = universe_factory(topology, traj_path)
        else:
            u = mda.Universe(str(topology), str(traj_path))
        features_df, sites_df, monomer_sels, stored_sites, d1_atoms_df = (
            generate_endpoint_features(u, feat_cfg, traj_id=traj_id)
        )
        # Sites are topology-defined (prmtop); write the QC PNG once.
        write_endpoint_features_csv(
            features_df,
            features_dir,
            traj_id,
            sites_df=sites_df,
            d1_atoms_df=d1_atoms_df,
            universe=u,
            monomer_selections=monomer_sels,
            stored_sites=stored_sites,
            write_site_plot=(i == 0),
        )
        if i == 0:
            last_sites_df = sites_df
            last_d1_atoms_df = d1_atoms_df
            last_universe = u
            last_monomer_sels = list(monomer_sels)
            last_stored_sites = stored_sites

    det = detection or ChangepointConfig(groups=("endpoint",))
    if "endpoint" not in det.groups:
        det = replace(det, groups=("endpoint",))
    clus = clustering or SegmentClusteringConfig(groups=("endpoint",))
    if "endpoint" not in clus.groups:
        clus = replace(clus, groups=("endpoint",))

    pipe = ChangepointPipeline(
        features_dir,
        output_dir,
        detection=det,
        sweep=sweep,
        clustering=clus,
        features_suffix="_endpoint_features.csv",
    )
    artifacts = pipe.run_all(
        with_sweep=with_sweep,
        skip_clustering=skip_clustering,
        skip_summarize=skip_summarize,
    )
    artifacts["features_dir"] = features_dir

    if not skip_clustering:
        compare_arts = compare_endpoint_clusters_to_deformation(
            output_dir,
            features_dir,
            rmsd_from=Path(rmsd_from) if rmsd_from else None,
            plot_dir=pipe.plot_dir,
        )
        artifacts.update(compare_arts)

        if not skip_transition_attribution:
            attr_cfg = transition_attribution or EndpointTransitionAttributionConfig(
                include_paper_d1=include_paper_d1,
                paper_d1_open_lo=paper_d1_open_lo,
                paper_d1_open_hi=paper_d1_open_hi,
            )
            attr_arts = pipe.attribute_endpoint_transitions(
                config=attr_cfg,
                sites_df=last_sites_df,
                d1_atoms_df=last_d1_atoms_df,
                universe=last_universe,
                monomer_selections=last_monomer_sels,
                stored_sites=last_stored_sites,
            )
            artifacts.update(attr_arts)

    return artifacts
