"""Orchestrator for the GSA feature-group changepoint analysis pipeline."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Optional

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
from .reporting import summarize_changepoint_results
from .segment_clustering import (
    SegmentClusteringConfig,
    cluster_all_groups,
    load_segment_stats,
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

        self._tables: Optional[ChangepointTables] = None
        self._sweep_result: Optional[PenaltySweepResult] = None
        self._clustered: Optional[pd.DataFrame] = None
        self._artifacts: dict[str, Path] = {}

    def discover_feature_csvs(self) -> list[Path]:
        """Return sorted ``*_gsa_features.csv`` paths under ``features_dir``."""
        return discover_feature_csvs(self.features_dir)

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
                f"No *_gsa_features.csv files found in {self.features_dir}"
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
                f"No *_gsa_features.csv files found in {self.features_dir}"
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
        )
        self._artifacts.update(artifacts)
        return artifacts

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
