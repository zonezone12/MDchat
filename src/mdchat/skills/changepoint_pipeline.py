"""MDChat skills for the cohort-level GSA feature-group changepoint pipeline.

These operate on pre-computed ``*_gsa_features.csv`` directories (no universe required),
unlike the single-array ``detect_changepoints`` skill.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..registry import get_default_registry
from ..skill import Parameter, ParamType, Skill, SkillResult

if TYPE_CHECKING:
    from ..context import AnalysisContext

_GROUP_ENUM = ["gsa", "iodine", "na_water", "combined"]


def _output_dir(context: "AnalysisContext", override: Optional[str] = None) -> str:
    if override:
        return override
    return getattr(context, "output_dir", None) or "output/changepoints"


class ChangepointFeatureGroupsSkill(Skill):
    name = "changepoint_feature_groups"
    description = (
        "Run multivariate Pelt changepoint detection on four GSA feature groups "
        "(gsa, iodine, na_water, combined) for all *_gsa_features.csv files in a "
        "directory. Writes all_breakpoints.csv, all_segment_stats.csv, and "
        "changepoint_timing_comparison.csv. Does not require a loaded universe."
    )
    category = "changepoint"
    parameters = [
        Parameter(
            "input_dir",
            ParamType.FILE_PATH,
            "Directory containing *_gsa_features.csv files.",
            required=False,
            default="output/gsa_features",
        ),
        Parameter(
            "output_dir",
            ParamType.FILE_PATH,
            "Directory for changepoint CSV outputs.",
            required=False,
            default=None,
        ),
        Parameter(
            "method",
            ParamType.STRING,
            "Ruptures search method.",
            required=False,
            default="Pelt",
            enum_values=["Pelt", "Binseg", "BottomUp", "Window", "Dynp"],
        ),
        Parameter(
            "cost_model",
            ParamType.STRING,
            "Ruptures cost function.",
            required=False,
            default="rbf",
        ),
        Parameter(
            "penalty",
            ParamType.FLOAT,
            "Pelt penalty (omit for log(n) auto).",
            required=False,
            default=None,
            min_value=0.0,
        ),
        Parameter(
            "min_size",
            ParamType.INTEGER,
            "Minimum segment length in frames.",
            required=False,
            default=10,
            min_value=1,
        ),
        Parameter(
            "jump",
            ParamType.INTEGER,
            "Ruptures subsample step.",
            required=False,
            default=5,
            min_value=1,
        ),
        Parameter(
            "tolerance_frames",
            ParamType.INTEGER,
            "Frame tolerance for cross-group timing comparison.",
            required=False,
            default=50,
            min_value=1,
        ),
        Parameter(
            "groups",
            ParamType.ARRAY,
            "Feature groups to analyse.",
            required=False,
            default=["gsa", "iodine", "na_water", "combined"],
            items_type=ParamType.STRING,
        ),
        Parameter(
            "normalize",
            ParamType.BOOLEAN,
            "Z-score normalise features before detection.",
            required=False,
            default=True,
        ),
    ]
    requires: List[str] = []
    produces = ["changepoint_tables"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from pathlib import Path

        from src.ChangepointAnalysis import ChangepointConfig, detect_cohort_changepoints
        from src.ChangepointAnalysis.detection import discover_feature_csvs

        input_dir = Path(params.get("input_dir") or "output/gsa_features")
        output_dir = Path(_output_dir(context, params.get("output_dir")))
        groups = params.get("groups") or ["gsa", "iodine", "na_water", "combined"]
        if isinstance(groups, str):
            groups = [g.strip() for g in groups.split(",") if g.strip()]

        csv_files = discover_feature_csvs(input_dir)
        if not csv_files:
            return SkillResult(
                success=False,
                summary=f"No *_gsa_features.csv found in {input_dir}",
                error="no feature csvs",
            )

        config = ChangepointConfig(
            method=params.get("method") or "Pelt",
            cost_model=params.get("cost_model") or "rbf",
            penalty=params.get("penalty"),
            min_size=int(params.get("min_size", 10)),
            jump=int(params.get("jump", 5)),
            tolerance_frames=int(params.get("tolerance_frames", 50)),
            normalize=bool(params.get("normalize", True)),
            groups=tuple(groups),
        )
        try:
            tables = detect_cohort_changepoints(
                csv_files, config, output_dir=output_dir
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                summary=f"Changepoint detection failed: {exc}",
                error=str(exc),
            )

        artifacts = {
            "all_breakpoints": str(output_dir / "all_breakpoints.csv"),
            "all_segment_stats": str(output_dir / "all_segment_stats.csv"),
            "changepoint_timing_comparison": str(
                output_dir / "changepoint_timing_comparison.csv"
            ),
        }
        context.set("changepoint_tables", {
            "n_breakpoints": len(tables.breakpoints),
            "n_segments": len(tables.segment_stats),
            "output_dir": str(output_dir),
        })
        return SkillResult(
            success=True,
            data={"changepoint_tables": context.get("changepoint_tables")},
            artifacts=artifacts,
            summary=(
                f"Detected changepoints for {len(csv_files)} trajectories "
                f"({len(tables.breakpoints)} breakpoints, "
                f"{len(tables.segment_stats)} segments) → {output_dir}"
            ),
        )


class SweepChangepointPenaltySkill(Skill):
    name = "sweep_changepoint_penalty"
    description = (
        "Sweep Pelt penalty values on *_gsa_features.csv files and recommend "
        "the elbow of total breakpoints vs penalty. Optionally re-run detection "
        "at the recommended penalty. Does not require a loaded universe."
    )
    category = "changepoint"
    parameters = [
        Parameter(
            "input_dir",
            ParamType.FILE_PATH,
            "Directory containing *_gsa_features.csv files.",
            required=False,
            default="output/gsa_features",
        ),
        Parameter(
            "output_dir",
            ParamType.FILE_PATH,
            "Sweep summaries and plots directory.",
            required=False,
            default="output/changepoints/penalty_sweep",
        ),
        Parameter(
            "n_penalties",
            ParamType.INTEGER,
            "Number of log-spaced penalty values.",
            required=False,
            default=15,
            min_value=3,
        ),
        Parameter(
            "run_final",
            ParamType.BOOLEAN,
            "Re-run detection at the recommended penalty after the sweep.",
            required=False,
            default=False,
        ),
        Parameter(
            "final_output_dir",
            ParamType.FILE_PATH,
            "Output dir for --run_final detection.",
            required=False,
            default="output/changepoints",
        ),
        Parameter(
            "max_trajectories",
            ParamType.INTEGER,
            "Limit trajectories for a faster exploratory sweep.",
            required=False,
            default=None,
            min_value=1,
        ),
        Parameter(
            "no_plots",
            ParamType.BOOLEAN,
            "Skip writing sweep plots.",
            required=False,
            default=False,
        ),
    ]
    requires: List[str] = []
    produces = ["penalty_sweep_result"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from pathlib import Path

        from src.ChangepointAnalysis import ChangepointConfig, PenaltySweepConfig
        from src.ChangepointAnalysis.detection import discover_feature_csvs
        from src.ChangepointAnalysis.penalty_sweep import (
            run_final_detection,
            sweep_penalties,
        )

        input_dir = Path(params.get("input_dir") or "output/gsa_features")
        output_dir = Path(
            params.get("output_dir") or "output/changepoints/penalty_sweep"
        )
        csv_files = discover_feature_csvs(input_dir)
        if not csv_files:
            return SkillResult(
                success=False,
                summary=f"No *_gsa_features.csv found in {input_dir}",
                error="no feature csvs",
            )

        max_traj = params.get("max_trajectories")
        if max_traj is not None:
            csv_files = csv_files[: int(max_traj)]

        sweep_cfg = PenaltySweepConfig(
            n_penalties=int(params.get("n_penalties", 15)),
            max_trajectories=int(max_traj) if max_traj is not None else None,
            no_plots=bool(params.get("no_plots", False)),
            detection=ChangepointConfig(),
        )
        try:
            result = sweep_penalties(csv_files, sweep_cfg, output_dir=output_dir)
        except Exception as exc:
            return SkillResult(
                success=False,
                summary=f"Penalty sweep failed: {exc}",
                error=str(exc),
            )

        artifacts: Dict[str, str] = {
            "penalty_recommendation": str(output_dir / "penalty_recommendation.txt"),
            "penalty_sweep_summary": str(output_dir / "penalty_sweep_summary.csv"),
        }

        if params.get("run_final") and result.recommended_penalty is not None:
            final_dir = Path(params.get("final_output_dir") or "output/changepoints")
            run_final_detection(
                csv_files,
                penalty=result.recommended_penalty,
                detection=ChangepointConfig(),
                output_dir=final_dir,
            )
            artifacts["final_all_breakpoints"] = str(final_dir / "all_breakpoints.csv")

        payload = {
            "recommended_penalty": result.recommended_penalty,
            "recommended_range": result.recommended_range,
            "n_trajectories": result.n_trajectories,
            "output_dir": str(output_dir),
        }
        context.set("penalty_sweep_result", payload)
        return SkillResult(
            success=True,
            data={"penalty_sweep_result": payload},
            artifacts=artifacts,
            summary=(
                f"Penalty sweep on {result.n_trajectories} trajectories → "
                f"recommended penalty={result.recommended_penalty}"
            ),
        )


class ClusterChangepointSegmentsSkill(Skill):
    name = "cluster_changepoint_segments"
    description = (
        "Hierarchically cluster changepoint-defined segments from "
        "all_segment_stats.csv by feature group. Writes dendrograms, PCA, "
        "and cohort transition tables under {output_dir}/. Does not require "
        "a loaded universe."
    )
    category = "changepoint"
    parameters = [
        Parameter(
            "changepoints_dir",
            ParamType.FILE_PATH,
            "Directory with all_segment_stats.csv.",
            required=False,
            default="output/changepoints",
        ),
        Parameter(
            "output_dir",
            ParamType.FILE_PATH,
            "Cluster output directory (default: {changepoints_dir}/clusters).",
            required=False,
            default=None,
        ),
        Parameter(
            "n_clusters",
            ParamType.INTEGER,
            "Fixed number of clusters.",
            required=False,
            default=5,
            min_value=2,
        ),
        Parameter(
            "linkage",
            ParamType.STRING,
            "SciPy linkage method.",
            required=False,
            default="ward",
        ),
        Parameter(
            "groups",
            ParamType.ARRAY,
            "Feature groups to cluster.",
            required=False,
            default=["gsa", "iodine", "na_water", "combined"],
            items_type=ParamType.STRING,
        ),
        Parameter(
            "skip_pca",
            ParamType.BOOLEAN,
            "Skip PCA scatter plots.",
            required=False,
            default=False,
        ),
    ]
    requires: List[str] = []
    produces = ["changepoint_clusters"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from pathlib import Path

        from src.ChangepointAnalysis import SegmentClusteringConfig
        from src.ChangepointAnalysis.segment_clustering import (
            cluster_all_groups,
            load_segment_stats,
        )

        cp_dir = Path(params.get("changepoints_dir") or "output/changepoints")
        out_dir = Path(params.get("output_dir") or (cp_dir / "clusters"))
        groups = params.get("groups") or ["gsa", "iodine", "na_water", "combined"]
        if isinstance(groups, str):
            groups = [g.strip() for g in groups.split(",") if g.strip()]

        try:
            df = load_segment_stats(cp_dir)
        except FileNotFoundError as exc:
            return SkillResult(
                success=False,
                summary=str(exc),
                error="missing segment stats",
            )

        config = SegmentClusteringConfig(
            groups=tuple(groups),
            linkage=params.get("linkage") or "ward",
            n_clusters=int(params.get("n_clusters", 5)),
            skip_pca=bool(params.get("skip_pca", False)),
        )
        try:
            clustered = cluster_all_groups(df, out_dir, config)
        except Exception as exc:
            return SkillResult(
                success=False,
                summary=f"Segment clustering failed: {exc}",
                error=str(exc),
            )

        artifacts = {
            "all_segments_clustered": str(out_dir / "all_segments_clustered.csv"),
            "cluster_counts_by_group": str(out_dir / "cluster_counts_by_group.csv"),
        }
        payload = {
            "n_segments": len(clustered),
            "output_dir": str(out_dir),
        }
        context.set("changepoint_clusters", payload)
        return SkillResult(
            success=True,
            data={"changepoint_clusters": payload},
            artifacts=artifacts,
            summary=f"Clustered {len(clustered)} segments → {out_dir}",
        )


class SummarizeChangepointResultsSkill(Skill):
    name = "summarize_changepoint_results"
    description = (
        "Write cohort timing/regime summary CSVs and generate Jaccard heatmap, "
        "breakpoint histogram, cohort overview, and timeline plots from "
        "changepoint detection outputs. Does not require a loaded universe."
    )
    category = "changepoint"
    parameters = [
        Parameter(
            "changepoints_dir",
            ParamType.FILE_PATH,
            "Directory with changepoint CSV outputs.",
            required=False,
            default="output/changepoints",
        ),
        Parameter(
            "features_dir",
            ParamType.FILE_PATH,
            "Directory with *_gsa_features.csv for timeline traces.",
            required=False,
            default="output/gsa_features",
        ),
        Parameter(
            "plot_dir",
            ParamType.FILE_PATH,
            "Plot output directory (default: {changepoints_dir}/plots).",
            required=False,
            default=None,
        ),
        Parameter(
            "skip_tables",
            ParamType.BOOLEAN,
            "Only regenerate plots, not summary CSVs.",
            required=False,
            default=False,
        ),
        Parameter(
            "skip_individual_timelines",
            ParamType.BOOLEAN,
            "Skip per-trajectory timeline plots.",
            required=False,
            default=False,
        ),
        Parameter(
            "skip_cluster_rep_timelines",
            ParamType.BOOLEAN,
            "Skip cluster-medoid timeline plots.",
            required=False,
            default=False,
        ),
    ]
    requires: List[str] = []
    produces = ["changepoint_summary_artifacts"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from pathlib import Path

        from src.ChangepointAnalysis import summarize_changepoint_results

        cp_dir = Path(params.get("changepoints_dir") or "output/changepoints")
        features_dir = Path(params.get("features_dir") or "output/gsa_features")
        plot_dir = Path(params.get("plot_dir") or (cp_dir / "plots"))

        try:
            written = summarize_changepoint_results(
                cp_dir,
                features_dir,
                plot_dir=plot_dir,
                skip_tables=bool(params.get("skip_tables", False)),
                skip_individual_timelines=bool(
                    params.get("skip_individual_timelines", False)
                ),
                skip_cluster_rep_timelines=bool(
                    params.get("skip_cluster_rep_timelines", False)
                ),
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                summary=f"Summarize failed: {exc}",
                error=str(exc),
            )

        artifacts = {k: str(v) for k, v in written.items()}
        context.set("changepoint_summary_artifacts", artifacts)
        return SkillResult(
            success=True,
            data={"changepoint_summary_artifacts": artifacts},
            artifacts=artifacts,
            summary=f"Wrote {len(artifacts)} summary/plot artifacts under {cp_dir}",
        )


class RunChangepointPipelineSkill(Skill):
    name = "run_changepoint_pipeline"
    description = (
        "Run the full GSA feature-group changepoint pipeline: detect → "
        "(optional penalty sweep) → segment clustering → summarize. "
        "Operates on *_gsa_features.csv directories; does not require a loaded universe."
    )
    category = "changepoint"
    parameters = [
        Parameter(
            "features_dir",
            ParamType.FILE_PATH,
            "Directory with *_gsa_features.csv files.",
            required=False,
            default="output/gsa_features",
        ),
        Parameter(
            "output_dir",
            ParamType.FILE_PATH,
            "Pipeline output directory.",
            required=False,
            default="output/changepoints",
        ),
        Parameter(
            "with_sweep",
            ParamType.BOOLEAN,
            "Run penalty sweep first and use the recommended elbow penalty.",
            required=False,
            default=False,
        ),
        Parameter(
            "n_penalties",
            ParamType.INTEGER,
            "Number of log-spaced penalties when with_sweep is true.",
            required=False,
            default=15,
            min_value=3,
        ),
        Parameter(
            "penalty",
            ParamType.FLOAT,
            "Fixed Pelt penalty (ignored if with_sweep is true).",
            required=False,
            default=None,
            min_value=0.0,
        ),
        Parameter(
            "n_clusters",
            ParamType.INTEGER,
            "Number of segment clusters.",
            required=False,
            default=5,
            min_value=2,
        ),
        Parameter(
            "skip_clustering",
            ParamType.BOOLEAN,
            "Skip the segment clustering stage.",
            required=False,
            default=False,
        ),
        Parameter(
            "skip_summarize",
            ParamType.BOOLEAN,
            "Skip the summarize/plot stage.",
            required=False,
            default=False,
        ),
    ]
    requires: List[str] = []
    produces = ["changepoint_pipeline_artifacts"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from pathlib import Path

        from src.ChangepointAnalysis import (
            ChangepointConfig,
            ChangepointPipeline,
            PenaltySweepConfig,
            SegmentClusteringConfig,
        )

        features_dir = Path(params.get("features_dir") or "output/gsa_features")
        output_dir = Path(params.get("output_dir") or "output/changepoints")
        with_sweep = bool(params.get("with_sweep", False))

        detection = ChangepointConfig(penalty=params.get("penalty"))
        clustering = SegmentClusteringConfig(
            n_clusters=int(params.get("n_clusters", 5)),
        )
        sweep = None
        if with_sweep:
            sweep = PenaltySweepConfig(
                n_penalties=int(params.get("n_penalties", 15)),
                detection=detection,
            )

        pipe = ChangepointPipeline(
            features_dir,
            output_dir,
            detection=detection,
            sweep=sweep,
            clustering=clustering,
        )
        try:
            artifacts = pipe.run_all(
                with_sweep=with_sweep,
                skip_clustering=bool(params.get("skip_clustering", False)),
                skip_summarize=bool(params.get("skip_summarize", False)),
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                summary=f"Changepoint pipeline failed: {exc}",
                error=str(exc),
            )

        art_str = {k: str(v) for k, v in artifacts.items()}
        context.set("changepoint_pipeline_artifacts", art_str)
        return SkillResult(
            success=True,
            data={"changepoint_pipeline_artifacts": art_str},
            artifacts=art_str,
            summary=f"Changepoint pipeline complete → {output_dir} ({len(art_str)} artifacts)",
        )


class EndpointChangepointSkill(Skill):
    name = "run_endpoint_changepoint"
    description = (
        "Extract endpoint-site distances (ring-system centroids + atom sites) from "
        "GSA monomer selections, then run the changepoint pipeline on the endpoint "
        "feature group. Use to check whether deformation regimes recovered from "
        "endpoint distances align with the paper's A/B/C1/C2 metastructures. "
        "Optionally overlays assembly_rmsd_to_ref from a prior GSA features directory."
    )
    category = "changepoint"
    parameters = [
        Parameter(
            "topology",
            ParamType.FILE_PATH,
            "Shared topology file (prmtop, pdb, …).",
            required=True,
        ),
        Parameter(
            "trajectories",
            ParamType.ARRAY,
            "Trajectory path(s) or glob pattern(s).",
            required=True,
            items_type=ParamType.STRING,
        ),
        Parameter(
            "output_dir",
            ParamType.FILE_PATH,
            "Pipeline output directory.",
            required=False,
            default="output/endpoint_changepoints",
        ),
        Parameter(
            "features_dir",
            ParamType.FILE_PATH,
            "Where to write *_endpoint_features.csv.",
            required=False,
            default=None,
        ),
        Parameter(
            "gsa_resname",
            ParamType.STRING,
            "GSA amphiphile residue name.",
            required=False,
            default="MOL",
        ),
        Parameter(
            "n_monomers",
            ParamType.INTEGER,
            "Expected number of GSA monomers.",
            required=False,
            default=6,
            min_value=2,
        ),
        Parameter(
            "use_ring_centroids",
            ParamType.BOOLEAN,
            "Collapse fused ring systems to centroids (cation–π style).",
            required=False,
            default=True,
        ),
        Parameter(
            "include_site_pairs",
            ParamType.BOOLEAN,
            "Also emit per-site-pair distance columns.",
            required=False,
            default=False,
        ),
        Parameter(
            "step",
            ParamType.INTEGER,
            "Trajectory frame stride.",
            required=False,
            default=1,
            min_value=1,
        ),
        Parameter(
            "time_per_frame_ps",
            ParamType.FLOAT,
            "Time between consecutive trajectory frames (ps).",
            required=False,
            default=1.0,
            min_value=0.0,
        ),
        Parameter(
            "penalty",
            ParamType.FLOAT,
            "Pelt penalty (omit for auto).",
            required=False,
            default=None,
            min_value=0.0,
        ),
        Parameter(
            "with_sweep",
            ParamType.BOOLEAN,
            "Run penalty sweep first.",
            required=False,
            default=False,
        ),
        Parameter(
            "n_clusters",
            ParamType.INTEGER,
            "Number of segment clusters.",
            required=False,
            default=5,
            min_value=2,
        ),
        Parameter(
            "rmsd_from",
            ParamType.FILE_PATH,
            "Optional *_gsa_features.csv directory for RMSD overlay.",
            required=False,
            default=None,
        ),
        Parameter(
            "skip_clustering",
            ParamType.BOOLEAN,
            "Skip segment clustering.",
            required=False,
            default=False,
        ),
        Parameter(
            "skip_summarize",
            ParamType.BOOLEAN,
            "Skip summarize/plot stage.",
            required=False,
            default=False,
        ),
    ]
    requires: List[str] = []
    produces = ["endpoint_changepoint_artifacts"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from pathlib import Path

        from src.ChangepointAnalysis import (
            ChangepointConfig,
            PenaltySweepConfig,
            SegmentClusteringConfig,
        )
        from src.ChangepointAnalysis.pipeline import run_endpoint_changepoint

        topology = params.get("topology")
        trajectories = params.get("trajectories")
        if not topology or not trajectories:
            return SkillResult(
                success=False,
                summary="topology and trajectories are required",
                error="missing required parameters",
            )
        if isinstance(trajectories, str):
            trajectories = [trajectories]

        detection = ChangepointConfig(
            groups=("endpoint",),
            penalty=params.get("penalty"),
        )
        clustering = SegmentClusteringConfig(
            groups=("endpoint",),
            n_clusters=int(params.get("n_clusters", 5)),
        )
        with_sweep = bool(params.get("with_sweep", False))
        sweep = None
        if with_sweep:
            sweep = PenaltySweepConfig(detection=detection)

        try:
            artifacts = run_endpoint_changepoint(
                topology,
                trajectories,
                output_dir=params.get("output_dir") or "output/endpoint_changepoints",
                features_dir=params.get("features_dir"),
                gsa_resname=params.get("gsa_resname") or "MOL",
                n_monomers=int(params.get("n_monomers", 6)),
                use_ring_centroids=bool(params.get("use_ring_centroids", True)),
                include_site_pairs=bool(params.get("include_site_pairs", False)),
                step=int(params.get("step", 1)),
                time_per_frame_ps=float(params.get("time_per_frame_ps", 1.0)),
                detection=detection,
                clustering=clustering,
                with_sweep=with_sweep,
                sweep=sweep,
                skip_clustering=bool(params.get("skip_clustering", False)),
                skip_summarize=bool(params.get("skip_summarize", False)),
                rmsd_from=params.get("rmsd_from"),
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                summary=f"Endpoint changepoint pipeline failed: {exc}",
                error=str(exc),
            )

        art_str = {k: str(v) for k, v in artifacts.items()}
        context.set("endpoint_changepoint_artifacts", art_str)
        out = params.get("output_dir") or "output/endpoint_changepoints"
        return SkillResult(
            success=True,
            data={"endpoint_changepoint_artifacts": art_str},
            artifacts=art_str,
            summary=(
                f"Endpoint-site changepoint pipeline complete → {out} "
                f"({len(art_str)} artifacts)"
            ),
        )


_registry = get_default_registry()
_registry.register(ChangepointFeatureGroupsSkill())
_registry.register(SweepChangepointPenaltySkill())
_registry.register(ClusterChangepointSegmentsSkill())
_registry.register(SummarizeChangepointResultsSkill())
_registry.register(RunChangepointPipelineSkill())
_registry.register(EndpointChangepointSkill())
