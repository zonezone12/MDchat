"""Simulation quality scoring skill."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class ScoreSimulationSkill(Skill):
    name = "score_simulation"
    description = (
        "Score the overall quality of an MD simulation on a 0-to-1 scale. "
        "Evaluates guest entry/exit behavior, volume dynamics, endpoint-volume "
        "correlations, and structural dynamics. Higher scores indicate more "
        "scientifically interesting simulations with meaningful events."
    )
    category = "scoring"
    parameters = [
        Parameter("min_volume_change_pct", ParamType.FLOAT,
                  "Minimum volume change (%) to consider meaningful.",
                  required=False, default=10.0, min_value=0, max_value=100),
        Parameter("min_correlation", ParamType.FLOAT,
                  "Minimum |r| for significant endpoint-volume correlation.",
                  required=False, default=0.5, min_value=0, max_value=1),
        Parameter("save_csv", ParamType.BOOLEAN,
                  "Whether to save score details as CSV.",
                  required=False, default=True),
    ]
    requires = ["universe"]
    produces = ["simulation_score", "score_breakdown"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        from src.FrameSelection import FrameSelection

        min_vol = params.get("min_volume_change_pct", 10.0)
        min_corr = params.get("min_correlation", 0.5)
        save_csv = params.get("save_csv", True)

        fs = FrameSelection()

        guest_stats = context.get("guest_stats")
        volume = context.get("volume")
        correlation_df = context.get("correlation_df")
        endpoint_dists = context.get("endpoint_distances")
        endpoint_metrics_df = context.get("endpoint_metrics_df")
        cube_metrics_df = context.get("cube_metrics_df")

        score, details = fs.score_simulation(
            guest_stats=guest_stats,
            volume=volume,
            correlation_df=correlation_df,
            endpoint_dists_array=endpoint_dists,
            endpoint_metrics_df=endpoint_metrics_df,
            cube_metrics_df=cube_metrics_df,
            min_volume_change_pct=min_vol,
            min_correlation=min_corr,
        )

        context.set("simulation_score", score)
        context.set("score_breakdown", details)

        summary_text = fs.get_simulation_score_summary()

        artifacts = {}
        if save_csv:
            csv_path = os.path.join(context.output_dir, "simulation_score.csv")
            fs.save_simulation_score_csv(csv_path)
            artifacts["score_csv"] = csv_path

        return SkillResult(
            success=True,
            data={
                "simulation_score": score,
                "score_breakdown": details,
            },
            artifacts=artifacts,
            summary=summary_text,
        )


get_default_registry().register(ScoreSimulationSkill())
