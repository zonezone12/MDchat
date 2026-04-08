"""Frame selection skill wrapping FrameSelection.select_meaningful_frames."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class SelectFramesSkill(Skill):
    name = "select_frames"
    description = (
        "Automatically select the most scientifically meaningful frames from "
        "the trajectory based on clustering medoids, change points, frame "
        "scores, and (optionally) endpoint extrema. Returns a curated list "
        "of frame indices suitable for detailed inspection or PDB export."
    )
    category = "selection"
    parameters = [
        Parameter("max_frames", ParamType.INTEGER,
                  "Maximum number of frames to select.",
                  required=False, default=30, min_value=1, max_value=500),
    ]
    requires = ["pca_scores"]
    produces = ["selected_frame_indices"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        from src.FrameSelection import FrameSelection

        max_frames = params.get("max_frames", 30)

        pcs = context.get("pca_scores")
        medoids = context.get("medoid_indices")
        cpd_idx = context.get("change_point_indices")
        labels = context.get("cluster_labels")
        rmsd = context.get("rmsd_array")
        rg = context.get("rg_array")
        strain = context.get("strain_array")
        endpoint_dists = context.get("endpoint_distances")
        endpoint_metrics_df = context.get("endpoint_metrics_df")

        fs = FrameSelection()

        scores = None
        if rmsd is not None and rg is not None:
            scores, _ = fs.score_frames(
                rmsd, rg, pcs,
                cpd_idx=cpd_idx or [],
                strain=strain,
            )

        if medoids is not None:
            medoids_arr = np.asarray(medoids)
        else:
            medoids_arr = None

        selected, score_df = fs.select_meaningful_frames(
            medoids=medoids_arr,
            cpd_idx=cpd_idx,
            scores=scores,
            pcs=pcs,
            max_frames=max_frames,
            endpoint_dists_array=endpoint_dists,
            endpoint_metrics_df=endpoint_metrics_df,
        )

        context.set("selected_frame_indices", selected)

        summary = (
            f"Selected {len(selected)} meaningful frames "
            f"(max_frames={max_frames}).\n"
            f"Frame indices: {selected[:20]}"
        )
        if len(selected) > 20:
            summary += f"... ({len(selected) - 20} more)"

        return SkillResult(
            success=True,
            data={"selected_frame_indices": selected},
            summary=summary,
        )


get_default_registry().register(SelectFramesSkill())
