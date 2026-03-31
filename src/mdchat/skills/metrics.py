"""Trajectory metrics skill (RMSD)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class ComputeRMSDSkill(Skill):
    name = "compute_rmsd"
    description = (
        "Compute the Root Mean Square Deviation (RMSD) of selected atoms over "
        "the trajectory relative to a reference frame. RMSD measures how much "
        "the structure deviates from the reference — increasing RMSD indicates "
        "conformational change."
    )
    category = "metrics"
    parameters = [
        Parameter("selection", ParamType.ATOM_SELECTION,
                  "MDAnalysis atom selection string (e.g., 'protein', 'resname GSA', "
                  "'backbone'). Defaults to all non-solvent heavy atoms.",
                  required=False, default="not water and not name H*"),
        Parameter("ref_frame", ParamType.INTEGER,
                  "Reference frame index for RMSD calculation (0-based).",
                  required=False, default=0, min_value=0),
    ]
    requires = ["universe"]
    produces = ["rmsd_array", "rmsd_summary"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        from src.TrajectoryMetrics import TrajectoryMetrics

        u = context.universe
        sel_str = params.get("selection", "not water and not name H*")
        ref_frame = params.get("ref_frame", 0)

        tm = TrajectoryMetrics()
        rmsd = tm.compute_rmsd(u, sel_str, ref_frame=ref_frame)

        context.set("rmsd_array", rmsd)

        mean_rmsd = float(np.nanmean(rmsd))
        max_rmsd = float(np.nanmax(rmsd))
        final_rmsd = float(rmsd[-1]) if len(rmsd) > 0 else 0.0

        summary_text = (
            f"RMSD computed for selection '{sel_str}' ({len(rmsd)} frames). "
            f"Mean: {mean_rmsd:.2f} A, Max: {max_rmsd:.2f} A, "
            f"Final: {final_rmsd:.2f} A. "
        )
        if max_rmsd > 5.0:
            summary_text += "Large deviations detected — significant conformational change."
        elif max_rmsd < 2.0:
            summary_text += "Structure remains relatively stable throughout."
        else:
            summary_text += "Moderate conformational changes observed."

        context.set("rmsd_summary", summary_text)

        return SkillResult(
            success=True,
            data={
                "rmsd_array": rmsd,
                "rmsd_summary": summary_text,
            },
            summary=summary_text,
        )


get_default_registry().register(ComputeRMSDSkill())
