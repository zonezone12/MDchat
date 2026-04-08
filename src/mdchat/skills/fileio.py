"""Frame export skill wrapping FileIO."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class ExportFramesSkill(Skill):
    name = "export_frames"
    description = (
        "Export specific trajectory frames as PDB files. Useful for "
        "visualizing representative structures, cluster medoids, or frames "
        "at key events (e.g., guest entry, maximum RMSD)."
    )
    category = "fileio"
    parameters = [
        Parameter("frame_indices", ParamType.ARRAY,
                  "List of frame indices to export (0-based). "
                  "If omitted, exports medoid frames from clustering.",
                  required=False, items_type=ParamType.INTEGER),
        Parameter("selection", ParamType.ATOM_SELECTION,
                  "Atom selection to include in exported PDBs. "
                  "If omitted, all atoms are exported.",
                  required=False, default=None),
        Parameter("prefix", ParamType.STRING,
                  "Output filename prefix.",
                  required=False, default="exported"),
    ]
    requires = ["universe"]
    produces = []

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        from src.FileIO import FileIO

        u = context.universe
        frame_indices = params.get("frame_indices")
        selection = params.get("selection")
        prefix = params.get("prefix", "exported")

        if not frame_indices:
            medoids = context.get("medoid_indices")
            if medoids is not None:
                import numpy as np
                frame_indices = medoids.tolist() if hasattr(medoids, 'tolist') else list(medoids)
            else:
                return SkillResult(
                    success=False,
                    error="No frame_indices provided and no medoid_indices in context. "
                          "Provide frame indices or run cluster_frames first.",
                    summary="Cannot export — no frame indices specified.",
                )

        n_frames = u.trajectory.n_frames
        invalid = [i for i in frame_indices if i < 0 or i >= n_frames]
        if invalid:
            return SkillResult(
                success=False,
                error=f"Frame indices out of range (0–{n_frames-1}): {invalid}",
                summary=f"Invalid frame indices: {invalid}",
            )

        out_prefix = os.path.join(context.output_dir, prefix)
        fio = FileIO()
        fio.save_frames_as_pdb(u, frame_indices, out_prefix, sel=selection)

        out_dir = f"{out_prefix}_frames"
        n_exported = len(frame_indices)

        return SkillResult(
            success=True,
            artifacts={"frames_directory": out_dir},
            summary=(
                f"Exported {n_exported} frames as PDB files to {out_dir}/. "
                f"Frames: {frame_indices[:10]}"
                f"{'...' if n_exported > 10 else ''}"
            ),
        )


get_default_registry().register(ExportFramesSkill())
