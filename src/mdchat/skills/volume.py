"""Volume analysis skill wrapping VolumeAnalyzer."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class ComputeVolumeSkill(Skill):
    name = "compute_volume"
    description = (
        "Compute the molecular volume (and internal cavity volume) over the "
        "trajectory using a voxel-based approach. Tracks how the enclosed "
        "volume changes over time — useful for cage molecules, channels, or "
        "binding pockets."
    )
    category = "volume"
    parameters = [
        Parameter("selection", ParamType.ATOM_SELECTION,
                  "MDAnalysis atom selection defining the molecule whose volume "
                  "to compute (e.g., 'resname GSA', 'protein', 'resname MOF', "
                  "'all'). If omitted, uses the session main selection.",
                  required=False, default=None),
        Parameter("spacing", ParamType.FLOAT,
                  "Grid spacing in Angstrom (smaller = more accurate but slower).",
                  required=False, default=1.0, min_value=0.3, max_value=5.0),
        Parameter("probe_radius", ParamType.FLOAT,
                  "Probe radius in Angstrom for cavity detection.",
                  required=False, default=1.4, min_value=0.5, max_value=5.0),
        Parameter("stride", ParamType.INTEGER,
                  "Analyze every N-th frame to speed up calculation.",
                  required=False, default=1, min_value=1),
        Parameter("save_csv", ParamType.BOOLEAN,
                  "Save volume time series as CSV.",
                  required=False, default=True),
    ]
    requires = ["universe"]
    produces = ["volume_array", "cavity_volume_array"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        from src.VolumeAnalyzer import VolumeAnalyzer

        u = context.universe
        selection = params.get("selection") or context.main_selection
        spacing = params.get("spacing", 1.0)
        probe_radius = params.get("probe_radius", 1.4)
        stride = params.get("stride", 1)
        save_csv = params.get("save_csv", True)

        va = VolumeAnalyzer(
            u, selection=selection,
            spacing=spacing, probe_radius=probe_radius,
        )
        results = va.analyze_trajectory(stride=stride)

        target_vols = np.array([r["target_volume"] for r in results])
        cavity_vols = np.array([r["cavity_volume"] for r in results])
        frames = np.array([r["frame"] for r in results])

        context.set("volume_array", target_vols)
        context.set("volume", target_vols)
        context.set("cavity_volume_array", cavity_vols)

        artifacts = {}
        if save_csv:
            import pandas as pd
            df = pd.DataFrame({
                "frame": frames,
                "target_volume": target_vols,
                "cavity_volume": cavity_vols,
            })
            csv_path = os.path.join(context.output_dir, "volume_timeseries.csv")
            df.to_csv(csv_path, index=False)
            artifacts["volume_csv"] = csv_path

        mean_v = float(np.nanmean(target_vols))
        std_v = float(np.nanstd(target_vols))
        mean_c = float(np.nanmean(cavity_vols))

        summary = (
            f"Volume computed for '{selection}' ({len(results)} frames, "
            f"stride={stride}). "
            f"Target volume: mean={mean_v:.0f} A^3, std={std_v:.0f} A^3. "
            f"Cavity volume: mean={mean_c:.0f} A^3. "
        )
        rel_fluct = std_v / mean_v if mean_v > 0 else 0
        if rel_fluct > 0.1:
            summary += "Significant volume fluctuations — possible structural transitions."
        elif rel_fluct > 0.03:
            summary += "Moderate breathing motions detected."
        else:
            summary += "Volume is stable throughout the trajectory."

        return SkillResult(
            success=True,
            data={
                "volume_array": target_vols,
                "cavity_volume_array": cavity_vols,
            },
            artifacts=artifacts,
            summary=summary,
        )


get_default_registry().register(ComputeVolumeSkill())
