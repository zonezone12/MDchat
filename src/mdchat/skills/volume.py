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
        "binding pockets. Optional detect_plateau_target / detect_plateau_cavity "
        "run plateau detection without calling compute_volume twice; use "
        "detect_motion_plateau to retune thresholds on cached volume only."
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
        Parameter(
            "detect_plateau_target", ParamType.BOOLEAN,
            "If True, run plateau detection on the target (enclosed) volume series.",
            required=False, default=False,
        ),
        Parameter(
            "detect_plateau_cavity", ParamType.BOOLEAN,
            "If True, run plateau detection on the cavity volume series.",
            required=False, default=False,
        ),
        Parameter(
            "plateau_window", ParamType.INTEGER,
            "Plateau: sliding window length (samples) for local standard deviation.",
            required=False, default=50, min_value=2,
        ),
        Parameter(
            "rel_std_threshold", ParamType.FLOAT,
            "Plateau: tolerance as a fraction of the global std of the series.",
            required=False, default=0.12, min_value=1e-6, max_value=1.0,
        ),
        Parameter(
            "abs_std_max", ParamType.FLOAT,
            "Plateau: optional absolute cap on window std (Å³ for volume series). "
            "Effective threshold is max(abs_std_max, rel_std_threshold * global_std).",
            required=False, default=None, min_value=0.0,
        ),
    ]
    requires = ["universe"]
    produces = [
        "volume_array",
        "cavity_volume_array",
        "plateau_detection",
        "plateau_start_frame",
        "steady_state_representative_frame",
    ]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        from src.VolumeAnalyzer import VolumeAnalyzer
        from ..plateau_helpers import apply_plateau_volume_optional

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

        target_vols = np.array([r["target_volume_A3"] for r in results])
        cavity_vols = np.array([r["cavity_volume_A3"] for r in results])
        frames = np.array([r["frame"] for r in results])

        context.set("volume_array", target_vols)
        context.set("volume", target_vols)
        context.set("cavity_volume_array", cavity_vols)
        context.set("volume_frames", frames)

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

        plateau_addon, plateau_data, plateau_art = apply_plateau_volume_optional(
            context,
            params,
            universe=u,
            target_vols=target_vols,
            cavity_vols=cavity_vols,
            frames=frames,
            selection=selection,
            save_csv=bool(save_csv),
        )
        summary = summary + plateau_addon
        artifacts.update(plateau_art)

        data = {
            "volume_array": target_vols,
            "cavity_volume_array": cavity_vols,
        }
        data.update(plateau_data)

        return SkillResult(
            success=True,
            data=data,
            artifacts=artifacts,
            summary=summary,
        )


get_default_registry().register(ComputeVolumeSkill())
