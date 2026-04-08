"""GSA nanocube analysis skill wrapping GSAnalyzer."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class GSANanocubeMetricsSkill(Skill):
    name = "gsa_nanocube_metrics"
    description = (
        "Compute nanocube geometry metrics (face planarity, edge lengths, "
        "volume, guest-to-center distance) over the trajectory for GSA-type "
        "cage molecules. Requires face selection strings that define the "
        "cube faces."
    )
    category = "gsa"
    parameters = [
        Parameter("face_selections", ParamType.ARRAY,
                  "List of MDAnalysis selection strings for each cube face "
                  "(e.g., ['resid 1', 'resid 2', 'resid 3', 'resid 4', "
                  "'resid 5', 'resid 6']).",
                  items_type=ParamType.STRING),
        Parameter("guest_selection", ParamType.ATOM_SELECTION,
                  "Atom selection for the guest molecule "
                  "(e.g., 'name I', 'resname LIG', 'resname CO2').",
                  required=False, default=None),
        Parameter("save_csv", ParamType.BOOLEAN,
                  "Save metrics DataFrame as CSV.",
                  required=False, default=True),
    ]
    requires = ["universe"]
    produces = ["cube_metrics_df", "volume_array"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        from src.task import GSAnalyzer

        u = context.universe
        face_sels = params["face_selections"]
        guest_sel = params.get("guest_selection")
        save_csv = params.get("save_csv", True)

        gsa = GSAnalyzer()
        out_prefix = os.path.join(context.output_dir, "gsa_nanocube")
        df = gsa.gsa_nanocube_metrics(
            u, face_sels,
            guest_sel=guest_sel,
            out_prefix=out_prefix if save_csv else None,
        )

        context.set("cube_metrics_df", df)

        if "volume" in df.columns:
            vol = df["volume"].values
            context.set("volume_array", vol)
            context.set("volume", vol)

        artifacts = {}
        if save_csv:
            csv_path = out_prefix + ".csv"
            if os.path.isfile(csv_path):
                artifacts["gsa_csv"] = csv_path

        n_frames = len(df)
        summary = f"Nanocube metrics computed for {n_frames} frames.\n"
        for col in ["volume", "mean_planarity_rms", "mean_edge_length"]:
            if col in df.columns:
                arr = df[col].values
                summary += (
                    f"  {col}: mean={np.nanmean(arr):.2f}, "
                    f"std={np.nanstd(arr):.2f}\n"
                )
        if guest_sel and "guest_min_center_dist" in df.columns:
            gd = df["guest_min_center_dist"].values
            summary += (
                f"  Guest-center distance: "
                f"min={np.nanmin(gd):.2f} A at frame "
                f"{int(np.nanargmin(gd))}"
            )

        return SkillResult(
            success=True,
            data={"cube_metrics_df": df},
            artifacts=artifacts,
            summary=summary,
        )


get_default_registry().register(GSANanocubeMetricsSkill())
