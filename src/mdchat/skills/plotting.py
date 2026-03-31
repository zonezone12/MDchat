"""Plotting skills — generate visualizations from analysis data."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class PlotTimeseriesSkill(Skill):
    name = "plot_timeseries"
    description = (
        "Plot a time series array (e.g., RMSD, radius of gyration, volume) "
        "over trajectory frames and save it as a PNG image."
    )
    category = "plotting"
    parameters = [
        Parameter("data_key", ParamType.STRING,
                  "Context key for the data array to plot (e.g., 'rmsd_array')."),
        Parameter("title", ParamType.STRING,
                  "Plot title.",
                  required=False, default="Time Series"),
        Parameter("ylabel", ParamType.STRING,
                  "Y-axis label.",
                  required=False, default="Value"),
        Parameter("filename", ParamType.STRING,
                  "Output filename (without directory). "
                  "Saved in the session output directory.",
                  required=False, default="timeseries.png"),
    ]
    requires = []
    produces = []

    def validate(self, context):
        return True, ""

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np

        data_key = params["data_key"]
        data = context.get(data_key)
        if data is None:
            return SkillResult(
                success=False,
                error=f"No data found in context for key '{data_key}'. "
                      f"Available keys: {context.list_keys()}",
                summary=f"Cannot plot — data key '{data_key}' not found.",
            )

        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(data)
            except Exception:
                return SkillResult(
                    success=False,
                    error=f"Data for '{data_key}' cannot be converted to a numpy array.",
                    summary=f"Cannot plot — data key '{data_key}' is not array-like.",
                )

        title = params.get("title", "Time Series")
        ylabel = params.get("ylabel", "Value")
        filename = params.get("filename", "timeseries.png")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return SkillResult(
                success=False,
                error="matplotlib is required for plotting but not installed.",
                summary="Cannot plot — matplotlib not available.",
            )

        fig, ax = plt.subplots(figsize=(10, 4))
        frames = np.arange(len(data))
        ax.plot(frames, data, linewidth=0.8)
        ax.set_xlabel("Frame")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path = os.path.join(context.output_dir, filename)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        return SkillResult(
            success=True,
            artifacts={"plot": out_path},
            summary=f"Saved plot to {out_path}",
        )


class PlotEndpointDistancesSkill(Skill):
    name = "plot_endpoint_distances"
    description = (
        "Plot endpoint pairwise distances over trajectory frames. "
        "Requires that endpoint distances have been computed first."
    )
    category = "plotting"
    parameters = [
        Parameter("filename", ParamType.STRING,
                  "Output filename prefix.",
                  required=False, default="endpoint_distances"),
    ]
    requires = ["endpoint_distances", "residue_selections"]
    produces = []

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        from src.Plotter import Plotter

        dists = context.get("endpoint_distances")
        sel_list = context.get("residue_selections")
        prefix = params.get("filename", "endpoint_distances")
        out_prefix = os.path.join(context.output_dir, prefix)

        plotter = Plotter()
        plotter.plot_endpoint_distances(dists, sel_list, out_prefix)

        png_path = out_prefix + ".png"
        if os.path.isfile(png_path):
            return SkillResult(
                success=True,
                artifacts={"endpoint_distances_plot": png_path},
                summary=f"Endpoint distance plot saved to {png_path}",
            )

        return SkillResult(
            success=True,
            summary=(
                f"Endpoint distance plotting completed (output prefix: {out_prefix}). "
                "Check the output directory for generated files."
            ),
        )


get_default_registry().register(PlotTimeseriesSkill())
get_default_registry().register(PlotEndpointDistancesSkill())
