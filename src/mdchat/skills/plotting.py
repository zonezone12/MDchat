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


class PlotVolumeSkill(Skill):
    name = "plot_volume"
    description = (
        "Plot the molecular volume time series over the trajectory. "
        "Shows target volume (and optionally cavity volume) evolution."
    )
    category = "plotting"
    parameters = [
        Parameter("filename", ParamType.STRING,
                  "Output filename prefix.",
                  required=False, default="volume_change"),
    ]
    requires = ["volume_array"]
    produces = []

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        from src.Plotter import Plotter

        volume = context.get("volume_array")
        prefix = params.get("filename", "volume_change")
        out_prefix = os.path.join(context.output_dir, prefix)

        plotter = Plotter()
        plotter.plot_volume_change(volume, out_prefix)

        png_path = out_prefix + ".png"
        if os.path.isfile(png_path):
            return SkillResult(
                success=True,
                artifacts={"volume_plot": png_path},
                summary=f"Volume plot saved to {png_path}",
            )
        return SkillResult(
            success=True,
            summary=f"Volume plotting completed (prefix: {out_prefix}).",
        )


class PlotGuestTimelineSkill(Skill):
    name = "plot_guest_timeline"
    description = (
        "Plot guest entry/exit events as a timeline. Shows when the guest "
        "molecule is inside or outside the host."
    )
    category = "plotting"
    parameters = [
        Parameter("filename", ParamType.STRING,
                  "Output filename prefix.",
                  required=False, default="guest_events"),
    ]
    requires = ["guest_stats"]
    produces = []

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        from src.Plotter import Plotter

        stats = context.get("guest_stats")
        prefix = params.get("filename", "guest_events")
        out_prefix = os.path.join(context.output_dir, prefix)

        plotter = Plotter()
        plotter.plot_guest_entering_events(stats, out_prefix)

        png_path = out_prefix + ".png"
        if os.path.isfile(png_path):
            return SkillResult(
                success=True,
                artifacts={"guest_timeline_plot": png_path},
                summary=f"Guest timeline plot saved to {png_path}",
            )
        return SkillResult(
            success=True,
            summary=f"Guest timeline plotting completed (prefix: {out_prefix}).",
        )


class PlotRMSFSkill(Skill):
    name = "plot_rmsf"
    description = (
        "Plot per-residue RMSF as a bar chart. Highlights flexible regions "
        "above a threshold."
    )
    category = "plotting"
    parameters = [
        Parameter("threshold", ParamType.FLOAT,
                  "RMSF threshold (Angstrom) above which residues are highlighted.",
                  required=False, default=3.0, min_value=0),
        Parameter("filename", ParamType.STRING,
                  "Output PNG filename.",
                  required=False, default="rmsf_profile.png"),
    ]
    requires = ["rmsf_array"]
    produces = []

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np

        rmsf = context.get("rmsf_array")
        atom_info = context.get("rmsf_atom_info")
        threshold = params.get("threshold", 3.0)
        filename = params.get("filename", "rmsf_profile.png")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return SkillResult(
                success=False,
                error="matplotlib is required for plotting.",
                summary="Cannot plot — matplotlib not available.",
            )

        resids = (
            np.array([a["resid"] for a in atom_info])
            if atom_info
            else np.arange(len(rmsf))
        )

        colors = ["#dc2626" if v > threshold else "#2563eb" for v in rmsf]

        fig, ax = plt.subplots(figsize=(max(10, len(rmsf) * 0.04), 4.5))
        ax.bar(resids, rmsf, color=colors, width=0.8, edgecolor="none")
        ax.axhline(y=np.mean(rmsf), ls="--", color="gray", lw=1,
                    label=f"mean = {np.mean(rmsf):.2f} A")
        if threshold > 0:
            ax.axhline(y=threshold, ls=":", color="#dc2626", lw=1,
                        label=f"threshold = {threshold:.1f} A")
        ax.set_xlabel("Residue ID")
        ax.set_ylabel("RMSF (A)")
        ax.set_title("Per-residue RMSF")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        out_path = os.path.join(context.output_dir, filename)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

        return SkillResult(
            success=True,
            artifacts={"rmsf_plot": out_path},
            summary=f"RMSF profile plot saved to {out_path}",
        )


class PlotCorrelationSkill(Skill):
    name = "plot_correlation"
    description = (
        "Plot the top endpoint–volume correlations, showing how endpoint "
        "distances covary with molecular volume."
    )
    category = "plotting"
    parameters = [
        Parameter("top_n", ParamType.INTEGER,
                  "Number of top-correlated pairs to plot.",
                  required=False, default=5, min_value=1),
        Parameter("filename", ParamType.STRING,
                  "Output filename prefix.",
                  required=False, default="endpoint_volume_corr"),
    ]
    requires = ["endpoint_distances", "volume_array", "correlation_df"]
    produces = []

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        from src.Plotter import Plotter

        dists = context.get("endpoint_distances")
        volume = context.get("volume_array")
        sel_list = context.get("residue_selections", [])
        corr_df = context.get("correlation_df")
        top_n = params.get("top_n", 5)
        prefix = params.get("filename", "endpoint_volume_corr")
        out_prefix = os.path.join(context.output_dir, prefix)

        plotter = Plotter()
        plotter.plot_endpoint_volume_correlation(
            dists, volume, sel_list, corr_df, out_prefix, top_n=top_n,
        )

        png_path = out_prefix + ".png"
        if os.path.isfile(png_path):
            return SkillResult(
                success=True,
                artifacts={"correlation_plot": png_path},
                summary=f"Endpoint–volume correlation plot saved to {png_path}",
            )
        return SkillResult(
            success=True,
            summary=f"Correlation plotting completed (prefix: {out_prefix}).",
        )


_registry = get_default_registry()
_registry.register(PlotTimeseriesSkill())
_registry.register(PlotEndpointDistancesSkill())
_registry.register(PlotVolumeSkill())
_registry.register(PlotGuestTimelineSkill())
_registry.register(PlotRMSFSkill())
_registry.register(PlotCorrelationSkill())
