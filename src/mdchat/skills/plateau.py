"""Plateau detection on structural metrics — steady-state / equilibrated region."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry
from ..plateau_helpers import (
    analyze_plateau_series,
    format_standalone_plateau_summary,
    plateau_detection_payload,
    plateau_set_context_defaults,
    write_plateau_csv,
)

if TYPE_CHECKING:
    from ..context import AnalysisContext


def _resolve_selection(context: "AnalysisContext", params: dict) -> str:
    sel = params.get("selection")
    if sel is not None:
        return sel
    return context.main_selection


class DetectMotionPlateauSkill(Skill):
    name = "detect_motion_plateau"
    description = (
        "Detect when large-scale motion has leveled off (plateau) using a sliding-window "
        "fluctuation criterion on RMSD, radius of gyration (Rg), molecular volume, or "
        "cavity volume. Use this when you want plateau analysis alone, different "
        "thresholds without recomputing RMSD/volume, or cavity volume without a full "
        "compute_volume run. For a single call, prefer optional detect_plateau on "
        "compute_rmsd, compute_rg, or compute_volume."
    )
    category = "metrics"
    parameters = [
        Parameter(
            "metric",
            ParamType.STRING,
            "Time series to analyze: 'rmsd', 'rg', 'volume' (enclosed molecular volume), "
            "or 'cavity_volume' (internal cavity volume).",
            required=False,
            default="rmsd",
            enum_values=["rmsd", "rg", "volume", "cavity_volume"],
        ),
        Parameter(
            "selection",
            ParamType.ATOM_SELECTION,
            "Atom selection for RMSD/Rg/volume when the metric must be computed. "
            "If omitted, uses the session main selection.",
            required=False,
            default=None,
        ),
        Parameter(
            "ref_frame",
            ParamType.INTEGER,
            "Reference frame index for RMSD (0-based) when computing RMSD.",
            required=False,
            default=0,
            min_value=0,
        ),
        Parameter(
            "window",
            ParamType.INTEGER,
            "Sliding window length in frames for local fluctuation (std).",
            required=False,
            default=50,
            min_value=2,
        ),
        Parameter(
            "rel_std_threshold",
            ParamType.FLOAT,
            "Plateau tolerance as a fraction of the global std of the time series "
            "(e.g. 0.12 means 12%).",
            required=False,
            default=0.12,
            min_value=1e-6,
            max_value=1.0,
        ),
        Parameter(
            "abs_std_max",
            ParamType.FLOAT,
            "Optional absolute cap on window standard deviation (RMSD/Rg: Å; volume: Å³). "
            "Effective threshold is max(abs_std_max, rel_std_threshold * global_std). "
            "Omit for relative-only.",
            required=False,
            default=None,
            min_value=0.0,
        ),
        Parameter(
            "spacing",
            ParamType.FLOAT,
            "Volume grid spacing (Å) when computing volume or cavity_volume.",
            required=False,
            default=1.0,
            min_value=0.3,
            max_value=5.0,
        ),
        Parameter(
            "probe_radius",
            ParamType.FLOAT,
            "Probe radius (Å) for cavity detection when computing volume metrics.",
            required=False,
            default=1.4,
            min_value=0.5,
            max_value=5.0,
        ),
        Parameter(
            "stride",
            ParamType.INTEGER,
            "Analyze every N-th trajectory frame when computing volume metrics.",
            required=False,
            default=1,
            min_value=1,
        ),
        Parameter(
            "prefer_cached",
            ParamType.BOOLEAN,
            "If True, use rmsd_array, rg_array, volume_array, or cavity_volume_array from "
            "context when present instead of recomputing.",
            required=False,
            default=True,
        ),
        Parameter(
            "save_csv",
            ParamType.BOOLEAN,
            "Write plateau_detection.csv with frame index, metric, and rolling std.",
            required=False,
            default=True,
        ),
    ]
    requires = ["universe"]
    produces = [
        "plateau_detection",
        "plateau_summary",
        "plateau_start_frame",
        "steady_state_representative_frame",
    ]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        import numpy as np
        from src.TrajectoryMetrics import TrajectoryMetrics

        metric = (params.get("metric") or "rmsd").lower().strip()
        allowed = ("rmsd", "rg", "volume", "cavity_volume")
        if metric not in allowed:
            return SkillResult(
                success=False,
                summary=f"metric must be one of {allowed}.",
                error="invalid metric",
            )

        u = context.universe
        n_traj = int(u.trajectory.n_frames)
        prefer = bool(params.get("prefer_cached", True))
        sel_str = _resolve_selection(context, params)
        window = int(params.get("window", 50))
        rel = float(params.get("rel_std_threshold", 0.12))
        abs_max = params.get("abs_std_max")
        abs_max_f = float(abs_max) if abs_max is not None else None
        stride = int(params.get("stride", 1))

        y: np.ndarray
        label: str
        series_frames: np.ndarray | None = None

        if metric == "rmsd":
            if prefer and context.has("rmsd_array"):
                y = np.asarray(context.get("rmsd_array"), dtype=np.float64)
                label = "RMSD (cached)"
            else:
                ref_frame = int(params.get("ref_frame", 0))
                tm = TrajectoryMetrics()
                y = np.asarray(
                    tm.compute_rmsd(u, sel_str, ref_frame=ref_frame), dtype=np.float64
                )
                context.set("rmsd_array", y)
                label = f"RMSD (computed, ref_frame={ref_frame}, sel='{sel_str}')"
            series_frames = np.arange(len(y), dtype=np.int64)

        elif metric == "rg":
            if prefer and context.has("rg_array"):
                y = np.asarray(context.get("rg_array"), dtype=np.float64)
                label = "Rg (cached)"
            else:
                tm = TrajectoryMetrics()
                y = np.asarray(tm.radius_of_gyration(u, sel_str), dtype=np.float64)
                context.set("rg_array", y)
                label = f"Rg (computed, sel='{sel_str}')"
            series_frames = np.arange(len(y), dtype=np.int64)

        elif metric == "volume":
            if prefer and context.has("volume_array"):
                y = np.asarray(context.get("volume_array"), dtype=np.float64)
                label = "Target volume (cached)"
            elif prefer and context.has("volume"):
                y = np.asarray(context.get("volume"), dtype=np.float64)
                label = "Target volume (cached)"
            else:
                from src.VolumeAnalyzer import VolumeAnalyzer

                spacing = float(params.get("spacing", 1.0))
                probe_radius = float(params.get("probe_radius", 1.4))
                va = VolumeAnalyzer(
                    u,
                    selection=sel_str,
                    spacing=spacing,
                    probe_radius=probe_radius,
                )
                results = va.analyze_trajectory(stride=stride)
                y = np.array([r["target_volume"] for r in results], dtype=np.float64)
                cavity = np.array([r["cavity_volume"] for r in results], dtype=np.float64)
                frames = np.array([r["frame"] for r in results], dtype=np.int64)
                context.set("volume_array", y)
                context.set("volume", y)
                context.set("cavity_volume_array", cavity)
                context.set("volume_frames", frames)
                series_frames = frames
                label = (
                    f"Target volume (computed, stride={stride}, spacing={spacing}, "
                    f"sel='{sel_str}')"
                )
            if series_frames is None:
                vf = context.get("volume_frames")
                if vf is not None:
                    series_frames = np.asarray(vf, dtype=np.int64)
                elif len(y) == n_traj:
                    series_frames = np.arange(n_traj, dtype=np.int64)
                else:
                    series_frames = np.arange(len(y), dtype=np.int64)

        else:  # cavity_volume
            if prefer and context.has("cavity_volume_array"):
                y = np.asarray(context.get("cavity_volume_array"), dtype=np.float64)
                label = "Cavity volume (cached)"
            else:
                from src.VolumeAnalyzer import VolumeAnalyzer

                spacing = float(params.get("spacing", 1.0))
                probe_radius = float(params.get("probe_radius", 1.4))
                va = VolumeAnalyzer(
                    u,
                    selection=sel_str,
                    spacing=spacing,
                    probe_radius=probe_radius,
                )
                results = va.analyze_trajectory(stride=stride)
                y = np.array([r["cavity_volume"] for r in results], dtype=np.float64)
                tv = np.array([r["target_volume"] for r in results], dtype=np.float64)
                frames = np.array([r["frame"] for r in results], dtype=np.int64)
                context.set("volume_array", tv)
                context.set("volume", tv)
                context.set("cavity_volume_array", y)
                context.set("volume_frames", frames)
                series_frames = frames
                label = (
                    f"Cavity volume (computed, stride={stride}, spacing={spacing}, "
                    f"sel='{sel_str}')"
                )
            if series_frames is None:
                vf = context.get("volume_frames")
                if vf is not None:
                    series_frames = np.asarray(vf, dtype=np.int64)
                elif len(y) == n_traj:
                    series_frames = np.arange(n_traj, dtype=np.int64)
                else:
                    series_frames = np.arange(len(y), dtype=np.int64)

        assert series_frames is not None

        pa = analyze_plateau_series(
            y,
            series_frames,
            n_traj=n_traj,
            metric=metric,
            window=window,
            rel_std_threshold=rel,
            abs_std_max=abs_max_f,
        )

        summary_text = format_standalone_plateau_summary(
            label=label,
            n_samples=len(y),
            window=window,
            pa=pa,
            metric=metric,
        )
        plateau_set_context_defaults(context, pa, summary_text)

        artifacts: Dict[str, str] = {}
        if params.get("save_csv", True):
            col = {
                "rmsd": "rmsd",
                "rg": "rg",
                "volume": "target_volume",
                "cavity_volume": "cavity_volume",
            }[metric]
            fn, path = write_plateau_csv(
                context.output_dir, col, y, series_frames, pa.rolling_std_list
            )
            artifacts[fn] = path

        payload = plateau_detection_payload(pa, metric, sel_str)

        data_out: Dict[str, Any] = {
            "plateau_detection": payload,
            "plateau_summary": summary_text,
            "plateau_start_frame": pa.plateau_traj,
            "steady_state_representative_frame": pa.steady_traj,
        }

        return SkillResult(
            success=pa.success,
            data=data_out,
            artifacts=artifacts,
            summary=summary_text,
        )


_registry = get_default_registry()
_registry.register(DetectMotionPlateauSkill())
