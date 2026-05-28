"""Changepoint detection skill — regime-change analysis on computed time series."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class DetectChangepointsSkill(Skill):
    name = "detect_changepoints"
    description = (
        "Detect regime changes (changepoints) in a previously computed time series "
        "(RMSD, Rg, volume, cavity_volume, or any named array in the session context) "
        "using the ruptures library. Useful for identifying structural transitions, "
        "equilibration boundaries, or phase changes in MD metrics. "
        "Requires that the target metric has already been computed and stored in context."
    )
    category = "metrics"
    parameters = [
        Parameter(
            "metric",
            ParamType.STRING,
            "Context key of the time series to analyze. Common values: "
            "'rmsd_array', 'rg_array', 'volume_array', 'cavity_volume_array'. "
            "Any 1-D numeric array stored in the session context can be used.",
            required=True,
        ),
        Parameter(
            "method",
            ParamType.STRING,
            "Search algorithm: 'Pelt' (penalized, automatic number), "
            "'Binseg' (binary segmentation), 'BottomUp', 'Window', 'Dynp' "
            "(dynamic programming, exact), 'KernelCPD' (kernel change-point detection).",
            required=False,
            default="Pelt",
            enum_values=["Pelt", "Binseg", "BottomUp", "Window", "Dynp", "KernelCPD"],
        ),
        Parameter(
            "cost_model",
            ParamType.STRING,
            "Cost function: 'l1', 'l2', 'rbf' (radial basis function — good general "
            "default), 'normal', 'ar' (autoregressive), 'linear', 'rank', 'mahalanobis'.",
            required=False,
            default="rbf",
            enum_values=["l1", "l2", "rbf", "normal", "ar", "linear", "rank", "mahalanobis"],
        ),
        Parameter(
            "penalty",
            ParamType.FLOAT,
            "Penalty value for penalized methods (Pelt, or pen-based predict). "
            "Higher = fewer breakpoints. If omitted, a BIC-like log(n)*variance heuristic "
            "is used automatically.",
            required=False,
            default=None,
            min_value=0.0,
        ),
        Parameter(
            "n_bkps",
            ParamType.INTEGER,
            "Exact number of breakpoints to find (for Binseg, BottomUp, Dynp, Window, "
            "KernelCPD). Mutually exclusive with penalty for those methods. "
            "Omit to let the algorithm decide via penalty.",
            required=False,
            default=None,
            min_value=1,
        ),
        Parameter(
            "min_size",
            ParamType.INTEGER,
            "Minimum segment length in frames.",
            required=False,
            default=2,
            min_value=1,
        ),
        Parameter(
            "jump",
            ParamType.INTEGER,
            "Subsample step for search algorithms (larger = faster but coarser).",
            required=False,
            default=5,
            min_value=1,
        ),
        Parameter(
            "window_width",
            ParamType.INTEGER,
            "Window width for the Window method.",
            required=False,
            default=100,
            min_value=10,
        ),
        Parameter(
            "save_csv",
            ParamType.BOOLEAN,
            "Save changepoint results to a CSV file in the session output directory.",
            required=False,
            default=True,
        ),
    ]
    requires = ["universe"]
    produces = ["changepoint_result", "changepoint_summary"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        import numpy as np

        metric_key = params.get("metric")
        if not metric_key:
            return SkillResult(
                success=False,
                summary="Parameter 'metric' is required (context key for the signal).",
                error="missing metric",
            )

        if not context.has(metric_key):
            available = [
                k
                for k in ("rmsd_array", "rg_array", "volume_array", "cavity_volume_array")
                if context.has(k)
            ]
            hint = (
                f" Available metric arrays: {available}." if available
                else " No metric arrays found — run a compute skill first."
            )
            return SkillResult(
                success=False,
                summary=f"Context key '{metric_key}' not found.{hint}",
                error="metric not in context",
            )

        signal = np.asarray(context.get(metric_key), dtype=np.float64).ravel()
        if signal.size < 10:
            return SkillResult(
                success=False,
                summary=f"Signal '{metric_key}' has only {signal.size} points — too short.",
                error="signal too short",
            )

        from src.utils.ruptures_utils import detect_changepoints, ChangePointResult

        method = params.get("method", "Pelt")
        cost_model = params.get("cost_model", "rbf")
        penalty = params.get("penalty")
        n_bkps = params.get("n_bkps")
        min_size = int(params.get("min_size", 2))
        jump = int(params.get("jump", 5))
        window_width = int(params.get("window_width", 100))

        penalty_f = float(penalty) if penalty is not None else None
        n_bkps_i = int(n_bkps) if n_bkps is not None else None

        try:
            result: ChangePointResult = detect_changepoints(
                signal,
                method=method,
                cost_model=cost_model,
                penalty=penalty_f,
                n_bkps=n_bkps_i,
                min_size=min_size,
                jump=jump,
                window_width=window_width,
            )
        except Exception as exc:
            return SkillResult(
                success=False,
                summary=f"ruptures detection failed: {exc}",
                error=str(exc),
            )

        summary_text = result.summary()

        context.set("changepoint_result", {
            "metric": metric_key,
            "breakpoints": result.breakpoints,
            "n_breakpoints": result.n_breakpoints,
            "method": result.method,
            "cost_model": result.cost_model,
            "segment_means": result.segment_means,
            "segment_stds": result.segment_stds,
            "segment_ranges": result.segment_ranges,
        })
        context.set("changepoint_summary", summary_text)

        artifacts: Dict[str, str] = {}
        if params.get("save_csv", True):
            import pandas as pd
            import os

            rows = []
            for i, (start, end) in enumerate(result.segment_ranges):
                rows.append({
                    "segment": i + 1,
                    "start_frame": start,
                    "end_frame": end,
                    "n_frames": end - start,
                    "mean": result.segment_means[i],
                    "std": result.segment_stds[i],
                })
            df = pd.DataFrame(rows)

            out_dir = getattr(context, "output_dir", None) or "."
            os.makedirs(out_dir, exist_ok=True)
            csv_name = f"changepoints_{metric_key}.csv"
            csv_path = os.path.join(out_dir, csv_name)
            df.to_csv(csv_path, index=False)
            artifacts[csv_name] = csv_path

        data_out = {
            "changepoint_result": context.get("changepoint_result"),
            "changepoint_summary": summary_text,
        }

        return SkillResult(
            success=True,
            data=data_out,
            artifacts=artifacts,
            summary=summary_text,
        )


_registry = get_default_registry()
_registry.register(DetectChangepointsSkill())
