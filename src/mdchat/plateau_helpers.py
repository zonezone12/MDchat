"""
Shared plateau detection helpers for MDChat skills.

Wraps :func:`src.PlateauDetection.detect_motion_plateau` with trajectory-frame
mapping and optional CSV export.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np

    from .context import AnalysisContext


def sample_index_to_trajectory_frame(
    series_frames: Any, idx: Optional[int]
) -> Optional[int]:
    """Map index into a time series sample to the corresponding trajectory frame."""
    import numpy as np

    if idx is None:
        return None
    sf = np.asarray(series_frames)
    idx = int(np.clip(idx, 0, sf.size - 1))
    return int(sf[idx])


def index_matches_trajectory_frames(
    y_len: int,
    n_traj: int,
    series_frames: Any,
    metric: str,
) -> bool:
    """False when strided volume fell back to 0..L-1 without real frame indices."""
    import numpy as np

    if metric not in ("volume", "cavity_volume"):
        return True
    sf = np.asarray(series_frames, dtype=np.int64)
    if y_len != n_traj and np.array_equal(sf, np.arange(y_len, dtype=np.int64)):
        return False
    return True


@dataclass
class PlateauAnalysis:
    """Result of running plateau detection on a 1D series."""

    result_ext: Dict[str, Any]
    plateau_traj: Optional[int]
    steady_traj: Optional[int]
    plateau_idx: Optional[int]
    steady_idx: Optional[int]
    rolling_std: Any
    rolling_std_list: List[Any]
    index_matches_traj: bool
    threshold: float
    raw: Dict[str, Any]
    success: bool


def analyze_plateau_series(
    y: "np.ndarray",
    series_frames: "np.ndarray",
    *,
    n_traj: int,
    metric: str,
    window: int = 50,
    rel_std_threshold: float = 0.12,
    abs_std_max: Optional[float] = None,
) -> PlateauAnalysis:
    """Run detect_motion_plateau and map sample indices to trajectory frames."""
    import numpy as np
    from src.PlateauDetection import detect_motion_plateau

    y = np.asarray(y, dtype=np.float64).ravel()
    series_frames = np.asarray(series_frames, dtype=np.int64)

    raw = detect_motion_plateau(
        y,
        window=window,
        rel_std_threshold=rel_std_threshold,
        abs_std_max=abs_std_max,
    )

    rs = raw.get("rolling_std")
    if hasattr(rs, "tolist"):
        rs_list = rs.tolist()
    else:
        rs_list = list(rs) if rs is not None else []

    plateau_idx = raw.get("plateau_start_frame")
    steady_idx = raw.get("steady_state_representative_frame")
    tau = float(raw.get("threshold", float("nan")))

    plateau_traj = sample_index_to_trajectory_frame(series_frames, plateau_idx)
    steady_traj = sample_index_to_trajectory_frame(series_frames, steady_idx)

    result_ext = dict(raw)
    result_ext["plateau_start_sample_index"] = plateau_idx
    result_ext["steady_state_sample_index"] = steady_idx
    result_ext["plateau_start_trajectory_frame"] = plateau_traj
    result_ext["steady_state_trajectory_frame"] = steady_traj
    result_ext["index_matches_trajectory_frames"] = index_matches_trajectory_frames(
        len(y), n_traj, series_frames, metric
    )

    return PlateauAnalysis(
        result_ext=result_ext,
        plateau_traj=plateau_traj,
        steady_traj=steady_traj,
        plateau_idx=plateau_idx,
        steady_idx=steady_idx,
        rolling_std=rs,
        rolling_std_list=rs_list,
        index_matches_traj=result_ext["index_matches_trajectory_frames"],
        threshold=tau,
        raw=raw,
        success=bool(raw.get("success")),
    )


def format_plateau_addon_for_compute_summary(
    *,
    label: str,
    n_samples: int,
    window: int,
    pa: PlateauAnalysis,
    metric: str,
) -> str:
    """Short plateau paragraph to append after compute_rmsd / compute_rg / compute_volume."""
    gstd = pa.raw.get("global_std", float("nan"))
    unit_note = ""
    if metric in ("volume", "cavity_volume"):
        unit_note = " (global std in Å³)"
    parts = [
        f"Plateau ({label}): {n_samples} samples, window={window}, "
        f"threshold τ≈{pa.threshold:.4g}{unit_note}, global std={gstd:.4g}{unit_note}. "
        f"Mode: {pa.raw.get('plateau_mode', '?')}. {pa.raw.get('message', '')}"
    ]
    if not pa.index_matches_traj and metric in ("volume", "cavity_volume"):
        parts.append(
            " Plateau frame indices use trajectory frame numbers from volume_frames when strided."
        )
    if pa.plateau_traj is not None:
        parts.append(
            f" Suggested steady-state trajectory frame: {pa.steady_traj} "
            f"(plateau onset ~frame {pa.plateau_traj})."
        )
    return " ".join(parts).strip()


def format_standalone_plateau_summary(
    *,
    label: str,
    n_samples: int,
    window: int,
    pa: PlateauAnalysis,
    metric: str,
) -> str:
    """Full summary text for the detect_motion_plateau skill."""
    gstd = pa.raw.get("global_std", float("nan"))
    unit_note = ""
    if metric in ("volume", "cavity_volume"):
        unit_note = " (global std in Å³)"
    summary_parts = [
        f"{label}: {n_samples} samples, window={window}, threshold τ≈{pa.threshold:.4g}{unit_note}, "
        f"global std={gstd:.4g}{unit_note}.",
        f"Mode: {pa.raw.get('plateau_mode', '?')}.",
        pa.raw.get("message", ""),
    ]
    if not pa.index_matches_traj and metric in ("volume", "cavity_volume"):
        summary_parts.append(
            "Plateau indices refer to trajectory frame numbers via the volume stride "
            "(see plateau_start_trajectory_frame)."
        )
    if pa.plateau_traj is not None:
        summary_parts.append(
            f"Suggested steady-state trajectory frame: {pa.steady_traj} "
            f"(plateau onset ~frame {pa.plateau_traj})."
        )
    return " ".join(summary_parts).strip()


def plateau_set_context_defaults(context: "AnalysisContext", pa: PlateauAnalysis, summary_text: str) -> None:
    """Store primary plateau keys used across MDChat."""
    context.set("plateau_detection", pa.result_ext)
    context.set("plateau_start_frame", pa.plateau_traj)
    context.set("steady_state_representative_frame", pa.steady_traj)
    context.set("plateau_summary", summary_text)


def write_plateau_csv(
    output_dir: str,
    metric_column: str,
    y: "np.ndarray",
    series_frames: "np.ndarray",
    rolling_std_list: List[Any],
    csv_filename: str = "plateau_detection.csv",
) -> Tuple[str, str]:
    """Write plateau CSV; returns (filename, full_path)."""
    import numpy as np
    import pandas as pd

    y = np.asarray(y, dtype=np.float64).ravel()
    series_frames = np.asarray(series_frames, dtype=np.int64)
    n = len(y)
    traj_col = [int(series_frames[i]) for i in range(n)]
    roll_aligned = [None] * n
    for i in range(min(len(rolling_std_list), n)):
        roll_aligned[i] = rolling_std_list[i]
    path = os.path.join(output_dir, csv_filename)
    pd.DataFrame(
        {
            "trajectory_frame": traj_col,
            "sample_index": list(range(n)),
            metric_column: y.tolist(),
            "rolling_std_window_start": roll_aligned,
        }
    ).to_csv(path, index=False)
    return csv_filename, path


def plateau_detection_payload(
    pa: PlateauAnalysis,
    metric: str,
    selection_used: str,
) -> Dict[str, Any]:
    """Full plateau_detection object for SkillResult (extends raw detector output)."""
    out = dict(pa.result_ext)
    out["metric"] = metric
    out["selection_used"] = selection_used
    return out


def apply_plateau_to_metric_compute(
    context: "AnalysisContext",
    params: Dict[str, Any],
    *,
    universe: Any,
    y: "np.ndarray",
    metric: str,
    selection: str,
    compute_label: str,
    save_csv: bool,
    csv_filename: str = "plateau_detection.csv",
) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
    """
    If ``detect_plateau`` is set in *params*, run plateau and update context.
    Returns ``(summary_addon, extra_data, extra_artifacts)``.
    """
    import numpy as np

    if not params.get("detect_plateau", False):
        return "", {}, {}

    n_traj = int(universe.trajectory.n_frames)
    sf = np.arange(len(y), dtype=np.int64)
    pw = int(params.get("plateau_window", 50))
    rel = float(params.get("rel_std_threshold", 0.12))
    abs_max = params.get("abs_std_max")
    abs_f = float(abs_max) if abs_max is not None else None

    pa = analyze_plateau_series(
        y,
        sf,
        n_traj=n_traj,
        metric=metric,
        window=pw,
        rel_std_threshold=rel,
        abs_std_max=abs_f,
    )
    addon = format_plateau_addon_for_compute_summary(
        label=compute_label,
        n_samples=len(y),
        window=pw,
        pa=pa,
        metric=metric,
    )
    plateau_set_context_defaults(context, pa, addon.strip())

    extra: Dict[str, Any] = {
        "plateau_detection": plateau_detection_payload(pa, metric, selection),
        "plateau_start_frame": pa.plateau_traj,
        "steady_state_representative_frame": pa.steady_traj,
    }
    artifacts: Dict[str, str] = {}
    if save_csv:
        col = "rmsd" if metric == "rmsd" else "rg"
        fn, path = write_plateau_csv(
            context.output_dir,
            col,
            y,
            sf,
            pa.rolling_std_list,
            csv_filename=csv_filename,
        )
        artifacts[fn] = path
    return " " + addon, extra, artifacts


def apply_plateau_volume_optional(
    context: "AnalysisContext",
    params: Dict[str, Any],
    *,
    universe: Any,
    target_vols: "np.ndarray",
    cavity_vols: "np.ndarray",
    frames: "np.ndarray",
    selection: str,
    save_csv: bool,
) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
    """
    Run plateau on target and/or cavity volume series when flags are set in *params*.
    Updates context with ``plateau_detection`` (target takes precedence if both),
    plus ``plateau_detection_target_volume`` / ``plateau_detection_cavity_volume``.
    """
    import numpy as np

    do_t = bool(params.get("detect_plateau_target", False))
    do_c = bool(params.get("detect_plateau_cavity", False))
    if not do_t and not do_c:
        return "", {}, {}

    n_traj = int(universe.trajectory.n_frames)
    sf = np.asarray(frames, dtype=np.int64)
    tv = np.asarray(target_vols, dtype=np.float64).ravel()
    cv = np.asarray(cavity_vols, dtype=np.float64).ravel()
    pw = int(params.get("plateau_window", 50))
    rel = float(params.get("rel_std_threshold", 0.12))
    abs_max = params.get("abs_std_max")
    abs_f = float(abs_max) if abs_max is not None else None

    addons: List[str] = []
    extra: Dict[str, Any] = {}
    artifacts: Dict[str, str] = {}
    pa_t = pa_c = None

    if do_t:
        pa_t = analyze_plateau_series(
            tv,
            sf,
            n_traj=n_traj,
            metric="volume",
            window=pw,
            rel_std_threshold=rel,
            abs_std_max=abs_f,
        )
        addons.append(
            format_plateau_addon_for_compute_summary(
                label="target volume",
                n_samples=len(tv),
                window=pw,
                pa=pa_t,
                metric="volume",
            )
        )
        pld = plateau_detection_payload(pa_t, "volume", selection)
        extra["plateau_target_volume"] = pld
        context.set("plateau_detection_target_volume", pld)
        if save_csv:
            fn, path = write_plateau_csv(
                context.output_dir,
                "target_volume",
                tv,
                sf,
                pa_t.rolling_std_list,
                csv_filename="plateau_detection_target_volume.csv",
            )
            artifacts[fn] = path

    if do_c:
        pa_c = analyze_plateau_series(
            cv,
            sf,
            n_traj=n_traj,
            metric="cavity_volume",
            window=pw,
            rel_std_threshold=rel,
            abs_std_max=abs_f,
        )
        addons.append(
            format_plateau_addon_for_compute_summary(
                label="cavity volume",
                n_samples=len(cv),
                window=pw,
                pa=pa_c,
                metric="cavity_volume",
            )
        )
        pld_c = plateau_detection_payload(pa_c, "cavity_volume", selection)
        extra["plateau_cavity_volume"] = pld_c
        context.set("plateau_detection_cavity_volume", pld_c)
        if save_csv:
            fn, path = write_plateau_csv(
                context.output_dir,
                "cavity_volume",
                cv,
                sf,
                pa_c.rolling_std_list,
                csv_filename="plateau_detection_cavity_volume.csv",
            )
            artifacts[fn] = path

    summary_join = (" ".join(addons)).strip()
    if pa_t is not None:
        plateau_set_context_defaults(context, pa_t, summary_join)
        context.set("plateau_detection", plateau_detection_payload(pa_t, "volume", selection))
    elif pa_c is not None:
        plateau_set_context_defaults(context, pa_c, summary_join)
        context.set("plateau_detection", plateau_detection_payload(pa_c, "cavity_volume", selection))

    return " " + summary_join if summary_join else "", extra, artifacts
