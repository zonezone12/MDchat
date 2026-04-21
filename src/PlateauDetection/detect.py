"""
Detect plateau regions in any 1D time series (e.g. RMSD, Rg, molecular volume in Å³).

Used to locate when large-scale motion has subsided so representative
“steady state” structures can be taken from the latter part of the run.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _rolling_std(y: np.ndarray, window: int) -> np.ndarray:
    """Sample standard deviation in each length-*window* segment; length n - window + 1."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    if window < 2 or n < window:
        return np.array([])
    c1 = np.empty(n + 1, dtype=np.float64)
    c2 = np.empty(n + 1, dtype=np.float64)
    c1[0] = 0.0
    c2[0] = 0.0
    c1[1:] = np.cumsum(y)
    c2[1:] = np.cumsum(y * y)
    s1 = c1[window:] - c1[:-window]
    s2 = c2[window:] - c2[:-window]
    # Sample variance: (sum x^2 - (sum x)^2 / n) / (n - 1)
    var = (s2 - (s1 * s1) / window) / max(window - 1, 1)
    return np.sqrt(np.maximum(var, 0.0))


def detect_motion_plateau(
    y: np.ndarray,
    window: int = 50,
    rel_std_threshold: float = 0.12,
    abs_std_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Find the first frame after which sliding-window fluctuations stay below a threshold
    (stable tail). This is a common heuristic for “equilibrated” or steady-state motion.

    For each window start index *i*, we compute std(y[i : i+window]). The plateau begins
    at the smallest *i* such that **all** subsequent window std values on the trajectory
    are <= tau, where::

        tau = max(abs_std_max, rel_std_threshold * nanstd(y))

    Parameters
    ----------
    y
        Metric time series (e.g. RMSD or Rg), one value per frame.
    window
        Sliding window length in frames (must be >= 2).
    rel_std_threshold
        Plateau tolerance as a fraction of the global standard deviation of *y*.
    abs_std_max
        Optional absolute cap on window std (e.g. 0.5 Angstrom for RMSD). The effective
        threshold is max(abs_std_max, rel * global_std) when given; otherwise rel * global_std.

    Returns
    -------
    dict with keys:
        success : bool
        plateau_start_frame : int or None
            First frame index *i* such that all windows from *i* to the end satisfy the
            threshold (None if not found or input invalid).
        steady_state_representative_frame : int or None
            Suggested single frame for a representative structure (midpoint of plateau).
        threshold : float
            Effective std threshold used.
        global_std : float
        rolling_std : ndarray
            Per-window rolling std (length n - window + 1).
        plateau_mode : str
            ``full_tail`` if a full-tail plateau was found, else ``longest_segment`` or
            ``insufficient_data``.
        message : str
            Short explanation.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.size
    out: Dict[str, Any] = {
        "success": False,
        "plateau_start_frame": None,
        "steady_state_representative_frame": None,
        "threshold": float("nan"),
        "global_std": float("nan"),
        "rolling_std": np.array([]),
        "plateau_mode": "insufficient_data",
        "message": "",
    }

    if n < 4:
        out["message"] = "Time series too short for plateau detection (need at least 4 frames)."
        return out

    w = int(window)
    if w < 2:
        w = 2
    if w > n // 2:
        w = max(2, n // 2)

    gstd = float(np.nanstd(y))
    if not np.isfinite(gstd) or gstd < 1e-12:
        gstd = float(np.nanmean(np.abs(y))) + 1e-12

    rel = float(rel_std_threshold)
    if rel <= 0:
        rel = 0.12

    if abs_std_max is not None and abs_std_max > 0:
        tau = max(float(abs_std_max), rel * gstd)
    else:
        tau = rel * gstd

    rs = _rolling_std(y, w)
    if rs.size == 0:
        out["message"] = "Could not compute rolling statistics."
        return out

    out["global_std"] = gstd
    out["threshold"] = tau
    out["rolling_std"] = rs

    stable = rs <= tau

    # Full tail: first index i such that stable[i:].all()
    plateau_start = None
    for i in range(rs.size):
        if bool(np.all(stable[i:])):
            plateau_start = i
            break

    if plateau_start is not None:
        out["success"] = True
        out["plateau_mode"] = "full_tail"
        out["plateau_start_frame"] = int(plateau_start)
        mid = int(round((plateau_start + (n - 1)) / 2.0))
        out["steady_state_representative_frame"] = mid
        out["message"] = (
            f"Stable tail from frame {plateau_start} onward "
            f"(all {w}-frame window std ≤ {tau:.4g})."
        )
        return out

    # Fallback: longest contiguous run of low rolling std
    best_len = 0
    best_start = 0
    i = 0
    while i < rs.size:
        if not stable[i]:
            i += 1
            continue
        j = i
        while j < rs.size and stable[j]:
            j += 1
        run_len = j - i
        if run_len > best_len:
            best_len = run_len
            best_start = i
        i = j

    if best_len == 0:
        out["plateau_mode"] = "none"
        out["message"] = (
            f"No plateau found: try a larger rel_std_threshold (now τ={tau:.4g}), "
            f"a smaller window, or abs_std_max."
        )
        return out

    out["plateau_mode"] = "longest_segment"
    out["success"] = True
    out["plateau_start_frame"] = int(best_start)
    seg_end_frame = int(best_start + best_len + w - 2)
    seg_end_frame = min(seg_end_frame, n - 1)
    mid = int(round((best_start + seg_end_frame) / 2.0))
    out["steady_state_representative_frame"] = mid
    out["message"] = (
        f"No full-tail plateau; longest low-fluctuation segment starts at window index "
        f"{best_start} (~frame {best_start}), length {best_len} windows. "
        f"Representative frame suggestion: {mid}."
    )
    return out
