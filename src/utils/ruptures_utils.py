"""
Changepoint detection utilities built on the `ruptures` library.

Provides a unified API for detecting structural breaks / regime changes
in 1-D time series (RMSD, Rg, volume, etc.) from MD trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class ChangePointResult:
    """Container for changepoint detection results."""

    breakpoints: List[int]
    n_breakpoints: int
    method: str
    cost_model: str
    penalty: Optional[float]
    signal_length: int
    segment_means: List[float] = field(default_factory=list)
    segment_stds: List[float] = field(default_factory=list)
    segment_ranges: List[tuple] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Changepoint detection ({self.method}, cost={self.cost_model})",
            f"  Signal length: {self.signal_length} frames",
            f"  Breakpoints found: {self.n_breakpoints}",
        ]
        if self.penalty is not None:
            lines.append(f"  Penalty: {self.penalty}")
        for i, (start, end) in enumerate(self.segment_ranges):
            mean_s = f"{self.segment_means[i]:.4f}" if self.segment_means else "?"
            std_s = f"{self.segment_stds[i]:.4f}" if self.segment_stds else "?"
            lines.append(
                f"  Segment {i + 1}: frames [{start}, {end}) — "
                f"mean={mean_s}, std={std_s}"
            )
        return "\n".join(lines)


SEARCH_METHODS = ("Pelt", "Binseg", "BottomUp", "Window", "Dynp", "KernelCPD")
COST_MODELS = ("l1", "l2", "rbf", "normal", "ar", "linear", "rank", "mahalanobis")


def detect_changepoints(
    signal: np.ndarray,
    *,
    method: str = "Pelt",
    cost_model: str = "rbf",
    penalty: Optional[float] = None,
    n_bkps: Optional[int] = None,
    min_size: int = 2,
    jump: int = 5,
    window_width: int = 100,
    epsilon: Optional[float] = None,
) -> ChangePointResult:
    """
    Detect changepoints in a 1-D time series using the `ruptures` library.

    Parameters
    ----------
    signal : array-like, shape (n_samples,) or (n_samples, n_features)
        Input time series.
    method : str
        Search method. One of: Pelt, Binseg, BottomUp, Window, Dynp, KernelCPD.
    cost_model : str
        Cost function model: l1, l2, rbf, normal, ar, linear, rank, mahalanobis.
    penalty : float, optional
        Penalty value for Pelt / penalized methods. If None and n_bkps is also None,
        a heuristic log-penalty is applied.
    n_bkps : int, optional
        Exact number of breakpoints to detect (used by Binseg, BottomUp, Dynp).
        Mutually exclusive with penalty for methods that require one or the other.
    min_size : int
        Minimum segment length.
    jump : int
        Subsample step for speed (Binseg, BottomUp, Window).
    window_width : int
        Window width for Window-based search.
    epsilon : float, optional
        Reconstruction budget for Pelt (alternative to penalty).

    Returns
    -------
    ChangePointResult
        Dataclass containing breakpoints, segment statistics, and metadata.
    """
    import ruptures as rpt

    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal.reshape(-1, 1)
    n = signal.shape[0]

    method_lower = method.lower()
    algo: Any

    if method_lower == "pelt":
        algo = rpt.Pelt(model=cost_model, min_size=min_size, jump=jump).fit(signal)
    elif method_lower == "binseg":
        algo = rpt.Binseg(model=cost_model, min_size=min_size, jump=jump).fit(signal)
    elif method_lower == "bottomup":
        algo = rpt.BottomUp(model=cost_model, min_size=min_size, jump=jump).fit(signal)
    elif method_lower == "window":
        algo = rpt.Window(
            model=cost_model, width=window_width, min_size=min_size, jump=jump
        ).fit(signal)
    elif method_lower == "dynp":
        algo = rpt.Dynp(model=cost_model, min_size=min_size, jump=jump).fit(signal)
    elif method_lower == "kernelcpd":
        algo = rpt.KernelCPD(kernel=cost_model, min_size=min_size, jump=jump).fit(signal)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {SEARCH_METHODS}"
        )

    if method_lower == "pelt":
        pen = penalty if penalty is not None else np.log(n) * signal.var() * signal.shape[1]
        if epsilon is not None:
            breakpoints = algo.predict(epsilon=epsilon)
        else:
            breakpoints = algo.predict(pen=pen)
    elif method_lower in ("binseg", "bottomup", "dynp"):
        if n_bkps is not None:
            breakpoints = algo.predict(n_bkps=n_bkps)
        elif penalty is not None:
            breakpoints = algo.predict(pen=penalty)
        else:
            pen = np.log(n) * signal.var() * signal.shape[1]
            breakpoints = algo.predict(pen=pen)
    elif method_lower == "window":
        if n_bkps is not None:
            breakpoints = algo.predict(n_bkps=n_bkps)
        elif penalty is not None:
            breakpoints = algo.predict(pen=penalty)
        else:
            pen = np.log(n) * signal.var() * signal.shape[1]
            breakpoints = algo.predict(pen=pen)
    elif method_lower == "kernelcpd":
        if n_bkps is not None:
            breakpoints = algo.predict(n_bkps=n_bkps)
        elif penalty is not None:
            breakpoints = algo.predict(pen=penalty)
        else:
            pen = np.log(n) * signal.var() * signal.shape[1]
            breakpoints = algo.predict(pen=pen)
    else:
        breakpoints = algo.predict(pen=penalty or 1.0)

    # ruptures includes the signal length as the last breakpoint — remove it
    if breakpoints and breakpoints[-1] == n:
        bkps = breakpoints[:-1]
    else:
        bkps = list(breakpoints)

    # Compute per-segment statistics
    boundaries = [0] + bkps + [n]
    seg_means: List[float] = []
    seg_stds: List[float] = []
    seg_ranges: List[tuple] = []
    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        seg = signal[s:e].ravel() if signal.shape[1] == 1 else signal[s:e]
        seg_means.append(float(np.mean(seg)))
        seg_stds.append(float(np.std(seg)))
        seg_ranges.append((s, e))

    return ChangePointResult(
        breakpoints=bkps,
        n_breakpoints=len(bkps),
        method=method,
        cost_model=cost_model,
        penalty=penalty,
        signal_length=n,
        segment_means=seg_means,
        segment_stds=seg_stds,
        segment_ranges=seg_ranges,
    )


def detect_changepoints_multiseries(
    signals: Dict[str, np.ndarray],
    **kwargs,
) -> Dict[str, ChangePointResult]:
    """
    Run changepoint detection on multiple named time series.

    Parameters
    ----------
    signals : dict[str, ndarray]
        Mapping of metric name -> 1-D signal.
    **kwargs
        Forwarded to `detect_changepoints`.

    Returns
    -------
    dict[str, ChangePointResult]
    """
    return {name: detect_changepoints(sig, **kwargs) for name, sig in signals.items()}
