"""Guest entry/exit events from cavity occupancy, and their timing against breakpoints.

``n_guest_inside_cavity`` flickers at the cavity boundary (most occupied runs
last 1–2 frames), so occupancy changes are debounced: the state flips only when
the new occupancy persists for at least ``min_dwell`` frames, and the event is
placed at the start of that run.

Co-timing uses the same greedy 1-to-1 matching (±tolerance frames) as the
breakpoint Jaccard, scored as the fraction of guest events matched to a
breakpoint, against a null that circularly shifts the guest events within
their own trajectory.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .detection import _match_breakpoints


def occupancy_runs(occupied: Sequence[bool]) -> list[tuple[bool, int, int]]:
    """Runs of constant occupancy as ``(state, start, length)``."""
    occ = np.asarray(occupied, dtype=bool)
    if occ.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(occ.astype(int))) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, occ.size]
    return [(bool(occ[s]), int(s), int(e - s)) for s, e in zip(starts, ends)]


def guest_events(
    occupied: Sequence[bool],
    *,
    min_dwell: int = 20,
) -> pd.DataFrame:
    """Debounced entry/exit events and the sustained dwells between them.

    Returns one row per sustained dwell: ``state`` (True = guest inside),
    ``start`` (the event frame that began it; 0 for the initial dwell),
    ``length`` and ``event`` (``entry`` / ``exit`` / ``start``). The last dwell
    is right-censored at the end of the trajectory (``censored=True``).
    """
    runs = occupancy_runs(occupied)
    if not runs:
        return pd.DataFrame(columns=["state", "start", "length", "event", "censored"])
    n = sum(r[2] for r in runs)
    state = runs[0][0]
    dwells: list[list] = [[state, 0, "start"]]
    for st, start, length in runs[1:]:
        if st != state and length >= int(min_dwell):
            state = st
            dwells.append([state, start, "entry" if state else "exit"])
    out = pd.DataFrame(dwells, columns=["state", "start", "event"])
    out["length"] = np.diff(np.r_[out["start"].to_numpy(), n])
    out["censored"] = False
    out.loc[out.index[-1], "censored"] = True
    return out[["state", "start", "length", "event", "censored"]]


def event_recall(
    events: Sequence[int],
    breakpoints: Sequence[int],
    tolerance: int,
) -> float:
    """Fraction of *events* matched 1-to-1 to a breakpoint within ±tolerance."""
    if len(events) == 0:
        return float("nan")
    return len(_match_breakpoints(list(events), list(breakpoints), tolerance)) / len(events)


def shifted_recall_null(
    events: Sequence[int],
    breakpoints: Sequence[int],
    n_frames: int,
    tolerance: int,
    *,
    n_shift: int = 100,
    margin: int = 250,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Recall after circularly shifting *events* by random offsets within the trajectory."""
    rng = rng or np.random.default_rng(0)
    ev = np.asarray(events, dtype=int)
    if ev.size == 0 or n_frames <= 2 * margin:
        return np.array([])
    shifts = rng.integers(margin, n_frames - margin, size=int(n_shift))
    return np.array(
        [event_recall(sorted(((ev + s) % n_frames).tolist()), breakpoints, tolerance) for s in shifts]
    )
