"""Debounced guest entry/exit events and their breakpoint co-timing."""

import numpy as np

from src.ChangepointAnalysis.guest_events import (
    event_recall,
    guest_events,
    occupancy_runs,
    shifted_recall_null,
)


def test_occupancy_runs():
    assert occupancy_runs([0, 0, 1, 1, 1, 0]) == [(False, 0, 2), (True, 2, 3), (False, 5, 1)]


def test_flicker_is_ignored_and_sustained_changes_are_events():
    occ = np.zeros(200, dtype=bool)
    occ[30:32] = True  # 2-frame flicker
    occ[100:160] = True  # sustained visit
    ev = guest_events(occ, min_dwell=20)
    assert ev["event"].tolist() == ["start", "entry", "exit"]
    assert ev["start"].tolist() == [0, 100, 160]
    assert ev["length"].tolist() == [100, 60, 40]
    assert ev["censored"].tolist() == [False, False, True]


def test_entry_that_lasts_to_the_end_is_censored():
    occ = np.r_[np.zeros(50, dtype=bool), np.ones(50, dtype=bool)]
    ev = guest_events(occ, min_dwell=20)
    assert ev.iloc[-1]["event"] == "entry" and bool(ev.iloc[-1]["censored"])


def test_recall_and_shift_null():
    assert event_recall([100, 500], [110, 900], 50) == 0.5
    null = shifted_recall_null([100], [100], 5000, 50, n_shift=200, rng=np.random.default_rng(0))
    assert null.size == 200 and null.mean() < 0.1
