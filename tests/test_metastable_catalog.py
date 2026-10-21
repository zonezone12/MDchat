"""Orientation-copy merge and chemical names for endpoint clusters."""

import numpy as np

from src.ChangepointAnalysis.metastable_catalog import (
    belt_kind,
    belt_label,
    contact_pairs,
    dwell_stats,
    family_ids,
    opening_label,
    pair_relation,
    structure_name,
)


def _row(**overrides):
    base = {
        "pct_A": 0.0,
        "pct_any_d1_elongated": 0.0,
        "pct_d1_elongated_e0_M0-M4": 0.0,
        "pct_d1_elongated_e5_M5-M0": 0.0,
        "pct_pi_open_e0_M0-M1": 0.0,
        "pct_pi_open_e5_M5-M3": 0.0,
    }
    base.update(overrides)
    return base


def test_family_ids_merge_a_chain_and_keep_a_weak_pair_apart():
    # 0–1 and 1–2 are copies; 0–2 is only related. Single linkage still joins them.
    # Cluster 3 stays apart.
    families = family_ids(
        [0, 1, 2, 3],
        [(0, 1, 0.96), (1, 2, 0.93), (0, 2, 0.82), (0, 3, 0.73)],
        threshold=0.90,
    )
    assert families[0] == families[1] == families[2] == 0
    assert families[3] == 3


def test_structure_name_distinguishes_compact_belt_and_replica():
    assert structure_name(_row(pct_A=96.0, pct_any_d1_elongated=0.03)) == "compact cage"
    assert (
        structure_name(
            _row(pct_A=2.0, pct_any_d1_elongated=95.0, **{"pct_d1_elongated_e5_M5-M0": 74.0})
        )
        == "belt broken along M5–M0"
    )
    assert (
        structure_name(
            _row(pct_A=32.0, pct_any_d1_elongated=8.0, **{"pct_pi_open_e5_M5-M3": 55.0})
        )
        == "intact belt, M5–M3 often open"
    )
    assert "single replica" in structure_name(_row(pct_any_d1_elongated=80.0), single_replica=True)


def test_belt_and_opening_labels_ignore_cluster_count():
    assert belt_label([]) == "intact belt"
    assert belt_label(["M5-M0"]) == "edge M5-M0"
    assert belt_label(["M0-M4", "M1-M5"]) == "edges M0-M4+M1-M5"
    assert belt_label(["M0-M4", "M1-M5", "M2-M3"]) == "3 elongated edges"
    assert opening_label(1) == "1 pole open"
    assert opening_label(2) == "2 poles open"
    assert opening_label(4) == "3+ poles open"


def test_dwell_stats_split_on_a_gap_and_a_label_change():
    traj = np.array(["a"] * 5)
    frame = np.array([0, 1, 2, 10, 11])
    label = np.array(["edge", "edge", "intact", "edge", "edge"])
    median, persistent = dwell_stats(traj, frame, label, min_persist=2)
    # runs of 2, 1, and 2 ps. Two of the three frames in the long runs... 
    # lengths 2, 1, 2. Frames in runs >= 2: 4/5 = 80%.
    assert median == 2.0
    assert persistent == 80.0


def test_pair_relation_from_motif_columns():
    # Pole contacts and belt edges as the motif CSVs name them; what is neither is opposite.
    cols = [
        "pct_pi_open_e0_M0-M1", "pct_pi_open_e1_M1-M2", "pct_pi_open_e2_M2-M0",
        "pct_pi_open_e3_M3-M4", "pct_pi_open_e4_M4-M5", "pct_pi_open_e5_M5-M3",
        "pct_d1_elongated_e0_M0-M4", "pct_d1_elongated_e1_M1-M5", "pct_d1_elongated_e2_M2-M3",
        "pct_d1_elongated_e3_M3-M1", "pct_d1_elongated_e4_M4-M2", "pct_d1_elongated_e5_M5-M0",
        "pct_A",
    ]
    pole = contact_pairs(cols, "pct_pi_open_")
    belt = contact_pairs(cols, "pct_d1_elongated_")
    assert len(pole) == 6 and len(belt) == 6 and not pole & belt
    assert pair_relation(1, 0, pole, belt) == "same pole"
    assert pair_relation(4, 0, pole, belt) == "equatorial"
    assert [pair_relation(i, i + 3, pole, belt) for i in range(3)] == ["opposite"] * 3


def test_belt_kind_bins_frame_labels():
    assert belt_kind(belt_label([])) == "intact"
    assert belt_kind(belt_label(["M5-M0"])) == "one_edge"
    assert belt_kind(belt_label(["M0-M4", "M5-M0"])) == "two_plus_edges"
    assert belt_kind(belt_label(["M0-M4", "M1-M5", "M2-M3"])) == "two_plus_edges"


def test_structure_name_keeps_a_split_axis_and_a_partial_edge():
    split = _row(
        pct_any_d1_elongated=92.0,
        **{"pct_d1_elongated_e4_M4-M2": 49.0, "pct_d1_elongated_e1_M1-M5": 43.0},
    )
    assert structure_name(split) == "belt broken along M4–M2 or M1–M5"
    partial = _row(pct_any_d1_elongated=48.0, **{"pct_d1_elongated_e4_M4-M2": 48.0})
    assert structure_name(partial) == "partly belt broken along M4–M2"
