"""Label-free k selection on held-out cage geometry."""

import numpy as np
import pandas as pd

from src.ChangepointAnalysis.cluster_k_diagnostics import (
    _r2_from_stats,
    cluster_sufficient_stats,
    held_out_geometry_columns,
    select_k_one_se,
    top_replica_frame_shares,
)


def test_top_replica_frame_shares_flags_a_one_run_cluster():
    segs = pd.DataFrame(
        {
            "cluster_label": [0, 0, 0, 1, 1, 1],
            "traj_id": ["a", "a", "a", "a", "b", "c"],
            "n_frames": [100, 200, 300, 100, 100, 200],
        }
    )
    shares = top_replica_frame_shares(segs)
    assert shares[0] == 1.0  # three segments, all from run "a"
    assert shares[1] == 0.5


def _direct_r2(z, labels):
    grand = z.mean(axis=0)
    res = sum(((z[labels == c] - z[labels == c].mean(axis=0)) ** 2).sum() for c in np.unique(labels))
    return 1.0 - res / ((z - grand) ** 2).sum()


def test_in_sample_r2_matches_direct():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(300, 4))
    traj = rng.integers(0, 6, size=300)
    lab = rng.integers(0, 3, size=300)
    z[lab == 1] += 1.5
    N, S, Q = cluster_sufficient_stats(z, traj, lab, 6, 3)
    ones = np.ones(6)
    assert np.isclose(_r2_from_stats(N, S, Q, ones, ones), _direct_r2(z, lab))


def test_out_of_bag_ignores_a_one_trajectory_cluster():
    # Cluster 2 exists only in trajectory 0. Fitted without trajectory 0 it has
    # no mean, so it cannot help predict; fitted on it alone it does.
    rng = np.random.default_rng(1)
    z = rng.normal(size=(400, 3))
    traj = np.repeat(np.arange(4), 100)
    lab = np.where(traj == 0, 2, rng.integers(0, 2, size=400))
    z[lab == 2] += 3.0
    N, S, Q = cluster_sufficient_stats(z, traj, lab, 4, 3)
    w_fit = np.array([0.0, 1.0, 1.0, 1.0])
    w_eval = np.array([1.0, 0.0, 0.0, 0.0])
    oob = _r2_from_stats(N, S, Q, w_fit, w_eval)
    in_sample = _r2_from_stats(N, S, Q, np.ones(4), np.ones(4))
    assert oob <= 0.0 + 1e-9
    assert in_sample > 0.5


def test_held_out_columns_exclude_clustering_duplicates():
    cols = held_out_geometry_columns(
        ["assembly_rg", "endpoint_dist_mean", "endpoint_dist_std", "salt_bridge_total", "n_guest_bulk"]
    )
    assert cols == ["assembly_rg", "salt_bridge_total"]


def test_select_k_one_se_respects_gate_and_picks_smallest():
    by_k = pd.DataFrame(
        {
            "cohort": ["X"] * 4,
            "k": [2, 3, 4, 5],
            "chemical_separation": [False, True, True, True],
            "score": [0.9, 0.40, 0.50, 0.52],
            "se": [0.01, 0.01, 0.01, 0.05],
        }
    )
    picked = select_k_one_se(by_k, score_col="score", se_col="se").iloc[0]
    assert picked["best_k"] == 5  # k=2 fails the gate despite the top score
    assert picked["selected_k"] == 4  # within 0.05 of 0.52; k=3 is not
