"""Single-replica endpoint clusters: composition, Rg window, feature margins."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ChangepointAnalysis.cluster_replicas import (
    cluster_feature_margins,
    cluster_replica_composition,
    replica_window_rg,
)


def test_cluster_replica_composition_top_replica_shares() -> None:
    segs = pd.DataFrame(
        {
            "traj_id": ["a", "a", "b", "b", "c"],
            "group": ["endpoint"] * 4 + ["gsa"],
            "cluster_label": [0, 0, 0, 1, 1],
            "n_frames": [100, 100, 50, 300, 999],
        }
    )
    comp = cluster_replica_composition(segs).set_index("cluster_label")
    assert comp.loc[0, "n_segments"] == 3
    assert comp.loc[0, "n_traj"] == 2
    assert comp.loc[0, "n_frames"] == 250
    assert comp.loc[0, "top_traj"] == "a"
    assert comp.loc[0, "top_traj_frame_share"] == pytest.approx(0.8)
    assert comp.loc[0, "top_traj_segment_share"] == pytest.approx(2 / 3)
    assert comp.loc[1, "top_traj"] == "b"
    assert comp.loc[1, "top_traj_frame_share"] == pytest.approx(1.0)


def test_replica_window_rg_covers_stride_gaps_to_run_end() -> None:
    replica = pd.DataFrame(
        {"start_frame": [20, 0, 40], "end_frame": [35, 15, 55], "cluster_label": [0, 1, 0]}
    )
    rg = np.where(np.arange(60) < 20, 10.0, 12.0)
    rg[37] = 13.0
    gsa = pd.DataFrame({"frame": np.arange(60), "assembly_rg": rg})

    late = replica_window_rg(gsa, replica, 0)
    assert (late["window_first_frame"], late["window_last_frame"]) == (20, 55)
    assert late["n_gsa_frames_in"] == 40
    assert late["n_gsa_frames_out"] == 20
    assert late["rg_mean_in"] == pytest.approx((12.0 * 39 + 13.0) / 40)
    assert late["rg_max_in"] == 13.0
    assert late["rg_mean_out"] == 10.0
    assert late["rg_max_frame"] == 37
    assert late["rg_max_in_window"]
    assert late["ends_in_cluster"]

    early = replica_window_rg(gsa, replica, 1)
    assert early["n_gsa_frames_in"] == 20
    assert not early["rg_max_in_window"]
    assert not early["ends_in_cluster"]


def test_cluster_feature_margins_signed_and_sorted() -> None:
    segs = pd.DataFrame(
        {
            "cluster_label": [0, 0, 1, 2],
            "endpoint_dist_0_1_mean_mean": [19.0, 21.0, 15.0, 16.0],
            "endpoint_dist_0_2_mean_mean": [10.0, 10.0, 12.0, 14.0],
            "endpoint_dist_1_2_mean_mean": [13.0, 13.0, 12.0, 14.0],
            "endpoint_dist_0_1_std_mean": [9.0, 9.0, 1.0, 1.0],
        }
    )
    margins = cluster_feature_margins(segs, 0)
    assert list(margins["feature"]) == [
        "endpoint_dist_0_1_mean_mean",
        "endpoint_dist_0_2_mean_mean",
        "endpoint_dist_1_2_mean_mean",
    ]
    assert list(margins["margin"]) == pytest.approx([4.0, -2.0, 0.0])
    assert margins.loc[0, "cluster_mean"] == pytest.approx(20.0)
    assert cluster_feature_margins(segs, 7).empty


def test_isolated_replicas_by_k_tracks_one_run_across_cuts():
    import pandas as pd

    from src.ChangepointAnalysis.cluster_replicas import isolated_replicas_by_k

    def segs(labels):
        # run "a" has two 100-frame segments; runs b-f one 300-frame segment each
        return pd.DataFrame(
            {
                "traj_id": ["a", "a", "b", "c", "d", "e", "f"],
                "segment_id": [0, 1, 0, 0, 0, 0, 0],
                "start_frame": [0, 500, 0, 0, 0, 0, 0],
                "end_frame": [495, 995, 995, 995, 995, 995, 995],
                "n_frames": [100, 100, 300, 300, 300, 300, 300],
                "cluster_label": labels,
            }
        )

    # k=2: run a sits inside a multi-run cluster. k=3: its second half becomes a cluster of its own.
    out = isolated_replicas_by_k(
        {2: segs([0, 0, 0, 0, 1, 1, 1]), 3: segs([0, 2, 0, 0, 1, 1, 1])},
        min_top_share=0.5,
    )
    assert out["k"].tolist() == [3]
    row = out.iloc[0]
    assert row["traj_id"] == "a" and row["top_traj_frame_share"] == 1.0
    assert (row["window_first_frame"], row["window_last_frame"]) == (500, 995)
