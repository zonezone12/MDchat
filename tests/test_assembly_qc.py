"""Tests for the pre-clustering assembly QC rule (plan.md G2)."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ChangepointAnalysis import (
    SegmentClusteringConfig,
    assembly_qc_failures,
    cluster_all_groups,
    drop_trajectories,
    scan_assembly_qc,
    scan_assembly_qc_dir,
)


def _write_gsa_features(path: Path, traj_id: str, rg, lcc) -> Path:
    n = len(rg)
    pd.DataFrame(
        {
            "traj_id": [traj_id] * n,
            "frame": np.arange(n),
            "assembly_rg": rg,
            "largest_connected_component_size": lcc,
            "n_guest_inside_cavity": np.zeros(n),
        }
    ).to_csv(path, index=False)
    return path


def test_scan_assembly_qc_flags_dissociation_at_first_bad_frame(tmp_path: Path) -> None:
    ok = _write_gsa_features(
        tmp_path / "BMHpH_1_mdcrd_v_gsa_features.csv",
        "BMHpH_1_mdcrd_v",
        rg=[12.0, 12.5, 13.0, 12.8],
        lcc=[6, 6, 6, 6],
    )
    broken = _write_gsa_features(
        tmp_path / "BMHpH_2_mdcrd_v_gsa_features.csv",
        "BMHpH_2_mdcrd_v",
        rg=[12.0, 12.4, 15.6, 18.0],
        lcc=[6, 6, 5, 5],
    )
    qc = scan_assembly_qc([ok, broken])
    by_id = qc.set_index("traj_id")

    assert bool(by_id.loc["BMHpH_1_mdcrd_v", "qc_pass"])
    assert pd.isna(by_id.loc["BMHpH_1_mdcrd_v", "first_fail_frame"])
    row = by_id.loc["BMHpH_2_mdcrd_v"]
    assert not bool(row["qc_pass"])
    assert row["first_fail_frame"] == 2
    assert row["assembly_rg_max"] == pytest.approx(18.0)
    assert row["min_component_size"] == 5
    assert row["cube"] == "BMHpH"
    assert assembly_qc_failures(qc) == ("BMHpH_2_mdcrd_v",)


def test_scan_assembly_qc_component_loss_alone_fails(tmp_path: Path) -> None:
    path = _write_gsa_features(
        tmp_path / "BHHpM_3_mdcrd_v_gsa_features.csv",
        "BHHpM_3_mdcrd_v",
        rg=[12.0, 12.1, 12.2],
        lcc=[6, 5, 6],
    )
    qc = scan_assembly_qc([path])
    assert not bool(qc.loc[0, "qc_pass"])
    assert qc.loc[0, "n_frames_rg_fail"] == 0
    assert qc.loc[0, "n_frames_component_fail"] == 1
    assert qc.loc[0, "first_fail_frame"] == 1


def test_scan_assembly_qc_requires_both_columns(tmp_path: Path) -> None:
    path = tmp_path / "BHHpM_4_mdcrd_v_gsa_features.csv"
    pd.DataFrame(
        {"traj_id": ["BHHpM_4_mdcrd_v"], "frame": [0], "assembly_rg": [12.0]}
    ).to_csv(path, index=False)
    with pytest.raises(KeyError, match="largest_connected_component_size"):
        scan_assembly_qc([path])


def test_scan_assembly_qc_dir_rejects_mixed_cubes(tmp_path: Path) -> None:
    _write_gsa_features(
        tmp_path / "BHHpM_1_mdcrd_v_gsa_features.csv", "BHHpM_1_mdcrd_v", rg=[12.0], lcc=[6]
    )
    _write_gsa_features(
        tmp_path / "BMHpH_1_mdcrd_v_gsa_features.csv", "BMHpH_1_mdcrd_v", rg=[12.0], lcc=[6]
    )
    with pytest.raises(ValueError, match="mixes cubes"):
        scan_assembly_qc_dir(tmp_path)


def test_drop_trajectories_ignores_cube_prefix_and_keeps_reruns() -> None:
    segs = pd.DataFrame(
        {
            "traj_id": ["563849_mdcrd_v", "563849_re_mdcrd_v", "109345_mdcrd_v"],
            "segment_id": [0, 0, 0],
        }
    )
    out = drop_trajectories(segs, ["BMHpH_563849_mdcrd_v"])
    assert list(out["traj_id"]) == ["563849_re_mdcrd_v", "109345_mdcrd_v"]
    assert drop_trajectories(segs, []) is segs


def test_cluster_all_groups_honours_exclude_traj_ids(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for t in range(4):
        for s in range(3):
            high = s % 2 == 1
            rows.append(
                {
                    "traj_id": f"{t}_mdcrd_v",
                    "group": "gsa",
                    "segment_id": s,
                    "start_frame": s * 100,
                    "end_frame": s * 100 + 99,
                    "n_frames": 100 + 10 * s,
                    "assembly_rg_mean": (14.0 if high else 12.0) + rng.normal(0, 0.1),
                    "assembly_rg_std": abs(rng.normal(0.2, 0.05)),
                    "assembly_rmsd_to_ref_mean": (3.0 if high else 1.0) + rng.normal(0, 0.1),
                    "assembly_rmsd_to_ref_std": abs(rng.normal(0.1, 0.02)),
                    "octahedrality_score_mean": rng.normal(0.8, 0.05),
                    "octahedrality_score_std": abs(rng.normal(0.02, 0.005)),
                }
            )
    config = SegmentClusteringConfig(
        groups=("gsa",),
        n_clusters=2,
        save_all_k=False,
        skip_pca=True,
        silhouette_k_max=3,
        exclude_traj_ids=("BMHpH_0_mdcrd_v",),
    )
    clustered = cluster_all_groups(pd.DataFrame(rows), tmp_path / "clusters", config)

    assert "0_mdcrd_v" not in set(clustered["traj_id"])
    assert len(clustered) == 9
