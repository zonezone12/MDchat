"""Murata apo-nanocube RMSD filter and peak matching."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ChangepointAnalysis.murata_rmsd import (
    apo_frame_mask,
    attach_open_cation_pi,
    filter_non_encapsulated,
    find_rmsd_histogram_peaks,
    open_count_bin,
    summarize_apo_rmsd,
)


def test_apo_frame_mask_drops_encapsulated() -> None:
    df = pd.DataFrame(
        {
            "traj_id": ["a", "a", "b"],
            "n_guest_inside_cavity": [0.0, 1.0, 0.0],
            "assembly_rmsd_to_ref": [1.0, 2.7, 1.5],
        }
    )
    mask = apo_frame_mask(df)
    assert list(mask) == [True, False, True]
    apo = filter_non_encapsulated(df, traj_filter="frames")
    assert list(apo["assembly_rmsd_to_ref"]) == [1.0, 1.5]
    all_frames = filter_non_encapsulated(df, traj_filter="all")
    assert len(all_frames) == 3


def test_never_encapsulated_traj_filter() -> None:
    df = pd.DataFrame(
        {
            "traj_id": ["a", "a", "b", "b"],
            "n_guest_inside_cavity": [0.0, 1.0, 0.0, 0.0],
            "assembly_rmsd_to_ref": [1.0, 2.0, 1.4, 1.6],
        }
    )
    apo = filter_non_encapsulated(df, traj_filter="never")
    assert set(apo["traj_id"]) == {"b"}
    assert len(apo) == 2


def test_find_rmsd_peaks_near_murata_values() -> None:
    rng = np.random.default_rng(0)
    vals = np.concatenate(
        [
            rng.normal(1.0, 0.08, 4000),
            rng.normal(1.5, 0.08, 3000),
            rng.normal(2.7, 0.10, 2000),
        ]
    )
    vals = vals[vals > 0]
    peaks = find_rmsd_histogram_peaks(vals, bin_width=0.05, smooth_sigma=1.5)
    matched = peaks[peaks["matched"]]
    found = set(float(x) for x in matched["nearest_murata_A"])
    assert {1.0, 1.5, 2.7} <= found
    for target in (1.0, 1.5, 2.7):
        row = matched.loc[matched["nearest_murata_A"] == target].iloc[0]
        assert abs(row["peak_A"] - target) < 0.2


def test_summarize_apo_rmsd_counts() -> None:
    df = pd.DataFrame(
        {
            "traj_id": ["t"] * 10,
            "n_guest_inside_cavity": [0.0] * 7 + [1.0] * 3,
            "assembly_rmsd_to_ref": [1.0] * 7 + [3.0] * 3,
        }
    )
    row = summarize_apo_rmsd(df, cohort="BHHpH")
    assert row["n_frames"] == 10
    assert row["n_apo_frames"] == 7
    assert row["n_encapsulated_frames"] == 3
    assert row["rmsd_median_A"] == 1.0
    all_row = summarize_apo_rmsd(df, cohort="BHHpH", traj_filter="all")
    assert all_row["guest_filter"] == "all"
    assert all_row["n_apo_frames"] == 10


def test_drop_reference_zero_rmsd() -> None:
    from src.ChangepointAnalysis.murata_rmsd import drop_reference_rmsd

    df = pd.DataFrame(
        {
            "traj_id": ["t", "t"],
            "n_guest_inside_cavity": [0.0, 0.0],
            "assembly_rmsd_to_ref": [0.0, 1.2],
        }
    )
    out = drop_reference_rmsd(df)
    assert list(out["assembly_rmsd_to_ref"]) == [1.2]


def test_attach_open_cation_pi_strips_cube_prefix() -> None:
    rmsd = pd.DataFrame(
        {
            "traj_id": ["BHHpM_109345_mdcrd_v", "BHHpM_109345_mdcrd_v"],
            "frame": [0, 1],
            "assembly_rmsd_to_ref": [1.0, 1.5],
            "n_guest_inside_cavity": [0.0, 0.0],
        }
    )
    opened = pd.DataFrame(
        {
            "traj_id": ["109345_mdcrd_v", "109345_mdcrd_v"],
            "frame": [0, 1],
            "n_open_cation_pi": [0, 1],
        }
    )
    out = attach_open_cation_pi(rmsd, opened, cube="BHHpM")
    assert list(out["n_open_cation_pi"]) == [0, 1]
    assert list(open_count_bin(out["n_open_cation_pi"])) == [0.0, 1.0]


def test_attach_guest_occupancy_joins_on_frame() -> None:
    import tempfile
    from pathlib import Path

    from src.ChangepointAnalysis.murata_rmsd import attach_guest_occupancy

    rmsd = pd.DataFrame(
        {
            "frame": [0, 1, 2],
            "rmsd_cation_pi": [0.0, 1.0, 1.5],
        }
    )
    gsa = pd.DataFrame({"frame": [0, 1, 2], "n_guest_inside_cavity": [0, 1, 0]})
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "gsa.csv"
        gsa.to_csv(path, index=False)
        out = attach_guest_occupancy(rmsd, path)
    assert list(out["n_guest_inside_cavity"]) == [0, 1, 0]
