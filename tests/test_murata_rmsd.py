"""Murata apo-nanocube RMSD filter and peak matching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def test_time_ns_from_time_ps() -> None:
    from src.ChangepointAnalysis.murata_rmsd import time_ns_from_frames

    df = pd.DataFrame({"time_ps": [0.0, 1000.0, 5000.0]})
    assert list(time_ns_from_frames(df)) == [0.0, 1.0, 5.0]


def test_plot_rmsd_vs_time_writes_png(tmp_path) -> None:
    from src.ChangepointAnalysis.murata_rmsd import plot_rmsd_vs_time

    t = np.arange(0, 200, dtype=float)
    df = pd.DataFrame(
        {
            "traj_id": ["a"] * 100 + ["b"] * 100,
            "time_ps": np.concatenate([t[:100], t[:100]]),
            "rmsd_cation_pi": np.concatenate(
                [1.0 + 0.1 * np.sin(t[:100] / 10), 1.5 + 0.1 * np.cos(t[:100] / 10)]
            ),
        }
    )
    path = tmp_path / "rmsd_vs_time.png"
    out = plot_rmsd_vs_time(df, path, rmsd_col="rmsd_cation_pi")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_load_motif_rmsd_csvs_keeps_unit_distance_columns(tmp_path) -> None:
    from src.ChangepointAnalysis.murata_rmsd import load_motif_rmsd_csvs

    path = tmp_path / "t_motif_rmsd.csv"
    pd.DataFrame(
        {
            "cohort": ["BMMpM"],
            "traj_id": ["t"],
            "frame": [0],
            "time_ps": [0.0],
            "assembly_rmsd_to_ref": [0.0],
            "rmsd_cation_pi": [0.0],
            "rmsd_equator": [0.0],
            "cation_pi_pole_e0_m0pole_m1ph": [5.2],
            "cation_pi_eq_e0_m5eq_m1ph": [4.8],
            "rmsd_cation_pi_e0": [0.1],
            "equator_d1_e0_m0r2_m1r3": [4.8],
            "equator_d2_e0_m0cpy_m1r3": [6.1],
            "cation_pi_angle_e0_m1ph": [170.0],
            "equator_angle_e0_m1c3": [95.0],
            "rmsd_equator_e0": [0.2],
            "n_open_cation_pi": [0],
            "n_guest_inside_cavity": [0.0],
        }
    ).to_csv(path, index=False)
    loaded = load_motif_rmsd_csvs([path])
    assert "cation_pi_pole_e0_m0pole_m1ph" in loaded.columns
    assert "cation_pi_eq_e0_m5eq_m1ph" in loaded.columns
    assert "equator_d1_e0_m0r2_m1r3" in loaded.columns
    assert "equator_d2_e0_m0cpy_m1r3" in loaded.columns
    assert "cation_pi_angle_e0_m1ph" in loaded.columns
    assert "equator_angle_e0_m1c3" in loaded.columns
    assert "rmsd_cation_pi_e0" in loaded.columns
    assert "n_open_cation_pi" in loaded.columns


def test_write_trajectory_motif_rmsd_writes_units_after_guest_merge(
    tmp_path, monkeypatch
) -> None:
    from src.ChangepointAnalysis import murata_rmsd as mod

    df = pd.DataFrame(
        {
            "traj_id": ["t"],
            "frame": [0],
            "rmsd_cation_pi": [0.0],
            "cation_pi_pole_e0_m0pole_m1ph": [5.0],
        }
    )
    df.attrs["motif_units"] = pd.DataFrame(
        [
            {
                "kind": "cation_pi",
                "edge": 0,
                "mon_a": 0,
                "mon_b": 1,
                "mon_c": 5,
                "distance_column": "cation_pi_pole_e0_m0pole_m1ph",
                "eq_distance_column": "cation_pi_eq_e0_m5eq_m1ph",
                "rmsd_column": "rmsd_cation_pi_e0",
            }
        ]
    )
    monkeypatch.setattr(mod, "compute_trajectory_motif_rmsd", lambda *a, **k: df)
    gsa = tmp_path / "gsa.csv"
    pd.DataFrame({"frame": [0], "n_guest_inside_cavity": [1]}).to_csv(gsa, index=False)
    out = tmp_path / "t_motif_rmsd.csv"
    mod.write_trajectory_motif_rmsd(
        "topo", "traj", "t", "BMMpM", str(out), gsa_csv=str(gsa)
    )
    units = tmp_path / "t_motif_units.csv"
    assert units.is_file()
    assert list(pd.read_csv(units)["distance_column"]) == ["cation_pi_pole_e0_m0pole_m1ph"]
    assert pd.read_csv(out)["n_guest_inside_cavity"].iloc[0] == 1


def test_plot_per_unit_motif_series_writes_png(tmp_path) -> None:
    from src.ChangepointAnalysis.murata_rmsd import plot_per_unit_motif_series

    t = np.arange(0, 50, dtype=float)
    df = pd.DataFrame(
        {
            "traj_id": ["a"] * 50,
            "time_ps": t,
            "cation_pi_pole_e0_m0pole_m1ph": 5.0 + 0.1 * np.sin(t / 8),
            "cation_pi_eq_e0_m5eq_m1ph": 4.5 + 0.1 * np.cos(t / 8),
            "cation_pi_pole_e1_m1pole_m2ph": 6.0 + 0.1 * np.cos(t / 8),
            "rmsd_cation_pi_e0": 0.5 + 0.05 * np.sin(t / 10),
            "equator_d1_e0_m0r2_m1r3": 4.5 + 0.1 * np.sin(t / 7),
            "equator_d2_e0_m0cpy_m1r3": 6.0 + 0.1 * np.cos(t / 7),
            "cation_pi_angle_e0_m1ph": 170.0 + np.sin(t / 8),
            "equator_angle_e0_m1c3": 90.0 + np.cos(t / 9),
        }
    )
    written = plot_per_unit_motif_series(df, tmp_path, cube="BMMpM")
    assert written
    assert all(p.is_file() and p.stat().st_size > 0 for p in written)


def test_compute_emits_per_unit_distance_columns() -> None:
    from src.ChangepointAnalysis.murata_rmsd import compute_trajectory_motif_rmsd

    root = Path(__file__).resolve().parent.parent
    topo = root / "traj" / "BMMpM_ca.prmtop"
    trajs = sorted(root.glob("traj/BMMpM_*mdcrd_v*"))
    if not topo.exists() or not trajs:
        pytest.skip("BMMpM topology/trajectory not present")
    df = compute_trajectory_motif_rmsd(
        topo, trajs[0], traj_id="smoke", start=0, stop=3, step=1
    )
    pole = [c for c in df.columns if str(c).startswith("cation_pi_pole_e")]
    eq_pi = [c for c in df.columns if str(c).startswith("cation_pi_eq_e")]
    d1 = [c for c in df.columns if str(c).startswith("equator_d1_e")]
    d2 = [c for c in df.columns if str(c).startswith("equator_d2_e")]
    pi_ang = [c for c in df.columns if str(c).startswith("cation_pi_angle_e")]
    eq_ang = [c for c in df.columns if str(c).startswith("equator_angle_e")]
    assert len(pole) == 6
    assert len(eq_pi) == 6
    assert len(d1) == 6
    assert len(d2) == 6
    assert len(pi_ang) == 6
    assert len(eq_ang) == 6
    assert "n_open_cation_pi" in df.columns
    units = df.attrs.get("motif_units")
    assert isinstance(units, pd.DataFrame)
    assert len(units) == 12
    cation = units[units["kind"] == "cation_pi"]
    assert "mon_c" in cation.columns
    assert (cation["mon_a"] != cation["mon_b"]).all()
    assert (cation["mon_b"] != cation["mon_c"]).all()
    assert (cation["mon_a"] != cation["mon_c"]).all()
    assert df.loc[0, pole[0]] > 0
    assert df.loc[0, eq_pi[0]] > 0
    assert df.loc[0, d2[0]] > 0
    assert 0.0 <= float(df.loc[0, pi_ang[0]]) <= 180.0
    assert 0.0 <= float(df.loc[0, eq_ang[0]]) <= 180.0

