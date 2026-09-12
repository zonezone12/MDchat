"""Murata G0 metastructure criteria: cation–π, d2, A/B/C1/C2 labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ChangepointAnalysis.murata_criteria import (
    CATION_PI_OPEN_LO,
    MURATA_RMSD_MOTIFS,
    attach_murata_metastructure_labels,
    cation_pi_column_name,
    classify_murata_metastructure,
    classify_murata_metastructure_columns,
    lock_cation_pi_units,
    motif_atom_indices,
    occupancy_table,
)
from src.ChangepointAnalysis.murata_d1 import match_hull_sites_to_roles, role_site_index


def test_classify_murata_metastructure_table() -> None:
    assert classify_murata_metastructure(0, 0) == "A"
    assert classify_murata_metastructure(0, 2) == "A"
    assert classify_murata_metastructure(1, 0) == "B"
    assert classify_murata_metastructure(1, 1) == "B"
    assert classify_murata_metastructure(2, 0) == "C1"
    assert classify_murata_metastructure(2, 1) == "C2"
    assert classify_murata_metastructure(2, 2) == "other"
    assert classify_murata_metastructure(3, 0) == "other"


def test_classify_murata_metastructure_columns() -> None:
    labels = classify_murata_metastructure_columns(
        np.array([0, 1, 2, 2, 2, 4]),
        np.array([0, 0, 0, 1, 2, 0]),
    )
    assert list(labels) == ["A", "B", "C1", "C2", "other", "other"]


def test_lock_cation_pi_counts_openings() -> None:
    n = 6
    data: dict[str, list[float]] = {"traj_id": ["t", "t"], "frame": [0, 1]}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Cycle i → (i+1) is compact (~5 Å); everything else is far.
            partner = (i + 1) % n
            data[cation_pi_column_name(i, j)] = (
                [5.0, 5.0] if j == partner else [12.0, 12.0]
            )
    # Open two cycle contacts on frame 1.
    data[cation_pi_column_name(0, 1)] = [5.0, 7.0]
    data[cation_pi_column_name(1, 2)] = [5.0, 8.0]
    df = pd.DataFrame(data)
    pair_cols = [(cation_pi_column_name(i, j), i, j) for i in range(n) for j in range(n) if i != j]
    out, edges = lock_cation_pi_units(df, pair_cols, n_monomers=n)
    assert len(edges) == 6
    assert out.loc[0, "n_open_cation_pi"] == 0
    assert out.loc[1, "n_open_cation_pi"] == 2
    labeled = attach_murata_metastructure_labels(
        out.assign(murata_d1_n_elongated=[0, 0])
    )
    assert labeled.loc[0, "murata_metastructure"] == "A"
    assert labeled.loc[1, "murata_metastructure"] == "C1"


def test_lock_cation_pi_open_threshold() -> None:
    assert CATION_PI_OPEN_LO == 6.5
    df = pd.DataFrame(
        {
            cation_pi_column_name(0, 1): [6.49, 6.50],
            cation_pi_column_name(1, 0): [5.0, 5.0],
        }
    )
    pair_cols = [
        (cation_pi_column_name(0, 1), 0, 1),
        (cation_pi_column_name(1, 0), 1, 0),
    ]
    out, edges = lock_cation_pi_units(df, pair_cols, n_monomers=2)
    assert edges == [(0, 1), (1, 0)]
    assert out.loc[0, "n_open_cation_pi"] == 0
    assert out.loc[1, "n_open_cation_pi"] == 1


def test_lock_equatorial_d1_counts_compact_and_elongated() -> None:
    from src.ChangepointAnalysis.endpoint_features import paper_d1_column_name
    from src.ChangepointAnalysis.murata_criteria import lock_equatorial_d1

    n = 6
    data: dict[str, list[float]] = {"frame": [0, 1]}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            partner = (i + 1) % n
            data[paper_d1_column_name(i, "r2", j, "r3")] = (
                [5.0, 5.0] if j == partner else [11.0, 11.0]
            )
    data[paper_d1_column_name(0, "r2", 1, "r3")] = [5.0, 7.5]
    df = pd.DataFrame(data)
    out, edges = lock_equatorial_d1(df, n_monomers=n)
    assert len(edges) == 6
    assert out.loc[0, "murata_d1_n_compact"] == 6
    assert out.loc[0, "murata_d1_n_elongated"] == 0
    assert out.loc[1, "murata_d1_n_elongated"] == 1


def test_occupancy_table_percentages() -> None:
    labels = pd.Series(["A", "A", "B", "other"])
    occ = occupancy_table(labels).set_index("metastructure")
    assert occ.loc["A", "percent"] == 50.0
    assert occ.loc["B", "percent"] == 25.0
    assert occ.loc["C1", "n_frames"] == 0


def test_match_hull_ring_role_ph() -> None:
    sites = pd.DataFrame(
        [
            {"traj_id": "t", "monomer": 0, "site_index": 0, "kind": "ring", "atom_ids": "1 2 3 4 5 6"},
            {"traj_id": "t", "monomer": 0, "site_index": 5, "kind": "ring", "atom_ids": "10 11 12 13 14 15"},
            {"traj_id": "t", "monomer": 0, "site_index": 3, "kind": "atom", "atom_ids": "4"},
        ]
    )
    roles = pd.DataFrame(
        [
            {
                "monomer": 0,
                "role": "ph",
                "kind": "ring",
                "methyl": False,
                "ipso_mda_id": None,
                "subst_mda_id": None,
                "atom_ids": "1 2 3 4 5 6",
            },
            {
                "monomer": 0,
                "role": "py_pole",
                "kind": "ring",
                "methyl": False,
                "ipso_mda_id": None,
                "subst_mda_id": None,
                "atom_ids": "10 11 12 13 14 15",
            },
        ]
    )
    matched = match_hull_sites_to_roles(sites, roles, required_roles=("ph", "py_pole"))
    assert role_site_index(matched, 0, "ph") == 0
    assert role_site_index(matched, 0, "py_pole") == 5


def _motif_roles_df() -> pd.DataFrame:
    rows = []
    for mon in (0, 1):
        base = 20 * mon
        rows.extend(
            [
                {
                    "monomer": mon,
                    "role": "ph",
                    "atom_indices": f"{base} {base + 1} {base + 2}",
                    "cpy_index": None,
                    "ipso_index": None,
                },
                {
                    "monomer": mon,
                    "role": "py_pole",
                    "atom_indices": f"{base + 3} {base + 4} {base + 5}",
                    "cpy_index": None,
                    "ipso_index": None,
                },
                {
                    "monomer": mon,
                    "role": "py_eq",
                    "atom_indices": f"{base + 6} {base + 7} {base + 8}",
                    "cpy_index": base + 6,
                    "ipso_index": None,
                },
                {
                    "monomer": mon,
                    "role": "r2",
                    "atom_indices": f"{base + 9} {base + 10} {base + 11}",
                    "cpy_index": None,
                    "ipso_index": base + 9,
                },
                {
                    "monomer": mon,
                    "role": "r3",
                    "atom_indices": f"{base + 12} {base + 13} {base + 14}",
                    "cpy_index": None,
                    "ipso_index": base + 12,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_motif_atom_indices_cation_pi_and_equator() -> None:
    roles = _motif_roles_df()
    cation = motif_atom_indices(roles, MURATA_RMSD_MOTIFS["cation_pi"])
    equator = motif_atom_indices(roles, MURATA_RMSD_MOTIFS["equator"])
    assert set(cation) == {0, 1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23, 24, 25, 26, 27, 28}
    assert set(equator) == {
        6, 7, 8, 9, 10, 11, 12, 13, 14,
        26, 27, 28, 29, 30, 31, 32, 33, 34,
    }
    assert set(cation) & set(equator) == {6, 7, 8, 26, 27, 28}


def test_motif_rmsd_rises_when_ph_moves() -> None:
    from types import SimpleNamespace

    from src.ChangepointAnalysis.murata_criteria import MurataCriteriaFrameObserver
    from src.TrajectoryMetrics.TrajectoryMetrics import rmsd_value_aligned

    roles = _motif_roles_df()
    n_atoms = 40
    ref = np.zeros((n_atoms, 3), dtype=float)
    for i in range(n_atoms):
        ref[i, 0] = float(i)
    cation = motif_atom_indices(roles, MURATA_RMSD_MOTIFS["cation_pi"])
    equator = motif_atom_indices(roles, MURATA_RMSD_MOTIFS["equator"])
    motif_refs = {
        "cation_pi": (cation, ref[cation].copy()),
        "equator": (equator, ref[equator].copy()),
    }
    obs = MurataCriteriaFrameObserver(
        roles, n_frame_rows=2, rmsd_by_motif=motif_refs
    )
    u0 = SimpleNamespace(atoms=SimpleNamespace(positions=ref.copy()))
    obs.on_frame(None, 0, u0)
    moved = ref.copy()
    moved[0:3, 1] = 2.0  # Ph of monomer 0, cation–π only
    u1 = SimpleNamespace(atoms=SimpleNamespace(positions=moved))
    obs.on_frame(None, 1, u1)
    df = obs.to_dataframe()
    assert df.loc[0, "rmsd_cation_pi"] < 1e-6
    assert df.loc[0, "rmsd_equator"] < 1e-6
    assert df.loc[1, "rmsd_cation_pi"] > 0.3
    assert df.loc[1, "rmsd_equator"] < 1e-6
    assert abs(
        df.loc[1, "rmsd_cation_pi"] - rmsd_value_aligned(moved[cation], ref[cation])
    ) < 1e-9


def test_motif_atom_counts_on_topology() -> None:
    pytest.importorskip("rdkit")
    import MDAnalysis as mda

    from src.ChangepointAnalysis.murata_criteria import (
        motif_atom_indices,
        resolve_murata_criteria_atoms,
    )
    from src.utils.gsa_selections import resolve_selections

    topo = Path(__file__).resolve().parent.parent / "traj" / "BMMpM_ca.prmtop"
    if not topo.exists():
        pytest.skip(f"{topo} not present")
    u = mda.Universe(str(topo))
    sels = resolve_selections(u, gsa_resname="MOL", n_monomers=6, auto_tooth=False)
    roles = resolve_murata_criteria_atoms(u, sels.monomer_selections)
    cation = motif_atom_indices(roles, MURATA_RMSD_MOTIFS["cation_pi"])
    equator = motif_atom_indices(roles, MURATA_RMSD_MOTIFS["equator"])
    # 6 monomers × 3 six-membered rings (pole Py+, Ph, eq Py+).
    assert len(cation) == 108
    # Equator adds R2/R3 aryl rings (and methyl carbons on BMM).
    assert len(equator) >= 108
    assert len(set(cation) & set(equator)) == 36

