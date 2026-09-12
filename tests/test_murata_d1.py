"""Remap hull-index endpoint pairs onto Murata equatorial d1."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ChangepointAnalysis.murata_d1 import (
    assign_equatorial_r2_r3_edges,
    classify_murata_d1,
    d1_edge_columns,
    match_hull_sites_to_roles,
    remap_d1_frame_table,
    role_site_index,
    site_pair_column,
)


def test_classify_murata_d1_windows() -> None:
    assert classify_murata_d1(4.0) == "short"
    assert classify_murata_d1(4.5) == "compact"
    assert classify_murata_d1(5.5) == "compact"
    assert classify_murata_d1(6.0) == "intermediate"
    assert classify_murata_d1(7.0) == "elongated"
    assert classify_murata_d1(float("nan")) == "unknown"


def test_assign_equatorial_r2_r3_is_derangement() -> None:
    cost = np.full((6, 6), 20.0)
    for i in range(6):
        cost[i, (i + 1) % 6] = 5.0
    edges = assign_equatorial_r2_r3_edges(cost)
    assert len(edges) == 6
    src = [i for i, _ in edges]
    dst = [j for _, j in edges]
    assert sorted(src) == list(range(6))
    assert sorted(dst) == list(range(6))
    assert all(i != j for i, j in edges)
    assert edges == [(i, (i + 1) % 6) for i in range(6)]


def test_match_hull_prefers_atom_site_containing_ipso() -> None:
    sites = pd.DataFrame(
        [
            {"traj_id": "t", "monomer": 0, "site_index": 0, "kind": "ring", "atom_ids": "1 2 3 4 5 6"},
            {"traj_id": "t", "monomer": 0, "site_index": 3, "kind": "atom", "atom_ids": "4"},
            {"traj_id": "t", "monomer": 0, "site_index": 7, "kind": "ring", "atom_ids": "10 11 12"},
        ]
    )
    roles = pd.DataFrame(
        [
            {
                "monomer": 0,
                "role": "r2",
                "kind": "atom",
                "methyl": False,
                "ipso_mda_id": 4,
                "subst_mda_id": 4,
                "atom_ids": "1 2 3 4 5 6",
            },
            {
                "monomer": 0,
                "role": "r3",
                "kind": "atom",
                "methyl": False,
                "ipso_mda_id": 11,
                "subst_mda_id": 11,
                "atom_ids": "10 11 12",
            },
        ]
    )
    matched = match_hull_sites_to_roles(sites, roles)
    assert role_site_index(matched, 0, "r2") == 3
    assert role_site_index(matched, 0, "r3") == 7
    assert matched.set_index("role").loc["r2", "proxy"] == "ipso_or_para"


def test_remap_d1_frame_table_counts() -> None:
    matched = pd.DataFrame(
        [
            {"monomer": 0, "role": "r2", "site_index": 3},
            {"monomer": 1, "role": "r3", "site_index": 4},
            {"monomer": 1, "role": "r2", "site_index": 3},
            {"monomer": 0, "role": "r3", "site_index": 4},
        ]
    )
    edges = [(0, 1), (1, 0)]
    edge_cols = d1_edge_columns(matched, edges)
    col_a = site_pair_column(0, 3, 1, 4)
    col_b = site_pair_column(1, 3, 0, 4)
    feats = pd.DataFrame(
        {
            "traj_id": ["t", "t"],
            "frame": [0, 1],
            col_a: [5.0, 8.0],
            col_b: [5.2, 4.0],
        }
    )
    out = remap_d1_frame_table(feats, edge_cols)
    assert out.loc[0, "murata_d1_n_compact"] == 2
    assert out.loc[1, "murata_d1_n_elongated"] == 1
    assert out.loc[1, "murata_d1_n_short"] == 1
