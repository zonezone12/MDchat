"""NGL HTML for locked Murata cation–π / equator units."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ChangepointAnalysis.murata_criteria import (
    cation_pi_unit_label,
    equator_unit_label,
)
from src.ChangepointAnalysis.murata_units_ngl import (
    MONOMER_COLORS,
    build_murata_unit_payloads,
    classify_cation_pi_state,
    edges_from_units_df,
    monomer_color,
    render_murata_units_html,
    write_murata_units_ngl_html,
)


def test_unit_labels_match_requested_wording() -> None:
    assert cation_pi_unit_label(0, 2, 5) == "Cation–π unit 0: π(pole) Py⁺ 2 → Ph 5"
    assert equator_unit_label(3, 1, 4) == (
        "Equator unit 3: d1 R2 1 → R3 4 · d2 CPy 1 → R3 4"
    )


def test_classify_cation_pi_state_threshold() -> None:
    assert classify_cation_pi_state(6.49) == "closed"
    assert classify_cation_pi_state(6.50) == "open"


def test_edges_from_units_df_preserves_edge_order() -> None:
    df = pd.DataFrame(
        [
            {"kind": "equator", "edge": 1, "mon_a": 2, "mon_b": 3},
            {"kind": "cation_pi", "edge": 1, "mon_a": 1, "mon_b": 2},
            {"kind": "cation_pi", "edge": 0, "mon_a": 0, "mon_b": 1},
            {"kind": "equator", "edge": 0, "mon_a": 0, "mon_b": 5},
        ]
    )
    cation, equator = edges_from_units_df(df)
    assert cation == [(0, 1), (1, 2)]
    assert equator == [(0, 5), (2, 3)]


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
                    "ipso_index": None,
                },
                {
                    "monomer": mon,
                    "role": "py_pole",
                    "atom_indices": f"{base + 3} {base + 4} {base + 5}",
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
                    "ipso_index": base + 9,
                },
                {
                    "monomer": mon,
                    "role": "r3",
                    "atom_indices": f"{base + 12} {base + 13} {base + 14}",
                    "ipso_index": base + 12,
                },
            ]
        )
    return pd.DataFrame(rows)


def _toy_universe():
    pytest.importorskip("MDAnalysis")
    import MDAnalysis as mda
    from MDAnalysis.coordinates.memory import MemoryReader

    n_atoms = 40
    u = mda.Universe.empty(n_atoms, n_residues=1, atom_resindex=[0] * n_atoms, trajectory=True)
    names = [f"C{i}" for i in range(n_atoms)]
    u.add_TopologyAttr("name", names)
    u.add_TopologyAttr("type", ["C"] * n_atoms)
    u.add_TopologyAttr("element", ["C"] * n_atoms)
    u.add_TopologyAttr("resname", ["MOL"])
    u.add_TopologyAttr("resid", [1])
    u.add_TopologyAttr("chainID", ["X"] * n_atoms)
    u.add_TopologyAttr("altLoc", [""] * n_atoms)
    u.add_TopologyAttr("occupancy", np.ones(n_atoms))
    u.add_TopologyAttr("tempfactor", np.zeros(n_atoms))
    pos = np.zeros((1, n_atoms, 3), dtype=np.float64)
    # Compact cation–π / equator 0→1 at ~1 Å; reverse pair far away.
    pos[0, 3:6] = (0.0, 0.0, 0.0)  # m0 py_pole
    pos[0, 20:23] = (1.0, 0.0, 0.0)  # m1 ph  → unit 0 closed (~1 Å)
    pos[0, 23:26] = (10.0, 0.0, 0.0)  # m1 py_pole
    pos[0, 0:3] = (20.0, 0.0, 0.0)  # m0 ph  → unit 1 open (~10 Å)
    pos[0, 9] = (0.0, 1.0, 0.0)  # m0 r2 ipso
    pos[0, 32] = (1.0, 1.0, 0.0)  # m1 r3 ipso  → equator 0 closed
    pos[0, 29] = (10.0, 1.0, 0.0)  # m1 r2 ipso
    pos[0, 12] = (20.0, 1.0, 0.0)  # m0 r3 ipso  → equator 1 elongated
    u.load_new(pos, format=MemoryReader)
    return u


def test_build_payloads_uses_requested_labels_and_states() -> None:
    u = _toy_universe()
    roles = _motif_roles_df()
    index_to_pos = {int(a.index): i for i, a in enumerate(u.atoms)}
    pairs = build_murata_unit_payloads(
        u,
        roles,
        cation_edges=[(0, 1), (1, 0)],
        equator_edges=[(0, 1), (1, 0)],
        index_to_pos=index_to_pos,
    )
    labels = [p["label"] for p in pairs]
    cation0 = next(p for p in pairs if p["kind"] == "cation_pi" and p["edge"] == 0)
    cation1 = next(p for p in pairs if p["kind"] == "cation_pi" and p["edge"] == 1)
    assert cation0["label"].startswith("Cation–π unit 0: π(pole) Py⁺ 0 → Ph 1")
    assert cation1["label"].startswith("Cation–π unit 1: π(pole) Py⁺ 1 → Ph 0")
    assert "Equator unit 0: d1 R2 0 → R3 1 · d2 CPy 0 → R3 1" in labels
    assert "Equator unit 1: d1 R2 1 → R3 0 · d2 CPy 1 → R3 0" in labels
    compact = cation0
    opened = cation1
    assert compact["state"] == "closed"
    assert compact["distance"] == pytest.approx(1.0)
    assert compact["color_i"] == monomer_color(0)
    assert compact["color_j"] == monomer_color(1)
    assert "distance_eq" in compact
    assert "mon_c" in compact
    assert "angle_deg" in compact
    assert 0.0 <= compact["angle_deg"] <= 180.0
    assert opened["state"] == "open"
    assert opened["distance"] == pytest.approx(10.0)
    eq_open = next(
        p for p in pairs if p["label"] == "Equator unit 1: d1 R2 1 → R3 0 · d2 CPy 1 → R3 0"
    )
    assert eq_open["state"] == "elongated"
    assert "distance_d2" in eq_open
    assert "state_eq" in compact
    assert "angle_deg" in eq_open
    eq_compact = next(
        p for p in pairs if p["label"] == "Equator unit 0: d1 R2 0 → R3 1 · d2 CPy 0 → R3 1"
    )
    assert eq_compact["atoms_k"] == [6]
    assert eq_open["atoms_k"] == [26]
    assert eq_compact["color_k"] == "#00d4aa"


def test_render_html_contains_sidebar_and_ngl() -> None:
    html = render_murata_units_html(
        title="Murata units",
        meta_text="Frame 0",
        pdb_string="ATOM      1  C   MOL     1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
        pairs=[
            {
                "rank": 1,
                "kind": "cation_pi",
                "edge": 0,
                "label": "Cation–π unit 0: π(pole) Py⁺ 0 → Ph 1",
                "distance": 5.2,
                "state": "closed",
                "color": "#e6194b",
                "color_i": "#911eb4",
                "color_j": "#f5c542",
                "atoms_i": [1, 2],
                "atoms_j": [3, 4],
                "atoms_unit": [1, 2, 3, 4],
                "centroid_i": [0.0, 0.0, 0.0],
                "centroid_j": [1.0, 0.0, 0.0],
                "midpoint": [0.5, 0.0, 0.0],
                "labelVisible": False,
            }
        ],
        site_groups=[{"label": "Py-pole", "color": "#911eb4", "atoms": [1, 2]}],
    )
    assert "ngl@2.3.1" in html
    assert "Cation–π unit 0: π(pole) Py⁺ 0 → Ph 1" in html
    assert "Show all" in html
    assert "Cation–π" in html
    assert "Equator" in html
    assert "Monomer backbone" in html
    assert "var monomerGroups" in html


def test_write_html_from_toy_universe(tmp_path: Path) -> None:
    u = _toy_universe()
    out = tmp_path / "murata_units.html"
    path = write_murata_units_ngl_html(
        u,
        out,
        frame=0,
        selection="all",
        roles_df=_motif_roles_df(),
        cation_edges=[(0, 1)],
        equator_edges=[(0, 1)],
        title="Toy units",
    )
    text = path.read_text(encoding="utf-8")
    assert path.is_file()
    assert "Cation–π unit 0: π(pole) Py⁺ 0 → Ph 1" in text
    assert "Equator unit 0: d1 R2 0 → R3 1 · d2 CPy 0 → R3 1" in text
    assert "π(pole)" in text
    assert "show-d1" in text
    assert "show-d2" in text
    assert "CPy (para to benzene linker)" in text
    assert "#00d4aa" in text
    assert "ngl@2.3.1" in text
    assert "Monomer backbone" in text
    assert '"label": "M0"' in text
    assert MONOMER_COLORS[0] in text
