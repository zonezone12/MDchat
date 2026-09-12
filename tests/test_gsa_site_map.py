"""Canonical GSA site map: R1/R2/R3 roles are stable across B* cohorts."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Murata substitution vs our cohort names (user-confirmed BMHpM = 2₆).
_EXPECTED_METHYL = {
    "BHHpH": {"r1": False, "r2": False, "r3": False},
    "BHHpM": {"r1": True, "r2": False, "r3": False},
    "BMHpH": {"r1": False, "r2": False, "r3": True},
    "BMHpM": {"r1": True, "r2": False, "r3": True},
    "BMMpH": {"r1": False, "r2": True, "r3": True},
    "BMMpM": {"r1": True, "r2": True, "r3": True},
}


def _monomer0_mol(cube: str):
    import MDAnalysis as mda

    from src.utils.gsa_selections import resolve_selections

    topo = ROOT / "traj" / f"{cube}_ca.prmtop"
    if not topo.exists():
        pytest.skip(f"{topo} not present")
    u = mda.Universe(str(topo))
    sels = resolve_selections(u, gsa_resname="MOL", n_monomers=6, auto_tooth=False)
    sel = u.select_atoms(sels.monomer_selections[0])
    mol = sel.convert_to("RDKIT")
    assert mol is not None
    return u, sels, mol


def test_canonical_roles_match_methyl_pattern() -> None:
    pytest.importorskip("rdkit")

    from src.EndpointAnalyzer.gsa_site_map import CANONICAL_ROLES, classify_gsa_monomer

    present = [c for c in _EXPECTED_METHYL if (ROOT / "traj" / f"{c}_ca.prmtop").exists()]
    if len(present) < 2:
        pytest.skip("Need at least two B* topologies")

    role_index: dict[str, list[str]] = {}
    for cube in present:
        _, _, mol = _monomer0_mol(cube)
        gsa = classify_gsa_monomer(mol)
        assert gsa.roles() == list(CANONICAL_ROLES)
        role_index[cube] = gsa.roles()
        for role, expect_me in _EXPECTED_METHYL[cube].items():
            assert gsa.by_role(role).methyl is expect_me, f"{cube} {role}"
            if expect_me:
                assert gsa.by_role(role).ipso_rdkit != gsa.by_role(role).subst_rdkit
            else:
                assert gsa.by_role(role).ipso_rdkit == gsa.by_role(role).subst_rdkit
        py_eq = gsa.by_role("py_eq")
        assert py_eq.cpy_rdkit is not None
        assert py_eq.cpy_rdkit in py_eq.rdkit_atoms
        py_pole = gsa.by_role("py_pole")
        assert py_pole.cpy_rdkit is not None
        assert py_pole.cpy_rdkit in py_pole.rdkit_atoms

    # The bug: BMHpM vs BMMpH looked like the same 4+4 hull but s4/s5 swapped.
    if "BMHpM" in role_index and "BMMpH" in role_index:
        assert role_index["BMHpM"] == role_index["BMMpH"] == list(CANONICAL_ROLES)


def test_find_residue_sites_uses_canonical_gsa() -> None:
    pytest.importorskip("rdkit")
    import MDAnalysis as mda

    from src.EndpointAnalyzer.EndpointAnalyzer import EndpointAnalyzer
    from src.EndpointAnalyzer.gsa_site_map import CANONICAL_ROLES
    from src.utils.gsa_selections import resolve_selections

    topo = ROOT / "traj" / "BMHpM_ca.prmtop"
    if not topo.exists():
        pytest.skip("BMHpM topology not present")
    u = mda.Universe(str(topo))
    sels = resolve_selections(u, gsa_resname="MOL", n_monomers=6, auto_tooth=False)
    _, sites = EndpointAnalyzer.find_residue_endpoint_sites(u, sels.monomer_selections[0])
    assert len(sites) == len(CANONICAL_ROLES)
    # r2 / r3 / r1 are singleton atom sites (indices 3, 4, 6)
    assert [len(s) for s in sites][3] == 1
    assert [len(s) for s in sites][4] == 1
    assert [len(s) for s in sites][6] == 1


def test_paper_d1_uses_r2_r3_not_hull_s3_s7() -> None:
    pytest.importorskip("rdkit")
    import MDAnalysis as mda

    from src.ChangepointAnalysis.endpoint_features import resolve_paper_d1_atoms
    from src.EndpointAnalyzer.EndpointAnalyzer import EndpointAnalyzer
    from src.utils.gsa_selections import resolve_selections

    topo = ROOT / "traj" / "BMMpM_ca.prmtop"
    if not topo.exists():
        pytest.skip("BMMpM topology not present")
    u = mda.Universe(str(topo))
    sels = resolve_selections(u, gsa_resname="MOL", n_monomers=6, auto_tooth=False)
    stored = []
    for mon_sel in sels.monomer_selections:
        _, sites = EndpointAnalyzer.find_residue_endpoint_sites(u, mon_sel)
        stored.append(sites)

    d1 = resolve_paper_d1_atoms(
        u, sels.monomer_selections, stored, traj_id="topo"
    )
    assert set(d1["role"]) == {"r2", "r3"}
    assert len(d1) == 12
    m0 = d1[d1["monomer"] == 0].set_index("role")
    # Fully methylated: substituent (methyl C) is not the ipso ring carbon.
    assert int(m0.loc["r2", "endpoint_atom_id"]) != int(m0.loc["r2", "d1_ring_atom_id"])
    assert int(m0.loc["r3", "endpoint_atom_id"]) != int(m0.loc["r3", "d1_ring_atom_id"])
    # Old hull s3/s7 on this monomer were R1 (70/65) and a shifted R3 (112/103).
    # Canonical R2 ipso is the ring carbon of the core-C1 aryl (MDA id 36).
    assert int(m0.loc["r2", "d1_ring_atom_id"]) == 36
    assert int(m0.loc["r3", "d1_ring_atom_id"]) == 103


def test_paper_d1_hydrogen_ipso_equals_endpoint() -> None:
    pytest.importorskip("rdkit")
    import MDAnalysis as mda

    from src.ChangepointAnalysis.endpoint_features import resolve_paper_d1_atoms
    from src.EndpointAnalyzer.EndpointAnalyzer import EndpointAnalyzer
    from src.utils.gsa_selections import resolve_selections

    topo = ROOT / "traj" / "BHHpH_ca.prmtop"
    if not topo.exists():
        pytest.skip("BHHpH topology not present")
    u = mda.Universe(str(topo))
    sels = resolve_selections(u, gsa_resname="MOL", n_monomers=6, auto_tooth=False)
    stored = [
        EndpointAnalyzer.find_residue_endpoint_sites(u, s)[1]
        for s in sels.monomer_selections
    ]
    d1 = resolve_paper_d1_atoms(u, sels.monomer_selections, stored, traj_id="topo")
    assert (d1["endpoint_kind"] == "hydrogen").all()
    assert (d1["endpoint_atom_id"] == d1["d1_ring_atom_id"]).all()
