"""Tests for topology / geometry / interaction layers of endpoint detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rdkit")

from rdkit import Chem

from src.EndpointAnalyzer.endpoints_finder import EndpointsFinder
from src.EndpointAnalyzer.layered_sites import (
    CATION_PI_DIST_MAX,
    CONTACT_CUTOFF,
    PI_PI_DIST_MAX,
    PI_PI_DIST_MIN,
    compare_hull_and_layered,
    compute_site_geometry,
    contact_run_stats,
    count_contacts,
    discover_topology_sites,
    fused_ring_systems,
    hydrogen_bond_count,
    cation_pi_metrics,
    pi_pi_metrics,
    score_site_pair,
    summarize_interface_stability,
)

ROOT = Path(__file__).resolve().parent.parent


def _mol(smiles: str, add_h: bool = False):
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, smiles
    if add_h:
        mol = Chem.AddHs(mol)
    return mol


def _kinds(catalog) -> dict[str, int]:
    out: dict[str, int] = {}
    for site in catalog.sites:
        out[site.kind] = out.get(site.kind, 0) + 1
    return out


def _labels(catalog) -> set[str]:
    tags: set[str] = set()
    for site in catalog.sites:
        tags.update(site.labels)
    return tags


def _regular_hexagon(z: float = 0.0, radius: float = 1.39) -> np.ndarray:
    ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    return np.column_stack(
        [radius * np.cos(ang), radius * np.sin(ang), np.full(6, z)]
    )


# ── Layer 1 ────────────────────────────────────────────────────────────────


def test_fused_naphthalene_is_one_system() -> None:
    mol = _mol("c1ccc2ccccc2c1")
    systems = fused_ring_systems(mol)
    assert len(systems) == 1
    assert len(systems[0]) == 10


def test_biphenyl_has_two_rings_not_fused() -> None:
    mol = _mol("c1ccccc1-c2ccccc2")
    catalog = discover_topology_sites(mol)
    rings = [s for s in catalog.sites if s.kind == "fused_ring"]
    assert len(rings) == 2
    assert all(len(s.atom_indices) == 6 for s in rings)
    assert catalog.n_rotatable_bonds >= 1
    assert "phenyl" in _labels(catalog)


def test_toluene_ring_plus_methyl_no_fixed_k() -> None:
    mol = _mol("Cc1ccccc1")
    catalog = discover_topology_sites(mol)
    kinds = _kinds(catalog)
    assert kinds.get("fused_ring", 0) == 1
    assert kinds.get("functional_group", 0) >= 1
    assert any("methyl" in s.labels for s in catalog.sites)
    # Site count follows the graph: at least ring + methyl, never a quota of 8.
    assert 2 <= len(catalog.sites) < 8


def test_site_count_grows_with_substituents() -> None:
    benzene = discover_topology_sites(_mol("c1ccccc1"))
    toluene = discover_topology_sites(_mol("Cc1ccccc1"))
    xylene = discover_topology_sites(_mol("Cc1ccc(C)cc1"))
    assert len(benzene.sites) == 1
    assert len(toluene.sites) > len(benzene.sites)
    assert len(xylene.sites) > len(toluene.sites)


def test_hexaaryl_has_seven_rings_not_eight_landmarks() -> None:
    # Hexaphenylbenzene: core + six phenyls. Topology must return all seven
    # rings rather than a hull-capped landmark count.
    smiles = (
        "c1(-c2ccccc2)c(-c2ccccc2)c(-c2ccccc2)"
        "c(-c2ccccc2)c(-c2ccccc2)c1-c1ccccc1"
    )
    catalog = discover_topology_sites(_mol(smiles))
    n_rings = sum(1 for s in catalog.sites if s.kind == "fused_ring")
    assert n_rings == 7
    assert len(catalog.sites) >= 7


def test_pyridine_and_pyridinium_tags() -> None:
    pyridine = discover_topology_sites(_mol("c1ccncc1"))
    assert "pyridine" in _labels(pyridine)
    assert "pyridinium" not in _labels(pyridine)

    pyridinium = discover_topology_sites(_mol("C[n+]1ccccc1"))
    assert "pyridinium" in _labels(pyridinium)
    assert any(s.charge > 0 for s in pyridinium.sites)
    assert any("methyl" in s.labels for s in pyridinium.sites)


def test_carboxyl_is_its_own_site() -> None:
    catalog = discover_topology_sites(_mol("OC(=O)c1cccnc1"))
    assert any("carboxyl" in s.labels for s in catalog.sites)
    assert any("pyridine" in s.labels or "aromatic" in s.labels for s in catalog.sites)
    assert len(catalog.sites) >= 2


def test_connectivity_neighbors_biphenyl() -> None:
    catalog = discover_topology_sites(_mol("c1ccccc1-c2ccccc2"))
    rings = [s for s in catalog.sites if s.kind == "fused_ring"]
    assert len(rings) == 2
    assert rings[1].site_index in rings[0].neighbors
    assert rings[0].site_index in rings[1].neighbors


def test_peripheral_flag_marks_tips() -> None:
    catalog = discover_topology_sites(_mol("OC(=O)c1ccc(cc1)c2ccc(cc2)C(=O)O"))
    assert any(s.peripheral for s in catalog.sites)
    # Internal linkers may exist; at least the carboxyls should be peripheral.
    carboxyls = [s for s in catalog.sites if "carboxyl" in s.labels]
    assert carboxyls
    assert all(s.peripheral for s in carboxyls)


def test_layered_finder_ignores_ring_max_cap() -> None:
    mol = _mol(
        "c1(-c2ccccc2)c(-c2ccccc2)c(-c2ccccc2)"
        "c(-c2ccccc2)c(-c2ccccc2)c1-c1ccccc1"
    )
    hull = EndpointsFinder(ring_max_per_ring=3, site_source="hull")
    layered = EndpointsFinder(site_source="layered")
    hull_sites = hull.find_endpoint_sites(mol)
    layered_sites = layered.find_endpoint_sites(mol)
    assert len(layered_sites) == 7
    # Hull path is allowed to return a different, often smaller, set.
    assert len(hull_sites) >= 1


def test_hull_atoms_map_into_layered_sites() -> None:
    mol = _mol("OC(=O)c1ccc(cc1)c2ccc(cc2)C(=O)O")
    finder = EndpointsFinder(step_back_from_terminals=True)
    hull_atoms = finder.find_endpoints(mol)
    report = compare_hull_and_layered(mol, hull_atoms)
    assert report["n_hull_atoms"] > 0
    assert report["n_layered_sites"] >= 4  # two rings + two carboxyls
    assert report["n_hull_atoms_unmatched"] == 0


def test_layered_site_info_has_metadata() -> None:
    mol = _mol("Cc1ccccc1")
    info = EndpointsFinder(site_source="layered").find_endpoint_site_info(mol)
    assert info
    assert all("kind" in row and "labels" in row and "peripheral" in row for row in info)


# ── Layer 2 ────────────────────────────────────────────────────────────────


def test_benzene_centroid_and_plane_normal() -> None:
    coords = _regular_hexagon()
    geom = compute_site_geometry(coords, list(range(6)), aromatic=True)
    assert np.allclose(geom.centroid, [0.0, 0.0, 0.0], atol=1e-12)
    assert geom.plane_normal is not None
    assert abs(geom.plane_normal[2]) > 0.99
    assert geom.planarity_rmsd < 1e-10
    assert geom.rg > 0
    assert geom.asphericity > 0


def test_fragment_principal_axis_follows_long_axis() -> None:
    # Two points on x, one near origin — long axis should be ±x.
    coords = np.array(
        [[-2.0, 0.0, 0.0], [0.0, 0.05, 0.0], [2.0, 0.0, 0.0]], dtype=float
    )
    geom = compute_site_geometry(coords, [0, 1, 2], aromatic=False)
    axis = geom.principal_axes[0]
    assert abs(axis[0]) > 0.95


def test_layered_geometry_on_rdkit_mol() -> None:
    from rdkit.Chem import AllChem

    from src.EndpointAnalyzer.layered_sites import compute_catalog_geometries, mol_xyz

    mol = _mol("c1ccccc1")
    AllChem.Compute2DCoords(mol)
    catalog = discover_topology_sites(mol)
    mol_geom, site_geoms = compute_catalog_geometries(catalog, mol_xyz(mol))
    assert mol_geom.n_atoms == mol.GetNumAtoms()
    assert len(site_geoms) == 1
    assert site_geoms[0].plane_normal is not None
    assert site_geoms[0].n_atoms == 6


# ── Layer 3 ────────────────────────────────────────────────────────────────


def test_pi_pi_stacked_hexagons() -> None:
    a = _regular_hexagon(z=0.0)
    b = _regular_hexagon(z=3.8)
    ga = compute_site_geometry(a, list(range(6)), aromatic=True)
    gb = compute_site_geometry(b, list(range(6)), aromatic=True)
    hit, dist, angle, offset = pi_pi_metrics(
        ga.centroid, ga.plane_normal, gb.centroid, gb.plane_normal
    )
    assert hit
    assert PI_PI_DIST_MIN <= dist <= PI_PI_DIST_MAX
    assert angle < 5.0
    assert offset < 0.2


def test_pi_pi_too_far_is_not_stacking() -> None:
    a = _regular_hexagon(z=0.0)
    b = _regular_hexagon(z=8.0)
    ga = compute_site_geometry(a, list(range(6)), aromatic=True)
    gb = compute_site_geometry(b, list(range(6)), aromatic=True)
    hit, dist, _, _ = pi_pi_metrics(
        ga.centroid, ga.plane_normal, gb.centroid, gb.plane_normal
    )
    assert not hit
    assert dist > PI_PI_DIST_MAX


def test_cation_pi_above_ring() -> None:
    ring = _regular_hexagon()
    geom = compute_site_geometry(ring, list(range(6)), aromatic=True)
    cation = np.array([[0.0, 0.0, 4.0]])
    hit, dist, angle = cation_pi_metrics(cation, geom.centroid, geom.plane_normal)
    assert hit
    assert dist < CATION_PI_DIST_MAX
    assert angle < 5.0


def test_cation_pi_in_plane_is_rejected() -> None:
    ring = _regular_hexagon()
    geom = compute_site_geometry(ring, list(range(6)), aromatic=True)
    cation = np.array([[6.0, 0.0, 0.0]])
    hit, dist, angle = cation_pi_metrics(cation, geom.centroid, geom.plane_normal)
    assert dist < CATION_PI_DIST_MAX or dist > 0
    assert angle > 45.0
    assert not hit


def test_hydrogen_bond_linear_water_like() -> None:
    donor = np.array([[0.0, 0.0, 0.0]])
    hydrogen = [np.array([[0.96, 0.0, 0.0]])]
    acceptor = np.array([[2.8, 0.0, 0.0]])
    n = hydrogen_bond_count(donor, hydrogen, acceptor)
    assert n == 1

    bent_h = [np.array([[0.2, 0.94, 0.0]])]
    n_bent = hydrogen_bond_count(donor, bent_h, acceptor)
    assert n_bent == 0


def test_contact_count_cutoff() -> None:
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b_near = np.array([[0.0, 0.0, 3.0]])
    b_far = np.array([[0.0, 0.0, 8.0]])
    assert count_contacts(a, b_near, cutoff=CONTACT_CUTOFF) == 2
    assert count_contacts(a, b_far, cutoff=CONTACT_CUTOFF) == 0


def test_score_site_pair_pi_pi_and_contact() -> None:
    catalog_a = discover_topology_sites(_mol("c1ccccc1"))
    catalog_b = discover_topology_sites(_mol("c1ccccc1"))
    a = _regular_hexagon(z=0.0)
    b = _regular_hexagon(z=3.8)
    ga = compute_site_geometry(a, list(range(6)), aromatic=True)
    gb = compute_site_geometry(b, list(range(6)), aromatic=True)
    rec = score_site_pair(
        catalog_a.sites[0], ga, a, catalog_b.sites[0], gb, b
    )
    assert rec.pi_pi
    assert rec.contact
    assert rec.n_contacts > 0
    assert rec.min_distance < 4.0


def test_contact_run_stats_and_interface_stability() -> None:
    flags = np.array([0, 1, 1, 1, 0, 0, 1, 1], dtype=bool)
    stats = contact_run_stats(flags)
    assert stats["n_on"] == 5
    assert stats["n_runs"] == 2
    assert stats["mean_run_length"] == pytest.approx(2.5)
    assert stats["n_switches"] == 3
    assert stats["persistence"] == pytest.approx(5 / 8)

    from src.EndpointAnalyzer.layered_sites import InteractionRecord

    def _rec(frame_on: bool) -> InteractionRecord:
        return InteractionRecord(
            residue_i=0,
            residue_j=1,
            site_i=0,
            site_j=0,
            min_distance=3.5 if frame_on else 8.0,
            n_contacts=2 if frame_on else 0,
            contact=frame_on,
            n_hbonds=0,
            cation_pi=False,
            cation_pi_distance=float("nan"),
            cation_pi_angle=float("nan"),
            pi_pi=frame_on,
            pi_pi_distance=3.8 if frame_on else 8.0,
            pi_pi_angle=0.0,
            pi_pi_offset=0.0,
        )

    frames = [[_rec(bool(v))] for v in flags]
    rows = summarize_interface_stability(frames)
    assert len(rows) == 1
    assert rows[0]["contact_n_runs"] == 2
    assert rows[0]["pi_pi_persistence"] == pytest.approx(5 / 8)


def test_layered_interaction_observer_two_benzenes() -> None:
    mda = pytest.importorskip("MDAnalysis")

    n = 12
    n_frames = 6
    frames = []
    ring_a = _regular_hexagon(z=0.0)
    ring_b = _regular_hexagon(z=3.8)
    for t in range(n_frames):
        # First three frames stacked; last three pulled apart.
        z_shift = 0.0 if t < 3 else 5.0
        frames.append(np.vstack([ring_a, ring_b + np.array([0.0, 0.0, z_shift])]))
    xyz = np.stack(frames)

    u = mda.Universe.empty(
        n,
        n_residues=2,
        atom_resindex=[0] * 6 + [1] * 6,
        trajectory=True,
    )
    u.add_TopologyAttr("name", ["C"] * 12)
    u.add_TopologyAttr("type", ["C"] * 12)
    u.add_TopologyAttr("elements", ["C"] * 12)
    u.add_TopologyAttr("resname", ["BNZ", "BNZ"])
    u.add_TopologyAttr("resid", [1, 2])
    u.add_TopologyAttr("id", list(range(1, 13)))
    u.add_TopologyAttr("masses", [12.0] * 12)
    bonds = [(i, (i + 1) % 6) for i in range(6)]
    bonds += [(i + 6, 6 + ((i + 1) % 6)) for i in range(6)]
    u.add_TopologyAttr("bonds", bonds)
    u.load_new(xyz, order="fac")

    from src.EndpointAnalyzer.layered_sites import (
        LayeredInteractionObserver,
        mol_from_atomgroup,
    )

    try:
        mol = mol_from_atomgroup(u.select_atoms("resid 1"))
    except Exception as exc:
        pytest.skip(f"RDKit conversion of synthetic benzene failed: {exc}")
    if mol is None:
        pytest.skip("RDKit conversion returned None")

    class _Iter:
        def __init__(self, universe):
            self.universe = universe

        def get_n_frames(self):
            return len(self.universe.trajectory)

    observer = LayeredInteractionObserver(["resid 1", "resid 2"])
    iterator = _Iter(u)
    observer.on_frame_start(iterator)
    assert sum(len(c.sites) for c in observer.catalogs) >= 2
    for ts in u.trajectory:
        observer.on_frame(ts, ts.frame, u)
    observer.on_frame_end(iterator)

    frames_df = observer.get_frame_summary_df()
    assert len(frames_df) == n_frames
    # Stacked half should report π–π; separated half should not.
    pipi = frames_df["n_pi_pi"].to_numpy()
    assert pipi[:3].sum() >= 1
    assert pipi[3:].sum() == 0
    stability = observer.get_stability_df()
    assert not stability.empty
    assert float(stability["contact_n_switches"].max()) >= 1


def test_skills_registered() -> None:
    from src.mdchat.registry import get_default_registry
    import src.mdchat.skills.endpoints  # noqa: F401

    registry = get_default_registry()
    assert registry.get("find_layered_endpoints") is not None
    assert registry.get("compute_layered_interactions") is not None


def test_gsa_monomer_site_count_is_not_fixed_eight() -> None:
    """If a GSA topology is present, layered sites must follow substitution."""
    mda = pytest.importorskip("MDAnalysis")
    from src.utils.gsa_selections import resolve_selections

    counts = {}
    for cube in ("BHHpH", "BMMpM"):
        topo = ROOT / "traj" / f"{cube}_ca.prmtop"
        if not topo.exists():
            continue
        u = mda.Universe(str(topo))
        sels = resolve_selections(u, gsa_resname="MOL", n_monomers=6, auto_tooth=False)
        sel = u.select_atoms(sels.monomer_selections[0])
        mol = sel.convert_to("RDKIT")
        assert mol is not None
        catalog = discover_topology_sites(mol)
        counts[cube] = len(catalog.sites)
        assert any(s.kind == "fused_ring" for s in catalog.sites)
        # Not the hull/GSA quota of eight.
        assert len(catalog.sites) != 8 or catalog.n_fused_systems != 8

    if "BHHpH" in counts and "BMMpM" in counts:
        # Three extra methyls on BMMpM must add functional-group sites.
        assert counts["BMMpM"] > counts["BHHpH"]
    if not counts:
        pytest.skip("No B* topologies in traj/")
