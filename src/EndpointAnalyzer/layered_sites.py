"""Three-layer endpoint / fragment model for molecular assemblies.

Layer 1 — Molecular topology
    Enumerate fused ring systems, rigid fragments (rotatable-bond partition),
    and functional groups from the molecular graph.  The number of sites is
    whatever the graph contains; nothing here asks for *k* landmarks.

Layer 2 — Three-dimensional geometry
    For each site (and for the whole molecule) compute the centroid, principal
    axes, gyration-tensor shape descriptors, and — for aromatic systems —
    the ring-plane normal.

Layer 3 — Dynamic intermolecular interactions
    From those sites, score contacts, hydrogen bonds, cation–π, and π–π
    stacking between residues, and summarise interface persistence over a
    trajectory.

The 2D convex-hull finder in :mod:`endpoints_finder` remains the default
atom-tip detector.  This module is the topology-first replacement used when
``EndpointsFinder(site_source="layered")`` is selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── Geometric cutoffs (documented in docs/improve_endpoint.md) ──────────────

CONTACT_CUTOFF = 4.5  # Å, heavy-atom contact
HBOND_DA_CUTOFF = 3.5  # Å, donor–acceptor heavy-atom distance
HBOND_ANGLE_MIN = 120.0  # degrees, D–H–A
CATION_PI_DIST_MAX = 6.5  # Å, Murata-style cation–π distance
CATION_PI_ANGLE_MAX = 45.0  # degrees, |90 − angle(normal, cation−centroid)| wait: angle between cation-centroid vector and normal
PI_PI_DIST_MIN = 3.0  # Å
PI_PI_DIST_MAX = 5.5  # Å
PI_PI_PARALLEL_COS = float(np.cos(np.deg2rad(30.0)))  # |n·n'| ≥ this → sandwich
PI_PI_OFFSET_MAX = 2.0  # Å, in-plane displacement for a stacked pair

# Lipinski-style rotatable bond: single, non-ring, both atoms heavy & not terminal.
_ROTATABLE_SMARTS = "[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]"

# Functional-group SMARTS.  Matches wholly inside a ring tag that ring;
# matches that extend outside become their own sites.
FUNCTIONAL_GROUP_SMARTS: Tuple[Tuple[str, str], ...] = (
    ("pyridinium", "[nX3+,nX4+,NX4+]"),
    ("ammonium", "[NX4+]"),
    ("carboxylate", "C(=O)[O-]"),
    ("carboxyl", "C(=O)[OH]"),
    ("nitro", "[N+](=O)[O-]"),
    ("cyano", "C#N"),
    ("trifluoromethyl", "[CX4](F)(F)F"),
    ("methyl", "[CH3]"),
    ("hydroxyl", "[OX2H]"),
    ("amine", "[NX3;H2,H1;!$(NC=O)]"),
    ("carbonyl", "[CX3]=O"),
    ("ether", "[OD2]([#6])[#6]"),
)


# ═══════════════════════════════════════════════════════════════════════════
#  Data types
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TopologySite:
    """One chemically defined site (variable count; no landmark quota)."""

    site_index: int
    kind: str  # fused_ring | rigid_fragment | functional_group
    atom_indices: List[int]
    labels: List[str] = field(default_factory=list)
    aromatic: bool = False
    charge: int = 0
    peripheral: bool = False
    neighbors: List[int] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.atom_indices)

    @property
    def representative_atom(self) -> int:
        return int(min(self.atom_indices)) if self.atom_indices else -1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "site_index": self.site_index,
            "kind": self.kind,
            "atom_indices": list(self.atom_indices),
            "n_atoms": self.n_atoms,
            "labels": list(self.labels),
            "aromatic": bool(self.aromatic),
            "charge": int(self.charge),
            "peripheral": bool(self.peripheral),
            "neighbors": list(self.neighbors),
            "representative_atom": self.representative_atom,
        }


@dataclass
class TopologyCatalog:
    """Layer-1 result: sites plus the site–site connectivity graph."""

    sites: List[TopologySite]
    n_rotatable_bonds: int = 0
    n_fused_systems: int = 0
    n_rigid_fragments: int = 0
    n_functional_groups: int = 0

    def atom_groups(self) -> List[List[int]]:
        return [list(s.atom_indices) for s in self.sites]

    def to_records(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.sites]


@dataclass
class SiteGeometry:
    """Layer-2 descriptors of one site in a single conformation."""

    centroid: np.ndarray
    principal_axes: np.ndarray  # (3, 3) rows = axes, descending eigenvalue
    eigenvalues: np.ndarray
    rg: float
    asphericity: float
    acylindricity: float
    relative_shape_anisotropy: float
    plane_normal: Optional[np.ndarray] = None
    planarity_rmsd: float = float("nan")
    n_atoms: int = 0

    def to_dict(self, prefix: str = "") -> Dict[str, float]:
        p = prefix
        out: Dict[str, float] = {
            f"{p}centroid_x": float(self.centroid[0]),
            f"{p}centroid_y": float(self.centroid[1]),
            f"{p}centroid_z": float(self.centroid[2]),
            f"{p}axis1_x": float(self.principal_axes[0, 0]),
            f"{p}axis1_y": float(self.principal_axes[0, 1]),
            f"{p}axis1_z": float(self.principal_axes[0, 2]),
            f"{p}rg": float(self.rg),
            f"{p}asphericity": float(self.asphericity),
            f"{p}acylindricity": float(self.acylindricity),
            f"{p}relative_shape_anisotropy": float(self.relative_shape_anisotropy),
            f"{p}planarity_rmsd": float(self.planarity_rmsd),
            f"{p}n_atoms": float(self.n_atoms),
        }
        if self.plane_normal is not None:
            n = self.plane_normal
            out[f"{p}normal_x"] = float(n[0])
            out[f"{p}normal_y"] = float(n[1])
            out[f"{p}normal_z"] = float(n[2])
        else:
            out[f"{p}normal_x"] = float("nan")
            out[f"{p}normal_y"] = float("nan")
            out[f"{p}normal_z"] = float("nan")
        return out


@dataclass
class MoleculeGeometry:
    centroid: np.ndarray
    principal_axes: np.ndarray
    eigenvalues: np.ndarray
    rg: float
    asphericity: float
    acylindricity: float
    relative_shape_anisotropy: float
    n_atoms: int


@dataclass
class InteractionRecord:
    """One residue-i site-a ↔ residue-j site-b interaction in one frame."""

    residue_i: int
    residue_j: int
    site_i: int
    site_j: int
    min_distance: float
    n_contacts: int
    contact: bool
    n_hbonds: int
    cation_pi: bool
    cation_pi_distance: float
    cation_pi_angle: float
    pi_pi: bool
    pi_pi_distance: float
    pi_pi_angle: float
    pi_pi_offset: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "residue_i": self.residue_i,
            "residue_j": self.residue_j,
            "site_i": self.site_i,
            "site_j": self.site_j,
            "min_distance": self.min_distance,
            "n_contacts": self.n_contacts,
            "contact": int(self.contact),
            "n_hbonds": self.n_hbonds,
            "cation_pi": int(self.cation_pi),
            "cation_pi_distance": self.cation_pi_distance,
            "cation_pi_angle": self.cation_pi_angle,
            "pi_pi": int(self.pi_pi),
            "pi_pi_distance": self.pi_pi_distance,
            "pi_pi_angle": self.pi_pi_angle,
            "pi_pi_offset": self.pi_pi_offset,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 1 — topology
# ═══════════════════════════════════════════════════════════════════════════


def _require_rdkit():
    from rdkit import Chem  # noqa: F401

    return Chem


def mol_from_atomgroup(sel: Any):
    """Convert an MDAnalysis AtomGroup to RDKit, allowing implicit H if needed."""
    try:
        mol = sel.convert_to("RDKIT")
    except TypeError:
        mol = sel.convert_to("RDKIT", NoImplicit=False)
    except Exception:
        mol = sel.convert_to("RDKIT", NoImplicit=False)
    if mol is None:
        raise ValueError("RDKit conversion failed")
    return mol


def fused_ring_systems(mol: Any) -> List[List[int]]:
    """Merge rings that share atoms into fused systems (union-find)."""
    rings = list(mol.GetRingInfo().AtomRings())
    if not rings:
        return []

    parent = list(range(len(rings)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    atom_to_rings: Dict[int, List[int]] = {}
    for rid, ring in enumerate(rings):
        for atom_idx in ring:
            atom_to_rings.setdefault(int(atom_idx), []).append(rid)
    for rids in atom_to_rings.values():
        for k in range(1, len(rids)):
            union(rids[0], rids[k])

    systems: Dict[int, set[int]] = {}
    for rid, ring in enumerate(rings):
        systems.setdefault(find(rid), set()).update(int(a) for a in ring)
    return [sorted(atoms) for atoms in systems.values()]


def rotatable_bonds(mol: Any) -> List[Tuple[int, int]]:
    """Heavy-atom rotatable bonds (Lipinski SMARTS)."""
    Chem = _require_rdkit()
    pattern = Chem.MolFromSmarts(_ROTATABLE_SMARTS)
    if pattern is None:
        return []
    hits = mol.GetSubstructMatches(pattern)
    return [(int(a), int(b)) for (a, b) in hits]


def rigid_heavy_fragments(mol: Any) -> List[List[int]]:
    """Connected components of the heavy-atom graph after cutting rotatable bonds."""
    rot = {frozenset(pair) for pair in rotatable_bonds(mol)}
    heavy = [
        int(atom.GetIdx())
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]
    if not heavy:
        return []
    heavy_set = set(heavy)
    parent = {i: i for i in heavy}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for atom in mol.GetAtoms():
        i = int(atom.GetIdx())
        if i not in heavy_set:
            continue
        for nbr in atom.GetNeighbors():
            j = int(nbr.GetIdx())
            if j not in heavy_set or j < i:
                continue
            if frozenset((i, j)) in rot:
                continue
            union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in heavy:
        groups.setdefault(find(i), []).append(i)
    return [sorted(atoms) for atoms in groups.values()]


def _site_charge(mol: Any, atoms: Sequence[int]) -> int:
    return int(sum(int(mol.GetAtomWithIdx(int(i)).GetFormalCharge()) for i in atoms))


def _all_aromatic(mol: Any, atoms: Sequence[int]) -> bool:
    if not atoms:
        return False
    return all(mol.GetAtomWithIdx(int(i)).GetIsAromatic() for i in atoms)


def _classify_ring(mol: Any, atoms: Sequence[int]) -> List[str]:
    labels: List[str] = []
    atom_objs = [mol.GetAtomWithIdx(int(i)) for i in atoms]
    symbols = [a.GetSymbol() for a in atom_objs]
    aromatic = _all_aromatic(mol, atoms)
    if aromatic:
        labels.append("aromatic")
    n_atoms = [a for a in atom_objs if a.GetSymbol() == "N"]
    if any(int(a.GetFormalCharge()) > 0 for a in n_atoms) or any(
        a.GetSymbol() == "N" and a.GetIsAromatic() and a.GetDegree() == 3
        for a in atom_objs
    ):
        labels.append("pyridinium")
    elif any(a.GetSymbol() == "N" and a.GetIsAromatic() for a in atom_objs):
        labels.append("pyridine")
    hetero = {s for s in symbols if s not in ("C", "H")}
    if aromatic and len(atoms) == 6 and not hetero:
        labels.append("phenyl")
    elif hetero:
        labels.append("heteroaryl" if aromatic else "heterocycle")
    if len(atoms) > 6:
        labels.append("fused")
    return labels


def _heavy_atoms(mol: Any, atoms: Sequence[int]) -> List[int]:
    return [
        int(i)
        for i in atoms
        if mol.GetAtomWithIdx(int(i)).GetAtomicNum() > 1
    ]


def _tag_site(site: TopologySite, label: str) -> None:
    if label not in site.labels:
        site.labels.append(label)


def _site_containing(sites: List[TopologySite], atoms: Sequence[int]) -> Optional[int]:
    target = set(int(i) for i in atoms)
    if not target:
        return None
    for site in sites:
        if target <= set(site.atom_indices):
            return site.site_index
    return None


def _coords_2d(mol: Any) -> np.ndarray:
    from rdkit.Chem import AllChem

    m = type(mol)(mol)
    AllChem.Compute2DCoords(m)
    conf = m.GetConformer()
    return np.array(
        [
            [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
            for i in range(m.GetNumAtoms())
        ],
        dtype=float,
    )


def _convex_hull_indices(xy: np.ndarray) -> List[int]:
    if len(xy) == 0:
        return []
    if len(xy) <= 2:
        return list(range(len(xy)))
    try:
        from scipy.spatial import ConvexHull

        if len(np.unique(xy, axis=0)) < 3:
            return list(range(len(xy)))
        hull = ConvexHull(xy)
        return sorted(set(int(v) for v in hull.vertices.tolist()))
    except Exception:
        return list(range(len(xy)))


def _site_eccentricities(n: int, edges: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Graph eccentricity of each node; isolated nodes get 0."""
    adj: List[List[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    ecc = np.zeros(n, dtype=int)
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = [src]
        far = 0
        while q:
            u = q.pop(0)
            far = max(far, dist[u])
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1
                    q.append(v)
        ecc[src] = far
    return ecc


def discover_topology_sites(mol: Any) -> TopologyCatalog:
    """Layer 1: enumerate rings, rigid fragments, and functional groups.

    The site count is determined by the molecular graph.  No *k* is imposed.
    """
    Chem = _require_rdkit()
    if mol is None:
        raise ValueError("mol is None")
    # Work on a copy so Sanitize/ring perception cannot mutate the caller.
    work = Chem.Mol(mol)

    ring_systems = fused_ring_systems(work)
    sites: List[TopologySite] = []
    covered: set[int] = set()

    for atoms in ring_systems:
        labels = _classify_ring(work, atoms)
        site = TopologySite(
            site_index=len(sites),
            kind="fused_ring",
            atom_indices=list(atoms),
            labels=labels,
            aromatic="aromatic" in labels,
            charge=_site_charge(work, atoms),
        )
        sites.append(site)
        covered.update(atoms)

    n_fused = len(sites)

    # Functional groups: tag containing ring, or add a new site.
    n_fg = 0
    for name, smarts in FUNCTIONAL_GROUP_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        for match in work.GetSubstructMatches(pattern):
            heavy = _heavy_atoms(work, match)
            if not heavy:
                continue
            host = _site_containing(sites, heavy)
            if host is not None:
                _tag_site(sites[host], name)
                if name in ("pyridinium", "ammonium"):
                    sites[host].charge = max(sites[host].charge, 1)
                continue
            # Skip if this match is a subset of a larger FG already added
            # (e.g. methyl inside trifluoromethyl, carbonyl inside carboxyl).
            if _site_containing(sites, heavy) is not None:
                continue
            site = TopologySite(
                site_index=len(sites),
                kind="functional_group",
                atom_indices=list(heavy),
                labels=[name],
                aromatic=False,
                charge=_site_charge(work, heavy),
            )
            sites.append(site)
            covered.update(heavy)
            n_fg += 1

    rot_bonds = rotatable_bonds(work)
    fragments = rigid_heavy_fragments(work)
    n_rigid_added = 0
    for frag in fragments:
        leftover = [i for i in frag if i not in covered]
        if not leftover:
            continue
        site = TopologySite(
            site_index=len(sites),
            kind="rigid_fragment",
            atom_indices=list(leftover),
            labels=["rigid"],
            aromatic=_all_aromatic(work, leftover),
            charge=_site_charge(work, leftover),
        )
        sites.append(site)
        covered.update(leftover)
        n_rigid_added += 1

    # Connectivity: sites share a bond.
    atom_to_site: Dict[int, int] = {}
    for site in sites:
        for a in site.atom_indices:
            atom_to_site[int(a)] = site.site_index

    edges: set[Tuple[int, int]] = set()
    for bond in work.GetBonds():
        a = int(bond.GetBeginAtomIdx())
        b = int(bond.GetEndAtomIdx())
        sa = atom_to_site.get(a)
        sb = atom_to_site.get(b)
        if sa is None or sb is None or sa == sb:
            continue
        lo, hi = (sa, sb) if sa < sb else (sb, sa)
        edges.add((lo, hi))
    for sa, sb in edges:
        sites[sa].neighbors.append(sb)
        sites[sb].neighbors.append(sa)
    for site in sites:
        site.neighbors = sorted(set(site.neighbors))

    # Peripheral: high eccentricity in the site graph, or on the 2D hull of
    # site centroids.  Isolated sites are peripheral by default.
    n = len(sites)
    if n == 0:
        return TopologyCatalog(sites=[], n_rotatable_bonds=len(rot_bonds))
    if n == 1:
        sites[0].peripheral = True
    else:
        ecc = _site_eccentricities(n, sorted(edges))
        radius = int(ecc.max()) if n else 0
        xy = _coords_2d(work)
        centroids = []
        for site in sites:
            pts = xy[site.atom_indices]
            centroids.append(pts.mean(axis=0) if len(pts) else np.zeros(2))
        centroids_arr = np.asarray(centroids, dtype=float)
        hull_sites = set(_convex_hull_indices(centroids_arr))
        for site in sites:
            leaf = (not site.neighbors) or (int(ecc[site.site_index]) >= radius and radius > 0)
            site.peripheral = bool(leaf or site.site_index in hull_sites)

    sites.sort(key=lambda s: (min(s.atom_indices) if s.atom_indices else 0, s.kind, s.n_atoms))
    # Reindex after sort so site_index matches list order (deterministic columns).
    remap = {old.site_index: new_i for new_i, old in enumerate(sites)}
    for new_i, site in enumerate(sites):
        site.site_index = new_i
        site.neighbors = sorted(remap[j] for j in site.neighbors if j in remap)

    return TopologyCatalog(
        sites=sites,
        n_rotatable_bonds=len(rot_bonds),
        n_fused_systems=n_fused,
        n_rigid_fragments=n_rigid_added + n_fused,
        n_functional_groups=n_fg,
    )


def compare_hull_and_layered(
    mol: Any,
    hull_atom_ids: Sequence[int],
) -> Dict[str, Any]:
    """How hull atoms sit inside the topology catalog (no *k* alignment)."""
    catalog = discover_topology_sites(mol)
    hull_set = set(int(i) for i in hull_atom_ids)
    covered = set()
    host_counts: Dict[int, int] = {}
    unmatched: List[int] = []
    for atom in hull_set:
        host = None
        for site in catalog.sites:
            if atom in site.atom_indices:
                host = site.site_index
                break
        if host is None:
            unmatched.append(atom)
        else:
            covered.add(atom)
            host_counts[host] = host_counts.get(host, 0) + 1
    return {
        "n_hull_atoms": len(hull_set),
        "n_layered_sites": len(catalog.sites),
        "n_hull_atoms_in_a_site": len(covered),
        "n_hull_atoms_unmatched": len(unmatched),
        "n_layered_sites_hosting_hull": len(host_counts),
        "kind_counts": _count_kinds(catalog),
        "unmatched_hull_atoms": unmatched,
    }


def _count_kinds(catalog: TopologyCatalog) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for site in catalog.sites:
        out[site.kind] = out.get(site.kind, 0) + 1
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 2 — 3D geometry
# ═══════════════════════════════════════════════════════════════════════════


def gyration_descriptors(coords: np.ndarray) -> Dict[str, Any]:
    """Centroid, principal axes, Rg, asphericity, acylindricity, κ²."""
    pts = np.asarray(coords, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got {pts.shape}")
    if len(pts) == 0:
        nan3 = np.full(3, np.nan)
        nan33 = np.full((3, 3), np.nan)
        return {
            "centroid": nan3,
            "principal_axes": nan33,
            "eigenvalues": nan3,
            "rg": float("nan"),
            "asphericity": float("nan"),
            "acylindricity": float("nan"),
            "relative_shape_anisotropy": float("nan"),
        }
    centroid = pts.mean(axis=0)
    d = pts - centroid
    if len(pts) == 1:
        axes = np.eye(3)
        eigvals = np.zeros(3)
        return {
            "centroid": centroid,
            "principal_axes": axes,
            "eigenvalues": eigvals,
            "rg": 0.0,
            "asphericity": 0.0,
            "acylindricity": 0.0,
            "relative_shape_anisotropy": 0.0,
        }
    cov = (d.T @ d) / len(pts)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    axes = eigvecs[:, order].T  # rows = axes
    # Right-handed frame
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    rg2 = float(eigvals.sum())
    rg = float(np.sqrt(max(rg2, 0.0)))
    l1, l2, l3 = (float(eigvals[2]), float(eigvals[1]), float(eigvals[0]))  # λ1≤λ2≤λ3
    asphericity = l3 - 0.5 * (l1 + l2)
    acylindricity = l2 - l1
    denom = rg2 * rg2 if rg2 > 0 else 1.0
    kappa2 = (asphericity**2 + 0.75 * acylindricity**2) / denom
    return {
        "centroid": centroid,
        "principal_axes": axes,
        "eigenvalues": eigvals,
        "rg": rg,
        "asphericity": float(asphericity),
        "acylindricity": float(acylindricity),
        "relative_shape_anisotropy": float(kappa2),
    }


def plane_normal_and_rmsd(coords: np.ndarray) -> Tuple[np.ndarray, float]:
    """Best-fit plane normal (unit) and RMSD of points from that plane."""
    pts = np.asarray(coords, dtype=float)
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0]), float("nan")
    desc = gyration_descriptors(pts)
    # Smallest-eigenvalue axis is the plane normal.
    normal = np.asarray(desc["principal_axes"][2], dtype=float)
    nrm = np.linalg.norm(normal)
    if nrm < 1e-12:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / nrm
    # Flip so z-component (or first nonzero) is non-negative — deterministic sign.
    for k in (2, 1, 0):
        if abs(normal[k]) > 1e-12:
            if normal[k] < 0:
                normal = -normal
            break
    d = pts - desc["centroid"]
    rmsd = float(np.sqrt(np.mean((d @ normal) ** 2)))
    return normal, rmsd


def mol_xyz(mol: Any) -> np.ndarray:
    """(N, 3) coordinates from the RDKit conformer; z=0 if only 2D is present."""
    if mol.GetNumConformers() == 0:
        from rdkit.Chem import AllChem

        AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    xyz = np.zeros((n, 3), dtype=float)
    for i in range(n):
        p = conf.GetAtomPosition(i)
        xyz[i] = [p.x, p.y, p.z]
    return xyz


def compute_site_geometry(
    coords: np.ndarray,
    atom_indices: Sequence[int],
    aromatic: bool = False,
) -> SiteGeometry:
    """Layer 2 for one site.  Aromatic sites always get a plane normal."""
    idx = [int(i) for i in atom_indices]
    if not idx:
        empty = gyration_descriptors(np.zeros((0, 3)))
        return SiteGeometry(
            centroid=empty["centroid"],
            principal_axes=empty["principal_axes"],
            eigenvalues=empty["eigenvalues"],
            rg=empty["rg"],
            asphericity=empty["asphericity"],
            acylindricity=empty["acylindricity"],
            relative_shape_anisotropy=empty["relative_shape_anisotropy"],
            n_atoms=0,
        )
    pts = np.asarray(coords, dtype=float)[idx]
    desc = gyration_descriptors(pts)
    normal = None
    planarity = float("nan")
    if aromatic or len(idx) >= 3:
        normal, planarity = plane_normal_and_rmsd(pts)
        if not aromatic:
            # Keep the normal only when the site is reasonably planar.
            span = float(np.sqrt(max(desc["rg"], 0.0)))
            if not (planarity < 0.25 or (span > 0 and planarity / max(span, 1e-6) < 0.2)):
                normal = None
    return SiteGeometry(
        centroid=desc["centroid"],
        principal_axes=desc["principal_axes"],
        eigenvalues=desc["eigenvalues"],
        rg=desc["rg"],
        asphericity=desc["asphericity"],
        acylindricity=desc["acylindricity"],
        relative_shape_anisotropy=desc["relative_shape_anisotropy"],
        plane_normal=normal,
        planarity_rmsd=planarity,
        n_atoms=len(idx),
    )


def compute_molecule_geometry(coords: np.ndarray) -> MoleculeGeometry:
    desc = gyration_descriptors(coords)
    return MoleculeGeometry(
        centroid=desc["centroid"],
        principal_axes=desc["principal_axes"],
        eigenvalues=desc["eigenvalues"],
        rg=desc["rg"],
        asphericity=desc["asphericity"],
        acylindricity=desc["acylindricity"],
        relative_shape_anisotropy=desc["relative_shape_anisotropy"],
        n_atoms=int(len(coords)),
    )


def compute_catalog_geometries(
    catalog: TopologyCatalog,
    coords: np.ndarray,
) -> Tuple[MoleculeGeometry, List[SiteGeometry]]:
    mol_geom = compute_molecule_geometry(coords)
    site_geoms = [
        compute_site_geometry(coords, site.atom_indices, aromatic=site.aromatic)
        for site in catalog.sites
    ]
    return mol_geom, site_geoms


# ═══════════════════════════════════════════════════════════════════════════
#  Layer 3 — interactions
# ═══════════════════════════════════════════════════════════════════════════


def min_pairwise_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    if len(coords_a) == 0 or len(coords_b) == 0:
        return float("nan")
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    return float(np.sqrt(np.min(np.sum(diff * diff, axis=2))))


def count_contacts(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    cutoff: float = CONTACT_CUTOFF,
) -> int:
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return int(np.sum(d2 < cutoff * cutoff))


def hydrogen_bond_count(
    donor_heavy: np.ndarray,
    donor_hydrogens: Sequence[np.ndarray],
    acceptor_heavy: np.ndarray,
    da_cutoff: float = HBOND_DA_CUTOFF,
    angle_min: float = HBOND_ANGLE_MIN,
) -> int:
    """Count D–H···A hydrogen bonds with a distance + angle criterion.

    *donor_hydrogens[k]* is an ``(n_H, 3)`` array of hydrogens bound to
    donor *k*.  Empty arrays skip that donor.
    """
    if len(donor_heavy) == 0 or len(acceptor_heavy) == 0:
        return 0
    if len(donor_hydrogens) != len(donor_heavy):
        raise ValueError("donor_hydrogens must align with donor_heavy")
    n = 0
    cos_min = float(np.cos(np.deg2rad(angle_min)))
    for d_xyz, h_xyz in zip(donor_heavy, donor_hydrogens):
        if h_xyz is None or len(h_xyz) == 0:
            continue
        for a_xyz in acceptor_heavy:
            if np.linalg.norm(a_xyz - d_xyz) > da_cutoff:
                continue
            for h in h_xyz:
                u = d_xyz - h
                v = a_xyz - h
                nu = np.linalg.norm(u)
                nv = np.linalg.norm(v)
                if nu < 1e-8 or nv < 1e-8:
                    continue
                cos = float(np.dot(u, v) / (nu * nv))
                # D–H–A angle: cos of angle at H.  We want angle ≥ angle_min,
                # i.e. cos ≤ cos(angle_min) for angle_min > 90.
                if cos <= cos_min:
                    n += 1
    return n


def cation_pi_metrics(
    cation_xyz: np.ndarray,
    aromatic_centroid: np.ndarray,
    aromatic_normal: np.ndarray,
    dist_max: float = CATION_PI_DIST_MAX,
    angle_max: float = CATION_PI_ANGLE_MAX,
) -> Tuple[bool, float, float]:
    """Return ``(is_cation_pi, distance, angle_to_normal_deg)``.

    *angle_to_normal_deg* is the angle between the cation→centroid vector and
    the ring normal (0° = cation sits on the normal, above the ring face).
    """
    cation_xyz = np.asarray(cation_xyz, dtype=float).reshape(-1, 3)
    if len(cation_xyz) == 0:
        return False, float("nan"), float("nan")
    vecs = cation_xyz - aromatic_centroid
    dists = np.linalg.norm(vecs, axis=1)
    k = int(np.argmin(dists))
    dist = float(dists[k])
    vec = vecs[k]
    nv = np.linalg.norm(vec)
    nn = np.linalg.norm(aromatic_normal)
    if nv < 1e-8 or nn < 1e-8:
        return False, dist, float("nan")
    cos = float(np.abs(np.dot(vec, aromatic_normal) / (nv * nn)))
    angle = float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))
    hit = bool(dist <= dist_max and angle <= angle_max)
    return hit, dist, angle


def pi_pi_metrics(
    centroid_a: np.ndarray,
    normal_a: np.ndarray,
    centroid_b: np.ndarray,
    normal_b: np.ndarray,
    dist_min: float = PI_PI_DIST_MIN,
    dist_max: float = PI_PI_DIST_MAX,
    parallel_cos: float = PI_PI_PARALLEL_COS,
    offset_max: float = PI_PI_OFFSET_MAX,
) -> Tuple[bool, float, float, float]:
    """Return ``(is_pi_pi, centroid_distance, angle_deg, in_plane_offset)``."""
    delta = np.asarray(centroid_b, dtype=float) - np.asarray(centroid_a, dtype=float)
    dist = float(np.linalg.norm(delta))
    na = np.asarray(normal_a, dtype=float)
    nb = np.asarray(normal_b, dtype=float)
    na = na / max(np.linalg.norm(na), 1e-12)
    nb = nb / max(np.linalg.norm(nb), 1e-12)
    cos = float(np.clip(np.abs(np.dot(na, nb)), 0.0, 1.0))
    angle = float(np.degrees(np.arccos(cos)))
    # Offset: component of delta orthogonal to the mean normal.
    mean_n = na + nb
    mn = np.linalg.norm(mean_n)
    mean_n = mean_n / mn if mn > 1e-12 else na
    parallel = float(np.dot(delta, mean_n))
    offset = float(np.sqrt(max(dist * dist - parallel * parallel, 0.0)))
    hit = bool(
        dist_min <= dist <= dist_max
        and cos >= parallel_cos
        and offset <= offset_max
    )
    return hit, dist, angle, offset


def _pi_stack_eligible(site: TopologySite, geom: SiteGeometry) -> bool:
    """Aromatic RDKit rings, or sufficiently planar ≥5-atom sites (for H-less MDA)."""
    if geom.plane_normal is None:
        return False
    if site.aromatic:
        return True
    return geom.n_atoms >= 5 and (
        geom.planarity_rmsd < 0.25 or np.isnan(geom.planarity_rmsd)
    )


def _cation_points(
    site: TopologySite,
    geom: SiteGeometry,
    coords: np.ndarray,
    mol: Optional[Any] = None,
) -> np.ndarray:
    """Points that can act as the cation in cation–π (N+ / charged site)."""
    if site.charge <= 0 and "pyridinium" not in site.labels and "ammonium" not in site.labels:
        return np.zeros((0, 3))
    if mol is None:
        return geom.centroid.reshape(1, 3)
    pts = []
    for i in site.atom_indices:
        atom = mol.GetAtomWithIdx(int(i))
        if atom.GetFormalCharge() > 0 or (
            atom.GetSymbol() == "N" and "pyridinium" in site.labels
        ):
            pts.append(coords[int(i)])
    if not pts:
        return geom.centroid.reshape(1, 3)
    return np.asarray(pts, dtype=float)


def score_site_pair(
    site_i: TopologySite,
    geom_i: SiteGeometry,
    coords_i: np.ndarray,
    site_j: TopologySite,
    geom_j: SiteGeometry,
    coords_j: np.ndarray,
    residue_i: int = 0,
    residue_j: int = 1,
    mol_i: Optional[Any] = None,
    mol_j: Optional[Any] = None,
    hbond_count: int = 0,
    contact_cutoff: float = CONTACT_CUTOFF,
) -> InteractionRecord:
    """Layer-3 scores for one site pair in one frame."""
    min_d = min_pairwise_distance(coords_i, coords_j)
    n_contacts = count_contacts(coords_i, coords_j, cutoff=contact_cutoff)

    cat_pi = False
    cat_d = float("nan")
    cat_ang = float("nan")
    # Try both directions: cation on i vs aromatic on j, and vice versa.
    candidates: List[Tuple[bool, float, float]] = []
    if _pi_stack_eligible(site_j, geom_j):
        pts = _cation_points(site_i, geom_i, coords_i, mol_i)
        if len(pts):
            candidates.append(
                cation_pi_metrics(pts, geom_j.centroid, geom_j.plane_normal)
            )
    if _pi_stack_eligible(site_i, geom_i):
        pts = _cation_points(site_j, geom_j, coords_j, mol_j)
        if len(pts):
            candidates.append(
                cation_pi_metrics(pts, geom_i.centroid, geom_i.plane_normal)
            )
    if candidates:
        # Prefer a hit; otherwise the closest approach.
        hits = [c for c in candidates if c[0]]
        pick = min(hits or candidates, key=lambda c: c[1])
        cat_pi, cat_d, cat_ang = pick

    pipi = False
    pipi_d = float("nan")
    pipi_ang = float("nan")
    pipi_off = float("nan")
    if _pi_stack_eligible(site_i, geom_i) and _pi_stack_eligible(site_j, geom_j):
        pipi, pipi_d, pipi_ang, pipi_off = pi_pi_metrics(
            geom_i.centroid,
            geom_i.plane_normal,
            geom_j.centroid,
            geom_j.plane_normal,
        )

    return InteractionRecord(
        residue_i=int(residue_i),
        residue_j=int(residue_j),
        site_i=int(site_i.site_index),
        site_j=int(site_j.site_index),
        min_distance=min_d,
        n_contacts=n_contacts,
        contact=bool(n_contacts > 0),
        n_hbonds=int(hbond_count),
        cation_pi=bool(cat_pi),
        cation_pi_distance=cat_d,
        cation_pi_angle=cat_ang,
        pi_pi=bool(pipi),
        pi_pi_distance=pipi_d,
        pi_pi_angle=pipi_ang,
        pi_pi_offset=pipi_off,
    )


def donor_acceptor_heavy_indices(mol: Any) -> Tuple[List[int], List[int]]:
    """N/O donors (have H) and N/O acceptors from the RDKit molecule."""
    donors: List[int] = []
    acceptors: List[int] = []
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z not in (7, 8):
            continue
        idx = int(atom.GetIdx())
        acceptors.append(idx)
        n_h = int(atom.GetTotalNumHs())
        if n_h == 0:
            n_h = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
        if n_h > 0:
            donors.append(idx)
    return donors, acceptors


def hydrogen_positions_for_donors(
    mol: Any,
    coords: np.ndarray,
    donor_indices: Sequence[int],
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for d in donor_indices:
        hs = [
            coords[int(n.GetIdx())]
            for n in mol.GetAtomWithIdx(int(d)).GetNeighbors()
            if n.GetAtomicNum() == 1
        ]
        out.append(np.asarray(hs, dtype=float) if hs else np.zeros((0, 3)))
    return out


def score_residue_pair_interactions(
    catalog_i: TopologyCatalog,
    geoms_i: Sequence[SiteGeometry],
    coords_i: np.ndarray,
    catalog_j: TopologyCatalog,
    geoms_j: Sequence[SiteGeometry],
    coords_j: np.ndarray,
    residue_i: int,
    residue_j: int,
    mol_i: Optional[Any] = None,
    mol_j: Optional[Any] = None,
    contact_cutoff: float = CONTACT_CUTOFF,
) -> List[InteractionRecord]:
    """All site–site interaction records between two residues in one frame.

    ``coords_i`` / ``coords_j`` are each ``(n_atoms, 3)`` in that residue's
    RDKit atom-index order.
    """
    hbond_by_sites: Dict[Tuple[int, int], int] = {}
    if mol_i is not None and mol_j is not None:
        hbond_by_sites = _hbonds_by_site_pair(
            mol_i, catalog_i, coords_i, mol_j, catalog_j, coords_j
        )

    records: List[InteractionRecord] = []
    for si, gi in zip(catalog_i.sites, geoms_i):
        ci = coords_i[si.atom_indices]
        for sj, gj in zip(catalog_j.sites, geoms_j):
            cj = coords_j[sj.atom_indices]
            hb = hbond_by_sites.get((si.site_index, sj.site_index), 0)
            records.append(
                score_site_pair(
                    si, gi, ci, sj, gj, cj,
                    residue_i=residue_i,
                    residue_j=residue_j,
                    mol_i=mol_i,
                    mol_j=mol_j,
                    hbond_count=hb,
                    contact_cutoff=contact_cutoff,
                )
            )
    return records


def _atom_to_site_map(catalog: TopologyCatalog) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    for site in catalog.sites:
        for a in site.atom_indices:
            mapping[int(a)] = site.site_index
    return mapping


def _hbonds_by_site_pair(
    mol_i: Any,
    cat_i: TopologyCatalog,
    coords_i: np.ndarray,
    mol_j: Any,
    cat_j: TopologyCatalog,
    coords_j: np.ndarray,
) -> Dict[Tuple[int, int], int]:
    """H-bond counts keyed by (site_i, site_j).  mol_i/j share *coords* layout
    only when they are slices of one molecule; for two residues the caller
    must pass coordinates in each mol's index space.

    Here each mol has its own coords array of length ``mol.GetNumAtoms()``.
    """
    # The function signature takes coords_i / coords_j separately — use them.
    counts: Dict[Tuple[int, int], int] = {}
    map_i = _atom_to_site_map(cat_i)
    map_j = _atom_to_site_map(cat_j)

    def _count(mol_d, coords_d, map_d, mol_a, coords_a, map_a, swap: bool) -> None:
        donors, _ = donor_acceptor_heavy_indices(mol_d)
        _, acceptors = donor_acceptor_heavy_indices(mol_a)
        if not donors or not acceptors:
            return
        hpos = hydrogen_positions_for_donors(mol_d, coords_d, donors)
        for di, d_idx in enumerate(donors):
            d_site = map_d.get(int(d_idx))
            if d_site is None:
                continue
            d_xyz = coords_d[int(d_idx)].reshape(1, 3)
            hs = hpos[di]
            for a_idx in acceptors:
                a_site = map_a.get(int(a_idx))
                if a_site is None:
                    continue
                a_xyz = coords_a[int(a_idx)].reshape(1, 3)
                n = hydrogen_bond_count(d_xyz, [hs], a_xyz)
                if n <= 0:
                    continue
                key = (a_site, d_site) if swap else (d_site, a_site)
                counts[key] = counts.get(key, 0) + n

    _count(mol_i, coords_i, map_i, mol_j, coords_j, map_j, swap=False)
    _count(mol_j, coords_j, map_j, mol_i, coords_i, map_i, swap=True)
    return counts


def contact_run_stats(flags: np.ndarray) -> Dict[str, float]:
    """Persistence, mean run length, and switch count of a boolean series."""
    flags = np.asarray(flags, dtype=bool)
    n = len(flags)
    if n == 0:
        return {
            "persistence": float("nan"),
            "n_frames": 0.0,
            "n_on": 0.0,
            "n_runs": 0.0,
            "mean_run_length": float("nan"),
            "max_run_length": float("nan"),
            "n_switches": 0.0,
        }
    persistence = float(flags.mean())
    n_on = int(flags.sum())
    switches = int(np.sum(flags[1:] != flags[:-1])) if n > 1 else 0
    runs: List[int] = []
    run = 0
    for v in flags:
        if v:
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return {
        "persistence": persistence,
        "n_frames": float(n),
        "n_on": float(n_on),
        "n_runs": float(len(runs)),
        "mean_run_length": float(np.mean(runs)) if runs else 0.0,
        "max_run_length": float(max(runs)) if runs else 0.0,
        "n_switches": float(switches),
    }


def summarize_interface_stability(
    records_by_frame: Sequence[Sequence[InteractionRecord]],
) -> List[Dict[str, Any]]:
    """Per site-pair contact / H-bond / cation–π / π–π persistence."""
    if not records_by_frame:
        return []
    keys: List[Tuple[int, int, int, int]] = []
    seen = set()
    for rec in records_by_frame[0]:
        key = (rec.residue_i, rec.residue_j, rec.site_i, rec.site_j)
        if key not in seen:
            seen.add(key)
            keys.append(key)

    by_key: Dict[Tuple[int, int, int, int], List[InteractionRecord]] = {
        k: [] for k in keys
    }
    for frame_recs in records_by_frame:
        for rec in frame_recs:
            key = (rec.residue_i, rec.residue_j, rec.site_i, rec.site_j)
            if key in by_key:
                by_key[key].append(rec)

    rows: List[Dict[str, Any]] = []
    for key, recs in by_key.items():
        ri, rj, si, sj = key
        contact_flags = np.array([r.contact for r in recs], dtype=bool)
        hbond_flags = np.array([r.n_hbonds > 0 for r in recs], dtype=bool)
        cat_flags = np.array([r.cation_pi for r in recs], dtype=bool)
        pipi_flags = np.array([r.pi_pi for r in recs], dtype=bool)
        row: Dict[str, Any] = {
            "residue_i": ri,
            "residue_j": rj,
            "site_i": si,
            "site_j": sj,
        }
        for name, flags in (
            ("contact", contact_flags),
            ("hbond", hbond_flags),
            ("cation_pi", cat_flags),
            ("pi_pi", pipi_flags),
        ):
            stats = contact_run_stats(flags)
            for sk, sv in stats.items():
                row[f"{name}_{sk}"] = sv
        dists = np.array([r.min_distance for r in recs], dtype=float)
        row["min_distance_mean"] = float(np.nanmean(dists))
        row["min_distance_min"] = float(np.nanmin(dists))
        row["min_distance_max"] = float(np.nanmax(dists))
        rows.append(row)
    return rows


def frame_interaction_summary(records: Sequence[InteractionRecord]) -> Dict[str, float]:
    """Assembly-level interaction counts for one frame."""
    if not records:
        return {
            "n_site_pairs": 0.0,
            "n_contacts": 0.0,
            "n_hbonds": 0.0,
            "n_cation_pi": 0.0,
            "n_pi_pi": 0.0,
            "contact_min_distance": float("nan"),
        }
    return {
        "n_site_pairs": float(len(records)),
        "n_contacts": float(sum(r.n_contacts for r in records)),
        "n_hbonds": float(sum(r.n_hbonds for r in records)),
        "n_cation_pi": float(sum(1 for r in records if r.cation_pi)),
        "n_pi_pi": float(sum(1 for r in records if r.pi_pi)),
        "contact_min_distance": float(
            np.nanmin([r.min_distance for r in records])
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Trajectory observer
# ═══════════════════════════════════════════════════════════════════════════


class LayeredInteractionObserver:
    """Frame observer: Layer 1 once, Layers 2–3 every frame.

    Implements the :class:`~src.TrajectoryIterator.FrameObserver` contract
    without importing it at module load (keeps RDKit-only tests light).
    """

    def __init__(
        self,
        residue_sel_list: List[str],
        contact_cutoff: float = CONTACT_CUTOFF,
        n_frame_rows: Optional[int] = None,
    ):
        self.residue_sel_list = list(residue_sel_list)
        self.contact_cutoff = float(contact_cutoff)
        self._n_frame_rows = n_frame_rows
        self.results: Dict[str, Any] = {
            "frame_summaries": [],
            "records_by_frame": [],
            "catalogs": [],
        }
        self.catalogs: List[TopologyCatalog] = []
        self._site_mda_indices: List[List[List[int]]] = []
        self._rdkit_to_mda: List[List[int]] = []
        self._mols: List[Any] = []
        self.n_frames = 0
        self._initialized = False

    def get_selections_needed(self) -> List[str]:
        return list(self.residue_sel_list)

    def on_frame_start(self, iterator: Any) -> None:
        u = iterator.universe
        self.n_frames = (
            int(self._n_frame_rows)
            if self._n_frame_rows is not None
            else int(iterator.get_n_frames())
        )
        self.catalogs = []
        self._site_mda_indices = []
        self._rdkit_to_mda = []
        self._mols = []
        for sel_str in self.residue_sel_list:
            sel = u.select_atoms(sel_str)
            if len(sel) == 0:
                self.catalogs.append(TopologyCatalog(sites=[]))
                self._site_mda_indices.append([])
                self._rdkit_to_mda.append([])
                self._mols.append(None)
                continue
            mol = mol_from_atomgroup(sel)
            catalog = discover_topology_sites(mol)
            self.catalogs.append(catalog)
            self._mols.append(mol)
            rdkit_to_mda = [int(sel[k].index) for k in range(mol.GetNumAtoms())]
            self._rdkit_to_mda.append(rdkit_to_mda)
            groups: List[List[int]] = []
            for site in catalog.sites:
                mda_idx = [rdkit_to_mda[rdkit_i] for rdkit_i in site.atom_indices]
                groups.append(mda_idx)
            self._site_mda_indices.append(groups)
        self.results["catalogs"] = [c.to_records() for c in self.catalogs]
        self.results["frame_summaries"] = []
        self.results["records_by_frame"] = []
        self._initialized = True

    def on_frame(self, ts: Any, frame_idx: int, universe: Any) -> None:
        if not self._initialized:
            return
        geoms_by_res: List[List[SiteGeometry]] = []
        site_coords_by_res: List[List[np.ndarray]] = []
        mol_coords_by_res: List[Optional[np.ndarray]] = []
        for res_i, catalog in enumerate(self.catalogs):
            site_coords: List[np.ndarray] = []
            geoms: List[SiteGeometry] = []
            mapping = self._rdkit_to_mda[res_i]
            mol = self._mols[res_i]
            mol_xyz = None
            if mol is not None and mapping:
                mol_xyz = universe.atoms[mapping].positions
            for site, mda_idx in zip(catalog.sites, self._site_mda_indices[res_i]):
                xyz = universe.atoms[mda_idx].positions if mda_idx else np.zeros((0, 3))
                site_coords.append(xyz)
                geoms.append(
                    compute_site_geometry(
                        xyz, list(range(len(mda_idx))), aromatic=site.aromatic
                    )
                )
            geoms_by_res.append(geoms)
            site_coords_by_res.append(site_coords)
            mol_coords_by_res.append(mol_xyz)

        frame_records: List[InteractionRecord] = []
        n_res = len(self.catalogs)
        for i in range(n_res):
            for j in range(i + 1, n_res):
                hbond_by_sites: Dict[Tuple[int, int], int] = {}
                mol_i = self._mols[i]
                mol_j = self._mols[j]
                ci_all = mol_coords_by_res[i]
                cj_all = mol_coords_by_res[j]
                if mol_i is not None and mol_j is not None and ci_all is not None and cj_all is not None:
                    hbond_by_sites = _hbonds_by_site_pair(
                        mol_i, self.catalogs[i], ci_all,
                        mol_j, self.catalogs[j], cj_all,
                    )
                for si, gi, ci in zip(
                    self.catalogs[i].sites, geoms_by_res[i], site_coords_by_res[i]
                ):
                    for sj, gj, cj in zip(
                        self.catalogs[j].sites, geoms_by_res[j], site_coords_by_res[j]
                    ):
                        rec = score_site_pair(
                            si, gi, ci, sj, gj, cj,
                            residue_i=i,
                            residue_j=j,
                            mol_i=mol_i,
                            mol_j=mol_j,
                            hbond_count=hbond_by_sites.get((si.site_index, sj.site_index), 0),
                            contact_cutoff=self.contact_cutoff,
                        )
                        frame_records.append(rec)
        summary = frame_interaction_summary(frame_records)
        summary["frame"] = float(getattr(ts, "frame", frame_idx))
        self.results["frame_summaries"].append(summary)
        self.results["records_by_frame"].append(frame_records)

    def on_frame_end(self, iterator: Any) -> None:
        recs = self.results.get("records_by_frame") or []
        self.results["interface_stability"] = summarize_interface_stability(recs)

    def get_frame_summary_df(self):
        import pandas as pd

        rows = self.results.get("frame_summaries") or []
        return pd.DataFrame(rows)

    def get_stability_df(self):
        import pandas as pd

        rows = self.results.get("interface_stability") or []
        return pd.DataFrame(rows)

    def get_records_df(self):
        import pandas as pd

        rows = []
        for f_idx, recs in enumerate(self.results.get("records_by_frame") or []):
            for rec in recs:
                d = rec.to_dict()
                d["frame_row"] = f_idx
                rows.append(d)
        return pd.DataFrame(rows)
