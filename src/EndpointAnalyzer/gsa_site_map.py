"""Canonical GSA monomer site map (Murata R1 / R2 / R3, not hull order).

The Hiraoka/Murata GSA is a hexaarylbenzene: a six-carbon core with six
aryl substituents. Around that core the unique pattern is

    Ph-linker — R1 — Py-linker — R3 — Py-linker — R2

``R1`` is the pole (cube vertex). ``R2`` and ``R3`` interlock on equatorial
edges; Murata's d1 is the ipso-carbon pair C2–C3 of those two aryls.

Hull-order site indices (``s3``/``s7``) are not chemically stable: adding a
methyl converts a ring site into an atom site and shifts every later index.
This module assigns the same role → site-index on every B* cohort.

R1 vs R2 (both sit between Ph and a Py+ in the hexagon) are disambiguated by
core-carbon atom index: on the Amber templates in this repo the higher-index
core carbon is the pole (R1). Tests lock that against the known methyl pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

CANONICAL_ROLES: tuple[str, ...] = (
    "ph",
    "ph_para",
    "py_eq",
    "r2",
    "r3",
    "py_pole",
    "r1",
    "r1_ring",
)

# paper-d1 uses the two equatorial ipso carbons
D1_ROLES: tuple[str, str] = ("r2", "r3")

_ROLE_KIND: dict[str, str] = {
    "ph": "ring",
    "ph_para": "atom",
    "py_eq": "ring",
    "r2": "atom",
    "r3": "atom",
    "py_pole": "ring",
    "r1": "atom",
    "r1_ring": "ring",
}

_PLOT_LABELS: dict[str, str] = {
    "ph": "Ph",
    "ph_para": "Ph-p",
    "py_eq": "Py-eq",
    "r2": "R2",
    "r3": "R3",
    "py_pole": "Py-pole",
    "r1": "R1",
    "r1_ring": "R1-ring",
}


class GSASiteMapError(ValueError):
    """Raised when a monomer is not a Murata-style hexaaryl GSA."""


@dataclass
class GSASite:
    role: str
    kind: str
    rdkit_atoms: list[int]
    ipso_rdkit: Optional[int] = None
    subst_rdkit: Optional[int] = None
    methyl: bool = False
    # Para-to-N carbon on Py+ rings (Murata CPy for d2).
    cpy_rdkit: Optional[int] = None

    @property
    def plot_label(self) -> str:
        return _PLOT_LABELS.get(self.role, self.role)


@dataclass
class GSAMonomerMap:
    sites: list[GSASite] = field(default_factory=list)

    def by_role(self, role: str) -> GSASite:
        for s in self.sites:
            if s.role == role:
                return s
        raise KeyError(role)

    def rdkit_site_groups(self) -> list[list[int]]:
        return [list(s.rdkit_atoms) for s in self.sites]

    def roles(self) -> list[str]:
        return [s.role for s in self.sites]


def plot_label_for_role(role: str, kind: str = "") -> str:
    """Human label for QC plots (``R1``, ``R2``, ``Ph``, …)."""
    if role in _PLOT_LABELS:
        return _PLOT_LABELS[role]
    letter = "A" if str(kind).lower().startswith("atom") else "R"
    return f"{role}:{letter}"


def _six_rings(mol: Any) -> list[tuple[int, ...]]:
    return [r for r in mol.GetRingInfo().AtomRings() if len(r) == 6]


def _ring_sets(ring_tuples: Sequence[tuple[int, ...]]) -> list[set[int]]:
    return [set(r) for r in ring_tuples]


def _has_n(mol: Any, atoms: Sequence[int]) -> bool:
    return any(mol.GetAtomWithIdx(int(i)).GetSymbol() == "N" for i in atoms)


def _find_core_index(mol: Any, rings: Sequence[set[int]]) -> int:
    for i, r in enumerate(rings):
        n_out = 0
        for aidx in r:
            atom = mol.GetAtomWithIdx(int(aidx))
            if any(
                (n.GetIdx() not in r and n.IsInRing()) for n in atom.GetNeighbors()
            ):
                n_out += 1
        if n_out == 6:
            return i
    raise GSASiteMapError("No hexaarylbenzene core (6 aryl attachments) found")


def _subst_ring_for_core_atom(
    mol: Any,
    core: set[int],
    core_atom: int,
    rings: Sequence[set[int]],
) -> tuple[int, int]:
    atom = mol.GetAtomWithIdx(int(core_atom))
    for n in atom.GetNeighbors():
        ni = int(n.GetIdx())
        if ni in core:
            continue
        for j, rj in enumerate(rings):
            if ni in rj:
                return j, ni
    raise GSASiteMapError(f"Core atom {core_atom} has no substituent ring")


def _other_ring_links(
    mol: Any,
    ring: set[int],
    core: set[int],
    rings: Sequence[set[int]],
) -> list[int]:
    out: list[int] = []
    for aidx in ring:
        atom = mol.GetAtomWithIdx(int(aidx))
        for n in atom.GetNeighbors():
            ni = int(n.GetIdx())
            if ni in ring or ni in core:
                continue
            for j, rj in enumerate(rings):
                if ni in rj:
                    out.append(j)
    return out


def _para_atom(ring_tuple: tuple[int, ...], attach: int) -> int:
    order = list(ring_tuple)
    pos = order.index(int(attach))
    return int(order[(pos + 3) % 6])


def _pyridinium_n_and_para(mol: Any, ring_tuple: tuple[int, ...]) -> tuple[int, int]:
    """Return ``(nitrogen, para-to-N carbon)`` on a six-membered Py+ ring."""
    n_idx = None
    for aidx in ring_tuple:
        if mol.GetAtomWithIdx(int(aidx)).GetSymbol() == "N":
            n_idx = int(aidx)
            break
    if n_idx is None:
        raise GSASiteMapError("Py+ ring has no nitrogen")
    return n_idx, _para_atom(ring_tuple, n_idx)


def _r_substituent(
    mol: Any,
    ring: set[int],
    core: set[int],
    attach: int,
    ring_tuple: tuple[int, ...],
) -> tuple[int, int, bool]:
    """Return ``(ipso, subst, is_methyl)`` for an R-bearing aryl.

    *ipso* is the ring carbon that bonds to R (Murata C2/C3/pole carbon).
    *subst* is the methyl carbon when R=CH3, otherwise the ipso carbon (R=H).
    """
    for aidx in ring:
        atom = mol.GetAtomWithIdx(int(aidx))
        for n in atom.GetNeighbors():
            ni = int(n.GetIdx())
            if ni in ring or ni in core:
                continue
            if n.GetSymbol() == "C" and not n.IsInRing():
                hcount = sum(1 for x in n.GetNeighbors() if x.GetSymbol() == "H")
                if hcount >= 2:
                    return int(aidx), ni, True
    ipso = _para_atom(ring_tuple, attach)
    return ipso, ipso, False


def classify_gsa_monomer(mol: Any) -> GSAMonomerMap:
    """Map one GSA monomer to canonical R1/R2/R3 / Ph / Py+ sites."""
    ring_tuples = _six_rings(mol)
    if len(ring_tuples) < 7:
        raise GSASiteMapError(f"Expected ≥7 six-membered rings, got {len(ring_tuples)}")
    rings = _ring_sets(ring_tuples)
    core_idx = _find_core_index(mol, rings)
    core = rings[core_idx]
    core_order = list(ring_tuples[core_idx])

    items: list[dict[str, Any]] = []
    for ca in core_order:
        srid, attach = _subst_ring_for_core_atom(mol, core, int(ca), rings)
        links = _other_ring_links(mol, rings[srid], core, rings)
        if _has_n(mol, rings[srid]) or any(_has_n(mol, rings[x]) for x in links):
            kind = "PyL"
        elif links:
            kind = "PhL"
        else:
            kind = "R"
        items.append(
            {
                "core": int(ca),
                "srid": srid,
                "attach": attach,
                "kind": kind,
                "links": links,
                "role": None,
            }
        )

    n = len(items)
    if n != 6:
        raise GSASiteMapError(f"Core has {n} substituents, expected 6")
    n_r = sum(1 for it in items if it["kind"] == "R")
    n_py = sum(1 for it in items if it["kind"] == "PyL")
    n_ph = sum(1 for it in items if it["kind"] == "PhL")
    if not (n_r == 3 and n_py == 2 and n_ph == 1):
        raise GSASiteMapError(
            f"Core pattern is R×{n_r} PyL×{n_py} PhL×{n_ph}, expected 3/2/1"
        )

    r3_i = None
    for i, it in enumerate(items):
        if it["kind"] != "R":
            continue
        left = items[(i - 1) % n]["kind"]
        right = items[(i + 1) % n]["kind"]
        if left == "PyL" and right == "PyL":
            r3_i = i
            break
    if r3_i is None:
        raise GSASiteMapError("Could not find R3 (R-aryl between the two Py+ linkers)")
    items[r3_i]["role"] = "r3"

    rest = [i for i, it in enumerate(items) if it["kind"] == "R" and it["role"] is None]
    rest.sort(key=lambda i: items[i]["core"], reverse=True)
    items[rest[0]]["role"] = "r1"
    items[rest[1]]["role"] = "r2"

    for i, it in enumerate(items):
        if it["kind"] != "PyL":
            continue
        nbrs = [items[(i - 1) % n], items[(i + 1) % n]]
        nbr_roles = {x.get("role") for x in nbrs}
        if "r1" in nbr_roles:
            it["role"] = "py_pole"
        elif "r2" in nbr_roles:
            it["role"] = "py_eq"
        else:
            raise GSASiteMapError("Py-linker is not adjacent to R1 or R2")

    phl = next(it for it in items if it["kind"] == "PhL")
    if not phl["links"]:
        raise GSASiteMapError("Ph-linker has no terminal phenyl")
    ph_rid = int(phl["links"][0])

    by_role = {it["role"]: it for it in items if it["role"]}

    def _ring_site(role: str, rid: int, *, cpy_rdkit: Optional[int] = None) -> GSASite:
        return GSASite(
            role=role,
            kind="ring",
            rdkit_atoms=sorted(rings[rid]),
            cpy_rdkit=cpy_rdkit,
        )

    def _r_atom_site(role: str) -> GSASite:
        it = by_role[role]
        rid = int(it["srid"])
        ipso, subst, methyl = _r_substituent(
            mol, rings[rid], core, int(it["attach"]), ring_tuples[rid]
        )
        return GSASite(
            role=role,
            kind="atom",
            rdkit_atoms=[int(subst)],
            ipso_rdkit=int(ipso),
            subst_rdkit=int(subst),
            methyl=bool(methyl),
        )

    # Terminal Ph para = opposite the linker attachment.
    ph_attach = None
    for aidx in rings[ph_rid]:
        atom = mol.GetAtomWithIdx(int(aidx))
        if any(int(n.GetIdx()) in rings[int(phl["srid"])] for n in atom.GetNeighbors()):
            ph_attach = int(aidx)
            break
    if ph_attach is None:
        raise GSASiteMapError("Cannot find Ph attachment to linker")
    ph_para = _para_atom(ring_tuples[ph_rid], ph_attach)

    py_eq_rid = None
    py_pole_rid = None
    for it in items:
        if it["role"] == "py_eq":
            n_links = [x for x in it["links"] if _has_n(mol, rings[x])]
            py_eq_rid = n_links[0] if n_links else it["srid"]
        elif it["role"] == "py_pole":
            n_links = [x for x in it["links"] if _has_n(mol, rings[x])]
            py_pole_rid = n_links[0] if n_links else it["srid"]
    if py_eq_rid is None or py_pole_rid is None:
        raise GSASiteMapError("Missing Py+ rings")

    _, cpy_eq = _pyridinium_n_and_para(mol, ring_tuples[int(py_eq_rid)])
    _, cpy_pole = _pyridinium_n_and_para(mol, ring_tuples[int(py_pole_rid)])

    r1_it = by_role["r1"]
    sites = [
        _ring_site("ph", ph_rid),
        GSASite(role="ph_para", kind="atom", rdkit_atoms=[int(ph_para)]),
        _ring_site("py_eq", int(py_eq_rid), cpy_rdkit=int(cpy_eq)),
        _r_atom_site("r2"),
        _r_atom_site("r3"),
        _ring_site("py_pole", int(py_pole_rid), cpy_rdkit=int(cpy_pole)),
        _r_atom_site("r1"),
        _ring_site("r1_ring", int(r1_it["srid"])),
    ]
    got = tuple(s.role for s in sites)
    if got != CANONICAL_ROLES:
        raise GSASiteMapError(f"Site order {got} != {CANONICAL_ROLES}")
    return GSAMonomerMap(sites=sites)


def canonical_gsa_sites_rdkit(mol: Any) -> list[list[int]]:
    """RDKit atom-index groups in canonical order (for EndpointAnalyzer)."""
    return classify_gsa_monomer(mol).rdkit_site_groups()


def try_classify_gsa_monomer(mol: Any) -> Optional[GSAMonomerMap]:
    try:
        return classify_gsa_monomer(mol)
    except (GSASiteMapError, ValueError, KeyError, IndexError):
        return None
