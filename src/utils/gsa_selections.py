"""
Hybrid selection resolver for GSA nanocube feature extraction.

Provides a dataclass ``GSAFeatureSelections`` that holds all MDAnalysis
selection strings needed by the feature functions, and a
``resolve_selections`` helper that fills in missing fields with
topology-aware heuristics while preserving any explicit overrides.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from typing import Any, List, Optional


@dataclass
class GSAFeatureSelections:
    """All MDAnalysis selection strings needed by GSA feature functions.

    Every field defaults to ``None``.  ``resolve_selections`` will fill in
    missing fields using topology-based heuristics.
    """

    assembly_sel: Optional[str] = None
    monomer_selections: Optional[List[str]] = None

    hydrophilic_sel: Optional[str] = None
    hydrophobic_sel: Optional[str] = None

    headgroup_sel: Optional[str] = None
    tail_sel: Optional[str] = None
    core_sel: Optional[str] = None

    tooth_selections: Optional[List[str]] = None
    inner_atom_sel: Optional[str] = None

    water_sel: Optional[str] = None
    ion_sel: Optional[str] = None
    guest_sel: Optional[str] = None

    hbond_donor_sel: Optional[str] = None
    hbond_acceptor_sel: Optional[str] = None

    cation_sel: Optional[str] = None
    anion_sel: Optional[str] = None


# ---------------------------------------------------------------------------
# Auto-derivation helpers
# ---------------------------------------------------------------------------

_WATER_RESNAMES = ("WAT", "HOH", "TIP3", "TIP3P", "TIP4P", "SPC", "SPCE", "OPC")
_NA_RESNAMES = ("Na+", "SOD", "NA")
_IODIDE_RESNAMES = ("I-", "IOD", "IB", "I")
_CATION_RESNAMES = ("Na+", "SOD", "NA", "K+", "K", "POT",
                     "Mg2+", "MG", "Ca2+", "CAL", "Zn2+", "ZN")
_ANION_RESNAMES = ("Cl-", "CLA", "CL", "I-", "IOD", "IB", "I",
                    "Br-", "BR", "F-")

_DUMMY_NAMES = ("EPW", "LP", "MW", "EP")


def _or_sel(resnames: tuple[str, ...]) -> str:
    return " or ".join(f"resname {r}" for r in resnames)


def _derive_monomers(
    universe: Any,
    gsa_resname: str,
    n_monomers: int,
) -> List[str]:
    """One selection per GSA residue; fall back to KMeans if only one residue."""
    residues = [r for r in universe.residues if r.resname == gsa_resname]
    if not residues:
        raise ValueError(
            f"No residues with resname '{gsa_resname}' found in universe. "
            "Provide explicit monomer_selections."
        )

    if len(residues) >= n_monomers:
        sels = [f"resname {gsa_resname} and resid {r.resid}" for r in residues[:n_monomers]]
        if len(residues) > n_monomers:
            warnings.warn(
                f"Found {len(residues)} '{gsa_resname}' residues but n_monomers={n_monomers}; "
                f"using first {n_monomers}.",
                stacklevel=3,
            )
        return sels

    if len(residues) == 1:
        warnings.warn(
            f"Only one '{gsa_resname}' residue found; attempting atom-name periodicity "
            "split or KMeans on atom positions.",
            stacklevel=3,
        )
        return _split_single_residue(universe, gsa_resname, n_monomers)

    warnings.warn(
        f"Found {len(residues)} '{gsa_resname}' residues, fewer than n_monomers={n_monomers}. "
        "Using all as separate monomers.",
        stacklevel=3,
    )
    return [f"resname {gsa_resname} and resid {r.resid}" for r in residues]


def _split_single_residue(
    universe: Any,
    gsa_resname: str,
    n_monomers: int,
) -> List[str]:
    """Split one big residue into n_monomers by atom-name periodicity or KMeans."""
    import numpy as np

    ag = universe.select_atoms(f"resname {gsa_resname}")
    names = [a.name for a in ag]
    n_atoms = len(names)
    atoms_per = n_atoms // n_monomers

    if atoms_per >= 2 and n_atoms == atoms_per * n_monomers:
        chunk_ok = True
        ref_chunk = names[:atoms_per]
        for k in range(1, n_monomers):
            if names[k * atoms_per : (k + 1) * atoms_per] != ref_chunk:
                chunk_ok = False
                break
        if chunk_ok:
            sels = []
            for k in range(n_monomers):
                start = ag[k * atoms_per].index
                stop = ag[(k + 1) * atoms_per - 1].index
                sels.append(f"index {start}-{stop}")
            return sels

    from sklearn.cluster import KMeans

    coords = ag.positions
    labels = KMeans(n_clusters=n_monomers, n_init=20, random_state=0).fit_predict(coords)
    sels = []
    for k in range(n_monomers):
        idx = np.where(labels == k)[0]
        global_idx = [str(ag[int(i)].index) for i in idx]
        sels.append("index " + " ".join(global_idx))
    return sels


def _derive_hydrophilic(gsa_resname: str) -> str:
    return f"resname {gsa_resname} and (name N* or name O*)"


def _derive_hydrophobic(gsa_resname: str) -> str:
    return f"resname {gsa_resname} and name C* and not name Cl*"


def _derive_water() -> str:
    parts = [f"resname {r}" for r in _WATER_RESNAMES]
    dummy_excl = " and ".join(f"not name {n}" for n in _DUMMY_NAMES)
    return f"({' or '.join(parts)}) and ({dummy_excl})"


def _derive_ion() -> str:
    parts = [f"resname {r}" for r in (*_NA_RESNAMES, *_IODIDE_RESNAMES)]
    return " or ".join(parts)


def _derive_guest(universe: Any, gsa_resname: str) -> Optional[str]:
    """Auto-detect guest: non-water, non-ion, non-GSA residues."""
    gsa_resnames = {gsa_resname}
    skip = set(_WATER_RESNAMES) | set(_NA_RESNAMES) | set(_CATION_RESNAMES) | set(_ANION_RESNAMES) | gsa_resnames
    candidates = set()
    for r in universe.residues:
        if r.resname not in skip:
            candidates.add(r.resname)
    if len(candidates) == 1:
        rn = candidates.pop()
        warnings.warn(
            f"Auto-detected guest resname '{rn}'. Override with guest_sel if wrong.",
            stacklevel=3,
        )
        return f"resname {rn}"
    if len(candidates) > 1:
        warnings.warn(
            f"Multiple non-solvent/non-ion residue names found: {candidates}. "
            "Cannot auto-detect guest; set guest_sel explicitly.",
            stacklevel=3,
        )
    return None


def _derive_cation_anion(universe: Any) -> tuple[Optional[str], Optional[str]]:
    """Derive cation/anion selections from charges or resname heuristics."""
    try:
        charges = universe.atoms.charges
        has_charges = True
    except Exception:
        has_charges = False

    if has_charges:
        cat_resnames: set[str] = set()
        ani_resnames: set[str] = set()
        for r in universe.residues:
            ag = r.atoms
            total_q = float(ag.charges.sum())
            if total_q > 0.5:
                cat_resnames.add(r.resname)
            elif total_q < -0.5:
                ani_resnames.add(r.resname)
        cat_sel = (" or ".join(f"resname {n}" for n in sorted(cat_resnames))) if cat_resnames else None
        ani_sel = (" or ".join(f"resname {n}" for n in sorted(ani_resnames))) if ani_resnames else None
        return cat_sel, ani_sel

    cat_parts = [f"resname {r}" for r in _CATION_RESNAMES]
    ani_parts = [f"resname {r}" for r in _ANION_RESNAMES]
    cat_found = " or ".join(cat_parts)
    ani_found = " or ".join(ani_parts)
    return cat_found, ani_found


def _derive_hbond_donors(gsa_resname: str) -> str:
    return f"resname {gsa_resname} and (name N* or name O*) and not name NE* NZ*"


def _derive_hbond_acceptors(gsa_resname: str) -> str:
    return f"resname {gsa_resname} and (name N* or name O*)"


def _derive_inner_atoms(gsa_resname: str) -> str:
    return f"resname {gsa_resname} and not name H*"


def _derive_tooth_selections(
    universe: Any,
    monomer_selections: List[str],
) -> Optional[List[str]]:
    """Per-monomer gear-tooth atom groups via EndpointAnalyzer convex-hull endpoints."""
    try:
        from src.EndpointAnalyzer import EndpointAnalyzer, EndpointsFinder
    except ImportError as exc:
        warnings.warn(
            f"Cannot auto-detect gear teeth: EndpointAnalyzer unavailable ({exc}). "
            "Install the rdkit extra or provide explicit tooth_selections.",
            stacklevel=3,
        )
        return None

    if hasattr(universe.trajectory, "__len__") and len(universe.trajectory) > 0:
        universe.trajectory[0]

    finder = EndpointsFinder()
    tooth_sels: List[str] = []
    n_found = 0
    for mon_sel in monomer_selections:
        try:
            _, ep_ids = EndpointAnalyzer.find_residue_endpoints(
                universe, mon_sel, finder
            )
            if ep_ids:
                tooth_sels.append("id " + " ".join(str(i) for i in ep_ids))
                n_found += 1
            else:
                tooth_sels.append(f"({mon_sel}) and index -1")
        except Exception as exc:
            warnings.warn(
                f"Tooth detection failed for '{mon_sel}': {exc}",
                stacklevel=3,
            )
            tooth_sels.append(f"({mon_sel}) and index -1")

    if n_found == 0:
        warnings.warn(
            "Auto tooth detection found no endpoints on any monomer.",
            stacklevel=3,
        )
        return None

    warnings.warn(
        f"Auto-detected gear teeth on {n_found}/{len(monomer_selections)} monomers "
        "via EndpointAnalyzer.",
        stacklevel=3,
    )
    return tooth_sels


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

def resolve_selections(
    universe: Any,
    explicit: Optional[GSAFeatureSelections] = None,
    gsa_resname: str = "MOL",
    n_monomers: int = 6,
    *,
    auto_tooth: bool = True,
) -> GSAFeatureSelections:
    """Fill in missing selections with topology-aware heuristics.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Loaded universe (topology must be available; trajectory optional).
    explicit : GSAFeatureSelections, optional
        Partially filled selections.  Any non-None field is kept as-is.
    gsa_resname : str
        Residue name of the GSA amphiphile building blocks (default ``"MOL"``).
    n_monomers : int
        Expected number of monomers forming the nanocube (default 6).
    auto_tooth : bool
        When True and ``tooth_selections`` is unset, derive per-monomer tooth
        atom groups with :class:`~src.EndpointAnalyzer.EndpointAnalyzer`.

    Returns
    -------
    GSAFeatureSelections
        Fully resolved selections (some optional fields may remain None if
        auto-detection fails and no explicit value was given).
    """
    sel = GSAFeatureSelections() if explicit is None else GSAFeatureSelections(
        **{f.name: getattr(explicit, f.name) for f in fields(explicit)}
    )

    if sel.assembly_sel is None:
        sel.assembly_sel = f"resname {gsa_resname}"

    if sel.monomer_selections is None:
        sel.monomer_selections = _derive_monomers(universe, gsa_resname, n_monomers)
        warnings.warn(
            f"Auto-derived {len(sel.monomer_selections)} monomer selections from "
            f"resname '{gsa_resname}'.",
            stacklevel=2,
        )

    if sel.hydrophilic_sel is None:
        sel.hydrophilic_sel = _derive_hydrophilic(gsa_resname)

    if sel.hydrophobic_sel is None:
        sel.hydrophobic_sel = _derive_hydrophobic(gsa_resname)

    if sel.water_sel is None:
        sel.water_sel = _derive_water()

    if sel.ion_sel is None:
        sel.ion_sel = _derive_ion()

    if sel.guest_sel is None:
        sel.guest_sel = _derive_guest(universe, gsa_resname)

    if sel.cation_sel is None or sel.anion_sel is None:
        cat, ani = _derive_cation_anion(universe)
        if sel.cation_sel is None:
            sel.cation_sel = cat
        if sel.anion_sel is None:
            sel.anion_sel = ani

    if sel.hbond_donor_sel is None:
        sel.hbond_donor_sel = _derive_hbond_donors(gsa_resname)

    if sel.hbond_acceptor_sel is None:
        sel.hbond_acceptor_sel = _derive_hbond_acceptors(gsa_resname)

    if sel.inner_atom_sel is None:
        sel.inner_atom_sel = _derive_inner_atoms(gsa_resname)

    if sel.tooth_selections is None and auto_tooth and sel.monomer_selections:
        sel.tooth_selections = _derive_tooth_selections(
            universe, sel.monomer_selections
        )

    return sel
