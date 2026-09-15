"""Murata metastructure criteria: cation–π, d2, RMSD, and A/B/C1/C2 labels.

G0 implements Murata 2026 operationally:

* cation–π sandwich = Ph between **π(pole)** (pole Py⁺ of monomer *i*) and
  **π(equator)** (equatorial Py⁺ of a third monomer *k*); **open** is π(pole) ≥ 6.5 Å
* d1 = C2 (R2 of *i*) to C3 (R3 of *j*) on a locked equatorial edge
* d2 = CPy (eq Py⁺ carbon **para to the phenylene linker**) to C3 (R3 of *j*)
* labels A / B / C1 / C2 / other from opened cation–π count and elongated d1

The six cation–π units and six d1 edges are locked by Hungarian assignment on
median distances (same machinery as ``murata_d1.assign_equatorial_r2_r3_edges``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .murata_d1 import (
    D1_ELONGATED_LO,
    assign_equatorial_r2_r3_edges,
    canonical_role_atoms_mda,
    match_hull_sites_to_roles,
    role_site_index,
    site_pair_column,
)
from .transition_attribution import parse_site_pair_feature

CATION_PI_OPEN_LO = 6.5
METASTRUCTURES = ("A", "B", "C1", "C2", "other")

# Murata 2026 Table 1, non-encapsulated trajectories. 2₆′ has no cohort here.
MURATA_TABLE1_PCT: dict[str, dict[str, float]] = {
    "1₆": {"A": 78.4, "B": 8.7, "C1": 2.3, "C2": 10.4, "other": 0.2},
    "2₆": {"A": 45.3, "B": 13.8, "C1": 1.7, "C2": 38.6, "other": 0.5},
    "2₆′": {"A": 19.7, "B": 13.3, "C1": 19.1, "C2": 38.9, "other": 9.0},
    "3₆": {"A": 7.2, "B": 15.8, "C1": 4.7, "C2": 26.3, "other": 46.1},
}
COHORT_TO_MURATA_SYSTEM: dict[str, str] = {
    "BMMpM": "1₆",
    "BMHpM": "2₆",
    "BHHpM": "3₆",
}

# Murata RMSD is **not** whole-cube Kabsch. It is local heavy-atom RMSD of
# the motif that opens: cation–π sandwich (pole Py+, Ph, equatorial Py+) or
# the equatorial interlocking belt (Py+, R2, R3, neighboring Py+).
MURATA_RMSD_MOTIFS: dict[str, tuple[str, ...]] = {
    "cation_pi": ("py_pole", "ph", "py_eq"),
    "equator": ("py_eq", "r2", "r3"),
}


def classify_murata_metastructure(
    n_open_cation_pi: int,
    n_d1_elongated: int = 0,
) -> str:
    """Murata Table-style label from opened cation–π count and elongated d1.

    C1 is two openings with **zero** d1 ≥ 7.0 Å; C2 is two openings with
    **exactly one** elongated d1. Anything else (including ≥ 3 openings, or
    two openings with two-plus elongated d1) is ``other``.
    """
    n_open = int(n_open_cation_pi)
    n_elong = int(n_d1_elongated)
    if n_open == 0:
        return "A"
    if n_open == 1:
        return "B"
    if n_open == 2:
        if n_elong == 0:
            return "C1"
        if n_elong == 1:
            return "C2"
    return "other"


def classify_murata_metastructure_columns(
    n_open: np.ndarray,
    n_elongated: np.ndarray,
) -> np.ndarray:
    """Vectorized labels for aligned ``n_open`` / ``n_elongated`` arrays."""
    n_open = np.asarray(n_open, dtype=int)
    n_elongated = np.asarray(n_elongated, dtype=int)
    if n_open.shape != n_elongated.shape:
        raise ValueError("n_open and n_elongated must have the same shape")
    out = np.full(n_open.shape, "other", dtype=object)
    out[n_open == 0] = "A"
    out[n_open == 1] = "B"
    two = n_open == 2
    out[two & (n_elongated == 0)] = "C1"
    out[two & (n_elongated == 1)] = "C2"
    return out


def cation_pi_column_name(mon_pole: int, mon_ph: int) -> str:
    return f"cation_pi_m{int(mon_pole)}pole_m{int(mon_ph)}ph"


def d2_column_name(monomer: int) -> str:
    return f"d2_m{int(monomer)}"


def cation_pi_pair_columns(n_monomers: int = 6) -> list[tuple[str, int, int]]:
    """All directed pole-Py⁺(i) → Ph(j) columns, i≠j."""
    out: list[tuple[str, int, int]] = []
    for i in range(int(n_monomers)):
        for j in range(int(n_monomers)):
            if i == j:
                continue
            out.append((cation_pi_column_name(i, j), i, j))
    return out


def _id_to_index(universe: Any) -> dict[int, int]:
    return {int(a.id): int(a.index) for a in universe.atoms}


def _parse_id_list(raw: Any) -> list[int]:
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return []
    text = str(raw).replace("|", " ").replace(",", " ").strip()
    if not text or text.lower() == "nan":
        return []
    out: list[int] = []
    for tok in text.split():
        try:
            out.append(int(float(tok)))
        except ValueError:
            continue
    return out


def resolve_murata_criteria_atoms(
    universe: Any,
    monomer_selections: Sequence[str],
) -> pd.DataFrame:
    """MDA atom ids/indices for Ph, Py-pole, CPy, and R3 ipso on each monomer."""
    roles = canonical_role_atoms_mda(universe, monomer_selections)
    id_to_index = _id_to_index(universe)
    rows: list[dict[str, Any]] = []
    for mon in sorted(set(int(m) for m in roles["monomer"])):
        sub = roles[roles["monomer"].astype(int) == mon].set_index("role")
        for role in ("ph", "py_pole", "py_eq", "r2", "r3"):
            if role not in sub.index:
                raise KeyError(f"Monomer {mon} missing role {role}")
            rrow = sub.loc[role]
            atom_ids = _parse_id_list(rrow.get("atom_ids"))
            cpy = rrow.get("cpy_mda_id")
            ipso = rrow.get("ipso_mda_id")
            rows.append(
                {
                    "monomer": mon,
                    "role": role,
                    "kind": str(rrow.get("kind", "")),
                    "atom_ids": " ".join(str(a) for a in atom_ids),
                    "atom_indices": " ".join(
                        str(id_to_index[a]) for a in atom_ids if a in id_to_index
                    ),
                    "cpy_mda_id": int(cpy) if pd.notna(cpy) else None,
                    "cpy_index": (
                        int(id_to_index[int(cpy)])
                        if pd.notna(cpy) and int(cpy) in id_to_index
                        else None
                    ),
                    "ipso_mda_id": int(ipso) if pd.notna(ipso) else None,
                    "ipso_index": (
                        int(id_to_index[int(ipso)])
                        if pd.notna(ipso) and int(ipso) in id_to_index
                        else None
                    ),
                }
            )
    return pd.DataFrame(rows)


def motif_atom_indices(
    roles_df: pd.DataFrame,
    roles: Sequence[str],
) -> np.ndarray:
    """Sorted unique MDA indices for ``roles`` on every monomer."""
    idx: list[int] = []
    monomers = sorted(set(int(m) for m in roles_df["monomer"]))
    for mon in monomers:
        for role in roles:
            idx.extend(_indices_for(roles_df, mon, role))
    if not idx:
        raise ValueError(f"No atoms for Murata RMSD roles {tuple(roles)}")
    return np.unique(np.asarray(idx, dtype=int))


def _role_indices_by_monomer(roles_df: pd.DataFrame, role: str) -> dict[int, list[int]]:
    monomers = sorted(set(int(m) for m in roles_df["monomer"]))
    return {int(m): _indices_for(roles_df, m, role) for m in monomers}


def _coms_from_indices(pos: np.ndarray, by_mon: dict[int, list[int]]) -> np.ndarray:
    monomers = sorted(by_mon)
    return np.stack([np.asarray(pos[by_mon[m]], dtype=float).mean(axis=0) for m in monomers])


def _role_point(pos: np.ndarray, roles_df: pd.DataFrame, monomer: int, role: str) -> np.ndarray:
    """Ipso coordinate when mapped, otherwise the role-atom COM."""
    try:
        return np.asarray(pos[_scalar_index(roles_df, monomer, role, "ipso_index")], dtype=float)
    except KeyError:
        idx = _indices_for(roles_df, monomer, role)
        return np.asarray(pos[idx], dtype=float).mean(axis=0)


def _cpy_point(pos: np.ndarray, roles_df: pd.DataFrame, monomer: int) -> np.ndarray:
    """CPy (para to the Py⁺–benzene linker), else the Py⁺ ring COM."""
    try:
        return np.asarray(pos[_scalar_index(roles_df, monomer, "py_eq", "cpy_index")], dtype=float)
    except KeyError:
        idx = _indices_for(roles_df, monomer, "py_eq")
        return np.asarray(pos[idx], dtype=float).mean(axis=0)


def lock_intermolecular_edges(src_coms: np.ndarray, dst_coms: np.ndarray) -> list[tuple[int, int]]:
    """Hungarian 1-1 pairing of intermolecular COMs (diagonal forbidden)."""
    n = int(len(src_coms))
    if len(dst_coms) != n:
        raise ValueError("src and dst COM arrays must have the same length")
    cost = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cost[i, j] = float(np.linalg.norm(src_coms[i] - dst_coms[j]))
    return assign_equatorial_r2_r3_edges(cost)


def _map_edge_monomers(
    edges: Sequence[tuple[int, int]], monomers: Sequence[int]
) -> list[tuple[int, int]]:
    mons = [int(m) for m in monomers]
    return [(mons[int(i)], mons[int(j)]) for i, j in edges]


def lock_cation_pi_edges(pos: np.ndarray, roles_df: pd.DataFrame) -> list[tuple[int, int]]:
    """Six pole-Py⁺ → Ph pairs from the current coordinates."""
    by_pole = _role_indices_by_monomer(roles_df, "py_pole")
    monomers = sorted(by_pole)
    pole = _coms_from_indices(pos, by_pole)
    ph = _coms_from_indices(pos, _role_indices_by_monomer(roles_df, "ph"))
    return _map_edge_monomers(lock_intermolecular_edges(pole, ph), monomers)


def lock_cation_pi_eq_edges(
    pos: np.ndarray,
    roles_df: pd.DataFrame,
    *,
    forbidden_src_for_dst: Optional[dict[int, int]] = None,
) -> list[tuple[int, int]]:
    """Six equatorial-Py⁺ → Ph pairs (the other face of each sandwich)."""
    by_eq = _role_indices_by_monomer(roles_df, "py_eq")
    monomers = sorted(by_eq)
    eq = _coms_from_indices(pos, by_eq)
    ph = _coms_from_indices(pos, _role_indices_by_monomer(roles_df, "ph"))
    n = len(monomers)
    cost = np.full((n, n), np.inf, dtype=float)
    forbid = {int(k): int(v) for k, v in (forbidden_src_for_dst or {}).items()}
    for i, src in enumerate(monomers):
        for j, dst in enumerate(monomers):
            if src == dst:
                continue
            if forbid.get(int(dst)) == int(src):
                continue
            cost[i, j] = float(np.linalg.norm(eq[i] - ph[j]))
    return _map_edge_monomers(assign_equatorial_r2_r3_edges(cost), monomers)


def assign_equatorial_pi_partners(
    pos: np.ndarray,
    roles_df: pd.DataFrame,
    pole_ph_edges: Sequence[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    """Complete ``(pole, ph)`` edges with the equatorial Py⁺ on the same Ph.

    Each sandwich is three gears: pole Py⁺(*i*) · Ph(*j*) · eq Py⁺(*k*).
    For six monomers, *k* is forbidden from equaling the pole partner of that Ph.
    """
    ordered = [(int(a), int(b)) for a, b in pole_ph_edges]
    ph_to_pole = {ph: pole for pole, ph in ordered}
    forbid = ph_to_pole if len(ph_to_pole) > 2 else None
    eq_edges = lock_cation_pi_eq_edges(
        pos, roles_df, forbidden_src_for_dst=forbid
    )
    ph_to_eq = {int(ph): int(eq) for eq, ph in eq_edges}
    missing = [ph for _, ph in ordered if ph not in ph_to_eq]
    if missing:
        raise ValueError(f"No equatorial Py⁺ assigned for Ph monomers {missing}")
    return [(pole, ph, ph_to_eq[ph]) for pole, ph in ordered]


def lock_cation_pi_sandwiches(
    pos: np.ndarray, roles_df: pd.DataFrame
) -> list[tuple[int, int, int]]:
    """Six ``(pole_mon, ph_mon, eq_mon)`` sandwiches from the current coordinates."""
    return assign_equatorial_pi_partners(
        pos, roles_df, lock_cation_pi_edges(pos, roles_df)
    )


def lock_equator_edges(pos: np.ndarray, roles_df: pd.DataFrame) -> list[tuple[int, int]]:
    """Six R2 → R3 pairs from the current coordinates."""
    monomers = sorted(set(int(m) for m in roles_df["monomer"]))
    r2 = np.stack([_role_point(pos, roles_df, m, "r2") for m in monomers])
    r3 = np.stack([_role_point(pos, roles_df, m, "r3") for m in monomers])
    return _map_edge_monomers(lock_intermolecular_edges(r2, r3), monomers)


def cation_pi_unit_indices(
    roles_df: pd.DataFrame,
    pole_mon: int,
    ph_mon: int,
    eq_mon: Optional[int] = None,
) -> np.ndarray:
    """Atoms of one sandwich: pole Py⁺(*i*), Ph(*j*), equatorial Py⁺(*k*)."""
    if eq_mon is None:
        eq_mon = ph_mon
    idx = (
        _indices_for(roles_df, pole_mon, "py_pole")
        + _indices_for(roles_df, ph_mon, "ph")
        + _indices_for(roles_df, eq_mon, "py_eq")
    )
    return np.unique(np.asarray(idx, dtype=int))


def equator_unit_indices(roles_df: pd.DataFrame, r2_mon: int, r3_mon: int) -> np.ndarray:
    """Atoms of one equatorial edge: eq Py⁺(*i*), R2(*i*), R3(*j*)."""
    idx = (
        _indices_for(roles_df, r2_mon, "py_eq")
        + _indices_for(roles_df, r2_mon, "r2")
        + _indices_for(roles_df, r3_mon, "r3")
    )
    return np.unique(np.asarray(idx, dtype=int))


def cation_pi_pole_column(edge: int, pole_mon: int, ph_mon: int) -> str:
    return f"cation_pi_pole_e{int(edge)}_m{int(pole_mon)}pole_m{int(ph_mon)}ph"


def cation_pi_eq_column(edge: int, eq_mon: int, ph_mon: int) -> str:
    return f"cation_pi_eq_e{int(edge)}_m{int(eq_mon)}eq_m{int(ph_mon)}ph"


def cation_pi_unit_column(edge: int, pole_mon: int, ph_mon: int) -> str:
    """Backward-compatible alias for the π(pole) column."""
    return cation_pi_pole_column(edge, pole_mon, ph_mon)


def equator_d1_column(edge: int, r2_mon: int, r3_mon: int) -> str:
    return f"equator_d1_e{int(edge)}_m{int(r2_mon)}r2_m{int(r3_mon)}r3"


def equator_d2_column(edge: int, r2_mon: int, r3_mon: int) -> str:
    return f"equator_d2_e{int(edge)}_m{int(r2_mon)}cpy_m{int(r3_mon)}r3"


def equator_d2_distance(
    pos: np.ndarray, roles_df: pd.DataFrame, r2_mon: int, r3_mon: int
) -> float:
    """CPy of R2's equatorial Py⁺ (para to the benzene linker) to C3 (R3 ipso)."""
    cpy = _cpy_point(pos, roles_df, r2_mon)
    c3 = _role_point(pos, roles_df, r3_mon, "r3")
    return float(np.linalg.norm(cpy - c3))


def angle_at_vertex_deg(
    a: np.ndarray, vertex: np.ndarray, b: np.ndarray
) -> float:
    """Angle A–vertex–B in degrees (0–180)."""
    va = np.asarray(a, dtype=float) - np.asarray(vertex, dtype=float)
    vb = np.asarray(b, dtype=float) - np.asarray(vertex, dtype=float)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    cos = float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def cation_pi_angle_column(edge: int, ph_mon: int) -> str:
    return f"cation_pi_angle_e{int(edge)}_m{int(ph_mon)}ph"


def equator_angle_column(edge: int, r3_mon: int) -> str:
    return f"equator_angle_e{int(edge)}_m{int(r3_mon)}c3"


def equator_d1_d2_angle_deg(
    pos: np.ndarray, roles_df: pd.DataFrame, r2_mon: int, r3_mon: int
) -> float:
    """Angle at C3 between d1 (C2–C3) and d2 (CPy–C3)."""
    r2 = _role_point(pos, roles_df, r2_mon, "r2")
    r3 = _role_point(pos, roles_df, r3_mon, "r3")
    cpy = _cpy_point(pos, roles_df, r2_mon)
    return angle_at_vertex_deg(r2, r3, cpy)


def cation_pi_unit_label(
    edge: int,
    pole_mon: int,
    ph_mon: int,
    eq_mon: Optional[int] = None,
) -> str:
    """Sidebar / plot label: π(pole) and optional π(eq) faces of one sandwich."""
    text = (
        f"Cation–π unit {int(edge)}: π(pole) Py⁺ {int(pole_mon)} → Ph {int(ph_mon)}"
    )
    if eq_mon is None:
        return text
    return f"{text} · π(eq) Py⁺ {int(eq_mon)} → Ph {int(ph_mon)}"


def equator_unit_label(edge: int, r2_mon: int, r3_mon: int) -> str:
    """Sidebar / plot label: ``Equator unit k: d1 R2 i → R3 j · d2 CPy i → R3 j``."""
    return (
        f"Equator unit {int(edge)}: d1 R2 {int(r2_mon)} → R3 {int(r3_mon)} "
        f"· d2 CPy {int(r2_mon)} → R3 {int(r3_mon)}"
    )


def snapshot_motif_rmsd_refs(
    universe: Any,
    roles_df: pd.DataFrame,
    *,
    motifs: Optional[dict[str, Sequence[str]]] = None,
    ref_frame: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Atom indices and frame-``ref_frame`` coords for each Murata RMSD motif."""
    current = int(getattr(universe.trajectory, "frame", 0))
    universe.trajectory[int(ref_frame)]
    pos = np.asarray(universe.atoms.positions, dtype=float)
    wanted = motifs if motifs is not None else MURATA_RMSD_MOTIFS
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, roles in wanted.items():
        idx = motif_atom_indices(roles_df, roles)
        out[str(name)] = (idx, pos[idx].copy())
    if current != int(ref_frame) and 0 <= current < len(universe.trajectory):
        universe.trajectory[current]
    return out


def _indices_for(roles_df: pd.DataFrame, monomer: int, role: str) -> list[int]:
    hit = roles_df[
        (roles_df["monomer"].astype(int) == int(monomer))
        & (roles_df["role"].astype(str) == str(role))
    ]
    if hit.empty:
        raise KeyError(f"No {role} atoms for monomer {monomer}")
    return [int(x) for x in str(hit.iloc[0]["atom_indices"]).split() if x]


def _scalar_index(roles_df: pd.DataFrame, monomer: int, role: str, column: str) -> int:
    hit = roles_df[
        (roles_df["monomer"].astype(int) == int(monomer))
        & (roles_df["role"].astype(str) == str(role))
    ]
    if hit.empty or pd.isna(hit.iloc[0][column]):
        raise KeyError(f"No {column} for monomer {monomer} role {role}")
    return int(hit.iloc[0][column])


class MurataCriteriaFrameObserver:
    """Accumulate cation–π (ring COM), intramolecular d2, and motif RMSD."""

    def __init__(
        self,
        roles_df: pd.DataFrame,
        *,
        n_frame_rows: int,
        rmsd_atom_indices: Optional[Sequence[int]] = None,
        rmsd_ref_coords: Optional[np.ndarray] = None,
        rmsd_by_motif: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None,
        cation_pi_open_lo: float = CATION_PI_OPEN_LO,
    ):
        from src.Aggregator import ResultsGroup

        self.results: dict[str, Any] = {}
        self._ResultsGroup = ResultsGroup
        self._n_frame_rows = int(n_frame_rows)
        self.cation_pi_open_lo = float(cation_pi_open_lo)
        monomers = sorted(set(int(m) for m in roles_df["monomer"]))
        self.monomers = monomers
        self.ph_indices = [_indices_for(roles_df, m, "ph") for m in monomers]
        self.py_pole_indices = [_indices_for(roles_df, m, "py_pole") for m in monomers]
        self.cpy_index = [_scalar_index(roles_df, m, "py_eq", "cpy_index") for m in monomers]
        self.r3_ipso_index = [_scalar_index(roles_df, m, "r3", "ipso_index") for m in monomers]
        self.cation_pi_specs = cation_pi_pair_columns(len(monomers))
        n_pi = len(self.cation_pi_specs)
        n_mon = len(monomers)
        self.results["cation_pi_dists"] = np.full(
            (self._n_frame_rows, n_pi), np.nan, dtype=float
        )
        self.results["d2_dists"] = np.full(
            (self._n_frame_rows, n_mon), np.nan, dtype=float
        )
        self.results["assembly_rmsd"] = np.full(self._n_frame_rows, np.nan, dtype=float)
        self._rmsd_idx = (
            np.asarray(rmsd_atom_indices, dtype=int)
            if rmsd_atom_indices is not None
            else None
        )
        self._rmsd_ref = (
            np.asarray(rmsd_ref_coords, dtype=float)
            if rmsd_ref_coords is not None
            else None
        )
        self._rmsd_motifs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (midx, mref) in (rmsd_by_motif or {}).items():
            key = f"rmsd_{name}"
            self.results[key] = np.full(self._n_frame_rows, np.nan, dtype=float)
            self._rmsd_motifs[str(name)] = (
                np.asarray(midx, dtype=int),
                np.asarray(mref, dtype=float),
            )

    def get_selections_needed(self) -> list[str]:
        return []

    def on_frame_start(self, iterator: Any) -> None:
        return None

    def on_frame(self, ts: Any, frame_idx: int, universe: Any) -> None:
        idx = int(frame_idx)
        if idx < 0 or idx >= self._n_frame_rows:
            return
        pos = np.asarray(universe.atoms.positions, dtype=float)
        pole_com = np.stack(
            [pos[inds].mean(axis=0) for inds in self.py_pole_indices], axis=0
        )
        ph_com = np.stack([pos[inds].mean(axis=0) for inds in self.ph_indices], axis=0)
        dists = self.results["cation_pi_dists"]
        for k, (_, i, j) in enumerate(self.cation_pi_specs):
            delta = pole_com[i] - ph_com[j]
            dists[idx, k] = float(np.linalg.norm(delta))
        d2 = self.results["d2_dists"]
        for m, (cpy, c3) in enumerate(zip(self.cpy_index, self.r3_ipso_index)):
            d2[idx, m] = float(np.linalg.norm(pos[cpy] - pos[c3]))
        if (
            (self._rmsd_idx is not None and self._rmsd_ref is not None)
            or self._rmsd_motifs
        ):
            from src.TrajectoryMetrics.TrajectoryMetrics import rmsd_value_aligned

            if self._rmsd_idx is not None and self._rmsd_ref is not None:
                self.results["assembly_rmsd"][idx] = float(
                    rmsd_value_aligned(pos[self._rmsd_idx], self._rmsd_ref)
                )
            for name, (midx, mref) in self._rmsd_motifs.items():
                self.results[f"rmsd_{name}"][idx] = float(
                    rmsd_value_aligned(pos[midx], mref)
                )

    def on_frame_end(self, iterator: Any) -> None:
        return None

    def _get_aggregator(self) -> Any:
        lookup = {
            "cation_pi_dists": self._ResultsGroup.ndarray_merge_nonnan,
            "d2_dists": self._ResultsGroup.ndarray_merge_nonnan,
            "assembly_rmsd": self._ResultsGroup.ndarray_merge_nonnan,
        }
        for name in self._rmsd_motifs:
            lookup[f"rmsd_{name}"] = self._ResultsGroup.ndarray_merge_nonnan
        return self._ResultsGroup(lookup=lookup)

    def merge_results(self, other: Any) -> None:
        if other is None:
            return
        payload = other if isinstance(other, dict) else getattr(other, "results", None)
        if not payload:
            return
        keys = ["cation_pi_dists", "d2_dists", "assembly_rmsd"]
        keys.extend(f"rmsd_{name}" for name in self._rmsd_motifs)
        for key in keys:
            other_arr = payload.get(key)
            if other_arr is None:
                continue
            self_arr = self.results[key]
            valid = ~np.isnan(other_arr)
            self_arr[valid] = other_arr[valid]

    def to_dataframe(self) -> pd.DataFrame:
        data: dict[str, Any] = {}
        dists = np.asarray(self.results["cation_pi_dists"], dtype=float)
        for k, (col, _, _) in enumerate(self.cation_pi_specs):
            data[col] = dists[:, k]
        d2 = np.asarray(self.results["d2_dists"], dtype=float)
        for m, mon in enumerate(self.monomers):
            data[d2_column_name(mon)] = d2[:, m]
        data["d2_min"] = np.nanmin(d2, axis=1)
        data["d2_max"] = np.nanmax(d2, axis=1)
        data["assembly_rmsd_to_ref"] = np.asarray(
            self.results["assembly_rmsd"], dtype=float
        )
        for name in self._rmsd_motifs:
            data[f"rmsd_{name}"] = np.asarray(
                self.results[f"rmsd_{name}"], dtype=float
            )
        return pd.DataFrame(data)


def _lookup_pair_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col in df.columns:
        return df[col]
    parsed = parse_site_pair_feature(col)
    if parsed is None:
        return None
    mi, sa, mj, sb = parsed
    alt = site_pair_column(mj, sb, mi, sa)
    if alt in df.columns:
        return df[alt]
    return None


def cation_pi_candidate_site_columns(
    matched: pd.DataFrame,
) -> list[tuple[str, int, int]]:
    """Hull ``endpoint_dist_{i}s{a}_{j}s{b}`` columns for pole-Py⁺(i) → Ph(j)."""
    monomers = sorted(set(int(m) for m in matched["monomer"]))
    out: list[tuple[str, int, int]] = []
    for i in monomers:
        si = role_site_index(matched, i, "py_pole")
        for j in monomers:
            if i == j:
                continue
            sj = role_site_index(matched, j, "ph")
            out.append((site_pair_column(i, si, j, sj), i, j))
    return out


def lock_cation_pi_units(
    features_df: pd.DataFrame,
    pair_cols: Sequence[tuple[str, int, int]],
    *,
    n_monomers: Optional[int] = None,
    open_lo: float = CATION_PI_OPEN_LO,
    edges: Optional[Sequence[tuple[int, int]]] = None,
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Hungarian-lock six pole-Py⁺→Ph units and count openings ≥ ``open_lo``.

    Pass ``edges`` to reuse a previously locked pairing instead of re-assigning.
    """
    n_mon = int(n_monomers) if n_monomers is not None else (
        1 + max(max(i, j) for _, i, j in pair_cols)
    )
    cost = np.full((n_mon, n_mon), np.inf, dtype=float)
    values: dict[tuple[int, int], np.ndarray] = {}
    for col, i, j in pair_cols:
        series = None
        if col in features_df.columns:
            series = features_df[col]
        else:
            alt = cation_pi_column_name(i, j)
            if alt in features_df.columns:
                series = features_df[alt]
            else:
                series = _lookup_pair_series(features_df, col)
        if series is None:
            continue
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        values[(i, j)] = arr
        cost[i, j] = float(np.nanmedian(arr))
    if edges is None:
        edges = assign_equatorial_r2_r3_edges(cost)
    else:
        edges = [(int(i), int(j)) for i, j in edges]
    out = features_df.copy()
    locked = np.full((len(features_df), len(edges)), np.nan, dtype=float)
    for k, (i, j) in enumerate(edges):
        arr = values.get((i, j))
        if arr is None:
            raise KeyError(f"Missing cation–π pair monomer {i} pole → {j} Ph")
        locked[:, k] = arr
        label = f"cation_pi_e{k}_m{i}pole_m{j}ph"
        out[label] = arr
        out[cation_pi_column_name(i, j)] = arr
    opened = locked >= float(open_lo)
    out["n_open_cation_pi"] = np.sum(opened, axis=1).astype(int)
    out["cation_pi_min"] = np.nanmin(locked, axis=1)
    out["cation_pi_max"] = np.nanmax(locked, axis=1)
    return out, list(edges)


def lock_equatorial_d1(
    features_df: pd.DataFrame,
    *,
    n_monomers: int = 6,
    edges: Optional[Sequence[tuple[int, int]]] = None,
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Lock six R2→R3 ipso contacts from ``paper_d1_m{i}r2_m{j}r3`` columns."""
    from .endpoint_features import paper_d1_column_name
    from .murata_d1 import D1_COMPACT_HI, D1_COMPACT_LO, D1_ELONGATED_LO

    candidates: list[tuple[str, int, int]] = []
    for i in range(int(n_monomers)):
        for j in range(int(n_monomers)):
            if i == j:
                continue
            candidates.append((paper_d1_column_name(i, "r2", j, "r3"), i, j))
    cost = np.full((n_monomers, n_monomers), np.inf, dtype=float)
    values: dict[tuple[int, int], np.ndarray] = {}
    for col, i, j in candidates:
        if col not in features_df.columns:
            continue
        arr = pd.to_numeric(features_df[col], errors="coerce").to_numpy(dtype=float)
        values[(i, j)] = arr
        cost[i, j] = float(np.nanmedian(arr))
    if not values:
        return features_df, list(edges or [])
    if edges is None:
        edges = assign_equatorial_r2_r3_edges(cost)
    else:
        edges = [(int(a), int(b)) for a, b in edges]
    out = features_df.copy()
    locked = np.full((len(features_df), len(edges)), np.nan, dtype=float)
    for k, (i, j) in enumerate(edges):
        arr = values.get((i, j))
        if arr is None:
            raise KeyError(f"Missing paper d1 pair m{i}r2–m{j}r3")
        locked[:, k] = arr
        out[f"d1_e{k}_m{i}r2_m{j}r3"] = arr
    compact = (locked >= D1_COMPACT_LO) & (locked <= D1_COMPACT_HI)
    elongated = locked >= D1_ELONGATED_LO
    out["murata_d1_n_compact"] = np.sum(compact, axis=1).astype(int)
    out["murata_d1_n_elongated"] = np.sum(elongated, axis=1).astype(int)
    out["murata_d1_min"] = np.nanmin(locked, axis=1)
    return out, list(edges)


def attach_murata_metastructure_labels(
    frames: pd.DataFrame,
    *,
    n_open_col: str = "n_open_cation_pi",
    n_elongated_col: str = "murata_d1_n_elongated",
) -> pd.DataFrame:
    """Add ``murata_metastructure`` from already-counted openings and d1."""
    if n_open_col not in frames.columns:
        raise KeyError(n_open_col)
    out = frames.copy()
    n_open = pd.to_numeric(out[n_open_col], errors="coerce").fillna(-1).to_numpy(dtype=int)
    if n_elongated_col in out.columns:
        n_elong = (
            pd.to_numeric(out[n_elongated_col], errors="coerce")
            .fillna(0)
            .to_numpy(dtype=int)
        )
    else:
        n_elong = np.zeros(len(out), dtype=int)
    out["murata_metastructure"] = classify_murata_metastructure_columns(n_open, n_elong)
    return out


def snapshot_assembly_rmsd_ref(
    universe: Any,
    *,
    selection: str = "resname MOL",
    ref_frame: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Atom indices and coordinates of ``selection`` at ``ref_frame``."""
    current = int(getattr(universe.trajectory, "frame", 0))
    universe.trajectory[int(ref_frame)]
    sel = universe.select_atoms(selection)
    indices = np.asarray(sel.indices, dtype=int)
    coords = np.asarray(sel.positions, dtype=float).copy()
    if current != int(ref_frame) and 0 <= current < len(universe.trajectory):
        universe.trajectory[current]
    return indices, coords


def occupancy_table(labels: pd.Series) -> pd.DataFrame:
    """Percent occupancy of A/B/C1/C2/other (Murata Table 1 layout)."""
    counts = labels.astype(str).value_counts()
    n = int(len(labels))
    rows = []
    for name in METASTRUCTURES:
        c = int(counts.get(name, 0))
        rows.append(
            {
                "metastructure": name,
                "n_frames": c,
                "percent": (100.0 * c / n) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def occupancy_vs_murata(
    labels: pd.Series,
    *,
    cohort: str,
    guest_filter: str,
) -> pd.DataFrame:
    """Our A/B/C1/C2/other % next to Murata Table 1 when the cohort maps."""
    occ = occupancy_table(labels)
    system = COHORT_TO_MURATA_SYSTEM.get(str(cohort))
    ref = MURATA_TABLE1_PCT.get(system or "", {})
    out = occ.copy()
    out.insert(0, "cohort", cohort)
    out.insert(1, "murata_system", system or "")
    out.insert(2, "guest_filter", guest_filter)
    out["our_percent"] = out["percent"]
    out["murata_percent"] = [
        ref.get(str(name), np.nan) for name in out["metastructure"]
    ]
    out["delta_percent"] = out["our_percent"] - out["murata_percent"]
    return out


def remap_endpoint_cohort_cation_pi(
    changepoints_dir: Path | str,
    topology: Path | str,
    output_dir: Path | str,
    *,
    d1_frames_csv: Optional[Path | str] = None,
) -> dict[str, Path]:
    """Match hull Ph / Py-pole sites, lock six cation–π units, optionally label."""
    import MDAnalysis as mda

    from src.utils.gsa_selections import resolve_selections

    changepoints_dir = Path(changepoints_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = changepoints_dir / "endpoint_features"
    sites_path = feat_dir / "endpoint_sites.csv"
    if not sites_path.is_file():
        raise FileNotFoundError(sites_path)

    u = mda.Universe(str(topology))
    sels = resolve_selections(u, gsa_resname="MOL", n_monomers=6, auto_tooth=False)
    sites_df = pd.read_csv(sites_path)
    role_atoms = canonical_role_atoms_mda(u, sels.monomer_selections)
    matched = match_hull_sites_to_roles(
        sites_df, role_atoms, required_roles=("py_pole", "ph")
    )
    matched_path = output_dir / "cation_pi_role_site_map.csv"
    matched.to_csv(matched_path, index=False)

    csvs = sorted(feat_dir.glob("*_endpoint_features.csv"))
    if not csvs:
        raise FileNotFoundError(f"No feature CSVs in {feat_dir}")
    candidates = cation_pi_candidate_site_columns(matched)
    needed: set[str] = {"traj_id", "frame", "time_ps"}
    for col, _, _ in candidates:
        needed.add(col)
        parsed = parse_site_pair_feature(col)
        if parsed is not None:
            mi, sa, mj, sb = parsed
            needed.add(site_pair_column(mj, sb, mi, sa))

    sample_header = pd.read_csv(csvs[0], nrows=0)
    sample_cols = [c for c in needed if c in sample_header.columns]
    sample = pd.read_csv(csvs[0], usecols=sample_cols)
    locked_sample, edges = lock_cation_pi_units(sample, candidates)
    edge_rows = [
        {
            "edge": k,
            "monomer_pole": i,
            "monomer_ph": j,
            "site_pole": role_site_index(matched, i, "py_pole"),
            "site_ph": role_site_index(matched, j, "ph"),
            "feature": site_pair_column(
                i,
                role_site_index(matched, i, "py_pole"),
                j,
                role_site_index(matched, j, "ph"),
            ),
            "median_A": float(np.nanmedian(locked_sample[f"cation_pi_e{k}_m{i}pole_m{j}ph"])),
        }
        for k, (i, j) in enumerate(edges)
    ]
    edges_path = output_dir / "cation_pi_edges.csv"
    pd.DataFrame(edge_rows).to_csv(edges_path, index=False)

    frames: list[pd.DataFrame] = []
    for csv_path in csvs:
        hdr = pd.read_csv(csv_path, nrows=0)
        cols = [c for c in needed if c in hdr.columns]
        raw = pd.read_csv(csv_path, usecols=cols)
        if "traj_id" not in raw.columns:
            raw.insert(0, "traj_id", csv_path.name.replace("_endpoint_features.csv", ""))
        locked, _ = lock_cation_pi_units(
            raw, candidates, n_monomers=6, edges=edges
        )
        frames.append(locked)
    all_frames = pd.concat(frames, ignore_index=True)

    d1_path = Path(d1_frames_csv) if d1_frames_csv else None
    if d1_path is not None and d1_path.is_file():
        d1 = pd.read_csv(d1_path, usecols=lambda c: c in {
            "traj_id", "frame", "murata_d1_n_elongated", "murata_d1_n_compact",
        })
        all_frames = all_frames.merge(d1, on=["traj_id", "frame"], how="left")
        all_frames = attach_murata_metastructure_labels(all_frames)

    frames_path = output_dir / "murata_cation_pi_frames.csv"
    all_frames.to_csv(frames_path, index=False)

    summary = {
        "n_trajectories": int(all_frames["traj_id"].nunique()) if "traj_id" in all_frames else 1,
        "n_frames": int(len(all_frames)),
        "mean_n_open_cation_pi": float(all_frames["n_open_cation_pi"].mean()),
        "frac_frames_any_open": float((all_frames["n_open_cation_pi"] > 0).mean()),
        "cation_pi_min_mean": float(all_frames["cation_pi_min"].mean()),
    }
    if "murata_metastructure" in all_frames.columns:
        occ = occupancy_table(all_frames["murata_metastructure"])
        for _, row in occ.iterrows():
            summary[f"pct_{row['metastructure']}"] = float(row["percent"])
    summary_path = output_dir / "murata_cation_pi_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    return {
        "cation_pi_role_site_map": matched_path,
        "cation_pi_edges": edges_path,
        "murata_cation_pi_frames": frames_path,
        "murata_cation_pi_summary": summary_path,
    }
