"""
Analysis Context for MDChat.

Holds per-conversation state: loaded Universe, file paths, computed results,
and generated artifacts. Acts as a shared blackboard between skills.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from collections import Counter
from typing import Any, Dict, List, Optional, Set

from .residue_catalog import format_catalog_for_prompt


class _SkillLogEntry:
    """Record of a single skill invocation."""

    __slots__ = ("skill_name", "params", "timestamp", "elapsed_s",
                 "success", "summary", "artifacts")

    def __init__(
        self,
        skill_name: str,
        params: Dict[str, Any],
        timestamp: datetime,
        elapsed_s: float = 0.0,
        success: bool = True,
        summary: str = "",
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        self.skill_name = skill_name
        self.params = params
        self.timestamp = timestamp
        self.elapsed_s = elapsed_s
        self.success = success
        self.summary = summary
        self.artifacts = artifacts or {}


class AnalysisContext:
    """Mutable state container shared across skills within a chat session."""

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self._store: Dict[str, Any] = {}
        self._artifacts: Dict[str, str] = {}  # name -> file path
        self._history: List[str] = []  # ordered list of skill names executed
        self._log: List[_SkillLogEntry] = []
        self.session_start = datetime.now()
        self.output_dir = output_dir or os.path.join(
            tempfile.gettempdir(), f"mdchat_{self.session_start:%Y%m%d_%H%M%S}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    # ---- basic key-value state ----

    def set(self, key: str, value: Any) -> None:
        self._store[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._store

    def list_keys(self) -> List[str]:
        return list(self._store.keys())

    # ---- convenience properties for common state ----

    @property
    def universe(self):
        """The loaded MDAnalysis Universe, or None."""
        return self._store.get("universe")

    @universe.setter
    def universe(self, u) -> None:
        self._store["universe"] = u

    @property
    def topology_path(self) -> Optional[str]:
        return self._store.get("topology_path")

    @property
    def trajectory_path(self) -> Optional[str]:
        return self._store.get("trajectory_path")

    @property
    def main_selection(self) -> str:
        """Session-level atom selection for the primary structure of interest.

        After a two-step load, set with ``set_main_selection`` once the user
        picks a structure. Until then (see ``main_selection_pending``), this
        falls back to ``'all'`` so skills still run; prefer calling
        ``set_main_selection`` before RMSD/Rg-style analyses.
        """
        return self._store.get("main_selection", "all")

    @main_selection.setter
    def main_selection(self, sel: str) -> None:
        self._store["main_selection"] = sel

    @property
    def main_selection_pending(self) -> bool:
        """True after ``load_trajectory`` until ``set_main_selection`` completes."""
        return bool(self._store.get("main_selection_pending"))

    # ---- system auto-detection ----

    _WATER_RESNAMES = frozenset({
        "HOH", "WAT", "SOL", "TIP3", "TIP4", "TIP5", "SPC", "T3P", "TP3",
        "TP4", "TP5", "OPC", "TIP",
    })
    _COMMON_ION_RESNAMES = frozenset({
        "Na+", "Cl-", "K+", "Na", "CL", "Cl", "K", "MG", "CA", "ZN", "FE",
        "NA", "SOD", "CLA", "POT", "MG2", "CAL",
    })
    # Standard amino acids and common force-field spellings (buffer / free AA).
    _STANDARD_AA_RESNAMES = frozenset({
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        "ASH", "GLH", "CYX", "CYM", "HID", "HIE", "HIP", "HSD", "HSE", "HSP",
        "MSE", "SEC", "PYL", "ORN", "DAB",
    })
    # If this many (or more) residues use standard AA names, treat as polymer /
    # peptide and do not strip amino acids as “free”.
    _MIN_AA_POLYMER_RESIDUES = 12
    # Any single AA type appears this many times → likely protein / peptide.
    _MIN_AA_REPEAT_FOR_POLYMER = 12
    # Per-resname residue count above this is never labeled “free buffer”.
    _FREE_AA_MAX_PER_RESNAME = 24
    # If the 2nd-largest candidate fragment has at least this fraction of the
    # largest fragment’s atom count, treat sizes as ambiguous (e.g. symmetric
    # dimers) and fall back to the resname heuristic.
    _FRAGMENT_SIZE_AMBIGUITY = 0.88

    @staticmethod
    def _fragment_unique_resnames(ag) -> Set[str]:
        return {str(rn).strip() for rn in ag.resnames}

    def _fragment_is_water_only(self, ag) -> bool:
        rns = self._fragment_unique_resnames(ag)
        return bool(rns) and rns <= self._WATER_RESNAMES

    def _fragment_is_ion_only(self, ag) -> bool:
        rns = self._fragment_unique_resnames(ag)
        return bool(rns) and rns <= self._COMMON_ION_RESNAMES

    def _fragment_based_main_selection(self, u) -> Optional[str]:
        """Use bonded fragments (after ``guess_bonds`` if needed) to pick main species.

        PDB and other topologies often lack bonds; ``atoms.guess_bonds()`` then
        exposes ``fragments`` so the largest non-solvent / non-ion fragment
        (e.g. protein vs. separate MG / ATP-Mg complexes) can be selected via
        ``same fragment as index …``. Mutates the universe by adding guessed
        bonds when none were present.
        """
        from MDAnalysis.exceptions import NoDataError

        try:
            n_bonds = len(u.bonds)
        except NoDataError:
            n_bonds = 0

        if n_bonds == 0:
            try:
                u.atoms.guess_bonds()
            except Exception:
                return None

        try:
            frags = tuple(u.atoms.fragments)
        except NoDataError:
            return None
        except Exception:
            return None

        if not frags:
            return None

        candidates = [
            f for f in frags
            if not self._fragment_is_water_only(f)
            and not self._fragment_is_ion_only(f)
        ]
        if not candidates:
            candidates = [f for f in frags if not self._fragment_is_water_only(f)]
        if not candidates:
            candidates = list(frags)

        ordered = sorted(candidates, key=lambda f: -f.n_atoms)
        if len(ordered) >= 2:
            a0, a1 = ordered[0].n_atoms, ordered[1].n_atoms
            if a1 >= a0 * self._FRAGMENT_SIZE_AMBIGUITY:
                return None

        main = ordered[0]
        if main.n_atoms == 0:
            return None
        try:
            ref_ix = int(main[0].index)
        except Exception:
            return None
        return f"same fragment as index {ref_ix}"

    def detect_main_selection(self) -> str:
        """Infer a sensible atom selection for the primary structure.

        Prefer the **largest bonded fragment** (after ``guess_bonds`` if the
        topology has no bonds), which matches PDB+XTC systems where ions and
        small ligands are separate fragments from the protein.

        Otherwise falls back to excluding common solvent, ions, and heuristic
        “free” amino acids when a larger non–amino-acid solute dominates.
        """
        if self.universe is None:
            return "all"

        u = self.universe
        frag_sel = self._fragment_based_main_selection(u)
        if frag_sel is not None:
            return frag_sel

        residue_counts: Counter[str] = Counter()
        atoms_by_resname: Dict[str, int] = {}
        for res in u.residues:
            rn = res.resname.strip()
            residue_counts[rn] += 1
            atoms_by_resname[rn] = atoms_by_resname.get(rn, 0) + res.atoms.n_atoms

        resnames = set(residue_counts.keys())
        water_present = resnames & self._WATER_RESNAMES
        ions_present = resnames & self._COMMON_ION_RESNAMES

        bulk_exclude = water_present | ions_present
        solute_atoms = sum(
            atoms_by_resname[rn] for rn in resnames if rn not in bulk_exclude
        )
        aa_in_sys: Set[str] = resnames & self._STANDARD_AA_RESNAMES
        atoms_aa = sum(atoms_by_resname[rn] for rn in aa_in_sys if rn not in bulk_exclude)
        atoms_non_aa = solute_atoms - atoms_aa
        n_aa_res = sum(residue_counts[rn] for rn in aa_in_sys if rn not in bulk_exclude)
        max_aa_repeat = max(
            (residue_counts[rn] for rn in aa_in_sys if rn not in bulk_exclude),
            default=0,
        )

        polymer_like = (
            n_aa_res >= self._MIN_AA_POLYMER_RESIDUES
            or max_aa_repeat >= self._MIN_AA_REPEAT_FOR_POLYMER
            or atoms_aa > atoms_non_aa
        )
        free_amino_acid_present: Set[str] = set()
        if aa_in_sys and not polymer_like and atoms_non_aa > atoms_aa:
            free_amino_acid_present = {
                rn for rn in aa_in_sys
                if rn not in bulk_exclude
                and residue_counts[rn] <= self._FREE_AA_MAX_PER_RESNAME
            }

        exclude_parts: List[str] = []
        if water_present:
            exclude_parts.append(" ".join(sorted(water_present)))
        if ions_present:
            exclude_parts.append(" ".join(sorted(ions_present)))
        if free_amino_acid_present:
            exclude_parts.append(" ".join(sorted(free_amino_acid_present)))

        if exclude_parts:
            all_exclude = " ".join(exclude_parts)
            return f"not resname {all_exclude}"

        return "all"

    # ---- artifacts (generated files) ----

    def add_artifact(self, name: str, path: str) -> None:
        self._artifacts[name] = path

    def get_artifacts(self) -> Dict[str, str]:
        return dict(self._artifacts)

    # ---- execution history / work log ----

    def record_execution(
        self,
        skill_name: str,
        params: Optional[Dict[str, Any]] = None,
        elapsed_s: float = 0.0,
        success: bool = True,
        summary: str = "",
        artifacts: Optional[Dict[str, str]] = None,
    ) -> None:
        self._history.append(skill_name)
        self._log.append(_SkillLogEntry(
            skill_name=skill_name,
            params=params or {},
            timestamp=datetime.now(),
            elapsed_s=elapsed_s,
            success=success,
            summary=summary,
            artifacts=artifacts,
        ))

    def get_history(self) -> List[str]:
        return list(self._history)

    def get_log(self) -> List[_SkillLogEntry]:
        return list(self._log)

    # ---- state summary for the LLM system prompt ----

    def get_state_summary(self) -> str:
        """Produce a concise summary of the current analysis state."""
        lines: List[str] = []

        if self.universe is not None:
            u = self.universe
            lines.append(
                f"Trajectory loaded: {self.topology_path or '?'} / "
                f"{self.trajectory_path or '?'}  "
                f"({u.trajectory.n_frames} frames, {u.atoms.n_atoms} atoms)"
            )
            if self.main_selection_pending:
                sug = self._store.get("suggested_main_selection", "")
                lines.append(
                    "Main selection: not confirmed yet. Suggested default: "
                    f"{sug!r}. Confirm with the user, then run set_main_selection "
                    "with their choice (until then tools fall back to 'all')."
                )
            else:
                lines.append(f"Main selection: '{self.main_selection}'")
            cat = self._store.get("residue_catalog")
            cat_path = self._store.get("residue_catalog_path")
            if isinstance(cat, list) and cat:
                lines.append(
                    f"Residue selection catalog: {len(cat)} groups"
                    + (f" ({cat_path})" if cat_path else "")
                )
                lines.append(format_catalog_for_prompt(cat, max_lines=24))
        else:
            lines.append("No trajectory loaded yet.")

        data_keys = [k for k in self._store if k not in {
            "universe", "topology_path", "trajectory_path", "trajectory_format",
            "residue_catalog", "residue_catalog_path",
            "suggested_main_selection", "main_selection_pending", "main_selection",
        }]
        if data_keys:
            lines.append(f"Computed data available: {', '.join(data_keys)}")

        if self._artifacts:
            lines.append(
                "Artifacts: " + ", ".join(
                    f"{k} ({v})" for k, v in self._artifacts.items()
                )
            )

        if self._history:
            lines.append(f"Skills executed so far: {' -> '.join(self._history)}")

        return "\n".join(lines)
