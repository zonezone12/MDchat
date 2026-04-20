"""
Single-pass trajectory metrics via FrameObserver + TrajectoryIterator.

Computes RMSD, Rg, and pairwise minimum contact distances in one trajectory
pass when registered together on the same iterator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
except ImportError:
    import sys

    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

from ..Aggregator import ResultsGroup
from ..TrajectoryIterator import FrameObserver, TrajectoryIterator

MetricKind = Literal["rmsd", "rg", "contacts"]


@dataclass(frozen=True)
class MetricPassSpec:
    """One metric to accumulate during a single iterator pass."""

    result_key: str
    kind: MetricKind
    selection: str
    ref_frame: int = 0
    selection_b: Optional[str] = None


class StackedMetricsObserver(FrameObserver):
    """
    Observer that evaluates multiple RMSD / Rg / contact metrics in one pass.

    Arrays are sized to *n_frames* (the number of frames that will be visited,
    i.e. the sliced iteration length).
    """

    def __init__(self, specs: List[MetricPassSpec], n_frames: int) -> None:
        super().__init__()
        if n_frames < 1:
            raise ValueError("n_frames must be >= 1")
        self.specs = list(specs)
        self.n_frames = int(n_frames)
        self._result_keys: List[str] = [s.result_key for s in self.specs]
        self._rmsd_ref: dict[str, np.ndarray] = {}
        self._sel_cache: dict[str, mda.AtomGroup] = {}
        self._arrays: dict[str, np.ndarray] = {}

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude MDAnalysis AtomGroups from pickling (parallel workers)."""
        state = self.__dict__.copy()
        state["_sel_cache"] = {}
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _ensure_sel_caches(self, universe: mda.Universe) -> None:
        """Rebuild atom selections after unpickling (parallel worker processes)."""
        if self._sel_cache:
            return
        for spec in self.specs:
            if spec.kind == "rmsd":
                sel = universe.select_atoms(spec.selection)
                self._sel_cache[spec.result_key] = sel
            elif spec.kind == "rg":
                sel = universe.select_atoms(spec.selection)
                self._sel_cache[spec.result_key] = sel
            elif spec.kind == "contacts":
                self._sel_cache[spec.result_key + "__a"] = universe.select_atoms(
                    spec.selection
                )
                self._sel_cache[spec.result_key + "__b"] = universe.select_atoms(
                    spec.selection_b or ""
                )

    def _get_aggregator(self) -> Optional[ResultsGroup]:
        if not self._result_keys:
            return None
        lookup = {
            k: ResultsGroup.ndarray_merge_nonnan for k in self._result_keys
        }
        return ResultsGroup(lookup=lookup)

    def get_selections_needed(self) -> List[str]:
        out: List[str] = []
        for s in self.specs:
            out.append(s.selection)
            if s.kind == "contacts" and s.selection_b:
                out.append(s.selection_b)
        return list(dict.fromkeys(out))

    def on_frame_start(self, iterator: TrajectoryIterator) -> None:
        u = iterator.universe
        for spec in self.specs:
            if spec.kind == "rmsd":
                ref_idx = spec.ref_frame
                if ref_idx < 0 or ref_idx >= len(u.trajectory):
                    raise ValueError(
                        f"ref_frame {ref_idx} out of range for this trajectory"
                    )
                u.trajectory[ref_idx]
                sel = u.select_atoms(spec.selection)
                if len(sel) == 0:
                    raise ValueError(f"Empty selection for RMSD: {spec.selection!r}")
                self._sel_cache[spec.result_key] = sel
                self._rmsd_ref[spec.result_key] = sel.positions.copy()
                arr = np.full(self.n_frames, np.nan, dtype=np.float64)
                self._arrays[spec.result_key] = arr
                self.results[spec.result_key] = arr
            elif spec.kind == "rg":
                sel = u.select_atoms(spec.selection)
                if len(sel) == 0:
                    raise ValueError(f"Empty selection for Rg: {spec.selection!r}")
                self._sel_cache[spec.result_key] = sel
                arr = np.full(self.n_frames, np.nan, dtype=np.float64)
                self._arrays[spec.result_key] = arr
                self.results[spec.result_key] = arr
            elif spec.kind == "contacts":
                a = u.select_atoms(spec.selection)
                b = u.select_atoms(spec.selection_b or "")
                if len(a) == 0 or len(b) == 0:
                    raise ValueError(
                        f"Empty selection for contacts: {spec.selection!r} / "
                        f"{spec.selection_b!r}"
                    )
                self._sel_cache[spec.result_key + "__a"] = a
                self._sel_cache[spec.result_key + "__b"] = b
                arr = np.full(self.n_frames, np.nan, dtype=np.float64)
                self._arrays[spec.result_key] = arr
                self.results[spec.result_key] = arr

    def on_frame(
        self,
        ts: mda.coordinates.base.Timestep,
        frame_idx: int,
        universe: mda.Universe,
    ) -> None:
        self._ensure_sel_caches(universe)
        for spec in self.specs:
            if spec.kind == "rmsd":
                sel = self._sel_cache[spec.result_key]
                ref = self._rmsd_ref[spec.result_key]
                _, rmsd_val = align.rotation_matrix(sel.positions, ref)
                self._arrays[spec.result_key][frame_idx] = rmsd_val
            elif spec.kind == "rg":
                sel = self._sel_cache[spec.result_key]
                coords = sel.positions
                com = sel.center_of_mass()
                rg2 = ((coords - com) ** 2).sum(axis=1).mean()
                self._arrays[spec.result_key][frame_idx] = float(np.sqrt(rg2))
            elif spec.kind == "contacts":
                a = self._sel_cache[spec.result_key + "__a"]
                b = self._sel_cache[spec.result_key + "__b"]
                da = a.positions[:, None, :]
                db = b.positions[None, :, :]
                diff = da - db
                dd = np.sqrt((diff * diff).sum(axis=2))
                self._arrays[spec.result_key][frame_idx] = float(dd.min())

    def on_frame_end(self, iterator: TrajectoryIterator) -> None:
        for key, arr in self._arrays.items():
            self.results[key] = arr
