from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import MDAnalysis as mda
except ImportError:
    import sys

    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

from ..TrajectoryIterator import FrameObserver, TrajectoryIterator
from .TrajectoryMetrics import pca_from_coords, rmsf_from_coords


@dataclass(frozen=True)
class CoordinateAccumulationSpec:
    result_key: str
    kind: str  # "rmsf" | "pca"
    selection: str
    n_components: int = 5


class CoordinateAccumulatorObserver(FrameObserver):
    """Accumulate per-frame coordinates and compute RMSF/PCA after iteration."""

    def __init__(self, specs: List[CoordinateAccumulationSpec], n_frames: int) -> None:
        super().__init__()
        if n_frames < 1:
            raise ValueError("n_frames must be >= 1")
        self.specs = list(specs)
        self.n_frames = int(n_frames)
        self._sel_cache: Dict[str, mda.AtomGroup] = {}
        self._coords_by_selection: Dict[str, np.ndarray] = {}
        self._cursor_by_selection: Dict[str, int] = {}
        self.results: Dict[str, object] = {}

    def get_selections_needed(self) -> List[str]:
        return list(dict.fromkeys(s.selection for s in self.specs))

    def _ensure_sel_caches(self, universe: mda.Universe) -> None:
        if self._sel_cache:
            return
        for spec in self.specs:
            if spec.selection in self._sel_cache:
                continue
            sel = universe.select_atoms(spec.selection)
            if len(sel) == 0:
                raise ValueError(f"Empty selection for {spec.kind}: {spec.selection!r}")
            self._sel_cache[spec.selection] = sel

    def on_frame_start(self, iterator: TrajectoryIterator) -> None:
        u = iterator.universe
        self._ensure_sel_caches(u)
        for selection, sel in self._sel_cache.items():
            self._coords_by_selection[selection] = np.empty(
                (self.n_frames, len(sel), 3),
                dtype=np.float64,
            )
            self._cursor_by_selection[selection] = 0

    def on_frame(
        self,
        ts: mda.coordinates.base.Timestep,
        frame_idx: int,
        universe: mda.Universe,
    ) -> None:
        self._ensure_sel_caches(universe)
        for selection, sel in self._sel_cache.items():
            cursor = self._cursor_by_selection[selection]
            if cursor >= self.n_frames:
                continue
            self._coords_by_selection[selection][cursor, :, :] = sel.positions
            self._cursor_by_selection[selection] = cursor + 1

    def on_frame_end(self, iterator: TrajectoryIterator) -> None:
        for spec in self.specs:
            coords = self._coords_by_selection[spec.selection]
            if spec.kind == "rmsf":
                self.results[spec.result_key] = rmsf_from_coords(coords)
            elif spec.kind == "pca":
                self.results[spec.result_key] = pca_from_coords(
                    coords,
                    n_components=spec.n_components,
                )
            else:
                raise ValueError(f"Unsupported accumulation kind: {spec.kind!r}")

