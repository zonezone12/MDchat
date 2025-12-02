from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np

try:
    import MDAnalysis as mda
except ImportError:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise


class FrameGatherer:
    """Gather frame data from trajectory in a single iteration."""

    @staticmethod
    def estimate_memory_usage(
        u: mda.Universe, selection_strings: List[str], bytes_per_float: int = 8
    ) -> float:
        """Estimate memory usage in GB for gathering coordinates."""
        n_frames = len(u.trajectory)
        total_atoms = 0

        for sel_str in selection_strings:
            try:
                sel = u.select_atoms(sel_str)
                total_atoms += len(sel)
            except Exception:
                total_atoms += len(u.atoms) // max(len(selection_strings), 1)

        memory_bytes = n_frames * total_atoms * 3 * bytes_per_float
        return memory_bytes / (1024**3)

    @staticmethod
    def should_use_memory_efficient(
        u: mda.Universe, selection_strings: List[str], memory_limit_gb: float = 2.0
    ) -> bool:
        """Determine if memory-efficient (on-the-fly) processing should be used."""
        estimated_memory = FrameGatherer.estimate_memory_usage(u, selection_strings)
        return estimated_memory > memory_limit_gb

    def __init__(
        self,
        u: mda.Universe,
        selection_strings: List[str],
        force_memory_efficient: Optional[bool] = None,
        memory_limit_gb: float = 2.0,
    ):
        self.universe = u
        self.selection_strings = selection_strings
        self.n_frames = len(u.trajectory)
        self.memory_efficient = False
        self.coordinates: Dict[str, np.ndarray] = {}
        self.frame_indices: np.ndarray = np.array([])
        self.times: np.ndarray = np.array([])
        self._gathered = False

        if force_memory_efficient is None:
            self.memory_efficient = FrameGatherer.should_use_memory_efficient(
                u, selection_strings, memory_limit_gb
            )
        else:
            self.memory_efficient = force_memory_efficient

        if self.memory_efficient:
            self._gather_metadata()
        else:
            self.gather()

    def _gather_metadata(self) -> None:
        """Gather only frame indices and times (memory-efficient)."""
        self.frame_indices = np.zeros(self.n_frames, dtype=int)
        self.times = np.zeros(self.n_frames)

        for frame_idx, ts in enumerate(self.universe.trajectory):
            self.frame_indices[frame_idx] = ts.frame
            self.times[frame_idx] = ts.time

        self._gathered = True

    def gather(self) -> None:
        """Iterate through all frames and gather coordinates for all selections."""
        if self._gathered:
            return

        for sel_str in self.selection_strings:
            sel = self.universe.select_atoms(sel_str)
            n_atoms = len(sel)
            self.coordinates[sel_str] = np.zeros((self.n_frames, n_atoms, 3))

        self.frame_indices = np.zeros(self.n_frames, dtype=int)
        self.times = np.zeros(self.n_frames)

        for frame_idx, ts in enumerate(self.universe.trajectory):
            self.frame_indices[frame_idx] = ts.frame
            self.times[frame_idx] = ts.time
            for sel_str in self.selection_strings:
                sel = self.universe.select_atoms(sel_str)
                self.coordinates[sel_str][frame_idx] = sel.positions.copy()

        self._gathered = True

    def get_coordinates(self, sel_str: str) -> np.ndarray:
        """Get gathered coordinates for a selection string."""
        if self.memory_efficient:
            if sel_str not in self.coordinates:
                sel = self.universe.select_atoms(sel_str)
                n_atoms = len(sel)
                coords = np.zeros((self.n_frames, n_atoms, 3))

                self.universe.trajectory[0]
                for frame_idx, _ in enumerate(self.universe.trajectory):
                    coords[frame_idx] = sel.positions.copy()

                self.coordinates[sel_str] = coords
            return self.coordinates[sel_str]

        if not self._gathered:
            self.gather()
        if sel_str not in self.coordinates:
            raise ValueError(
                f"Selection '{sel_str}' was not included in gathering. "
                f"Available selections: {list(self.coordinates.keys())}"
            )
        return self.coordinates[sel_str]

    def get_frame_indices(self) -> np.ndarray:
        if not self._gathered:
            self.gather()
        return self.frame_indices

    def get_times(self) -> np.ndarray:
        if not self._gathered:
            self.gather()
        return self.times

    def get_n_frames(self) -> int:
        return self.n_frames

    def clear(self) -> None:
        """Clear gathered data to free memory."""
        self.coordinates.clear()
        self.frame_indices = np.array([])
        self.times = np.array([])
        self._gathered = False


