from __future__ import annotations

import warnings
from typing import Optional

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
except ImportError:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise


class AlignedTrajectory:
    """Store and manage an aligned MDAnalysis Universe.

    By default, aligns the trajectory to the first frame (trajectory[0]) of
    the main structure.
    """

    def __init__(
        self,
        universe: mda.Universe,
        align_sel: Optional[str] = None,
        ref_frame: int = 0,
        in_memory: bool = False,  # Changed default to False to avoid loading full trajectory in memory
    ):
        self.original_universe = universe
        self.align_sel = (
            align_sel
            if align_sel is not None
            else "not water and not name I and not name Na+"
        )
        self.ref_frame = ref_frame
        self.in_memory = in_memory
        self.aligned_universe: Optional[mda.Universe] = None
        self._alignment_performed = False

        # Perform alignment immediately
        self.align()

    def align(self) -> None:
        """Perform trajectory alignment."""
        if self._alignment_performed and self.aligned_universe is not None:
            return

        try:
            # Create a copy of the universe for alignment using the same files
            if hasattr(self.original_universe, "filename") and hasattr(
                self.original_universe.trajectory, "filename"
            ):
                try:
                    self.aligned_universe = mda.Universe(
                    self.original_universe.filename,
                    self.original_universe.trajectory.filename,format=self.original_universe.trajectory.format[0]
                )
                except Exception as e:
                    warnings.warn(f"Failed to create aligned universe: {e}. Using original universe.")
                    self.aligned_universe = self.original_universe
            else:
                warnings.warn(
                    "Cannot create independent copy. "
                    "Alignment will modify original universe."
                )
                self.aligned_universe = self.original_universe

            # Align all frames to the reference frame
            align.AlignTraj(
                self.aligned_universe,
                self.original_universe,
                select=self.align_sel,
                ref_frame=self.ref_frame,
                in_memory=self.in_memory,
            ).run()

            self._alignment_performed = True
        except Exception as e:  # pragma: no cover - defensive
            warnings.warn(f"Alignment failed: {e}. Using original universe.")
            self.aligned_universe = self.original_universe
            self._alignment_performed = False

    def get_aligned_universe(self) -> mda.Universe:
        """Return the aligned universe (aligns on demand if needed)."""
        if self.aligned_universe is None:
            self.align()
        return (
            self.aligned_universe
            if self.aligned_universe is not None
            else self.original_universe
        )

    def realign(
        self, align_sel: Optional[str] = None, ref_frame: Optional[int] = None
    ) -> None:
        """Realign the trajectory with new parameters."""
        if align_sel is not None:
            self.align_sel = align_sel
        if ref_frame is not None:
            self.ref_frame = ref_frame

        self._alignment_performed = False
        self.aligned_universe = None
        self.align()

    def __getattr__(self, name):
        """Delegate attribute access to the aligned universe for convenience."""
        if self.aligned_universe is None:
            self.align()
        if self.aligned_universe is not None:
            return getattr(self.aligned_universe, name)
        return getattr(self.original_universe, name)


