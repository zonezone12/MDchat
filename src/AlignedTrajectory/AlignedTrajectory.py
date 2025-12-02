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
    the main structure. It can also create aligned universes for the first
    and last frames.
    """

    def __init__(
        self,
        universe: mda.Universe,
        align_sel: Optional[str] = None,
        ref_frame: int = 0,
        align_first_and_last: bool = True,
        in_memory: bool = True,
    ):
        self.original_universe = universe
        self.align_sel = (
            align_sel
            if align_sel is not None
            else "not water and not name I and not name Na+"
        )
        self.ref_frame = ref_frame
        self.align_first_and_last = align_first_and_last
        self.in_memory = in_memory
        self.aligned_universe: Optional[mda.Universe] = None
        self.first_frame_universe: Optional[mda.Universe] = None
        self.last_frame_universe: Optional[mda.Universe] = None
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
                self.aligned_universe = mda.Universe(
                    self.original_universe.filename,
                    self.original_universe.trajectory.filename,
                )
            else:
                warnings.warn(
                    "Cannot create independent copy. "
                    "Alignment will modify original universe."
                )
                self.aligned_universe = self.original_universe

            # Align all frames to the reference frame
            align.AlignTraj(
                self.aligned_universe,
                self.aligned_universe,
                select=self.align_sel,
                ref_frame=self.ref_frame,
                in_memory=self.in_memory,
            ).run()

            if self.align_first_and_last and hasattr(
                self.original_universe, "filename"
            ) and hasattr(self.original_universe.trajectory, "filename"):
                # First-frame universe
                self.first_frame_universe = mda.Universe(
                    self.original_universe.filename,
                    self.original_universe.trajectory.filename,
                )
                self.first_frame_universe.trajectory[0]
                align.AlignTraj(
                    self.first_frame_universe,
                    self.first_frame_universe,
                    select=self.align_sel,
                    ref_frame=0,
                    in_memory=self.in_memory,
                ).run()

                # Last-frame universe aligned to first
                self.last_frame_universe = mda.Universe(
                    self.original_universe.filename,
                    self.original_universe.trajectory.filename,
                )
                self.last_frame_universe.trajectory[-1]
                if self.first_frame_universe is not None:
                    align.AlignTraj(
                        self.last_frame_universe,
                        self.first_frame_universe,
                        select=self.align_sel,
                        ref_frame=0,
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

    def get_first_frame_universe(self) -> Optional[mda.Universe]:
        """Get the universe with first frame aligned (if available)."""
        return self.first_frame_universe

    def get_last_frame_universe(self) -> Optional[mda.Universe]:
        """Get the universe with last frame aligned to first (if available)."""
        return self.last_frame_universe

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
        self.first_frame_universe = None
        self.last_frame_universe = None
        self.align()

    def __getattr__(self, name):
        """Delegate attribute access to the aligned universe for convenience."""
        if self.aligned_universe is None:
            self.align()
        if self.aligned_universe is not None:
            return getattr(self.aligned_universe, name)
        return getattr(self.original_universe, name)


