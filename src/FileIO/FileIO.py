from __future__ import annotations

import os
from typing import List, Optional

try:
    import MDAnalysis as mda
except ImportError:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise


class FileIO:
    """Handle file input/output operations."""

    def __init__(self) -> None:
        self.output_prefix: Optional[str] = None
        self.saved_frames: List[int] = []
        self.output_directory: Optional[str] = None

    def save_frames_as_pdb(
        self,
        u: mda.Universe,
        frame_indices: List[int],
        out_prefix: str,
        sel: Optional[str] = None,
    ) -> None:
        """Save selected frames as PDB files."""
        self.output_prefix = out_prefix
        output_dir = f"{out_prefix}_frames"
        self.output_directory = output_dir
        os.makedirs(output_dir, exist_ok=True)

        ag = u.select_atoms(sel) if sel is not None else u.atoms
        for idx in frame_indices:
            u.trajectory[idx]
            ag.write(f"{output_dir}/frame_{idx:06d}.pdb")
        self.saved_frames.extend(frame_indices)


