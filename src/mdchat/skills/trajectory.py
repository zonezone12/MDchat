"""Trajectory loading and alignment skill."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class LoadTrajectorySkill(Skill):
    name = "load_trajectory"
    description = (
        "Load an MD trajectory from topology and coordinate files, "
        "optionally aligning it to a reference frame. Supports Amber "
        "(.prmtop/.nc), GROMACS (.tpr/.xtc), PDB, and other formats "
        "recognized by MDAnalysis."
    )
    category = "trajectory"
    parameters = [
        Parameter("topology", ParamType.FILE_PATH,
                  "Path to the topology file (e.g., .prmtop, .tpr, .psf, .pdb)"),
        Parameter("trajectory", ParamType.FILE_PATH,
                  "Path to the trajectory file (e.g., .nc, .xtc, .dcd, .trr)"),
        Parameter("format", ParamType.STRING,
                  "Format of the trajectory file (e.g., 'TRJ', 'TRR', 'DCD', 'XTC')",
                  required=False, default="TRJ"),
        Parameter("align", ParamType.BOOLEAN,
                  "Whether to align the trajectory to the first frame",
                  required=False, default=True),
        Parameter("align_selection", ParamType.ATOM_SELECTION,
                  "MDAnalysis atom selection string for alignment "
                  "(e.g., 'protein', 'resname GSA', 'all'). "
                  "If omitted, auto-detected from system composition.",
                  required=False, default=None),
    ]
    requires = []
    produces = ["universe", "topology_path", "trajectory_path",
                "n_frames", "n_atoms"]

    def validate(self, context: AnalysisContext):
        return True, ""

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import MDAnalysis as mda
        from src.AlignedTrajectory import AlignedTrajectory

        topology = params["topology"]
        trajectory = params["trajectory"]
        do_align = params.get("align", True)
        align_sel = params.get("align_selection")

        for f in (topology, trajectory):
            if not os.path.isfile(f):
                return SkillResult(
                    success=False,
                    error=f"File not found: {f}",
                    summary=f"Cannot load trajectory — file not found: {f}",
                )

        try:
            u = mda.Universe(topology, trajectory,format="TRJ")
        except Exception as exc:
            return SkillResult(
                success=False,
                error=f"MDAnalysis could not read the files: {exc}",
                summary=(
                    f"Failed to load trajectory. MDAnalysis error: {exc}. "
                    "Check that the topology and trajectory formats are compatible."
                ),
            )

        context.set("universe", u)
        main_sel = context.detect_main_selection()
        context.main_selection = main_sel

        if align_sel is None:
            align_sel = main_sel

        if do_align:
            try:
                aligned = AlignedTrajectory(u, align_sel=align_sel)
                u = aligned.get_aligned_universe()
                context.set("universe", u)
            except Exception as exc:
                return SkillResult(
                    success=False,
                    error=f"Alignment failed: {exc}",
                    summary=(
                        f"Trajectory loaded but alignment failed: {exc}. "
                        f"Try a different align_selection or set align=false."
                    ),
                )

        context.set("topology_path", topology)
        context.set("trajectory_path", trajectory)
        context.set("n_frames", u.trajectory.n_frames)
        context.set("n_atoms", u.atoms.n_atoms)

        n_frames = u.trajectory.n_frames
        n_atoms = u.atoms.n_atoms
        residues = u.residues
        resnames = list(set(residues.resnames))

        summary = (
            f"Loaded trajectory: {n_frames} frames, {n_atoms} atoms, "
            f"{len(residues)} residues. "
            f"Residue types: {', '.join(sorted(resnames)[:10])}"
            f"{'...' if len(resnames) > 10 else ''}. "
            f"{'Aligned' if do_align else 'Not aligned'} to first frame. "
            f"Main selection: '{main_sel}'."
        )

        return SkillResult(
            success=True,
            data={
                "universe": u,
                "topology_path": topology,
                "trajectory_path": trajectory,
                "n_frames": n_frames,
                "n_atoms": n_atoms,
                "main_selection": main_sel,
            },
            summary=summary,
        )


get_default_registry().register(LoadTrajectorySkill())
