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
        "Load an MD trajectory from topology and coordinate files. "
        "Does not align by default — after loading, ask the user which "
        "structure to use, then call `set_main_selection` with their "
        "MDAnalysis selection (see residue catalog in / status). "
        "For a one-shot load+align, pass align=true and align_selection. "
        "Supports Amber (.prmtop/.nc), GROMACS (.tpr/.xtc), PDB, and other "
        "formats recognized by MDAnalysis."
    )
    category = "trajectory"
    parameters = [
        Parameter("topology", ParamType.FILE_PATH,
                  "Path to the topology file (e.g., .prmtop, .tpr, .psf, .pdb)"),
        Parameter("trajectory", ParamType.FILE_PATH,
                  "Path to the trajectory file (e.g., .nc, .xtc, .dcd, .trr)"),
        Parameter(
            "format",
            ParamType.STRING,
            "MDAnalysis trajectory format (e.g. 'XTC', 'TRR', 'DCD', 'TRJ'). "
            "Omit for automatic detection from the trajectory filename.",
            required=False,
            default=None,
        ),
        Parameter("align", ParamType.BOOLEAN,
                  "Whether to align the trajectory to the first frame. "
                  "Requires align_selection when true.",
                  required=False, default=False),
        Parameter("align_selection", ParamType.ATOM_SELECTION,
                  "MDAnalysis atom selection for alignment (e.g. 'protein', "
                  "'resname GSA'). Required when align=true. "
                  "When align=false, omit this and use set_main_selection next.",
                  required=False, default=None),
    ]
    requires = []
    produces = [
        "universe",
        "topology_path",
        "trajectory_path",
        "n_frames",
        "n_atoms",
        "residue_catalog",
        "residue_catalog_path",
    ]

    def validate(self, context: AnalysisContext):
        return True, ""

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import MDAnalysis as mda
        from src.AlignedTrajectory import AlignedTrajectory

        topology = params["topology"]
        trajectory = params["trajectory"]
        do_align = params.get("align", False)
        align_sel = params.get("align_selection")
        trj_format = params.get("format")

        for f in (topology, trajectory):
            if not os.path.isfile(f):
                return SkillResult(
                    success=False,
                    error=f"File not found: {f}",
                    summary=f"Cannot load trajectory — file not found: {f}",
                )

        try:
            # Never default to format="TRJ": that forces ASCII/Amber-style parsing
            # and breaks binary trajectories (XTC, DCD, …) — on Windows locale
            # decoders (e.g. cp950) this surfaces as UnicodeDecodeError on XTC magic.
            u = (
                mda.Universe(topology, trajectory, format=trj_format)
                if trj_format
                else mda.Universe(topology, trajectory)
            )
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
        suggested = context.detect_main_selection()
        context.set("suggested_main_selection", suggested)

        if do_align and not align_sel:
            return SkillResult(
                success=False,
                error="align=true requires align_selection",
                summary=(
                    "Cannot align without align_selection. Either set align=false "
                    "and call set_main_selection after the user picks a "
                    "structure, or pass align_selection for a one-shot aligned "
                    "load."
                ),
            )

        if align_sel is not None:
            context.main_selection = align_sel
            context.set("main_selection_pending", False)
        else:
            context._store.pop("main_selection", None)
            context.set("main_selection_pending", True)

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
        if trj_format:
            context.set("trajectory_format", trj_format)
        else:
            context._store.pop("trajectory_format", None)
        context.set("n_frames", u.trajectory.n_frames)
        context.set("n_atoms", u.atoms.n_atoms)

        n_frames = u.trajectory.n_frames
        n_atoms = u.atoms.n_atoms
        residues = u.residues
        resnames = list(set(residues.resnames))

        from ..residue_catalog import (
            build_residue_catalog,
            format_catalog_for_prompt,
            write_residue_catalog_csv,
        )

        catalog_rows = build_residue_catalog(u)
        context.set("residue_catalog", catalog_rows)
        catalog_path = os.path.join(context.output_dir, "residue_catalog.csv")
        write_residue_catalog_csv(catalog_path, catalog_rows)
        context.set("residue_catalog_path", catalog_path)

        if context.main_selection_pending:
            next_step = (
                f"Next: ask which structure to use for alignment and metrics, "
                f"then call set_main_selection with an MDAnalysis selection "
                f"(suggested: {suggested!r})."
            )
            main_line = (
                f"Main selection pending; suggested default {suggested!r}."
            )
        else:
            if do_align:
                next_step = (
                    "Aligned to first frame; main_selection is ready for "
                    "downstream tools."
                )
            else:
                next_step = (
                    "main_selection is set but coordinates were not aligned "
                    "(align=false). Call set_main_selection with align=true to "
                    "align, or proceed with raw coordinates."
                )
            main_line = f"Main selection: '{context.main_selection}'."

        summary = (
            f"Loaded trajectory: {n_frames} frames, {n_atoms} atoms, "
            f"{len(residues)} residues. "
            f"Residue types: {', '.join(sorted(resnames)[:10])}"
            f"{'...' if len(resnames) > 10 else ''}. "
            f"{'Aligned' if do_align else 'Not aligned'} to first frame. "
            f"{main_line} "
            f"Built residue selection catalog: {len(catalog_rows)} (segid,resname) groups — "
            f"use the `selection` column in {catalog_path} or see /status for a preview. "
            f"{next_step}"
        )

        catalog_preview = format_catalog_for_prompt(catalog_rows, max_lines=12)

        return SkillResult(
            success=True,
            data={
                "universe": u,
                "topology_path": topology,
                "trajectory_path": trajectory,
                "n_frames": n_frames,
                "n_atoms": n_atoms,
                "main_selection": context.main_selection,
                "main_selection_pending": context.main_selection_pending,
                "suggested_main_selection": suggested,
                "residue_catalog": catalog_rows,
                "residue_catalog_path": catalog_path,
                "residue_catalog_preview": catalog_preview,
            },
            summary=summary,
            artifacts={"residue_catalog_csv": catalog_path},
        )


class SetMainSelectionSkill(Skill):
    name = "set_main_selection"
    description = (
        "After load_trajectory, set the session main atom selection from the "
        "user’s choice and optionally align the trajectory to the first frame. "
        "Uses the universe already in memory (no second full load). "
        "To use different topology/trajectory files, call load_trajectory again."
    )
    category = "trajectory"
    parameters = [
        Parameter(
            "selection",
            ParamType.ATOM_SELECTION,
            "MDAnalysis atom selection for the primary structure (e.g. "
            "'protein', 'resname GSA', or a catalog `selection` string).",
        ),
        Parameter(
            "align",
            ParamType.BOOLEAN,
            "Align the trajectory to the first frame using this selection.",
            required=False,
            default=True,
        ),
    ]
    requires = ["universe"]
    produces = ["universe", "n_frames", "n_atoms", "main_selection"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        from src.AlignedTrajectory import AlignedTrajectory

        selection = (params.get("selection") or "").strip()
        do_align = params.get("align", True)
        u = context.universe

        if u is None:
            return SkillResult(
                success=False,
                error="no universe loaded",
                summary="Load a trajectory first with load_trajectory.",
            )

        if not selection:
            return SkillResult(
                success=False,
                error="selection is empty",
                summary="Provide a non-empty MDAnalysis atom selection string.",
            )

        try:
            sel_atoms = u.select_atoms(selection)
        except Exception as exc:
            return SkillResult(
                success=False,
                error=f"invalid selection: {exc}",
                summary=f"MDAnalysis could not parse the selection: {exc}",
            )
        if sel_atoms.n_atoms == 0:
            return SkillResult(
                success=False,
                error="selection is empty",
                summary=(
                    "That selection matches zero atoms. Try a residue catalog "
                    "string from /status or a broader selection."
                ),
            )

        if do_align:
            try:
                aligned = AlignedTrajectory(u, align_sel=selection)
                u = aligned.get_aligned_universe()
            except Exception as exc:
                return SkillResult(
                    success=False,
                    error=f"Alignment failed: {exc}",
                    summary=(
                        f"Alignment failed: {exc}. Try a different selection or "
                        "set align=false to keep raw coordinates."
                    ),
                )

        context.set("universe", u)
        context.main_selection = selection
        context.set("main_selection_pending", False)
        n_frames = u.trajectory.n_frames
        n_atoms = u.atoms.n_atoms
        context.set("n_frames", n_frames)
        context.set("n_atoms", n_atoms)

        summary = (
            f"Main selection set to {selection!r}. "
            f"{'Aligned' if do_align else 'Not aligned'} to first frame "
            f"({n_frames} frames, {n_atoms} atoms)."
        )
        return SkillResult(
            success=True,
            data={
                "universe": u,
                "n_frames": n_frames,
                "n_atoms": n_atoms,
                "main_selection": selection,
            },
            summary=summary,
        )


get_default_registry().register(LoadTrajectorySkill())
get_default_registry().register(SetMainSelectionSkill())
