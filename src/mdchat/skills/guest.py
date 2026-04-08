"""Guest molecule tracking skill wrapping guest_entering."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class TrackGuestSkill(Skill):
    name = "track_guest"
    description = (
        "Track whether a guest molecule (e.g., an ion, small ligand) enters "
        "or exits a host structure (e.g., a nanocage, binding pocket) during "
        "the simulation. Reports entry/exit events, residence times, and "
        "overall statistics."
    )
    category = "guest"
    parameters = [
        Parameter("host_selection", ParamType.ATOM_SELECTION,
                  "MDAnalysis selection for the host structure "
                  "(e.g., 'resname GSA', 'protein', 'resname MOF', "
                  "'resname ZIF')."),
        Parameter("guest_selection", ParamType.ATOM_SELECTION,
                  "MDAnalysis selection for the guest molecule(s) "
                  "(e.g., 'name I', 'resname LIG', 'resname CO2', "
                  "'resname H2O')."),
        Parameter("method", ParamType.STRING,
                  "Detection method: 'distance' (COM-based) or 'volume' "
                  "(VolumeAnalyzer, more accurate but slower).",
                  required=False, default="distance",
                  enum_values=["distance", "volume"]),
        Parameter("distance_threshold", ParamType.FLOAT,
                  "Distance threshold in Angstrom for the distance method. "
                  "If omitted, auto-calculated from host geometry.",
                  required=False, default=None),
        Parameter("save_csv", ParamType.BOOLEAN,
                  "Save guest events as CSV.",
                  required=False, default=True),
    ]
    requires = ["universe"]
    produces = ["guest_stats"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        from src.utils.guest_in import guest_entering

        u = context.universe
        host_sel = params["host_selection"]
        guest_sel = params["guest_selection"]
        method = params.get("method", "distance")
        dist_thresh = params.get("distance_threshold")
        save_csv = params.get("save_csv", True)

        kwargs = dict(
            universe=u,
            host_sel=host_sel,
            guest_sel=guest_sel,
            method=method,
            return_stats=True,
        )
        if dist_thresh is not None:
            kwargs["distance_threshold"] = dist_thresh

        stats = guest_entering(**kwargs)

        context.set("guest_stats", stats)

        artifacts = {}
        if save_csv and isinstance(stats, dict):
            import pandas as pd
            events = stats.get("events", [])
            if events:
                df = pd.DataFrame(events)
                csv_path = os.path.join(context.output_dir, "guest_events.csv")
                df.to_csv(csv_path, index=False)
                artifacts["guest_events_csv"] = csv_path

        if not isinstance(stats, dict):
            entry_frame = stats
            summary = (
                f"Guest entry detected at frame {entry_frame}."
                if entry_frame is not None
                else "No guest entry detected during the trajectory."
            )
            return SkillResult(success=True, summary=summary)

        n_entries = stats.get("n_entries", 0)
        n_exits = stats.get("n_exits", 0)
        total_inside = stats.get("total_frames_inside", 0)
        n_frames = u.trajectory.n_frames
        pct_inside = (total_inside / n_frames * 100) if n_frames > 0 else 0

        summary = (
            f"Guest tracking: '{guest_sel}' in '{host_sel}' "
            f"({method} method, {n_frames} frames).\n"
            f"  Entries: {n_entries}, Exits: {n_exits}\n"
            f"  Frames inside: {total_inside} ({pct_inside:.1f}%)\n"
        )
        if n_entries == 0:
            summary += "  No entry events — guest remained outside the host."
        elif n_exits == 0 and n_entries > 0:
            summary += "  Guest entered and remained inside the host."
        else:
            summary += f"  Guest entered and exited multiple times."

        return SkillResult(
            success=True,
            data={"guest_stats": stats},
            artifacts=artifacts,
            summary=summary,
        )


get_default_registry().register(TrackGuestSkill())
