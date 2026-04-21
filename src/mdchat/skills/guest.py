"""Guest molecule tracking skill wrapping guest_entering."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


def _save_guest_entering_artifacts(stats: dict, out_prefix: str) -> dict[str, str]:
    """
    Write the same CSV layout as ``scripts/run_volume_endpoint_guest_analysis.py``:
    ``{prefix}_guest_entering_stats.csv`` and ``{prefix}_guest_entering_events.csv``.
    """
    artifacts: dict[str, str] = {}

    durations = stats.get("durations_inside") or []
    guest_stats_df = pd.DataFrame(
        {
            "metric": [
                "first_entry_frame",
                "first_entry_time",
                "n_entries",
                "n_exits",
                "total_time_inside",
                "total_time_outside",
                "avg_stay_duration",
                "max_stay_duration",
                "min_stay_duration",
            ],
            "value": [
                stats.get("first_entry_frame"),
                stats.get("first_entry_time"),
                stats.get("n_entries", 0),
                stats.get("n_exits", 0),
                stats.get("total_time_inside", 0.0),
                stats.get("total_time_outside", 0.0),
                float(np.mean(durations)) if len(durations) else None,
                float(np.max(durations)) if len(durations) else None,
                float(np.min(durations)) if len(durations) else None,
            ],
        }
    )
    stats_path = f"{out_prefix}_guest_entering_stats.csv"
    guest_stats_df.to_csv(stats_path, index=False)
    artifacts["guest_entering_stats_csv"] = stats_path

    if stats.get("entry_frames"):
        events_df = pd.DataFrame(
            {
                "event_type": ["entry"] * len(stats["entry_frames"])
                + ["exit"] * len(stats["exit_frames"]),
                "frame": stats["entry_frames"] + stats["exit_frames"],
                "time": stats["entry_times"] + stats["exit_times"],
                "guest_indices": stats["entry_guest_indices"]
                + stats["exit_guest_indices"],
            }
        )
        events_df = events_df.sort_values("frame")
        events_path = f"{out_prefix}_guest_entering_events.csv"
        events_df.to_csv(events_path, index=False)
        artifacts["guest_entering_events_csv"] = events_path

    return artifacts


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
                  "Save guest stats/events CSV (same names as batch scripts: "
                  "{prefix}_guest_entering_stats.csv, {prefix}_guest_entering_events.csv).",
                  required=False, default=True),
        Parameter("filename_prefix", ParamType.STRING,
                  "Output basename in the session folder (same default as plot_guest_timeline).",
                  required=False, default="guest_events"),
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
        filename_prefix = (params.get("filename_prefix") or "guest_events").strip() or "guest_events"

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

        artifacts: dict[str, str] = {}
        if save_csv and isinstance(stats, dict):
            out_prefix = os.path.join(context.output_dir, filename_prefix)
            artifacts.update(_save_guest_entering_artifacts(stats, out_prefix))

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
        total_time_inside = float(stats.get("total_time_inside") or 0.0)
        n_frames = u.trajectory.n_frames

        summary = (
            f"Guest tracking: '{guest_sel}' in '{host_sel}' "
            f"({method} method, {n_frames} frames).\n"
            f"  Entries: {n_entries}, Exits: {n_exits}\n"
            f"  Total time inside host: {total_time_inside:.1f} ps\n"
        )
        if n_entries == 0:
            summary += "  No entry events — guest remained outside the host."
        elif n_exits == 0 and n_entries > 0:
            summary += "  Guest entered and remained inside the host."
        else:
            summary += f"  Guest entered and exited multiple times."

        if artifacts:
            summary += (
                "\n  Saved: "
                + ", ".join(os.path.basename(p) for p in artifacts.values())
            )

        return SkillResult(
            success=True,
            data={"guest_stats": stats},
            artifacts=artifacts,
            summary=summary,
        )


get_default_registry().register(TrackGuestSkill())
