"""Work-log skill — exports a reproducible record of the current session."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class ExportWorkLogSkill(Skill):
    name = "export_work_log"
    description = (
        "Export a structured work log of everything done in this session. "
        "Produces a JSON file (machine-readable, for replication) and a "
        "Markdown file (human-readable summary). Each entry records the "
        "skill name, parameters used, execution time, result summary, "
        "and output artifacts."
    )
    category = "session"
    parameters = [
        Parameter(
            "title", ParamType.STRING,
            "Short title for the session (used in the log header).",
            required=False, default="MDChat session",
        ),
        Parameter(
            "notes", ParamType.STRING,
            "Free-form notes to include in the log (e.g. purpose, system info).",
            required=False, default="",
        ),
    ]
    requires: list = []
    produces = ["work_log_path"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        from datetime import datetime

        title = params.get("title", "MDChat session")
        notes = params.get("notes", "")
        log_entries = context.get_log()

        if not log_entries:
            return SkillResult(
                success=True,
                summary="No skills have been executed yet — nothing to log.",
            )

        session_start = context.session_start.isoformat()
        now = datetime.now()

        records = []
        for entry in log_entries:
            rec = {
                "skill": entry.skill_name,
                "params": _sanitize_params(entry.params),
                "timestamp": entry.timestamp.isoformat(),
                "elapsed_s": round(entry.elapsed_s, 3),
                "success": entry.success,
                "summary": entry.summary,
                "artifacts": entry.artifacts,
            }
            records.append(rec)

        log_data = {
            "title": title,
            "session_start": session_start,
            "session_end": now.isoformat(),
            "output_dir": context.output_dir,
            "notes": notes,
            "steps": records,
            "all_artifacts": dict(context.get_artifacts()),
        }

        json_path = os.path.join(context.output_dir, "work_log.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, default=str)

        md_path = os.path.join(context.output_dir, "work_log.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(_render_markdown(log_data))

        context.set("work_log_path", json_path)
        artifacts = {"work_log.json": json_path, "work_log.md": md_path}

        return SkillResult(
            success=True,
            data={"work_log_path": json_path},
            artifacts=artifacts,
            summary=(
                f"Work log exported ({len(records)} steps). "
                f"JSON: {json_path} | Markdown: {md_path}"
            ),
        )


def _sanitize_params(params: dict) -> dict:
    """Strip large arrays / non-serializable objects from param dict."""
    clean = {}
    for k, v in params.items():
        try:
            json.dumps(v)
            clean[k] = v
        except (TypeError, ValueError, OverflowError):
            clean[k] = repr(v)[:200]
    return clean


def _render_markdown(log_data: dict) -> str:
    lines = [
        f"# {log_data['title']}",
        "",
        f"**Session**: {log_data['session_start']} → {log_data['session_end']}  ",
        f"**Output dir**: `{log_data['output_dir']}`  ",
    ]
    if log_data.get("notes"):
        lines += ["", f"> {log_data['notes']}", ""]

    lines += ["", "## Steps", ""]
    for i, step in enumerate(log_data["steps"], 1):
        status = "OK" if step["success"] else "FAIL"
        lines.append(f"### {i}. `{step['skill']}` [{status}] ({step['elapsed_s']:.1f}s)")
        lines.append("")
        if step["params"]:
            lines.append("**Parameters:**")
            lines.append("")
            for pk, pv in step["params"].items():
                lines.append(f"- `{pk}`: `{pv}`")
            lines.append("")
        lines.append(f"**Result:** {step['summary']}")
        lines.append("")
        if step["artifacts"]:
            lines.append("**Files:**")
            lines.append("")
            for ak, av in step["artifacts"].items():
                lines.append(f"- `{ak}` → `{av}`")
            lines.append("")

    all_arts = log_data.get("all_artifacts", {})
    if all_arts:
        lines += ["## All Generated Files", ""]
        for ak, av in all_arts.items():
            lines.append(f"- `{ak}` → `{av}`")
        lines.append("")

    lines += [
        "---",
        "*Generated by MDChat `export_work_log` skill.*",
        "",
    ]
    return "\n".join(lines)


_registry = get_default_registry()
_registry.register(ExportWorkLogSkill())
