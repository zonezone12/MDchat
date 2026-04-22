"""
Skill Registry for MDChat.

Manages discovery, registration, and lookup of skills. Generates
tool schemas for the LLM (Anthropic Messages API and Gemini declarations).
"""

from __future__ import annotations

import importlib
import logging
import textwrap
from collections import defaultdict
from typing import Dict, List, Optional, TYPE_CHECKING

from .skill import Skill

if TYPE_CHECKING:
    from .context import AnalysisContext

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Central catalogue of all available MDChat skills."""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        if not skill.name:
            raise ValueError("Skill must have a non-empty name")
        if skill.name in self._skills:
            logger.warning("Overwriting existing skill '%s'", skill.name)
        self._skills[skill.name] = skill
        logger.debug("Registered skill '%s'", skill.name)

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def list_skills(self) -> List[Skill]:
        return list(self._skills.values())

    def get_available(self, context: AnalysisContext) -> List[Skill]:
        """Return only skills whose prerequisites are satisfied."""
        available = []
        for skill in self._skills.values():
            ok, _ = skill.validate(context)
            if ok:
                available.append(skill)
        return available

    def to_tool_definitions(self) -> List[dict]:
        """Tool schemas for every registered skill (Claude + Gemini)."""
        return [s.to_tool_schema() for s in self._skills.values()]

    def get_skills_summary(self) -> str:
        """Human-readable summary of all registered skills (for system prompt)."""
        lines = []
        for s in self._skills.values():
            reqs = ", ".join(s.requires) if s.requires else "none"
            lines.append(f"- **{s.name}** [{s.category}]: {s.description}  (requires: {reqs})")
        return "\n".join(lines)

    def format_analysis_method_catalog(
        self,
        context: Optional["AnalysisContext"] = None,
        *,
        wrap_width: int = 88,
    ) -> str:
        """Full multi-line catalog: every skill, grouped by category, for /analysis."""
        by_cat: Dict[str, List[Skill]] = defaultdict(list)
        for s in sorted(self._skills.values(), key=lambda x: (x.category, x.name)):
            by_cat[s.category].append(s)

        blocks: List[str] = []
        for cat in sorted(by_cat.keys()):
            blocks.append(f"## {cat}\n")
            for s in by_cat[cat]:
                if context is not None:
                    ok, _ = s.validate(context)
                    status = "ready" if ok else "needs prereqs"
                else:
                    status = "n/a (not in MDChat session)"
                blocks.append(f"### {s.name}\n")
                blocks.append(f"**Status (this session):** {status}\n\n")
                desc = " ".join(s.description.split())
                blocks.append(textwrap.fill(desc, width=wrap_width) + "\n\n")
                reqs = ", ".join(s.requires) if s.requires else "none"
                prods = ", ".join(s.produces) if s.produces else "none"
                blocks.append(f"**Requires:** {reqs}  \n**Produces:** {prods}\n\n")
        return "".join(blocks).rstrip() + "\n"

    def auto_discover(self) -> None:
        """Import the built-in skills package so skills self-register."""
        try:
            importlib.import_module("src.mdchat.skills")
            logger.debug("Auto-discovered skills from src.mdchat.skills")
        except Exception:
            logger.exception("Failed to auto-discover skills")


# Module-level singleton
_default_registry = SkillRegistry()


def get_default_registry() -> SkillRegistry:
    return _default_registry
