"""
Shared pieces for LLM backends (Anthropic, Gemini): system prompt and skill execution.
"""

from __future__ import annotations

import logging
from typing import Any

from .context import AnalysisContext
from .registry import SkillRegistry
from .skill import SkillResult

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 15
MAX_TOKENS = 4096

SYSTEM_PROMPT_TEMPLATE = """\
You are **MDChat**, an expert assistant for Molecular Dynamics trajectory analysis.

Your role is to help chemist researchers analyze their MD simulation data through \
natural language conversation. You have access to a set of analytical *skills* that \
you can call as tools. Each skill performs a specific analysis on trajectory data.

## How to operate

1. **Understand** the user's question and decide which skill(s) to invoke.
2. **Check prerequisites**: look at the current analysis state below to see what \
data is already available. If a required prerequisite is missing, call the skill \
that produces it first (e.g., load a trajectory before computing RMSD).
3. **Call skills** with appropriate parameters extracted from the conversation. \
If you're unsure about a parameter value, ask the user instead of guessing.
4. **Interpret results** in chemically meaningful language. Don't just repeat \
numbers — explain what they mean for the molecular system.
5. **Suggest follow-up** analyses when appropriate.

## Current analysis state

{context_state}

## Guidelines

- Never fabricate analysis results. Only report what the skills return.
- When a skill fails, explain the error and suggest how to fix it.
- When multiple skills are needed, chain them in the correct dependency order.
- Be concise but scientifically precise.
- If the user asks something outside MD analysis, politely redirect.
- Reference generated artifact file paths so the user can find their plots/data.
"""


class EngineMixin:
    """Shared skill runner and system prompt (used by Anthropic and Gemini engines)."""

    registry: SkillRegistry
    context: AnalysisContext
    callback: Any

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            context_state=self.context.get_state_summary(),
        )

    def _execute_skill(self, tool_name: str, tool_input: dict) -> str:
        skill = self.registry.get(tool_name)
        if skill is None:
            msg = f"Error: Unknown skill '{tool_name}'."
            self.callback.on_skill_end(tool_name, False, msg)
            return msg

        ok, reason = skill.validate(self.context)
        if not ok:
            msg = f"Skill '{tool_name}' cannot run: {reason}"
            self.callback.on_skill_end(tool_name, False, msg)
            return msg

        self.callback.on_skill_start(tool_name, tool_input)
        logger.info("Executing skill '%s' with params: %s", tool_name, tool_input)

        try:
            result: SkillResult = skill.execute(self.context, **tool_input)
        except Exception as exc:
            logger.exception("Skill '%s' raised an unhandled exception", tool_name)
            result = SkillResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                summary=f"Skill '{tool_name}' failed with an unexpected error: {exc}",
            )

        self.context.record_execution(tool_name)

        if result.artifacts:
            for name, path in result.artifacts.items():
                self.context.add_artifact(name, path)

        if result.success:
            for key in skill.produces:
                if key in result.data:
                    self.context.set(key, result.data[key])

        self.callback.on_skill_end(tool_name, result.success, result.summary)
        return result.to_tool_result()


def tool_rounds_exceeded_message() -> str:
    return (
        "I reached the maximum number of tool-use rounds. "
        "Please try a simpler request or break it into steps."
    )
