"""
LLM Engine for MDChat.

Manages conversation with Anthropic Claude using the tool-use API.
Implements the agentic loop: user message -> LLM reasoning -> tool calls ->
skill execution -> feed results back -> repeat until text response.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol

import anthropic

from .context import AnalysisContext
from .registry import SkillRegistry
from .skill import SkillResult

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096
MAX_TOOL_ROUNDS = 15

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


class EventCallback(Protocol):
    """Protocol for UI callbacks during the agentic loop."""

    def on_thinking(self) -> None: ...
    def on_skill_start(self, name: str, params: dict) -> None: ...
    def on_skill_end(self, name: str, success: bool, summary: str) -> None: ...
    def on_text_chunk(self, text: str) -> None: ...


class _NullCallback:
    """No-op implementation of EventCallback."""

    def on_thinking(self) -> None: pass
    def on_skill_start(self, name: str, params: dict) -> None: pass
    def on_skill_end(self, name: str, success: bool, summary: str) -> None: pass
    def on_text_chunk(self, text: str) -> None: pass


class ChatEngine:
    """Manages the conversation loop with Anthropic Claude."""

    def __init__(
        self,
        registry: SkillRegistry,
        context: AnalysisContext,
        api_key: Optional[str] = None,
        model: str = MODEL,
        callback: Optional[EventCallback] = None,
    ) -> None:
        self.registry = registry
        self.context = context
        self.model = model
        self.messages: List[Dict[str, Any]] = []
        self.callback: Any = callback or _NullCallback()
        self.client = (
            anthropic.Anthropic(api_key=api_key) if api_key
            else anthropic.Anthropic()
        )

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            context_state=self.context.get_state_summary(),
        )

    def _execute_skill(self, tool_name: str, tool_input: dict) -> str:
        """Look up and run a skill, returning the result string for the LLM."""
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

        result: SkillResult = skill.execute(self.context, **tool_input)
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

    def send_message(self, user_text: str) -> str:
        """Send a user message and return the final assistant text response.

        Runs the full agentic loop: the LLM may call tools multiple times
        before producing a text response.
        """
        self.messages.append({"role": "user", "content": user_text})
        tools = self.registry.to_tool_definitions()

        for _round in range(MAX_TOOL_ROUNDS):
            self.callback.on_thinking()

            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=self._build_system_prompt(),
                tools=tools,
                messages=self.messages,
            )

            assistant_content: List[Dict[str, Any]] = []
            tool_use_blocks = []

            for block in response.content:
                if block.type == "text":
                    assistant_content.append({
                        "type": "text",
                        "text": block.text,
                    })
                elif block.type == "tool_use":
                    assistant_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
                    tool_use_blocks.append(block)

            self.messages.append({
                "role": "assistant",
                "content": assistant_content,
            })

            if response.stop_reason != "tool_use" or not tool_use_blocks:
                text_parts = [
                    b["text"] for b in assistant_content if b["type"] == "text"
                ]
                full_text = "\n".join(text_parts) if text_parts else ""
                self.callback.on_text_chunk(full_text)
                return full_text

            tool_results = []
            for block in tool_use_blocks:
                result_str = self._execute_skill(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            self.messages.append({
                "role": "user",
                "content": tool_results,
            })

        return (
            "I reached the maximum number of tool-use rounds. "
            "Please try a simpler request or break it into steps."
        )

    def reset(self) -> None:
        """Clear conversation history (keeps context state)."""
        self.messages.clear()
