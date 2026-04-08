"""
LLM engines for MDChat: Anthropic Claude and Google Gemini (google-genai).

Each engine runs an agentic loop: user message → model → tool calls →
skill execution → feed results back → repeat until a text response.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import anthropic

from .context import AnalysisContext
from .engine_common import (
    EngineMixin,
    MAX_TOOL_ROUNDS,
    MAX_TOKENS,
    tool_rounds_exceeded_message,
)
from .registry import SkillRegistry

ANTHROPIC_MODEL_DEFAULT = "claude-sonnet-4-20250514"


class _NullCallback:
    def on_thinking(self) -> None:
        pass

    def on_skill_start(self, name: str, params: dict) -> None:
        pass

    def on_skill_end(self, name: str, success: bool, summary: str) -> None:
        pass

    def on_text_chunk(self, text: str) -> None:
        pass


class AnthropicChatEngine(EngineMixin):
    """Conversation loop with Anthropic Claude (Messages API + tools)."""

    def __init__(
        self,
        registry: SkillRegistry,
        context: AnalysisContext,
        api_key: Optional[str] = None,
        model: str = ANTHROPIC_MODEL_DEFAULT,
        callback: Any = None,
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

    def send_message(self, user_text: str) -> str:
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

        return tool_rounds_exceeded_message()

    def reset(self) -> None:
        self.messages.clear()


# Backwards compatibility
ChatEngine = AnthropicChatEngine


def create_chat_engine(
    provider: str,
    registry: SkillRegistry,
    context: AnalysisContext,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    callback: Any = None,
) -> Union[AnthropicChatEngine, "GeminiChatEngine"]:
    """Build the configured LLM engine (anthropic or gemini)."""
    p = (provider or "anthropic").strip().lower()
    if p in ("anthropic", "claude"):
        return AnthropicChatEngine(
            registry,
            context,
            api_key=api_key,
            model=model or ANTHROPIC_MODEL_DEFAULT,
            callback=callback,
        )
    if p in ("gemini", "google"):
        try:
            from .gemini_engine import GEMINI_MODEL_DEFAULT, GeminiChatEngine
        except ImportError as exc:
            raise ImportError(
                "The Gemini provider requires the google-genai package. "
                'Install with: pip install -e ".[gemini]" '
                '(or pip install "google-genai>=1.0.0").'
            ) from exc

        return GeminiChatEngine(
            registry,
            context,
            api_key=api_key,
            model=model or GEMINI_MODEL_DEFAULT,
            callback=callback,
        )
    raise ValueError(
        f"Unknown MDChat provider '{provider}'. Use 'anthropic' or 'gemini'."
    )


__all__ = [
    "ANTHROPIC_MODEL_DEFAULT",
    "AnthropicChatEngine",
    "ChatEngine",
    "create_chat_engine",
]
