"""
Gemini backend for MDChat using the ``google-genai`` SDK (function calling).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from .context import AnalysisContext
from .engine_common import (
    EngineMixin,
    MAX_TOOL_ROUNDS,
    MAX_TOKENS,
    MODEL_EMPTY_TEXT_FALLBACK,
    tool_rounds_exceeded_message,
)
from .registry import SkillRegistry

logger = logging.getLogger(__name__)

GEMINI_MODEL_DEFAULT = "gemini-3.1-pro-preview"


class _NullCallback:
    def on_thinking(self) -> None:
        pass

    def on_skill_start(self, name: str, params: dict) -> None:
        pass

    def on_skill_end(self, name: str, success: bool, summary: str) -> None:
        pass

    def on_text_chunk(self, text: str) -> None:
        pass


def _args_to_dict(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    try:
        return dict(args)
    except Exception:
        return {}


class GeminiChatEngine(EngineMixin):
    """Conversation loop with Google Gemini (generate_content + tools)."""

    def __init__(
        self,
        registry: SkillRegistry,
        context: AnalysisContext,
        api_key: Optional[str] = None,
        model: str = GEMINI_MODEL_DEFAULT,
        callback: Any = None,
    ) -> None:
        self.registry = registry
        self.context = context
        self.model = model
        self._contents: List[types.Content] = []
        self.callback: Any = callback or _NullCallback()
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    def _build_tools(self) -> List[types.Tool]:
        declarations: List[types.FunctionDeclaration] = []
        for t in self.registry.to_tool_definitions():
            declarations.append(
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=t["input_schema"],
                )
            )
        return [types.Tool(function_declarations=declarations)]

    def send_message(self, user_text: str) -> str:
        self._contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=user_text)],
            )
        )
        tools = self._build_tools()
        config = types.GenerateContentConfig(
            max_output_tokens=MAX_TOKENS,
            system_instruction=self._build_system_prompt(),
            tools=tools,
        )

        for _ in range(MAX_TOOL_ROUNDS):
            self.callback.on_thinking()

            response = self._client.models.generate_content(
                model=self.model,
                contents=self._contents,
                config=config,
            )

            candidates = getattr(response, "candidates", None) or []
            if not candidates:
                fallback = (getattr(response, "text", None) or "").strip()
                self.callback.on_text_chunk(fallback)
                return fallback or "No response from the model."

            cand = candidates[0]
            mod_content = getattr(cand, "content", None)
            parts = list(getattr(mod_content, "parts", None) or [])

            if mod_content is not None:
                self._contents.append(mod_content)

            function_calls = [p for p in parts if getattr(p, "function_call", None)]
            text_parts = [
                p.text for p in parts
                if getattr(p, "text", None)
            ]

            if not function_calls:
                full_text = "\n".join(t for t in text_parts if t).strip()
                if not full_text and getattr(response, "text", None):
                    full_text = (response.text or "").strip()
                self.callback.on_text_chunk(full_text)
                return full_text or MODEL_EMPTY_TEXT_FALLBACK

            response_parts: List[types.Part] = []
            for part in function_calls:
                fc = part.function_call
                name = fc.name or ""
                args = _args_to_dict(fc.args)
                logger.info("Gemini tool call: %s %s", name, args)
                result_str, skill_ok = self._execute_skill(name, args)
                fr = types.FunctionResponse(
                    name=name,
                    response={"result": result_str},
                    id=getattr(fc, "id", None),
                )
                response_parts.append(types.Part(function_response=fr))

            self._contents.append(
                types.Content(role="user", parts=response_parts),
            )

        return tool_rounds_exceeded_message()

    def reset(self) -> None:
        self._contents.clear()


__all__ = ["GEMINI_MODEL_DEFAULT", "GeminiChatEngine"]
