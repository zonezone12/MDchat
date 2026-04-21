"""
Persist MDChat LLM conversation state per session directory.

Saves provider-native message buffers so multi-turn tool use survives restarts
when using the same --output-dir with --resume.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

CONVERSATION_FILENAME = "mdchat_conversation.json"
TRANSCRIPT_FILENAME = "mdchat_transcript.md"


def conversation_path(output_dir: str) -> str:
    return os.path.join(output_dir, CONVERSATION_FILENAME)


def _atomic_write_json(path: str, data: dict[str, Any]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def clear_conversation_file(output_dir: str) -> None:
    path = conversation_path(output_dir)
    if os.path.isfile(path):
        try:
            os.remove(path)
        except OSError:
            pass


def save_conversation(output_dir: str, provider: str, engine: Any) -> None:
    """Snapshot engine message state to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    p = (provider or "anthropic").strip().lower()
    model = getattr(engine, "model", None)
    if hasattr(engine, "messages"):
        data: dict[str, Any] = {
            "format_version": 1,
            "provider": "anthropic",
            "model": model,
            "anthropic_messages": engine.messages,
        }
    elif hasattr(engine, "_contents"):
        contents = engine._contents
        data = {
            "format_version": 1,
            "provider": "gemini",
            "model": model,
            "gemini_contents": [c.model_dump(mode="json") for c in contents],
        }
    else:
        return
    _atomic_write_json(conversation_path(output_dir), data)


def load_conversation(
    output_dir: str, provider: str, engine: Any
) -> tuple[bool, str]:
    """Restore conversation from disk if present and provider matches."""
    path = conversation_path(output_dir)
    if not os.path.isfile(path):
        return False, ""

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"Could not read saved conversation: {exc}"

    saved_provider = str(data.get("provider", "")).strip().lower()
    want = (provider or "anthropic").strip().lower()
    if saved_provider in ("anthropic", "claude"):
        saved_key = "anthropic"
    elif saved_provider in ("gemini", "google"):
        saved_key = "gemini"
    else:
        saved_key = saved_provider

    if want in ("anthropic", "claude"):
        want_key = "anthropic"
    elif want in ("gemini", "google"):
        want_key = "gemini"
    else:
        want_key = want

    if saved_key != want_key:
        return (
            False,
            f"Saved chat is for provider {data.get('provider')!r}; "
            f"current run is {provider!r}. Start with matching MDCHAT_PROVIDER or "
            "remove mdchat_conversation.json.",
        )

    smodel = data.get("model")
    if isinstance(smodel, str) and smodel.strip() and hasattr(engine, "model"):
        engine.model = smodel.strip()

    if saved_key == "anthropic":
        msgs = data.get("anthropic_messages")
        if not isinstance(msgs, list):
            return False, "Invalid anthropic_messages in save file."
        engine.messages = msgs
        n = len(msgs)
    else:
        from google.genai import types

        raw = data.get("gemini_contents")
        if not isinstance(raw, list):
            return False, "Invalid gemini_contents in save file."
        engine._contents = [types.Content.model_validate(d) for d in raw]
        n = len(engine._contents)

    return True, f"Restored {n} API message block(s) from {CONVERSATION_FILENAME}."


def append_transcript(output_dir: str, user_text: str, assistant_text: str) -> None:
    """Append a readable markdown exchange (optional human log)."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, TRANSCRIPT_FILENAME)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    block = (
        f"\n## {ts}\n\n"
        f"**You:**\n\n{user_text}\n\n"
        f"**MDChat:**\n\n{assistant_text}\n\n"
        f"---\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(block)


def read_transcript_tail(output_dir: str, max_chars: int = 12000) -> str:
    path = os.path.join(output_dir, TRANSCRIPT_FILENAME)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return "...[truncated]\n\n" + text[-max_chars:]


__all__ = [
    "CONVERSATION_FILENAME",
    "TRANSCRIPT_FILENAME",
    "append_transcript",
    "clear_conversation_file",
    "conversation_path",
    "load_conversation",
    "read_transcript_tail",
    "save_conversation",
]
