"""
Curated model ids per provider for interactive /model selection.

Override the list with MDCHAT_MODEL_CHOICES=id1,id2,... (comma-separated, active provider only).
"""

from __future__ import annotations

import os


def _split_env_choices(raw: str | None) -> list[str] | None:
    if raw is None or not str(raw).strip():
        return None
    out = [p.strip() for p in str(raw).split(",") if p.strip()]
    return out or None


def _anthropic_defaults() -> list[str]:
    from .llm import ANTHROPIC_MODEL_DEFAULT

    return [
        ANTHROPIC_MODEL_DEFAULT,
        "claude-opus-4-7",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ]


def _gemini_defaults() -> list[str]:
    from .gemini_engine import GEMINI_MODEL_DEFAULT

    return [
        GEMINI_MODEL_DEFAULT,
        "gemini-3.1-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]


def model_choices_for_provider(provider: str) -> list[str]:
    """Return selectable model ids for the given provider (env override or built-in list)."""
    env = _split_env_choices(os.environ.get("MDCHAT_MODEL_CHOICES"))
    if env is not None:
        return env

    p = (provider or "anthropic").strip().lower()
    if p in ("anthropic", "claude"):
        return _anthropic_defaults()
    if p in ("gemini", "google"):
        return _gemini_defaults()
    return []


__all__ = ["model_choices_for_provider"]
