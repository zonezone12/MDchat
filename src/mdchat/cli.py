"""
Rich terminal chat interface for MDChat.

Run with:
    python -m src.mdchat       # from repo root
    mdchat                     # after pip install -e '.[chat]'
    F5 in VS Code / Cursor     # via launch.json
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Protocol

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .chat_persistence import (
    CONVERSATION_FILENAME,
    TRANSCRIPT_FILENAME,
    append_transcript,
    clear_conversation_file,
    conversation_path,
    load_conversation,
    read_transcript_tail,
    save_conversation,
)
from .context import AnalysisContext
from .engine_common import HELP_TEXT, format_welcome
from .llm import create_chat_engine
from .model_catalog import model_choices_for_provider
from .registry import SkillRegistry, get_default_registry


class _ChatEngine(Protocol):
    def send_message(self, user_text: str) -> str: ...
    def reset(self) -> None: ...

console = Console()


def _default_session_output_dir(base: str = "output") -> str:
    """Per CLAUDE.md session layout: ``<base>/YYYY-MM-DD_<slug>/`` (unique per run)."""
    now = datetime.now()
    slug = f"mdchat-{now:%H%M%S}"
    return os.path.join(base, f"{now:%Y-%m-%d}_{slug}")


def _is_resolved_default_output_dir(path: str) -> bool:
    """True when ``path`` is the cwd's ``output`` folder (.env often sets ``./output``)."""
    try:
        resolved = os.path.normpath(os.path.abspath(os.path.expanduser(path.strip())))
        default_out = os.path.normpath(os.path.abspath("output"))
        return resolved == default_out
    except OSError:
        return False


class _RichCallback:
    """Live feedback for the VS Code / Cursor integrated terminal."""

    def __init__(self) -> None:
        self._status = None
        self._start: float = 0

    def on_thinking(self) -> None:
        pass  # handled by the outer console.status context manager

    def on_skill_start(self, name: str, params: dict) -> None:
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        if len(param_str) > 80:
            param_str = param_str[:77] + "..."
        console.print(f"  [bold yellow]>> {name}[/bold yellow]({param_str})")
        self._start = time.monotonic()

    def on_skill_end(self, name: str, success: bool, summary: str) -> None:
        elapsed = time.monotonic() - self._start
        tag = "[bold green]OK[/bold green]" if success else "[bold red]FAIL[/bold red]"
        console.print(f"  {tag} [dim]{name} ({elapsed:.1f}s)[/dim]")

    def on_text_chunk(self, text: str) -> None:
        pass  # final text is rendered by the main loop


def _print_model_picker(current: str | None, choices: list[str]) -> None:
    console.print(f"[bold]Current model:[/bold] {current!r}")
    if not choices:
        console.print(
            "[dim]No catalog (set MDCHAT_MODEL_CHOICES=id1,id2 in .env). "
            "Use: /model <api-model-id>[/dim]"
        )
        return
    console.print("[bold]Models[/bold] [dim](/model <n> or /model <id>)[/dim]")
    for i, mid in enumerate(choices, start=1):
        tag = " [cyan]*[/cyan]" if mid == current else ""
        console.print(f"  {i:2}. {mid}{tag}")
    console.print(
        "[dim]Customize list: MDCHAT_MODEL_CHOICES in .env (comma-separated)[/dim]"
    )


def _handle_slash_command(
    cmd: str,
    engine: _ChatEngine,
    context: AnalysisContext,
    registry: SkillRegistry,
    model_choices: list[str],
    session_dir: str,
    provider: str,
) -> bool:
    """Handle a slash command. Returns True if the command was recognized."""
    parts = cmd.strip().split()
    verb = parts[0].lower()

    if verb in ("/quit", "/exit"):
        console.print("[bold]Goodbye![/bold]")
        sys.exit(0)

    if verb == "/status":
        summary = context.get_state_summary() or "Empty context."
        console.print(Panel(summary, title="Analysis State", border_style="blue"))
        return True

    if verb == "/skills":
        console.print(Panel.fit("[bold]Registered Skills[/bold]", border_style="blue"))
        for s in registry.list_skills():
            ok, _ = s.validate(context)
            status = "[green]ready[/green]" if ok else "[dim]needs prereqs[/dim]"
            console.print(f"  {s.name:30s} {status}  {s.description[:60]}")
        console.print()
        return True

    if verb == "/model":
        current = getattr(engine, "model", None)
        if len(parts) < 2:
            _print_model_picker(current, model_choices)
            return True
        token = parts[1].strip()
        if not hasattr(engine, "model"):
            console.print("[red]This backend does not support /model.[/red]")
            return True
        new_model: str | None = None
        if token.isdigit() and model_choices:
            idx = int(token)
            if 1 <= idx <= len(model_choices):
                new_model = model_choices[idx - 1]
            else:
                console.print(
                    f"[red]Pick 1–{len(model_choices)} or use /model <api-model-id>.[/red]"
                )
                return True
        if new_model is None:
            new_model = " ".join(parts[1:]).strip()
        if not new_model:
            console.print("[red]Model id cannot be empty.[/red]")
            return True
        engine.model = new_model
        console.print(
            f"[green]Model set to[/green] {new_model!r} [dim](this session only)[/dim]"
        )
        return True

    if verb == "/reset":
        engine.reset()
        clear_conversation_file(session_dir)
        console.print(
            "[yellow]Conversation history cleared[/yellow] "
            "[dim](saved chat file removed for this session folder)[/dim]"
        )
        return True

    if verb == "/history":
        tail = read_transcript_tail(session_dir)
        if not tail.strip():
            console.print(
                "[dim]No transcript yet. Chat is logged to mdchat_transcript.md "
                "after each reply when using a session output directory.[/dim]"
            )
        else:
            console.print(Panel(tail, title="Recent chat (transcript)", border_style="blue"))
        return True

    if verb == "/load":
        if len(parts) < 3:
            console.print("[red]Usage: /load <topology> <trajectory>[/red]")
            return True
        topo, traj = parts[1], parts[2]
        load_prompt = f"Load the trajectory with topology={topo} and trajectory={traj}"
        with console.status("[bold cyan]Loading trajectory...[/bold cyan]", spinner="dots"):
            try:
                response = engine.send_message(load_prompt)
            except Exception as exc:
                console.print(f"\n[bold red]Error:[/bold red] {exc}")
                return True
        append_transcript(session_dir, load_prompt, response)
        save_conversation(session_dir, provider, engine)
        console.print()
        console.print(Panel(Markdown(response), title="MDChat", border_style="cyan"))
        console.print()
        return True

    if verb == "/help":
        console.print(Markdown(HELP_TEXT))
        return True

    return False


def run_cli(
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
    output_dir: str | None = None,
) -> None:
    """Main entry point for the MDChat CLI."""

    # -- registry --
    registry = get_default_registry()
    registry.auto_discover()
    n_skills = len(registry.list_skills())
    if output_dir is None or (isinstance(output_dir, str) and not output_dir.strip()):
        session_dir = _default_session_output_dir()
    elif isinstance(output_dir, str) and _is_resolved_default_output_dir(output_dir):
        # MDCHAT_OUTPUT_DIR=./output means "artifacts under output/", not "flat output/ only"
        base = os.path.normpath(os.path.abspath(os.path.expanduser(output_dir.strip())))
        session_dir = _default_session_output_dir(base=base)
    else:
        session_dir = output_dir.strip()
    context = AnalysisContext(output_dir=session_dir)
    model_choices = model_choices_for_provider(provider)

    # -- engine --
    callback = _RichCallback()
    engine = create_chat_engine(
        provider,
        registry,
        context,
        api_key=api_key,
        model=model,
        callback=callback,
    )
    if os.path.isfile(conversation_path(session_dir)):
        ok, info = load_conversation(session_dir, provider, engine)
        if ok:
            console.print(f"[bold green]{info}[/bold green]")
        elif info:
            console.print(f"[yellow]{info}[/yellow]")

    welcome_text = format_welcome(
        n_skills=n_skills,
        output_dir=context.output_dir,
        provider=provider,
        model=getattr(engine, "model", None),
    )
    console.print(Text(welcome_text, style="bold cyan"))
    console.print(
        f"[dim]Session folder:[/dim] {session_dir}\n"
        f"[dim]Chat state:[/dim] {CONVERSATION_FILENAME} · "
        f"[dim]readable log:[/dim] {TRANSCRIPT_FILENAME}\n"
    )

    # -- main loop --
    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold]Goodbye![/bold]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            if _handle_slash_command(
                user_input,
                engine,
                context,
                registry,
                model_choices,
                session_dir,
                provider,
            ):
                continue

        with console.status("[bold cyan]Thinking...[/bold cyan]", spinner="dots"):
            try:
                response = engine.send_message(user_input)
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled.[/yellow]")
                continue
            except Exception as exc:
                console.print(f"\n[bold red]Error:[/bold red] {exc}")
                continue

        append_transcript(session_dir, user_input, response)
        save_conversation(session_dir, provider, engine)

        console.print()
        console.print(Panel(Markdown(response), title="MDChat", border_style="cyan"))
        console.print()
