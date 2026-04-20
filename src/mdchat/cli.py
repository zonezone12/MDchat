"""
Rich terminal chat interface for MDChat.

Run with:
    python -m src.mdchat       # from repo root
    mdchat                     # after pip install -e '.[chat]'
    F5 in VS Code / Cursor     # via launch.json
"""

from __future__ import annotations

import sys
import time
from typing import Protocol

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from .context import AnalysisContext
from .engine_common import BANNER, HELP_TEXT, format_welcome
from .llm import create_chat_engine
from .registry import SkillRegistry, get_default_registry


class _ChatEngine(Protocol):
    def send_message(self, user_text: str) -> str: ...
    def reset(self) -> None: ...

console = Console()


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


def _handle_slash_command(
    cmd: str,
    engine: _ChatEngine,
    context: AnalysisContext,
    registry: SkillRegistry,
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

    if verb == "/reset":
        engine.reset()
        console.print("[yellow]Conversation history cleared.[/yellow]")
        return True

    if verb == "/load":
        if len(parts) < 3:
            console.print("[red]Usage: /load <topology> <trajectory>[/red]")
            return True
        topo, traj = parts[1], parts[2]
        with console.status("[bold cyan]Loading trajectory...[/bold cyan]", spinner="dots"):
            response = engine.send_message(
                f"Load the trajectory with topology={topo} and trajectory={traj}"
            )
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

    # -- context --
    context = AnalysisContext(output_dir=output_dir)

    # -- print welcome --
    console.print(Text(BANNER, style="bold cyan"))
    console.print(Markdown(HELP_TEXT))
    console.print(f"[dim]{n_skills} skills registered.[/dim]")
    console.print(f"[dim]Output directory: {context.output_dir}[/dim]")
    console.print(f"[dim]Provider: {provider}[/dim]\n")

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
            if _handle_slash_command(user_input, engine, context, registry):
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

        console.print()
        console.print(Panel(Markdown(response), title="MDChat", border_style="cyan"))
        console.print()
