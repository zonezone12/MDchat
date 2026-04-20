"""Guest entry/exit timeline GIF generation (CLI and programmatic entry point)."""

from __future__ import annotations

from typing import Optional


def create_guest_timeline_gif(
    csv_path: str,
    out_prefix: str,
    *,
    fps: float = 20.0,
    frame_step: int = 5,
    max_frames: Optional[int] = None,
    show_connections: bool = False,
    n_jobs: Optional[int] = None,
) -> None:
    """Build an animated GIF from guest event CSV via :class:`~src.Plotter.Plotter.Plotter`."""
    from .Plotter import Plotter

    Plotter().plot_guest_entry_exit_timeline_gif(
        csv_path=csv_path,
        out_prefix=out_prefix,
        show_connections=show_connections,
        max_frames=max_frames,
        fps=fps,
        frame_step=frame_step,
        n_jobs=n_jobs,
    )
