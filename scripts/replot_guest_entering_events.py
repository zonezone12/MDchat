#!/usr/bin/env python3
"""
Re-render ``{prefix}_guest_entering_events.png`` from
``{prefix}_guest_entering_events.csv`` (and optional ``*_guest_entering_stats.csv``).

Example::

    python scripts/replot_guest_entering_events.py ^
        --events_csv output/run/run_guest_entering_events.csv ^
        --out_prefix output/run/guest_timeline
"""

from __future__ import annotations

import argparse
import os
import sys

# Repo root on path
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.Plotter.Plotter import (  # noqa: E402
    Plotter,
    guest_residence_stats_from_entering_events_csv,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate guest entering/exiting PNG from CSV.")
    p.add_argument(
        "--events_csv",
        required=True,
        help="Path to *_guest_entering_events.csv",
    )
    p.add_argument(
        "--out_prefix",
        required=True,
        help="Output prefix (same as analysis run: writes {prefix}_guest_entering_events.png)",
    )
    p.add_argument(
        "--stats_csv",
        default=None,
        help="Optional *_guest_entering_stats.csv for stat box metadata",
    )
    p.add_argument(
        "--no-durations",
        action="store_true",
        help="Omit green 'inside host' shading",
    )
    args = p.parse_args()

    stats = guest_residence_stats_from_entering_events_csv(
        args.events_csv,
        args.stats_csv,
    )
    if not stats.get("entry_frames") and not stats.get("exit_frames"):
        print("No entry/exit events found in CSV; nothing to plot.", file=sys.stderr)
        sys.exit(1)

    plotter = Plotter()
    plotter.plot_guest_entering_events(
        stats,
        args.out_prefix,
        show_durations=not args.no_durations,
    )
    out = f"{args.out_prefix}_guest_entering_events.png"
    print(out if os.path.isfile(out) else f"Expected output at {out}")


if __name__ == "__main__":
    main()
