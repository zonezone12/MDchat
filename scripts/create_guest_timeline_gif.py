#!/usr/bin/env python3
"""
CLI: animated GIF of guest entry/exit timeline from ``guest_entering_events.csv``.

Implementation: ``src.Plotter.guest_timeline_gif.create_guest_timeline_gif``.

Usage:
    python scripts/create_guest_timeline_gif.py --csv_path guest_entering_events.csv --out_prefix output/run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.Plotter import create_guest_timeline_gif


def main() -> int:
    p = argparse.ArgumentParser(
        description="Create animated GIF of guest entry/exit timeline from CSV files."
    )
    p.add_argument(
        "--csv_path",
        required=True,
        help="Path to guest_entering_events.csv (guest_entering_stats.csv in the same dir is used if present)",
    )
    p.add_argument(
        "--out_prefix",
        required=True,
        help="Output prefix; writes ``{out_prefix}_guest_entry_exit_timeline.gif``",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="GIF frames per second (default: 20)",
    )
    p.add_argument(
        "--frame_step",
        type=int,
        default=5,
        help="Use every Nth trajectory frame in the animation (default: 5)",
    )
    p.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional cap on frame index for the x-axis",
    )
    p.add_argument(
        "--show_connections",
        action="store_true",
        help="Draw horizontal lines connecting entry-exit pairs",
    )
    p.add_argument(
        "--n_jobs",
        type=int,
        default=None,
        help="Parallel workers for frame rendering (default: all CPUs; 1 = sequential)",
    )

    args = p.parse_args()

    print(f"Reading guest events from: {args.csv_path}")
    print(f"Creating animated GIF: {args.out_prefix}_guest_entry_exit_timeline.gif")
    print(f"FPS: {args.fps}, frame step: {args.frame_step}")
    if args.n_jobs:
        print(f"Parallel workers: {args.n_jobs}")

    create_guest_timeline_gif(
        csv_path=args.csv_path,
        out_prefix=args.out_prefix,
        fps=args.fps,
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        show_connections=args.show_connections,
        n_jobs=args.n_jobs,
    )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
