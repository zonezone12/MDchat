#!/usr/bin/env python3
"""
Standalone script to create an animated GIF of guest entry/exit timeline from CSV files.

This script reads the guest_entering_events.csv file (and automatically looks for
guest_entering_stats.csv in the same directory for metadata) and creates an animated GIF version
of the timeline plot.

Usage:
    python scripts/create_guest_timeline_gif.py --csv_path output_guest_entering_events.csv --out_prefix output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import warnings
from src.Plotter import Plotter

def main():
    p = argparse.ArgumentParser(
        description="Create animated GIF of guest entry/exit timeline from CSV files."
    )
    p.add_argument('--csv_path', required=True, 
                   help='Path to guest_entering_events.csv file (will also look for guest_entering_stats.csv for metadata)')
    p.add_argument('--out_prefix', required=True, 
                   help='Output file prefix for the GIF')
    p.add_argument('--fps', type=float, default=20.0,
                   help='Frames per second for the GIF animation (default: 10.0)')
    p.add_argument('--frame_step', type=int, default=5,
                   help='Step size for animation frames (1 = every frame, 10 = every 10th frame, etc., default: 1)')
    p.add_argument('--max_frames', type=int, default=None,
                   help='Optional maximum frame number for x-axis limit')
    p.add_argument('--show_connections', action='store_true',
                   help='Draw horizontal lines connecting entry-exit pairs')
    p.add_argument('--n_jobs', type=int, default=None,
                   help='Number of parallel workers for frame generation. If None, uses all available CPU cores. If 1, runs sequentially.')
    
    args = p.parse_args()
    
    plotter = Plotter()
    
    print(f"Reading guest events from: {args.csv_path}")
    print(f"Creating animated GIF: {args.out_prefix}_guest_entry_exit_timeline.gif")
    print(f"FPS: {args.fps}, Frame step: {args.frame_step}")
    if args.n_jobs:
        print(f"Parallel workers: {args.n_jobs}")
    
    plotter.plot_guest_entry_exit_timeline_gif(
        csv_path=args.csv_path,
        out_prefix=args.out_prefix,
        show_connections=args.show_connections,
        max_frames=args.max_frames,
        fps=args.fps,
        frame_step=args.frame_step,
        n_jobs=args.n_jobs,
    )
    
    print("\nDone!")

if __name__ == '__main__':
    main()

#python create_guest_timeline_gif.py --csv_path output/BMMpM_guest_entering_events.csv --out_prefix output/BMMpM_test_guest_timeline