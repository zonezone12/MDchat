#!/usr/bin/env python3
"""
Simple script to calculate volume from MD trajectory files.

Usage:
    python scripts/calculate_volume.py topology.pdb [trajectory.xtc] [options]

Examples:
    # Calculate volume for a single PDB file
    python scripts/calculate_volume.py structure.pdb

    # Calculate volume for trajectory frames
    python scripts/calculate_volume.py topology.pdb trajectory.xtc

    # Calculate volume for specific frame
    python scripts/calculate_volume.py topology.pdb trajectory.xtc --frame 0

    # Custom atom selection
    python scripts/calculate_volume.py topology.pdb trajectory.xtc --selection "protein"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import MDAnalysis as mda

from src.VolumeAnalyzer.VolumeAnalyzer import VolumeAnalyzer

try:
    import pandas as pd
except ImportError:
    pd = None


def main():
    parser = argparse.ArgumentParser(
        description="Calculate volume from MD trajectory files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "topology",
        help="Topology file (PDB, GRO, etc.)"
    )
    
    parser.add_argument(
        "trajectory",
        nargs="?",
        default=None,
        help="Trajectory file (XTC, TRR, etc.). If not provided, only the first frame from topology is analyzed."
    )
    
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Specific frame index to analyze (default: all frames or frame 0 if no trajectory)"
    )
    
    parser.add_argument(
        "--selection",
        type=str,
        default="not water and not name I and not name Na+",
        help="Atom selection string (default: 'not water and not name I and not name Na+')"
    )
    
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.5,
        help="Grid spacing in Å (default: 1.0)"
    )
    
    parser.add_argument(
        "--probe-radius",
        type=float,
        default=1.4,
        dest="probe_radius",
        help="Probe radius in Å (default: 1.4, ~water size)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file to save results (optional)"
    )
    
    args = parser.parse_args()
    
    # Load universe
    print(f"Loading topology: {args.topology}")
    if args.trajectory:
        print(f"Loading trajectory: {args.trajectory}")
        u = mda.Universe(args.topology, args.trajectory,format="TRJ")
    else:
        u = mda.Universe(args.topology)
        print("No trajectory provided, analyzing single frame from topology file")
    
    # Create VolumeAnalyzer
    print(f"Creating VolumeAnalyzer with selection: '{args.selection}'")
    print(f"  Grid spacing: {args.spacing} Å")
    print(f"  Probe radius: {args.probe_radius} Å")
    
    try:
        analyzer = VolumeAnalyzer(
            universe=u,
            selection=args.selection,
            spacing=args.spacing,
            probe_radius=args.probe_radius,
        )
    except Exception as e:
        print(f"Error creating VolumeAnalyzer: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Calculate volume
    results = []
    
    if args.frame is not None:
        # Single frame
        if args.frame >= len(u.trajectory):
            print(f"Error: Frame {args.frame} is out of range (trajectory has {len(u.trajectory)} frames)", file=sys.stderr)
            sys.exit(1)
        
        print(f"\nCalculating volume for frame {args.frame}...")
        target_vol, cavity_vol = analyzer.compute_frame(args.frame)
        total_vol = target_vol + cavity_vol
        
        results.append({
            "frame": args.frame,
            "time_ps": u.trajectory[args.frame].time,
            "target_volume_A3": target_vol,
            "cavity_volume_A3": cavity_vol,
            "total_volume_A3": total_vol
        })
        
        print(f"\nResults for frame {args.frame}:")
        print(f"  Target volume:  {target_vol:.2f} Å³")
        print(f"  Cavity volume:  {cavity_vol:.2f} Å³")
        print(f"  Total volume:   {total_vol:.2f} Å³")
        
    else:
        # All frames (or single frame if no trajectory)
        if args.trajectory:
            print(f"\nCalculating volume for all {len(u.trajectory)} frames...")
            results = analyzer.analyze_trajectory(stride=1)
            
            # Add total volume
            for r in results:
                r["total_volume_A3"] = r["target_volume_A3"] + r["cavity_volume_A3"]
            
            # Print summary
            target_vols = [r["target_volume_A3"] for r in results]
            cavity_vols = [r["cavity_volume_A3"] for r in results]
            total_vols = [r["total_volume_A3"] for r in results]
            
            print(f"\nSummary over {len(results)} frames:")
            print(f"  Target volume:  {np.mean(target_vols):.2f} ± {np.std(target_vols):.2f} Å³")
            print(f"  Cavity volume:  {np.mean(cavity_vols):.2f} ± {np.std(cavity_vols):.2f} Å³")
            print(f"  Total volume:   {np.mean(total_vols):.2f} ± {np.std(total_vols):.2f} Å³")
            print(f"  Range:          {np.min(total_vols):.2f} - {np.max(total_vols):.2f} Å³")
        else:
            # Single frame from topology
            print(f"\nCalculating volume for frame 0...")
            target_vol, cavity_vol = analyzer.compute_frame(0)
            total_vol = target_vol + cavity_vol
            
            results.append({
                "frame": 0,
                "time_ps": 0.0,
                "target_volume_A3": target_vol,
                "cavity_volume_A3": cavity_vol,
                "total_volume_A3": total_vol
            })
            
            print(f"\nResults:")
            print(f"  Target volume:  {target_vol:.2f} Å³")
            print(f"  Cavity volume:  {cavity_vol:.2f} Å³")
            print(f"  Total volume:   {total_vol:.2f} Å³")
    
    # Save to CSV if requested
    if args.output:
        if pd is None:
            print("Warning: pandas not available. Cannot save CSV file.", file=sys.stderr)
        else:
            df = pd.DataFrame(results)
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

# Calculate volume for a single PDB file
#python calculate_volume.py BMMpM.pdb

# Calculate volume for all frames in a trajectory
#python calculate_volume.py topology.pdb trajectory.xtc

# Calculate volume for a specific frame
#python calculate_volume.py topology.pdb trajectory.xtc --frame 0

# Custom atom selection
#python calculate_volume.py topology.pdb trajectory.xtc --selection "protein"

# Save results to CSV
#python calculate_volume.py topology.pdb trajectory.xtc --output volumes.csv