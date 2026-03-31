#!/usr/bin/env python3
"""
Script to run only Volume and Endpoint Analyzer parts of the trajectory analysis.

This script focuses on:
1. Endpoint-based analysis for specified residues
2. Volume analysis (cube volume from face selections)
3. Correlation analysis between endpoint distances and volume
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import MDAnalysis as mda

from src.AlignedTrajectory import AlignedTrajectory
from src.EndpointAnalyzer import EndpointAnalyzer, EndpointAnalyzerObserver
from src.TrajectoryIterator import TrajectoryIterator
from src.Plotter import Plotter
from src.task import GSAnalyzerObserver

def main():
    p = argparse.ArgumentParser(
        description="Run volume and endpoint analysis on MD trajectories."
    )
    p.add_argument('--top', required=True, help='Topology file (PDB/PSF/PRMTOP/etc.)')
    p.add_argument('--traj', required=True, nargs='+', help='One or more trajectory files (XTC/DCD/TRR/etc.)')
    p.add_argument('--out_prefix', required=True, help='Prefix for outputs (CSV, plots, etc.).')
    p.add_argument('--align_sel', default='resid 1', help='Selection used for trajectory alignment/superposition (default: resid 1).')
    
    # Endpoint-based analysis arguments
    p.add_argument('--endpoint_residues', nargs='+', required=True, 
                   help='List of residue selection strings for endpoint analysis (e.g., "resid 1" "resid 2").')
    p.add_argument('--endpoint_angle_tol', type=float, default=15.0,
                   help='Angle tolerance in degrees for endpoint finding (default: 15.0).')
    p.add_argument('--endpoint_alpha', type=float, default=0.2,
                   help='Graph farness weight for endpoint finding (default: 0.2).')
    
    # Cube volume computation arguments
    p.add_argument('--cube_faces', nargs='+', default=None,
                   help='List of face selection strings for cube volume computation (e.g., "resid 1-10" "resid 11-20").')
    p.add_argument('--guest_sel', default=None,
                   help='Selection for guest molecule (e.g., "I-").')
    p.add_argument('--plot_top_correlations', type=int, default=5,
                   help='Number of top endpoint pairs to plot for volume correlation (default: 5).')
    
    args = p.parse_args()

    # Load trajectory
    print("Loading trajectory...")
    try:
        u = mda.Universe(args.top, *args.traj, format="TRJ")
    except Exception as e:
        warnings.warn(f"Trajectory loading failed: {e}")
        u = mda.Universe(args.top, *args.traj)
    if u is None:
        raise ValueError("Trajectory loading failed")
    
    print(f"Loaded trajectory with {len(u.trajectory)} frames")
    
    # Align trajectory to remove global motion using AlignedTrajectory class
    # Note: We align the full universe first, then filter atoms later for analysis
    print("\nAligning trajectory...")
    try:
        aligned_traj = AlignedTrajectory(
            universe=u,
            align_sel=args.align_sel,
            ref_frame=0,
        )
        u = aligned_traj.get_aligned_universe()
        print("Trajectory alignment completed.")
    except Exception as e:
        warnings.warn(f"Alignment failed or selection invalid: {e}. Using original universe.")
        aligned_traj = None
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_prefix) if os.path.dirname(args.out_prefix) else '.', exist_ok=True)
    
    # Create TrajectoryIterator for single-pass architecture
    print(f"\n{'='*60}")
    print("Setting up single-pass trajectory iteration...")
    print(f"{'='*60}")
    iterator = TrajectoryIterator(u)
    
    # 1) Endpoint-based analysis observer
    endpoint_metrics_df = None
    endpoint_dists_array = None
    endpoint_analyzer = None
    endpoint_observer = None

    print(f"\n{'='*60}")
    print("Setting up endpoint analysis observer...")
    print(f"{'='*60}")
    print(f"Analyzing {len(args.endpoint_residues)} residues:")
    for i, sel in enumerate(args.endpoint_residues):
        print(f"  {i}: {sel}")
    
    try:
        from src.EndpointAnalyzer.endpoints_finder import EndpointsFinder
        endpoints_finder = EndpointsFinder(
            angle_tol_deg=args.endpoint_angle_tol,
            alpha=args.endpoint_alpha
        )
        endpoint_observer = EndpointAnalyzerObserver(
            residue_sel_list=args.endpoint_residues,
            endpoints_finder=endpoints_finder
        )
        iterator.subscribe(endpoint_observer)
        print("Endpoint observer subscribed successfully.")
    except Exception as e:
        warnings.warn(f"Failed to create endpoint observer: {e}")
        import traceback
        traceback.print_exc()
        endpoint_observer = None
    
    # 2) Volume analysis observer (cube volume from face selections)
    volume = None
    cube_metrics_df = None
    volume_observer = None
    
    if args.cube_faces and len(args.cube_faces) > 0:
        print(f"\n{'='*60}")
        print("Setting up volume analysis observer...")
        print(f"{'='*60}")
        print(f"Using {len(args.cube_faces)} face selections:")
        for i, sel in enumerate(args.cube_faces):
            print(f"  Face {i+1}: {sel}")
        if args.guest_sel:
            print(f"Guest selection: {args.guest_sel}")
        
        try:
            volume_observer = GSAnalyzerObserver(
                face_sel_list=args.cube_faces,
                guest_sel=args.guest_sel,
                out_prefix=args.out_prefix
            )
            iterator.subscribe(volume_observer)
            print("Volume observer subscribed successfully.")
        except Exception as e:
            warnings.warn(f"Failed to create volume observer: {e}")
            import traceback
            traceback.print_exc()
            volume_observer = None
    else:
        print("\nNo cube faces provided. Skipping volume analysis.")
        print("Use --cube_faces to enable volume computation.")
    
    # Perform single-pass iteration
    print(f"\n{'='*60}")
    print("Iterating trajectory (single pass)...")
    print(f"{'='*60}")
    iterator.iterate()
    print(f"Completed iteration of {iterator.get_n_frames()} frames.")
    
    # Extract results from observers
    if endpoint_observer is not None:
        endpoint_metrics_df = endpoint_observer.get_endpoint_metrics()
        endpoint_dists_array = endpoint_observer.get_endpoint_distances()
        endpoint_analyzer = EndpointAnalyzer()  # For utility methods
        
        print(f"\nEndpoint metrics computed successfully.")
        print(f"  Found {len(endpoint_metrics_df.columns)} metric columns")
        if endpoint_dists_array:
            print(f"  Computed distances for {endpoint_dists_array.get('n_residues', 0)} residues")
        
        # Save endpoint metrics
        if endpoint_metrics_df is not None and len(endpoint_metrics_df) > 0:
            endpoint_metrics_df.to_csv(f"{args.out_prefix}_endpoint_metrics.csv", index=False)
            print(f"\nEndpoint metrics saved to {args.out_prefix}_endpoint_metrics.csv")
    
    if volume_observer is not None:
        cube_metrics_df = volume_observer.get_metrics_df()
        volume = volume_observer.get_volume()
        
        if volume is not None and len(volume) > 0:
            print(f"\nCube volume computed successfully.")
            print(f"  Mean volume: {np.nanmean(volume):.2f} Å³")
            print(f"  Std volume: {np.nanstd(volume):.2f} Å³")
            print(f"  Min volume: {np.nanmin(volume):.2f} Å³")
            print(f"  Max volume: {np.nanmax(volume):.2f} Å³")
    
    # 3) Correlation analysis between endpoint distances and volume
    correlation_df = None
    
    if volume is not None and endpoint_dists_array is not None and endpoint_analyzer is not None:
        print(f"\n{'='*60}")
        print("Computing correlations between endpoint distances and cube volume...")
        print(f"{'='*60}")
        
        if len(volume) != len(u.trajectory):
            warnings.warn(f"Volume array length ({len(volume)}) doesn't match trajectory length ({len(u.trajectory)}).")
        else:
            try:
                correlation_df = endpoint_analyzer.compute_endpoint_volume_correlation(
                    endpoint_dists_array, volume, args.endpoint_residues
                )
                
                if not correlation_df.empty:
                    correlation_df.to_csv(f"{args.out_prefix}_endpoint_volume_correlation.csv", index=False)
                    print(f"\nCorrelation results saved to {args.out_prefix}_endpoint_volume_correlation.csv")
                    print(f"  Found {len(correlation_df)} endpoint pair correlations")
                    
                    # Analyze variation of endpoint pair distances
                    if isinstance(endpoint_dists_array, dict):
                        print("\nAnalyzing variation of endpoint pair distances...")
                        variation_df = endpoint_analyzer.analyze_endpoint_pair_variation(
                            endpoint_dists_array, args.endpoint_residues
                        )
                        if not variation_df.empty:
                            variation_df.to_csv(f"{args.out_prefix}_endpoint_variation.csv", index=False)
                            print(f"Variation analysis saved to {args.out_prefix}_endpoint_variation.csv")
                            print(f"  Found {len(variation_df)} endpoint pairs with variation data")
                        
                        # Identify key endpoint pairs for expansion/shrinkage
                        print("\nIdentifying key endpoint pairs for expansion/shrinkage...")
                        key_pairs_df = endpoint_analyzer.identify_key_endpoint_pairs_for_expansion(
                            endpoint_dists_array, volume, args.endpoint_residues,
                            variation_threshold=0.1, correlation_threshold=0.7, top_n=10
                        )
                        if not key_pairs_df.empty:
                            key_pairs_df.to_csv(f"{args.out_prefix}_key_endpoint_pairs.csv", index=False)
                            print(f"\nKey endpoint pairs saved to {args.out_prefix}_key_endpoint_pairs.csv")
                            print(f"  Found {len(key_pairs_df)} key endpoint pairs (variation_score >= 0.1, |correlation| >= 0.7)")
                            print("\nTop 5 key endpoint pairs:")
                            for idx, (_, pair) in enumerate(key_pairs_df.head(5).iterrows(), 1):
                                print(f"\n  {idx}. Residue {int(pair['residue_i'])} ({pair['residue_i_sel']}) - "
                                      f"Residue {int(pair['residue_j'])} ({pair['residue_j_sel']})")
                                print(f"     Endpoint pair: ep{int(pair['ep_i_idx'])} - ep{int(pair['ep_j_idx'])}")
                                print(f"     Correlation: {pair['correlation']:.4f} (p={pair['p_value']:.4e})")
                                print(f"     Variation score: {pair['variation_score']:.4f} (CV: {pair['cv']:.4f}, Range: {pair['range_distance']:.2f} Å)")
                                print(f"     Mean distance: {pair['mean_distance']:.2f} Å (std: {pair['std_distance']:.2f} Å)")
                    
                    # Report the single best pair
                    if not correlation_df['correlation'].isna().all():
                        best_pair = correlation_df.iloc[0]
                        print(f"\n{'='*60}")
                        print("Best single endpoint pair (highest correlation):")
                        print(f"{'='*60}")
                        print(f"  Residue {int(best_pair['residue_i'])} ({best_pair['residue_i_sel']}) - "
                              f"Residue {int(best_pair['residue_j'])} ({best_pair['residue_j_sel']})")
                        if 'ep_i_idx' in best_pair and pd.notna(best_pair['ep_i_idx']):
                            print(f"  Endpoint indices: ep{int(best_pair['ep_i_idx'])} - ep{int(best_pair['ep_j_idx'])}")
                        print(f"  Correlation: {best_pair['correlation']:.4f}")
                        print(f"  P-value: {best_pair['p_value']:.4e}")
                        print(f"  Valid data points: {int(best_pair['n_valid_points'])}")
                    
                    # Plot endpoint distances
                    plotter = Plotter()
                    print("\nPlotting endpoint distances over frames...")
                    plotter.plot_endpoint_distances(endpoint_dists_array, args.endpoint_residues, args.out_prefix)
                    
                    # Plot top correlations
                    print(f"Plotting top {args.plot_top_correlations} endpoint-volume correlations...")
                    plotter.plot_endpoint_volume_correlation(
                        endpoint_dists_array, volume, args.endpoint_residues,
                        correlation_df, args.out_prefix, top_n=args.plot_top_correlations
                    )
                    #plot volume change
                    print("\nPlotting volume change over frames...")
                    plotter.plot_volume_change(volume, args.out_prefix)
                    #plot residue endpoints
                    print("\nPlotting residue endpoints...")
                    plotter.plot_residue_endpoints(u, args.endpoint_residues, args.out_prefix)
            except Exception as e:
                warnings.warn(f"Failed to compute endpoint-volume correlations: {e}")
                import traceback
                traceback.print_exc()
    elif volume is None and endpoint_dists_array is not None:
        print("\nWarning: Cube volume not computed. Provide --cube_faces to enable correlation analysis.")
    elif endpoint_dists_array is None and volume is not None:
        print("\nWarning: Endpoint distances not computed. Cannot perform correlation analysis.")
    
    # Summary
    print(f"\n{'='*60}")
    print("Analysis Summary")
    print(f"{'='*60}")
    print(f"Output prefix: {args.out_prefix}")
    print(f"Trajectory frames: {len(u.trajectory)}")
    
    if endpoint_metrics_df is not None:
        print(f"\nEndpoint Analysis:")
        print(f"  ✓ Endpoint metrics computed: {args.out_prefix}_endpoint_metrics.csv")
        if correlation_df is not None and not correlation_df.empty:
            print(f"  ✓ Correlation analysis: {args.out_prefix}_endpoint_volume_correlation.csv")
            if isinstance(endpoint_dists_array, dict):
                print(f"  ✓ Variation analysis: {args.out_prefix}_endpoint_variation.csv")
                print(f"  ✓ Key pairs analysis: {args.out_prefix}_key_endpoint_pairs.csv")
            print(f"  ✓ Endpoint distance plot: {args.out_prefix}_endpoint_distances.png")
            print(f"  ✓ Correlation plot: {args.out_prefix}_endpoint_volume_correlation.png")
    
    if cube_metrics_df is not None:
        print(f"\nVolume Analysis:")
        print(f"  ✓ Cube metrics: {args.out_prefix}_gsa_nanocube.csv")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

#python run_volume_endpoint_analysis.py --top C:\Users\zonezone\Desktop\YCU_research\BHHpH_ca.prmtop --traj C:\Users\zonezone\Desktop\YCU_research\BHHpH_ca_mdcrd_v --out_prefix test\BHHpH --endpoint_residues "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" --cube_faces "resid 1" "resid 2" "resid 3" "resid 4" "resid 5" "resid 6" --guest_sel "resname IOD" --plot_top_correlations 10