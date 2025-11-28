#!/usr/bin/env python3
"""
Script to run only Volume and Endpoint Analyzer parts of the trajectory analysis.

This script focuses on:
1. Endpoint-based analysis for specified residues
2. Volume analysis (cube volume from face selections)
3. Correlation analysis between endpoint distances and volume
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd

try:
    import MDAnalysis as mda
except ImportError as e:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

# Import classes from the workflow module
from trajectory_deformation_workflow import (
    EndpointAnalyzer,
    Plotter,
    HAS_ENDPOINTS_FINDER,
    HAS_VOLUME_ANALYZER,
)

from gs_analyzer import GSAnalyzer

def main():
    p = argparse.ArgumentParser(
        description="Run volume and endpoint analysis on MD trajectories."
    )
    p.add_argument('--top', required=True, help='Topology file (PDB/PSF/PRMTOP/etc.)')
    p.add_argument('--traj', required=True, nargs='+', help='One or more trajectory files (XTC/DCD/TRR/etc.)')
    p.add_argument('--out_prefix', required=True, help='Prefix for outputs (CSV, plots, etc.).')
    
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
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_prefix) if os.path.dirname(args.out_prefix) else '.', exist_ok=True)
    
    # 1) Endpoint-based analysis
    endpoint_metrics_df = None
    endpoint_dists_array = None
    endpoint_analyzer = None
    
    if HAS_ENDPOINTS_FINDER:
        print(f"\n{'='*60}")
        print("Computing endpoint-based metrics...")
        print(f"{'='*60}")
        print(f"Analyzing {len(args.endpoint_residues)} residues:")
        for i, sel in enumerate(args.endpoint_residues):
            print(f"  {i}: {sel}")
        
            endpoint_analyzer = EndpointAnalyzer()
            endpoint_metrics_df = endpoint_analyzer.compute_endpoint_metrics(
                u, args.endpoint_residues)
            endpoint_dists_array = endpoint_analyzer.compute_endpoint_distances(
                u, args.endpoint_residues)
            
            print(f"Endpoint metrics computed successfully.")
            print(f"  Found {len(endpoint_metrics_df.columns)} metric columns")
            print(f"  Computed distances for {endpoint_dists_array.get('n_residues', 0)} residues")
            

    else:
        warnings.warn("EndpointsFinder not available. Skipping endpoint analysis.")
        print("Install required dependencies: pip install rdkit")
    
    # Save endpoint metrics
    if endpoint_metrics_df is not None and len(endpoint_metrics_df) > 0:
        endpoint_metrics_df.to_csv(f"{args.out_prefix}_endpoint_metrics.csv", index=False)
        print(f"\nEndpoint metrics saved to {args.out_prefix}_endpoint_metrics.csv")
    
    # 2) Volume analysis (cube volume from face selections)
    volume = None
    cube_metrics_df = None
    
    if args.cube_faces and len(args.cube_faces) > 0:
        print(f"\n{'='*60}")
        print("Computing cube volume from face selections...")
        print(f"{'='*60}")
        print(f"Using {len(args.cube_faces)} face selections:")
        for i, sel in enumerate(args.cube_faces):
            print(f"  Face {i+1}: {sel}")
        if args.guest_sel:
            print(f"Guest selection: {args.guest_sel}")
        
        try:
            gsa_analyzer = GSAnalyzer()
            cube_metrics_df = gsa_analyzer.gsa_nanocube_metrics(
                u, 
                face_sel_list=args.cube_faces, 
                guest_sel=args.guest_sel, 
                out_prefix=args.out_prefix
            )
            
            if 'volume' in cube_metrics_df.columns:
                volume = cube_metrics_df['volume'].values
                print(f"\nCube volume computed successfully.")
                print(f"  Mean volume: {np.nanmean(volume):.2f} Å³")
                print(f"  Std volume: {np.nanstd(volume):.2f} Å³")
                print(f"  Min volume: {np.nanmin(volume):.2f} Å³")
                print(f"  Max volume: {np.nanmax(volume):.2f} Å³")
            else:
                # Fallback: compute from edge_mean if available
                if 'edge_mean' in cube_metrics_df.columns:
                    volume = cube_metrics_df['edge_mean'].values ** 3
                    print(f"\nCube volume computed from edge_mean.")
                    print(f"  Mean volume: {np.nanmean(volume):.2f} Å³")
                else:
                    warnings.warn("No volume column found in cube metrics.")
                    volume = None
                    
        except Exception as e:
            warnings.warn(f"Failed to compute volume from cube faces: {e}")
            import traceback
            traceback.print_exc()
            volume = None
    else:
        print("\nNo cube faces provided. Skipping volume analysis.")
        print("Use --cube_faces to enable volume computation.")
    
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

