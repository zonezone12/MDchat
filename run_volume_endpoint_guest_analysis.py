#!/usr/bin/env python3
"""
Script to run Volume, Endpoint, and Guest Entering Analyzer parts of the trajectory analysis.

This script focuses on:
1. Endpoint-based analysis for specified residues
2. Volume analysis (cube volume from face selections) with integrated guest tracking
3. Correlation analysis between endpoint distances and volume
4. Guest entering/exiting analysis (integrated in GSAnalyzerObserver)
"""

import argparse
import os
import warnings
import datetime

import numpy as np
import pandas as pd

try:
    import MDAnalysis as mda
except ImportError as e:
    import sys

    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

try:
    import psutil
except ImportError:
    psutil = None
    warnings.warn("psutil not available. Memory logging will be disabled.")

# Import classes from the new modular src package
from src.AlignedTrajectory import AlignedTrajectory
from src.EndpointAnalyzer import EndpointAnalyzer, EndpointAnalyzerObserver
from src.TrajectoryIterator import TrajectoryIterator
from src.Plotter import Plotter
from src.task import GSAnalyzerObserver
from src.FrameSelection import FrameSelection
from src.utils.monitor import log_memory, auto_limit_workers



def main():
    p = argparse.ArgumentParser(
        description="Run volume, endpoint, and guest entering analysis on MD trajectories."
    )
    p.add_argument('--top', required=True, help='Topology file (PDB/PSF/PRMTOP/etc.)')
    p.add_argument('--traj', required=True, nargs='+', help='One or more trajectory files (XTC/DCD/TRR/etc.)')
    p.add_argument('--out_prefix', required=True, help='Prefix for outputs (CSV, plots, etc.).')
    p.add_argument('--align_sel', default='resid 1-6', help='Selection used for trajectory alignment/superposition (default: resid 1-6).')
    p.add_argument('--in_memory', action='store_true', default=False, help='Load the trajectory into memory before analysis.')
    # Endpoint-based analysis arguments
    p.add_argument('--endpoint_residues', nargs='+', default=['resid 1','resid 2','resid 3','resid 4','resid 5','resid 6'], 
                   help='List of residue selection strings for endpoint analysis (e.g., "resid 1" "resid 2", default: resid 1-6).')
    p.add_argument('--endpoint_angle_tol', type=float, default=15.0,
                   help='Angle tolerance in degrees for endpoint finding (default: 15.0).')
    p.add_argument('--endpoint_alpha', type=float, default=0.2,
                   help='Graph farness weight for endpoint finding (default: 0.2).')
    
    # Cube volume computation arguments (GSAnalyzerObserver handles both volume and guest tracking)
    p.add_argument('--cube_faces', nargs='+', default=['resid 1', 'resid 2', 'resid 3', 'resid 4', 'resid 5', 'resid 6'],
                   help='List of face selection strings for cube volume computation (e.g., "resid 1-10" "resid 11-20", default: resid 1-6).')
    p.add_argument('--guest_sel', default='name I',
                   help='Selection for guest molecule (e.g., "name I"). Used for both volume analysis and guest tracking.')
    p.add_argument('--plot_top_correlations', type=int, default=5,
                   help='Number of top endpoint pairs to plot for volume correlation (default: 5).')
    
    # Guest entering analysis arguments (now handled by GSAnalyzerObserver)
    p.add_argument('--guest_tracking_method', choices=['distance', 'volume'], default='distance',
                   help='Method to determine if guest is inside host: "distance" (faster) or "volume" (more accurate, default: distance).')
    p.add_argument('--guest_distance_threshold', type=float, default=None,
                   help='Distance threshold in Angstrom for guest entering detection (if None, auto-calculate).')
    
    # Parallel processing arguments
    p.add_argument('--n_jobs', type=int, default=-1,
                   help='Number of parallel jobs. Default: 1 (sequential). Use > 1 for parallel processing. Use -1 to use all CPU cores (WARNING: may cause OOM on large trajectories).')
    p.add_argument('--use_dask', action='store_true',
                   help='Use Dask Distributed for parallel processing (requires dask installed).')
    p.add_argument('--auto_limit_workers', action='store_true', default=True,
                   help='Automatically limit number of workers based on available memory. Recommended for HPC environments.')
    
    args = p.parse_args()
    log_memory("Start of script")
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
    log_memory("After loading universe")
    # Align trajectory to remove global motion using AlignedTrajectory class
    # Note: We align the full universe first, then filter atoms later for analysis
    print("\nAligning trajectory...")
    try:
        aligned_traj = AlignedTrajectory(
            universe=u,
            align_sel=args.align_sel,
            ref_frame=0,
            in_memory=args.in_memory,
        )
        u = aligned_traj.get_aligned_universe()
        print("Trajectory alignment completed.")
    except Exception as e:
        warnings.warn(f"Alignment failed or selection invalid: {e}. Using original universe.")
        aligned_traj = None
    log_memory("After aligning trajectory")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out_prefix) if os.path.dirname(args.out_prefix) else '.', exist_ok=True)
    
    # Create TrajectoryIterator for single-pass architecture
    print(f"\n{'='*60}")
    print("Setting up single-pass trajectory iteration...")
    print(f"{'='*60}")
    iterator = TrajectoryIterator(u, use_dask=args.use_dask)
    log_memory("After creating trajectory iterator")
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
    log_memory("After creating endpoint observer")
    # 2) Volume analysis observer with integrated guest tracking (GSAnalyzerObserver)
    volume = None
    cube_metrics_df = None
    volume_observer = None
    guest_stats = None
    if args.cube_faces and len(args.cube_faces) > 0:
        print(f"\n{'='*60}")
        print("Setting up volume analysis observer with guest tracking...")
        print(f"{'='*60}")
        print(f"Using {len(args.cube_faces)} face selections:")
        for i, sel in enumerate(args.cube_faces):
            print(f"  Face {i+1}: {sel}")
        if args.guest_sel:
            print(f"Guest selection: {args.guest_sel}")
            print(f"Guest tracking method: {args.guest_tracking_method}")
            if args.guest_distance_threshold:
                print(f"Distance threshold: {args.guest_distance_threshold} Å")
            else:
                print("Distance threshold: auto-calculate")
        
        try:
            volume_observer = GSAnalyzerObserver(
                face_sel_list=args.cube_faces,
                guest_sel=args.guest_sel,
                out_prefix=args.out_prefix,
                guest_tracking_method=args.guest_tracking_method,
                guest_distance_threshold=args.guest_distance_threshold,
            )
            iterator.subscribe(volume_observer)
            print("Volume observer with guest tracking subscribed successfully.")
        except Exception as e:
            warnings.warn(f"Failed to create volume observer: {e}")
            import traceback
            traceback.print_exc()
            volume_observer = None
    else:
        print("\nNo cube faces provided. Skipping volume analysis.")
        print("Use --cube_faces to enable volume computation.")
    log_memory("After creating volume observer")
    # Perform single-pass iteration
    print(f"\n{'='*60}")
    print("Iterating trajectory (single pass)...")
    print(f"{'='*60}")
    
    # Determine n_jobs for iteration
    n_jobs = args.n_jobs if args.n_jobs is not None else 1
    
    # Auto-limit workers based on memory if requested
    if args.auto_limit_workers and n_jobs != 1:
        print("\nAuto-limiting workers based on available memory...")
        n_jobs = auto_limit_workers(n_jobs, u)
        print(f"Using {n_jobs} worker(s) for parallel processing.")
    elif n_jobs == -1:
        # Warn about using all cores without memory checking
        warnings.warn(
            "Using all CPU cores (n_jobs=-1) without memory checking. "
            "This may cause OOM on large trajectories. "
            "Consider using --auto_limit_workers or set --n_jobs to a specific number."
        )
    
    iterator.iterate(n_jobs=n_jobs)
    print(f"Completed iteration of {iterator.get_n_frames()} frames.")
    log_memory("After performing single-pass iteration")
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
        
        # Diagnostic information
        if volume is None or len(volume) == 0:
            print(f"\nWarning: Volume array is empty!")
            print(f"  Metrics DataFrame shape: {cube_metrics_df.shape if cube_metrics_df is not None else 'None'}")
            print(f"  Metrics DataFrame columns: {list(cube_metrics_df.columns) if cube_metrics_df is not None and len(cube_metrics_df) > 0 else 'None'}")
            # Get diagnostic information from observer
            if hasattr(volume_observer, 'get_diagnostic_info'):
                diag_info = volume_observer.get_diagnostic_info()
                print(f"  Observer diagnostic information:")
                for key, value in diag_info.items():
                    print(f"    {key}: {value}")
            else:
                # Fallback to checking attributes directly
                if hasattr(volume_observer, '_initialized'):
                    print(f"  Observer initialized: {volume_observer._initialized}")
                if hasattr(volume_observer, 'rows'):
                    print(f"  Number of rows collected: {len(volume_observer.rows)}")
            warnings.warn(
                f"Volume array is empty. This may indicate that the volume observer "
                f"did not process any frames. Check that the observer was properly "
                f"subscribed and that frame processing completed without errors."
            )
        elif volume is not None and len(volume) > 0:
            print(f"\nCube volume computed successfully.")
            print(f"  Mean volume: {np.nanmean(volume):.2f} Å³")
            print(f"  Std volume: {np.nanstd(volume):.2f} Å³")
            print(f"  Min volume: {np.nanmin(volume):.2f} Å³")
            print(f"  Max volume: {np.nanmax(volume):.2f} Å³")
        
        # Get guest residence statistics from GSAnalyzerObserver
        if args.guest_sel:
            guest_stats = volume_observer.get_guest_residence_stats()
            
            print(f"\n{'='*60}")
            print("Guest Entering Analysis Results")
            print(f"{'='*60}")
            
            if guest_stats and guest_stats.get('n_entries', 0) > 0:
                print(f"First entry frame: {guest_stats['first_entry_frame']}")
                if guest_stats['first_entry_time'] is not None:
                    print(f"First entry time: {guest_stats['first_entry_time']:.2f} ps")
                print(f"Total entries: {guest_stats['n_entries']}")
                print(f"Total exits: {guest_stats['n_exits']}")
                print(f"Total time inside: {guest_stats['total_time_inside']:.2f} ps")
                print(f"Total time outside: {guest_stats['total_time_outside']:.2f} ps")
                
                if guest_stats['durations_inside']:
                    print(f"Average stay duration: {np.mean(guest_stats['durations_inside']):.2f} ps")
                    print(f"Longest stay duration: {np.max(guest_stats['durations_inside']):.2f} ps")
                    print(f"Shortest stay duration: {np.min(guest_stats['durations_inside']):.2f} ps")
                
                # Show longest duration guest indices
                volume_observer.print_longest_duration_guest_indices()
            else:
                print("Guest never entered the host.")
            
            # Save guest entering statistics
            if guest_stats:
                guest_stats_df = pd.DataFrame({
                    'metric': ['first_entry_frame', 'first_entry_time', 'n_entries', 'n_exits',
                              'total_time_inside', 'total_time_outside',
                              'avg_stay_duration', 'max_stay_duration', 'min_stay_duration'],
                    'value': [
                        guest_stats.get('first_entry_frame'),
                        guest_stats.get('first_entry_time'),
                        guest_stats.get('n_entries', 0),
                        guest_stats.get('n_exits', 0),
                        guest_stats.get('total_time_inside', 0.0),
                        guest_stats.get('total_time_outside', 0.0),
                        np.mean(guest_stats['durations_inside']) if guest_stats.get('durations_inside') else None,
                        np.max(guest_stats['durations_inside']) if guest_stats.get('durations_inside') else None,
                        np.min(guest_stats['durations_inside']) if guest_stats.get('durations_inside') else None,
                    ]
                })
                guest_stats_df.to_csv(f"{args.out_prefix}_guest_entering_stats.csv", index=False)
                print(f"\nGuest entering statistics saved to {args.out_prefix}_guest_entering_stats.csv")
                
                # Save detailed entry/exit events
                if guest_stats.get('entry_frames'):
                    events_df = pd.DataFrame({
                        'event_type': ['entry'] * len(guest_stats['entry_frames']) + ['exit'] * len(guest_stats['exit_frames']),
                        'frame': guest_stats['entry_frames'] + guest_stats['exit_frames'],
                        'time': guest_stats['entry_times'] + guest_stats['exit_times'],
                        'guest_indices': guest_stats['entry_guest_indices'] + guest_stats['exit_guest_indices']
                    })
                    events_df = events_df.sort_values('frame')
                    events_df.to_csv(f"{args.out_prefix}_guest_entering_events.csv", index=False)
                    print(f"Guest entering events saved to {args.out_prefix}_guest_entering_events.csv")
    log_memory("After extracting results from observers")
    # 3) Correlation analysis between endpoint distances and volume
    correlation_df = None
    
    if volume is not None and endpoint_dists_array is not None and endpoint_analyzer is not None:
        print(f"\n{'='*60}")
        print("Computing correlations between endpoint distances and cube volume...")
        print(f"{'='*60}")
        
        if len(volume) == 0:
            warnings.warn(
                f"Volume array is empty (length 0). Cannot compute correlations. "
                f"This may indicate that the volume observer did not process any frames. "
                f"Check that the observer was properly subscribed and that frame processing completed without errors."
            )
        elif len(volume) != len(u.trajectory):
            warnings.warn(
                f"Volume array length ({len(volume)}) doesn't match trajectory length ({len(u.trajectory)}). "
                f"Skipping correlation analysis."
            )
            
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
                    plotter.plot_residue_endpoints(u, endpoint_observer, args.endpoint_residues, args.out_prefix)
                   #plot guest entry/exit timeline
                    print("\nPlotting guest entry/exit timeline...")
                    plotter.plot_guest_entry_exit_timeline(guest_stats, args.out_prefix)
            except Exception as e:
                warnings.warn(f"Failed to compute endpoint-volume correlations: {e}")
                import traceback
                traceback.print_exc()
    elif volume is None and endpoint_dists_array is not None:
        print("\nWarning: Cube volume not computed. Provide --cube_faces to enable correlation analysis.")
    elif endpoint_dists_array is None and volume is not None:
        print("\nWarning: Endpoint distances not computed. Cannot perform correlation analysis.")
    log_memory("After computing correlations between endpoint distances and volume")
    
    # 4) Simulation scoring and validation
    simulation_score = None
    score_details = None
    if volume is not None or guest_stats is not None or correlation_df is not None:
        print(f"\n{'='*60}")
        print("Scoring simulation quality...")
        print(f"{'='*60}")
        
        try:
            frame_selection = FrameSelection()
            simulation_score, score_details = frame_selection.score_simulation(
                guest_stats=guest_stats,
                volume=volume,
                correlation_df=correlation_df,
                endpoint_dists_array=endpoint_dists_array,
                endpoint_metrics_df=endpoint_metrics_df,
                cube_metrics_df=cube_metrics_df,
                min_guest_entry=args.guest_sel is not None,  # Require guest entry if guest_sel provided
                min_volume_change_pct=10.0,
                min_correlation=0.5,
            )
            
            # Print summary
            print(frame_selection.get_simulation_score_summary())
            
            # Save score as CSV
            # Extract trajectory ID from out_prefix (last component of path)
            trajectory_id = os.path.basename(args.out_prefix) if args.out_prefix else None
            if trajectory_id:
                # Remove any common suffixes to get clean ID
                trajectory_id = trajectory_id.replace('_test', '').replace('_analysis', '')
            
            frame_selection.save_simulation_score_csv(
                output_path=f"{args.out_prefix}_simulation_score.csv",
                trajectory_id=trajectory_id,
                guest_stats=guest_stats,
                volume=volume,
                correlation_df=correlation_df,
            )
            print(f"\nSimulation score saved to {args.out_prefix}_simulation_score.csv")
            
        except Exception as e:
            warnings.warn(f"Failed to score simulation: {e}")
            import traceback
            traceback.print_exc()
    
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
    
    if guest_stats is not None:
        print(f"\nGuest Entering Analysis:")
        print(f"  ✓ Guest entering statistics: {args.out_prefix}_guest_entering_stats.csv")
        if guest_stats.get('entry_frames'):
            print(f"  ✓ Guest entering events: {args.out_prefix}_guest_entering_events.csv")
    
    if simulation_score is not None:
        print(f"\nSimulation Quality Scoring:")
        print(f"  ✓ Overall score: {simulation_score:.3f} / 1.000")
        print(f"  ✓ Valid: {'Yes' if score_details and score_details.get('is_valid') else 'No'}")
        print(f"  ✓ Score details: {args.out_prefix}_simulation_score.csv")
        if score_details:
            print(f"    - Guest Entry Score: {score_details.get('guest_entry_score', 0):.3f}")
            print(f"    - Volume Dynamics Score: {score_details.get('volume_dynamics_score', 0):.3f}")
            print(f"    - Correlation Score: {score_details.get('correlation_score', 0):.3f}")
            print(f"    - Structural Dynamics Score: {score_details.get('structural_stability_score', 0):.3f}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()

# Example usage:
# python run_volume_endpoint_guest_analysis.py --top C:\Users\zonezone\Desktop\YCU_research\BMMpM_ca.prmtop --traj C:\Users\zonezone\Desktop\YCU_research\BMMpM_mdcrd_test.pdb --out_prefix output/test --align_sel "resid 1-6" --guest_sel "name I" --guest_tracking_method distance --plot_top_correlations 2
