#!/usr/bin/env python3
"""
Main entry point for Trajectory Deformation Analysis & Auto-Selection Pipeline

This script provides the CLI interface for the trajectory analysis workflow.
All utility functions are imported from trajectory_deformation_workflow module.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
except ImportError as e:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

# Import classes from the workflow module
from trajectory_deformation_workflow import (
    AlignedTrajectory,
    TrajectoryMetrics,
    ClusteringAnalysis,
    FrameSelection,
    FileIO,
    GSAnalyzer,
    EndpointAnalyzer,
    Plotter,
    HAS_ENDPOINTS_FINDER,
    EndpointsFinder  # type: ignore
    VolumeAnalyzer
)

from volume_analyser import VolumeAnalyzer

def main():
    p = argparse.ArgumentParser(description="Detect structural deformation and auto-select meaningful frames from trajectories.")
    p.add_argument('--top', required=True, help='Topology file (PDB/PSF/PRMTOP/etc.)')
    p.add_argument('--traj', required=True, nargs='+', help='One or more trajectory files (XTC/DCD/TRR/etc.)')
    p.add_argument('--align_sel', default='resid 1', help='Selection used for alignment/superposition.')
    p.add_argument('--rmsd_sel', default='all', help='Selection for RMSD (default: all atoms).')
    p.add_argument('--rmsf_sel', default='all', help='Selection for RMSF (default: all atoms).')
    p.add_argument('--pca_sel', default='all', help='Selection for PCA positional fluctuations.')
    p.add_argument('--n_pc', type=int, default=5, help='Number of PCs to compute (default: 5).')
    p.add_argument('--contact_selA', default=None, help='Selection A for minimal contact distance (optional).')
    p.add_argument('--contact_selB', default=None, help='Selection B for minimal contact distance (optional).')
    p.add_argument('--lag', type=int, default=5, help='Lag (frames) for strain proxy and derivatives.')
    p.add_argument('--strain_window', type=int, default=10, help='Window size for local affine strain proxy.')
    p.add_argument('--cpd_model', default='rbf', choices=['l2','rbf','linear'], help='Change-point detection cost model.')
    p.add_argument('--cpd_n', type=int, default=6, help='Number of change-points (approximate).')
    p.add_argument('--out_prefix', required=True, help='Prefix for outputs (CSV, frames, etc.).')
    p.add_argument('--save_pdb_sel', default='all', help='Selection to save for representative frames.')
    p.add_argument('--max_frames', type=int, default=30, help='Max frames to save for quick review.')
    p.add_argument('--load_results', default=None, help='Prefix for loading previously saved results (CSV files).')
    # Endpoint-based analysis arguments
    p.add_argument('--endpoint_residues', nargs='+', default=None, 
                   help='List of residue selection strings for endpoint analysis (e.g., "resid 1" "resid 2").')
    p.add_argument('--endpoint_angle_tol', type=float, default=15.0,
                   help='Angle tolerance in degrees for endpoint finding (default: 15.0).')
    p.add_argument('--endpoint_alpha', type=float, default=0.2,
                   help='Graph farness weight for endpoint finding (default: 0.2).')
    # Cube volume computation arguments (for correlation analysis)
    p.add_argument('--cube_faces', nargs='+', default=None,
                   help='List of face selection strings for cube volume computation (e.g., "resid 1-10" "resid 11-20").')
    p.add_argument('--plot_top_correlations', type=int, default=5,
                   help='Number of top endpoint pairs to plot for volume correlation (default: 5).')
    args = p.parse_args()

    # Load trajectory
    try:
        u = mda.Universe(args.top, *args.traj)
    except Exception as e:
        warnings.warn(f"Trajectory loading failed: {e}")
        u = mda.Universe(args.top, *args.traj, format="TRJ")
    if u is None:
        raise ValueError("Trajectory loading failed")
    
    # 1) Align trajectory to remove global motion using AlignedTrajectory class
    # Note: We align the full universe first, then filter atoms later for analysis
    print("Aligning trajectory...")
    try:
        aligned_traj = AlignedTrajectory(
            universe=u,
            align_sel=args.align_sel,
            ref_frame=0,
            align_first_and_last=True,
            in_memory=True
        )
        u = aligned_traj.get_aligned_universe()
        print("Trajectory alignment completed.")
    except Exception as e:
        warnings.warn(f"Alignment failed or selection invalid: {e}. Using original universe.")
        aligned_traj = None
    
    # 2) Global metrics (using aligned universe)
    metrics = TrajectoryMetrics()
    print("Computing RMSD...")
    rmsd_vals = metrics.compute_rmsd(u, args.rmsd_sel)

    print("Computing Rg...")
    rg_vals = metrics.radius_of_gyration(u, args.rmsd_sel)

    # Optional contacts
    contact_vals = None
    if args.contact_selA and args.contact_selB:
        print("Computing contact distances...")
        contact_vals = metrics.contact_distances(u, args.contact_selA, args.contact_selB)

    # 3) RMSF (per-atom) - using pre-aligned trajectory
    print("Computing RMSF...")
    try:
        rmsf_vals = metrics.compute_rmsf(u, args.rmsf_sel, aligned=True)
    except Exception as e:
        warnings.warn(f"RMSF failed: {e}")
        rmsf_vals = None

    # 4) PCA on positional fluctuations - using pre-aligned trajectory
    print("Computing PCA...")
    pcs, pca_model = metrics.pca_on_fluctuations(u, args.pca_sel, n_components=args.n_pc, aligned=True)

    # 5) Strain proxy
    print("Computing local affine strain proxy...")
    try:
        strain_vals = metrics.local_affine_strain_proxy(u, args.pca_sel, window=args.strain_window, lag=args.lag)
    except Exception as e:
        warnings.warn(f"Strain proxy failed: {e}")
        strain_vals = None

    # 5b) Endpoint-based metrics (if residue selections provided)
    endpoint_metrics_df = None
    endpoint_dists_array = None
    endpoint_analyzer = None
    if args.endpoint_residues and HAS_ENDPOINTS_FINDER:
        print(f"Computing endpoint-based metrics for {len(args.endpoint_residues)} residues...")
        try:
            endpoints_finder = EndpointsFinder(
                angle_tol_deg=args.endpoint_angle_tol,
                alpha=args.endpoint_alpha
            )
            endpoint_analyzer = EndpointAnalyzer()
            endpoint_metrics_df = endpoint_analyzer.compute_endpoint_metrics(u, args.endpoint_residues, endpoints_finder)
            endpoint_dists_array = endpoint_analyzer.compute_endpoint_distances(u, args.endpoint_residues, endpoints_finder)
            print(f"Endpoint metrics computed successfully. Found {len(endpoint_metrics_df.columns)} metrics.")
        except Exception as e:
            warnings.warn(f"Endpoint metrics computation failed: {e}")
            endpoint_metrics_df = None
            endpoint_dists_array = None
            endpoint_analyzer = None
    elif args.endpoint_residues and not HAS_ENDPOINTS_FINDER:
        warnings.warn("Endpoint residues specified but EndpointsFinder not available. Skipping endpoint analysis.")

    # 6) Change-point detection on fused signal
    clustering = ClusteringAnalysis()
    print("Running change-point detection...")
    # Build a single 1D signal by fusing informative terms
    fused = np.abs(clustering.zscore(pcs[:, 0])) + 0.5*np.abs(clustering.zscore(pcs[:, 1])) + 0.5*np.abs(clustering.zscore(rmsd_vals)) + 0.3*np.abs(clustering.zscore(rg_vals))
    if contact_vals is not None:
        fused += 0.3*np.abs(clustering.zscore(contact_vals))
    cpd_idx = clustering.change_points(fused, model=args.cpd_model, n_bkps=args.cpd_n)

    # 7) Cluster in PC space and get medoids
    print("Clustering frames in PC space...")
    labels, medoids = clustering.cluster_frames(pcs[:, :min(5, pcs.shape[1])])

    # 8) Score frames (key states + transitions)
    frame_selection = FrameSelection()
    print("Scoring frames...")
    scores, score_df = frame_selection.score_frames(rmsd_vals, rg_vals, pcs, cpd_idx, strain_vals)
    score_df['frame'] = np.arange(len(scores))
    score_df['label'] = labels
    score_df['is_medoid'] = 0
    score_df.loc[medoids, 'is_medoid'] = 1
    score_df['is_cpd'] = 0
    score_df.loc[cpd_idx, 'is_cpd'] = 1

    # 9) Select meaningful frames
    load_results = getattr(args, 'load_results', None)
    if load_results:
        scores_csv = load_results + "_scores.csv"
        metrics_csv = load_results + "_metrics.csv"
        endpoint_metrics_csv = load_results + "_endpoint_metrics.csv"
        print(f"Loading results from {load_results}")
    else:
        scores_csv = None
        metrics_csv = None
        endpoint_metrics_csv = None
        print("No results to load. Running analysis from scratch.")
    
    chosen_sorted, score_df = frame_selection.select_meaningful_frames(
        medoids=medoids,
        cpd_idx=cpd_idx,
        scores=scores,
        pcs=pcs,
        max_frames=args.max_frames,
        score_df=score_df,
        distance_threshold=0.75,
        endpoint_dists_array=endpoint_dists_array,
        endpoint_metrics_df=endpoint_metrics_df,
        scores_csv=scores_csv,
        metrics_csv=metrics_csv,
        endpoint_metrics_csv=endpoint_metrics_csv
    )

    # 10) Save reports
    os.makedirs(os.path.dirname(args.out_prefix) if os.path.dirname(args.out_prefix) else '.', exist_ok=True)
    metrics = {
        'frame': np.arange(len(rmsd_vals)),
        'rmsd': rmsd_vals,
        'rg': rg_vals,
        'pc1': pcs[:, 0],
        'pc2': pcs[:, 1] if pcs.shape[1] > 1 else np.zeros_like(rmsd_vals),
        'pc3': pcs[:, 2] if pcs.shape[1] > 2 else np.zeros_like(rmsd_vals),
    }
    if contact_vals is not None:
        metrics['contact_min'] = contact_vals
    if strain_vals is not None:
        metrics['strain'] = strain_vals

    metrics_df = pd.DataFrame(metrics)
    
    # Merge endpoint metrics if available
    if endpoint_metrics_df is not None and len(endpoint_metrics_df) > 0:
        # Merge on frame column
        metrics_df = metrics_df.merge(endpoint_metrics_df, on='frame', how='left')
        # Also save endpoint metrics separately
        endpoint_metrics_df.to_csv(f"{args.out_prefix}_endpoint_metrics.csv", index=False)
        print(f"Endpoint metrics saved to {args.out_prefix}_endpoint_metrics.csv")
    
    # Compute cube volume and correlation with endpoint distances
    volume = None
    correlation_df = None
    if endpoint_dists_array is not None and args.endpoint_residues:
        # Try to get volume from cube faces if provided
        if args.cube_faces and len(args.cube_faces) > 0:
            print("Computing cube volume from face selections...")
            try:
                gsa_analyzer = GSAnalyzer()
                cube_metrics_df = gsa_analyzer.gsa_nanocube_metrics(u, args.cube_faces, out_prefix=None)
                if 'volume' in cube_metrics_df.columns:
                    volume = cube_metrics_df['volume'].values
                    print(f"Cube volume computed from {len(args.cube_faces)} faces.")
                else:
                    # Fallback: compute from edge_mean if available
                    if 'edge_mean' in cube_metrics_df.columns:
                        volume = cube_metrics_df['edge_mean'].values ** 3
                        print("Cube volume computed from edge_mean.")
            except Exception as e:
                warnings.warn(f"Failed to compute volume from cube faces: {e}")
                volume = None
        
        # If volume is available, compute correlations and plot
        if volume is not None and len(volume) == len(u.trajectory) and endpoint_analyzer is not None:
            print("Computing correlations between endpoint distances and cube volume...")
            try:
                correlation_df = endpoint_analyzer.compute_endpoint_volume_correlation(
                    endpoint_dists_array, volume, args.endpoint_residues
                )
                if not correlation_df.empty:
                    correlation_df.to_csv(f"{args.out_prefix}_endpoint_volume_correlation.csv", index=False)
                    print(f"Correlation results saved to {args.out_prefix}_endpoint_volume_correlation.csv")
                    
                    # Analyze variation of endpoint pair distances
                    if isinstance(endpoint_dists_array, dict):
                        print("Analyzing variation of endpoint pair distances...")
                        variation_df = endpoint_analyzer.analyze_endpoint_pair_variation(
                            endpoint_dists_array, args.endpoint_residues
                        )
                        if not variation_df.empty:
                            variation_df.to_csv(f"{args.out_prefix}_endpoint_variation.csv", index=False)
                            print(f"Variation analysis saved to {args.out_prefix}_endpoint_variation.csv")
                            print(f"  Found {len(variation_df)} endpoint pairs with variation data")
                        
                        # Identify key endpoint pairs for expansion/shrinkage using variation analysis
                        key_pairs_df = endpoint_analyzer.identify_key_endpoint_pairs_for_expansion(
                            endpoint_dists_array, volume, args.endpoint_residues,
                            variation_threshold=0.1, correlation_threshold=0.7, top_n=10
                        )
                        if not key_pairs_df.empty:
                            key_pairs_df.to_csv(f"{args.out_prefix}_key_endpoint_pairs.csv", index=False)
                            print(f"\nKey endpoint pairs for cube expansion/shrinkage (saved to {args.out_prefix}_key_endpoint_pairs.csv):")
                            print(f"  Found {len(key_pairs_df)} key endpoint pairs (variation_score >= 0.1, |correlation| >= 0.7)")
                            for idx, (_, pair) in enumerate(key_pairs_df.head(5).iterrows(), 1):
                                print(f"\n  {idx}. Residue {int(pair['residue_i'])} ({pair['residue_i_sel']}) - "
                                      f"Residue {int(pair['residue_j'])} ({pair['residue_j_sel']})")
                                print(f"     Endpoint pair: ep{int(pair['ep_i_idx'])} - ep{int(pair['ep_j_idx'])}")
                                print(f"     Correlation: {pair['correlation']:.4f} (p={pair['p_value']:.4e})")
                                print(f"     Variation score: {pair['variation_score']:.4f} (CV: {pair['cv']:.4f}, Range: {pair['range_distance']:.2f} Å)")
                                print(f"     Mean distance: {pair['mean_distance']:.2f} Å (std: {pair['std_distance']:.2f} Å)")
                    
                    # Also report the single best pair for backward compatibility
                    if not correlation_df['correlation'].isna().all():
                        best_pair = correlation_df.iloc[0]
                        print(f"\nBest single endpoint pair (highest correlation):")
                        print(f"  Residue {int(best_pair['residue_i'])} ({best_pair['residue_i_sel']}) - "
                              f"Residue {int(best_pair['residue_j'])} ({best_pair['residue_j_sel']})")
                        if 'ep_i_idx' in best_pair and pd.notna(best_pair['ep_i_idx']):
                            print(f"  Endpoint indices: ep{int(best_pair['ep_i_idx'])} - ep{int(best_pair['ep_j_idx'])}")
                        print(f"  Correlation: {best_pair['correlation']:.4f}")
                        print(f"  P-value: {best_pair['p_value']:.4e}")
                        print(f"  Valid data points: {int(best_pair['n_valid_points'])}")
                    
                    # Plot endpoint distances
                    plotter = Plotter()
                    print("Plotting endpoint distances over frames...")
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
        elif volume is None:
            print("Warning: Cube volume not computed. Provide --cube_faces to enable correlation analysis.")
    
    metrics_df.to_csv(f"{args.out_prefix}_metrics.csv", index=False)
    score_df.to_csv(f"{args.out_prefix}_scores.csv", index=False)

    # 11) Save selected frames as PDBs for inspection
    file_io = FileIO()
    print(f"Saving up to {len(chosen_sorted)} representative frames to PDB...")
    file_io.save_frames_as_pdb(u, chosen_sorted, args.out_prefix, sel=args.save_pdb_sel)

    # 12) Write a quick summary
    with open(f"{args.out_prefix}_SUMMARY.txt", 'w') as fh:
        fh.write("Deformation analysis summary\n")
        fh.write(f"Frames: {len(rmsd_vals)}\n")
        fh.write(f"Change-points: {cpd_idx}\n")
        fh.write(f"Clusters: {len(set(labels)) - (1 if -1 in set(labels) else 0)}\n")
        fh.write(f"Medoids: {medoids.tolist()}\n")
        fh.write(f"Selected (top {len(chosen_sorted)}): {chosen_sorted}\n")
        fh.write("\nHow were frames selected?\n")
        fh.write("- Union of: cluster medoids (state representatives), change-points (putative transitions), and top-scoring frames across PC extremes / RMSD changes / Rg extrema / strain.\n")
        fh.write("- Non-redundancy enforced in PC space via distance threshold.\n")

    print("Done. Outputs:")
    print(f"  Metrics CSV: {args.out_prefix}_metrics.csv")
    print(f"  Scores CSV:  {args.out_prefix}_scores.csv")
    if endpoint_metrics_df is not None:
        print(f"  Endpoint metrics CSV: {args.out_prefix}_endpoint_metrics.csv")
    if correlation_df is not None and not correlation_df.empty:
        print(f"  Endpoint-volume correlation CSV: {args.out_prefix}_endpoint_volume_correlation.csv")
        if isinstance(endpoint_dists_array, dict):
            print(f"  Endpoint variation CSV: {args.out_prefix}_endpoint_variation.csv")
            print(f"  Key endpoint pairs CSV: {args.out_prefix}_key_endpoint_pairs.csv")
        print(f"  Endpoint distance plot: {args.out_prefix}_endpoint_distances.png")
        print(f"  Endpoint-volume correlation plot: {args.out_prefix}_endpoint_volume_correlation.png")
    print(f"  Summary:     {args.out_prefix}_SUMMARY.txt")
    print(f"  Frames dir:  {args.out_prefix}_frames/")


if __name__ == '__main__':
    main()

