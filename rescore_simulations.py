#!/usr/bin/env python3
"""
Script to rescore MD simulations from existing CSV files using the updated scoring function.

This script:
1. Reads existing simulation score CSV files
2. Attempts to load related data files (correlation, endpoint metrics, guest stats)
3. Reconstructs the necessary data structures for rescoring
4. Applies the updated scoring function
5. Saves the updated scores

Usage:
    python rescore_simulations.py --input_dir output/ --output output/rescored_scores.csv
    python rescore_simulations.py --csv_files output/traj1_simulation_score.csv output/traj2_simulation_score.csv
    python rescore_simulations.py --input_dir output/ --overwrite
"""

import argparse
import os
import glob
from pathlib import Path
from typing import List, Optional, Dict, Any
import warnings

import numpy as np
import pandas as pd

from src.FrameSelection.FrameSelection import FrameSelection


def find_related_files(score_csv_path: str) -> Dict[str, Optional[str]]:
    """
    Find related data files for a given simulation score CSV.
    
    Args:
        score_csv_path: Path to simulation score CSV file
        
    Returns:
        Dictionary with paths to related files (correlation, endpoint_metrics, guest_stats, volume)
    """
    base_dir = os.path.dirname(score_csv_path)
    base_name = os.path.basename(score_csv_path)
    
    # Extract prefix (remove _simulation_score.csv)
    prefix = base_name.replace('_simulation_score.csv', '')
    if not prefix:
        # Try to extract from full path
        prefix = os.path.splitext(os.path.basename(score_csv_path))[0]
        prefix = prefix.replace('_simulation_score', '')
    
    related_files = {
        'correlation': None,
        'endpoint_metrics': None,
        'guest_stats': None,
        'volume': None,
    }
    
    # Search for related files
    if base_dir:
        search_dir = base_dir
    else:
        search_dir = '.'
    
    # Look for correlation file
    correlation_patterns = [
        f"{prefix}_endpoint_volume_correlation.csv",
        f"*{prefix}*endpoint_volume_correlation.csv",
        f"{prefix}*correlation.csv",
    ]
    for pattern in correlation_patterns:
        matches = glob.glob(os.path.join(search_dir, pattern))
        if matches:
            related_files['correlation'] = matches[0]
            break
    
    # Look for endpoint metrics file
    endpoint_patterns = [
        f"{prefix}_endpoint_metrics.csv",
        f"*{prefix}*endpoint_metrics.csv",
    ]
    for pattern in endpoint_patterns:
        matches = glob.glob(os.path.join(search_dir, pattern))
        if matches:
            related_files['endpoint_metrics'] = matches[0]
            break
    
    # Look for guest stats file
    guest_patterns = [
        f"{prefix}_guest_entering_stats.csv",
        f"*{prefix}*guest_entering_stats.csv",
        f"{prefix}*guest_stats.csv",
    ]
    for pattern in guest_patterns:
        matches = glob.glob(os.path.join(search_dir, pattern))
        if matches:
            related_files['guest_stats'] = matches[0]
            break
    
    # Look for volume file (if saved separately)
    volume_patterns = [
        f"{prefix}_volume.csv",
        f"*{prefix}*volume.csv",
    ]
    for pattern in volume_patterns:
        matches = glob.glob(os.path.join(search_dir, pattern))
        if matches:
            related_files['volume'] = matches[0]
            break
    
    return related_files


def reconstruct_guest_stats(score_row: pd.Series) -> Optional[Dict[str, Any]]:
    """
    Reconstruct guest_stats dictionary from CSV row.
    
    Args:
        score_row: Row from simulation score CSV
        
    Returns:
        Dictionary with guest statistics or None if not available
    """
    guest_stats = {}
    
    # Extract guest entry/exit data
    if 'n_entries' in score_row:
        guest_stats['n_entries'] = int(score_row['n_entries']) if not pd.isna(score_row['n_entries']) else 0
    else:
        return None
    
    if 'n_exits' in score_row:
        guest_stats['n_exits'] = int(score_row['n_exits']) if not pd.isna(score_row['n_exits']) else 0
    else:
        guest_stats['n_exits'] = 0
    
    if 'total_time_inside_ps' in score_row:
        guest_stats['total_time_inside'] = float(score_row['total_time_inside_ps']) if not pd.isna(score_row['total_time_inside_ps']) else 0.0
    else:
        guest_stats['total_time_inside'] = 0.0
    
    if 'total_time_outside_ps' in score_row:
        guest_stats['total_time_outside'] = float(score_row['total_time_outside_ps']) if not pd.isna(score_row['total_time_outside_ps']) else 0.0
    else:
        guest_stats['total_time_outside'] = 0.0
    
    if 'first_entry_frame' in score_row:
        guest_stats['first_entry_frame'] = int(score_row['first_entry_frame']) if not pd.isna(score_row['first_entry_frame']) else None
    else:
        guest_stats['first_entry_frame'] = None
    
    if 'first_entry_time_ps' in score_row:
        guest_stats['first_entry_time'] = float(score_row['first_entry_time_ps']) if not pd.isna(score_row['first_entry_time_ps']) else None
    else:
        guest_stats['first_entry_time'] = None
    
    return guest_stats


def reconstruct_volume_from_metrics(score_row: pd.Series, n_frames: int = 1000) -> Optional[np.ndarray]:
    """
    Reconstruct approximate volume array from volume metrics.
    
    This is a simplified reconstruction - for accurate rescoring, 
    the original volume array should be available.
    
    Args:
        score_row: Row from simulation score CSV
        n_frames: Number of frames (default: 1000, approximate)
        
    Returns:
        Approximate volume array or None
    """
    if 'volume_mean_A3' not in score_row or pd.isna(score_row['volume_mean_A3']):
        return None
    
    vol_mean = float(score_row['volume_mean_A3'])
    vol_std = float(score_row['volume_std_A3']) if 'volume_std_A3' in score_row and not pd.isna(score_row['volume_std_A3']) else 0.0
    vol_min = float(score_row['volume_min_A3']) if 'volume_min_A3' in score_row and not pd.isna(score_row['volume_min_A3']) else vol_mean
    vol_max = float(score_row['volume_max_A3']) if 'volume_max_A3' in score_row and not pd.isna(score_row['volume_max_A3']) else vol_mean
    
    # Create a simple approximation: sinusoidal variation around mean
    # This is not accurate but allows rescoring to proceed
    if vol_std > 0:
        t = np.linspace(0, 4 * np.pi, n_frames)
        variation = np.sin(t) * vol_std
        volume = vol_mean + variation
        # Clip to min/max range
        volume = np.clip(volume, vol_min, vol_max)
    else:
        # Constant volume
        volume = np.full(n_frames, vol_mean)
    
    return volume


def load_data_for_rescoring(score_csv_path: str, score_row: pd.Series) -> Dict[str, Any]:
    """
    Load all necessary data for rescoring from files and CSV row.
    
    Args:
        score_csv_path: Path to simulation score CSV
        score_row: Row from the CSV
        
    Returns:
        Dictionary with loaded data (guest_stats, volume, correlation_df, endpoint_metrics_df)
    """
    data = {
        'guest_stats': None,
        'volume': None,
        'correlation_df': None,
        'endpoint_metrics_df': None,
    }
    
    # Find related files
    related_files = find_related_files(score_csv_path)
    
    # Load guest stats
    if related_files['guest_stats'] and os.path.exists(related_files['guest_stats']):
        try:
            guest_df = pd.read_csv(related_files['guest_stats'])
            # Convert DataFrame to dict (assuming single row or aggregate stats)
            if len(guest_df) > 0:
                data['guest_stats'] = guest_df.iloc[0].to_dict()
        except Exception as e:
            warnings.warn(f"Failed to load guest stats from {related_files['guest_stats']}: {e}")
    
    # If guest stats file not found, reconstruct from CSV row
    if data['guest_stats'] is None:
        data['guest_stats'] = reconstruct_guest_stats(score_row)
    
    # Load correlation dataframe
    if related_files['correlation'] and os.path.exists(related_files['correlation']):
        try:
            data['correlation_df'] = pd.read_csv(related_files['correlation'])
        except Exception as e:
            warnings.warn(f"Failed to load correlation data from {related_files['correlation']}: {e}")
    
    # Load endpoint metrics dataframe
    if related_files['endpoint_metrics'] and os.path.exists(related_files['endpoint_metrics']):
        try:
            data['endpoint_metrics_df'] = pd.read_csv(related_files['endpoint_metrics'])
        except Exception as e:
            warnings.warn(f"Failed to load endpoint metrics from {related_files['endpoint_metrics']}: {e}")
    
    # Load volume array
    if related_files['volume'] and os.path.exists(related_files['volume']):
        try:
            vol_df = pd.read_csv(related_files['volume'])
            if 'volume' in vol_df.columns:
                data['volume'] = vol_df['volume'].values
            elif len(vol_df.columns) > 0:
                data['volume'] = vol_df.iloc[:, 0].values
        except Exception as e:
            warnings.warn(f"Failed to load volume from {related_files['volume']}: {e}")
    
    # If volume file not found, try to reconstruct from metrics (approximate)
    if data['volume'] is None:
        data['volume'] = reconstruct_volume_from_metrics(score_row)
        if data['volume'] is not None:
            warnings.warn(f"Volume array reconstructed approximately from metrics for {score_row.get('trajectory_id', 'unknown')}")
    
    return data


def rescore_simulation(score_csv_path: str, score_row: pd.Series, 
                      min_guest_entry: bool = True,
                      min_volume_change_pct: float = 10.0,
                      min_correlation: float = 0.5) -> Dict[str, Any]:
    """
    Rescore a single simulation from CSV data.
    
    Args:
        score_csv_path: Path to simulation score CSV
        score_row: Row from the CSV
        min_guest_entry: If True, simulation is invalid if guest never entered
        min_volume_change_pct: Minimum volume change percentage
        min_correlation: Minimum correlation threshold
        
    Returns:
        Dictionary with new score and details
    """
    # Load data
    data = load_data_for_rescoring(score_csv_path, score_row)
    
    # Create FrameSelection instance and rescore
    frame_selection = FrameSelection()
    
    trajectory_id = score_row.get('trajectory_id', 'unknown')
    
    try:
        new_score, score_details = frame_selection.score_simulation(
            guest_stats=data['guest_stats'],
            volume=data['volume'],
            correlation_df=data['correlation_df'],
            endpoint_metrics_df=data['endpoint_metrics_df'],
            min_guest_entry=min_guest_entry,
            min_volume_change_pct=min_volume_change_pct,
            min_correlation=min_correlation,
        )
        
        return {
            'trajectory_id': trajectory_id,
            'new_overall_score': new_score,
            'old_overall_score': score_row.get('overall_score', np.nan),
            'score_change': new_score - score_row.get('overall_score', 0),
            'guest_entry_score': score_details['guest_entry_score'],
            'volume_dynamics_score': score_details['volume_dynamics_score'],
            'correlation_score': score_details['correlation_score'],
            'structural_dynamics_score': score_details['structural_stability_score'],
            'is_valid': score_details['is_valid'],
            'success': True,
        }
    except Exception as e:
        warnings.warn(f"Failed to rescore {trajectory_id}: {e}")
        return {
            'trajectory_id': trajectory_id,
            'new_overall_score': np.nan,
            'old_overall_score': score_row.get('overall_score', np.nan),
            'score_change': np.nan,
            'success': False,
            'error': str(e),
        }


def find_simulation_score_files(
    input_dir: str,
    pattern: str = "*_simulation_score.csv",
    recursive: bool = True
) -> List[str]:
    """
    Find all simulation score CSV files in a directory.
    
    Args:
        input_dir: Directory to search
        pattern: Glob pattern to match
        recursive: If True, search recursively
        
    Returns:
        List of file paths
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    if recursive:
        pattern = f"**/{pattern}"
    
    csv_files = list(input_path.glob(pattern))
    csv_files = [str(f) for f in csv_files if f.is_file()]
    
    return sorted(csv_files)


def main():
    parser = argparse.ArgumentParser(
        description="Rescore MD simulations from existing CSV files using updated scoring function.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rescore all simulations in output directory
  python rescore_simulations.py --input_dir output/
  
  # Rescore with custom output file
  python rescore_simulations.py --input_dir output/ --output output/rescored_scores.csv
  
  # Rescore specific CSV files
  python rescore_simulations.py --csv_files output/traj1_simulation_score.csv output/traj2_simulation_score.csv
  
  # Overwrite original files
  python rescore_simulations.py --input_dir output/ --overwrite
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_dir',
        type=str,
        help='Directory containing simulation score CSV files (searches for *_simulation_score.csv)'
    )
    input_group.add_argument(
        '--csv_files',
        nargs='+',
        help='List of specific CSV file paths to rescore'
    )
    
    # Search options
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_simulation_score.csv',
        help='Glob pattern to match CSV files (default: *_simulation_score.csv)'
    )
    parser.add_argument(
        '--no_recursive',
        action='store_true',
        help='Do not search recursively in subdirectories'
    )
    
    # Output options
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file path for rescored results (default: rescored_simulation_scores.csv)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite original CSV files with new scores'
    )
    
    # Scoring parameters
    parser.add_argument(
        '--min_volume_change_pct',
        type=float,
        default=10.0,
        help='Minimum volume change percentage (default: 10.0)'
    )
    parser.add_argument(
        '--min_correlation',
        type=float,
        default=0.5,
        help='Minimum correlation threshold (default: 0.5)'
    )
    parser.add_argument(
        '--require_guest_entry',
        action='store_true',
        help='Require guest entry for valid simulation'
    )
    
    args = parser.parse_args()
    
    # Find CSV files
    if args.input_dir:
        print(f"Searching for simulation score files in: {args.input_dir}")
        csv_files = find_simulation_score_files(
            args.input_dir,
            pattern=args.pattern,
            recursive=not args.no_recursive
        )
        print(f"Found {len(csv_files)} simulation score file(s)")
    else:
        csv_files = args.csv_files
        print(f"Using {len(csv_files)} specified CSV file(s)")
    
    if not csv_files:
        print("ERROR: No simulation score CSV files found!")
        return 1
    
    # Display found files
    print("\nFiles to rescore:")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f}")
    print()
    
    # Rescore each file
    all_results = []
    
    for csv_file in csv_files:
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(csv_file)}")
        print(f"{'='*80}")
        
        try:
            # Read CSV
            df = pd.read_csv(csv_file)
            
            if len(df) == 0:
                print(f"  Warning: Empty CSV file, skipping")
                continue
            
            # Process each row (usually just one row per file)
            for idx, row in df.iterrows():
                result = rescore_simulation(
                    csv_file,
                    row,
                    min_guest_entry=args.require_guest_entry,
                    min_volume_change_pct=args.min_volume_change_pct,
                    min_correlation=args.min_correlation,
                )
                
                if result['success']:
                    print(f"  Trajectory: {result['trajectory_id']}")
                    print(f"    Old score: {result['old_overall_score']:.4f}")
                    print(f"    New score: {result['new_overall_score']:.4f}")
                    print(f"    Change:    {result['score_change']:+.4f}")
                else:
                    print(f"  Trajectory: {result['trajectory_id']}")
                    print(f"    Error: {result.get('error', 'Unknown error')}")
                
                all_results.append(result)
        
        except Exception as e:
            print(f"  ERROR: Failed to process {csv_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        print("\nNo results to save.")
        return 1
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Update original CSV files if requested
    if args.overwrite:
        print(f"\n{'='*80}")
        print("Updating original CSV files...")
        print(f"{'='*80}")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                trajectory_id = df.iloc[0].get('trajectory_id', 'unknown')
                
                # Find matching result
                matching_result = results_df[results_df['trajectory_id'] == trajectory_id]
                if len(matching_result) > 0 and matching_result.iloc[0]['success']:
                    result = matching_result.iloc[0]
                    # Update scores in original DataFrame
                    df['overall_score'] = result['new_overall_score']
                    df['guest_entry_score'] = result['guest_entry_score']
                    df['volume_dynamics_score'] = result['volume_dynamics_score']
                    df['correlation_score'] = result['correlation_score']
                    if 'structural_dynamics_score' in df.columns:
                        df['structural_dynamics_score'] = result['structural_dynamics_score']
                    df['is_valid'] = result['is_valid']
                    
                    # Save updated file
                    df.to_csv(csv_file, index=False)
                    print(f"  Updated: {csv_file}")
                else:
                    print(f"  Skipped: {csv_file} (no valid result)")
            
            except Exception as e:
                print(f"  Error updating {csv_file}: {e}")
    
    # Save comparison results
    output_path = args.output if args.output else "rescored_simulation_scores.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Rescoring complete!")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}\n")
    
    # Display summary
    successful = results_df[results_df['success'] == True]
    if len(successful) > 0:
        print("Summary:")
        print(f"  Total processed: {len(results_df)}")
        print(f"  Successful: {len(successful)}")
        print(f"  Failed: {len(results_df) - len(successful)}")
        print(f"\nScore Statistics:")
        print(f"  Old score mean: {successful['old_overall_score'].mean():.4f}")
        print(f"  New score mean: {successful['new_overall_score'].mean():.4f}")
        print(f"  Mean change: {successful['score_change'].mean():+.4f}")
        print(f"  Score range (old): {successful['old_overall_score'].min():.4f} - {successful['old_overall_score'].max():.4f}")
        print(f"  Score range (new): {successful['new_overall_score'].min():.4f} - {successful['new_overall_score'].max():.4f}")
        print(f"  Score std (old): {successful['old_overall_score'].std():.4f}")
        print(f"  Score std (new): {successful['new_overall_score'].std():.4f}")
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

