#!/usr/bin/env python3
"""
Script to load and compare simulation scores from multiple MD trajectories.

This script:
1. Finds all simulation score CSV files (matching pattern *_simulation_score.csv)
2. Loads and combines them into a single DataFrame
3. Ranks simulations by overall score (or other criteria)
4. Displays top N simulations
5. Saves the ranked comparison to CSV

Usage:
    python compare_simulation_scores.py --input_dir output/ --output output/all_simulations_ranked.csv
    python compare_simulation_scores.py --input_dir output/ --top_n 10 --sort_by volume_dynamics_score
    python compare_simulation_scores.py --csv_files output/traj1_simulation_score.csv output/traj2_simulation_score.csv
"""

import argparse
import os
import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.FrameSelection.FrameSelection import load_and_compare_simulation_scores


def find_simulation_score_files(
    input_dir: str,
    pattern: str = "*_simulation_score.csv",
    recursive: bool = True
) -> List[str]:
    """
    Find all simulation score CSV files in a directory.
    
    Args:
        input_dir: Directory to search
        pattern: Glob pattern to match (default: "*_simulation_score.csv")
        recursive: If True, search recursively in subdirectories
    
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


def display_top_simulations(
    df: pd.DataFrame,
    top_n: int = 10,
    columns: Optional[List[str]] = None
) -> None:
    """
    Display top N simulations in a formatted table.
    
    Args:
        df: DataFrame with simulation scores
        top_n: Number of top simulations to display
        columns: List of columns to display (if None, shows key columns)
    """
    if columns is None:
        columns = [
            'rank',
            'trajectory_id',
            'overall_score',
            'is_valid',
            'guest_entry_score',
            'volume_dynamics_score',
            'correlation_score',
            'structural_dynamics_score',
            'n_entries',
            'n_exits',
            'entry_exit_diff',
            'volume_change_pct',
            'max_correlation',
        ]
    
    # Filter to only existing columns
    available_columns = [col for col in columns if col in df.columns]
    
    if not available_columns:
        print("No matching columns found. Available columns:")
        print(df.columns.tolist())
        return
    
    top_df = df.head(top_n)[available_columns]
    
    print(f"\n{'='*80}")
    print(f"Top {min(top_n, len(df))} Simulations (sorted by {df.index.name if df.index.name else 'overall_score'})")
    print(f"{'='*80}\n")
    
    # Set pandas display options for better formatting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    print(top_df.to_string(index=False))
    print(f"\n{'='*80}\n")


def display_summary_statistics(df: pd.DataFrame) -> None:
    """Display summary statistics for all simulations."""
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}\n")
    
    print(f"Total simulations: {len(df)}")
    print(f"Valid simulations: {df['is_valid'].sum() if 'is_valid' in df.columns else 'N/A'}")
    print(f"Invalid simulations: {(~df['is_valid']).sum() if 'is_valid' in df.columns else 'N/A'}")
    
    if 'overall_score' in df.columns:
        print(f"\nOverall Score Statistics:")
        print(f"  Mean: {df['overall_score'].mean():.4f}")
        print(f"  Median: {df['overall_score'].median():.4f}")
        print(f"  Std: {df['overall_score'].std():.4f}")
        print(f"  Min: {df['overall_score'].min():.4f}")
        print(f"  Max: {df['overall_score'].max():.4f}")
    
    if 'n_entries' in df.columns:
        print(f"\nGuest Entry Statistics:")
        print(f"  Simulations with guest entries: {(df['n_entries'] > 0).sum()}")
        print(f"  Mean entries per simulation: {df['n_entries'].mean():.2f}")
        print(f"  Mean exits per simulation: {df['n_exits'].mean():.2f}")
        print(f"  Simulations with more entries than exits: {(df['entry_exit_diff'] > 0).sum() if 'entry_exit_diff' in df.columns else 'N/A'}")
    
    if 'volume_change_pct' in df.columns:
        valid_vol = df['volume_change_pct'].dropna()
        if len(valid_vol) > 0:
            print(f"\nVolume Dynamics Statistics:")
            print(f"  Mean volume change: {valid_vol.mean():.2f}%")
            print(f"  Simulations with >10% volume change: {(valid_vol > 10.0).sum()}")
    
    if 'max_correlation' in df.columns:
        valid_corr = df['max_correlation'].dropna()
        if len(valid_corr) > 0:
            print(f"\nCorrelation Statistics:")
            print(f"  Mean max correlation: {valid_corr.mean():.4f}")
            print(f"  Simulations with correlation >0.5: {(valid_corr > 0.5).sum()}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare simulation scores from multiple MD trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all simulations in output directory
  python compare_simulation_scores.py --input_dir output/
  
  # Compare with custom output file and show top 20
  python compare_simulation_scores.py --input_dir output/ --output ranked.csv --top_n 20
  
  # Sort by volume dynamics instead of overall score
  python compare_simulation_scores.py --input_dir output/ --sort_by volume_dynamics_score
  
  # Compare specific CSV files
  python compare_simulation_scores.py --csv_files output/traj1_simulation_score.csv output/traj2_simulation_score.csv
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
        help='List of specific CSV file paths to compare'
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
        help='Output CSV file path for ranked results (default: all_simulations_ranked.csv)'
    )
    parser.add_argument(
        '--sort_by',
        type=str,
        default='overall_score',
        help='Column name to sort by (default: overall_score)'
    )
    parser.add_argument(
        '--ascending',
        action='store_true',
        help='Sort in ascending order (default: descending, highest scores first)'
    )
    
    # Display options
    parser.add_argument(
        '--top_n',
        type=int,
        default=10,
        help='Number of top simulations to display (default: 10)'
    )
    parser.add_argument(
        '--no_summary',
        action='store_true',
        help='Do not display summary statistics'
    )
    parser.add_argument(
        '--columns',
        nargs='+',
        help='Specific columns to display (default: key columns)'
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
        print("Make sure you have run run_volume_endpoint_guest_analysis.py first.")
        return 1
    
    # Display found files
    print("\nFiles to compare:")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f}")
    print()
    
    # Load and compare
    try:
        output_path = args.output if args.output else "all_simulations_ranked.csv"
        
        print(f"Loading and comparing simulations...")
        print(f"Sorting by: {args.sort_by}")
        print(f"Order: {'ascending' if args.ascending else 'descending'}")
        
        comparison_df = load_and_compare_simulation_scores(
            csv_paths=csv_files,
            output_path=output_path,
            sort_by=args.sort_by,
            ascending=args.ascending,
        )
        
        print(f"\n✓ Successfully loaded {len(comparison_df)} simulation(s)")
        print(f"✓ Ranked results saved to: {output_path}")
        
        # Display summary statistics
        if not args.no_summary:
            display_summary_statistics(comparison_df)
        
        # Display top simulations
        display_top_simulations(
            comparison_df,
            top_n=args.top_n,
            columns=args.columns
        )
        
        # Additional information
        print(f"\nFull ranked results saved to: {output_path}")
        print(f"Use this file to identify which simulations to investigate first.\n")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: Failed to compare simulation scores: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())

#python compare_simulation_scores.py --input_dir output/ --top_n 10