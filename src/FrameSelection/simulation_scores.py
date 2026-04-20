"""Batch discovery, comparison, and rescoring of simulation score CSV artifacts."""

from __future__ import annotations

import glob
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .FrameSelection import FrameSelection


def find_simulation_score_files(
    input_dir: str,
    pattern: str = "*_simulation_score.csv",
    recursive: bool = True,
) -> List[str]:
    """Return sorted paths to simulation score CSV files under ``input_dir``."""
    input_path = Path(input_dir)

    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    glob_pat = f"**/{pattern}" if recursive else pattern

    csv_files = list(input_path.glob(glob_pat))
    return sorted(str(f) for f in csv_files if f.is_file())


def load_and_compare_simulation_scores(
    csv_paths: List[str],
    output_path: Optional[str] = None,
    sort_by: str = "overall_score",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Load and compare multiple simulation score CSV files.

    Args:
        csv_paths: Paths to simulation score CSV files.
        output_path: Optional path to save the combined/ranked results.
        sort_by: Column to sort by (default ``overall_score``).
        ascending: If True, sort ascending; default is descending (best first).

    Returns:
        DataFrame with all scores, sorted; includes a ``rank`` column.
    """
    all_scores: List[pd.DataFrame] = []

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                all_scores.append(df)
            else:
                warnings.warn(f"Empty CSV file: {csv_path}")
        except Exception as e:
            warnings.warn(f"Failed to load {csv_path}: {e}")

    if not all_scores:
        raise ValueError("No valid simulation score CSV files could be loaded")

    combined_df = pd.concat(all_scores, ignore_index=True)

    if sort_by in combined_df.columns:
        combined_df = combined_df.sort_values(by=sort_by, ascending=ascending, na_position="last")
    else:
        warnings.warn(
            f"Column '{sort_by}' not found. Available columns: {list(combined_df.columns)}"
        )
        warnings.warn("Sorting by 'overall_score' instead")
        if "overall_score" in combined_df.columns:
            combined_df = combined_df.sort_values(
                by="overall_score", ascending=False, na_position="last"
            )

    combined_df.insert(0, "rank", range(1, len(combined_df) + 1))

    if output_path:
        combined_df.to_csv(output_path, index=False)

    return combined_df


def display_top_simulations(
    df: pd.DataFrame,
    top_n: int = 10,
    columns: Optional[List[str]] = None,
) -> None:
    """Print the top ``top_n`` rows (CLI-friendly table)."""
    if columns is None:
        columns = [
            "rank",
            "trajectory_id",
            "overall_score",
            "is_valid",
            "guest_entry_score",
            "volume_dynamics_score",
            "correlation_score",
            "structural_dynamics_score",
            "n_entries",
            "n_exits",
            "entry_exit_diff",
            "volume_change_pct",
            "max_correlation",
        ]

    available_columns = [col for col in columns if col in df.columns]

    if not available_columns:
        print("No matching columns found. Available columns:")
        print(df.columns.tolist())
        return

    top_df = df.head(top_n)[available_columns]

    print(f"\n{'='*80}")
    sort_name = df.index.name if df.index.name else "overall_score"
    print(f"Top {min(top_n, len(df))} Simulations (sorted by {sort_name})")
    print(f"{'='*80}\n")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 50)

    print(top_df.to_string(index=False))
    print(f"\n{'='*80}\n")


def display_summary_statistics(df: pd.DataFrame) -> None:
    """Print summary statistics for a combined simulation score table."""
    print(f"\n{'='*80}")
    print("Summary Statistics")
    print(f"{'='*80}\n")

    print(f"Total simulations: {len(df)}")
    if "is_valid" in df.columns:
        print(f"Valid simulations: {df['is_valid'].sum()}")
        print(f"Invalid simulations: {(~df['is_valid']).sum()}")
    else:
        print("Valid simulations: N/A")
        print("Invalid simulations: N/A")

    if "overall_score" in df.columns:
        print("\nOverall Score Statistics:")
        print(f"  Mean: {df['overall_score'].mean():.4f}")
        print(f"  Median: {df['overall_score'].median():.4f}")
        print(f"  Std: {df['overall_score'].std():.4f}")
        print(f"  Min: {df['overall_score'].min():.4f}")
        print(f"  Max: {df['overall_score'].max():.4f}")

    if "n_entries" in df.columns:
        print("\nGuest Entry Statistics:")
        print(f"  Simulations with guest entries: {(df['n_entries'] > 0).sum()}")
        print(f"  Mean entries per simulation: {df['n_entries'].mean():.2f}")
        print(f"  Mean exits per simulation: {df['n_exits'].mean():.2f}")
        if "entry_exit_diff" in df.columns:
            print(
                "  Simulations with more entries than exits: "
                f"{(df['entry_exit_diff'] > 0).sum()}"
            )
        else:
            print("  Simulations with more entries than exits: N/A")

    if "volume_change_pct" in df.columns:
        valid_vol = df["volume_change_pct"].dropna()
        if len(valid_vol) > 0:
            print("\nVolume Dynamics Statistics:")
            print(f"  Mean volume change: {valid_vol.mean():.2f}%")
            print(f"  Simulations with >10% volume change: {(valid_vol > 10.0).sum()}")

    if "max_correlation" in df.columns:
        valid_corr = df["max_correlation"].dropna()
        if len(valid_corr) > 0:
            print("\nCorrelation Statistics:")
            print(f"  Mean max correlation: {valid_corr.mean():.4f}")
            print(f"  Simulations with correlation >0.5: {(valid_corr > 0.5).sum()}")

    print(f"\n{'='*80}\n")


def find_related_files(score_csv_path: str) -> Dict[str, Optional[str]]:
    """Find correlation, endpoint, guest, and volume CSV paths for a score file."""
    base_dir = os.path.dirname(score_csv_path)
    base_name = os.path.basename(score_csv_path)

    prefix = base_name.replace("_simulation_score.csv", "")
    if not prefix:
        prefix = os.path.splitext(os.path.basename(score_csv_path))[0]
        prefix = prefix.replace("_simulation_score", "")

    related_files: Dict[str, Optional[str]] = {
        "correlation": None,
        "endpoint_metrics": None,
        "guest_stats": None,
        "volume": None,
    }

    search_dir = base_dir if base_dir else "."

    correlation_patterns = [
        f"{prefix}_endpoint_volume_correlation.csv",
        f"*{prefix}*endpoint_volume_correlation.csv",
        f"{prefix}*correlation.csv",
    ]
    for pattern in correlation_patterns:
        matches = glob.glob(os.path.join(search_dir, pattern))
        if matches:
            related_files["correlation"] = matches[0]
            break

    endpoint_patterns = [
        f"{prefix}_endpoint_metrics.csv",
        f"*{prefix}*endpoint_metrics.csv",
    ]
    for pattern in endpoint_patterns:
        matches = glob.glob(os.path.join(search_dir, pattern))
        if matches:
            related_files["endpoint_metrics"] = matches[0]
            break

    guest_patterns = [
        f"{prefix}_guest_entering_stats.csv",
        f"*{prefix}*guest_entering_stats.csv",
        f"{prefix}*guest_stats.csv",
    ]
    for pattern in guest_patterns:
        matches = glob.glob(os.path.join(search_dir, pattern))
        if matches:
            related_files["guest_stats"] = matches[0]
            break

    volume_patterns = [
        f"{prefix}_volume.csv",
        f"*{prefix}*volume.csv",
    ]
    for pattern in volume_patterns:
        matches = glob.glob(os.path.join(search_dir, pattern))
        if matches:
            related_files["volume"] = matches[0]
            break

    return related_files


def reconstruct_guest_stats(score_row: pd.Series) -> Optional[Dict[str, Any]]:
    """Reconstruct guest_stats dict from a simulation score CSV row."""
    guest_stats: Dict[str, Any] = {}

    if "n_entries" not in score_row:
        return None

    guest_stats["n_entries"] = (
        int(score_row["n_entries"]) if not pd.isna(score_row["n_entries"]) else 0
    )

    if "n_exits" in score_row:
        guest_stats["n_exits"] = (
            int(score_row["n_exits"]) if not pd.isna(score_row["n_exits"]) else 0
        )
    else:
        guest_stats["n_exits"] = 0

    if "total_time_inside_ps" in score_row:
        guest_stats["total_time_inside"] = (
            float(score_row["total_time_inside_ps"])
            if not pd.isna(score_row["total_time_inside_ps"])
            else 0.0
        )
    else:
        guest_stats["total_time_inside"] = 0.0

    if "total_time_outside_ps" in score_row:
        guest_stats["total_time_outside"] = (
            float(score_row["total_time_outside_ps"])
            if not pd.isna(score_row["total_time_outside_ps"])
            else 0.0
        )
    else:
        guest_stats["total_time_outside"] = 0.0

    if "first_entry_frame" in score_row:
        guest_stats["first_entry_frame"] = (
            int(score_row["first_entry_frame"])
            if not pd.isna(score_row["first_entry_frame"])
            else None
        )
    else:
        guest_stats["first_entry_frame"] = None

    if "first_entry_time_ps" in score_row:
        guest_stats["first_entry_time"] = (
            float(score_row["first_entry_time_ps"])
            if not pd.isna(score_row["first_entry_time_ps"])
            else None
        )
    else:
        guest_stats["first_entry_time"] = None

    return guest_stats


def reconstruct_volume_from_metrics(
    score_row: pd.Series, n_frames: int = 1000
) -> Optional[np.ndarray]:
    """Approximate a volume time series from aggregate metrics (for rescoring only)."""
    if "volume_mean_A3" not in score_row or pd.isna(score_row["volume_mean_A3"]):
        return None

    vol_mean = float(score_row["volume_mean_A3"])
    vol_std = (
        float(score_row["volume_std_A3"])
        if "volume_std_A3" in score_row and not pd.isna(score_row["volume_std_A3"])
        else 0.0
    )
    vol_min = (
        float(score_row["volume_min_A3"])
        if "volume_min_A3" in score_row and not pd.isna(score_row["volume_min_A3"])
        else vol_mean
    )
    vol_max = (
        float(score_row["volume_max_A3"])
        if "volume_max_A3" in score_row and not pd.isna(score_row["volume_max_A3"])
        else vol_mean
    )

    if vol_std > 0:
        t = np.linspace(0, 4 * np.pi, n_frames)
        variation = np.sin(t) * vol_std
        volume = np.clip(vol_mean + variation, vol_min, vol_max)
    else:
        volume = np.full(n_frames, vol_mean)

    return volume


def load_data_for_rescoring(score_csv_path: str, score_row: pd.Series) -> Dict[str, Any]:
    """Load guest stats, volume, correlation, and endpoint metrics for rescoring."""
    data: Dict[str, Any] = {
        "guest_stats": None,
        "volume": None,
        "correlation_df": None,
        "endpoint_metrics_df": None,
    }

    related_files = find_related_files(score_csv_path)

    if related_files["guest_stats"] and os.path.exists(related_files["guest_stats"]):
        try:
            guest_df = pd.read_csv(related_files["guest_stats"])
            if len(guest_df) > 0:
                if "metric" in guest_df.columns and "value" in guest_df.columns:
                    data["guest_stats"] = {}
                    for _, row in guest_df.iterrows():
                        metric = row["metric"]
                        value = row["value"]
                        if pd.isna(value):
                            data["guest_stats"][metric] = None
                        elif metric in ["first_entry_frame"]:
                            data["guest_stats"][metric] = int(value) if not pd.isna(value) else None
                        elif metric in [
                            "first_entry_time",
                            "total_time_inside",
                            "total_time_outside",
                            "avg_stay_duration",
                            "max_stay_duration",
                            "min_stay_duration",
                        ]:
                            data["guest_stats"][metric] = (
                                float(value) if not pd.isna(value) else None
                            )
                        elif metric in ["n_entries", "n_exits"]:
                            data["guest_stats"][metric] = int(value) if not pd.isna(value) else 0
                        else:
                            data["guest_stats"][metric] = value
                else:
                    data["guest_stats"] = guest_df.iloc[0].to_dict()
        except Exception as e:
            warnings.warn(f"Failed to load guest stats from {related_files['guest_stats']}: {e}")

    if data["guest_stats"] is None:
        data["guest_stats"] = reconstruct_guest_stats(score_row)

    if data["guest_stats"] is not None:
        n_entries = data["guest_stats"].get("n_entries", "missing")
        if isinstance(n_entries, (int, float)) and n_entries == 0:
            warnings.warn(
                "Warning: n_entries is 0 for "
                f"{score_row.get('trajectory_id', 'unknown')} - guest_entry_score will be 0"
            )

    if related_files["correlation"] and os.path.exists(related_files["correlation"]):
        try:
            data["correlation_df"] = pd.read_csv(related_files["correlation"])
        except Exception as e:
            warnings.warn(
                f"Failed to load correlation data from {related_files['correlation']}: {e}"
            )

    if related_files["endpoint_metrics"] and os.path.exists(related_files["endpoint_metrics"]):
        try:
            data["endpoint_metrics_df"] = pd.read_csv(related_files["endpoint_metrics"])
        except Exception as e:
            warnings.warn(
                f"Failed to load endpoint metrics from {related_files['endpoint_metrics']}: {e}"
            )

    if related_files["volume"] and os.path.exists(related_files["volume"]):
        try:
            vol_df = pd.read_csv(related_files["volume"])
            if "volume" in vol_df.columns:
                data["volume"] = vol_df["volume"].values
            elif len(vol_df.columns) > 0:
                data["volume"] = vol_df.iloc[:, 0].values
        except Exception as e:
            warnings.warn(f"Failed to load volume from {related_files['volume']}: {e}")

    if data["volume"] is None:
        data["volume"] = reconstruct_volume_from_metrics(score_row)
        if data["volume"] is not None:
            warnings.warn(
                "Volume array reconstructed approximately from metrics for "
                f"{score_row.get('trajectory_id', 'unknown')}"
            )

    return data


def rescore_simulation(
    score_csv_path: str,
    score_row: pd.Series,
    min_guest_entry: bool = True,
    min_volume_change_pct: float = 10.0,
    min_correlation: float = 0.5,
) -> Dict[str, Any]:
    """Recompute simulation score from on-disk artifacts and CSV row fields."""
    data = load_data_for_rescoring(score_csv_path, score_row)
    frame_selection = FrameSelection()

    trajectory_id = score_row.get("trajectory_id", "unknown")

    try:
        new_score, score_details = frame_selection.score_simulation(
            guest_stats=data["guest_stats"],
            volume=data["volume"],
            correlation_df=data["correlation_df"],
            endpoint_metrics_df=data["endpoint_metrics_df"],
            min_guest_entry=min_guest_entry,
            min_volume_change_pct=min_volume_change_pct,
            min_correlation=min_correlation,
        )

        return {
            "trajectory_id": trajectory_id,
            "new_overall_score": new_score,
            "old_overall_score": score_row.get("overall_score", np.nan),
            "score_change": new_score - score_row.get("overall_score", 0),
            "guest_entry_score": score_details["guest_entry_score"],
            "volume_dynamics_score": score_details["volume_dynamics_score"],
            "correlation_score": score_details["correlation_score"],
            "structural_dynamics_score": score_details["structural_stability_score"],
            "is_valid": score_details["is_valid"],
            "success": True,
        }
    except Exception as e:
        warnings.warn(f"Failed to rescore {trajectory_id}: {e}")
        return {
            "trajectory_id": trajectory_id,
            "new_overall_score": np.nan,
            "old_overall_score": score_row.get("overall_score", np.nan),
            "score_change": np.nan,
            "success": False,
            "error": str(e),
        }


def rescore_simulation_csv_files(
    csv_files: List[str],
    *,
    min_guest_entry: bool = False,
    min_volume_change_pct: float = 10.0,
    min_correlation: float = 0.5,
) -> pd.DataFrame:
    """Rescore every row in each simulation score CSV; return a results DataFrame."""
    all_results: List[Dict[str, Any]] = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        if len(df) == 0:
            continue

        for _, row in df.iterrows():
            result = rescore_simulation(
                csv_file,
                row,
                min_guest_entry=min_guest_entry,
                min_volume_change_pct=min_volume_change_pct,
                min_correlation=min_correlation,
            )
            all_results.append(result)

    return pd.DataFrame(all_results)


def overwrite_simulation_score_sources(
    results_df: pd.DataFrame,
    csv_files: List[str],
) -> None:
    """Replace scores in original ``*_simulation_score.csv`` files where rescoring succeeded."""
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            trajectory_id = df.iloc[0].get("trajectory_id", "unknown")

            matching = results_df[results_df["trajectory_id"] == trajectory_id]
            if len(matching) > 0 and matching.iloc[0]["success"]:
                result = matching.iloc[0]
                df["overall_score"] = result["new_overall_score"]
                df["guest_entry_score"] = result["guest_entry_score"]
                df["volume_dynamics_score"] = result["volume_dynamics_score"]
                df["correlation_score"] = result["correlation_score"]
                if "structural_dynamics_score" in df.columns:
                    df["structural_dynamics_score"] = result["structural_dynamics_score"]
                df["is_valid"] = result["is_valid"]

                df.to_csv(csv_file, index=False)
                print(f"  Updated: {csv_file}")
            else:
                print(f"  Skipped: {csv_file} (no valid result)")

        except Exception as e:
            print(f"  Error updating {csv_file}: {e}")
