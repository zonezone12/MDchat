"""
Changepoint detection on four feature groups from pre-computed GSA feature CSVs.

Groups:
  gsa       — nanocube cage geometry and monomer structure
  iodine    — iodine (IOD) guest dynamics and contacts
  na_water  — cavity Na+/water solvent environment
  combined  — all numeric features

For each *_gsa_features.csv in --input-dir, runs multivariate Pelt (ruptures) on
each group separately, compares changepoint timing across groups, and writes:
  {traj_id}_breakpoints.csv       — tidy breakpoints table (group × breakpoint)
  {traj_id}_segment_stats.csv     — per-segment feature summaries
  changepoint_timing_comparison.csv — cross-group Jaccard / timing offset
  all_breakpoints.csv             — breakpoints from all trajectories merged
  all_segment_stats.csv           — segment stats from all trajectories merged

Example
-------
python scripts/changepoint_feature_groups.py \\
    --input-dir output/gsa_features \\
    --output-dir output/changepoints
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.ruptures_utils import detect_changepoints
from src.utils.run_log import RunContext, log_event, step


# ---------------------------------------------------------------------------
# Feature group column lists
# ---------------------------------------------------------------------------

_METADATA_COLS: frozenset[str] = frozenset({"traj_id", "frame", "time_ps"})

# A — GSA cage/assembly/monomer geometry (excludes guest and solvent counters)
GSA_COLS: list[str] = [
    "assembly_rg",
    "assembly_gyration_eig_1", "assembly_gyration_eig_2", "assembly_gyration_eig_3",
    "assembly_asphericity", "assembly_acylindricity", "assembly_relative_shape_anisotropy",
    "assembly_rmsd_to_ref", "assembly_frame_to_frame_rmsd",
    "radial_distance_mean", "radial_distance_std",
    "com_dist_mean", "com_dist_std",
    "com_neighbor_dist_mean", "com_opposite_dist_mean", "com_opposite_neighbor_ratio",
    "octahedrality_score",
    "amphiphile_axis_radial_cos_mean", "amphiphile_axis_radial_cos_std",
    "monomer_twist_angle_std",
    "total_inter_monomer_contacts", "inter_contact_mean", "inter_contact_std",
    "n_active_interfaces", "n_broken_interfaces", "weakest_interface_contact_count",
    "contact_graph_density", "largest_connected_component_size",
    "cavity_min_heavy_atom_distance_from_center",
    "cavity_inner_atom_distance_mean", "cavity_inner_atom_distance_std",
    "hydrophilic_radial_distance_mean", "hydrophobic_radial_distance_mean",
    "hydrophilic_minus_hydrophobic_radial_distance",
    "monomer_rg_mean", "monomer_rg_std", "monomer_rmsd_mean", "monomer_rmsd_std",
    "monomer_deformation_max",
    "tooth_contact_total", "tooth_contact_min", "tooth_contact_std",
    "endpoint_dist_mean", "endpoint_dist_std",
    "gear_phase_offset_mean", "gear_phase_offset_std",
    "neighbor_relative_rotation_angle_mean", "neighbor_relative_rotation_angle_std",
    "inter_monomer_hbond_total", "inter_monomer_hbond_min",
    "hydrophobic_contact_total", "polar_contact_total", "salt_bridge_total",
    "pore_opening_distance_1", "pore_opening_distance_2", "pore_opening_distance_3",
]

# B — Iodine (IOD) guest dynamics and cage interactions
IODINE_COLS: list[str] = [
    "n_guest_inside_cavity", "n_guest_near_inner_surface",
    "n_guest_near_outer_surface", "n_guest_bulk",
    "guest_radial_distance_mean", "guest_radial_distance_std",
    "guest_radial_distance_min", "guest_radial_distance_max",
    "guest_spatial_dipole_magnitude",
    "n_guest_gsa_contacts", "guest_gsa_min_distance",
    "n_guest_hydrophobic_contacts", "n_guest_hydrophilic_contacts",
    "guest_hydrophobic_preference_ratio",
    "guest_interface_count_total", "guest_interface_count_mean",
    "n_guest_bridging_2_monomers", "n_guest_bridging_3plus_monomers",
    "guest_bridge_count_total",
    "guest_per_monomer_contact_mean", "guest_per_monomer_contact_std",
    "guest_contacted_monomer_count", "guest_mediated_interface_count",
]

# C — Na+ and water solvent environment
NA_WATER_COLS: list[str] = [
    "cavity_water_count", "cavity_ion_count",
    "headgroup_water_contacts", "hydrophobic_water_contacts",
    "water_bridge_count_between_monomers", "ion_bridge_count_between_monomers",
    "water_count_near_guest", "guest_water_contact_count",
]

# Representative columns for segment summary output (subset per group)
_SUMMARY_COLS: dict[str, list[str]] = {
    "gsa": [
        "assembly_rg", "assembly_rmsd_to_ref", "octahedrality_score",
        "total_inter_monomer_contacts", "n_active_interfaces", "n_broken_interfaces",
        "hydrophilic_minus_hydrophobic_radial_distance", "endpoint_dist_mean",
        "inter_monomer_hbond_total", "salt_bridge_total",
        "pore_opening_distance_1", "monomer_deformation_max",
    ],
    "iodine": [
        "n_guest_inside_cavity", "n_guest_near_inner_surface",
        "n_guest_near_outer_surface", "n_guest_bulk",
        "guest_radial_distance_mean", "guest_gsa_min_distance",
        "guest_bridge_count_total", "guest_mediated_interface_count",
        "guest_hydrophobic_preference_ratio", "guest_contacted_monomer_count",
    ],
    "na_water": [
        "cavity_water_count", "cavity_ion_count",
        "headgroup_water_contacts", "hydrophobic_water_contacts",
        "water_bridge_count_between_monomers", "ion_bridge_count_between_monomers",
        "water_count_near_guest", "guest_water_contact_count",
    ],
    "combined": [
        "assembly_rg", "assembly_rmsd_to_ref", "octahedrality_score",
        "total_inter_monomer_contacts", "n_active_interfaces",
        "hydrophilic_minus_hydrophobic_radial_distance",
        "cavity_water_count", "cavity_ion_count",
        "n_guest_inside_cavity", "guest_radial_distance_mean",
        "inter_monomer_hbond_total", "salt_bridge_total",
    ],
}


# ---------------------------------------------------------------------------
# Signal preparation
# ---------------------------------------------------------------------------

def _build_signal_matrix(
    df: pd.DataFrame,
    cols: list[str],
    normalize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build (n_frames, n_features) float64 array ready for ruptures.

    Filters to available columns, fills NaN/inf with column means, drops
    zero-variance columns, and optionally z-score normalises.

    Returns (matrix, used_column_names). Returns empty array if nothing usable.
    """
    available = [c for c in cols if c in df.columns]
    if not available:
        return np.empty((len(df), 0), dtype=np.float64), []

    mat = df[available].to_numpy(dtype=np.float64)
    mat = np.where(np.isfinite(mat), mat, np.nan)
    col_means = np.nanmean(mat, axis=0)
    for j in range(mat.shape[1]):
        fill = col_means[j] if np.isfinite(col_means[j]) else 0.0
        mat[np.isnan(mat[:, j]), j] = fill

    var = mat.var(axis=0)
    keep = var > 0
    mat = mat[:, keep]
    used_cols = [c for c, k in zip(available, keep) if k]

    if mat.shape[1] == 0:
        return np.empty((len(df), 0), dtype=np.float64), []

    if normalize:
        mat = StandardScaler().fit_transform(mat)

    return mat, used_cols


# ---------------------------------------------------------------------------
# Segment statistics
# ---------------------------------------------------------------------------

def _segment_stats(
    df: pd.DataFrame,
    s_start: int,
    s_end: int,
    summary_cols: list[str],
) -> dict[str, float]:
    """Mean and std of summary_cols over df rows [s_start, s_end)."""
    slice_df = df.iloc[s_start:s_end]
    out: dict[str, float] = {}
    for col in summary_cols:
        if col not in df.columns:
            continue
        vals = slice_df[col].to_numpy(dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue
        out[f"{col}_mean"] = float(np.mean(finite))
        out[f"{col}_std"] = float(np.std(finite))
    return out


# ---------------------------------------------------------------------------
# Timing comparison between two breakpoint sets
# ---------------------------------------------------------------------------

def _compare_breakpoints(
    bkps_a: list[int],
    bkps_b: list[int],
    tolerance: int,
    time_arr: np.ndarray,
) -> dict[str, float]:
    """Pairwise timing similarity between two sets of signal-index breakpoints.

    Returns n_bkps_a, n_bkps_b, n_shared (within ±tolerance frames), Jaccard,
    and mean_timing_offset in both frames and ps.
    """
    n_a, n_b = len(bkps_a), len(bkps_b)

    shared = sum(
        1 for b in bkps_a if any(abs(b - c) <= tolerance for c in bkps_b)
    )
    union_size = len(set(bkps_a) | set(bkps_b))

    if n_a == 0 and n_b == 0:
        jaccard = 1.0
    elif union_size == 0:
        jaccard = 0.0
    else:
        jaccard = shared / union_size

    if n_a > 0 and n_b > 0:
        arr_b = np.array(bkps_b, dtype=np.float64)
        min_dists_frames = [float(np.abs(b - arr_b).min()) for b in bkps_a]
        mean_off_frames = float(np.mean(min_dists_frames))
        if len(time_arr) >= 2:
            frame_step_ps = float(time_arr[1] - time_arr[0])
        else:
            frame_step_ps = 1.0
        mean_off_ps = mean_off_frames * frame_step_ps
    else:
        mean_off_frames = float("nan")
        mean_off_ps = float("nan")

    return {
        "n_bkps_a": n_a,
        "n_bkps_b": n_b,
        "n_shared": shared,
        "jaccard": jaccard,
        "mean_timing_offset_frames": mean_off_frames,
        "mean_timing_offset_ps": mean_off_ps,
    }


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _auto_penalty(n: int, normalize: bool) -> float:
    """BIC-inspired penalty for normalized multivariate Pelt.

    After z-score normalization each feature has unit variance, so we use
    log(n) rather than log(n)*d*var to avoid over-penalising high-dimensional
    groups.  For un-normalised signals we fall back to None (let detect_changepoints
    apply its own heuristic).
    """
    return float(np.log(n)) if normalize else None  # type: ignore[return-value]


def process_csv(
    csv_path: Path,
    *,
    method: str,
    cost_model: str,
    penalty: Optional[float],
    n_bkps: Optional[int],
    min_size: int,
    jump: int,
    tolerance_frames: int,
    groups: list[str],
    normalize: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run changepoint detection on one features CSV.

    Returns:
        breakpoints_df    — tidy table of all breakpoints
        segment_stats_df  — per-segment summaries
        comparison_df     — cross-group pairwise timing metrics
    """
    df = pd.read_csv(csv_path)
    # Strip the _gsa_features suffix to get a clean trajectory identifier.
    stem = csv_path.stem
    traj_id = stem[:-len("_gsa_features")] if stem.endswith("_gsa_features") else stem
    n_signal = len(df)

    frame_arr = (
        df["frame"].to_numpy(dtype=np.int64)
        if "frame" in df.columns
        else np.arange(n_signal, dtype=np.int64)
    )
    time_arr = (
        df["time_ps"].to_numpy(dtype=np.float64)
        if "time_ps" in df.columns
        else np.arange(n_signal, dtype=np.float64)
    )

    all_numeric_cols = [
        c for c in df.columns
        if c not in _METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    col_map: dict[str, list[str]] = {
        "gsa": GSA_COLS,
        "iodine": IODINE_COLS,
        "na_water": NA_WATER_COLS,
        "combined": all_numeric_cols,
    }

    bkp_rows: list[dict] = []
    seg_rows: list[dict] = []
    group_bkps: dict[str, list[int]] = {}

    for grp in groups:
        with step(f"group={grp}"):
            cols = col_map[grp]
            signal, used_cols = _build_signal_matrix(df, cols, normalize=normalize)

            if signal.shape[1] == 0:
                log_event(
                    "warning",
                    f"{traj_id}: group '{grp}' — no usable columns, skipping",
                    component="changepoint_feature_groups",
                )
                group_bkps[grp] = []
                continue

            log_event(
                "info",
                f"{traj_id}: group '{grp}' — {signal.shape[1]} cols, {n_signal} frames",
                component="changepoint_feature_groups",
            )

            effective_penalty = (
                penalty
                if penalty is not None or n_bkps is not None
                else _auto_penalty(n_signal, normalize)
            )
            cp = detect_changepoints(
                signal,
                method=method,
                cost_model=cost_model,
                penalty=effective_penalty,
                n_bkps=n_bkps,
                min_size=min_size,
                jump=jump,
            )
            group_bkps[grp] = cp.breakpoints

            # Breakpoint rows — map signal indices back to trajectory frames/times
            for idx, sig_idx in enumerate(cp.breakpoints):
                safe_idx = min(sig_idx, n_signal - 1)
                bkp_rows.append({
                    "traj_id": traj_id,
                    "group": grp,
                    "breakpoint_idx": idx,
                    "signal_index": sig_idx,
                    "frame": int(frame_arr[safe_idx]),
                    "time_ps": float(time_arr[safe_idx]),
                    "n_cols_used": len(used_cols),
                })

            # Segment stat rows
            summary_cols = _SUMMARY_COLS.get(grp, [])
            boundaries = [0] + cp.breakpoints + [n_signal]
            for seg_id, (s_start, s_end) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            ):
                start_frame = int(frame_arr[s_start])
                end_frame = int(frame_arr[min(s_end - 1, n_signal - 1)])
                stats = _segment_stats(df, s_start, s_end, summary_cols)
                row: dict = {
                    "traj_id": traj_id,
                    "group": grp,
                    "segment_id": seg_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "start_ps": float(time_arr[s_start]),
                    "end_ps": float(time_arr[min(s_end - 1, n_signal - 1)]),
                    "n_frames": s_end - s_start,
                    "n_cols_used": len(used_cols),
                }
                row.update(stats)
                seg_rows.append(row)

    # Cross-group pairwise timing comparison
    cmp_rows: list[dict] = []
    for grp_a, grp_b in itertools.combinations(groups, 2):
        if grp_a not in group_bkps or grp_b not in group_bkps:
            continue
        metrics = _compare_breakpoints(
            group_bkps[grp_a], group_bkps[grp_b], tolerance_frames, time_arr
        )
        cmp_rows.append({
            "traj_id": traj_id,
            "group_a": grp_a,
            "group_b": grp_b,
            "tolerance_frames": tolerance_frames,
            **metrics,
        })

    return (
        pd.DataFrame(bkp_rows),
        pd.DataFrame(seg_rows),
        pd.DataFrame(cmp_rows),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Changepoint detection on GSA feature groups from pre-computed CSVs."
    )
    parser.add_argument(
        "--input-dir",
        default="output/gsa_features",
        help="Directory containing *_gsa_features.csv files (default: output/gsa_features)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/changepoints",
        help="Output directory for all result CSVs (default: output/changepoints)",
    )
    parser.add_argument(
        "--method",
        default="Pelt",
        help="Ruptures search method: Pelt (default), Binseg, BottomUp, Window, Dynp",
    )
    parser.add_argument(
        "--cost-model",
        default="rbf",
        help="Ruptures cost function: rbf (default), l1, l2, normal, ar, linear, rank",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=None,
        help="Penalty value for Pelt (omit to use log(n)*var heuristic)",
    )
    parser.add_argument(
        "--n-bkps",
        type=int,
        default=None,
        help="Fixed number of breakpoints; requires --method Binseg or Dynp",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=10,
        help="Minimum segment length in frames (default: 10)",
    )
    parser.add_argument(
        "--jump",
        type=int,
        default=5,
        help="Ruptures subsample step for speed (default: 5)",
    )
    parser.add_argument(
        "--tolerance-frames",
        type=int,
        default=50,
        help="Frame tolerance for timing comparison; breakpoints within ±N are 'shared' (default: 50)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["gsa", "iodine", "na_water", "combined"],
        choices=["gsa", "iodine", "na_water", "combined"],
        help="Feature groups to analyse (default: all four)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip z-score normalisation before changepoint detection",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*_gsa_features.csv"))
    if not csv_files:
        print(f"No *_gsa_features.csv files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    with RunContext.from_namespace(args, name="changepoint_feature_groups"):
        log_event(
            "info",
            f"Found {len(csv_files)} feature CSV(s); groups={args.groups}; "
            f"method={args.method}; cost={args.cost_model}",
            component="changepoint_feature_groups",
        )

        all_bkp: list[pd.DataFrame] = []
        all_seg: list[pd.DataFrame] = []
        all_cmp: list[pd.DataFrame] = []

        for csv_path in csv_files:
            traj_label = csv_path.stem
            with step(f"trajectory={traj_label}"):
                log_event(
                    "info",
                    f"Processing {csv_path.name}",
                    component="changepoint_feature_groups",
                )
                bkp_df, seg_df, cmp_df = process_csv(
                    csv_path,
                    method=args.method,
                    cost_model=args.cost_model,
                    penalty=args.penalty,
                    n_bkps=args.n_bkps,
                    min_size=args.min_size,
                    jump=args.jump,
                    tolerance_frames=args.tolerance_frames,
                    groups=args.groups,
                    normalize=not args.no_normalize,
                )

                # Derive clean traj_id for file naming (strip _gsa_features suffix)
                traj_id = traj_label[:-len("_gsa_features")] if traj_label.endswith("_gsa_features") else traj_label

                bkp_out = output_dir / f"{traj_id}_breakpoints.csv"
                seg_out = output_dir / f"{traj_id}_segment_stats.csv"
                bkp_df.to_csv(bkp_out, index=False)
                seg_df.to_csv(seg_out, index=False)
                log_event(
                    "info",
                    f"  → {bkp_out.name} ({len(bkp_df)} breakpoints), "
                    f"{seg_out.name} ({len(seg_df)} segments)",
                    component="changepoint_feature_groups",
                )

                all_bkp.append(bkp_df)
                all_seg.append(seg_df)
                all_cmp.append(cmp_df)

        # Aggregate outputs
        merged_bkp = pd.concat(all_bkp, ignore_index=True) if all_bkp else pd.DataFrame()
        merged_seg = pd.concat(all_seg, ignore_index=True) if all_seg else pd.DataFrame()
        merged_cmp = pd.concat(all_cmp, ignore_index=True) if all_cmp else pd.DataFrame()

        merged_bkp.to_csv(output_dir / "all_breakpoints.csv", index=False)
        merged_seg.to_csv(output_dir / "all_segment_stats.csv", index=False)
        merged_cmp.to_csv(output_dir / "changepoint_timing_comparison.csv", index=False)

        log_event(
            "info",
            f"Wrote all_breakpoints.csv ({len(merged_bkp)} rows), "
            f"all_segment_stats.csv ({len(merged_seg)} rows), "
            f"changepoint_timing_comparison.csv ({len(merged_cmp)} rows)",
            component="changepoint_feature_groups",
        )

    # Print manifest
    print("\nOutput files:")
    for f in sorted(output_dir.glob("*.csv")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
