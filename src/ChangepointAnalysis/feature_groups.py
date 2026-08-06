"""Feature-group column lists for GSA changepoint analysis.

Single source of truth for the four analysis groups (gsa, iodine, na_water,
combined) and their summary / regime headline columns.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

METADATA_COLS: frozenset[str] = frozenset({"traj_id", "frame", "time_ps"})

DEFAULT_GROUPS: tuple[str, ...] = ("gsa", "iodine", "na_water", "combined")

# Extended set including the endpoint-site metastructure group.
ALL_GROUPS: tuple[str, ...] = ("gsa", "iodine", "na_water", "combined", "endpoint")

GROUP_COLORS: dict[str, str] = {
    "gsa": "#1f77b4",
    "iodine": "#d62728",
    "na_water": "#2ca02c",
    "combined": "#9467bd",
    "endpoint": "#ff7f0e",
}

GROUP_MARKERS: dict[str, str] = {
    "gsa": "|",
    "iodine": "v",
    "na_water": "s",
    "combined": "D",
    "endpoint": "o",
}

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
SUMMARY_COLS: dict[str, list[str]] = {
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
    "endpoint": [
        "endpoint_dist_mean", "endpoint_dist_min",
        "endpoint_dist_max", "endpoint_dist_std",
    ],
}

# Headline segment metrics for regime-direction checks (early vs late segments).
# Also used as SEGMENT_KEY_COLS in cohort regime summaries.
REGIME_COLS: dict[str, list[str]] = {
    "gsa": [
        "assembly_rg_mean",
        "assembly_rmsd_to_ref_mean",
        "octahedrality_score_mean",
        "total_inter_monomer_contacts_mean",
    ],
    "iodine": [
        "n_guest_inside_cavity_mean",
        "n_guest_bulk_mean",
        "guest_radial_distance_mean_mean",
        "guest_contacted_monomer_count_mean",
    ],
    "na_water": [
        "cavity_water_count_mean",
        "cavity_ion_count_mean",
        "guest_water_contact_count_mean",
    ],
    "combined": [
        "assembly_rg_mean",
        "n_guest_inside_cavity_mean",
        "cavity_ion_count_mean",
    ],
    "endpoint": [
        "endpoint_dist_mean_mean",
        "endpoint_dist_min_mean",
        "endpoint_dist_max_mean",
    ],
}


def resolve_group_columns(df: pd.DataFrame, group: str) -> list[str]:
    """Return the column list for *group*, intersecting with available columns.

    For ``combined``, returns all numeric non-metadata columns present in *df*.
    For ``endpoint``, returns every column whose name starts with ``endpoint_dist_``.
    """
    if group == "combined":
        return [
            c
            for c in df.columns
            if c not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
    if group == "endpoint":
        return [
            c
            for c in df.columns
            if c.startswith("endpoint_dist_")
            and c not in METADATA_COLS
            and pd.api.types.is_numeric_dtype(df[c])
        ]
    col_map = {
        "gsa": GSA_COLS,
        "iodine": IODINE_COLS,
        "na_water": NA_WATER_COLS,
    }
    if group not in col_map:
        raise ValueError(
            f"Unknown feature group '{group}'. Choose from: {ALL_GROUPS}"
        )
    return [c for c in col_map[group] if c in df.columns]


def summary_feature_columns(group: str) -> list[str]:
    """Expand summary base columns to mean + std CSV column names."""
    bases = SUMMARY_COLS.get(group, [])
    cols: list[str] = []
    for base in bases:
        cols.append(f"{base}_mean")
        cols.append(f"{base}_std")
    return cols


def traj_id_from_features_stem(
    stem: str,
    *,
    suffix: str | None = None,
) -> str:
    """Strip a features-CSV stem suffix.

    If *suffix* is given, only that suffix is stripped. Otherwise known
    suffixes (``_gsa_features``, ``_endpoint_features``) are tried in order.
    """
    if suffix is not None:
        # Accept both ``_gsa_features`` and ``_gsa_features.csv`` forms.
        bare = suffix[:-4] if suffix.endswith(".csv") else suffix
        if stem.endswith(bare):
            return stem[: -len(bare)]
        return stem
    for known in ("_gsa_features", "_endpoint_features"):
        if stem.endswith(known):
            return stem[: -len(known)]
    return stem


def features_csv_path(
    features_dir: Path | str,
    traj_id: str,
    *,
    suffix: str = "_gsa_features.csv",
) -> Path:
    """Build ``<features_dir>/<traj_id><suffix>``."""
    if not suffix.startswith("_") and not suffix.startswith("."):
        suffix = f"_{suffix}"
    if not suffix.endswith(".csv"):
        suffix = f"{suffix}.csv"
    return Path(features_dir) / f"{traj_id}{suffix}"


def resolve_features_csv(
    features_dir: Path | str,
    traj_id: str,
    *,
    suffix: str | None = None,
) -> Optional[Path]:
    """Locate a features CSV for *traj_id*, trying *suffix* then known defaults."""
    features_dir = Path(features_dir)
    candidates: list[str] = []
    if suffix is not None:
        bare = suffix if suffix.endswith(".csv") else f"{suffix}.csv"
        if not bare.startswith("_"):
            bare = f"_{bare}"
        candidates.append(bare)
    candidates.extend(["_gsa_features.csv", "_endpoint_features.csv"])
    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        path = features_dir / f"{traj_id}{cand}"
        if path.exists():
            return path
    return None


def all_numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """All numeric non-metadata columns in *df*."""
    return [
        c
        for c in df.columns
        if c not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]


def group_column_map(df: pd.DataFrame, groups: Sequence[str] | None = None) -> dict[str, list[str]]:
    """Map group name -> candidate columns (not filtered to available)."""
    all_numeric = all_numeric_feature_columns(df)
    endpoint_cols = [c for c in all_numeric if c.startswith("endpoint_dist_")]
    full = {
        "gsa": GSA_COLS,
        "iodine": IODINE_COLS,
        "na_water": NA_WATER_COLS,
        "combined": all_numeric,
        "endpoint": endpoint_cols,
    }
    if groups is None:
        return full
    return {g: full[g] for g in groups if g in full}
