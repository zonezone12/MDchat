"""Feature-group column lists for GSA changepoint analysis.

Single source of truth for the four analysis groups (gsa, iodine, na_water,
combined) and their summary / regime headline columns.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

METADATA_COLS: frozenset[str] = frozenset({"traj_id", "frame", "time_ps"})

DEFAULT_GROUPS: tuple[str, ...] = ("gsa", "iodine", "na_water", "combined")

GROUP_COLORS: dict[str, str] = {
    "gsa": "#1f77b4",
    "iodine": "#d62728",
    "na_water": "#2ca02c",
    "combined": "#9467bd",
}

GROUP_MARKERS: dict[str, str] = {
    "gsa": "|",
    "iodine": "v",
    "na_water": "s",
    "combined": "D",
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
}


def resolve_group_columns(df: pd.DataFrame, group: str) -> list[str]:
    """Return the column list for *group*, intersecting with available columns.

    For ``combined``, returns all numeric non-metadata columns present in *df*.
    """
    if group == "combined":
        return [
            c
            for c in df.columns
            if c not in METADATA_COLS and pd.api.types.is_numeric_dtype(df[c])
        ]
    col_map = {
        "gsa": GSA_COLS,
        "iodine": IODINE_COLS,
        "na_water": NA_WATER_COLS,
    }
    if group not in col_map:
        raise ValueError(f"Unknown feature group '{group}'. Choose from: {DEFAULT_GROUPS}")
    return [c for c in col_map[group] if c in df.columns]


def summary_feature_columns(group: str) -> list[str]:
    """Expand summary base columns to mean + std CSV column names."""
    bases = SUMMARY_COLS.get(group, [])
    cols: list[str] = []
    for base in bases:
        cols.append(f"{base}_mean")
        cols.append(f"{base}_std")
    return cols


def traj_id_from_features_stem(stem: str) -> str:
    """Strip ``_gsa_features`` suffix from a features CSV stem."""
    suffix = "_gsa_features"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


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
    full = {
        "gsa": GSA_COLS,
        "iodine": IODINE_COLS,
        "na_water": NA_WATER_COLS,
        "combined": all_numeric,
    }
    if groups is None:
        return full
    return {g: full[g] for g in groups if g in full}
