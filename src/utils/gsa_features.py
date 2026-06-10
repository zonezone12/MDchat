"""
Stateless per-frame feature functions for GSA nanocube analysis.

Every public function takes NumPy arrays (positions, COMs, etc.) and returns
scalars or flat dicts whose keys match the column names in
``GSA_features_table.txt`` and ``guest_GSA_features_table.txt``.

Heavy imports (scipy, sklearn) are deferred to the functions that need them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Re-export the existing RMSD helper for convenience
from src.TrajectoryMetrics.TrajectoryMetrics import rmsd_value_aligned


# ═══════════════════════════════════════════════════════════════════════════
#  Tier 1 — Whole-assembly shape
# ═══════════════════════════════════════════════════════════════════════════

def gyration_tensor_descriptors(coords: np.ndarray) -> Dict[str, float]:
    """Gyration tensor eigenvalues and derived shape descriptors.

    Parameters
    ----------
    coords : (N, 3) array
        Cartesian positions of assembly atoms.

    Returns
    -------
    dict with keys:
        assembly_rg, assembly_gyration_eig_1..3,
        assembly_asphericity, assembly_acylindricity,
        assembly_relative_shape_anisotropy
    """
    com = coords.mean(axis=0)
    d = coords - com
    S = (d[:, :, None] * d[:, None, :]).mean(axis=0)  # (3, 3)
    eigvals = np.sort(np.linalg.eigvalsh(S))  # ascending λ1 ≤ λ2 ≤ λ3
    l1, l2, l3 = eigvals

    rg2 = eigvals.sum()
    rg = float(np.sqrt(max(rg2, 0.0)))
    asphericity = l3 - 0.5 * (l1 + l2)
    acylindricity = l2 - l1
    denom = rg2 * rg2 if rg2 > 0 else 1.0
    kappa2 = (asphericity ** 2 + 0.75 * acylindricity ** 2) / denom

    return {
        "assembly_rg": rg,
        "assembly_gyration_eig_1": float(l1),
        "assembly_gyration_eig_2": float(l2),
        "assembly_gyration_eig_3": float(l3),
        "assembly_asphericity": float(asphericity),
        "assembly_acylindricity": float(acylindricity),
        "assembly_relative_shape_anisotropy": float(kappa2),
    }


def assembly_rmsd_to_ref(
    coords: np.ndarray,
    ref_coords: np.ndarray,
) -> float:
    """RMSD of current assembly coords against a reference (after optimal alignment)."""
    return float(rmsd_value_aligned(coords, ref_coords))


def assembly_frame_to_frame_rmsd(
    coords: np.ndarray,
    prev_coords: Optional[np.ndarray],
) -> float:
    """RMSD between current and previous frame (NaN if no previous)."""
    if prev_coords is None:
        return float("nan")
    return float(rmsd_value_aligned(coords, prev_coords))


# ═══════════════════════════════════════════════════════════════════════════
#  Tier 1 — Six-monomer geometry
# ═══════════════════════════════════════════════════════════════════════════

def compute_monomer_coms(monomer_coords_list: List[np.ndarray]) -> np.ndarray:
    """Return (n_monomers, 3) COM array from a list of per-monomer coordinate arrays."""
    return np.array([c.mean(axis=0) for c in monomer_coords_list])


def radial_distance_stats(
    monomer_coms: np.ndarray,
    assembly_center: np.ndarray,
) -> Dict[str, float]:
    """Distance from each monomer COM to the assembly center."""
    dists = np.linalg.norm(monomer_coms - assembly_center, axis=1)
    return {
        "radial_distance_mean": float(np.mean(dists)),
        "radial_distance_std": float(np.std(dists)),
        "radial_distance_min": float(np.min(dists)),
        "radial_distance_max": float(np.max(dists)),
    }


def pairwise_com_stats(monomer_coms: np.ndarray) -> Dict[str, float]:
    """All-pairs COM distances."""
    n = len(monomer_coms)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(float(np.linalg.norm(monomer_coms[i] - monomer_coms[j])))
    dists_arr = np.array(dists)
    return {
        "com_dist_mean": float(np.mean(dists_arr)),
        "com_dist_std": float(np.std(dists_arr)),
        "com_dist_min": float(np.min(dists_arr)),
        "com_dist_max": float(np.max(dists_arr)),
    }


def neighbor_opposite_stats(monomer_coms: np.ndarray) -> Dict[str, float]:
    """Neighbor and opposite-pair COM distance statistics.

    For a cube-like 6-monomer arrangement each monomer has 4 neighbors
    (the closest 4) and 1 opposite (the farthest).
    """
    n = len(monomer_coms)
    dist_mat = np.linalg.norm(
        monomer_coms[:, None, :] - monomer_coms[None, :, :], axis=2
    )
    neighbor_dists = []
    opposite_dists = []
    for i in range(n):
        d = dist_mat[i].copy()
        d[i] = np.inf
        order = np.argsort(d)
        others = [j for j in order if j != i]
        neighbors = [d[j] for j in others[:min(4, n - 1)]]
        neighbor_dists.extend(neighbors)
        opposite_dists.append(float(d[others[-1]]))

    neigh = np.array(neighbor_dists)
    oppo = np.array(opposite_dists)

    neigh_mean = float(np.mean(neigh))
    oppo_mean = float(np.mean(oppo))
    ratio = oppo_mean / neigh_mean if neigh_mean > 1e-12 else float("nan")

    return {
        "com_neighbor_dist_mean": neigh_mean,
        "com_neighbor_dist_std": float(np.std(neigh)),
        "com_opposite_dist_mean": oppo_mean,
        "com_opposite_dist_std": float(np.std(oppo)),
        "com_opposite_neighbor_ratio": ratio,
    }


def octahedrality_score(monomer_coms: np.ndarray) -> float:
    """How close 6 COMs are to a perfect octahedron.

    Defined as 1 - (std / mean) of the 12 nearest-neighbor distances.
    Perfect octahedron => score == 1.
    """
    n = len(monomer_coms)
    if n < 4:
        return float("nan")
    dist_mat = np.linalg.norm(
        monomer_coms[:, None, :] - monomer_coms[None, :, :], axis=2
    )
    nn_dists = []
    for i in range(n):
        d = dist_mat[i].copy()
        d[i] = np.inf
        nn = np.sort(d)[:4]
        nn_dists.extend(nn.tolist())
    nn_arr = np.array(nn_dists)
    m = nn_arr.mean()
    if m < 1e-12:
        return float("nan")
    return float(1.0 - nn_arr.std() / m)


# ═══════════════════════════════════════════════════════════════════════════
#  Tier 1 — Monomer orientation
# ═══════════════════════════════════════════════════════════════════════════

def amphiphile_axes(
    monomer_coords_list: List[np.ndarray],
    headgroup_coords_list: Optional[List[np.ndarray]] = None,
    tail_coords_list: Optional[List[np.ndarray]] = None,
) -> np.ndarray:
    """Per-monomer amphiphile axis vectors (n_monomers, 3).

    If headgroup/tail coords are given the axis points from tail COM to
    headgroup COM.  Otherwise the first principal component is used.
    """
    axes = []
    for i, mc in enumerate(monomer_coords_list):
        if headgroup_coords_list and tail_coords_list:
            hc = headgroup_coords_list[i].mean(axis=0)
            tc = tail_coords_list[i].mean(axis=0)
            v = hc - tc
        else:
            centered = mc - mc.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            v = Vt[0]
        norm = np.linalg.norm(v)
        axes.append(v / norm if norm > 1e-12 else v)
    return np.array(axes)


def amphiphile_axis_radial_cos_stats(
    axes: np.ndarray,
    monomer_coms: np.ndarray,
    assembly_center: np.ndarray,
) -> Dict[str, float]:
    """Cosine between each monomer's amphiphile axis and its radial direction."""
    radial = monomer_coms - assembly_center
    norms = np.linalg.norm(radial, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    radial_hat = radial / norms

    cosines = np.abs(np.sum(axes * radial_hat, axis=1))
    return {
        "amphiphile_axis_radial_cos_mean": float(np.mean(cosines)),
        "amphiphile_axis_radial_cos_std": float(np.std(cosines)),
        "amphiphile_axis_radial_cos_min": float(np.min(cosines)),
        "amphiphile_axis_radial_cos_max": float(np.max(cosines)),
    }


def monomer_twist_angle_std(
    axes: np.ndarray,
    prev_axes: Optional[np.ndarray],
) -> float:
    """Std-dev of per-monomer twist angles relative to previous frame (degrees).

    Returns NaN when no previous-frame axes are available.
    """
    if prev_axes is None:
        return float("nan")
    dots = np.sum(axes * prev_axes, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.degrees(np.arccos(np.abs(dots)))
    return float(np.std(angles))


# ═══════════════════════════════════════════════════════════════════════════
#  Tier 1 — Inter-monomer contacts
# ═══════════════════════════════════════════════════════════════════════════

def inter_monomer_contact_features(
    monomer_coords_list: List[np.ndarray],
    cutoff: float = 4.5,
    active_threshold: int = 1,
) -> Dict[str, float]:
    """Contact counts, graph density, largest connected component for monomer pairs.

    A contact between monomer *i* and *j* exists when any atom pair is within
    *cutoff* angstrom.
    """
    n = len(monomer_coords_list)
    pair_counts: List[int] = []
    adj = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            diff = monomer_coords_list[i][:, None, :] - monomer_coords_list[j][None, :, :]
            d2 = (diff * diff).sum(axis=2)
            count = int(np.sum(d2 < cutoff * cutoff))
            pair_counts.append(count)
            if count >= active_threshold:
                adj[i, j] = 1
                adj[j, i] = 1

    pc = np.array(pair_counts) if pair_counts else np.array([0])
    n_active = int(np.sum(adj[np.triu_indices(n, k=1)]))
    n_possible = n * (n - 1) // 2
    density = n_active / n_possible if n_possible > 0 else 0.0

    lcc = _largest_connected_component(adj, n)

    return {
        "total_inter_monomer_contacts": int(pc.sum()),
        "inter_contact_mean": float(np.mean(pc)),
        "inter_contact_std": float(np.std(pc)),
        "inter_contact_min": int(np.min(pc)),
        "inter_contact_max": int(np.max(pc)),
        "n_active_interfaces": n_active,
        "n_broken_interfaces": n_possible - n_active,
        "weakest_interface_contact_count": int(np.min(pc)),
        "contact_graph_density": float(density),
        "largest_connected_component_size": lcc,
    }


def _largest_connected_component(adj: np.ndarray, n: int) -> int:
    """BFS on a small adjacency matrix to find the largest CC size."""
    visited = [False] * n
    best = 0
    for start in range(n):
        if visited[start]:
            continue
        size = 0
        queue = [start]
        visited[start] = True
        while queue:
            node = queue.pop(0)
            size += 1
            for nb in range(n):
                if adj[node, nb] and not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)
        best = max(best, size)
    return best


# ═══════════════════════════════════════════════════════════════════════════
#  Tier 1 — Cavity / volume proxies
# ═══════════════════════════════════════════════════════════════════════════

def cavity_solvent_counts(
    assembly_center: np.ndarray,
    water_positions: Optional[np.ndarray],
    ion_positions: Optional[np.ndarray],
    cavity_radius: float = 8.0,
) -> Dict[str, int]:
    """Count waters and ions within *cavity_radius* of assembly center."""
    wc = 0
    if water_positions is not None and len(water_positions) > 0:
        wd = np.linalg.norm(water_positions - assembly_center, axis=1)
        wc = int(np.sum(wd < cavity_radius))

    ic = 0
    if ion_positions is not None and len(ion_positions) > 0:
        iond = np.linalg.norm(ion_positions - assembly_center, axis=1)
        ic = int(np.sum(iond < cavity_radius))

    return {"cavity_water_count": wc, "cavity_ion_count": ic}


def cavity_inner_atom_features(
    assembly_center: np.ndarray,
    inner_atom_positions: np.ndarray,
) -> Dict[str, float]:
    """Distance statistics for inner-facing heavy atoms relative to assembly center."""
    if len(inner_atom_positions) == 0:
        nan = float("nan")
        return {
            "cavity_min_heavy_atom_distance_from_center": nan,
            "cavity_inner_atom_distance_mean": nan,
            "cavity_inner_atom_distance_std": nan,
            "cavity_inner_atom_distance_min": nan,
            "cavity_inner_atom_distance_max": nan,
        }
    dists = np.linalg.norm(inner_atom_positions - assembly_center, axis=1)
    return {
        "cavity_min_heavy_atom_distance_from_center": float(np.min(dists)),
        "cavity_inner_atom_distance_mean": float(np.mean(dists)),
        "cavity_inner_atom_distance_std": float(np.std(dists)),
        "cavity_inner_atom_distance_min": float(np.min(dists)),
        "cavity_inner_atom_distance_max": float(np.max(dists)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Tier 1 — Amphiphile segregation
# ═══════════════════════════════════════════════════════════════════════════

def segregation_features(
    assembly_center: np.ndarray,
    hydrophilic_positions: np.ndarray,
    hydrophobic_positions: np.ndarray,
    headgroup_positions: Optional[np.ndarray] = None,
    hydrophobic_assembly_positions: Optional[np.ndarray] = None,
    water_positions: Optional[np.ndarray] = None,
    contact_cutoff: float = 3.5,
) -> Dict[str, float]:
    """Radial segregation between hydrophilic and hydrophobic moieties."""
    phi_r = _mean_radial(assembly_center, hydrophilic_positions)
    pho_r = _mean_radial(assembly_center, hydrophobic_positions)

    result: Dict[str, float] = {
        "hydrophilic_radial_distance_mean": phi_r,
        "hydrophobic_radial_distance_mean": pho_r,
        "hydrophilic_minus_hydrophobic_radial_distance": phi_r - pho_r,
    }

    hg_positions = headgroup_positions if headgroup_positions is not None else hydrophilic_positions
    ho_positions = hydrophobic_assembly_positions if hydrophobic_assembly_positions is not None else hydrophobic_positions

    if water_positions is not None and len(water_positions) > 0:
        result["headgroup_water_contacts"] = float(
            _count_contacts(hg_positions, water_positions, contact_cutoff)
        )
        result["hydrophobic_water_contacts"] = float(
            _count_contacts(ho_positions, water_positions, contact_cutoff)
        )
    else:
        result["headgroup_water_contacts"] = 0.0
        result["hydrophobic_water_contacts"] = 0.0

    return result


def _mean_radial(center: np.ndarray, positions: np.ndarray) -> float:
    if len(positions) == 0:
        return float("nan")
    return float(np.mean(np.linalg.norm(positions - center, axis=1)))


def _count_contacts(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    cutoff: float,
) -> int:
    """Count atom pairs within cutoff (brute-force, fine for moderate sizes)."""
    if len(coords_a) == 0 or len(coords_b) == 0:
        return 0
    if len(coords_a) * len(coords_b) > 5_000_000:
        return _count_contacts_chunked(coords_a, coords_b, cutoff)
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    d2 = (diff * diff).sum(axis=2)
    return int(np.sum(d2 < cutoff * cutoff))


def _count_contacts_chunked(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    cutoff: float,
    chunk_size: int = 500,
) -> int:
    """Memory-friendly contact counting for large atom groups."""
    total = 0
    c2 = cutoff * cutoff
    for i in range(0, len(coords_a), chunk_size):
        chunk = coords_a[i : i + chunk_size]
        diff = chunk[:, None, :] - coords_b[None, :, :]
        d2 = (diff * diff).sum(axis=2)
        total += int(np.sum(d2 < c2))
    return total


# ═══════════════════════════════════════════════════════════════════════════
#  Tier 1 — Monomer deformation
# ═══════════════════════════════════════════════════════════════════════════

def monomer_deformation_features(
    monomer_coords_list: List[np.ndarray],
    monomer_ref_coords_list: Optional[List[np.ndarray]] = None,
) -> Dict[str, float]:
    """Per-monomer Rg and RMSD (vs reference) statistics."""
    rgs = []
    for mc in monomer_coords_list:
        com = mc.mean(axis=0)
        rg2 = ((mc - com) ** 2).sum(axis=1).mean()
        rgs.append(float(np.sqrt(max(rg2, 0.0))))
    rg_arr = np.array(rgs)

    result: Dict[str, float] = {
        "monomer_rg_mean": float(np.mean(rg_arr)),
        "monomer_rg_std": float(np.std(rg_arr)),
    }

    if monomer_ref_coords_list is not None:
        rmsds = []
        for mc, ref in zip(monomer_coords_list, monomer_ref_coords_list):
            if mc.shape == ref.shape:
                rmsds.append(float(rmsd_value_aligned(mc, ref)))
            else:
                rmsds.append(float("nan"))
        rmsd_arr = np.array(rmsds)
        result["monomer_rmsd_mean"] = float(np.nanmean(rmsd_arr))
        result["monomer_rmsd_std"] = float(np.nanstd(rmsd_arr))
        result["monomer_deformation_max"] = float(np.nanmax(rmsd_arr))
    else:
        result["monomer_rmsd_mean"] = float("nan")
        result["monomer_rmsd_std"] = float("nan")
        result["monomer_deformation_max"] = float("nan")

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  Tier 2 — Gear / interlocking
# ═══════════════════════════════════════════════════════════════════════════

def pairwise_endpoint_distance_matrix(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
) -> np.ndarray:
    """All-pairs distance matrix between two endpoint (tooth) atom groups.

    Same layout as :meth:`EndpointAnalyzer.compute_endpoint_distances` per
    residue pair for a single frame: shape ``(n_ep_a, n_ep_b)``.
    """
    if len(coords_a) == 0 or len(coords_b) == 0:
        return np.empty((0, 0))
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    return np.linalg.norm(diff, axis=2)


def contacts_from_endpoint_distance_matrix(
    dist_matrix: np.ndarray,
    cutoff: float,
) -> int:
    """Count endpoint pairs within *cutoff* in an endpoint distance matrix."""
    if dist_matrix.size == 0:
        return 0
    return int(np.sum(dist_matrix < cutoff))


def tooth_contact_features_from_endpoint_matrices(
    dist_matrices: List[np.ndarray],
    cutoff: float,
) -> Dict[str, float]:
    """Aggregate tooth-contact stats from neighbor endpoint distance matrices."""
    nan = float("nan")
    if not dist_matrices:
        return {
            "tooth_contact_total": nan,
            "tooth_contact_min": nan,
            "tooth_contact_std": nan,
        }
    counts = np.array(
        [contacts_from_endpoint_distance_matrix(m, cutoff) for m in dist_matrices],
        dtype=float,
    )
    return {
        "tooth_contact_total": float(counts.sum()),
        "tooth_contact_min": float(counts.min()),
        "tooth_contact_std": float(counts.std()),
    }


def neighbor_monomer_pairs(
    monomer_coms: np.ndarray,
    n_neighbors: int = 4,
) -> List[Tuple[int, int]]:
    """Undirected monomer pairs among spatial nearest neighbors (COM-based)."""
    n = len(monomer_coms)
    com_dist = np.linalg.norm(
        monomer_coms[:, None, :] - monomer_coms[None, :, :], axis=2
    )
    pairs: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for i in range(n):
        d = com_dist[i].copy()
        d[i] = np.inf
        neighbors = np.argsort(d)[:n_neighbors]
        for j in neighbors:
            pair = (i, j) if i < j else (j, i)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
    return pairs


def neighbor_endpoint_distance_matrices(
    tooth_coords_list: List[np.ndarray],
    monomer_coms: np.ndarray,
    n_neighbors: int = 4,
) -> List[np.ndarray]:
    """Endpoint distance matrices for each spatially neighboring monomer pair."""
    matrices: List[np.ndarray] = []
    for i, j in neighbor_monomer_pairs(monomer_coms, n_neighbors=n_neighbors):
        matrices.append(
            pairwise_endpoint_distance_matrix(
                tooth_coords_list[i], tooth_coords_list[j]
            )
        )
    return matrices


def all_monomer_endpoint_distance_matrices(
    tooth_coords_list: List[np.ndarray],
) -> Dict[Tuple[int, int], np.ndarray]:
    """Endpoint distance matrix for every monomer pair ``(i, j)`` with ``i < j``."""
    n = len(tooth_coords_list)
    return {
        (i, j): pairwise_endpoint_distance_matrix(
            tooth_coords_list[i], tooth_coords_list[j]
        )
        for i in range(n)
        for j in range(i + 1, n)
    }


def endpoint_distance_matrix_record_features(
    dist_by_pair: Optional[Dict[Tuple[int, int], np.ndarray]],
    n_monomers: int,
) -> Dict[str, float]:
    """Per-pair and global endpoint-distance stats for CSV export.

    Column names match :meth:`EndpointAnalyzer.compute_endpoint_metrics`:
    ``endpoint_dist_{i}_{j}_{min,mean,max}`` plus assembly-wide
    ``endpoint_dist_{mean,min,max,std}``.
    """
    nan = float("nan")
    result: Dict[str, float] = {}
    all_vals: List[float] = []

    for i in range(n_monomers):
        for j in range(i + 1, n_monomers):
            key_min = f"endpoint_dist_{i}_{j}_min"
            key_mean = f"endpoint_dist_{i}_{j}_mean"
            key_max = f"endpoint_dist_{i}_{j}_max"
            dist_matrix = dist_by_pair.get((i, j)) if dist_by_pair else None
            if dist_matrix is not None and dist_matrix.size > 0:
                result[key_min] = float(np.min(dist_matrix))
                result[key_mean] = float(np.mean(dist_matrix))
                result[key_max] = float(np.max(dist_matrix))
                all_vals.extend(dist_matrix.ravel().tolist())
            else:
                result[key_min] = nan
                result[key_mean] = nan
                result[key_max] = nan

    if all_vals:
        arr = np.asarray(all_vals, dtype=float)
        result["endpoint_dist_mean"] = float(np.mean(arr))
        result["endpoint_dist_min"] = float(np.min(arr))
        result["endpoint_dist_max"] = float(np.max(arr))
        result["endpoint_dist_std"] = float(np.std(arr))
    else:
        result["endpoint_dist_mean"] = nan
        result["endpoint_dist_min"] = nan
        result["endpoint_dist_max"] = nan
        result["endpoint_dist_std"] = nan

    return result


def gear_interlocking_features(
    monomer_coords_list: List[np.ndarray],
    tooth_coords_list: Optional[List[np.ndarray]] = None,
    monomer_coms: Optional[np.ndarray] = None,
    assembly_center: Optional[np.ndarray] = None,
    cutoff: float = 4.5,
    endpoint_dist_by_pair: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
) -> Dict[str, float]:
    """Gear tooth contacts and phase offsets between neighboring monomers.

    Tooth-contact features are derived from the endpoint distance matrix
    (all pairwise distances between auto-detected tooth/endpoint atoms on
    neighboring monomers), matching
    :class:`~src.EndpointAnalyzer.EndpointAnalyzer` pair layout.  Each
    neighbor-pair matrix entry below *cutoff* counts as one tooth contact.

    Pass precomputed *endpoint_dist_by_pair* to reuse matrices from another
    pass; otherwise they are built from *tooth_coords_list*.

    Returned columns are ordered: tooth-contact aggregates, per-pair endpoint
    distance matrix summaries, then gear-phase metrics.

    If *tooth_coords_list* is None and no matrices are supplied, tooth-contact
    and endpoint-distance features are NaN (Tier 2 optional).
    """
    nan = float("nan")
    n = len(monomer_coords_list)

    if tooth_coords_list is None and endpoint_dist_by_pair is None:
        result: Dict[str, float] = {
            "tooth_contact_total": nan,
            "tooth_contact_min": nan,
            "tooth_contact_std": nan,
        }
        result.update(endpoint_distance_matrix_record_features(None, n))
        result.update({
            "gear_phase_offset_mean": nan,
            "gear_phase_offset_std": nan,
            "neighbor_relative_rotation_angle_mean": nan,
            "neighbor_relative_rotation_angle_std": nan,
        })
        return result

    if monomer_coms is None:
        monomer_coms = compute_monomer_coms(monomer_coords_list)
    if assembly_center is None:
        assembly_center = monomer_coms.mean(axis=0)

    if endpoint_dist_by_pair is None:
        if tooth_coords_list is None:
            result = {
                "tooth_contact_total": nan,
                "tooth_contact_min": nan,
                "tooth_contact_std": nan,
            }
            result.update(endpoint_distance_matrix_record_features(None, n))
            result.update({
                "gear_phase_offset_mean": nan,
                "gear_phase_offset_std": nan,
                "neighbor_relative_rotation_angle_mean": nan,
                "neighbor_relative_rotation_angle_std": nan,
            })
            return result
        endpoint_dist_by_pair = all_monomer_endpoint_distance_matrices(
            tooth_coords_list
        )

    neighbor_mats = [
        endpoint_dist_by_pair[pair]
        for pair in neighbor_monomer_pairs(monomer_coms)
        if pair in endpoint_dist_by_pair
    ]

    result = tooth_contact_features_from_endpoint_matrices(neighbor_mats, cutoff)
    result.update(endpoint_distance_matrix_record_features(endpoint_dist_by_pair, n))

    if tooth_coords_list is None:
        result.update({
            "gear_phase_offset_mean": nan,
            "gear_phase_offset_std": nan,
            "neighbor_relative_rotation_angle_mean": nan,
            "neighbor_relative_rotation_angle_std": nan,
        })
        return result

    # Phase offsets via projection angle around radial axis
    phases = _gear_phases(tooth_coords_list, monomer_coms, assembly_center)
    if phases is not None:
        offsets = []
        angles = []
        for i, j in neighbor_monomer_pairs(monomer_coms):
            offsets.append(abs(phases[i] - phases[j]))
            angles.append(abs(phases[i] - phases[j]) % 180.0)
        off_arr = np.array(offsets) if offsets else np.array([nan])
        ang_arr = np.array(angles) if angles else np.array([nan])
        result["gear_phase_offset_mean"] = float(np.nanmean(off_arr))
        result["gear_phase_offset_std"] = float(np.nanstd(off_arr))
        result["neighbor_relative_rotation_angle_mean"] = float(np.nanmean(ang_arr))
        result["neighbor_relative_rotation_angle_std"] = float(np.nanstd(ang_arr))
    else:
        result.update({
            "gear_phase_offset_mean": nan,
            "gear_phase_offset_std": nan,
            "neighbor_relative_rotation_angle_mean": nan,
            "neighbor_relative_rotation_angle_std": nan,
        })

    return result


def _gear_phases(
    tooth_coords_list: List[np.ndarray],
    monomer_coms: np.ndarray,
    assembly_center: np.ndarray,
) -> Optional[np.ndarray]:
    """Per-monomer gear phase angle (degrees) around the radial axis."""
    phases = []
    for i, tc in enumerate(tooth_coords_list):
        if len(tc) == 0:
            phases.append(float("nan"))
            continue
        radial = monomer_coms[i] - assembly_center
        rn = np.linalg.norm(radial)
        if rn < 1e-12:
            phases.append(float("nan"))
            continue
        radial_hat = radial / rn
        tooth_com = tc.mean(axis=0) - monomer_coms[i]
        proj = tooth_com - np.dot(tooth_com, radial_hat) * radial_hat
        pn = np.linalg.norm(proj)
        if pn < 1e-12:
            phases.append(0.0)
            continue
        ref = np.cross(radial_hat, np.array([0, 0, 1.0]))
        ref_n = np.linalg.norm(ref)
        if ref_n < 1e-12:
            ref = np.cross(radial_hat, np.array([0, 1.0, 0]))
            ref_n = np.linalg.norm(ref)
        ref = ref / ref_n
        cos_a = np.clip(np.dot(proj / pn, ref), -1, 1)
        phases.append(float(np.degrees(np.arccos(cos_a))))
    return np.array(phases)


# ═══════════════════════════════════════════════════════════════════════════
#  Tier 2 — Interface chemistry (geometric proxies)
# ═══════════════════════════════════════════════════════════════════════════

def interface_chemistry_features(
    monomer_coords_list: List[np.ndarray],
    donor_coords_list: Optional[List[np.ndarray]] = None,
    acceptor_coords_list: Optional[List[np.ndarray]] = None,
    hydrophobic_coords_list: Optional[List[np.ndarray]] = None,
    polar_coords_list: Optional[List[np.ndarray]] = None,
    cation_positions: Optional[np.ndarray] = None,
    anion_positions: Optional[np.ndarray] = None,
    water_positions: Optional[np.ndarray] = None,
    ion_positions: Optional[np.ndarray] = None,
    hbond_cutoff: float = 3.5,
    hydrophobic_cutoff: float = 4.5,
    salt_bridge_cutoff: float = 4.0,
    bridge_cutoff: float = 3.5,
) -> Dict[str, float]:
    """Approximate inter-monomer H-bonds, hydrophobic/polar contacts, salt bridges,
    water/ion bridges.  All geometry-based (distance only, no angle criterion).
    """
    nan = float("nan")
    n = len(monomer_coords_list)

    result: Dict[str, float] = {
        "inter_monomer_hbond_total": 0.0,
        "inter_monomer_hbond_min": nan,
        "hydrophobic_contact_total": 0.0,
        "polar_contact_total": 0.0,
        "salt_bridge_total": 0.0,
        "water_bridge_count_between_monomers": 0.0,
        "ion_bridge_count_between_monomers": 0.0,
    }

    if donor_coords_list and acceptor_coords_list:
        hb_counts = []
        for i in range(n):
            for j in range(i + 1, n):
                c = _count_contacts(donor_coords_list[i], acceptor_coords_list[j], hbond_cutoff)
                c += _count_contacts(donor_coords_list[j], acceptor_coords_list[i], hbond_cutoff)
                hb_counts.append(c)
        hb_arr = np.array(hb_counts) if hb_counts else np.array([0])
        result["inter_monomer_hbond_total"] = float(hb_arr.sum())
        result["inter_monomer_hbond_min"] = float(hb_arr.min())

    if hydrophobic_coords_list:
        hpc = 0
        for i in range(n):
            for j in range(i + 1, n):
                hpc += _count_contacts(
                    hydrophobic_coords_list[i], hydrophobic_coords_list[j], hydrophobic_cutoff
                )
        result["hydrophobic_contact_total"] = float(hpc)

    if polar_coords_list:
        pc = 0
        for i in range(n):
            for j in range(i + 1, n):
                pc += _count_contacts(polar_coords_list[i], polar_coords_list[j], hbond_cutoff)
        result["polar_contact_total"] = float(pc)

    if cation_positions is not None and anion_positions is not None:
        result["salt_bridge_total"] = float(
            _count_contacts(cation_positions, anion_positions, salt_bridge_cutoff)
        )

    if water_positions is not None and len(water_positions) > 0:
        wbc = _count_bridges(monomer_coords_list, water_positions, bridge_cutoff)
        result["water_bridge_count_between_monomers"] = float(wbc)

    if ion_positions is not None and len(ion_positions) > 0:
        ibc = _count_bridges(monomer_coords_list, ion_positions, bridge_cutoff)
        result["ion_bridge_count_between_monomers"] = float(ibc)

    return result


def _count_bridges(
    monomer_coords_list: List[np.ndarray],
    bridge_positions: np.ndarray,
    cutoff: float,
) -> int:
    """Count bridge molecules touching 2+ monomers simultaneously."""
    n = len(monomer_coords_list)
    touching = np.zeros((len(bridge_positions), n), dtype=bool)
    for mi, mc in enumerate(monomer_coords_list):
        if len(mc) == 0:
            continue
        for bi, bp in enumerate(bridge_positions):
            d = np.linalg.norm(mc - bp, axis=1).min()
            if d < cutoff:
                touching[bi, mi] = True
    return int(np.sum(touching.sum(axis=1) >= 2))


# ═══════════════════════════════════════════════════════════════════════════
#  Tier 2 — Specific distances / pore openings
# ═══════════════════════════════════════════════════════════════════════════

def pore_opening_distances(monomer_coms: np.ndarray) -> Dict[str, float]:
    """Distances across the 3 opposite-face pairs (for a 6-face cube).

    Sorted descending so pore_opening_distance_1 is the largest.
    """
    n = len(monomer_coms)
    if n < 6:
        return {
            "pore_opening_distance_1": float("nan"),
            "pore_opening_distance_2": float("nan"),
            "pore_opening_distance_3": float("nan"),
        }
    dist_mat = np.linalg.norm(
        monomer_coms[:, None, :] - monomer_coms[None, :, :], axis=2
    )
    used = set()
    pairs = []
    for _ in range(3):
        best_d = -1.0
        best_pair = (0, 0)
        for i in range(n):
            if i in used:
                continue
            for j in range(i + 1, n):
                if j in used:
                    continue
                if dist_mat[i, j] > best_d:
                    best_d = dist_mat[i, j]
                    best_pair = (i, j)
        pairs.append(best_d)
        used.add(best_pair[0])
        used.add(best_pair[1])

    pairs.sort(reverse=True)
    while len(pairs) < 3:
        pairs.append(float("nan"))
    return {
        "pore_opening_distance_1": float(pairs[0]),
        "pore_opening_distance_2": float(pairs[1]),
        "pore_opening_distance_3": float(pairs[2]),
    }


def selected_group_pair_distances(
    group_a_positions: np.ndarray,
    group_b_positions: np.ndarray,
) -> Dict[str, float]:
    """COM distance and minimum atom-pair distance between two groups."""
    if len(group_a_positions) == 0 or len(group_b_positions) == 0:
        return {"group_com_distance": float("nan"), "group_min_distance": float("nan")}
    com_a = group_a_positions.mean(axis=0)
    com_b = group_b_positions.mean(axis=0)
    diff = group_a_positions[:, None, :] - group_b_positions[None, :, :]
    d2 = (diff * diff).sum(axis=2)
    return {
        "group_com_distance": float(np.linalg.norm(com_a - com_b)),
        "group_min_distance": float(np.sqrt(d2.min())),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Guest features — Location
# ═══════════════════════════════════════════════════════════════════════════

def guest_location_features(
    guest_positions: np.ndarray,
    assembly_center: np.ndarray,
    inner_radius: float = 5.0,
    outer_radius: float = 12.0,
    surface_shell: float = 3.0,
) -> Dict[str, float]:
    """Classify guest molecules into radial shells and compute spatial statistics.

    Shells:
    - inside cavity: d < inner_radius
    - near inner surface: inner_radius ≤ d < inner_radius + surface_shell
    - near outer surface: outer_radius - surface_shell ≤ d < outer_radius
    - bulk: d ≥ outer_radius
    """
    if len(guest_positions) == 0:
        return _empty_guest_location()

    dists = np.linalg.norm(guest_positions - assembly_center, axis=1)

    inside = dists < inner_radius
    near_inner = (dists >= inner_radius) & (dists < inner_radius + surface_shell)
    near_outer = (dists >= outer_radius - surface_shell) & (dists < outer_radius)
    bulk = dists >= outer_radius

    result: Dict[str, float] = {
        "n_guest_inside_cavity": float(np.sum(inside)),
        "n_guest_near_inner_surface": float(np.sum(near_inner)),
        "n_guest_near_outer_surface": float(np.sum(near_outer)),
        "n_guest_bulk": float(np.sum(bulk)),
        "guest_radial_distance_mean": float(np.mean(dists)),
        "guest_radial_distance_std": float(np.std(dists)),
        "guest_radial_distance_min": float(np.min(dists)),
        "guest_radial_distance_max": float(np.max(dists)),
    }

    if len(guest_positions) >= 2:
        dipole_vec = guest_positions.mean(axis=0) - assembly_center
        result["guest_spatial_dipole_magnitude"] = float(np.linalg.norm(dipole_vec))
    else:
        result["guest_spatial_dipole_magnitude"] = float(dists[0]) if len(dists) == 1 else float("nan")

    return result


def _empty_guest_location() -> Dict[str, float]:
    nan = float("nan")
    return {
        "n_guest_inside_cavity": 0.0,
        "n_guest_near_inner_surface": 0.0,
        "n_guest_near_outer_surface": 0.0,
        "n_guest_bulk": 0.0,
        "guest_radial_distance_mean": nan,
        "guest_radial_distance_std": nan,
        "guest_radial_distance_min": nan,
        "guest_radial_distance_max": nan,
        "guest_spatial_dipole_magnitude": nan,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Guest features — GSA contacts
# ═══════════════════════════════════════════════════════════════════════════

def guest_gsa_contact_features(
    guest_positions: np.ndarray,
    assembly_positions: np.ndarray,
    hydrophobic_positions: Optional[np.ndarray] = None,
    hydrophilic_positions: Optional[np.ndarray] = None,
    cutoff: float = 4.0,
) -> Dict[str, float]:
    """Guest-to-assembly contact counts and hydrophobic preference."""
    nan = float("nan")
    if len(guest_positions) == 0 or len(assembly_positions) == 0:
        return {
            "n_guest_gsa_contacts": 0,
            "guest_gsa_min_distance": nan,
            "n_guest_hydrophobic_contacts": 0,
            "n_guest_hydrophilic_contacts": 0,
            "guest_hydrophobic_preference_ratio": nan,
        }

    n_contacts = _count_contacts(guest_positions, assembly_positions, cutoff)

    diff = guest_positions[:, None, :] - assembly_positions[None, :, :]
    d2 = (diff * diff).sum(axis=2)
    min_d = float(np.sqrt(d2.min()))

    n_pho = 0
    if hydrophobic_positions is not None and len(hydrophobic_positions) > 0:
        n_pho = _count_contacts(guest_positions, hydrophobic_positions, cutoff)

    n_phi = 0
    if hydrophilic_positions is not None and len(hydrophilic_positions) > 0:
        n_phi = _count_contacts(guest_positions, hydrophilic_positions, cutoff)

    pref = n_pho / n_phi if n_phi > 0 else (float("inf") if n_pho > 0 else nan)

    return {
        "n_guest_gsa_contacts": n_contacts,
        "guest_gsa_min_distance": min_d,
        "n_guest_hydrophobic_contacts": n_pho,
        "n_guest_hydrophilic_contacts": n_phi,
        "guest_hydrophobic_preference_ratio": float(pref),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Guest features — Monomer interface
# ═══════════════════════════════════════════════════════════════════════════

def guest_monomer_interface_features(
    guest_positions: np.ndarray,
    monomer_coords_list: List[np.ndarray],
    cutoff: float = 4.0,
) -> Dict[str, float]:
    """Per-monomer guest contacts, bridging statistics."""
    nan = float("nan")
    if len(guest_positions) == 0:
        return {
            "guest_interface_count_total": 0,
            "guest_interface_count_mean": nan,
            "guest_interface_count_std": nan,
            "guest_interface_count_max": 0,
            "n_guest_bridging_2_monomers": 0,
            "n_guest_bridging_3plus_monomers": 0,
            "guest_bridge_count_total": 0,
        }

    n_mon = len(monomer_coords_list)
    per_mon = []
    per_guest_touching = np.zeros((len(guest_positions), n_mon), dtype=bool)

    for mi, mc in enumerate(monomer_coords_list):
        c = _count_contacts(guest_positions, mc, cutoff)
        per_mon.append(c)
        if len(mc) > 0:
            for gi, gp in enumerate(guest_positions):
                d = np.linalg.norm(mc - gp, axis=1).min()
                if d < cutoff:
                    per_guest_touching[gi, mi] = True

    pm_arr = np.array(per_mon)
    touching_counts = per_guest_touching.sum(axis=1)

    return {
        "guest_interface_count_total": int(pm_arr.sum()),
        "guest_interface_count_mean": float(np.mean(pm_arr)),
        "guest_interface_count_std": float(np.std(pm_arr)),
        "guest_interface_count_max": int(pm_arr.max()),
        "n_guest_bridging_2_monomers": int(np.sum(touching_counts == 2)),
        "n_guest_bridging_3plus_monomers": int(np.sum(touching_counts >= 3)),
        "guest_bridge_count_total": int(np.sum(touching_counts >= 2)),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Guest features — Cavity water / coupling
# ═══════════════════════════════════════════════════════════════════════════

def guest_cavity_and_water_features(
    guest_positions: np.ndarray,
    assembly_center: np.ndarray,
    water_positions: Optional[np.ndarray],
    cavity_radius: float = 8.0,
    water_contact_cutoff: float = 3.5,
) -> Dict[str, float]:
    """Guest-cavity water overlap and water contacts near guest."""
    n_inside = 0
    if len(guest_positions) > 0:
        gd = np.linalg.norm(guest_positions - assembly_center, axis=1)
        n_inside = int(np.sum(gd < cavity_radius))

    cavity_wc = 0
    water_near_guest = 0
    guest_water_contacts = 0
    if water_positions is not None and len(water_positions) > 0:
        wd = np.linalg.norm(water_positions - assembly_center, axis=1)
        cavity_wc = int(np.sum(wd < cavity_radius))

        if len(guest_positions) > 0:
            water_near_guest = _count_contacts(guest_positions, water_positions, water_contact_cutoff)
            guest_water_contacts = water_near_guest

    return {
        "n_guest_inside_cavity": float(n_inside),
        "cavity_water_count": float(cavity_wc),
        "water_count_near_guest": float(water_near_guest),
        "guest_water_contact_count": float(guest_water_contacts),
    }


def guest_coupling_features(
    guest_positions: np.ndarray,
    monomer_coords_list: List[np.ndarray],
    cutoff: float = 4.0,
) -> Dict[str, float]:
    """Guest-per-monomer contact stats and mediated interfaces."""
    nan = float("nan")
    if len(guest_positions) == 0:
        return {
            "guest_per_monomer_contact_mean": nan,
            "guest_per_monomer_contact_std": nan,
            "guest_contacted_monomer_count": 0,
            "guest_mediated_interface_count": 0,
        }

    n_mon = len(monomer_coords_list)
    per_mon = []
    for mc in monomer_coords_list:
        per_mon.append(_count_contacts(guest_positions, mc, cutoff))
    pm_arr = np.array(per_mon)

    contacted = int(np.sum(pm_arr > 0))
    mediated = contacted * (contacted - 1) // 2 if contacted >= 2 else 0

    return {
        "guest_per_monomer_contact_mean": float(np.mean(pm_arr)),
        "guest_per_monomer_contact_std": float(np.std(pm_arr)),
        "guest_contacted_monomer_count": contacted,
        "guest_mediated_interface_count": mediated,
    }
