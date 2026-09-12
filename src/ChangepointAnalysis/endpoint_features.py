"""Endpoint-site feature extraction for metastructure changepoint analysis.

Computes per-frame distances between chemically meaningful endpoint *sites*
(fused ring-system centroids and singleton atom sites) via a single
``TrajectoryIterator`` pass, then flattens them into feature CSVs suitable for
``ChangepointPipeline(groups=('endpoint',))``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class EndpointFeatureConfig:
    """Configuration for endpoint-site feature extraction."""

    gsa_resname: str = "MOL"
    n_monomers: int = 6
    monomer_selections: Optional[list[str]] = None
    use_ring_centroids: bool = True
    ring_min_gap_deg: Optional[float] = 35.0
    ring_max_per_ring: int = 3
    step_back_from_terminals: bool = True
    start: Optional[int] = None
    stop: Optional[int] = None
    step: int = 1
    time_per_frame_ps: float = 1.0
    pair_summaries: tuple[str, ...] = ("min", "mean", "max")
    include_site_pairs: bool = False
    # Paper d1 = Murata C2–C3 (R2/R3 equatorial ipso carbons), not hull s3/s7.
    include_paper_d1: bool = True
    paper_d1_roles: tuple[str, str] = ("r2", "r3")
    # Deprecated hull-index fallback when a monomer is not a GSA.
    paper_d1_s3_site: int = 3
    paper_d1_s7_site: int = 7
    paper_d1_open_lo: float = 4.5
    paper_d1_open_hi: float = 5.5
    # Murata G0: cation–π (pole Py+ COM–Ph COM), d2, RMSD, A/B/C1/C2 labels.
    include_murata_criteria: bool = True
    murata_cation_pi_open_lo: float = 6.5
    murata_rmsd_selection: str = "resname MOL"
    murata_rmsd_ref_frame: int = 0
    # TrajectoryIterator parallelization for the endpoint-site distance pass.
    # None = platform default (1 on Windows, -1 elsewhere); 1 = sequential;
    # -1 = all CPUs; >1 = that many workers.
    n_jobs: Optional[int] = None
    use_dask: bool = False
    max_workers_for_io: Optional[int] = None


def _default_n_jobs() -> int:
    """Windows coords-parallel path is usually I/O-bound; prefer sequential there."""
    import sys

    return 1 if sys.platform.startswith("win") else -1


PAPER_D1_COL_PREFIX = "paper_d1_"


def _d1_site_keys(
    d1_atoms_df: pd.DataFrame,
    *,
    s3_site: int = 3,
    s7_site: int = 7,
    roles: Optional[tuple[str, str]] = None,
) -> tuple[str, tuple[Any, Any]]:
    """Column and the two site keys used to pair d1 atoms."""
    if (
        roles is not None
        and "role" in d1_atoms_df.columns
        and set(d1_atoms_df["role"].astype(str)) >= set(roles)
    ):
        return "role", (str(roles[0]), str(roles[1]))
    if "role" in d1_atoms_df.columns and {"r2", "r3"} <= set(
        d1_atoms_df["role"].astype(str)
    ):
        return "role", ("r2", "r3")
    return "endpoint_site", (int(s3_site), int(s7_site))


def _paper_d1_pair_layout(
    d1_atoms_df: pd.DataFrame,
    universe: Any,
    *,
    s3_site: int = 3,
    s7_site: int = 7,
    roles: Optional[tuple[str, str]] = None,
) -> tuple[list[int], list[tuple[str, int, int]]]:
    """Return (MDA atom indices by slot, list of (col, slot_a, slot_b))."""
    key, (site_a, site_b) = _d1_site_keys(
        d1_atoms_df, s3_site=s3_site, s7_site=s7_site, roles=roles
    )
    ring_index: dict[int, dict[Any, int]] = {}
    id_to_index = {int(a.id): int(a.index) for a in universe.atoms}
    for _, row in d1_atoms_df.iterrows():
        mon = int(row["monomer"])
        site = row[key] if key == "role" else int(row[key])
        if key == "role":
            site = str(site)
        ring_id = int(row["d1_ring_atom_id"])
        if ring_id not in id_to_index:
            raise ValueError(f"d1 ring atom id {ring_id} not in universe")
        ring_index.setdefault(mon, {})[site] = id_to_index[ring_id]

    monomers = sorted(ring_index)
    for mon in monomers:
        for site in (site_a, site_b):
            if site not in ring_index[mon]:
                raise ValueError(f"Missing d1 ring atom for monomer {mon} site {site}")

    slots: list[tuple[int, Any]] = []
    atom_idxs: list[int] = []
    for mon in monomers:
        for site in (site_a, site_b):
            slots.append((mon, site))
            atom_idxs.append(ring_index[mon][site])
    slot_pos = {slot: k for k, slot in enumerate(slots)}

    pair_specs: list[tuple[str, int, int]] = []
    for i in monomers:
        for j in monomers:
            if i == j:
                continue
            for si, sj in ((site_a, site_b), (site_b, site_a)):
                col = paper_d1_column_name(i, si, j, sj)
                pair_specs.append((col, slot_pos[(i, si)], slot_pos[(j, sj)]))
    return atom_idxs, pair_specs


class PaperD1FrameObserver:
    """Accumulate paper-d1 distances during the same TrajectoryIterator pass."""

    def __init__(
        self,
        atom_idxs: Sequence[int],
        pair_specs: Sequence[tuple[str, int, int]],
        *,
        n_frame_rows: int,
        open_lo: float = 4.5,
        open_hi: float = 5.5,
    ):
        from src.Aggregator import ResultsGroup

        self.results: dict[str, Any] = {}
        self.atom_idxs = [int(i) for i in atom_idxs]
        self.pair_specs = list(pair_specs)
        self.colnames = [c for c, _, _ in self.pair_specs]
        self.open_lo = float(open_lo)
        self.open_hi = float(open_hi)
        self._n_frame_rows = int(n_frame_rows)
        self._ResultsGroup = ResultsGroup
        n_pairs = len(self.pair_specs)
        self.results["d1_dists"] = np.full(
            (self._n_frame_rows, n_pairs), np.nan, dtype=float
        )

    def get_selections_needed(self) -> list[str]:
        return []

    def on_frame_start(self, iterator: Any) -> None:
        return None

    def on_frame(self, ts: Any, frame_idx: int, universe: Any) -> None:
        idx = int(frame_idx)
        if idx < 0 or idx >= self._n_frame_rows:
            return
        pos = np.asarray(universe.atoms[self.atom_idxs].positions, dtype=float)
        dists = self.results["d1_dists"]
        for k, (_, ia, ib) in enumerate(self.pair_specs):
            dists[idx, k] = float(np.linalg.norm(pos[ia] - pos[ib]))

    def on_frame_end(self, iterator: Any) -> None:
        return None

    def _get_aggregator(self) -> Any:
        return self._ResultsGroup(
            lookup={"d1_dists": self._ResultsGroup.ndarray_merge_nonnan}
        )

    def merge_results(self, other: Any) -> None:
        if other is None:
            return
        other_dists = None
        if isinstance(other, dict):
            other_dists = other.get("d1_dists")
        elif hasattr(other, "results"):
            other_dists = other.results.get("d1_dists")
        if other_dists is None:
            return
        self_arr = self.results["d1_dists"]
        valid = ~np.isnan(other_dists)
        self_arr[valid] = other_dists[valid]

    def to_dataframe(self) -> pd.DataFrame:
        dists = np.asarray(self.results["d1_dists"], dtype=float)
        data: dict[str, Any] = {}
        for k, col in enumerate(self.colnames):
            data[col] = dists[:, k]
        with np.errstate(all="ignore"):
            data["paper_d1_min"] = np.nanmin(dists, axis=1)
            data["paper_d1_n_open"] = np.sum(
                (dists >= self.open_lo) & (dists <= self.open_hi), axis=1
            ).astype(int)
            data["paper_d1_n_closed"] = np.sum(dists < self.open_lo, axis=1).astype(int)
            data["paper_d1_n_elongated"] = np.sum(dists > self.open_hi, axis=1).astype(
                int
            )
        return pd.DataFrame(data)


def all_pairs_to_metrics_df(
    all_pairs: dict,
    n_res: int,
    n_frames: int,
    *,
    summaries: Sequence[str] = ("min", "mean", "max"),
    include_site_pairs: bool = False,
) -> pd.DataFrame:
    """Flatten ``all_pairs[(i, j)]`` arrays into a per-frame metrics DataFrame.

    Columns
    -------
    ``endpoint_dist_{i}_{j}_{min,mean,max}``
        Aggregate over all site-pairs between monomers *i* and *j*.
    ``endpoint_dist_{mean,min,max,std}``
        Assembly-wide aggregates over every site-pair distance.
    ``endpoint_dist_{i}s{a}_{j}s{b}`` (optional)
        Raw distance between site *a* of monomer *i* and site *b* of monomer *j*.
    """
    summary_set = set(summaries)
    data: dict[str, Any] = {"frame": np.arange(n_frames, dtype=int)}
    assembly_blocks: list[np.ndarray] = []

    for i in range(n_res):
        for j in range(i + 1, n_res):
            prefix = f"endpoint_dist_{i}_{j}"
            if (i, j) not in all_pairs:
                for s in summaries:
                    data[f"{prefix}_{s}"] = np.full(n_frames, np.nan)
                continue

            pair_array = np.asarray(all_pairs[(i, j)], dtype=float)
            if pair_array.ndim == 2:
                # (n_frames, n_pairs) legacy shape
                flat = pair_array
            elif pair_array.ndim == 3:
                flat = pair_array.reshape(n_frames, -1)
            else:
                raise ValueError(
                    f"Unexpected all_pairs[{(i, j)}] shape {pair_array.shape}"
                )

            with np.errstate(all="ignore"):
                if "mean" in summary_set:
                    data[f"{prefix}_mean"] = np.nanmean(flat, axis=1)
                if "min" in summary_set:
                    data[f"{prefix}_min"] = np.nanmin(flat, axis=1)
                if "max" in summary_set:
                    data[f"{prefix}_max"] = np.nanmax(flat, axis=1)
                if "std" in summary_set:
                    data[f"{prefix}_std"] = np.nanstd(flat, axis=1)
            assembly_blocks.append(flat)

            if include_site_pairs and pair_array.ndim == 3:
                n_a, n_b = pair_array.shape[1], pair_array.shape[2]
                for a in range(n_a):
                    for b in range(n_b):
                        col = pair_array[:, a, b]
                        data[f"endpoint_dist_{i}s{a}_{j}s{b}"] = np.where(
                            np.isfinite(col), col, np.nan
                        )

    if assembly_blocks:
        all_flat = np.concatenate(assembly_blocks, axis=1)
        with np.errstate(all="ignore"):
            data["endpoint_dist_mean"] = np.nanmean(all_flat, axis=1)
            data["endpoint_dist_min"] = np.nanmin(all_flat, axis=1)
            data["endpoint_dist_max"] = np.nanmax(all_flat, axis=1)
            data["endpoint_dist_std"] = np.nanstd(all_flat, axis=1)
    else:
        data["endpoint_dist_mean"] = np.full(n_frames, np.nan)
        data["endpoint_dist_min"] = np.full(n_frames, np.nan)
        data["endpoint_dist_max"] = np.full(n_frames, np.nan)
        data["endpoint_dist_std"] = np.full(n_frames, np.nan)

    return pd.DataFrame(data)


def _count_iterated_frames(
    n_traj_frames: int,
    start: Optional[int],
    stop: Optional[int],
    step: int,
) -> int:
    s = slice(start, stop, step)
    return len(range(*s.indices(n_traj_frames)))


def generate_endpoint_features(
    universe: Any,
    config: Optional[EndpointFeatureConfig] = None,
    *,
    traj_id: str = "traj",
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    list[list[list[int]]],
    Optional[pd.DataFrame],
]:
    """Extract per-frame endpoint-site distances from *universe*.

    Returns
    -------
    features_df
        Per-frame distance features with ``traj_id``, ``frame``, ``time_ps``.
        When ``include_paper_d1`` is True, also includes ``paper_d1_*`` columns.
    sites_df
        Site map (one row per monomer site) for traceability.
    monomer_selections
        Resolved per-monomer MDAnalysis selection strings.
    stored_sites
        Per monomer → per site → MDA atom ids (for QC plots).
    d1_atoms_df
        Paper-d1 ring-neighbor map, or ``None`` when disabled.
    """
    from src.EndpointAnalyzer import EndpointAnalyzer, EndpointAnalyzerObserver, EndpointsFinder
    from src.TrajectoryIterator import TrajectoryIterator
    from src.utils.gsa_selections import GSAFeatureSelections, resolve_selections

    cfg = config or EndpointFeatureConfig()
    n_jobs = cfg.n_jobs if cfg.n_jobs is not None else _default_n_jobs()

    explicit = None
    if cfg.monomer_selections:
        explicit = GSAFeatureSelections(monomer_selections=list(cfg.monomer_selections))
    sels = resolve_selections(
        universe,
        explicit=explicit,
        gsa_resname=cfg.gsa_resname,
        n_monomers=cfg.n_monomers,
        auto_tooth=False,
    )
    monomer_sels = sels.monomer_selections
    if not monomer_sels or len(monomer_sels) < 2:
        raise ValueError(
            "Need at least two monomer selections for endpoint-site distances."
        )

    finder = EndpointsFinder(
        ring_min_gap_deg=cfg.ring_min_gap_deg,
        ring_max_per_ring=cfg.ring_max_per_ring,
        step_back_from_terminals=cfg.step_back_from_terminals,
        extend_to_ring_atoms=True,
    )

    n_traj = len(universe.trajectory)
    n_rows = _count_iterated_frames(n_traj, cfg.start, cfg.stop, cfg.step)

    # Pre-resolve sites so paper-d1 can ride the same iterator pass.
    preview_sites: list[list[list[int]]] = []
    for sel_str in monomer_sels:
        if cfg.use_ring_centroids:
            _, sites = EndpointAnalyzer.find_residue_endpoint_sites(
                universe, sel_str, finder
            )
        else:
            _, ep_indices = EndpointAnalyzer.find_residue_endpoints(
                universe, sel_str, finder
            )
            sites = [[aid] for aid in ep_indices]
        preview_sites.append(sites)

    d1_atoms_df: Optional[pd.DataFrame] = None
    d1_observer: Optional[PaperD1FrameObserver] = None
    murata_observer: Any = None
    if cfg.include_paper_d1:
        d1_atoms_df = resolve_paper_d1_atoms(
            universe,
            monomer_sels,
            preview_sites,
            traj_id=traj_id,
            s3_site=cfg.paper_d1_s3_site,
            s7_site=cfg.paper_d1_s7_site,
            roles=cfg.paper_d1_roles,
            finder=finder,
        )
        atom_idxs, pair_specs = _paper_d1_pair_layout(
            d1_atoms_df,
            universe,
            s3_site=cfg.paper_d1_s3_site,
            s7_site=cfg.paper_d1_s7_site,
            roles=cfg.paper_d1_roles,
        )
        d1_observer = PaperD1FrameObserver(
            atom_idxs,
            pair_specs,
            n_frame_rows=n_rows,
            open_lo=cfg.paper_d1_open_lo,
            open_hi=cfg.paper_d1_open_hi,
        )

    if cfg.include_murata_criteria:
        from src.ChangepointAnalysis.murata_criteria import (
            MurataCriteriaFrameObserver,
            resolve_murata_criteria_atoms,
            snapshot_assembly_rmsd_ref,
            snapshot_motif_rmsd_refs,
        )

        try:
            murata_roles = resolve_murata_criteria_atoms(universe, monomer_sels)
            rmsd_idx, rmsd_ref = snapshot_assembly_rmsd_ref(
                universe,
                selection=cfg.murata_rmsd_selection,
                ref_frame=int(cfg.murata_rmsd_ref_frame),
            )
            motif_refs = snapshot_motif_rmsd_refs(
                universe,
                murata_roles,
                ref_frame=int(cfg.murata_rmsd_ref_frame),
            )
            murata_observer = MurataCriteriaFrameObserver(
                murata_roles,
                n_frame_rows=n_rows,
                rmsd_atom_indices=rmsd_idx,
                rmsd_ref_coords=rmsd_ref,
                rmsd_by_motif=motif_refs,
                cation_pi_open_lo=cfg.murata_cation_pi_open_lo,
            )
        except Exception:
            murata_observer = None

    observer = EndpointAnalyzerObserver(
        residue_sel_list=list(monomer_sels),
        endpoints_finder=finder,
        n_frame_rows=n_rows,
        use_ring_centroids=cfg.use_ring_centroids,
    )
    iterator = TrajectoryIterator(universe, use_dask=bool(cfg.use_dask))
    iterator.subscribe(observer)
    if d1_observer is not None:
        iterator.subscribe(d1_observer)
    if murata_observer is not None:
        iterator.subscribe(murata_observer)
    iterator.iterate(
        start=cfg.start,
        stop=cfg.stop,
        step=cfg.step,
        n_jobs=n_jobs,
        max_workers_for_io=cfg.max_workers_for_io,
    )

    dist_info = observer.get_endpoint_distances()
    features = all_pairs_to_metrics_df(
        dist_info["all_pairs"],
        dist_info["n_residues"],
        dist_info["n_frames"],
        summaries=cfg.pair_summaries,
        include_site_pairs=cfg.include_site_pairs,
    )

    # Map iterated frame indices to trajectory frame numbers / time
    frame_indices = list(range(*slice(cfg.start, cfg.stop, cfg.step).indices(n_traj)))
    if len(frame_indices) != len(features):
        # Fall back to sequential numbering if lengths diverge
        frame_indices = list(range(len(features)))
    features.insert(0, "traj_id", traj_id)
    features["frame"] = frame_indices
    features["time_ps"] = np.asarray(frame_indices, dtype=float) * float(cfg.time_per_frame_ps)
    # Keep column order: traj_id, frame, time_ps, then the rest
    cols = ["traj_id", "frame", "time_ps"] + [
        c for c in features.columns if c not in ("traj_id", "frame", "time_ps")
    ]
    features = features[cols]

    sites_df = _build_sites_dataframe(
        universe,
        monomer_sels,
        finder,
        traj_id=traj_id,
        use_ring_centroids=cfg.use_ring_centroids,
        stored_sites=observer.stored_site_indices,
    )

    if cfg.include_paper_d1 and d1_atoms_df is not None and d1_observer is not None:
        # Annotate site map with paper-d1 ring neighbor for s3/s7 rows.
        sites_df = sites_df.copy()
        sites_df["d1_ring_atom_id"] = ""
        sites_df["d1_endpoint_atom_id"] = ""
        for _, drow in d1_atoms_df.iterrows():
            mask = (
                (sites_df["monomer"] == int(drow["monomer"]))
                & (sites_df["site_index"] == int(drow["endpoint_site"]))
            )
            sites_df.loc[mask, "d1_ring_atom_id"] = str(int(drow["d1_ring_atom_id"]))
            sites_df.loc[mask, "d1_endpoint_atom_id"] = str(
                int(drow["endpoint_atom_id"])
            )

        d1_feat = d1_observer.to_dataframe()
        if len(d1_feat) != len(features):
            raise RuntimeError(
                f"paper d1 rows ({len(d1_feat)}) != feature rows ({len(features)})"
            )
        features = pd.concat(
            [features.reset_index(drop=True), d1_feat.reset_index(drop=True)],
            axis=1,
        )

    if murata_observer is not None:
        from src.ChangepointAnalysis.murata_criteria import (
            attach_murata_metastructure_labels,
            cation_pi_pair_columns,
            lock_cation_pi_units,
            lock_equatorial_d1,
        )

        murata_feat = murata_observer.to_dataframe()
        if len(murata_feat) != len(features):
            raise RuntimeError(
                f"murata criteria rows ({len(murata_feat)}) != "
                f"feature rows ({len(features)})"
            )
        features = pd.concat(
            [features.reset_index(drop=True), murata_feat.reset_index(drop=True)],
            axis=1,
        )
        n_mon = len(monomer_sels)
        features, _ = lock_cation_pi_units(
            features,
            cation_pi_pair_columns(n_mon),
            n_monomers=n_mon,
            open_lo=cfg.murata_cation_pi_open_lo,
        )
        features, _ = lock_equatorial_d1(features, n_monomers=n_mon)
        features = attach_murata_metastructure_labels(features)

    return (
        features,
        sites_df,
        list(monomer_sels),
        list(observer.stored_site_indices),
        d1_atoms_df,
    )


def _build_sites_dataframe(
    universe: Any,
    monomer_sels: list[str],
    finder: Any,
    *,
    traj_id: str,
    use_ring_centroids: bool,
    stored_sites: list[list[list[int]]],
) -> pd.DataFrame:
    """Build a site-map table from stored MDA atom-id groups."""
    rows: list[dict[str, Any]] = []
    from src.EndpointAnalyzer.gsa_site_map import (
        plot_label_for_role,
        try_classify_gsa_monomer,
    )

    for mon_idx, sites in enumerate(stored_sites):
        roles: list[str] = [""] * len(sites)
        sel_str = monomer_sels[mon_idx] if mon_idx < len(monomer_sels) else ""
        if sel_str:
            try:
                sel = universe.select_atoms(sel_str)
                mol = sel.convert_to("RDKIT") if len(sel) else None
                gsa = try_classify_gsa_monomer(mol) if mol is not None else None
                if gsa is not None and len(gsa.sites) == len(sites):
                    roles = gsa.roles()
            except Exception:
                roles = [""] * len(sites)
        for site_idx, atom_ids in enumerate(sites):
            kind = "ring" if (use_ring_centroids and len(atom_ids) > 1) else "atom"
            role = roles[site_idx] if site_idx < len(roles) else ""
            rows.append(
                {
                    "traj_id": traj_id,
                    "monomer": mon_idx,
                    "monomer_selection": sel_str,
                    "site_index": site_idx,
                    "role": role,
                    "label": plot_label_for_role(role, kind) if role else "",
                    "kind": kind,
                    "n_atoms": len(atom_ids),
                    "atom_ids": " ".join(str(a) for a in atom_ids),
                }
            )
    return pd.DataFrame(rows)


def _mda_id_to_local_index(sel: Any, atom_id: int) -> int:
    """Map MDA atom id → local index within *sel* AtomGroup."""
    ids = sel.ids
    hits = np.flatnonzero(ids == int(atom_id))
    if hits.size == 0:
        raise ValueError(f"Atom id {atom_id} not found in selection {sel}")
    return int(hits[0])


def resolve_bonded_ring_neighbor_mda_id(
    mol: Any,
    sel: Any,
    endpoint_atom_mda_id: int,
) -> int:
    """Return the MDA id of the unique ring atom bonded to *endpoint_atom_mda_id*.

    Raises
    ------
    ValueError
        If the endpoint atom is missing, has no bonded ring neighbor, or has
        more than one bonded ring neighbor (ambiguous).
    """
    local = _mda_id_to_local_index(sel, endpoint_atom_mda_id)
    atom = mol.GetAtomWithIdx(local)
    ring_nbrs = [nbr for nbr in atom.GetNeighbors() if nbr.IsInRing()]
    if not ring_nbrs:
        raise ValueError(
            f"Endpoint atom id={endpoint_atom_mda_id} has no bonded ring neighbor"
        )
    if len(ring_nbrs) > 1:
        raise ValueError(
            f"Endpoint atom id={endpoint_atom_mda_id} has {len(ring_nbrs)} bonded "
            f"ring neighbors {[int(sel[n.GetIdx()].id) for n in ring_nbrs]}; "
            "paper d1 requires a unique one-step-in ring atom"
        )
    return int(sel[ring_nbrs[0].GetIdx()].id)


def _site_mda_to_rdkit_locals(sel: Any, site_mda_ids: Sequence[int]) -> list[int]:
    return [_mda_id_to_local_index(sel, int(aid)) for aid in site_mda_ids]


def _exocyclic_tip_candidates(
    mol: Any, site_locals: set[int]
) -> list[tuple[int, int]]:
    """Return ``(tip_local, ring_attachment_local)`` for exocyclic C tips on a ring site."""
    pairs: list[tuple[int, int]] = []
    for r in site_locals:
        ring_atom = mol.GetAtomWithIdx(int(r))
        if not ring_atom.IsInRing():
            continue
        for nbr in ring_atom.GetNeighbors():
            nidx = int(nbr.GetIdx())
            if nidx in site_locals:
                continue
            if nbr.GetSymbol() == "C" and not nbr.IsInRing():
                pairs.append((nidx, int(r)))
    return pairs


def _resolve_paper_d1_endpoint_pair(
    mol: Any,
    sel: Any,
    site_mda_ids: Sequence[int],
    *,
    finder: Any | None = None,
) -> tuple[int, int, str]:
    """Return ``(endpoint_mda_id, d1_ring_mda_id, endpoint_kind)`` for one s3/s7 site."""
    if len(site_mda_ids) == 1:
        endpoint_id = int(site_mda_ids[0])
        ring_id = resolve_bonded_ring_neighbor_mda_id(mol, sel, endpoint_id)
        return endpoint_id, ring_id, "atom"

    site_locals = set(_site_mda_to_rdkit_locals(sel, site_mda_ids))
    exo = _exocyclic_tip_candidates(mol, site_locals)
    if exo:
        chosen = exo
        if len(exo) > 1 and finder is not None:
            _, cands = finder._candidate_endpoints(mol)
            cand_set = set(cands)
            filtered = [(t, r) for t, r in exo if t in cand_set or r in cand_set]
            if len(filtered) == 1:
                chosen = filtered
            elif filtered:
                chosen = filtered
        if len(chosen) > 1:
            from rdkit.Chem import GetDistanceMatrix

            D = GetDistanceMatrix(mol)
            tip_local, ring_local = max(chosen, key=lambda tr: float(D[tr[0]].max()))
        else:
            tip_local, ring_local = chosen[0]
        return int(sel[tip_local].id), int(sel[ring_local].id), "exocyclic_tip"

    # Ring-only site (e.g. BHHpM without methyl): endpoint hull landed on the ring.
    from src.EndpointAnalyzer import EndpointsFinder

    ep_finder = finder or EndpointsFinder(step_back_from_terminals=True)
    _, cands = ep_finder._candidate_endpoints(mol)
    in_site = [c for c in cands if c in site_locals]
    if not in_site:
        for r in site_locals:
            atom = mol.GetAtomWithIdx(int(r))
            if atom.IsInRing() and any(n.GetSymbol() == "H" for n in atom.GetNeighbors()):
                in_site.append(int(r))
    if not in_site:
        raise ValueError(
            f"Cannot resolve paper-d1 endpoint on ring site with "
            f"{len(site_mda_ids)} atoms {list(site_mda_ids)}"
        )
    if len(in_site) > 1:
        from rdkit.Chem import GetDistanceMatrix

        D = GetDistanceMatrix(mol)
        ep_local = max(in_site, key=lambda i: float(D[i].max()))
    else:
        ep_local = in_site[0]
    endpoint_id = int(sel[ep_local].id)
    atom = mol.GetAtomWithIdx(int(ep_local))
    if atom.IsInRing():
        # No exocyclic tip (e.g. BHHpM): the tooth / hull ring carbon *is*
        # the paper-d1 atom — do not step further into the ring.
        return endpoint_id, endpoint_id, "ring_site"
    ring_id = resolve_bonded_ring_neighbor_mda_id(mol, sel, endpoint_id)
    return endpoint_id, ring_id, "ring_site"


def resolve_paper_d1_atoms(
    universe: Any,
    monomer_selections: Sequence[str],
    stored_sites: Sequence[Sequence[Sequence[int]]],
    *,
    traj_id: str = "traj",
    s3_site: int = 3,
    s7_site: int = 7,
    roles: Optional[tuple[str, str]] = ("r2", "r3"),
    finder: Any | None = None,
) -> pd.DataFrame:
    """Map each monomer's equatorial R2/R3 ipso carbons (Murata C2–C3).

    Prefers the canonical GSA site map. Falls back to hull indices ``s3``/``s7``
    only when a monomer is not a hexaaryl GSA.
    """
    from src.EndpointAnalyzer.gsa_site_map import (
        CANONICAL_ROLES,
        D1_ROLES,
        try_classify_gsa_monomer,
    )

    if len(monomer_selections) != len(stored_sites):
        raise ValueError(
            f"monomer_selections ({len(monomer_selections)}) != "
            f"stored_sites ({len(stored_sites)})"
        )

    d1_roles = tuple(roles) if roles else D1_ROLES
    rows: list[dict[str, Any]] = []
    for mon_idx, (mon_sel, sites) in enumerate(zip(monomer_selections, stored_sites)):
        sel = universe.select_atoms(mon_sel)
        if len(sel) == 0:
            raise ValueError(f"Empty monomer selection for monomer {mon_idx}: {mon_sel}")
        mol = sel.convert_to("RDKIT")
        if mol is None:
            raise ValueError(f"RDKit conversion failed for monomer {mon_idx}: {mon_sel}")

        gsa = try_classify_gsa_monomer(mol)
        if gsa is not None:
            for role in d1_roles:
                site = gsa.by_role(str(role))
                if site.ipso_rdkit is None or site.subst_rdkit is None:
                    raise ValueError(
                        f"Monomer {mon_idx} role {role} is missing ipso/substituent"
                    )
                endpoint_id = int(sel[int(site.subst_rdkit)].id)
                ring_id = int(sel[int(site.ipso_rdkit)].id)
                rows.append(
                    {
                        "traj_id": traj_id,
                        "monomer": mon_idx,
                        "monomer_selection": mon_sel,
                        "endpoint_site": CANONICAL_ROLES.index(str(role)),
                        "role": str(role),
                        "endpoint_kind": "methyl" if site.methyl else "hydrogen",
                        "endpoint_atom_id": endpoint_id,
                        "d1_ring_atom_id": ring_id,
                        "d1_label": f"m{mon_idx}{role}_ring",
                        "n_site_atoms": 1,
                    }
                )
            continue

        for endpoint_site in (int(s3_site), int(s7_site)):
            if endpoint_site >= len(sites):
                raise ValueError(
                    f"Monomer {mon_idx} has {len(sites)} sites; "
                    f"paper d1 needs site index {endpoint_site}"
                )
            site_atoms = list(sites[endpoint_site])
            endpoint_id, ring_id, kind = _resolve_paper_d1_endpoint_pair(
                mol, sel, site_atoms, finder=finder
            )
            rows.append(
                {
                    "traj_id": traj_id,
                    "monomer": mon_idx,
                    "monomer_selection": mon_sel,
                    "endpoint_site": endpoint_site,
                    "role": "",
                    "endpoint_kind": kind,
                    "endpoint_atom_id": endpoint_id,
                    "d1_ring_atom_id": ring_id,
                    "d1_label": f"m{mon_idx}s{endpoint_site}_ring",
                    "n_site_atoms": len(site_atoms),
                }
            )
    return pd.DataFrame(rows)


def _paper_d1_token(site: Any) -> str:
    if isinstance(site, str) and not str(site).isdigit():
        return str(site)
    return f"s{int(site)}"


def paper_d1_column_name(mon_i: int, site_i: Any, mon_j: int, site_j: Any) -> str:
    """``paper_d1_m{i}r2_m{j}r3`` (canonical) or ``paper_d1_m{i}s{a}_m{j}s{b}``."""
    return (
        f"{PAPER_D1_COL_PREFIX}m{int(mon_i)}{_paper_d1_token(site_i)}"
        f"_m{int(mon_j)}{_paper_d1_token(site_j)}"
    )


def list_paper_d1_columns(columns: Sequence[str]) -> list[str]:
    """Return directional paper-d1 pair columns (exclude summary aggregates)."""
    out = []
    for c in columns:
        if not str(c).startswith(PAPER_D1_COL_PREFIX):
            continue
        if c in ("paper_d1_min", "paper_d1_n_open", "paper_d1_n_closed", "paper_d1_n_elongated"):
            continue
        if "_m" in c[len(PAPER_D1_COL_PREFIX) :]:
            out.append(str(c))
    return out


def classify_paper_d1_state(
    distance: float,
    *,
    open_lo: float = 4.5,
    open_hi: float = 5.5,
) -> str:
    """Classify a d1 distance as closed / open / elongated."""
    if not np.isfinite(distance):
        return "unknown"
    if distance < open_lo:
        return "closed"
    if distance <= open_hi:
        return "open"
    return "elongated"


def compute_paper_d1_distances(
    universe: Any,
    d1_atoms_df: pd.DataFrame,
    *,
    frame_indices: Sequence[int],
    s3_site: int = 3,
    s7_site: int = 7,
    roles: Optional[tuple[str, str]] = None,
    open_lo: float = 4.5,
    open_hi: float = 5.5,
) -> pd.DataFrame:
    """Compute per-frame directional paper-d1 distances for the given frames.

    Canonical GSA columns are ``paper_d1_m{i}r2_m{j}r3`` / ``…r3…r2``.
    Hull fallback still emits ``paper_d1_m{i}s3_m{j}s7``.
    """
    if d1_atoms_df.empty:
        raise ValueError("d1_atoms_df is empty")

    atom_idxs, pair_specs = _paper_d1_pair_layout(
        d1_atoms_df, universe, s3_site=s3_site, s7_site=s7_site, roles=roles
    )

    n_frames = len(frame_indices)
    n_slots = len(atom_idxs)
    coords = np.empty((n_frames, n_slots, 3), dtype=float)
    # Prefer AtomGroup.positions when available (MDAnalysis); fall back to
    # per-atom .position for lightweight test doubles.
    ag = None
    try:
        candidate = universe.atoms[atom_idxs]
        if hasattr(candidate, "positions"):
            ag = candidate
    except Exception:
        ag = None
    for fi, frame in enumerate(frame_indices):
        universe.trajectory[int(frame)]
        if ag is not None:
            coords[fi] = np.asarray(ag.positions, dtype=float)
        else:
            for k, atom_idx in enumerate(atom_idxs):
                coords[fi, k] = np.asarray(
                    universe.atoms[int(atom_idx)].position, dtype=float
                )

    # (n_frames, n_pairs) — vectorized over frames after the I/O sweep
    diffs = np.stack(
        [coords[:, ia, :] - coords[:, ib, :] for _, ia, ib in pair_specs],
        axis=1,
    )
    dists = np.linalg.norm(diffs, axis=2)

    rows: list[dict[str, Any]] = []
    for fi, frame in enumerate(frame_indices):
        row: dict[str, Any] = {"frame": int(frame)}
        frame_d = dists[fi]
        for k, (col, _, _) in enumerate(pair_specs):
            row[col] = float(frame_d[k])
        row["paper_d1_min"] = float(np.min(frame_d)) if frame_d.size else np.nan
        row["paper_d1_n_open"] = int(np.sum((frame_d >= open_lo) & (frame_d <= open_hi)))
        row["paper_d1_n_closed"] = int(np.sum(frame_d < open_lo))
        row["paper_d1_n_elongated"] = int(np.sum(frame_d > open_hi))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_paper_d1_by_segments(
    features_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    *,
    open_lo: float = 4.5,
    open_hi: float = 5.5,
    group: str = "endpoint",
) -> pd.DataFrame:
    """Per-segment mean d1 for each directional pair + open-window counts."""
    d1_cols = list_paper_d1_columns(features_df.columns)
    if not d1_cols or "frame" not in features_df.columns:
        return pd.DataFrame()

    segs = segments_df
    if "group" in segs.columns:
        segs = segs[segs["group"] == group]
    if segs.empty:
        return pd.DataFrame()

    out_rows: list[dict[str, Any]] = []
    for _, seg in segs.iterrows():
        traj = str(seg["traj_id"])
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        feat = features_df
        if "traj_id" in feat.columns:
            feat = feat[feat["traj_id"] == traj]
        mask = (feat["frame"] >= start) & (feat["frame"] <= end)
        sub = feat.loc[mask]
        base = {
            "traj_id": traj,
            "group": group,
            "segment_id": int(seg.get("segment_id", -1)),
            "cluster_label": int(seg["cluster_label"])
            if "cluster_label" in seg and pd.notna(seg["cluster_label"])
            else np.nan,
            "start_frame": start,
            "end_frame": end,
            "n_frames_feat": int(len(sub)),
        }
        if sub.empty:
            continue
        # Assembly summaries
        if "paper_d1_min" in sub.columns:
            base["paper_d1_min_mean"] = float(sub["paper_d1_min"].mean())
        if "paper_d1_n_open" in sub.columns:
            base["paper_d1_n_open_mean"] = float(sub["paper_d1_n_open"].mean())

        open_pairs: list[str] = []
        closed_pairs: list[str] = []
        elongated_pairs: list[str] = []
        for col in d1_cols:
            mean_d = float(sub[col].mean())
            state = classify_paper_d1_state(mean_d, open_lo=open_lo, open_hi=open_hi)
            base[f"{col}_mean"] = mean_d
            base[f"{col}_state"] = state
            if state == "open":
                open_pairs.append(col)
            elif state == "closed":
                closed_pairs.append(col)
            elif state == "elongated":
                elongated_pairs.append(col)
        base["open_pairs"] = ";".join(open_pairs)
        base["closed_pairs"] = ";".join(closed_pairs)
        base["n_open_pairs"] = len(open_pairs)
        base["n_closed_pairs"] = len(closed_pairs)
        base["n_elongated_pairs"] = len(elongated_pairs)
        out_rows.append(base)
    return pd.DataFrame(out_rows)


def _draw_endpoint_site_panel(
    ax: Any,
    universe: Any,
    mon_sel: str,
    sites: Sequence[Sequence[int]],
    *,
    panel_title: str,
) -> None:
    """Render one monomer 2D structure with ring (blue) and atom (orange) sites."""
    from rdkit.Chem import Draw

    from src.EndpointAnalyzer import EndpointsFinder

    from src.EndpointAnalyzer.gsa_site_map import plot_label_for_role, try_classify_gsa_monomer

    sel = universe.select_atoms(mon_sel)
    mol = sel.convert_to("RDKIT")
    if mol is None:
        ax.text(0.5, 0.5, f"RDKit failed\n{mon_sel}", ha="center", va="center")
        ax.axis("off")
        return

    ef = EndpointsFinder()
    m2d, xy = ef.to_2d_coords(mol)
    mda_to_rdkit = {int(sel.atoms[i].id): i for i in range(len(sel))}
    gsa = try_classify_gsa_monomer(mol)
    role_names = gsa.roles() if gsa is not None and len(gsa.sites) == len(sites) else []

    # Ring sites = multi-atom; atom sites = singletons
    ring_mda: list[int] = []
    atom_mda: list[int] = []
    site_labels: list[tuple[int, str, str]] = []  # rdkit_idx, label, color
    for site_idx, atom_ids in enumerate(sites):
        ids = [int(a) for a in atom_ids]
        is_ring = len(ids) > 1
        color = "steelblue" if is_ring else "darkorange"
        kind = "R" if is_ring else "A"
        if is_ring:
            ring_mda.extend(ids)
        else:
            atom_mda.extend(ids)
        role = role_names[site_idx] if site_idx < len(role_names) else ""
        label = plot_label_for_role(role, kind) if role else f"s{site_idx}:{kind}"
        # Label at the site representative (min id → centroid proxy atom)
        rep = min(ids) if ids else None
        if rep is not None and rep in mda_to_rdkit:
            site_labels.append((mda_to_rdkit[rep], label, color))

    ring_rdkit = [mda_to_rdkit[i] for i in ring_mda if i in mda_to_rdkit]
    atom_rdkit = [mda_to_rdkit[i] for i in atom_mda if i in mda_to_rdkit]

    highlight_colors: dict[int, tuple[float, float, float]] = {}
    for idx in ring_rdkit:
        highlight_colors[idx] = (0.2, 0.45, 0.85)  # blue — ring system
    # Atom sites overwrite ring color so R=H para carbons (R1 on R1-ring) stay orange.
    for idx in atom_rdkit:
        highlight_colors[idx] = (1.0, 0.5, 0.0)  # orange — atom site
    all_highlight = list(highlight_colors.keys())

    img = None
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        import io

        from PIL import Image

        drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
        drawer.DrawMolecule(
            m2d,
            highlightAtoms=all_highlight,
            highlightAtomColors=highlight_colors,
        )
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    except Exception:
        img = Draw.MolToImage(
            m2d,
            size=(800, 600),
            highlightAtoms=ring_rdkit or all_highlight,
            highlightColor=(0.2, 0.45, 0.85),
        )
        if atom_rdkit:
            from PIL import Image

            base = np.array(img.convert("RGBA"))
            overlay = np.array(
                Draw.MolToImage(
                    m2d,
                    size=(800, 600),
                    highlightAtoms=atom_rdkit,
                    highlightColor=(1.0, 0.5, 0.0),
                ).convert("RGBA")
            )
            mask = overlay[:, :, 3] > 0
            base[mask] = overlay[mask]
            img = Image.fromarray(base)

    ax.imshow(img, extent=[0, 800, 600, 0])
    ax.axis("off")
    n_ring = sum(1 for s in sites if len(s) > 1)
    n_atom = len(sites) - n_ring
    ax.set_title(
        f"{panel_title}  ({n_ring} ring, {n_atom} atom)",
        fontsize=11,
        fontweight="bold",
    )

    if len(xy) == 0:
        return

    # Ring labels first, atom labels last so orange R1 sits on top of R1-ring.
    site_labels.sort(key=lambda t: 1 if t[2] == "darkorange" else 0)
    x_coords = xy[:, 0]
    y_coords = xy[:, 1]
    x_min, x_max = float(x_coords.min()), float(x_coords.max())
    y_min, y_max = float(y_coords.min()), float(y_coords.max())
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    padding = 0.15
    scale = min(
        (800 * (1 - 2 * padding)) / x_range if x_range > 0 else 1.0,
        (600 * (1 - 2 * padding)) / y_range if y_range > 0 else 1.0,
    )
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    offset_x = 400 - center_x * scale
    offset_y = 300 + center_y * scale

    for rdkit_idx, label, color in site_labels:
        if rdkit_idx >= len(xy):
            continue
        x, y = xy[rdkit_idx]
        ax.annotate(
            label,
            xy=(x * scale + offset_x, -y * scale + offset_y),
            fontsize=9,
            color=color,
            fontweight="bold",
            zorder=3 if color == "darkorange" else 2,
        )


def plot_endpoint_sites(
    universe: Any,
    monomer_selections: Sequence[str],
    stored_sites: Sequence[Sequence[Sequence[int]]],
    output_path: Path | str,
    *,
    dpi: int = 150,
    title: Optional[str] = None,
) -> Path:
    """Save a multi-panel PNG of endpoint sites per monomer for visual QC.

    Blue highlights = fused ring-system sites (centroid distances).
    Orange highlights = singleton atom sites (e.g. gear-tooth / ipso carbons).
    Labels ``R1`` / ``R2`` / ``R3`` / ``Ph`` / ``Py-eq`` / ``Py-pole`` mark
    Murata chemical roles (canonical GSA map). Hull fallback still uses
    ``s{k}:R`` / ``s{k}:A``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(monomer_selections)
    if n == 0:
        raise ValueError("No monomer selections provided for endpoint-site plot.")
    if len(stored_sites) != n:
        raise ValueError(
            f"stored_sites length ({len(stored_sites)}) != "
            f"monomer_selections length ({n})"
        )

    if title is None:
        title = (
            "Endpoint sites — canonical GSA roles "
            "(R1=pole, R2/R3=equator, Ph / Py+ rings)"
        )

    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.atleast_1d(axes).flatten())

    for idx, mon_sel in enumerate(monomer_selections):
        _draw_endpoint_site_panel(
            axes_flat[idx],
            universe,
            mon_sel,
            stored_sites[idx],
            panel_title=f"Monomer {idx}",
        )

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    legend_handles = [
        Patch(
            facecolor=(0.2, 0.45, 0.85),
            edgecolor="navy",
            label="Ring-system site (centroid)",
        ),
        Patch(
            facecolor=(1.0, 0.5, 0.0),
            edgecolor="darkorange",
            label="Atom site (tooth / tip)",
        ),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_endpoint_features_csv(
    features_df: pd.DataFrame,
    out_dir: Path | str,
    traj_id: str,
    *,
    sites_df: Optional[pd.DataFrame] = None,
    d1_atoms_df: Optional[pd.DataFrame] = None,
    universe: Any = None,
    monomer_selections: Optional[Sequence[str]] = None,
    stored_sites: Optional[Sequence[Sequence[Sequence[int]]]] = None,
    write_site_plot: bool = False,
    write_shared_maps: bool = True,
    write_features: bool = True,
) -> dict[str, Path]:
    """Write ``<traj_id>_endpoint_features.csv`` and optionally sites map + QC PNG.

    The site QC plot is topology-only (one ``endpoint_sites.png`` for the
    shared prmtop). Pass ``write_site_plot=True`` once; later calls skip if
    the file already exists.

    When extracting trajectories in parallel, set ``write_shared_maps=False``
    in workers and write ``endpoint_sites.csv`` / ``paper_d1_atoms.csv`` once
    from the parent process to avoid empty-file races.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    if write_features:
        feat_path = out_dir / f"{traj_id}_endpoint_features.csv"
        features_df.to_csv(feat_path, index=False)
        written["features"] = feat_path

    if write_shared_maps and sites_df is not None and not sites_df.empty:
        sites_path = out_dir / "endpoint_sites.csv"
        _merge_traj_table_csv(sites_path, sites_df, traj_id=traj_id)
        written["sites"] = sites_path

    if write_shared_maps and d1_atoms_df is not None and not d1_atoms_df.empty:
        d1_path = out_dir / "paper_d1_atoms.csv"
        _merge_traj_table_csv(d1_path, d1_atoms_df, traj_id=traj_id)
        written["paper_d1_atoms"] = d1_path

    if (
        write_site_plot
        and universe is not None
        and monomer_selections is not None
        and stored_sites is not None
        and len(stored_sites) > 0
    ):
        cohort_plot = out_dir / "endpoint_sites.png"
        if not cohort_plot.exists():
            plot_endpoint_sites(
                universe,
                monomer_selections,
                stored_sites,
                cohort_plot,
            )
        written["sites_plot"] = cohort_plot

    return written


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Read a CSV, treating missing/empty/corrupt partial writes as empty."""
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def _merge_traj_table_csv(
    path: Path,
    new_df: pd.DataFrame,
    *,
    traj_id: str,
) -> None:
    """Append/replace rows for ``traj_id`` with an atomic rewrite."""
    existing = _read_csv_or_empty(path)
    if not existing.empty and "traj_id" in existing.columns:
        existing = existing[existing["traj_id"] != traj_id]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    combined.to_csv(tmp, index=False)
    tmp.replace(path)
