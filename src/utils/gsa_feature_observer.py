"""
GSAFeatureObserver — FrameObserver producing a per-frame feature DataFrame.

Subscribes to ``TrajectoryIterator``, calls the stateless functions in
``gsa_features`` on every frame, and collects rows into a DataFrame whose
columns match ``GSA_features_table.txt`` + ``guest_GSA_features_table.txt``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np

from src.Aggregator import ResultsGroup
from src.TrajectoryIterator import FrameObserver, TrajectoryIterator

from . import gsa_features as F
from .gsa_selections import GSAFeatureSelections, resolve_selections


class GSAFeatureObserver(FrameObserver):
    """Collect Tier-1, Tier-2, and guest GSA features for every visited frame.

    Parameters
    ----------
    selections : GSAFeatureSelections, optional
        Pre-filled (or partial) selections.  Missing fields are resolved via
        ``resolve_selections`` on the first call to ``on_frame_start``.
    gsa_resname : str
        Passed to ``resolve_selections`` if auto-derivation is needed.
    n_monomers : int
        Expected number of monomers (default 6).
    include_tier1 : bool
        Compute Tier-1 features (default True).
    include_tier2 : bool
        Compute Tier-2 gear/interlocking + interface chemistry (default True).
    include_guest : bool
        Compute guest-related features (default True).
    ref_frame : int
        Frame index used as RMSD reference for assembly and monomers.
    contact_cutoff : float
        Inter-monomer atom-pair contact cutoff (angstrom).
    cavity_radius : float
        Radius around assembly center for cavity solvent counts.
    inner_radius, outer_radius, surface_shell : float
        Radial shell boundaries for guest location classification.
    traj_id : str
        Label written into the ``traj_id`` column of every row.
    out_prefix : str, optional
        If given, ``on_frame_end`` writes a CSV to ``{out_prefix}_gsa_features.csv``.
    """

    def __init__(
        self,
        selections: Optional[GSAFeatureSelections] = None,
        gsa_resname: str = "MOL",
        n_monomers: int = 6,
        *,
        include_tier1: bool = True,
        include_tier2: bool = True,
        include_guest: bool = True,
        ref_frame: int = 0,
        contact_cutoff: float = 4.5,
        cavity_radius: float = 8.0,
        inner_radius: float = 5.0,
        outer_radius: float = 12.0,
        surface_shell: float = 3.0,
        traj_id: str = "",
        out_prefix: Optional[str] = None,
        auto_tooth: bool = True,
    ):
        super().__init__()
        self._raw_selections = selections
        self._gsa_resname = gsa_resname
        self._n_monomers = n_monomers
        self._auto_tooth = auto_tooth
        self.include_tier1 = include_tier1
        self.include_tier2 = include_tier2
        self.include_guest = include_guest
        self._ref_frame = ref_frame
        self._contact_cutoff = contact_cutoff
        self._cavity_radius = cavity_radius
        self._inner_radius = inner_radius
        self._outer_radius = outer_radius
        self._surface_shell = surface_shell
        self._traj_id = traj_id
        self._out_prefix = out_prefix

        self.results["rows"] = []
        self.results["frame_call_count"] = 0
        self.results["frame_exception_count"] = 0

        # Populated in on_frame_start
        self.sel: Optional[GSAFeatureSelections] = None
        self._ag_cache: Dict[str, Any] = {}
        self._assembly_ref_coords: Optional[np.ndarray] = None
        self._monomer_ref_coords: Optional[List[np.ndarray]] = None
        self._prev_assembly_coords: Optional[np.ndarray] = None
        self._prev_axes: Optional[np.ndarray] = None
        self._initialized = False

    # ------------------------------------------------------------------
    #  Pickling support (parallel workers)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_ag_cache"] = {}
        state["_initialized"] = False
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    #  ResultsGroup aggregation
    # ------------------------------------------------------------------

    def _get_aggregator(self) -> ResultsGroup:
        return ResultsGroup(lookup={
            "rows": ResultsGroup.list_extend_sorted("frame"),
            "frame_call_count": ResultsGroup.sum_values,
            "frame_exception_count": ResultsGroup.sum_values,
        })

    # ------------------------------------------------------------------
    #  Observer lifecycle
    # ------------------------------------------------------------------

    def on_frame_start(self, iterator: TrajectoryIterator) -> None:
        u = iterator.universe
        self.sel = resolve_selections(
            u,
            explicit=self._raw_selections,
            gsa_resname=self._gsa_resname,
            n_monomers=self._n_monomers,
            auto_tooth=self._auto_tooth and self.include_tier2,
        )
        self._build_ag_cache(u)

        # Reference coordinates for RMSD
        ref = self._ref_frame
        if ref < 0 or ref >= len(u.trajectory):
            ref = 0
        u.trajectory[ref]
        self._assembly_ref_coords = self._get_positions("assembly").copy()
        self._monomer_ref_coords = [
            self._get_monomer_positions(u, i).copy()
            for i in range(len(self.sel.monomer_selections))
        ]

        self._prev_assembly_coords = None
        self._prev_axes = None
        self.results["rows"] = []
        self.results["frame_call_count"] = 0
        self.results["frame_exception_count"] = 0
        self._initialized = True

    def on_frame(
        self,
        ts: Any,
        frame_idx: int,
        universe: Any,
    ) -> None:
        if not self._initialized:
            self._lazy_init(universe)

        self.results["frame_call_count"] += 1

        try:
            row = self._compute_row(ts, frame_idx, universe)
            self.results["rows"].append(row)
        except Exception as e:
            self.results["frame_exception_count"] += 1
            warnings.warn(
                f"GSAFeatureObserver frame {getattr(ts, 'frame', frame_idx)}: {e}"
            )
            self.results["rows"].append({
                "traj_id": self._traj_id,
                "frame": getattr(ts, "frame", frame_idx),
                "time_ps": getattr(ts, "time", float("nan")),
            })

    def on_frame_end(self, iterator: TrajectoryIterator) -> None:
        if self._out_prefix:
            import pandas as pd
            df = self.get_features_df()
            path = f"{self._out_prefix}_gsa_features.csv"
            df.to_csv(path, index=False)

    # ------------------------------------------------------------------
    #  Public helpers
    # ------------------------------------------------------------------

    def get_features_df(self):
        """Return collected rows as a DataFrame."""
        import pandas as pd
        rows = self.results.get("rows", [])
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_selections_needed(self) -> List[str]:
        if self.sel is None:
            return []
        out = [self.sel.assembly_sel or ""]
        out.extend(self.sel.monomer_selections or [])
        out.extend(self.sel.tooth_selections or [])
        for attr in (
            "hydrophilic_sel", "hydrophobic_sel", "headgroup_sel", "tail_sel",
            "inner_atom_sel", "water_sel", "ion_sel", "guest_sel",
            "hbond_donor_sel", "hbond_acceptor_sel", "cation_sel", "anion_sel",
        ):
            v = getattr(self.sel, attr, None)
            if v:
                out.append(v)
        return [s for s in out if s]

    # ------------------------------------------------------------------
    #  Internal plumbing
    # ------------------------------------------------------------------

    def _lazy_init(self, universe: Any) -> None:
        """Re-initialise caches in a worker process after unpickling."""
        self.sel = resolve_selections(
            universe,
            explicit=self._raw_selections,
            gsa_resname=self._gsa_resname,
            n_monomers=self._n_monomers,
            auto_tooth=self._auto_tooth and self.include_tier2,
        )
        self._build_ag_cache(universe)

        ref = self._ref_frame
        if ref < 0 or ref >= len(universe.trajectory):
            ref = 0
        universe.trajectory[ref]
        self._assembly_ref_coords = self._get_positions("assembly").copy()
        self._monomer_ref_coords = [
            self._get_monomer_positions(universe, i).copy()
            for i in range(len(self.sel.monomer_selections))
        ]
        self._initialized = True

    def _build_ag_cache(self, universe: Any) -> None:
        self._ag_cache = {}
        sel = self.sel
        if sel is None:
            return
        _try_cache = [
            ("assembly", sel.assembly_sel),
            ("hydrophilic", sel.hydrophilic_sel),
            ("hydrophobic", sel.hydrophobic_sel),
            ("headgroup", sel.headgroup_sel),
            ("tail", sel.tail_sel),
            ("inner_atom", sel.inner_atom_sel),
            ("water", sel.water_sel),
            ("ion", sel.ion_sel),
            ("guest", sel.guest_sel),
            ("hbond_donor", sel.hbond_donor_sel),
            ("hbond_acceptor", sel.hbond_acceptor_sel),
            ("cation", sel.cation_sel),
            ("anion", sel.anion_sel),
        ]
        for key, sel_str in _try_cache:
            if sel_str:
                try:
                    self._ag_cache[key] = universe.select_atoms(sel_str)
                except Exception as e:
                    warnings.warn(f"Selection '{sel_str}' for '{key}' failed: {e}")

        if sel.monomer_selections:
            for i, ms in enumerate(sel.monomer_selections):
                try:
                    self._ag_cache[f"monomer_{i}"] = universe.select_atoms(ms)
                except Exception as e:
                    warnings.warn(f"Monomer selection {i} '{ms}' failed: {e}")

        if sel.tooth_selections:
            for i, ts in enumerate(sel.tooth_selections):
                try:
                    self._ag_cache[f"tooth_{i}"] = universe.select_atoms(ts)
                except Exception as e:
                    warnings.warn(f"Tooth selection {i} '{ts}' failed: {e}")

    def _get_tooth_coords_list(self, n_mon: int) -> Optional[List[np.ndarray]]:
        result: List[np.ndarray] = []
        any_nonempty = False
        for i in range(n_mon):
            ag = self._ag_cache.get(f"tooth_{i}")
            if ag is not None and len(ag) > 0:
                result.append(ag.positions)
                any_nonempty = True
            else:
                result.append(np.empty((0, 3)))
        return result if any_nonempty else None

    def _get_positions(self, key: str) -> np.ndarray:
        ag = self._ag_cache.get(key)
        if ag is not None and len(ag) > 0:
            return ag.positions
        return np.empty((0, 3), dtype=np.float64)

    def _get_monomer_positions(self, universe: Any, idx: int) -> np.ndarray:
        return self._get_positions(f"monomer_{idx}")

    def _compute_row(self, ts: Any, frame_idx: int, universe: Any) -> Dict[str, Any]:
        sel = self.sel
        n_mon = len(sel.monomer_selections) if sel.monomer_selections else 0

        row: Dict[str, Any] = {
            "traj_id": self._traj_id,
            "frame": getattr(ts, "frame", frame_idx),
            "time_ps": getattr(ts, "time", float("nan")),
        }

        assembly_pos = self._get_positions("assembly")
        assembly_center = assembly_pos.mean(axis=0) if len(assembly_pos) > 0 else np.zeros(3)

        monomer_coords = [self._get_monomer_positions(universe, i) for i in range(n_mon)]
        monomer_coms = F.compute_monomer_coms(monomer_coords) if monomer_coords else np.empty((0, 3))

        # ── Tier 1 ────────────────────────────────────────────────────
        if self.include_tier1:
            if len(assembly_pos) > 0:
                row.update(F.gyration_tensor_descriptors(assembly_pos))
                if self._assembly_ref_coords is not None:
                    row["assembly_rmsd_to_ref"] = F.assembly_rmsd_to_ref(
                        assembly_pos, self._assembly_ref_coords
                    )
                row["assembly_frame_to_frame_rmsd"] = F.assembly_frame_to_frame_rmsd(
                    assembly_pos, self._prev_assembly_coords
                )

            if len(monomer_coms) > 0:
                row.update(F.radial_distance_stats(monomer_coms, assembly_center))
                row.update(F.pairwise_com_stats(monomer_coms))
                row.update(F.neighbor_opposite_stats(monomer_coms))
                row["octahedrality_score"] = F.octahedrality_score(monomer_coms)

            # Orientation
            headgroup_coords = self._per_monomer_subgroup("headgroup", n_mon, universe)
            tail_coords = self._per_monomer_subgroup("tail", n_mon, universe)
            axes = F.amphiphile_axes(monomer_coords, headgroup_coords, tail_coords)
            row.update(F.amphiphile_axis_radial_cos_stats(axes, monomer_coms, assembly_center))
            row["monomer_twist_angle_std"] = F.monomer_twist_angle_std(axes, self._prev_axes)

            # Inter-monomer contacts
            row.update(F.inter_monomer_contact_features(
                monomer_coords, cutoff=self._contact_cutoff
            ))

            # Cavity / volume proxies
            water_pos = self._get_positions("water")
            ion_pos = self._get_positions("ion")
            row.update(F.cavity_solvent_counts(
                assembly_center, water_pos, ion_pos, self._cavity_radius
            ))
            inner_pos = self._get_positions("inner_atom")
            row.update(F.cavity_inner_atom_features(assembly_center, inner_pos))

            # Segregation
            phi_pos = self._get_positions("hydrophilic")
            pho_pos = self._get_positions("hydrophobic")
            hg_pos = self._get_positions("headgroup")
            row.update(F.segregation_features(
                assembly_center,
                phi_pos,
                pho_pos,
                headgroup_positions=hg_pos if len(hg_pos) > 0 else None,
                water_positions=water_pos if len(water_pos) > 0 else None,
            ))

            # Monomer deformation
            row.update(F.monomer_deformation_features(
                monomer_coords, self._monomer_ref_coords
            ))

            # Update previous-frame state
            self._prev_assembly_coords = assembly_pos.copy() if len(assembly_pos) > 0 else None
            self._prev_axes = axes.copy()

        # ── Tier 2 ────────────────────────────────────────────────────
        if self.include_tier2:
            tooth_coords = self._get_tooth_coords_list(n_mon)
            endpoint_by_pair = (
                F.all_monomer_endpoint_distance_matrices(tooth_coords)
                if tooth_coords is not None
                else None
            )
            row.update(F.gear_interlocking_features(
                monomer_coords,
                tooth_coords_list=tooth_coords,
                monomer_coms=monomer_coms,
                assembly_center=assembly_center,
                cutoff=self._contact_cutoff,
                endpoint_dist_by_pair=endpoint_by_pair,
            ))

            donor_coords = self._per_monomer_subgroup("hbond_donor", n_mon, universe)
            acceptor_coords = self._per_monomer_subgroup("hbond_acceptor", n_mon, universe)
            hydrophobic_m = self._per_monomer_subgroup("hydrophobic", n_mon, universe)
            polar_m = self._per_monomer_subgroup("hydrophilic", n_mon, universe)

            cation_pos = self._get_positions("cation")
            anion_pos = self._get_positions("anion")
            row.update(F.interface_chemistry_features(
                monomer_coords,
                donor_coords_list=donor_coords,
                acceptor_coords_list=acceptor_coords,
                hydrophobic_coords_list=hydrophobic_m,
                polar_coords_list=polar_m,
                cation_positions=cation_pos if len(cation_pos) > 0 else None,
                anion_positions=anion_pos if len(anion_pos) > 0 else None,
                water_positions=water_pos if len(water_pos) > 0 else None,
                ion_positions=ion_pos if len(ion_pos) > 0 else None,
            ))

            row.update(F.pore_opening_distances(monomer_coms))

        # ── Guest ─────────────────────────────────────────────────────
        if self.include_guest:
            guest_pos = self._get_positions("guest")
            if len(guest_pos) > 0:
                row.update(F.guest_location_features(
                    guest_pos, assembly_center,
                    self._inner_radius, self._outer_radius, self._surface_shell,
                ))
                row.update(F.guest_gsa_contact_features(
                    guest_pos, assembly_pos,
                    hydrophobic_positions=self._get_positions("hydrophobic"),
                    hydrophilic_positions=self._get_positions("hydrophilic"),
                ))
                row.update(F.guest_monomer_interface_features(
                    guest_pos, monomer_coords,
                ))
                row.update(F.guest_cavity_and_water_features(
                    guest_pos, assembly_center,
                    water_positions=water_pos if len(water_pos) > 0 else None,
                    cavity_radius=self._cavity_radius,
                ))
                row.update(F.guest_coupling_features(
                    guest_pos, monomer_coords,
                ))

        return row

    def _per_monomer_subgroup(
        self,
        group_key: str,
        n_mon: int,
        universe: Any,
    ) -> Optional[List[np.ndarray]]:
        """Intersect a global group selection with each monomer selection."""
        group_ag = self._ag_cache.get(group_key)
        if group_ag is None or len(group_ag) == 0:
            return None

        result = []
        group_ix = set(group_ag.indices)
        for i in range(n_mon):
            mon_ag = self._ag_cache.get(f"monomer_{i}")
            if mon_ag is None:
                result.append(np.empty((0, 3)))
                continue
            common_ix = group_ix & set(mon_ag.indices)
            if common_ix:
                mask = np.isin(mon_ag.indices, list(common_ix))
                result.append(mon_ag.positions[mask])
            else:
                result.append(np.empty((0, 3)))
        return result


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════

def compute_gsa_features(
    universe: Any,
    selections: Optional[GSAFeatureSelections] = None,
    gsa_resname: str = "MOL",
    n_monomers: int = 6,
    *,
    include_tier1: bool = True,
    include_tier2: bool = True,
    include_guest: bool = True,
    ref_frame: int = 0,
    contact_cutoff: float = 4.5,
    cavity_radius: float = 8.0,
    inner_radius: float = 5.0,
    outer_radius: float = 12.0,
    surface_shell: float = 3.0,
    traj_id: str = "",
    out_prefix: Optional[str] = None,
    auto_tooth: bool = True,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: Optional[int] = None,
    n_jobs: int = 1,
):
    """One-call feature extraction over a trajectory.

    Returns a ``pandas.DataFrame`` with one row per visited frame and columns
    matching the GSA feature tables.
    """
    import pandas as pd

    observer = GSAFeatureObserver(
        selections=selections,
        gsa_resname=gsa_resname,
        n_monomers=n_monomers,
        include_tier1=include_tier1,
        include_tier2=include_tier2,
        include_guest=include_guest,
        ref_frame=ref_frame,
        contact_cutoff=contact_cutoff,
        cavity_radius=cavity_radius,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        surface_shell=surface_shell,
        traj_id=traj_id,
        out_prefix=out_prefix,
        auto_tooth=auto_tooth,
    )
    iterator = TrajectoryIterator(universe)
    iterator.subscribe(observer)
    iterator.iterate(start=start, stop=stop, step=step, n_jobs=n_jobs)
    return observer.get_features_df()
