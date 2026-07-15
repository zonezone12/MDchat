"""
Imamura et al. nanocube MSM workflow utilities.

Reproduces the coarse-grained self-assembly analysis pipeline described in
Imamura, Yamamoto & Sato, *Chem. Phys. Lett.* **742**, 137135 (2020):

1. Truncate trajectories at nanocube formation.
2. Build 168-D feature vectors from sorted inter-bead distances
   (type-1 central benzenes → 15 distances; type-4 methyls → 153).
3. PCA or time-lagged ICA (tlICA) to five components.
4. Two-stage clustering: MiniBatchKMeans microstates → Ward macrostates.
5. Lagged transition counting for a kinetic network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import json

import numpy as np

from src.Aggregator import ResultsGroup
from src.TrajectoryIterator import FrameObserver, TrajectoryIterator
from src.utils import gsa_features as F
from src.utils.cluster_inspection import (
    plot_transition_heatmap,
    plot_transition_network_diagram,
    transition_matrix_dataframe,
    transition_probability_matrix,
)
from src.utils.gsa_selections import GSAFeatureSelections, resolve_selections
from src.utils.rdkit_utils import (
    RingCenterCalculator,
    all_methyl_atom_indices,
    filter_endpoints_to_methyl_atoms,
    select_type4_by_bond_distance,
    select_type4_by_centroid_distance,
)

# Imamura paper defaults (6-monomer nanocube, 18 methyl beads).
DEFAULT_N_TYPE1_BEADS = 6
DEFAULT_N_TYPE4_BEADS = 18
DEFAULT_N_TYPE1_DISTANCES = 15   # C(6, 2)
DEFAULT_N_TYPE4_DISTANCES = 153  # C(18, 2)
DEFAULT_FEATURE_DIM = 168
DEFAULT_PCA_COMPONENTS = 5
DEFAULT_N_MICROCLUSTERS = 1500
DEFAULT_N_MACROSTATES = 22
DEFAULT_TRANSITION_LAG_NS = 2.0
DEFAULT_DIMENSIONALITY_REDUCTION = "pca"
DEFAULT_TLICA_REGULARIZATION = 1e-6


@dataclass
class ImamuraTICAModel:
    """
  Time-lagged ICA (TICA) projection model with a PCA-compatible interface.

    ``components_`` has shape ``(n_components, n_features)`` on standardized
    features, matching :class:`sklearn.decomposition.PCA`.
    """

    components_: np.ndarray
    eigenvalues_: np.ndarray
    mean_: np.ndarray
    scale_: np.ndarray
    lag_frames: int
    lag_ns: float
    implied_timescales_ns: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_
        return Xs @ self.components_.T


@dataclass
class ImamuraMSMConfig:
    """Hyperparameters for the Imamura MSM pipeline."""

    n_type1_beads: int = DEFAULT_N_TYPE1_BEADS
    n_type4_beads: int = DEFAULT_N_TYPE4_BEADS
    n_pca_components: int = DEFAULT_PCA_COMPONENTS
    n_microclusters: int = DEFAULT_N_MICROCLUSTERS
    n_macrostates: int = DEFAULT_N_MACROSTATES
    transition_lag_ns: float = DEFAULT_TRANSITION_LAG_NS
    dimensionality_reduction: str = DEFAULT_DIMENSIONALITY_REDUCTION
    tlica_lag_ns: Optional[float] = None
    tlica_regularization: float = DEFAULT_TLICA_REGULARIZATION
    formation_min_active_interfaces: int = 15
    formation_sustain_frames: int = 1
    contact_cutoff: float = 4.5
    kmeans_batch_size: int = 1000
    kmeans_random_state: int = 0


@dataclass
class ImamuraBeadSpec:
    """
    Bead definitions for Imamura distance features.

    Type-1 beads are the geometric centers of each monomer's central benzene ring
    (one bead per monomer). Type-4 beads are methyl carbons or endpoint atoms
    depending on :attr:`type4_source`.
    """

    type1_ring_groups: Optional[List[Tuple[int, ...]]] = None
    type4_atom_ids: Optional[Tuple[int, ...]] = None
    type1_selection: Optional[str] = None
    type4_selection: Optional[str] = None
    type4_source: str = "methyl"
    endpoint_extend_ring: bool = True
    endpoint_exclude_center_benzene: bool = True
    type4_rank: str = "bond"

    @property
    def uses_ring_centroids(self) -> bool:
        return self.type1_ring_groups is not None

    def validate(self, *, n_type1: int, n_type4: int) -> None:
        if self.uses_ring_centroids:
            if self.type1_ring_groups is None or self.type4_atom_ids is None:
                raise ValueError("Incomplete ImamuraBeadSpec ring-centroid definition")
            if len(self.type1_ring_groups) != n_type1:
                raise ValueError(
                    f"type-1 ring groups ({len(self.type1_ring_groups)}) != {n_type1}"
                )
            if len(self.type4_atom_ids) != n_type4:
                raise ValueError(
                    f"type-4 atoms ({len(self.type4_atom_ids)}) != {n_type4}"
                )
        else:
            if not self.type1_selection or not self.type4_selection:
                raise ValueError(
                    "Provide type1/type4 MDAnalysis selections or ring-centroid groups"
                )


@dataclass
class ImamuraFeatureResult:
    """Per-frame Imamura feature table for one or more trajectories."""

    dataframe: Any  # pandas.DataFrame
    feature_names: List[str]
    formation_frames: Dict[str, int] = field(default_factory=dict)
    bead_spec: Optional[ImamuraBeadSpec] = None
    pair_trace: Optional["ImamuraPairTrace"] = None


@dataclass
class ImamuraPairTrace:
    """Compact per-frame mapping from sorted CV ranks to canonical bead pairs."""

    traj_ids: np.ndarray
    frames: np.ndarray
    v1_pair_indices: np.ndarray
    v4_pair_indices: np.ndarray


@dataclass
class ImamuraClusteringResult:
    """Outputs from dimensionality reduction + two-stage clustering."""

    pca_model: Any
    pca_scores: np.ndarray
    micro_labels: np.ndarray
    macro_labels: np.ndarray
    micro_to_macro: Dict[int, int]
    linkage_matrix: np.ndarray
    explained_variance_ratio: np.ndarray
    dimensionality_reduction: str = DEFAULT_DIMENSIONALITY_REDUCTION
    projection_eigenvalues: Optional[np.ndarray] = None
    implied_timescales_ns: Optional[np.ndarray] = None


@dataclass
class ImamuraTransitionResult:
    """Lagged transition counts between macrostates."""

    count_matrix: np.ndarray
    probability_matrix: np.ndarray
    count_df: Any
    prob_df: Any
    long_df: Any
    lag_frames: int
    lag_ns: float


def bead_positions_from_spec(universe: Any, bead_spec: ImamuraBeadSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Return (type1_positions, type4_positions) for the current universe frame."""
    if bead_spec.uses_ring_centroids:
        type1_rows = []
        for ring_ids in bead_spec.type1_ring_groups or []:
            ring_pos = universe.atoms[list(ring_ids)].positions
            type1_rows.append(ring_pos.mean(axis=0))
        type1 = np.asarray(type1_rows, dtype=np.float64)
        type4 = universe.atoms[list(bead_spec.type4_atom_ids or ())].positions
        return type1, type4

    type1 = universe.select_atoms(bead_spec.type1_selection).positions
    type4 = universe.select_atoms(bead_spec.type4_selection).positions
    return type1, type4


@dataclass
class ImamuraBeadResolutionOptions:
    """Auto-bead resolution knobs for RDKit + EndpointAnalyzer mode."""

    type4_source: str = "methyl"
    endpoint_extend_ring: bool = True
    endpoint_exclude_center_benzene: bool = True
    type4_rank: str = "bond"
    type4_per_monomer: Optional[int] = None

    def effective_type4_per_monomer(self) -> Optional[int]:
        """
        Per-monomer type-4 bead cap.

        * ``methyl`` — default 3 when unset.
        * ``endpoint`` — ``None`` keeps all endpoint candidates; an explicit
          positive integer trims with ``type4_rank``.
        """
        if self.type4_source == "endpoint":
            return self.type4_per_monomer
        return self.type4_per_monomer if self.type4_per_monomer is not None else 3


def make_imamura_endpoints_finder(
    *,
    extend_to_ring_atoms: bool = True,
) -> Any:
    """Construct :class:`EndpointsFinder` for Imamura type-4 bead picking."""
    from src.EndpointAnalyzer import EndpointsFinder

    return EndpointsFinder(extend_to_ring_atoms=extend_to_ring_atoms)


def _type4_methyl_candidates(
    mol: Any,
    endpoint_rdkit_indices: Sequence[int],
    *,
    ring_calculator: RingCenterCalculator,
) -> List[int]:
    """Methyl carbons linked to endpoints (fallback: all ``[CH3]`` in the monomer)."""
    candidates = filter_endpoints_to_methyl_atoms(
        mol, endpoint_rdkit_indices, ring_calculator=ring_calculator
    )
    if candidates:
        return candidates
    return all_methyl_atom_indices(mol, ring_calculator=ring_calculator)


def _exclude_center_ring_atoms(
    candidates: Sequence[int],
    center_ring_rdkit_indices: Sequence[int],
) -> List[int]:
    ring_set = set(center_ring_rdkit_indices)
    return [i for i in candidates if i not in ring_set]


def resolve_monomer_imamura_beads(
    universe: Any,
    mon_sel: str,
    *,
    ring_calc: Optional[RingCenterCalculator] = None,
    endpoints_finder: Optional[Any] = None,
    resolution: Optional[ImamuraBeadResolutionOptions] = None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Per-monomer Imamura beads via RDKit substructure + endpoint analysis.

    Type-1 uses the geometric center of :meth:`RingCenterCalculator.find_center_benzene_ring`.

    Type-4 source (``resolution.type4_source``):

    * ``methyl`` — methyl carbons from endpoint-linked ``[CH3]`` groups
    * ``endpoint`` — raw :class:`EndpointsFinder` output (optionally excluding
      the central benzene ring when ``endpoint_exclude_center_benzene`` is True)

    Trim to ``effective_type4_per_monomer()`` beads when that value is set.
    For ``endpoint`` source with no cap, **all** filtered endpoint atoms are kept.

    * ``bond`` — bond-graph distance from central benzene (nearest for methyl,
      outermost for endpoint)
    * ``centroid3d`` — 3D distance from the central ring centroid (farthest kept)
    """
    from src.EndpointAnalyzer import EndpointAnalyzer

    opts = resolution or ImamuraBeadResolutionOptions()
    n_type4 = opts.effective_type4_per_monomer()
    ring_calc = ring_calc or RingCenterCalculator()
    if endpoints_finder is None:
        endpoints_finder = make_imamura_endpoints_finder(
            extend_to_ring_atoms=opts.endpoint_extend_ring,
        )

    sel = universe.select_atoms(mon_sel)
    if len(sel) == 0:
        raise ValueError(f"Empty monomer selection: {mon_sel}")

    mol = sel.convert_to("RDKIT")
    if mol is None:
        raise ValueError(f"RDKit conversion failed for {mon_sel}")

    center_ring = ring_calc.find_center_benzene_ring(mol, sel.positions)
    if center_ring is None:
        raise ValueError(f"No center benzene ring found for {mon_sel}")

    ring_mda_ids = tuple(int(sel[i].id) for i in center_ring.rdkit_indices)

    _, ep_mda_ids = EndpointAnalyzer.find_residue_endpoints(
        universe, mon_sel, endpoints_finder
    )
    rdkit_for_mda = {int(sel.atoms[i].id): i for i in range(len(sel))}
    ep_rdkit = [rdkit_for_mda[mid] for mid in ep_mda_ids if mid in rdkit_for_mda]

    if opts.type4_source == "methyl":
        candidates = _type4_methyl_candidates(
            mol, ep_rdkit, ring_calculator=ring_calc
        )
    elif opts.type4_source == "endpoint":
        candidates = list(ep_rdkit)
    else:
        raise ValueError(
            f"Unknown type4_source {opts.type4_source!r}; use 'methyl' or 'endpoint'"
        )

    if opts.endpoint_exclude_center_benzene:
        candidates = _exclude_center_ring_atoms(
            candidates, center_ring.rdkit_indices
        )

    if not candidates:
        raise ValueError(f"Monomer '{mon_sel}': no type-4 candidates after filtering.")

    if n_type4 is None:
        type4_rdkit = sorted(set(candidates))
    elif opts.type4_rank == "centroid3d":
        type4_rdkit = select_type4_by_centroid_distance(
            candidates,
            center_ring.rdkit_indices,
            sel.positions,
            n_type4,
        )
    elif opts.type4_rank == "bond":
        type4_rdkit = select_type4_by_bond_distance(
            mol,
            candidates,
            center_ring.rdkit_indices,
            sel.positions,
            n_type4,
            outermost=opts.type4_source == "endpoint",
            ring_calculator=ring_calc,
        )
    else:
        raise ValueError(
            f"Unknown type4_rank {opts.type4_rank!r}; use 'bond' or 'centroid3d'"
        )

    if n_type4 is not None and len(type4_rdkit) != n_type4:
        raise ValueError(
            f"Monomer '{mon_sel}': type-4 selection yielded "
            f"{len(type4_rdkit)} atoms; expected {n_type4}."
        )

    type4_mda_ids = tuple(int(sel[i].id) for i in type4_rdkit)
    return ring_mda_ids, type4_mda_ids


def resolve_imamura_bead_spec(
    universe: Any,
    *,
    explicit_type1: Optional[str] = None,
    explicit_type4: Optional[str] = None,
    gsa_resname: str = "MOL",
    n_monomers: int = 6,
    bead_mode: str = "rdkit",
    type1_atom_name: str = "B1",
    type4_atom_name: str = "B4",
    resolution: Optional[ImamuraBeadResolutionOptions] = None,
) -> ImamuraBeadSpec:
    """
    Build :class:`ImamuraBeadSpec` for type-1 and type-4 Imamura beads.

    Default ``bead_mode='rdkit'`` uses central benzene ring centroids (type-1)
    and type-4 beads from ``resolution`` (methyl-filtered or raw endpoints).
    ``bead_mode='cg'`` uses CG bead names per monomer.
    """
    if explicit_type1 and explicit_type4:
        return ImamuraBeadSpec(
            type1_selection=explicit_type1,
            type4_selection=explicit_type4,
        )

    opts = resolution or ImamuraBeadResolutionOptions()
    type4_limit = opts.effective_type4_per_monomer()

    sel = resolve_selections(
        universe,
        gsa_resname=gsa_resname,
        n_monomers=n_monomers,
        auto_tooth=False,
    )
    monomer_sels = sel.monomer_selections or []

    if bead_mode == "cg":
        type1_ids: List[str] = []
        type4_ids: List[str] = []
        for mon_sel in monomer_sels:
            ag = universe.select_atoms(mon_sel)
            t1 = ag.select_atoms(f"name {type1_atom_name}")
            t4 = ag.select_atoms(f"name {type4_atom_name}")
            if len(t1) != 1:
                raise ValueError(
                    f"Monomer '{mon_sel}' has {len(t1)} type-1 atoms "
                    f"(name {type1_atom_name}); expected 1."
                )
            type1_ids.append(str(t1[0].index))
            if type4_limit is not None and len(t4) != type4_limit:
                raise ValueError(
                    f"Monomer '{mon_sel}' has {len(t4)} type-4 atoms "
                    f"(name {type4_atom_name}); expected {type4_limit}."
                )
            type4_ids.extend(str(a.index) for a in t4)
        return ImamuraBeadSpec(
            type1_selection="index " + " ".join(type1_ids),
            type4_selection="index " + " ".join(type4_ids),
        )

    if bead_mode != "rdkit":
        raise ValueError(f"Unknown bead_mode {bead_mode!r}; use 'rdkit' or 'cg'")

    ring_calc = RingCenterCalculator()
    ep_finder = make_imamura_endpoints_finder(
        extend_to_ring_atoms=opts.endpoint_extend_ring,
    )
    type1_groups: List[Tuple[int, ...]] = []
    type4_ids: List[int] = []

    for mon_sel in monomer_sels:
        ring_ids, type4_monomer_ids = resolve_monomer_imamura_beads(
            universe,
            mon_sel,
            ring_calc=ring_calc,
            endpoints_finder=ep_finder,
            resolution=opts,
        )
        type1_groups.append(ring_ids)
        type4_ids.extend(type4_monomer_ids)

    return ImamuraBeadSpec(
        type1_ring_groups=type1_groups,
        type4_atom_ids=tuple(type4_ids),
        type4_source=opts.type4_source,
        endpoint_extend_ring=opts.endpoint_extend_ring,
        endpoint_exclude_center_benzene=opts.endpoint_exclude_center_benzene,
        type4_rank=opts.type4_rank,
    )


def resolve_imamura_bead_selections(
    universe: Any,
    **kwargs: Any,
) -> Tuple[str, str]:
    """Legacy helper returning MDAnalysis selection strings only."""
    spec = resolve_imamura_bead_spec(universe, **kwargs)
    if spec.uses_ring_centroids:
        type1_ids = [str(ring_ids[0]) for ring_ids in spec.type1_ring_groups or []]
        type4_ids = [str(i) for i in spec.type4_atom_ids or ()]
        return "index " + " ".join(type1_ids), "index " + " ".join(type4_ids)
    return spec.type1_selection or "", spec.type4_selection or ""


def write_imamura_bead_spec(
    bead_spec: ImamuraBeadSpec,
    output_path: Union[str, Path],
) -> Path:
    """Write resolved bead indices for reproducibility."""
    import json

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if bead_spec.uses_ring_centroids:
        payload: Dict[str, Any] = {
            "mode": "ring_centroids",
            "type1_ring_groups": [list(g) for g in bead_spec.type1_ring_groups or []],
            "type4_atom_ids": list(bead_spec.type4_atom_ids or ()),
            "type4_source": bead_spec.type4_source,
            "endpoint_extend_ring": bead_spec.endpoint_extend_ring,
            "endpoint_exclude_center_benzene": bead_spec.endpoint_exclude_center_benzene,
            "type4_rank": bead_spec.type4_rank,
        }
    else:
        payload = {
            "mode": "selections",
            "type1_selection": bead_spec.type1_selection,
            "type4_selection": bead_spec.type4_selection,
        }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_imamura_bead_spec(path: Union[str, Path]) -> ImamuraBeadSpec:
    """Load a bead spec written by :func:`write_imamura_bead_spec`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("mode") == "ring_centroids":
        return ImamuraBeadSpec(
            type1_ring_groups=[tuple(g) for g in data["type1_ring_groups"]],
            type4_atom_ids=tuple(data["type4_atom_ids"]),
            type4_source=data.get("type4_source", "methyl"),
            endpoint_extend_ring=data.get("endpoint_extend_ring", True),
            endpoint_exclude_center_benzene=data.get(
                "endpoint_exclude_center_benzene", True
            ),
            type4_rank=data.get("type4_rank", "bond"),
        )
    return ImamuraBeadSpec(
        type1_selection=data.get("type1_selection"),
        type4_selection=data.get("type4_selection"),
    )


def _monomer_bead_atom_ids(
    universe: Any,
    mon_sel: str,
    bead_spec: ImamuraBeadSpec,
    monomer_index: int,
) -> Tuple[List[int], List[int]]:
    """MDAnalysis atom IDs for type-1 ring and type-4 methyl beads in one monomer."""
    ag = universe.select_atoms(mon_sel)
    mon_id_set = {int(i) for i in ag.ids}

    if bead_spec.uses_ring_centroids:
        rings = bead_spec.type1_ring_groups or []
        if monomer_index >= len(rings):
            return [], []
        ring_ids = [int(i) for i in rings[monomer_index] if int(i) in mon_id_set]
        methyl_ids = [
            int(i) for i in (bead_spec.type4_atom_ids or ()) if int(i) in mon_id_set
        ]
        return ring_ids, methyl_ids

    if not bead_spec.type1_selection or not bead_spec.type4_selection:
        return [], []
    t1 = universe.select_atoms(f"({bead_spec.type1_selection}) and ({mon_sel})")
    t4 = universe.select_atoms(f"({bead_spec.type4_selection}) and ({mon_sel})")
    return [int(a.id) for a in t1.atoms], [int(a.id) for a in t4.atoms]


def _draw_monomer_bead_panel(
    ax: Any,
    universe: Any,
    mon_sel: str,
    ring_mda_ids: Sequence[int],
    methyl_mda_ids: Sequence[int],
    *,
    panel_title: str,
    importance_ranks: Optional[Dict[int, int]] = None,
) -> None:
    """Render one monomer 2D structure with Imamura bead highlights."""
    from rdkit.Chem import Draw

    from src.EndpointAnalyzer import EndpointsFinder

    sel = universe.select_atoms(mon_sel)
    mol = sel.convert_to("RDKIT")
    if mol is None:
        ax.text(0.5, 0.5, f"RDKit failed\n{mon_sel}", ha="center", va="center")
        ax.axis("off")
        return

    ef = EndpointsFinder()
    m2d, xy = ef.to_2d_coords(mol)
    mda_to_rdkit = {int(sel.atoms[i].id): i for i in range(len(sel))}

    ring_rdkit = [mda_to_rdkit[i] for i in ring_mda_ids if i in mda_to_rdkit]
    methyl_rdkit = [mda_to_rdkit[i] for i in methyl_mda_ids if i in mda_to_rdkit]

    highlight_colors: Dict[int, Tuple[float, float, float]] = {}
    ring_set = set(ring_rdkit)
    for idx in ring_rdkit:
        highlight_colors[idx] = (0.0, 0.7, 0.0)
    for idx in methyl_rdkit:
        if idx not in ring_set:
            highlight_colors[idx] = (1.0, 0.5, 0.0)
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
        # Fallback: single-color highlight (cannot distinguish T1 vs T4).
        ring_only = [i for i in all_highlight if i in ring_rdkit]
        t4_only = [i for i in all_highlight if i in methyl_rdkit and i not in ring_rdkit]
        img = Draw.MolToImage(
            m2d,
            size=(800, 600),
            highlightAtoms=ring_only or all_highlight,
            highlightColor=(0.0, 0.7, 0.0),
        )
        if t4_only:
            from PIL import Image
            import numpy as np

            base = np.array(img.convert("RGBA"))
            overlay = np.array(
                Draw.MolToImage(
                    m2d,
                    size=(800, 600),
                    highlightAtoms=t4_only,
                    highlightColor=(1.0, 0.5, 0.0),
                ).convert("RGBA")
            )
            mask = overlay[:, :, 3] > 0
            base[mask] = overlay[mask]
            img = Image.fromarray(base)

    ax.imshow(img, extent=[0, 800, 600, 0])
    ax.axis("off")
    ax.set_title(panel_title, fontsize=11, fontweight="bold")

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

    for rdkit_idx, mda_id in zip(ring_rdkit, ring_mda_ids):
        if rdkit_idx >= len(xy):
            continue
        x, y = xy[rdkit_idx]
        rank = (importance_ranks or {}).get(int(mda_id))
        if rank is not None:
            label = f"#{rank} T1:{mda_id}"
            color = "crimson"
            fontsize = 11
        else:
            label = f"T1:{mda_id}"
            color = "darkgreen"
            fontsize = 9
        ax.annotate(
            label,
            xy=(x * scale + offset_x, -y * scale + offset_y),
            fontsize=fontsize,
            color=color,
            fontweight="bold",
        )
    for rdkit_idx, mda_id in zip(methyl_rdkit, methyl_mda_ids):
        if rdkit_idx >= len(xy):
            continue
        x, y = xy[rdkit_idx]
        rank = (importance_ranks or {}).get(int(mda_id))
        if rank is not None:
            label = f"#{rank} T4:{mda_id}"
            color = "crimson"
            fontsize = 11
        else:
            label = f"T4:{mda_id}"
            color = "darkorange"
            fontsize = 9
        ax.annotate(
            label,
            xy=(x * scale + offset_x, -y * scale + offset_y),
            fontsize=fontsize,
            color=color,
            fontweight="bold",
        )


def plot_imamura_bead_spec(
    universe: Any,
    monomer_selections: Sequence[str],
    bead_spec: ImamuraBeadSpec,
    output_path: Union[str, Path],
    *,
    dpi: int = 150,
    title: Optional[str] = None,
    importance_ranks: Optional[Dict[int, int]] = None,
) -> Path:
    """
    Save a multi-panel PNG of resolved Imamura type-1 and type-4 beads per monomer.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    type4_kind = "methyl" if bead_spec.type4_source == "methyl" else "endpoint"
    if title is None:
        if importance_ranks:
            title = (
                f"Imamura beads — top ranked drivers "
                f"(green=type-1, orange=type-4 {type4_kind}; #rank = importance)"
            )
        else:
            title = (
                f"Imamura auto-beads (green=type-1 benzene, orange=type-4 {type4_kind})"
            )

    n = len(monomer_selections)
    if n == 0:
        raise ValueError("No monomer selections provided for bead plot.")

    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.atleast_1d(axes).flatten())

    for idx, mon_sel in enumerate(monomer_selections):
        ring_ids, methyl_ids = _monomer_bead_atom_ids(
            universe, mon_sel, bead_spec, idx
        )
        _draw_monomer_bead_panel(
            axes_flat[idx],
            universe,
            mon_sel,
            ring_ids,
            methyl_ids,
            panel_title=f"Monomer {idx + 1}",
            importance_ranks=importance_ranks,
        )

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    legend_handles = [
        Patch(facecolor=(0.0, 0.7, 0.0), edgecolor="darkgreen", label="Type-1 (central benzene)"),
        Patch(
            facecolor=(1.0, 0.5, 0.0),
            edgecolor="darkorange",
            label=f"Type-4 ({type4_kind})",
        ),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_imamura_bead_importance_artifacts(
    output_dir: Union[str, Path],
    feature_result: ImamuraFeatureResult,
    clustering: ImamuraClusteringResult,
    universe: Any,
    monomer_selections: Sequence[str],
    bead_spec: ImamuraBeadSpec,
    *,
    top_n: int = 5,
    importance: str = "cluster",
    cluster_level: str = "macro",
    cluster_method: str = "pca_centroid",
) -> Dict[str, Path]:
    """
    Trace Imamura beads that drive clustering (default) or PCA variance.

    Parameters
    ----------
    importance
        ``cluster`` — features that separate macro/micro states (recommended).
        ``pca`` — features with highest PCA loadings (legacy).
    cluster_level
        ``macro`` or ``micro`` centroid level from two-stage clustering.
    cluster_method
        ``pca_centroid`` — back-project PCA centroids from
        ``two_stage_imamura_clustering`` (recommended);
        ``anova`` / ``between_var`` — raw-feature separation using labels only.
    """
    output_dir = Path(output_dir)
    type1_ref, type4_ref = bead_positions_from_spec(universe, bead_spec)

    if importance == "pca":
        bead_df, trace_df = compute_bead_pca_importance(
            clustering,
            feature_result.feature_names,
            bead_spec,
            type1_ref,
            type4_ref,
        )
        prefix = "pca"
        plot_title = (
            f"Imamura beads — top PCA variance drivers "
            f"(#rank = loading-weighted importance)"
        )
        ngl_title = "Imamura beads — top PCA drivers (3D)"
    elif importance == "cluster":
        labels = (
            clustering.macro_labels
            if cluster_level == "macro"
            else clustering.micro_labels
        )
        bead_df, trace_df = compute_bead_cluster_importance(
            feature_result.dataframe[feature_result.feature_names].to_numpy(),
            labels,
            feature_result.feature_names,
            bead_spec,
            type1_ref,
            type4_ref,
            clustering=clustering,
            method=cluster_method,
            cluster_level=cluster_level,
        )
        prefix = "cluster"
        plot_title = (
            f"Imamura beads — top {cluster_level}state cluster drivers "
            f"({cluster_method}; #rank = importance)"
        )
        ngl_title = f"Imamura beads — top {cluster_level}state drivers (3D)"
    else:
        raise ValueError(f"Unknown importance mode {importance!r}; use 'cluster' or 'pca'.")

    ranked_atoms = top_ranked_bead_atom_ids(bead_df, top_n=top_n)

    written: Dict[str, Path] = {}
    bead_path = output_dir / f"bead_{prefix}_importance.csv"
    bead_df.to_csv(bead_path, index=False)
    written[f"bead_{prefix}_importance"] = bead_path

    trace_path = output_dir / f"{prefix}_feature_bead_trace.csv"
    trace_df.to_csv(trace_path, index=False)
    written[f"{prefix}_feature_bead_trace"] = trace_path

    plot_path = output_dir / f"auto_beads_{prefix}_top5.png"
    plot_imamura_bead_spec(
        universe,
        monomer_selections,
        bead_spec,
        plot_path,
        importance_ranks=ranked_atoms,
        title=plot_title,
    )
    written[f"auto_beads_{prefix}_top5"] = plot_path

    ngl_path = output_dir / f"auto_beads_{prefix}_top5_3d.html"
    html_path, pdb_path = write_imamura_bead_pca_ngl_html(
        universe,
        bead_spec,
        bead_df,
        ngl_path,
        top_n=top_n,
        title=ngl_title,
    )
    written[f"auto_beads_{prefix}_top5_3d"] = html_path
    if pdb_path is not None:
        written[f"bead_{prefix}_structure"] = pdb_path

    return written


def write_imamura_bead_pca_artifacts(
    output_dir: Union[str, Path],
    feature_result: ImamuraFeatureResult,
    clustering: ImamuraClusteringResult,
    universe: Any,
    monomer_selections: Sequence[str],
    bead_spec: ImamuraBeadSpec,
    *,
    top_n: int = 5,
) -> Dict[str, Path]:
    """Legacy alias: PCA-variance bead attribution only."""
    return write_imamura_bead_importance_artifacts(
        output_dir,
        feature_result,
        clustering,
        universe,
        monomer_selections,
        bead_spec,
        top_n=top_n,
        importance="pca",
    )


def expected_distance_counts(n_type1: int, n_type4: int) -> Tuple[int, int, int]:
    """Return (n_type1_distances, n_type4_distances, total_dim)."""
    n1 = n_type1 * (n_type1 - 1) // 2
    n4 = n_type4 * (n_type4 - 1) // 2
    return n1, n4, n1 + n4


def imamura_feature_names(
    n_type1: int = DEFAULT_N_TYPE1_BEADS,
    n_type4: int = DEFAULT_N_TYPE4_BEADS,
) -> List[str]:
    """Column names for sorted inter-bead distance blocks."""
    n1, n4, _ = expected_distance_counts(n_type1, n_type4)
    names = [f"v1_{i:02d}" for i in range(1, n1 + 1)]
    names.extend(f"v4_{i:03d}" for i in range(1, n4 + 1))
    return names


def sorted_pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """
    All unique pairwise distances for *positions*, sorted descending.

    Parameters
    ----------
    positions : (N, 3)
        Bead coordinates for one frame.

    Returns
    -------
    (C(N, 2),) array
    """
    distances, _ = sorted_pairwise_distances_with_identity(positions)
    return distances


def sorted_pairwise_distances_with_identity(
    positions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted distances and canonical pair indices for one bead block.

    Canonical pair indices follow ``combinations(range(n_beads), 2)`` order.
    The returned identity array therefore records exactly which physical pair
    occupies each sorted Imamura CV rank in this frame.
    """
    n = positions.shape[0]
    if n < 2:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.uint16),
        )
    pairs = np.asarray(list(combinations(range(n), 2)), dtype=np.int64)
    deltas = positions[pairs[:, 0]] - positions[pairs[:, 1]]
    distances = np.linalg.norm(deltas, axis=1).astype(np.float64, copy=False)
    order = np.argsort(-distances, kind="stable")
    index_dtype = np.uint16 if len(pairs) <= np.iinfo(np.uint16).max else np.uint32
    return distances[order], order.astype(index_dtype, copy=False)


def build_imamura_feature_vector(
    type1_positions: np.ndarray,
    type4_positions: np.ndarray,
    *,
    n_type1: int = DEFAULT_N_TYPE1_BEADS,
    n_type4: int = DEFAULT_N_TYPE4_BEADS,
) -> np.ndarray:
    """Combine sorted type-1 and type-4 distance vectors into v_f (168-D)."""
    vf, _, _ = build_imamura_feature_vector_with_trace(
        type1_positions,
        type4_positions,
        n_type1=n_type1,
        n_type4=n_type4,
    )
    return vf


def build_imamura_feature_vector_with_trace(
    type1_positions: np.ndarray,
    type4_positions: np.ndarray,
    *,
    n_type1: int = DEFAULT_N_TYPE1_BEADS,
    n_type4: int = DEFAULT_N_TYPE4_BEADS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one Imamura vector and its exact rank-to-pair identity arrays."""
    if type1_positions.shape[0] != n_type1:
        raise ValueError(
            f"type-1 selection yielded {type1_positions.shape[0]} beads; "
            f"expected {n_type1}"
        )
    if type4_positions.shape[0] != n_type4:
        raise ValueError(
            f"type-4 selection yielded {type4_positions.shape[0]} beads; "
            f"expected {n_type4}"
        )

    v1, v1_pair_indices = sorted_pairwise_distances_with_identity(type1_positions)
    v4, v4_pair_indices = sorted_pairwise_distances_with_identity(type4_positions)
    n1, n4, total = expected_distance_counts(n_type1, n_type4)
    if v1.size != n1 or v4.size != n4:
        raise ValueError(
            f"Distance block sizes ({v1.size}, {v4.size}) != expected ({n1}, {n4})"
        )
    vf = np.concatenate([v1, v4])
    if vf.size != total:
        raise ValueError(f"Feature vector length {vf.size} != expected {total}")
    return vf, v1_pair_indices, v4_pair_indices


def _monomer_coords_list(universe: Any, monomer_selections: Sequence[str]) -> List[np.ndarray]:
    return [
        universe.select_atoms(sel).positions.copy()
        for sel in monomer_selections
    ]


def is_nanocube_assembled(
    stats: Dict[str, float],
    *,
    n_monomers: int,
    min_active_interfaces: int,
) -> bool:
    """
    Return True when the monomer contact graph indicates a formed nanocube.

    A cube is considered assembled when either:

    * all monomers belong to one connected contact component (typical for
      trajectories that start from an already-formed cage), or
    * ``n_active_interfaces >= min_active_interfaces`` (strict Imamura-style
      criterion; all face–face pairs active at the chosen cutoff).
    """
    lcc = int(stats["largest_connected_component_size"])
    n_active = int(stats["n_active_interfaces"])
    if lcc >= n_monomers:
        return True
    return n_active >= min_active_interfaces


def detect_formation_frame(
    universe: Any,
    monomer_selections: Sequence[str],
    *,
    n_monomers: Optional[int] = None,
    contact_cutoff: float = 4.5,
    min_active_interfaces: int = 15,
    sustain_frames: int = 1,
    stride: int = 1,
    start: int = 0,
    stop: Optional[int] = None,
) -> int:
    """
    First trajectory frame where the nanocube is fully assembled.

    Uses the inter-monomer contact graph from GSA tier-1 features.  Formation
    is detected when :func:`is_nanocube_assembled` is satisfied for
    *sustain_frames* consecutive visited frames.  Pre-assembled trajectories
    therefore return frame 0 as soon as all monomers share one connected
    component (even if fewer than 15/15 atom-pair interfaces exceed the cutoff).
    """
    if sustain_frames < 1:
        raise ValueError("sustain_frames must be >= 1")

    n_mon = n_monomers if n_monomers is not None else len(monomer_selections)
    n_traj = len(universe.trajectory)
    stop_eff = n_traj if stop is None else min(stop, n_traj)
    streak = 0
    first_candidate: Optional[int] = None

    for frame in range(start, stop_eff, stride):
        universe.trajectory[frame]
        monomer_coords = _monomer_coords_list(universe, monomer_selections)
        stats = F.inter_monomer_contact_features(
            monomer_coords,
            cutoff=contact_cutoff,
            active_threshold=1,
        )
        if is_nanocube_assembled(
            stats,
            n_monomers=n_mon,
            min_active_interfaces=min_active_interfaces,
        ):
            if streak == 0:
                first_candidate = frame
            streak += 1
            if streak >= sustain_frames:
                return int(first_candidate if first_candidate is not None else frame)
        else:
            streak = 0
            first_candidate = None

    raise RuntimeError(
        "Nanocube formation frame not found — relax "
        "--formation-min-interfaces, omit --truncate-at-formation for "
        "pre-assembled trajectories, or pass explicit formation frames."
    )


def _trajectory_id(traj_path: Path) -> str:
    """Identify a trajectory by its containing folder (e.g. ``BMMpM/run_000/mdcrd_v`` → ``run_000``)."""
    folder = traj_path.parent.name
    return folder if folder else traj_path.stem

def detect_formation_frames(
    topology: Union[str, Path],
    trajectory_paths: Sequence[Union[str, Path]],
    monomer_selections: Sequence[str],
    *,
    n_monomers: Optional[int] = None,
    traj_format: Optional[str] = None,
    contact_cutoff: float = 4.5,
    min_active_interfaces: int = 15,
    sustain_frames: int = 1,
    stride: int = 1,
) -> Dict[str, int]:
    """Run formation detection independently for each trajectory file."""
    import MDAnalysis as mda

    n_mon = n_monomers if n_monomers is not None else len(monomer_selections)
    out: Dict[str, int] = {}
    for traj_path in trajectory_paths:
        traj_path = Path(traj_path)
        kwargs: Dict[str, Any] = {}
        if traj_format:
            kwargs["format"] = traj_format
        u = mda.Universe(str(topology), str(traj_path), **kwargs)
        frame = detect_formation_frame(
            u,
            monomer_selections,
            n_monomers=n_mon,
            contact_cutoff=contact_cutoff,
            min_active_interfaces=min_active_interfaces,
            sustain_frames=sustain_frames,
            stride=stride,
        )
        out[_trajectory_id(traj_path)] = frame
    return out


class ImamuraBeadDistanceObserver(FrameObserver):
    """FrameObserver that records Imamura 168-D distance feature vectors."""

    def __init__(
        self,
        bead_spec: ImamuraBeadSpec,
        *,
        n_type1: int = DEFAULT_N_TYPE1_BEADS,
        n_type4: int = DEFAULT_N_TYPE4_BEADS,
        traj_id: str = "",
        start_frame: int = 0,
        record_pair_trace: bool = False,
    ):
        super().__init__()
        self.bead_spec = bead_spec
        self.n_type1 = n_type1
        self.n_type4 = n_type4
        self.traj_id = traj_id
        self.start_frame = start_frame
        self.record_pair_trace = record_pair_trace
        self.feature_names = imamura_feature_names(n_type1, n_type4)
        bead_spec.validate(n_type1=n_type1, n_type4=n_type4)

        self.results["rows"] = []
        self.results["pair_trace_rows"] = []
        self.results["frame_call_count"] = 0

    def get_selections_needed(self) -> List[str]:
        if self.bead_spec.uses_ring_centroids:
            return []
        return [
            self.bead_spec.type1_selection or "",
            self.bead_spec.type4_selection or "",
        ]

    def _get_aggregator(self) -> Optional[ResultsGroup]:
        return ResultsGroup(lookup={
            "rows": ResultsGroup.list_extend_sorted("frame"),
            "pair_trace_rows": ResultsGroup.list_extend_sorted("frame"),
            "frame_call_count": ResultsGroup.sum_values,
        })

    def on_frame_start(self, iterator: TrajectoryIterator) -> None:
        pass

    def on_frame(self, ts: Any, frame_idx: int, universe: Any) -> None:
        type1, type4 = bead_positions_from_spec(universe, self.bead_spec)
        if self.record_pair_trace:
            vf, v1_pair_indices, v4_pair_indices = (
                build_imamura_feature_vector_with_trace(
                    type1,
                    type4,
                    n_type1=self.n_type1,
                    n_type4=self.n_type4,
                )
            )
        else:
            vf = build_imamura_feature_vector(
                type1,
                type4,
                n_type1=self.n_type1,
                n_type4=self.n_type4,
            )
        time_ps = float(getattr(ts, "time", frame_idx))
        row = {
            "traj_id": self.traj_id,
            "frame": int(frame_idx),
            "time_ps": time_ps,
        }
        for name, val in zip(self.feature_names, vf):
            row[name] = float(val)
        self.results["rows"].append(row)
        if self.record_pair_trace:
            self.results["pair_trace_rows"].append(
                {
                    "frame": int(frame_idx),
                    "v1_pair_indices": v1_pair_indices,
                    "v4_pair_indices": v4_pair_indices,
                }
            )
        self.results["frame_call_count"] += 1

    def on_frame_end(self, iterator: TrajectoryIterator) -> None:
        pass


def compute_imamura_features(
    universe: Any,
    bead_spec: ImamuraBeadSpec,
    *,
    n_type1: int = DEFAULT_N_TYPE1_BEADS,
    n_type4: int = DEFAULT_N_TYPE4_BEADS,
    traj_id: str = "",
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: Optional[int] = None,
    n_jobs: int = 1,
    record_pair_trace: bool = False,
) -> Any:
    """Extract Imamura features for one trajectory via TrajectoryIterator."""
    import pandas as pd

    observer = ImamuraBeadDistanceObserver(
        bead_spec,
        n_type1=n_type1,
        n_type4=n_type4,
        traj_id=traj_id,
        start_frame=start or 0,
        record_pair_trace=record_pair_trace,
    )
    iterator = TrajectoryIterator(universe)
    iterator.subscribe(observer)
    iterator.iterate(start=start, stop=stop, step=step, n_jobs=n_jobs)

    rows = observer.results.get("rows", [])
    if not rows:
        return pd.DataFrame(columns=["traj_id", "frame", "time_ps", *observer.feature_names])
    dataframe = pd.DataFrame(rows)
    if record_pair_trace:
        pair_rows = observer.results.get("pair_trace_rows", [])
        if len(pair_rows) != len(dataframe):
            raise RuntimeError(
                "Pair-identity trace row count does not match Imamura feature rows"
            )
        dataframe.attrs["imamura_pair_trace"] = ImamuraPairTrace(
            traj_ids=np.full(len(pair_rows), str(traj_id)),
            frames=np.asarray([row["frame"] for row in pair_rows], dtype=np.int64),
            v1_pair_indices=np.stack(
                [row["v1_pair_indices"] for row in pair_rows]
            ),
            v4_pair_indices=np.stack(
                [row["v4_pair_indices"] for row in pair_rows]
            ),
        )
    return dataframe


def extract_imamura_features_multi(
    topology: Union[str, Path],
    trajectory_paths: Sequence[Union[str, Path]],
    bead_spec: ImamuraBeadSpec,
    *,
    traj_format: Optional[str] = None,
    formation_frames: Optional[Dict[str, int]] = None,
    truncate_at_formation: bool = False,
    stride: int = 1,
    n_type1: int = DEFAULT_N_TYPE1_BEADS,
    n_type4: int = DEFAULT_N_TYPE4_BEADS,
    n_jobs: int = 1,
    record_pair_trace: bool = False,
) -> ImamuraFeatureResult:
    """Extract features from multiple trajectories, optionally post-formation."""
    import MDAnalysis as mda
    import pandas as pd

    feature_names = imamura_feature_names(n_type1, n_type4)
    frames_map = formation_frames or {}
    parts: List[Any] = []
    pair_trace_parts: List[ImamuraPairTrace] = []

    for traj_path in trajectory_paths:
        traj_path = Path(traj_path)
        traj_id = _trajectory_id(traj_path)
        kwargs: Dict[str, Any] = {}
        if traj_format:
            kwargs["format"] = traj_format
        u = mda.Universe(str(topology), str(traj_path), **kwargs)

        start = 0
        if truncate_at_formation:
            if traj_id not in frames_map:
                raise KeyError(
                    f"Missing formation frame for {traj_id!r}. "
                    "Run detection first or pass --no-truncate."
                )
            start = int(frames_map[traj_id])

        df = compute_imamura_features(
            u,
            bead_spec,
            n_type1=n_type1,
            n_type4=n_type4,
            traj_id=traj_id,
            start=start,
            step=stride,
            n_jobs=n_jobs,
            record_pair_trace=record_pair_trace,
        )
        if not df.empty:
            pair_trace = df.attrs.get("imamura_pair_trace")
            if pair_trace is not None:
                pair_trace_parts.append(pair_trace)
            parts.append(df)

    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["traj_id", "frame", "time_ps", *feature_names]
    )
    combined_pair_trace: Optional[ImamuraPairTrace] = None
    if pair_trace_parts:
        combined_pair_trace = ImamuraPairTrace(
            traj_ids=np.concatenate([trace.traj_ids for trace in pair_trace_parts]),
            frames=np.concatenate([trace.frames for trace in pair_trace_parts]),
            v1_pair_indices=np.concatenate(
                [trace.v1_pair_indices for trace in pair_trace_parts], axis=0
            ),
            v4_pair_indices=np.concatenate(
                [trace.v4_pair_indices for trace in pair_trace_parts], axis=0
            ),
        )
    return ImamuraFeatureResult(
        dataframe=combined,
        feature_names=feature_names,
        formation_frames=frames_map,
        bead_spec=bead_spec,
        pair_trace=combined_pair_trace,
    )


def fit_imamura_pca(
    feature_matrix: np.ndarray,
    *,
    n_components: int = DEFAULT_PCA_COMPONENTS,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """PCA on standardized Imamura feature vectors."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)
    X = np.nan_to_num(X, nan=0.0)

    n_comp = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp, random_state=0)
    scores = pca.fit_transform(X)
    return pca, scores, pca.explained_variance_ratio_


def imamura_projection_prefix(dimensionality_reduction: str) -> str:
    """Column prefix for reduced coordinates (``PC`` or ``TIC``)."""
    if dimensionality_reduction == "tlica":
        return "TIC"
    return "PC"


def imamura_projection_column_names(
    dimensionality_reduction: str,
    n_components: int,
) -> List[str]:
    prefix = imamura_projection_prefix(dimensionality_reduction)
    return [f"{prefix}{i + 1}" for i in range(n_components)]


def _tica_trajectory_segments(
    X: np.ndarray,
    traj_ids: Optional[np.ndarray],
) -> List[np.ndarray]:
    if traj_ids is None:
        return [X]
    import pandas as pd

    segments: List[np.ndarray] = []
    for tid in pd.unique(traj_ids):
        segments.append(X[traj_ids == tid])
    return segments


def _tica_covariance_matrices(
    X: np.ndarray,
    *,
    traj_ids: Optional[np.ndarray] = None,
    lag_frames: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mean-free instantaneous and symmetrized time-lagged covariances.

    Lag pairs are formed only within each trajectory segment.
    """
    lag = max(1, int(lag_frames))
    n_features = X.shape[1]
    C0 = np.zeros((n_features, n_features), dtype=np.float64)
    Ctau = np.zeros((n_features, n_features), dtype=np.float64)
    n0 = 0
    ntau = 0

    for seg in _tica_trajectory_segments(X, traj_ids):
        if len(seg) < 2:
            continue
        centered = seg - seg.mean(axis=0)
        C0 += centered.T @ centered
        n0 += len(centered)
        if len(centered) > lag:
            x_t = centered[:-lag]
            x_tau = centered[lag:]
            Ctau += x_t.T @ x_tau
            ntau += len(x_t)

    if n0 < 2:
        raise ValueError("Need at least two frames for tlICA.")
    C0 /= max(n0 - 1, 1)
    if ntau < 1:
        raise ValueError(
            f"Need at least {lag + 1} frames per trajectory for tlICA lag={lag}."
        )
    Ctau /= ntau
    Ctau = 0.5 * (Ctau + Ctau.T)
    return C0, Ctau


def fit_imamura_tlica(
    feature_matrix: np.ndarray,
    *,
    n_components: int = DEFAULT_PCA_COMPONENTS,
    traj_ids: Optional[np.ndarray] = None,
    lag_frames: int = 1,
    lag_ns: float = DEFAULT_TRANSITION_LAG_NS,
    regularization: float = DEFAULT_TLICA_REGULARIZATION,
) -> Tuple[ImamuraTICAModel, np.ndarray, np.ndarray]:
    """
    Time-lagged independent component analysis (TICA / tlICA).

    Solves the generalized eigenvalue problem
    ``C(tau) v = lambda C(0) v`` on standardized features and returns the
    leading slow modes.
    """
    from scipy.linalg import eigh
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_matrix)
    X = np.nan_to_num(X, nan=0.0)

    C0, Ctau = _tica_covariance_matrices(
        X,
        traj_ids=traj_ids,
        lag_frames=lag_frames,
    )
    n_features = C0.shape[0]
    reg = float(regularization) * np.trace(C0) / max(n_features, 1)
    C0 = C0 + reg * np.eye(n_features)

    eigvals, eigvecs = eigh(Ctau, C0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    n_comp = min(n_components, eigvecs.shape[1])
    components = eigvecs[:, :n_comp].T
    scores = X @ components.T

    lag_ns_eff = float(lag_ns)
    timescales = np.full(n_comp, np.nan, dtype=np.float64)
    for i, lam in enumerate(eigvals[:n_comp]):
        mag = min(abs(float(lam)), 1.0 - 1e-12)
        if mag > 1e-12:
            timescales[i] = -lag_ns_eff / np.log(mag)

    evr = np.clip(eigvals[:n_comp], 0.0, None)
    evr_sum = float(evr.sum())
    if evr_sum > 0:
        evr = evr / evr_sum

    model = ImamuraTICAModel(
        components_=components,
        eigenvalues_=eigvals[:n_comp],
        mean_=scaler.mean_,
        scale_=scaler.scale_,
        lag_frames=int(lag_frames),
        lag_ns=lag_ns_eff,
        implied_timescales_ns=timescales,
    )
    return model, scores, evr


def _sorted_bead_pair_indices(positions: np.ndarray) -> List[Tuple[int, int]]:
    """Bead-index pairs in descending distance order (Imamura sorted blocks)."""
    pairs: List[Tuple[float, int, int]] = []
    for i, j in combinations(range(positions.shape[0]), 2):
        d = float(np.linalg.norm(positions[i] - positions[j]))
        pairs.append((d, i, j))
    pairs.sort(key=lambda item: item[0], reverse=True)
    return [(i, j) for _, i, j in pairs]


def imamura_feature_bead_pair_map(
    type1_positions: np.ndarray,
    type4_positions: np.ndarray,
) -> Dict[str, Tuple[str, int, int]]:
    """
    Map Imamura feature column names to bead pairs for one reference frame.

    Sorted-distance features lose bead identity across frames; this uses the
    reference geometry to recover which bead pair each ``v1_*`` / ``v4_*`` slot
    represents.
    """
    mapping: Dict[str, Tuple[str, int, int]] = {}
    for slot, (i, j) in enumerate(_sorted_bead_pair_indices(type1_positions), start=1):
        mapping[f"v1_{slot:02d}"] = ("type1", i, j)
    for slot, (i, j) in enumerate(_sorted_bead_pair_indices(type4_positions), start=1):
        mapping[f"v4_{slot:03d}"] = ("type4", i, j)
    return mapping


def pca_feature_importance(
    pca_model: Any,
    explained_variance_ratio: np.ndarray,
) -> np.ndarray:
    """Variance-weighted squared loadings per raw Imamura feature."""
    weights = explained_variance_ratio / explained_variance_ratio.sum()
    loadings = pca_model.components_
    importance = np.zeros(loadings.shape[1], dtype=np.float64)
    for i, weight in enumerate(weights):
        importance += weight * (loadings[i] ** 2)
    return importance


def _occupied_micro_centroids_pca(
    clustering: ImamuraClusteringResult,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Microcluster centroids in PCA space, matching ``two_stage_imamura_clustering``.

    Returns occupied micro ids, centroid matrix (n_micro, n_pc), frame counts.
    """
    micro = np.asarray(clustering.micro_labels, dtype=int)
    scores = clustering.pca_scores
    occupied = sorted(set(int(x) for x in micro.tolist()))
    centroids = np.array(
        [scores[micro == k].mean(axis=0) for k in occupied],
        dtype=np.float64,
    )
    weights = np.array([np.sum(micro == k) for k in occupied], dtype=np.float64)
    return occupied, centroids, weights


def _macro_centroids_pca_from_two_stage(
    clustering: ImamuraClusteringResult,
) -> np.ndarray:
    """
    Macrostates centroids in PCA space by frame-weighted merging of micro centroids.

    This mirrors the Ward step in ``two_stage_imamura_clustering``: macrostates are
    defined from microcluster centroids, not by re-running KMeans on macro labels.
    """
    occupied, centroids, weights = _occupied_micro_centroids_pca(clustering)
    micro_to_macro = clustering.micro_to_macro
    macro_ids = sorted(set(int(v) for v in micro_to_macro.values()))
    macro_centroids: List[np.ndarray] = []
    for macro in macro_ids:
        idx = [i for i, micro in enumerate(occupied) if micro_to_macro[micro] == macro]
        if not idx:
            continue
        w = weights[idx]
        c = centroids[idx]
        if w.sum() > 0:
            macro_centroids.append(np.average(c, axis=0, weights=w))
    if not macro_centroids:
        return np.zeros((0, centroids.shape[1]), dtype=np.float64)
    return np.asarray(macro_centroids, dtype=np.float64)


def _backproject_pca_centroid_spread(
    centroids_pca: np.ndarray,
    components: np.ndarray,
) -> np.ndarray:
    """Variance of back-projected centroid coordinates across clusters."""
    if centroids_pca.shape[0] < 2:
        return np.zeros(components.shape[1], dtype=np.float64)
    c_feat = centroids_pca @ components
    var = c_feat.var(axis=0)
    total = var.sum()
    if total > 0:
        return var / total
    return var


def cluster_feature_importance(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    method: str = "pca_centroid",
    clustering: Optional[ImamuraClusteringResult] = None,
    cluster_level: str = "macro",
) -> np.ndarray:
    """
    Score features by how much they separate cluster assignments.

    Methods
    -------
    pca_centroid
        Spread of PCA-space cluster centroids back-projected to features.
        Uses the same geometry as ``two_stage_imamura_clustering``:
        micro = occupied MiniBatchKMeans centroids; macro = frame-weighted
        merge of those centroids via ``micro_to_macro``. **Recommended.**
    anova
        ANOVA F-statistic on raw features vs labels (uses labels only).
    between_var
        Variance of per-cluster raw-feature means (uses labels only).
    """
    X = np.nan_to_num(feature_matrix, nan=0.0)
    labels = np.asarray(labels, dtype=int)
    n_features = X.shape[1]

    if method == "pca_centroid":
        if clustering is None:
            raise ValueError("pca_centroid method requires clustering result.")
        components = clustering.pca_model.components_
        if cluster_level == "macro":
            centroids_pca = _macro_centroids_pca_from_two_stage(clustering)
        else:
            _, centroids_pca, _ = _occupied_micro_centroids_pca(clustering)
        return _backproject_pca_centroid_spread(centroids_pca, components)

    if method == "anova":
        from sklearn.feature_selection import f_classif

        scores, _ = f_classif(X, labels)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        total = scores.sum()
        if total > 0:
            return scores / total
        return scores

    if method == "between_var":
        centroids = []
        for lab in sorted(set(labels.tolist())):
            mask = labels == lab
            if mask.any():
                centroids.append(X[mask].mean(axis=0))
        if len(centroids) < 2:
            return np.zeros(n_features, dtype=np.float64)
        c = np.asarray(centroids, dtype=np.float64)
        var = c.var(axis=0)
        total = var.sum()
        if total > 0:
            return var / total
        return var

    raise ValueError(
        f"Unknown cluster feature method {method!r}; "
        "use 'pca_centroid', 'anova', or 'between_var'."
    )


def _aggregate_feature_importance_to_beads(
    feat_imp: np.ndarray,
    feature_names: Sequence[str],
    pair_map: Dict[str, Tuple[str, int, int]],
    bead_spec: ImamuraBeadSpec,
    *,
    importance_source: str,
    trace_extra: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Map per-feature scores to beads via the reference-frame pair map."""
    import pandas as pd
    from collections import defaultdict

    trace_rows: List[Dict[str, Any]] = []
    bead_scores: Dict[Tuple[str, int], float] = defaultdict(float)
    extra = trace_extra or {}

    for col_idx, fname in enumerate(feature_names):
        if fname not in pair_map:
            continue
        kind, i, j = pair_map[fname]
        importance = float(feat_imp[col_idx])
        bead_scores[(kind, i)] += importance
        bead_scores[(kind, j)] += importance
        atom_i = _bead_slot_atom_ids(bead_spec, kind, i)
        atom_j = _bead_slot_atom_ids(bead_spec, kind, j)
        trace_rows.append({
            "feature": fname,
            "bead_kind": kind,
            "bead_i": i,
            "bead_j": j,
            "bead_i_label": _bead_slot_label(bead_spec, kind, i),
            "bead_j_label": _bead_slot_label(bead_spec, kind, j),
            "atom_i_ids": ",".join(str(a) for a in atom_i),
            "atom_j_ids": ",".join(str(a) for a in atom_j),
            "feature_importance": importance,
            "importance_source": importance_source,
            **extra,
        })

    bead_rows: List[Dict[str, Any]] = []
    for (kind, idx), score in bead_scores.items():
        atom_ids = _bead_slot_atom_ids(bead_spec, kind, idx)
        if kind == "type1":
            monomer = idx + 1
        else:
            monomer = _type4_monomer_index(bead_spec, idx) + 1
        bead_rows.append({
            "bead_kind": kind,
            "bead_index": idx,
            "bead_label": _bead_slot_label(bead_spec, kind, idx),
            "monomer": monomer,
            "atom_ids": ",".join(str(a) for a in atom_ids),
            "importance": score,
            "importance_source": importance_source,
        })

    bead_df = pd.DataFrame(bead_rows)
    if not bead_df.empty:
        bead_df = bead_df.sort_values("importance", ascending=False).reset_index(drop=True)
        bead_df.insert(0, "rank", range(1, len(bead_df) + 1))

    trace_df = pd.DataFrame(trace_rows)
    if not trace_df.empty:
        trace_df = trace_df.sort_values("feature_importance", ascending=False).reset_index(
            drop=True
        )

    return bead_df, trace_df


def compute_bead_cluster_importance(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: Sequence[str],
    bead_spec: ImamuraBeadSpec,
    reference_type1: np.ndarray,
    reference_type4: np.ndarray,
    *,
    clustering: Optional[ImamuraClusteringResult] = None,
    method: str = "anova",
    cluster_level: str = "macro",
) -> Tuple[Any, Any]:
    """
    Trace cluster-separating Imamura features back to beads.

    Uses per-cluster differences in inter-bead distance features (or
    PCA-backprojected microcluster centroids) and maps sorted-distance
    slots to bead pairs in a reference frame.
    """
    pair_map = imamura_feature_bead_pair_map(reference_type1, reference_type4)
    feat_imp = cluster_feature_importance(
        feature_matrix,
        labels,
        method=method,
        clustering=clustering,
        cluster_level=cluster_level,
    )
    source = f"cluster_{cluster_level}_{method}"
    bead_df, trace_df = _aggregate_feature_importance_to_beads(
        feat_imp,
        feature_names,
        pair_map,
        bead_spec,
        importance_source=source,
    )
    if not trace_df.empty:
        trace_df["cluster_level"] = cluster_level
        trace_df["cluster_method"] = method
    if not bead_df.empty:
        bead_df["cluster_level"] = cluster_level
        bead_df["cluster_method"] = method
    return bead_df, trace_df


def _bead_slot_atom_ids(
    bead_spec: ImamuraBeadSpec,
    bead_kind: str,
    bead_index: int,
) -> Tuple[int, ...]:
    if bead_kind == "type1":
        if bead_spec.uses_ring_centroids:
            groups = bead_spec.type1_ring_groups or []
            if bead_index < len(groups):
                return tuple(int(a) for a in groups[bead_index])
        return ()
    t4 = bead_spec.type4_atom_ids or ()
    if bead_index < len(t4):
        return (int(t4[bead_index]),)
    return ()


def _type4_monomer_index(bead_spec: ImamuraBeadSpec, type4_index: int) -> int:
    n_monomers = len(bead_spec.type1_ring_groups or ())
    n_type4 = len(bead_spec.type4_atom_ids or ())
    if n_monomers < 1 or n_type4 < 1:
        return 0
    per_monomer = n_type4 // n_monomers
    if per_monomer < 1:
        return 0
    return type4_index // per_monomer


def _bead_slot_label(
    bead_spec: ImamuraBeadSpec,
    bead_kind: str,
    bead_index: int,
) -> str:
    if bead_kind == "type1":
        return f"T1-M{bead_index + 1}"
    atom_ids = _bead_slot_atom_ids(bead_spec, bead_kind, bead_index)
    if atom_ids:
        return f"T4:{atom_ids[0]}"
    return f"T4[{bead_index}]"


def compute_bead_pca_importance(
    clustering: ImamuraClusteringResult,
    feature_names: Sequence[str],
    bead_spec: ImamuraBeadSpec,
    reference_type1: np.ndarray,
    reference_type4: np.ndarray,
) -> Tuple[Any, Any]:
    """
    Trace PCA loadings back to Imamura beads via a reference-frame pair map.

    Returns
    -------
    bead_importance_df, feature_trace_df
        Per-bead aggregated scores and per-feature bead-pair attribution.
    """
    pair_map = imamura_feature_bead_pair_map(reference_type1, reference_type4)
    feat_imp = pca_feature_importance(
        clustering.pca_model,
        clustering.explained_variance_ratio,
    )
    loadings = clustering.pca_model.components_
    evr = clustering.explained_variance_ratio

    bead_df, trace_df = _aggregate_feature_importance_to_beads(
        feat_imp,
        feature_names,
        pair_map,
        bead_spec,
        importance_source="pca_variance",
    )

    if not trace_df.empty:
        pc_rows: List[Dict[str, Any]] = []
        for _, row in trace_df.iterrows():
            fname = row["feature"]
            col_idx = feature_names.index(fname) if fname in feature_names else -1
            if col_idx < 0:
                continue
            kind, i, j = pair_map[fname]
            for pc_idx in range(loadings.shape[0]):
                pc_rows.append({
                    **row.to_dict(),
                    "pc": f"PC{pc_idx + 1}",
                    "loading": float(loadings[pc_idx, col_idx]),
                    "pc_variance_weight": float(evr[pc_idx]),
                    "weighted_loading_sq": float(
                        evr[pc_idx] * (loadings[pc_idx, col_idx] ** 2)
                    ),
                })
        import pandas as pd
        trace_df = pd.DataFrame(pc_rows)
        if not trace_df.empty:
            trace_df = trace_df.sort_values(
                ["feature_importance", "weighted_loading_sq"],
                ascending=False,
            ).reset_index(drop=True)

    return bead_df, trace_df


def top_ranked_bead_atom_ids(
    bead_importance_df: Any,
    *,
    top_n: int = 5,
) -> Dict[int, int]:
    """Map MDAnalysis atom id → PCA importance rank (1 = highest)."""
    ranked: Dict[int, int] = {}
    if bead_importance_df is None or bead_importance_df.empty:
        return ranked
    for _, row in bead_importance_df.head(top_n).iterrows():
        rank = int(row["rank"])
        for atom_s in str(row["atom_ids"]).split(","):
            atom_s = atom_s.strip()
            if atom_s:
                ranked[int(atom_s)] = rank
    return ranked


BEAD_PCA_RANK_COLORS = [
    "#dc143c",
    "#1e90ff",
    "#32cd32",
    "#ff8c00",
    "#8a2be2",
    "#ffd700",
]

IMAMURA_BEAD_PCA_NGL_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; overflow: hidden; font-family: system-ui, sans-serif; }}
  #viewport {{ width: 100vw; height: 100vh; }}
  #sidebar {{
    position: absolute; top: 12px; left: 12px; width: 300px; max-height: calc(100vh - 24px);
    overflow-y: auto; color: #e8e8e8; background: rgba(0,0,0,0.72);
    padding: 12px 14px; border-radius: 10px; backdrop-filter: blur(8px);
    font-size: 13px; line-height: 1.45;
  }}
  #sidebar h2 {{ font-size: 14px; margin-bottom: 8px; font-weight: 600; }}
  #sidebar .meta {{ color: #aaa; font-size: 12px; margin-bottom: 10px; }}
  .rank-row {{
    display: flex; align-items: flex-start; gap: 8px; padding: 6px 0;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }}
  .rank-row:last-child {{ border-bottom: none; }}
  .swatch {{ width: 14px; height: 14px; border-radius: 50%; flex-shrink: 0; margin-top: 2px; }}
  .rank-label {{ font-weight: 600; }}
  .rank-detail {{ color: #bbb; font-size: 12px; }}
  #controls {{
    position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%);
    color: #bbb; font-size: 12px; background: rgba(0,0,0,0.45);
    padding: 6px 16px; border-radius: 6px; pointer-events: none;
    backdrop-filter: blur(6px);
  }}
</style>
<script src="https://unpkg.com/ngl@2.3.1/dist/ngl.js"></script>
</head>
<body>
<div id="viewport"></div>
<div id="sidebar">
  <h2>{title}</h2>
  <div class="meta">{meta_text}</div>
  <div id="rank-list"></div>
</div>
<div id="controls">Scroll to zoom &middot; Click-drag to rotate &middot; Right-drag to pan</div>
<script>
document.addEventListener("DOMContentLoaded", function() {{
  var highlights = {highlights_json};
  var type1Positions = {type1_positions_json};
  var type4Positions = {type4_positions_json};

  var stage = new NGL.Stage("viewport", {{
    backgroundColor: "#1a1a2e",
    ambientIntensity: 0.35,
    quality: "high"
  }});
  window.addEventListener("resize", function() {{ stage.handleResize(); }});

  var listEl = document.getElementById("rank-list");
  highlights.forEach(function(h) {{
    var row = document.createElement("div");
    row.className = "rank-row";
    var swatch = document.createElement("div");
    swatch.className = "swatch";
    swatch.style.background = h.color;
    var text = document.createElement("div");
    text.innerHTML =
      '<div class="rank-label">#' + h.rank + ' ' + h.label + '</div>' +
      '<div class="rank-detail">Monomer ' + h.monomer + ' &middot; ' +
      h.kind + ' &middot; atom(s) ' + h.atom_ids + '</div>';
    row.appendChild(swatch);
    row.appendChild(text);
    listEl.appendChild(row);
  }});

  var pdbBlob = new Blob([`{pdb_string}`], {{ type: "text/plain" }});
  stage.loadFile(pdbBlob, {{ ext: "pdb", defaultRepresentation: false }}).then(function(comp) {{
    comp.addRepresentation("licorice", {{
      color: "grey",
      radiusScale: 1.2,
      opacity: 0.75
    }});
    if (type1Positions.length) {{
      comp.addRepresentation("ball+stick", {{
        sele: "@" + type1Positions.join(","),
        color: "#2ecc71",
        radiusScale: 1.4,
        opacity: 0.55
      }});
    }}
    if (type4Positions.length) {{
      comp.addRepresentation("ball+stick", {{
        sele: "@" + type4Positions.join(","),
        color: "#ff9933",
        radiusScale: 1.0,
        opacity: 0.35
      }});
    }}
    highlights.forEach(function(h) {{
      var sele = "@" + h.positions.join(",");
      comp.addRepresentation("spacefill", {{
        sele: sele,
        color: h.color,
        radiusScale: 2.0,
        opacity: 0.55
      }});
      comp.addRepresentation("label", {{
        sele: sele,
        color: h.color,
        labelType: "atomname",
        radiusScale: 2.2,
        fontSize: 14,
        showBorder: true,
        borderColor: "black"
      }});
    }});
    comp.autoView();
  }});
}});
</script>
</body>
</html>
"""


def _bead_pca_ngl_highlight_payloads(
    bead_importance_df: Any,
    id_to_pos: Dict[int, int],
    *,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """Build NGL highlight metadata for ranked Imamura beads."""
    highlights: List[Dict[str, Any]] = []
    if bead_importance_df is None or bead_importance_df.empty:
        return highlights

    for _, row in bead_importance_df.head(top_n).iterrows():
        rank = int(row["rank"])
        atom_ids = [int(x) for x in str(row["atom_ids"]).split(",") if x.strip()]
        positions = sorted(id_to_pos[aid] for aid in atom_ids if aid in id_to_pos)
        if not positions:
            continue
        color = BEAD_PCA_RANK_COLORS[(rank - 1) % len(BEAD_PCA_RANK_COLORS)]
        highlights.append({
            "rank": rank,
            "label": str(row["bead_label"]),
            "monomer": int(row["monomer"]),
            "kind": str(row["bead_kind"]),
            "atom_ids": ",".join(str(a) for a in atom_ids),
            "color": color,
            "positions": positions,
        })
    return highlights


def write_imamura_bead_pca_ngl_html(
    universe: Any,
    bead_spec: ImamuraBeadSpec,
    bead_importance_df: Any,
    output_path: Union[str, Path],
    *,
    selection: str = "resname MOL",
    frame: int = 0,
    top_n: int = 5,
    write_pdb: bool = True,
    title: str = "Imamura beads — top drivers (3D)",
) -> Tuple[Path, Optional[Path]]:
    """
    Interactive NGL viewer for the nanocube with top-ranked beads highlighted.

    Returns (html_path, pdb_path_or_none).
    """
    import os
    import tempfile

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    universe.trajectory[frame]
    atoms = universe.select_atoms(selection)
    if len(atoms) == 0:
        raise ValueError(f"Selection '{selection}' matched 0 atoms at frame {frame}.")

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".pdb",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp_path = tmp.name
    try:
        atoms.write(tmp_path)
        pdb_string = Path(tmp_path).read_text(encoding="utf-8")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    pdb_string = pdb_string.replace("\\", "\\\\").replace("`", "\\`").replace("${", "\\${")
    id_to_pos = {int(atom.id): pos for pos, atom in enumerate(atoms)}

    highlights = _bead_pca_ngl_highlight_payloads(
        bead_importance_df,
        id_to_pos,
        top_n=top_n,
    )
    ranked_pos_set = {p for h in highlights for p in h["positions"]}

    type1_positions: List[int] = []
    if bead_spec.uses_ring_centroids:
        for ring in bead_spec.type1_ring_groups or []:
            for aid in ring:
                pos = id_to_pos.get(int(aid))
                if pos is not None and pos not in ranked_pos_set:
                    type1_positions.append(pos)
    elif bead_spec.type1_selection:
        t1 = universe.select_atoms(f"({bead_spec.type1_selection}) and ({selection})")
        type1_positions = sorted(
            id_to_pos[int(a.id)]
            for a in t1
            if int(a.id) in id_to_pos and id_to_pos[int(a.id)] not in ranked_pos_set
        )

    type4_positions: List[int] = []
    for aid in bead_spec.type4_atom_ids or ():
        pos = id_to_pos.get(int(aid))
        if pos is not None and pos not in ranked_pos_set:
            type4_positions.append(pos)
    type4_positions.sort()

    meta = (
        f"Frame {frame} &bull; {len(atoms)} atoms &bull; "
        f"Top {top_n} ranked beads (spacefill) &bull; "
        f"green=type-1, faint orange=type-4"
    )
    html = IMAMURA_BEAD_PCA_NGL_HTML.format(
        title=title,
        meta_text=meta,
        highlights_json=json.dumps(highlights),
        type1_positions_json=json.dumps(type1_positions),
        type4_positions_json=json.dumps(type4_positions),
        pdb_string=pdb_string,
    )
    output_path.write_text(html, encoding="utf-8")

    pdb_out: Optional[Path] = None
    if write_pdb:
        pdb_out = output_path.with_suffix(".pdb")
        atoms.write(str(pdb_out))

    return output_path, pdb_out


def two_stage_imamura_clustering(
    pca_scores: np.ndarray,
    *,
    n_microclusters: int = DEFAULT_N_MICROCLUSTERS,
    n_macrostates: int = DEFAULT_N_MACROSTATES,
    batch_size: int = 1000,
    random_state: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, int], np.ndarray]:
    """
    MiniBatchKMeans microclustering followed by Ward merging to macrostates.

    Returns micro_labels, macro_labels, micro_to_macro map, linkage_matrix.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.cluster import MiniBatchKMeans

    n_samples = pca_scores.shape[0]
    n_micro = min(n_microclusters, n_samples)
    if n_micro < 2:
        micro = np.zeros(n_samples, dtype=int)
        macro = np.zeros(n_samples, dtype=int)
        return micro, macro, {0: 0}, np.zeros((0, 4))

    km = MiniBatchKMeans(
        n_clusters=n_micro,
        batch_size=min(batch_size, max(n_samples, 1)),
        random_state=random_state,
        n_init=3,
    )
    micro_labels = km.fit_predict(pca_scores)

    occupied = sorted(set(int(x) for x in micro_labels))
    centroids = np.array([
        pca_scores[micro_labels == k].mean(axis=0) for k in occupied
    ], dtype=np.float64)

    n_macro = min(n_macrostates, len(occupied))
    if n_macro < 2:
        micro_to_macro = {k: 0 for k in occupied}
        macro_labels = np.zeros(n_samples, dtype=int)
        return micro_labels, macro_labels, micro_to_macro, np.zeros((0, 4))

    Z = linkage(centroids, method="ward")
    local_macro = fcluster(Z, t=n_macro, criterion="maxclust") - 1
    micro_to_macro = {occupied[i]: int(local_macro[i]) for i in range(len(occupied))}
    macro_labels = np.array([micro_to_macro[int(m)] for m in micro_labels], dtype=int)
    return micro_labels, macro_labels, micro_to_macro, Z


def cluster_imamura_features(
    feature_matrix: np.ndarray,
    *,
    config: Optional[ImamuraMSMConfig] = None,
    traj_ids: Optional[np.ndarray] = None,
    tlica_lag_frames: int = 1,
    tlica_lag_ns: Optional[float] = None,
) -> ImamuraClusteringResult:
    """Dimensionality reduction (PCA or tlICA) + two-stage clustering."""
    cfg = config or ImamuraMSMConfig()
    method = cfg.dimensionality_reduction

    if method == "tlica":
        lag_ns = (
            float(tlica_lag_ns)
            if tlica_lag_ns is not None
            else float(cfg.tlica_lag_ns or cfg.transition_lag_ns)
        )
        model, scores, evr = fit_imamura_tlica(
            feature_matrix,
            n_components=cfg.n_pca_components,
            traj_ids=traj_ids,
            lag_frames=tlica_lag_frames,
            lag_ns=lag_ns,
            regularization=cfg.tlica_regularization,
        )
        proj_eigs = model.eigenvalues_
        timescales = model.implied_timescales_ns
    elif method == "pca":
        model, scores, evr = fit_imamura_pca(
            feature_matrix,
            n_components=cfg.n_pca_components,
        )
        proj_eigs = None
        timescales = None
    else:
        raise ValueError(
            f"Unknown dimensionality_reduction {method!r}; use 'pca' or 'tlica'."
        )

    micro, macro, micro_to_macro, Z = two_stage_imamura_clustering(
        scores,
        n_microclusters=cfg.n_microclusters,
        n_macrostates=cfg.n_macrostates,
        batch_size=cfg.kmeans_batch_size,
        random_state=cfg.kmeans_random_state,
    )
    return ImamuraClusteringResult(
        pca_model=model,
        pca_scores=scores,
        micro_labels=micro,
        macro_labels=macro,
        micro_to_macro=micro_to_macro,
        linkage_matrix=Z,
        explained_variance_ratio=evr,
        dimensionality_reduction=method,
        projection_eigenvalues=proj_eigs,
        implied_timescales_ns=timescales,
    )


@dataclass
class ImamuraClusteringMetrics:
    """Quality metrics for one Imamura two-stage clustering fit."""

    n_micro_requested: int
    n_micro_occupied: int
    n_macro_requested: int
    n_macro_occupied: int
    pca_variance_sum: float
    silhouette_macro: float
    davies_bouldin_macro: float
    calinski_harabasz_macro: float
    min_macro_population_frac: float
    macro_population_gini: float
    ward_cut_height: float
    composite_score: float


def _population_gini(counts: np.ndarray) -> float:
    """Gini coefficient of state populations (0 = uniform, 1 = maximally skewed)."""
    x = np.asarray(counts, dtype=np.float64)
    x = x[x > 0]
    if x.size < 2:
        return 0.0
    x.sort()
    n = x.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float(
        (2.0 * np.sum(index * x) / (n * np.sum(x))) - (n + 1.0) / n
    )


def _subsample_rows(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    max_rows: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    if n <= max_rows:
        return X, labels
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_rows, replace=False)
    return X[idx], labels[idx]


def _ward_cut_height(linkage_matrix: np.ndarray, n_clusters: int) -> float:
    if linkage_matrix.size == 0 or n_clusters < 2:
        return 0.0
    n_leaves = linkage_matrix.shape[0] + 1
    n_clusters = min(n_clusters, n_leaves)
    if n_clusters < 2:
        return 0.0
    merge_idx = n_leaves - n_clusters
    if merge_idx < 0 or merge_idx >= linkage_matrix.shape[0]:
        return float(linkage_matrix[-1, 2]) if linkage_matrix.shape[0] else 0.0
    return float(linkage_matrix[merge_idx, 2])


def evaluate_imamura_clustering(
    pca_scores: np.ndarray,
    micro_labels: np.ndarray,
    macro_labels: np.ndarray,
    *,
    n_micro_requested: int,
    n_macro_requested: int,
    pca_variance_sum: float,
    linkage_matrix: Optional[np.ndarray] = None,
    silhouette_max_rows: int = 8000,
    random_state: int = 0,
) -> ImamuraClusteringMetrics:
    """
    Score one Imamura clustering fit for parameter sweeps.

    ``composite_score`` rewards high macro silhouette, full macro occupancy,
    and balanced populations; it is used to rank grid points.
    """
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )

    n_micro_occupied = len(set(int(x) for x in micro_labels))
    macro_ids, macro_counts = np.unique(macro_labels, return_counts=True)
    n_macro_occupied = int(len(macro_ids))
    n_samples = macro_labels.shape[0]
    min_macro_frac = (
        float(macro_counts.min() / n_samples) if macro_counts.size else 0.0
    )
    gini = _population_gini(macro_counts)

    sil = float("nan")
    db = float("nan")
    ch = float("nan")
    if n_macro_occupied >= 2 and n_samples > n_macro_occupied:
        X_eval, y_eval = _subsample_rows(
            pca_scores,
            macro_labels,
            max_rows=silhouette_max_rows,
            random_state=random_state,
        )
        try:
            sil = float(silhouette_score(X_eval, y_eval))
        except ValueError:
            sil = float("nan")
        try:
            db = float(davies_bouldin_score(X_eval, y_eval))
        except ValueError:
            db = float("nan")
        try:
            ch = float(calinski_harabasz_score(X_eval, y_eval))
        except ValueError:
            ch = float("nan")

    occupancy_macro = n_macro_occupied / max(n_macro_requested, 1)
    occupancy_micro = n_micro_occupied / max(n_micro_requested, 1)
    balance = max(0.0, 1.0 - gini)
    sil_term = sil if np.isfinite(sil) else -1.0
    composite = (
        sil_term
        * occupancy_macro
        * occupancy_micro
        * balance
        * np.sqrt(max(min_macro_frac, 1e-6))
    )

    ward_h = 0.0
    if linkage_matrix is not None:
        ward_h = _ward_cut_height(linkage_matrix, n_macro_occupied)

    return ImamuraClusteringMetrics(
        n_micro_requested=n_micro_requested,
        n_micro_occupied=n_micro_occupied,
        n_macro_requested=n_macro_requested,
        n_macro_occupied=n_macro_occupied,
        pca_variance_sum=float(pca_variance_sum),
        silhouette_macro=sil,
        davies_bouldin_macro=db,
        calinski_harabasz_macro=ch,
        min_macro_population_frac=min_macro_frac,
        macro_population_gini=gini,
        ward_cut_height=ward_h,
        composite_score=float(composite),
    )


def sweep_imamura_clustering(
    feature_matrix: np.ndarray,
    *,
    n_micro_values: Sequence[int],
    n_macro_values: Sequence[int],
    n_pca_components: int = DEFAULT_PCA_COMPONENTS,
    dimensionality_reduction: str = DEFAULT_DIMENSIONALITY_REDUCTION,
    traj_ids: Optional[np.ndarray] = None,
    tlica_lag_frames: int = 1,
    tlica_lag_ns: float = DEFAULT_TRANSITION_LAG_NS,
    tlica_regularization: float = DEFAULT_TLICA_REGULARIZATION,
    kmeans_batch_size: int = 1000,
    kmeans_random_state: int = 0,
    silhouette_max_rows: int = 8000,
) -> Any:
    """
    Grid search over MiniBatchKMeans micro count and Ward macro count.

    Returns a :class:`pandas.DataFrame` with one row per (n_micro, n_macro).
    PCA or tlICA is fit once; only the two-stage clustering step is repeated.
    """
    import pandas as pd

    if dimensionality_reduction == "tlica":
        _, proj_scores, evr = fit_imamura_tlica(
            feature_matrix,
            n_components=n_pca_components,
            traj_ids=traj_ids,
            lag_frames=tlica_lag_frames,
            lag_ns=tlica_lag_ns,
            regularization=tlica_regularization,
        )
    elif dimensionality_reduction == "pca":
        _, proj_scores, evr = fit_imamura_pca(
            feature_matrix,
            n_components=n_pca_components,
        )
    else:
        raise ValueError(
            f"Unknown dimensionality_reduction {dimensionality_reduction!r}; "
            "use 'pca' or 'tlica'."
        )
    proj_var = float(np.sum(evr))

    rows: List[Dict[str, Any]] = []
    for n_micro in n_micro_values:
        for n_macro in n_macro_values:
            micro, macro, _, Z = two_stage_imamura_clustering(
                proj_scores,
                n_microclusters=int(n_micro),
                n_macrostates=int(n_macro),
                batch_size=kmeans_batch_size,
                random_state=kmeans_random_state,
            )
            metrics = evaluate_imamura_clustering(
                proj_scores,
                micro,
                macro,
                n_micro_requested=int(n_micro),
                n_macro_requested=int(n_macro),
                pca_variance_sum=proj_var,
                linkage_matrix=Z,
                silhouette_max_rows=silhouette_max_rows,
                random_state=kmeans_random_state,
            )
            rows.append({
                "dimensionality_reduction": dimensionality_reduction,
                "n_pca": n_pca_components,
                "tlica_lag_ns": tlica_lag_ns if dimensionality_reduction == "tlica" else np.nan,
                "tlica_lag_frames": (
                    int(tlica_lag_frames) if dimensionality_reduction == "tlica" else np.nan
                ),
                "n_micro": int(n_micro),
                "n_macro": int(n_macro),
                **{
                    k: v
                    for k, v in metrics.__dict__.items()
                    if k not in ("n_micro_requested", "n_macro_requested")
                },
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["composite_score", "silhouette_macro"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return df


def load_imamura_features_csv(
    path: Union[str, Path],
    *,
    n_type1_beads: int = DEFAULT_N_TYPE1_BEADS,
    n_type4_beads: int = DEFAULT_N_TYPE4_BEADS,
) -> Tuple[Any, List[str]]:
    """Load ``imamura_features.csv`` and return (dataframe, feature column names)."""
    import pandas as pd

    df = pd.read_csv(path)
    feat_names = [c for c in df.columns if c.startswith("v1_") or c.startswith("v4_")]
    if not feat_names:
        feat_names = imamura_feature_names(n_type1_beads, n_type4_beads)
    return df, feat_names


def imamura_pair_index_table(
    bead_spec: ImamuraBeadSpec,
    *,
    n_type1: int,
    n_type4: int,
    universe: Any = None,
) -> Any:
    """Describe canonical pair indices stored in :class:`ImamuraPairTrace`."""
    import pandas as pd

    def _atom_values(kind: str, bead_index: int) -> Tuple[int, ...]:
        if bead_spec.uses_ring_centroids:
            if kind == "v1":
                rings = bead_spec.type1_ring_groups or []
                return (
                    tuple(int(value) for value in rings[bead_index])
                    if bead_index < len(rings)
                    else ()
                )
            atoms = bead_spec.type4_atom_ids or ()
            return (int(atoms[bead_index]),) if bead_index < len(atoms) else ()
        if universe is None:
            return ()
        selection = (
            bead_spec.type1_selection if kind == "v1" else bead_spec.type4_selection
        )
        group = universe.select_atoms(selection) if selection else []
        if bead_index >= len(group):
            return ()
        return (int(group[bead_index].ix),)

    def _monomer_index(kind: str, bead_index: int) -> Optional[int]:
        if kind == "v1":
            return bead_index if bead_index < n_type1 else None
        if n_type1 > 0 and n_type4 % n_type1 == 0:
            return bead_index // (n_type4 // n_type1)
        return None

    rows: List[Dict[str, Any]] = []
    for block, count in (("v1", n_type1), ("v4", n_type4)):
        for pair_index, (bead_i, bead_j) in enumerate(
            combinations(range(count), 2)
        ):
            values_i = _atom_values(block, bead_i)
            values_j = _atom_values(block, bead_j)
            row: Dict[str, Any] = {
                "feature_block": block,
                "pair_index": pair_index,
                "bead_i": bead_i,
                "bead_j": bead_j,
                "bead_i_monomer_index": _monomer_index(block, bead_i),
                "bead_j_monomer_index": _monomer_index(block, bead_j),
                "bead_i_spec_atom_values": ";".join(map(str, values_i)),
                "bead_j_spec_atom_values": ";".join(map(str, values_j)),
            }
            if universe is not None and values_i and values_j:
                atoms_i = universe.atoms[list(values_i)]
                atoms_j = universe.atoms[list(values_j)]
                row.update(
                    {
                        "bead_i_atom_ids": ";".join(
                            str(int(value)) for value in atoms_i.ids
                        ),
                        "bead_j_atom_ids": ";".join(
                            str(int(value)) for value in atoms_j.ids
                        ),
                        "bead_i_atom_names": ";".join(map(str, atoms_i.names)),
                        "bead_j_atom_names": ";".join(map(str, atoms_j.names)),
                        "bead_i_resids": ";".join(
                            str(int(value)) for value in atoms_i.resids
                        ),
                        "bead_j_resids": ";".join(
                            str(int(value)) for value in atoms_j.resids
                        ),
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def write_imamura_pair_trace(
    trace: ImamuraPairTrace,
    output_dir: Union[str, Path],
    bead_spec: ImamuraBeadSpec,
    *,
    n_type1: int,
    n_type4: int,
    universe: Any = None,
) -> Dict[str, Path]:
    """Persist compact exact pair identities without rereading trajectories."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "pair_identity_trace.npz"
    np.savez(
        trace_path,
        format_version=np.asarray([1], dtype=np.uint8),
        traj_ids=np.asarray(trace.traj_ids, dtype=str),
        frames=np.asarray(trace.frames, dtype=np.int64),
        v1_pair_indices=trace.v1_pair_indices,
        v4_pair_indices=trace.v4_pair_indices,
    )
    map_path = output_dir / "pair_index_map.csv"
    imamura_pair_index_table(
        bead_spec,
        n_type1=n_type1,
        n_type4=n_type4,
        universe=universe,
    ).to_csv(map_path, index=False)
    return {"pair_identity_trace": trace_path, "pair_index_map": map_path}


def load_imamura_pair_trace(path: Union[str, Path]) -> ImamuraPairTrace:
    """Load a trace written by :func:`write_imamura_pair_trace`."""
    with np.load(path, allow_pickle=False) as data:
        return ImamuraPairTrace(
            traj_ids=data["traj_ids"].astype(str),
            frames=data["frames"].astype(np.int64, copy=False),
            v1_pair_indices=data["v1_pair_indices"],
            v4_pair_indices=data["v4_pair_indices"],
        )


def _infer_dt_ps(universe: Any, sample_frames: Sequence[int]) -> float:
    """Estimate trajectory spacing in ps from consecutive frame times."""
    if len(universe.trajectory) < 2:
        return 1.0
    idx = [i for i in sample_frames if 0 <= i < len(universe.trajectory)]
    if len(idx) < 2:
        idx = [0, min(1, len(universe.trajectory) - 1)]
    t0 = float(universe.trajectory[idx[0]].time)
    t1 = float(universe.trajectory[idx[1]].time)
    dt = t1 - t0
    if dt <= 0:
        dt = float(getattr(universe.trajectory, "dt", 0.0) or 0.0)
    return dt if dt > 0 else 1.0


def lag_frames_from_ns(
    lag_ns: float,
    dt_ps: float,
    stride: int = 1,
) -> int:
    """Convert lag time in ns to frame index offset (minimum 1)."""
    lag_ps = lag_ns * 1000.0
    effective_dt = dt_ps * max(stride, 1)
    if effective_dt <= 0:
        return 1
    return max(1, int(round(lag_ps / effective_dt)))


def count_lagged_transitions(
    labeled_df: Any,
    n_states: int,
    *,
    lag_frames: int = 1,
    label_col: str = "macro_label",
    traj_col: str = "traj_id",
    frame_col: str = "frame",
) -> ImamuraTransitionResult:
    """
    Count state transitions using a lag time in frames (Imamura: 2 ns).

    A transition ``i → j`` is counted when ``label[t] == i`` and
    ``label[t + lag_frames] == j`` within each trajectory.
    """
    import pandas as pd

    counts: Dict[Tuple[int, int], int] = {}
    lag = max(1, int(lag_frames))

    for _, grp in labeled_df.groupby(traj_col):
        grp = grp.sort_values(frame_col)
        labels = grp[label_col].astype(int).to_numpy()
        for i in range(len(labels) - lag):
            a, b = int(labels[i]), int(labels[i + lag])
            if a == b:
                continue
            if a < 0 or b < 0 or a >= n_states or b >= n_states:
                continue
            counts[(a, b)] = counts.get((a, b), 0) + 1

    rows = [
        {"from_state": a, "to_state": b, "count": n}
        for (a, b), n in sorted(counts.items())
    ]
    long_df = pd.DataFrame(rows, columns=["from_state", "to_state", "count"])
    count_mat = np.zeros((n_states, n_states), dtype=np.int64)
    for _, row in long_df.iterrows():
        count_mat[int(row["from_state"]), int(row["to_state"])] = int(row["count"])

    prob_mat = transition_probability_matrix(count_mat)
    count_df = transition_matrix_dataframe(count_mat)
    prob_df = transition_matrix_dataframe(prob_mat, value_name="probability")

    return ImamuraTransitionResult(
        count_matrix=count_mat,
        probability_matrix=prob_mat,
        count_df=count_df,
        prob_df=prob_df,
        long_df=long_df,
        lag_frames=lag,
        lag_ns=float("nan"),
    )


def plot_macrostate_dendrogram(
    linkage_matrix: np.ndarray,
    output_path: Union[str, Path],
    *,
    n_labels: int = 30,
    title: str = "Ward dendrogram (microcluster centroids)",
) -> Path:
    """Save a dendrogram for the Ward merge step."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    if linkage_matrix.size:
        dendro_kwargs: Dict[str, Any] = {"color_threshold": None}
        if linkage_matrix.shape[0] > n_labels:
            dendro_kwargs["truncate_mode"] = "lastp"
            dendro_kwargs["p"] = n_labels
        dendrogram(linkage_matrix, ax=ax, **dendro_kwargs)
    ax.set_title(title)
    ax.set_xlabel("Microcluster index")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def microstate_pca_centroids(labeled_df: "Any") -> "Any":
    """Per-microcluster mean projection coordinates, macro label, and frame count."""
    import pandas as pd

    if any(c.startswith("TIC") for c in labeled_df.columns):
        pc_cols = [c for c in labeled_df.columns if c.startswith("TIC")]
    else:
        pc_cols = [c for c in labeled_df.columns if c.startswith("PC")]

    required = {"micro_label", "macro_label", *pc_cols[:2]}
    missing = required - set(labeled_df.columns)
    if missing:
        raise ValueError(f"labeled_df missing columns: {sorted(missing)}")
    agg: Dict[str, tuple] = {
        "macro_label": ("macro_label", "first"),
        "n_frames": ("micro_label", "size"),
    }
    for col in pc_cols:
        agg[col] = (col, "mean")

    return (
        labeled_df.groupby("micro_label", as_index=False)
        .agg(**agg)
        .sort_values("micro_label")
        .reset_index(drop=True)
    )


def plot_imamura_pca_microstates(
    labeled_df: "Any",
    output_path: Union[str, Path],
    *,
    explained_variance_ratio: Optional[np.ndarray] = None,
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    title: str = "Imamura microclusters in PCA space",
    annotate: bool = False,
) -> Path:
    """
    Scatter PC1 vs PC2 for microcluster centroids, colored by macrostate label.

    Each point is the mean PCA coordinate of one MiniBatchKMeans microcluster;
    color encodes the Ward-merged macrostate.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    centroids = microstate_pca_centroids(labeled_df)
    if centroids.empty:
        raise ValueError("No microcluster centroids to plot.")

    n_macro = int(centroids["macro_label"].max()) + 1
    fig, ax = plt.subplots(figsize=(9, 7))

    sizes = np.clip(centroids["n_frames"].to_numpy(dtype=float), 1.0, None)
    size_pts = 20.0 + 180.0 * (sizes / sizes.max())

    scatter = ax.scatter(
        centroids[pc_x],
        centroids[pc_y],
        c=centroids["macro_label"],
        s=size_pts,
        cmap="tab20" if n_macro <= 20 else "nipy_spectral",
        vmin=0,
        vmax=max(n_macro - 1, 1),
        alpha=0.85,
        edgecolors="0.25",
        linewidths=0.4,
    )

    if annotate:
        for _, row in centroids.iterrows():
            ax.annotate(
                str(int(row["micro_label"])),
                (row[pc_x], row[pc_y]),
                fontsize=6,
                ha="center",
                va="center",
                color="0.15",
            )

    xlabel, ylabel = pc_x, pc_y
    if explained_variance_ratio is not None and len(explained_variance_ratio) >= 2:
        xlabel = f"{pc_x} ({explained_variance_ratio[0]:.1%} var.)"
        ylabel = f"{pc_y} ({explained_variance_ratio[1]:.1%} var.)"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Macrostate label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_imamura_artifacts(
    output_dir: Union[str, Path],
    feature_result: ImamuraFeatureResult,
    clustering: ImamuraClusteringResult,
    transitions: ImamuraTransitionResult,
    *,
    config: Optional[ImamuraMSMConfig] = None,
    universe: Any = None,
    monomer_selections: Optional[Sequence[str]] = None,
    bead_spec: Optional[ImamuraBeadSpec] = None,
    top_pca_beads: int = 5,
) -> Dict[str, Path]:
    """Write CSV/plots for the full Imamura MSM pipeline."""
    import pandas as pd

    cfg = config or ImamuraMSMConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    feat_path = output_dir / "imamura_features.csv"
    feature_result.dataframe.to_csv(feat_path, index=False)
    written["features"] = feat_path

    effective_bead_spec = bead_spec or feature_result.bead_spec
    if feature_result.pair_trace is not None and effective_bead_spec is not None:
        trace = feature_result.pair_trace
        table = feature_result.dataframe
        if len(trace.frames) != len(table):
            raise ValueError(
                "Pair-identity trace length does not match Imamura feature table"
            )
        if "traj_id" in table and not np.array_equal(
            trace.traj_ids.astype(str),
            table["traj_id"].astype(str).to_numpy(),
        ):
            raise ValueError(
                "Pair-identity trace trajectory IDs do not match feature rows"
            )
        if "frame" in table and not np.array_equal(
            trace.frames,
            table["frame"].to_numpy(dtype=np.int64),
        ):
            raise ValueError(
                "Pair-identity trace frame indices do not match feature rows"
            )
        written.update(
            write_imamura_pair_trace(
                trace,
                output_dir,
                effective_bead_spec,
                n_type1=cfg.n_type1_beads,
                n_type4=cfg.n_type4_beads,
                universe=universe,
            )
        )

    if feature_result.formation_frames:
        ff_path = output_dir / "formation_frames.csv"
        pd.DataFrame([
            {"traj_id": k, "formation_frame": v}
            for k, v in sorted(feature_result.formation_frames.items())
        ]).to_csv(ff_path, index=False)
        written["formation_frames"] = ff_path

    labeled = feature_result.dataframe.copy()
    labeled["micro_label"] = clustering.micro_labels
    labeled["macro_label"] = clustering.macro_labels
    proj_cols = imamura_projection_column_names(
        clustering.dimensionality_reduction,
        clustering.pca_scores.shape[1],
    )
    for i, col in enumerate(proj_cols):
        labeled[col] = clustering.pca_scores[:, i]

    labeled_path = output_dir / "frame_states.csv"
    labeled.to_csv(labeled_path, index=False)
    written["frame_states"] = labeled_path

    if clustering.dimensionality_reduction == "tlica":
        dim_info = pd.DataFrame({
            "component": proj_cols,
            "eigenvalue": clustering.projection_eigenvalues,
            "normalized_weight": clustering.explained_variance_ratio,
            "implied_timescale_ns": clustering.implied_timescales_ns,
            "tlica_lag_ns": clustering.pca_model.lag_ns,
            "tlica_lag_frames": clustering.pca_model.lag_frames,
        })
        dim_path = output_dir / "tlica_eigenvalues.csv"
        dim_info.to_csv(dim_path, index=False)
        written["tlica_eigenvalues"] = dim_path
    else:
        dim_info = pd.DataFrame({
            "component": proj_cols,
            "explained_variance_ratio": clustering.explained_variance_ratio,
        })
        dim_path = output_dir / "pca_explained_variance.csv"
        dim_info.to_csv(dim_path, index=False)
        written["pca_explained_variance"] = dim_path

    micro_macro = pd.DataFrame([
        {"micro_label": micro, "macro_label": macro}
        for micro, macro in sorted(clustering.micro_to_macro.items())
    ])
    mm_path = output_dir / "micro_to_macro.csv"
    micro_macro.to_csv(mm_path, index=False)
    written["micro_to_macro"] = mm_path

    n_states = cfg.n_macrostates
    micro_pca = microstate_pca_centroids(labeled)
    micro_pca_path = output_dir / "micro_pca_centroids.csv"
    micro_pca.to_csv(micro_pca_path, index=False)
    written["micro_pca_centroids"] = micro_pca_path

    pca_micro = plot_imamura_pca_microstates(
        labeled,
        output_dir / "pca_microstates.png",
        explained_variance_ratio=clustering.explained_variance_ratio,
        pc_x=proj_cols[0],
        pc_y=proj_cols[1] if len(proj_cols) > 1 else proj_cols[0],
        title=(
            f"Imamura microclusters ({proj_cols[0]} vs {proj_cols[1]}, "
            f"{clustering.dimensionality_reduction}, "
            f"{len(micro_pca)} micros → {cfg.n_macrostates} macrostates)"
        ),
    )
    written["pca_microstates"] = pca_micro

    trans = transitions.long_df.rename(
        columns={"from_state": "from_cluster", "to_state": "to_cluster"}
    )
    t_path = output_dir / "state_transitions.csv"
    trans.to_csv(t_path, index=False)
    written["transitions"] = t_path

    c_path = output_dir / "state_transition_matrix.csv"
    transitions.count_df.to_csv(c_path)
    written["transition_matrix"] = c_path

    p_path = output_dir / "state_transition_prob.csv"
    transitions.prob_df.to_csv(p_path)
    written["transition_prob"] = p_path

    heat = plot_transition_heatmap(
        trans,
        n_states,
        output_dir / "transition_matrix.png",
        title=f"Imamura macrostate transitions (lag={transitions.lag_frames} frames)",
    )
    written["transition_heatmap"] = heat

    net = plot_transition_network_diagram(
        transitions.count_matrix,
        output_dir / "transition_network.png",
        title=(
            f"Imamura transition network "
            f"({n_states} states, lag={transitions.lag_frames} frames)"
        ),
    )
    written["transition_network"] = net

    if clustering.linkage_matrix.size:
        dendro = plot_macrostate_dendrogram(
            clustering.linkage_matrix,
            output_dir / "microcluster_ward_dendrogram.png",
            title=f"Ward merge of {len(clustering.micro_to_macro)} microclusters",
        )
        written["dendrogram"] = dendro

    summary_path = output_dir / "pipeline_summary.txt"
    n_frames = len(labeled)
    total_time_us = float("nan")
    if "time_ps" in labeled.columns and n_frames > 1:
        by_traj = labeled.groupby("traj_id")["time_ps"]
        total_ps = sum(float(g.max() - g.min()) for _, g in by_traj)
        total_time_us = total_ps / 1e6

    if clustering.dimensionality_reduction == "tlica":
        dim_line = (
            f"tlICA components: {clustering.pca_scores.shape[1]} "
            f"(lag={clustering.pca_model.lag_frames} frames, "
            f"{clustering.pca_model.lag_ns:g} ns)"
        )
        eig_line = (
            f"Leading TIC eigenvalues: "
            f"{', '.join(f'{v:.3f}' for v in clustering.projection_eigenvalues[:5])}"
            if clustering.projection_eigenvalues is not None
            else "Leading TIC eigenvalues: n/a"
        )
    else:
        dim_line = f"PCA components: {clustering.pca_scores.shape[1]}"
        eig_line = (
            f"Explained variance (PC1–PC{clustering.pca_scores.shape[1]}): "
            f"{clustering.explained_variance_ratio.sum():.3f}"
        )

    summary_path.write_text(
        "\n".join([
            "Imamura MSM pipeline summary",
            "==============================",
            f"Frames analyzed: {n_frames}",
            f"Trajectories: {labeled['traj_id'].nunique() if n_frames else 0}",
            f"Feature dimension: {len(feature_result.feature_names)}",
            f"Dimensionality reduction: {clustering.dimensionality_reduction}",
            dim_line,
            eig_line,
            f"Microclusters: {cfg.n_microclusters}",
            f"Macrostates: {cfg.n_macrostates}",
            f"Transition lag: {transitions.lag_frames} frames",
            f"Total post-formation time (approx): {total_time_us:.3f} µs"
            if not np.isnan(total_time_us)
            else "Total post-formation time: n/a",
        ]),
        encoding="utf-8",
    )
    written["summary"] = summary_path

    spec = bead_spec or feature_result.bead_spec
    if universe is not None and monomer_selections and spec is not None:
        bead_art = write_imamura_bead_importance_artifacts(
            output_dir,
            feature_result,
            clustering,
            universe,
            monomer_selections,
            spec,
            top_n=top_pca_beads,
            importance="cluster",
        )
        written.update(bead_art)

    return written


def run_imamura_msm_pipeline(
    topology: Union[str, Path],
    trajectory_paths: Sequence[Union[str, Path]],
    bead_spec: ImamuraBeadSpec,
    output_dir: Union[str, Path],
    *,
    config: Optional[ImamuraMSMConfig] = None,
    traj_format: Optional[str] = None,
    formation_frames: Optional[Dict[str, int]] = None,
    truncate_at_formation: bool = False,
    detect_formation: bool = True,
    monomer_selections: Optional[Sequence[str]] = None,
    stride: int = 1,
    n_jobs: int = 1,
    dt_ps: Optional[float] = None,
) -> Dict[str, Any]:
    """
    End-to-end Imamura MSM workflow for multiple trajectories.

    Returns a dict with feature, clustering, transition results and artifact paths.
    """
    import MDAnalysis as mda

    cfg = config or ImamuraMSMConfig()
    traj_paths = [Path(p) for p in trajectory_paths]
    frames_map = dict(formation_frames or {})

    if truncate_at_formation and detect_formation and not frames_map:
        if monomer_selections is None:
            u0 = mda.Universe(
                str(topology),
                str(traj_paths[0]),
                **({"format": traj_format} if traj_format else {}),
            )
            resolved = resolve_selections(u0, gsa_resname="MOL", n_monomers=6, auto_tooth=False)
            monomer_selections = resolved.monomer_selections
        frames_map = detect_formation_frames(
            topology,
            traj_paths,
            monomer_selections,
            n_monomers=len(monomer_selections) if monomer_selections else None,
            traj_format=traj_format,
            contact_cutoff=cfg.contact_cutoff,
            min_active_interfaces=cfg.formation_min_active_interfaces,
            sustain_frames=cfg.formation_sustain_frames,
            stride=stride,
        )

    features = extract_imamura_features_multi(
        topology,
        traj_paths,
        bead_spec,
        traj_format=traj_format,
        formation_frames=frames_map if truncate_at_formation else None,
        truncate_at_formation=truncate_at_formation,
        stride=stride,
        n_type1=cfg.n_type1_beads,
        n_type4=cfg.n_type4_beads,
        n_jobs=n_jobs,
    )
    features.formation_frames = frames_map

    feat_cols = features.feature_names
    X = features.dataframe[feat_cols].to_numpy(dtype=np.float64)
    if X.shape[0] < 2:
        raise ValueError("Need at least two frames for dimensionality reduction.")

    traj_ids = (
        features.dataframe["traj_id"].to_numpy()
        if "traj_id" in features.dataframe.columns
        else None
    )
    tlica_lag_ns = cfg.tlica_lag_ns or cfg.transition_lag_ns
    tlica_lag_frames = lag_frames_from_ns(tlica_lag_ns, dt_ps or 1.0, stride=stride)
    clustering = cluster_imamura_features(
        X,
        config=cfg,
        traj_ids=traj_ids,
        tlica_lag_frames=tlica_lag_frames,
        tlica_lag_ns=tlica_lag_ns,
    )

    if dt_ps is None:
        u_ref = mda.Universe(
            str(topology),
            str(traj_paths[0]),
            **({"format": traj_format} if traj_format else {}),
        )
        sample = features.dataframe["frame"].head(2).tolist() if "frame" in features.dataframe else [0, 1]
        dt_ps = _infer_dt_ps(u_ref, sample)

    lag = lag_frames_from_ns(cfg.transition_lag_ns, dt_ps, stride=stride)
    labeled = features.dataframe.copy()
    labeled["macro_label"] = clustering.macro_labels

    transitions = count_lagged_transitions(
        labeled,
        cfg.n_macrostates,
        lag_frames=lag,
    )
    transitions.lag_ns = cfg.transition_lag_ns

    u_plot = mda.Universe(
        str(topology),
        str(traj_paths[0]),
        **({"format": traj_format} if traj_format else {}),
    )
    if monomer_selections is None:
        resolved = resolve_selections(u_plot, gsa_resname="MOL", n_monomers=6, auto_tooth=False)
        monomer_selections = resolved.monomer_selections

    artifacts = write_imamura_artifacts(
        output_dir,
        features,
        clustering,
        transitions,
        config=cfg,
        universe=u_plot,
        monomer_selections=monomer_selections or [],
        bead_spec=bead_spec,
    )

    return {
        "features": features,
        "clustering": clustering,
        "transitions": transitions,
        "artifacts": artifacts,
        "formation_frames": frames_map,
        "dt_ps": dt_ps,
        "lag_frames": lag,
    }
