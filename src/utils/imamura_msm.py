"""
Imamura et al. nanocube MSM workflow utilities.

Reproduces the coarse-grained self-assembly analysis pipeline described in
Imamura, Yamamoto & Sato, *Chem. Phys. Lett.* **742**, 137135 (2020):

1. Truncate trajectories at nanocube formation.
2. Build 168-D feature vectors from sorted inter-bead distances
   (type-1 central benzenes → 15 distances; type-4 methyls → 153).
3. PCA to five components.
4. Two-stage clustering: MiniBatchKMeans microstates → Ward macrostates.
5. Lagged transition counting for a kinetic network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
    select_central_methyl_atoms,
    select_endpoint_type4_atoms,
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


@dataclass
class ImamuraMSMConfig:
    """Hyperparameters for the Imamura MSM pipeline."""

    n_type1_beads: int = DEFAULT_N_TYPE1_BEADS
    n_type4_beads: int = DEFAULT_N_TYPE4_BEADS
    n_pca_components: int = DEFAULT_PCA_COMPONENTS
    n_microclusters: int = DEFAULT_N_MICROCLUSTERS
    n_macrostates: int = DEFAULT_N_MACROSTATES
    transition_lag_ns: float = DEFAULT_TRANSITION_LAG_NS
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


@dataclass
class ImamuraClusteringResult:
    """Outputs from PCA + two-stage clustering."""

    pca_model: Any
    pca_scores: np.ndarray
    micro_labels: np.ndarray
    macro_labels: np.ndarray
    micro_to_macro: Dict[int, int]
    linkage_matrix: np.ndarray
    explained_variance_ratio: np.ndarray


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
    type4_per_monomer: int = 3


def make_imamura_endpoints_finder(
    *,
    extend_to_ring_atoms: bool = True,
) -> Any:
    """Construct :class:`EndpointsFinder` for Imamura type-4 bead picking."""
    from src.EndpointAnalyzer import EndpointsFinder

    return EndpointsFinder(extend_to_ring_atoms=extend_to_ring_atoms)


def resolve_monomer_imamura_beads(
    universe: Any,
    mon_sel: str,
    *,
    ring_calc: Optional[RingCenterCalculator] = None,
    endpoints_finder: Optional[Any] = None,
    resolution: Optional[ImamuraBeadResolutionOptions] = None,
    n_methyl_per_monomer: int = 3,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Per-monomer Imamura beads via RDKit substructure + endpoint analysis.

    Type-1 uses the geometric center of :meth:`RingCenterCalculator.find_center_benzene_ring`.

    Type-4 source (``resolution.type4_source``):

    * ``methyl`` — bond-step methyl selection (:func:`select_central_methyl_atoms`)
    * ``endpoint`` — raw :class:`EndpointsFinder` output (optionally excluding
      the central benzene ring when ``endpoint_exclude_center_benzene`` is True)
    """
    from src.EndpointAnalyzer import EndpointAnalyzer

    opts = resolution or ImamuraBeadResolutionOptions(type4_per_monomer=n_methyl_per_monomer)
    n_type4 = opts.type4_per_monomer
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
        type4_rdkit = select_central_methyl_atoms(
            mol,
            sel.positions,
            n_type4,
            center_ring_rdkit_indices=center_ring.rdkit_indices,
            ring_calculator=ring_calc,
            endpoint_rdkit_indices=ep_rdkit,
        )
        if len(type4_rdkit) != n_type4:
            raise ValueError(
                f"Monomer '{mon_sel}': methyl selection yielded "
                f"{len(type4_rdkit)} atoms; expected {n_type4}."
            )
    elif opts.type4_source == "endpoint":
        type4_rdkit = select_endpoint_type4_atoms(
            ep_rdkit,
            center_ring.rdkit_indices,
            n=n_type4,
            exclude_center_benzene=opts.endpoint_exclude_center_benzene,
            positions=sel.positions,
        )
    else:
        raise ValueError(
            f"Unknown type4_source {opts.type4_source!r}; use 'methyl' or 'endpoint'"
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
    n_methyl_per_monomer: int = 3,
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

    opts = resolution or ImamuraBeadResolutionOptions(
        type4_per_monomer=n_methyl_per_monomer,
    )

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
            if len(t4) != opts.type4_per_monomer:
                raise ValueError(
                    f"Monomer '{mon_sel}' has {len(t4)} type-4 atoms "
                    f"(name {type4_atom_name}); expected {opts.type4_per_monomer}."
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
            n_methyl_per_monomer=n_methyl_per_monomer,
        )
        type1_groups.append(ring_ids)
        type4_ids.extend(type4_monomer_ids)

    return ImamuraBeadSpec(
        type1_ring_groups=type1_groups,
        type4_atom_ids=tuple(type4_ids),
        type4_source=opts.type4_source,
        endpoint_extend_ring=opts.endpoint_extend_ring,
        endpoint_exclude_center_benzene=opts.endpoint_exclude_center_benzene,
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
        }
    else:
        payload = {
            "mode": "selections",
            "type1_selection": bead_spec.type1_selection,
            "type4_selection": bead_spec.type4_selection,
        }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


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
    for idx in ring_rdkit:
        highlight_colors[idx] = (0.0, 0.7, 0.0)
    for idx in methyl_rdkit:
        highlight_colors[idx] = (1.0, 0.5, 0.0)
    all_highlight = list(dict.fromkeys(ring_rdkit + methyl_rdkit))

    img = None
    try:
        from rdkit.Chem import rdMolDraw2D
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
            highlightAtoms=all_highlight,
            highlightColor=(1.0, 0.5, 0.0),
        )

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
        ax.annotate(
            f"T1:{mda_id}",
            xy=(x * scale + offset_x, -y * scale + offset_y),
            fontsize=9,
            color="darkgreen",
            fontweight="bold",
        )
    for rdkit_idx, mda_id in zip(methyl_rdkit, methyl_mda_ids):
        if rdkit_idx >= len(xy):
            continue
        x, y = xy[rdkit_idx]
        ax.annotate(
            f"T4:{mda_id}",
            xy=(x * scale + offset_x, -y * scale + offset_y),
            fontsize=9,
            color="darkorange",
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
    n = positions.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)

    dists: List[float] = []
    for i, j in combinations(range(n), 2):
        d = float(np.linalg.norm(positions[i] - positions[j]))
        dists.append(d)
    arr = np.asarray(dists, dtype=np.float64)
    arr.sort()
    arr = arr[::-1]
    return arr


def build_imamura_feature_vector(
    type1_positions: np.ndarray,
    type4_positions: np.ndarray,
    *,
    n_type1: int = DEFAULT_N_TYPE1_BEADS,
    n_type4: int = DEFAULT_N_TYPE4_BEADS,
) -> np.ndarray:
    """Combine sorted type-1 and type-4 distance vectors into v_f (168-D)."""
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

    v1 = sorted_pairwise_distances(type1_positions)
    v4 = sorted_pairwise_distances(type4_positions)
    n1, n4, total = expected_distance_counts(n_type1, n_type4)
    if v1.size != n1 or v4.size != n4:
        raise ValueError(
            f"Distance block sizes ({v1.size}, {v4.size}) != expected ({n1}, {n4})"
        )
    vf = np.concatenate([v1, v4])
    if vf.size != total:
        raise ValueError(f"Feature vector length {vf.size} != expected {total}")
    return vf


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
        out[traj_path.stem] = frame
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
    ):
        super().__init__()
        self.bead_spec = bead_spec
        self.n_type1 = n_type1
        self.n_type4 = n_type4
        self.traj_id = traj_id
        self.start_frame = start_frame
        self.feature_names = imamura_feature_names(n_type1, n_type4)
        bead_spec.validate(n_type1=n_type1, n_type4=n_type4)

        self.results["rows"] = []
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
            "frame_call_count": ResultsGroup.sum_values,
        })

    def on_frame_start(self, iterator: TrajectoryIterator) -> None:
        pass

    def on_frame(self, ts: Any, frame_idx: int, universe: Any) -> None:
        type1, type4 = bead_positions_from_spec(universe, self.bead_spec)
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
) -> Any:
    """Extract Imamura features for one trajectory via TrajectoryIterator."""
    import pandas as pd

    observer = ImamuraBeadDistanceObserver(
        bead_spec,
        n_type1=n_type1,
        n_type4=n_type4,
        traj_id=traj_id,
        start_frame=start or 0,
    )
    iterator = TrajectoryIterator(universe)
    iterator.subscribe(observer)
    iterator.iterate(start=start, stop=stop, step=step, n_jobs=n_jobs)

    rows = observer.results.get("rows", [])
    if not rows:
        return pd.DataFrame(columns=["traj_id", "frame", "time_ps", *observer.feature_names])
    return pd.DataFrame(rows)


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
) -> ImamuraFeatureResult:
    """Extract features from multiple trajectories, optionally post-formation."""
    import MDAnalysis as mda
    import pandas as pd

    feature_names = imamura_feature_names(n_type1, n_type4)
    frames_map = formation_frames or {}
    parts: List[Any] = []

    for traj_path in trajectory_paths:
        traj_path = Path(traj_path)
        traj_id = traj_path.stem
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
        )
        if not df.empty:
            parts.append(df)

    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["traj_id", "frame", "time_ps", *feature_names]
    )
    return ImamuraFeatureResult(
        dataframe=combined,
        feature_names=feature_names,
        formation_frames=frames_map,
        bead_spec=bead_spec,
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
) -> ImamuraClusteringResult:
    """PCA + two-stage clustering on a feature matrix."""
    cfg = config or ImamuraMSMConfig()
    pca, scores, evr = fit_imamura_pca(
        feature_matrix,
        n_components=cfg.n_pca_components,
    )
    micro, macro, micro_to_macro, Z = two_stage_imamura_clustering(
        scores,
        n_microclusters=cfg.n_microclusters,
        n_macrostates=cfg.n_macrostates,
        batch_size=cfg.kmeans_batch_size,
        random_state=cfg.kmeans_random_state,
    )
    return ImamuraClusteringResult(
        pca_model=pca,
        pca_scores=scores,
        micro_labels=micro,
        macro_labels=macro,
        micro_to_macro=micro_to_macro,
        linkage_matrix=Z,
        explained_variance_ratio=evr,
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
        dendrogram(
            linkage_matrix,
            ax=ax,
            truncate_mode="lastp" if linkage_matrix.shape[0] > n_labels else None,
            p=n_labels if linkage_matrix.shape[0] > n_labels else None,
            color_threshold=None,
        )
    ax.set_title(title)
    ax.set_xlabel("Microcluster index")
    ax.set_ylabel("Ward distance")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def microstate_pca_centroids(labeled_df: "Any") -> "Any":
    """Per-microcluster mean PC coordinates, macro label, and frame count."""
    import pandas as pd

    required = {"micro_label", "macro_label", "PC1", "PC2"}
    missing = required - set(labeled_df.columns)
    if missing:
        raise ValueError(f"labeled_df missing columns: {sorted(missing)}")

    pc_cols = [c for c in labeled_df.columns if c.startswith("PC")]
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
    for i in range(clustering.pca_scores.shape[1]):
        labeled[f"PC{i + 1}"] = clustering.pca_scores[:, i]

    labeled_path = output_dir / "frame_states.csv"
    labeled.to_csv(labeled_path, index=False)
    written["frame_states"] = labeled_path

    pca_info = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(clustering.explained_variance_ratio))],
        "explained_variance_ratio": clustering.explained_variance_ratio,
    })
    pca_path = output_dir / "pca_explained_variance.csv"
    pca_info.to_csv(pca_path, index=False)
    written["pca_explained_variance"] = pca_path

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
        title=(
            f"Imamura microclusters (PC1 vs PC2, "
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

    summary_path.write_text(
        "\n".join([
            "Imamura MSM pipeline summary",
            "==============================",
            f"Frames analyzed: {n_frames}",
            f"Trajectories: {labeled['traj_id'].nunique() if n_frames else 0}",
            f"Feature dimension: {len(feature_result.feature_names)}",
            f"PCA components: {clustering.pca_scores.shape[1]}",
            f"Explained variance (PC1–PC5): "
            f"{clustering.explained_variance_ratio.sum():.3f}",
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
        raise ValueError("Need at least two frames for PCA/clustering.")

    clustering = cluster_imamura_features(X, config=cfg)

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

    artifacts = write_imamura_artifacts(
        output_dir,
        features,
        clustering,
        transitions,
        config=cfg,
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
