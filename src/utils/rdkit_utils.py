from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any, Sequence, TYPE_CHECKING

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdDistGeom
import MDAnalysis as mda

from ..TrajectoryIterator import FrameObserver
from ..Aggregator import ResultsGroup

if TYPE_CHECKING:
    from ..TrajectoryIterator import TrajectoryIterator

def get_3d_coordinates_from_smiles(smiles: str) -> tuple[np.ndarray, list[str]]:
    # Parse SMILES and generate 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates using ETKDGv3
    ps = rdDistGeom.ETKDGv3()
    ps.randomSeed = 0xa100f
    status = rdDistGeom.EmbedMolecule(mol, ps)
    if status != 0:
        # If ETKDGv3 fails, try basic embedding
        status = rdDistGeom.EmbedMolecule(mol)
        if status != 0:
            raise RuntimeError(f"Failed to generate 3D coordinates for SMILES: {smiles}")
    
    # Optimize geometry
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        # If MMFF fails, try UFF
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except:
            pass  # Continue without optimization
    
    # Extract coordinates and elements
    conf = mol.GetConformer()
    n_atoms_mol = mol.GetNumAtoms()
    coords = np.zeros((n_atoms_mol, 3), dtype=float)
    elements_list = []
    
    for i in range(n_atoms_mol):
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]
        atom = mol.GetAtomWithIdx(i)
        elements_list.append(atom.GetSymbol())
    return mol, coords, elements_list


def smiles_to_universe(smiles: str) -> mda.Universe:
    mol, coords, elements_list = get_3d_coordinates_from_smiles(smiles)
    u = mda.Universe(mol)
    return u


@dataclass
class SubstructureInfo:
    """Information about a detected substructure (ring or SMARTS match)."""
    name: str                    # Identifier (e.g., "ring_0", "benzene_1")
    rdkit_indices: Tuple[int, ...]  # Atom indices in RDKit molecule
    mda_indices: Optional[Tuple[int, ...]] = None  # Atom indices in MDAnalysis selection
    smarts: Optional[str] = None  # SMARTS pattern if from pattern matching


@dataclass
class RingMetrics:
    """Metrics computed for a ring/substructure at a single frame."""
    center: np.ndarray           # Geometric center (x, y, z)
    planarity_rmsd: float        # RMSD from best-fit plane (0 = perfectly planar)
    normal_vector: np.ndarray    # Unit normal vector of best-fit plane
    radius: float                # Average distance from center to ring atoms


class RingCenterCalculator:
    """
    Calculate geometric centers and metrics for rings and substructures.
    
    Supports both automatic ring detection and custom SMARTS pattern matching.
    
    Example:
        >>> calc = RingCenterCalculator()
        >>> mol = Chem.MolFromSmiles("c1ccccc1")  # benzene
        >>> rings = calc.find_all_rings(mol)
        >>> # Or use SMARTS patterns
        >>> matches = calc.match_smarts_pattern(mol, "c1ccccc1")
    """
    
    # Common SMARTS patterns for convenience
    COMMON_PATTERNS = {
        "benzene": "c1ccccc1",
        "phenyl": "c1ccccc1",
        "pyridine": "c1ccncc1",
        "naphthalene": "c1ccc2ccccc2c1",
        "cyclopentane": "C1CCCC1",
        "cyclohexane": "C1CCCCC1",
        "methyl": "[CH3]",
        "trifluoromethyl": "[CF3]",
        "amino": "[NH2]",
        "hydroxyl": "[OH]",
        "carboxyl": "C(=O)[OH]",
        "carbonyl": "C=O",
        "nitro": "[N+](=O)[O-]",
        "cyano": "C#N",
    }
    
    def __init__(self):
        """Initialize RingCenterCalculator."""
        pass
    
    def find_all_rings(self, mol: Chem.Mol) -> List[SubstructureInfo]:
        """
        Automatically detect all rings in a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            List of SubstructureInfo for each ring found
        """
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        rings = []
        for i, ring_atoms in enumerate(atom_rings):
            info = SubstructureInfo(
                name=f"ring_{i}",
                rdkit_indices=tuple(ring_atoms),
                smarts=None
            )
            rings.append(info)
        
        return rings
    
    def match_smarts_pattern(
        self, 
        mol: Chem.Mol, 
        smarts: str,
        name: Optional[str] = None
    ) -> List[SubstructureInfo]:
        """
        Match a SMARTS pattern against a molecule.
        
        Args:
            mol: RDKit molecule object
            smarts: SMARTS pattern string
            name: Optional name for the pattern (default: derived from smarts)
            
        Returns:
            List of SubstructureInfo for each match found
        """
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            raise ValueError(f"Invalid SMARTS pattern: {smarts}")
        
        matches = mol.GetSubstructMatches(pattern)
        
        if name is None:
            # Try to find common pattern name
            for pattern_name, pattern_smarts in self.COMMON_PATTERNS.items():
                if smarts == pattern_smarts:
                    name = pattern_name
                    break
            else:
                name = "pattern"
        
        results = []
        for i, match in enumerate(matches):
            info = SubstructureInfo(
                name=f"{name}_{i}",
                rdkit_indices=tuple(match),
                smarts=smarts
            )
            results.append(info)
        
        return results
    
    def match_common_pattern(
        self, 
        mol: Chem.Mol, 
        pattern_name: str
    ) -> List[SubstructureInfo]:
        """
        Match a common pre-defined pattern by name.
        
        Args:
            mol: RDKit molecule object
            pattern_name: Name of pattern (e.g., "benzene", "methyl")
            
        Returns:
            List of SubstructureInfo for each match found
            
        Raises:
            ValueError: If pattern_name is not recognized
        """
        if pattern_name not in self.COMMON_PATTERNS:
            available = ", ".join(self.COMMON_PATTERNS.keys())
            raise ValueError(
                f"Unknown pattern: {pattern_name}. Available: {available}"
            )
        
        smarts = self.COMMON_PATTERNS[pattern_name]
        return self.match_smarts_pattern(mol, smarts, name=pattern_name)

    def find_center_benzene_ring(
        self,
        mol: Chem.Mol,
        positions: np.ndarray,
    ) -> Optional[SubstructureInfo]:
        """Return the benzene ring whose center is closest to the molecule COM."""
        benzenes = self.match_common_pattern(mol, "benzene")
        if not benzenes:
            return None

        com = np.mean(positions, axis=0)
        return min(
            benzenes,
            key=lambda ring: np.linalg.norm(
                self.calculate_substructure_center(positions, ring.rdkit_indices) - com
            ),
        )
    
    @staticmethod
    def calculate_substructure_center(
        positions: np.ndarray,
        atom_indices: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Compute geometric center of atoms at given indices.
        
        Args:
            positions: Array of atom positions (n_atoms, 3)
            atom_indices: Tuple of atom indices to include
            
        Returns:
            Center coordinates (3,)
        """
        subset_positions = positions[list(atom_indices)]
        return np.mean(subset_positions, axis=0)
    
    @staticmethod
    def calculate_ring_metrics(
        positions: np.ndarray,
        atom_indices: Tuple[int, ...]
    ) -> RingMetrics:
        """
        Compute metrics for a ring/substructure.
        
        Metrics include:
        - center: Geometric center
        - planarity_rmsd: RMSD from best-fit plane (lower = more planar)
        - normal_vector: Unit normal of best-fit plane
        - radius: Average distance from center to atoms
        
        Args:
            positions: Array of atom positions (n_atoms, 3)
            atom_indices: Tuple of atom indices in the ring
            
        Returns:
            RingMetrics dataclass with computed values
        """
        subset_positions = positions[list(atom_indices)]
        n_atoms = len(atom_indices)
        
        # Geometric center
        center = np.mean(subset_positions, axis=0)
        
        # Average radius
        distances_from_center = np.linalg.norm(subset_positions - center, axis=1)
        radius = np.mean(distances_from_center)
        
        # For planarity and normal vector, we need at least 3 atoms
        if n_atoms < 3:
            # Can't define a plane with < 3 atoms
            return RingMetrics(
                center=center,
                planarity_rmsd=0.0,
                normal_vector=np.array([0.0, 0.0, 1.0]),
                radius=radius
            )
        
        # Compute best-fit plane using SVD
        # Center the points
        centered = subset_positions - center
        
        # SVD to find principal components
        try:
            U, S, Vt = np.linalg.svd(centered)
            # Normal vector is the last row of Vt (smallest singular value direction)
            normal = Vt[-1]
            # Ensure unit vector
            normal = normal / np.linalg.norm(normal)
            
            # Planarity: RMSD of distances from plane
            # Distance from point to plane = |dot(point - center, normal)|
            distances_from_plane = np.abs(np.dot(centered, normal))
            planarity_rmsd = np.sqrt(np.mean(distances_from_plane ** 2))
        except np.linalg.LinAlgError:
            # SVD failed - return default values
            normal = np.array([0.0, 0.0, 1.0])
            planarity_rmsd = 0.0
        
        return RingMetrics(
            center=center,
            planarity_rmsd=planarity_rmsd,
            normal_vector=normal,
            radius=radius
        )

    def find_and_map_substructures(
        self,
        mol: Chem.Mol,
        mda_selection: mda.AtomGroup,
        smarts_patterns: Optional[List[str]] = None,
        include_rings: bool = True
    ) -> List[SubstructureInfo]:
        """
        Find substructures and map RDKit indices to MDAnalysis indices.
        
        Args:
            mol: RDKit molecule object
            mda_selection: MDAnalysis AtomGroup corresponding to the molecule
            smarts_patterns: Optional list of SMARTS patterns to match
            include_rings: If True, also include auto-detected rings
            
        Returns:
            List of SubstructureInfo with both RDKit and MDA indices mapped
        """
        substructures = []
        
        if include_rings:
            rings = self.find_all_rings(mol)
            substructures.extend(rings)
        
        if smarts_patterns:
            for smarts in smarts_patterns:
                matches = self.match_smarts_pattern(mol, smarts)
                substructures.extend(matches)
        
        mda_atom_indices = mda_selection.indices
        
        for sub in substructures:
            mda_indices = tuple(mda_atom_indices[i] for i in sub.rdkit_indices)
            sub.mda_indices = mda_indices
        
        return substructures


def filter_endpoints_to_methyl_atoms(
    mol: Chem.Mol,
    endpoint_rdkit_indices: Sequence[int],
    *,
    ring_calculator: Optional[RingCenterCalculator] = None,
) -> List[int]:
    """
    Resolve type-4 methyl carbons from EndpointsFinder output.

    1. Endpoints that directly match ``[CH3]``.
    2. Methyl carbons bonded to any endpoint heavy atom (benzylic endpoints).
    """
    calc = ring_calculator or RingCenterCalculator()
    methyl_matches = calc.match_common_pattern(mol, "methyl")
    methyl_atoms = {idx for match in methyl_matches for idx in match.rdkit_indices}
    endpoint_set = set(endpoint_rdkit_indices)

    direct = sorted(i for i in endpoint_rdkit_indices if i in methyl_atoms)
    if direct:
        return direct

    bonded: List[int] = []
    for idx in methyl_atoms:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in endpoint_set:
                bonded.append(idx)
                break
    return sorted(set(bonded))


def all_methyl_atom_indices(
    mol: Chem.Mol,
    *,
    ring_calculator: Optional[RingCenterCalculator] = None,
) -> List[int]:
    """All ``[CH3]`` carbon indices in *mol* (RDKit local indices)."""
    calc = ring_calculator or RingCenterCalculator()
    methyl_matches = calc.match_common_pattern(mol, "methyl")
    return sorted({idx for match in methyl_matches for idx in match.rdkit_indices})


def bond_steps_from_atoms(
    mol: Chem.Mol,
    source_rdkit_indices: Sequence[int],
) -> List[int]:
    """Graph distance from the nearest *source* atom to every atom in *mol*."""
    n_atoms = mol.GetNumAtoms()
    visited = [-1] * n_atoms
    queue: deque[Tuple[int, int]] = deque()

    for atom_idx in source_rdkit_indices:
        if 0 <= atom_idx < n_atoms and visited[atom_idx] == -1:
            visited[atom_idx] = 0
            queue.append((atom_idx, 0))

    while queue:
        atom_idx, depth = queue.popleft()
        for nbr in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            j = nbr.GetIdx()
            if visited[j] == -1:
                visited[j] = depth + 1
                queue.append((j, depth + 1))

    return visited


def select_central_methyl_atoms(
    mol: Chem.Mol,
    positions: np.ndarray,
    n: int = 3,
    *,
    center_ring_rdkit_indices: Optional[Sequence[int]] = None,
    ring_calculator: Optional[RingCenterCalculator] = None,
    endpoint_rdkit_indices: Optional[Sequence[int]] = None,
    use_endpoint_filter: bool = True,
) -> List[int]:
    """
    Select *n* methyl carbons nearest the central benzene in bond-graph distance.

    Candidates are endpoint-linked methyls (``filter_endpoints_to_methyl_atoms``)
    when *endpoint_rdkit_indices* is provided and *use_endpoint_filter* is True;
    otherwise all ``[CH3]`` carbons in the molecule.

    Methyls at the minimum bond-step count from the center benzene are kept.
    If that yields more than *n*, ties break by 3D distance to the ring centroid.
    """
    calc = ring_calculator or RingCenterCalculator()

    if center_ring_rdkit_indices is None:
        center_ring = calc.find_center_benzene_ring(mol, positions)
        if center_ring is None:
            raise ValueError("No central benzene ring found for methyl selection.")
        center_ring_rdkit_indices = center_ring.rdkit_indices

    if use_endpoint_filter and endpoint_rdkit_indices is not None:
        candidates = filter_endpoints_to_methyl_atoms(
            mol, endpoint_rdkit_indices, ring_calculator=calc
        )
    else:
        candidates = all_methyl_atom_indices(mol, ring_calculator=calc)

    if not candidates:
        raise ValueError("No methyl candidates found for central methyl selection.")

    steps = bond_steps_from_atoms(mol, center_ring_rdkit_indices)
    candidate_steps = [steps[i] for i in candidates if steps[i] >= 0]
    if not candidate_steps:
        raise ValueError("No methyl candidates reachable from the central benzene ring.")

    min_steps = min(candidate_steps)
    at_min = [i for i in candidates if steps[i] == min_steps]
    if len(at_min) <= n:
        return sorted(at_min)

    ring_center = calc.calculate_substructure_center(
        positions, tuple(center_ring_rdkit_indices)
    )
    return sorted(
        at_min,
        key=lambda i: np.linalg.norm(positions[i] - ring_center),
    )[:n]


def select_type4_by_centroid_distance(
    candidate_rdkit_indices: Sequence[int],
    center_ring_rdkit_indices: Sequence[int],
    positions: np.ndarray,
    n: int,
    *,
    farthest: bool = True,
) -> List[int]:
    """
    Keep *n* candidates ranked by 3D distance to the central benzene centroid.

    By default the farthest atoms are kept (outer substituents in the assembled cube).
    """
    candidates = sorted(set(candidate_rdkit_indices))
    if len(candidates) < n:
        raise ValueError(
            f"Centroid type-4 selection found {len(candidates)} candidates; expected {n}."
        )
    if len(candidates) == n:
        return candidates

    center = positions[list(center_ring_rdkit_indices)].mean(axis=0)
    ranked = sorted(
        candidates,
        key=lambda i: float(np.linalg.norm(positions[i] - center)),
        reverse=farthest,
    )
    return sorted(ranked[:n])


def select_type4_by_bond_distance(
    mol: Chem.Mol,
    candidate_rdkit_indices: Sequence[int],
    center_ring_rdkit_indices: Sequence[int],
    positions: np.ndarray,
    n: int,
    *,
    outermost: bool = True,
    ring_calculator: Optional[RingCenterCalculator] = None,
) -> List[int]:
    """
    Keep *n* candidates ranked by bond-graph distance from the central benzene.

    *outermost=True* keeps atoms at the maximum bond-step (hull-style endpoints).
    *outermost=False* keeps atoms at the minimum bond-step (nearest substituent methyls).
    Ties break by 3D distance to the ring centroid.
    """
    candidates = sorted(set(candidate_rdkit_indices))
    if len(candidates) < n:
        raise ValueError(
            f"Bond-step type-4 selection found {len(candidates)} candidates; expected {n}."
        )
    if len(candidates) == n:
        return candidates

    steps = bond_steps_from_atoms(mol, center_ring_rdkit_indices)
    reachable = [i for i in candidates if steps[i] >= 0]
    if not reachable:
        raise ValueError("No type-4 candidates reachable from the central benzene ring.")

    step_values = [steps[i] for i in reachable]
    target_step = max(step_values) if outermost else min(step_values)
    at_target = [i for i in reachable if steps[i] == target_step]
    if len(at_target) <= n:
        return sorted(at_target)

    calc = ring_calculator or RingCenterCalculator()
    ring_center = calc.calculate_substructure_center(
        positions, tuple(center_ring_rdkit_indices)
    )
    return sorted(
        at_target,
        key=lambda i: float(np.linalg.norm(positions[i] - ring_center)),
        reverse=True,
    )[:n]


def select_endpoint_type4_atoms(
    endpoint_rdkit_indices: Sequence[int],
    center_ring_rdkit_indices: Sequence[int],
    *,
    n: int,
    exclude_center_benzene: bool = True,
    positions: Optional[np.ndarray] = None,
) -> List[int]:
    """
    Pick type-4 beads directly from EndpointsFinder output.

    When *exclude_center_benzene* is True, atoms belonging to the central
    benzene ring (already used for type-1 centroids) are removed first.
    If more than *n* candidates remain, the farthest from the ring centroid
    are kept (requires *positions*).
    """
    ring_set = set(center_ring_rdkit_indices)
    candidates = sorted(set(endpoint_rdkit_indices))
    if exclude_center_benzene:
        candidates = [i for i in candidates if i not in ring_set]

    if positions is None:
        if len(candidates) == n:
            return candidates
        if len(candidates) > n:
            return candidates[:n]
        raise ValueError(
            f"Endpoint type-4 selection found {len(candidates)} atoms after "
            f"filtering (exclude_center_benzene={exclude_center_benzene}); "
            f"expected {n}. Raw endpoint count: {len(set(endpoint_rdkit_indices))}."
        )

    return select_type4_by_centroid_distance(
        candidates,
        center_ring_rdkit_indices,
        positions,
        n,
    )


class SubstructureCenterObserver(FrameObserver):
    """
    Observer for tracking ring/substructure centers and metrics over a trajectory.
    
    Implements the FrameObserver pattern to integrate with TrajectoryIterator.
    Computes geometric centers, planarity, normal vectors, and pairwise distances
    for rings and custom SMARTS-defined substructures.
    
    Example:
        >>> from MD_analysis.src.TrajectoryIterator import TrajectoryIterator
        >>> from MD_analysis.src.utils.rdkit_utils import SubstructureCenterObserver
        >>> 
        >>> # Create observer for a specific residue
        >>> observer = SubstructureCenterObserver(
        ...     selection="resname LIG",
        ...     smarts_patterns=["c1ccccc1"],  # benzene rings
        ...     include_rings=True  # also auto-detect rings
        ... )
        >>> 
        >>> # Register with iterator
        >>> iterator = TrajectoryIterator(universe)
        >>> iterator.subscribe(observer)
        >>> iterator.iterate()
        >>> 
        >>> # Access results
        >>> centers = observer.results['centers']
        >>> distances = observer.results['distances']
    """
    
    def __init__(
        self,
        selection: str,
        smarts_patterns: Optional[List[str]] = None,
        include_rings: bool = True,
        compute_distances: bool = True,
        compute_metrics: bool = True
    ):
        """
        Initialize SubstructureCenterObserver.
        
        Args:
            selection: MDAnalysis selection string for the molecule(s) to analyze
            smarts_patterns: Optional list of SMARTS patterns to match
            include_rings: If True, auto-detect all rings in the molecule
            compute_distances: If True, compute pairwise distances between centers
            compute_metrics: If True, compute planarity, normal vectors, radius
        """
        super().__init__()
        
        self.selection = selection
        self.smarts_patterns = smarts_patterns or []
        self.include_rings = include_rings
        self.compute_distances = compute_distances
        self.compute_metrics = compute_metrics
        
        # Will be populated in on_frame_start
        self._calculator = RingCenterCalculator()
        self._substructures: List[SubstructureInfo] = []
        self._initialized = False
        self._n_substructures = 0
        
        # Initialize results structure
        self.results = {
            'centers': [],           # List of dicts per frame
            'distances': [],         # List of dicts per frame
            'planarity': [],         # List of dicts per frame
            'normal_vectors': [],    # List of dicts per frame
            'substructure_info': [], # Metadata about detected substructures
        }
    
    def get_selections_needed(self) -> List[str]:
        """Return list of selection strings needed by this observer."""
        return [self.selection]
    
    def on_frame_start(self, iterator: 'TrajectoryIterator') -> None:
        """
        Initialize substructure detection before iteration.
        
        Converts the selection to RDKit, finds substructures,
        and maps indices between RDKit and MDAnalysis.
        """
        universe = iterator.universe
        
        # Get the atom selection
        try:
            sel = universe.select_atoms(self.selection)
        except Exception as e:
            warnings.warn(f"Failed to select atoms with '{self.selection}': {e}")
            self._initialized = False
            return
        
        if len(sel) == 0:
            warnings.warn(f"Selection '{self.selection}' returned no atoms")
            self._initialized = False
            return
        
        # Convert to RDKit molecule
        try:
            mol = sel.convert_to("RDKIT")
        except Exception as e:
            warnings.warn(f"Failed to convert selection to RDKit: {e}")
            self._initialized = False
            return
        
        if mol is None:
            warnings.warn("RDKit conversion returned None")
            self._initialized = False
            return
        
        # Find and map substructures
        try:
            self._substructures = self._calculator.find_and_map_substructures(
                mol=mol,
                mda_selection=sel,
                smarts_patterns=self.smarts_patterns,
                include_rings=self.include_rings
            )
        except Exception as e:
            warnings.warn(f"Failed to find substructures: {e}")
            self._initialized = False
            return
        
        self._n_substructures = len(self._substructures)
        
        if self._n_substructures == 0:
            warnings.warn("No substructures found in the selection")
            self._initialized = False
            return
        
        # Store substructure metadata in results
        self.results['substructure_info'] = [
            {
                'name': sub.name,
                'rdkit_indices': sub.rdkit_indices,
                'mda_indices': sub.mda_indices,
                'smarts': sub.smarts
            }
            for sub in self._substructures
        ]
        
        self._initialized = True
    
    def on_frame(
        self,
        ts: mda.coordinates.base.Timestep,
        frame_idx: int,
        universe: mda.Universe
    ) -> None:
        """
        Process a single frame during iteration.
        
        Computes centers and metrics for each substructure at this frame.
        """
        if not self._initialized:
            return
        
        # Get all atom positions
        positions = universe.atoms.positions
        
        # Compute centers for each substructure
        frame_centers = {}
        frame_planarity = {}
        frame_normals = {}
        
        for sub in self._substructures:
            if sub.mda_indices is None:
                continue
            
            try:
                if self.compute_metrics:
                    # Compute full metrics
                    metrics = RingCenterCalculator.calculate_ring_metrics(
                        positions, sub.mda_indices
                    )
                    frame_centers[sub.name] = metrics.center.copy()
                    frame_planarity[sub.name] = metrics.planarity_rmsd
                    frame_normals[sub.name] = metrics.normal_vector.copy()
                else:
                    # Just compute center
                    center = RingCenterCalculator.calculate_substructure_center(
                        positions, sub.mda_indices
                    )
                    frame_centers[sub.name] = center.copy()
            except Exception as e:
                warnings.warn(
                    f"Failed to compute metrics for {sub.name} at frame {frame_idx}: {e}"
                )
        
        # Store centers with frame info
        self.results['centers'].append({
            'frame': frame_idx,
            'time': ts.time,
            **{f"{name}_center": center for name, center in frame_centers.items()}
        })
        
        # Store metrics if computed
        if self.compute_metrics:
            self.results['planarity'].append({
                'frame': frame_idx,
                'time': ts.time,
                **{f"{name}_planarity": val for name, val in frame_planarity.items()}
            })
            self.results['normal_vectors'].append({
                'frame': frame_idx,
                'time': ts.time,
                **{f"{name}_normal": vec for name, vec in frame_normals.items()}
            })
        
        # Compute pairwise distances between centers
        if self.compute_distances and len(frame_centers) > 1:
            distances = {'frame': frame_idx, 'time': ts.time}
            names = list(frame_centers.keys())
            for i, name1 in enumerate(names):
                for name2 in names[i+1:]:
                    c1 = frame_centers[name1]
                    c2 = frame_centers[name2]
                    dist = np.linalg.norm(c1 - c2)
                    distances[f"{name1}_to_{name2}"] = dist
            self.results['distances'].append(distances)
    
    def on_frame_end(self, iterator: 'TrajectoryIterator') -> None:
        """Finalize results after iteration."""
        # Results are already stored in self.results
        pass
    
    def _get_aggregator(self) -> ResultsGroup:
        """
        Return ResultsGroup for parallel processing result merging.
        
        Defines how results from parallel workers should be combined.
        """
        return ResultsGroup(lookup={
            'centers': ResultsGroup.list_extend_sorted('frame'),
            'distances': ResultsGroup.list_extend_sorted('frame'),
            'planarity': ResultsGroup.list_extend_sorted('frame'),
            'normal_vectors': ResultsGroup.list_extend_sorted('frame'),
            # substructure_info is the same across workers, keep first
        })
    
    def get_centers_array(self, substructure_name: str) -> np.ndarray:
        """
        Get centers for a specific substructure as a numpy array.
        
        Args:
            substructure_name: Name of the substructure (e.g., "ring_0", "benzene_0")
            
        Returns:
            Array of shape (n_frames, 3) with center coordinates
        """
        key = f"{substructure_name}_center"
        centers = []
        for frame_data in self.results['centers']:
            if key in frame_data:
                centers.append(frame_data[key])
        return np.array(centers)
    
    def get_distances_array(self, name1: str, name2: str) -> np.ndarray:
        """
        Get distances between two substructures as a numpy array.
        
        Args:
            name1: Name of first substructure
            name2: Name of second substructure
            
        Returns:
            Array of shape (n_frames,) with distances
        """
        # Try both orderings
        key1 = f"{name1}_to_{name2}"
        key2 = f"{name2}_to_{name1}"
        
        distances = []
        for frame_data in self.results['distances']:
            if key1 in frame_data:
                distances.append(frame_data[key1])
            elif key2 in frame_data:
                distances.append(frame_data[key2])
        return np.array(distances)
    
    def get_planarity_array(self, substructure_name: str) -> np.ndarray:
        """
        Get planarity RMSD for a specific substructure as a numpy array.
        
        Args:
            substructure_name: Name of the substructure
            
        Returns:
            Array of shape (n_frames,) with planarity RMSD values
        """
        key = f"{substructure_name}_planarity"
        values = []
        for frame_data in self.results['planarity']:
            if key in frame_data:
                values.append(frame_data[key])
        return np.array(values)
    
    def get_normal_vectors_array(self, substructure_name: str) -> np.ndarray:
        """
        Get normal vectors for a specific substructure as a numpy array.
        
        Args:
            substructure_name: Name of the substructure
            
        Returns:
            Array of shape (n_frames, 3) with normal vectors
        """
        key = f"{substructure_name}_normal"
        vectors = []
        for frame_data in self.results['normal_vectors']:
            if key in frame_data:
                vectors.append(frame_data[key])
        return np.array(vectors)
    
    def get_substructure_names(self) -> List[str]:
        """Get list of detected substructure names."""
        return [info['name'] for info in self.results.get('substructure_info', [])]

