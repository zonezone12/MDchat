from typing import List, Optional, Tuple
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolops import GetDistanceMatrix


class EndpointsFinder:
    """Class for finding molecular endpoints using 2D coordinates and convex hull analysis."""
    
    def __init__(
        self,
        angle_tol_deg: float = 15.0,
        use_graph_farness: bool = True,
        alpha: float = 0.2,  # low weight: mostly visual, slight topo preference
        ring_min_gap_deg: Optional[float] = 35.0,
        ring_max_per_ring: int = 3,
        step_back_from_terminals: bool = True,
        extend_to_ring_atoms: bool = True
    ):
        """
        Initialize the EndpointsFinder with configuration parameters.
        
        Args:
            angle_tol_deg: Angle tolerance in degrees for finding opposite endpoints
            use_graph_farness: Whether to use topological graph farness weighting
            alpha: Weight for graph farness (0.0 = visual only, 1.0 = topology only)
            ring_min_gap_deg: Minimum angular gap between endpoints in rings (None to disable)
            ring_max_per_ring: Maximum number of endpoints per ring
            step_back_from_terminals: If True, step back from H atoms or atoms with only one neighbor
            extend_to_ring_atoms: If True, extend endpoints on rings to include all ring atoms
        """
        self.angle_tol_deg = angle_tol_deg
        self.use_graph_farness = use_graph_farness
        self.alpha = alpha
        self.ring_min_gap_deg = ring_min_gap_deg
        self.ring_max_per_ring = ring_max_per_ring
        self.step_back_from_terminals = step_back_from_terminals
        self.extend_to_ring_atoms = extend_to_ring_atoms
    
    def to_2d_coords(self, mol: Chem.Mol) -> Tuple[Chem.Mol, np.ndarray]:
        """Convert molecule to 2D coordinates."""
        m = Chem.Mol(mol)
        AllChem.Compute2DCoords(m)
        conf = m.GetConformer()
        xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(m.GetNumAtoms())])
        return m, xy
    
    def _convex_hull_indices_monotone_chain(self, xy: np.ndarray) -> List[int]:
        """Compute convex hull indices using monotone chain algorithm (fallback)."""
        pts = np.asarray(xy, float)
        n = len(pts)
        if n <= 3:
            return list(range(n))
        order = np.lexsort((pts[:,1], pts[:,0]))
        P = pts[order]
        def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        lower = []
        for idx in order:
            while len(lower) >= 2 and cross(pts[lower[-2]], pts[lower[-1]], pts[idx]) <= 0:
                lower.pop()
            lower.append(idx)
        upper = []
        for idx in order[::-1]:
            while len(upper) >= 2 and cross(pts[upper[-2]], pts[upper[-1]], pts[idx]) <= 0:
                upper.pop()
            upper.append(idx)
        hull = lower[:-1] + upper[:-1]
        return sorted(set(hull))
    
    def _get_endpoints_id_from_2d_coords(self, xy: np.ndarray) -> List[int]:
        """Get endpoint indices from 2D coordinates using convex hull."""
        try:
            from scipy.spatial import ConvexHull
            if len(xy) < 3:
                return list(range(len(xy)))
            hull = ConvexHull(xy)
            return sorted(set(hull.vertices.tolist()))
        except Exception:
            return self._convex_hull_indices_monotone_chain(xy)
    
    def _graph_farness(self, mol: Chem.Mol) -> np.ndarray:
        """Calculate topological graph farness for each atom."""
        D = GetDistanceMatrix(mol)
        return D.max(axis=1).astype(float)
    
    def _center_endpoints_opposite_ends(self, xy: np.ndarray, endpointsId: List[int]) -> List[Optional[int]]:
        """Find opposite endpoints on the other side of the centroid."""
        center = xy.mean(axis=0)
        endpoints = xy[endpointsId]
        vectors = center - endpoints
        opp = []
        for v in vectors:
            norm_v = np.linalg.norm(v)
            if norm_v == 0:
                opp.append(None)
                continue
            candidates = []
            for i, xy_i in enumerate(xy):
                opp_xy = xy_i - center
                norm_opp = np.linalg.norm(opp_xy)
                if norm_opp == 0:
                    continue
                dot = np.dot(v, opp_xy)
                if dot > 0:  # other side of center
                    cos = dot / (norm_v * norm_opp)
                    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
                    if angle < self.angle_tol_deg:
                        candidates.append((angle, norm_opp, i))  # angle, dist, id
            if candidates:
                candidates.sort(key=lambda x: (-x[1]))  # max dist
                best_i = candidates[0][2]
                opp.append(best_i)
            else:
                opp.append(None)
        return opp
    
    def _ring_spacing_filter(self, mol: Chem.Mol, xy: np.ndarray, ids: List[int]) -> List[int]:
        """Filter endpoints by ring spacing constraints."""
        ringinfo = mol.GetRingInfo()
        rings = ringinfo.AtomRings()
        atom2rings = {i: [] for i in range(mol.GetNumAtoms())}
        for rid, ring in enumerate(rings):
            for a in ring:
                atom2rings[a].append(rid)
        c = xy.mean(axis=0)
        ang = np.degrees(np.arctan2(xy[:,1]-c[1], xy[:,0]-c[0])) % 360.0
        
        selected, per_ring = [], {}
        def gap_ok(rid, idx):
            arr = per_ring.setdefault(rid, [])
            for j in arr:
                d = abs((ang[idx] - ang[j] + 180) % 360 - 180)
                if d < self.ring_min_gap_deg:
                    return False
            if len(arr) >= self.ring_max_per_ring:
                return False
            arr.append(idx)
            return True
        
        for i in ids:
            rids = atom2rings.get(i, [])
            if not rids:
                selected.append(i)
            else:
                kept = any(gap_ok(rid, i) for rid in rids)
                if kept:
                    selected.append(i)
        return sorted(set(selected))
    
    def find_endpoints(self, mol: Chem.Mol) -> List[int]:
        """
        Find endpoint atom indices for a given molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            List of atom indices representing endpoints
        """
        m2d, xy = self.to_2d_coords(mol)
        hull_ids = self._get_endpoints_id_from_2d_coords(xy)  # "visual tips"
        opp_ids = self._center_endpoints_opposite_ends(xy, hull_ids)
        
        # union hull + valid opposites
        endpoints = set(hull_ids) | {i for i in opp_ids if i is not None}
        
        # optional topo weighting: bias toward topologically far atoms
        if self.use_graph_farness:
            f = self._graph_farness(m2d)
            f = (f - f.min()) / (np.ptp(f) + 1e-12)
            # re-rank hull tips by farness and keep top 80–100% (gentle pruning)
            ranked = sorted(endpoints, key=lambda i: self.alpha * f[i] + (1 - self.alpha) * 1.0, reverse=True)
            endpoints = set(ranked)  # no strong pruning by default
        
        # ring spacing throttle (optional)
        if self.ring_min_gap_deg is not None and self.ring_max_per_ring > 0:
            endpoints = set(self._ring_spacing_filter(m2d, xy, list(endpoints)))
        
        # step back from terminals (optional)
        if self.step_back_from_terminals:
            endpoints = set(self._step_back_from_terminals(m2d, list(endpoints)))
        
        # extend to ring atoms (optional)
        if self.extend_to_ring_atoms:
            endpoints = set(self._extend_to_ring_atoms(m2d, list(endpoints)))
        
        return sorted(endpoints)
    
    def _step_back_from_terminals(self, mol: Chem.Mol, endpoints: List[int]) -> List[int]:
        """
        Step back from terminal atoms (H or atoms with only one neighbor) to their neighbor atoms.
        Internal method used when step_back_from_terminals option is enabled.
        
        Args:
            mol: RDKit molecule object
            endpoints: List of endpoint atom indices
            
        Returns:
            List of updated endpoint indices (stepped back from terminals)
        """
        updated_endpoints = []
        
        for atom_idx in endpoints:
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # Check if atom is H or has only one neighbor
            is_hydrogen = atom.GetSymbol() == 'H'
            num_neighbors = atom.GetDegree()
            
            if is_hydrogen or num_neighbors == 1:
                # Get the neighbor atom
                neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
                if neighbors:
                    # Step back to the neighbor atom
                    updated_endpoints.append(neighbors[0])
                else:
                    # No neighbors (shouldn't happen, but keep original if it does)
                    updated_endpoints.append(atom_idx)
            else:
                # Keep the original endpoint
                updated_endpoints.append(atom_idx)
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for idx in updated_endpoints:
            if idx not in seen:
                seen.add(idx)
                result.append(idx)
        
        return sorted(result)
    
    def _extend_to_ring_atoms(self, mol: Chem.Mol, endpoints: List[int]) -> List[int]:
        """
        Extend endpoints that are on rings to include all atoms in those rings.
        If an endpoint is on multiple rings (fused rings), all atoms from all rings are included.
        
        Args:
            mol: RDKit molecule object
            endpoints: List of endpoint atom indices
            
        Returns:
            List of extended endpoint indices (includes all ring atoms for endpoints on rings)
        """
        ringinfo = mol.GetRingInfo()
        rings = ringinfo.AtomRings()
        
        # Build a mapping from atom index to rings it belongs to
        atom2rings = {i: [] for i in range(mol.GetNumAtoms())}
        for rid, ring in enumerate(rings):
            for atom_idx in ring:
                atom2rings[atom_idx].append(rid)
        
        # Collect all atoms to include
        extended_endpoints = set(endpoints)
        
        for endpoint_idx in endpoints:
            # Check if this endpoint is on any ring
            ring_ids = atom2rings.get(endpoint_idx, [])
            if ring_ids:
                # Add all atoms from all rings this endpoint belongs to
                for ring_id in ring_ids:
                    ring_atoms = set(rings[ring_id])
                    extended_endpoints.update(ring_atoms)
        
        return sorted(extended_endpoints)

