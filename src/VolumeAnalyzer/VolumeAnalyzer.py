"""
VolumeAnalyzer: Analyze target volume and internal cavity volume over an MD trajectory.

Uses a voxel grid around a selected atom group (e.g. target).
Atoms are treated as spheres with VDW radii + probe radius.
- target volume: union of these inflated spheres (solvent-excluded volume).
- Cavity volume: empty voxels not connected to the outer boundary (enclosed voids).
"""

from __future__ import annotations

import io
import warnings
from typing import Optional

import numpy as np

try:
    import MDAnalysis as mda
except ImportError:
    raise ImportError("MDAnalysis is required. pip install MDAnalysis")

try:
    from scipy import ndimage
except ImportError:
    raise ImportError("scipy is required. pip install scipy")

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:
    plt = None
    ListedColormap = None
    warnings.warn("matplotlib not available. Plotting will be disabled.")

try:
    from rdkit.Chem import AllChem, Draw
except ImportError:
    AllChem = None
    Draw = None
    warnings.warn("rdkit not available. Some features will be disabled.")

try:
    import plotly.graph_objects as go
except ImportError:
    go = None
    warnings.warn("plotly not available. Interactive 3D visualization will be disabled.")

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None
    warnings.warn("imageio not available. GIF generation will be disabled.")

# Import plotly_molecule with fallback for different import contexts
try:
    from MD_analysis.plotly_molecule import make_molecule_components
except ImportError:
    try:
        from ..plotly_molecule import make_molecule_components
    except ImportError:
        try:
            from plotly_molecule import make_molecule_components
        except ImportError:
            make_molecule_components = None
            warnings.warn("plotly_molecule not available. Interactive 3D visualization will be limited.")

try:
    from skimage import measure
except ImportError:
    measure = None
    warnings.warn("scikit-image is required for mesh volume computation. pip install scikit-image")


class VolumeAnalyzer:
    """
    Analyze target volume and internal cavity volume over an MD trajectory.

    - Uses a voxel grid around a selected atom group (e.g. target).
    - Atoms are treated as spheres with VDW radii + probe radius.
    - target volume: union of these inflated spheres (solvent-excluded volume).
    - Cavity volume: empty voxels not connected to the outer boundary (enclosed voids).
    """

    # Van der Waals radii (Å) for common elements
    DEFAULT_VDW_RADII = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "P": 1.80,
        "F": 1.47,
        "CL": 1.75,
        "BR": 1.85,
        "I": 1.98,
        "NA+": 1.02,
    }
    
    # Constants for volume computation
    DEFAULT_DENSITY_THRESHOLD = 0.5
    DEFAULT_SIGMA_FACTOR = 0.6
    DEFAULT_FALLBACK_RADIUS = 1.8
    GAUSSIAN_SIZE_MULTIPLIER = 3.0
    DEFAULT_SELECTION = "not water and not name I and not name Na+"

    def __init__(
        self,
        universe: mda.Universe,
        selection: str | None = None,
        spacing: float = 0.3,
        probe_radius: float = 1.4,
        margin: float | None = None,
        radii_scale: float = 1.0,
        radii_offset: float = 0.0,
        smooth_surface: bool = False,
        smooth_iterations: int = 1,
        vdw_radii_table: dict | None = None,
        use_gaussian_step: bool = True,
        density_threshold: float = DEFAULT_DENSITY_THRESHOLD,
        sigma_factor: float = DEFAULT_SIGMA_FACTOR,
        apply_binary_opening: bool = True,
    ):
        """
        Parameters
        ----------
        universe : MDAnalysis.Universe
            MDAnalysis Universe with topology and trajectory loaded.
        selection : str or None
            Atom selection string for MDAnalysis. If None, uses default selection
            (excludes water, I, and Na+).
        spacing : float
            Grid spacing in Å (smaller = more accurate, slower). Typical: 0.3–1.0.
            Must be > 0.
        probe_radius : float
            Probe radius in Å. 1.4 Å ~ water; increase for ligand-sized probe.
            Must be >= 0.
        margin : float or None
            Extra padding added around target bounding box (Å).
            If None, it will be derived automatically from radii + spacing.
            Must be >= 0 if provided.
        radii_scale : float
            Global multiplicative factor applied to VDW radii. Must be > 0.
        radii_offset : float
            Global additive padding (Å) added to radii (after scaling).
        smooth_surface : bool
            If True, apply binary closing to the occupancy grid to seal small gaps.
        smooth_iterations : int
            Number of closing iterations (1–2 usually enough). Must be >= 0.
        vdw_radii_table : dict or None
            Optional per-element VDW radii, e.g. {"C": 1.75, "N": 1.6}.
            Overrides DEFAULT_VDW_RADII for those elements.
        use_gaussian_step : bool
            If True, use Gaussian-based density accumulation for smoother, more
            accurate volume computation. If False, use binary occupancy method.
        density_threshold : float
            Threshold for converting density grid to binary mask when using
            Gaussian method. Default: 0.5. Must be > 0.
        sigma_factor : float
            Controls sharpness of Gaussian blobs. Typical: 0.5–0.7. Default: 0.6.
            Only used when use_gaussian_step=True.
        apply_binary_opening : bool
            If True, apply binary opening after closing during surface smoothing.
            Helps remove small isolated regions. Default: True.
        """
        # Input validation
        if spacing <= 0:
            raise ValueError(f"spacing must be > 0, got {spacing}")
        if probe_radius < 0:
            raise ValueError(f"probe_radius must be >= 0, got {probe_radius}")
        if margin is not None and margin < 0:
            raise ValueError(f"margin must be >= 0 if provided, got {margin}")
        if radii_scale <= 0:
            raise ValueError(f"radii_scale must be > 0, got {radii_scale}")
        if smooth_iterations < 0:
            raise ValueError(f"smooth_iterations must be >= 0, got {smooth_iterations}")
        if density_threshold <= 0:
            raise ValueError(f"density_threshold must be > 0, got {density_threshold}")
        if sigma_factor <= 0:
            raise ValueError(f"sigma_factor must be > 0, got {sigma_factor}")
        
        # Store file paths instead of Universe to avoid pickling full trajectory
        # Universe will be created on-demand in compute_frame
        try:
            self.top_file = universe.filename
            self.traj_file = universe.trajectory.filename
            # Get trajectory format
            if hasattr(universe.trajectory, 'format'):
                self.traj_format = universe.trajectory.format[0] if isinstance(universe.trajectory.format, (list, tuple)) else universe.trajectory.format
            else:
                self.traj_format = None
            # Handle multiple trajectory files
            if isinstance(self.traj_file, (list, tuple)):
                if len(self.traj_file) > 1:
                    warnings.warn(f"Multiple trajectory files detected. Using first file: {self.traj_file[0]}")
                self.traj_file = self.traj_file[0]
            self._universe = None  # Will be created on-demand
            self._use_file_paths = True
        except (AttributeError, TypeError):
            # Fallback: store universe reference if file paths not available
            self._universe = universe
            self.top_file = None
            self.traj_file = None
            self.traj_format = None
            self._use_file_paths = False
        
        self.vdw_radii_table = self.DEFAULT_VDW_RADII.copy()
        # Update with custom VDW radii
        if vdw_radii_table is not None:
            self.vdw_radii_table.update({k.upper(): v for k, v in vdw_radii_table.items()})

        self.selection = self.DEFAULT_SELECTION if selection is None else selection
        self.spacing = spacing
        self.probe_radius = probe_radius
        self.margin = margin  # if None, computed per-frame
        self.radii_scale = radii_scale
        self.radii_offset = radii_offset
        self.smooth_surface = smooth_surface
        self.smooth_iterations = smooth_iterations
        self.use_gaussian_step = use_gaussian_step
        self.density_threshold = density_threshold
        self.sigma_factor = sigma_factor
        self.apply_binary_opening = apply_binary_opening
        
        # Get atom group from universe to precompute VDW radii
        # This is done once to cache the radii, but we'll recreate ag in compute_frame
        if self._use_file_paths:
            # Create temporary universe to get atom group for VDW radii
            try:
                if self.traj_format:
                    temp_u = mda.Universe(self.top_file, self.traj_file, format=self.traj_format)
                else:
                    temp_u = mda.Universe(self.top_file, self.traj_file)
                temp_ag = temp_u.select_atoms(self.selection)
                if temp_ag.n_atoms == 0:
                    raise ValueError(f"Selection '{self.selection}' returned no atoms.")
                # Precompute VDW radii for the selected atoms (fixed over trajectory)
                self.vdw_radii = self._get_vdw_radii(temp_ag)
                del temp_u, temp_ag  # Clean up
            except Exception as e:
                warnings.warn(f"Failed to precompute VDW radii from file paths: {e}. Will compute on-demand.")
                self.vdw_radii = None
        else:
            ag = self._universe.select_atoms(self.selection)
            if ag.n_atoms == 0:
                raise ValueError(f"Selection '{self.selection}' returned no atoms.")
            # Precompute VDW radii for the selected atoms (fixed over trajectory)
            self.vdw_radii = self._get_vdw_radii(ag)
        
        # Will store grid axes of the last computed frame for visualization
        self._last_grid_axes = None  # (x, y, z)

    # ----------- element / radii helpers -----------

    def _get_element(self, atom) -> str | None:
        """Best-effort guess of the element symbol from an MDAnalysis Atom."""
        elem = getattr(atom, "element", None)
        if elem is None or elem == "":
            name = atom.name.strip()
            elem = "".join([c for c in name if c.isalpha()])
        elem = elem.upper()
        if len(elem) >= 2 and elem[:2] in self.vdw_radii_table:
            return elem[:2]
        elif len(elem) >= 1 and elem[0] in self.vdw_radii_table:
            return elem[0]
        return None

    def _get_vdw_radii(self, atomgroup) -> np.ndarray:
        """Return np.array of base VDW radii for an AtomGroup (before scale/offset)."""
        radii = []
        for atom in atomgroup.atoms:
            elem = self._get_element(atom)
            if elem is None:
                radii.append(self.DEFAULT_FALLBACK_RADIUS)
                warnings.warn(
                    f"Unknown element for atom {atom.name} (resname {atom.resname}, "
                    f"resid {atom.resid}), using default radius {self.DEFAULT_FALLBACK_RADIUS} Å",
                    UserWarning,
                    stacklevel=2
                )
            else:
                radii.append(self.vdw_radii_table.get(elem, self.DEFAULT_FALLBACK_RADIUS))
        return np.array(radii, dtype=float)

    # ----------- grid / occupancy helpers -----------

    @staticmethod
    def _build_grid(
        coords: np.ndarray, spacing: float, margin: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int], np.ndarray]:
        """
        Build a 3D grid around a set of coordinates.

        Parameters
        ----------
        coords : np.ndarray
            (n_atoms, 3) array of coordinates.
        spacing : float
            Grid spacing in Å.
        margin : float
            Margin around bounding box in Å.

        Returns
        -------
        x, y, z : np.ndarray
            1D arrays of grid coordinates.
        shape : tuple[int, int, int]
            (nx, ny, nz) grid dimensions.
        origin : np.ndarray
            (xmin, ymin, zmin) origin coordinates.
        """
        if coords.size == 0:
            raise ValueError("coords cannot be empty")
        min_coord = coords.min(axis=0) - margin
        max_coord = coords.max(axis=0) + margin

        x = np.arange(min_coord[0], max_coord[0] + spacing, spacing)
        y = np.arange(min_coord[1], max_coord[1] + spacing, spacing)
        z = np.arange(min_coord[2], max_coord[2] + spacing, spacing)

        return x, y, z, (len(x), len(y), len(z)), min_coord

    @staticmethod
    def _mark_occupancy(
        inside: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        coords: np.ndarray,
        radii_eff: np.ndarray,
    ):
        """
        Mark grid voxels that fall inside any atom (with effective radius radii_eff).

        inside : 3D Boolean array (modified in place).
        x, y, z : 1D grid axes (Å).
        coords : (n_atoms, 3) array (Å).
        radii_eff : (n_atoms,) effective radii per atom (Å).
        """
        nx, ny, nz = inside.shape

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]

        for (ax, ay, az), r in zip(coords, radii_eff):
            r2 = r * r

            ix_min = max(int(np.floor((ax - r - x[0]) / dx)), 0)
            ix_max = min(int(np.ceil((ax + r - x[0]) / dx)), nx - 1)
            iy_min = max(int(np.floor((ay - r - y[0]) / dy)), 0)
            iy_max = min(int(np.ceil((ay + r - y[0]) / dy)), ny - 1)
            iz_min = max(int(np.floor((az - r - z[0]) / dz)), 0)
            iz_max = min(int(np.ceil((az + r - z[0]) / dz)), nz - 1)

            xs = x[ix_min : ix_max + 1]
            ys = y[iy_min : iy_max + 1]
            zs = z[iz_min : iz_max + 1]

            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
            dist2 = (X - ax) ** 2 + (Y - ay) ** 2 + (Z - az) ** 2

            mask_local = dist2 <= r2
            inside[ix_min : ix_max + 1, iy_min : iy_max + 1, iz_min : iz_max + 1] |= (
                mask_local
            )
    
    def _mark_occupancy_gaussian_step(
        self,
        density_grid: np.ndarray,           # float grid, modified in-place
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        z_axis: np.ndarray,
        coord: np.ndarray,                   # single atom position (3,)
        radius_eff: float,
        sigma_factor: float | None = None,  # controls sharpness, 0.5–0.7 is ideal
    ):
        """
        Add one atom's Gaussian blob to the density grid.
        This is used for smooth, anti-aliased volume that fixes underestimation.
        """
        if sigma_factor is None:
            sigma_factor = self.sigma_factor
        
        spacing = self.spacing
        ox, oy, oz = x_axis[0], y_axis[0], z_axis[0]
        nx, ny, nz = density_grid.shape
    
        cx, cy, cz = coord
        r_grid = radius_eff / spacing
        sigma = r_grid * sigma_factor
        
        if sigma <= 0:
            return  # Skip if invalid sigma
    
        # Size of local Gaussian patch
        size = int(self.GAUSSIAN_SIZE_MULTIPLIER * sigma) + 1
        if size <= 0:
            return
    
        # Local coordinates
        xx, yy, zz = np.meshgrid(
            np.arange(-size, size + 1),
            np.arange(-size, size + 1),
            np.arange(-size, size + 1),
            indexing='ij'
        )
        gauss = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    
        # Convert atom center to grid indices
        ix_c = (cx - ox) / spacing
        iy_c = (cy - oy) / spacing
        iz_c = (cz - oz) / spacing
    
        i0 = int(ix_c - size)
        i1 = int(ix_c + size + 1)
        j0 = int(iy_c - size)
        j1 = int(iy_c + size + 1)
        k0 = int(iz_c - size)
        k1 = int(iz_c + size + 1)
    
        # Clip to grid bounds
        gi0, gi1 = max(i0, 0), min(i1, nx)
        gj0, gj1 = max(j0, 0), min(j1, ny)
        gk0, gk1 = max(k0, 0), min(k1, nz)
    
        if gi0 >= gi1 or gj0 >= gj1 or gk0 >= gk1:
            return
    
        # Local slice offsets
        li0, li1 = gi0 - i0, gi1 - i0
        lj0, lj1 = gj0 - j0, gj1 - j0
        lk0, lk1 = gk0 - k0, gk1 - k0
    
        density_grid[gi0:gi1, gj0:gj1, gk0:gk1] += gauss[li0:li1, lj0:lj1, lk0:lk1]
    
    
    @staticmethod
    def _mesh_volume_signed(verts, faces):
        """
        Calculate the exact volume of a closed mesh using the signed tetrahedron method.
        (Divergence Theorem)
        """
        # Vertices for each face triangle
        v1 = verts[faces[:, 0]]
        v2 = verts[faces[:, 1]]
        v3 = verts[faces[:, 2]]
        
        # Cross product => area * normal
        cross = np.cross(v2, v3)
        
        # Dot product with v1 / 6.0
        return np.abs(np.sum(v1 * cross)) / 6.0   
    
    def _get_rdkit_mol_from_ag(self, ag: Optional[mda.AtomGroup] = None) -> Optional[object]:
        """
        Try to convert an AtomGroup to an RDKit Mol via AtomGroup.convert_to("RDKIT").

        Parameters
        ----------
        ag : Optional[mda.AtomGroup]
            AtomGroup to convert. If None, creates one from current universe.

        Returns
        -------
        mol : rdkit.Chem.Mol or None
            RDKit molecule object, or None if conversion fails or RDKit unavailable.
        """
        if AllChem is None:
            return None
        if ag is None:
            # Get atom group from universe
            u = self._get_universe()
            ag = u.select_atoms(self.selection)
        mol = None
        try:
            # MDAnalysisConverters must be installed for this to work:
            #   pip install MDAnalysisConverters
            mol = ag.convert_to("RDKIT")
        except Exception as e:
            warnings.warn(
                f"Failed to convert AtomGroup to RDKit Mol: {e}",
                UserWarning,
                stacklevel=2
            )
            return None

        if mol is None:
            return None

        # Ensure there is at least one conformer (needed for some RDKit ops)
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)

        return mol

    # ----------- cavity detection -----------

    @staticmethod
    def _detect_cavities_from_inside(inside: np.ndarray) -> np.ndarray:
        """
        Identify enclosed cavities as empty voxels not connected to the grid boundary.

        Parameters
        ----------
        inside : 3D Boolean array
            True = target (inflated by probe), False = empty.

        Returns
        -------
        cavities : 3D Boolean array
            True = cavity voxels (enclosed empty space).
        """
        empty = ~inside

        nx, ny, nz = empty.shape
        outside_seed = np.zeros_like(empty, dtype=bool)

        # Mark empty voxels at the boundary as seeds for "outside"
        outside_seed[0, :, :] |= empty[0, :, :]
        outside_seed[nx - 1, :, :] |= empty[nx - 1, :, :]
        outside_seed[:, 0, :] |= empty[:, 0, :]
        outside_seed[:, ny - 1, :] |= empty[:, ny - 1, :]
        outside_seed[:, :, 0] |= empty[:, :, 0]
        outside_seed[:, :, nz - 1] |= empty[:, :, nz - 1]

        # 6-connected neighborhood
        structure = ndimage.generate_binary_structure(rank=3, connectivity=1)

        # Grow from outside_seed but only inside "empty" region
        outside_region = ndimage.binary_propagation(
            outside_seed, structure=structure, mask=empty
        )

        # Cavities = empty voxels that are NOT in outside_region
        cavities = empty & (~outside_region)
        return cavities
    
    # ----------- per-frame computation -----------
    def _effective_radii(self) -> np.ndarray:
        """
        Apply radii scaling & offset, then add probe radius.
        """
        r = self.vdw_radii * self.radii_scale + self.radii_offset
        return r + self.probe_radius
    
    def _get_universe(self, frame_index: Optional[int] = None) -> mda.Universe:
        """Get Universe, creating on-demand if using file paths."""
        if self._use_file_paths:
            # Create Universe on-demand (lightweight, doesn't load full trajectory)
            try:
                if self.traj_format:
                    u = mda.Universe(self.top_file, self.traj_file, format=self.traj_format)
                else:
                    u = mda.Universe(self.top_file, self.traj_file)
                if frame_index is not None:
                    u.trajectory[frame_index]
                return u
            except Exception as e:
                raise RuntimeError(f"Failed to create Universe from file paths: {e}")
        else:
            if self._universe is None:
                raise RuntimeError("Universe not available")
            if frame_index is not None:
                self._universe.trajectory[frame_index]
            return self._universe
    
    def compute_frame(
        self,
        frame_index: int,
        return_masks: bool = False,
        use_mesh_volume: bool = True,
        universe: Optional[mda.Universe] = None,
    ) -> tuple[float, float] | tuple[float, float, np.ndarray, np.ndarray]:
        """
        Compute target and cavity volume for a single frame.

        Parameters
        ----------
        frame_index : int
            Index of the frame in universe.trajectory. Must be >= 0 and < len(trajectory).
        return_masks : bool
            If True, also return the inside and cavity masks for visualization.
        use_mesh_volume : bool
            If True, use marching cubes mesh volume (more accurate).
        universe : Optional[mda.Universe]
            Optional Universe already positioned at frame_index. If provided, uses this
            instead of creating/loading from file paths. Useful when Universe is already
            available (e.g., from TrajectoryIterator).

        Returns
        -------
        If return_masks is False:
            target_volume_A3, cavity_volume_A3
        If return_masks is True:
            target_volume_A3, cavity_volume_A3, inside_mask, cavity_mask
        """
        # Use provided universe if available (more efficient)
        if universe is not None:
            u = universe
            # Verify frame_index is valid
            if frame_index < 0 or frame_index >= len(u.trajectory):
                raise ValueError(
                    f"frame_index {frame_index} out of range "
                    f"(trajectory has {len(u.trajectory)} frames)"
                )
            u.trajectory[frame_index]
        else:
            # Get universe (creates on-demand if using file paths)
            u = self._get_universe(frame_index)
            # Verify frame_index is valid
            if frame_index < 0 or frame_index >= len(u.trajectory):
                raise ValueError(
                    f"frame_index {frame_index} out of range "
                    f"(trajectory has {len(u.trajectory)} frames)"
                )
        
        # Get atom group and coordinates
        ag = u.select_atoms(self.selection)
        if ag.n_atoms == 0:
            raise ValueError(f"Selection '{self.selection}' returned no atoms.")
        coords = ag.positions.copy()
        
        # Get VDW radii (use precomputed if available, otherwise compute on-demand)
        if self.vdw_radii is None or len(self.vdw_radii) != ag.n_atoms:
            vdw_radii = self._get_vdw_radii(ag)
        else:
            vdw_radii = self.vdw_radii

        # Effective radii: VDW + probe
        radii_eff = vdw_radii * self.radii_scale + self.radii_offset + self.probe_radius

        # Margin: either user-specified or auto
        if self.margin is not None:
            margin = self.margin
        else:
            margin = np.max(radii_eff) + self.spacing

        x, y, z, shape, origin = self._build_grid(coords, self.spacing, margin)
        self._last_grid_axes = (x, y, z)
        # 1. Generate Density/Boolean Grid
        if self.use_gaussian_step:
            density = np.zeros(shape, dtype=float)
            for i in range(len(coords)):
                self._mark_occupancy_gaussian_step(density, x, y, z, coords[i], radii_eff[i])
            inside = density > self.density_threshold
            grid_for_meshing = density
        else:
            inside = np.zeros(shape, dtype=bool)
            self._mark_occupancy(inside, x, y, z, coords, radii_eff)
            grid_for_meshing = inside.astype(float)
        # Optional surface smoothing (important for rings / tight contacts)
        if self.smooth_surface and self.smooth_iterations > 0:
            struct = ndimage.generate_binary_structure(rank=3, connectivity=1)
            inside = ndimage.binary_closing(
                inside, structure=struct, iterations=self.smooth_iterations
            )
            # Binary opening removes small isolated regions
            if self.apply_binary_opening:
                inside = ndimage.binary_opening(inside, structure=struct, iterations=1)
        voxel_volume = self.spacing**3

        # 2. Detect Cavities (Topology is still best done on the grid, this is just a fast approximation)
        cavities = self._detect_cavities_from_inside(inside)
        # 3. Calculate Volumes
        if use_mesh_volume and measure is not None:
            # --- Precision Method: Marching Cubes ---
            # Determine isovalue based on method
            if self.use_gaussian_step:
                isovalue = self.density_threshold
            else:
                isovalue = 0.5
            
            # Target Volume
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    grid_for_meshing, level=isovalue, spacing=(self.spacing,)*3
                )
                target_volume = self._mesh_volume_signed(verts, faces)
            except (ValueError, RuntimeError):
                # Fallback if mesh fails (e.g. empty grid)
                target_volume = inside.sum() * voxel_volume

            # Cavity Volume
            # We mesh the cavity mask. Note: Cavity mask is boolean.
            try:
                if cavities.sum() > 0:
                    verts_c, faces_c, _, _ = measure.marching_cubes(
                        cavities.astype(float), level=0.5, spacing=(self.spacing,)*3
                    )
                    cavity_volume = self._mesh_volume_signed(verts_c, faces_c)
                else:
                    cavity_volume = 0.0
            except (ValueError, RuntimeError):
                cavity_volume = cavities.sum() * (self.spacing**3)

        else:
            # --- Standard Method: Voxel Counting ---
            voxel_volume = self.spacing**3
            target_volume = inside.sum() * voxel_volume
            cavity_volume = cavities.sum() * voxel_volume

        if return_masks:
            return target_volume, cavity_volume, inside, cavities
        else:
            return target_volume, cavity_volume

    # ----------- trajectory analysis -----------

    def analyze_trajectory(
        self,
        stride: int = 1,
        plot_frame: int | None = None,
        plot_axis: str = "z",
        plot_slice_index: int | None = None,
        plot_png: str | None = None,
    ) -> list[dict[str, float | int]]:
        """
        Analyze all frames in the trajectory (with optional plotting for one frame).

        Parameters
        ----------
        stride : int
            Analyze every `stride`-th frame. Must be >= 1.
        plot_frame : int or None
            If not None, generate a 2D sanity slice plot for this frame.
        plot_axis : {'x', 'y', 'z'}
            Axis normal to the slice for plotting (default: 'z').
        plot_slice_index : int or None
            Index of slice along chosen axis; if None, use middle slice.
        plot_png : str or None
            Output PNG filename for the slice plot. If None, default name is used.

        Returns
        -------
        results : list of dict
            Each dict: {
                "frame": int,
                "time_ps": float,
                "target_volume_A3": float,
                "cavity_volume_A3": float,
            }
        """
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        results = []

        # Get universe for iteration
        u = self._get_universe()
        for ts in u.trajectory[::stride]:
            frame_index = ts.frame
            time_ps = ts.time

            if plot_frame is not None and frame_index == plot_frame:
                target_vol, cavity_vol, inside, cavities = self.compute_frame(
                    frame_index, return_masks=True, universe=u
                )

                self.plot_cavity_slice(
                    inside=inside,
                    cavities=cavities,
                    axis=plot_axis,
                    index=plot_slice_index,
                    outfile=plot_png or f"cavity_frame{frame_index}_{plot_axis}.png",
                )
            else:
                target_vol, cavity_vol = self.compute_frame(frame_index, universe=u)

            results.append(
                {
                    "frame": frame_index,
                    "time_ps": float(time_ps),
                    "target_volume_A3": float(target_vol),
                    "cavity_volume_A3": float(cavity_vol),
                }
            )

        return results

    # ----------- plotting -----------

    @staticmethod
    def plot_cavity_slice(
        inside: np.ndarray,
        cavities: np.ndarray,
        axis: str = "z",
        index: int | None = None,
        outfile: str = "cavity_slice.png",
    ):
        """
        Plot a 2D slice through the 3D grid for sanity checking.

        Parameters
        ----------
        inside : 3D bool array
            True = target.
        cavities : 3D bool array
            True = cavity voxels.
        axis : {'x', 'y', 'z'}
            Axis normal to the slice.
        index : int or None
            Slice index along chosen axis. If None, use middle slice.
        outfile : str
            Output PNG filename.
        """
        if plt is None or ListedColormap is None:
            warnings.warn("matplotlib not available. Cannot plot cavity slice.")
            return

        nx, ny, nz = inside.shape
        axis = axis.lower()

        if axis == "z":
            if index is None or index < 0 or index >= nz:
                index = nz // 2
            prot2d = inside[:, :, index]
            cav2d = cavities[:, :, index]
        elif axis == "y":
            if index is None or index < 0 or index >= ny:
                index = ny // 2
            prot2d = inside[:, index, :]
            cav2d = cavities[:, index, :]
        elif axis == "x":
            if index is None or index < 0 or index >= nx:
                index = nx // 2
            prot2d = inside[index, :, :]
            cav2d = cavities[index, :, :]
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        arr = np.zeros_like(prot2d, dtype=int)
        arr[prot2d] = 1   # target
        arr[cav2d] = 2    # cavity

        cmap = ListedColormap(["white", "black", "red"])

        plt.figure(figsize=(5, 5))
        im = plt.imshow(arr.T, origin="lower", cmap=cmap, interpolation="nearest")
        plt.title(f"Cavity slice (axis={axis}, index={index})")
        plt.xlabel("grid index")
        plt.ylabel("grid index")
        cbar = plt.colorbar(im, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["Outside", "target", "Cavity"])
        plt.tight_layout()
        plt.savefig(outfile, dpi=300)
        plt.close()


    def plot_interactive_3d(
        self,
        frame_index: int = 0,
        voxel_stride: int = 1,
    ):
        """
        Interactive 3D visualization of:
        - molecule/target (atoms + bonds, with annotation menu)
        - target volume (inside mask) as a Volume isosurface
        - cavity volume (cavities mask) as a Volume isosurface
        
        Parameters
        ----------
        frame_index : int
            Frame index in universe.trajectory.
        voxel_stride : int
            Subsampling step for the grid in each dimension (>=1).
            1 = full grid; 2 = take every 2nd voxel, etc.
            
        Returns
        -------
        fig : go.Figure
        """
        if go is None or make_molecule_components is None:
            raise RuntimeError("plotly and plotly_molecule are required for interactive 3D visualization.")

        # Get universe for this frame
        u = self._get_universe(frame_index)
        
        # --- 1. Compute volumes & masks on the same grid used in compute_frame ---
        target_vol, cavity_vol, inside, cavities = self.compute_frame(
            frame_index, return_masks=True, universe=u
        )
        
        if self._last_grid_axes is None:
            raise RuntimeError(
                "Grid axes not stored; make sure compute_frame sets self._last_grid_axes."
            )
        x_axis, y_axis, z_axis = self._last_grid_axes
        
        # Optional grid subsampling for rendering speed
        if voxel_stride > 1:
            inside_sub = inside[::voxel_stride, ::voxel_stride, ::voxel_stride]
            cavities_sub = cavities[::voxel_stride, ::voxel_stride, ::voxel_stride]
            x_sub = x_axis[::voxel_stride]
            y_sub = y_axis[::voxel_stride]
            z_sub = z_axis[::voxel_stride]
        else:
            inside_sub = inside
            cavities_sub = cavities
            x_sub, y_sub, z_sub = x_axis, y_axis, z_axis
            
        # Build grid of voxel centers
        X, Y, Z = np.meshgrid(x_sub, y_sub, z_sub, indexing="ij")
        
        # Scalar fields for Volume plots (0/1 occupancy)
        values_target = inside_sub.astype(float)
        values_cavity = cavities_sub.astype(float)
        
        # Flatten for Plotly Volume
        Xr = X.ravel()
        Yr = Y.ravel()
        Zr = Z.ravel()
        Vr_target = values_target.ravel()
        Vr_cavity = values_cavity.ravel()
        
        # --- 2. Molecule representation from helper (atoms + bonds + menus) ---
        # Get coords/elements from universe (already at frame_index)
        ag = u.select_atoms(self.selection)
        coords = ag.positions.copy()
        elements = [self._get_element(atom) or "C" for atom in ag.atoms]
        
        atom_trace, bond_trace, annotations_id, annotations_length, updatemenus = \
            make_molecule_components(coords, elements)
        
        # --- 3. Volume traces for target & cavities ---
        # target volume: semi-transparent gray shell
        vol_target = go.Volume(
            x=Xr,
            y=Yr,
            z=Zr,
            value=Vr_target,
            isomin=0.5,   # "inside" voxels are 1; surface at ~0.5
            isomax=1.5,
            opacity=0.1,  # overall opacity of the volume
            surface_count=20,
            name="target volume",
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "grey"]],
            showscale=False,
            showlegend=True,
        )
        
        # Cavity volume: more opaque red shell
        vol_cavity = go.Volume(
            x=Xr,
            y=Yr,
            z=Zr,
            value=Vr_cavity,
            isomin=0.5,
            isomax=1.5,
            opacity=0.3,
            surface_count=20,
            name="Cavity volume",
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "red"]],
            showscale=False,
            showlegend=True,
        )
        
        # --- 4. Layout (same axis style as your molecule plot) ---
        axis_params = dict(
            showgrid=True,
            showbackground=True,
            showticklabels=True,
            zeroline=False,
            tickfont=dict(color='black'),
        )
        
        # existing annotation menu from make_molecule_components
        annotation_menus = updatemenus
        # Trace order: 0: atoms, 1: bonds, 2: target volume, 3: cavity volume
        volume_menu = dict(
            type="buttons",
            direction="right",
            x=0.5,
            y=1.08,
            xanchor="center",
            yanchor="bottom",
            buttons=[
                dict(
                    label="Show both vols",
                    method="update",
                    args=[{"visible": [True, True, True, True]}],
                ),
                dict(
                    label="Hide vols",
                    method="update",
                    args=[{"visible": [True, True, False, False]}],
                ),
                dict(
                   label="Only target vol",
                    method="update",
                    args=[{"visible": [True, True, True, False]}],
                ),
                dict(
                    label="Only cavity vol",
                    method="update",
                    args=[{"visible": [True, True, False, True]}],
                ),
            ],
        )        
        
        layout = dict(
            scene=dict(
                xaxis=axis_params,
                yaxis=axis_params,
                zaxis=axis_params,
                annotations=annotations_id,
                aspectmode="data",
            ),
            margin=dict(r=100, l=100, b=100, t=100),
            showlegend=True,
            updatemenus=updatemenus+[volume_menu],
            title=(
                f"Frame {frame_index} — "
                f"V_target ≈ {target_vol:.0f} Å³, "
                f"V_cavity ≈ {cavity_vol:.0f} Å³"
            ),
        )
        
        fig = go.Figure(
            data=[atom_trace, bond_trace, vol_target, vol_cavity],
            layout=layout,
        )
        return fig

    def make_volume_pipeline_gif(
        self,
        frame_index: int = 0,
        gif_path: str = "volume_pipeline.gif",
        atom_stride: int = 5,
        voxel_stride: int = 5,
        max_cavity_steps: int | None = None,
        elev: float = 20.0,
        azim: float = -60.0,
        fps: float = 10.0,
        show_inside_shell: bool = True,
        use_mesh_volume: bool = True,
    ):
        """
        Unified GIF for a single frame using the same gaussian approach as compute_frame:

        Phase 1: _mark_occupancy_gaussian_step (when use_gaussian_step=True)
            - Density grid accumulates gaussian blobs from each atom
            - Inside mask is derived by thresholding density (uses self.density_threshold)
        Phase 2: smoothing + cavity detection
            - Inside mask is optionally smoothed (binary closing + opening)
            - Outside region flood-fills empty space from boundary
            - Cavities = empty & ~outside_region

        Visualization:
            - gray shell  = thin surface of inside (optional)
            - red points  = cavity voxels (in cavity phase only)
            - colored dots= MDAnalysis atoms
            - Uses the same gaussian density approach as compute_frame for accuracy

        Parameters
        ----------
        frame_index : int
            Frame index in universe.trajectory. Must be >= 0 and < len(trajectory).
        gif_path : str
            Output path for the GIF file.
        atom_stride : int
            Render frame every N atoms during occupancy phase. Must be >= 1.
        voxel_stride : int
            Subsampling step for visualization grid. Must be >= 1.
        max_cavity_steps : int or None
            Maximum cavity detection steps. If None, uses grid dimensions.
        elev : float
            Elevation angle for 3D view.
        azim : float
            Azimuth angle for 3D view.
        fps : float
            Frames per second for GIF. Must be > 0.
        show_inside_shell : bool
            If True, show thin shell of inside volume for orientation.
        use_mesh_volume : bool
            If True and scikit-image is available, use marching cubes mesh volume
            for more accurate volume calculations. If False, use voxel counting.
        """
        if imageio is None:
            raise RuntimeError("imageio is required for GIF generation.")
        if plt is None:
            raise RuntimeError("matplotlib is required for GIF generation.")

        if voxel_stride < 1:
            raise ValueError("voxel_stride must be >= 1")
        if atom_stride < 1:
            raise ValueError("atom_stride must be >= 1")
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        # Get universe for this frame
        u = self._get_universe(frame_index)
        
        if frame_index < 0:
            raise ValueError(f"frame_index must be >= 0, got {frame_index}")
        if frame_index >= len(u.trajectory):
            raise ValueError(
                f"frame_index {frame_index} out of range "
                f"(trajectory has {len(u.trajectory)} frames)"
            )

        # --- 1. Set frame and build grid (as in compute_frame) ---
        ag = u.select_atoms(self.selection)
        coords = ag.positions.copy()
        
        # Get VDW radii (use precomputed if available, otherwise compute on-demand)
        if self.vdw_radii is None or len(self.vdw_radii) != ag.n_atoms:
            vdw_radii = self._get_vdw_radii(ag)
        else:
            vdw_radii = self.vdw_radii
        
        # Effective radii: VDW + probe
        radii_eff = vdw_radii * self.radii_scale + self.radii_offset + self.probe_radius

        if self.margin is not None:
            margin = self.margin
        else:
            margin = np.max(radii_eff) + self.spacing

        x_axis, y_axis, z_axis, shape, origin = self._build_grid(
            coords, self.spacing, margin
        )
        self._last_grid_axes = (x_axis, y_axis, z_axis)

        # Use the same approach as compute_frame
        use_gaussian = self.use_gaussian_step
        
        if use_gaussian:
            # Initialize density grid (same as compute_frame)
            density_grid = np.zeros(shape, dtype=float)
            inside_raw = np.zeros(shape, dtype=bool)
        else:
            # Fallback to binary occupancy
            inside_raw = np.zeros(shape, dtype=bool)
            density_grid = None
        
        n_atoms = coords.shape[0]

        # --- MD structure in subsampled grid coordinates ---
        x0, y0, z0 = x_axis[0], y_axis[0], z_axis[0]
        step_size = self.spacing * voxel_stride

        gx = (coords[:, 0] - x0) / step_size
        gy = (coords[:, 1] - y0) / step_size
        gz = (coords[:, 2] - z0) / step_size

        elements = [self._get_element(atom) or "C" for atom in ag.atoms]
        color_map = {
            "H": "lightgray",
            "C": "black",
            "N": "blue",
            "O": "red",
            "F": "green",
            "S": "orange",
            "P": "purple",
        }
        atom_colors = [color_map.get(el.upper(), "gray") for el in elements]

        nx_full, ny_full, nz_full = shape
        nx_sub = (nx_full - 1) // voxel_stride + 1
        ny_sub = (ny_full - 1) // voxel_stride + 1
        nz_sub = (nz_full - 1) // voxel_stride + 1

        images = []

        # Struct for shell erosion
        struct3 = ndimage.generate_binary_structure(3, 1)

        voxel_volume = self.spacing ** 3

        def compute_mesh_volumes(
            inside_mask: np.ndarray,
            cavity_mask: np.ndarray | None,
            grid_for_meshing: np.ndarray,
        ) -> tuple[float | None, float | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
            """
            Compute mesh volumes and surfaces for inside and cavity masks.
            
            Returns
            -------
            inside_vol_mesh, cavity_vol_mesh, inside_verts, inside_faces, cavity_verts, cavity_faces
                Mesh-calculated volumes and mesh geometry (vertices and faces).
                Returns None for volumes/geometry if mesh computation fails/unavailable.
            """
            if not (use_mesh_volume and measure is not None):
                return None, None, None, None, None, None
            
            # Determine isovalue based on method
            if use_gaussian:
                isovalue = self.density_threshold
            else:
                isovalue = 0.5
            
            # Calculate inside volume and mesh
            inside_vol_mesh = None
            inside_verts = None
            inside_faces = None
            try:
                verts, faces, _, _ = measure.marching_cubes(
                    grid_for_meshing, level=isovalue, spacing=(self.spacing,)*3
                )
                inside_vol_mesh = self._mesh_volume_signed(verts, faces)
                inside_verts = verts
                inside_faces = faces
            except (ValueError, RuntimeError):
                pass  # Fallback to voxel counting
            
            # Calculate cavity volume and mesh
            cavity_vol_mesh = None
            cavity_verts = None
            cavity_faces = None
            if cavity_mask is not None:
                try:
                    if cavity_mask.sum() > 0:
                        verts_c, faces_c, _, _ = measure.marching_cubes(
                            cavity_mask.astype(float), level=0.5, spacing=(self.spacing,)*3
                        )
                        cavity_vol_mesh = self._mesh_volume_signed(verts_c, faces_c)
                        cavity_verts = verts_c
                        cavity_faces = faces_c
                    else:
                        cavity_vol_mesh = 0.0
                except (ValueError, RuntimeError):
                    pass  # Fallback to voxel counting
            
            return inside_vol_mesh, cavity_vol_mesh, inside_verts, inside_faces, cavity_verts, cavity_faces

        def render_state(
            phase: str,
            step_idx: int,
            inside_mask: np.ndarray,
            density_grid_vis: np.ndarray | None = None,
            outside_region: np.ndarray | None = None,
            cavity_mask: np.ndarray | None = None,
            inside_vol_mesh: float | None = None,
            cavity_vol_mesh: float | None = None,
            inside_verts: np.ndarray | None = None,
            inside_faces: np.ndarray | None = None,
            cavity_verts: np.ndarray | None = None,
            cavity_faces: np.ndarray | None = None,
        ):
            """
            Render a single frame.

            - phase: "occupancy" or "cavity"
            - inside_mask: 3D bool of inside voxels (raw or smoothed)
            - density_grid_vis: 3D float array of density values (for gaussian visualization)
            - outside_region: 3D bool of outside flood (None in occupancy phase)
            - cavity_mask: 3D bool of cavities (None in occupancy phase)
            - inside_vol_mesh: float or None - mesh-calculated inside volume (if available)
            - cavity_vol_mesh: float or None - mesh-calculated cavity volume (if available)
            - inside_verts: np.ndarray or None - mesh vertices for inside surface
            - inside_faces: np.ndarray or None - mesh faces for inside surface
            - cavity_verts: np.ndarray or None - mesh vertices for cavity surface
            - cavity_faces: np.ndarray or None - mesh faces for cavity surface
            """
            # subsample
            vis_inside = inside_mask[::voxel_stride, ::voxel_stride, ::voxel_stride]

            if cavity_mask is not None:
                vis_cav = cavity_mask[::voxel_stride, ::voxel_stride, ::voxel_stride]
            else:
                vis_cav = np.zeros_like(vis_inside, dtype=bool)

            # For gaussian visualization, show density-based shell with opacity
            if use_gaussian and density_grid_vis is not None:
                vis_density = density_grid_vis[::voxel_stride, ::voxel_stride, ::voxel_stride]
                # Normalize density for visualization (0-1 range)
                if vis_density.max() > 0:
                    vis_density_norm = np.clip(vis_density / vis_density.max(), 0, 1)
                else:
                    vis_density_norm = vis_density
            else:
                vis_density_norm = None

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")

            # Convert mesh vertices from physical coordinates (Å) to grid indices
            # Mesh vertices are in physical space, need to convert to grid space for visualization
            x0, y0, z0 = x_axis[0], y_axis[0], z_axis[0]
            step_size = self.spacing * voxel_stride

            # Visualize inside mesh surface
            if inside_verts is not None and inside_faces is not None and show_inside_shell:
                # Convert mesh vertices from marching_cubes coordinates to visualization grid indices
                # marching_cubes with spacing returns vertices in physical coordinates
                # where the grid origin (first voxel center) is at approximately (spacing/2, spacing/2, spacing/2)
                # in the marching_cubes coordinate system (which starts at 0,0,0)
                # Our grid origin in physical space is at (x0, y0, z0)
                # To align with atoms (which use: (coords - [x0,y0,z0]) / step_size):
                # Convert marching_cubes coords to absolute physical coords, then to grid indices
                verts_absolute = inside_verts + np.array([x0, y0, z0])
                verts_grid = (verts_absolute - np.array([x0, y0, z0])) / step_size
                # Extract x, y, z coordinates for plot_trisurf
                x_mesh = verts_grid[:, 0]
                y_mesh = verts_grid[:, 1]
                z_mesh = verts_grid[:, 2]
                # Plot mesh surface with semi-transparent gray
                ax.plot_trisurf(
                    x_mesh, y_mesh, z_mesh,
                    triangles=inside_faces,
                    alpha=0.2,
                    color='gray',
                    edgecolor='none',
                    shade=True,
                )
            elif show_inside_shell:
                # Fallback to voxel visualization if mesh not available
                inner = ndimage.binary_erosion(
                    vis_inside, structure=struct3, border_value=0
                )
                shell = vis_inside & (~inner)
                if np.any(shell):
                    if use_gaussian and vis_density_norm is not None:
                        # Use density-based opacity for more accurate visualization (vectorized)
                        shell_colors = np.zeros(shell.shape + (4,), dtype=float)
                        # Vectorized assignment: set RGB to gray and opacity based on density
                        shell_colors[shell, 0] = 0.8  # R
                        shell_colors[shell, 1] = 0.8  # G
                        shell_colors[shell, 2] = 0.8  # B
                        shell_colors[shell, 3] = 0.1 + 0.15 * vis_density_norm[shell]  # A
                        ax.voxels(shell, facecolors=shell_colors, edgecolor=None)
                    else:
                        # Binary visualization (fallback)
                        shell_colors = np.zeros(shell.shape + (4,), dtype=float)
                        shell_colors[..., 3] = 0.0
                        shell_colors[shell] = (0.8, 0.8, 0.8, 0.15)
                        ax.voxels(shell, facecolors=shell_colors, edgecolor=None)

            # Visualize cavity mesh surface
            if cavity_verts is not None and cavity_faces is not None:
                # Convert mesh vertices from marching_cubes coordinates to visualization grid indices
                # Same conversion as inside mesh
                verts_cav_physical = cavity_verts + np.array([x0, y0, z0])
                verts_cav_grid = (verts_cav_physical - np.array([x0, y0, z0])) / step_size
                # Extract x, y, z coordinates for plot_trisurf
                x_cav = verts_cav_grid[:, 0]
                y_cav = verts_cav_grid[:, 1]
                z_cav = verts_cav_grid[:, 2]
                # Plot cavity mesh surface with semi-transparent red
                ax.plot_trisurf(
                    x_cav, y_cav, z_cav,
                    triangles=cavity_faces,
                    alpha=0.4,
                    color='red',
                    edgecolor='none',
                    shade=True,
                )
            else:
                # Fallback to scatter plot if mesh not available
                cav_ix, cav_iy, cav_iz = np.where(vis_cav)
                if cav_ix.size > 0:
                    ax.scatter(
                        cav_ix,
                        cav_iy,
                        cav_iz,
                        s=25,
                        c="red",
                        depthshade=False,
                    )

            # MD structure
            ax.scatter(gx, gy, gz, s=10, c=atom_colors, depthshade=False)

            ax.set_xlim(0, nx_sub)
            ax.set_ylim(0, ny_sub)
            ax.set_zlim(0, nz_sub)
            ax.set_box_aspect((nx_sub, ny_sub, nz_sub))

            ax.view_init(elev=elev, azim=azim)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            # Calculate volumes - use mesh volume if available, otherwise voxel counting
            if inside_vol_mesh is not None:
                inside_vol = inside_vol_mesh
            else:
                inside_vol = inside_mask.sum() * voxel_volume
            
            if cavity_vol_mesh is not None:
                cav_vol = cavity_vol_mesh
            elif cavity_mask is not None:
                cav_vol = cavity_mask.sum() * voxel_volume
            else:
                cav_vol = 0.0
            
            total_vol = inside_vol + cav_vol

            method_str = "Gaussian" if use_gaussian else "Binary"
            volume_method = "Mesh" if (inside_vol_mesh is not None or cavity_vol_mesh is not None) else "Voxel"
            title_phase = "Occupancy build" if phase == "occupancy" else "Cavity detection"
            ax.set_title(
                f"{title_phase} ({method_str}, {volume_method}) – step {step_idx}\n"
                f"V_total ≈ {total_vol:.0f} Å³, "
                f"V_cavity ≈ {cav_vol:.0f} Å³",
                fontsize=8,
            )

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150)
            plt.close(fig)
            buf.seek(0)
            img = imageio.imread(buf)
            images.append(img)

        # ------------------------
        # PHASE 1: _mark_occupancy
        # ------------------------

        # Prepare grid_for_meshing for mesh volume calculation
        if use_gaussian:
            grid_for_meshing = density_grid
        else:
            grid_for_meshing = inside_raw.astype(float)

        # initial empty grid
        inside_vol_mesh, cavity_vol_mesh, inside_verts, inside_faces, cavity_verts, cavity_faces = compute_mesh_volumes(
            inside_raw, None, grid_for_meshing
        )
        render_state(
            phase="occupancy",
            step_idx=0,
            inside_mask=inside_raw,
            density_grid_vis=density_grid if use_gaussian else None,
            outside_region=None,
            cavity_mask=None,
            inside_vol_mesh=inside_vol_mesh,
            cavity_vol_mesh=cavity_vol_mesh,
            inside_verts=inside_verts,
            inside_faces=inside_faces,
            cavity_verts=cavity_verts,
            cavity_faces=cavity_faces,
        )

        for i in range(n_atoms):
            # stamp a single atom using the same method as compute_frame
            if use_gaussian:
                # Use the same gaussian step function as compute_frame
                self._mark_occupancy_gaussian_step(
                    density_grid, x_axis, y_axis, z_axis, 
                    coords[i], radii_eff[i]
                )
                # Update inside_raw from density_grid by thresholding (same as compute_frame)
                inside_raw = density_grid > self.density_threshold
            else:
                # Fallback to binary occupancy
                self._mark_occupancy(
                    inside_raw,
                    x_axis,
                    y_axis,
                    z_axis,
                    coords[i : i + 1],
                    radii_eff[i : i + 1],
                )
            n_done = i + 1
            if (n_done % atom_stride == 0) or (n_done == n_atoms):
                # Update grid_for_meshing
                if use_gaussian:
                    grid_for_meshing = density_grid
                else:
                    grid_for_meshing = inside_raw.astype(float)
                
                inside_vol_mesh, cavity_vol_mesh, inside_verts, inside_faces, cavity_verts, cavity_faces = compute_mesh_volumes(
                    inside_raw, None, grid_for_meshing
                )
                render_state(
                    phase="occupancy",
                    step_idx=n_done,
                    inside_mask=inside_raw,
                    density_grid_vis=density_grid if use_gaussian else None,
                    outside_region=None,
                    cavity_mask=None,
                    inside_vol_mesh=inside_vol_mesh,
                    cavity_vol_mesh=cavity_vol_mesh,
                    inside_verts=inside_verts,
                    inside_faces=inside_faces,
                    cavity_verts=cavity_verts,
                    cavity_faces=cavity_faces,
                )


        # ------------------------
        # PHASE 2: smoothing + cavities
        # ------------------------

        # optional smoothing (same as compute_frame)
        if self.smooth_surface and self.smooth_iterations > 0:
            struct = ndimage.generate_binary_structure(rank=3, connectivity=1)
            inside_smooth = ndimage.binary_closing(
                inside_raw, structure=struct, iterations=self.smooth_iterations
            )
            # Binary opening removes small isolated regions
            if self.apply_binary_opening:
                inside_smooth = ndimage.binary_opening(inside_smooth, structure=struct, iterations=1)
        else:
            inside_smooth = inside_raw.copy()

        # Update grid_for_meshing after smoothing
        if use_gaussian:
            # For Gaussian, we still use the density grid (smoothing affects inside mask, not density)
            grid_for_meshing = density_grid
        else:
            # For binary, update to smoothed mask
            grid_for_meshing = inside_smooth.astype(float)

        empty = ~inside_smooth

        # seeds on boundary
        nx, ny, nz = empty.shape
        outside_region = np.zeros_like(empty, dtype=bool)
        outside_region[0, :, :] |= empty[0, :, :]
        outside_region[nx - 1, :, :] |= empty[nx - 1, :, :]
        outside_region[:, 0, :] |= empty[:, 0, :]
        outside_region[:, ny - 1, :] |= empty[:, ny - 1, :]
        outside_region[:, :, 0] |= empty[:, :, 0]
        outside_region[:, :, nz - 1] |= empty[:, :, nz - 1]

        struct_out = ndimage.generate_binary_structure(rank=3, connectivity=1)

        if max_cavity_steps is None:
            max_cavity_steps = nx + ny + nz

        # initial cavity mask
        cav_full = empty & (~outside_region)
        inside_vol_mesh, cavity_vol_mesh, inside_verts, inside_faces, cavity_verts, cavity_faces = compute_mesh_volumes(
            inside_smooth, cav_full, grid_for_meshing
        )
        render_state(
            phase="cavity",
            step_idx=0,
            inside_mask=inside_smooth,
            density_grid_vis=density_grid if use_gaussian else None,
            outside_region=outside_region,
            cavity_mask=cav_full,
            inside_vol_mesh=inside_vol_mesh,
            cavity_vol_mesh=cavity_vol_mesh,
            inside_verts=inside_verts,
            inside_faces=inside_faces,
            cavity_verts=cavity_verts,
            cavity_faces=cavity_faces,
        )

        for k in range(1, max_cavity_steps + 1):
            dilated = ndimage.binary_dilation(outside_region, structure=struct_out)
            outside_next = (dilated & empty) | outside_region

            cav_full = empty & (~outside_next)
            
            # Compute mesh volumes for current cavity state
            inside_vol_mesh, cavity_vol_mesh, inside_verts, inside_faces, cavity_verts, cavity_faces = compute_mesh_volumes(
                inside_smooth, cav_full, grid_for_meshing
            )

            render_state(
                phase="cavity",
                step_idx=k,
                inside_mask=inside_smooth,
                density_grid_vis=density_grid if use_gaussian else None,
                outside_region=outside_next,
                cavity_mask=cav_full,
                inside_vol_mesh=inside_vol_mesh,
                cavity_vol_mesh=cavity_vol_mesh,
                inside_verts=inside_verts,
                inside_faces=inside_faces,
                cavity_verts=cavity_verts,
                cavity_faces=cavity_faces,
            )

            if np.array_equal(outside_next, outside_region):
                break

            outside_region = outside_next

        if not images:
            raise RuntimeError("No frames collected for GIF.")

        imageio.mimsave(gif_path, images, duration=1.0 / fps)
        # Note: File save success is indicated by function return without exception

