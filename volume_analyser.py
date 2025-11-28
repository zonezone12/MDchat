import numpy as np
import MDAnalysis as mda

from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rdkit.Chem import AllChem, Draw

# Import plotly_molecule with fallback for different import contexts
try:
    from MD_analysis.plotly_molecule import make_molecule_components
except ImportError:
    try:
        from .plotly_molecule import make_molecule_components
    except ImportError:
        # Fallback to direct import (works when run from within the directory)
        from plotly_molecule import make_molecule_components

import plotly.graph_objects as go
import io
import imageio.v2 as imageio

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

    def __init__(
        self,
        universe: mda.Universe,
        selection: str = "not water and not name I and not name Na+",
        spacing: float = 1.0,
        probe_radius: float = 1.4,
        margin: float | None = None,
        radii_scale: float = 1.0,
        radii_offset: float = 0.0,
        smooth_surface: bool = False,
        smooth_iterations: int = 1,
        vdw_radii_table: dict | None = None,
    ):
        """
        Parameters
        ----------
        universe : MDAnalysis.Universe
            MDAnalysis Universe with topology and trajectory loaded.
        selection : str
            Atom selection string for MDAnalysis (default: target heavy atoms).
        spacing : float
            Grid spacing in Å (smaller = more accurate, slower). Typical: 0.5–1.0.
        probe_radius : float
            Probe radius in Å. 1.4 Å ~ water; increase for ligand-sized probe.
        margin : float or None
            Extra padding added around target bounding box (Å).
            If None, it will be derived automatically from radii + spacing.
        radii_scale : float
            Global multiplicative factor applied to VDW radii.
        radii_offset : float
            Global additive padding (Å) added to radii (after scaling).
        smooth_surface : bool
            If True, apply binary closing to the occupancy grid to seal small gaps.
        smooth_iterations : int
            Number of closing iterations (1–2 usually enough).
        vdw_radii_table : dict or None
            Optional per-element VDW radii, e.g. {"C": 1.75, "N": 1.6}.
            Overrides DEFAULT_VDW_RADII for those elements.
        """
        self.universe = universe
        self.vdw_radii_table = self.DEFAULT_VDW_RADII.copy()
        # Update with custom VDW radii
        if vdw_radii_table is not None:
            self.vdw_radii_table.update({k.upper(): v for k, v in vdw_radii_table.items()})

        self.selection = "not water and not name I and not name Na+" if selection is None else selection
        self.spacing = spacing
        self.probe_radius = probe_radius
        self.margin = margin  # if None, computed per-frame
        self.radii_scale = radii_scale
        self.radii_offset = radii_offset
        self.smooth_surface = smooth_surface
        self.smooth_iterations = smooth_iterations
        
        self.ag = self.universe.select_atoms(self.selection)

        if self.ag.n_atoms == 0:
            raise ValueError(f"Selection '{self.selection}' returned no atoms.")

        # Precompute VDW radii for the selected atoms (fixed over trajectory)
        self.vdw_radii = self._get_vdw_radii(self.ag)
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
                radii.append(1.8)  # fallback default
                print(f"Warning: Unknown element for atom {atom.name} (resname {atom.resname}, resid {atom.resid}), using default radius 1.8 Å")
            else:
                radii.append(self.vdw_radii_table.get(elem, 1.8))
        return np.array(radii, dtype=float)

    # ----------- grid / occupancy helpers -----------

    @staticmethod
    def _build_grid(coords: np.ndarray, spacing: float, margin: float):
        """
        Build a 3D grid around a set of coordinates.

        Returns
        -------
        x, y, z : 1D arrays
        shape   : (nx, ny, nz)
        origin  : (xmin, ymin, zmin)
        """
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
    def _get_rdkit_mol_from_ag(self):
        """
        Try to convert self.ag to an RDKit Mol via AtomGroup.convert_to("RDKIT").

        Returns
        -------
        mol : rdkit.Chem.Mol or None
        """
        mol = None
        try:
            # MDAnalysisConverters must be installed for this to work:
            #   pip install MDAnalysisConverters
            mol = self.ag.convert_to("RDKIT")
        except Exception as e:
            print(f"Warning: failed to convert AtomGroup to RDKit Mol: {e}")
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
    
    def compute_frame(
        self,
        frame_index: int,
        return_masks: bool = False,
    ):
        """
        Compute target and cavity volume for a single frame.

        Parameters
        ----------
        frame_index : int
            Index of the frame in universe.trajectory.
        return_masks : bool
            If True, also return the inside and cavity masks for visualization.

        Returns
        -------
        If return_masks is False:
            target_volume_A3, cavity_volume_A3
        If return_masks is True:
            target_volume_A3, cavity_volume_A3, inside_mask, cavity_mask
        """
        self.universe.trajectory[frame_index]
        coords = self.ag.positions.copy()

        # Effective radii: VDW + probe
        radii_eff = self._effective_radii()

        # Margin: either user-specified or auto
        if self.margin is not None:
            margin = self.margin
        else:
            margin = np.max(radii_eff) + self.spacing

        x, y, z, shape, origin = self._build_grid(coords, self.spacing, margin)
        self._last_grid_axes = (x, y, z)
        inside = np.zeros(shape, dtype=bool)

        self._mark_occupancy(inside, x, y, z, coords, radii_eff)
        
        # Optional surface smoothing (important for rings / tight contacts)
        if self.smooth_surface and self.smooth_iterations > 0:
            struct = ndimage.generate_binary_structure(rank=3, connectivity=1)
            inside = ndimage.binary_closing(
                inside, structure=struct, iterations=self.smooth_iterations
            )
        voxel_volume = self.spacing**3

        target_volume = inside.sum() * voxel_volume

        cavities = self._detect_cavities_from_inside(inside)
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
    ):
        """
        Analyze all frames in the trajectory (with optional plotting for one frame).

        Parameters
        ----------
        stride : int
            Analyze every `stride`-th frame.
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
        results = []

        for ts in self.universe.trajectory[::stride]:
            frame_index = ts.frame
            time_ps = ts.time

            if plot_frame is not None and frame_index == plot_frame:
                target_vol, cavity_vol, inside, cavities = self.compute_frame(
                    frame_index, return_masks=True
                )

                self.plot_cavity_slice(
                    inside=inside,
                    cavities=cavities,
                    axis=plot_axis,
                    index=plot_slice_index,
                    outfile=plot_png or f"cavity_frame{frame_index}_{plot_axis}.png",
                )
            else:
                target_vol, cavity_vol = self.compute_frame(frame_index)

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
        # --- 1. Compute volumes & masks on the same grid used in compute_frame ---
        target_vol, cavity_vol, inside, cavities = self.compute_frame(
            frame_index, return_masks=True
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
        # Move trajectory to frame and get coords/elements
        self.universe.trajectory[frame_index]
        coords = self.ag.positions.copy()
        elements = [self._get_element(atom) or "C" for atom in self.ag.atoms]
        
        atom_trace, bond_trace, annotations_id, annotations_length, updatemenus = \
            make_molecule_components(coords, elements)
        
        # --- 3. Volume traces for target & cavities ---
        # target volume: semi-transparent gray shell
        vol_target = go.Volume(
            x=Xr,
            y=Yr,
            z=Zr,
            value=Vr_target,
            isomin=0.5,   # “inside” voxels are 1; surface at ~0.5
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
        atom_stride: int = 1,
        voxel_stride: int = 1,
        max_cavity_steps: int | None = None,
        elev: float = 20.0,
        azim: float = -60.0,
        fps: float = 10.0,
        show_inside_shell: bool = True,
    ):
        """
        Unified GIF for a single frame:

        Phase 1: _mark_occupancy
            - Inside grid grows as atoms are stamped (raw inside mask)
        Phase 2: smoothing + cavity detection
            - Inside mask is optionally smoothed (binary closing)
            - Outside region flood-fills empty space from boundary
            - Cavities = empty & ~outside_region

        Visualization:
            - gray shell  = thin surface of inside (optional)
            - red points  = cavity voxels (in cavity phase only)
            - colored dots= MDAnalysis atoms
            - RDKit 2D inset from self.ag.convert_to("RDKIT"), if available
        """
        if voxel_stride < 1:
            raise ValueError("voxel_stride must be >= 1")

        import io
        import imageio.v2 as imageio
        import numpy as np
        from scipy import ndimage

        # --- 1. Set frame and build grid (as in compute_frame) ---
        self.universe.trajectory[frame_index]
        coords = self.ag.positions.copy()
        radii_eff = self._effective_radii()

        if self.margin is not None:
            margin = self.margin
        else:
            margin = np.max(radii_eff) + self.spacing

        x_axis, y_axis, z_axis, shape, origin = self._build_grid(
            coords, self.spacing, margin
        )
        self._last_grid_axes = (x_axis, y_axis, z_axis)

        inside_raw = np.zeros(shape, dtype=bool)

        n_atoms = coords.shape[0]

        # --- MD structure in subsampled grid coordinates ---
        x0, y0, z0 = x_axis[0], y_axis[0], z_axis[0]
        step_size = self.spacing * voxel_stride

        gx = (coords[:, 0] - x0) / step_size
        gy = (coords[:, 1] - y0) / step_size
        gz = (coords[:, 2] - z0) / step_size

        elements = [self._get_element(atom) or "C" for atom in self.ag.atoms]
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

        nx_full, ny_full, nz_full = inside_raw.shape
        nx_sub = (nx_full - 1) // voxel_stride + 1
        ny_sub = (ny_full - 1) // voxel_stride + 1
        nz_sub = (nz_full - 1) // voxel_stride + 1

        images = []

        # Struct for shell erosion
        struct3 = ndimage.generate_binary_structure(3, 1)

        voxel_volume = self.spacing ** 3

        def render_state(
            phase: str,
            step_idx: int,
            inside_mask: np.ndarray,
            outside_region: np.ndarray | None,
            cavity_mask: np.ndarray | None,
        ):
            """
            Render a single frame.

            - phase: "occupancy" or "cavity"
            - inside_mask: 3D bool of inside voxels (raw or smoothed)
            - outside_region: 3D bool of outside flood (None in occupancy phase)
            - cavity_mask: 3D bool of cavities (None in occupancy phase)
            """
            # subsample
            vis_inside = inside_mask[::voxel_stride, ::voxel_stride, ::voxel_stride]

            if cavity_mask is not None:
                vis_cav = cavity_mask[::voxel_stride, ::voxel_stride, ::voxel_stride]
            else:
                vis_cav = np.zeros_like(vis_inside, dtype=bool)

            # shell of inside (for orientation)
            if show_inside_shell:
                inner = ndimage.binary_erosion(
                    vis_inside, structure=struct3, border_value=0
                )
                shell = vis_inside & (~inner)
            else:
                shell = np.zeros_like(vis_inside, dtype=bool)

            # cavity voxels indices (scatter)
            cav_ix, cav_iy, cav_iz = np.where(vis_cav)

            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111, projection="3d")

            # inside shell as faint voxels
            if show_inside_shell and np.any(shell):
                shell_colors = np.zeros(shell.shape + (4,), dtype=float)
                shell_colors[..., 3] = 0.0
                shell_colors[shell] = (0.8, 0.8, 0.8, 0.15)
                ax.voxels(shell, facecolors=shell_colors, edgecolor=None)

            # cavity as red scatter (cannot be occluded)
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

            inside_vol = inside_mask.sum() * voxel_volume
            if cavity_mask is not None:
                cav_vol = cavity_mask.sum() * voxel_volume
                cav_vox = cavity_mask.sum()
            else:
                cav_vol = 0.0
                cav_vox = 0

            title_phase = "Occupancy build" if phase == "occupancy" else "Cavity detection"
            ax.set_title(
                f"{title_phase} – step {step_idx}\n"
                f"V_inside ≈ {inside_vol:.0f} Å³, "
                f"cavity voxels: {cav_vox}, "
                f"Vcav ≈ {cav_vol:.0f} Å³",
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

        # initial empty grid
        render_state(
            phase="occupancy",
            step_idx=0,
            inside_mask=inside_raw,
            outside_region=None,
            cavity_mask=None,
        )

        for i in range(n_atoms):
            # stamp a single atom
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
                render_state(
                    phase="occupancy",
                    step_idx=n_done,
                    inside_mask=inside_raw,
                    outside_region=None,
                    cavity_mask=None,
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
        else:
            inside_smooth = inside_raw.copy()

        empty = ~inside_smooth

        # seeds on boundary
        nx, ny, nz = empty.shape
        outside_region = np.zeros_like(empty, dtype=bool)
        outside_region[0, :, :] |= empty[0, :, :]
        outside_region[nx - 1, :, :] |= empty[nx - 1, :, :]
        outside_region[:, 0, :] |= empty[:, 0, :]
        outside_region[:, ny - 1, :] |= empty[:, ny - 1, :]
        outside_region[:, :, 0] |= empty[:, :, 0]
        outside_region[:, :, nz - 1] |= empty[:, :, 0]

        struct_out = ndimage.generate_binary_structure(rank=3, connectivity=1)

        if max_cavity_steps is None:
            max_cavity_steps = nx + ny + nz

        # initial cavity mask
        cav_full = empty & (~outside_region)
        render_state(
            phase="cavity",
            step_idx=0,
            inside_mask=inside_smooth,
            outside_region=outside_region,
            cavity_mask=cav_full,
        )

        for k in range(1, max_cavity_steps + 1):
            dilated = ndimage.binary_dilation(outside_region, structure=struct_out)
            outside_next = (dilated & empty) | outside_region

            cav_full = empty & (~outside_next)

            render_state(
                phase="cavity",
                step_idx=k,
                inside_mask=inside_smooth,
                outside_region=outside_next,
                cavity_mask=cav_full,
            )

            if np.array_equal(outside_next, outside_region):
                break

            outside_region = outside_next

        if not images:
            raise RuntimeError("No frames collected for GIF.")

        imageio.mimsave(gif_path, images, duration=1.0 / fps)
        print(f"Saved volume pipeline GIF to: {gif_path}")