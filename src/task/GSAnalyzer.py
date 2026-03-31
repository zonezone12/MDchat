#!/usr/bin/env python3
"""
GSAnalyzer Module
-----------------

GSA nanocube specific analysis functions for computing geometric metrics.
"""

import warnings
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import pandas as pd

try:
    import MDAnalysis as mda
except ImportError as e:
    raise ImportError("MDAnalysis is required. pip install MDAnalysis") from e

from sklearn.cluster import KMeans
from src.VolumeAnalyzer import VolumeAnalyzer
from src.TrajectoryIterator import FrameObserver, TrajectoryIterator, ResultsGroup

# Import Client for type hints (optional dependency)
try:
    from dask.distributed import Client
except ImportError:
    Client = None  # type: ignore




class GSAnalyzer:
    """Analyze GSA nanocube structures and compute geometric metrics."""
    
    def __init__(self):
        """Initialize GSAnalyzer with storage for computed results."""
        self.nanocube_metrics_df: Optional[pd.DataFrame] = None
        self.face_selections: Optional[List[str]] = None
        self.planar_rms_cache: dict = {}
        self.guest_residence_stats: Optional[Dict] = None
    
    def residue_planar_rms(self, atoms):
        """Compute planar RMS for a residue/atom group.
        
        Returns:
            tuple: (planar_rms, face_normal_vector) where:
                - planar_rms: Root mean square deviation from best-fit plane
                - face_normal_vector: Normal vector of the best-fit plane
        """
        P = atoms.positions
        if len(P) < 3:
            return np.nan, np.array([np.nan, np.nan, np.nan])
        P0 = P - P.mean(axis=0)
        U, S, Vt = np.linalg.svd(P0, full_matrices=False)
        face_normal_vector = Vt[-1]
        dist = np.abs(P0 @ face_normal_vector)
        planar_rms = float(np.sqrt((dist**2).mean()))
        return planar_rms, face_normal_vector
    
    def gsa_nanocube_metrics(self, u: mda.Universe,
                             face_sel_list: List[str],
                             corner_sel_list: Optional[List[str]] = None,
                             guest_sel: Optional[str] = None,
                             out_prefix: Optional[str] = None,
                             gatherer: Optional[object] = None,
                             guest_tracking_method: str = "distance",
                             guest_distance_threshold: Optional[float] = None,
                             use_dask: bool = False,
                             n_jobs: Optional[int] = None,
                             dask_client: Optional[Client] = None) -> pd.DataFrame:
        """
        Compute geometric observables for GSA nanocube behavior.
        
        This method now uses TrajectoryIterator with GSAnalyzerObserver for efficient
        single-pass trajectory iteration. Guest tracking functionality from guest_in.py
        is integrated to provide detailed guest residence statistics.
        
        Args:
            u: MDAnalysis Universe containing the trajectory
            face_sel_list: List of face selection strings
            corner_sel_list: Optional list of corner selection strings (currently unused)
            guest_sel: Optional guest selection string
            out_prefix: Optional output prefix for saving results
            gatherer: DEPRECATED - This parameter is ignored. Use TrajectoryIterator instead.
            guest_tracking_method: Method to determine if guest is inside host
                                 - "distance": Use distance from guest to host center (faster)
                                 - "volume": Use VolumeAnalyzer to check if guest is inside host volume (more accurate)
            guest_distance_threshold: Distance threshold in Angstrom for distance method.
                                    If None, automatically calculated from host size.
            use_dask: If True, prefer Dask Distributed for parallelization
            n_jobs: Number of parallel jobs. If None, uses all available CPU cores.
                   If 1, runs sequentially. If -1, uses all available CPU cores.
            dask_client: Optional Dask Distributed Client for cluster computing.
        
        Returns:
            DataFrame with geometric metrics and guest tracking information
        """
        if gatherer is not None:
            warnings.warn(
                "The 'gatherer' parameter is deprecated. "
                "gsa_nanocube_metrics now uses TrajectoryIterator internally. "
                "The gatherer parameter will be ignored.",
                DeprecationWarning
            )
        
        # Create observer with guest tracking capabilities
        observer = GSAnalyzerObserver(
            face_sel_list=face_sel_list,
            corner_sel_list=corner_sel_list,
            guest_sel=guest_sel,
            out_prefix=out_prefix,
            guest_tracking_method=guest_tracking_method,
            guest_distance_threshold=guest_distance_threshold,
        )
        
        # Create iterator and subscribe observer
        iterator = TrajectoryIterator(u, use_dask=use_dask)
        iterator.subscribe(observer)
        
        # Determine n_jobs: if None, use -1 to use all cores; otherwise pass as-is
        if n_jobs is None:
            n_jobs = -1
        
        # Iterate through trajectory
        iterator.iterate(n_jobs=n_jobs, dask_client=dask_client)
        
        # Get results
        df = observer.get_metrics_df()
        self.nanocube_metrics_df = df
        self.face_selections = face_sel_list
        
        return df
    
    def amber_preset_selections(self, u: mda.Universe,
                                gsa_resnames: List[str] = ["GSA"],
                                water_resnames: List[str] = ["WAT", "HOH", "TIP3", "TIP3P"],
                                na_resnames: List[str] = ["Na+", "SOD", "NA"],
                                i_resnames: List[str] = ["I-", "IOD", "IB"]) -> dict:
        """Return a dict of robust MDAnalysis selection strings for Amber systems."""
        gsa_or = " or ".join([f"resname {r}" for r in gsa_resnames]) if gsa_resnames else "resname GSA"
        wat_or = " or ".join([f"resname {r}" for r in water_resnames])
        na_or = " or ".join([f"resname {r}" for r in na_resnames])
        i_or = " or ".join([f"resname {r}" for r in i_resnames])
        
        sels = {
            'water': f"({wat_or})",
            'sodium': f"(({na_or}) and (name Na* or type Na))",
            'iodide': f"(({i_or}) and (name I* or type I))",
            'gsa_all': f"({gsa_or})",
            'non_solvent_non_ion': f"not ({wat_or}) and not ({na_or}) and not ({i_or})",
        }
        return sels
    
    def gsa_auto_faces_by_kmeans(self, u: mda.Universe,
                                 gsa_sel: str = "resname GSA",
                                 n_faces: int = 6,
                                 group_by: str = 'residue') -> List[str]:
        """Cluster GSA building blocks into ~6 faces by KMeans."""
        ag = u.select_atoms(gsa_sel)
        if group_by == 'residue':
            groups = list({a.residue for a in ag.atoms})
            coms = np.array([res.atoms.center_of_mass() for res in groups])
            labels = KMeans(n_clusters=n_faces, n_init=20, random_state=0).fit_predict(coms)
            faces = []
            for k in range(n_faces):
                resid_list = [str(res.resid) for i, res in enumerate(groups) if labels[i] == k]
                if len(resid_list) == 0:
                    faces.append("resid -1")
                else:
                    faces.append("resid " + " ".join(resid_list))
            result_faces = faces
        else:
            coms = ag.positions
            labels = KMeans(n_clusters=n_faces, n_init=20, random_state=0).fit_predict(coms)
            faces = []
            for k in range(n_faces):
                idx = np.where(labels == k)[0]
                if len(idx) == 0:
                    faces.append("index -1")
                else:
                    faces.append("index " + " ".join(map(str, idx.tolist())))
            result_faces = faces
        
        self.face_selections = result_faces
        return result_faces
    
    def faces_from_atomname_blocks(self, u: 'mda.Universe',
                                   gsa_reslabel: str = 'MOL',
                                   n_faces: int = 6,
                                   residues_per_face: int | None = None,
                                   try_detect_period: bool = True) -> list[str]:
        """Construct ~6 nanocube face selections from GSA building blocks."""
        gsa_residues = [res for res in u.residues if res.resname.upper().startswith(gsa_reslabel.upper())]
        if not gsa_residues:
            raise ValueError(f"No residues with resname '{gsa_reslabel}' found in universe.")
        
        gsa_resids = [res.resid for res in gsa_residues]
        
        if residues_per_face is None and try_detect_period and len(gsa_residues) >= n_faces:
            flat_names = [atom.name for res in gsa_residues for atom in res.atoms]
            
            def smallest_period(seq: list[str], max_p: int = 512) -> int | None:
                for p in range(1, min(max_p, len(seq)//2) + 1):
                    ok = True
                    for i in range(len(seq) - p):
                        if seq[i] != seq[i + p]:
                            ok = False
                            break
                    if ok:
                        return p
                return None
            
            avg_atoms_per_res = int(round(np.mean([len(res.atoms) for res in gsa_residues])))
            p = smallest_period(flat_names)
            if p is not None and avg_atoms_per_res > 0:
                rp = max(1, int(round(p / avg_atoms_per_res)))
                residues_per_face = rp
        
        if residues_per_face is not None and len(gsa_resids) >= residues_per_face * n_faces:
            faces = []
            for k in range(n_faces):
                chunk = gsa_resids[k*residues_per_face : (k+1)*residues_per_face]
                if not chunk:
                    faces.append('resid -1')
                else:
                    faces.append('resid ' + ' '.join(map(str, chunk)))
            return faces
        
        coms = np.array([res.atoms.center_of_mass() for res in gsa_residues])
        if len(coms) < n_faces:
            raise ValueError(f"Not enough GSA residues ({len(coms)}) to form {n_faces} faces.")
        labels = KMeans(n_clusters=n_faces, n_init=20, random_state=0).fit_predict(coms)
        faces = []
        for k in range(n_faces):
            group_resids = [str(gsa_residues[i].resid) for i in range(len(gsa_residues)) if labels[i] == k]
            if not group_resids:
                faces.append('resid -1')
            else:
                faces.append('resid ' + " ".join(group_resids))
        self.face_selections = faces
        return faces
    
    def get_longest_duration_guest_indices(self) -> Optional[Dict]:
        """
        Get the guest indices for the longest duration_inside stay.
        
        Returns:
            Dictionary with:
            - duration: The longest duration (ps)
            - guest_indices: List of guest atom indices for this stay
            - entry_frame: Frame when guest entered
            - entry_time: Time (ps) when guest entered
            - exit_frame: Frame when guest exited
            - exit_time: Time (ps) when guest exited
            - duration_index: Index in durations_inside list
            None if no guest residence stats available or no entries
        """
        if self.guest_residence_stats is None:
            return None
        
        durations_inside = self.guest_residence_stats.get('durations_inside', [])
        if not durations_inside:
            return None
        
        # Find index of longest duration
        max_duration_idx = np.argmax(durations_inside)
        max_duration = durations_inside[max_duration_idx]
        
        # Get corresponding guest indices and entry/exit info
        entry_guest_indices = self.guest_residence_stats.get('entry_guest_indices', [])
        entry_frames = self.guest_residence_stats.get('entry_frames', [])
        entry_times = self.guest_residence_stats.get('entry_times', [])
        exit_frames = self.guest_residence_stats.get('exit_frames', [])
        exit_times = self.guest_residence_stats.get('exit_times', [])
        
        return {
            'duration': max_duration,
            'guest_indices': entry_guest_indices[max_duration_idx] if max_duration_idx < len(entry_guest_indices) else [],
            'entry_frame': entry_frames[max_duration_idx] if max_duration_idx < len(entry_frames) else None,
            'entry_time': entry_times[max_duration_idx] if max_duration_idx < len(entry_times) else None,
            'exit_frame': exit_frames[max_duration_idx] if max_duration_idx < len(exit_frames) else None,
            'exit_time': exit_times[max_duration_idx] if max_duration_idx < len(exit_times) else None,
            'duration_index': max_duration_idx,
        }
    
    def print_longest_duration_guest_indices(self) -> None:
        """
        Print information about the guest indices for the longest duration_inside stay.
        """
        result = self.get_longest_duration_guest_indices()
        if result is None:
            print("No guest residence statistics available or no entries found.")
            return
        
        print(f"\n{'='*60}")
        print("Longest Duration Inside - Guest Indices")
        print(f"{'='*60}")
        print(f"Duration: {result['duration']:.2f} ps")
        print(f"Guest indices: {result['guest_indices']}")
        print(f"Entry frame: {result['entry_frame']}")
        if result['entry_time'] is not None:
            print(f"Entry time: {result['entry_time']:.2f} ps")
        print(f"Exit frame: {result['exit_frame']}")
        if result['exit_time'] is not None:
            print(f"Exit time: {result['exit_time']:.2f} ps")
        print(f"Duration index: {result['duration_index']}")
        print(f"{'='*60}\n")


class GSAnalyzerObserver(FrameObserver):
    """
    Observer implementation of GSAnalyzer for single-pass trajectory iteration.
    
    Processes nanocube metrics during trajectory iteration without storing
    coordinates in memory. Includes integrated guest tracking functionality
    from guest_in.py for monitoring guest entry/exit events and residence times.
    
    Uses ResultsGroup for declarative result aggregation in parallel processing.
    Results are stored in self.results dict with keys: 'rows', 'entry_events',
    'exit_events', 'first_frame', 'last_frame', 'first_time', 'last_time',
    'frame_call_count', 'frame_exception_count'.
    """
    
    def __init__(
        self,
        face_sel_list: List[str],
        corner_sel_list: Optional[List[str]] = None,
        guest_sel: Optional[str] = None,
        out_prefix: Optional[str] = None,
        guest_tracking_method: str = "distance",
        guest_distance_threshold: Optional[float] = None,
    ):
        """
        Initialize GSAnalyzerObserver.
        
        Args:
            face_sel_list: List of face selection strings
            corner_sel_list: Optional list of corner selection strings
            guest_sel: Optional guest selection string
            out_prefix: Optional output prefix for saving results
            guest_tracking_method: Method to determine if guest is inside host
                                 - "distance": Use distance from guest to host center (faster)
                                 - "volume": Use VolumeAnalyzer to check if guest is inside host volume (more accurate)
            guest_distance_threshold: Distance threshold in Angstrom for distance method.
                                    If None, automatically calculated from host size.
        """
        super().__init__()  # Initialize results dict from FrameObserver
        
        self.face_sel_list = face_sel_list
        self.corner_sel_list = corner_sel_list or []
        self.guest_sel = guest_sel
        self.out_prefix = out_prefix
        
        # Guest tracking parameters
        self.guest_tracking_method = guest_tracking_method
        self.guest_distance_threshold = guest_distance_threshold
        
        # Storage for results (using self.results for ResultsGroup pattern)
        self.results['rows'] = []
        self.results['entry_events'] = []
        self.results['exit_events'] = []
        self.results['first_frame'] = None
        self.results['first_time'] = None
        self.results['last_frame'] = None
        self.results['last_time'] = None
        self.results['frame_call_count'] = 0
        self.results['frame_exception_count'] = 0
        
        self.volume_analyzer: Optional[VolumeAnalyzer] = None
        self._initialized = False
        
        # Reference to analyzer for utility methods
        self._analyzer = GSAnalyzer()
        
        # Guest tracking state (integrated from GuestEnteringObserver)
        self._guest_volume_analyzer: Optional[VolumeAnalyzer] = None
        self._inside_guest_indices: set = set()
    
    # ===== Properties for backward compatibility =====
    
    @property
    def rows(self) -> List[dict]:
        """Property to access rows from results dict for backward compatibility."""
        return self.results.get('rows', [])
    
    @rows.setter
    def rows(self, value: List[dict]) -> None:
        """Setter for rows to store in results dict."""
        self.results['rows'] = value
    
    @property
    def entry_events(self) -> List[Dict]:
        """Property to access entry_events from results dict for backward compatibility."""
        return self.results.get('entry_events', [])
    
    @entry_events.setter
    def entry_events(self, value: List[Dict]) -> None:
        """Setter for entry_events to store in results dict."""
        self.results['entry_events'] = value
    
    @property
    def exit_events(self) -> List[Dict]:
        """Property to access exit_events from results dict for backward compatibility."""
        return self.results.get('exit_events', [])
    
    @exit_events.setter
    def exit_events(self, value: List[Dict]) -> None:
        """Setter for exit_events to store in results dict."""
        self.results['exit_events'] = value
    
    @property
    def _first_frame(self) -> Optional[int]:
        """Property to access _first_frame from results dict for backward compatibility."""
        return self.results.get('first_frame')
    
    @_first_frame.setter
    def _first_frame(self, value: Optional[int]) -> None:
        """Setter for _first_frame to store in results dict."""
        self.results['first_frame'] = value
    
    @property
    def _first_time(self) -> Optional[float]:
        """Property to access _first_time from results dict for backward compatibility."""
        return self.results.get('first_time')
    
    @_first_time.setter
    def _first_time(self, value: Optional[float]) -> None:
        """Setter for _first_time to store in results dict."""
        self.results['first_time'] = value
    
    @property
    def _last_frame(self) -> Optional[int]:
        """Property to access _last_frame from results dict for backward compatibility."""
        return self.results.get('last_frame')
    
    @_last_frame.setter
    def _last_frame(self, value: Optional[int]) -> None:
        """Setter for _last_frame to store in results dict."""
        self.results['last_frame'] = value
    
    @property
    def _last_time(self) -> Optional[float]:
        """Property to access _last_time from results dict for backward compatibility."""
        return self.results.get('last_time')
    
    @_last_time.setter
    def _last_time(self, value: Optional[float]) -> None:
        """Setter for _last_time to store in results dict."""
        self.results['last_time'] = value
    
    @property
    def _frame_call_count(self) -> int:
        """Property to access _frame_call_count from results dict for backward compatibility."""
        return self.results.get('frame_call_count', 0)
    
    @_frame_call_count.setter
    def _frame_call_count(self, value: int) -> None:
        """Setter for _frame_call_count to store in results dict."""
        self.results['frame_call_count'] = value
    
    @property
    def _frame_exception_count(self) -> int:
        """Property to access _frame_exception_count from results dict for backward compatibility."""
        return self.results.get('frame_exception_count', 0)
    
    @_frame_exception_count.setter
    def _frame_exception_count(self, value: int) -> None:
        """Setter for _frame_exception_count to store in results dict."""
        self.results['frame_exception_count'] = value
    
    def __getstate__(self):
        """Custom pickling: exclude VolumeAnalyzer instances (they contain Universe references)."""
        state = self.__dict__.copy()
        # Remove VolumeAnalyzer instances - they'll be reinitialized in on_frame_start
        state['volume_analyzer'] = None
        state['_guest_volume_analyzer'] = None
        # Reset initialized flag since VolumeAnalyzers will be recreated
        state['_initialized'] = False
        return state
    
    def __setstate__(self, state):
        """Custom unpickling: restore state (VolumeAnalyzers will be reinitialized in on_frame_start)."""
        self.__dict__.update(state)
    
    def _get_aggregator(self) -> ResultsGroup:
        """
        Return ResultsGroup for declarative result aggregation.
        
        Defines how results from parallel workers should be merged:
        - rows: Extend and sort by frame
        - entry_events/exit_events: Extend and sort by frame
        - first_frame/first_time: Take minimum
        - last_frame/last_time: Take maximum
        - frame_call_count/frame_exception_count: Sum
        """
        return ResultsGroup(lookup={
            'rows': ResultsGroup.list_extend_sorted('frame'),
            'entry_events': ResultsGroup.list_extend_sorted('frame'),
            'exit_events': ResultsGroup.list_extend_sorted('frame'),
            'first_frame': ResultsGroup.min_value,
            'first_time': ResultsGroup.min_value,
            'last_frame': ResultsGroup.max_value,
            'last_time': ResultsGroup.max_value,
            'frame_call_count': ResultsGroup.sum_values,
            'frame_exception_count': ResultsGroup.sum_values,
        })
    
    def get_selections_needed(self) -> List[str]:
        """Return list of selection strings needed by this observer."""
        selections = list(self.face_sel_list)
        if self.guest_sel:
            selections.append(self.guest_sel)
        return selections
    
    def _is_guest_inside_distance(self, universe: mda.Universe, cube_center: np.ndarray) -> Tuple[bool, List[int]]:
        """Check if guest is inside host using distance method.
        
        Returns:
            tuple: (is_inside, guest_indices) where:
                - is_inside: True if any guest atom is within threshold
                - guest_indices: List of guest atom indices that are inside
        """
        if self.guest_sel is None:
            return False, []
        
        try:
            guest = universe.select_atoms(self.guest_sel)
            if len(guest) == 0:
                return False, []
            
            guest_positions = guest.positions
            inside_indices = []
            
            # Calculate distance from each guest atom to cube center
            for idx, guest_pos in enumerate(guest_positions):
                distance = np.linalg.norm(guest_pos - cube_center)
                if distance <= self.guest_distance_threshold:
                    inside_indices.append(guest[idx].id)
            
            return len(inside_indices) > 0, inside_indices
        except Exception:
            return False, []
    
    def _is_guest_inside_volume(self, universe: mda.Universe, frame_idx: int) -> Tuple[bool, List[int], Optional[float]]:
        """Check if guest is inside host using VolumeAnalyzer.
        
        Args:
            universe: MDAnalysis Universe (already positioned at current frame)
            frame_idx: Zero-based frame index (not frame number)
        
        Returns:
            tuple: (is_inside, guest_indices, volume) where:
                - is_inside: True if any guest atom is inside host volume
                - guest_indices: List of guest atom indices that are inside
                - volume: Total volume (target + cavity) or None if computation failed
        """
        if self._guest_volume_analyzer is None or self.guest_sel is None:
            return False, [], None
        
        try:
            # Get host volume mask (using combined face selection)
            # Pass universe to avoid creating new Universe in compute_frame
            # Use frame_idx (0-based index) for consistency with in_memory trajectories
            target_vol, cavity_vol, inside_mask, cavities = self._guest_volume_analyzer.compute_frame(
                frame_idx, return_masks=True, universe=universe
            )
            volume = target_vol + cavity_vol
            
            # Get guest atom positions
            guest = universe.select_atoms(self.guest_sel)
            if len(guest) == 0:
                return False, [], volume
            
            guest_positions = guest.positions
            inside_indices = []
            
            # Get grid information
            if self._guest_volume_analyzer._last_grid_axes is None:
                return False, [], volume
            
            x, y, z = self._guest_volume_analyzer._last_grid_axes
            # Grid origin is at the first grid point (x[0], y[0], z[0])
            origin = np.array([x[0], y[0], z[0]])
            spacing = self._guest_volume_analyzer.spacing
            
            # Check if guest atoms are inside the host volume
            for atom_idx, pos in enumerate(guest_positions):
                # Convert position to grid indices
                grid_idx = ((pos - origin) / spacing).astype(int)
                
                # Check bounds
                if (0 <= grid_idx[0] < inside_mask.shape[0] and
                    0 <= grid_idx[1] < inside_mask.shape[1] and
                    0 <= grid_idx[2] < inside_mask.shape[2]):
                    if inside_mask[grid_idx[0], grid_idx[1], grid_idx[2]]:
                        inside_indices.append(guest[atom_idx].id)
            
            return len(inside_indices) > 0, inside_indices, volume
        except Exception as e:
            warnings.warn(f"Volume-based guest check failed at frame {frame_idx}: {e}")
            return False, [], None
    
    def on_frame_start(self, iterator: TrajectoryIterator) -> None:
        """Initialize data structures before iteration."""
        u = iterator.universe
        
        # Validate face selections before proceeding
        u.trajectory[0]  # Go to first frame for validation
        invalid_faces = []
        for i, face_sel in enumerate(self.face_sel_list):
            try:
                ag = u.select_atoms(face_sel)
                if len(ag) == 0:
                    invalid_faces.append(f"Face {i+1} ({face_sel}): empty selection")
            except Exception as e:
                invalid_faces.append(f"Face {i+1} ({face_sel}): {e}")
        
        if invalid_faces:
            warnings.warn(
                f"Some face selections are invalid:\n  " + "\n  ".join(invalid_faces) +
                "\nThis may cause volume computation to fail."
            )
        
        # Initialize VolumeAnalyzer for volume computation if available
        combined_sel = " or ".join([f"({s})" for s in self.face_sel_list])
        try:
            self.volume_analyzer = VolumeAnalyzer(
                universe=u,
                selection=combined_sel,
                spacing=1.0,
                probe_radius=1.4
            )
        except Exception as e:
            warnings.warn(
                f"Failed to initialize VolumeAnalyzer: {e}. "
                "Falling back to edge-based volume computation."
            )
            self.volume_analyzer = None
        
        # Initialize guest tracking
        if self.guest_sel:
            # Initialize guest volume analyzer if using volume method
            if self.guest_tracking_method == "volume":
                combined_sel = " or ".join([f"({s})" for s in self.face_sel_list])
                try:
                    self._guest_volume_analyzer = VolumeAnalyzer(
                        universe=u,
                        selection=combined_sel,
                        spacing=0.5,  # Fine grid for accurate detection
                        probe_radius=1.4
                    )
                except Exception as e:
                    warnings.warn(
                        f"Failed to initialize guest VolumeAnalyzer: {e}. "
                        "Falling back to distance-based method."
                    )
                    self.guest_tracking_method = "distance"
                    self._guest_volume_analyzer = None
            
            # Calculate distance threshold if not provided and using distance method
            if self.guest_distance_threshold is None and self.guest_tracking_method == "distance":
                # Use first frame to estimate host size
                u.trajectory[0]
                try:
                    # Combine all face selections to get host
                    combined_sel = " or ".join([f"({s})" for s in self.face_sel_list])
                    host = u.select_atoms(combined_sel)
                    if len(host) > 0:
                        host_coords = host.positions
                        host_com = host.center_of_geometry()
                        # Use maximum distance from COM to any host atom as threshold
                        max_dist = np.max(np.linalg.norm(host_coords - host_com, axis=1))
                        # Add some margin (20% of max distance)
                        self.guest_distance_threshold = max_dist * 0.8
                    else:
                        warnings.warn("Host selection is empty. Using default threshold of 10.0 Å.")
                        self.guest_distance_threshold = 10.0
                except Exception as e:
                    warnings.warn(f"Failed to calculate distance threshold: {e}. Using default 10.0 Å.")
                    self.guest_distance_threshold = 10.0
        
        self.rows = []
        self._initialized = True
        self._frame_call_count = 0  # Reset debug counters
        self._frame_exception_count = 0
        
        # Reset guest tracking state
        self._inside_guest_indices = set()
        self.entry_events = []
        self.exit_events = []
        self._first_frame = None
        self._first_time = None
        self._last_frame = None
        self._last_time = None
    
    def on_frame(self, ts: mda.coordinates.base.Timestep, frame_idx: int,
                 universe: mda.Universe) -> None:
        """
        Process a single frame during iteration.
        
        Note: Use frame_idx (0-based index) for trajectory access, not ts.frame (frame number).
        This is especially important when using in_memory=True, where frame numbers might
        not match indices correctly.
        """
        self._frame_call_count += 1  # Debug: track calls
        
        # Lazy initialization for worker processes (VolumeAnalyzer was excluded from pickling)
        # This happens when the observer is unpickled in a worker process
        if not self._initialized:
            try:
                # Initialize VolumeAnalyzer if needed (for worker processes after unpickling)
                if self.volume_analyzer is None:
                    combined_sel = " or ".join([f"({s})" for s in self.face_sel_list])
                    try:
                        # Ensure universe is at a valid frame before initializing
                        if len(universe.trajectory) > 0:
                            universe.trajectory[0]
                        # Validate selection before creating VolumeAnalyzer
                        test_ag = universe.select_atoms(combined_sel)
                        if len(test_ag) == 0:
                            warnings.warn(f"Selection '{combined_sel}' returned no atoms. VolumeAnalyzer will use edge-based volume.")
                            self.volume_analyzer = None
                        else:
                            # Ensure coordinates are properly formatted
                            coords = test_ag.positions
                            if not isinstance(coords, np.ndarray):
                                coords = np.asarray(coords, dtype=np.float64, order='C')
                            elif not coords.flags['C_CONTIGUOUS']:
                                coords = np.ascontiguousarray(coords, dtype=np.float64)
                            
                            self.volume_analyzer = VolumeAnalyzer(
                                universe=universe,
                                selection=combined_sel,
                                spacing=1.0,
                                probe_radius=1.4
                            )
                    except Exception as e:
                        warnings.warn(f"Failed to initialize VolumeAnalyzer in worker: {e}")
                        import traceback
                        traceback.print_exc()
                        self.volume_analyzer = None
                
                # Initialize guest volume analyzer if needed
                if self.guest_sel and self.guest_tracking_method == "volume" and self._guest_volume_analyzer is None:
                    combined_sel = " or ".join([f"({s})" for s in self.face_sel_list])
                    try:
                        self._guest_volume_analyzer = VolumeAnalyzer(
                            universe=universe,
                            selection=combined_sel,
                            spacing=0.5,
                            probe_radius=1.4
                        )
                    except Exception as e:
                        warnings.warn(f"Failed to initialize guest VolumeAnalyzer in worker: {e}")
                        self.guest_tracking_method = "distance"
                        self._guest_volume_analyzer = None
                
                # Calculate distance threshold if needed
                if self.guest_sel and self.guest_distance_threshold is None and self.guest_tracking_method == "distance":
                    try:
                        combined_sel = " or ".join([f"({s})" for s in self.face_sel_list])
                        host = universe.select_atoms(combined_sel)
                        if len(host) > 0:
                            host_coords = host.positions
                            host_com = host.center_of_geometry()
                            max_dist = np.max(np.linalg.norm(host_coords - host_com, axis=1))
                            self.guest_distance_threshold = max_dist * 0.8
                        else:
                            self.guest_distance_threshold = 10.0
                    except Exception as e:
                        warnings.warn(f"Failed to calculate distance threshold in worker: {e}")
                        self.guest_distance_threshold = 10.0
                
                # Initialize rows if not already done
                if not hasattr(self, 'rows') or self.rows is None:
                    self.rows = []
                
                self._initialized = True
            except Exception as e:
                # Even if initialization partially fails, mark as initialized to avoid infinite retries
                warnings.warn(f"Error during lazy initialization in worker: {e}. Continuing with partial initialization.")
                if not hasattr(self, 'rows') or self.rows is None:
                    self.rows = []
                self._initialized = True
        
        # Get face centers (universe is already at current frame)
        fcent = []
        for face_sel in self.face_sel_list:
            try:
                ag = universe.select_atoms(face_sel)
                fcent.append(ag.center_of_geometry())
            except Exception as e:
                warnings.warn(f"Failed to get face center for {face_sel} at frame {ts.frame}: {e}")
                fcent.append(np.array([np.nan, np.nan, np.nan]))
        
        fcent = np.array(fcent)
        
        # Compute edges
        edges = []
        if len(fcent) >= 4:
            for i in range(len(fcent)):
                d = np.linalg.norm(fcent - fcent[i], axis=1)
                nn = np.argsort(d)[1:5]
                for j in nn:
                    if i < j:
                        edges.append((i, j))
        edges = list(set(edges))
        
        edge_len = [np.linalg.norm(fcent[i] - fcent[j]) for (i, j) in edges] if edges else [np.nan]
        edge_mean = np.nanmean(edge_len)
        
        # Planarity per face
        planar_rms_list = []
        face_norms_list = []
        for face_sel in self.face_sel_list:
            try:
                ag = universe.select_atoms(face_sel)
                planar_rms, face_normal_vector = self._analyzer.residue_planar_rms(ag)
                planar_rms_list.append(planar_rms)
                face_norms_list.append(face_normal_vector)
            except Exception as e:
                warnings.warn(f"Failed to compute planarity for {face_sel} at frame {ts.frame}: {e}")
                planar_rms_list.append(np.nan)
                face_norms_list.append(np.array([np.nan, np.nan, np.nan]))
        
        face_norms_array = np.array(face_norms_list)
        face_norms_angle = np.degrees(np.arccos(np.clip(np.dot(face_norms_array, face_norms_array.T), -1.0, 1.0)))
        face_angle_dict = {}
        for i, face_i in enumerate(self.face_sel_list):
            for j, face_j in enumerate(self.face_sel_list):
                if i != j:
                    face_angle_dict[f'{face_i}{face_j}_angle'] = face_norms_angle[i, j]
        
        cube_center = fcent.mean(axis=0) if len(fcent) else np.array([np.nan, np.nan, np.nan])
        
        # Guest tracking and metrics
        guest_min = np.nan
        guest_is_inside = False
        guest_inside_count = 0
        volume_from_guest_tracking = None  # Cache volume if computed during guest tracking
        
        if self.guest_sel:
            try:
                guest = universe.select_atoms(self.guest_sel)
                if len(guest):
                    # Calculate minimum distance to center
                    guest_min = float(np.linalg.norm(guest.positions - cube_center, axis=1).min())
                    
                    # Track guest entry/exit events
                    if self.guest_tracking_method == "volume" and self._guest_volume_analyzer is not None:
                        # Use frame_idx (0-based index) instead of ts.frame for consistency with in_memory trajectories
                        guest_is_inside, current_guest_indices, volume_from_guest_tracking = self._is_guest_inside_volume(universe, frame_idx)
                    else:
                        guest_is_inside, current_guest_indices = self._is_guest_inside_distance(universe, cube_center)
                    
                    guest_inside_count = len(current_guest_indices)
                    
                    # Track first and last frame/time
                    if self._first_frame is None:
                        self._first_frame = ts.frame
                        self._first_time = ts.time
                    self._last_frame = ts.frame
                    self._last_time = ts.time
                    
                    # Detect entry/exit events
                    current_guest_indices_set = set(current_guest_indices)
                    newly_entered = current_guest_indices_set - self._inside_guest_indices
                    newly_exited = self._inside_guest_indices - current_guest_indices_set
                    
                    if newly_entered:
                        self.entry_events.append({
                            'frame': ts.frame,
                            'time': ts.time,
                            'guest_indices': sorted(list(newly_entered))
                        })
                    
                    if newly_exited:
                        self.exit_events.append({
                            'frame': ts.frame,
                            'time': ts.time,
                            'guest_indices': sorted(list(newly_exited))
                        })
                    
                    self._inside_guest_indices = current_guest_indices_set
            except Exception as e:
                warnings.warn(f"Guest tracking failed at frame {ts.frame}: {e}")
        
        # Compute volume - reuse if already computed during guest tracking
        try:
            if volume_from_guest_tracking is not None:
                volume = volume_from_guest_tracking
            elif self.volume_analyzer is not None:
                try:
                    # Use frame_idx (0-based index) instead of ts.frame (frame number)
                    # This is important for in_memory trajectories where frame numbers
                    # might not match indices correctly
                    target_volume, cavity_volume = self.volume_analyzer.compute_frame(
                        frame_idx, return_masks=False, universe=universe
                    )
                    volume = target_volume + cavity_volume
                except Exception as e:
                    warnings.warn(f"VolumeAnalyzer failed for frame {frame_idx} (frame number {ts.frame}): {e}. Using edge-based volume.")
                    volume = edge_mean ** 3 if not np.isnan(edge_mean) else np.nan
            else:
                volume = edge_mean ** 3 if not np.isnan(edge_mean) else np.nan
            
            # Always add a row, even if some values are NaN
            self.rows.append({
                'frame': ts.frame,
                'edge_mean': edge_mean,
                'edge_min': float(np.nanmin(edge_len)) if len(edge_len) else np.nan,
                'edge_max': float(np.nanmax(edge_len)) if len(edge_len) else np.nan,
                'volume': volume,
                'planarity_mean': float(np.nanmean(planar_rms_list)),
                'planarity_max': float(np.nanmax(planar_rms_list)),
                'guest_min_center_dist': guest_min,
                'guest_is_inside': guest_is_inside,
                'guest_inside_count': guest_inside_count,
                **face_angle_dict,
            })
        except Exception as e:
            # If there's an exception, still try to add a minimal row with frame info
            self._frame_exception_count += 1  # Debug: track exceptions
            warnings.warn(f"Failed to process frame {ts.frame} in GSAnalyzerObserver: {e}")
            import traceback
            traceback.print_exc()
            # Add a minimal row with frame number and NaN values
            self.rows.append({
                'frame': ts.frame,
                'edge_mean': np.nan,
                'edge_min': np.nan,
                'edge_max': np.nan,
                'volume': np.nan,
                'planarity_mean': np.nan,
                'planarity_max': np.nan,
                'guest_min_center_dist': np.nan,
                'guest_is_inside': False,
                'guest_inside_count': 0,
            })
    
    def on_frame_end(self, iterator: TrajectoryIterator) -> None:
        """Finalize results after iteration."""
        # If guest is still inside at the end, record the final frame as exit
        if self.guest_sel and len(self._inside_guest_indices) > 0:
            if self._last_frame is not None and self._last_time is not None:
                self.exit_events.append({
                    'frame': self._last_frame,
                    'time': self._last_time,
                    'guest_indices': sorted(list(self._inside_guest_indices))
                })
        
        # Convert rows to DataFrame
        df = pd.DataFrame(self.rows)
        self._analyzer.nanocube_metrics_df = df
        self._analyzer.face_selections = self.face_sel_list
        
        # Compute and store guest residence statistics
        if self.guest_sel:
            self._analyzer.guest_residence_stats = self.get_guest_residence_stats()
        
        # Save if out_prefix is provided
        if self.out_prefix:
            df.to_csv(f"{self.out_prefix}_gsa_nanocube.csv", index=False)
 
    def get_metrics_df(self) -> pd.DataFrame:
        """Get computed metrics DataFrame."""
        return pd.DataFrame(self.rows) if self.rows else pd.DataFrame()
    
    def get_volume(self) -> np.ndarray:
        """Get volume array from computed metrics."""
        df = self.get_metrics_df()
        if df is None or len(df) == 0:
            return np.array([])
        if 'volume' in df.columns:
            return df['volume'].values
        elif 'edge_mean' in df.columns:
            return df['edge_mean'].values ** 3
        else:
            return np.array([])
    
    def get_diagnostic_info(self) -> Dict:
        """Get diagnostic information about the observer state."""
        return {
            'initialized': getattr(self, '_initialized', False),
            'n_rows': len(self.rows) if hasattr(self, 'rows') else 0,
            'has_volume_analyzer': self.volume_analyzer is not None,
            'has_guest_volume_analyzer': self._guest_volume_analyzer is not None,
            'metrics_df_shape': self.get_metrics_df().shape if hasattr(self, 'rows') and len(self.rows) > 0 else (0, 0),
            'frame_call_count': getattr(self, '_frame_call_count', 0),
            'frame_exception_count': getattr(self, '_frame_exception_count', 0),
        }
    
    def get_guest_residence_stats(self) -> Dict:
        """
        Get detailed statistics about guest residence inside/outside host.
        
        Returns:
            Dictionary with:
            - entry_frames: List of frame indices when guest entered
            - entry_times: List of times (ps) when guest entered
            - entry_guest_indices: List of guest atom indices that entered at each event
            - exit_frames: List of frame indices when guest exited
            - exit_times: List of times (ps) when guest exited
            - exit_guest_indices: List of guest atom indices that exited at each event
            - durations_inside: List of durations (ps) for each stay inside
            - durations_outside: List of durations (ps) for each stay outside
            - total_time_inside: Total time (ps) guest spent inside
            - total_time_outside: Total time (ps) guest spent outside
            - n_entries: Number of times guest entered
            - n_exits: Number of times guest exited
            - first_entry_frame: First entry frame (None if never entered)
            - first_entry_time: First entry time (None if never entered)
        """
        if not self.guest_sel:
            return {}
        
        # Calculate durations for each stay inside
        durations_inside = []
        for i, entry in enumerate(self.entry_events):
            if i < len(self.exit_events):
                duration = self.exit_events[i]['time'] - entry['time']
                durations_inside.append(duration)
        
        # Calculate durations for each stay outside
        durations_outside = []
        
        if self._first_time is None or self._last_time is None:
            # No time information available
            pass
        elif len(self.entry_events) == 0:
            # Guest never entered - entire trajectory is outside
            durations_outside.append(self._last_time - self._first_time)
        else:
            # Guest entered at least once
            # Time before first entry (if guest started outside)
            if self.entry_events[0]['time'] > self._first_time:
                duration = self.entry_events[0]['time'] - self._first_time
                durations_outside.append(duration)
            
            # Time between exits and next entries
            for i, exit_event in enumerate(self.exit_events):
                if i + 1 < len(self.entry_events):
                    duration = self.entry_events[i + 1]['time'] - exit_event['time']
                    durations_outside.append(duration)
            
            # Time after last exit (if guest ended outside)
            # If we have equal or more exits than entries, guest ended outside
            if len(self.exit_events) >= len(self.entry_events) and len(self.exit_events) > 0:
                last_exit = self.exit_events[-1]
                if last_exit['time'] < self._last_time:
                    duration = self._last_time - last_exit['time']
                    durations_outside.append(duration)
        
        # Calculate totals
        total_time_inside = sum(durations_inside) if durations_inside else 0.0
        total_time_outside = sum(durations_outside) if durations_outside else 0.0
        
        first_entry_frame = self.entry_events[0]['frame'] if self.entry_events else None
        
        return {
            'entry_frames': [e['frame'] for e in self.entry_events],
            'entry_times': [e['time'] for e in self.entry_events],
            'entry_guest_indices': [e.get('guest_indices', []) for e in self.entry_events],
            'exit_frames': [e['frame'] for e in self.exit_events],
            'exit_times': [e['time'] for e in self.exit_events],
            'exit_guest_indices': [e.get('guest_indices', []) for e in self.exit_events],
            'durations_inside': durations_inside,
            'durations_outside': durations_outside,
            'total_time_inside': total_time_inside,
            'total_time_outside': total_time_outside,
            'n_entries': len(self.entry_events),
            'n_exits': len(self.exit_events),
            'first_entry_frame': first_entry_frame,
            'first_entry_time': self.entry_events[0]['time'] if self.entry_events else None,
        }
    
    def get_longest_duration_guest_indices(self) -> Optional[Dict]:
        """
        Get the guest indices for the longest duration_inside stay.
        
        Returns:
            Dictionary with:
            - duration: The longest duration (ps)
            - guest_indices: List of guest atom indices for this stay
            - entry_frame: Frame when guest entered
            - entry_time: Time (ps) when guest entered
            - exit_frame: Frame when guest exited
            - exit_time: Time (ps) when guest exited
            - duration_index: Index in durations_inside list
            None if no guest residence stats available or no entries
        """
        if self._analyzer.guest_residence_stats is None:
            return None
        return self._analyzer.get_longest_duration_guest_indices()
    
    def print_longest_duration_guest_indices(self) -> None:
        """
        Print information about the guest indices for the longest duration_inside stay.
        """
        if self._analyzer.guest_residence_stats is None:
            print("No guest residence statistics available or no entries found.")
            return
        self._analyzer.print_longest_duration_guest_indices()
    
    def merge_results(self, other: 'GSAnalyzerObserver') -> None:
        """
        Merge results from another observer instance (used in parallel processing).
        
        This is the legacy merge method. Prefer using _get_aggregator() for new code.
        When _get_aggregator() is defined, the TrajectoryIterator will use it instead.
        This method is kept for backward compatibility.
        
        Args:
            other: Another GSAnalyzerObserver instance with results to merge
        """
        if other is None:
            return
        
        # Use the aggregator to merge results if other has results dict
        if hasattr(other, 'results'):
            aggregator = self._get_aggregator()
            n_rows_before = len(self.rows)
            aggregator.merge(self.results, other.results)
            
            # Debug output
            if len(self.rows) > n_rows_before:
                print(f"GSAnalyzerObserver: Merged {len(self.rows) - n_rows_before} rows from worker (total: {len(self.rows)})")