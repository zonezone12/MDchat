"""
Function to calculate when a guest enters a host during trajectory analysis.
"""

from typing import Optional, Dict, List, Union, Tuple
import numpy as np
import warnings
import time

try:
    import MDAnalysis as mda
except ImportError:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

# Import Client for type hints (optional dependency)
try:
    from dask.distributed import Client
except ImportError:
    Client = None  # type: ignore

from src.TrajectoryIterator import TrajectoryIterator, FrameObserver


class GuestEnteringObserver(FrameObserver):
    """
    Observer to track when a guest enters a host during trajectory iteration.
    """
    
    def __init__(self, host_sel: str, guest_sel: str, 
                 method: str = "distance",
                 distance_threshold: Optional[float] = None,
                 use_volume_analyzer: bool = False):
        """
        Initialize GuestEnteringObserver.
        
        Args:
            host_sel: MDAnalysis selection string for host atoms
            guest_sel: MDAnalysis selection string for guest atoms
            method: Method to determine if guest is inside host
                   - "distance": Use distance from guest COM to host COM
                   - "volume": Use VolumeAnalyzer to check if guest is inside host volume
            distance_threshold: Distance threshold in Angstrom (if None, auto-calculate)
            use_volume_analyzer: If True, use VolumeAnalyzer for more accurate detection
        """
        self.host_sel = host_sel
        self.guest_sel = guest_sel
        self.method = method
        self.distance_threshold = distance_threshold
        self.use_volume_analyzer = use_volume_analyzer
        
        # State tracking
        self.entry_frame: Optional[int] = None  # First entry frame (for backward compatibility)
        self._was_inside = False
        self._volume_analyzer = None
        self._initialized = False
        
        # Track which guest atom indices are currently inside
        self._inside_guest_indices: set = set()
        
        # Detailed tracking for in/out durations
        self.entry_events: List[Dict] = []  # List of entry events: {frame, time, guest_indices}
        self.exit_events: List[Dict] = []   # List of exit events: {frame, time, guest_indices}
        self._current_entry_frame: Optional[int] = None
        self._current_entry_time: Optional[float] = None
        self._first_frame: Optional[int] = None
        self._first_time: Optional[float] = None
        self._last_frame: Optional[int] = None
        self._last_time: Optional[float] = None
    
    def get_selections_needed(self) -> list[str]:
        """Return list of selection strings needed by this observer."""
        return [self.host_sel, self.guest_sel]
    
    def on_frame_start(self, iterator: TrajectoryIterator) -> None:
        """Initialize before iteration starts."""
        universe = iterator.universe
        
        # Initialize VolumeAnalyzer if requested
        if self.use_volume_analyzer or self.method == "volume":
            try:
                from src.VolumeAnalyzer import VolumeAnalyzer
                self._volume_analyzer = VolumeAnalyzer(
                    universe=universe,
                    selection=self.host_sel,
                    spacing=0.5,  # Fine grid for accurate detection
                    probe_radius=1.4
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to initialize VolumeAnalyzer: {e}. "
                    "Falling back to distance-based method."
                )
                self.method = "distance"
                self._volume_analyzer = None
        
        # Calculate distance threshold if not provided
        if self.distance_threshold is None and self.method == "distance":
            # Use first frame to estimate host size
            universe.trajectory[0]
            try:
                host = universe.select_atoms(self.host_sel)
                if len(host) > 0:
                    host_coords = host.positions
                    host_com = host.center_of_geometry()
                    # Use maximum distance from COM to any host atom as threshold
                    max_dist = np.max(np.linalg.norm(host_coords - host_com, axis=1))
                    # Add some margin (20% of max distance)
                    self.distance_threshold = max_dist * 0.8
                else:
                    warnings.warn("Host selection is empty. Using default threshold of 10.0 Å.")
                    self.distance_threshold = 10.0
            except Exception as e:
                warnings.warn(f"Failed to calculate distance threshold: {e}. Using default 10.0 Å.")
                self.distance_threshold = 10.0
        
        self.entry_frame = None
        self._was_inside = False
        self._initialized = True
        
        # Reset detailed tracking
        self._inside_guest_indices = set()
        self.entry_events = []
        self.exit_events = []
        self._current_entry_frame = None
        self._current_entry_time = None
        self._first_frame = None
        self._first_time = None
        self._last_frame = None
        self._last_time = None
    
    def _is_guest_inside_distance(self, universe: mda.Universe) -> Tuple[bool, List[int]]:
        """Check if guest is inside host using distance method.
        
        For multiple guest atoms (e.g., Iodine atoms), calculates the distance
        between each individual guest atom and the host center.
        
        Returns:
            tuple: (is_inside, guest_indices) where:
                - is_inside: True if any guest atom is within threshold
                - guest_indices: List of guest atom indices that are inside
        """
        try:
            host = universe.select_atoms(self.host_sel)
            guest = universe.select_atoms(self.guest_sel)
            
            if len(host) == 0 or len(guest) == 0:
                return False, []
            
            host_com = host.center_of_geometry()
            guest_positions = guest.positions
            inside_indices = []
            
            # Calculate distance from each guest atom to host center
            for idx, guest_pos in enumerate(guest_positions):
                distance = np.linalg.norm(guest_pos - host_com)
                if distance <= self.distance_threshold:
                    inside_indices.append(guest[idx].id)
            
            return len(inside_indices) > 0, inside_indices
        except Exception:
            return False, []
    
    def _is_guest_inside_volume(self, universe: mda.Universe, frame_idx: int) -> Tuple[bool, List[int]]:
        """Check if guest is inside host using VolumeAnalyzer.
        
        Returns:
            tuple: (is_inside, guest_indices) where:
                - is_inside: True if any guest atom is inside host volume
                - guest_indices: List of guest atom indices that are inside
        """
        if self._volume_analyzer is None:
            return False, []
        
        try:
            # Get host volume mask
            # Pass universe to avoid creating new Universe in compute_frame
            target_vol, cavity_vol, inside_mask, cavities = self._volume_analyzer.compute_frame(
                frame_idx, return_masks=True, universe=universe
            )
            
            # Get guest atom positions
            guest = universe.select_atoms(self.guest_sel)
            if len(guest) == 0:
                return False, []
            
            guest_positions = guest.positions
            inside_indices = []
            
            # Get grid information
            if self._volume_analyzer._last_grid_axes is None:
                return False, []
            
            x, y, z = self._volume_analyzer._last_grid_axes
            # Grid origin is at the first grid point (x[0], y[0], z[0])
            origin = np.array([x[0], y[0], z[0]])
            spacing = self._volume_analyzer.spacing
            
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
            
            return len(inside_indices) > 0, inside_indices
        except Exception as e:
            warnings.warn(f"Volume-based check failed at frame {frame_idx}: {e}")
            return False, []
    
    def on_frame(self, ts: mda.coordinates.base.Timestep, frame_idx: int,
                 universe: mda.Universe) -> None:
        """Process a single frame during iteration."""
        if not self._initialized:
            return
        
        # Track first and last frame/time
        if self._first_frame is None:
            self._first_frame = ts.frame
            self._first_time = ts.time
        self._last_frame = ts.frame
        self._last_time = ts.time
        
        # Check if guest is inside host
        if self.method == "volume" and self._volume_analyzer is not None:
            is_inside, current_guest_indices = self._is_guest_inside_volume(universe, ts.frame)
        else:
            is_inside, current_guest_indices = self._is_guest_inside_distance(universe)
        
        # Convert to set for easier comparison
        current_guest_indices_set = set(current_guest_indices)
        
        # Detect new entries: atoms that just entered
        newly_entered = current_guest_indices_set - self._inside_guest_indices
        if newly_entered:
            if self.entry_frame is None:
                self.entry_frame = ts.frame  # First entry (for backward compatibility)
            self._current_entry_frame = ts.frame
            self._current_entry_time = ts.time
            self.entry_events.append({
                'frame': ts.frame,
                'time': ts.time,
                'guest_indices': sorted(list(newly_entered))
            })
        
        # Detect exits: atoms that just exited
        newly_exited = self._inside_guest_indices - current_guest_indices_set
        if newly_exited:
            if self._current_entry_frame is not None:
                self.exit_events.append({
                    'frame': ts.frame,
                    'time': ts.time,
                    'guest_indices': sorted(list(newly_exited))
                })
                # Only clear current entry if all atoms have exited
                if len(current_guest_indices_set) == 0:
                    self._current_entry_frame = None
                    self._current_entry_time = None
        
        # Update tracking
        self._was_inside = is_inside
        self._inside_guest_indices = current_guest_indices_set
    
    def on_frame_end(self, iterator: TrajectoryIterator) -> None:
        """Finalize after iteration completes."""
        # If guest is still inside at the end, record the final frame as exit
        if self._was_inside and self._current_entry_frame is not None:
            if self._last_frame is not None and self._last_time is not None:
                self.exit_events.append({
                    'frame': self._last_frame,
                    'time': self._last_time,
                    'guest_indices': sorted(list(self._inside_guest_indices))
                })
    
    def get_entry_frame(self) -> Optional[int]:
        """Get the frame index where guest entered host (first entry)."""
        return self.entry_frame
    
    def merge_results(self, other: 'GuestEnteringObserver') -> None:
        """
        Merge results from another observer instance (used in parallel processing).
        
        Args:
            other: Another GuestEnteringObserver instance with results to merge
        """
        if other is None:
            return
        
        # Merge entry events (sorted by frame)
        self.entry_events.extend(other.entry_events)
        self.entry_events.sort(key=lambda x: x['frame'])
        
        # Merge exit events (sorted by frame)
        self.exit_events.extend(other.exit_events)
        self.exit_events.sort(key=lambda x: x['frame'])
        
        # Update first entry frame if needed
        if other.entry_frame is not None:
            if self.entry_frame is None or other.entry_frame < self.entry_frame:
                self.entry_frame = other.entry_frame
        
        # Update first/last frame and time
        if other._first_frame is not None:
            if self._first_frame is None or other._first_frame < self._first_frame:
                self._first_frame = other._first_frame
                self._first_time = other._first_time
        
        if other._last_frame is not None:
            if self._last_frame is None or other._last_frame > self._last_frame:
                self._last_frame = other._last_frame
                self._last_time = other._last_time
    
    def get_residence_stats(self) -> Dict:
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
            'first_entry_frame': self.entry_frame,
            'first_entry_time': self.entry_events[0]['time'] if self.entry_events else None,
        }


def guest_entering(universe: mda.Universe, host_sel: str, guest_sel: str,
                   method: str = "distance",
                   distance_threshold: Optional[float] = None,
                   use_volume_analyzer: bool = False,
                   return_stats: bool = False,
                   use_dask: bool = False,
                   n_jobs: Optional[int] = None,
                   dask_client: Optional[Client] = None) -> Union[Optional[int], Dict]:
    """
    Calculate the frame index when a guest enters a host and optionally monitor
    residence times inside/outside the host.
    
    This function uses TrajectoryIterator to efficiently iterate through
    the trajectory and detect when the guest enters/exits the host.
    
    TrajectoryIterator handles all parallelization internally (Dask or multiprocessing).
    
    Args:
        universe: MDAnalysis Universe containing the trajectory
        host_sel: MDAnalysis selection string for host atoms
        guest_sel: MDAnalysis selection string for guest atoms
        method: Method to determine if guest is inside host
               - "distance": Use distance from guest COM to host COM (faster)
               - "volume": Use VolumeAnalyzer to check if guest is inside host volume (more accurate)
        distance_threshold: Distance threshold in Angstrom for distance method.
                          If None, automatically calculated from host size.
        use_volume_analyzer: If True, use VolumeAnalyzer for more accurate detection
                            (overrides method parameter)
        return_stats: If True, return detailed statistics dictionary instead of just frame index
        use_dask: If True, prefer Dask Distributed for parallelization (requires dask installed).
                 If False, use multiprocessing. TrajectoryIterator will handle the fallback.
        n_jobs: Number of parallel jobs. If None, uses all available CPU cores.
               If 1, runs sequentially. If > 1, uses parallel processing.
               If -1, uses all available CPU cores.
               Ignored if dask_client is provided (uses client's workers).
        dask_client: Optional Dask Distributed Client for cluster computing.
                    If provided, uses this client instead of creating a local one.
                    If None and use_dask=True, TrajectoryIterator creates a local Dask cluster.
                    Example: Client("tcp://scheduler:8786") for HPC cluster.
    
    Returns:
        If return_stats=False:
            Frame index (int) when guest enters host, or None if guest never enters.
        If return_stats=True:
            Dictionary with detailed statistics including:
            - entry_frames: List of entry frame indices
            - entry_times: List of entry times (ps)
            - entry_guest_indices: List of guest atom indices that entered at each event
            - exit_frames: List of exit frame indices
            - exit_times: List of exit times (ps)
            - exit_guest_indices: List of guest atom indices that exited at each event
            - durations_inside: List of durations (ps) for each stay inside
            - durations_outside: List of durations (ps) for each stay outside
            - total_time_inside: Total time (ps) spent inside
            - total_time_outside: Total time (ps) spent outside
            - n_entries: Number of entry events
            - n_exits: Number of exit events
            - first_entry_frame: First entry frame (None if never entered)
            - first_entry_time: First entry time (None if never entered)
    
    Example:
        >>> import MDAnalysis as mda
        >>> u = mda.Universe("topology.pdb", "trajectory.xtc")
        >>> 
        >>> # Simple usname IOD")
        >>> print(f"Guest entered at frame {frame}")age: get first entry frame
        >>> frame = guest_entering(u, "resname GSA", "res
        >>> 
        >>> # Detailed usage: get residence statistics
        >>> stats = guest_entering(u, "resname GSA", "resname IOD", return_stats=True)
        >>> print(f"Total time inside: {stats['total_time_inside']:.2f} ps")
        >>> print(f"Number of entries: {stats['n_entries']}")
        >>> print(f"Average stay duration: {np.mean(stats['durations_inside']):.2f} ps")
        >>> 
        >>> # Parallel processing with 4 workers
        >>> frame = guest_entering(u, "resname GSA", "resname IOD", n_jobs=4)
        >>> 
        >>> # Use Dask cluster
        >>> from dask.distributed import Client
        >>> client = Client("tcp://scheduler:8786")
        >>> frame = guest_entering(u, "resname GSA", "resname IOD", dask_client=client)
    """
    # Create observer
    observer = GuestEnteringObserver(
        host_sel=host_sel,
        guest_sel=guest_sel,
        method=method,
        distance_threshold=distance_threshold,
        use_volume_analyzer=use_volume_analyzer
    )
    
    # Create iterator and subscribe observer
    iterator = TrajectoryIterator(universe, use_dask=use_dask)
    iterator.subscribe(observer)
    
    # Iterate through trajectory and time the calculation
    start_time = time.time()
    
    # Determine n_jobs: if None, use -1 to use all cores; otherwise pass as-is
    if n_jobs is None:
        n_jobs = -1
    
    # Let TrajectoryIterator handle all parallelization
    iterator.iterate(n_jobs=n_jobs, dask_client=dask_client)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Report calculation time
    if elapsed_time < 60:
        print(f"Calculation completed in {elapsed_time:.2f} seconds")
    else:
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"Calculation completed in {minutes} minute(s) {seconds:.2f} seconds")
    
    # Return appropriate result
    if return_stats:
        return observer.get_residence_stats()
    else:
        return observer.get_entry_frame()

