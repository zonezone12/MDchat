"""
TrajectoryIterator: Single-pass trajectory iteration using Observer Pattern.

This module provides a TrajectoryIterator that iterates through a trajectory once
and notifies registered observers (subscribers) for each frame. This eliminates
the need to store coordinates in memory since MDAnalysis already provides them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, Tuple
import warnings
import pickle
import os
import threading
import hashlib

import numpy as np

try:
    import MDAnalysis as mda
except ImportError:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

# Import ResultsGroup from separate Aggregator module
from ..Aggregator import ResultsGroup

# Try to import Dask Distributed (preferred)
try:
    from dask.distributed import Client, as_completed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    Client = None
    as_completed = None

# Fallback to multiprocessing if Dask is not available
try:
    import multiprocessing as mp
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False
    mp = None

# Worker-local storage for Universe instances (to avoid recreation overhead)
# This is a module-level dictionary that will be keyed by (worker_id, file_hash)
# For multiprocessing: use process name
# For Dask: use worker address
_universe_cache: Dict[Tuple[str, str], Any] = {}

# Global storage for fork-based sharing (Linux copy-on-write optimization)
# When using fork(), child processes inherit parent's memory space
# If they only READ the data, it's shared via copy-on-write (no memory duplication)
_shared_universe: Optional[mda.Universe] = None
_shared_frame_indices: Optional[List[int]] = None
_shared_observers: Optional[List[Any]] = None

def _can_use_fork() -> bool:
    """Check if we can use fork-based multiprocessing (Linux COW optimization)."""
    import sys
    import platform
    
    # Fork is efficient on Linux, available but less efficient on macOS
    # Not available on Windows
    if platform.system() == 'Windows':
        return False
    
    # Check current start method
    if MULTIPROCESSING_AVAILABLE:
        try:
            current_method = mp.get_start_method(allow_none=True)
            if current_method is None:
                # Not set yet - on Linux default is fork
                return platform.system() == 'Linux'
            return current_method == 'fork'
        except Exception:
            return False
    return False

_cache_lock = threading.Lock()


def _get_worker_id() -> str:
    """Get unique identifier for current worker process."""
    try:
        # Try Dask worker first
        if DASK_AVAILABLE:
            try:
                from dask.distributed import get_worker
                worker = get_worker()
                if worker is not None:
                    return f"dask_{worker.address}"
            except (ImportError, ValueError, RuntimeError):
                pass
    except:
        pass
    
    # Fallback to multiprocessing process name
    if MULTIPROCESSING_AVAILABLE:
        try:
            return f"mp_{mp.current_process().name}_{os.getpid()}"
        except:
            pass
    
    # Last resort: use thread ID
    return f"thread_{threading.current_thread().ident}"


def _get_file_hash(top_file: str, traj_file: str) -> str:
    """Generate hash for file paths to use as cache key."""
    combined = f"{top_file}|{traj_file}"
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def _get_cached_universe(top_file: str, traj_file: str, traj_format: Optional[str]) -> Optional[Any]:
    """Get cached Universe instance for current worker, or None if not cached."""
    worker_id = _get_worker_id()
    file_hash = _get_file_hash(top_file, traj_file)
    cache_key = (worker_id, file_hash)
    
    with _cache_lock:
        return _universe_cache.get(cache_key)


def _cache_universe(universe: Any, top_file: str, traj_file: str) -> None:
    """Cache Universe instance for current worker."""
    worker_id = _get_worker_id()
    file_hash = _get_file_hash(top_file, traj_file)
    cache_key = (worker_id, file_hash)
    
    with _cache_lock:
        _universe_cache[cache_key] = universe


def _clear_worker_cache(worker_id: Optional[str] = None) -> None:
    """Clear Universe cache for a specific worker or all workers."""
    with _cache_lock:
        if worker_id is None:
            _universe_cache.clear()
        else:
            keys_to_remove = [k for k in _universe_cache.keys() if k[0] == worker_id]
            for key in keys_to_remove:
                del _universe_cache[key]



class FrameObserver(ABC):
    """
    Abstract base class for frame observers in the Observer Pattern.
    
    Observers can store results in the `results` dict and define aggregation
    strategies via `_get_aggregator()` for parallel processing. Alternatively,
    they can override `merge_results()` for custom merge logic.
    """
    
    def __init__(self):
        """Initialize observer with empty results dictionary."""
        self.results: Dict[str, Any] = {}
    
    @abstractmethod
    def on_frame_start(self, iterator: 'TrajectoryIterator') -> None:
        """Called once before iteration starts."""
        pass
    
    @abstractmethod
    def on_frame(self, ts: mda.coordinates.base.Timestep, frame_idx: int, 
                 universe: mda.Universe) -> None:
        """
        Called for each frame during iteration.
        
        Args:
            ts: MDAnalysis Timestep object for the current frame
            frame_idx: Zero-based index of the current frame
            universe: MDAnalysis Universe (already positioned at current frame)
        """
        pass
    
    @abstractmethod
    def on_frame_end(self, iterator: 'TrajectoryIterator') -> None:
        """Called once after iteration completes."""
        pass
    
    def get_selections_needed(self) -> List[str]:
        """
        Return list of selection strings this observer needs.
        Used for optimization and validation.
        """
        return []
    
    def _get_aggregator(self) -> Optional[ResultsGroup]:
        """
        Return a ResultsGroup defining how to merge results from parallel workers.
        
        Override this method to define declarative aggregation strategies for
        the results stored in self.results. If None is returned, the legacy
        merge_results() method will be used instead.
        
        Example:
            def _get_aggregator(self):
                return ResultsGroup(lookup={
                    'rows': ResultsGroup.list_extend_sorted('frame'),
                    'frame_count': ResultsGroup.sum_values,
                })
        
        Returns:
            ResultsGroup instance with aggregation lookup, or None for legacy behavior
        """
        return None
    
    def merge_results(self, other: 'FrameObserver') -> None:
        """
        Merge results from another observer instance (used in parallel processing).
        
        This method is called when _get_aggregator() returns None. If an aggregator
        is defined, it will be used instead of this method.
        
        By default, this uses the aggregator if available, otherwise does nothing.
        Subclasses should either override _get_aggregator() (preferred) or this method.
        
        Args:
            other: Another observer instance with results to merge
        """
        # Try to use aggregator if available
        aggregator = self._get_aggregator()
        if aggregator is not None and hasattr(other, 'results'):
            aggregator.merge(self.results, other.results)


class TrajectoryIterator:
    """
    Single-pass trajectory iterator using Observer Pattern.
    
    Iterates through a trajectory once and notifies all registered observers
    for each frame. This eliminates redundant iterations and memory storage.
    
    Supports both sequential and parallel processing. For parallel processing,
    observers must be pickle-able and should implement merge_results() to
    combine results from parallel workers.
    
    Parallel processing is implemented using Dask Distributed, which supports both
    local multi-core processing and distributed cluster computing (HPC). Falls back
    to multiprocessing if Dask is not available.
    
    Example:
        >>> iterator = TrajectoryIterator(u)
        >>> endpoint_observer = EndpointAnalyzerObserver(...)
        >>> volume_observer = VolumeAnalyzerObserver(...)
        >>> iterator.subscribe(endpoint_observer)
        >>> iterator.subscribe(volume_observer)
        >>> iterator.iterate()  # Sequential processing
        >>> iterator.iterate(n_jobs=4)  # Parallel processing with 4 workers
        >>> iterator.iterate(n_jobs=-1)  # Use all available CPU cores
        
        # With Dask Distributed cluster
        >>> from dask.distributed import Client
        >>> client = Client("tcp://scheduler:8786")  # HPC cluster
        >>> iterator.iterate(dask_client=client)  # Use cluster
    """
    
    def __init__(self, universe: mda.Universe, use_dask: bool = False):
        """
        Initialize TrajectoryIterator.
        
        Args:
            universe: MDAnalysis Universe to iterate
            use_dask: If True, prefer Dask Distributed for parallel processing.
                     Falls back to multiprocessing if Dask is unavailable.
                     If False, use multiprocessing directly.
        """
        self.universe = universe
        self.observers: List[FrameObserver] = []
        self.n_frames = len(universe.trajectory)
        self.frame_indices: List[int] = []
        self.times: List[float] = []
        self._iterated = False
        self.use_dask = use_dask and DASK_AVAILABLE
        self._dask_client: Optional[Client] = None
        self._dask_client_managed = False  # Track if we created the client
    
    def subscribe(self, observer: FrameObserver) -> None:
        """
        Subscribe an observer to receive frame events.
        
        Args:
            observer: FrameObserver instance to subscribe
        """
        if observer not in self.observers:
            self.observers.append(observer)
    
    def unsubscribe(self, observer: FrameObserver) -> None:
        """
        Unsubscribe an observer from frame events.
        
        Args:
            observer: FrameObserver instance to unsubscribe
        """
        if observer in self.observers:
            self.observers.remove(observer)
    
    def iterate(self, start: Optional[int] = None, stop: Optional[int] = None, 
                step: Optional[int] = None, n_jobs: Optional[int] = None,
                dask_client: Optional[Client] = None,
                preload_coordinates: bool = False,
                max_workers_for_io: Optional[int] = None) -> None:
        """
        Iterate through trajectory once and notify all observers.
        
        Args:
            start: Start frame index (default: 0)
            stop: Stop frame index (default: n_frames)
            step: Step size (default: 1)  
            n_jobs: Number of parallel jobs. If None or 1, runs sequentially.
                   If > 1, uses parallel processing (Dask or multiprocessing).
                   If -1, uses all available CPU cores.
                   Ignored if dask_client is provided (uses client's workers).
            dask_client: Optional Dask Distributed Client for cluster computing.
                        If provided, uses this client instead of creating a local one.
                        If None and use_dask=True, creates a local Dask cluster.
                        Example: Client("tcp://scheduler:8786") for HPC cluster.
            preload_coordinates: If True, pre-load all coordinates into memory.
                                Only use if you have sufficient RAM. Default: False.
            max_workers_for_io: Maximum number of workers for I/O-bound operations.
                               If None, automatically limits to 4 for I/O-bound tasks.
                               Set to a higher value if using fast storage (e.g., SSD arrays).
        """
        if self._iterated:
            warnings.warn(
                "TrajectoryIterator has already been iterated. "
                "Reset observers if you need to iterate again."
            )
        
        # Determine number of jobs
        if n_jobs is None or n_jobs == 1:
            self._iterate_sequential(start, stop, step, preload_coordinates)
        else:
            # Limit workers for I/O-bound operations
            if max_workers_for_io is None:
                max_workers_for_io = 16  # Default limit for I/O-bound tasks
            
            if n_jobs == -1:
                n_jobs = os.cpu_count() or 2
            
            # Apply I/O worker limit
            if n_jobs > max_workers_for_io:
                original_n_jobs = n_jobs
                n_jobs = max_workers_for_io
                warnings.warn(
                    f"Limiting workers from {original_n_jobs} to {n_jobs} to avoid I/O contention. "
                    f"Set max_workers_for_io to override this limit."
                )
            
            # Use provided client or determine parallel backend
            if dask_client is not None:
                self._dask_client = dask_client
                self._dask_client_managed = False
                self._iterate_parallel_dask(start, stop, step, n_jobs, dask_client, preload_coordinates)
            elif self.use_dask:
                self._iterate_parallel_dask(start, stop, step, n_jobs, None, preload_coordinates)
            else:
                # Fallback to multiprocessing
                if not MULTIPROCESSING_AVAILABLE:
                    warnings.warn(
                        "Neither Dask nor multiprocessing available. "
                        "Falling back to sequential processing."
                    )
                    self._iterate_sequential(start, stop, step, preload_coordinates)
                else:
                    self._iterate_parallel_multiprocessing(start, stop, step, n_jobs, preload_coordinates)
        
        # Clean up managed Dask client if we created it
        if self._dask_client_managed and self._dask_client is not None:
            try:
                self._dask_client.close()
            except Exception:
                pass
            self._dask_client = None
            self._dask_client_managed = False
        
        self._iterated = True
    
    def _iterate_sequential(self, start: Optional[int] = None, 
                           stop: Optional[int] = None, 
                           step: Optional[int] = None,
                           preload_coordinates: bool = False) -> None:
        """Sequential iteration (original implementation)."""
        # Notify observers that iteration is starting
        for observer in self.observers:
            observer.on_frame_start(self)
        
        # Reset frame tracking
        self.frame_indices = []
        self.times = []
        
        # Iterate through trajectory
        frame_idx = 0
        for ts in self.universe.trajectory[start:stop:step]:
            self.frame_indices.append(ts.frame)
            self.times.append(ts.time)
            
            # Notify all observers
            for observer in self.observers:
                try:
                    observer.on_frame(ts, frame_idx, self.universe)
                except Exception as e:
                    warnings.warn(
                        f"Observer {type(observer).__name__} raised exception "
                        f"on frame {frame_idx}: {e}"
                    )
            
            frame_idx += 1
        
        # Notify observers that iteration is complete
        for observer in self.observers:
            observer.on_frame_end(self)
    
    def _iterate_parallel_dask(self, start: Optional[int] = None,
                               stop: Optional[int] = None,
                               step: Optional[int] = None,
                               n_jobs: Optional[int] = None,
                               client: Optional[Client] = None,
                               preload_coordinates: bool = False) -> None:
        """Parallel iteration using Dask Distributed."""
        if not DASK_AVAILABLE:
            warnings.warn(
                "Dask not available. Falling back to sequential processing."
            )
            self._iterate_sequential(start, stop, step, preload_coordinates)
            return
        
        # Get topology file path (only need topology, not trajectory file)
        try:
            top_file = self.universe.filename
        except (AttributeError, TypeError) as e:
            warnings.warn(
                f"Cannot get topology file path from Universe for parallel processing: {e}. "
                "Falling back to sequential processing."
            )
            self._iterate_sequential(start, stop, step)
            return
        
        # Collect all frame indices to process
        traj = self.universe.trajectory
        n_frames = len(traj)
        s = slice(start, stop, step)
        indices = list(range(n_frames))[s]
        frame_indices_to_process = indices

        if len(frame_indices_to_process) == 0:
            return
        
        # Extract coordinates and times from trajectory in main process
        # This avoids workers having to re-open trajectory files
        print("Extracting coordinates from trajectory (main process)...")
        all_coords = {}  # {frame_idx: positions_array}
        all_times = {}   # {frame_idx: time}
        all_frame_nums = {}  # {frame_idx: frame_number}
        dimensions = None
        
        for frame_idx in frame_indices_to_process:
            traj[frame_idx]
            ts = traj.ts
            # Copy positions as C-contiguous array to ensure proper serialization
            all_coords[frame_idx] = np.ascontiguousarray(ts.positions.copy(), dtype=np.float64)
            all_times[frame_idx] = ts.time
            all_frame_nums[frame_idx] = ts.frame
            if dimensions is None and ts.dimensions is not None:
                dimensions = ts.dimensions.copy()
        
        times_to_process = [all_times[idx] for idx in frame_indices_to_process]
        print(f"Extracted coordinates for {len(all_coords)} frames")
        
        # Validate that all observers can be pickled before starting parallel processing
        unpickleable_observers = []
        for observer in self.observers:
            try:
                pickled = pickle.dumps(observer)
                pickle.loads(pickled)
            except Exception as e:
                unpickleable_observers.append((type(observer).__name__, str(e)))
        
        if unpickleable_observers:
            warnings.warn(
                f"The following observers cannot be pickled and will not work with parallel processing:\n" +
                "\n".join([f"  - {name}: {error}" for name, error in unpickleable_observers]) +
                "\nFalling back to sequential processing."
            )
            self._iterate_sequential(start, stop, step, preload_coordinates)
            return
        
        # Notify observers that iteration is starting
        for observer in self.observers:
            observer.on_frame_start(self)
        
        # Split frames into batches for parallel processing
        n_frames_to_process = len(frame_indices_to_process)
        
        # Determine number of workers
        if client is None:
            # Create local Dask client
            if n_jobs is None or n_jobs == -1:
                n_jobs = os.cpu_count() or 2
            client = Client(n_workers=n_jobs, threads_per_worker=1)
            self._dask_client = client
            self._dask_client_managed = True
            print(f"Created local Dask cluster with {n_jobs} workers")
            print(f"Dask dashboard: {client.dashboard_link}")
        else:
            # Use provided client
            self._dask_client = client
            self._dask_client_managed = False
            n_workers = len(client.scheduler_info()['workers'])
            print(f"Using Dask cluster with {n_workers} workers")
            if hasattr(client, 'dashboard_link'):
                print(f"Dask dashboard: {client.dashboard_link}")
        
        n_workers = len(client.scheduler_info()['workers'])
        
        # Calculate batch size - simpler now since we're not recreating Universes
        target_batches_per_worker = 1.5
        target_total_batches = max(1, int(n_workers * target_batches_per_worker))
        batch_size = max(1, n_frames_to_process // target_total_batches)
        
        batches = []
        for i in range(0, n_frames_to_process, batch_size):
            batch_end = min(i + batch_size, n_frames_to_process)
            batch_frame_indices = frame_indices_to_process[i:batch_end]
            
            # Extract coords/times for this batch
            batch_coords = {idx: all_coords[idx] for idx in batch_frame_indices}
            batch_times = {idx: all_times[idx] for idx in batch_frame_indices}
            batch_frames = {local_i: all_frame_nums[idx] for local_i, idx in enumerate(batch_frame_indices)}
            
            batches.append((i, batch_end, batch_coords, batch_times, batch_frames))
        
        print(f"Processing {n_frames_to_process} frames in {len(batches)} batches "
              f"(~{batch_size} frames per batch, {n_workers} workers)")
        
        # Submit batches to Dask cluster
        try:
            futures = []
            for batch_start_idx, batch_end_idx, batch_coords, batch_times, batch_frames in batches:
                future = client.submit(
                    _process_frame_batch_with_coords,
                    top_file,
                    batch_coords,
                    batch_times,
                    batch_frames,
                    batch_start_idx,
                    self.observers,
                    dimensions
                )
                futures.append(future)
            
            # Gather results as they complete (with progress tracking)
            results = []
            completed = 0
            total = len(futures)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if completed % max(1, total // 10) == 0:
                        print(f"Progress: {completed}/{total} batches completed ({100*completed/total:.1f}%)")
                except Exception as e:
                    warnings.warn(
                        f"Error processing batch in Dask worker: {e}. "
                        "This batch will be skipped."
                    )
                    results.append(None)
            
            print(f"Completed all {total} batches")
            
        except Exception as e:
            warnings.warn(
                f"Dask parallel processing failed: {e}. "
                "Falling back to sequential processing."
            )
            self._iterate_sequential(start, stop, step, preload_coordinates)
            return
        
        # Merge results from all batches
        self.frame_indices = frame_indices_to_process
        self.times = times_to_process
        
        # Merge observer results from parallel workers using ResultsGroup or legacy method
        for batch_results in results:
            if batch_results is None:
                continue
            for observer_idx, observer_result in enumerate(batch_results):
                if observer_idx < len(self.observers) and observer_result is not None:
                    try:
                        observer = self.observers[observer_idx]
                        aggregator = observer._get_aggregator()
                        if aggregator is not None and hasattr(observer_result, 'results'):
                            # Use declarative ResultsGroup aggregation
                            aggregator.merge(observer.results, observer_result.results)
                        else:
                            # Fall back to legacy merge_results method
                            observer.merge_results(observer_result)
                    except Exception as e:
                        warnings.warn(
                            f"Failed to merge results for observer "
                            f"{type(self.observers[observer_idx]).__name__}: {e}"
                        )
        
        # Notify observers that iteration is complete
        for observer in self.observers:
            observer.on_frame_end(self)
    
    def _iterate_parallel_multiprocessing(self, start: Optional[int] = None,
                                         stop: Optional[int] = None,
                                         step: Optional[int] = None,
                                         n_jobs: int = 2,
                                         preload_coordinates: bool = False) -> None:
        """Parallel iteration using multiprocessing (fallback)."""
        if not MULTIPROCESSING_AVAILABLE:
            warnings.warn(
                "Multiprocessing not available. Falling back to sequential processing."
            )
            self._iterate_sequential(start, stop, step, preload_coordinates)
            return
        
        # Check if we can use fork-based copy-on-write (Linux optimization)
        use_fork = _can_use_fork()
        
        if use_fork:
            self._iterate_parallel_fork(start, stop, step, n_jobs)
        else:
            self._iterate_parallel_coords(start, stop, step, n_jobs, preload_coordinates)
    
    def _iterate_parallel_fork(self, start: Optional[int] = None,
                               stop: Optional[int] = None,
                               step: Optional[int] = None,
                               n_jobs: int = 2) -> None:
        """
        Parallel iteration using fork-based copy-on-write (Linux only).
        
        This is the most efficient approach on Linux:
        - Parent process loads trajectory (once)
        - Child processes share memory via copy-on-write
        - No data copying if children only READ the trajectory
        """
        global _shared_universe, _shared_observers
        
        print("Using fork-based parallel processing (Linux copy-on-write optimization)")
        
        # Collect frame indices
        traj = self.universe.trajectory
        n_frames_total = len(traj)
        s = slice(start, stop, step)
        frame_indices_to_process = list(range(n_frames_total))[s]
        
        if len(frame_indices_to_process) == 0:
            return
        
        # Store times for later
        times_to_process = []
        for idx in frame_indices_to_process:
            traj[idx]
            times_to_process.append(traj.ts.time)
        
        # Validate observers can be pickled (still needed for their state)
        unpickleable_observers = []
        for observer in self.observers:
            try:
                pickled = pickle.dumps(observer)
                pickle.loads(pickled)
            except Exception as e:
                unpickleable_observers.append((type(observer).__name__, str(e)))
        
        if unpickleable_observers:
            warnings.warn(
                f"The following observers cannot be pickled:\n" +
                "\n".join([f"  - {name}: {error}" for name, error in unpickleable_observers]) +
                "\nFalling back to coordinate-based parallel processing."
            )
            self._iterate_parallel_coords(start, stop, step, n_jobs, False)
            return
        
        # Notify observers that iteration is starting
        for observer in self.observers:
            observer.on_frame_start(self)
        
        # Set up shared state for fork
        _shared_universe = self.universe
        _shared_observers = self.observers
        
        # Split frames into batches
        n_frames = len(frame_indices_to_process)
        batch_size = max(1, n_frames // n_jobs)
        
        batches = []
        for i in range(0, n_frames, batch_size):
            batch_end = min(i + batch_size, n_frames)
            batch_frame_indices = frame_indices_to_process[i:batch_end]
            batches.append((batch_frame_indices, i))  # (frame_indices, batch_start_idx)
        
        print(f"Processing {n_frames} frames in {len(batches)} batches (~{batch_size} per batch, {n_jobs} workers)")
        print("Workers will share trajectory memory via copy-on-write (no memory duplication)")
        
        # Process batches in parallel using fork
        try:
            with mp.Pool(processes=n_jobs) as pool:
                results = pool.starmap(_process_frame_batch_fork, batches)
        except Exception as e:
            warnings.warn(f"Fork-based parallel processing failed: {e}. Falling back to sequential.")
            _shared_universe = None
            _shared_observers = None
            self._iterate_sequential(start, stop, step, False)
            return
        finally:
            # Clean up shared state
            _shared_universe = None
            _shared_observers = None
        
        # Merge results using ResultsGroup or legacy method
        self.frame_indices = frame_indices_to_process
        self.times = times_to_process
        
        for batch_results in results:
            if batch_results is None:
                continue
            for observer_idx, observer_result in enumerate(batch_results):
                if observer_idx < len(self.observers) and observer_result is not None:
                    try:
                        observer = self.observers[observer_idx]
                        aggregator = observer._get_aggregator()
                        if aggregator is not None and hasattr(observer_result, 'results'):
                            # Use declarative ResultsGroup aggregation
                            aggregator.merge(observer.results, observer_result.results)
                        else:
                            # Fall back to legacy merge_results method
                            observer.merge_results(observer_result)
                    except Exception as e:
                        warnings.warn(f"Failed to merge results: {e}")
        
        # Notify observers that iteration is complete
        for observer in self.observers:
            observer.on_frame_end(self)
    
    def _iterate_parallel_coords(self, start: Optional[int] = None,
                                 stop: Optional[int] = None,
                                 step: Optional[int] = None,
                                 n_jobs: int = 2,
                                 preload_coordinates: bool = False) -> None:
        """
        Parallel iteration by extracting coordinates (Windows/macOS or fallback).
        
        Extracts coordinates in main process and sends to workers.
        Less efficient than fork but works on all platforms.
        """
        # Get topology file path
        try:
            top_file = self.universe.filename
        except (AttributeError, TypeError) as e:
            warnings.warn(f"Cannot get topology file path: {e}. Falling back to sequential.")
            self._iterate_sequential(start, stop, step, preload_coordinates)
            return
        
        # Collect frame indices
        traj = self.universe.trajectory
        n_frames_total = len(traj)
        s = slice(start, stop, step)
        frame_indices_to_process = list(range(n_frames_total))[s]

        if len(frame_indices_to_process) == 0:
            return
        
        # Extract coordinates in main process
        print("Extracting coordinates from trajectory (main process)...")
        all_coords = {}
        all_times = {}
        all_frame_nums = {}
        dimensions = None
        
        for frame_idx in frame_indices_to_process:
            traj[frame_idx]
            ts = traj.ts
            all_coords[frame_idx] = np.ascontiguousarray(ts.positions.copy(), dtype=np.float64)
            all_times[frame_idx] = ts.time
            all_frame_nums[frame_idx] = ts.frame
            if dimensions is None and ts.dimensions is not None:
                dimensions = ts.dimensions.copy()
        
        times_to_process = [all_times[idx] for idx in frame_indices_to_process]
        print(f"Extracted coordinates for {len(all_coords)} frames")
        
        # Validate observers
        unpickleable_observers = []
        for observer in self.observers:
            try:
                pickle.dumps(observer)
                pickle.loads(pickle.dumps(observer))
            except Exception as e:
                unpickleable_observers.append((type(observer).__name__, str(e)))
        
        if unpickleable_observers:
            warnings.warn(
                f"Observers cannot be pickled:\n" +
                "\n".join([f"  - {name}: {error}" for name, error in unpickleable_observers]) +
                "\nFalling back to sequential processing."
            )
            self._iterate_sequential(start, stop, step, preload_coordinates)
            return
        
        # Notify observers
        for observer in self.observers:
            observer.on_frame_start(self)
        
        # Split into batches
        n_frames = len(frame_indices_to_process)
        batch_size = max(1, n_frames // n_jobs)
        
        batches = []
        for i in range(0, n_frames, batch_size):
            batch_end = min(i + batch_size, n_frames)
            batch_frame_indices = frame_indices_to_process[i:batch_end]
            batch_coords = {idx: all_coords[idx] for idx in batch_frame_indices}
            batch_times = {idx: all_times[idx] for idx in batch_frame_indices}
            batch_frames = {local_i: all_frame_nums[idx] for local_i, idx in enumerate(batch_frame_indices)}
            batches.append((i, batch_end, batch_coords, batch_times, batch_frames))
        
        print(f"Processing {n_frames} frames in {len(batches)} batches (~{batch_size} per batch, {n_jobs} workers)")
        
        # Process in parallel
        try:
            with mp.Pool(processes=n_jobs) as pool:
                results = pool.starmap(
                    _process_frame_batch_with_coords,
                    [(top_file, batch_coords, batch_times, batch_frames, batch_start_idx, 
                      self.observers, dimensions) 
                     for batch_start_idx, _, batch_coords, batch_times, batch_frames in batches]
                )
        except Exception as e:
            warnings.warn(f"Parallel processing failed: {e}. Falling back to sequential.")
            self._iterate_sequential(start, stop, step, preload_coordinates)
            return
        
        # Merge results using ResultsGroup or legacy method
        self.frame_indices = frame_indices_to_process
        self.times = times_to_process
        
        for batch_results in results:
            if batch_results is None:
                continue
            for observer_idx, observer_result in enumerate(batch_results):
                if observer_idx < len(self.observers) and observer_result is not None:
                    try:
                        observer = self.observers[observer_idx]
                        aggregator = observer._get_aggregator()
                        if aggregator is not None and hasattr(observer_result, 'results'):
                            # Use declarative ResultsGroup aggregation
                            aggregator.merge(observer.results, observer_result.results)
                        else:
                            # Fall back to legacy merge_results method
                            observer.merge_results(observer_result)
                    except Exception as e:
                        warnings.warn(f"Failed to merge results: {e}")
        
        for observer in self.observers:
            observer.on_frame_end(self)
    
    def _get_trajectory_format(self) -> Optional[str]:
        """Extract trajectory format from Universe."""
        traj_format = None
        
        # First, try to get format from trajectory object
        if hasattr(self.universe.trajectory, 'format'):
            format_attr = self.universe.trajectory.format
            if isinstance(format_attr, (list, tuple)):
                traj_format = format_attr[0] if format_attr else None
            elif isinstance(format_attr, str):
                traj_format = format_attr
            else:
                traj_format = str(format_attr)
        
        # If format is a single character or seems wrong, try class name
        if not traj_format or len(traj_format) == 1:
            if hasattr(self.universe.trajectory, '__class__'):
                # Try to infer from class name (e.g., MDCRDReader -> MDCRD)
                class_name = self.universe.trajectory.__class__.__name__
                # Remove 'Reader' suffix if present
                if class_name.endswith('Reader'):
                    traj_format = class_name[:-6]
                elif 'MDCRD' in class_name.upper():
                    traj_format = 'MDCRD'
                elif 'XTC' in class_name.upper():
                    traj_format = 'XTC'
                elif 'TRR' in class_name.upper():
                    traj_format = 'TRR'
                elif 'DCD' in class_name.upper():
                    traj_format = 'DCD'
        
        # Handle empty string format
        if traj_format == '' or (traj_format and len(traj_format) == 1):
            traj_format = None
        
        # If format is still None or seems wrong, try to infer from filename
        if not traj_format or len(traj_format) <= 2:
            try:
                traj_file = self.universe.trajectory.filename
                if isinstance(traj_file, (list, tuple)):
                    traj_file = traj_file[0]
                traj_file_lower = str(traj_file).lower()
                # Check filename for format hints
                if 'mdcrd' in traj_file_lower:
                    traj_format = 'MDCRD'
                elif traj_file_lower.endswith('.xtc'):
                    traj_format = 'XTC'
                elif traj_file_lower.endswith('.trr'):
                    traj_format = 'TRR'
                elif traj_file_lower.endswith('.dcd'):
                    traj_format = 'DCD'
                elif traj_file_lower.endswith('.pdb'):
                    # PDB files might actually be MDCRD if filename contains 'mdcrd'
                    if 'mdcrd' in traj_file_lower:
                        traj_format = 'MDCRD'
                    else:
                        traj_format = 'PDB'
            except Exception:
                pass  # If we can't infer, return None and let MDAnalysis try
        
        return traj_format
    
    def get_frame_indices(self) -> List[int]:
        """Get list of frame indices from last iteration."""
        return self.frame_indices.copy()
    
    def get_times(self) -> List[float]:
        """Get list of times from last iteration."""
        return self.times.copy()
    
    def get_n_frames(self) -> int:
        """Get total number of frames in trajectory."""
        return self.n_frames
    
    def reset(self) -> None:
        """Reset iterator state (allows re-iteration)."""
        self._iterated = False
        self.frame_indices = []
        self.times = []
        
        # Clean up Dask client if we managed it
        if self._dask_client_managed and self._dask_client is not None:
            try:
                self._dask_client.close()
            except Exception:
                pass
            self._dask_client = None
            self._dask_client_managed = False
    
    def get_dask_client(self) -> Optional[Client]:
        """
        Get the current Dask client (if using Dask).
        
        Returns:
            Dask Client instance or None if not using Dask
        """
        return self._dask_client
    
    def get_dask_dashboard_link(self) -> Optional[str]:
        """
        Get the Dask dashboard link for monitoring (if using Dask).
        
        Returns:
            Dashboard URL string or None if not available
        """
        if self._dask_client is not None and hasattr(self._dask_client, 'dashboard_link'):
            return self._dask_client.dashboard_link
        return None


def _process_frame_batch_with_coords(
    top_file: str,
    batch_coords: Dict[int, np.ndarray],  # {frame_idx: positions_array}
    batch_times: Dict[int, float],  # {frame_idx: time}
    batch_frames: Dict[int, int],  # {local_idx: frame_number}
    batch_start_idx: int,
    observers: List[FrameObserver],
    dimensions: Optional[np.ndarray] = None,  # Box dimensions
) -> List[Any]:
    """
    Process a batch of frames in a worker process using pre-extracted coordinates.
    
    This approach avoids re-opening trajectory files in workers. The main process
    extracts coordinates once and passes them to workers as numpy arrays.
    
    Args:
        top_file: Topology file path (only topology, no trajectory needed)
        batch_coords: Dictionary mapping frame indices to position arrays
        batch_times: Dictionary mapping frame indices to simulation times
        batch_frames: Dictionary mapping local indices to frame numbers
        batch_start_idx: Starting index for this batch (for frame_idx parameter)
        observers: List of observer instances (will be pickled and recreated)
        dimensions: Optional box dimensions array
    
    Returns:
        List of observer results (one per observer)
    """
    import numpy as np
    
    try:
        # Get or create cached Universe from topology only
        worker_id = _get_worker_id()
        cache_key = (worker_id, top_file)
        
        with _cache_lock:
            u = _universe_cache.get(cache_key)
        
        if u is None:
            # Create Universe from topology file only (no trajectory)
            u = mda.Universe(top_file)
            
            # Load an in-memory trajectory so we can set positions
            # Use the first frame's coordinates to initialize
            first_frame_idx = min(batch_coords.keys())
            first_coords = batch_coords[first_frame_idx]
            # Create a single-frame in-memory trajectory
            u.load_new(first_coords[np.newaxis, :, :], format='MEMORY')
            
            with _cache_lock:
                _universe_cache[cache_key] = u
        
        # Recreate observers in worker process (they need to be pickle-able)
        worker_observers = []
        for observer in observers:
            try:
                pickled = pickle.dumps(observer)
                worker_observer = pickle.loads(pickled)
                worker_observers.append(worker_observer)
                if worker_observer is None:
                    warnings.warn(
                        f"Observer {type(observer).__name__} unpickled to None in worker process"
                    )
            except Exception as e:
                warnings.warn(
                    f"Observer {type(observer).__name__} is not pickle-able in worker: {e}"
                )
                worker_observers.append(None)
        
        if all(obs is None for obs in worker_observers):
            warnings.warn("All observers failed to unpickle. This batch will return None.")
            return [None] * len(observers)
        
        # Create a minimal Timestep-like object to pass to observers
        class MinimalTimestep:
            """Minimal timestep object for passing frame info to observers."""
            def __init__(self, frame_num, time_val, positions, dims):
                self.frame = frame_num
                self.time = time_val
                self._pos = positions
                self.dimensions = dims
            
            @property
            def positions(self):
                return self._pos
        
        # Process each frame in the batch
        frame_indices = sorted(batch_coords.keys())
        for local_idx, frame_idx in enumerate(frame_indices):
            global_idx = batch_start_idx + local_idx
            frame_num = batch_frames.get(local_idx, frame_idx)
            
            try:
                # Get coordinates for this frame
                positions = batch_coords[frame_idx]
                time_val = batch_times.get(frame_idx, 0.0)
                
                # Set positions on the Universe's atoms
                u.atoms.positions = positions
                
                # Create timestep object
                ts = MinimalTimestep(frame_num, time_val, positions, dimensions)
                
                # Notify all observers
                for worker_observer in worker_observers:
                    if worker_observer is not None:
                        try:
                            worker_observer.on_frame(ts, global_idx, u)
                        except Exception as e:
                            warnings.warn(
                                f"Observer {type(worker_observer).__name__} raised exception "
                                f"on frame {global_idx}: {e}"
                            )
                            import traceback
                            traceback.print_exc()
            except Exception as e:
                warnings.warn(f"Failed to process frame {frame_idx} (global_idx {global_idx}): {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return worker_observers
        
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        warnings.warn(f"Error in worker process: {e}\nTraceback:\n{error_traceback}")
        return [None] * len(observers)


def _process_frame_batch_fork(
    frame_indices: List[int],
    batch_start_idx: int,
) -> List[Any]:
    """
    Process a batch of frames using fork-based copy-on-write sharing (Linux only).
    
    This function accesses the shared Universe directly from the parent process
    via Linux's copy-on-write memory sharing. Workers only READ the trajectory
    data, so no memory is actually copied - it's shared.
    
    Args:
        frame_indices: List of frame indices to process in this batch
        batch_start_idx: Starting index for this batch (for frame_idx parameter)
    
    Returns:
        List of observer results (one per observer)
    """
    global _shared_universe, _shared_observers
    
    try:
        # Access shared Universe directly (copy-on-write - no memory copy if read-only)
        u = _shared_universe
        if u is None:
            warnings.warn("Shared universe is None in worker. Fork sharing may have failed.")
            return [None] * len(_shared_observers) if _shared_observers else []
        
        # Recreate observers (they need their own state for results)
        worker_observers = []
        for observer in _shared_observers:
            try:
                pickled = pickle.dumps(observer)
                worker_observer = pickle.loads(pickled)
                worker_observers.append(worker_observer)
            except Exception as e:
                warnings.warn(f"Observer {type(observer).__name__} is not pickle-able: {e}")
                worker_observers.append(None)
        
        if all(obs is None for obs in worker_observers):
            return [None] * len(_shared_observers)
        
        # Process each frame - READ trajectory directly (copy-on-write efficient)
        for local_idx, frame_idx in enumerate(frame_indices):
            global_idx = batch_start_idx + local_idx
            
            try:
                # Seek to frame - this only reads data, doesn't modify the trajectory
                # Copy-on-write means this read is from shared memory
                u.trajectory[frame_idx]
                ts = u.trajectory.ts
                
                # Notify all observers
                for worker_observer in worker_observers:
                    if worker_observer is not None:
                        try:
                            worker_observer.on_frame(ts, global_idx, u)
                        except Exception as e:
                            warnings.warn(
                                f"Observer {type(worker_observer).__name__} raised exception "
                                f"on frame {global_idx}: {e}"
                            )
            except Exception as e:
                warnings.warn(f"Failed to process frame {frame_idx}: {e}")
                continue
        
        return worker_observers
        
    except Exception as e:
        import traceback
        warnings.warn(f"Error in fork worker: {e}\n{traceback.format_exc()}")
        return [None] * len(_shared_observers) if _shared_observers else []
