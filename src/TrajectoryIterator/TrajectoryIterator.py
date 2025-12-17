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

try:
    import MDAnalysis as mda
except ImportError:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

# Try to import Dask Distributed (preferred)
try:
    import dask
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
    """Abstract base class for frame observers in the Observer Pattern."""
    
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
    
    def merge_results(self, other: 'FrameObserver') -> None:
        """
        Merge results from another observer instance (used in parallel processing).
        
        By default, this does nothing. Subclasses should override this method
        if they need to merge state from parallel workers.
        
        Args:
            other: Another observer instance with results to merge
        """
        pass


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
    
    def __init__(self, universe: mda.Universe, use_dask: bool = True):
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
        
        # Pre-load coordinates if requested (only for sequential-like processing)
        preloaded_coords = None
        if preload_coordinates:
            try:
                print("Pre-loading coordinates into memory...")
                traj = self.universe.trajectory
                s = slice(start, stop, step)
                indices = list(range(len(traj)))[s]
                preloaded_coords = {}
                for idx in indices:
                    traj[idx]
                    preloaded_coords[idx] = traj.ts.copy()
                print(f"Pre-loaded {len(preloaded_coords)} frames into memory")
            except Exception as e:
                warnings.warn(f"Failed to pre-load coordinates: {e}. Continuing without pre-loading.")
                preloaded_coords = None
        
        # Get trajectory file paths and format for creating new Universe instances
        try:
            top_file = self.universe.filename
            traj_file = self.universe.trajectory.filename
            
            # Get trajectory format to preserve it in worker processes
            traj_format = self._get_trajectory_format()
            
            # Handle multiple trajectory files
            if isinstance(traj_file, (list, tuple)):
                if len(traj_file) > 1:
                    warnings.warn(
                        f"Multiple trajectory files detected. Using first file for parallel processing: {traj_file[0]}"
                    )
                traj_file = traj_file[0]
                
        except (AttributeError, TypeError) as e:
            warnings.warn(
                f"Cannot get file paths from Universe for parallel processing: {e}. "
                "Falling back to sequential processing."
            )
            self._iterate_sequential(start, stop, step)
            return
        
        # Collect all frame indices to process efficiently
        traj = self.universe.trajectory
        n_frames = len(traj)
        s = slice(start, stop, step)
        indices = list(range(n_frames))[s]
        frame_indices_to_process = indices

        # Derive times without reading every frame (assumes evenly spaced frames)
        times_to_process: List[float] = []
        if frame_indices_to_process:
            dt = getattr(traj, "dt", None)
            first_idx = frame_indices_to_process[0]
            try:
                traj[first_idx]  # single seek to get starting time
                t0 = traj.ts.time
            except Exception:
                dt = None
                t0 = None

            if dt is not None and t0 is not None:
                times_to_process = [t0 + (idx - first_idx) * dt for idx in frame_indices_to_process]

        if len(frame_indices_to_process) == 0:
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
        
        # Calculate batch size based on number of workers and overhead estimation
        # Account for Universe creation overhead: ~0.1-1 second per Universe creation
        # Use larger batches to amortize this overhead
        n_workers = len(client.scheduler_info()['workers'])
        
        # Estimate Universe creation overhead (conservative: 0.5 seconds)
        # This means we want batches large enough that Universe creation is < 5% of total time
        universe_overhead_seconds = 0.5
        min_frames_per_universe = 100  # Minimum frames to process per Universe creation
        
        # Calculate optimal batch size accounting for overhead
        # Target: 1-2 batches per worker to minimize scheduling overhead
        target_batches_per_worker = 1.2  # Slightly fewer batches per worker
        target_total_batches = max(1, int(n_workers * target_batches_per_worker))
        
        # Minimum batch size: ensure Universe creation overhead is minimal
        # For small trajectories, use larger minimum to avoid overhead
        if n_frames_to_process < 1000:
            min_batch_size = max(100, n_frames_to_process // max(2, n_workers))
        else:
            min_batch_size = max(min_frames_per_universe, n_frames_to_process // (n_workers * 4))
        
        batch_size = max(min_batch_size, n_frames_to_process // target_total_batches)
        
        # Warn if we still have too many batches (indicates lightweight computation)
        estimated_batches = (n_frames_to_process + batch_size - 1) // batch_size
        if estimated_batches > n_workers * 3 and n_frames_to_process > 1000:
            warnings.warn(
                f"Estimated {estimated_batches} batches for {n_frames_to_process} frames with {n_workers} workers. "
                f"Dask overhead may outweigh benefits for lightweight computations. "
                "Consider using sequential processing (n_jobs=1) or multiprocessing (use_dask=False) instead."
            )
        
        batches = []
        
        for i in range(0, n_frames_to_process, batch_size):
            batch_end = min(i + batch_size, n_frames_to_process)
            batches.append((i, batch_end, frame_indices_to_process[i:batch_end]))
        
        print(f"Processing {n_frames_to_process} frames in {len(batches)} batches "
              f"(~{batch_size} frames per batch, {n_workers} workers)")
        
        # Submit batches to Dask cluster
        try:
            futures = []
            for batch_start_idx, batch_end_idx, batch_frames in batches:
                future = client.submit(
                    _process_frame_batch,
                    top_file,
                    traj_file,
                    traj_format,
                    batch_frames,
                    batch_start_idx,
                    self.observers,
                    preloaded_coords
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
        
        # Merge observer results from parallel workers
        for batch_results in results:
            if batch_results is None:
                continue
            for observer_idx, observer_result in enumerate(batch_results):
                if observer_idx < len(self.observers) and observer_result is not None:
                    try:
                        self.observers[observer_idx].merge_results(observer_result)
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
        
        # Pre-load coordinates if requested
        preloaded_coords = None
        if preload_coordinates:
            try:
                print("Pre-loading coordinates into memory...")
                traj = self.universe.trajectory
                s = slice(start, stop, step)
                indices = list(range(len(traj)))[s]
                preloaded_coords = {}
                for idx in indices:
                    traj[idx]
                    preloaded_coords[idx] = traj.ts.copy()
                print(f"Pre-loaded {len(preloaded_coords)} frames into memory")
            except Exception as e:
                warnings.warn(f"Failed to pre-load coordinates: {e}. Continuing without pre-loading.")
                preloaded_coords = None
        
        # Get trajectory file paths and format for creating new Universe instances
        try:
            top_file = self.universe.filename
            traj_file = self.universe.trajectory.filename
            
            # Get trajectory format to preserve it in worker processes
            traj_format = self._get_trajectory_format()
                
            # Handle multiple trajectory files (convert to list if needed)
            if isinstance(traj_file, (list, tuple)):
                # For multiple files, we'll need to pass them all
                # For now, use the first file and warn
                if len(traj_file) > 1:
                    warnings.warn(
                        f"Multiple trajectory files detected. Using first file for parallel processing: {traj_file[0]}"
                    )
                traj_file = traj_file[0]
                
        except (AttributeError, TypeError) as e:
            warnings.warn(
                f"Cannot get file paths from Universe for parallel processing: {e}. "
                "Falling back to sequential processing."
            )
            self._iterate_sequential(start, stop, step, preload_coordinates)
            return
        
        # Collect all frame indices to process efficiently
        traj = self.universe.trajectory
        n_frames = len(traj)
        # handle None for start/stop/step as slice does
        s = slice(start, stop, step)
        indices = list(range(n_frames))[s]
        frame_indices_to_process = indices

        # Derive times without reading every frame (assumes evenly spaced frames)
        times_to_process: List[float] = []
        if frame_indices_to_process:
            dt = getattr(traj, "dt", None)
            first_idx = frame_indices_to_process[0]
            try:
                traj[first_idx]  # single seek to get starting time
                t0 = traj.ts.time
            except Exception:
                dt = None
                t0 = None

            if dt is not None and t0 is not None:
                times_to_process = [t0 + (idx - first_idx) * dt for idx in frame_indices_to_process]

        if len(frame_indices_to_process) == 0:
            return
        
        # Notify observers that iteration is starting
        for observer in self.observers:
            observer.on_frame_start(self)
        
        # Split frames into batches for parallel processing
        # Improved batch size calculation accounting for Universe creation overhead
        n_frames = len(frame_indices_to_process)
        
        # Estimate Universe creation overhead and calculate optimal batch size
        # Target: minimize Universe recreations while keeping batches balanced
        # For small trajectories, use larger minimum batch size
        if n_frames < 1000:
            min_batch_size = max(100, n_frames // max(2, n_jobs))
        else:
            # For larger trajectories, ensure each worker gets substantial work
            # to amortize Universe creation overhead
            min_batch_size = max(200, n_frames // (n_jobs * 2))
        
        # Calculate batch size: ensure we don't create too many small batches
        batch_size = max(min_batch_size, n_frames // n_jobs)
        
        batches = []
        
        for i in range(0, n_frames, batch_size):
            batch_end = min(i + batch_size, n_frames)
            batches.append((i, batch_end, frame_indices_to_process[i:batch_end]))
        
        # Process batches in parallel
        try:
            with mp.Pool(processes=n_jobs) as pool:
                results = pool.starmap(
                    _process_frame_batch,
                    [(top_file, traj_file, traj_format, batch_frames, batch_start_idx, 
                      self.observers, preloaded_coords) for batch_start_idx, batch_end_idx, batch_frames in batches]
                )
        except Exception as e:
            warnings.warn(
                f"Parallel processing failed: {e}. Falling back to sequential processing."
            )
            self._iterate_sequential(start, stop, step, preload_coordinates)
            return
        
        # Merge results from all batches
        self.frame_indices = frame_indices_to_process
        self.times = times_to_process
        
        # Merge observer results from parallel workers
        for batch_results in results:
            if batch_results is None:
                continue
            for observer_idx, observer_result in enumerate(batch_results):
                if observer_idx < len(self.observers) and observer_result is not None:
                    try:
                        self.observers[observer_idx].merge_results(observer_result)
                    except Exception as e:
                        warnings.warn(
                            f"Failed to merge results for observer "
                            f"{type(self.observers[observer_idx]).__name__}: {e}"
                        )
        
        # Notify observers that iteration is complete
        for observer in self.observers:
            observer.on_frame_end(self)
    
    def _get_trajectory_format(self) -> Optional[str]:
        """Extract trajectory format from Universe."""
        traj_format = None
        if hasattr(self.universe.trajectory, 'format'):
            traj_format = self.universe.trajectory.format[0]
        elif hasattr(self.universe.trajectory, '__class__'):
            # Try to infer from class name (e.g., MDCRDReader -> MDCRD)
            class_name = self.universe.trajectory.__class__.__name__
            # Remove 'Reader' suffix if present
            if class_name.endswith('Reader'):
                traj_format = class_name[:-6]
        
        # Handle empty string format
        if traj_format == '':
            traj_format = None
        
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


def _process_frame_batch(top_file: str, traj_file: str, traj_format: Optional[str],
                         frame_indices: List[int], batch_start_idx: int, 
                         observers: List[FrameObserver],
                         preloaded_coords: Optional[Dict[int, Any]] = None) -> List[Any]:
    """
    Process a batch of frames in a worker process.
    
    Args:
        top_file: Topology file path
        traj_file: Trajectory file path
        traj_format: Trajectory format (e.g., 'MDCRD', 'XTC', 'TRR', etc.)
        frame_indices: List of frame indices to process
        batch_start_idx: Starting index for this batch (for frame_idx parameter)
        observers: List of observer instances (will be pickled and recreated)
        preloaded_coords: Optional dictionary of pre-loaded coordinates (frame_idx -> Timestep)
    
    Returns:
        List of observer results (one per observer)
    """
    try:
        # Try to get cached Universe first (reuse across batches in same worker)
        u = _get_cached_universe(top_file, traj_file, traj_format)
        
        if u is None:
            # Create new Universe in worker process with explicit format
            # If format is provided, use it; otherwise let MDAnalysis try to auto-detect
            try:
                if traj_format:
                    u = mda.Universe(top_file, traj_file, format=traj_format)
                else:
                    u = mda.Universe(top_file, traj_file)
            except (ValueError, OSError) as e:
                # If format detection fails, try common formats for the file
                # This is a fallback for cases where format wasn't properly detected
                if 'format' in str(e).lower() or 'reader' in str(e).lower():
                    # Try to infer format from filename extension or common patterns
                    traj_lower = str(traj_file).lower()
                    if 'mdcrd' in traj_lower or traj_lower.endswith('.mdcrd'):
                        u = mda.Universe(top_file, traj_file, format='MDCRD')
                    elif traj_lower.endswith('.xtc'):
                        u = mda.Universe(top_file, traj_file, format='XTC')
                    elif traj_lower.endswith('.trr'):
                        u = mda.Universe(top_file, traj_file, format='TRR')
                    elif traj_lower.endswith('.dcd'):
                        u = mda.Universe(top_file, traj_file, format='DCD')
                    else:
                        # Re-raise the original error if we can't infer format
                        raise
                else:
                    raise
            
            # Cache the Universe for reuse in subsequent batches
            _cache_universe(u, top_file, traj_file)
        
        # Recreate observers in worker process (they need to be pickle-able)
        worker_observers = []
        for observer in observers:
            try:
                # Try to pickle and unpickle the observer
                pickled = pickle.dumps(observer)
                worker_observer = pickle.loads(pickled)
                worker_observers.append(worker_observer)
            except Exception as e:
                warnings.warn(
                    f"Observer {type(observer).__name__} is not pickle-able, "
                    f"skipping parallel processing: {e}"
                )
                worker_observers.append(None)
        
        # Note: on_frame_start and on_frame_end are called in the main process,
        # not in worker processes. Workers only process frames.
        
        # Process each frame in the batch
        for local_idx, frame_num in enumerate(frame_indices):
            global_idx = batch_start_idx + local_idx
            
            # Use preloaded coordinates if available, otherwise seek to frame
            if preloaded_coords is not None and frame_num in preloaded_coords:
                ts = preloaded_coords[frame_num]
                # Still need to position Universe for observers that use it
                u.trajectory[frame_num]
            else:
                u.trajectory[frame_num]  # Seek to frame
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
        
        # Return observer instances (they contain the results)
        return worker_observers
        
    except Exception as e:
        warnings.warn(f"Error in worker process: {e}")
        return [None] * len(observers)

