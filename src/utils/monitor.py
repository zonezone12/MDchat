import psutil
import os
import datetime
import warnings
import MDAnalysis as mda

def log_memory(label=""):
    """Log current memory usage (RSS and VMS) if psutil is available."""
    if psutil is None:
        return
    try:
        process = psutil.Process(os.getpid())
        rss_gb = process.memory_info().rss / 1e9
        vms_gb = process.memory_info().vms / 1e9
        print(f"[{datetime.datetime.now()}] MEM {label:30s} → RSS: {rss_gb:7.1f} GB | VMS: {vms_gb:7.1f} GB", flush=True)
    except Exception as e:
        warnings.warn(f"Failed to log memory: {e}")
        


def estimate_memory_per_worker(universe: mda.Universe, n_atoms: int = None) -> float:
    """
    Estimate memory usage per worker process in GB.
    
    Each worker loads the trajectory, so we estimate based on:
    - Universe overhead: ~0.5 GB
    - Trajectory data: depends on number of atoms and frames
    - Observer overhead: ~0.1-0.3 GB per observer
    
    Args:
        universe: MDAnalysis Universe
        n_atoms: Number of atoms (if None, uses universe.atoms.n_atoms)
    
    Returns:
        Estimated memory per worker in GB
    """
    if n_atoms is None:
        n_atoms = universe.atoms.n_atoms
    
    # Rough estimate: 0.5 GB base + 0.1 GB per 10k atoms
    base_memory_gb = 0.5
    atom_memory_gb = (n_atoms / 10000) * 0.1
    observer_overhead_gb = 0.3  # For endpoint + volume observers
    
    return base_memory_gb + atom_memory_gb + observer_overhead_gb


def auto_limit_workers(requested_n_jobs: int, universe: mda.Universe, 
                       available_memory_gb: float = None) -> int:
    """
    Automatically limit number of workers based on available memory.
    
    Args:
        requested_n_jobs: Requested number of workers (-1 means all cores)
        universe: MDAnalysis Universe
        available_memory_gb: Available memory in GB (if None, tries to detect)
    
    Returns:
        Limited number of workers
    """
    if requested_n_jobs == 1:
        return 1  # Sequential processing, no limit needed
    
    # Get available memory
    if available_memory_gb is None and psutil is not None:
        try:
            mem = psutil.virtual_memory()
            # Use 80% of available memory to leave some headroom
            available_memory_gb = (mem.available / 1e9) * 0.8
        except Exception:
            available_memory_gb = None
    
    if available_memory_gb is None:
        # Can't detect memory, use conservative default
        warnings.warn(
            "Cannot detect available memory. Using conservative limit of 4 workers. "
            "Set --n_jobs explicitly or increase SLURM --mem allocation."
        )
        return min(4, requested_n_jobs if requested_n_jobs > 0 else 4)
    
    # Estimate memory per worker
    memory_per_worker = estimate_memory_per_worker(universe)
    
    # Calculate max workers based on available memory
    # Leave 2 GB headroom for main process
    max_workers_by_memory = max(1, int((available_memory_gb - 2.0) / memory_per_worker))
    
    # Get CPU count if requested all cores
    if requested_n_jobs == -1:
        try:
            cpu_count = os.cpu_count() or 2
        except Exception:
            cpu_count = 2
        requested_n_jobs = cpu_count
    
    # Limit to minimum of requested and memory-limited
    limited_n_jobs = min(requested_n_jobs, max_workers_by_memory)
    
    if limited_n_jobs < requested_n_jobs:
        warnings.warn(
            f"Limiting workers from {requested_n_jobs} to {limited_n_jobs} based on "
            f"available memory ({available_memory_gb:.1f} GB). "
            f"Estimated {memory_per_worker:.2f} GB per worker. "
            f"To use more workers, increase SLURM --mem allocation or set --n_jobs explicitly."
        )
    
    return limited_n_jobs