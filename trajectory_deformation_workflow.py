#!/usr/bin/env python3
"""
Trajectory Deformation Analysis & Auto-Selection Pipeline
--------------------------------------------------------

This module provides classes for trajectory analysis organized by functionality.
All utility functions have been organized into appropriate classes for better management.

Dependencies (install with pip)
===============================
- MDAnalysis>=2.5.0
- numpy, pandas, scikit-learn
- ruptures (for change-point detection)
- hdbscan (optional; will fallback to k-means)
- rdkit (required for endpoint-based analysis)
- endpoints_finder module (required for endpoint-based analysis)
"""
import os
import sys
import warnings
from typing import Optional, Tuple, List, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

# Plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    warnings.warn("matplotlib not available. Plotting will be disabled.")

# Third-party scientific packages
try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms, pca as mda_pca
except ImportError as e:
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise
from MDAnalysis.analysis import gnm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from rdkit.Chem import Draw
# Optional packages
try:
    import hdbscan  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False

try:
    import ruptures as rpt
    HAS_RUPTURES = True
except Exception:
    HAS_RUPTURES = False


# Import EndpointsFinder
try:
    # Try new location first (in src/EndpointAnalyzer)
    from src.EndpointAnalyzer import EndpointsFinder
except ImportError:
    try:
        # Fallback to absolute import (works when imported from outside the package)
        from MD_analysis.endpoints_finder import EndpointsFinder
    except ImportError:
        # Fallback to relative import (works when run from within the directory)
        from endpoints_finder import EndpointsFinder

# Import VolumeAnalyzer for volume computation
try:
    # Try absolute import first (works when imported from outside the package)
    from MD_analysis.volume_analyser import VolumeAnalyzer
except ImportError:
    # Fallback to relative import (works when run from within the directory)
    from volume_analyser import VolumeAnalyzer

# Import GSAnalyzer from separate module
try:
    # Try new location first (in src/task)
    from src.task import GSAnalyzer
except ImportError:
    try:
        # Fallback to absolute import (works when imported from outside the package)
        from MD_analysis.gs_analyzer import GSAnalyzer
    except ImportError:
        try:
            # Fallback to relative import (works when run from within the directory)
            from gs_analyzer import GSAnalyzer
        except ImportError:
            warnings.warn("gs_analyzer module not found. GSAnalyzer will not be available.")
            GSAnalyzer = None  # type: ignore


# ============================================================================
# Class: AlignedTrajectory
# Purpose: Store and manage aligned MDAnalysis Universe
# ============================================================================

class AlignedTrajectory:
    """Store and manage an aligned MDAnalysis Universe.
    
    By default, aligns the trajectory to the first frame (trajectory[0]) of the main structure.
    Can also align first and last frames for comparison.
    """
    
    def __init__(self, 
                 universe: mda.Universe,
                 align_sel: Optional[str] = None,
                 ref_frame: int = 0,
                 align_first_and_last: bool = True,
                 in_memory: bool = True):
        """
        Initialize AlignedTrajectory with a universe and perform alignment.
        
        By default, aligns the trajectory to the first frame (trajectory[0]) and 
        creates separate aligned universes for first and last frames.
        
        Args:
            universe: MDAnalysis Universe to align
            align_sel: Selection string for alignment (default: "not water and not name I and not name Na+")
            ref_frame: Reference frame index for alignment (default: 0, first frame)
            align_first_and_last: If True, also create separate universes for first 
                                 and last frames aligned to each other (default: True)
            in_memory: Whether to align in memory (default: True)
        """
        self.original_universe = universe
        self.align_sel = align_sel if align_sel is not None else "not water and not name I and not name Na+"
        self.ref_frame = ref_frame
        self.align_first_and_last = align_first_and_last
        self.in_memory = in_memory
        self.aligned_universe: Optional[mda.Universe] = None
        self.first_frame_universe: Optional[mda.Universe] = None
        self.last_frame_universe: Optional[mda.Universe] = None
        self._alignment_performed = False
        
        # Perform alignment
        self.align()
    
    def align(self):
        """Perform trajectory alignment."""
        if self._alignment_performed and self.aligned_universe is not None:
            return  # Already aligned
        
        try:
            # Create a copy of the universe for alignment
            # Use the same topology and trajectory files
            if hasattr(self.original_universe, 'filename') and hasattr(self.original_universe.trajectory, 'filename'):
                # If we have file paths, create new universe from files
                self.aligned_universe = mda.Universe(
                    self.original_universe.filename,
                    self.original_universe.trajectory.filename
                )
            else:
                # Otherwise, work with the universe in-place (will modify original)
                # For safety, we'll create a copy by writing and reloading if possible
                warnings.warn("Cannot create independent copy. Alignment will modify original universe.")
                self.aligned_universe = self.original_universe
            
            # Perform alignment: align all frames to reference frame (default: frame 0)
            align.AlignTraj(
                self.aligned_universe, 
                self.aligned_universe,
                select=self.align_sel, 
                ref_frame=self.ref_frame,
                in_memory=self.in_memory
            ).run()
            
            # If align_first_and_last is True, also create separate universes for first and last frames
            if self.align_first_and_last:
                # First frame universe (aligned to itself)
                if hasattr(self.original_universe, 'filename') and hasattr(self.original_universe.trajectory, 'filename'):
                    self.first_frame_universe = mda.Universe(
                        self.original_universe.filename,
                        self.original_universe.trajectory.filename
                    )
                    self.first_frame_universe.trajectory[0]
                    align.AlignTraj(
                        self.first_frame_universe,
                        self.first_frame_universe,
                        select=self.align_sel,
                        ref_frame=0,
                        in_memory=self.in_memory
                    ).run()
                
                # Last frame universe (aligned to first frame)
                if hasattr(self.original_universe, 'filename') and hasattr(self.original_universe.trajectory, 'filename'):
                    self.last_frame_universe = mda.Universe(
                        self.original_universe.filename,
                        self.original_universe.trajectory.filename
                    )
                    # Align last frame universe to first frame universe
                    self.last_frame_universe.trajectory[-1]
                    if self.first_frame_universe is not None:
                        align.AlignTraj(
                            self.last_frame_universe,
                            self.first_frame_universe,
                            select=self.align_sel,
                            ref_frame=0,
                            in_memory=self.in_memory
                        ).run()
            
            self._alignment_performed = True
        except Exception as e:
            warnings.warn(f"Alignment failed: {e}. Using original universe.")
            self.aligned_universe = self.original_universe
            self._alignment_performed = False
    
    def get_aligned_universe(self) -> mda.Universe:
        """Get the aligned universe."""
        if self.aligned_universe is None:
            self.align()
        return self.aligned_universe if self.aligned_universe is not None else self.original_universe
    
    def get_first_frame_universe(self) -> Optional[mda.Universe]:
        """Get the universe with first frame aligned (only if align_first_and_last=True)."""
        return self.first_frame_universe
    
    def get_last_frame_universe(self) -> Optional[mda.Universe]:
        """Get the universe with last frame aligned to first (only if align_first_and_last=True)."""
        return self.last_frame_universe
    
    def realign(self, align_sel: Optional[str] = None, ref_frame: Optional[int] = None):
        """Realign the trajectory with new parameters."""
        if align_sel is not None:
            self.align_sel = align_sel
        if ref_frame is not None:
            self.ref_frame = ref_frame
        
        self._alignment_performed = False
        self.aligned_universe = None
        self.first_frame_universe = None
        self.last_frame_universe = None
        self.align()
    
    def __getattr__(self, name):
        """Delegate attribute access to aligned_universe for convenience."""
        if self.aligned_universe is None:
            self.align()
        if self.aligned_universe is not None:
            return getattr(self.aligned_universe, name)
        return getattr(self.original_universe, name)


# ============================================================================
# Class: MetricRegistry
# Purpose: Container for metric definitions allowing dynamic metric selection
# ============================================================================

class MetricRegistry:
    """Registry for metric definitions that can be dynamically selected."""
    
    def __init__(self):
        """Initialize the metric registry."""
        self._metrics: dict = {}
        self._default_metrics: List[str] = []
    
    def register(self, name: str, metric_func, default: bool = False):
        """
        Register a metric function.
        
        Args:
            name: Name identifier for the metric
            metric_func: Function that computes the metric. Should accept:
                - universe: mda.Universe
                - frame_data: dict with frame-specific data
                - config: dict with metric configuration
                Returns: tuple (frame_result, needs_postprocessing)
            default: Whether this metric should be included by default
        """
        self._metrics[name] = metric_func
        if default:
            if name not in self._default_metrics:
                self._default_metrics.append(name)
    
    def unregister(self, name: str):
        """Unregister a metric."""
        if name in self._metrics:
            del self._metrics[name]
        if name in self._default_metrics:
            self._default_metrics.remove(name)
    
    def get(self, name: str):
        """Get a metric function by name."""
        return self._metrics.get(name)
    
    def list_metrics(self) -> List[str]:
        """List all registered metric names."""
        return list(self._metrics.keys())
    
    def get_default_metrics(self) -> List[str]:
        """Get list of default metric names."""
        return self._default_metrics.copy()
    
    def has_metric(self, name: str) -> bool:
        """Check if a metric is registered."""
        return name in self._metrics


# ============================================================================
# Class: FrameProcessor
# Purpose: Process frames and compute metrics on-the-fly (memory-efficient)
# ============================================================================

class FrameProcessor:
    """Process frames in a single iteration, computing metrics on-the-fly.
    
    This class is memory-efficient as it only stores final results, not all
    coordinates. Use this for large trajectories where memory is a concern.
    
    Metrics can be selected dynamically using the metric registry system.
    """
    
    def __init__(self, u: mda.Universe, metric_registry: Optional[MetricRegistry] = None):
        """
        Initialize FrameProcessor.
        
        Args:
            u: MDAnalysis Universe to process
            metric_registry: Optional MetricRegistry instance. If None, creates a default one.
        """
        self.universe = u
        self.n_frames = len(u.trajectory)
        self.results: dict = {}
        self.frame_indices: np.ndarray = np.array([])
        self.times: np.ndarray = np.array([])
        self._processed = False
        
        # Initialize metric registry
        if metric_registry is None:
            self.metric_registry = MetricRegistry()
            self._register_builtin_metrics()
        else:
            self.metric_registry = metric_registry
    
    def _register_builtin_metrics(self):
        """Register built-in metrics with the registry."""
        # Register each metric with its computation function (using lambda to bind self)
        self.metric_registry.register('rmsd', lambda u, fd, cfg: self._compute_rmsd_metric(u, fd, cfg), default=True)
        self.metric_registry.register('rmsf', lambda u, fd, cfg: self._compute_rmsf_metric(u, fd, cfg), default=True)
        self.metric_registry.register('rg', lambda u, fd, cfg: self._compute_rg_metric(u, fd, cfg), default=True)
        self.metric_registry.register('pca', lambda u, fd, cfg: self._compute_pca_metric(u, fd, cfg), default=True)
        self.metric_registry.register('strain', lambda u, fd, cfg: self._compute_strain_metric(u, fd, cfg), default=True)
        self.metric_registry.register('contacts', lambda u, fd, cfg: self._compute_contacts_metric(u, fd, cfg), default=False)
    
    def _compute_rmsd_metric(self, universe, frame_data, config):
        """Compute RMSD for a single frame."""
        atoms = universe.select_atoms(config['sel'])
        ref_pos = config['ref_pos']
        R, rmsd_val = align.rotation_matrix(atoms.positions, ref_pos)
        return rmsd_val, False  # False = no post-processing needed
    
    def _compute_rmsf_metric(self, universe, frame_data, config):
        """Store coordinates for RMSF computation (needs post-processing)."""
        atoms = universe.select_atoms(config['sel'])
        return atoms.positions.copy(), True  # True = needs post-processing
    
    def _compute_rg_metric(self, universe, frame_data, config):
        """Compute radius of gyration for a single frame."""
        atoms = universe.select_atoms(config['sel'])
        com = atoms.center_of_mass()
        rg2 = ((atoms.positions - com) ** 2).sum(axis=1).mean()
        return np.sqrt(rg2), False
    
    def _compute_pca_metric(self, universe, frame_data, config):
        """Store coordinates for PCA computation (needs post-processing)."""
        atoms = universe.select_atoms(config['sel'])
        return atoms.positions.copy().reshape(-1), True
    
    def _compute_strain_metric(self, universe, frame_data, config):
        """Store coordinates for strain computation (needs post-processing)."""
        atoms = universe.select_atoms(config['sel'])
        return atoms.positions.copy(), True
    
    def _compute_contacts_metric(self, universe, frame_data, config):
        """Compute contact distance for a single frame."""
        selA = config.get('selA')
        selB = config.get('selB')
        if selA and selB:
            atomsA = universe.select_atoms(selA)
            atomsB = universe.select_atoms(selB)
            if len(atomsA) > 0 and len(atomsB) > 0:
                da = atomsA.positions[:, None, :]
                db = atomsB.positions[None, :, :]
                diff = da - db
                dd = np.sqrt((diff * diff).sum(axis=2))
                return dd.min(), False
        return None, False
    
    def process_metrics(self,
                       metrics_to_compute: Optional[List[str]] = None,
                       metric_configs: Optional[dict] = None,
                       **kwargs) -> dict:
        """
        Compute selected metrics in a single iteration through the trajectory.
        
        This method allows you to dynamically choose which metrics to compute,
        rather than computing all metrics at once.
        
        Args:
            metrics_to_compute: List of metric names to compute. If None, uses default metrics.
                               Available metrics: 'rmsd', 'rmsf', 'rg', 'pca', 'strain', 'contacts'
            metric_configs: Dictionary mapping metric names to their configurations.
                          Each config should contain the necessary parameters:
                          - 'rmsd': {'sel': str, 'ref_frame': int}
                          - 'rmsf': {'sel': str}
                          - 'rg': {'sel': str}
                          - 'pca': {'sel': str, 'n_components': int}
                          - 'strain': {'sel': str, 'window': int, 'lag': int}
                          - 'contacts': {'selA': str, 'selB': str}
            **kwargs: Additional parameters for backward compatibility with process_all_metrics
        
        Returns:
            Dictionary with computed metrics
        
        Examples:
            >>> # Compute only RMSD and radius of gyration
            >>> processor = FrameProcessor(universe)
            >>> results = processor.process_metrics(
            ...     metrics_to_compute=['rmsd', 'rg'],
            ...     metric_configs={
            ...         'rmsd': {'sel': 'protein', 'ref_frame': 0},
            ...         'rg': {'sel': 'protein'}
            ...     }
            ... )
            
            >>> # Compute default metrics with custom selections
            >>> results = processor.process_metrics(
            ...     metric_configs={
            ...         'rmsd': {'sel': 'backbone', 'ref_frame': 0},
            ...         'rg': {'sel': 'protein'}
            ...     }
            ... )
            
            >>> # List available metrics
            >>> print(processor.list_available_metrics())
            ['rmsd', 'rmsf', 'rg', 'pca', 'strain', 'contacts']
        """
        if self._processed:
            return self.results
        
        # Handle backward compatibility: if old-style parameters are provided, use them
        if metrics_to_compute is None and metric_configs is None:
            # Check if old-style parameters are provided
            if any(k in kwargs for k in ['rmsd_sel', 'rmsf_sel', 'rg_sel', 'pca_sel']):
                return self.process_all_metrics(**kwargs)
            # Otherwise use default metrics
            metrics_to_compute = self.metric_registry.get_default_metrics()
        
        if metrics_to_compute is None:
            metrics_to_compute = self.metric_registry.get_default_metrics()
        
        if metric_configs is None:
            metric_configs = {}
        
        # Validate requested metrics
        for metric_name in metrics_to_compute:
            if not self.metric_registry.has_metric(metric_name):
                raise ValueError(f"Metric '{metric_name}' is not registered. "
                               f"Available metrics: {self.metric_registry.list_metrics()}")
        
        # Prepare metric configurations
        configs = {}
        frame_data_storage = {}  # For metrics that need post-processing
        
        # Initialize frame tracking
        self.frame_indices = np.zeros(self.n_frames, dtype=int)
        self.times = np.zeros(self.n_frames)
        
        # Prepare configurations for each metric
        for metric_name in metrics_to_compute:
            config = metric_configs.get(metric_name, {}).copy()
            
            if metric_name == 'rmsd':
                sel = config.get('sel', kwargs.get('rmsd_sel'))
                ref_frame = config.get('ref_frame', kwargs.get('rmsd_ref_frame', 0))
                if sel is None:
                    raise ValueError("rmsd requires 'sel' in metric_configs or rmsd_sel parameter")
                atoms = self.universe.select_atoms(sel)
                self.universe.trajectory[ref_frame]
                config['sel'] = sel
                config['ref_pos'] = atoms.positions.copy()
                configs[metric_name] = config
                frame_data_storage[metric_name] = []
            
            elif metric_name == 'rmsf':
                sel = config.get('sel', kwargs.get('rmsf_sel'))
                if sel is None:
                    raise ValueError("rmsf requires 'sel' in metric_configs or rmsf_sel parameter")
                config['sel'] = sel
                configs[metric_name] = config
                frame_data_storage[metric_name] = []
            
            elif metric_name == 'rg':
                sel = config.get('sel', kwargs.get('rg_sel'))
                if sel is None:
                    raise ValueError("rg requires 'sel' in metric_configs or rg_sel parameter")
                config['sel'] = sel
                configs[metric_name] = config
                frame_data_storage[metric_name] = []
            
            elif metric_name == 'pca':
                sel = config.get('sel', kwargs.get('pca_sel'))
                n_components = config.get('n_components', kwargs.get('pca_n_components', 5))
                if sel is None:
                    raise ValueError("pca requires 'sel' in metric_configs or pca_sel parameter")
                config['sel'] = sel
                config['n_components'] = n_components
                configs[metric_name] = config
                frame_data_storage[metric_name] = []
            
            elif metric_name == 'strain':
                sel = config.get('sel', kwargs.get('strain_sel', kwargs.get('pca_sel')))
                window = config.get('window', kwargs.get('strain_window', 10))
                lag = config.get('lag', kwargs.get('strain_lag', 1))
                if sel is None:
                    raise ValueError("strain requires 'sel' in metric_configs or strain_sel/pca_sel parameter")
                config['sel'] = sel
                config['window'] = window
                config['lag'] = lag
                configs[metric_name] = config
                frame_data_storage[metric_name] = []
            
            elif metric_name == 'contacts':
                selA = config.get('selA', kwargs.get('contact_selA'))
                selB = config.get('selB', kwargs.get('contact_selB'))
                if selA is None or selB is None:
                    warnings.warn("contacts metric requires both selA and selB. Skipping.")
                    # Don't add to configs, so it won't be computed
                    continue
                config['selA'] = selA
                config['selB'] = selB
                configs[metric_name] = config
                frame_data_storage[metric_name] = []
        
        # SINGLE ITERATION - compute selected metrics on-the-fly
        for frame_idx, ts in enumerate(self.universe.trajectory):
            self.frame_indices[frame_idx] = ts.frame
            self.times[frame_idx] = ts.time
            
            # Compute each requested metric
            for metric_name in metrics_to_compute:
                if metric_name not in configs:
                    continue
                
                metric_func = self.metric_registry.get(metric_name)
                if metric_func is None:
                    continue
                
                try:
                    result, needs_postprocessing = metric_func(self.universe, {}, configs[metric_name])
                    if needs_postprocessing:
                        frame_data_storage[metric_name].append(result)
                    else:
                        if metric_name not in frame_data_storage:
                            frame_data_storage[metric_name] = []
                        frame_data_storage[metric_name].append(result)
                except Exception as e:
                    warnings.warn(f"Error computing {metric_name} at frame {frame_idx}: {e}")
                    if metric_name not in frame_data_storage:
                        frame_data_storage[metric_name] = []
                    frame_data_storage[metric_name].append(np.nan)
        
        # Post-process metrics that need it
        self.results = {}
        
        for metric_name in metrics_to_compute:
            if metric_name not in frame_data_storage or len(frame_data_storage[metric_name]) == 0:
                continue
            
            if metric_name == 'rmsd':
                self.results['rmsd'] = np.array(frame_data_storage[metric_name])
            
            elif metric_name == 'rmsf':
                rmsf_coords = np.array(frame_data_storage[metric_name])  # (T, n, 3)
                mean = rmsf_coords.mean(axis=0)
                diffsq = (rmsf_coords - mean) ** 2
                rmsf_vals = np.sqrt(diffsq.sum(axis=2).mean(axis=0))
                self.results['rmsf'] = rmsf_vals
                del rmsf_coords
            
            elif metric_name == 'rg':
                self.results['rg'] = np.array(frame_data_storage[metric_name])
            
            elif metric_name == 'pca':
                pca_coords = np.array(frame_data_storage[metric_name])  # (T, 3N)
                Xc = pca_coords - pca_coords.mean(axis=0)
                n_components = configs[metric_name]['n_components']
                pca = PCA(n_components=n_components, svd_solver="auto")
                pcs = pca.fit_transform(Xc)
                self.results['pcs'] = pcs
                self.results['pca_model'] = pca
                del pca_coords
            
            elif metric_name == 'strain':
                strain_coords = np.array(frame_data_storage[metric_name])  # (T, n, 3)
                T, n, _ = strain_coords.shape
                strain = np.full(T, np.nan)
                window = configs[metric_name]['window']
                lag = configs[metric_name]['lag']
                for t in range(0, T - lag):
                    if t < window:
                        continue
                    A = strain_coords[t - window:t, :, :].reshape(-1, 3)
                    B = strain_coords[t - window + lag:t + lag, :, :].reshape(-1, 3)
                    A_aug = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
                    Xsol, *_ = np.linalg.lstsq(A_aug, B, rcond=None)
                    F = Xsol[:3, :]
                    C = F.T @ F
                    E = 0.5 * (C - np.eye(3))
                    strain[t] = np.linalg.norm(E, ord='fro')
                self.results['strain'] = strain
                del strain_coords
            
            elif metric_name == 'contacts':
                contact_vals = frame_data_storage[metric_name]
                valid_vals = [v for v in contact_vals if v is not None]
                if valid_vals:
                    self.results['contacts'] = np.array(valid_vals)
                else:
                    self.results['contacts'] = None
        
        self._processed = True
        return self.results
    
    def process_all_metrics(self,
                          rmsd_sel: str,
                          rmsf_sel: str,
                          rg_sel: str,
                          pca_sel: str,
                          contact_selA: Optional[str] = None,
                          contact_selB: Optional[str] = None,
                          strain_sel: Optional[str] = None,
                          strain_window: int = 10,
                          strain_lag: int = 1,
                          pca_n_components: int = 5,
                          rmsd_ref_frame: int = 0) -> dict:
        """
        Compute all metrics in a single iteration through the trajectory.
        
        This method is maintained for backward compatibility. It uses the new
        metric registry system internally.
        
        Args:
            rmsd_sel: Selection for RMSD computation
            rmsf_sel: Selection for RMSF computation
            rg_sel: Selection for radius of gyration
            pca_sel: Selection for PCA
            contact_selA: First selection for contact distances (optional)
            contact_selB: Second selection for contact distances (optional)
            strain_sel: Selection for strain computation (optional, defaults to pca_sel)
            strain_window: Window size for strain
            strain_lag: Lag for strain
            pca_n_components: Number of PCA components
            rmsd_ref_frame: Reference frame for RMSD
            
        Returns:
            Dictionary with computed metrics
        """
        # Determine which metrics to compute
        metrics_to_compute = ['rmsd', 'rmsf', 'rg', 'pca', 'strain']
        if contact_selA and contact_selB:
            metrics_to_compute.append('contacts')
        
        # Prepare metric configurations
        metric_configs = {
            'rmsd': {'sel': rmsd_sel, 'ref_frame': rmsd_ref_frame},
            'rmsf': {'sel': rmsf_sel},
            'rg': {'sel': rg_sel},
            'pca': {'sel': pca_sel, 'n_components': pca_n_components},
            'strain': {
                'sel': strain_sel if strain_sel is not None else pca_sel,
                'window': strain_window,
                'lag': strain_lag
            }
        }
        
        if contact_selA and contact_selB:
            metric_configs['contacts'] = {'selA': contact_selA, 'selB': contact_selB}
        
        # Use the new process_metrics method
        return self.process_metrics(
            metrics_to_compute=metrics_to_compute,
            metric_configs=metric_configs
        )
    
    def get_frame_indices(self) -> np.ndarray:
        """Get frame indices for all processed frames."""
        return self.frame_indices
    
    def get_times(self) -> np.ndarray:
        """Get time values for all processed frames."""
        return self.times
    
    def get_n_frames(self) -> int:
        """Get number of frames."""
        return self.n_frames
    
    def list_available_metrics(self) -> List[str]:
        """List all available metric names."""
        return self.metric_registry.list_metrics()
    
    def register_custom_metric(self, name: str, metric_func, default: bool = False):
        """
        Register a custom metric function.
        
        Args:
            name: Name identifier for the metric
            metric_func: Function that computes the metric. Should accept:
                - universe: mda.Universe
                - frame_data: dict with frame-specific data (currently unused, for future use)
                - config: dict with metric configuration
                Returns: tuple (frame_result, needs_postprocessing)
                where needs_postprocessing is True if the result needs post-processing
            default: Whether this metric should be included by default
        """
        self.metric_registry.register(name, metric_func, default=default)
    
    def reset_processing(self):
        """Reset the processed state to allow reprocessing with different metrics."""
        self._processed = False
        self.results = {}
        self.frame_indices = np.array([])
        self.times = np.array([])


# ============================================================================
# Class: FrameGatherer
# Purpose: Gather frame data in a single iteration through the trajectory
# ============================================================================

class FrameGatherer:
    """Gather frame data from trajectory in a single iteration.
    
    This class iterates through all frames once and collects coordinates
    for multiple selections, allowing other functions to use pre-gathered
    data instead of iterating multiple times.
    
    Example:
        >>> # Instead of each function iterating separately:
        >>> # metrics.compute_rmsd(u, sel)  # iterates frames
        >>> # metrics.compute_rmsf(u, sel)   # iterates frames again
        >>> # metrics.radius_of_gyration(u, sel)  # iterates frames again
        >>> 
        >>> # Use FrameGatherer to iterate once:
        >>> gatherer = FrameGatherer(u, [sel_str1, sel_str2])
        >>> rmsd = metrics.compute_rmsd(u, sel_str1, gatherer=gatherer)
        >>> rmsf = metrics.compute_rmsf(u, sel_str1, gatherer=gatherer)
        >>> rg = metrics.radius_of_gyration(u, sel_str1, gatherer=gatherer)
    """
    
    @staticmethod
    def estimate_memory_usage(u: mda.Universe, selection_strings: List[str], 
                             bytes_per_float: int = 8) -> float:
        """
        Estimate memory usage in GB for gathering coordinates.
        
        Args:
            u: MDAnalysis Universe
            selection_strings: List of selection strings
            bytes_per_float: Bytes per float (default: 8 for float64)
            
        Returns:
            Estimated memory usage in GB
        """
        n_frames = len(u.trajectory)
        total_atoms = 0
        
        for sel_str in selection_strings:
            try:
                sel = u.select_atoms(sel_str)
                total_atoms += len(sel)
            except Exception:
                # If selection fails, estimate based on universe size
                total_atoms += len(u.atoms) // len(selection_strings)
        
        # Memory = n_frames × n_atoms × 3 coordinates × bytes_per_float
        memory_bytes = n_frames * total_atoms * 3 * bytes_per_float
        memory_gb = memory_bytes / (1024 ** 3)
        
        return memory_gb
    
    @staticmethod
    def should_use_memory_efficient(u: mda.Universe, selection_strings: List[str],
                                    memory_limit_gb: float = 2.0) -> bool:
        """
        Determine if memory-efficient (on-the-fly) processing should be used.
        
        Args:
            u: MDAnalysis Universe
            selection_strings: List of selection strings
            memory_limit_gb: Memory limit in GB (default: 2.0)
            
        Returns:
            True if memory-efficient mode should be used
        """
        estimated_memory = FrameGatherer.estimate_memory_usage(u, selection_strings)
        return estimated_memory > memory_limit_gb
    
    def __init__(self, u: mda.Universe, selection_strings: List[str],
                 force_memory_efficient: Optional[bool] = None,
                 memory_limit_gb: float = 2.0):
        """
        Initialize FrameGatherer and gather data from trajectory.
        
        Args:
            u: MDAnalysis Universe to gather data from
            selection_strings: List of selection strings to gather coordinates for
            force_memory_efficient: If True, use memory-efficient mode. If None, auto-detect.
            memory_limit_gb: Memory limit in GB for auto-detection (default: 2.0)
        """
        self.universe = u
        self.selection_strings = selection_strings
        self.n_frames = len(u.trajectory)
        self.memory_efficient = False
        
        # Auto-detect memory-efficient mode if not forced
        if force_memory_efficient is None:
            self.memory_efficient = FrameGatherer.should_use_memory_efficient(
                u, selection_strings, memory_limit_gb
            )
        else:
            self.memory_efficient = force_memory_efficient
        
        if self.memory_efficient:
            # Memory-efficient mode: don't store coordinates
            self.coordinates = {}  # Empty - will compute on-the-fly
            self.frame_indices = np.array([])
            self.times = np.array([])
            self._gathered = False
            # Just gather frame indices and times
            self._gather_metadata()
        else:
            # Standard mode: store all coordinates
            self.coordinates: dict = {}  # {sel_str: np.ndarray of shape (T, n_atoms, 3)}
            self.frame_indices: np.ndarray = np.array([])
            self.times: np.ndarray = np.array([])
            self._gathered = False
            # Gather data immediately
            self.gather()
    
    def _gather_metadata(self):
        """Gather only frame indices and times (memory-efficient)."""
        self.frame_indices = np.zeros(self.n_frames, dtype=int)
        self.times = np.zeros(self.n_frames)
        
        for frame_idx, ts in enumerate(self.universe.trajectory):
            self.frame_indices[frame_idx] = ts.frame
            self.times[frame_idx] = ts.time
        
        self._gathered = True
    
    def gather(self):
        """Iterate through all frames and gather coordinates for all selections."""
        if self._gathered:
            return
        
        # Initialize storage for each selection
        for sel_str in self.selection_strings:
            sel = self.universe.select_atoms(sel_str)
            n_atoms = len(sel)
            self.coordinates[sel_str] = np.zeros((self.n_frames, n_atoms, 3))
        
        # Initialize frame tracking arrays
        self.frame_indices = np.zeros(self.n_frames, dtype=int)
        self.times = np.zeros(self.n_frames)
        
        # Iterate through all frames once
        for frame_idx, ts in enumerate(self.universe.trajectory):
            self.frame_indices[frame_idx] = ts.frame
            self.times[frame_idx] = ts.time
            
            # Gather coordinates for each selection
            for sel_str in self.selection_strings:
                sel = self.universe.select_atoms(sel_str)
                self.coordinates[sel_str][frame_idx] = sel.positions.copy()
        
        self._gathered = True
    
    def get_coordinates(self, sel_str: str) -> np.ndarray:
        """
        Get gathered coordinates for a selection string.
        
        In memory-efficient mode, this will gather coordinates on-demand.
        
        Args:
            sel_str: Selection string
            
        Returns:
            Array of shape (T, n_atoms, 3) with coordinates for all frames
        """
        if self.memory_efficient:
            # Memory-efficient mode: gather on-demand
            if sel_str not in self.coordinates:
                # Gather this selection now
                sel = self.universe.select_atoms(sel_str)
                n_atoms = len(sel)
                coords = np.zeros((self.n_frames, n_atoms, 3))
                
                # Reset trajectory to beginning before iterating
                # (in case _gather_metadata() already iterated through it)
                self.universe.trajectory[0]
                
                for frame_idx, ts in enumerate(self.universe.trajectory):
                    coords[frame_idx] = sel.positions.copy()
                
                self.coordinates[sel_str] = coords
            return self.coordinates[sel_str]
        else:
            # Standard mode: return pre-gathered coordinates
            if not self._gathered:
                self.gather()
            if sel_str not in self.coordinates:
                raise ValueError(f"Selection '{sel_str}' was not included in gathering. "
                               f"Available selections: {list(self.coordinates.keys())}")
            return self.coordinates[sel_str]
    
    def get_frame_indices(self) -> np.ndarray:
        """Get frame indices for all gathered frames."""
        if not self._gathered:
            self.gather()
        return self.frame_indices
    
    def get_times(self) -> np.ndarray:
        """Get time values for all gathered frames."""
        if not self._gathered:
            self.gather()
        return self.times
    
    def get_n_frames(self) -> int:
        """Get number of frames."""
        return self.n_frames
    
    def clear(self):
        """Clear gathered data to free memory."""
        self.coordinates.clear()
        self.frame_indices = np.array([])
        self.times = np.array([])
        self._gathered = False


# ============================================================================
# Class: TrajectoryMetrics
# Purpose: Basic trajectory metric computations (RMSD, RMSF, Rg, contacts, PCA, strain)
# ============================================================================

class TrajectoryMetrics:
    """Compute basic trajectory metrics like RMSD, RMSF, radius of gyration, etc.
    
    Note: For accurate results, trajectories should be pre-aligned using AlignedTrajectory
    before computing metrics. The compute_rmsf() and pca_on_fluctuations() methods
    assume the trajectory is already aligned by default.
    """
    
    def __init__(self):
        """Initialize TrajectoryMetrics with storage for computed results."""
        self.rmsd_cache: dict = {}
        self.rmsf_cache: dict = {}
        self.rg_cache: dict = {}
        self.contact_cache: dict = {}
        self.pca_cache: dict = {}
        self.strain_cache: dict = {}
        self._universe: Optional[mda.Universe] = None
        self._alignment_sel: Optional[str] = None
    
    def set_universe(self, u: mda.Universe, align_sel: Optional[str] = None):
        """Set the universe and alignment selection for caching."""
        self._universe = u
        self._alignment_sel = align_sel
    
    def compute_rmsd(self, u: mda.Universe, sel_str: str, ref_frame: int = 0, 
                     gatherer: Optional['FrameGatherer'] = None) -> np.ndarray:
        """Compute RMSD for each frame relative to reference frame.
        
        Args:
            u: MDAnalysis Universe
            sel_str: Selection string
            ref_frame: Reference frame index (default: 0)
            gatherer: Optional FrameGatherer instance with pre-gathered coordinates
        """
        cache_key = (sel_str, ref_frame)
        if cache_key in self.rmsd_cache:
            return self.rmsd_cache[cache_key]
        
        if gatherer is not None:
            # Use pre-gathered coordinates
            coords = gatherer.get_coordinates(sel_str)
            frame_indices = gatherer.get_frame_indices()
            # Find the index in gathered array that corresponds to ref_frame
            ref_idx = np.where(frame_indices == ref_frame)[0]
            if len(ref_idx) == 0:
                raise ValueError(f"Reference frame {ref_frame} not found in gathered data")
            ref_idx = ref_idx[0]
            ref = coords[ref_idx].copy()
            rmsds = []
            for frame_coords in coords:
                R, rmsd_val = align.rotation_matrix(frame_coords, ref)
                rmsds.append(rmsd_val)
            result = np.array(rmsds)
        else:
            # Original iteration-based approach
            sel = u.select_atoms(sel_str)
            ref = sel.positions.copy()
            rmsds = []
            for ts in u.trajectory:
                R, rmsd_val = align.rotation_matrix(sel.positions, ref)
                rmsds.append(rmsd_val)
            result = np.array(rmsds)
        
        self.rmsd_cache[cache_key] = result
        return result
    
    def compute_rmsf(self, u: mda.Universe, sel_str: str, aligned: bool = True,
                     gatherer: Optional['FrameGatherer'] = None) -> np.ndarray:
        """Compute RMSF (per-atom root mean square fluctuation).
        
        Args:
            u: MDAnalysis Universe (should be pre-aligned via AlignedTrajectory)
            sel_str: Selection string for atoms to compute RMSF
            aligned: If True, assumes trajectory is already aligned (default: True)
                    If False, will align internally (not recommended)
            gatherer: Optional FrameGatherer instance with pre-gathered coordinates
        """
        cache_key = sel_str
        if cache_key in self.rmsf_cache:
            return self.rmsf_cache[cache_key]
        
        if gatherer is not None:
            # Use pre-gathered coordinates
            coords = gatherer.get_coordinates(sel_str)
        else:
            # Original iteration-based approach
            sel = u.select_atoms(sel_str)
            
            # Only align if explicitly requested (trajectory should be pre-aligned)
            if not aligned:
                warnings.warn("Computing RMSF on unaligned trajectory. Consider using AlignedTrajectory first.")
                with mda.lib.util.tempdir.in_tempdir():
                    align.AlignTraj(u, u, select=sel_str, in_memory=True).run()
            
            coords = []
            for ts in u.trajectory:
                coords.append(sel.positions.copy())
            coords = np.array(coords)  # (T, n, 3)
        
        mean = coords.mean(axis=0)
        diffsq = (coords - mean) ** 2
        rmsf = np.sqrt(diffsq.sum(axis=2).mean(axis=0))
        self.rmsf_cache[cache_key] = rmsf
        return rmsf
    
    def radius_of_gyration(self, u: mda.Universe, sel_str: str,
                          gatherer: Optional['FrameGatherer'] = None) -> np.ndarray:
        """Compute radius of gyration for each frame.
        
        Args:
            u: MDAnalysis Universe
            sel_str: Selection string
            gatherer: Optional FrameGatherer instance with pre-gathered coordinates
        """
        cache_key = sel_str
        if cache_key in self.rg_cache:
            return self.rg_cache[cache_key]
        
        if gatherer is not None:
            # Use pre-gathered coordinates
            coords = gatherer.get_coordinates(sel_str)
            rgs = []
            for frame_coords in coords:
                com = frame_coords.mean(axis=0)
                rg2 = ((frame_coords - com) ** 2).sum(axis=1).mean()
                rgs.append(np.sqrt(rg2))
            result = np.array(rgs)
        else:
            # Original iteration-based approach
            sel = u.select_atoms(sel_str)
            rgs = []
            for ts in u.trajectory:
                coords = sel.positions
                com = sel.center_of_mass()
                rg2 = ((coords - com) ** 2).sum(axis=1).mean()
                rgs.append(np.sqrt(rg2))
            result = np.array(rgs)
        
        self.rg_cache[cache_key] = result
        return result
    
    def contact_distances(self, u: mda.Universe, selA: str, selB: str,
                         gatherer: Optional['FrameGatherer'] = None) -> np.ndarray:
        """Compute minimal contact distance between two selections for each frame.
        
        Args:
            u: MDAnalysis Universe
            selA: First selection string
            selB: Second selection string
            gatherer: Optional FrameGatherer instance with pre-gathered coordinates
        """
        cache_key = (selA, selB)
        if cache_key in self.contact_cache:
            return self.contact_cache[cache_key]
        
        if gatherer is not None:
            # Use pre-gathered coordinates
            coordsA = gatherer.get_coordinates(selA)
            coordsB = gatherer.get_coordinates(selB)
            dists = []
            for frame_idx in range(gatherer.get_n_frames()):
                da = coordsA[frame_idx][:, None, :]
                db = coordsB[frame_idx][None, :, :]
                diff = da - db
                dd = np.sqrt((diff * diff).sum(axis=2))
                dists.append(dd.min())
            result = np.array(dists)
        else:
            # Original iteration-based approach
            A = u.select_atoms(selA)
            B = u.select_atoms(selB)
            if len(A) == 0 or len(B) == 0:
                raise ValueError("Empty selection for contacts.")
            dists = []
            for ts in u.trajectory:
                da = A.positions[:, None, :]
                db = B.positions[None, :, :]
                diff = da - db
                dd = np.sqrt((diff * diff).sum(axis=2))
                dists.append(dd.min())
            result = np.array(dists)
        
        self.contact_cache[cache_key] = result
        return result
    
    def pca_on_fluctuations(self, u: mda.Universe, sel_str: str, n_components: int = 5, 
                            aligned: bool = True, gatherer: Optional['FrameGatherer'] = None) -> Tuple[np.ndarray, PCA]:
        """Return PC projections (T x n_comp) and the fitted PCA model.
        
        Args:
            u: MDAnalysis Universe (should be pre-aligned via AlignedTrajectory)
            sel_str: Selection string for atoms to use in PCA
            n_components: Number of principal components to compute
            aligned: If True, assumes trajectory is already aligned (default: True)
                    If False, will align internally (not recommended)
            gatherer: Optional FrameGatherer instance with pre-gathered coordinates
        """
        cache_key = (sel_str, n_components)
        if cache_key in self.pca_cache:
            return self.pca_cache[cache_key]
        
        if gatherer is not None:
            # Use pre-gathered coordinates
            coords = gatherer.get_coordinates(sel_str)
            X = coords.reshape(coords.shape[0], -1)  # shape (T, 3N)
        else:
            # Original iteration-based approach
            sel = u.select_atoms(sel_str)
            
            # Only align if explicitly requested (trajectory should be pre-aligned)
            if not aligned:
                warnings.warn("Computing PCA on unaligned trajectory. Consider using AlignedTrajectory first.")
                align.AlignTraj(u, u, select=sel_str, in_memory=True).run()
            
            coords = []
            for ts in u.trajectory:
                coords.append(sel.positions.copy().reshape(-1))
            X = np.array(coords)  # shape (T, 3N)
        
        Xc = X - X.mean(axis=0)
        pca = PCA(n_components=n_components, svd_solver="auto")
        pcs = pca.fit_transform(Xc)
        result = (pcs, pca)
        self.pca_cache[cache_key] = result
        return result
    
    def local_affine_strain_proxy(self, u: mda.Universe, sel_str: str, window: int = 10, lag: int = 1,
                                  gatherer: Optional['FrameGatherer'] = None) -> np.ndarray:
        """Compute a lightweight proxy of local strain.
        
        Args:
            u: MDAnalysis Universe
            sel_str: Selection string
            window: Window size for strain computation
            lag: Lag for strain computation
            gatherer: Optional FrameGatherer instance with pre-gathered coordinates
        """
        cache_key = (sel_str, window, lag)
        if cache_key in self.strain_cache:
            return self.strain_cache[cache_key]
        
        if gatherer is not None:
            # Use pre-gathered coordinates
            X = gatherer.get_coordinates(sel_str)  # (T, n, 3)
        else:
            # Original iteration-based approach
            sel = u.select_atoms(sel_str)
            coords = []
            for ts in u.trajectory:
                coords.append(sel.positions.copy())
            X = np.array(coords)  # (T, n, 3)
        
        T, n, _ = X.shape
        strain = np.full(T, np.nan)
        for t in range(0, T - lag):
            if t < window:
                continue
            A = X[t - window:t, :, :].reshape(-1, 3)
            B = X[t - window + lag:t + lag, :, :].reshape(-1, 3)
            A_aug = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
            Xsol, *_ = np.linalg.lstsq(A_aug, B, rcond=None)
            F = Xsol[:3, :]
            C = F.T @ F
            E = 0.5 * (C - np.eye(3))
            strain[t] = np.linalg.norm(E, ord='fro')
        self.strain_cache[cache_key] = strain
        return strain
    
    def clear_cache(self):
        """Clear all cached results."""
        self.rmsd_cache.clear()
        self.rmsf_cache.clear()
        self.rg_cache.clear()
        self.contact_cache.clear()
        self.pca_cache.clear()
        self.strain_cache.clear()


# ============================================================================
# Class: ClusteringAnalysis
# Purpose: Clustering and change-point detection
# ============================================================================

class ClusteringAnalysis:
    """Perform clustering analysis and change-point detection on trajectory data."""
    
    def __init__(self):
        """Initialize ClusteringAnalysis with storage for computed results."""
        self.clustering_results: dict = {}
        self.change_points_cache: dict = {}
        self.labels: Optional[np.ndarray] = None
        self.medoids: Optional[np.ndarray] = None
    
    def zscore(self, x: np.ndarray) -> np.ndarray:
        """Compute z-score normalization."""
        # Simple computation, no caching needed (zscore is fast)
        m, s = np.nanmean(x), np.nanstd(x)
        if s == 0 or np.isnan(s):
            return np.zeros_like(x)
        return (x - m) / s
    
    def choose_k_by_silhouette(self, X, kmin=2, kmax=10) -> int:
        """Choose optimal k for KMeans using silhouette score."""
        best_k, best_score = kmin, -1
        for k in range(kmin, min(kmax, len(X) - 1) + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
            labels = km.labels_
            if len(set(labels)) == 1:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        return best_k
    
    def cluster_frames(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster frame embeddings (e.g., PCs). Returns (labels, medoid_indices)."""
        cache_key = id(embeddings)
        if cache_key in self.clustering_results:
            cached_labels, cached_medoids, cached_id = self.clustering_results[cache_key]
            if cached_id == id(embeddings) and np.array_equal(cached_labels.shape, (len(embeddings),)):
                self.labels = cached_labels
                self.medoids = cached_medoids
                return cached_labels, cached_medoids
        
        X = embeddings
        if HAS_HDBSCAN:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(10, len(X)//100+1), min_samples=None)
            labels = clusterer.fit_predict(X)
            medoids = []
            for cl in sorted(set(labels)):
                if cl == -1:
                    continue
                idx = np.where(labels == cl)[0]
                centroid = X[idx].mean(axis=0)
                d = np.linalg.norm(X[idx] - centroid, axis=1)
                medoids.append(idx[np.argmin(d)])
            result_labels = labels
            result_medoids = np.array(medoids, dtype=int)
        else:
            k = self.choose_k_by_silhouette(X)
            km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
            labels = km.labels_
            medoids = []
            for cl in range(k):
                idx = np.where(labels == cl)[0]
                centroid = X[idx].mean(axis=0)
                d = np.linalg.norm(X[idx] - centroid, axis=1)
                medoids.append(idx[np.argmin(d)])
            result_labels = labels
            result_medoids = np.array(medoids, dtype=int)
        
        self.labels = result_labels
        self.medoids = result_medoids
        self.clustering_results[cache_key] = (result_labels, result_medoids, id(embeddings))
        return result_labels, result_medoids
    
    def change_points(self, signal: np.ndarray, model: str = "rbf", n_bkps: int = 5) -> List[int]:
        """Return change-point indices for a 1D signal."""
        cache_key = (id(signal), model, n_bkps)
        if cache_key in self.change_points_cache:
            cached_result, cached_id = self.change_points_cache[cache_key]
            if cached_id == id(signal) and len(cached_result) > 0:
                return cached_result
        
        if HAS_RUPTURES:
            algo = rpt.Binseg(model=model).fit(signal.reshape(-1, 1))
            bkps = algo.predict(n_bkps=n_bkps)
            result = sorted(set([b for b in bkps if b < len(signal)]))
        else:
            # Fallback: pick top-N peaks of absolute derivative
            deriv = np.abs(np.gradient(signal))
            N = max(2, n_bkps)
            idx = np.argsort(deriv)[-N:]
            result = sorted(set(idx.tolist()))
        
        self.change_points_cache[cache_key] = (result, id(signal))
        return result
    
    def clear_cache(self):
        """Clear all cached results."""
        self.clustering_results.clear()
        self.change_points_cache.clear()
        self.labels = None
        self.medoids = None
    
    def get_labels(self) -> Optional[np.ndarray]:
        """Get the most recently computed cluster labels."""
        return self.labels
    
    def get_medoids(self) -> Optional[np.ndarray]:
        """Get the most recently computed medoids."""
        return self.medoids


# ============================================================================
# Class: FrameSelection
# Purpose: Frame scoring and selection
# ============================================================================

class FrameSelection:
    """Score and select meaningful frames from trajectory."""
    
    def __init__(self):
        """Initialize FrameSelection with storage for computed results."""
        self.scores: Optional[np.ndarray] = None
        self.score_df: Optional[pd.DataFrame] = None
        self.selected_frames: Optional[List[int]] = None
        self.medoids: Optional[np.ndarray] = None
        self.cpd_idx: Optional[List[int]] = None
        self.pcs: Optional[np.ndarray] = None
    
    def score_frames(self, rmsd: np.ndarray,
                     rg: np.ndarray,
                     pcs: np.ndarray,
                     cpd_idx: List[int],
                     strain: Optional[np.ndarray] = None,
                     w: Tuple[float, float, float, float] = (0.35, 0.2, 0.35, 0.1)) -> Tuple[np.ndarray, pd.DataFrame]:
        """Combine multiple criteria into a single score."""
        T = len(rmsd)
        clustering = ClusteringAnalysis()
        
        # Store inputs for later use
        self.pcs = pcs
        
        # 1) PC extremes
        pcZ = clustering.zscore(pcs[:, :min(3, pcs.shape[1])])
        pc_ext = np.max(np.abs(pcZ), axis=1)
        
        # 2) RMSD derivative magnitude
        rmsd_deriv = np.abs(np.gradient(rmsd))
        rmsd_dZ = clustering.zscore(rmsd_deriv)
        
        # 3) Rg extrema
        rgZ = np.abs(clustering.zscore(rg))
        
        # 4) Strain (optional)
        if strain is None:
            strainZ = np.zeros(T)
        else:
            strainZ = clustering.zscore(strain)
            strainZ = np.nan_to_num(strainZ)
        
        # Change-point bonus
        bonus = np.zeros(T)
        for i in cpd_idx:
            if 0 <= i < T:
                bonus[i] += 1.0
        
        score = w[0]*pc_ext + w[1]*rmsd_dZ + w[2]*rgZ + w[3]*strainZ + bonus
        df = pd.DataFrame({
            'pc_extreme': pc_ext,
            'rmsd_d': rmsd_dZ,
            'rg_extreme': rgZ,
            'strain': strainZ,
            'cpd_bonus': bonus,
            'score': score
        })
        self.scores = score
        self.score_df = df
        return score, df
    
    def load_data_from_csv(self, scores_csv: Optional[str] = None,
                           metrics_csv: Optional[str] = None,
                           endpoint_metrics_csv: Optional[str] = None) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load data from previously saved CSV files."""
        medoids = None
        cpd_idx = None
        scores = None
        pcs = None
        score_df = None
        endpoint_metrics_df = None
        
        if scores_csv and os.path.exists(scores_csv):
            try:
                score_df = pd.read_csv(scores_csv)
                print(f"Loaded scores from {scores_csv}")
                if 'is_medoid' in score_df.columns:
                    medoids = score_df[score_df['is_medoid'] == 1]['frame'].values
                    medoids = np.array(medoids, dtype=int)
                if 'is_cpd' in score_df.columns:
                    cpd_idx = score_df[score_df['is_cpd'] == 1]['frame'].tolist()
                    cpd_idx = [int(x) for x in cpd_idx]
                if 'score' in score_df.columns:
                    scores = score_df['score'].values
            except Exception as e:
                warnings.warn(f"Failed to load scores CSV {scores_csv}: {e}")
        
        if metrics_csv and os.path.exists(metrics_csv):
            try:
                metrics_df = pd.read_csv(metrics_csv)
                print(f"Loaded metrics from {metrics_csv}")
                pc_cols = [col for col in metrics_df.columns if col.startswith('pc') and col[2:].isdigit()]
                if len(pc_cols) > 0:
                    pc_cols = sorted(pc_cols, key=lambda x: int(x[2:]))
                    pcs = metrics_df[pc_cols].values
                    print(f"Extracted {len(pc_cols)} principal components from metrics CSV")
            except Exception as e:
                warnings.warn(f"Failed to load metrics CSV {metrics_csv}: {e}")
        
        if endpoint_metrics_csv and os.path.exists(endpoint_metrics_csv):
            try:
                endpoint_metrics_df = pd.read_csv(endpoint_metrics_csv)
                print(f"Loaded endpoint metrics from {endpoint_metrics_csv}")
            except Exception as e:
                warnings.warn(f"Failed to load endpoint metrics CSV {endpoint_metrics_csv}: {e}")
        
        # Store loaded data in instance
        if medoids is not None:
            self.medoids = medoids
        if cpd_idx is not None:
            self.cpd_idx = cpd_idx
        if scores is not None:
            self.scores = scores
        if pcs is not None:
            self.pcs = pcs
        if score_df is not None:
            self.score_df = score_df
        
        return medoids, cpd_idx, scores, pcs, score_df, endpoint_metrics_df
    
    def select_meaningful_frames(self, medoids: Optional[np.ndarray] = None,
                                 cpd_idx: Optional[List[int]] = None,
                                 scores: Optional[np.ndarray] = None,
                                 pcs: Optional[np.ndarray] = None,
                                 max_frames: int = 30,
                                 score_df: Optional[pd.DataFrame] = None,
                                 distance_threshold: float = 0.75,
                                 endpoint_dists_array: Optional[Union[np.ndarray, dict]] = None,
                                 endpoint_metrics_df: Optional[pd.DataFrame] = None,
                                 scores_csv: Optional[str] = None,
                                 metrics_csv: Optional[str] = None,
                                 endpoint_metrics_csv: Optional[str] = None) -> Tuple[List[int], Optional[pd.DataFrame]]:
        """Select meaningful frames from trajectory."""
        # Use instance state if available
        if medoids is None and self.medoids is not None:
            medoids = self.medoids
        if cpd_idx is None and self.cpd_idx is not None:
            cpd_idx = self.cpd_idx
        if scores is None and self.scores is not None:
            scores = self.scores
        if pcs is None and self.pcs is not None:
            pcs = self.pcs
        if score_df is None and self.score_df is not None:
            score_df = self.score_df
        
        # Load data from CSV files if provided
        if (scores_csv or metrics_csv or endpoint_metrics_csv) and (medoids is None or cpd_idx is None or scores is None or pcs is None):
            print("Loading data from CSV files...")
            loaded_medoids, loaded_cpd_idx, loaded_scores, loaded_pcs, loaded_score_df, loaded_endpoint_metrics_df = self.load_data_from_csv(
                scores_csv=scores_csv,
                metrics_csv=metrics_csv,
                endpoint_metrics_csv=endpoint_metrics_csv
            )
            if medoids is None and loaded_medoids is not None:
                medoids = loaded_medoids
            if cpd_idx is None and loaded_cpd_idx is not None:
                cpd_idx = loaded_cpd_idx
            if scores is None and loaded_scores is not None:
                scores = loaded_scores
            if pcs is None and loaded_pcs is not None:
                pcs = loaded_pcs
            if score_df is None and loaded_score_df is not None:
                score_df = loaded_score_df
            if endpoint_metrics_df is None and loaded_endpoint_metrics_df is not None:
                endpoint_metrics_df = loaded_endpoint_metrics_df
        
        # Validate required inputs
        if medoids is None:
            raise ValueError("medoids must be provided either directly or via scores_csv")
        if cpd_idx is None:
            raise ValueError("cpd_idx must be provided either directly or via scores_csv")
        if scores is None:
            raise ValueError("scores must be provided either directly or via scores_csv")
        if pcs is None:
            raise ValueError("pcs must be provided either directly or via metrics_csv")
        
        # Start with union of medoids and change-points
        chosen = set(medoids.tolist() + cpd_idx)
        
        # Add frames with extreme endpoint distances if available
        if endpoint_metrics_df is not None and len(endpoint_metrics_df) > 0:
            if 'frame' not in endpoint_metrics_df.columns:
                endpoint_metrics_df = endpoint_metrics_df.copy()
                endpoint_metrics_df['frame'] = endpoint_metrics_df.index
            
            if 'endpoint_dist_min' in endpoint_metrics_df.columns:
                min_dist = endpoint_metrics_df['endpoint_dist_min'].values
                valid_min = ~np.isnan(min_dist)
                if np.any(valid_min):
                    min_idx = np.nanargmin(min_dist)
                    min_frame = int(endpoint_metrics_df.iloc[min_idx]['frame'])
                    chosen.add(min_frame)
            
            if 'endpoint_dist_max' in endpoint_metrics_df.columns:
                max_dist = endpoint_metrics_df['endpoint_dist_max'].values
                valid_max = ~np.isnan(max_dist)
                if np.any(valid_max):
                    max_idx = np.nanargmax(max_dist)
                    max_frame = int(endpoint_metrics_df.iloc[max_idx]['frame'])
                    chosen.add(max_frame)
            
            if 'endpoint_dist_mean' in endpoint_metrics_df.columns:
                mean_dist = endpoint_metrics_df['endpoint_dist_mean'].values
                valid_mean = ~np.isnan(mean_dist)
                if np.any(valid_mean):
                    deriv = np.abs(np.gradient(mean_dist))
                    deriv[~valid_mean] = 0
                    top_deriv_indices = np.argsort(deriv)[-3:][::-1]
                    for idx in top_deriv_indices:
                        if valid_mean[idx] and len(chosen) < max_frames:
                            frame_num = int(endpoint_metrics_df.iloc[idx]['frame'])
                            chosen.add(frame_num)
        
        # Handle endpoint_dists_array
        if endpoint_dists_array is not None:
            if isinstance(endpoint_dists_array, dict):
                all_pairs = endpoint_dists_array.get('all_pairs', {})
                T = endpoint_dists_array.get('n_frames', 0)
                frame_stats = []
                for t in range(T):
                    all_pair_dists = []
                    for (i, j), pair_array in all_pairs.items():
                        if i < j:
                            frame_pair_dists = pair_array[t]
                            valid_dists = frame_pair_dists[~np.isnan(frame_pair_dists)]
                            all_pair_dists.extend(valid_dists.tolist())
                    if len(all_pair_dists) > 0:
                        all_pair_dists = np.array(all_pair_dists)
                        frame_stats.append({
                            'frame': t,
                            'min': np.min(all_pair_dists),
                            'max': np.max(all_pair_dists)
                        })
                if len(frame_stats) > 0:
                    stats_df = pd.DataFrame(frame_stats)
                    if 'min' in stats_df.columns:
                        min_frame = int(stats_df.loc[stats_df['min'].idxmin(), 'frame'])
                        if len(chosen) < max_frames:
                            chosen.add(min_frame)
                    if 'max' in stats_df.columns:
                        max_frame = int(stats_df.loc[stats_df['max'].idxmax(), 'frame'])
                        if len(chosen) < max_frames:
                            chosen.add(max_frame)
        
        # Sort frames by score and add top-scoring non-redundant frames
        order = np.argsort(scores)[::-1]
        
        def far_from_set(i: int, chosen_list: List[int], thr: float = 1.0) -> bool:
            """Check if frame i is far enough from already chosen frames in PC space."""
            if len(chosen_list) == 0:
                return True
            X = pcs[:, :3]
            v = X[i]
            for j in chosen_list:
                if np.linalg.norm(v - X[j]) < thr:
                    return False
            return True
        
        for i in order:
            if len(chosen) >= max_frames:
                break
            if far_from_set(i, list(chosen), thr=distance_threshold):
                chosen.add(i)
        
        chosen_sorted = sorted(chosen)
        
        # Update score_df if provided
        if score_df is not None:
            score_df['selected'] = 0
            if 'frame' in score_df.columns:
                mask = score_df['frame'].isin(chosen_sorted)
                score_df.loc[mask, 'selected'] = 1
            else:
                score_df.loc[chosen_sorted, 'selected'] = 1
        
        # Store results in instance
        self.selected_frames = chosen_sorted
        self.score_df = score_df
        if medoids is not None:
            self.medoids = medoids
        if cpd_idx is not None:
            self.cpd_idx = cpd_idx
        if scores is not None:
            self.scores = scores
        if pcs is not None:
            self.pcs = pcs
        
        return chosen_sorted, score_df


# ============================================================================
# Class: FileIO
# Purpose: File input/output operations
# ============================================================================

class FileIO:
    """Handle file input/output operations."""
    
    def __init__(self):
        """Initialize FileIO with storage for file paths and configurations."""
        self.output_prefix: Optional[str] = None
        self.saved_frames: List[int] = []
        self.output_directory: Optional[str] = None
    
    def save_frames_as_pdb(self, u: mda.Universe, frame_indices: List[int], out_prefix: str, sel: Optional[str] = None):
        """Save selected frames as PDB files."""
        self.output_prefix = out_prefix
        output_dir = f"{out_prefix}_frames"
        self.output_directory = output_dir
        os.makedirs(output_dir, exist_ok=True)
        if sel is None:
            ag = u.atoms
        else:
            ag = u.select_atoms(sel)
        for idx in frame_indices:
            u.trajectory[idx]
            ag.write(f"{output_dir}/frame_{idx:06d}.pdb")
        self.saved_frames.extend(frame_indices)


# ============================================================================
# GSAnalyzer is now imported from gs_analyzer.py module
# ============================================================================


# ============================================================================
# Class: EndpointAnalyzer
# Purpose: Endpoint-based analysis functions
# ============================================================================

class EndpointAnalyzer:
    """Analyze molecular endpoints and compute endpoint-based metrics."""
    
    def __init__(self):
        """Initialize EndpointAnalyzer with storage for results."""
        self.endpoints_distance_dict: Optional[dict] = None
        self.endpoint_metrics_df: Optional[pd.DataFrame] = None
        self.residue_sel_list: Optional[List[str]] = None
        self._endpoints_finder: Optional['EndpointsFinder'] = None
    
    def clear_cache(self):
        """Clear all cached results."""
        self.endpoints_distance_dict = None
        self.endpoint_metrics_df = None
        self.residue_sel_list = None
        self._endpoints_finder = None
    
    @staticmethod
    def find_residue_endpoints(u: mda.Universe,
                               sel_str: str,
                               endpoints_finder: Optional['EndpointsFinder'] = None) -> Tuple[np.ndarray, List[int]]:
        """Find the endpoints of a residue using EndpointsFinder."""
        if endpoints_finder is None:
            endpoints_finder = EndpointsFinder()
        
        sel = u.select_atoms(sel_str)
        if len(sel) == 0:
            return np.array([np.nan, np.nan, np.nan]), []
        
        
        mol = sel.convert_to('RDKIT')
        if mol is None:
            raise ValueError("RDKit conversion failed")
        
        endpoint_indices = endpoints_finder.find_endpoints(mol)
        #Change into  mda.Universe global index
        mda_endpoint_indices = []
        for rdkit_idx in endpoint_indices:
            mda_endpoint_indices.append(int(sel[rdkit_idx].id))
            
        center = sel.center_of_mass()
        return center, mda_endpoint_indices if mda_endpoint_indices else []
    
    def compute_endpoint_distances(self, u: mda.Universe,
                                   residue_sel_list: List[str],
                                   endpoints_finder: Optional['EndpointsFinder'] = None,
                                   gatherer: Optional['FrameGatherer'] = None) -> dict:
        """Compute distances between endpoints of different residues over trajectory."""
        # Store configuration
        self.residue_sel_list = residue_sel_list
        if endpoints_finder is not None:
            self._endpoints_finder = endpoints_finder
        
        if endpoints_finder is None:
            endpoints_finder = EndpointsFinder()
        
        n_res = len(residue_sel_list)
        T = len(u.trajectory)
        all_pairs = {}
        
        # Find endpoint indices once on the initial frame (residues don't change)
        stored_ep_indices = []
        u.trajectory[0]  # Go to initial frame
        for sel_str in residue_sel_list:
            try:
                _, ep_indices = EndpointAnalyzer.find_residue_endpoints(u, sel_str, endpoints_finder)
                stored_ep_indices.append(ep_indices)
            except Exception as e:
                warnings.warn(f"Failed to find endpoints for {sel_str} on initial frame: {e}")
                stored_ep_indices.append([])
        
        # Determine max number of endpoints per residue from stored indices
        max_ep_per_residue = [len(ep_indices) for ep_indices in stored_ep_indices]
        
        # Initialize distance arrays for all residue pairs
        for i in range(n_res):
            for j in range(i + 1, n_res):
                max_ep_i = max_ep_per_residue[i]
                max_ep_j = max_ep_per_residue[j]
                
                if max_ep_i > 0 and max_ep_j > 0:
                    pair_distances = np.full((T, max_ep_i, max_ep_j), np.nan)
                    all_pairs[(i, j)] = pair_distances
                    all_pairs[(j, i)] = np.full((T, max_ep_j, max_ep_i), np.nan)
        
        # Now iterate through all frames and compute distances directly
        if gatherer is not None:
            # Use pre-gathered coordinates
            frame_indices = gatherer.get_frame_indices()
            for frame_idx, frame_num in enumerate(frame_indices):
                frame = int(frame_num)
                
                # Get endpoint positions for all residues at this frame
                frame_ep_positions = []
                for idx, sel_str in enumerate(residue_sel_list):
                    try:
                        ep_indices = stored_ep_indices[idx]
                        
                        if len(ep_indices) > 0:
                            # Get coordinates from gatherer
                            residue_coords = gatherer.get_coordinates(sel_str)[frame_idx]
                            # Map endpoint indices to positions within the residue selection
                            sel = u.select_atoms(sel_str)
                            # Find local indices of endpoints within the selection
                            sel_atom_ids = sel.atoms.ids
                            local_ep_indices = []
                            for ep_id in ep_indices:
                                if ep_id in sel_atom_ids:
                                    local_idx = np.where(sel_atom_ids == ep_id)[0]
                                    if len(local_idx) > 0:
                                        local_ep_indices.append(local_idx[0])
                            
                            if len(local_ep_indices) > 0:
                                ep_positions = residue_coords[local_ep_indices]
                                frame_ep_positions.append(ep_positions)
                            else:
                                frame_ep_positions.append(None)
                        else:
                            frame_ep_positions.append(None)
                    except Exception as e:
                        raise ValueError(f"Failed to get endpoint positions for {sel_str} at frame {frame}: {e}")
                
                # Compute distances for all residue pairs at this frame
                for i in range(n_res):
                    for j in range(i + 1, n_res):
                        ep_i = frame_ep_positions[i]
                        ep_j = frame_ep_positions[j]
                        
                        if ep_i is not None and ep_j is not None and (i, j) in all_pairs:
                            diff = ep_i[:, None, :] - ep_j[None, :, :]
                            dists = np.linalg.norm(diff, axis=2)
                            n_ep_i_actual, n_ep_j_actual = dists.shape
                            all_pairs[(i, j)][frame_idx, :n_ep_i_actual, :n_ep_j_actual] = dists
                            all_pairs[(j, i)][frame_idx, :n_ep_j_actual, :n_ep_i_actual] = dists.T
        else:
            # Original iteration-based approach
            for ts in u.trajectory:
                frame = ts.frame
                
                # Get endpoint positions for all residues at this frame
                frame_ep_positions = []
                for idx, sel_str in enumerate(residue_sel_list):
                    try:
                        ep_indices = stored_ep_indices[idx]
                        
                        if len(ep_indices) > 0:
                            ep_positions = u.atoms[ep_indices].positions
                            frame_ep_positions.append(ep_positions)
                        else:
                            frame_ep_positions.append(None)
                    except Exception as e:
                        raise ValueError(f"Failed to get endpoint positions for {sel_str} at frame {frame}: {e}")
                
                # Compute distances for all residue pairs at this frame
                for i in range(n_res):
                    for j in range(i + 1, n_res):
                        ep_i = frame_ep_positions[i]
                        ep_j = frame_ep_positions[j]
                        
                        if ep_i is not None and ep_j is not None and (i, j) in all_pairs:
                            diff = ep_i[:, None, :] - ep_j[None, :, :]
                            dists = np.linalg.norm(diff, axis=2)
                            n_ep_i_actual, n_ep_j_actual = dists.shape
                            all_pairs[(i, j)][frame, :n_ep_i_actual, :n_ep_j_actual] = dists
                            all_pairs[(j, i)][frame, :n_ep_j_actual, :n_ep_i_actual] = dists.T
        
        self.endpoints_distance_dict = {
            'all_pairs': all_pairs,
            'n_residues': n_res,
            'n_frames': T
        }
        return self.endpoints_distance_dict
    
    def compute_endpoint_metrics(self, u: mda.Universe,
                                residue_sel_list: List[str],
                                endpoints_finder: Optional['EndpointsFinder'] = None,
                                gatherer: Optional['FrameGatherer'] = None) -> pd.DataFrame:
        """Compute comprehensive endpoint-based metrics for residues over trajectory."""
        # Store configuration
        self.residue_sel_list = residue_sel_list
        if endpoints_finder is not None:
            self._endpoints_finder = endpoints_finder
        
        endpoint_dists_dict = self.compute_endpoint_distances(u, residue_sel_list, endpoints_finder, gatherer=gatherer)
        all_pairs = endpoint_dists_dict['all_pairs']
        T = endpoint_dists_dict['n_frames']
        n_res = endpoint_dists_dict['n_residues']
        
        rows = []
        for frame in range(T):
            row = {'frame': frame}
            all_pair_dists = []
            
            for i in range(n_res):
                for j in range(i + 1, n_res):
                    if (i, j) in all_pairs:
                        pair_array = all_pairs[(i, j)][frame]
                        valid_dists = pair_array[~np.isnan(pair_array)]
                        all_pair_dists.extend(valid_dists.tolist())
                        
                        if len(valid_dists) > 0:
                            row[f'endpoint_dist_{i}_{j}_mean'] = float(np.mean(valid_dists))
                            row[f'endpoint_dist_{i}_{j}_min'] = float(np.min(valid_dists))
                            row[f'endpoint_dist_{i}_{j}_max'] = float(np.max(valid_dists))
                        else:
                            row[f'endpoint_dist_{i}_{j}_mean'] = np.nan
                            row[f'endpoint_dist_{i}_{j}_min'] = np.nan
                            row[f'endpoint_dist_{i}_{j}_max'] = np.nan
            
            if len(all_pair_dists) > 0:
                all_pair_dists = np.array(all_pair_dists)
                row['endpoint_dist_mean'] = float(np.mean(all_pair_dists))
                row['endpoint_dist_min'] = float(np.min(all_pair_dists))
                row['endpoint_dist_max'] = float(np.max(all_pair_dists))
                row['endpoint_dist_std'] = float(np.std(all_pair_dists))
            else:
                row['endpoint_dist_mean'] = np.nan
                row['endpoint_dist_min'] = np.nan
                row['endpoint_dist_max'] = np.nan
                row['endpoint_dist_std'] = np.nan
            
            rows.append(row)
        
        self.endpoint_metrics_df = pd.DataFrame(rows)
        return self.endpoint_metrics_df
    
    @staticmethod
    def compute_endpoint_volume_correlation(endpoint_dists_array: Union[np.ndarray, dict],
                                            volume: np.ndarray,
                                            residue_sel_list: List[str]) -> pd.DataFrame:
        """Compute correlation between each endpoint pair distance and cube volume."""
        if endpoint_dists_array is None:
            return pd.DataFrame(columns=['residue_i', 'residue_j', 'ep_i_idx', 'ep_j_idx', 'correlation', 'p_value', 'n_valid_points'])
        
        valid_volume_mask = ~np.isnan(volume)
        if not np.any(valid_volume_mask):
            warnings.warn("No valid volume values. Cannot compute correlations.")
            return pd.DataFrame(columns=['residue_i', 'residue_j', 'ep_i_idx', 'ep_j_idx', 'correlation', 'p_value', 'n_valid_points'])
        
        try:
            from scipy.stats import pearsonr
        except ImportError:
            warnings.warn("scipy not available. Cannot compute correlations.")
            return pd.DataFrame(columns=['residue_i', 'residue_j', 'ep_i_idx', 'ep_j_idx', 'correlation', 'p_value', 'n_valid_points'])
        
        correlations = []
        
        if isinstance(endpoint_dists_array, dict):
            all_pairs = endpoint_dists_array.get('all_pairs', {})
            T = endpoint_dists_array.get('n_frames', 0)
            n_res = endpoint_dists_array.get('n_residues', 0)
            
            for (i, j), pair_array in all_pairs.items():
                if i >= j:
                    continue
                
                n_ep_i, n_ep_j = pair_array.shape[1], pair_array.shape[2]
                
                for ep_i in range(n_ep_i):
                    for ep_j in range(n_ep_j):
                        dists = pair_array[:, ep_i, ep_j]
                        valid_mask = valid_volume_mask & (~np.isnan(dists))
                        
                        if np.sum(valid_mask) < 3:
                            correlations.append({
                                'residue_i': i,
                                'residue_j': j,
                                'ep_i_idx': ep_i,
                                'ep_j_idx': ep_j,
                                'residue_i_sel': residue_sel_list[i] if i < len(residue_sel_list) else f"residue_{i}",
                                'residue_j_sel': residue_sel_list[j] if j < len(residue_sel_list) else f"residue_{j}",
                                'correlation': np.nan,
                                'p_value': np.nan,
                                'n_valid_points': np.sum(valid_mask)
                            })
                            continue
                        
                        valid_dists = dists[valid_mask]
                        valid_vol = volume[valid_mask]
                        
                        try:
                            corr, p_val = pearsonr(valid_dists, valid_vol)
                            correlations.append({
                                'residue_i': i,
                                'residue_j': j,
                                'ep_i_idx': ep_i,
                                'ep_j_idx': ep_j,
                                'residue_i_sel': residue_sel_list[i] if i < len(residue_sel_list) else f"residue_{i}",
                                'residue_j_sel': residue_sel_list[j] if j < len(residue_sel_list) else f"residue_{j}",
                                'correlation': corr,
                                'p_value': p_val,
                                'n_valid_points': np.sum(valid_mask)
                            })
                        except Exception as e:
                            warnings.warn(f"Correlation computation failed for pair ({i}, {j}), endpoints ({ep_i}, {ep_j}): {e}")
                            correlations.append({
                                'residue_i': i,
                                'residue_j': j,
                                'ep_i_idx': ep_i,
                                'ep_j_idx': ep_j,
                                'residue_i_sel': residue_sel_list[i] if i < len(residue_sel_list) else f"residue_{i}",
                                'residue_j_sel': residue_sel_list[j] if j < len(residue_sel_list) else f"residue_{j}",
                                'correlation': np.nan,
                                'p_value': np.nan,
                                'n_valid_points': np.sum(valid_mask)
                            })
        
        df = pd.DataFrame(correlations)
        if df.empty or 'correlation' not in df.columns:
            return pd.DataFrame(columns=['residue_i', 'residue_j', 'ep_i_idx', 'ep_j_idx', 'residue_i_sel', 'residue_j_sel', 'correlation', 'p_value', 'n_valid_points'])
        
        df['abs_correlation'] = np.abs(df['correlation'].fillna(0))
        df = df.sort_values('abs_correlation', ascending=False)
        df = df.drop('abs_correlation', axis=1)
        
        return df
    
    @staticmethod
    def analyze_endpoint_pair_variation(endpoint_dists_dict: dict,
                                        residue_sel_list: List[str]) -> pd.DataFrame:
        """Analyze variation of endpoint pair distances through time."""
        if not isinstance(endpoint_dists_dict, dict):
            warnings.warn("endpoint_dists_dict must be a dictionary from compute_endpoint_distances")
            return pd.DataFrame()
        
        all_pairs = endpoint_dists_dict.get('all_pairs', {})
        T = endpoint_dists_dict.get('n_frames', 0)
        n_res = endpoint_dists_dict.get('n_residues', 0)
        
        if T == 0 or n_res == 0:
            return pd.DataFrame()
        
        variation_data = []
        
        for (i, j), pair_array in all_pairs.items():
            if i >= j:
                continue
            
            n_ep_i, n_ep_j = pair_array.shape[1], pair_array.shape[2]
            
            for ep_i in range(n_ep_i):
                for ep_j in range(n_ep_j):
                    dists = pair_array[:, ep_i, ep_j]
                    valid_dists = dists[~np.isnan(dists)]
                    
                    if len(valid_dists) < 3:
                        continue
                    
                    mean_dist = np.mean(valid_dists)
                    std_dist = np.std(valid_dists)
                    min_dist = np.min(valid_dists)
                    max_dist = np.max(valid_dists)
                    range_dist = max_dist - min_dist
                    cv = std_dist / mean_dist if mean_dist > 0 else 0
                    variation_score = cv * (range_dist / mean_dist if mean_dist > 0 else 0)
                    
                    variation_data.append({
                        'residue_i': i,
                        'residue_j': j,
                        'ep_i_idx': ep_i,
                        'ep_j_idx': ep_j,
                        'residue_i_sel': residue_sel_list[i] if i < len(residue_sel_list) else f"residue_{i}",
                        'residue_j_sel': residue_sel_list[j] if j < len(residue_sel_list) else f"residue_{j}",
                        'mean_distance': mean_dist,
                        'std_distance': std_dist,
                        'min_distance': min_dist,
                        'max_distance': max_dist,
                        'range_distance': range_dist,
                        'cv': cv,
                        'variation_score': variation_score,
                        'n_valid_points': len(valid_dists)
                    })
        
        df = pd.DataFrame(variation_data)
        if df.empty:
            return df
        
        df = df.sort_values('variation_score', ascending=False)
        return df
    
    @staticmethod
    def identify_key_endpoint_pairs_for_expansion(endpoint_dists_dict: dict,
                                                  volume: np.ndarray,
                                                  residue_sel_list: List[str],
                                                  variation_threshold: float = 0.1,
                                                  correlation_threshold: float = 0.7,
                                                  top_n: int = 10) -> pd.DataFrame:
        """Identify key endpoint pairs that represent cube expansion/shrinkage."""
        if not isinstance(endpoint_dists_dict, dict):
            warnings.warn("endpoint_dists_dict must be a dictionary from compute_endpoint_distances")
            return pd.DataFrame()
        
        all_pairs = endpoint_dists_dict.get('all_pairs', {})
        T = endpoint_dists_dict.get('n_frames', 0)
        n_res = endpoint_dists_dict.get('n_residues', 0)
        
        if T == 0 or n_res == 0:
            return pd.DataFrame()
        
        try:
            from scipy.stats import pearsonr
        except ImportError:
            warnings.warn("scipy not available. Cannot compute correlations.")
            return pd.DataFrame()
        
        variation_df = EndpointAnalyzer.analyze_endpoint_pair_variation(endpoint_dists_dict, residue_sel_list)
        if variation_df.empty:
            return pd.DataFrame()
        
        candidates = variation_df[variation_df['variation_score'] >= variation_threshold].copy()
        if candidates.empty:
            warnings.warn(f"No endpoint pairs found with variation_score >= {variation_threshold}")
            return pd.DataFrame()
        
        valid_volume_mask = ~np.isnan(volume)
        if not np.any(valid_volume_mask):
            warnings.warn("No valid volume values.")
            return pd.DataFrame()
        
        key_pairs = []
        
        for _, row in candidates.iterrows():
            i = int(row['residue_i'])
            j = int(row['residue_j'])
            ep_i = int(row['ep_i_idx'])
            ep_j = int(row['ep_j_idx'])
            
            if (i, j) not in all_pairs:
                continue
            
            pair_array = all_pairs[(i, j)]
            dists = pair_array[:, ep_i, ep_j]
            valid_mask = valid_volume_mask & (~np.isnan(dists))
            
            if np.sum(valid_mask) < 3:
                continue
            
            valid_dists = dists[valid_mask]
            valid_vol = volume[valid_mask]
            
            try:
                corr, p_val = pearsonr(valid_dists, valid_vol)
                
                if abs(corr) >= correlation_threshold:
                    key_pairs.append({
                        'residue_i': i,
                        'residue_j': j,
                        'ep_i_idx': ep_i,
                        'ep_j_idx': ep_j,
                        'residue_i_sel': residue_sel_list[i] if i < len(residue_sel_list) else f"residue_{i}",
                        'residue_j_sel': residue_sel_list[j] if j < len(residue_sel_list) else f"residue_{j}",
                        'correlation': corr,
                        'p_value': p_val,
                        'variation_score': row['variation_score'],
                        'cv': row['cv'],
                        'range_distance': row['range_distance'],
                        'mean_distance': row['mean_distance'],
                        'std_distance': row['std_distance'],
                        'n_valid_points': np.sum(valid_mask)
                    })
            except Exception:
                continue
        
        df = pd.DataFrame(key_pairs)
        if df.empty:
            return df
        
        df['abs_correlation'] = np.abs(df['correlation'])
        df = df.sort_values('abs_correlation', ascending=False)
        df = df.head(top_n)
        df = df.drop('abs_correlation', axis=1)
        
        return df


# ============================================================================
# Class: Plotter
# Purpose: Plotting functions
# ============================================================================

class Plotter:
    """Handle plotting operations for trajectory analysis."""
    
    def __init__(self):
        """Initialize Plotter with storage for plot configurations and outputs."""
        self.output_prefix: Optional[str] = None
        self.plots_generated: List[str] = []
        self.figure_size: Tuple[int, int] = (12, 8)
        self.dpi: int = 300
    
    def plot_endpoint_distances(self, endpoint_dists_array: Union[np.ndarray, dict],
                                residue_sel_list: List[str],
                                out_prefix: str) -> None:
        """Plot distances between endpoint pairs over frames."""
        if not HAS_MATPLOTLIB:
            warnings.warn("matplotlib not available. Skipping endpoint distance plots.")
            return
        
        if endpoint_dists_array is None:
            warnings.warn("Endpoint distances array is None. Skipping plots.")
            return
        
        if isinstance(endpoint_dists_array, dict):
            all_pairs = endpoint_dists_array.get('all_pairs', {})
            T = endpoint_dists_array.get('n_frames', 0)
            n_res = endpoint_dists_array.get('n_residues', 0)
            
            if not all_pairs:
                warnings.warn("Endpoint distances array is empty. Skipping plots.")
                return
            
            frames = np.arange(T)
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            for i in range(n_res):
                for j in range(i + 1, n_res):
                    if (i, j) in all_pairs:
                        pair_array = all_pairs[(i, j)]
                        mean_dists = []
                        for t in range(T):
                            frame_dists = pair_array[t]
                            valid_dists = frame_dists[~np.isnan(frame_dists)]
                            if len(valid_dists) > 0:
                                mean_dists.append(np.mean(valid_dists))
                            else:
                                mean_dists.append(np.nan)
                        mean_dists = np.array(mean_dists)
                        if not np.all(np.isnan(mean_dists)):
                            label = f"Res {i}-{j} (mean)"
                            ax.plot(frames, mean_dists, label=label, alpha=0.7, linewidth=1.5)
            
            ax.set_xlabel('Frame', fontsize=12)
            ax.set_ylabel('Endpoint Distance (Å)', fontsize=12)
            ax.set_title('Endpoint Pair Distances Over Frames (Mean)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
            plt.tight_layout()
            plot_path = f"{out_prefix}_endpoint_distances.png"
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            self.output_prefix = out_prefix
            self.plots_generated.append(plot_path)
            print(f"Endpoint distance plot saved to {plot_path}")
    
    def plot_endpoint_volume_correlation(self, endpoint_dists_array: Union[np.ndarray, dict],
                                         volume: np.ndarray,
                                         residue_sel_list: List[str],
                                         correlation_df: pd.DataFrame,
                                         out_prefix: str,
                                         top_n: int = 5) -> None:
        """Plot the top N endpoint pairs with highest correlation to volume."""
        if not HAS_MATPLOTLIB:
            warnings.warn("matplotlib not available. Skipping correlation plots.")
            return
        
        if endpoint_dists_array is None or correlation_df.empty:
            warnings.warn("No correlation data to plot.")
            return
        
        top_pairs = correlation_df.head(top_n)
        n_plots = min(top_n, len(top_pairs))
        if n_plots == 0:
            return
        
        fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
        if n_plots == 1:
            axes = [axes]
        
        frames = np.arange(len(volume))
        is_dict_format = isinstance(endpoint_dists_array, dict)
        if is_dict_format:
            all_pairs = endpoint_dists_array.get('all_pairs', {})
        
        for idx, (_, row) in enumerate(top_pairs.iterrows()):
            i, j = int(row['residue_i']), int(row['residue_j'])
            corr = row['correlation']
            p_val = row['p_value']
            ep_i_idx = row.get('ep_i_idx', None)
            ep_j_idx = row.get('ep_j_idx', None)
            
            ax = axes[idx]
            ax2 = ax.twinx()
            
            if is_dict_format and ep_i_idx is not None and ep_j_idx is not None and (i, j) in all_pairs:
                pair_array = all_pairs[(i, j)]
                dists = pair_array[:, int(ep_i_idx), int(ep_j_idx)]
                ep_label = f"ep{int(ep_i_idx)}-ep{int(ep_j_idx)}"
            elif is_dict_format:
                if (i, j) in all_pairs:
                    pair_array = all_pairs[(i, j)]
                    mean_dists = []
                    for t in range(len(volume)):
                        frame_dists = pair_array[t]
                        valid_dists = frame_dists[~np.isnan(frame_dists)]
                        if len(valid_dists) > 0:
                            mean_dists.append(np.mean(valid_dists))
                        else:
                            mean_dists.append(np.nan)
                    dists = np.array(mean_dists)
                    ep_label = "mean"
                else:
                    warnings.warn(f"Cannot extract distances for pair ({i}, {j})")
                    continue
            else:
                dists = endpoint_dists_array[:, i, j]
                ep_label = "min"
            
            line1 = ax.plot(frames, dists, 'b-',
                           label=f'Endpoint Distance ({row["residue_i_sel"]}-{row["residue_j_sel"]}, {ep_label})',
                           linewidth=2, alpha=0.8)
            line2 = ax2.plot(frames, volume, 'r-', label='Cube Volume', linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Frame', fontsize=11)
            ax.set_ylabel('Endpoint Distance (Å)', fontsize=11, color='b')
            ax2.set_ylabel('Volume (Å³)', fontsize=11, color='r')
            ax.tick_params(axis='y', labelcolor='b')
            ax2.tick_params(axis='y', labelcolor='r')
            
            title = f'Pair {i}-{j}'
            if ep_i_idx is not None and ep_j_idx is not None:
                title += f' (ep{int(ep_i_idx)}-ep{int(ep_j_idx)})'
            title += f': r={corr:.3f}, p={p_val:.3e}'
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', fontsize=9)
        
        plt.tight_layout()
        plot_path = f"{out_prefix}_endpoint_volume_correlation.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Endpoint-volume correlation plot saved to {plot_path}")
    
    def plot_volume_change(self, volume: np.ndarray,
                           out_prefix: str,
                           frame_indices: Optional[np.ndarray] = None,
                           times: Optional[np.ndarray] = None,
                           xlabel: str = 'Frame',
                           ylabel: str = 'Volume (Å³)',
                           title: Optional[str] = None,
                           show_stats: bool = True) -> None:
        """Plot volume change through frames.
        
        Args:
            volume: Array of volume values for each frame
            out_prefix: Output prefix for the plot file
            frame_indices: Optional array of frame indices (default: np.arange(len(volume)))
            times: Optional array of time values (if provided, xlabel will be 'Time (ps)')
            xlabel: Label for x-axis (default: 'Frame')
            ylabel: Label for y-axis (default: 'Volume (Å³)')
            title: Optional title for the plot (default: 'Volume Change Through Frames')
            show_stats: If True, display statistics in the plot (default: True)
        """
        if not HAS_MATPLOTLIB:
            warnings.warn("matplotlib not available. Skipping volume change plot.")
            return
        
        if volume is None or len(volume) == 0:
            warnings.warn("Volume array is None or empty. Skipping plot.")
            return
        
        # Determine x-axis values
        if times is not None and len(times) == len(volume):
            x_values = times
            xlabel = 'Time (ps)'
        elif frame_indices is not None and len(frame_indices) == len(volume):
            x_values = frame_indices
        else:
            x_values = np.arange(len(volume))
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot volume
        ax.plot(x_values, volume, 'b-', linewidth=2, alpha=0.8, label='Volume')
        
        # Add statistics if requested
        valid_volume = volume[~np.isnan(volume)]
        if show_stats and len(valid_volume) > 0:
            mean_vol = np.mean(valid_volume)
            std_vol = np.std(valid_volume)
            min_vol = np.min(valid_volume)
            max_vol = np.max(valid_volume)
            
            # Add horizontal lines for mean and std
            ax.axhline(mean_vol, color='r', linestyle='--', alpha=0.5, linewidth=1, label=f'Mean: {mean_vol:.2f} Å³')
            ax.axhline(mean_vol + std_vol, color='orange', linestyle=':', alpha=0.5, linewidth=1, label=f'Mean ± Std: {mean_vol:.2f} ± {std_vol:.2f} Å³')
            ax.axhline(mean_vol - std_vol, color='orange', linestyle=':', alpha=0.5, linewidth=1)
            
            # Add text box with statistics
            stats_text = f'Mean: {mean_vol:.2f} Å³\nStd: {std_vol:.2f} Å³\nMin: {min_vol:.2f} Å³\nMax: {max_vol:.2f} Å³'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Set labels and title
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if title is None:
            title = 'Volume Change Through Frames'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add legend if stats are shown
        if show_stats and len(valid_volume) > 0:
            ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        plot_path = f"{out_prefix}_volume_change.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Volume change plot saved to {plot_path}")
   
    def plot_residue_endpoints(self, u: mda.Universe, residue_sel: str = 'resid 1', out_prefix: str = 'residue_endpoints') -> None:
        """Plot the endpoints of a residue."""
        sel = u.select_atoms(residue_sel)
        mol = sel.convert_to('RDKIT')
        if mol is None:
            raise ValueError("RDKit conversion failed")
        
        mtest=Draw.MolToImage(EndpointsFinder().to_2d_coords(mol)[0], size=(600, 400), highlightAtoms=EndpointsFinder().find_endpoints(mol), highlightColor=(1, 0, 0))  # Red highlight
        mtest.save(out_prefix+'.png',format='png')
