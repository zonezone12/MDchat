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

if TYPE_CHECKING:
    from endpoints_finder import EndpointsFinder

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

# Import EndpointsFinder for residue endpoint analysis
try:
    from endpoints_finder import EndpointsFinder
    HAS_ENDPOINTS_FINDER = True
except ImportError:
    HAS_ENDPOINTS_FINDER = False
    EndpointsFinder = None  # type: ignore
    warnings.warn("endpoints_finder module not found. Endpoint-based metrics will be disabled.")

# Import VolumeAnalyzer for volume computation
try:
    from volume_analyser import VolumeAnalyzer
    HAS_VOLUME_ANALYZER = True
except ImportError:
    HAS_VOLUME_ANALYZER = False
    VolumeAnalyzer = None  # type: ignore
    warnings.warn("volume_analyser module not found. Volume computation will fallback to edge-based method.")


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
    
    def compute_rmsd(self, u: mda.Universe, sel_str: str, ref_frame: int = 0) -> np.ndarray:
        """Compute RMSD for each frame relative to reference frame."""
        cache_key = (sel_str, ref_frame)
        if cache_key in self.rmsd_cache:
            return self.rmsd_cache[cache_key]
        
        sel = u.select_atoms(sel_str)
        ref = sel.positions.copy()
        rmsds = []
        for ts in u.trajectory:
            R, rmsd_val = align.rotation_matrix(sel.positions, ref)
            rmsds.append(rmsd_val)
        result = np.array(rmsds)
        self.rmsd_cache[cache_key] = result
        return result
    
    def compute_rmsf(self, u: mda.Universe, sel_str: str, aligned: bool = True) -> np.ndarray:
        """Compute RMSF (per-atom root mean square fluctuation).
        
        Args:
            u: MDAnalysis Universe (should be pre-aligned via AlignedTrajectory)
            sel_str: Selection string for atoms to compute RMSF
            aligned: If True, assumes trajectory is already aligned (default: True)
                    If False, will align internally (not recommended)
        """
        cache_key = sel_str
        if cache_key in self.rmsf_cache:
            return self.rmsf_cache[cache_key]
        
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
    
    def radius_of_gyration(self, u: mda.Universe, sel_str: str) -> np.ndarray:
        """Compute radius of gyration for each frame."""
        cache_key = sel_str
        if cache_key in self.rg_cache:
            return self.rg_cache[cache_key]
        
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
    
    def contact_distances(self, u: mda.Universe, selA: str, selB: str) -> np.ndarray:
        """Compute minimal contact distance between two selections for each frame."""
        cache_key = (selA, selB)
        if cache_key in self.contact_cache:
            return self.contact_cache[cache_key]
        
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
    
    def pca_on_fluctuations(self, u: mda.Universe, sel_str: str, n_components: int = 5, aligned: bool = True) -> Tuple[np.ndarray, PCA]:
        """Return PC projections (T x n_comp) and the fitted PCA model.
        
        Args:
            u: MDAnalysis Universe (should be pre-aligned via AlignedTrajectory)
            sel_str: Selection string for atoms to use in PCA
            n_components: Number of principal components to compute
            aligned: If True, assumes trajectory is already aligned (default: True)
                    If False, will align internally (not recommended)
        """
        cache_key = (sel_str, n_components)
        if cache_key in self.pca_cache:
            return self.pca_cache[cache_key]
        
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
    
    def local_affine_strain_proxy(self, u: mda.Universe, sel_str: str, window: int = 10, lag: int = 1) -> np.ndarray:
        """Compute a lightweight proxy of local strain."""
        cache_key = (sel_str, window, lag)
        if cache_key in self.strain_cache:
            return self.strain_cache[cache_key]
        
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
# Class: GSAnalyzer
# Purpose: GSA nanocube specific analysis functions
# ============================================================================

class GSAnalyzer:
    """Analyze GSA nanocube structures and compute geometric metrics."""
    
    def __init__(self):
        """Initialize GSAnalyzer with storage for computed results."""
        self.nanocube_metrics_df: Optional[pd.DataFrame] = None
        self.face_selections: Optional[List[str]] = None
        self.planar_rms_cache: dict = {}
    
    def residue_planar_rms(self, atoms):
        """Compute planar RMS for a residue/atom group."""
        P = atoms.positions
        if len(P) < 3:
            return np.nan
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
                             out_prefix: Optional[str] = None) -> pd.DataFrame:
        """Compute geometric observables for GSA nanocube behavior."""
        faces = [u.select_atoms(s) for s in face_sel_list]
        corners = [u.select_atoms(s) for s in (corner_sel_list or [])]
        guest = u.select_atoms(guest_sel) if guest_sel else None
        
        # Combine all face selections for volume computation using VolumeAnalyzer
        combined_sel = " or ".join([f"({s})" for s in face_sel_list])
        
        # Initialize VolumeAnalyzer if available
        volume_analyzer = None
        if HAS_VOLUME_ANALYZER and VolumeAnalyzer is not None:
            try:
                volume_analyzer = VolumeAnalyzer(
                    universe=u,
                    selection=combined_sel,
                    spacing=1.0,
                    probe_radius=1.4
                )
            except Exception as e:
                warnings.warn(f"Failed to initialize VolumeAnalyzer: {e}. Falling back to edge-based volume computation.")
                volume_analyzer = None
        
        rows = []
        for ts in u.trajectory:
            fcent = np.array([ag.center_of_geometry() for ag in faces])
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
            for ag in faces:
                planar_rms, face_normal_vector = self.residue_planar_rms(ag)
                planar_rms_list.append(planar_rms)
                face_norms_list.append(face_normal_vector)
            
            face_norms_array = np.array(face_norms_list)
            face_norms_angle = np.degrees(np.arccos(np.clip(np.dot(face_norms_array, face_norms_array.T), -1.0, 1.0)))
            face_angle_dict = {}
            for i, face_i in enumerate(face_sel_list):
                for j, face_j in enumerate(face_sel_list):
                    if i != j:
                        face_angle_dict[f'{face_i}{face_j}_angle'] = face_norms_angle[i, j]
            
            cube_center = fcent.mean(axis=0) if len(fcent) else np.array([np.nan, np.nan, np.nan])
            guest_min = np.nan
            if guest is not None and len(guest):
                guest_min = float(np.linalg.norm(guest.positions - cube_center, axis=1).min())
            
            # Compute volume using VolumeAnalyzer if available, otherwise fallback to edge-based method
            if volume_analyzer is not None:
                try:
                    target_volume, cavity_volume = volume_analyzer.compute_frame(ts.frame, return_masks=False)
                    volume = target_volume+cavity_volume
                except Exception as e:
                    warnings.warn(f"VolumeAnalyzer failed for frame {ts.frame}: {e}. Using edge-based volume.")
                    volume = edge_mean ** 3 if not np.isnan(edge_mean) else np.nan
            else:
                volume = edge_mean ** 3 if not np.isnan(edge_mean) else np.nan
            
            rows.append({
                'frame': ts.frame,
                'edge_mean': edge_mean,
                'edge_min': float(np.nanmin(edge_len)) if len(edge_len) else np.nan,
                'edge_max': float(np.nanmax(edge_len)) if len(edge_len) else np.nan,
                'volume': volume,
                'planarity_mean': float(np.nanmean(planar_rms_list)),
                'planarity_max': float(np.nanmax(planar_rms_list)),
                'guest_min_center_dist': guest_min,
                **face_angle_dict,
            })
        df = pd.DataFrame(rows)
        self.nanocube_metrics_df = df
        self.face_selections = face_sel_list
        if out_prefix:
            df.to_csv(f"{out_prefix}_gsa_nanocube.csv", index=False)
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
        if not HAS_ENDPOINTS_FINDER:
            sel = u.select_atoms(sel_str)
            center = sel.center_of_geometry()
            distances = np.linalg.norm(sel.positions - center, axis=1)
            endpoints = np.argsort(distances)[::-1][:4].tolist()
            return center, endpoints
        
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
                                   endpoints_finder: Optional['EndpointsFinder'] = None) -> dict:
        """Compute distances between endpoints of different residues over trajectory."""
        # Store configuration
        self.residue_sel_list = residue_sel_list
        if endpoints_finder is not None:
            self._endpoints_finder = endpoints_finder
        
        if not HAS_ENDPOINTS_FINDER:
            warnings.warn("EndpointsFinder not available. Returning empty structure.")
            n_res = len(residue_sel_list)
            T = len(u.trajectory)
            self.endpoints_distance_dict = {
                'all_pairs': {},
                'n_residues': n_res,
                'n_frames': T
            }
            return self.endpoints_distance_dict
        
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
        for ts in u.trajectory:
            frame = ts.frame
            
            # Get endpoint positions for all residues at this frame
            frame_ep_positions = []
            for idx, sel_str in enumerate(residue_sel_list):
                try:
                    ep_indices = stored_ep_indices[idx]
                    
                    if len(ep_indices) > 0:
                        ep_positions = u.positions[ep_indices]
                        frame_ep_positions.append(ep_positions)
                    else:
                        frame_ep_positions.append(None)
                except Exception as e:
                    warnings.warn(f"Failed to get endpoint positions for {sel_str} at frame {frame}: {e}")
                    frame_ep_positions.append(None)
            
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
                                endpoints_finder: Optional['EndpointsFinder'] = None) -> pd.DataFrame:
        """Compute comprehensive endpoint-based metrics for residues over trajectory."""
        # Store configuration
        self.residue_sel_list = residue_sel_list
        if endpoints_finder is not None:
            self._endpoints_finder = endpoints_finder
        
        if not HAS_ENDPOINTS_FINDER:
            warnings.warn("EndpointsFinder not available. Returning empty DataFrame.")
            return pd.DataFrame({'frame': range(len(u.trajectory))})
        
        endpoint_dists_dict = self.compute_endpoint_distances(u, residue_sel_list, endpoints_finder)
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
