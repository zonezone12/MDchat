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


# ============================================================================
# Class: TrajectoryMetrics
# Purpose: Basic trajectory metric computations (RMSD, RMSF, Rg, contacts, PCA, strain)
# ============================================================================

class TrajectoryMetrics:
    """Compute basic trajectory metrics like RMSD, RMSF, radius of gyration, etc."""
    
    @staticmethod
    def compute_rmsd(u: mda.Universe, sel_str: str, ref_frame: int = 0) -> np.ndarray:
        """Compute RMSD for each frame relative to reference frame."""
        sel = u.select_atoms(sel_str)
        ref = sel.positions.copy()
        rmsds = []
        for ts in u.trajectory:
            R, rmsd_val = align.rotation_matrix(sel.positions, ref)
            rmsds.append(rmsd_val)
        return np.array(rmsds)
    
    @staticmethod
    def compute_rmsf(u: mda.Universe, sel_str: str) -> np.ndarray:
        """Compute RMSF (per-atom root mean square fluctuation)."""
        sel = u.select_atoms(sel_str)
        with mda.lib.util.tempdir.in_tempdir():
            align.AlignTraj(u, u, select=sel_str, in_memory=True).run()
        coords = []
        for ts in u.trajectory:
            coords.append(sel.positions.copy())
        coords = np.array(coords)  # (T, n, 3)
        mean = coords.mean(axis=0)
        diffsq = (coords - mean) ** 2
        rmsf = np.sqrt(diffsq.sum(axis=2).mean(axis=0))
        return rmsf
    
    @staticmethod
    def radius_of_gyration(u: mda.Universe, sel_str: str) -> np.ndarray:
        """Compute radius of gyration for each frame."""
        sel = u.select_atoms(sel_str)
        rgs = []
        for ts in u.trajectory:
            coords = sel.positions
            com = sel.center_of_mass()
            rg2 = ((coords - com) ** 2).sum(axis=1).mean()
            rgs.append(np.sqrt(rg2))
        return np.array(rgs)
    
    @staticmethod
    def contact_distances(u: mda.Universe, selA: str, selB: str) -> np.ndarray:
        """Compute minimal contact distance between two selections for each frame."""
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
        return np.array(dists)
    
    @staticmethod
    def pca_on_fluctuations(u: mda.Universe, sel_str: str, n_components: int = 5) -> Tuple[np.ndarray, PCA]:
        """Return PC projections (T x n_comp) and the fitted PCA model."""
        sel = u.select_atoms(sel_str)
        align.AlignTraj(u, u, select=sel_str, in_memory=True).run()
        coords = []
        for ts in u.trajectory:
            coords.append(sel.positions.copy().reshape(-1))
        X = np.array(coords)  # shape (T, 3N)
        Xc = X - X.mean(axis=0)
        pca = PCA(n_components=n_components, svd_solver="auto")
        pcs = pca.fit_transform(Xc)
        return pcs, pca
    
    @staticmethod
    def local_affine_strain_proxy(u: mda.Universe, sel_str: str, window: int = 10, lag: int = 1) -> np.ndarray:
        """Compute a lightweight proxy of local strain."""
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
        return strain


# ============================================================================
# Class: ClusteringAnalysis
# Purpose: Clustering and change-point detection
# ============================================================================

class ClusteringAnalysis:
    """Perform clustering analysis and change-point detection on trajectory data."""
    
    @staticmethod
    def zscore(x: np.ndarray) -> np.ndarray:
        """Compute z-score normalization."""
        m, s = np.nanmean(x), np.nanstd(x)
        if s == 0 or np.isnan(s):
            return np.zeros_like(x)
        return (x - m) / s
    
    @staticmethod
    def choose_k_by_silhouette(X, kmin=2, kmax=10) -> int:
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
    
    @staticmethod
    def cluster_frames(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cluster frame embeddings (e.g., PCs). Returns (labels, medoid_indices)."""
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
            return labels, np.array(medoids, dtype=int)
        else:
            k = ClusteringAnalysis.choose_k_by_silhouette(X)
            km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
            labels = km.labels_
            medoids = []
            for cl in range(k):
                idx = np.where(labels == cl)[0]
                centroid = X[idx].mean(axis=0)
                d = np.linalg.norm(X[idx] - centroid, axis=1)
                medoids.append(idx[np.argmin(d)])
            return labels, np.array(medoids, dtype=int)
    
    @staticmethod
    def change_points(signal: np.ndarray, model: str = "rbf", n_bkps: int = 5) -> List[int]:
        """Return change-point indices for a 1D signal."""
        if HAS_RUPTURES:
            algo = rpt.Binseg(model=model).fit(signal.reshape(-1, 1))
            bkps = algo.predict(n_bkps=n_bkps)
            return sorted(set([b for b in bkps if b < len(signal)]))
        # Fallback: pick top-N peaks of absolute derivative
        deriv = np.abs(np.gradient(signal))
        N = max(2, n_bkps)
        idx = np.argsort(deriv)[-N:]
        return sorted(set(idx.tolist()))


# ============================================================================
# Class: FrameSelection
# Purpose: Frame scoring and selection
# ============================================================================

class FrameSelection:
    """Score and select meaningful frames from trajectory."""
    
    @staticmethod
    def score_frames(rmsd: np.ndarray,
                     rg: np.ndarray,
                     pcs: np.ndarray,
                     cpd_idx: List[int],
                     strain: Optional[np.ndarray] = None,
                     w: Tuple[float, float, float, float] = (0.35, 0.2, 0.35, 0.1)) -> Tuple[np.ndarray, pd.DataFrame]:
        """Combine multiple criteria into a single score."""
        T = len(rmsd)
        clustering = ClusteringAnalysis()
        
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
        return score, df
    
    @staticmethod
    def load_data_from_csv(scores_csv: Optional[str] = None,
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
        
        return medoids, cpd_idx, scores, pcs, score_df, endpoint_metrics_df
    
    @staticmethod
    def select_meaningful_frames(medoids: Optional[np.ndarray] = None,
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
        # Load data from CSV files if provided
        if (scores_csv or metrics_csv or endpoint_metrics_csv) and (medoids is None or cpd_idx is None or scores is None or pcs is None):
            print("Loading data from CSV files...")
            loaded_medoids, loaded_cpd_idx, loaded_scores, loaded_pcs, loaded_score_df, loaded_endpoint_metrics_df = FrameSelection.load_data_from_csv(
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
        
        return chosen_sorted, score_df


# ============================================================================
# Class: FileIO
# Purpose: File input/output operations
# ============================================================================

class FileIO:
    """Handle file input/output operations."""
    
    @staticmethod
    def save_frames_as_pdb(u: mda.Universe, frame_indices: List[int], out_prefix: str, sel: Optional[str] = None):
        """Save selected frames as PDB files."""
        os.makedirs(f"{out_prefix}_frames", exist_ok=True)
        if sel is None:
            ag = u.atoms
        else:
            ag = u.select_atoms(sel)
        for idx in frame_indices:
            u.trajectory[idx]
            ag.write(f"{out_prefix}_frames/frame_{idx:06d}.pdb")


# ============================================================================
# Class: GSAnalyzer
# Purpose: GSA nanocube specific analysis functions
# ============================================================================

class GSAnalyzer:
    """Analyze GSA nanocube structures and compute geometric metrics."""
    
    @staticmethod
    def residue_planar_rms(atoms):
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
    
    @staticmethod
    def gsa_nanocube_metrics(u: mda.Universe,
                             face_sel_list: List[str],
                             corner_sel_list: Optional[List[str]] = None,
                             guest_sel: Optional[str] = None,
                             out_prefix: Optional[str] = None) -> pd.DataFrame:
        """Compute geometric observables for GSA nanocube behavior."""
        faces = [u.select_atoms(s) for s in face_sel_list]
        corners = [u.select_atoms(s) for s in (corner_sel_list or [])]
        guest = u.select_atoms(guest_sel) if guest_sel else None
        
        def center_of_face(ag):
            return ag.center_of_geometry()
        
        rows = []
        for ts in u.trajectory:
            fcent = np.array([center_of_face(ag) for ag in faces])
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
                planar_rms, face_normal_vector = GSAnalyzer.residue_planar_rms(ag)
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
        if out_prefix:
            df.to_csv(f"{out_prefix}_gsa_nanocube.csv", index=False)
        return df
    
    @staticmethod
    def amber_preset_selections(u: mda.Universe,
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
    
    @staticmethod
    def gsa_auto_faces_by_kmeans(u: mda.Universe,
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
            return faces
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
            return faces
    
    @staticmethod
    def faces_from_atomname_blocks(u: 'mda.Universe',
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
        return faces


# ============================================================================
# Class: EndpointAnalyzer
# Purpose: Endpoint-based analysis functions
# ============================================================================

class EndpointAnalyzer:
    """Analyze molecular endpoints and compute endpoint-based metrics."""
    
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
        
        mda_endpoint_indices = []
        for rdkit_idx in endpoint_indices:
            atom = mol.GetAtomWithIdx(rdkit_idx)
            if atom.HasProp('_MDAnalysis_index'):
                mda_idx = atom.GetIntProp('_MDAnalysis_index')
                sel_idx = np.where(sel.indices == mda_idx)[0]
                if len(sel_idx) > 0:
                    mda_endpoint_indices.append(int(sel_idx[0]))
        
        center = sel.center_of_geometry()
        return center, mda_endpoint_indices if mda_endpoint_indices else []
    
    @staticmethod
    def compute_endpoint_distances(u: mda.Universe,
                                   residue_sel_list: List[str],
                                   endpoints_finder: Optional['EndpointsFinder'] = None) -> dict:
        """Compute distances between endpoints of different residues over trajectory."""
        if not HAS_ENDPOINTS_FINDER:
            warnings.warn("EndpointsFinder not available. Returning empty structure.")
            n_res = len(residue_sel_list)
            T = len(u.trajectory)
            return {
                'all_pairs': {},
                'n_residues': n_res,
                'n_frames': T
            }
        
        if endpoints_finder is None:
            endpoints_finder = EndpointsFinder()
        
        n_res = len(residue_sel_list)
        T = len(u.trajectory)
        all_pairs = {}
        
        all_residue_endpoints = []
        all_residue_positions = []
        
        for ts in u.trajectory:
            frame = ts.frame
            frame_endpoints = []
            frame_positions = []
            
            for sel_str in residue_sel_list:
                try:
                    center, ep_indices = EndpointAnalyzer.find_residue_endpoints(u, sel_str, endpoints_finder)
                    sel = u.select_atoms(sel_str)
                    if len(ep_indices) > 0 and len(sel) > 0:
                        ep_positions = sel.positions[ep_indices]
                        frame_endpoints.append(ep_positions)
                        frame_positions.append(center)
                    else:
                        frame_endpoints.append(None)
                        frame_positions.append(center)
                except Exception as e:
                    warnings.warn(f"Failed to find endpoints for {sel_str} at frame {frame}: {e}")
                    frame_endpoints.append(None)
                    frame_positions.append(np.array([np.nan, np.nan, np.nan]))
            
            all_residue_endpoints.append(frame_endpoints)
            all_residue_positions.append(frame_positions)
        
        for i in range(n_res):
            for j in range(i + 1, n_res):
                max_ep_i = 0
                max_ep_j = 0
                for frame in range(T):
                    if all_residue_endpoints[frame][i] is not None:
                        max_ep_i = max(max_ep_i, len(all_residue_endpoints[frame][i]))
                    if all_residue_endpoints[frame][j] is not None:
                        max_ep_j = max(max_ep_j, len(all_residue_endpoints[frame][j]))
                
                if max_ep_i > 0 and max_ep_j > 0:
                    pair_distances = np.full((T, max_ep_i, max_ep_j), np.nan)
                    
                    for frame in range(T):
                        ep_i = all_residue_endpoints[frame][i]
                        ep_j = all_residue_endpoints[frame][j]
                        
                        if ep_i is not None and ep_j is not None:
                            diff = ep_i[:, None, :] - ep_j[None, :, :]
                            dists = np.linalg.norm(diff, axis=2)
                            n_ep_i_actual, n_ep_j_actual = dists.shape
                            pair_distances[frame, :n_ep_i_actual, :n_ep_j_actual] = dists
                    
                    all_pairs[(i, j)] = pair_distances
                    all_pairs[(j, i)] = pair_distances.transpose(0, 2, 1)
        
        return {
            'all_pairs': all_pairs,
            'n_residues': n_res,
            'n_frames': T
        }
    
    @staticmethod
    def compute_endpoint_metrics(u: mda.Universe,
                                residue_sel_list: List[str],
                                endpoints_finder: Optional['EndpointsFinder'] = None) -> pd.DataFrame:
        """Compute comprehensive endpoint-based metrics for residues over trajectory."""
        if not HAS_ENDPOINTS_FINDER:
            warnings.warn("EndpointsFinder not available. Returning empty DataFrame.")
            return pd.DataFrame({'frame': range(len(u.trajectory))})
        
        endpoint_dists_dict = EndpointAnalyzer.compute_endpoint_distances(u, residue_sel_list, endpoints_finder)
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
        
        return pd.DataFrame(rows)
    
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
    
    @staticmethod
    def plot_endpoint_distances(endpoint_dists_array: Union[np.ndarray, dict],
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
            fig, ax = plt.subplots(figsize=(12, 8))
            
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
            plt.savefig(f"{out_prefix}_endpoint_distances.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Endpoint distance plot saved to {out_prefix}_endpoint_distances.png")
    
    @staticmethod
    def plot_endpoint_volume_correlation(endpoint_dists_array: Union[np.ndarray, dict],
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
        plt.savefig(f"{out_prefix}_endpoint_volume_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Endpoint-volume correlation plot saved to {out_prefix}_endpoint_volume_correlation.png")
