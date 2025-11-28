#!/usr/bin/env python3
"""
GSAnalyzer Module
-----------------

GSA nanocube specific analysis functions for computing geometric metrics.
"""

import warnings
from typing import Optional, List
import numpy as np
import pandas as pd

try:
    import MDAnalysis as mda
except ImportError as e:
    raise ImportError("MDAnalysis is required. pip install MDAnalysis") from e

from sklearn.cluster import KMeans

# Import VolumeAnalyzer for volume computation
try:
    # Try absolute import first (works when imported from outside the package)
    from MD_analysis.volume_analyser import VolumeAnalyzer
    HAS_VOLUME_ANALYZER = True
except ImportError:
    try:
        # Fallback to relative import (works when run from within the directory)
        from volume_analyser import VolumeAnalyzer
        HAS_VOLUME_ANALYZER = True
    except ImportError:
        HAS_VOLUME_ANALYZER = False
        VolumeAnalyzer = None  # type: ignore
        raise ImportError("volume_analyser module not found. Volume computation will fallback to edge-based method.")


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
                             out_prefix: Optional[str] = None,
                             gatherer: Optional[object] = None) -> pd.DataFrame:
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
        if gatherer is not None:
            # Use pre-gathered coordinates
            frame_indices = gatherer.get_frame_indices()
            for frame_idx, frame_num in enumerate(frame_indices):
                ts_frame = int(frame_num)
                # Get face centers from gathered coordinates
                fcent = []
                for face_sel in face_sel_list:
                    try:
                        face_coords = gatherer.get_coordinates(face_sel)[frame_idx]
                        fcent.append(face_coords.mean(axis=0))
                    except (ValueError, KeyError):
                        # Fallback to original method if selection not in gatherer
                        ag = u.select_atoms(face_sel)
                        u.trajectory[ts_frame]
                        fcent.append(ag.center_of_geometry())
                fcent = np.array(fcent)
                
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
                for idx, face_sel in enumerate(face_sel_list):
                    try:
                        face_coords = gatherer.get_coordinates(face_sel)[frame_idx]
                        # Manually compute planar RMS from coordinates
                        P = face_coords
                        if len(P) < 3:
                            planar_rms = np.nan
                            face_normal_vector = np.array([np.nan, np.nan, np.nan])
                        else:
                            P0 = P - P.mean(axis=0)
                            U, S, Vt = np.linalg.svd(P0, full_matrices=False)
                            face_normal_vector = Vt[-1]
                            dist = np.abs(P0 @ face_normal_vector)
                            planar_rms = float(np.sqrt((dist**2).mean()))
                        planar_rms_list.append(planar_rms)
                        face_norms_list.append(face_normal_vector)
                    except (ValueError, KeyError):
                        # Fallback to original method
                        ag = faces[idx]
                        u.trajectory[ts_frame]
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
                if guest is not None:
                    u.trajectory[ts_frame]
                    if len(guest):
                        guest_min = float(np.linalg.norm(guest.positions - cube_center, axis=1).min())
                
                # Compute volume using VolumeAnalyzer if available, otherwise fallback to edge-based method
                if volume_analyzer is not None:
                    try:
                        target_volume, cavity_volume = volume_analyzer.compute_frame(ts_frame, return_masks=False)
                        volume = target_volume+cavity_volume
                    except Exception as e:
                        warnings.warn(f"VolumeAnalyzer failed for frame {ts_frame}: {e}. Using edge-based volume.")
                        volume = edge_mean ** 3 if not np.isnan(edge_mean) else np.nan
                else:
                    volume = edge_mean ** 3 if not np.isnan(edge_mean) else np.nan
                
                rows.append({
                    'frame': ts_frame,
                    'edge_mean': edge_mean,
                    'edge_min': float(np.nanmin(edge_len)) if len(edge_len) else np.nan,
                    'edge_max': float(np.nanmax(edge_len)) if len(edge_len) else np.nan,
                    'volume': volume,
                    'planarity_mean': float(np.nanmean(planar_rms_list)),
                    'planarity_max': float(np.nanmax(planar_rms_list)),
                    'guest_min_center_dist': guest_min,
                    **face_angle_dict,
                })
        else:
            # Original iteration-based approach
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

