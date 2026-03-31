from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    import MDAnalysis as mda
except ImportError:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

from .endpoints_finder import EndpointsFinder  # type: ignore
from ..TrajectoryIterator import FrameObserver, TrajectoryIterator  # type: ignore
from ..Aggregator import ResultsGroup  # type: ignore

if TYPE_CHECKING:
    from ..FrameGatherer import FrameGatherer  # type: ignore


class EndpointAnalyzer:
    """Analyze molecular endpoints and compute endpoint-based metrics."""

    def __init__(self) -> None:
        self.endpoints_distance_dict: Optional[dict] = None
        self.endpoint_metrics_df: Optional[pd.DataFrame] = None
        self.residue_sel_list: Optional[List[str]] = None
        self._endpoints_finder: Optional[EndpointsFinder] = None  # type: ignore[name-defined]

    def clear_cache(self) -> None:
        self.endpoints_distance_dict = None
        self.endpoint_metrics_df = None
        self.residue_sel_list = None
        self._endpoints_finder = None

    @staticmethod
    def find_residue_endpoints(
        u: mda.Universe,
        sel_str: str,
        endpoints_finder: Optional[EndpointsFinder] = None,  # type: ignore[name-defined]
    ) -> Tuple[np.ndarray, List[int]]:
        """Find the endpoints of a residue using EndpointsFinder."""

        if endpoints_finder is None:
            endpoints_finder = EndpointsFinder()  # type: ignore[call-arg]

        sel = u.select_atoms(sel_str)
        if len(sel) == 0:
            return np.array([np.nan, np.nan, np.nan]), []

        mol = sel.convert_to("RDKIT")
        if mol is None:
            raise ValueError("RDKit conversion failed")

        endpoint_indices = endpoints_finder.find_endpoints(mol)
        mda_endpoint_indices = []
        for rdkit_idx in endpoint_indices:
            mda_endpoint_indices.append(int(sel[rdkit_idx].id))

        center = sel.center_of_mass()
        return center, mda_endpoint_indices if mda_endpoint_indices else []

    def compute_endpoint_distances(
        self,
        u: mda.Universe,
        residue_sel_list: List[str],
        endpoints_finder: Optional[EndpointsFinder] = None,  # type: ignore[name-defined]
        gatherer: Optional["FrameGatherer"] = None,  # type: ignore[name-defined]
    ) -> dict:
        """Compute distances between endpoints of different residues over trajectory."""
        from ..FrameGatherer import FrameGatherer  # local import to avoid cycle

        self.residue_sel_list = residue_sel_list
        if endpoints_finder is not None:
            self._endpoints_finder = endpoints_finder


        if endpoints_finder is None:
            endpoints_finder = EndpointsFinder()  # type: ignore[call-arg]

        n_res = len(residue_sel_list)
        T = len(u.trajectory)
        all_pairs: dict = {}
        # Find endpoint indices once on the initial frame (residues don't change)
        stored_ep_indices: List[List[int]] = []
        u.trajectory[0] # Go to initial frame
        for sel_str in residue_sel_list:
            try:
                _, ep_indices = EndpointAnalyzer.find_residue_endpoints(
                    u, sel_str, endpoints_finder
                )
                stored_ep_indices.append(ep_indices)
            except Exception as e:
                warnings.warn(
                    f"Failed to find endpoints for {sel_str} on initial frame: {e}"
                )
                stored_ep_indices.append([])

        # Determine max number of endpoints per residue from stored indices
        max_ep_per_residue = [len(ep_indices) for ep_indices in stored_ep_indices]

        for i in range(n_res):
            for j in range(i + 1, n_res):
                max_ep_i = max_ep_per_residue[i]
                max_ep_j = max_ep_per_residue[j]
                if max_ep_i > 0 and max_ep_j > 0:
                    pair_distances = np.full((T, max_ep_i, max_ep_j), np.nan)
                    all_pairs[(i, j)] = pair_distances
                    all_pairs[(j, i)] = np.full((T, max_ep_j, max_ep_i), np.nan)

        if gatherer is not None:
            frame_indices = gatherer.get_frame_indices()
            for frame_idx, frame_num in enumerate(frame_indices):
                frame = int(frame_num)
                frame_ep_positions = []
                for idx, sel_str in enumerate(residue_sel_list):
                    try:
                        ep_indices = stored_ep_indices[idx]
                        if len(ep_indices) > 0:
                            residue_coords = gatherer.get_coordinates(sel_str)[
                                frame_idx
                            ]
                            sel = u.select_atoms(sel_str)
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
                        raise ValueError(
                            f"Failed to get endpoint positions for {sel_str} at frame {frame}: {e}"
                        )

                for i in range(n_res):
                    for j in range(i + 1, n_res):
                        ep_i = frame_ep_positions[i]
                        ep_j = frame_ep_positions[j]
                        if ep_i is not None and ep_j is not None and (i, j) in all_pairs:
                            diff = ep_i[:, None, :] - ep_j[None, :, :]
                            dists = np.linalg.norm(diff, axis=2)
                            n_ep_i_actual, n_ep_j_actual = dists.shape
                            all_pairs[(i, j)][
                                frame_idx, :n_ep_i_actual, :n_ep_j_actual
                            ] = dists
                            all_pairs[(j, i)][
                                frame_idx, :n_ep_j_actual, :n_ep_i_actual
                            ] = dists.T
        else:
            for ts in u.trajectory:
                frame = ts.frame
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
                        raise ValueError(
                            f"Failed to get endpoint positions for {sel_str} at frame {frame}: {e}"
                        )

                for i in range(n_res):
                    for j in range(i + 1, n_res):
                        ep_i = frame_ep_positions[i]
                        ep_j = frame_ep_positions[j]
                        if ep_i is not None and ep_j is not None and (i, j) in all_pairs:
                            diff = ep_i[:, None, :] - ep_j[None, :, :]
                            dists = np.linalg.norm(diff, axis=2)
                            n_ep_i_actual, n_ep_j_actual = dists.shape
                            all_pairs[(i, j)][
                                frame, :n_ep_i_actual, :n_ep_j_actual
                            ] = dists
                            all_pairs[(j, i)][
                                frame, :n_ep_j_actual, :n_ep_i_actual
                            ] = dists.T

        self.endpoints_distance_dict = {
            "all_pairs": all_pairs,
            "n_residues": n_res,
            "n_frames": T,
        }
        return self.endpoints_distance_dict

    def compute_endpoint_metrics(
        self,
        u: mda.Universe,
        residue_sel_list: List[str],
        endpoints_finder: Optional[EndpointsFinder] = None,  # type: ignore[name-defined]
        gatherer: Optional["FrameGatherer"] = None,  # type: ignore[name-defined]
    ) -> pd.DataFrame:
        """Compute comprehensive endpoint-based metrics for residues over trajectory."""
        endpoint_dists_dict = self.compute_endpoint_distances(
            u, residue_sel_list, endpoints_finder, gatherer=gatherer
        )
        all_pairs = endpoint_dists_dict["all_pairs"]
        T = endpoint_dists_dict["n_frames"]
        n_res = endpoint_dists_dict["n_residues"]

        rows = []
        for frame in range(T):
            row = {"frame": frame}
            all_pair_dists = []
            for i in range(n_res):
                for j in range(i + 1, n_res):
                    if (i, j) in all_pairs:
                        pair_array = all_pairs[(i, j)][frame]
                        valid_dists = pair_array[~np.isnan(pair_array)]
                        all_pair_dists.extend(valid_dists.tolist())
                        if len(valid_dists) > 0:
                            row[f"endpoint_dist_{i}_{j}_mean"] = float(
                                np.mean(valid_dists)
                            )
                            row[f"endpoint_dist_{i}_{j}_min"] = float(
                                np.min(valid_dists)
                            )
                            row[f"endpoint_dist_{i}_{j}_max"] = float(
                                np.max(valid_dists)
                            )
                        else:
                            row[f"endpoint_dist_{i}_{j}_mean"] = np.nan
                            row[f"endpoint_dist_{i}_{j}_min"] = np.nan
                            row[f"endpoint_dist_{i}_{j}_max"] = np.nan

            if len(all_pair_dists) > 0:
                all_pair_dists = np.array(all_pair_dists)
                row["endpoint_dist_mean"] = float(np.mean(all_pair_dists))
                row["endpoint_dist_min"] = float(np.min(all_pair_dists))
                row["endpoint_dist_max"] = float(np.max(all_pair_dists))
                row["endpoint_dist_std"] = float(np.std(all_pair_dists))
            else:
                row["endpoint_dist_mean"] = np.nan
                row["endpoint_dist_min"] = np.nan
                row["endpoint_dist_max"] = np.nan
                row["endpoint_dist_std"] = np.nan

            rows.append(row)

        self.endpoint_metrics_df = pd.DataFrame(rows)
        return self.endpoint_metrics_df

    @staticmethod
    def compute_endpoint_volume_correlation(
        endpoint_dists_array: Union[np.ndarray, dict],
        volume: np.ndarray,
        residue_sel_list: List[str],
    ) -> pd.DataFrame:
        """Compute correlation between each endpoint pair distance and cube volume."""
        if endpoint_dists_array is None:
            return pd.DataFrame(
                columns=[
                    "residue_i",
                    "residue_j",
                    "ep_i_idx",
                    "ep_j_idx",
                    "correlation",
                    "p_value",
                    "n_valid_points",
                ]
            )

        valid_volume_mask = ~np.isnan(volume)
        if not np.any(valid_volume_mask):
            warnings.warn("No valid volume values. Cannot compute correlations.")
            return pd.DataFrame(
                columns=[
                    "residue_i",
                    "residue_j",
                    "ep_i_idx",
                    "ep_j_idx",
                    "correlation",
                    "p_value",
                    "n_valid_points",
                ]
            )

        try:
            from scipy.stats import pearsonr
        except ImportError:
            warnings.warn("scipy not available. Cannot compute correlations.")
            return pd.DataFrame(
                columns=[
                    "residue_i",
                    "residue_j",
                    "ep_i_idx",
                    "ep_j_idx",
                    "correlation",
                    "p_value",
                    "n_valid_points",
                ]
            )

        correlations = []
        if isinstance(endpoint_dists_array, dict):
            all_pairs = endpoint_dists_array.get("all_pairs", {})
            T = endpoint_dists_array.get("n_frames", 0)
            _ = T  # unused but kept for clarity

            for (i, j), pair_array in all_pairs.items():
                if i >= j:
                    continue
                n_ep_i, n_ep_j = pair_array.shape[1], pair_array.shape[2]
                for ep_i in range(n_ep_i):
                    for ep_j in range(n_ep_j):
                        dists = pair_array[:, ep_i, ep_j]
                        valid_mask = valid_volume_mask & (~np.isnan(dists))
                        if np.sum(valid_mask) < 3:
                            correlations.append(
                                {
                                    "residue_i": i,
                                    "residue_j": j,
                                    "ep_i_idx": ep_i,
                                    "ep_j_idx": ep_j,
                                    "residue_i_sel": residue_sel_list[i]
                                    if i < len(residue_sel_list)
                                    else f"residue_{i}",
                                    "residue_j_sel": residue_sel_list[j]
                                    if j < len(residue_sel_list)
                                    else f"residue_{j}",
                                    "correlation": np.nan,
                                    "p_value": np.nan,
                                    "n_valid_points": np.sum(valid_mask),
                                }
                            )
                            continue

                        valid_dists = dists[valid_mask]
                        valid_vol = volume[valid_mask]
                        try:
                            corr, p_val = pearsonr(valid_dists, valid_vol)
                            correlations.append(
                                {
                                    "residue_i": i,
                                    "residue_j": j,
                                    "ep_i_idx": ep_i,
                                    "ep_j_idx": ep_j,
                                    "residue_i_sel": residue_sel_list[i]
                                    if i < len(residue_sel_list)
                                    else f"residue_{i}",
                                    "residue_j_sel": residue_sel_list[j]
                                    if j < len(residue_sel_list)
                                    else f"residue_{j}",
                                    "correlation": corr,
                                    "p_value": p_val,
                                    "n_valid_points": np.sum(valid_mask),
                                }
                            )
                        except Exception as e:
                            warnings.warn(
                                f"Correlation computation failed for pair ({i}, {j}), "
                                f"endpoints ({ep_i}, {ep_j}): {e}"
                            )
                            correlations.append(
                                {
                                    "residue_i": i,
                                    "residue_j": j,
                                    "ep_i_idx": ep_i,
                                    "ep_j_idx": ep_j,
                                    "residue_i_sel": residue_sel_list[i]
                                    if i < len(residue_sel_list)
                                    else f"residue_{i}",
                                    "residue_j_sel": residue_sel_list[j]
                                    if j < len(residue_sel_list)
                                    else f"residue_{j}",
                                    "correlation": np.nan,
                                    "p_value": np.nan,
                                    "n_valid_points": np.sum(valid_mask),
                                }
                            )

        df = pd.DataFrame(correlations)
        if df.empty or "correlation" not in df.columns:
            return pd.DataFrame(
                columns=[
                    "residue_i",
                    "residue_j",
                    "ep_i_idx",
                    "ep_j_idx",
                    "residue_i_sel",
                    "residue_j_sel",
                    "correlation",
                    "p_value",
                    "n_valid_points",
                ]
            )

        df["abs_correlation"] = np.abs(df["correlation"].fillna(0))
        df = df.sort_values("abs_correlation", ascending=False)
        df = df.drop("abs_correlation", axis=1)
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


class EndpointAnalyzerObserver(FrameObserver):
    """
    Observer implementation of EndpointAnalyzer for single-pass trajectory iteration.
    
    Processes endpoint distances during trajectory iteration without storing
    coordinates in memory.
    
    Uses ResultsGroup for declarative result aggregation in parallel processing.
    Results are stored in self.results['all_pairs'].
    """
    
    def __init__(
        self,
        residue_sel_list: List[str],
        endpoints_finder: Optional[EndpointsFinder] = None,
    ):
        """
        Initialize EndpointAnalyzerObserver.
        
        Args:
            residue_sel_list: List of residue selection strings
            endpoints_finder: Optional EndpointsFinder instance
        """
        super().__init__()  # Initialize results dict from FrameObserver
        
        self.residue_sel_list = residue_sel_list
        self.endpoints_finder = endpoints_finder or EndpointsFinder()
        
        # Storage for results (using self.results for ResultsGroup pattern)
        self.stored_ep_indices: List[List[int]] = []
        self.results['all_pairs'] = {}  # Store in results dict for aggregation
        self.n_res: int = 0
        self.n_frames: int = 0
        self._initialized = False
        
        # Reference to analyzer for utility methods
        self._analyzer = EndpointAnalyzer()
    
    @property
    def all_pairs(self) -> dict:
        """Property to access all_pairs from results dict for backward compatibility."""
        return self.results.get('all_pairs', {})
    
    @all_pairs.setter
    def all_pairs(self, value: dict) -> None:
        """Setter for all_pairs to store in results dict."""
        self.results['all_pairs'] = value
    
    def get_selections_needed(self) -> List[str]:
        """Return list of selection strings needed by this observer."""
        return self.residue_sel_list.copy()
    
    def on_frame_start(self, iterator: TrajectoryIterator) -> None:
        """Initialize data structures before iteration."""
        self.n_res = len(self.residue_sel_list)
        self.n_frames = iterator.get_n_frames()
        
        # Find endpoint indices once on the initial frame
        u = iterator.universe
        #seems not needed to be at the initial frame in the iterator also save the time
        #u.trajectory[0]  # Go to initial frame 
        
        self.stored_ep_indices = []
        for idx, sel_str in enumerate(self.residue_sel_list):
            try:
                _, ep_indices = EndpointAnalyzer.find_residue_endpoints(
                    u, sel_str, self.endpoints_finder
                )
                self.stored_ep_indices.append(ep_indices)
                if len(ep_indices) == 0:
                    warnings.warn(
                        f"No endpoints found for residue {idx} ({sel_str}) on initial frame. "
                        f"Check selection string and EndpointsFinder parameters."
                    )
                else:
                    print(f"  Residue {idx} ({sel_str}): Found {len(ep_indices)} endpoint(s)")
            except Exception as e:
                warnings.warn(
                    f"Failed to find endpoints for {sel_str} on initial frame: {e}"
                )
                import traceback
                traceback.print_exc()
                self.stored_ep_indices.append([])
        
        # Determine max number of endpoints per residue
        max_ep_per_residue = [len(ep_indices) for ep_indices in self.stored_ep_indices]
        
        # Initialize distance arrays
        self.all_pairs = {}
        pairs_initialized = 0
        for i in range(self.n_res):
            for j in range(i + 1, self.n_res):
                max_ep_i = max_ep_per_residue[i]
                max_ep_j = max_ep_per_residue[j]
                if max_ep_i > 0 and max_ep_j > 0:
                    pair_distances = np.full((self.n_frames, max_ep_i, max_ep_j), np.nan)
                    self.all_pairs[(i, j)] = pair_distances
                    self.all_pairs[(j, i)] = np.full((self.n_frames, max_ep_j, max_ep_i), np.nan)
                    pairs_initialized += 1
        
        if pairs_initialized == 0:
            warnings.warn(
                f"No endpoint pairs initialized. This can happen if:\n"
                f"  1. No endpoints were found for any residue\n"
                f"  2. Only one residue was provided (need at least 2 for pairs)\n"
                f"  3. Endpoints were found but all residues have 0 endpoints\n"
                f"  Found endpoints per residue: {max_ep_per_residue}"
            )
        else:
            print(f"  Initialized {pairs_initialized} endpoint pair(s) for distance computation")
        
        self._initialized = True
    
    def on_frame(self, ts: mda.coordinates.base.Timestep, frame_idx: int,
                 universe: mda.Universe) -> None:
        """Process a single frame during iteration."""
        if not self._initialized:
            return
        
        frame_ep_positions = []
        for idx, sel_str in enumerate(self.residue_sel_list):
            try:
                ep_indices = self.stored_ep_indices[idx]
                if len(ep_indices) > 0:
                    # Get endpoint positions directly from universe (already at current frame)
                    ep_positions = universe.atoms[ep_indices].positions
                    frame_ep_positions.append(ep_positions)
                else:
                    frame_ep_positions.append(None)
            except Exception as e:
                warnings.warn(
                    f"Failed to get endpoint positions for {sel_str} at frame {ts.frame}: {e}"
                )
                frame_ep_positions.append(None)
        
        # Compute distances for all pairs
        for i in range(self.n_res):
            for j in range(i + 1, self.n_res):
                ep_i = frame_ep_positions[i]
                ep_j = frame_ep_positions[j]
                if ep_i is not None and ep_j is not None and (i, j) in self.all_pairs:
                    diff = ep_i[:, None, :] - ep_j[None, :, :]
                    dists = np.linalg.norm(diff, axis=2)
                    n_ep_i_actual, n_ep_j_actual = dists.shape
                    self.all_pairs[(i, j)][
                        frame_idx, :n_ep_i_actual, :n_ep_j_actual
                    ] = dists
                    self.all_pairs[(j, i)][
                        frame_idx, :n_ep_j_actual, :n_ep_i_actual
                    ] = dists.T
    
    def on_frame_end(self, iterator: TrajectoryIterator) -> None:
        """Finalize results after iteration."""
        # Results are already stored in self.results['all_pairs']
        pass
    
    def _get_aggregator(self) -> ResultsGroup:
        """
        Return ResultsGroup for declarative result aggregation.
        
        Uses dict_merge_nonnan for all_pairs to merge non-NaN values
        from worker results.
        """
        return ResultsGroup(lookup={
            'all_pairs': ResultsGroup.dict_merge_nonnan,
        })
    
    def merge_results(self, other: 'EndpointAnalyzerObserver') -> None:
        """
        Merge results from another observer instance (used in parallel processing).
        
        Args:
            other: Another EndpointAnalyzerObserver instance with results to merge
        """
        if not isinstance(other, EndpointAnalyzerObserver):
            return
        
        if not self._initialized or not other._initialized:
            return
        
        # Merge all_pairs dictionaries
        # For each pair, copy non-NaN values from other into self
        for (i, j), other_array in other.all_pairs.items():
            if (i, j) in self.all_pairs:
                self_array = self.all_pairs[(i, j)]
                # Copy non-NaN values from other_array to self_array
                # Use np.where to only update where other_array has valid values
                valid_mask = ~np.isnan(other_array)
                self_array[valid_mask] = other_array[valid_mask]
    
    def get_endpoint_distances(self) -> dict:
        """
        Get computed endpoint distances dictionary.
        
        Returns:
            Dictionary with 'all_pairs', 'n_residues', 'n_frames' keys
        """
        return {
            "all_pairs": self.all_pairs,
            "n_residues": self.n_res,
            "n_frames": self.n_frames,
        }
    
    def get_endpoint_metrics(self) -> pd.DataFrame:
        """
        Compute endpoint metrics DataFrame from accumulated distances.
        
        Returns:
            DataFrame with endpoint metrics per frame
        """
        if not self._initialized:
            return pd.DataFrame({"frame": range(self.n_frames)})
        
        all_pairs = self.all_pairs
        rows = []
        for frame in range(self.n_frames):
            row = {"frame": frame}
            all_pair_dists = []
            for i in range(self.n_res):
                for j in range(i + 1, self.n_res):
                    if (i, j) in all_pairs:
                        pair_array = all_pairs[(i, j)][frame]
                        valid_dists = pair_array[~np.isnan(pair_array)]
                        all_pair_dists.extend(valid_dists.tolist())
                        if len(valid_dists) > 0:
                            row[f"endpoint_dist_{i}_{j}_mean"] = float(np.mean(valid_dists))
                            row[f"endpoint_dist_{i}_{j}_min"] = float(np.min(valid_dists))
                            row[f"endpoint_dist_{i}_{j}_max"] = float(np.max(valid_dists))
                        else:
                            row[f"endpoint_dist_{i}_{j}_mean"] = np.nan
                            row[f"endpoint_dist_{i}_{j}_min"] = np.nan
                            row[f"endpoint_dist_{i}_{j}_max"] = np.nan
            
            if len(all_pair_dists) > 0:
                all_pair_dists = np.array(all_pair_dists)
                row["endpoint_dist_mean"] = float(np.mean(all_pair_dists))
                row["endpoint_dist_min"] = float(np.min(all_pair_dists))
                row["endpoint_dist_max"] = float(np.max(all_pair_dists))
                row["endpoint_dist_std"] = float(np.std(all_pair_dists))
            else:
                row["endpoint_dist_mean"] = np.nan
                row["endpoint_dist_min"] = np.nan
                row["endpoint_dist_max"] = np.nan
                row["endpoint_dist_std"] = np.nan
            
            rows.append(row)
        
        return pd.DataFrame(rows)