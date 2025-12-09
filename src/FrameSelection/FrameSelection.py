from __future__ import annotations

from typing import List, Optional, Tuple, Union, Dict, Any
import warnings

import numpy as np
import pandas as pd

from ..ClusteringAnalysis import ClusteringAnalysis


class FrameSelection:
    """Score and select meaningful frames from trajectory."""

    def __init__(self) -> None:
        self.scores: Optional[np.ndarray] = None
        self.score_df: Optional[pd.DataFrame] = None
        self.selected_frames: Optional[List[int]] = None
        self.medoids: Optional[np.ndarray] = None
        self.cpd_idx: Optional[List[int]] = None
        self.pcs: Optional[np.ndarray] = None
        # Storage for simulation-level scoring
        self.simulation_score: Optional[float] = None
        self.simulation_score_details: Optional[Dict[str, Any]] = None

    def score_frames(
        self,
        rmsd: np.ndarray,
        rg: np.ndarray,
        pcs: np.ndarray,
        cpd_idx: List[int],
        strain: Optional[np.ndarray] = None,
        w: Tuple[float, float, float, float] = (0.35, 0.2, 0.35, 0.1),
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        T = len(rmsd)
        clustering = ClusteringAnalysis()
        self.pcs = pcs

        pcZ = clustering.zscore(pcs[:, : min(3, pcs.shape[1])])
        pc_ext = np.max(np.abs(pcZ), axis=1)

        rmsd_deriv = np.abs(np.gradient(rmsd))
        rmsd_dZ = clustering.zscore(rmsd_deriv)

        rgZ = np.abs(clustering.zscore(rg))

        if strain is None:
            strainZ = np.zeros(T)
        else:
            strainZ = clustering.zscore(strain)
            strainZ = np.nan_to_num(strainZ)

        bonus = np.zeros(T)
        for i in cpd_idx:
            if 0 <= i < T:
                bonus[i] += 1.0

        score = w[0] * pc_ext + w[1] * rmsd_dZ + w[2] * rgZ + w[3] * strainZ + bonus
        df = pd.DataFrame(
            {
                "pc_extreme": pc_ext,
                "rmsd_d": rmsd_dZ,
                "rg_extreme": rgZ,
                "strain": strainZ,
                "cpd_bonus": bonus,
                "score": score,
            }
        )
        self.scores = score
        self.score_df = df
        return score, df

    def select_meaningful_frames(
        self,
        medoids: Optional[np.ndarray] = None,
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
        endpoint_metrics_csv: Optional[str] = None,
    ) -> Tuple[List[int], Optional[pd.DataFrame]]:
        """Select meaningful frames from trajectory."""
        # Use stored state if not provided
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

        # Basic validation
        if medoids is None:
            raise ValueError("medoids must be provided either directly or via scores_csv")
        if cpd_idx is None:
            raise ValueError("cpd_idx must be provided either directly or via scores_csv")
        if scores is None:
            raise ValueError("scores must be provided either directly or via scores_csv")
        if pcs is None:
            raise ValueError("pcs must be provided either directly or via metrics_csv")

        chosen = set(medoids.tolist() + cpd_idx)

        # Endpoint-based augmentation if available
        if endpoint_metrics_df is not None and len(endpoint_metrics_df) > 0:
            if "frame" not in endpoint_metrics_df.columns:
                endpoint_metrics_df = endpoint_metrics_df.copy()
                endpoint_metrics_df["frame"] = endpoint_metrics_df.index

            if "endpoint_dist_min" in endpoint_metrics_df.columns:
                min_dist = endpoint_metrics_df["endpoint_dist_min"].values
                valid_min = ~np.isnan(min_dist)
                if np.any(valid_min):
                    min_idx = np.nanargmin(min_dist)
                    min_frame = int(endpoint_metrics_df.iloc[min_idx]["frame"])
                    chosen.add(min_frame)

            if "endpoint_dist_max" in endpoint_metrics_df.columns:
                max_dist = endpoint_metrics_df["endpoint_dist_max"].values
                valid_max = ~np.isnan(max_dist)
                if np.any(valid_max):
                    max_idx = np.nanargmax(max_dist)
                    max_frame = int(endpoint_metrics_df.iloc[max_idx]["frame"])
                    chosen.add(max_frame)

            if "endpoint_dist_mean" in endpoint_metrics_df.columns:
                mean_dist = endpoint_metrics_df["endpoint_dist_mean"].values
                valid_mean = ~np.isnan(mean_dist)
                if np.any(valid_mean):
                    deriv = np.abs(np.gradient(mean_dist))
                    deriv[~valid_mean] = 0
                    top_deriv_indices = np.argsort(deriv)[-3:][::-1]
                    for idx in top_deriv_indices:
                        if valid_mean[idx] and len(chosen) < max_frames:
                            frame_num = int(endpoint_metrics_df.iloc[idx]["frame"])
                            chosen.add(frame_num)

        # Handle endpoint_dists_array dict format if provided
        if endpoint_dists_array is not None and isinstance(endpoint_dists_array, dict):
            all_pairs = endpoint_dists_array.get("all_pairs", {})
            T = endpoint_dists_array.get("n_frames", 0)
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
                    frame_stats.append(
                        {
                            "frame": t,
                            "min": np.min(all_pair_dists),
                            "max": np.max(all_pair_dists),
                        }
                    )
            if len(frame_stats) > 0:
                stats_df = pd.DataFrame(frame_stats)
                if "min" in stats_df.columns:
                    min_frame = int(stats_df.loc[stats_df["min"].idxmin(), "frame"])
                    if len(chosen) < max_frames:
                        chosen.add(min_frame)
                if "max" in stats_df.columns:
                    max_frame = int(stats_df.loc[stats_df["max"].idxmax(), "frame"])
                    if len(chosen) < max_frames:
                        chosen.add(max_frame)

        order = np.argsort(scores)[::-1]

        def far_from_set(i: int, chosen_list: List[int], thr: float = 1.0) -> bool:
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

        if score_df is not None:
            score_df["selected"] = 0
            if "frame" in score_df.columns:
                mask = score_df["frame"].isin(chosen_sorted)
                score_df.loc[mask, "selected"] = 1
            else:
                score_df.loc[chosen_sorted, "selected"] = 1

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

    def score_simulation(
        self,
        guest_stats: Optional[Dict[str, Any]] = None,
        volume: Optional[np.ndarray] = None,
        correlation_df: Optional[pd.DataFrame] = None,
        endpoint_dists_array: Optional[Union[np.ndarray, dict]] = None,
        endpoint_metrics_df: Optional[pd.DataFrame] = None,
        cube_metrics_df: Optional[pd.DataFrame] = None,
        min_guest_entry: bool = True,
        min_volume_change_pct: float = 10.0,
        min_correlation: float = 0.5,
        min_guest_residence_time: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Score an entire MD simulation based on multiple criteria.
        
        This method evaluates whether a simulation is "meaningful" by checking:
        - Guest entry/exit behavior
        - Volume dynamics
        - Endpoint-volume correlations
        - Structural stability
        
        Args:
            guest_stats: Dictionary with guest residence statistics from GSAnalyzerObserver
            volume: Array of volume values over trajectory
            correlation_df: DataFrame with endpoint-volume correlations
            endpoint_dists_array: Dictionary or array with endpoint distances
            endpoint_metrics_df: DataFrame with endpoint metrics
            cube_metrics_df: DataFrame with cube/volume metrics
            min_guest_entry: If True, simulation is invalid if guest never entered
            min_volume_change_pct: Minimum volume change percentage to be considered meaningful
            min_correlation: Minimum correlation threshold for endpoint-volume pairs
            min_guest_residence_time: Minimum total residence time (ps) for guest inside
            weights: Dictionary of weights for different scoring components.
                    Default: {'guest_entry': 0.3, 'volume_dynamics': 0.25, 
                            'correlation': 0.25, 'structural_stability': 0.2}
        
        Returns:
            Tuple of (total_score, score_details_dict) where:
            - total_score: Overall score (0.0 to 1.0, higher is better)
            - score_details: Dictionary with component scores and reasons
        """
        if weights is None:
            weights = {
                'guest_entry': 0.3,
                'volume_dynamics': 0.25,
                'correlation': 0.25,
                'structural_stability': 0.2
            }
        
        score_details = {
            'guest_entry_score': 0.0,
            'volume_dynamics_score': 0.0,
            'correlation_score': 0.0,
            'structural_stability_score': 0.0,
            'reasons': [],
            'warnings': [],
            'is_valid': True
        }
        
        total_score = 0.0
        
        # 1) Guest Entry/Exit Scoring
        guest_score = 0.0
        if guest_stats is not None:
            n_entries = guest_stats.get('n_entries', 0)
            n_exits = guest_stats.get('n_exits', 0)
            total_time_inside = guest_stats.get('total_time_inside', 0.0)
            total_time_outside = guest_stats.get('total_time_outside', 0.0)
            first_entry_frame = guest_stats.get('first_entry_frame')
            
            if n_entries > 0:
                # Guest entered - base score
                guest_score += 0.4
                score_details['reasons'].append(f"Guest entered host ({n_entries} entry events)")
                
                # Bonus for multiple entries (indicates dynamic behavior)
                if n_entries > 1:
                    guest_score += min(0.2, (n_entries - 1) * 0.05)
                    score_details['reasons'].append(f"Multiple entry events indicate dynamic behavior")
                
                # Residence time scoring
                if total_time_inside > 0:
                    # Calculate residence fraction
                    total_time = total_time_inside + total_time_outside
                    if total_time > 0:
                        residence_fraction = total_time_inside / total_time
                        guest_score += 0.2 * residence_fraction
                        score_details['reasons'].append(
                            f"Guest residence fraction: {residence_fraction:.2%}"
                        )
                    
                    # Check minimum residence time if specified
                    if min_guest_residence_time is not None:
                        if total_time_inside >= min_guest_residence_time:
                            guest_score += 0.2
                            score_details['reasons'].append(
                                f"Guest residence time ({total_time_inside:.1f} ps) exceeds minimum"
                            )
                        else:
                            score_details['warnings'].append(
                                f"Guest residence time ({total_time_inside:.1f} ps) below minimum "
                                f"({min_guest_residence_time:.1f} ps)"
                            )
            else:
                # Guest never entered
                if min_guest_entry:
                    score_details['is_valid'] = False
                    score_details['reasons'].append("Guest never entered host (simulation invalid)")
                else:
                    score_details['warnings'].append("Guest never entered host")
        else:
            score_details['warnings'].append("Guest statistics not available")
        
        score_details['guest_entry_score'] = min(1.0, guest_score)
        total_score += weights['guest_entry'] * score_details['guest_entry_score']
        
        # 2) Volume Dynamics Scoring
        volume_score = 0.0
        if volume is not None and len(volume) > 0:
            valid_volume = volume[~np.isnan(volume)]
            if len(valid_volume) > 0:
                vol_mean = np.nanmean(valid_volume)
                vol_std = np.nanstd(valid_volume)
                vol_min = np.nanmin(valid_volume)
                vol_max = np.nanmax(valid_volume)
                vol_range = vol_max - vol_min
                
                # Calculate volume change percentage
                if vol_mean > 0:
                    vol_change_pct = (vol_range / vol_mean) * 100.0
                    
                    # Base score for having volume data
                    volume_score += 0.3
                    
                    # Score based on volume variation
                    if vol_change_pct >= min_volume_change_pct:
                        volume_score += 0.4
                        score_details['reasons'].append(
                            f"Significant volume dynamics ({vol_change_pct:.1f}% change)"
                        )
                    elif vol_change_pct >= min_volume_change_pct * 0.5:
                        volume_score += 0.2
                        score_details['reasons'].append(
                            f"Moderate volume dynamics ({vol_change_pct:.1f}% change)"
                        )
                    else:
                        score_details['warnings'].append(
                            f"Limited volume dynamics ({vol_change_pct:.1f}% change, "
                            f"threshold: {min_volume_change_pct}%)"
                        )
                    
                    # Bonus for volume stability (low std relative to mean)
                    cv = vol_std / vol_mean if vol_mean > 0 else np.inf
                    if cv < 0.1:  # Coefficient of variation < 10%
                        volume_score += 0.3
                        score_details['reasons'].append("Stable volume (low variation)")
                    elif cv > 0.3:
                        score_details['warnings'].append(f"High volume variation (CV={cv:.2f})")
                else:
                    score_details['warnings'].append("Invalid volume data (mean <= 0)")
            else:
                score_details['warnings'].append("No valid volume data")
        else:
            score_details['warnings'].append("Volume data not available")
        
        score_details['volume_dynamics_score'] = min(1.0, volume_score)
        total_score += weights['volume_dynamics'] * score_details['volume_dynamics_score']
        
        # 3) Endpoint-Volume Correlation Scoring
        correlation_score = 0.0
        if correlation_df is not None and not correlation_df.empty:
            # Check for significant correlations
            if 'correlation' in correlation_df.columns:
                abs_correlations = correlation_df['correlation'].abs()
                max_corr = abs_correlations.max()
                n_significant = (abs_correlations >= min_correlation).sum()
                
                # Base score for having correlation data
                correlation_score += 0.2
                
                if max_corr >= min_correlation:
                    # Strong correlation found
                    correlation_score += 0.5
                    score_details['reasons'].append(
                        f"Strong endpoint-volume correlation (max: {max_corr:.3f})"
                    )
                    
                    # Bonus for multiple significant correlations
                    if n_significant > 1:
                        correlation_score += min(0.3, (n_significant - 1) * 0.1)
                        score_details['reasons'].append(
                            f"Multiple significant correlations ({n_significant} pairs)"
                        )
                else:
                    score_details['warnings'].append(
                        f"Weak correlations (max: {max_corr:.3f}, threshold: {min_correlation})"
                    )
            else:
                score_details['warnings'].append("Correlation data missing 'correlation' column")
        else:
            score_details['warnings'].append("Correlation data not available")
        
        score_details['correlation_score'] = min(1.0, correlation_score)
        total_score += weights['correlation'] * score_details['correlation_score']
        
        # 4) Structural Stability Scoring
        stability_score = 0.0
        if endpoint_metrics_df is not None and not endpoint_metrics_df.empty:
            # Check endpoint distance stability
            if 'endpoint_dist_mean' in endpoint_metrics_df.columns:
                mean_dists = endpoint_metrics_df['endpoint_dist_mean'].values
                valid_dists = mean_dists[~np.isnan(mean_dists)]
                
                if len(valid_dists) > 1:
                    dist_std = np.nanstd(valid_dists)
                    dist_mean = np.nanmean(valid_dists)
                    
                    # Base score for having endpoint data
                    stability_score += 0.3
                    
                    # Score based on reasonable variation (not too stable, not too chaotic)
                    if dist_mean > 0:
                        cv = dist_std / dist_mean
                        if 0.05 < cv < 0.2:  # Moderate, meaningful variation
                            stability_score += 0.5
                            score_details['reasons'].append(
                                f"Reasonable structural variation (CV={cv:.3f})"
                            )
                        elif cv < 0.05:  # Too stable (might be stuck)
                            stability_score += 0.2
                            score_details['warnings'].append(
                                f"Very stable structure (CV={cv:.3f}, might be stuck)"
                            )
                        else:  # Too chaotic
                            stability_score += 0.2
                            score_details['warnings'].append(
                                f"High structural variation (CV={cv:.3f}, might be unstable)"
                            )
        elif cube_metrics_df is not None and not cube_metrics_df.empty:
            # Fallback to cube metrics for stability
            if 'edge_mean' in cube_metrics_df.columns:
                edges = cube_metrics_df['edge_mean'].values
                valid_edges = edges[~np.isnan(edges)]
                if len(valid_edges) > 1:
                    edge_cv = np.nanstd(valid_edges) / np.nanmean(valid_edges)
                    stability_score = 0.5 if 0.05 < edge_cv < 0.2 else 0.3
                    score_details['reasons'].append("Structural stability assessed from edge metrics")
        else:
            score_details['warnings'].append("Structural stability metrics not available")
        
        score_details['structural_stability_score'] = min(1.0, stability_score)
        total_score += weights['structural_stability'] * score_details['structural_stability_score']
        
        # Store results
        self.simulation_score = total_score
        self.simulation_score_details = score_details
        
        return total_score, score_details

    def get_simulation_score_summary(self) -> str:
        """
        Get a human-readable summary of the simulation score.
        
        Returns:
            Formatted string with score breakdown
        """
        if self.simulation_score is None or self.simulation_score_details is None:
            return "Simulation not scored yet. Call score_simulation() first."
        
        details = self.simulation_score_details
        lines = [
            "=" * 60,
            "Simulation Quality Score",
            "=" * 60,
            f"Overall Score: {self.simulation_score:.3f} / 1.000",
            f"Valid: {'Yes' if details['is_valid'] else 'No'}",
            "",
            "Component Scores:",
            f"  Guest Entry:        {details['guest_entry_score']:.3f} / 1.000",
            f"  Volume Dynamics:   {details['volume_dynamics_score']:.3f} / 1.000",
            f"  Correlation:       {details['correlation_score']:.3f} / 1.000",
            f"  Structural Stability: {details['structural_stability_score']:.3f} / 1.000",
            "",
        ]
        
        if details['reasons']:
            lines.append("Positive Indicators:")
            for reason in details['reasons']:
                lines.append(f"  ✓ {reason}")
            lines.append("")
        
        if details['warnings']:
            lines.append("Warnings:")
            for warning in details['warnings']:
                lines.append(f"  ⚠ {warning}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)


