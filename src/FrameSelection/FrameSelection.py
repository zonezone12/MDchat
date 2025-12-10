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
        - Structural dynamics (higher variation indicates abnormal/interesting events)
        
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
                    Note: 'structural_stability' actually scores for structural dynamics/variation
                          (higher variation = higher score, as it indicates abnormal events)
        
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
        
        # 1) Guest Entry/Exit Scoring (more granular, continuous scoring)
        guest_score = 0.0
        if guest_stats is not None:
            n_entries = guest_stats.get('n_entries', 0)
            n_exits = guest_stats.get('n_exits', 0)
            total_time_inside = guest_stats.get('total_time_inside', 0.0)
            total_time_outside = guest_stats.get('total_time_outside', 0.0)
            first_entry_frame = guest_stats.get('first_entry_frame')
            
            if n_entries > 0:
                # Guest entered - base score (continuous, not binary)
                # Use sigmoid-like function: more entries = higher score, but with diminishing returns
                entry_base = 0.3 + 0.2 * (1 - np.exp(-0.3 * n_entries))  # 0.3-0.5 range
                guest_score += entry_base
                score_details['reasons'].append(f"Guest entered host ({n_entries} entry events)")
                
                # Multiple entries bonus (continuous, exponential decay)
                if n_entries > 1:
                    multi_entry_bonus = 0.15 * (1 - np.exp(-0.2 * (n_entries - 1)))
                    guest_score += multi_entry_bonus
                    score_details['reasons'].append(f"Multiple entry events indicate dynamic behavior")
                
                # Entry/exit balance scoring (continuous function)
                if n_entries > 0:
                    entry_exit_ratio = n_entries / (n_entries + n_exits) if (n_entries + n_exits) > 0 else 0.5
                    # Prefer more entries than exits, but reward balance too
                    if n_entries > n_exits:
                        balance_score = 0.25 * (1 + (n_entries - n_exits) / max(n_entries, 1)) * entry_exit_ratio
                    elif n_entries == n_exits:
                        balance_score = 0.15 * entry_exit_ratio
                    else:
                        balance_score = 0.05 * entry_exit_ratio
                    guest_score += balance_score
                    score_details['reasons'].append(
                        f"Entry/exit balance: {n_entries} entries, {n_exits} exits"
                    )
                
                # Residence time scoring (continuous, magnitude-aware)
                if total_time_inside > 0:
                    total_time = total_time_inside + total_time_outside
                    if total_time > 0:
                        residence_fraction = total_time_inside / total_time
                        # Use power function to emphasize higher residence fractions
                        residence_score = 0.25 * (residence_fraction ** 0.7)  # Non-linear scaling
                        guest_score += residence_score
                        score_details['reasons'].append(
                            f"Guest residence fraction: {residence_fraction:.2%}"
                        )
                    
                    # Minimum residence time check (continuous scoring)
                    if min_guest_residence_time is not None:
                        if total_time_inside >= min_guest_residence_time:
                            # Bonus increases with excess time
                            excess_ratio = total_time_inside / min_guest_residence_time
                            time_bonus = 0.15 * min(1.0, np.log(1 + excess_ratio) / np.log(2))
                            guest_score += time_bonus
                            score_details['reasons'].append(
                                f"Guest residence time ({total_time_inside:.1f} ps) exceeds minimum"
                            )
                        else:
                            # Partial credit for approaching minimum
                            partial_credit = 0.1 * (total_time_inside / min_guest_residence_time) ** 0.5
                            guest_score += partial_credit
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
        
        # 2) Volume Dynamics Scoring (continuous, magnitude-aware)
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
                    volume_score += 0.2
                    
                    # Continuous scoring based on volume variation magnitude
                    # Use sigmoid-like function for smooth transition
                    # Normalize to [0, 1] range with min_volume_change_pct as reference
                    normalized_change = vol_change_pct / max(min_volume_change_pct, 1.0)
                    # Use tanh for smooth S-curve: 0.5 at threshold, approaches 1.0 for high values
                    variation_score = 0.5 * (1 + np.tanh(2 * (normalized_change - 0.5)))
                    volume_score += 0.4 * variation_score
                    
                    if vol_change_pct >= min_volume_change_pct:
                        score_details['reasons'].append(
                            f"Significant volume dynamics ({vol_change_pct:.1f}% change)"
                        )
                    elif vol_change_pct >= min_volume_change_pct * 0.5:
                        score_details['reasons'].append(
                            f"Moderate volume dynamics ({vol_change_pct:.1f}% change)"
                        )
                    else:
                        score_details['warnings'].append(
                            f"Limited volume dynamics ({vol_change_pct:.1f}% change, "
                            f"threshold: {min_volume_change_pct}%)"
                        )
                    
                    # Volume stability scoring (continuous, not binary)
                    cv = vol_std / vol_mean if vol_mean > 0 else np.inf
                    if cv < np.inf:
                        # Optimal CV is around 0.05-0.15 (some variation but not too chaotic)
                        # Score peaks at CV=0.1, decreases for both very low and very high CV
                        optimal_cv = 0.1
                        stability_score = 0.4 * np.exp(-((cv - optimal_cv) / 0.15) ** 2)
                        volume_score += stability_score
                        if cv < 0.1:
                            score_details['reasons'].append(f"Stable volume (CV={cv:.3f})")
                        elif cv > 0.3:
                            score_details['warnings'].append(f"High volume variation (CV={cv:.3f})")
                        else:
                            score_details['reasons'].append(f"Moderate volume variation (CV={cv:.3f})")
                else:
                    score_details['warnings'].append("Invalid volume data (mean <= 0)")
            else:
                score_details['warnings'].append("No valid volume data")
        else:
            score_details['warnings'].append("Volume data not available")
        
        score_details['volume_dynamics_score'] = min(1.0, volume_score)
        total_score += weights['volume_dynamics'] * score_details['volume_dynamics_score']
        
        # 3) Endpoint-Volume Correlation Scoring (continuous, magnitude-aware)
        correlation_score = 0.0
        if correlation_df is not None and not correlation_df.empty:
            # Check for significant correlations
            if 'correlation' in correlation_df.columns:
                abs_correlations = correlation_df['correlation'].abs()
                max_corr = abs_correlations.max()
                n_significant = (abs_correlations >= min_correlation).sum()
                
                # Base score for having correlation data
                correlation_score += 0.15
                
                # Continuous scoring based on maximum correlation magnitude
                # Use power function to emphasize stronger correlations
                # Normalize: 0.5 correlation = 0.5 score, 1.0 correlation = 1.0 score
                if max_corr > 0:
                    # Use power function: stronger correlations get exponentially higher scores
                    max_corr_score = 0.5 * (max_corr ** 1.5)  # Power > 1 emphasizes high values
                    correlation_score += max_corr_score
                    
                    if max_corr >= min_correlation:
                        score_details['reasons'].append(
                            f"Strong endpoint-volume correlation (max: {max_corr:.3f})"
                        )
                    else:
                        score_details['warnings'].append(
                            f"Weak correlations (max: {max_corr:.3f}, threshold: {min_correlation})"
                        )
                
                # Multiple correlations bonus (continuous, with diminishing returns)
                if n_significant > 0:
                    # Use logarithmic scaling: more correlations = better, but with diminishing returns
                    multi_corr_bonus = 0.25 * np.log(1 + n_significant) / np.log(10)  # log10 scaling
                    correlation_score += multi_corr_bonus
                    if n_significant > 1:
                        score_details['reasons'].append(
                            f"Multiple significant correlations ({n_significant} pairs)"
                        )
                
                # Bonus for average correlation strength (not just max)
                if len(abs_correlations) > 0:
                    mean_corr = abs_correlations.mean()
                    if mean_corr > 0:
                        mean_corr_bonus = 0.1 * (mean_corr ** 1.2)
                        correlation_score += mean_corr_bonus
            else:
                score_details['warnings'].append("Correlation data missing 'correlation' column")
        else:
            score_details['warnings'].append("Correlation data not available")
        
        score_details['correlation_score'] = min(1.0, correlation_score)
        total_score += weights['correlation'] * score_details['correlation_score']
        
        # 4) Structural Stability Scoring (continuous, magnitude-aware)
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
                    stability_score += 0.25
                    
                    # Score based on variation: Higher variation = higher score (abnormal events are interesting)
                    if dist_mean > 0:
                        cv = dist_std / dist_mean
                        # Use continuous function: optimal CV around 0.15-0.25, but reward higher values too
                        # Use a combination: base score + variation bonus
                        if cv >= 0.2:
                            # High variation - very interesting
                            variation_score = 0.5 + 0.2 * min(1.0, (cv - 0.2) / 0.3)  # 0.5-0.7 range
                            stability_score += variation_score
                            score_details['reasons'].append(
                                f"High structural variation detected (CV={cv:.3f}) - abnormal events present"
                            )
                        elif 0.05 < cv < 0.2:
                            # Moderate variation - good dynamics
                            variation_score = 0.3 + 0.2 * ((cv - 0.05) / 0.15)  # 0.3-0.5 range
                            stability_score += variation_score
                            score_details['reasons'].append(
                                f"Moderate structural variation (CV={cv:.3f})"
                            )
                        else:
                            # Too stable - less interesting
                            variation_score = 0.1 + 0.2 * (cv / 0.05)  # 0.1-0.3 range
                            stability_score += variation_score
                            score_details['warnings'].append(
                                f"Very stable structure (CV={cv:.3f}, might be stuck or lack dynamics)"
                            )
        elif cube_metrics_df is not None and not cube_metrics_df.empty:
            # Fallback to cube metrics for stability
            if 'edge_mean' in cube_metrics_df.columns:
                edges = cube_metrics_df['edge_mean'].values
                valid_edges = edges[~np.isnan(edges)]
                if len(valid_edges) > 1:
                    edge_cv = np.nanstd(valid_edges) / np.nanmean(valid_edges)
                    # Continuous scoring: higher CV = higher score, but with smooth transitions
                    if edge_cv >= 0.2:
                        stability_score = 0.7 + 0.2 * min(1.0, (edge_cv - 0.2) / 0.3)  # 0.7-0.9 range
                        score_details['reasons'].append(
                            f"High structural variation from edge metrics (CV={edge_cv:.3f}) - abnormal events"
                        )
                    elif edge_cv >= 0.05:
                        stability_score = 0.5 + 0.2 * ((edge_cv - 0.05) / 0.15)  # 0.5-0.7 range
                        score_details['reasons'].append("Moderate structural variation from edge metrics")
                    else:
                        stability_score = 0.3 + 0.2 * (edge_cv / 0.05)  # 0.3-0.5 range
                        score_details['warnings'].append("Low structural variation from edge metrics")
        else:
            score_details['warnings'].append("Structural stability metrics not available")
        
        score_details['structural_stability_score'] = min(1.0, stability_score)
        total_score += weights['structural_stability'] * score_details['structural_stability_score']
        
        # Apply non-linear transformation to make scores more discriminative
        # Use power function to spread out scores: higher scores get more separation
        # This makes differences between trajectories more significant
        if total_score > 0:
            # Apply power transformation: score^1.2 spreads out high scores more
            # This means a score of 0.9 becomes ~0.88, 0.8 becomes ~0.77, etc.
            # But we need to preserve the 0-1 range, so we normalize
            # Actually, let's use a different approach: scale to emphasize differences
            # Use a sigmoid-like transformation that increases separation in the middle-high range
            # For scores > 0.5, apply slight expansion: score -> score + 0.1*(score-0.5)^2
            if total_score > 0.5:
                expansion = 0.08 * ((total_score - 0.5) ** 1.5)
                total_score = min(1.0, total_score + expansion)
        
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
            f"  Structural Dynamics: {details['structural_stability_score']:.3f} / 1.000",
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

    def save_simulation_score_csv(
        self,
        output_path: str,
        trajectory_id: Optional[str] = None,
        guest_stats: Optional[Dict[str, Any]] = None,
        volume: Optional[np.ndarray] = None,
        correlation_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Save simulation score summary as CSV for easy comparison across trajectories.
        
        Args:
            output_path: Path to save the CSV file (e.g., "output/simulation_score.csv")
            trajectory_id: Optional identifier for this trajectory (e.g., "traj_001")
            guest_stats: Optional guest statistics dictionary (for extracting metrics)
            volume: Optional volume array (for extracting volume statistics)
            correlation_df: Optional correlation DataFrame (for extracting correlation statistics)
        """
        if self.simulation_score is None or self.simulation_score_details is None:
            raise ValueError("Simulation not scored yet. Call score_simulation() first.")
        
        details = self.simulation_score_details
        
        # Extract key metrics for comparison
        n_entries = guest_stats.get('n_entries', 0) if guest_stats else 0
        n_exits = guest_stats.get('n_exits', 0) if guest_stats else 0
        total_time_inside = guest_stats.get('total_time_inside', 0.0) if guest_stats else 0.0
        total_time_outside = guest_stats.get('total_time_outside', 0.0) if guest_stats else 0.0
        first_entry_frame = guest_stats.get('first_entry_frame') if guest_stats else None
        first_entry_time = guest_stats.get('first_entry_time') if guest_stats else None
        
        # Volume statistics
        if volume is not None and len(volume) > 0:
            valid_volume = volume[~np.isnan(volume)]
            if len(valid_volume) > 0:
                vol_mean = float(np.nanmean(valid_volume))
                vol_std = float(np.nanstd(valid_volume))
                vol_min = float(np.nanmin(valid_volume))
                vol_max = float(np.nanmax(valid_volume))
                vol_range = vol_max - vol_min
                vol_change_pct = (vol_range / vol_mean * 100.0) if vol_mean > 0 else 0.0
                vol_cv = (vol_std / vol_mean) if vol_mean > 0 else 0.0
            else:
                vol_mean = vol_std = vol_min = vol_max = vol_range = vol_change_pct = vol_cv = np.nan
        else:
            vol_mean = vol_std = vol_min = vol_max = vol_range = vol_change_pct = vol_cv = np.nan
        
        # Correlation statistics
        max_correlation = np.nan
        n_significant_correlations = 0
        if correlation_df is not None and not correlation_df.empty:
            if 'correlation' in correlation_df.columns:
                abs_correlations = correlation_df['correlation'].abs()
                max_correlation = float(abs_correlations.max())
                n_significant_correlations = int((abs_correlations >= 0.5).sum())
        
        # Combine reasons and warnings (semicolon-separated for CSV)
        reasons_str = "; ".join(details['reasons']) if details['reasons'] else ""
        warnings_str = "; ".join(details['warnings']) if details['warnings'] else ""
        
        # Create DataFrame with single row
        score_data = {
            'trajectory_id': trajectory_id if trajectory_id else 'unknown',
            'overall_score': round(self.simulation_score, 4),
            'is_valid': details['is_valid'],
            'guest_entry_score': round(details['guest_entry_score'], 4),
            'volume_dynamics_score': round(details['volume_dynamics_score'], 4),
            'correlation_score': round(details['correlation_score'], 4),
            'structural_dynamics_score': round(details['structural_stability_score'], 4),
            # Guest metrics
            'n_entries': n_entries,
            'n_exits': n_exits,
            'entry_exit_diff': n_entries - n_exits,
            'total_time_inside_ps': round(total_time_inside, 2) if total_time_inside > 0 else 0.0,
            'total_time_outside_ps': round(total_time_outside, 2) if total_time_outside > 0 else 0.0,
            'residence_fraction': round(total_time_inside / (total_time_inside + total_time_outside), 4) 
                                   if (total_time_inside + total_time_outside) > 0 else 0.0,
            'first_entry_frame': first_entry_frame if first_entry_frame is not None else np.nan,
            'first_entry_time_ps': round(first_entry_time, 2) if first_entry_time is not None else np.nan,
            # Volume metrics
            'volume_mean_A3': round(vol_mean, 2) if not np.isnan(vol_mean) else np.nan,
            'volume_std_A3': round(vol_std, 2) if not np.isnan(vol_std) else np.nan,
            'volume_min_A3': round(vol_min, 2) if not np.isnan(vol_min) else np.nan,
            'volume_max_A3': round(vol_max, 2) if not np.isnan(vol_max) else np.nan,
            'volume_change_pct': round(vol_change_pct, 2) if not np.isnan(vol_change_pct) else np.nan,
            'volume_cv': round(vol_cv, 4) if not np.isnan(vol_cv) else np.nan,
            # Correlation metrics
            'max_correlation': round(max_correlation, 4) if not np.isnan(max_correlation) else np.nan,
            'n_significant_correlations': n_significant_correlations,
            # Summary text
            'positive_indicators': reasons_str,
            'warnings': warnings_str,
        }
        
        df = pd.DataFrame([score_data])
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        return


def load_and_compare_simulation_scores(
    csv_paths: List[str],
    output_path: Optional[str] = None,
    sort_by: str = 'overall_score',
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Load and compare multiple simulation score CSV files.
    
    This function is useful for batch analysis - load all trajectory score CSVs
    and rank them to identify which simulations to investigate first.
    
    Args:
        csv_paths: List of paths to simulation score CSV files
        output_path: Optional path to save the combined/ranked results
        sort_by: Column name to sort by (default: 'overall_score')
        ascending: If True, sort ascending (default: False, highest scores first)
    
    Returns:
        DataFrame with all simulation scores, sorted by the specified column
    
    Example:
        >>> from src.FrameSelection import load_and_compare_simulation_scores
        >>> import glob
        >>> 
        >>> # Find all simulation score CSV files
        >>> csv_files = glob.glob("output/*_simulation_score.csv")
        >>> 
        >>> # Load and compare
        >>> comparison_df = load_and_compare_simulation_scores(
        ...     csv_files,
        ...     output_path="output/all_simulations_ranked.csv",
        ...     sort_by='overall_score'
        ... )
        >>> 
        >>> # Print top 5
        >>> print(comparison_df.head(5)[['trajectory_id', 'overall_score', 'is_valid']])
    """
    all_scores = []
    
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                all_scores.append(df)
            else:
                warnings.warn(f"Empty CSV file: {csv_path}")
        except Exception as e:
            warnings.warn(f"Failed to load {csv_path}: {e}")
    
    if not all_scores:
        raise ValueError("No valid simulation score CSV files could be loaded")
    
    # Combine all DataFrames
    combined_df = pd.concat(all_scores, ignore_index=True)
    
    # Sort by specified column
    if sort_by in combined_df.columns:
        combined_df = combined_df.sort_values(by=sort_by, ascending=ascending, na_position='last')
    else:
        warnings.warn(f"Column '{sort_by}' not found. Available columns: {list(combined_df.columns)}")
        warnings.warn("Sorting by 'overall_score' instead")
        if 'overall_score' in combined_df.columns:
            combined_df = combined_df.sort_values(by='overall_score', ascending=False, na_last=True)
    
    # Add rank column
    combined_df.insert(0, 'rank', range(1, len(combined_df) + 1))
    
    # Save if output path provided
    if output_path:
        combined_df.to_csv(output_path, index=False)
    
    return combined_df


