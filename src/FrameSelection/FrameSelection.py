from __future__ import annotations

from typing import List, Optional, Tuple, Union

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


