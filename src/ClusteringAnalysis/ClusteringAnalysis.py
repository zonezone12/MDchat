from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Optional clustering / change-point detection
try:
    import hdbscan  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    hdbscan = None  # type: ignore
    HAS_HDBSCAN = False

try:
    import ruptures as rpt  # type: ignore
    HAS_RUPTURES = True
except Exception:
    rpt = None  # type: ignore
    HAS_RUPTURES = False


class ClusteringAnalysis:
    """Perform clustering analysis and change-point detection on trajectory data."""

    def __init__(self) -> None:
        self.clustering_results: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}
        self.change_points_cache: Dict[Tuple[int, str, int], Tuple[List[int], int]] = {}
        self.labels: Optional[np.ndarray] = None
        self.medoids: Optional[np.ndarray] = None

    def zscore(self, x: np.ndarray) -> np.ndarray:
        m, s = np.nanmean(x), np.nanstd(x)
        if s == 0 or np.isnan(s):
            return np.zeros_like(x)
        return (x - m) / s

    def choose_k_by_silhouette(self, X, kmin: int = 2, kmax: int = 10) -> int:
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
            cached_labels, cached_medoids, cached_id = self.clustering_results[
                cache_key
            ]
            if cached_id == id(embeddings) and np.array_equal(
                cached_labels.shape, (len(embeddings),)
            ):
                self.labels = cached_labels
                self.medoids = cached_medoids
                return cached_labels, cached_medoids

        X = embeddings
        if HAS_HDBSCAN and hdbscan is not None:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(10, len(X) // 100 + 1), min_samples=None
            )
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
        self.clustering_results[cache_key] = (
            result_labels,
            result_medoids,
            id(embeddings),
        )
        return result_labels, result_medoids

    def change_points(
        self, signal: np.ndarray, model: str = "rbf", n_bkps: int = 5
    ) -> List[int]:
        """Return change-point indices for a 1D signal."""
        cache_key = (id(signal), model, n_bkps)
        if cache_key in self.change_points_cache:
            cached_result, cached_id = self.change_points_cache[cache_key]
            if cached_id == id(signal) and len(cached_result) > 0:
                return cached_result

        if HAS_RUPTURES and rpt is not None:
            algo = rpt.Binseg(model=model).fit(signal.reshape(-1, 1))
            bkps = algo.predict(n_bkps=n_bkps)
            result = sorted(set([b for b in bkps if b < len(signal)]))
        else:
            deriv = np.abs(np.gradient(signal))
            N = max(2, n_bkps)
            idx = np.argsort(deriv)[-N:]
            result = sorted(set(idx.tolist()))

        self.change_points_cache[cache_key] = (result, id(signal))
        return result

    def clear_cache(self) -> None:
        self.clustering_results.clear()
        self.change_points_cache.clear()
        self.labels = None
        self.medoids = None

    def get_labels(self) -> Optional[np.ndarray]:
        return self.labels

    def get_medoids(self) -> Optional[np.ndarray]:
        return self.medoids


