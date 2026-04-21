from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
except ImportError:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

from ..FrameGatherer import FrameGatherer


def rmsd_value_aligned(frame_coords: np.ndarray, ref_coords: np.ndarray) -> float:
    _, rmsd_val = align.rotation_matrix(frame_coords, ref_coords)
    return float(rmsd_val)


def rg_value(coords: np.ndarray) -> float:
    com = coords.mean(axis=0)
    rg2 = ((coords - com) ** 2).sum(axis=1).mean()
    return float(np.sqrt(rg2))


def contact_min_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> float:
    da = coords_a[:, None, :]
    db = coords_b[None, :, :]
    diff = da - db
    dd = np.sqrt((diff * diff).sum(axis=2))
    return float(dd.min())


def rmsf_from_coords(coords: np.ndarray) -> np.ndarray:
    mean = coords.mean(axis=0)
    diffsq = (coords - mean) ** 2
    return np.sqrt(diffsq.sum(axis=2).mean(axis=0))


def pca_from_coords(coords: np.ndarray, n_components: int = 5) -> Tuple[np.ndarray, PCA]:
    X = coords.reshape(coords.shape[0], -1)
    Xc = X - X.mean(axis=0)
    pca = PCA(n_components=n_components, svd_solver="auto")
    pcs = pca.fit_transform(Xc)
    return pcs, pca


class TrajectoryMetrics:
    """Compute basic trajectory metrics like RMSD, RMSF, radius of gyration, etc."""

    def __init__(self) -> None:
        self.rmsd_cache = {}
        self.rmsf_cache = {}
        self.rg_cache = {}
        self.contact_cache = {}
        self.pca_cache = {}
        self.strain_cache = {}
        self._universe: Optional[mda.Universe] = None
        self._alignment_sel: Optional[str] = None

    def set_universe(self, u: mda.Universe, align_sel: Optional[str] = None) -> None:
        self._universe = u
        self._alignment_sel = align_sel

    def compute_rmsd(
        self,
        u: mda.Universe,
        sel_str: str,
        ref_frame: int = 0,
        gatherer: Optional[FrameGatherer] = None,
    ) -> np.ndarray:
        cache_key = (sel_str, ref_frame)
        if cache_key in self.rmsd_cache:
            return self.rmsd_cache[cache_key]

        if gatherer is not None:
            coords = gatherer.get_coordinates(sel_str)
            frame_indices = gatherer.get_frame_indices()
            ref_idx = np.where(frame_indices == ref_frame)[0]
            if len(ref_idx) == 0:
                raise ValueError(
                    f"Reference frame {ref_frame} not found in gathered data"
                )
            ref = coords[ref_idx[0]].copy()
            rmsds = []
            for frame_coords in coords:
                rmsds.append(rmsd_value_aligned(frame_coords, ref))
            result = np.array(rmsds)
        else:
            sel = u.select_atoms(sel_str)
            ref = sel.positions.copy()
            rmsds = []
            for _ in u.trajectory:
                rmsds.append(rmsd_value_aligned(sel.positions, ref))
            result = np.array(rmsds)

        self.rmsd_cache[cache_key] = result
        return result

    def compute_rmsf(
        self,
        u: mda.Universe,
        sel_str: str,
        aligned: bool = True,
        gatherer: Optional[FrameGatherer] = None,
    ) -> np.ndarray:
        cache_key = sel_str
        if cache_key in self.rmsf_cache:
            return self.rmsf_cache[cache_key]

        if gatherer is not None:
            coords = gatherer.get_coordinates(sel_str)
        else:
            sel = u.select_atoms(sel_str)
            if not aligned:
                warnings.warn(
                    "Computing RMSF on unaligned trajectory. "
                    "Consider using AlignedTrajectory first."
                )
                align.AlignTraj(u, u, select=sel_str, in_memory=True).run()

            coords = []
            for _ in u.trajectory:
                coords.append(sel.positions.copy())
            coords = np.array(coords)

        rmsf = rmsf_from_coords(coords)
        self.rmsf_cache[cache_key] = rmsf
        return rmsf

    def radius_of_gyration(
        self,
        u: mda.Universe,
        sel_str: str,
        gatherer: Optional[FrameGatherer] = None,
    ) -> np.ndarray:
        cache_key = sel_str
        if cache_key in self.rg_cache:
            return self.rg_cache[cache_key]

        if gatherer is not None:
            coords = gatherer.get_coordinates(sel_str)
            rgs = []
            for frame_coords in coords:
                rgs.append(rg_value(frame_coords))
            result = np.array(rgs)
        else:
            sel = u.select_atoms(sel_str)
            rgs = []
            for _ in u.trajectory:
                rgs.append(rg_value(sel.positions))
            result = np.array(rgs)

        self.rg_cache[cache_key] = result
        return result

    def contact_distances(
        self,
        u: mda.Universe,
        selA: str,
        selB: str,
        gatherer: Optional[FrameGatherer] = None,
    ) -> np.ndarray:
        cache_key = (selA, selB)
        if cache_key in self.contact_cache:
            return self.contact_cache[cache_key]

        if gatherer is not None:
            coordsA = gatherer.get_coordinates(selA)
            coordsB = gatherer.get_coordinates(selB)
            dists = []
            for frame_idx in range(gatherer.get_n_frames()):
                dists.append(contact_min_distance(coordsA[frame_idx], coordsB[frame_idx]))
            result = np.array(dists)
        else:
            A = u.select_atoms(selA)
            B = u.select_atoms(selB)
            if len(A) == 0 or len(B) == 0:
                raise ValueError("Empty selection for contacts.")
            dists = []
            for _ in u.trajectory:
                dists.append(contact_min_distance(A.positions, B.positions))
            result = np.array(dists)

        self.contact_cache[cache_key] = result
        return result

    def pca_on_fluctuations(
        self,
        u: mda.Universe,
        sel_str: str,
        n_components: int = 5,
        aligned: bool = True,
        gatherer: Optional[FrameGatherer] = None,
    ) -> Tuple[np.ndarray, PCA]:
        cache_key = (sel_str, n_components)
        if cache_key in self.pca_cache:
            return self.pca_cache[cache_key]

        if gatherer is not None:
            coords = gatherer.get_coordinates(sel_str)
        else:
            sel = u.select_atoms(sel_str)
            if not aligned:
                warnings.warn(
                    "Computing PCA on unaligned trajectory. "
                    "Consider using AlignedTrajectory first."
                )
                align.AlignTraj(u, u, select=sel_str, in_memory=True).run()

            coords = []
            for _ in u.trajectory:
                coords.append(sel.positions.copy())
            coords = np.array(coords)

        pcs, pca = pca_from_coords(coords, n_components=n_components)
        result = (pcs, pca)
        self.pca_cache[cache_key] = result
        return result

    def local_affine_strain_proxy(
        self,
        u: mda.Universe,
        sel_str: str,
        window: int = 10,
        lag: int = 1,
        gatherer: Optional[FrameGatherer] = None,
    ) -> np.ndarray:
        cache_key = (sel_str, window, lag)
        if cache_key in self.strain_cache:
            return self.strain_cache[cache_key]

        if gatherer is not None:
            X = gatherer.get_coordinates(sel_str)
        else:
            sel = u.select_atoms(sel_str)
            coords = []
            for _ in u.trajectory:
                coords.append(sel.positions.copy())
            X = np.array(coords)

        T, _, _ = X.shape
        strain = np.full(T, np.nan)
        for t in range(0, T - lag):
            if t < window:
                continue
            A = X[t - window : t, :, :].reshape(-1, 3)
            B = X[t - window + lag : t + lag, :, :].reshape(-1, 3)
            A_aug = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
            Xsol, *_ = np.linalg.lstsq(A_aug, B, rcond=None)
            F = Xsol[:3, :]
            C = F.T @ F
            E = 0.5 * (C - np.eye(3))
            strain[t] = np.linalg.norm(E, ord="fro")

        self.strain_cache[cache_key] = strain
        return strain

    def clear_cache(self) -> None:
        self.rmsd_cache.clear()
        self.rmsf_cache.clear()
        self.rg_cache.clear()
        self.contact_cache.clear()
        self.pca_cache.clear()
        self.strain_cache.clear()


