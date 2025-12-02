from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
except ImportError:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

from ..MetricRegistry import MetricRegistry


class FrameProcessor:
    """Process frames in a single iteration, computing metrics on-the-fly."""

    def __init__(self, u: mda.Universe, metric_registry: Optional[MetricRegistry] = None):
        self.universe = u
        self.n_frames = len(u.trajectory)
        self.results: Dict[str, object] = {}
        self.frame_indices: np.ndarray = np.array([])
        self.times: np.ndarray = np.array([])
        self._processed = False

        if metric_registry is None:
            self.metric_registry = MetricRegistry()
            self._register_builtin_metrics()
        else:
            self.metric_registry = metric_registry

    def _register_builtin_metrics(self) -> None:
        """Register built-in metrics with the registry."""
        self.metric_registry.register(
            "rmsd", lambda u, fd, cfg: self._compute_rmsd_metric(u, fd, cfg), default=True
        )
        self.metric_registry.register(
            "rmsf", lambda u, fd, cfg: self._compute_rmsf_metric(u, fd, cfg), default=True
        )
        self.metric_registry.register(
            "rg", lambda u, fd, cfg: self._compute_rg_metric(u, fd, cfg), default=True
        )
        self.metric_registry.register(
            "pca", lambda u, fd, cfg: self._compute_pca_metric(u, fd, cfg), default=True
        )
        self.metric_registry.register(
            "strain", lambda u, fd, cfg: self._compute_strain_metric(u, fd, cfg), default=True
        )
        self.metric_registry.register(
            "contacts",
            lambda u, fd, cfg: self._compute_contacts_metric(u, fd, cfg),
            default=False,
        )

    def _compute_rmsd_metric(self, universe, frame_data, config):
        atoms = universe.select_atoms(config["sel"])
        ref_pos = config["ref_pos"]
        _, rmsd_val = align.rotation_matrix(atoms.positions, ref_pos)
        return rmsd_val, False

    def _compute_rmsf_metric(self, universe, frame_data, config):
        atoms = universe.select_atoms(config["sel"])
        return atoms.positions.copy(), True

    def _compute_rg_metric(self, universe, frame_data, config):
        atoms = universe.select_atoms(config["sel"])
        com = atoms.center_of_mass()
        rg2 = ((atoms.positions - com) ** 2).sum(axis=1).mean()
        return np.sqrt(rg2), False

    def _compute_pca_metric(self, universe, frame_data, config):
        atoms = universe.select_atoms(config["sel"])
        return atoms.positions.copy().reshape(-1), True

    def _compute_strain_metric(self, universe, frame_data, config):
        atoms = universe.select_atoms(config["sel"])
        return atoms.positions.copy(), True

    def _compute_contacts_metric(self, universe, frame_data, config):
        selA = config.get("selA")
        selB = config.get("selB")
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

    def process_metrics(
        self,
        metrics_to_compute: Optional[List[str]] = None,
        metric_configs: Optional[dict] = None,
        **kwargs,
    ) -> Dict[str, object]:
        """Compute selected metrics in a single iteration through the trajectory."""
        if self._processed:
            return self.results

        if metrics_to_compute is None and metric_configs is None:
            if any(k in kwargs for k in ["rmsd_sel", "rmsf_sel", "rg_sel", "pca_sel"]):
                return self.process_all_metrics(**kwargs)
            metrics_to_compute = self.metric_registry.get_default_metrics()

        if metrics_to_compute is None:
            metrics_to_compute = self.metric_registry.get_default_metrics()

        if metric_configs is None:
            metric_configs = {}

        for metric_name in metrics_to_compute:
            if not self.metric_registry.has_metric(metric_name):
                raise ValueError(
                    f"Metric '{metric_name}' is not registered. "
                    f"Available metrics: {self.metric_registry.list_metrics()}"
                )

        configs: Dict[str, dict] = {}
        frame_data_storage: Dict[str, list] = {}

        self.frame_indices = np.zeros(self.n_frames, dtype=int)
        self.times = np.zeros(self.n_frames)

        # Prepare configurations for each metric
        for metric_name in metrics_to_compute:
            config = metric_configs.get(metric_name, {}).copy()

            if metric_name == "rmsd":
                sel = config.get("sel", kwargs.get("rmsd_sel"))
                ref_frame = config.get("ref_frame", kwargs.get("rmsd_ref_frame", 0))
                if sel is None:
                    raise ValueError(
                        "rmsd requires 'sel' in metric_configs or rmsd_sel parameter"
                    )
                atoms = self.universe.select_atoms(sel)
                self.universe.trajectory[ref_frame]
                config["sel"] = sel
                config["ref_pos"] = atoms.positions.copy()
                configs[metric_name] = config
                frame_data_storage[metric_name] = []

            elif metric_name == "rmsf":
                sel = config.get("sel", kwargs.get("rmsf_sel"))
                if sel is None:
                    raise ValueError(
                        "rmsf requires 'sel' in metric_configs or rmsf_sel parameter"
                    )
                config["sel"] = sel
                configs[metric_name] = config
                frame_data_storage[metric_name] = []

            elif metric_name == "rg":
                sel = config.get("sel", kwargs.get("rg_sel"))
                if sel is None:
                    raise ValueError(
                        "rg requires 'sel' in metric_configs or rg_sel parameter"
                    )
                config["sel"] = sel
                configs[metric_name] = config
                frame_data_storage[metric_name] = []

            elif metric_name == "pca":
                sel = config.get("sel", kwargs.get("pca_sel"))
                n_components = config.get(
                    "n_components", kwargs.get("pca_n_components", 5)
                )
                if sel is None:
                    raise ValueError(
                        "pca requires 'sel' in metric_configs or pca_sel parameter"
                    )
                config["sel"] = sel
                config["n_components"] = n_components
                configs[metric_name] = config
                frame_data_storage[metric_name] = []

            elif metric_name == "strain":
                sel = config.get("sel", kwargs.get("strain_sel", kwargs.get("pca_sel")))
                window = config.get("window", kwargs.get("strain_window", 10))
                lag = config.get("lag", kwargs.get("strain_lag", 1))
                if sel is None:
                    raise ValueError(
                        "strain requires 'sel' in metric_configs or "
                        "strain_sel/pca_sel parameter"
                    )
                config["sel"] = sel
                config["window"] = window
                config["lag"] = lag
                configs[metric_name] = config
                frame_data_storage[metric_name] = []

            elif metric_name == "contacts":
                selA = config.get("selA", kwargs.get("contact_selA"))
                selB = config.get("selB", kwargs.get("contact_selB"))
                if selA is None or selB is None:
                    warnings.warn(
                        "contacts metric requires both selA and selB. Skipping."
                    )
                    continue
                config["selA"] = selA
                config["selB"] = selB
                configs[metric_name] = config
                frame_data_storage[metric_name] = []

        # Single pass through the trajectory
        for frame_idx, ts in enumerate(self.universe.trajectory):
            self.frame_indices[frame_idx] = ts.frame
            self.times[frame_idx] = ts.time

            for metric_name in metrics_to_compute:
                if metric_name not in configs:
                    continue
                metric_func = self.metric_registry.get(metric_name)
                if metric_func is None:
                    continue
                try:
                    result, needs_postprocessing = metric_func(
                        self.universe, {}, configs[metric_name]
                    )
                    if needs_postprocessing:
                        frame_data_storage[metric_name].append(result)
                    else:
                        if metric_name not in frame_data_storage:
                            frame_data_storage[metric_name] = []
                        frame_data_storage[metric_name].append(result)
                except Exception as e:
                    warnings.warn(
                        f"Error computing {metric_name} at frame {frame_idx}: {e}"
                    )
                    if metric_name not in frame_data_storage:
                        frame_data_storage[metric_name] = []
                    frame_data_storage[metric_name].append(np.nan)

        # Post-process
        self.results = {}
        for metric_name in metrics_to_compute:
            if metric_name not in frame_data_storage or len(
                frame_data_storage[metric_name]
            ) == 0:
                continue

            if metric_name == "rmsd":
                self.results["rmsd"] = np.array(frame_data_storage[metric_name])

            elif metric_name == "rmsf":
                rmsf_coords = np.array(frame_data_storage[metric_name])
                mean = rmsf_coords.mean(axis=0)
                diffsq = (rmsf_coords - mean) ** 2
                rmsf_vals = np.sqrt(diffsq.sum(axis=2).mean(axis=0))
                self.results["rmsf"] = rmsf_vals
                del rmsf_coords

            elif metric_name == "rg":
                self.results["rg"] = np.array(frame_data_storage[metric_name])

            elif metric_name == "pca":
                pca_coords = np.array(frame_data_storage[metric_name])
                Xc = pca_coords - pca_coords.mean(axis=0)
                n_components = configs[metric_name]["n_components"]
                pca = PCA(n_components=n_components, svd_solver="auto")
                pcs = pca.fit_transform(Xc)
                self.results["pcs"] = pcs
                self.results["pca_model"] = pca
                del pca_coords

            elif metric_name == "strain":
                strain_coords = np.array(frame_data_storage[metric_name])
                T, _, _ = strain_coords.shape
                strain = np.full(T, np.nan)
                window = configs[metric_name]["window"]
                lag = configs[metric_name]["lag"]
                for t in range(0, T - lag):
                    if t < window:
                        continue
                    A = strain_coords[t - window : t, :, :].reshape(-1, 3)
                    B = strain_coords[t - window + lag : t + lag, :, :].reshape(-1, 3)
                    A_aug = np.concatenate([A, np.ones((A.shape[0], 1))], axis=1)
                    Xsol, *_ = np.linalg.lstsq(A_aug, B, rcond=None)
                    F = Xsol[:3, :]
                    C = F.T @ F
                    E = 0.5 * (C - np.eye(3))
                    strain[t] = np.linalg.norm(E, ord="fro")
                self.results["strain"] = strain
                del strain_coords

            elif metric_name == "contacts":
                contact_vals = frame_data_storage[metric_name]
                valid_vals = [v for v in contact_vals if v is not None]
                self.results["contacts"] = (
                    np.array(valid_vals) if valid_vals else None
                )

        self._processed = True
        return self.results

    def process_all_metrics(
        self,
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
        rmsd_ref_frame: int = 0,
    ) -> Dict[str, object]:
        """Backward-compatible wrapper to compute all metrics."""
        metrics_to_compute = ["rmsd", "rmsf", "rg", "pca", "strain"]
        if contact_selA and contact_selB:
            metrics_to_compute.append("contacts")

        metric_configs = {
            "rmsd": {"sel": rmsd_sel, "ref_frame": rmsd_ref_frame},
            "rmsf": {"sel": rmsf_sel},
            "rg": {"sel": rg_sel},
            "pca": {"sel": pca_sel, "n_components": pca_n_components},
            "strain": {
                "sel": strain_sel if strain_sel is not None else pca_sel,
                "window": strain_window,
                "lag": strain_lag,
            },
        }
        if contact_selA and contact_selB:
            metric_configs["contacts"] = {"selA": contact_selA, "selB": contact_selB}

        return self.process_metrics(
            metrics_to_compute=metrics_to_compute,
            metric_configs=metric_configs,
        )

    def get_frame_indices(self) -> np.ndarray:
        return self.frame_indices

    def get_times(self) -> np.ndarray:
        return self.times

    def get_n_frames(self) -> int:
        return self.n_frames

    def list_available_metrics(self) -> List[str]:
        return self.metric_registry.list_metrics()

    def register_custom_metric(self, name: str, metric_func, default: bool = False):
        self.metric_registry.register(name, metric_func, default=default)

    def reset_processing(self) -> None:
        self._processed = False
        self.results = {}
        self.frame_indices = np.array([])
        self.times = np.array([])


