from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit.Chem import Draw

try:
    import MDAnalysis as mda
except ImportError:
    import sys
    sys.stderr.write("MDAnalysis is required. pip install MDAnalysis\n")
    raise

# Optional plotting
try:
    import matplotlib.pyplot as plt  # type: ignore
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore
    warnings.warn("matplotlib not available. Plotting will be disabled.")

# Endpoint finder (now in EndpointAnalyzer module)
try:
    from src.EndpointAnalyzer import EndpointsFinder  # type: ignore
except ImportError:
    # Fallback to old locations for backward compatibility
    try:
        from MD_analysis.endpoints_finder import EndpointsFinder  # type: ignore
    except ImportError:
        try:
            from endpoints_finder import EndpointsFinder  # type: ignore
        except ImportError:
            EndpointsFinder = None  # type: ignore
            warnings.warn("endpoints_finder module not found. Endpoint-based metrics will be disabled.")


class Plotter:
    """Handle plotting operations for trajectory analysis."""

    def __init__(self) -> None:
        self.output_prefix: Optional[str] = None
        self.plots_generated: List[str] = []
        self.figure_size: Tuple[int, int] = (12, 8)
        self.dpi: int = 300

    def plot_endpoint_distances(
        self,
        endpoint_dists_array: Union[np.ndarray, dict],
        residue_sel_list: List[str],
        out_prefix: str,
    ) -> None:
        """Plot distances between endpoint pairs over frames."""
        if not HAS_MATPLOTLIB or plt is None:
            warnings.warn("matplotlib not available. Skipping endpoint distance plots.")
            return

        if endpoint_dists_array is None:
            warnings.warn("Endpoint distances array is None. Skipping plots.")
            return

        if isinstance(endpoint_dists_array, dict):
            all_pairs = endpoint_dists_array.get("all_pairs", {})
            T = endpoint_dists_array.get("n_frames", 0)
            n_res = endpoint_dists_array.get("n_residues", 0)

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
                            ax.plot(
                                frames,
                                mean_dists,
                                label=label,
                                alpha=0.7,
                                linewidth=1.5,
                            )

            ax.set_xlabel("Frame", fontsize=12)
            ax.set_ylabel("Endpoint Distance (Å)", fontsize=12)
            ax.set_title(
                "Endpoint Pair Distances Over Frames (Mean)",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)
            plt.tight_layout()
            plot_path = f"{out_prefix}_endpoint_distances.png"
            plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
            plt.close()
            self.output_prefix = out_prefix
            self.plots_generated.append(plot_path)
            print(f"Endpoint distance plot saved to {plot_path}")

    def plot_endpoint_volume_correlation(
        self,
        endpoint_dists_array: Union[np.ndarray, dict],
        volume: np.ndarray,
        residue_sel_list: List[str],
        correlation_df: pd.DataFrame,
        out_prefix: str,
        top_n: int = 5,
    ) -> None:
        """Plot the top N endpoint pairs with highest correlation to volume."""
        if not HAS_MATPLOTLIB or plt is None:
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
            all_pairs = endpoint_dists_array.get("all_pairs", {})

        for idx, (_, row) in enumerate(top_pairs.iterrows()):
            i, j = int(row["residue_i"]), int(row["residue_j"])
            corr = row["correlation"]
            p_val = row["p_value"]
            ep_i_idx = row.get("ep_i_idx", None)
            ep_j_idx = row.get("ep_j_idx", None)

            ax = axes[idx]
            ax2 = ax.twinx()

            if (
                is_dict_format
                and ep_i_idx is not None
                and ep_j_idx is not None
                and (i, j) in all_pairs
            ):
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

            line1 = ax.plot(
                frames,
                dists,
                "b-",
                label=f"Endpoint Distance ({row['residue_i_sel']}-{row['residue_j_sel']}, {ep_label})",
                linewidth=2,
                alpha=0.8,
            )
            line2 = ax2.plot(
                frames, volume, "r-", label="Cube Volume", linewidth=2, alpha=0.8
            )

            ax.set_xlabel("Frame", fontsize=11)
            ax.set_ylabel("Endpoint Distance (Å)", fontsize=11, color="b")
            ax2.set_ylabel("Volume (Å³)", fontsize=11, color="r")
            ax.tick_params(axis="y", labelcolor="b")
            ax2.tick_params(axis="y", labelcolor="r")

            title = f"Pair {i}-{j}"
            if ep_i_idx is not None and ep_j_idx is not None:
                title += f" (ep{int(ep_i_idx)}-ep{int(ep_j_idx)})"
            title += f": r={corr:.3f}, p={p_val:.3e}"
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc="upper left", fontsize=9)

        plt.tight_layout()
        plot_path = f"{out_prefix}_endpoint_volume_correlation.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Endpoint-volume correlation plot saved to {plot_path}")

    def plot_volume_change(
        self,
        volume: np.ndarray,
        out_prefix: str,
        frame_indices: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        xlabel: str = "Frame",
        ylabel: str = "Volume (Å³)",
        title: Optional[str] = None,
        show_stats: bool = True,
    ) -> None:
        """Plot volume change through frames."""
        if not HAS_MATPLOTLIB or plt is None:
            warnings.warn("matplotlib not available. Skipping volume change plot.")
            return

        if volume is None or len(volume) == 0:
            warnings.warn("Volume array is None or empty. Skipping plot.")
            return

        if times is not None and len(times) == len(volume):
            x_values = times
            xlabel = "Time (ps)"
        elif frame_indices is not None and len(frame_indices) == len(volume):
            x_values = frame_indices
        else:
            x_values = np.arange(len(volume))

        fig, ax = plt.subplots(figsize=self.figure_size)
        ax.plot(x_values, volume, "b-", linewidth=2, alpha=0.8, label="Volume")

        valid_volume = volume[~np.isnan(volume)]
        if show_stats and len(valid_volume) > 0:
            mean_vol = np.mean(valid_volume)
            std_vol = np.std(valid_volume)
            min_vol = np.min(valid_volume)
            max_vol = np.max(valid_volume)

            ax.axhline(
                mean_vol,
                color="r",
                linestyle="--",
                alpha=0.5,
                linewidth=1,
                label=f"Mean: {mean_vol:.2f} Å³",
            )
            ax.axhline(
                mean_vol + std_vol,
                color="orange",
                linestyle=":",
                alpha=0.5,
                linewidth=1,
                label=f"Mean ± Std: {mean_vol:.2f} ± {std_vol:.2f} Å³",
            )
            ax.axhline(
                mean_vol - std_vol,
                color="orange",
                linestyle=":",
                alpha=0.5,
                linewidth=1,
            )

            stats_text = (
                f"Mean: {mean_vol:.2f} Å³\n"
                f"Std: {std_vol:.2f} Å³\n"
                f"Min: {min_vol:.2f} Å³\n"
                f"Max: {max_vol:.2f} Å³"
            )
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if title is None:
            title = "Volume Change Through Frames"
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if show_stats and len(valid_volume) > 0:
            ax.legend(loc="best", fontsize=9)

        plt.tight_layout()
        plot_path = f"{out_prefix}_volume_change.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Volume change plot saved to {plot_path}")

    def plot_residue_endpoints(
        self,
        u: mda.Universe,
        residue_sel: str = "resid 1",
        out_prefix: str = "residue_endpoints",
    ) -> None:
        """Plot the endpoints of a residue."""
        sel = u.select_atoms(residue_sel)
        mol = sel.convert_to("RDKIT")
        if mol is None:
            raise ValueError("RDKit conversion failed")

        ef = EndpointsFinder() if EndpointsFinder is not None else None  # type: ignore[call-arg]
        if ef is None:
            raise RuntimeError("EndpointsFinder is not available for plotting.")

        m2d, _ = ef.to_2d_coords(mol)
        highlight_atoms = ef.find_endpoints(mol)
        img = Draw.MolToImage(
            m2d, size=(600, 400), highlightAtoms=highlight_atoms, highlightColor=(1, 0, 0)
        )
        img.save(out_prefix + ".png", format="png")


