from __future__ import annotations

import ast
import io
import os
import re
import warnings
from typing import List, Optional, Tuple, Union

# Try to import multiprocessing for parallel frame generation
try:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from multiprocessing import cpu_count
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False
    ProcessPoolExecutor = None
    as_completed = None
    cpu_count = None

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
    from matplotlib.animation import FuncAnimation  # type: ignore
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None  # type: ignore
    FuncAnimation = None  # type: ignore
    warnings.warn("matplotlib not available. Plotting will be disabled.")

# Optional GIF support
try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    imageio = None  # type: ignore
    warnings.warn("imageio not available. GIF generation will be disabled.")

# Endpoint finder (now in EndpointAnalyzer module)
from src.EndpointAnalyzer import EndpointsFinder, EndpointAnalyzer  # type: ignore
from src.EndpointAnalyzer import EndpointAnalyzerObserver  # type: ignore   


def _generate_timeline_frame(
    current_frame: int,
    entry_frames: List[int],
    exit_frames: List[int],
    entry_guest_indices: List[List[int]],
    exit_guest_indices: List[List[int]],
    sorted_guest_indices: List[int],
    guest_index_to_y: dict,
    guest_residence_stats: dict,
    x_min: float,
    x_max: float,
    figure_size: Tuple[int, int],
    dpi: int,
    show_connections: bool,
) -> np.ndarray:
    """
    Helper function to generate a single frame for the timeline GIF.
    This is a module-level function to enable multiprocessing.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    
    # Import imageio (needed in worker processes)
    try:
        import imageio.v2 as imageio
    except ImportError:
        import imageio
    
    fig, ax = plt.subplots(figsize=figure_size)
    
    # Filter events up to current frame
    current_entry_frames = [f for f in entry_frames if f <= current_frame]
    current_exit_frames = [f for f in exit_frames if f <= current_frame]
    current_entry_indices = [idx for i, idx in enumerate(entry_guest_indices) if entry_frames[i] <= current_frame]
    current_exit_indices = [idx for i, idx in enumerate(exit_guest_indices) if exit_frames[i] <= current_frame]

    # Plot entry-exit connections for each guest (up to current frame)
    if show_connections:
        # Build a timeline for each guest: list of (entry_frame, exit_frame) pairs
        guest_timelines = {gidx: [] for gidx in sorted_guest_indices}
        
        # Process entry events up to current frame
        for entry_frame, guest_indices in zip(entry_frames, entry_guest_indices):
            if entry_frame <= current_frame:
                for gidx in guest_indices:
                    if gidx in guest_timelines:
                        guest_timelines[gidx].append({'entry': entry_frame, 'exit': None})
        
        # Process exit events up to current frame and match with entries
        for exit_frame, guest_indices in zip(exit_frames, exit_guest_indices):
            if exit_frame <= current_frame:
                for gidx in guest_indices:
                    if gidx in guest_timelines:
                        # Find the most recent unmatched entry for this guest
                        for timeline_entry in reversed(guest_timelines[gidx]):
                            if timeline_entry['exit'] is None:
                                timeline_entry['exit'] = exit_frame
                                break
        
        # Draw horizontal lines for each guest's residence periods
        for gidx, timeline in guest_timelines.items():
            y_pos = guest_index_to_y[gidx]
            for period in timeline:
                entry_frame = period['entry']
                exit_frame = period['exit'] if period['exit'] is not None else current_frame
                ax.plot(
                    [entry_frame, exit_frame],
                    [y_pos, y_pos],
                    color='lightblue',
                    linewidth=3,
                    alpha=0.5,
                    zorder=1
                )

    # Plot entry events up to current frame
    entry_x = []
    entry_y = []
    for entry_frame, guest_indices in zip(entry_frames, entry_guest_indices):
        if entry_frame <= current_frame:
            for gidx in guest_indices:
                if gidx in guest_index_to_y:
                    entry_x.append(entry_frame)
                    entry_y.append(guest_index_to_y[gidx])

    if entry_x:
        ax.scatter(
            entry_x,
            entry_y,
            color='green',
            marker='^',
            s=150,
            zorder=3,
            label=f'Entry events (n={len(entry_x)})',
            edgecolors='darkgreen',
            linewidths=1.5
        )

    # Plot exit events up to current frame
    exit_x = []
    exit_y = []
    for exit_frame, guest_indices in zip(exit_frames, exit_guest_indices):
        if exit_frame <= current_frame:
            for gidx in guest_indices:
                if gidx in guest_index_to_y:
                    exit_x.append(exit_frame)
                    exit_y.append(guest_index_to_y[gidx])

    if exit_x:
        ax.scatter(
            exit_x,
            exit_y,
            color='red',
            marker='v',
            s=150,
            zorder=3,
            label=f'Exit events (n={len(exit_x)})',
            edgecolors='darkred',
            linewidths=1.5
        )

    # Add vertical line showing current frame
    ax.axvline(x=current_frame, color='black', linestyle='--', linewidth=2, alpha=0.5, zorder=2, label='Current frame')

    # Set labels and title
    ax.set_xlabel("Frame", fontsize=12)
    ax.set_ylabel("Guest Index", fontsize=12)
    ax.set_title(f"Guest Entry/Exit Timeline (Frame {current_frame})", fontsize=14, fontweight="bold")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, len(sorted_guest_indices) - 0.5)
    ax.set_yticks(range(len(sorted_guest_indices)))
    ax.set_yticklabels([str(gidx) for gidx in sorted_guest_indices])
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(loc='upper left', fontsize=9)

    # Add statistics text box
    stats_text = []
    if guest_residence_stats.get('n_entries', 0) > 0:
        stats_text.append(f"Total entries: {guest_residence_stats['n_entries']}")
        stats_text.append(f"Total exits: {guest_residence_stats['n_exits']}")
        stats_text.append(f"Unique guests: {len(sorted_guest_indices)}")
        if guest_residence_stats.get('first_entry_frame') is not None:
            stats_text.append(f"First entry: Frame {guest_residence_stats['first_entry_frame']}")
    
    if stats_text:
        ax.text(
            0.98,
            0.02,
            '\n'.join(stats_text),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

    plt.tight_layout()
    
    # Convert figure to image using BytesIO
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = imageio.imread(buf)
    return img


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
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)
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
        endpoint_observer: Optional[EndpointAnalyzerObserver] = None,
        residue_sel: Union[str, List[str]] = "resid 1",
        out_prefix: str = "residue_endpoints",
    ) -> None:
        """Plot the endpoints of a residue or multiple residues.
        
        Args:
            u: MDAnalysis Universe
            endpoint_observer: Optional EndpointAnalyzerObserver. If provided, uses stored_ep_indices for labeling
            residue_sel: Single residue selection string or list of residue selection strings
            out_prefix: Output file prefix
        """
        if not HAS_MATPLOTLIB or plt is None:
            warnings.warn("matplotlib not available. Skipping residue endpoint plots.")
            return

        ef = EndpointsFinder() if EndpointsFinder is not None else None  # type: ignore[call-arg]
        if ef is None:
            raise RuntimeError("EndpointsFinder is not available for plotting.")

        # Handle both string and list inputs
        if isinstance(residue_sel, str):
            residue_sel_list = [residue_sel]
        else:
            residue_sel_list = residue_sel

        n_residues = len(residue_sel_list)
        
        # Create subplots if multiple residues
        if n_residues == 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            axes = [ax]
        else:
            n_cols = min(2, n_residues)
            n_rows = (n_residues + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 8 * n_rows))
            # Flatten axes array if it's 2D
            if hasattr(axes, 'flatten'):
                axes = axes.flatten()
            elif not isinstance(axes, (list, np.ndarray)):
                axes = [axes]

        for idx, sel_str in enumerate(residue_sel_list):
            ax = axes[idx] if n_residues > 1 else axes[0]
            
            sel = u.select_atoms(sel_str)
            mol = sel.convert_to("RDKIT")
            if mol is None:
                warnings.warn(f"RDKit conversion failed for {sel_str}. Skipping.")
                ax.text(0.5, 0.5, f"Failed to convert\n{sel_str}", 
                       ha='center', va='center', transform=ax.transAxes)
                continue

            m2d, xy = ef.to_2d_coords(mol)
            
            # Get endpoint indices - use endpoint_observer if provided, otherwise find them
            if endpoint_observer is not None and hasattr(endpoint_observer, 'stored_ep_indices'):
                # Find the index of sel_str in endpoint_observer.residue_sel_list
                try:
                    residue_idx = endpoint_observer.residue_sel_list.index(sel_str)
                    mda_endpoint_indices = endpoint_observer.stored_ep_indices[residue_idx]
                except (ValueError, IndexError):
                    # If sel_str not found in observer's list, fall back to finding endpoints
                    mda_endpoint_indices = None
            else:
                mda_endpoint_indices = None
            
            if mda_endpoint_indices is not None and len(mda_endpoint_indices) > 0:
                # Use stored endpoint indices from observer
                # Map MDAnalysis universe indices to RDKit indices for highlighting
                universe_to_rdkit = {}
                for rdkit_idx in range(len(sel)):
                    universe_id = int(sel[rdkit_idx].id)
                    universe_to_rdkit[universe_id] = rdkit_idx
                
                # Get RDKit indices for highlighting, preserving order
                highlight_atoms = []
                valid_mda_indices = []
                for universe_idx in mda_endpoint_indices:
                    if universe_idx in universe_to_rdkit:
                        highlight_atoms.append(universe_to_rdkit[universe_idx])
                        valid_mda_indices.append(universe_idx)
                mda_endpoint_indices = valid_mda_indices  # Update to only valid indices
            else:
                # Fall back to finding endpoints using EndpointsFinder
                highlight_atoms = ef.find_endpoints(mol)  # RDKit indices for highlighting
                # Map RDKit indices to MDAnalysis universe indices
                mda_endpoint_indices = [int(sel[rdkit_idx].id) for rdkit_idx in highlight_atoms]
            
            # Create mapping for labeling (RDKit index -> MDAnalysis universe index)
            rdkit_to_universe = {rdkit_idx: int(sel[rdkit_idx].id) for rdkit_idx in highlight_atoms}
            
            # Try to use rdMolDraw2D for better coordinate control
            try:
                from rdkit.Chem import rdMolDraw2D
                import io
                try:
                    from PIL import Image
                except ImportError:
                    try:
                        from PIL import Image as PILImage
                        Image = PILImage
                    except ImportError:
                        Image = None
                
                if Image is None:
                    raise ImportError("PIL not available")
                
                # Create drawer
                drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
                drawer.SetDrawOptions(drawer.drawOptions())
                
                # Set up highlight colors
                highlight_colors = {atom_idx: (1.0, 0.0, 0.0) for atom_idx in highlight_atoms}
                
                # Draw molecule
                drawer.DrawMolecule(m2d, highlightAtoms=highlight_atoms, highlightAtomColors=highlight_colors)
                drawer.FinishDrawing()
                
                # Get the image
                img_data = drawer.GetDrawingText()
                img = Image.open(io.BytesIO(img_data))
                
            except (ImportError, AttributeError):
                # Fallback to MolToImage if rdMolDraw2D is not available
                img = Draw.MolToImage(
                    m2d, size=(800, 600), highlightAtoms=highlight_atoms, highlightColor=(1, 0, 0)
                )
            
            # Calculate coordinate transformation
            # Use the 2D coordinates from to_2d_coords
            x_coords = xy[:, 0]
            y_coords = xy[:, 1]
            x_min, x_max = x_coords.min(), x_coords.max()
            y_min, y_max = y_coords.min(), y_coords.max()
            
            x_range = x_max - x_min if x_max > x_min else 1.0
            y_range = y_max - y_min if y_max > y_min else 1.0
            padding_factor = 0.15  # RDKit typically uses ~15% padding
            
            # Calculate scale factors (drawer uses most of the image with some padding)
            scale_x = (800 * (1 - 2 * padding_factor)) / x_range if x_range > 0 else 1.0
            scale_y = (600 * (1 - 2 * padding_factor)) / y_range if y_range > 0 else 1.0
            scale = min(scale_x, scale_y)  # Use uniform scaling
            
            # Calculate offsets (center the molecule)
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            offset_x = 400 - center_x * scale
            offset_y = 300 + center_y * scale  # Note: y is inverted in image coordinates
            
            # Display image
            ax.imshow(img, extent=[0, 800, 600, 0])  # Set extent so coordinates match
            ax.axis('off')
            
            # Label each endpoint with its universe index
            # highlight_atoms and mda_endpoint_indices are now aligned
            for rdkit_idx, universe_idx in zip(highlight_atoms, mda_endpoint_indices):
                if rdkit_idx < len(xy):
                    x, y = xy[rdkit_idx]
                    
                    # Transform to image coordinates
                    x_pixel = x * scale + offset_x
                    y_pixel = -y * scale + offset_y  # Invert y for image coordinates
                    
                    # Add label with universe index
                    ax.annotate(
                        f"{universe_idx}",
                        xy=(x_pixel, y_pixel),
                        xytext=(8, -8),
                        textcoords='offset points',
                        fontsize=14,
                        fontweight='bold',
                        color='blue',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8, edgecolor='blue', linewidth=2),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2, connectionstyle='arc3,rad=0.1')
                    )
            
            # Set title
            title = f"Endpoints: {sel_str}"
            if n_residues > 1:
                title += f" (Residue {idx + 1})"
            ax.set_title(title, fontsize=12, fontweight='bold')

        # Hide unused subplots
        for idx in range(n_residues, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plot_path = f"{out_prefix}.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Residue endpoint plot saved to {plot_path}")

    def plot_guest_entering_events(
        self,
        guest_residence_stats: dict,
        out_prefix: str,
        show_exits: bool = True,
        show_durations: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Plot guest entering events as a function of frame.
        
        Args:
            guest_residence_stats: Dictionary from GSAnalyzerObserver.get_guest_residence_stats()
            out_prefix: Output file prefix
            show_exits: If True, also plot exit events
            show_durations: If True, show shaded regions for durations inside host
            max_frames: Optional maximum frame number for x-axis limit. If None, uses max frame from events
        """
        if not HAS_MATPLOTLIB or plt is None:
            warnings.warn("matplotlib not available. Skipping guest entering events plot.")
            return

        if not guest_residence_stats:
            warnings.warn("Guest residence stats is empty. Skipping plot.")
            return

        entry_frames = guest_residence_stats.get('entry_frames', [])
        exit_frames = guest_residence_stats.get('exit_frames', [])
        entry_times = guest_residence_stats.get('entry_times', [])
        exit_times = guest_residence_stats.get('exit_times', [])
        durations_inside = guest_residence_stats.get('durations_inside', [])

        if not entry_frames and not exit_frames:
            warnings.warn("No entry or exit events found. Skipping plot.")
            return

        # Determine frame range
        all_frames = entry_frames + exit_frames
        if all_frames:
            min_frame = min(all_frames)
            max_frame = max(all_frames)
            if max_frames is not None:
                max_frame = max(max_frame, max_frames)
            # Add some padding
            frame_range = max_frame - min_frame
            x_min = max(0, min_frame - frame_range * 0.05)
            x_max = max_frame + frame_range * 0.05
        else:
            x_min, x_max = 0, 100

        fig, ax = plt.subplots(figsize=self.figure_size)

        # Plot durations as shaded regions if requested
        if show_durations and durations_inside and entry_frames:
            for i, (entry_frame, duration) in enumerate(zip(entry_frames, durations_inside)):
                # Find corresponding exit frame
                if i < len(exit_frames):
                    exit_frame = exit_frames[i]
                else:
                    # Guest still inside at end of trajectory
                    exit_frame = max_frame if max_frames else entry_frame + int(duration)
                
                # Shade the region between entry and exit
                ax.axvspan(
                    entry_frame,
                    exit_frame,
                    alpha=0.2,
                    color='green',
                    label='Inside host' if i == 0 else ''
                )

        # Plot entry events
        if entry_frames:
            ax.scatter(
                entry_frames,
                [1] * len(entry_frames),
                color='green',
                marker='^',
                s=100,
                zorder=3,
                label=f'Entry events (n={len(entry_frames)})',
                edgecolors='darkgreen',
                linewidths=1.5
            )
            # Add frame labels for entry events
            for i, (frame, time) in enumerate(zip(entry_frames, entry_times)):
                ax.annotate(
                    f'Frame {frame}\n({time:.1f} ps)',
                    xy=(frame, 1),
                    xytext=(10, 20),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5, connectionstyle='arc3,rad=0.2')
                )

        # Plot exit events if requested
        if show_exits and exit_frames:
            ax.scatter(
                exit_frames,
                [0.5] * len(exit_frames),
                color='red',
                marker='v',
                s=100,
                zorder=3,
                label=f'Exit events (n={len(exit_frames)})',
                edgecolors='darkred',
                linewidths=1.5
            )
            # Add frame labels for exit events
            for i, (frame, time) in enumerate(zip(exit_frames, exit_times)):
                ax.annotate(
                    f'Frame {frame}\n({time:.1f} ps)',
                    xy=(frame, 0.5),
                    xytext=(10, -30),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5, connectionstyle='arc3,rad=-0.2')
                )

        # Add statistics text box
        stats_text = []
        if guest_residence_stats.get('n_entries', 0) > 0:
            stats_text.append(f"Total entries: {guest_residence_stats['n_entries']}")
            stats_text.append(f"Total exits: {guest_residence_stats['n_exits']}")
            if guest_residence_stats.get('first_entry_frame') is not None:
                stats_text.append(f"First entry: Frame {guest_residence_stats['first_entry_frame']}")
            if durations_inside:
                stats_text.append(f"Avg stay: {np.mean(durations_inside):.1f} ps")
                stats_text.append(f"Max stay: {np.max(durations_inside):.1f} ps")
            if guest_residence_stats.get('total_time_inside', 0) > 0:
                stats_text.append(f"Total inside: {guest_residence_stats['total_time_inside']:.1f} ps")
        
        if stats_text:
            ax.text(
                0.98,
                0.98,
                '\n'.join(stats_text),
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )

        ax.set_xlabel("Frame", fontsize=12)
        ax.set_ylabel("Event Type", fontsize=12)
        ax.set_title("Guest Entering/Exiting Events", fontsize=14, fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.2, 1.5)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(['Exit', 'Entry'])
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='upper left', fontsize=9)

        plt.tight_layout()
        plot_path = f"{out_prefix}_guest_entering_events.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Guest entering events plot saved to {plot_path}")

    def plot_guest_entry_exit_timeline(
        self,
        guest_residence_stats: dict,
        out_prefix: str,
        show_connections: bool = True,
        max_frames: Optional[int] = None,
    ) -> None:
        """
        Plot guest entry and exit frames with guest index on Y-axis and frame on X-axis.
        
        Args:
            guest_residence_stats: Dictionary from GSAnalyzerObserver.get_guest_residence_stats()
            out_prefix: Output file prefix
            show_connections: If True, draw horizontal lines connecting entry-exit pairs for each guest
            max_frames: Optional maximum frame number for x-axis limit. If None, uses max frame from events
        """
        if not HAS_MATPLOTLIB or plt is None:
            warnings.warn("matplotlib not available. Skipping guest entry/exit timeline plot.")
            return

        if not guest_residence_stats:
            warnings.warn("Guest residence stats is empty. Skipping plot.")
            return

        entry_frames = guest_residence_stats.get('entry_frames', [])
        exit_frames = guest_residence_stats.get('exit_frames', [])
        entry_guest_indices = guest_residence_stats.get('entry_guest_indices', [])
        exit_guest_indices = guest_residence_stats.get('exit_guest_indices', [])

        if not entry_frames and not exit_frames:
            warnings.warn("No entry or exit events found. Skipping plot.")
            return

        # Collect all unique guest indices
        all_guest_indices = set()
        for indices in entry_guest_indices:
            all_guest_indices.update(indices)
        for indices in exit_guest_indices:
            all_guest_indices.update(indices)
        
        if not all_guest_indices:
            warnings.warn("No guest indices found in events. Skipping plot.")
            return

        # Sort guest indices for consistent Y-axis ordering
        sorted_guest_indices = sorted(all_guest_indices)
        guest_index_to_y = {gidx: i for i, gidx in enumerate(sorted_guest_indices)}

        # Determine frame range
        all_frames = entry_frames + exit_frames
        if all_frames:
            min_frame = min(all_frames)
            max_frame = max(all_frames)
            if max_frames is not None:
                max_frame = max(max_frame, max_frames)
            # Add some padding
            frame_range = max_frame - min_frame
            x_min = max(0, min_frame - frame_range * 0.05)
            x_max = max_frame + frame_range * 0.05
        else:
            x_min, x_max = 0, 100

        fig, ax = plt.subplots(figsize=self.figure_size)

        # Plot entry-exit connections for each guest
        if show_connections:
            # Build a timeline for each guest: list of (entry_frame, exit_frame) pairs
            guest_timelines = {gidx: [] for gidx in sorted_guest_indices}
            
            # Process entry events
            for entry_frame, guest_indices in zip(entry_frames, entry_guest_indices):
                for gidx in guest_indices:
                    if gidx in guest_timelines:
                        guest_timelines[gidx].append({'entry': entry_frame, 'exit': None})
            
            # Process exit events and match with entries
            exit_idx = 0
            for exit_frame, guest_indices in zip(exit_frames, exit_guest_indices):
                for gidx in guest_indices:
                    if gidx in guest_timelines:
                        # Find the most recent unmatched entry for this guest
                        for timeline_entry in reversed(guest_timelines[gidx]):
                            if timeline_entry['exit'] is None:
                                timeline_entry['exit'] = exit_frame
                                break
            
            # Draw horizontal lines for each guest's residence periods
            for gidx, timeline in guest_timelines.items():
                y_pos = guest_index_to_y[gidx]
                for period in timeline:
                    entry_frame = period['entry']
                    exit_frame = period['exit'] if period['exit'] is not None else x_max
                    ax.plot(
                        [entry_frame, exit_frame],
                        [y_pos, y_pos],
                        color='lightblue',
                        linewidth=3,
                        alpha=0.5,
                        zorder=1
                    )

        # Plot entry events
        entry_x = []
        entry_y = []
        for entry_frame, guest_indices in zip(entry_frames, entry_guest_indices):
            for gidx in guest_indices:
                if gidx in guest_index_to_y:
                    entry_x.append(entry_frame)
                    entry_y.append(guest_index_to_y[gidx])

        if entry_x:
            ax.scatter(
                entry_x,
                entry_y,
                color='green',
                marker='^',
                s=150,
                zorder=3,
                label=f'Entry events (n={len(entry_x)})',
                edgecolors='darkgreen',
                linewidths=1.5
            )

        # Plot exit events
        exit_x = []
        exit_y = []
        for exit_frame, guest_indices in zip(exit_frames, exit_guest_indices):
            for gidx in guest_indices:
                if gidx in guest_index_to_y:
                    exit_x.append(exit_frame)
                    exit_y.append(guest_index_to_y[gidx])

        if exit_x:
            ax.scatter(
                exit_x,
                exit_y,
                color='red',
                marker='v',
                s=150,
                zorder=3,
                label=f'Exit events (n={len(exit_x)})',
                edgecolors='darkred',
                linewidths=1.5
            )

        # Set labels and title
        ax.set_xlabel("Frame", fontsize=12)
        ax.set_ylabel("Guest Index", fontsize=12)
        ax.set_title("Guest Entry/Exit Timeline", fontsize=14, fontweight="bold")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.5, len(sorted_guest_indices) - 0.5)
        ax.set_yticks(range(len(sorted_guest_indices)))
        ax.set_yticklabels([str(gidx) for gidx in sorted_guest_indices])
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(loc='upper left', fontsize=9)

        # Add statistics text box
        stats_text = []
        if guest_residence_stats.get('n_entries', 0) > 0:
            stats_text.append(f"Total entries: {guest_residence_stats['n_entries']}")
            stats_text.append(f"Total exits: {guest_residence_stats['n_exits']}")
            stats_text.append(f"Unique guests: {len(sorted_guest_indices)}")
            if guest_residence_stats.get('first_entry_frame') is not None:
                stats_text.append(f"First entry: Frame {guest_residence_stats['first_entry_frame']}")
        
        if stats_text:
            ax.text(
                0.98,
                0.02,
                '\n'.join(stats_text),
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )

        plt.tight_layout()
        plot_path = f"{out_prefix}_guest_entry_exit_timeline.png"
        plt.savefig(plot_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        self.output_prefix = out_prefix
        self.plots_generated.append(plot_path)
        print(f"Guest entry/exit timeline plot saved to {plot_path}")

    def plot_guest_entry_exit_timeline_gif(
        self,
        csv_path: str,
        out_prefix: str,
        show_connections: bool = True,
        max_frames: Optional[int] = None,
        fps: float = 5.0,
        frame_step: int = 5,
        n_jobs: Optional[int] = None,
    ) -> None:
        """
        Create an animated GIF version of guest entry/exit timeline plot from CSV files.
        
        Args:
            csv_path: Path to guest_entering_events.csv file (will also look for guest_entering_stats.csv for metadata)
            out_prefix: Output file prefix
            show_connections: If True, draw horizontal lines connecting entry-exit pairs for each guest
            max_frames: Optional maximum frame number for x-axis limit. If None, uses max frame from events
            fps: Frames per second for the GIF animation
            frame_step: Step size for animation frames (1 = every frame, 10 = every 10th frame, etc.)
            n_jobs: Number of parallel workers for frame generation. If None, uses all available CPU cores.
                   If 1, runs sequentially. Use > 1 for parallel processing.
        """
        if not HAS_MATPLOTLIB or plt is None:
            warnings.warn("matplotlib not available. Skipping guest entry/exit timeline GIF.")
            return
        
        if not HAS_IMAGEIO or imageio is None:
            warnings.warn("imageio not available. Skipping GIF generation.")
            return

        # Determine if csv_path is events or stats file
        csv_dir = os.path.dirname(csv_path)
        csv_basename = os.path.basename(csv_path)
        
        if "_guest_entering_events.csv" in csv_basename:
            events_csv_path = csv_path
            # Try to find stats file
            stats_csv_path = os.path.join(
                csv_dir,
                csv_basename.replace("_guest_entering_events.csv", "_guest_entering_stats.csv")
            )
        elif "_guest_entering_stats.csv" in csv_basename:
            # Legacy: if stats file is provided, look for events file
            stats_csv_path = csv_path
            events_csv_path = os.path.join(
                csv_dir,
                csv_basename.replace("_guest_entering_stats.csv", "_guest_entering_events.csv")
            )
        else:
            # Assume it's the events file
            events_csv_path = csv_path
            stats_csv_path = None

        # Helper function to parse numpy type strings like 'np.int64(708)'
        def parse_numpy_int_string(s):
            """Extract integer from numpy type string like 'np.int64(708)' or just return int"""
            if isinstance(s, (int, np.integer)):
                return int(s)
            if isinstance(s, str):
                # Match patterns like np.int64(708), np.int32(123), etc.
                match = re.search(r'np\.int\d+\((\d+)\)', s)
                if match:
                    return int(match.group(1))
                # Try direct conversion
                try:
                    return int(float(s))
                except:
                    return None
            try:
                return int(float(s))
            except:
                return None

        # Read events CSV file (primary source)
        entry_frames = []
        exit_frames = []
        entry_times = []
        exit_times = []
        entry_guest_indices = []
        exit_guest_indices = []
        
        if not os.path.exists(events_csv_path):
            warnings.warn(f"Events CSV file not found at {events_csv_path}. Cannot create GIF.")
            return
        
        try:
            events_df = pd.read_csv(events_csv_path)
            
            # Parse events
            for _, row in events_df.iterrows():
                event_type = str(row['event_type']).strip().lower()
                
                # Parse frame (handle numpy types)
                frame_val = row['frame']
                frame = parse_numpy_int_string(frame_val)
                if frame is None:
                    continue
                
                # Parse time
                time_val = row['time']
                try:
                    time = float(time_val)
                except:
                    continue
                
                # Parse guest_indices (handle numpy type strings)
                guest_indices_str = row.get('guest_indices', '[]')
                guest_indices = []
                
                if pd.isna(guest_indices_str) or guest_indices_str == '':
                    guest_indices = []
                elif isinstance(guest_indices_str, str):
                    # Remove quotes if present
                    guest_indices_str = guest_indices_str.strip('"\'')
                    
                    # Try to extract all np.int64(...) patterns
                    numpy_int_pattern = r'np\.int\d+\((\d+)\)'
                    matches = re.findall(numpy_int_pattern, guest_indices_str)
                    if matches:
                        guest_indices = [int(m) for m in matches]
                    else:
                        # Try ast.literal_eval for normal list format
                        try:
                            parsed = ast.literal_eval(guest_indices_str)
                            if isinstance(parsed, list):
                                guest_indices = [parse_numpy_int_string(x) for x in parsed if parse_numpy_int_string(x) is not None]
                            else:
                                idx = parse_numpy_int_string(parsed)
                                if idx is not None:
                                    guest_indices = [idx]
                        except:
                            # Fallback: try comma-separated
                            parts = guest_indices_str.strip('[]()').split(',')
                            for part in parts:
                                idx = parse_numpy_int_string(part.strip())
                                if idx is not None:
                                    guest_indices.append(idx)
                elif isinstance(guest_indices_str, list):
                    guest_indices = [parse_numpy_int_string(x) for x in guest_indices_str if parse_numpy_int_string(x) is not None]
                else:
                    idx = parse_numpy_int_string(guest_indices_str)
                    if idx is not None:
                        guest_indices = [idx]
                
                if event_type == 'entry':
                    entry_frames.append(frame)
                    entry_times.append(time)
                    entry_guest_indices.append(guest_indices)
                elif event_type == 'exit':
                    exit_frames.append(frame)
                    exit_times.append(time)
                    exit_guest_indices.append(guest_indices)
        except Exception as e:
            warnings.warn(f"Failed to read events CSV file {events_csv_path}: {e}")
            import traceback
            traceback.print_exc()
            return

        # Read stats CSV for additional metadata (optional)
        guest_residence_stats = {}
        if stats_csv_path and os.path.exists(stats_csv_path):
            try:
                stats_df = pd.read_csv(stats_csv_path)
                for _, row in stats_df.iterrows():
                    metric = row['metric']
                    value = row['value']
                    if pd.notna(value):
                        if metric in ['first_entry_frame', 'n_entries', 'n_exits']:
                            guest_residence_stats[metric] = int(value)
                        elif metric in ['first_entry_time', 'total_time_inside', 'total_time_outside',
                                       'avg_stay_duration', 'max_stay_duration', 'min_stay_duration']:
                            guest_residence_stats[metric] = float(value)
                        else:
                            guest_residence_stats[metric] = value
            except Exception as e:
                warnings.warn(f"Failed to read stats CSV file {stats_csv_path}: {e}. Continuing without stats metadata.")
        
        # Calculate stats from events if not available
        if 'n_entries' not in guest_residence_stats:
            guest_residence_stats['n_entries'] = len(entry_frames)
        if 'n_exits' not in guest_residence_stats:
            guest_residence_stats['n_exits'] = len(exit_frames)
        if 'first_entry_frame' not in guest_residence_stats and entry_frames:
            guest_residence_stats['first_entry_frame'] = min(entry_frames)

        guest_residence_stats['entry_frames'] = entry_frames
        guest_residence_stats['exit_frames'] = exit_frames
        guest_residence_stats['entry_times'] = entry_times
        guest_residence_stats['exit_times'] = exit_times
        guest_residence_stats['entry_guest_indices'] = entry_guest_indices
        guest_residence_stats['exit_guest_indices'] = exit_guest_indices

        if not entry_frames and not exit_frames:
            warnings.warn("No entry or exit events found. Skipping GIF.")
            return

        # Collect all unique guest indices
        all_guest_indices = set()
        for indices in entry_guest_indices:
            all_guest_indices.update(indices)
        for indices in exit_guest_indices:
            all_guest_indices.update(indices)
        
        if not all_guest_indices:
            warnings.warn("No guest indices found in events. Skipping GIF.")
            return

        # Sort guest indices for consistent Y-axis ordering
        sorted_guest_indices = sorted(all_guest_indices)
        guest_index_to_y = {gidx: i for i, gidx in enumerate(sorted_guest_indices)}

        # Determine frame range
        all_frames = entry_frames + exit_frames
        if all_frames:
            min_frame = min(all_frames)
            max_frame = max(all_frames)
            if max_frames is not None:
                max_frame = max(max_frame, max_frames)
            # Add some padding
            frame_range = max_frame - min_frame
            x_min = max(0, min_frame - frame_range * 0.05)
            x_max = max_frame + frame_range * 0.05
        else:
            x_min, x_max = 0, 100

        # Create frames for animation
        animation_frames = list(range(int(x_min), int(x_max) + 1, frame_step))
        if not animation_frames:
            animation_frames = [int(x_min), int(x_max)]

        print(f"Creating animated GIF with {len(animation_frames)} frames...")
        
        # Determine number of workers
        if n_jobs is None:
            if PARALLEL_AVAILABLE and cpu_count is not None:
                n_jobs = cpu_count()
            else:
                n_jobs = 1
        elif n_jobs == -1:
            if PARALLEL_AVAILABLE and cpu_count is not None:
                n_jobs = cpu_count()
            else:
                n_jobs = 1
        
        # Use parallel processing if n_jobs > 1 and parallel is available
        if n_jobs > 1 and PARALLEL_AVAILABLE and ProcessPoolExecutor is not None:
            print(f"Using {n_jobs} parallel workers for frame generation...")
            images_dict = {}  # Store images by frame index to maintain order
            
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all frame generation tasks
                future_to_frame = {
                    executor.submit(
                        _generate_timeline_frame,
                        current_frame,
                        entry_frames,
                        exit_frames,
                        entry_guest_indices,
                        exit_guest_indices,
                        sorted_guest_indices,
                        guest_index_to_y,
                        guest_residence_stats,
                        x_min,
                        x_max,
                        self.figure_size,
                        self.dpi,
                        show_connections,
                    ): current_frame
                    for current_frame in animation_frames
                }
                
                # Collect results as they complete
                completed = 0
                for future in as_completed(future_to_frame):
                    current_frame = future_to_frame[future]
                    try:
                        img = future.result()
                        images_dict[current_frame] = img
                        completed += 1
                        if completed % 10 == 0 or completed == len(animation_frames):
                            print(f"  Progress: {completed}/{len(animation_frames)} frames completed")
                    except Exception as e:
                        warnings.warn(f"Failed to generate frame {current_frame}: {e}")
            
            # Sort images by frame index to maintain order
            images = [images_dict[frame] for frame in animation_frames if frame in images_dict]
        else:
            # Sequential processing
            if n_jobs > 1:
                warnings.warn("Parallel processing requested but not available. Using sequential processing.")
            images = []
            for i, current_frame in enumerate(animation_frames):
                img = _generate_timeline_frame(
                    current_frame,
                    entry_frames,
                    exit_frames,
                    entry_guest_indices,
                    exit_guest_indices,
                    sorted_guest_indices,
                    guest_index_to_y,
                    guest_residence_stats,
                    x_min,
                    x_max,
                    self.figure_size,
                    self.dpi,
                    show_connections,
                )
                images.append(img)
                if (i + 1) % 10 == 0 or (i + 1) == len(animation_frames):
                    print(f"  Progress: {i + 1}/{len(animation_frames)} frames completed")

        if not images:
            warnings.warn("No frames collected for GIF.")
            return

        # Save GIF
        gif_path = f"{out_prefix}_guest_entry_exit_timeline.gif"
        imageio.mimsave(gif_path, images, duration=1.0 / fps)
        self.output_prefix = out_prefix
        self.plots_generated.append(gif_path)
        print(f"Guest entry/exit timeline GIF saved to {gif_path}")


