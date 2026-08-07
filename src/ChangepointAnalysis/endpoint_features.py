"""Endpoint-site feature extraction for metastructure changepoint analysis.

Computes per-frame distances between chemically meaningful endpoint *sites*
(fused ring-system centroids and singleton atom sites) via a single
``TrajectoryIterator`` pass, then flattens them into feature CSVs suitable for
``ChangepointPipeline(groups=('endpoint',))``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class EndpointFeatureConfig:
    """Configuration for endpoint-site feature extraction."""

    gsa_resname: str = "MOL"
    n_monomers: int = 6
    monomer_selections: Optional[list[str]] = None
    use_ring_centroids: bool = True
    ring_min_gap_deg: Optional[float] = 35.0
    ring_max_per_ring: int = 3
    step_back_from_terminals: bool = True
    start: Optional[int] = None
    stop: Optional[int] = None
    step: int = 1
    time_per_frame_ps: float = 1.0
    pair_summaries: tuple[str, ...] = ("min", "mean", "max")
    include_site_pairs: bool = False


def all_pairs_to_metrics_df(
    all_pairs: dict,
    n_res: int,
    n_frames: int,
    *,
    summaries: Sequence[str] = ("min", "mean", "max"),
    include_site_pairs: bool = False,
) -> pd.DataFrame:
    """Flatten ``all_pairs[(i, j)]`` arrays into a per-frame metrics DataFrame.

    Columns
    -------
    ``endpoint_dist_{i}_{j}_{min,mean,max}``
        Aggregate over all site-pairs between monomers *i* and *j*.
    ``endpoint_dist_{mean,min,max,std}``
        Assembly-wide aggregates over every site-pair distance.
    ``endpoint_dist_{i}s{a}_{j}s{b}`` (optional)
        Raw distance between site *a* of monomer *i* and site *b* of monomer *j*.
    """
    summary_set = set(summaries)
    rows: list[dict[str, Any]] = []
    for frame in range(n_frames):
        row: dict[str, Any] = {"frame": frame}
        all_pair_dists: list[float] = []
        for i in range(n_res):
            for j in range(i + 1, n_res):
                if (i, j) not in all_pairs:
                    for s in summaries:
                        row[f"endpoint_dist_{i}_{j}_{s}"] = np.nan
                    continue
                pair_array = all_pairs[(i, j)][frame]
                valid = pair_array[~np.isnan(pair_array)]
                all_pair_dists.extend(valid.tolist())
                if len(valid) > 0:
                    if "mean" in summary_set:
                        row[f"endpoint_dist_{i}_{j}_mean"] = float(np.mean(valid))
                    if "min" in summary_set:
                        row[f"endpoint_dist_{i}_{j}_min"] = float(np.min(valid))
                    if "max" in summary_set:
                        row[f"endpoint_dist_{i}_{j}_max"] = float(np.max(valid))
                    if "std" in summary_set:
                        row[f"endpoint_dist_{i}_{j}_std"] = float(np.std(valid))
                else:
                    for s in summaries:
                        row[f"endpoint_dist_{i}_{j}_{s}"] = np.nan

                if include_site_pairs and pair_array.ndim == 2:
                    n_a, n_b = pair_array.shape
                    for a in range(n_a):
                        for b in range(n_b):
                            val = pair_array[a, b]
                            row[f"endpoint_dist_{i}s{a}_{j}s{b}"] = (
                                float(val) if np.isfinite(val) else np.nan
                            )

        if all_pair_dists:
            arr = np.asarray(all_pair_dists, dtype=float)
            row["endpoint_dist_mean"] = float(np.mean(arr))
            row["endpoint_dist_min"] = float(np.min(arr))
            row["endpoint_dist_max"] = float(np.max(arr))
            row["endpoint_dist_std"] = float(np.std(arr))
        else:
            row["endpoint_dist_mean"] = np.nan
            row["endpoint_dist_min"] = np.nan
            row["endpoint_dist_max"] = np.nan
            row["endpoint_dist_std"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _count_iterated_frames(
    n_traj_frames: int,
    start: Optional[int],
    stop: Optional[int],
    step: int,
) -> int:
    s = slice(start, stop, step)
    return len(range(*s.indices(n_traj_frames)))


def generate_endpoint_features(
    universe: Any,
    config: Optional[EndpointFeatureConfig] = None,
    *,
    traj_id: str = "traj",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[list[list[int]]]]:
    """Extract per-frame endpoint-site distances from *universe*.

    Returns
    -------
    features_df
        Per-frame distance features with ``traj_id``, ``frame``, ``time_ps``.
    sites_df
        Site map (one row per monomer site) for traceability.
    monomer_selections
        Resolved per-monomer MDAnalysis selection strings.
    stored_sites
        Per monomer → per site → MDA atom ids (for QC plots).
    """
    from src.EndpointAnalyzer import EndpointAnalyzerObserver, EndpointsFinder
    from src.TrajectoryIterator import TrajectoryIterator
    from src.utils.gsa_selections import GSAFeatureSelections, resolve_selections

    cfg = config or EndpointFeatureConfig()

    explicit = None
    if cfg.monomer_selections:
        explicit = GSAFeatureSelections(monomer_selections=list(cfg.monomer_selections))
    sels = resolve_selections(
        universe,
        explicit=explicit,
        gsa_resname=cfg.gsa_resname,
        n_monomers=cfg.n_monomers,
        auto_tooth=False,
    )
    monomer_sels = sels.monomer_selections
    if not monomer_sels or len(monomer_sels) < 2:
        raise ValueError(
            "Need at least two monomer selections for endpoint-site distances."
        )

    finder = EndpointsFinder(
        ring_min_gap_deg=cfg.ring_min_gap_deg,
        ring_max_per_ring=cfg.ring_max_per_ring,
        step_back_from_terminals=cfg.step_back_from_terminals,
        extend_to_ring_atoms=True,
    )

    n_traj = len(universe.trajectory)
    n_rows = _count_iterated_frames(n_traj, cfg.start, cfg.stop, cfg.step)

    observer = EndpointAnalyzerObserver(
        residue_sel_list=list(monomer_sels),
        endpoints_finder=finder,
        n_frame_rows=n_rows,
        use_ring_centroids=cfg.use_ring_centroids,
    )
    iterator = TrajectoryIterator(universe)
    iterator.subscribe(observer)
    iterator.iterate(start=cfg.start, stop=cfg.stop, step=cfg.step)

    dist_info = observer.get_endpoint_distances()
    features = all_pairs_to_metrics_df(
        dist_info["all_pairs"],
        dist_info["n_residues"],
        dist_info["n_frames"],
        summaries=cfg.pair_summaries,
        include_site_pairs=cfg.include_site_pairs,
    )

    # Map iterated frame indices to trajectory frame numbers / time
    frame_indices = list(range(*slice(cfg.start, cfg.stop, cfg.step).indices(n_traj)))
    if len(frame_indices) != len(features):
        # Fall back to sequential numbering if lengths diverge
        frame_indices = list(range(len(features)))
    features.insert(0, "traj_id", traj_id)
    features["frame"] = frame_indices
    features["time_ps"] = np.asarray(frame_indices, dtype=float) * float(cfg.time_per_frame_ps)
    # Keep column order: traj_id, frame, time_ps, then the rest
    cols = ["traj_id", "frame", "time_ps"] + [
        c for c in features.columns if c not in ("traj_id", "frame", "time_ps")
    ]
    features = features[cols]

    sites_df = _build_sites_dataframe(
        universe,
        monomer_sels,
        finder,
        traj_id=traj_id,
        use_ring_centroids=cfg.use_ring_centroids,
        stored_sites=observer.stored_site_indices,
    )
    return features, sites_df, list(monomer_sels), list(observer.stored_site_indices)


def _build_sites_dataframe(
    universe: Any,
    monomer_sels: list[str],
    finder: Any,
    *,
    traj_id: str,
    use_ring_centroids: bool,
    stored_sites: list[list[list[int]]],
) -> pd.DataFrame:
    """Build a site-map table from stored MDA atom-id groups."""
    rows: list[dict[str, Any]] = []
    for mon_idx, sites in enumerate(stored_sites):
        for site_idx, atom_ids in enumerate(sites):
            kind = "ring" if (use_ring_centroids and len(atom_ids) > 1) else "atom"
            rows.append(
                {
                    "traj_id": traj_id,
                    "monomer": mon_idx,
                    "monomer_selection": monomer_sels[mon_idx] if mon_idx < len(monomer_sels) else "",
                    "site_index": site_idx,
                    "kind": kind,
                    "n_atoms": len(atom_ids),
                    "atom_ids": " ".join(str(a) for a in atom_ids),
                }
            )
    return pd.DataFrame(rows)


def _draw_endpoint_site_panel(
    ax: Any,
    universe: Any,
    mon_sel: str,
    sites: Sequence[Sequence[int]],
    *,
    panel_title: str,
) -> None:
    """Render one monomer 2D structure with ring (blue) and atom (orange) sites."""
    from rdkit.Chem import Draw

    from src.EndpointAnalyzer import EndpointsFinder

    sel = universe.select_atoms(mon_sel)
    mol = sel.convert_to("RDKIT")
    if mol is None:
        ax.text(0.5, 0.5, f"RDKit failed\n{mon_sel}", ha="center", va="center")
        ax.axis("off")
        return

    ef = EndpointsFinder()
    m2d, xy = ef.to_2d_coords(mol)
    mda_to_rdkit = {int(sel.atoms[i].id): i for i in range(len(sel))}

    # Ring sites = multi-atom; atom sites = singletons
    ring_mda: list[int] = []
    atom_mda: list[int] = []
    site_labels: list[tuple[int, str, str]] = []  # rdkit_idx, label, color
    for site_idx, atom_ids in enumerate(sites):
        ids = [int(a) for a in atom_ids]
        is_ring = len(ids) > 1
        color = "steelblue" if is_ring else "darkorange"
        kind = "R" if is_ring else "A"
        if is_ring:
            ring_mda.extend(ids)
        else:
            atom_mda.extend(ids)
        # Label at the site representative (min id → centroid proxy atom)
        rep = min(ids) if ids else None
        if rep is not None and rep in mda_to_rdkit:
            site_labels.append((mda_to_rdkit[rep], f"s{site_idx}:{kind}", color))

    ring_rdkit = [mda_to_rdkit[i] for i in ring_mda if i in mda_to_rdkit]
    atom_rdkit = [mda_to_rdkit[i] for i in atom_mda if i in mda_to_rdkit]

    highlight_colors: dict[int, tuple[float, float, float]] = {}
    for idx in ring_rdkit:
        highlight_colors[idx] = (0.2, 0.45, 0.85)  # blue — ring system
    for idx in atom_rdkit:
        if idx not in highlight_colors:
            highlight_colors[idx] = (1.0, 0.5, 0.0)  # orange — atom site
    all_highlight = list(highlight_colors.keys())

    img = None
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        import io

        from PIL import Image

        drawer = rdMolDraw2D.MolDraw2DCairo(800, 600)
        drawer.DrawMolecule(
            m2d,
            highlightAtoms=all_highlight,
            highlightAtomColors=highlight_colors,
        )
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))
    except Exception:
        img = Draw.MolToImage(
            m2d,
            size=(800, 600),
            highlightAtoms=ring_rdkit or all_highlight,
            highlightColor=(0.2, 0.45, 0.85),
        )
        if atom_rdkit:
            from PIL import Image

            base = np.array(img.convert("RGBA"))
            overlay = np.array(
                Draw.MolToImage(
                    m2d,
                    size=(800, 600),
                    highlightAtoms=atom_rdkit,
                    highlightColor=(1.0, 0.5, 0.0),
                ).convert("RGBA")
            )
            mask = overlay[:, :, 3] > 0
            base[mask] = overlay[mask]
            img = Image.fromarray(base)

    ax.imshow(img, extent=[0, 800, 600, 0])
    ax.axis("off")
    n_ring = sum(1 for s in sites if len(s) > 1)
    n_atom = len(sites) - n_ring
    ax.set_title(
        f"{panel_title}  ({n_ring} ring, {n_atom} atom)",
        fontsize=11,
        fontweight="bold",
    )

    if len(xy) == 0:
        return
    x_coords = xy[:, 0]
    y_coords = xy[:, 1]
    x_min, x_max = float(x_coords.min()), float(x_coords.max())
    y_min, y_max = float(y_coords.min()), float(y_coords.max())
    x_range = x_max - x_min if x_max > x_min else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    padding = 0.15
    scale = min(
        (800 * (1 - 2 * padding)) / x_range if x_range > 0 else 1.0,
        (600 * (1 - 2 * padding)) / y_range if y_range > 0 else 1.0,
    )
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    offset_x = 400 - center_x * scale
    offset_y = 300 + center_y * scale

    for rdkit_idx, label, color in site_labels:
        if rdkit_idx >= len(xy):
            continue
        x, y = xy[rdkit_idx]
        ax.annotate(
            label,
            xy=(x * scale + offset_x, -y * scale + offset_y),
            fontsize=9,
            color=color,
            fontweight="bold",
        )


def plot_endpoint_sites(
    universe: Any,
    monomer_selections: Sequence[str],
    stored_sites: Sequence[Sequence[Sequence[int]]],
    output_path: Path | str,
    *,
    dpi: int = 150,
    title: Optional[str] = None,
) -> Path:
    """Save a multi-panel PNG of endpoint sites per monomer for visual QC.

    Blue highlights = fused ring-system sites (centroid distances).
    Orange highlights = singleton atom sites (e.g. gear-tooth / ipso carbons).
    Labels ``s{k}:R`` / ``s{k}:A`` mark site index and kind.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(monomer_selections)
    if n == 0:
        raise ValueError("No monomer selections provided for endpoint-site plot.")
    if len(stored_sites) != n:
        raise ValueError(
            f"stored_sites length ({len(stored_sites)}) != "
            f"monomer_selections length ({n})"
        )

    if title is None:
        title = (
            "Endpoint sites — blue=ring system (centroid), "
            "orange=atom site (tooth / tip)"
        )

    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n == 1:
        axes_flat = [axes]
    else:
        axes_flat = list(np.atleast_1d(axes).flatten())

    for idx, mon_sel in enumerate(monomer_selections):
        _draw_endpoint_site_panel(
            axes_flat[idx],
            universe,
            mon_sel,
            stored_sites[idx],
            panel_title=f"Monomer {idx}",
        )

    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    legend_handles = [
        Patch(
            facecolor=(0.2, 0.45, 0.85),
            edgecolor="navy",
            label="Ring-system site (centroid)",
        ),
        Patch(
            facecolor=(1.0, 0.5, 0.0),
            edgecolor="darkorange",
            label="Atom site (tooth / tip)",
        ),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_endpoint_features_csv(
    features_df: pd.DataFrame,
    out_dir: Path | str,
    traj_id: str,
    *,
    sites_df: Optional[pd.DataFrame] = None,
    universe: Any = None,
    monomer_selections: Optional[Sequence[str]] = None,
    stored_sites: Optional[Sequence[Sequence[Sequence[int]]]] = None,
    write_site_plot: bool = False,
) -> dict[str, Path]:
    """Write ``<traj_id>_endpoint_features.csv`` and optionally sites map + QC PNG.

    The site QC plot is topology-only (one ``endpoint_sites.png`` for the
    shared prmtop). Pass ``write_site_plot=True`` once; later calls skip if
    the file already exists.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_path = out_dir / f"{traj_id}_endpoint_features.csv"
    features_df.to_csv(feat_path, index=False)
    written: dict[str, Path] = {"features": feat_path}

    if sites_df is not None and not sites_df.empty:
        sites_path = out_dir / "endpoint_sites.csv"
        if sites_path.exists():
            existing = pd.read_csv(sites_path)
            # Drop prior rows for this traj_id then append
            existing = existing[existing["traj_id"] != traj_id]
            combined = pd.concat([existing, sites_df], ignore_index=True)
        else:
            combined = sites_df
        combined.to_csv(sites_path, index=False)
        written["sites"] = sites_path

    if (
        write_site_plot
        and universe is not None
        and monomer_selections is not None
        and stored_sites is not None
        and len(stored_sites) > 0
    ):
        cohort_plot = out_dir / "endpoint_sites.png"
        if not cohort_plot.exists():
            plot_endpoint_sites(
                universe,
                monomer_selections,
                stored_sites,
                cohort_plot,
            )
        written["sites_plot"] = cohort_plot

    return written
