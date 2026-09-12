"""Murata-style nanocube RMSD distributions (non-encapsulated conformations).

Murata 2026: RMSD of the cube vs the initial NVT structure, restricted to
frames that do **not** contain an encapsulated iodide. Peaks near **1.0, 1.5,
and 2.7 Å** are the closed / one-open / two-open cation–π motifs.

Stored ``gsa_features_step1`` already has ``assembly_rmsd_to_ref`` (Kabsch
alignment of ``resname MOL`` to each replica's frame 0) and
``n_guest_inside_cavity``. This module filters and plots those; it does not
re-read trajectories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .gsa_cohort_run import KNOWN_CUBES, discover_gsa_feature_csvs_by_cube

MURATA_RMSD_PEAKS_A = (1.0, 1.5, 2.7)
MURATA_C1_RMSD_WINDOW_A = (1.6, 2.5)
OPEN_COUNT_LABELS = {
    0: "0 open (closed, ~1.0 Å)",
    1: "1 open (~1.5 Å)",
    2: "≥2 open (~2.7 Å)",
}

CUBE_COLORS: dict[str, str] = {
    "BHHpH": "#1f77b4",
    "BHHpM": "#5fa8d3",
    "BMHpH": "#2ca02c",
    "BMHpM": "#98df8a",
    "BMMpH": "#ff7f0e",
    "BMMpM": "#d62728",
}

_USECOLS = ("traj_id", "frame", "time_ps", "assembly_rmsd_to_ref", "n_guest_inside_cavity")


def apo_frame_mask(
    df: pd.DataFrame,
    *,
    guest_col: str = "n_guest_inside_cavity",
) -> pd.Series:
    """True where no iodide is inside the cavity (Murata's apo conformations)."""
    if guest_col not in df.columns:
        raise KeyError(guest_col)
    guest = pd.to_numeric(df[guest_col], errors="coerce").fillna(0.0)
    return guest < 1.0


def filter_non_encapsulated(
    df: pd.DataFrame,
    *,
    guest_col: str = "n_guest_inside_cavity",
    traj_filter: str = "frames",
) -> pd.DataFrame:
    """Keep nanocube frames (or whole replicas) without encapsulated iodide.

    ``traj_filter``:
    * ``frames`` — drop only frames with a guest inside (Murata RMSD caption).
    * ``never`` — keep only trajectories that never encapsulate (Table 1 style).
    * ``all`` — keep every frame (ignore iodide occupancy).
    """
    if traj_filter not in {"frames", "never", "all"}:
        raise ValueError("traj_filter must be 'frames', 'never', or 'all'")
    if traj_filter == "all":
        return df.copy()
    if "traj_id" not in df.columns:
        mask = apo_frame_mask(df, guest_col=guest_col)
        return df.loc[mask].copy()
    if traj_filter == "never":
        guest = pd.to_numeric(df[guest_col], errors="coerce").fillna(0.0)
        ever = df.assign(_g=guest).groupby(df["traj_id"].astype(str))["_g"].max()
        keep = set(ever.index[ever < 1.0])
        return df.loc[df["traj_id"].astype(str).isin(keep)].copy()
    return df.loc[apo_frame_mask(df, guest_col=guest_col)].copy()


def drop_reference_rmsd(
    df: pd.DataFrame,
    *,
    rmsd_col: str = "assembly_rmsd_to_ref",
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Drop the initial NVT frame (RMSD identically 0 vs itself)."""
    rmsd = pd.to_numeric(df[rmsd_col], errors="coerce")
    return df.loc[rmsd > float(eps)].copy()


def find_rmsd_histogram_peaks(
    values: np.ndarray | pd.Series,
    *,
    bin_width: float = 0.05,
    smooth_sigma: float = 2.0,
    min_prominence_frac: float = 0.05,
    expected: Sequence[float] = MURATA_RMSD_PEAKS_A,
    match_tol: float = 0.4,
) -> pd.DataFrame:
    """Local maxima of a smoothed RMSD histogram, matched to Murata's 1.0/1.5/2.7 Å."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    vals = np.asarray(pd.to_numeric(values, errors="coerce"), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 10:
        return pd.DataFrame(
            columns=["peak_A", "density", "nearest_murata_A", "delta_A", "matched"]
        )
    hi = max(float(np.nanmax(vals)), 4.0) + bin_width
    edges = np.arange(0.0, hi + bin_width, bin_width)
    counts, edges = np.histogram(vals, bins=edges, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    smooth = gaussian_filter1d(counts.astype(float), sigma=float(smooth_sigma))
    prom = max(float(np.nanmax(smooth)) * float(min_prominence_frac), 1e-6)
    idx, _ = find_peaks(smooth, prominence=prom)
    rows = []
    expected = tuple(float(x) for x in expected)
    for i in idx:
        peak = float(centers[i])
        nearest = min(expected, key=lambda x: abs(x - peak)) if expected else np.nan
        delta = float(peak - nearest) if expected else np.nan
        rows.append(
            {
                "peak_A": peak,
                "density": float(smooth[i]),
                "nearest_murata_A": nearest,
                "delta_A": delta,
                "matched": bool(abs(delta) <= match_tol) if expected else False,
            }
        )
    return pd.DataFrame(rows)


def summarize_apo_rmsd(
    df: pd.DataFrame,
    *,
    cohort: str,
    rmsd_col: str = "assembly_rmsd_to_ref",
    guest_col: str = "n_guest_inside_cavity",
    traj_filter: str = "frames",
) -> dict[str, object]:
    """One-row occupancy + peak summary for a cohort."""
    n_all = int(len(df))
    n_encap = int((~apo_frame_mask(df, guest_col=guest_col)).sum())
    apo = filter_non_encapsulated(df, guest_col=guest_col, traj_filter=traj_filter)
    apo = drop_reference_rmsd(apo, rmsd_col=rmsd_col)
    rmsd = pd.to_numeric(apo[rmsd_col], errors="coerce") if rmsd_col in apo.columns else pd.Series(dtype=float)
    rmsd = rmsd[np.isfinite(rmsd)]
    peaks = find_rmsd_histogram_peaks(rmsd)
    matched = peaks[peaks["matched"]] if not peaks.empty else peaks
    summary: dict[str, object] = {
        "guest_filter": traj_filter,
        "cohort": cohort,
        "rmsd_col": rmsd_col,
        "n_frames": n_all,
        "n_apo_frames": int(len(apo)),
        "n_encapsulated_frames": n_encap,
        "frac_apo": float(len(apo) / n_all) if n_all else np.nan,
        "n_trajectories": int(df["traj_id"].nunique()) if "traj_id" in df.columns else 1,
        "n_apo_trajectories": (
            int(apo["traj_id"].nunique()) if "traj_id" in apo.columns and len(apo) else 0
        ),
        "rmsd_median_A": float(np.median(rmsd)) if len(rmsd) else np.nan,
        "rmsd_p05_A": float(np.percentile(rmsd, 5)) if len(rmsd) else np.nan,
        "rmsd_p95_A": float(np.percentile(rmsd, 95)) if len(rmsd) else np.nan,
        "n_histogram_peaks": int(len(peaks)),
        "peak_positions_A": (
            ";".join(f"{p:.2f}" for p in peaks["peak_A"]) if len(peaks) else ""
        ),
    }
    for target in MURATA_RMSD_PEAKS_A:
        hit = matched.loc[matched["nearest_murata_A"] == target] if len(matched) else pd.DataFrame()
        if len(hit):
            best = hit.iloc[(hit["delta_A"].abs()).argmin()]
            summary[f"peak_near_{target:.1f}"] = float(best["peak_A"])
            summary[f"delta_near_{target:.1f}"] = float(best["delta_A"])
        else:
            summary[f"peak_near_{target:.1f}"] = np.nan
            summary[f"delta_near_{target:.1f}"] = np.nan
    return summary


def load_gsa_rmsd_frames(
    features_root: Path | str,
    *,
    cubes: Optional[Sequence[str]] = None,
) -> dict[str, pd.DataFrame]:
    """Load ``assembly_rmsd_to_ref`` + guest occupancy from GSA feature CSVs."""
    wanted = tuple(cubes) if cubes else KNOWN_CUBES
    by_cube = discover_gsa_feature_csvs_by_cube(features_root, cubes=wanted)
    out: dict[str, pd.DataFrame] = {}
    for cube, paths in by_cube.items():
        frames: list[pd.DataFrame] = []
        for path in paths:
            header = pd.read_csv(path, nrows=0)
            cols = [c for c in _USECOLS if c in header.columns]
            if "assembly_rmsd_to_ref" not in cols or "n_guest_inside_cavity" not in cols:
                continue
            df = pd.read_csv(path, usecols=cols)
            if "traj_id" not in df.columns:
                stem = path.name.replace("_gsa_features.csv", "")
                df.insert(0, "traj_id", stem)
            frames.append(df)
        if frames:
            stacked = pd.concat(frames, ignore_index=True)
            stacked.insert(0, "cohort", cube)
            out[cube] = stacked
    return out


def _norm_traj_id(traj_id: str, cube: str) -> str:
    text = str(traj_id)
    prefix = f"{cube}_"
    return text[len(prefix) :] if text.startswith(prefix) else text


def attach_open_cation_pi(
    rmsd_df: pd.DataFrame,
    open_df: pd.DataFrame,
    *,
    cube: str,
) -> pd.DataFrame:
    """Left-join ``n_open_cation_pi`` onto RMSD frames (traj_id + frame)."""
    if "n_open_cation_pi" not in open_df.columns:
        raise KeyError("n_open_cation_pi")
    left = rmsd_df.copy()
    right = open_df.copy()
    left["_tid"] = left["traj_id"].map(lambda t: _norm_traj_id(t, cube))
    right["_tid"] = right["traj_id"].map(lambda t: _norm_traj_id(t, cube))
    keep = ["_tid", "frame", "n_open_cation_pi"]
    extra = [c for c in ("murata_metastructure",) if c in right.columns]
    merged = left.merge(right[keep + extra], on=["_tid", "frame"], how="left")
    return merged.drop(columns=["_tid"])


def open_count_bin(n_open: pd.Series) -> pd.Series:
    """0 / 1 / ≥2 opened cation–π units (Murata's three RMSD motifs)."""
    n = pd.to_numeric(n_open, errors="coerce")
    out = pd.Series(np.full(len(n), np.nan), index=n_open.index)
    out.loc[n == 0] = 0
    out.loc[n == 1] = 1
    out.loc[n >= 2] = 2
    return out


def _filter_title(traj_filter: str) -> str:
    if traj_filter == "all":
        return "All frames (iodide not filtered)"
    if traj_filter == "never":
        return "Replicas that never encapsulate"
    return "Non-encapsulated frames"


def plot_murata_rmsd_distributions(
    frames_by_cohort: dict[str, pd.DataFrame],
    output_path: Path | str,
    *,
    traj_filter: str = "frames",
    rmsd_col: str = "assembly_rmsd_to_ref",
    xmax: float = 6.0,
    xlabel: str = "assembly RMSD to initial NVT (Å)",
) -> Path:
    """Per-cube apo RMSD histograms with Murata 1.0 / 1.5 / 2.7 Å markers."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohorts = [c for c in KNOWN_CUBES if c in frames_by_cohort] or list(frames_by_cohort)
    n = len(cohorts)
    ncols = 3
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)
    bins = np.arange(0.0, xmax + 0.05, 0.05)
    for i, cube in enumerate(cohorts):
        ax = axes[i // ncols][i % ncols]
        apo = filter_non_encapsulated(frames_by_cohort[cube], traj_filter=traj_filter)
        apo = drop_reference_rmsd(apo, rmsd_col=rmsd_col)
        vals = pd.to_numeric(apo[rmsd_col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals >= 0) & (vals <= xmax)]
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=CUBE_COLORS.get(cube, "0.4"),
            alpha=0.45,
            lw=0,
        )
        ax.hist(vals, bins=bins, density=True, histtype="step", color=CUBE_COLORS.get(cube, "0.2"), lw=1.2)
        for x in MURATA_RMSD_PEAKS_A:
            ax.axvline(x, color="0.25", ls="--", lw=0.9, zorder=3)
        ax.axvspan(*MURATA_C1_RMSD_WINDOW_A, color="0.85", zorder=0, lw=0)
        ax.set_xlim(0, xmax)
        ax.set_title(f"{cube}  n={len(vals):,}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(
        f"{_filter_title(traj_filter)}  (dashed: Murata 1.0 / 1.5 / 2.7 Å; "
        "shade: C1 1.6–2.5 Å)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_murata_rmsd_overlay(
    frames_by_cohort: dict[str, pd.DataFrame],
    output_path: Path | str,
    *,
    traj_filter: str = "frames",
    rmsd_col: str = "assembly_rmsd_to_ref",
    xmax: float = 6.0,
    xlabel: str = "assembly RMSD to initial NVT (Å)",
) -> Path:
    """Single-axis overlay of apo RMSD densities across cubes."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    bins = np.arange(0.0, xmax + 0.05, 0.05)
    for cube, df in frames_by_cohort.items():
        apo = filter_non_encapsulated(df, traj_filter=traj_filter)
        apo = drop_reference_rmsd(apo, rmsd_col=rmsd_col)
        vals = pd.to_numeric(apo[rmsd_col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals >= 0) & (vals <= xmax)]
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype="step",
            label=cube,
            color=CUBE_COLORS.get(cube, None),
            lw=1.5,
        )
    for x, lab in zip(MURATA_RMSD_PEAKS_A, ("closed ~1.0 Å", "1 open ~1.5 Å", "2 open ~2.7 Å")):
        ax.axvline(x, color="0.35", ls="--", lw=0.9, label=lab)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(f"{_filter_title(traj_filter)} vs Murata peaks")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_rmsd_by_open_count(
    frames_by_cohort: dict[str, pd.DataFrame],
    output_path: Path | str,
    *,
    traj_filter: str = "frames",
    rmsd_col: str = "assembly_rmsd_to_ref",
    xmax: float = 6.0,
) -> Path:
    """Apo RMSD split by opened cation–π count (0 / 1 / ≥2)."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#d62728"}
    bins = np.arange(0.0, xmax + 0.05, 0.05)
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    stacked = []
    for df in frames_by_cohort.values():
        if "n_open_cation_pi" not in df.columns:
            continue
        stacked.append(
            drop_reference_rmsd(
                filter_non_encapsulated(df, traj_filter=traj_filter),
                rmsd_col=rmsd_col,
            )
        )
    if not stacked:
        raise ValueError("No n_open_cation_pi column to split RMSD by cation–π count")
    all_apo = pd.concat(stacked, ignore_index=True)
    bins_open = open_count_bin(all_apo["n_open_cation_pi"])
    for k, label in OPEN_COUNT_LABELS.items():
        vals = pd.to_numeric(all_apo.loc[bins_open == k, rmsd_col], errors="coerce")
        vals = vals[np.isfinite(vals) & (vals >= 0) & (vals <= xmax)]
        if len(vals) == 0:
            continue
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype="step",
            label=f"{label}  n={len(vals):,}",
            color=colors[k],
            lw=1.6,
        )
    for x in MURATA_RMSD_PEAKS_A:
        ax.axvline(x, color="0.3", ls="--", lw=0.9)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("assembly RMSD to initial NVT (Å)")
    ax.set_ylabel("density")
    ax.set_title(f"{_filter_title(traj_filter)} by opened cation–π units")
    ax.legend(fontsize=8)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_rmsd_apo_vs_all(
    frames_by_cohort: dict[str, pd.DataFrame],
    output_path: Path | str,
    *,
    rmsd_col: str = "assembly_rmsd_to_ref",
    xmax: float = 6.0,
) -> Path:
    """Per-cube RMSD: all frames (gray) vs non-encapsulated frames (color)."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohorts = [c for c in KNOWN_CUBES if c in frames_by_cohort] or list(frames_by_cohort)
    n = len(cohorts)
    ncols = 3
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)
    bins = np.arange(0.0, xmax + 0.05, 0.05)
    for i, cube in enumerate(cohorts):
        ax = axes[i // ncols][i % ncols]
        raw = drop_reference_rmsd(frames_by_cohort[cube], rmsd_col=rmsd_col)
        all_vals = pd.to_numeric(raw[rmsd_col], errors="coerce").to_numpy(dtype=float)
        all_vals = all_vals[np.isfinite(all_vals) & (all_vals >= 0) & (all_vals <= xmax)]
        apo = drop_reference_rmsd(
            filter_non_encapsulated(frames_by_cohort[cube], traj_filter="frames"),
            rmsd_col=rmsd_col,
        )
        apo_vals = pd.to_numeric(apo[rmsd_col], errors="coerce").to_numpy(dtype=float)
        apo_vals = apo_vals[np.isfinite(apo_vals) & (apo_vals >= 0) & (apo_vals <= xmax)]
        ax.hist(
            all_vals,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color="0.75",
            alpha=0.9,
            lw=0,
            label=f"all n={len(all_vals):,}",
        )
        ax.hist(
            apo_vals,
            bins=bins,
            density=True,
            histtype="step",
            color=CUBE_COLORS.get(cube, "0.2"),
            lw=1.4,
            label=f"no I⁻ n={len(apo_vals):,}",
        )
        for x in MURATA_RMSD_PEAKS_A:
            ax.axvline(x, color="0.25", ls="--", lw=0.8, zorder=3)
        ax.set_xlim(0, xmax)
        ax.set_title(cube)
        ax.set_xlabel("assembly RMSD to initial NVT (Å)")
        ax.set_ylabel("density")
        ax.legend(fontsize=7, loc="upper right")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(
        "All frames (gray) vs non-encapsulated (color). Dashed: Murata 1.0 / 1.5 / 2.7 Å",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


MOTIF_XLABELS = {
    "rmsd_cation_pi": "cation–π (Py+, Ph, Py+) RMSD to initial NVT (Å)",
    "rmsd_equator": "equator (Py+, R2, R3, Py+) RMSD to initial NVT (Å)",
    "assembly_rmsd_to_ref": "whole-cube RMSD to initial NVT (Å)",
}


def compute_trajectory_motif_rmsd(
    topology: Path | str,
    trajectory: Path | str,
    *,
    traj_id: str,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: int = 1,
    ref_frame: int = 0,
    n_monomers: int = 6,
    traj_format: Optional[str] = None,
) -> pd.DataFrame:
    """Kabsch RMSD of Murata cation–π and equatorial motifs vs ``ref_frame``.

    Reads only the motif / MOL AtomGroups each frame (not the full solvent box).
    """
    from src.ChangepointAnalysis.murata_criteria import (
        MURATA_RMSD_MOTIFS,
        motif_atom_indices,
        resolve_murata_criteria_atoms,
    )
    from src.ChangepointAnalysis.pipeline import _load_universe
    from src.TrajectoryMetrics.TrajectoryMetrics import rmsd_value_aligned
    from src.utils.gsa_selections import resolve_selections

    u = _load_universe(topology, trajectory, traj_format=traj_format)
    n_traj = len(u.trajectory)
    frame_indices = list(range(*slice(start, stop, step).indices(n_traj)))
    if not frame_indices:
        raise ValueError(f"No frames in {trajectory}")
    sels = resolve_selections(
        u, gsa_resname="MOL", n_monomers=int(n_monomers), auto_tooth=False
    )
    roles = resolve_murata_criteria_atoms(u, sels.monomer_selections)
    u.trajectory[int(ref_frame)]
    groups: dict[str, Any] = {"assembly_rmsd_to_ref": u.select_atoms("resname MOL")}
    for name, role_names in MURATA_RMSD_MOTIFS.items():
        idx = motif_atom_indices(roles, role_names)
        groups[f"rmsd_{name}"] = u.atoms[idx]
    refs = {col: ag.positions.copy() for col, ag in groups.items()}
    rows: list[dict[str, object]] = []
    for ts in u.trajectory[start:stop:step]:
        row: dict[str, object] = {
            "traj_id": traj_id,
            "frame": int(ts.frame),
            "time_ps": float(getattr(ts, "time", ts.frame) or ts.frame),
        }
        for col, ag in groups.items():
            row[col] = rmsd_value_aligned(ag.positions, refs[col])
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["motif_atom_counts"] = {
        col: int(len(ag)) for col, ag in groups.items()
    }
    return out


def attach_guest_occupancy(
    rmsd_df: pd.DataFrame,
    gsa_csv: Path | str,
) -> pd.DataFrame:
    """Join ``n_guest_inside_cavity`` from a stored GSA features CSV."""
    gsa = pd.read_csv(
        gsa_csv, usecols=lambda c: c in {"frame", "n_guest_inside_cavity"}
    )
    if "frame" not in gsa.columns or "n_guest_inside_cavity" not in gsa.columns:
        raise KeyError(str(gsa_csv))
    return rmsd_df.merge(gsa.drop_duplicates("frame"), on="frame", how="left")


def write_trajectory_motif_rmsd(
    topology: str,
    trajectory: str,
    traj_id: str,
    cube: str,
    out_csv: str,
    gsa_csv: Optional[str] = None,
    start: Optional[int] = None,
    stop: Optional[int] = None,
    step: int = 1,
    ref_frame: int = 0,
    n_monomers: int = 6,
    traj_format: Optional[str] = None,
) -> dict[str, object]:
    """Worker: motif RMSD for one replica, write CSV, return the path.

    Importable top-level so ``ProcessPoolExecutor`` can pickle it on spawn.
    """
    df = compute_trajectory_motif_rmsd(
        topology,
        trajectory,
        traj_id=traj_id,
        start=start,
        stop=stop,
        step=step,
        ref_frame=ref_frame,
        n_monomers=n_monomers,
        traj_format=traj_format,
    )
    df.insert(0, "cohort", cube)
    if gsa_csv:
        df = attach_guest_occupancy(df, gsa_csv)
    elif "n_guest_inside_cavity" not in df.columns:
        df["n_guest_inside_cavity"] = 0.0
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return {"cube": cube, "traj_id": traj_id, "path": str(path), "n": int(len(df))}
