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

_MOTIF_RMSD_USECOLS = (
    "cohort",
    "traj_id",
    "frame",
    "time_ps",
    "assembly_rmsd_to_ref",
    "rmsd_cation_pi",
    "rmsd_equator",
    "n_guest_inside_cavity",
)


def time_ns_from_frames(df: pd.DataFrame) -> np.ndarray:
    """Time in ns. Prefer ``time_ps`` (1 ps/frame in these runs); else ``frame``."""
    if "time_ps" in df.columns:
        t = pd.to_numeric(df["time_ps"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(t).any():
            return t / 1000.0
    if "frame" in df.columns:
        return pd.to_numeric(df["frame"], errors="coerce").to_numpy(dtype=float) / 1000.0
    raise KeyError("Need time_ps or frame to plot RMSD vs time")


def discover_motif_rmsd_csvs_by_cube(
    output_root: Path | str,
    *,
    cubes: Optional[Sequence[str]] = None,
) -> dict[str, list[Path]]:
    """Find ``*_motif_rmsd.csv`` under ``output/{CUBE}_motif_rmsd`` (and the mixed folder)."""
    output_root = Path(output_root)
    wanted = tuple(cubes) if cubes else KNOWN_CUBES
    by_cube: dict[str, list[Path]] = {}
    extra = output_root / "murata_motif_rmsd"
    for cube in wanted:
        hits: list[Path] = []
        named = output_root / f"{cube}_motif_rmsd"
        if named.is_dir():
            hits.extend(named.rglob("*_motif_rmsd.csv"))
        nested = extra / cube
        if nested.is_dir():
            hits.extend(nested.rglob("*_motif_rmsd.csv"))
        uniq = sorted({p.resolve() for p in hits if p.is_file()})
        if uniq:
            by_cube[cube] = uniq
    return by_cube


def load_motif_rmsd_csvs(paths: Sequence[Path | str]) -> pd.DataFrame:
    """Stack per-replica motif RMSD CSVs."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        header = pd.read_csv(path, nrows=0)
        cols = [
            c
            for c in header.columns
            if c in _MOTIF_RMSD_USECOLS
            or str(c).startswith(
                ("cation_pi_pole_e", "cation_pi_eq_e", "cation_pi_e", "cation_pi_angle_e", "equator_d1_e", "equator_d2_e", "equator_angle_e", "rmsd_cation_pi_e", "rmsd_equator_e")
            )
            or str(c) == "n_open_cation_pi"
        ]
        if "traj_id" not in cols:
            continue
        df = pd.read_csv(path, usecols=cols)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_rmsd_vs_time(
    df: pd.DataFrame,
    output_path: Path | str,
    *,
    rmsd_col: str = "rmsd_cation_pi",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    color: str = "0.35",
) -> Path:
    """Spaghetti of replicas plus mean RMSD vs time in ns."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rmsd_col not in df.columns:
        raise KeyError(rmsd_col)
    work = df.loc[:, [c for c in ("traj_id", "time_ps", "frame", rmsd_col) if c in df.columns]].copy()
    work["_t_ns"] = time_ns_from_frames(work)
    work["_y"] = pd.to_numeric(work[rmsd_col], errors="coerce")
    work = work[np.isfinite(work["_t_ns"]) & np.isfinite(work["_y"])]
    fig, ax = plt.subplots(figsize=(8.5, 3.6), constrained_layout=True)
    n_traj = 0
    if "traj_id" in work.columns:
        for _, sub in work.groupby(work["traj_id"].astype(str), sort=False):
            sub = sub.sort_values("_t_ns")
            ax.plot(
                sub["_t_ns"],
                sub["_y"],
                color=color,
                lw=0.55,
                alpha=0.28,
                rasterized=True,
            )
            n_traj += 1
    else:
        ax.plot(work["_t_ns"], work["_y"], color=color, lw=0.8, alpha=0.7)
        n_traj = 1
    mean = work.groupby("_t_ns", sort=True)["_y"].mean()
    ax.plot(mean.index.to_numpy(), mean.to_numpy(), color="0.05", lw=1.35, label="mean", zorder=4)
    for x in MURATA_RMSD_PEAKS_A:
        ax.axhline(x, color="0.35", ls="--", lw=0.8, zorder=3)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel(ylabel or MOTIF_XLABELS.get(rmsd_col, f"{rmsd_col} (Å)"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title(title or f"{rmsd_col}  n_traj={n_traj}")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_rmsd_vs_time_three_motifs(
    df: pd.DataFrame,
    output_path: Path | str,
    *,
    color: str = "0.35",
    title: Optional[str] = None,
) -> Path:
    """Stacked cation–π / equator / whole-cube RMSD vs time (ns)."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in MOTIF_XLABELS if c in df.columns]
    if not cols:
        raise ValueError("No motif RMSD columns to plot")
    fig, axes = plt.subplots(
        len(cols), 1, figsize=(8.5, 2.7 * len(cols)), sharex=True, constrained_layout=True
    )
    if len(cols) == 1:
        axes = [axes]
    n_traj = int(df["traj_id"].nunique()) if "traj_id" in df.columns else 1
    work = df.copy()
    work["_t_ns"] = time_ns_from_frames(work)
    for ax, col in zip(axes, cols):
        y = pd.to_numeric(work[col], errors="coerce")
        ok = np.isfinite(work["_t_ns"]) & np.isfinite(y)
        sub = work.loc[ok]
        if "traj_id" in sub.columns:
            for _, traj in sub.groupby(sub["traj_id"].astype(str), sort=False):
                traj = traj.sort_values("_t_ns")
                ax.plot(
                    traj["_t_ns"],
                    pd.to_numeric(traj[col], errors="coerce"),
                    color=color,
                    lw=0.5,
                    alpha=0.25,
                    rasterized=True,
                )
        mean = sub.assign(_y=pd.to_numeric(sub[col], errors="coerce")).groupby("_t_ns")["_y"].mean()
        ax.plot(mean.index.to_numpy(), mean.to_numpy(), color="0.05", lw=1.3, zorder=4)
        for x in MURATA_RMSD_PEAKS_A:
            ax.axhline(x, color="0.35", ls="--", lw=0.75, zorder=3)
        ax.set_ylabel(MOTIF_XLABELS[col].replace(" to initial NVT (Å)", "\n(Å)"))
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
    axes[-1].set_xlabel("time (ns)")
    fig.suptitle(title or f"RMSD vs time  (n={n_traj} replicas; dashed 1.0 / 1.5 / 2.7 Å)")
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_rmsd_vs_time_mean_by_cube(
    frames_by_cohort: dict[str, pd.DataFrame],
    output_path: Path | str,
    *,
    rmsd_col: str = "rmsd_cation_pi",
    ylabel: Optional[str] = None,
) -> Path:
    """Mean RMSD vs time, one line per cube."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    cohorts = [c for c in KNOWN_CUBES if c in frames_by_cohort] or list(frames_by_cohort)
    for cube in cohorts:
        df = frames_by_cohort[cube]
        if rmsd_col not in df.columns:
            continue
        t = time_ns_from_frames(df)
        y = pd.to_numeric(df[rmsd_col], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(y)
        tmp = pd.DataFrame({"t": t[ok], "y": y[ok]})
        mean = tmp.groupby("t", sort=True)["y"].mean()
        ax.plot(
            mean.index.to_numpy(),
            mean.to_numpy(),
            color=CUBE_COLORS.get(cube, None),
            lw=1.5,
            label=cube,
        )
    for x in MURATA_RMSD_PEAKS_A:
        ax.axhline(x, color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel(ylabel or MOTIF_XLABELS.get(rmsd_col, f"{rmsd_col} (Å)"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_title("Mean RMSD vs time (dashed: Murata 1.0 / 1.5 / 2.7 Å)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def unit_series_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    """Columns like ``cation_pi_pole_e0_…`` / ``equator_d1_e0_…`` (not pooled RMSD)."""
    cols = sorted(c for c in df.columns if str(c).startswith(prefix))
    if prefix == "cation_pi_e":
        cols = [c for c in cols if not str(c).startswith("cation_pi_eq")]
    return cols


def _unit_legend_label(col: str) -> str:
    import re

    name = str(col)
    hit = re.search(r"_e(\d+)_m(\d+)\w+_m(\d+)", name)
    if hit:
        return f"e{hit.group(1)}  m{hit.group(2)}→m{hit.group(3)}"
    hit = re.search(r"_e(\d+)$", name)
    if hit:
        return f"e{hit.group(1)}"
    return name


def plot_unit_series_vs_time(
    df: pd.DataFrame,
    output_path: Path | str,
    *,
    prefix: str,
    ylabel: str,
    title: Optional[str] = None,
    hlines: Sequence[float] = (),
) -> Path:
    """Mean-over-replicas trace for each locked unit (six cation–π or d1 contacts)."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = unit_series_columns(df, prefix)
    if not cols:
        raise ValueError(f"No columns starting with {prefix!r}")
    work = df.copy()
    work["_t_ns"] = time_ns_from_frames(work)
    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    for col in cols:
        y = pd.to_numeric(work[col], errors="coerce")
        ok = np.isfinite(work["_t_ns"]) & np.isfinite(y)
        tmp = pd.DataFrame({"t": work.loc[ok, "_t_ns"], "y": y[ok]})
        mean = tmp.groupby("t", sort=True)["y"].mean()
        ax.plot(mean.index.to_numpy(), mean.to_numpy(), lw=1.3, label=_unit_legend_label(col))
    for x in hlines:
        ax.axhline(float(x), color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(left=0)
    if "°" in ylabel or "angle" in prefix:
        ax.set_ylim(0, 180)
    else:
        ax.set_ylim(bottom=0)
    ax.set_title(title or ylabel)
    ax.legend(fontsize=7, ncol=3)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_per_unit_motif_series(
    df: pd.DataFrame,
    plots_dir: Path | str,
    *,
    cube: str,
) -> list[Path]:
    """Write mean-over-replicas traces for the six locked units, if present."""
    plots_dir = Path(plots_dir)
    written: list[Path] = []
    specs = (
        ("cation_pi_pole_e", "π(pole) Py⁺–Ph (Å)", (6.5,), "cation_pi_pole"),
        ("cation_pi_eq_e", "π(equator) Py⁺–Ph (Å)", (6.5,), "cation_pi_eq"),
        ("cation_pi_e", "cation–π π(pole) (Å)", (6.5,), "cation_pi_units"),
        ("equator_d1_e", "equatorial d1 C2–C3 (Å)", (4.5, 5.5, 7.0), "equator_d1_units"),
        ("equator_d2_e", "equatorial d2 CPy–C3 (Å)", (), "equator_d2_units"),
        ("cation_pi_angle_e", "π(pole)–Ph–π(eq) angle (°)", (180.0,), "cation_pi_angle"),
        ("equator_angle_e", "d1–C3–d2 angle (°)", (), "equator_angle"),
        ("rmsd_cation_pi_e", "cation–π unit RMSD (Å)", MURATA_RMSD_PEAKS_A, "cation_pi_unit_rmsd"),
        ("rmsd_equator_e", "equator unit RMSD (Å)", MURATA_RMSD_PEAKS_A, "equator_unit_rmsd"),
    )
    for prefix, ylabel, hlines, tag in specs:
        if not unit_series_columns(df, prefix):
            continue
        path = plot_unit_series_vs_time(
            df,
            plots_dir / f"rmsd_vs_time_{tag}.png",
            prefix=prefix,
            ylabel=ylabel,
            title=f"{cube}  {ylabel}  (six locked units)",
            hlines=hlines,
        )
        written.append(path)
    return written


CONTACT_DISTANCE_SPECS: tuple[tuple[str, str, tuple[float, ...], str], ...] = (
    ("cation_pi_pole_e", "π(pole) Py⁺–Ph (Å)", (6.5,), "cation_pi_pole"),
    ("cation_pi_eq_e", "π(equator) Py⁺–Ph (Å)", (6.5,), "cation_pi_eq"),
    ("equator_d1_e", "equatorial d1 C2–C3 (Å)", (4.5, 5.5, 7.0), "equator_d1"),
    ("equator_d2_e", "equatorial d2 CPy–C3 (Å)", (), "equator_d2"),
)

_OPEN_PI_PANEL_TITLES = {
    0: "all cation–π closed",
    1: "one cation–π opened",
    2: "two cation–π opened",
}


def stacked_unit_values(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """All finite distances for locked units with ``prefix`` (six units × frames)."""
    cols = unit_series_columns(df, prefix)
    if not cols:
        return np.array([], dtype=float)
    vals = df[cols].to_numpy(dtype=float).ravel()
    return vals[np.isfinite(vals)]


def stacked_unit_values_with_open(
    df: pd.DataFrame, prefix: str
) -> tuple[np.ndarray, np.ndarray]:
    """Unit distances paired with the frame-level opened π count."""
    cols = unit_series_columns(df, prefix)
    if not cols or "n_open_cation_pi" not in df.columns:
        return np.array([], dtype=float), np.array([], dtype=float)
    vals = df[cols].to_numpy(dtype=float)
    n_open = pd.to_numeric(df["n_open_cation_pi"], errors="coerce").to_numpy(dtype=float)
    n_open = np.repeat(n_open, vals.shape[1])
    flat = vals.ravel()
    ok = np.isfinite(flat) & np.isfinite(n_open)
    return flat[ok], n_open[ok]


def _draw_distance_hist(
    ax,
    vals: np.ndarray,
    *,
    bins: np.ndarray,
    color: str,
    label: Optional[str] = None,
    filled: bool = False,
) -> None:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return
    if filled:
        ax.hist(
            vals,
            bins=bins,
            density=True,
            histtype="stepfilled",
            color=color,
            alpha=0.35,
            lw=0,
        )
    ax.hist(
        vals,
        bins=bins,
        density=True,
        histtype="step",
        color=color,
        lw=1.4,
        label=label,
    )


def _mark_distance_guides(ax, hlines: Sequence[float]) -> None:
    for x in hlines:
        ax.axvline(float(x), color="0.35", ls="--", lw=0.8, zorder=3)


def plot_contact_distance_overlay(
    frames_by_cohort: dict[str, pd.DataFrame],
    output_path: Path | str,
    *,
    prefix: str,
    xlabel: str,
    hlines: Sequence[float] = (),
    xmax: float = 12.0,
    traj_filter: str = "all",
) -> Path:
    """Overlay cubes: pooled histogram of one locked-unit distance."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bins = np.arange(0.0, xmax + 0.1, 0.1)
    fig, ax = plt.subplots(figsize=(8.0, 4.2), constrained_layout=True)
    for cube in [c for c in KNOWN_CUBES if c in frames_by_cohort] or list(frames_by_cohort):
        apo = filter_non_encapsulated(frames_by_cohort[cube], traj_filter=traj_filter)
        vals = stacked_unit_values(apo, prefix)
        vals = vals[(vals >= 0) & (vals <= xmax)]
        _draw_distance_hist(
            ax,
            vals,
            bins=bins,
            color=CUBE_COLORS.get(cube, "0.35"),
            label=f"{cube}  n={len(vals):,}",
        )
    _mark_distance_guides(ax, hlines)
    ax.set_xlim(0, xmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("distribution")
    ax.set_title(_filter_title(traj_filter))
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_contact_distance_by_cube(
    frames_by_cohort: dict[str, pd.DataFrame],
    output_path: Path | str,
    *,
    prefix: str,
    xlabel: str,
    hlines: Sequence[float] = (),
    xmax: float = 12.0,
    traj_filter: str = "all",
) -> Path:
    """Per-cube panels of one locked-unit distance."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohorts = [c for c in KNOWN_CUBES if c in frames_by_cohort] or list(frames_by_cohort)
    n = len(cohorts)
    ncols = 3
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows), squeeze=False)
    bins = np.arange(0.0, xmax + 0.1, 0.1)
    for i, cube in enumerate(cohorts):
        ax = axes[i // ncols][i % ncols]
        apo = filter_non_encapsulated(frames_by_cohort[cube], traj_filter=traj_filter)
        vals = stacked_unit_values(apo, prefix)
        vals = vals[(vals >= 0) & (vals <= xmax)]
        color = CUBE_COLORS.get(cube, "0.4")
        _draw_distance_hist(ax, vals, bins=bins, color=color, filled=True)
        _mark_distance_guides(ax, hlines)
        ax.set_xlim(0, xmax)
        ax.set_title(f"{cube}  n={len(vals):,}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("distribution")
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(_filter_title(traj_filter), fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_contact_distance_four_panel(
    frames_by_cohort: dict[str, pd.DataFrame],
    output_path: Path | str,
    *,
    xmax: float = 12.0,
    traj_filter: str = "all",
) -> Path:
    """One figure: pole π, equator π, d1, and d2 overlays."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    bins = np.arange(0.0, xmax + 0.1, 0.1)
    cubes = [c for c in KNOWN_CUBES if c in frames_by_cohort] or list(frames_by_cohort)
    for ax, (prefix, xlabel, hlines, _) in zip(axes.ravel(), CONTACT_DISTANCE_SPECS):
        for cube in cubes:
            apo = filter_non_encapsulated(frames_by_cohort[cube], traj_filter=traj_filter)
            vals = stacked_unit_values(apo, prefix)
            vals = vals[(vals >= 0) & (vals <= xmax)]
            _draw_distance_hist(
                ax,
                vals,
                bins=bins,
                color=CUBE_COLORS.get(cube, "0.35"),
                label=cube,
            )
        _mark_distance_guides(ax, hlines)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("distribution")
        ax.set_title(xlabel.split(" (")[0])
    axes[0, 1].legend(fontsize=7, ncol=2, loc="upper right")
    fig.suptitle(_filter_title(traj_filter), fontsize=11)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_d1_d2_by_open_cation_pi(
    frames_by_cohort: dict[str, pd.DataFrame],
    output_path: Path | str,
    *,
    xmax: float = 12.0,
    traj_filter: str = "all",
) -> Path:
    """Murata Fig. S11-style: d1 and d2 vs 0 / 1 / ≥2 opened pole π contacts."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("equator_d1_e", "d1 [Å]", (4.5, 5.5, 7.0)),
        ("equator_d2_e", "d2 [Å]", ()),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), constrained_layout=True)
    bins = np.arange(0.0, xmax + 0.1, 0.1)
    cubes = [c for c in KNOWN_CUBES if c in frames_by_cohort] or list(frames_by_cohort)
    for row, (prefix, xlabel, hlines) in enumerate(metrics):
        for col, n_open in enumerate((0, 1, 2)):
            ax = axes[row, col]
            for cube in cubes:
                apo = filter_non_encapsulated(
                    frames_by_cohort[cube], traj_filter=traj_filter
                )
                vals, opened = stacked_unit_values_with_open(apo, prefix)
                if vals.size == 0:
                    continue
                keep = open_count_bin(pd.Series(opened)).to_numpy() == n_open
                subset = vals[keep]
                subset = subset[(subset >= 0) & (subset <= xmax)]
                _draw_distance_hist(
                    ax,
                    subset,
                    bins=bins,
                    color=CUBE_COLORS.get(cube, "0.35"),
                    label=cube if row == 0 and col == 0 else None,
                )
            _mark_distance_guides(ax, hlines)
            ax.set_xlim(0, xmax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("distribution")
            if row == 0:
                ax.set_title(_OPEN_PI_PANEL_TITLES[n_open])
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle(
        f"{_filter_title(traj_filter)}  ·  d1 / d2 when pole π is closed or opened",
        fontsize=11,
    )
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def write_contact_distance_plots(
    frames_by_cohort: dict[str, pd.DataFrame],
    plots_dir: Path | str,
    *,
    traj_filter: str = "all",
    suffix: str = "",
) -> list[Path]:
    """Write pole/eq π, d1, d2 histograms and the S11-style open/closed split."""
    plots_dir = Path(plots_dir)
    written: list[Path] = []
    if not any(
        unit_series_columns(df, "cation_pi_pole_e")
        or unit_series_columns(df, "equator_d1_e")
        for df in frames_by_cohort.values()
    ):
        return written
    four = plot_contact_distance_four_panel(
        frames_by_cohort,
        plots_dir / f"dist_pole_eq_d1_d2{suffix}.png",
        traj_filter=traj_filter,
    )
    written.append(four)
    if any("n_open_cation_pi" in df.columns for df in frames_by_cohort.values()):
        split = plot_d1_d2_by_open_cation_pi(
            frames_by_cohort,
            plots_dir / f"dist_d1_d2_by_open_cation_pi{suffix}.png",
            traj_filter=traj_filter,
        )
        written.append(split)
    for prefix, xlabel, hlines, tag in CONTACT_DISTANCE_SPECS:
        if not any(unit_series_columns(df, prefix) for df in frames_by_cohort.values()):
            continue
        written.append(
            plot_contact_distance_overlay(
                frames_by_cohort,
                plots_dir / f"dist_{tag}_overlay{suffix}.png",
                prefix=prefix,
                xlabel=xlabel,
                hlines=hlines,
                traj_filter=traj_filter,
            )
        )
        written.append(
            plot_contact_distance_by_cube(
                frames_by_cohort,
                plots_dir / f"dist_{tag}_by_cube{suffix}.png",
                prefix=prefix,
                xlabel=xlabel,
                hlines=hlines,
                traj_filter=traj_filter,
            )
        )
    return written


def count_elongated_d1(
    df: pd.DataFrame,
    *,
    elongated_lo: float = 7.0,
) -> pd.Series:
    """Per-frame count of locked equatorial d1 values ≥ 7.0 Å."""
    cols = unit_series_columns(df, "equator_d1_e")
    if not cols:
        return pd.Series(np.zeros(len(df), dtype=int), index=df.index)
    vals = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return pd.Series(np.sum(vals >= float(elongated_lo), axis=1).astype(int), index=df.index)


def label_motif_metastructures(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``murata_d1_n_elongated`` and ``murata_metastructure`` on motif CSVs."""
    from src.ChangepointAnalysis.murata_criteria import attach_murata_metastructure_labels
    from src.ChangepointAnalysis.murata_d1 import D1_ELONGATED_LO

    out = df.copy()
    out["murata_d1_n_elongated"] = count_elongated_d1(out, elongated_lo=D1_ELONGATED_LO)
    return attach_murata_metastructure_labels(out)


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
    """Per-unit cation–π / equatorial distances and RMSD vs ``ref_frame``.

    Pooled ``rmsd_cation_pi`` / ``rmsd_equator`` are kept for comparison.
    Each Ph sandwich stores π(pole), π(equator), and the angle at Ph.
    Each equatorial edge stores d1, d2, and the angle at C3.
    """
    from src.ChangepointAnalysis.murata_criteria import (
        CATION_PI_OPEN_LO,
        MURATA_RMSD_MOTIFS,
        cation_pi_angle_column,
        cation_pi_eq_column,
        cation_pi_pole_column,
        cation_pi_unit_indices,
        equator_angle_column,
        equator_d1_column,
        equator_d1_d2_angle_deg,
        equator_d2_column,
        equator_d2_distance,
        equator_unit_indices,
        angle_at_vertex_deg,
        lock_cation_pi_sandwiches,
        lock_equator_edges,
        motif_atom_indices,
        resolve_murata_criteria_atoms,
        _indices_for,
        _role_point,
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
    monomers = sorted(set(int(m) for m in roles["monomer"]))
    u.trajectory[int(ref_frame)]
    pos0 = np.asarray(u.atoms.positions, dtype=float)
    sandwiches = lock_cation_pi_sandwiches(pos0, roles)
    equator_edges = lock_equator_edges(pos0, roles)

    groups: dict[str, Any] = {"assembly_rmsd_to_ref": u.select_atoms("resname MOL")}
    for name, role_names in MURATA_RMSD_MOTIFS.items():
        idx = motif_atom_indices(roles, role_names)
        groups[f"rmsd_{name}"] = u.atoms[idx]
    pole_dist_cols: list[str] = []
    eq_pi_dist_cols: list[str] = []
    cation_angle_cols: list[str] = []
    equator_d1_cols: list[str] = []
    equator_d2_cols: list[str] = []
    equator_angle_cols: list[str] = []
    unit_rows: list[dict[str, object]] = []
    pole_ags = {m: u.atoms[_indices_for(roles, m, "py_pole")] for m in monomers}
    ph_ags = {m: u.atoms[_indices_for(roles, m, "ph")] for m in monomers}
    eq_ags = {m: u.atoms[_indices_for(roles, m, "py_eq")] for m in monomers}
    for k, (pole_m, ph_m, eq_m) in enumerate(sandwiches):
        pole_col = cation_pi_pole_column(k, pole_m, ph_m)
        eq_col = cation_pi_eq_column(k, eq_m, ph_m)
        ang_col = cation_pi_angle_column(k, ph_m)
        rmsd_col = f"rmsd_cation_pi_e{k}"
        pole_dist_cols.append(pole_col)
        eq_pi_dist_cols.append(eq_col)
        cation_angle_cols.append(ang_col)
        groups[rmsd_col] = u.atoms[cation_pi_unit_indices(roles, pole_m, ph_m, eq_m)]
        unit_rows.append(
            {
                "kind": "cation_pi",
                "edge": k,
                "mon_a": pole_m,
                "mon_b": ph_m,
                "mon_c": eq_m,
                "distance_column": pole_col,
                "eq_distance_column": eq_col,
                "angle_column": ang_col,
                "rmsd_column": rmsd_col,
            }
        )
    for k, (i, j) in enumerate(equator_edges):
        d1_col = equator_d1_column(k, i, j)
        d2_col = equator_d2_column(k, i, j)
        ang_col = equator_angle_column(k, j)
        rmsd_col = f"rmsd_equator_e{k}"
        equator_d1_cols.append(d1_col)
        equator_d2_cols.append(d2_col)
        equator_angle_cols.append(ang_col)
        groups[rmsd_col] = u.atoms[equator_unit_indices(roles, i, j)]
        unit_rows.append(
            {
                "kind": "equator",
                "edge": k,
                "mon_a": i,
                "mon_b": j,
                "distance_column": d1_col,
                "d2_distance_column": d2_col,
                "angle_column": ang_col,
                "rmsd_column": rmsd_col,
            }
        )
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
        pos = np.asarray(u.atoms.positions, dtype=float)
        for k, (pole_m, ph_m, eq_m) in enumerate(sandwiches):
            pole = pole_ags[pole_m].positions.mean(axis=0)
            ph = ph_ags[ph_m].positions.mean(axis=0)
            eq = eq_ags[eq_m].positions.mean(axis=0)
            row[pole_dist_cols[k]] = float(np.linalg.norm(pole - ph))
            row[eq_pi_dist_cols[k]] = float(np.linalg.norm(eq - ph))
            row[cation_angle_cols[k]] = angle_at_vertex_deg(pole, ph, eq)
        for k, (i, j) in enumerate(equator_edges):
            r2 = _role_point(pos, roles, i, "r2")
            r3 = _role_point(pos, roles, j, "r3")
            row[equator_d1_cols[k]] = float(np.linalg.norm(r2 - r3))
            row[equator_d2_cols[k]] = equator_d2_distance(pos, roles, i, j)
            row[equator_angle_cols[k]] = equator_d1_d2_angle_deg(pos, roles, i, j)
        if pole_dist_cols:
            dists = np.array([row[c] for c in pole_dist_cols], dtype=float)
            row["n_open_cation_pi"] = int(np.sum(dists >= CATION_PI_OPEN_LO))
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["motif_atom_counts"] = {col: int(len(ag)) for col, ag in groups.items()}
    out.attrs["motif_units"] = pd.DataFrame(unit_rows)
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
    # Snapshot before guest merge: pandas drops DataFrame.attrs on merge.
    units = df.attrs.get("motif_units")
    if isinstance(units, pd.DataFrame):
        units = units.copy()
    df.insert(0, "cohort", cube)
    if gsa_csv:
        df = attach_guest_occupancy(df, gsa_csv)
    elif "n_guest_inside_cavity" not in df.columns:
        df["n_guest_inside_cavity"] = 0.0
    path = Path(out_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if isinstance(units, pd.DataFrame) and len(units):
        units_path = path.with_name(path.name.replace("_motif_rmsd.csv", "_motif_units.csv"))
        units.insert(0, "traj_id", traj_id)
        units.insert(0, "cohort", cube)
        units.to_csv(units_path, index=False)
    return {"cube": cube, "traj_id": traj_id, "path": str(path), "n": int(len(df))}
