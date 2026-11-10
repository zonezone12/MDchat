"""Figure 5: contact-breakpoint nesting and the cavity response at iodide entry.

Panel (a) is the mean, over replicas, of the fraction of contact breakpoints
matched one-to-one to a cage-geometry breakpoint within ±50 ps. Contact
breakpoints are the endpoint family in ``breakpoint_nesting.csv`` (pair
``endpoint-gsa``). The two nulls keep breakpoint counts and spacing: a
circular shift inside the same replica, and pairing one replica's contact
breakpoints with another replica's cage-geometry breakpoints.

Panel (b) aligns sustained iodide entries (new occupancy held at least 20 ps)
at t = 0. The line is the median cavity radius across replicas that have such
an entry, minus its mean over −200 to −100 ps, then an 11 ps centred rolling
mean. The nearest-iodide distance is not drawn: t = 0 is the frame where that
distance crosses the 8 Å occupancy boundary, so the drop is fixed by the
alignment. The shaded band is the central 95% of the same median curve when
each of those replicas is instead aligned at one random time (200 draws).

Example
-------
python scripts/plot_event_timing.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.guest_entry_response import (
    HALF,
    LAGS,
    _entries,
    _load_cube_series,
)
from src.ChangepointAnalysis.guest_events import aligned_slice
from src.utils.run_log import RunContext

# Same system colours as Figure 2.
CUBE_COLORS: dict[str, str] = {
    "BHHpH": "#1f77b4",
    "BHHpM": "#5fa8d3",
    "BMHpH": "#2ca02c",
    "BMHpM": "#98df8a",
    "BMMpH": "#ff7f0e",
    "BMMpM": "#d62728",
}
SERIES = (
    ("recall_sparse", "Observed", "#252525", ""),
    ("recall_sparse_null_shift", "Circular shift", "#bdbdbd", ""),
    ("recall_sparse_null_cross", "Cross-replica", "#ffffff", "///"),
)
CAVITY = "cavity_inner_atom_distance_mean"
EARLY = (-200, -100)
SMOOTH = 11
N_DRAW = 200
NULL_SEED = 0
LOAD_COLUMNS = [
    "traj_id",
    "frame",
    "n_guest_inside_cavity",
    "assembly_rg",
    "largest_connected_component_size",
    CAVITY,
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "event_timing_figure",
    )
    p.add_argument(
        "--manuscript-dir",
        type=Path,
        default=Path("docs/figures"),
        help="Also copy the figure here. Pass '' to skip.",
    )
    return p.parse_args()


def load_nesting(output_root: Path) -> pd.DataFrame:
    path = output_root / "breakpoint_nesting" / "breakpoint_nesting.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    nesting = pd.read_csv(path)
    sub = nesting[nesting["pair"] == "endpoint-gsa"].copy()
    missing = [cube for cube in KNOWN_CUBES if cube not in set(sub["cube"])]
    if missing:
        raise ValueError(f"endpoint-gsa nesting missing cubes: {missing}")
    return sub.set_index("cube").loc[list(KNOWN_CUBES)].reset_index()


def load_traces(output_root: Path) -> pd.DataFrame:
    path = output_root / "guest_cage_geometry" / "entry_aligned_traces.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    traces = pd.read_csv(path)
    keep = traces["feature"] == CAVITY
    traces = traces.loc[keep].copy()
    missing = [
        cube
        for cube in KNOWN_CUBES
        if cube not in set(traces["cube"])
    ]
    if missing:
        raise ValueError(f"entry-aligned traces missing cubes: {missing}")
    return traces


def _smoothed(lag: np.ndarray, value: np.ndarray) -> np.ndarray:
    order = np.argsort(lag)
    series = pd.Series(value[order]).rolling(SMOOTH, center=True, min_periods=1).mean()
    out = np.empty_like(value, dtype=float)
    out[order] = series.to_numpy()
    return out


def _entry_replicas(output_root: Path, cube: str) -> list[np.ndarray]:
    """Cavity-radius traces for replicas with a sustained entry.

    The same quality-control drop as ``guest_entry_response``: a replica whose
    assembly radius exceeds 15 Å, or whose contact graph falls below six
    monomers, is left out.
    """
    paths = sorted((output_root / "gsa_features_by_cube" / cube).glob("*_gsa_features.csv"))
    if not paths:
        raise FileNotFoundError(f"No gsa feature tables for {cube}")
    loaded = _load_cube_series(paths, LOAD_COLUMNS, cube)
    kept: list[np.ndarray] = []
    for series in loaded.values():
        guest = series.get("n_guest_inside_cavity")
        cavity = series.get(CAVITY)
        if guest is None or cavity is None or cavity.size <= 2 * HALF:
            continue
        if not _entries(guest >= 1, 20):
            continue
        kept.append(cavity)
    if not kept:
        raise ValueError(f"{cube} has no replica with a sustained iodide entry")
    return kept


def cavity_null_band(output_root: Path, cache_path: Path) -> pd.DataFrame:
    """Central 95% of the baseline-subtracted cavity median at random times.

    One draw aligns every replica that has a sustained entry at a single
    random frame, takes the median across replicas, and subtracts that
    curve's mean on [−200, −100] ps. The band is the 2.5 and 97.5
    percentiles of 200 such curves after the same 11 ps smooth as the
    observed line. The cache is reused only when those settings match.
    """
    if cache_path.is_file():
        cached = pd.read_csv(cache_path)
        settings_ok = (
            {"cube", "lag_ps", "p2_5", "p97_5"}.issubset(cached.columns)
            and set(cached["cube"]) == set(KNOWN_CUBES)
            and int(cached["n_draw"].iloc[0]) == N_DRAW
            and int(cached["seed"].iloc[0]) == NULL_SEED
            and int(cached["smooth"].iloc[0]) == SMOOTH
        )
        if settings_ok:
            return cached

    early = (LAGS >= EARLY[0]) & (LAGS <= EARLY[1])
    rng = np.random.default_rng(NULL_SEED)
    rows: list[dict] = []
    for cube in KNOWN_CUBES:
        replicas = _entry_replicas(output_root, cube)
        print(f"null {cube}: {len(replicas)} replicas", flush=True)
        curves = np.empty((N_DRAW, LAGS.size), dtype=float)
        lengths = [y.size for y in replicas]
        for draw in range(N_DRAW):
            aligned = np.empty((len(replicas), LAGS.size), dtype=float)
            for i, y in enumerate(replicas):
                origin = int(rng.integers(HALF, lengths[i] - HALF))
                aligned[i] = aligned_slice(y, origin, LAGS)
            median = np.nanmedian(aligned, axis=0)
            median = median - float(np.nanmean(median[early]))
            curves[draw] = _smoothed(LAGS.astype(float), median)
        low, high = np.nanpercentile(curves, [2.5, 97.5], axis=0)
        for lag, lo, hi in zip(LAGS, low, high):
            rows.append(
                {
                    "cube": cube,
                    "lag_ps": int(lag),
                    "p2_5": float(lo),
                    "p97_5": float(hi),
                    "n_replica": len(replicas),
                    "n_draw": N_DRAW,
                    "seed": NULL_SEED,
                    "smooth": SMOOTH,
                }
            )
    band = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    band.to_csv(cache_path, index=False)
    return band


def plot_figure(
    nesting: pd.DataFrame,
    traces: pd.DataFrame,
    band: pd.DataFrame,
    out: Path,
) -> Path:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(9.6, 7.4))
    gs = fig.add_gridspec(
        len(KNOWN_CUBES),
        2,
        width_ratios=[1.0, 1.28],
        wspace=0.28,
        hspace=0.38,
    )
    ax = fig.add_subplot(gs[:, 0])
    axes_b: list = []
    for i in range(len(KNOWN_CUBES)):
        axes_b.append(
            fig.add_subplot(
                gs[i, 1],
                sharex=axes_b[0] if axes_b else None,
                sharey=axes_b[0] if axes_b else None,
            )
        )
    x = np.arange(len(KNOWN_CUBES))
    width = 0.22
    gap = 0.26
    for i, (column, label, color, hatch) in enumerate(SERIES):
        heights = [
            float(nesting.loc[nesting["cube"] == cube, column].iloc[0])
            for cube in KNOWN_CUBES
        ]
        ax.bar(
            x + (i - 1) * gap,
            heights,
            width=width,
            color=color,
            edgecolor="0.15",
            linewidth=0.4,
            hatch=hatch,
            label=label,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(list(KNOWN_CUBES), rotation=30, ha="right")
    ax.set_ylabel("Matched fraction")
    ax.set_xlabel("System")
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.55, len(KNOWN_CUBES) - 0.45)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(
        fontsize=8,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=3,
        borderaxespad=0.0,
        columnspacing=0.8,
        handlelength=1.4,
        handleheight=0.9,
    )
    ax.text(-0.14, 1.02, "(a)", transform=ax.transAxes, fontweight="bold", va="bottom")

    y_span: list[float] = []
    prepared: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for cube in KNOWN_CUBES:
        cavity = traces[(traces["cube"] == cube) & (traces["feature"] == CAVITY)].sort_values("lag_ps")
        null = band[band["cube"] == cube].sort_values("lag_ps")
        lag_c = cavity["lag_ps"].to_numpy(dtype=float)
        med_c = cavity["median"].to_numpy(dtype=float)
        early = (lag_c >= EARLY[0]) & (lag_c <= EARLY[1])
        level = float(np.nanmean(med_c[early]))
        observed = _smoothed(lag_c, med_c - level)
        lo = null["p2_5"].to_numpy(dtype=float)
        hi = null["p97_5"].to_numpy(dtype=float)
        prepared.append((cube, lag_c, observed, lo, hi))
        y_span.extend(
            [
                float(np.nanmin(lo)),
                float(np.nanmax(hi)),
                float(np.nanmin(observed)),
                float(np.nanmax(observed)),
            ]
        )
    pad = 0.02
    y_lo, y_hi = min(y_span) - pad, max(y_span) + pad

    for row, (cube, lag_c, observed, lo, hi) in enumerate(prepared):
        ax_c = axes_b[row]
        color = CUBE_COLORS[cube]
        ax_c.fill_between(lag_c, lo, hi, color="0.75", alpha=0.7, lw=0, zorder=1)
        ax_c.plot(lag_c, observed, color=color, lw=1.45, zorder=3)
        ax_c.axvline(0, color="0.35", lw=0.7, zorder=2)
        ax_c.axhline(0, color="0.45", lw=0.6, zorder=2)
        ax_c.set_xlim(-200, 200)
        ax_c.set_ylim(y_lo, y_hi)
        ax_c.set_yticks([-0.1, 0.0, 0.1])
        ax_c.tick_params(labelsize=8)
        ax_c.text(
            0.99,
            0.86,
            cube,
            transform=ax_c.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=color,
        )
        if row < len(prepared) - 1:
            ax_c.tick_params(labelbottom=False)
        else:
            ax_c.set_xlabel("Time from sustained entry (ps)")
    axes_b[2].set_ylabel("Δ cavity radius (Å)")
    axes_b[0].text(-0.16, 1.08, "(b)", transform=axes_b[0].transAxes, fontweight="bold", va="bottom")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with RunContext.from_namespace(args, name="plot_event_timing"):
        nesting = load_nesting(output_root)
        traces = load_traces(output_root)
        band_path = out_dir / "cavity_null_band.csv"
        band = cavity_null_band(output_root, band_path)
        for cube in KNOWN_CUBES:
            cavity = traces[(traces["cube"] == cube) & (traces["feature"] == CAVITY)].sort_values("lag_ps")
            null = band[band["cube"] == cube].sort_values("lag_ps")
            lag = cavity["lag_ps"].to_numpy(dtype=float)
            med = cavity["median"].to_numpy(dtype=float)
            early = (lag >= EARLY[0]) & (lag <= EARLY[1])
            observed = _smoothed(lag, med - float(np.nanmean(med[early])))
            lo = null["p2_5"].to_numpy(dtype=float)
            hi = null["p97_5"].to_numpy(dtype=float)
            outside = (observed < lo) | (observed > hi)
            print(
                f"{cube}: null band {lo.min():+.3f} to {hi.max():+.3f} Å; "
                f"observed outside on {int(outside.sum())}/{outside.size} ps",
                flush=True,
            )
        nesting_path = out_dir / "contact_geometry_match.csv"
        nesting.to_csv(nesting_path, index=False)
        fig_path = plot_figure(nesting, traces, band, out_dir / "figure5_event_timing.png")
        manuscript_dir = str(args.manuscript_dir).strip()
        if manuscript_dir:
            dest_dir = Path(manuscript_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in (fig_path, fig_path.with_suffix(".pdf")):
                dest = dest_dir / src.name
                try:
                    dest.write_bytes(src.read_bytes())
                except OSError:
                    dest = dest_dir / f"{src.stem}_redraw{src.suffix}"
                    dest.write_bytes(src.read_bytes())
                print(f"Copied figure to {dest}")
        print(f"Wrote {fig_path}")
        print(f"Wrote {nesting_path}")
        print(f"Wrote {band_path}")
        show = nesting[
            [
                "cube",
                "n_traj",
                "recall_sparse",
                "recall_sparse_null_shift",
                "recall_sparse_null_cross",
                "excess_over_chance",
            ]
        ]
        print(show.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
