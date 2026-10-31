"""Cage geometry aligned to a sustained iodide entry.

For each visit that stays inside at least 20 ps, cage coordinates are read from
200 ps before the entry to 200 ps after. The change across the entry is the
mean on [0, 100) ps minus the mean on [-100, 0). The change already under way
is the mean on [-100, 0) minus the mean on [-200, -100). Their difference is
the kink at the entry. A null draws the same windows at random times in the
same replicas.

The guest's own radial distance is aligned the same way, as a check that the
entry is where the ion arrives.

Example
-------
python scripts/guest_entry_geometry_response.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.guest_events import aligned_slice, guest_events

HALF = 200
STEP = 100
RG_MAX = 15.0
MIN_COMPLETE = 20
LAGS = np.arange(-HALF, HALF + 1)
CAGE_FEATURES = [
    "assembly_rg",
    "cavity_inner_atom_distance_mean",
    "endpoint_dist_mean",
    "total_inter_monomer_contacts",
    "pore_opening_distance_1",
    "octahedrality_score",
]
GUEST_FEATURES = [
    "guest_radial_distance_min",
    "guest_gsa_min_distance",
    "n_guest_inside_cavity",
]
MOTIF_FEATURES = ["n_open_cation_pi", "equator_d1_mean"]
# Plotted as the change from the mean on [-200, -100) ps.
PLOT_FEATURES = [
    "guest_radial_distance_min",
    "guest_gsa_min_distance",
    "cavity_inner_atom_distance_mean",
    "assembly_rg",
    "total_inter_monomer_contacts",
    "n_open_cation_pi",
]
PLOT_LABELS = {
    "guest_radial_distance_min": "Δ radial min (Å)",
    "guest_gsa_min_distance": "Δ guest–cage min (Å)",
    "cavity_inner_atom_distance_mean": "Δ cavity radius (Å)",
    "assembly_rg": "Δ assembly Rg (Å)",
    "total_inter_monomer_contacts": "Δ contacts",
    "n_open_cation_pi": "Δ open poles",
}
COLORS = {
    "BHHpH": "#1b9e77",
    "BHHpM": "#d95f02",
    "BMHpH": "#7570b3",
    "BMHpM": "#e7298a",
    "BMMpH": "#66a61e",
    "BMMpM": "#e6ab02",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--gsa-features-root", type=Path, default=ROOT / "output" / "gsa_features_by_cube")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "guest_cage_geometry")
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument("--min-dwell", type=int, default=20)
    p.add_argument("--n-shift", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _strip(traj_id: str, cube: str) -> str:
    prefix = f"{cube}_"
    return traj_id[len(prefix):] if traj_id.startswith(prefix) else traj_id


def _window_means(trace: np.ndarray) -> tuple[float, float, float]:
    """Return (early, pre, post) means on the three 100 ps blocks of a ±200 ps slice."""
    early = trace[:STEP]
    pre = trace[HALF - STEP:HALF]
    post = trace[HALF:HALF + STEP]
    def _mean(block: np.ndarray) -> float:
        ok = block[np.isfinite(block)]
        if ok.size < MIN_COMPLETE:
            return np.nan
        return float(ok.mean())
    return _mean(early), _mean(pre), _mean(post)


def _one_sided_p(obs: float, null: np.ndarray) -> float:
    """Fraction of null values at least as far from the null median as obs, same side."""
    if not np.isfinite(obs) or null.size == 0:
        return float("nan")
    center = float(np.median(null))
    return float((1 + np.sum((null - center) * np.sign(obs - center) >= abs(obs - center) - 1e-12)) / (null.size + 1))


def _load_cube_series(paths: list[Path], columns: list[str], cube: str) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for path in paths:
        header = set(pd.read_csv(path, nrows=0).columns)
        use = [c for c in columns if c in header]
        df = pd.read_csv(path, usecols=use).sort_values("frame")
        if float(df["assembly_rg"].max()) > RG_MAX or float(df["largest_connected_component_size"].min()) < 6:
            continue
        frame = df["frame"].to_numpy(dtype=int)
        if frame.size == 0 or int(frame[0]) != 0 or not np.array_equal(frame, np.arange(frame.size)):
            raise RuntimeError(f"{path.name} frames are not 0..n-1")
        series = {c: df[c].to_numpy(dtype=float) for c in use if c not in ("frame", "traj_id")}
        pores = [series.get(f"pore_opening_distance_{i}") for i in (1, 2, 3)]
        if all(p is not None for p in pores):
            series["pore_opening_mean"] = np.nanmean(np.column_stack(pores), axis=1)
        out[_strip(str(df["traj_id"].iloc[0]), cube)] = series
    return out


def _d1_mean(path: Path, cube: str) -> tuple[str, np.ndarray, np.ndarray] | None:
    header = list(pd.read_csv(path, nrows=0).columns)
    d1_cols = [c for c in header if c.startswith("equator_d1_")]
    if "n_open_cation_pi" not in header or "frame" not in header or not d1_cols:
        return None
    df = pd.read_csv(path, usecols=["traj_id", "frame", "n_open_cation_pi", *d1_cols]).sort_values("frame")
    frame = df["frame"].to_numpy(dtype=int)
    if frame.size == 0 or int(frame[0]) != 0 or not np.array_equal(frame, np.arange(frame.size)):
        return None
    return (
        _strip(str(df["traj_id"].iloc[0]), cube),
        df["n_open_cation_pi"].to_numpy(dtype=float),
        np.nanmean(df[d1_cols].to_numpy(dtype=float), axis=1),
    )


def _entries(occupied: np.ndarray, min_dwell: int) -> list[int]:
    ev = guest_events(occupied, min_dwell=min_dwell)
    return [int(s) for s in ev.loc[ev["event"] == "entry", "start"].tolist()]


def _replica_traces(series: dict[str, np.ndarray], origins: list[int], feature: str) -> np.ndarray | None:
    y = series.get(feature)
    if y is None or not origins:
        return None
    rows = [aligned_slice(y, origin, LAGS) for origin in origins]
    stacked = np.vstack(rows)
    if not np.isfinite(stacked).any():
        return None
    return np.nanmean(stacked, axis=0)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plots").mkdir(exist_ok=True)

    base_cols = [
        "traj_id", "frame", "n_guest_inside_cavity", "assembly_rg",
        "largest_connected_component_size",
        *CAGE_FEATURES, *GUEST_FEATURES,
        "pore_opening_distance_2", "pore_opening_distance_3",
    ]
    trace_rows: list[dict] = []
    stat_rows: list[dict] = []
    null_rows: list[dict] = []

    for cube in args.cubes:
        paths = sorted((args.gsa_features_root / cube).glob("*_gsa_features.csv"))
        print(f"align {cube}: {len(paths)} trajectories", flush=True)
        series_by_traj = _load_cube_series(paths, base_cols, cube)
        motif_dir = args.output_root / f"{cube}_motif_rmsd"
        seen: set[str] = set()
        if motif_dir.is_dir():
            for path in sorted(motif_dir.rglob("*_motif_rmsd.csv")):
                loaded = _d1_mean(path, cube)
                if loaded is None:
                    continue
                tid, n_open, d1 = loaded
                if tid in seen or tid not in series_by_traj:
                    continue
                if n_open.size != series_by_traj[tid]["n_guest_inside_cavity"].size:
                    continue
                seen.add(tid)
                series_by_traj[tid]["n_open_cation_pi"] = n_open
                series_by_traj[tid]["equator_d1_mean"] = d1

        features = CAGE_FEATURES + ["pore_opening_mean", *GUEST_FEATURES, *MOTIF_FEATURES]
        # Per replica: entry origins and the series, for the null.
        replica_entries: list[tuple[dict[str, np.ndarray], list[int]]] = []
        cube_traces: dict[str, list[np.ndarray]] = {f: [] for f in features}
        for series in series_by_traj.values():
            guest = series.get("n_guest_inside_cavity")
            if guest is None:
                continue
            origins = _entries(guest >= 1, args.min_dwell)
            if not origins:
                continue
            replica_entries.append((series, origins))
            for feature in features:
                trace = _replica_traces(series, origins, feature)
                if trace is not None:
                    cube_traces[feature].append(trace)

        for feature, traces in cube_traces.items():
            if not traces:
                continue
            stack = np.vstack(traces)
            for lag, med, q25, q75 in zip(
                LAGS,
                np.nanmedian(stack, axis=0),
                np.nanpercentile(stack, 25, axis=0),
                np.nanpercentile(stack, 75, axis=0),
            ):
                trace_rows.append({
                    "cube": cube, "feature": feature, "lag_ps": int(lag),
                    "n_replica": int(np.isfinite(stack[:, int(lag) + HALF]).sum()),
                    "median": float(med), "q25": float(q25), "q75": float(q75),
                })

        for feature in features:
            steps, trends, kinks = [], [], []
            for series, origins in replica_entries:
                trace = _replica_traces(series, origins, feature)
                if trace is None:
                    continue
                early, pre, post = _window_means(trace)
                if not (np.isfinite(early) and np.isfinite(pre) and np.isfinite(post)):
                    continue
                steps.append(post - pre)
                trends.append(pre - early)
                kinks.append((post - pre) - (pre - early))
            steps_a = np.asarray(steps, dtype=float)
            trends_a = np.asarray(trends, dtype=float)
            kinks_a = np.asarray(kinks, dtype=float)
            if steps_a.size == 0:
                continue
            # Null: same three windows at a random time in each replica.
            null_step = np.empty(args.n_shift)
            null_kink = np.empty(args.n_shift)
            n_frames = [series["n_guest_inside_cavity"].size for series, _ in replica_entries]
            for draw in range(args.n_shift):
                draw_steps, draw_kinks = [], []
                for (series, _), n in zip(replica_entries, n_frames):
                    if n <= 2 * HALF:
                        continue
                    origin = int(rng.integers(HALF, n - HALF))
                    trace = _replica_traces(series, [origin], feature)
                    if trace is None:
                        continue
                    early, pre, post = _window_means(trace)
                    if not (np.isfinite(early) and np.isfinite(pre) and np.isfinite(post)):
                        continue
                    draw_steps.append(post - pre)
                    draw_kinks.append((post - pre) - (pre - early))
                null_step[draw] = np.median(draw_steps) if draw_steps else np.nan
                null_kink[draw] = np.median(draw_kinks) if draw_kinks else np.nan
            obs_step = float(np.median(steps_a))
            obs_kink = float(np.median(kinks_a))
            stat_rows.append({
                "cube": cube,
                "feature": feature,
                "n_replica": int(steps_a.size),
                "median_step": obs_step,
                "median_trend": float(np.median(trends_a)),
                "median_kink": obs_kink,
                "null_step_median": float(np.nanmedian(null_step)),
                "null_kink_median": float(np.nanmedian(null_kink)),
                "p_step": _one_sided_p(obs_step, null_step[np.isfinite(null_step)]),
                "p_kink": _one_sided_p(obs_kink, null_kink[np.isfinite(null_kink)]),
            })
            for draw, (s, k) in enumerate(zip(null_step, null_kink)):
                null_rows.append({"cube": cube, "feature": feature, "draw": draw, "null_step": s, "null_kink": k})

    traces = pd.DataFrame(trace_rows)
    stats = pd.DataFrame(stat_rows)
    traces.to_csv(args.out_dir / "entry_aligned_traces.csv", index=False)
    stats.to_csv(args.out_dir / "entry_aligned_step.csv", index=False)
    pd.DataFrame(null_rows).to_csv(args.out_dir / "entry_aligned_null.csv", index=False)

    _plot(traces, args.out_dir / "plots" / "entry_aligned_geometry.png")
    show = stats[stats["feature"].isin(PLOT_FEATURES + ["equator_d1_mean", "pore_opening_mean"])]
    print(show.round(4).to_string(index=False), flush=True)
    print(f"wrote {args.out_dir}", flush=True)
    return 0


def _plot(traces: pd.DataFrame, path: Path) -> None:
    if traces.empty:
        return
    fig, axes = plt.subplots(len(PLOT_FEATURES), 1, figsize=(8.2, 11.5), sharex=True)
    for ax, feature in zip(axes, PLOT_FEATURES):
        sub = traces[traces["feature"] == feature]
        for cube, g in sub.groupby("cube"):
            g = g.sort_values("lag_ps")
            base = g.loc[g["lag_ps"].between(-HALF, -STEP), "median"]
            if base.empty or not np.isfinite(base).any():
                continue
            level = float(np.nanmean(base.to_numpy()))
            lag = g["lag_ps"].to_numpy()
            med = g["median"].to_numpy() - level
            med = pd.Series(med).rolling(11, center=True, min_periods=1).mean().to_numpy()
            color = COLORS.get(cube, "0.4")
            ax.plot(lag, med, color=color, lw=1.4, label=cube)
        ax.axvline(0, color="0.3", lw=0.8)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylabel(PLOT_LABELS[feature], fontsize=8)
        ax.tick_params(labelsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=6, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    axes[-1].set_xlabel("Time from sustained entry (ps). Zero is the mean over −200 to −100 ps.")
    fig.suptitle("Cage geometry aligned to iodide entry", fontsize=12, y=1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
