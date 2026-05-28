"""
Test script for src/utils/ruptures_utils.py

Loads trajectory from traj/BMMpM_ca.prmtop + traj/BMMpM_891249_mdcrd_v.trj,
computes RMSD and Rg time series, then runs changepoint detection with
multiple methods and visualizes results.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda

from src.TrajectoryIterator import TrajectoryIterator
from src.TrajectoryMetrics import MetricPassSpec, StackedMetricsObserver
from src.utils.ruptures_utils import (
    detect_changepoints,
    detect_changepoints_multiseries,
    ChangePointResult,
)

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
TOPOLOGY = ROOT / "traj" / "BMMpM_ca.prmtop"
TRAJECTORY = ROOT / "traj" / "BMMpM_891249_mdcrd_v.trj"
SELECTION = "not water and not name I and not name Na+"
STRIDE = 10  # subsample for speed on large trajectories
OUTPUT_DIR = ROOT / "output" / "test_ruptures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_metrics(universe: mda.Universe, selection: str) -> tuple[np.ndarray, np.ndarray]:
    """Compute RMSD and Rg in a single iterator pass using StackedMetricsObserver."""
    n_frames = len(range(0, universe.trajectory.n_frames, STRIDE))

    specs = [
        MetricPassSpec(result_key="rmsd", kind="rmsd", selection=selection, ref_frame=0),
        MetricPassSpec(result_key="rg", kind="rg", selection=selection),
    ]
    observer = StackedMetricsObserver(specs, n_frames=n_frames)

    iterator = TrajectoryIterator(universe)
    iterator.subscribe(observer)
    iterator.iterate(step=STRIDE)

    rmsd = observer.results["rmsd"]
    rg = observer.results["rg"]
    return rmsd, rg


def plot_changepoints(
    signal: np.ndarray,
    result: ChangePointResult,
    title: str,
    ylabel: str,
    filename: str,
):
    """Plot signal with detected changepoints marked as vertical lines."""
    fig, ax = plt.subplots(figsize=(12, 4))
    frames = np.arange(len(signal)) * STRIDE
    signal_line, = ax.plot(frames, signal, lw=0.6, color="steelblue", alpha=0.8)

    first_bkp_line = None
    for bkp in result.breakpoints:
        line = ax.axvline(bkp * STRIDE, color="red", ls="--", lw=1.2, alpha=0.8)
        if first_bkp_line is None:
            first_bkp_line = line

    for i, (start, end) in enumerate(result.segment_ranges):
        ax.axhspan(
            result.segment_means[i] - result.segment_stds[i],
            result.segment_means[i] + result.segment_stds[i],
            xmin=(start) / len(signal),
            xmax=(end) / len(signal),
            alpha=0.1,
            color=f"C{i % 10}",
        )
        ax.hlines(
            result.segment_means[i],
            start * STRIDE,
            end * STRIDE,
            colors=f"C{i % 10}",
            lw=2,
            alpha=0.7,
        )

    ax.set_xlabel("Frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles = [signal_line]
    labels = ["signal"]
    if first_bkp_line is not None:
        handles.append(first_bkp_line)
        labels.append("changepoints")
    ax.legend(handles, labels, loc="upper right", fontsize=8)
    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print(f"Loading topology: {TOPOLOGY}")
    print(f"Loading trajectory: {TRAJECTORY}")
    print(f"Selection: {SELECTION}")
    print(f"Stride: {STRIDE}")
    print()

    u = mda.Universe(str(TOPOLOGY), str(TRAJECTORY))
    n_frames = u.trajectory.n_frames
    effective_n_frames = len(range(0, n_frames, STRIDE))
    print(f"Trajectory loaded: {n_frames} frames, {u.atoms.n_atoms} atoms")
    print(f"Effective frames after stride: {effective_n_frames}")
    print()

    # ── Compute metrics (single iterator pass) ─────────────────────
    print("Computing RMSD + Rg in single iterator pass...")
    rmsd, rg = compute_metrics(u, SELECTION)
    print(f"  RMSD shape: {rmsd.shape}, range: [{rmsd.min():.2f}, {rmsd.max():.2f}] Å")
    print(f"  Rg shape:   {rg.shape}, range: [{rg.min():.2f}, {rg.max():.2f}] Å")
    print()

    # ── Test 1: Pelt with RBF cost (auto penalty) ────────────────
    print("=" * 60)
    print("Test 1: Pelt + RBF (auto penalty) on RMSD")
    print("=" * 60)
    result_pelt = detect_changepoints(rmsd, method="Pelt", cost_model="rbf")
    print(result_pelt.summary())
    plot_changepoints(rmsd, result_pelt, "RMSD — Pelt + RBF", "RMSD (Å)", "rmsd_pelt_rbf.png")
    print()

    # ── Test 2: Binseg with L2, fixed n_bkps ─────────────────────
    print("=" * 60)
    print("Test 2: Binseg + L2 (n_bkps=3) on RMSD")
    print("=" * 60)
    result_binseg = detect_changepoints(rmsd, method="Binseg", cost_model="l2", n_bkps=3)
    print(result_binseg.summary())
    plot_changepoints(rmsd, result_binseg, "RMSD — Binseg + L2 (3 bkps)", "RMSD (Å)", "rmsd_binseg_l2.png")
    print()

    # ── Test 3: BottomUp on Rg ────────────────────────────────────
    print("=" * 60)
    print("Test 3: BottomUp + RBF (auto penalty) on Rg")
    print("=" * 60)
    result_bu = detect_changepoints(rg, method="BottomUp", cost_model="rbf")
    print(result_bu.summary())
    plot_changepoints(rg, result_bu, "Rg — BottomUp + RBF", "Rg (Å)", "rg_bottomup_rbf.png")
    print()

    # ── Test 4: Window method on RMSD ─────────────────────────────
    print("=" * 60)
    print("Test 4: Window + RBF (n_bkps=4) on RMSD")
    print("=" * 60)
    result_win = detect_changepoints(
        rmsd, method="Window", cost_model="rbf", n_bkps=4, window_width=50
    )
    print(result_win.summary())
    plot_changepoints(rmsd, result_win, "RMSD — Window + RBF (4 bkps)", "RMSD (Å)", "rmsd_window_rbf.png")
    print()

    # ── Test 5: KernelCPD on Rg ───────────────────────────────────
    print("=" * 60)
    print("Test 5: KernelCPD + RBF (n_bkps=2) on Rg")
    print("=" * 60)
    result_kern = detect_changepoints(rg, method="KernelCPD", cost_model="rbf", n_bkps=2)
    print(result_kern.summary())
    plot_changepoints(rg, result_kern, "Rg — KernelCPD (2 bkps)", "Rg (Å)", "rg_kernelcpd.png")
    print()

    # ── Test 6: Multi-series detection ────────────────────────────
    print("=" * 60)
    print("Test 6: Multi-series (RMSD + Rg) with Pelt + L2")
    print("=" * 60)
    multi_results = detect_changepoints_multiseries(
        {"RMSD": rmsd, "Rg": rg},
        method="Pelt",
        cost_model="l2",
    )
    for name, res in multi_results.items():
        print(f"\n--- {name} ---")
        print(res.summary())
    print()

    # ── Test 7: Penalty sensitivity sweep on RMSD ─────────────────
    print("=" * 60)
    print("Test 7: Penalty sweep (Pelt + RBF) on RMSD")
    print("=" * 60)
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    penalties = [1.0, 10.0, 100.0]
    for ax, pen in zip(axes, penalties):
        res = detect_changepoints(rmsd, method="Pelt", cost_model="rbf", penalty=pen)
        frames = np.arange(len(rmsd)) * STRIDE
        ax.plot(frames, rmsd, lw=0.5, color="steelblue")
        for bkp in res.breakpoints:
            ax.axvline(bkp * STRIDE, color="red", ls="--", lw=1)
        ax.set_ylabel("RMSD (Å)")
        ax.set_title(f"penalty={pen} → {res.n_breakpoints} breakpoints")
    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    sweep_path = OUTPUT_DIR / "rmsd_penalty_sweep.png"
    fig.savefig(sweep_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {sweep_path}")
    print()

    # ── Summary ───────────────────────────────────────────────────
    print("=" * 60)
    print("All tests completed successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
