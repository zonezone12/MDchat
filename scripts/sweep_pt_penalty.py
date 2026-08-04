"""
Penalty sweep for the NH4NO3 proton-transfer changepoint signals.

The main driver (run_proton_transfer_analysis.py) uses a fixed BIC-style default
penalty = log(n_frames) for Pelt. This script instead SWEEPS a grid of penalties
for each trajectory and signal, records the breakpoint count vs penalty, finds
the elbow (knee) of the curve, and plots it -- so the penalty is chosen by
structural stability rather than a single heuristic.

Two signals per trajectory (read from the already-computed features_*.csv):
  * delta              -> 1-D CV, cost model rbf
  * pairwise_distances -> z-scored all-pairwise matrix, cost model l2

Outputs (into --output-dir):
  penalty_sweep_delta.csv
  penalty_sweep_pairwise.csv
  penalty_recommendation.csv           (elbow penalty + n_bkps per tag/signal)
  penalty_sweep_delta.png
  penalty_sweep_pairwise.png

Example
-------
python scripts/sweep_pt_penalty.py --output-dir output/nh4no3_pt
python scripts/sweep_pt_penalty.py --output-dir output/nh4no3_pt --n-penalties 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ProtonTransfer import pairwise_feature_columns, zscore_matrix

_SIGNALS = {
    "delta": {"cost": "rbf", "label": "delta CV (1-D)"},
    "pairwise": {"cost": "l2", "label": "pairwise distances (z-scored)"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Penalty sweep for PT changepoints")
    p.add_argument("--output-dir", type=Path, default=ROOT / "output" / "nh4no3_pt")
    p.add_argument(
        "--tags",
        nargs="+",
        default=None,
        help="Subset of tags (default: all features_*.csv found)",
    )
    p.add_argument("--method", default="Pelt")
    p.add_argument("--min-size", type=int, default=20)
    p.add_argument("--jump", type=int, default=5)
    p.add_argument("--n-penalties", type=int, default=18)
    p.add_argument(
        "--penalty-min",
        type=float,
        default=None,
        help="Lower bound (default: 0.3 * log(n))",
    )
    p.add_argument(
        "--penalty-max",
        type=float,
        default=None,
        help="Upper bound (default: 40 * log(n))",
    )
    p.add_argument(
        "--signals",
        nargs="+",
        default=["delta", "pairwise"],
        choices=["delta", "pairwise"],
    )
    return p.parse_args()


def _discover_tags(out_dir: Path) -> List[str]:
    return [p.stem.replace("features_", "", 1) for p in sorted(out_dir.glob("features_*.csv"))]


def _penalty_grid(n: int, args: argparse.Namespace) -> np.ndarray:
    # Grid centered on the BIC-style default log(n): below it we get more
    # breakpoints, above it fewer. The fast solvers handle the full range.
    ref = float(np.log(max(n, 2)))
    lo = args.penalty_min if args.penalty_min is not None else 0.3 * ref
    hi = args.penalty_max if args.penalty_max is not None else 30.0 * ref
    lo = max(lo, 0.3)
    return np.unique(np.geomspace(lo, hi, num=args.n_penalties))


def _elbow(penalties: np.ndarray, counts: np.ndarray) -> Optional[Dict[str, float]]:
    """Knee of counts vs log10(penalty): farthest interior point from the chord."""
    if len(penalties) < 3:
        return None
    x = np.log10(penalties)
    y = counts.astype(float)
    xr = x.max() - x.min()
    yr = y.max() - y.min()
    if xr == 0 or yr == 0:
        return None
    xn = (x - x.min()) / xr
    yn = (y - y.min()) / yr
    x0, y0, x1, y1 = xn[0], yn[0], xn[-1], yn[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = np.hypot(dx, dy)
    if denom == 0:
        return None
    dist = np.abs(dy * xn - dx * yn + x1 * y0 - y1 * x0) / denom
    interior = np.arange(1, len(dist) - 1)
    idx = int(interior[np.argmax(dist[interior])])
    return {
        "elbow_penalty": float(penalties[idx]),
        "elbow_n_bkps": float(counts[idx]),
        "elbow_score": float(dist[idx]),
    }


def _load_signal(out_dir: Path, tag: str, signal: str) -> Optional[np.ndarray]:
    fpath = out_dir / f"features_{tag}.csv"
    if not fpath.is_file():
        return None
    df = pd.read_csv(fpath)
    if signal == "delta":
        return df["delta"].to_numpy(dtype=np.float64)
    cols = pairwise_feature_columns(df)
    return zscore_matrix(df[cols].to_numpy(dtype=np.float64))


def _fit_algo(signal: np.ndarray, cost: str, args: argparse.Namespace):
    """Fit a changepoint model once; predict at many penalties cheaply.

    rbf uses the fast exact KernelCPD solver (O(n) segment cost via the kernel
    Gram sums) instead of Pelt+CostRbf, which is prohibitively slow at n~5000
    when enumerating many breakpoints. Both optimize the same penalized-kernel
    objective, so the recommended penalty transfers to the driver's Pelt+rbf.
    l2 stays on Pelt (cheap, matches the driver's pairwise path).
    """
    import ruptures as rpt

    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim == 1:
        sig = sig.reshape(-1, 1)
    if cost == "rbf":
        algo = rpt.KernelCPD(kernel="rbf", min_size=args.min_size).fit(sig)
    else:
        algo = rpt.Pelt(model=cost, min_size=args.min_size, jump=args.jump).fit(sig)
    return algo, sig


def _n_bkps(algo, n: int, penalty: float) -> int:
    """Predict breakpoints at a penalty; return count excluding the final index."""
    bkps = [b for b in algo.predict(pen=float(penalty)) if b < n]
    return len(bkps)


def _sweep_signal(
    out_dir: Path,
    tags: List[str],
    signal: str,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    cost = _SIGNALS[signal]["cost"]
    rows: List[Dict[str, float]] = []
    elbows: Dict[str, Dict[str, float]] = {}
    for tag in tags:
        sig = _load_signal(out_dir, tag, signal)
        if sig is None:
            print(f"  SKIP {tag} ({signal}): no features CSV", flush=True)
            continue
        n = sig.shape[0]
        grid = _penalty_grid(n, args)
        ref = float(np.log(max(n, 2)))
        print(f"  fitting {tag} ({signal}, cost={cost}, n={n})...", flush=True)
        algo, _ = _fit_algo(sig, cost, args)  # fit ONCE, predict many penalties
        counts = []
        for pen in grid:
            c = _n_bkps(algo, n, float(pen))
            counts.append(c)
            rows.append(
                {
                    "tag": tag,
                    "signal": signal,
                    "penalty": float(pen),
                    "n_bkps": int(c),
                }
            )
        counts = np.asarray(counts, dtype=float)
        elb = _elbow(grid, counts)
        if elb is not None:
            elb["ref_penalty_logn"] = ref
            elb["heuristic_n_bkps"] = float(_n_bkps(algo, n, ref))
            elbows[tag] = elb
            print(
                f"  {tag} ({signal}): elbow penalty={elb['elbow_penalty']:.3g} "
                f"-> {int(elb['elbow_n_bkps'])} bkps "
                f"(log(n)={ref:.2f} -> {int(elb['heuristic_n_bkps'])} bkps)",
                flush=True,
            )
    return pd.DataFrame(rows), elbows


def _plot_signal(
    df: pd.DataFrame,
    elbows: Dict[str, Dict[str, float]],
    signal: str,
    out_path: Path,
) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    tags = sorted(df["tag"].unique())
    cmap = plt.get_cmap("tab10")
    ref_logn = None
    for i, tag in enumerate(tags):
        sub = df[df["tag"] == tag].sort_values("penalty")
        color = cmap(i % 10)
        ax.plot(
            sub["penalty"], sub["n_bkps"], "o-", ms=4, lw=1.4, color=color, label=tag
        )
        if tag in elbows:
            ep = elbows[tag]["elbow_penalty"]
            eb = elbows[tag]["elbow_n_bkps"]
            ax.scatter([ep], [eb], s=110, facecolors="none", edgecolors=color, linewidths=2.2)
            ref_logn = elbows[tag].get("ref_penalty_logn", ref_logn)
    if ref_logn is not None:
        ax.axvline(
            ref_logn,
            color="gray",
            ls="--",
            lw=1.3,
            label=f"current heuristic log(n)≈{ref_logn:.1f}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Pelt penalty")
    ax.set_ylabel("number of breakpoints")
    ax.set_title(
        f"Penalty sweep — {_SIGNALS[signal]['label']}\n"
        "(open circles = elbow/knee = recommended penalty)"
    )
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path.name}")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    if not out_dir.is_dir():
        print(f"Output dir not found: {out_dir}", file=sys.stderr)
        return 1
    tags = args.tags or _discover_tags(out_dir)
    if not tags:
        print(f"No features_*.csv in {out_dir}", file=sys.stderr)
        return 1
    print(f"Tags: {tags}")

    rec_rows: List[Dict[str, float]] = []
    for signal in args.signals:
        print(f"\n=== Sweeping {signal} ===")
        df, elbows = _sweep_signal(out_dir, tags, signal, args)
        df.to_csv(out_dir / f"penalty_sweep_{signal}.csv", index=False)
        _plot_signal(df, elbows, signal, out_dir / f"penalty_sweep_{signal}.png")
        for tag, elb in elbows.items():
            rec_rows.append(
                {
                    "tag": tag,
                    "signal": signal,
                    "cost_model": _SIGNALS[signal]["cost"],
                    "recommended_penalty": elb["elbow_penalty"],
                    "n_bkps_at_elbow": int(elb["elbow_n_bkps"]),
                    "heuristic_penalty_logn": elb["ref_penalty_logn"],
                    "n_bkps_at_heuristic": int(elb["heuristic_n_bkps"]),
                }
            )

    rec = pd.DataFrame(rec_rows)
    rec_path = out_dir / "penalty_recommendation.csv"
    rec.to_csv(rec_path, index=False)
    print(f"\nWrote {rec_path}")
    if not rec.empty:
        print(rec.to_string(index=False))
        for signal in args.signals:
            sub = rec[rec["signal"] == signal]
            if not sub.empty:
                med = float(np.median(sub["recommended_penalty"]))
                print(
                    f"\nSuggested shared penalty for '{signal}' "
                    f"(median across tags): {med:.3g}  "
                    f"-> run with: --penalty {med:.3g}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
