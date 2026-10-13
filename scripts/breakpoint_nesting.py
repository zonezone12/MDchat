"""Do breakpoints of one feature family fall on breakpoints of another more than chance?

For each cube and family pair, per trajectory: greedy 1-to-1 matches within
±``--tolerance`` frames (the breakpoint-Jaccard matcher), reported as

* ``recall_sparse``: matched pairs / breakpoints in the sparser of the two sets;
* ``jaccard`` and ``jaccard_ceiling`` = min(n_a, n_b) / max(n_a, n_b), the
  largest Jaccard the two counts allow.

Two nulls remove true co-timing but keep counts and spacing: circularly
shifting the second family within the same trajectory, and pairing trajectory
i's first family with trajectory j's second family (j != i, same cube).
``excess_over_chance`` = (observed − null) / (1 − null) for the shift null.

Breakpoints are the own-elbow detections: endpoint from
``OWN_ELBOW_ENDPOINT_DIRS``, ``gsa`` / ``iodine`` from
``gsa_changepoints_matched_{CUBE}``.

Example
-------
python scripts/breakpoint_nesting.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.ChangepointAnalysis.detection import _match_breakpoints
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir

PAIRS = (("endpoint", "gsa"), ("gsa", "iodine"), ("endpoint", "iodine"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "breakpoint_nesting")
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument("--tolerance", type=int, default=50, help="Frames (1 ps)")
    p.add_argument("--n-shift", type=int, default=100, help="Shift replicates per trajectory")
    p.add_argument("--n-cross-pairs", type=int, default=1500, help="Cross-trajectory pairs per cube")
    p.add_argument("--margin", type=int, default=250, help="Smallest circular shift (frames)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _strip(traj_id: str, cube: str) -> str:
    return traj_id[len(cube) + 1 :] if traj_id.startswith(f"{cube}_") else traj_id


def _load(cp_dir: Path, group: str, cube: str) -> dict[str, list[int]]:
    bk = pd.read_csv(cp_dir / "all_breakpoints.csv", usecols=["traj_id", "group", "frame"])
    bk = bk[bk["group"] == group]
    return {_strip(t, cube): sorted(g["frame"].astype(int).tolist()) for t, g in bk.groupby("traj_id")}


def _lengths(cp_dir: Path, group: str, cube: str) -> dict[str, int]:
    seg = pd.read_csv(cp_dir / "all_segment_stats.csv", usecols=["traj_id", "group", "end_frame"])
    seg = seg[seg["group"] == group]
    return {_strip(t, cube): int(v) + 1 for t, v in seg.groupby("traj_id")["end_frame"].max().items()}


def _scores(a: list[int], b: list[int], tolerance: int) -> tuple[float, float, float]:
    m = len(_match_breakpoints(a, b, tolerance))
    small, large = min(len(a), len(b)), max(len(a), len(b))
    return m / small, m / (len(a) + len(b) - m), small / large


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    for cube in args.cubes:
        ep_dir = endpoint_changepoints_dir(args.output_root, cube, own_elbow=True)
        g_dir = args.output_root / f"gsa_changepoints_matched_{cube}"
        if not (ep_dir.is_dir() and g_dir.is_dir()):
            print(f"skip {cube}: missing inputs", flush=True)
            continue
        sets = {
            "endpoint": _load(ep_dir, "endpoint", cube),
            "gsa": _load(g_dir, "gsa", cube),
            "iodine": _load(g_dir, "iodine", cube),
        }
        lengths = _lengths(g_dir, "gsa", cube)
        for fam_a, fam_b in PAIRS:
            A, B = sets[fam_a], sets[fam_b]
            common = sorted(set(A) & set(B))  # trajectories with ≥1 breakpoint in both
            obs = np.array([_scores(A[t], B[t], args.tolerance) for t in common])
            shift_null = []
            for t in common:
                n = lengths[t]
                for s in rng.integers(args.margin, n - args.margin, size=args.n_shift):
                    shifted = sorted(((np.asarray(B[t]) + s) % n).tolist())
                    shift_null.append(_scores(A[t], shifted, args.tolerance)[0])
            pairs = [(i, j) for i in common for j in common if i != j]
            pick = rng.choice(len(pairs), size=min(args.n_cross_pairs, len(pairs)), replace=False)
            cross = np.array([_scores(A[pairs[k][0]], B[pairs[k][1]], args.tolerance) for k in pick])
            null_shift = float(np.mean(shift_null))
            rows.append(
                {
                    "cube": cube,
                    "pair": f"{fam_a}-{fam_b}",
                    "n_traj": len(common),
                    "median_n_a": float(np.median([len(A[t]) for t in common])),
                    "median_n_b": float(np.median([len(B[t]) for t in common])),
                    "recall_sparse": float(obs[:, 0].mean()),
                    "recall_sparse_null_shift": null_shift,
                    "recall_sparse_null_cross": float(cross[:, 0].mean()),
                    "excess_over_chance": (float(obs[:, 0].mean()) - null_shift) / (1 - null_shift),
                    "median_jaccard": float(np.median(obs[:, 1])),
                    "median_jaccard_null_cross": float(np.median(cross[:, 1])),
                    "median_jaccard_ceiling": float(np.median(obs[:, 2])),
                }
            )
        print(f"{cube} done", flush=True)
    if not rows:
        return 1
    out = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "breakpoint_nesting.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}", flush=True)
    pd.set_option("display.width", 250)
    for pair, g in out.groupby("pair", sort=False):
        print(f"\n== {pair}")
        print(g.drop(columns="pair").round(3).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
