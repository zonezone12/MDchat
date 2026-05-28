"""
Per-trajectory worker for parallel metastable-state pipeline.

Segments one trajectory and writes intermediate artifacts for merge_metastable_states.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.metastable_states import (
    extract_representative_positions,
    segment_trajectory,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Segment one trajectory (parallel worker).")
    p.add_argument("--topology", required=True)
    p.add_argument("--trajectory", required=True)
    p.add_argument("--work-dir", required=True, help="Shared directory for per-traj artifacts")
    p.add_argument(
        "--selection",
        default="not water and not resname I and not resname Na+",
    )
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--ref-frame", type=int, default=0)
    p.add_argument("--signal-metric", choices=["rmsd", "rg"], default="rmsd")
    p.add_argument("--method", default="Pelt")
    p.add_argument("--cost-model", default="rbf")
    p.add_argument("--penalty", type=float, default=None)
    p.add_argument("--n-bkps", type=int, default=None)
    p.add_argument("--min-segment-frames", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import MDAnalysis as mda

    traj_path = Path(args.trajectory)
    traj_id = traj_path.stem
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    u = mda.Universe(str(args.topology), str(traj_path))
    segs = segment_trajectory(
        u,
        args.selection,
        traj_id,
        stride=args.stride,
        ref_frame=args.ref_frame,
        signal_metric=args.signal_metric,
        method=args.method,
        cost_model=args.cost_model,
        penalty=args.penalty,
        n_bkps=args.n_bkps,
        min_segment_frames=args.min_segment_frames,
    )
    if not segs:
        print(f"No segments for {traj_id}; skipping.")
        return

    positions, segs = extract_representative_positions(u, args.selection, segs)

    rows = [
        {
            "traj_file": s.traj_id,
            "segment_id": s.segment_id,
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
            "rep_frame": s.rep_frame,
            "n_frames": s.n_frames,
            "rmsd_mean": s.rmsd_mean,
            "rg_mean": s.rg_mean,
            "signal_start": s.signal_start,
            "signal_end": s.signal_end,
        }
        for s in segs
    ]
    seg_csv = work_dir / f"{traj_id}_segments.csv"
    pd.DataFrame(rows).to_csv(seg_csv, index=False)

    pos_npz = work_dir / f"{traj_id}_positions.npz"
    np.savez_compressed(pos_npz, positions=positions, traj_id=traj_id)

    print(f"Wrote {seg_csv} ({len(segs)} segments)")
    print(f"Wrote {pos_npz}")


if __name__ == "__main__":
    main()
