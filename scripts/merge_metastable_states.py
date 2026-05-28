"""
Merge per-trajectory segment artifacts and run hierarchical clustering.
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
    Segment,
    assign_cluster_labels,
    cluster_metastable_states,
    compute_pairwise_rmsd_matrix,
    format_cluster_summary,
    plot_dendrogram,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge parallel segment outputs and cluster.")
    p.add_argument("--work-dir", required=True, help="Directory with *_segments.csv and *_positions.npz")
    p.add_argument("--output-dir", required=True, help="Final results directory")
    p.add_argument("--linkage", default="ward")
    p.add_argument("--rmsd-cutoff", type=float, default=None)
    p.add_argument("--k-max", type=int, default=10)
    return p.parse_args()


def _load_segments(work_dir: Path) -> tuple[list[Segment], list[np.ndarray], list[str]]:
    segments: list[Segment] = []
    positions_list: list[np.ndarray] = []
    leaf_labels: list[str] = []

    for seg_csv in sorted(work_dir.glob("*_segments.csv")):
        traj_id = seg_csv.name.replace("_segments.csv", "")
        pos_npz = work_dir / f"{traj_id}_positions.npz"
        if not pos_npz.exists():
            print(f"Warning: missing {pos_npz}, skipping {traj_id}")
            continue

        df = pd.read_csv(seg_csv)
        pos = np.load(pos_npz)["positions"]

        for i, row in df.iterrows():
            segments.append(
                Segment(
                    traj_id=str(row["traj_file"]),
                    segment_id=int(row["segment_id"]),
                    start_frame=int(row["start_frame"]),
                    end_frame=int(row["end_frame"]),
                    rep_frame=int(row["rep_frame"]),
                    rmsd_mean=float(row["rmsd_mean"]),
                    rg_mean=float(row["rg_mean"]),
                    signal_start=int(row.get("signal_start", 0)),
                    signal_end=int(row.get("signal_end", 0)),
                )
            )
            leaf_labels.append(f"{row['traj_file']}:seg{row['segment_id']}")

        positions_list.append(pos)

    if not segments:
        raise RuntimeError(f"No segment data found in {work_dir}")

    return segments, positions_list, leaf_labels


def main() -> None:
    args = parse_args()
    work_dir = Path(args.work_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    segments, positions_list, leaf_labels = _load_segments(work_dir)
    positions = np.vstack(positions_list)

    dist_mat = compute_pairwise_rmsd_matrix(positions)
    clustering = cluster_metastable_states(
        dist_mat,
        method=args.linkage,
        rmsd_cutoff=args.rmsd_cutoff,
        k_max=args.k_max,
    )
    assign_cluster_labels(segments, clustering)

    seg_rows = [
        {
            "traj_file": s.traj_id,
            "segment_id": s.segment_id,
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
            "rep_frame": s.rep_frame,
            "n_frames": s.n_frames,
            "rmsd_mean": s.rmsd_mean,
            "rg_mean": s.rg_mean,
            "cluster_label": s.cluster_label,
        }
        for s in segments
    ]
    seg_csv = out_dir / "segments_all.csv"
    pd.DataFrame(seg_rows).to_csv(seg_csv, index=False)

    dist_df = pd.DataFrame(dist_mat, index=leaf_labels, columns=leaf_labels)
    dist_csv = out_dir / "distance_matrix.csv"
    dist_df.to_csv(dist_csv)

    dendro_path = out_dir / "dendrogram.png"
    plot_dendrogram(
        clustering.linkage_matrix,
        leaf_labels,
        dendro_path,
        rmsd_cutoff=clustering.cutoff_distance if args.rmsd_cutoff else None,
    )

    summary_text = format_cluster_summary(segments, clustering)
    summary_path = out_dir / "cluster_summary.txt"
    summary_path.write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"\nWrote {seg_csv}, {dist_csv}, {dendro_path}, {summary_path}")


if __name__ == "__main__":
    main()
