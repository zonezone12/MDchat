"""Murata-style RMSD of non-encapsulated nanocube conformations.

Uses stored ``assembly_rmsd_to_ref`` (cube vs each replica's initial NVT frame)
and drops frames with an iodide inside the cavity. Marks the paper peaks at
1.0 Å (all cation–π closed), 1.5 Å (one unit open), and 2.7 Å (two units open).

Does not re-read trajectories. Source: ``output/gsa_features_step1``.

Example
-------
python scripts/plot_murata_rmsd.py
python scripts/plot_murata_rmsd.py --traj-filter never
python scripts/plot_murata_rmsd.py --open-dir output/murata_criteria
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.murata_rmsd import (
    attach_open_cation_pi,
    drop_reference_rmsd,
    filter_non_encapsulated,
    load_gsa_rmsd_frames,
    plot_murata_rmsd_distributions,
    plot_murata_rmsd_overlay,
    plot_rmsd_apo_vs_all,
    plot_rmsd_by_open_count,
    summarize_apo_rmsd,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot RMSD of nanocube conformations that do not contain an "
            "encapsulated iodide, with Murata 1.0 / 1.5 / 2.7 Å peak markers."
        )
    )
    p.add_argument(
        "--features-dir",
        type=Path,
        default=ROOT / "output" / "gsa_features_step1",
        help="Directory of *_gsa_features.csv (default: output/gsa_features_step1)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "murata_rmsd",
        help="Output directory (default: output/murata_rmsd)",
    )
    p.add_argument(
        "--cubes",
        nargs="*",
        default=list(KNOWN_CUBES),
        help="Cohort names (default: all six B* cubes)",
    )
    p.add_argument(
        "--traj-filter",
        choices=("frames", "never", "all"),
        default="frames",
        help=(
            "frames: drop encapsulated frames. never: replicas that never "
            "encapsulate. all: ignore iodide (still written as a companion plot)."
        ),
    )
    p.add_argument(
        "--open-dir",
        type=Path,
        default=None,
        help=(
            "Optional murata_criteria output root. If {cube}/murata_cation_pi_frames.csv "
            "exists, overlay RMSD by opened cation–π count."
        ),
    )
    p.add_argument(
        "--write-frames",
        action="store_true",
        help="Write the filtered apo RMSD rows (can be large).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.features_dir.is_dir():
        print(f"Missing features dir: {args.features_dir}", flush=True)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading RMSD from {args.features_dir} …", flush=True)
    frames = load_gsa_rmsd_frames(args.features_dir, cubes=args.cubes)
    if not frames:
        print("No GSA feature CSVs with assembly_rmsd_to_ref found.", flush=True)
        return 1

    if args.open_dir:
        for cube in list(frames):
            path = args.open_dir / cube / "murata_cation_pi_frames.csv"
            if not path.is_file():
                print(f"  no cation–π frames for {cube} at {path}", flush=True)
                continue
            open_df = pd.read_csv(
                path,
                usecols=lambda c: c
                in {"traj_id", "frame", "n_open_cation_pi", "murata_metastructure"},
            )
            frames[cube] = attach_open_cation_pi(frames[cube], open_df, cube=cube)
            n_join = int(frames[cube]["n_open_cation_pi"].notna().sum())
            print(f"  joined n_open_cation_pi for {cube}: {n_join} frames", flush=True)

    summaries = [
        summarize_apo_rmsd(df, cohort=cube, traj_filter=args.traj_filter)
        for cube, df in frames.items()
    ]
    summaries.extend(
        summarize_apo_rmsd(df, cohort=cube, traj_filter="all")
        for cube, df in frames.items()
        if args.traj_filter != "all"
    )
    summary = pd.DataFrame(summaries)
    summary_path = args.out_dir / "murata_rmsd_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)

    panels = plot_murata_rmsd_distributions(
        frames, args.out_dir / "plots" / "murata_rmsd_by_cube.png", traj_filter=args.traj_filter
    )
    overlay = plot_murata_rmsd_overlay(
        frames, args.out_dir / "plots" / "murata_rmsd_overlay.png", traj_filter=args.traj_filter
    )
    print(f"wrote {panels}", flush=True)
    print(f"wrote {overlay}", flush=True)

    if args.traj_filter != "all":
        all_panels = plot_murata_rmsd_distributions(
            frames,
            args.out_dir / "plots" / "murata_rmsd_by_cube_all_frames.png",
            traj_filter="all",
        )
        all_overlay = plot_murata_rmsd_overlay(
            frames,
            args.out_dir / "plots" / "murata_rmsd_overlay_all_frames.png",
            traj_filter="all",
        )
        compare = plot_rmsd_apo_vs_all(
            frames, args.out_dir / "plots" / "murata_rmsd_apo_vs_all.png"
        )
        print(f"wrote {all_panels}", flush=True)
        print(f"wrote {all_overlay}", flush=True)
        print(f"wrote {compare}", flush=True)

    if any("n_open_cation_pi" in df.columns for df in frames.values()):
        try:
            by_open = plot_rmsd_by_open_count(
                frames,
                args.out_dir / "plots" / "murata_rmsd_by_open_count.png",
                traj_filter=args.traj_filter,
            )
            print(f"wrote {by_open}", flush=True)
        except ValueError as exc:
            print(f"skip open-count plot: {exc}", flush=True)

    if args.write_frames:
        apo_rows = [
            drop_reference_rmsd(
                filter_non_encapsulated(df, traj_filter=args.traj_filter)
            )
            for df in frames.values()
        ]
        apo_path = args.out_dir / "murata_rmsd_apo_frames.csv"
        pd.concat(apo_rows, ignore_index=True).to_csv(apo_path, index=False)
        print(f"wrote {apo_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
