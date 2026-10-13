"""G5: how robust are our Murata A/B/C1/C2/other labels in each mapped cohort?

For apo motif frames of BMMpM (1₆), BMHpM (2₆) and BHHpM (3₆) this writes:

* ``open_by_elongated.csv``: joint distribution of opened cation–π count and
  elongated d1 count, which is what the labels are built from;
* ``cutoff_sensitivity.csv``: label occupancy over a grid of the π(pole) open
  cutoff (6.5 Å) and the d1 elongated cutoff (7.0 Å);
* ``c2_near_cutoff.csv``: how close C2 frames sit to becoming "other", i.e. the
  third-widest π(pole) relative to the open cutoff and the second-longest d1
  relative to the elongated cutoff;
* ``occupancy_bootstrap.csv``: occupancy with a trajectory-bootstrap 95% interval;
* ``open_count_switching.csv``: how often the opened-π count moves between two
  and three or more, and how long the ≥3-open episodes last. This one uses all
  frames in time order, not only apo frames.

Replicas failing the assembly QC rule are dropped first unless
``--no-assembly-qc`` is given.

Example
-------
python scripts/murata_label_diagnostics.py
python scripts/murata_label_diagnostics.py --out-dir output/murata_g5/label_diagnostics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.ChangepointAnalysis.assembly_qc import (
    assembly_qc_failures,
    drop_trajectories,
    scan_assembly_qc_dir,
)
from src.ChangepointAnalysis.murata_criteria import (
    CATION_PI_OPEN_LO,
    COHORT_TO_MURATA_SYSTEM,
    METASTRUCTURES,
    MURATA_TABLE1_PCT,
    classify_murata_metastructure_columns,
)
from src.ChangepointAnalysis.murata_d1 import D1_ELONGATED_LO
from src.ChangepointAnalysis.murata_rmsd import (
    discover_motif_rmsd_csvs_by_cube,
    filter_non_encapsulated,
    label_motif_metastructures,
    load_motif_rmsd_csvs,
    unit_series_columns,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Murata label diagnostics for the mapped cohorts.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "murata_g5" / "label_diagnostics",
    )
    p.add_argument("--cubes", nargs="*", default=list(COHORT_TO_MURATA_SYSTEM))
    p.add_argument("--pi-cutoffs", type=float, nargs="+", default=[6.0, 6.25, 6.5, 6.75, 7.0])
    p.add_argument("--d1-cutoffs", type=float, nargs="+", default=[6.5, 6.75, 7.0, 7.25, 7.5])
    p.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Distance below a cutoff (Å) that counts as near it (default: 0.5)",
    )
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--qc-gsa-root",
        type=Path,
        default=ROOT / "output" / "gsa_features_by_cube",
        help="Root of per-cube *_gsa_features.csv folders used by the assembly QC rule",
    )
    p.add_argument(
        "--no-assembly-qc",
        action="store_true",
        help="Keep replicas that fail the assembly QC rule",
    )
    return p.parse_args()


def _sorted_desc(values: np.ndarray) -> np.ndarray:
    filled = np.where(np.isfinite(values), values, -np.inf)
    return -np.sort(-filled, axis=1)


def _percentages(labels: np.ndarray) -> dict[str, float]:
    counts = pd.Series(labels).value_counts()
    n = max(int(counts.sum()), 1)
    return {m: 100.0 * float(counts.get(m, 0)) / n for m in METASTRUCTURES}


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def _open_count_switching(frames: pd.DataFrame) -> dict[str, float]:
    ordered = frames.dropna(subset=["n_open_cation_pi"]).sort_values(["traj_id", "frame"])
    step_ps = float(ordered.groupby("traj_id")["time_ps"].diff().median())
    runs3, runs2, n_switch = [], [], 0
    for _, grp in ordered.groupby("traj_id"):
        n_open = grp["n_open_cation_pi"].to_numpy(dtype=int)
        runs3.append(_run_lengths(n_open >= 3))
        runs2.append(_run_lengths(n_open == 2))
        a, b = n_open[:-1], n_open[1:]
        n_switch += int((((a == 2) & (b >= 3)) | ((a >= 3) & (b == 2))).sum())
    r3 = np.concatenate(runs3) * step_ps
    r2 = np.concatenate(runs2) * step_ps
    return {
        "frame_step_ps": step_ps,
        "n_frames": len(ordered),
        "n_ge3_open_episodes": len(r3),
        "ge3_open_median_ps": float(np.median(r3)) if len(r3) else np.nan,
        "ge3_open_mean_ps": float(r3.mean()) if len(r3) else np.nan,
        "pct_ge3_open_le_2_frames": 100.0 * float((r3 <= 2 * step_ps).mean()) if len(r3) else np.nan,
        "two_open_median_ps": float(np.median(r2)) if len(r2) else np.nan,
        "two_open_mean_ps": float(r2.mean()) if len(r2) else np.nan,
        "switches_2_ge3_per_ns": n_switch / (len(ordered) * step_ps / 1000.0),
    }


def main() -> int:
    args = parse_args()
    by_cube = discover_motif_rmsd_csvs_by_cube(args.output_root, cubes=args.cubes)
    if not by_cube:
        print(f"No *_motif_rmsd.csv under {args.output_root}", flush=True)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    joint_rows, sens_rows, near_rows, boot_rows, switch_rows = [], [], [], [], []

    for cube, paths in by_cube.items():
        system = COHORT_TO_MURATA_SYSTEM.get(cube, "")
        print(f"{cube} ({system}): {len(paths)} CSVs", flush=True)
        df = load_motif_rmsd_csvs(paths)
        if not args.no_assembly_qc:
            gsa_dir = args.qc_gsa_root / cube
            if gsa_dir.is_dir():
                failed = assembly_qc_failures(scan_assembly_qc_dir(gsa_dir))
                if failed:
                    print(f"  assembly QC excludes {', '.join(failed)}", flush=True)
                    df = drop_trajectories(df, failed)
            else:
                print(f"  assembly QC skipped: {gsa_dir} not found", flush=True)
        frames = label_motif_metastructures(df)
        switch_rows.append({"cohort": cube, "murata_system": system, **_open_count_switching(frames)})
        apo = filter_non_encapsulated(frames, traj_filter="frames")
        apo = apo.dropna(subset=["murata_metastructure"]).reset_index(drop=True)
        pole = apo[unit_series_columns(apo, "cation_pi_pole_e")].to_numpy(dtype=float)
        d1 = apo[unit_series_columns(apo, "equator_d1_e")].to_numpy(dtype=float)
        labels = apo["murata_metastructure"].to_numpy()
        n_apo = len(apo)

        joint = (
            apo.groupby(["n_open_cation_pi", "murata_d1_n_elongated"])
            .size()
            .rename("n_frames")
            .reset_index()
            .rename(columns={"murata_d1_n_elongated": "n_d1_elongated"})
        )
        joint["percent"] = 100.0 * joint["n_frames"] / n_apo
        joint.insert(0, "murata_system", system)
        joint.insert(0, "cohort", cube)
        joint_rows.append(joint)

        for pi_lo in args.pi_cutoffs:
            n_open = (pole >= pi_lo).sum(axis=1)
            for d1_lo in args.d1_cutoffs:
                relabeled = classify_murata_metastructure_columns(n_open, (d1 >= d1_lo).sum(axis=1))
                sens_rows.append(
                    {
                        "cohort": cube,
                        "murata_system": system,
                        "pi_open_lo": pi_lo,
                        "d1_elongated_lo": d1_lo,
                        "is_published_cutoff": bool(
                            np.isclose(pi_lo, CATION_PI_OPEN_LO) and np.isclose(d1_lo, D1_ELONGATED_LO)
                        ),
                        **_percentages(relabeled),
                    }
                )

        c2 = labels == "C2"
        third_pi = _sorted_desc(pole)[c2, 2]
        second_d1 = _sorted_desc(d1)[c2, 1]
        pi_near = (third_pi >= CATION_PI_OPEN_LO - args.margin) & (third_pi < CATION_PI_OPEN_LO)
        d1_near = (second_d1 >= D1_ELONGATED_LO - args.margin) & (second_d1 < D1_ELONGATED_LO)
        row = {
            "cohort": cube,
            "murata_system": system,
            "n_apo_frames": n_apo,
            "n_c2_frames": int(c2.sum()),
            "margin_angstrom": args.margin,
            "pct_c2_third_pi_near_open": 100.0 * float(pi_near.mean()) if c2.any() else np.nan,
            "pct_c2_second_d1_near_elongated": 100.0 * float(d1_near.mean()) if c2.any() else np.nan,
            "pct_c2_either_near": 100.0 * float((pi_near | d1_near).mean()) if c2.any() else np.nan,
        }
        for q in (10, 25, 50, 75, 90):
            row[f"c2_third_pi_q{q}"] = float(np.quantile(third_pi, q / 100)) if c2.any() else np.nan
        for q in (10, 25, 50, 75, 90):
            row[f"c2_second_d1_q{q}"] = float(np.quantile(second_d1, q / 100)) if c2.any() else np.nan
        near_rows.append(row)

        per_traj = (
            pd.crosstab(apo["traj_id"], apo["murata_metastructure"])
            .reindex(columns=list(METASTRUCTURES), fill_value=0)
            .to_numpy(dtype=float)
        )
        picks = rng.integers(0, len(per_traj), size=(args.n_boot, len(per_traj)))
        sums = per_traj[picks].sum(axis=1)
        boot_pct = 100.0 * sums / sums.sum(axis=1, keepdims=True)
        ours = _percentages(labels)
        ref = MURATA_TABLE1_PCT.get(system, {})
        for i, m in enumerate(METASTRUCTURES):
            boot_rows.append(
                {
                    "cohort": cube,
                    "murata_system": system,
                    "metastructure": m,
                    "our_percent": ours[m],
                    "ci95_lo": float(np.percentile(boot_pct[:, i], 2.5)),
                    "ci95_hi": float(np.percentile(boot_pct[:, i], 97.5)),
                    "murata_percent": ref.get(m, np.nan),
                    "n_trajectories": len(per_traj),
                    "n_boot": args.n_boot,
                }
            )

    tables = {
        "open_by_elongated.csv": pd.concat(joint_rows, ignore_index=True),
        "cutoff_sensitivity.csv": pd.DataFrame(sens_rows),
        "c2_near_cutoff.csv": pd.DataFrame(near_rows),
        "occupancy_bootstrap.csv": pd.DataFrame(boot_rows),
        "open_count_switching.csv": pd.DataFrame(switch_rows),
    }
    for name, df in tables.items():
        path = args.out_dir / name
        df.to_csv(path, index=False)
        print(f"wrote {path}", flush=True)

    boot = tables["occupancy_bootstrap.csv"]
    show = boot.assign(
        ours=boot["our_percent"].round(1).astype(str)
        + " ["
        + boot["ci95_lo"].round(1).astype(str)
        + ", "
        + boot["ci95_hi"].round(1).astype(str)
        + "]"
    ).pivot(index="cohort", columns="metastructure", values="ours")
    print("\nApo occupancy % [trajectory-bootstrap 95% CI]:")
    print(show.reindex(columns=list(METASTRUCTURES)).to_string(), flush=True)
    print("\nC2 frames near a cutoff:")
    print(tables["c2_near_cutoff.csv"].round(2).to_string(index=False), flush=True)
    print("\nOpened-π count switching (all frames):")
    print(tables["open_count_switching.csv"].round(2).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
