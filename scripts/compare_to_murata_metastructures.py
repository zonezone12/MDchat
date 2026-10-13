"""G5: Murata Table 1 occupancy from motif-RMSD frames.

Labels every motif frame A/B/C1/C2/other from pole π ≥ 6.5 Å and elongated
d1 ≥ 7.0 Å, then compares the three mapped cohorts (BMMpM=1₆, BMHpM=2₆,
BHHpM=3₆). 2₆′ has no cohort here. Optionally cross-tabulates Murata labels
against stored endpoint cluster labels, and writes each cluster's make-up by
elongated-d1 and opened-π counts. Replicas failing the assembly QC rule
(Rg > 15 Å or a lost monomer, from ``--qc-gsa-root/{CUBE}``) are dropped first
unless ``--no-assembly-qc`` is given.

Example
-------
python scripts/compare_to_murata_metastructures.py
python scripts/compare_to_murata_metastructures.py --output-root output --out-dir output/murata_g5

# Cross-tabulate against the clusters of the own-elbow detections
python scripts/compare_to_murata_metastructures.py --own-elbow --out-dir output/murata_g5_own_elbow

# ... at each cube's G3-selected k instead of the stored k=5 cut
python scripts/compare_to_murata_metastructures.py --own-elbow \
    --k-selected output/endpoint_cluster_chemical_k_own_elbow/chemical_k_selected.csv \
    --out-dir output/murata_g5_own_elbow_selected_k
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
from src.ChangepointAnalysis.cluster_k_diagnostics import chosen_k_from_cohort, load_segments_at_k
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.murata_criteria import (
    COHORT_TO_MURATA_SYSTEM,
    METASTRUCTURES,
    occupancy_vs_murata,
)
from src.ChangepointAnalysis.murata_rmsd import (
    CUBE_COLORS,
    _norm_traj_id,
    discover_motif_rmsd_csvs_by_cube,
    filter_non_encapsulated,
    label_motif_metastructures,
    load_motif_rmsd_csvs,
)
from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare motif-frame metastructure occupancy to Murata Table 1."
    )
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "murata_g5")
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
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
    p.add_argument(
        "--own-elbow",
        action="store_true",
        help="Read endpoint clusters from the own-elbow detections (OWN_ELBOW_ENDPOINT_DIRS)",
    )
    k_choice = p.add_mutually_exclusive_group()
    k_choice.add_argument(
        "--k",
        type=int,
        help="Cross-tabulate the Ward cut at this k (default: the stored clustering)",
    )
    k_choice.add_argument(
        "--k-selected",
        type=Path,
        help="chemical_k_selected.csv from select_chemical_k.py; use each cube's selected_k",
    )
    return p.parse_args()


def _assign_endpoint_clusters(
    frames: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    cube: str,
) -> pd.DataFrame:
    out = frames.copy()
    out["_tid"] = out["traj_id"].map(lambda t: _norm_traj_id(t, cube))
    segs = segments.copy()
    if "group" in segs.columns:
        segs = segs[segs["group"].astype(str) == "endpoint"]
    segs["_tid"] = segs["traj_id"].map(lambda t: _norm_traj_id(t, cube))
    out["cluster_label"] = np.nan
    for tid, grp in segs.groupby("_tid"):
        mask = out["_tid"] == tid
        if not mask.any():
            continue
        fr = pd.to_numeric(out.loc[mask, "frame"], errors="coerce").to_numpy()
        lab = np.full(fr.shape, np.nan)
        for rec in grp.itertuples(index=False):
            start = float(rec.start_frame)
            end = float(rec.end_frame)
            m = (fr >= start) & (fr <= end)
            lab[m] = rec.cluster_label
        out.loc[mask, "cluster_label"] = lab
    return out.drop(columns=["_tid"])


def _cluster_composition(joined: pd.DataFrame, *, cube: str, k: int) -> pd.DataFrame:
    """Per endpoint cluster: % of its apo frames by elongated-d1 and opened-π counts.

    ``pct_of_intact_belt`` is the share of the cohort's frames with no elongated
    d1 that fall in the cluster.
    """
    joined = joined[joined["murata_metastructure"].notna()]
    labels = joined["cluster_label"].astype(int)
    intact = labels[pd.to_numeric(joined["murata_d1_n_elongated"], errors="coerce") == 0]
    parts = [
        labels.value_counts().sort_index().rename("n_frames"),
        (100.0 * intact.value_counts(normalize=True)).rename("pct_of_intact_belt"),
    ]
    for col, prefix in (("murata_d1_n_elongated", "pct_d1_elongated"), ("n_open_cation_pi", "pct_open")):
        cap = 2 if prefix == "pct_d1_elongated" else 3
        counts = pd.to_numeric(joined[col], errors="coerce").clip(upper=cap).astype(int)
        share = pd.crosstab(labels, counts, normalize="index").reindex(columns=range(cap + 1), fill_value=0.0)
        share.columns = [f"{prefix}_{c}" if c < cap else f"{prefix}_{c}plus" for c in share.columns]
        parts.append(100.0 * share)
    out = pd.concat(parts, axis=1).rename_axis("cluster_label").reset_index()
    out["pct_of_intact_belt"] = out["pct_of_intact_belt"].fillna(0.0)
    out.insert(0, "k", k)
    out.insert(0, "murata_system", COHORT_TO_MURATA_SYSTEM[cube])
    out.insert(0, "cohort", cube)
    return out


def plot_occupancy_vs_murata(cmp: pd.DataFrame, output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    mapped = cmp[cmp["murata_system"].astype(str) != ""].copy()
    mapped = mapped[mapped["guest_filter"] == "frames"]
    cohorts = [c for c in KNOWN_CUBES if c in set(mapped["cohort"])]
    if not cohorts:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.0 * len(cohorts), 3.8), squeeze=False)
    x = np.arange(len(METASTRUCTURES))
    width = 0.36
    for ax, cube in zip(axes[0], cohorts):
        sub = mapped[mapped["cohort"] == cube].set_index("metastructure")
        ours = [float(sub.loc[m, "our_percent"]) for m in METASTRUCTURES]
        refs = [float(sub.loc[m, "murata_percent"]) for m in METASTRUCTURES]
        ax.bar(x - width / 2, ours, width, color=CUBE_COLORS.get(cube, "0.4"), label="this work")
        ax.bar(x + width / 2, refs, width, color="0.65", label="Murata Table 1")
        system = str(sub["murata_system"].iloc[0])
        ax.set_xticks(x)
        ax.set_xticklabels(METASTRUCTURES)
        ax.set_ylim(0, 100)
        ax.set_ylabel("occupancy (%)")
        ax.set_title(f"{cube}  ({system})")
        ax.legend(fontsize=8)
    fig.suptitle("Non-encapsulated frames vs Murata Table 1", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    by_cube = discover_motif_rmsd_csvs_by_cube(args.output_root, cubes=args.cubes)
    if not by_cube:
        print(f"No *_motif_rmsd.csv under {args.output_root}", flush=True)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_k: dict[str, int] = {}
    if args.k_selected is not None:
        sel = pd.read_csv(args.k_selected)
        selected_k = dict(zip(sel["cohort"].astype(str), sel["selected_k"].astype(int)))
    occupancy_rows: list[pd.DataFrame] = []
    xtab_rows: list[pd.DataFrame] = []
    other_rows: list[pd.DataFrame] = []
    composition_rows: list[pd.DataFrame] = []
    qc_rows: list[pd.DataFrame] = []

    for cube, paths in by_cube.items():
        print(f"labeling {cube}: {len(paths)} CSVs …", flush=True)
        df = load_motif_rmsd_csvs(paths)
        if not args.no_assembly_qc:
            gsa_dir = args.qc_gsa_root / cube
            if gsa_dir.is_dir():
                qc = scan_assembly_qc_dir(gsa_dir)
                qc_rows.append(qc)
                failed = assembly_qc_failures(qc)
                if failed:
                    print(f"  assembly QC excludes {', '.join(failed)}", flush=True)
                    df = drop_trajectories(df, failed)
            else:
                print(f"  assembly QC skipped: {gsa_dir} not found", flush=True)
        df = label_motif_metastructures(df)
        filters = [("all", "all")]
        if "n_guest_inside_cavity" in df.columns:
            filters.extend([("frames", "frames"), ("never", "never")])
        for filt, name in filters:
            labeled = filter_non_encapsulated(df, traj_filter=filt)
            occupancy_rows.append(
                occupancy_vs_murata(
                    labeled["murata_metastructure"],
                    cohort=cube,
                    guest_filter=name,
                )
            )
        cp_dir = endpoint_changepoints_dir(args.output_root, cube, own_elbow=args.own_elbow)
        if cube not in COHORT_TO_MURATA_SYSTEM or not (cp_dir / "clusters" / "endpoint").is_dir():
            continue
        if args.k_selected is not None and cube not in selected_k:
            print(f"  {cube}: not in {args.k_selected}, using the stored clustering", flush=True)
        k = args.k or selected_k.get(cube) or chosen_k_from_cohort(cp_dir, group="endpoint")
        segs = load_segments_at_k(cp_dir, k, group="endpoint")
        if segs.empty:
            print(f"  {cube}: no endpoint clustering at k={k}", flush=True)
            continue
        apo = filter_non_encapsulated(df, traj_filter="frames")
        joined = _assign_endpoint_clusters(apo, segs, cube=cube)
        joined = joined.dropna(subset=["cluster_label"])
        if joined.empty:
            continue
        xtab = pd.crosstab(
            joined["murata_metastructure"],
            joined["cluster_label"].astype(int),
            normalize="index",
        ) * 100.0
        xtab = xtab.reindex(METASTRUCTURES)
        xtab.insert(0, "cohort", cube)
        xtab.insert(1, "murata_system", COHORT_TO_MURATA_SYSTEM[cube])
        xtab.insert(2, "k", k)
        xtab_rows.append(xtab.reset_index())
        composition_rows.append(_cluster_composition(joined, cube=cube, k=k))
        other = joined[joined["murata_metastructure"] == "other"]
        if not other.empty:
            counts = other["cluster_label"].astype(int).value_counts(normalize=True).sort_index()
            other_rows.append(
                pd.DataFrame(
                    {
                        "cohort": cube,
                        "murata_system": COHORT_TO_MURATA_SYSTEM[cube],
                        "k": k,
                        "cluster_label": counts.index.astype(int),
                        "percent_of_other": 100.0 * counts.to_numpy(),
                        "n_other_frames": int(len(other)),
                    }
                )
            )

    occ = pd.concat(occupancy_rows, ignore_index=True)
    occ_path = args.out_dir / "murata_table1_occupancy.csv"
    occ.to_csv(occ_path, index=False)
    print(f"wrote {occ_path}", flush=True)
    if qc_rows:
        qc_path = args.out_dir / "assembly_qc.csv"
        pd.concat(qc_rows, ignore_index=True).to_csv(qc_path, index=False)
        print(f"wrote {qc_path}", flush=True)
    mapped = occ[(occ["guest_filter"] == "frames") & (occ["murata_system"].astype(str) != "")]
    if not mapped.empty:
        show = mapped.pivot_table(
            index=["cohort", "murata_system"],
            columns="metastructure",
            values="our_percent",
        ).reindex(columns=list(METASTRUCTURES))
        ref = mapped.pivot_table(
            index=["cohort", "murata_system"],
            columns="metastructure",
            values="murata_percent",
        ).reindex(columns=list(METASTRUCTURES))

        def _ascii_index(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            out.index = pd.MultiIndex.from_tuples(
                [
                    (str(a), str(b).replace("₆", "6").replace("′", "'"))
                    for a, b in out.index
                ],
                names=out.index.names,
            )
            return out

        print("\nOur occupancy % (apo frames):\n", _ascii_index(show).round(1).to_string(), flush=True)
        print("\nMurata Table 1 %:\n", _ascii_index(ref).round(1).to_string(), flush=True)
        plot_path = plot_occupancy_vs_murata(
            occ, args.out_dir / "plots" / "occupancy_vs_murata_table1.png"
        )
        print(f"wrote {plot_path}", flush=True)
    if xtab_rows:
        xtab_path = args.out_dir / "murata_vs_endpoint_cluster.csv"
        pd.concat(xtab_rows, ignore_index=True).to_csv(xtab_path, index=False)
        print(f"wrote {xtab_path}", flush=True)
    if other_rows:
        other_path = args.out_dir / "murata_other_vs_endpoint_cluster.csv"
        pd.concat(other_rows, ignore_index=True).to_csv(other_path, index=False)
        print(f"wrote {other_path}", flush=True)
    if composition_rows:
        comp_path = args.out_dir / "endpoint_cluster_composition.csv"
        pd.concat(composition_rows, ignore_index=True).to_csv(comp_path, index=False)
        print(f"wrote {comp_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
