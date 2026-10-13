"""Automatic endpoint k without reference labels: chemical gate, then held-out cage geometry.

Each k from the G3 chemical scan must first pass ``chemical_separation`` (the
top site pairs mark ≥2 clusters with distinct contacts or open/closed signs).
Among those, k is scored by how much variance of the GSA cage-geometry
descriptors (never used to build the endpoint clusters) its clusters explain on
left-out replicas (out-of-bag R², replica bootstrap). The pick is the smallest
gated k whose out-of-bag R² is within one bootstrap SE of the best.

Murata A/B/C1/C2/other labels play no part in the choice. If the G3 NMI table
exists, its NMI at the selected k is copied in as an evaluation column only.

Writes to ``--out-dir``:

* ``physical_k_by_k.csv``: gate and geometry scores per cohort and k;
* ``physical_k_selected.csv``: selected k per cohort, with the gate-only k and,
  for comparison, the Murata-NMI pick and NMI at the selected k;
* ``plots/geometry_r2_vs_k.png``.

Example
-------
python scripts/select_physical_k.py --own-elbow \\
    --chem-dir output/endpoint_cluster_chemical_k_own_elbow \\
    --out-dir output/endpoint_cluster_physical_k_own_elbow
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.cluster_k_diagnostics import (
    _chemical_flag,
    load_held_out_geometry_frames,
    score_held_out_geometry_by_k,
    select_k_one_se,
)
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.murata_rmsd import CUBE_COLORS
from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select endpoint k by held-out cage geometry among chemically valid k."
    )
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--chem-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_chemical_k_own_elbow",
        help="Directory with chemical_separation_by_k.csv (and optionally the NMI table)",
    )
    p.add_argument(
        "--gsa-features-root",
        type=Path,
        default=ROOT / "output" / "gsa_features_by_cube",
        help="Per-cube folders of *_gsa_features.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow",
    )
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument("--k-min", type=int, default=None, help="Default: the chemical scan's smallest k")
    p.add_argument("--k-max", type=int, default=None, help="Default: the chemical scan's largest k")
    p.add_argument("--stride", type=int, default=5, help="Frame stride of the endpoint signal")
    p.add_argument("--n-bootstrap", type=int, default=500, help="Replica resamples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--own-elbow", action="store_true", help="Score the own-elbow clusters")
    return p.parse_args()


def plot_r2_vs_k(long_df: pd.DataFrame, picked: pd.DataFrame, output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = dict(zip(picked["cohort"].astype(str), picked["selected_k"]))
    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    for cube, sub in long_df.groupby("cohort"):
        sub = sub.sort_values("k")
        color = CUBE_COLORS.get(str(cube), "0.35")
        ax.errorbar(
            sub["k"], sub["geometry_r2_oob"], yerr=sub["geometry_r2_oob_se"], fmt="o-",
            color=color, lw=1.5, ms=5, capsize=2, label=cube,
        )
        passing = sub[_chemical_flag(sub["chemical_separation"])]
        ax.scatter(
            passing["k"], passing["geometry_r2_oob"], s=64, facecolors="none",
            edgecolors=color, lw=1.4, zorder=3,
        )
        pick = sub[sub["k"] == selected.get(str(cube))]
        ax.scatter(pick["k"], pick["geometry_r2_oob"], marker="*", s=180, color=color, zorder=4)
    ax.set_xlabel("k")
    ax.set_ylabel("Out-of-bag R² of held-out cage geometry")
    ax.set_xticks(sorted({int(k) for k in long_df["k"]}))
    ax.set_title("Open circles: chemical_separation=True; stars: selected k (one-SE). Bars: replica SE.")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    chem_path = args.chem_dir / "chemical_separation_by_k.csv"
    if not chem_path.is_file():
        print(f"Missing {chem_path} (run compare_endpoint_cluster_k.py --chemical-scan)", flush=True)
        return 1
    chem = pd.read_csv(chem_path)
    k_min = args.k_min if args.k_min is not None else int(chem["k"].min())
    k_max = args.k_max if args.k_max is not None else int(chem["k"].max())

    parts: list[pd.DataFrame] = []
    for cube in args.cubes:
        cp_dir = endpoint_changepoints_dir(args.output_root, cube, own_elbow=args.own_elbow)
        gsa_dir = args.gsa_features_root / cube
        if not cp_dir.is_dir() or not gsa_dir.is_dir():
            print(f"skip {cube}: missing {cp_dir} or {gsa_dir}", flush=True)
            continue
        print(f"held-out geometry {cube} ...", flush=True)
        frames = load_held_out_geometry_frames(gsa_dir, stride=args.stride)
        parts.append(
            score_held_out_geometry_by_k(
                cp_dir, frames, cube=cube, k_min=k_min, k_max=k_max,
                n_bootstrap=args.n_bootstrap, random_state=args.seed,
            )
        )
    if not parts:
        print("No cohorts scored.", flush=True)
        return 1
    keep = ["cohort", "k", "chemical_separation", "fail_reason", "silhouette", "cluster_sizes"]
    merged = chem[[c for c in keep if c in chem.columns]].merge(
        pd.concat(parts, ignore_index=True), on=["cohort", "k"], how="inner"
    )
    picked = select_k_one_se(merged, score_col="geometry_r2_oob", se_col="geometry_r2_oob_se")

    gated = merged[_chemical_flag(merged["chemical_separation"])]
    picked["gate_only_k"] = picked["cohort"].map(gated.groupby("cohort")["k"].min())
    nmi_path = args.chem_dir / "chemical_k_murata_nmi_by_k.csv"
    sel_path = args.chem_dir / "chemical_k_selected.csv"
    if nmi_path.is_file():  # evaluation only; never used to choose k
        nmi = pd.read_csv(nmi_path).set_index(["cohort", "k"])["murata_nmi"]
        picked["murata_nmi_at_selected_k"] = [
            nmi.get((c, k)) for c, k in zip(picked["cohort"], picked["selected_k"])
        ]
    if sel_path.is_file():
        picked["murata_nmi_selected_k"] = picked["cohort"].map(
            pd.read_csv(sel_path).set_index("cohort")["selected_k"]
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("physical_k_by_k.csv", merged), ("physical_k_selected.csv", picked)):
        path = args.out_dir / name
        table.to_csv(path, index=False)
        print(f"wrote {path}", flush=True)
    print(picked.to_string(index=False), flush=True)
    plot_r2_vs_k(merged, picked, args.out_dir / "plots" / "geometry_r2_vs_k.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
