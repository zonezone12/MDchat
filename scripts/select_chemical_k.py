"""Automatic endpoint k: chemical_separation first, then max Murata NMI.

Silhouette is not used to pick k. Reads the G3 chemical scan and motif-RMSD
frames (A/B/C1/C2/other), scores each k=2..10 by NMI vs those independent
labels, and writes ``selected_k`` per cohort.

Example
-------
python scripts/select_chemical_k.py
python scripts/select_chemical_k.py --chem-dir output/endpoint_cluster_chemical_k \\
    --output-root output --out-dir output/endpoint_cluster_chemical_k
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.cluster_k_diagnostics import (
    score_murata_nmi_by_k,
    select_k_by_chemical_information,
)
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.murata_rmsd import (
    CUBE_COLORS,
    discover_motif_rmsd_csvs_by_cube,
    label_motif_metastructures,
    load_motif_rmsd_csvs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select endpoint k by Murata-label NMI among chemically valid k."
    )
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--chem-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_chemical_k",
        help="Directory with chemical_separation_by_k.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: --chem-dir",
    )
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument("--k-min", type=int, default=2)
    p.add_argument("--k-max", type=int, default=10)
    return p.parse_args()


def plot_nmi_vs_k(long_df: pd.DataFrame, output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    for cube, sub in long_df.groupby("cohort"):
        sub = sub.sort_values("k")
        color = CUBE_COLORS.get(str(cube), "0.35")
        ax.plot(sub["k"], sub["murata_nmi"], "o-", color=color, lw=1.5, ms=5, label=cube)
        flag = sub["chemical_separation"]
        if flag.dtype == object or pd.api.types.is_string_dtype(flag):
            passing = sub[flag.astype(str).str.lower().isin(["true", "1", "yes"])]
        else:
            passing = sub[flag.astype(bool)]
        if not passing.empty:
            ax.scatter(
                passing["k"],
                passing["murata_nmi"],
                s=64,
                facecolors="none",
                edgecolors=color,
                lw=1.4,
                zorder=3,
            )
    ax.set_xlabel("k")
    ax.set_ylabel("NMI (cluster vs Murata A/B/C1/C2/other)")
    ax.set_xticks(sorted({int(k) for k in long_df["k"]}))
    ax.set_title("Open circles: chemical_separation=True. k is not chosen by silhouette.")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or args.chem_dir
    chem_path = args.chem_dir / "chemical_separation_by_k.csv"
    if not chem_path.is_file():
        print(f"Missing {chem_path} (run compare_endpoint_cluster_k.py --chemical-scan)", flush=True)
        return 1
    chem = pd.read_csv(chem_path)
    by_cube = discover_motif_rmsd_csvs_by_cube(args.output_root, cubes=args.cubes)
    nmi_parts: list[pd.DataFrame] = []
    for cube, paths in by_cube.items():
        cp_dir = args.output_root / f"endpoint_changepoints_{cube}"
        if not cp_dir.is_dir():
            print(f"skip {cube}: missing {cp_dir}", flush=True)
            continue
        print(f"NMI vs Murata labels {cube} ...", flush=True)
        frames = label_motif_metastructures(load_motif_rmsd_csvs(paths))
        nmi_parts.append(
            score_murata_nmi_by_k(
                cp_dir,
                frames,
                cube=cube,
                k_min=args.k_min,
                k_max=args.k_max,
            )
        )
    if not nmi_parts:
        print("No motif CSVs scored.", flush=True)
        return 1
    nmi = pd.concat(nmi_parts, ignore_index=True)
    merged = chem.merge(nmi, on=["cohort", "k"], how="left")
    picked = select_k_by_chemical_information(merged)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "chemical_k_murata_nmi_by_k.csv"
    pick_path = out_dir / "chemical_k_selected.csv"
    merged.to_csv(merged_path, index=False)
    picked.to_csv(pick_path, index=False)
    print(f"wrote {merged_path}", flush=True)
    print(f"wrote {pick_path}", flush=True)
    print(picked.to_string(index=False), flush=True)
    plot_nmi_vs_k(merged, out_dir / "plots" / "murata_nmi_vs_k.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
