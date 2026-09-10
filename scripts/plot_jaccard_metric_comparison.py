"""
Old vs 1-to-1 Jaccard figures for the changepoint timing metric writeup.

Recomputes both formulas from existing ``all_breakpoints.csv`` (no Pelt).
The current ``compare_breakpoints`` is the 1-to-1 definition; the pre-fix
fuzzy-A / exact-union formula is inlined here so the canvas comparison can
be regenerated after the timing CSVs were overwritten.

GSA rows come from ``gsa_changepoints_B*``. Endpoint rows pair
``endpoint_changepoints_B*`` with the matching GSA directory on the original
``frame`` column (endpoint is stride-5, GSA is every frame).

Example
-------
    python scripts/plot_jaccard_metric_comparison.py
    python scripts/plot_jaccard_metric_comparison.py --out-dir output/jaccard_metric_comparison
"""

from __future__ import annotations

import argparse
import itertools
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.detection import canonicalize_traj_id, compare_breakpoints
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.utils.run_log import RunContext

GSA_PAIR_ORDER = (
    ("combined", "gsa"),
    ("combined", "iodine"),
    ("gsa", "iodine"),
    ("combined", "na_water"),
    ("gsa", "na_water"),
    ("iodine", "na_water"),
)
ENDPOINT_PAIR_ORDER = (
    ("combined", "endpoint"),
    ("endpoint", "gsa"),
    ("endpoint", "iodine"),
    ("endpoint", "na_water"),
)
OLD_COLOR = "#7f7f7f"
NEW_COLOR = "#1f77b4"
DUMMY_T = np.array([0.0, 1.0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Plot old (fuzzy/exact) vs 1-to-1 Jaccard from GSA and endpoint "
            "all_breakpoints.csv."
        )
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root with gsa_changepoints_B* and endpoint_changepoints_B*",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/jaccard_metric_comparison"),
        help="Where CSVs and plots are written",
    )
    p.add_argument(
        "--tolerance-frames",
        type=int,
        default=50,
        help="Match window in original trajectory frames (default: 50)",
    )
    return p.parse_args()


def _legacy_compare(bkps_a: list[int], bkps_b: list[int], tolerance: int) -> dict[str, float]:
    """Pre-fix compare_breakpoints: fuzzy A→B count / exact-set union, Chamfer A→B."""
    n_a, n_b = len(bkps_a), len(bkps_b)
    shared = sum(1 for b in bkps_a if any(abs(b - c) <= tolerance for c in bkps_b))
    union_size = len(set(bkps_a) | set(bkps_b))
    if n_a == 0 and n_b == 0:
        jaccard = 1.0
    elif union_size == 0:
        jaccard = 0.0
    else:
        jaccard = shared / union_size
    if n_a > 0 and n_b > 0:
        arr_b = np.array(bkps_b, dtype=np.float64)
        offset = float(np.mean([np.abs(b - arr_b).min() for b in bkps_a]))
    else:
        offset = float("nan")
    return {"n_shared": shared, "jaccard": jaccard, "offset_frames": offset}


def _frames(sub: pd.DataFrame) -> list[int]:
    loc = "frame" if "frame" in sub.columns else "signal_index"
    return [int(x) for x in sub[loc].to_numpy()]


def _load_bkps(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty or "traj_id" not in df.columns or "group" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["traj_id"] = df["traj_id"].map(canonicalize_traj_id)
    df["group"] = df["group"].astype(str)
    return df


def _pair_row(
    *,
    cube: str,
    tid: str,
    ga: str,
    gb: str,
    a: list[int],
    b: list[int],
    tolerance: int,
    family: str,
) -> dict:
    old_ab = _legacy_compare(a, b, tolerance)
    old_ba = _legacy_compare(b, a, tolerance)
    new = compare_breakpoints(a, b, tolerance, DUMMY_T)
    return {
        "family": family,
        "cube": cube,
        "traj_id": str(tid),
        "group_a": ga,
        "group_b": gb,
        "n_bkps_a": new["n_bkps_a"],
        "n_bkps_b": new["n_bkps_b"],
        "jaccard_old": old_ab["jaccard"],
        "jaccard_old_reversed": old_ba["jaccard"],
        "jaccard_1to1": new["jaccard"],
        "n_shared_old": old_ab["n_shared"],
        "n_shared_old_reversed": old_ba["n_shared"],
        "n_shared_1to1": new["n_shared"],
        "offset_old_ab": old_ab["offset_frames"],
        "offset_old_ba": old_ba["offset_frames"],
        "offset_matched": new["mean_timing_offset_frames"],
    }


def _rows_from_table(
    df: pd.DataFrame,
    cube: str,
    tolerance: int,
    family: str,
    *,
    require_group: str | None = None,
) -> list[dict]:
    rows: list[dict] = []
    if df.empty:
        return rows
    for tid, tdf in df.groupby(df["traj_id"].astype(str)):
        groups = sorted(str(g) for g in tdf["group"].dropna().unique())
        if len(groups) < 2:
            continue
        for ga, gb in itertools.combinations(groups, 2):
            if require_group is not None and require_group not in (ga, gb):
                continue
            a = _frames(tdf[tdf["group"] == ga])
            b = _frames(tdf[tdf["group"] == gb])
            rows.append(
                _pair_row(
                    cube=cube,
                    tid=str(tid),
                    ga=ga,
                    gb=gb,
                    a=a,
                    b=b,
                    tolerance=tolerance,
                    family=family,
                )
            )
    return rows


def collect_gsa_rows(output_root: Path, tolerance: int) -> pd.DataFrame:
    rows: list[dict] = []
    for cube in KNOWN_CUBES:
        path = output_root / f"gsa_changepoints_{cube}" / "all_breakpoints.csv"
        if not path.is_file():
            continue
        rows.extend(_rows_from_table(_load_bkps(path), cube, tolerance, "gsa"))
    return pd.DataFrame(rows)


def collect_endpoint_vs_gsa_rows(output_root: Path, tolerance: int) -> pd.DataFrame:
    rows: list[dict] = []
    for cube in KNOWN_CUBES:
        ep_path = output_root / f"endpoint_changepoints_{cube}" / "all_breakpoints.csv"
        gsa_path = output_root / f"gsa_changepoints_{cube}" / "all_breakpoints.csv"
        if not ep_path.is_file() or not gsa_path.is_file():
            continue
        ep = _load_bkps(ep_path)
        gsa = _load_bkps(gsa_path)
        if ep.empty or gsa.empty:
            continue
        merged = pd.concat([ep, gsa], ignore_index=True)
        rows.extend(
            _rows_from_table(
                merged,
                cube,
                tolerance,
                "endpoint_vs_gsa",
                require_group="endpoint",
            )
        )
    return pd.DataFrame(rows)


def _pair_label(ga: str, gb: str) -> str:
    return f"{ga}–{gb}"


def summarize_pairs(rows: pd.DataFrame, pair_order: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    recs = []
    for ga, gb in pair_order:
        sub = rows[(rows["group_a"] == ga) & (rows["group_b"] == gb)]
        if sub.empty:
            continue
        recs.append(
            {
                "group_a": ga,
                "group_b": gb,
                "pair": _pair_label(ga, gb),
                "n": int(len(sub)),
                "median_jaccard_old": float(sub["jaccard_old"].median()),
                "median_jaccard_1to1": float(sub["jaccard_1to1"].median()),
                "mean_jaccard_old": float(sub["jaccard_old"].mean()),
                "mean_jaccard_1to1": float(sub["jaccard_1to1"].mean()),
                "median_n_shared_old": float(sub["n_shared_old"].median()),
                "median_n_shared_1to1": float(sub["n_shared_1to1"].median()),
                "median_offset_old_ab": float(sub["offset_old_ab"].median()),
                "median_offset_old_ba": float(sub["offset_old_ba"].median()),
                "median_offset_matched": float(sub["offset_matched"].median()),
                "frac_n_shared_asymmetric": float(
                    (sub["n_shared_old"] != sub["n_shared_old_reversed"]).mean()
                ),
            }
        )
    return pd.DataFrame(recs)


def summarize_by_cube(rows: pd.DataFrame, ga: str, gb: str) -> pd.DataFrame:
    sub = rows[(rows["group_a"] == ga) & (rows["group_b"] == gb)]
    if sub.empty:
        return pd.DataFrame(
            columns=["cube", "n", "median_jaccard_old", "median_jaccard_1to1"]
        )
    out = sub.groupby("cube", as_index=False).agg(
        n=("jaccard_1to1", "count"),
        median_jaccard_old=("jaccard_old", "median"),
        median_jaccard_1to1=("jaccard_1to1", "median"),
    )
    out["cube"] = pd.Categorical(out["cube"], list(KNOWN_CUBES), ordered=True)
    return out.sort_values("cube")


def impact_stats_row(rows: pd.DataFrame, family: str, ga: str, gb: str) -> dict:
    pair = rows[(rows["group_a"] == ga) & (rows["group_b"] == gb)]
    j_old = rows["jaccard_old"].to_numpy(dtype=float)
    j_new = rows["jaccard_1to1"].to_numpy(dtype=float)
    j_rev = rows["jaccard_old_reversed"].to_numpy(dtype=float)
    spearman = (
        float(pd.Series(j_old).corr(pd.Series(j_new), method="spearman"))
        if len(rows) > 1
        else float("nan")
    )
    return {
        "family": family,
        "n_traj_pairs": int(len(rows)),
        "headline_pair": _pair_label(ga, gb),
        "spearman_old_vs_1to1": spearman,
        "frac_n_shared_asymmetric": float(
            (rows["n_shared_old"] != rows["n_shared_old_reversed"]).mean()
        )
        if len(rows)
        else float("nan"),
        "frac_jaccard_asymmetry_gt_0_10": float((np.abs(j_old - j_rev) > 0.10).mean())
        if len(rows)
        else float("nan"),
        "median_jaccard_headline_old": float(pair["jaccard_old"].median())
        if len(pair)
        else float("nan"),
        "median_jaccard_headline_1to1": float(pair["jaccard_1to1"].median())
        if len(pair)
        else float("nan"),
    }


def _grouped_bars(
    *,
    categories: list[str],
    series: list[tuple[str, list[float], str]],
    ylabel: str,
    title: str,
    out: Path,
    ylim: tuple[float, float] | None = None,
    value_fmt: str = "{:.2f}",
) -> Path:
    x = np.arange(len(categories))
    width = 0.36 if len(series) == 2 else 0.24
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    offsets = np.linspace(-(len(series) - 1) / 2, (len(series) - 1) / 2, len(series)) * width
    for off, (label, vals, color) in zip(offsets, series):
        bars = ax.bar(x + off, vals, width, label=label, color=color)
        for rect, val in zip(bars, vals):
            if not np.isfinite(val):
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height(),
                value_fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_jaccard_by_pair(pairs: pd.DataFrame, out: Path, title: str) -> Path:
    return _grouped_bars(
        categories=list(pairs["pair"]),
        series=[
            ("Old formula", list(pairs["median_jaccard_old"]), OLD_COLOR),
            ("1-to-1 Jaccard", list(pairs["median_jaccard_1to1"]), NEW_COLOR),
        ],
        ylabel="Median Jaccard",
        title=title,
        out=out,
        ylim=(0.0, 0.85),
    )


def plot_jaccard_by_cube(by_cube: pd.DataFrame, out: Path, title: str) -> Path:
    return _grouped_bars(
        categories=[str(c) for c in by_cube["cube"]],
        series=[
            ("Old formula", list(by_cube["median_jaccard_old"]), OLD_COLOR),
            ("1-to-1 Jaccard", list(by_cube["median_jaccard_1to1"]), NEW_COLOR),
        ],
        ylabel="Median Jaccard",
        title=title,
        out=out,
        ylim=(0.0, 0.85),
    )


def plot_offset_by_pair(pairs: pd.DataFrame, out: Path, title: str) -> Path:
    return _grouped_bars(
        categories=list(pairs["pair"]),
        series=[
            ("Old Chamfer A→B", list(pairs["median_offset_old_ab"]), OLD_COLOR),
            ("Old Chamfer B→A", list(pairs["median_offset_old_ba"]), "#c44e52"),
            ("1-to-1 matched mean", list(pairs["median_offset_matched"]), NEW_COLOR),
        ],
        ylabel="Median offset (frames)",
        title=title,
        out=out,
        value_fmt="{:.0f}",
    )


def _write_family(
    *,
    rows: pd.DataFrame,
    pair_order: tuple[tuple[str, str], ...],
    out_dir: Path,
    plot_dir: Path,
    family: str,
    pair_stem: str,
    cube_ga: str,
    cube_gb: str,
    pair_title: str,
    cube_title: str,
    offset_title: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    pairs = summarize_pairs(rows, pair_order)
    by_cube = summarize_by_cube(rows, cube_ga, cube_gb)
    if family == "gsa":
        row_csv = "traj_pair_jaccard_old_vs_1to1.csv"
        pair_csv = "pair_summary.csv"
        cube_csv = "combined_gsa_by_cube.csv"
        pair_png = "jaccard_by_pair_old_vs_1to1.png"
        cube_png = "jaccard_combined_gsa_by_cube.png"
        offset_png = "offset_by_pair_old_vs_matched.png"
    else:
        row_csv = f"{family}_traj_pair_jaccard_old_vs_1to1.csv"
        pair_csv = f"{family}_pair_summary.csv"
        cube_csv = f"{cube_ga}_{cube_gb}_by_cube.csv"
        pair_png = f"{pair_stem}.png"
        cube_png = f"jaccard_{cube_ga}_{cube_gb}_by_cube.png"
        offset_png = "offset_endpoint_pairs_old_vs_matched.png"
    rows.to_csv(out_dir / row_csv, index=False)
    pairs.to_csv(out_dir / pair_csv, index=False)
    by_cube.to_csv(out_dir / cube_csv, index=False)
    written = {
        f"{family}_jaccard_by_pair": plot_jaccard_by_pair(
            pairs, plot_dir / pair_png, pair_title
        ),
        f"{family}_jaccard_by_cube": plot_jaccard_by_cube(
            by_cube, plot_dir / cube_png, cube_title
        ),
        f"{family}_offset_by_pair": plot_offset_by_pair(
            pairs, plot_dir / offset_png, offset_title
        ),
    }
    return pairs, by_cube, written


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    output_root = Path(args.output_root)
    with RunContext.from_namespace(args, name="plot_jaccard_metric_comparison"):
        gsa_rows = collect_gsa_rows(output_root, args.tolerance_frames)
        ep_rows = collect_endpoint_vs_gsa_rows(output_root, args.tolerance_frames)
        if gsa_rows.empty and ep_rows.empty:
            raise SystemExit(
                f"No all_breakpoints.csv under {output_root}/gsa_changepoints_* "
                f"or endpoint_changepoints_*"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        stats_rows: list[dict] = []

        if not gsa_rows.empty:
            gsa_pairs, gsa_cube, gsa_plots = _write_family(
                rows=gsa_rows,
                pair_order=GSA_PAIR_ORDER,
                out_dir=out_dir,
                plot_dir=plot_dir,
                family="gsa",
                pair_stem="jaccard_by_pair_old_vs_1to1",
                cube_ga="combined",
                cube_gb="gsa",
                pair_title="Median Jaccard by group pair · six GSA cubes pooled",
                cube_title="Median combined–gsa Jaccard by cube",
                offset_title="Timing offset by GSA group pair · old Chamfer vs matched-pair mean",
            )
            written.update(gsa_plots)
            stats_rows.append(impact_stats_row(gsa_rows, "gsa", "combined", "gsa"))
        else:
            gsa_pairs = pd.DataFrame()

        if not ep_rows.empty:
            ep_pairs, _ep_cube, ep_plots = _write_family(
                rows=ep_rows,
                pair_order=ENDPOINT_PAIR_ORDER,
                out_dir=out_dir,
                plot_dir=plot_dir,
                family="endpoint_vs_gsa",
                pair_stem="jaccard_endpoint_pairs_old_vs_1to1",
                cube_ga="endpoint",
                cube_gb="gsa",
                pair_title="Median Jaccard by pair · endpoint vs GSA groups, six B* cubes",
                cube_title="Median endpoint–gsa Jaccard by cube",
                offset_title=(
                    "Timing offset · endpoint vs GSA groups · old Chamfer vs matched-pair mean"
                ),
            )
            written.update(ep_plots)
            stats_rows.append(
                impact_stats_row(ep_rows, "endpoint_vs_gsa", "endpoint", "gsa")
            )
            # Combined is the other headline pair from the sparse BMMpM writeup.
            combined_cube = summarize_by_cube(ep_rows, "combined", "endpoint")
            if not combined_cube.empty:
                combined_cube.to_csv(out_dir / "combined_endpoint_by_cube.csv", index=False)
                written["jaccard_combined_endpoint"] = plot_jaccard_by_cube(
                    combined_cube,
                    plot_dir / "jaccard_combined_endpoint_by_cube.png",
                    "Median combined–endpoint Jaccard by cube",
                )
        else:
            ep_pairs = pd.DataFrame()

        stats = pd.DataFrame(stats_rows)
        stats.to_csv(out_dir / "impact_stats.csv", index=False)

        docs_fig = ROOT / "docs" / "figures"
        docs_fig.mkdir(parents=True, exist_ok=True)
        for path in list(written.values()):
            dest = docs_fig / Path(path).name
            shutil.copy2(path, dest)
            written[f"docs_{Path(path).stem}"] = dest

    print(f"Wrote GSA {len(gsa_rows)} traj-pairs, endpoint-vs-GSA {len(ep_rows)} → {out_dir}")
    for rec in stats_rows:
        print(
            f"  {rec['family']}: Spearman {rec['spearman_old_vs_1to1']:.3f}; "
            f"{rec['headline_pair']} median J "
            f"{rec['median_jaccard_headline_old']:.3f} → "
            f"{rec['median_jaccard_headline_1to1']:.3f}"
        )
    for path in written.values():
        print(f"  {path}")


if __name__ == "__main__":
    main()
