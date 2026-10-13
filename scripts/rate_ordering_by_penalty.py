"""Does the methylation ordering of event rates depend on the Pelt penalty?

Reads the existing penalty sweeps (no re-detection) and, at every swept
penalty, compares breakpoints per trajectory across the equatorial series
BMM → BMH → BHH within each pole series (pM, pH):

* ``rates_by_penalty.csv``: family, group, cube, penalty, penalty / log(n),
  breakpoints per trajectory;
* ``series_ratios_by_penalty.csv``: per penalty and series, the 1 : mid : top
  ratio and whether BHH > BMH > BMM holds; also the pole effect pM / pH per
  equatorial pattern;
* ``ordering_summary.csv``: per family/group and series, the share of
  penalties where the ordering holds and the range of each ratio term.

Endpoint sweeps: ``endpoint_changepoints_{CUBE}/penalty_sweep_median_n`` when
present (BMHpH, BMMpM after the G1a grid fix), else ``penalty_sweep``. GSA
sweeps: ``gsa_changepoints_{CUBE}/penalty_sweep`` (``gsa`` and ``iodine``).

Example
-------
python scripts/rate_ordering_by_penalty.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

SERIES = {"pM": ("BMMpM", "BMHpM", "BHHpM"), "pH": ("BMMpH", "BMHpH", "BHHpH")}
LOG_N = {"endpoint": float(np.log(1000)), "gsa": float(np.log(5000))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "rate_ordering_by_penalty")
    return p.parse_args()


def _sweep_rates(output_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    cubes = sorted({c for s in SERIES.values() for c in s})
    for cube in cubes:
        base = output_root / f"endpoint_changepoints_{cube}"
        sweep = base / "penalty_sweep_median_n"
        if not (sweep / "penalty_sweep_summary.csv").is_file():
            sweep = base / "penalty_sweep"
        s = pd.read_csv(sweep / "penalty_sweep_summary.csv")
        rows.append(
            pd.DataFrame(
                {
                    "family": "endpoint",
                    "group": "endpoint",
                    "cube": cube,
                    "penalty": s["penalty"].round(3),
                    "bkps_per_traj": s["total_breakpoints"] / s["n_trajectories"],
                    "sweep_dir": str(sweep.relative_to(output_root)),
                }
            )
        )
        g = pd.read_csv(output_root / f"gsa_changepoints_{cube}" / "penalty_sweep" / "penalty_sweep_summary.csv")
        for group in ("gsa", "iodine"):
            rows.append(
                pd.DataFrame(
                    {
                        "family": "gsa",
                        "group": group,
                        "cube": cube,
                        "penalty": g["penalty"].round(3),
                        "bkps_per_traj": g[f"n_bkps_{group}"] / g["n_trajectories"],
                        "sweep_dir": f"gsa_changepoints_{cube}/penalty_sweep",
                    }
                )
            )
    out = pd.concat(rows, ignore_index=True)
    out["rel_penalty"] = out["penalty"] / out["family"].map(LOG_N)
    return out


def main() -> int:
    args = parse_args()
    rates = _sweep_rates(args.output_root)
    ratio_rows: list[dict] = []
    for (family, group), sub in rates.groupby(["family", "group"], sort=False):
        wide = sub.pivot_table(index="penalty", columns="cube", values="bkps_per_traj")
        for pen, r in wide.iterrows():
            base = {"family": family, "group": group, "penalty": pen, "rel_penalty": pen / LOG_N[family]}
            for name, (lo, mid, top) in SERIES.items():
                ratio_rows.append(
                    {
                        **base,
                        "series": name,
                        "ratio_mid": r[mid] / r[lo],
                        "ratio_top": r[top] / r[lo],
                        "ordering_holds": bool(r[top] > r[mid] > r[lo]),
                    }
                )
            for eq in ("BHH", "BMH", "BMM"):
                ratio_rows.append(
                    {**base, "series": f"pole_{eq}", "pole_pM_over_pH": r[f"{eq}pM"] / r[f"{eq}pH"]}
                )
    ratios = pd.DataFrame(ratio_rows)
    ordered = ratios[ratios["series"].isin(SERIES)]
    summary = (
        ordered.groupby(["family", "group", "series"], sort=False)
        .agg(
            n_penalties=("penalty", "size"),
            frac_ordering_holds=("ordering_holds", "mean"),
            ratio_mid_min=("ratio_mid", "min"),
            ratio_mid_max=("ratio_mid", "max"),
            ratio_top_min=("ratio_top", "min"),
            ratio_top_max=("ratio_top", "max"),
        )
        .reset_index()
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (
        ("rates_by_penalty.csv", rates),
        ("series_ratios_by_penalty.csv", ratios),
        ("ordering_summary.csv", summary),
    ):
        table.to_csv(args.out_dir / name, index=False)
        print(f"wrote {args.out_dir / name}", flush=True)
    print(summary.round(2).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
