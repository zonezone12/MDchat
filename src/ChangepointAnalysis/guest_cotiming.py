"""Guest entry/exit events vs cage and endpoint breakpoints, and their timescales.

Events come from debounced ``n_guest_inside_cavity`` (see
``src/ChangepointAnalysis/guest_events.py``). For each cube and each
``--min-dwell``, reports:

* event counts and rates;
* recall: the fraction of guest events with a cage (``gsa``) or endpoint
  breakpoint within ±``--tolerance`` frames, against a circular-shift null;
* timescales: median sustained guest-inside dwell next to the median cage and
  endpoint segment lengths.

Breakpoints are the own-elbow detections used throughout the paper: GSA from
``gsa_changepoints_matched_{CUBE}``, endpoint from ``OWN_ELBOW_ENDPOINT_DIRS``.

Example
-------
python scripts/guest_event_cotiming.py
python scripts/guest_event_cotiming.py --min-dwell 10 20 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd

from .gsa_cohort_run import KNOWN_CUBES
from .guest_events import event_recall, guest_events, shifted_recall_null
from .reporting import endpoint_changepoints_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--gsa-features-root", type=Path, default=ROOT / "output" / "gsa_features_by_cube")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "guest_event_cotiming")
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument("--min-dwell", type=int, nargs="*", default=[10, 20, 50], help="Frames (1 ps)")
    p.add_argument("--tolerance", type=int, default=50, help="Frames, as in the breakpoint Jaccard")
    p.add_argument("--n-shift", type=int, default=999, help="Shift replicates for the null")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _strip(traj_id: str, cube: str) -> str:
    return traj_id[len(cube) + 1 :] if traj_id.startswith(f"{cube}_") else traj_id


def _breakpoints(cp_dir: Path, group: str, cube: str) -> tuple[dict[str, list[int]], pd.Series]:
    bk = pd.read_csv(cp_dir / "all_breakpoints.csv", usecols=["traj_id", "group", "frame"])
    bk = bk[bk["group"] == group]
    by_traj = {
        _strip(t, cube): sorted(g["frame"].astype(int).tolist()) for t, g in bk.groupby("traj_id")
    }
    seg = pd.read_csv(cp_dir / "all_segment_stats.csv", usecols=["group", "start_frame", "end_frame"])
    seg = seg[seg["group"] == group]
    return by_traj, (seg["end_frame"] - seg["start_frame"])


def guest_event_cotiming(
    *,
    output_root: Path | str = "output",
    gsa_features_root: Path | str = "output/gsa_features_by_cube",
    out_dir: Path | str = "output/guest_event_cotiming",
    cubes: Optional[Sequence[str]] = None,
    min_dwell: Sequence[int] = (10, 20, 50),
    tolerance: int = 50,
    n_shift: int = 999,
    seed: int = 0,
) -> dict[str, Path]:
    class _A:
        pass
    args = _A()
    args.output_root = Path(output_root)
    args.gsa_features_root = Path(gsa_features_root)
    args.out_dir = Path(out_dir)
    args.cubes = list(cubes) if cubes is not None else list(KNOWN_CUBES)
    args.min_dwell = list(min_dwell)
    args.tolerance = int(tolerance)
    args.n_shift = int(n_shift)
    args.seed = int(seed)
    return _run_cotiming(args)


def _run_cotiming(args) -> dict[str, Path]:
    rng = np.random.default_rng(args.seed)
    summary: list[dict] = []
    event_rows: list[pd.DataFrame] = []
    for cube in args.cubes:
        feat_dir = args.gsa_features_root / cube
        gsa_dir = args.output_root / f"gsa_changepoints_matched_{cube}"
        ep_dir = endpoint_changepoints_dir(args.output_root, cube, own_elbow=True)
        if not (feat_dir.is_dir() and gsa_dir.is_dir() and ep_dir.is_dir()):
            print(f"skip {cube}: missing inputs", flush=True)
            continue
        print(f"guest events {cube} ...", flush=True)
        gsa_bk, gsa_len = _breakpoints(gsa_dir, "gsa", cube)
        ep_bk, ep_len = _breakpoints(ep_dir, "endpoint", cube)
        occupancy = {}
        for path in sorted(feat_dir.glob("*_gsa_features.csv")):
            df = pd.read_csv(path, usecols=["traj_id", "frame", "n_guest_inside_cavity"])
            occupancy[_strip(str(df["traj_id"].iloc[0]), cube)] = (
                df.sort_values("frame")["n_guest_inside_cavity"].to_numpy() >= 1
            )
        total_ns = sum(len(o) for o in occupancy.values()) / 1000.0
        for min_dwell in args.min_dwell:
            rec = {"gsa": ([], []), "endpoint": ([], [])}
            # matched-event totals, observed and per shared shift replicate
            matched = {"gsa": [0, np.zeros(args.n_shift)], "endpoint": [0, np.zeros(args.n_shift)]}
            dwell_parts = []
            for tid, occ in occupancy.items():
                ev = guest_events(occ, min_dwell=min_dwell).assign(cube=cube, traj_id=tid, min_dwell=min_dwell)
                dwell_parts.append(ev)
                frames = ev.loc[ev["event"] != "start", "start"].tolist()
                if not frames:
                    continue
                for fam, bks in (("gsa", gsa_bk), ("endpoint", ep_bk)):
                    b = bks.get(tid, [])
                    obs = event_recall(frames, b, args.tolerance)
                    null = shifted_recall_null(
                        frames, b, len(occ), args.tolerance, n_shift=args.n_shift, rng=rng
                    )
                    rec[fam][0].extend([obs] * len(frames))  # event-weighted
                    rec[fam][1].extend([float(np.nanmean(null))] * len(frames) if null.size else [])
                    matched[fam][0] += obs * len(frames)
                    if null.size:
                        matched[fam][1] += null * len(frames)
            dwells = pd.concat(dwell_parts, ignore_index=True)
            events = dwells[dwells["event"] != "start"]
            inside = dwells[dwells["state"] & ~dwells["censored"]]
            entries = events[events["event"] == "entry"]
            first_entry = entries.groupby("traj_id")["start"].min()
            entry_dwells = dwells[dwells["event"] == "entry"]
            row = {
                "cube": cube,
                "min_dwell_ps": min_dwell,
                "n_traj": len(occupancy),
                "n_traj_with_entry": int(events.loc[events["event"] == "entry", "traj_id"].nunique()),
                "n_entries": int((events["event"] == "entry").sum()),
                "n_exits": int((events["event"] == "exit").sum()),
                "events_per_ns": len(events) / total_ns,
                "median_inside_dwell_ps": float(inside["length"].median()) if len(inside) else np.nan,
                "frac_entries_inside_to_end": (
                    float(entry_dwells["censored"].mean()) if len(entry_dwells) else np.nan
                ),
                "median_first_entry_ps": float(first_entry.median()) if len(first_entry) else np.nan,
                "median_cage_segment_ps": float(gsa_len.median()),
                "median_endpoint_segment_ps": float(ep_len.median()),
            }
            for fam in ("gsa", "endpoint"):
                obs, null = (np.asarray(x, dtype=float) for x in rec[fam])
                row[f"recall_{fam}"] = float(np.nanmean(obs)) if obs.size else np.nan
                row[f"recall_{fam}_null"] = float(np.nanmean(null)) if null.size else np.nan
                if obs.size and null.size:
                    row[f"excess_{fam}"] = (row[f"recall_{fam}"] - row[f"recall_{fam}_null"]) / (
                        1 - row[f"recall_{fam}_null"]
                    )
                    obs_n, null_n = matched[fam]
                    row[f"p_{fam}"] = (1 + int((null_n >= obs_n - 1e-9).sum())) / (args.n_shift + 1)
            summary.append(row)
            event_rows.append(dwells)
    if not summary:
        raise FileNotFoundError("No guest event co-timing rows")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(summary)
    written = {
        "guest_event_cotiming": args.out_dir / "guest_event_cotiming.csv",
        "guest_dwells": args.out_dir / "guest_dwells.csv",
    }
    out.to_csv(written["guest_event_cotiming"], index=False)
    pd.concat(event_rows, ignore_index=True).to_csv(written["guest_dwells"], index=False)
    return written


def main() -> int:
    try:
        written = _run_cotiming(parse_args())
    except FileNotFoundError:
        return 1
    for path in written.values():
        print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
