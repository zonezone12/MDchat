"""Which logged run wrote a folder? Search ``output/<run_id>/manifest.json``.

Scripts run under ``RunContext`` leave ``output/<run_id>/manifest.json`` with
their arguments. This lists the runs whose manifest mentions ``--contains``
(for example an output folder name) and/or that ``--name`` logged, oldest
first, with their non-empty arguments.

Example
-------
python scripts/find_run_manifests.py --contains endpoint_changepoints_g1a_BMHpH
python scripts/find_run_manifests.py --name compare_changepoint_timing --contains g1b --csv output/g1b_timing_runs.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.utils.run_log import find_runs

BOOKKEEPING = ("run_id", "name", "started_at", "events_path", "python", "pid")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List logged runs whose manifest matches.")
    p.add_argument("--root", type=Path, default=ROOT / "output", help="Folder holding <run_id>/manifest.json")
    p.add_argument("--contains", help="Text the manifest must mention, e.g. an output folder name")
    p.add_argument("--name", help="Run name, e.g. changepoint_feature_groups")
    p.add_argument("--csv", type=Path, help="Also write the matches to this CSV")
    return p.parse_args()


def run_arguments(run: dict) -> dict:
    return {k: v for k, v in run.items() if k not in BOOKKEEPING and v not in (None, False, [], "")}


def main() -> int:
    args = parse_args()
    runs = find_runs(args.root, contains=args.contains, name=args.name)
    if not runs:
        print(f"No manifest under {args.root} matches.", flush=True)
        return 1
    for run in runs:
        print(f"== {run.get('run_id')}  {run.get('name')}  {run.get('started_at')}", flush=True)
        for key, value in run_arguments(run).items():
            print(f"   {key} = {value}", flush=True)
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "run_id": run.get("run_id"),
                    "name": run.get("name"),
                    "started_at": run.get("started_at"),
                    "arguments": json.dumps(run_arguments(run), default=str, ensure_ascii=False),
                }
                for run in runs
            ]
        ).to_csv(args.csv, index=False)
        print(f"wrote {args.csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
