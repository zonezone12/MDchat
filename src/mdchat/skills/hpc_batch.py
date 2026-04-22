"""Export HPC-oriented batch scripts from the current session summary."""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, Any

from ..registry import get_default_registry
from ..skill import Parameter, ParamType, Skill, SkillResult

if TYPE_CHECKING:
    from ..context import AnalysisContext


class ExportHpcBatchScriptsSkill(Skill):
    name = "export_hpc_batch_scripts"
    description = (
        "Summarize this conversation/session and write cluster-ready scripts: "
        "(1) run volume/endpoint/guest analysis for each trajectory via "
        "`scripts/run_volume_endpoint_guest_analysis.py`, "
        "(2) aggregate and rank `*_simulation_score.csv` files via "
        "`scripts/compare_simulation_scores.py`. "
        "Also exports `replay_session_tasks.py`, a standalone Python runner that replays "
        "session tasks using repository `scripts/*.py` entry points (no `src/mdchat` imports). "
        "Produces a Markdown brief, manifest template, Python drivers, bash helpers, "
        "and optionally a Slurm array template. Edit paths on the cluster before submitting."
    )
    category = "session"
    parameters = [
        Parameter(
            "goal_summary",
            ParamType.STRING,
            "What you want from the batch (system, hypotheses, screening criteria). "
            "If empty, the skill summarizes from session state and executed skills.",
            required=False,
            default="",
        ),
        Parameter(
            "repo_root_env",
            ParamType.STRING,
            "Environment variable name pointing to the MD_analysis repo on the cluster.",
            required=False,
            default="MD_ANALYSIS_ROOT",
        ),
        Parameter(
            "results_subdir",
            ParamType.STRING,
            "Default results tree under $REPO for ranking (rank script's RESULTS_DIR).",
            required=False,
            default="output/hpc_batch",
        ),
        Parameter(
            "align_sel",
            ParamType.STRING,
            "Alignment selection passed to run_volume_endpoint_guest_analysis.",
            required=False,
            default="resid 1-6",
        ),
        Parameter(
            "guest_sel",
            ParamType.STRING,
            "Guest atom selection for GSAnalyzer.",
            required=False,
            default="name I",
        ),
        Parameter(
            "endpoint_residue",
            ParamType.STRING,
            'Comma-separated endpoint residue selections, e.g. "resid 1,resid 2".',
            required=False,
            default="resid 1,resid 2,resid 3,resid 4,resid 5,resid 6",
        ),
        Parameter(
            "analysis_n_jobs",
            ParamType.INTEGER,
            "Parallel workers inside each trajectory job (use 1 on small allocations).",
            required=False,
            default=1,
            min_value=1,
            max_value=512,
        ),
        Parameter(
            "include_slurm_template",
            ParamType.BOOLEAN,
            "Also write submit_hpc_array.slurm (Slurm array over manifest rows).",
            required=False,
            default=True,
        ),
    ]
    requires: list = []
    produces = ["hpc_export_manifest", "hpc_export_readme"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        goal = (params.get("goal_summary") or "").strip()
        repo_var = params.get("repo_root_env") or "MD_ANALYSIS_ROOT"
        results_sub = params.get("results_subdir") or "output/hpc_batch"
        align_sel = params.get("align_sel") or "resid 1-6"
        guest_sel = params.get("guest_sel") or "name I"
        ep_raw = params.get("endpoint_residue") or ""
        n_jobs = int(params.get("analysis_n_jobs") or 1)
        include_slurm = bool(params.get("include_slurm_template", True))

        residues = [s.strip() for s in ep_raw.split(",") if s.strip()]
        if not residues:
            residues = [f"resid {i}" for i in range(1, 7)]

        readme_path = os.path.join(context.output_dir, "HPC_BATCH_README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                _build_session_markdown(
                    context, goal, repo_var, align_sel, guest_sel, residues
                )
            )

        manifest_example = os.path.join(context.output_dir, "manifest.example.tsv")
        _write_manifest_example(manifest_example)

        py_path = os.path.join(context.output_dir, "run_manifest_trajectories.py")
        _write_run_manifest_py(
            py_path,
            repo_var=repo_var,
            align_sel=align_sel,
            guest_sel=guest_sel,
            residues=residues,
            n_jobs=n_jobs,
        )

        sh_single = os.path.join(context.output_dir, "run_one_trajectory.sh")
        _write_run_one_sh(
            sh_single,
            repo_var=repo_var,
            align_sel=align_sel,
            guest_sel=guest_sel,
            residues=residues,
            n_jobs=n_jobs,
        )

        sh_rank = os.path.join(context.output_dir, "rank_simulation_outputs.sh")
        _write_rank_sh(sh_rank, repo_var=repo_var, results_sub=results_sub)

        replay_path = os.path.join(context.output_dir, "replay_session_tasks.py")
        _write_replay_script(replay_path, context)

        artifacts: dict[str, str] = {
            "hpc_readme": readme_path,
            "manifest_example": manifest_example,
            "run_manifest": py_path,
            "run_one_trajectory": sh_single,
            "rank_outputs": sh_rank,
            "replay_session_tasks": replay_path,
        }

        if include_slurm:
            slurm_path = os.path.join(context.output_dir, "submit_hpc_array.slurm")
            _write_slurm(slurm_path, repo_var=repo_var)
            artifacts["slurm_array"] = slurm_path

        context.set("hpc_export_manifest", manifest_example)
        context.set("hpc_export_readme", readme_path)

        lines = [
            "Wrote HPC batch templates into the session output directory.",
            f"- Brief + session summary: {readme_path}",
            f"- Manifest template: {manifest_example}",
            f"- Driver: `python {py_path} --manifest your.tsv`",
            f"- Single job: `bash {sh_single}`",
            f"- Rank scores: `bash {sh_rank}`",
            f"- Session replay (scripts-style): `python {replay_path}`",
        ]
        if include_slurm:
            lines.append(f"- Slurm: `sbatch {artifacts['slurm_array']}` (edit paths first)")

        return SkillResult(
            success=True,
            data={
                "hpc_export_manifest": manifest_example,
                "hpc_export_readme": readme_path,
            },
            artifacts=artifacts,
            summary="\n".join(lines),
        )


def _build_session_markdown(
    context: AnalysisContext,
    goal_summary: str,
    repo_var: str,
    align_sel: str,
    guest_sel: str,
    residues: list[str],
) -> str:
    lines = [
        "# HPC batch export — MDChat",
        "",
        "Batch **per-trajectory** analysis (volume, endpoints, guest) then **rank** runs using "
        "`*_simulation_score.csv` produced under each `--out_prefix`.",
        "",
        "## Goal (from conversation)",
        "",
    ]
    if goal_summary:
        lines.append(goal_summary.strip())
    else:
        lines.append(
            "_No explicit goal text was supplied; add your screening criteria here on the cluster._"
        )

    lines += ["", "## Session-derived defaults", ""]

    if context.topology_path:
        lines.append(f"- **Topology seen in session**: `{context.topology_path}`")
    else:
        lines.append("- **Topology**: _not loaded — use `manifest.tsv` on the cluster._")

    if context.trajectory_path:
        lines.append(f"- **Trajectory seen in session**: `{context.trajectory_path}`")

    hist = context.get_history()
    if hist:
        lines.append(f"- **Skills executed**: {' → '.join(hist)}")
    else:
        lines.append("- **Skills executed**: _none recorded yet._")

    lines += [
        "",
        "### Embedded CLI defaults",
        "",
        f"- `{repo_var}` → path to MD_analysis checkout",
        f"- `--align_sel`: `{align_sel}`",
        f"- `--guest_sel`: `{guest_sel}`",
        f"- `--endpoint_residues` / `--cube_faces`: {' '.join(residues)}",
        "",
        "## Cluster workflow",
        "",
        f"1. Export `MD_analysis` (or sync these files into your job directory).",
        f"2. `export {repo_var}=/path/to/MD_analysis`",
        "3. Copy `manifest.example.tsv` → `manifest.tsv` and fill **topology**, **trajectories**, **out_prefix**.",
        "4. Run all rows: `python run_manifest_trajectories.py --manifest manifest.tsv`",
        "   Or use Slurm: edit `submit_hpc_array.slurm` then `sbatch submit_hpc_array.slurm`.",
        "5. Merge ranks: `bash rank_simulation_outputs.sh` → `all_simulations_ranked.csv`.",
        "",
        "## Files",
        "",
        "| File | Purpose |",
        "|------|---------|",
        "| `manifest.example.tsv` | Columns: topology, trajectories (space-separated), out_prefix |",
        "| `run_manifest_trajectories.py` | One analysis job per manifest row; optional `--task-index` |",
        "| `run_one_trajectory.sh` | Single topology + prefix + traj files |",
        "| `rank_simulation_outputs.sh` | Runs `scripts/compare_simulation_scores.py` recursively |",
        "| `replay_session_tasks.py` | Replays session tasks via `scripts/*.py` commands |",
        "| `submit_hpc_array.slurm` | Array job: one manifest row per task ID |",
        "",
        "## Choosing trajectories to inspect",
        "",
        "Open `all_simulations_ranked.csv`, sort by `overall_score` or domain columns "
        "(guest entry, volume dynamics, correlation). Follow up with targeted visualization "
        "or MDChat `export_work_log`-style logs for reproducibility.",
        "",
        "---",
        "*Skill `export_hpc_batch_scripts`.*",
        "",
    ]
    return "\n".join(lines)


def _write_manifest_example(path: str) -> None:
    rows = [
        ["topology", "trajectories", "out_prefix"],
        [
            "/path/on/cluster/system.prmtop",
            "/path/on/cluster/run01.nc",
            "/path/on/cluster/output/hpc_batch/run01",
        ],
        [
            "/path/on/cluster/system.prmtop",
            "/path/run02_part1.nc /path/run02_part2.nc",
            "/path/on/cluster/output/hpc_batch/run02",
        ],
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerows(rows)


def _write_run_manifest_py(
    path: str,
    *,
    repo_var: str,
    align_sel: str,
    guest_sel: str,
    residues: list[str],
    n_jobs: int,
) -> None:
    residues_literal = repr(residues)
    body_tmpl = """#!/usr/bin/env python3
'''Drive `scripts/run_volume_endpoint_guest_analysis.py` from a manifest.

Manifest columns (header required): topology, trajectories, out_prefix
  trajectories — space-separated trajectory paths (no spaces in paths).

Environment: {repo_var} — absolute path to MD_analysis repo root.

Options:
  --task-index K — run only manifest row K (0-based data rows). Pair with Slurm arrays.

Generated by MDChat export_hpc_batch_scripts.
'''
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="If set, run only this manifest row index (0-based).",
    )
    args = p.parse_args()

    repo = os.environ.get("{repo_var}", "").strip()
    if not repo:
        print("ERROR: set environment variable {repo_var}", file=sys.stderr)
        return 1

    script = Path(repo) / "scripts" / "run_volume_endpoint_guest_analysis.py"
    if not script.is_file():
        print(f"ERROR: missing {{script}}", file=sys.stderr)
        return 1

    manifest_path = Path(args.manifest)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        sample = f.readline()
        f.seek(0)
        delim = chr(9) if (chr(9) in sample) else ","
        rows = list(csv.DictReader(f, delimiter=delim))

    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(rows):
            print("ERROR: task-index out of range", file=sys.stderr)
            return 1
        rows = [rows[args.task_index]]

    endpoint_residues = __RESIDUES_LITERAL__

    for i, row in enumerate(rows):
        top = (row.get("topology") or "").strip()
        traj_cell = (row.get("trajectories") or row.get("traj") or "").strip()
        prefix = (row.get("out_prefix") or "").strip()
        if not top or not traj_cell or not prefix:
            print(f"Skipping row {{i}}: incomplete columns", file=sys.stderr)
            continue

        traj_parts = traj_cell.split()
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script),
            "--top",
            top,
            "--traj",
            *traj_parts,
            "--out_prefix",
            prefix,
            "--align_sel",
            "{align_sel}",
            "--guest_sel",
            "{guest_sel}",
            "--endpoint_residues",
            *endpoint_residues,
            "--cube_faces",
            *endpoint_residues,
            "--n_jobs",
            "{n_jobs}",
            "--auto_limit_workers",
        ]
        label = args.task_index if args.task_index is not None else i
        print(f"=== task {{label}} ===")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

    body = (
        body_tmpl.format(
            repo_var=repo_var,
            align_sel=align_sel,
            guest_sel=guest_sel,
            n_jobs=n_jobs,
        ).replace("__RESIDUES_LITERAL__", residues_literal)
    )

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass


def _write_run_one_sh(
    path: str,
    *,
    repo_var: str,
    align_sel: str,
    guest_sel: str,
    residues: list[str],
    n_jobs: int,
) -> None:
    res_args = " ".join(f'"{r}"' for r in residues)
    txt = f'''#!/usr/bin/env bash
set -euo pipefail
# Usage: run_one_trajectory.sh TOPOLOGY OUT_PREFIX TRAJ1 [TRAJ2 ...]
# Requires {repo_var} pointing at MD_analysis repo root.

REPO="${{{repo_var}:?export {repo_var}}}"
TOP="${{1:?topology}}"
PREFIX="${{2:?out_prefix}}"
shift 2
TRAJ=( "$@" )
if [[ ${{#TRAJ[@]}} -eq 0 ]]; then
  echo "Provide at least one trajectory file after OUT_PREFIX." >&2
  exit 1
fi

python "$REPO/scripts/run_volume_endpoint_guest_analysis.py" \\
  --top "$TOP" \\
  --traj "${{TRAJ[@]}}" \\
  --out_prefix "$PREFIX" \\
  --align_sel '{align_sel}' \\
  --guest_sel '{guest_sel}' \\
  --endpoint_residues {res_args} \\
  --cube_faces {res_args} \\
  --n_jobs {n_jobs} \\
  --auto_limit_workers
'''
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass


def _write_rank_sh(path: str, *, repo_var: str, results_sub: str) -> None:
    txt = f'''#!/usr/bin/env bash
set -euo pipefail
# Rank simulations: aggregates *_simulation_score.csv under RESULTS_DIR (recursive).

REPO="${{{repo_var}:?export {repo_var}}}"
RESULTS_DIR="${{RESULTS_DIR:-$REPO/{results_sub}}}"

python "$REPO/scripts/compare_simulation_scores.py" \\
  --input_dir "$RESULTS_DIR" \\
  --output "$RESULTS_DIR/all_simulations_ranked.csv" \\
  --top_n 25

echo "Ranked table: $RESULTS_DIR/all_simulations_ranked.csv"
'''
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass


def _write_slurm(path: str, *, repo_var: str) -> None:
    txt = """#!/bin/bash
#SBATCH --job-name=md_traj_batch
#SBATCH --output=logs/%%A_%%a.out
#SBATCH --error=logs/%%A_%%a.err
#SBATCH --array=0-9
# One task per manifest **data** row (0-based). Set --array=0-$((N-1)) for N rows.
# Submit from the directory that contains manifest.tsv and run_manifest_trajectories.py.

set -euo pipefail
mkdir -p logs

: "${%s:?Set %s to the MD_analysis repo root}"

MANIFEST="${MANIFEST:-manifest.tsv}"
python run_manifest_trajectories.py \\
  --manifest "$MANIFEST" \\
  --task-index "$SLURM_ARRAY_TASK_ID"
""" % (
        repo_var,
        repo_var,
    )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)


def _to_python_literal(value: Any) -> str:
    """Safe literal rendering for generated script payloads."""
    try:
        return repr(value)
    except Exception:
        return repr(str(value))


def _write_replay_script(path: str, context: "AnalysisContext") -> None:
    log_entries = context.get_log()
    rows: list[tuple[str, dict[str, Any]]] = []
    for entry in log_entries:
        params = entry.params if isinstance(entry.params, dict) else {}
        rows.append((entry.skill_name, params))

    payload_lines = []
    for skill_name, params in rows:
        payload_lines.append(
            f"    ({_to_python_literal(skill_name)}, {_to_python_literal(params)}),"
        )
    payload_block = "\n".join(payload_lines) if payload_lines else "    # (no executed skills)"

    txt = f"""#!/usr/bin/env python3
'''Replay session tasks via direct imports from src/ (no src/mdchat imports).

Generated by MDChat export_hpc_batch_scripts.
'''
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import MDAnalysis as mda
import pandas as pd

from src.AlignedTrajectory import AlignedTrajectory
from src.EndpointAnalyzer import EndpointAnalyzerObserver
from src.TrajectoryIterator import TrajectoryIterator
from src.FrameSelection import FrameSelection
from src.task import GSAnalyzerObserver
from src.FrameSelection.simulation_scores import (
    find_simulation_score_files,
    load_and_compare_simulation_scores,
)
from src.utils.monitor import auto_limit_workers


STEPS = [
{payload_block}
]


def _normalize_traj(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def step_load_trajectory(params: dict, state: dict) -> None:
    top = params.get("topology_path") or params.get("topology") or params.get("top") or state.get("topology_path")
    traj = _normalize_traj(params.get("trajectory_paths") or params.get("trajectory_path") or params.get("traj") or state.get("trajectory_paths"))
    if not top or not traj:
        raise ValueError("load_trajectory requires topology and trajectory path(s)")
    u = mda.Universe(str(top), *traj)
    state["universe"] = u
    state["topology_path"] = str(top)
    state["trajectory_paths"] = traj
    print(f"Loaded trajectory: {{top}} ({{len(u.trajectory)}} frames)")


def step_run_volume_endpoint_guest_analysis(params: dict, state: dict) -> None:
    u = state.get("universe")
    if u is None:
        step_load_trajectory(params, state)
        u = state["universe"]

    out_prefix = params.get("out_prefix") or str(Path(state["output_dir"]) / "replay_run")
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    align_sel = params.get("align_sel", "resid 1-6")
    endpoint_res = params.get("endpoint_residues") or ["resid 1", "resid 2", "resid 3", "resid 4", "resid 5", "resid 6"]
    cube_faces = params.get("cube_faces") or endpoint_res
    guest_sel = params.get("guest_sel", "name I")
    n_jobs = int(params.get("n_jobs", 1))
    auto_limit = bool(params.get("auto_limit_workers", True))

    try:
        aligned = AlignedTrajectory(universe=u, align_sel=str(align_sel), ref_frame=0, in_memory=False)
        u = aligned.get_aligned_universe()
        state["universe"] = u
    except Exception as exc:
        warnings.warn(f"Alignment skipped: {{exc}}")

    iterator = TrajectoryIterator(u, use_dask=bool(params.get("use_dask", False)))

    endpoint_observer = EndpointAnalyzerObserver(
        residue_sel_list=[str(x) for x in endpoint_res]
    )
    iterator.subscribe(endpoint_observer)

    gsa_observer = GSAnalyzerObserver(
        face_sel_list=[str(x) for x in cube_faces],
        guest_sel=str(guest_sel),
        out_prefix=str(out_prefix),
        guest_tracking_method=str(params.get("guest_tracking_method", "distance")),
        guest_distance_threshold=params.get("guest_distance_threshold"),
    )
    iterator.subscribe(gsa_observer)

    if auto_limit and n_jobs != 1:
        n_jobs = auto_limit_workers(n_jobs, u)
        print(f"Auto-limited workers to: {{n_jobs}}")

    iterator.iterate(n_jobs=n_jobs)

    endpoint_metrics_df = endpoint_observer.get_endpoint_metrics()
    if endpoint_metrics_df is not None and len(endpoint_metrics_df) > 0:
        endpoint_metrics_df.to_csv(f"{{out_prefix}}_endpoint_metrics.csv", index=False)

    cube_metrics_df = gsa_observer.get_metrics_df()
    volume = gsa_observer.get_volume()
    guest_stats = gsa_observer.get_guest_residence_stats()

    if cube_metrics_df is not None and len(cube_metrics_df) > 0:
        cube_metrics_df.to_csv(f"{{out_prefix}}_gsa_nanocube.csv", index=False)

    fs = FrameSelection()
    sim_score, score_details = fs.score_simulation(
        guest_stats=guest_stats,
        volume=volume,
        correlation_df=None,
        endpoint_dists_array=endpoint_observer.get_endpoint_distances(),
        endpoint_metrics_df=endpoint_metrics_df,
        cube_metrics_df=cube_metrics_df,
        min_guest_entry=True,
        min_volume_change_pct=10.0,
        min_correlation=0.5,
    )
    fs.save_simulation_score_csv(
        output_path=f"{{out_prefix}}_simulation_score.csv",
        trajectory_id=Path(out_prefix).name,
        guest_stats=guest_stats,
        volume=volume,
        correlation_df=None,
    )
    print(f"Analysis complete: {{out_prefix}} (score={{sim_score:.3f}}, valid={{score_details.get('is_valid') if score_details else False}})")


def step_compare_simulation_scores(params: dict, state: dict) -> None:
    input_dir = params.get("input_dir") or state.get("output_dir")
    output_csv = params.get("output") or str(Path(input_dir) / "all_simulations_ranked.csv")
    sort_by = str(params.get("sort_by", "overall_score"))
    ascending = bool(params.get("ascending", False))
    pattern = str(params.get("pattern", "*_simulation_score.csv"))
    csv_files = params.get("csv_files")
    if not csv_files:
        csv_files = find_simulation_score_files(input_dir, pattern=pattern, recursive=True)
    if not csv_files:
        raise ValueError(f"No simulation score files found under {{input_dir}}")
    df = load_and_compare_simulation_scores(
        csv_paths=csv_files,
        output_path=output_csv,
        sort_by=sort_by,
        ascending=ascending,
    )
    print(f"Ranked {{len(df)}} simulation(s) -> {{output_csv}}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", required=True, help="MD_analysis repository root.")
    p.add_argument("--output-dir", required=True, help="Replay output directory.")
    p.add_argument("--from-step", type=int, default=1, help="1-based start step.")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report-json", default="", help="Optional JSON report path.")
    args = p.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo_root))

    if args.from_step < 1:
        print("ERROR: --from-step must be >= 1", file=sys.stderr)
        return 1

    state = {{"output_dir": str(output_dir)}}
    report = []
    ok_all = True

    for idx, (skill_name, params) in enumerate(STEPS, start=1):
        if idx < args.from_step:
            continue
        params = params if isinstance(params, dict) else {{}}
        print(f"[RUN ] {{idx}} {{skill_name}}")
        if args.dry_run:
            report.append({{"step": idx, "skill": skill_name, "success": True, "dry_run": True}})
            continue
        try:
            if skill_name == "load_trajectory":
                step_load_trajectory(params, state)
            elif skill_name == "run_volume_endpoint_guest_analysis":
                step_run_volume_endpoint_guest_analysis(params, state)
            elif skill_name == "compare_simulation_scores":
                step_compare_simulation_scores(params, state)
            else:
                raise NotImplementedError(f"No src/ replay mapping for skill '{{skill_name}}'")
            report.append({{"step": idx, "skill": skill_name, "success": True}})
            print(f"[ OK ] {{idx}} {{skill_name}}")
        except Exception as exc:
            ok_all = False
            msg = f"{{type(exc).__name__}}: {{exc}}"
            report.append({{"step": idx, "skill": skill_name, "success": False, "error": msg}})
            print(f"[FAIL] {{idx}} {{skill_name}}: {{msg}}", file=sys.stderr)
            if not args.continue_on_error:
                break

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Replay report written: {{args.report_json}}")

    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
"""

    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass


_registry = get_default_registry()
_registry.register(ExportHpcBatchScriptsSkill())
