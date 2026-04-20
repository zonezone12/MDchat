"""Export HPC-oriented batch scripts from the current session summary."""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING

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
        "Produces a Markdown brief, manifest template, Python batch driver, bash helpers, "
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

        artifacts: dict[str, str] = {
            "hpc_readme": readme_path,
            "manifest_example": manifest_example,
            "run_manifest": py_path,
            "run_one_trajectory": sh_single,
            "rank_outputs": sh_rank,
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


_registry = get_default_registry()
_registry.register(ExportHpcBatchScriptsSkill())
