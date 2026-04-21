"""
Shared pieces for LLM backends (Anthropic, Gemini): system prompt and skill execution.
"""

from __future__ import annotations

import logging
from typing import Any

from .context import AnalysisContext
from .registry import SkillRegistry
from .skill import SkillResult

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 15
MAX_TOKENS = 4096

# Final model turn sometimes has no text (API shape, safety, etc.).
MODEL_EMPTY_TEXT_FALLBACK = (
    "The model returned no visible text. If a skill just finished, check "
    "**/status** and your output folder. Try asking again, or **/help**."
)

BANNER = r"""  __  __ ____   ____ _           _
 |  \/  |  _ \ / ___| |__   __ _| |_
 | |\/| | | | | |   | '_ \ / _` | __|
 | |  | | |_| | |___| | | | (_| | |_
 |_|  |_|____/ \____|_| |_|\__,_|\__|

 LLM-Powered Molecular Dynamics Analysis"""

HELP_TEXT = """**Commands:**
- Type a question in natural language to analyze your trajectory.
- `/load <topology> <trajectory>` -- Quick-load files into the session.
- `/status` -- Show current analysis state.
- `/skills` -- List available skills.
- `/model` -- Show the model list; `/model <n>` pick by number; `/model <id>` set API model id.
- `/history` -- Show recent chat from this session’s markdown transcript.
- `/reset` -- Clear conversation history and saved chat file (keeps loaded trajectory data).
- `/help` -- Show this help message.
- `/quit` or `/exit` -- Exit MDChat.

**Model:** Set `MDCHAT_MODEL` in `.env`, pass `--model` at startup, or `/model` for the picker. Optional `MDCHAT_MODEL_CHOICES=id1,id2,...` replaces the built-in list for your provider.

**Chat history:** With a fixed session directory (`--output-dir` / `MDCHAT_OUTPUT_DIR` pointing at one folder), turns are saved to `mdchat_conversation.json` (full context for the LLM) and `mdchat_transcript.md` (readable log). Reopening mdchat with the same directory reloads the conversation automatically.

**Quick start:**
1. Set your API key in `.env` — `ANTHROPIC_API_KEY` (Claude) or `GEMINI_API_KEY` / `GOOGLE_API_KEY` (Gemini). Set `MDCHAT_PROVIDER=gemini` for Gemini (see `.env.example`).
2. Type: `/load myfile.prmtop myfile.nc`
3. Ask: "Compute the RMSD and plot it."
"""


def format_welcome(
    n_skills: int, output_dir: str, provider: str, model: str | None = None
) -> str:
    """Return a plain-text welcome message for any UI to display."""
    model_line = model if model else "(provider default)"
    return (
        BANNER
        + f"\n\n{n_skills} skills registered."
        + f"\nOutput directory: {output_dir}"
        + f"\nProvider: {provider}"
        + f"\nModel: {model_line}\n"
        + "\n" + HELP_TEXT
    )


SYSTEM_PROMPT_TEMPLATE = """\
You are **MDChat**, an expert assistant for Molecular Dynamics trajectory analysis.

Your role is to help chemist researchers analyze their MD simulation data through \
natural language conversation. You have access to a set of analytical *skills* that \
you can call as tools. Each skill performs a specific analysis on trajectory data.

## How to operate

1. **Understand** the user's question and decide which skill(s) to invoke.
2. **Check prerequisites**: look at the current analysis state below to see what \
data is already available. If a required prerequisite is missing, call the skill \
that produces it first (e.g., load a trajectory before computing RMSD).
3. **Call skills** with appropriate parameters extracted from the conversation. \
If you're unsure about a parameter value, ask the user instead of guessing. \
**After `load_trajectory` succeeds:** Your **very next** reply to the user **must** \
be normal, visible chat text (not meta-instructions to yourself). In that message: \
(1) briefly confirm the load (frames, atoms); (2) quote the **suggested** \
`main_selection` string from the tool result; (3) **ask them directly** whether \
to use that suggestion for alignment and downstream metrics, or to name a \
different MDAnalysis selection (point them to **`/status`** and the residue \
catalog). **Do not** call **`set_main_selection`** until they have confirmed or \
given a selection — never silently assume. For a one-shot load with alignment, \
use **`load_trajectory`** with `align=true` and **`align_selection`** only when \
the user already specified the selection in chat. \
**Trajectory workflow (mandatory):** If the user wants **two or more** of RMSD, Rg, \
contacts, endpoint distances, or GSA nanocube metrics on the **same** loaded \
trajectory **in one turn**, you **must** call **run_trajectory_observer_pass** \
**exactly once** with **all** matching `include_*` / contact / endpoint / GSA \
parameters set in that single call — **do not** chain multiple `compute_*` or \
**gsa_nanocube_metrics** invocations that each reread the trajectory. If only **one** \
of those analyses is needed, **run_trajectory_observer_pass** with just that flag \
(or a single `compute_*` / **gsa_nanocube_metrics**) is fine — same physics, one pass. \
For parallel frame batches on the observer pass, set **n_jobs**>1 or **use_dask** as needed.
4. **Interpret results** in chemically meaningful language. Don't just repeat \
numbers — explain what they mean for the molecular system.
5. **Suggest follow-up** analyses when appropriate.
6. When the user wants **batch/HPC** scripts to analyze **many trajectories** and **rank** results, call **`export_hpc_batch_scripts`** (writes manifest + `run_manifest_trajectories.py` + ranking shell + optional Slurm).

## Current analysis state

{context_state}

## Guidelines

- Never fabricate analysis results. Only report what the skills return.
- When a skill fails, explain the error and suggest how to fix it.
- When multiple skills are needed, chain them in the correct dependency order.
- Be concise but scientifically precise.
- If the user asks something outside MD analysis, politely redirect.
- Reference generated artifact file paths so the user can find their plots/data.
- After **load_trajectory**, the session includes a **residue selection catalog** (see analysis state): use those `selection` strings for skills instead of guessing `resname`/`resid`.
- Write for the **human user**: never paste internal rubrics like “ask the user to…”; speak in second person (“Should I use … for alignment?”).
"""


class EngineMixin:
    """Shared skill runner and system prompt (used by Anthropic and Gemini engines)."""

    registry: SkillRegistry
    context: AnalysisContext
    callback: Any

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT_TEMPLATE.format(
            context_state=self.context.get_state_summary(),
        )

    def _execute_skill(self, tool_name: str, tool_input: dict) -> tuple[str, bool]:
        skill = self.registry.get(tool_name)
        if skill is None:
            msg = f"Error: Unknown skill '{tool_name}'."
            self.callback.on_skill_end(tool_name, False, msg)
            return msg, False

        ok, reason = skill.validate(self.context)
        if not ok:
            msg = f"Skill '{tool_name}' cannot run: {reason}"
            self.callback.on_skill_end(tool_name, False, msg)
            return msg, False

        self.callback.on_skill_start(tool_name, tool_input)
        logger.info("Executing skill '%s' with params: %s", tool_name, tool_input)

        import time as _time
        _t0 = _time.monotonic()
        try:
            result: SkillResult = skill.execute(self.context, **tool_input)
        except Exception as exc:
            logger.exception("Skill '%s' raised an unhandled exception", tool_name)
            result = SkillResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                summary=f"Skill '{tool_name}' failed with an unexpected error: {exc}",
            )
        _elapsed = _time.monotonic() - _t0

        self.context.record_execution(
            tool_name,
            params=tool_input,
            elapsed_s=_elapsed,
            success=result.success,
            summary=result.summary,
            artifacts=result.artifacts if result.artifacts else None,
        )

        if result.artifacts:
            for name, path in result.artifacts.items():
                self.context.add_artifact(name, path)

        if result.success:
            for key in skill.produces:
                if key in result.data:
                    self.context.set(key, result.data[key])

        self.callback.on_skill_end(tool_name, result.success, result.summary)
        return result.to_tool_result(), result.success


def tool_rounds_exceeded_message() -> str:
    return (
        "I reached the maximum number of tool-use rounds. "
        "Please try a simpler request or break it into steps."
    )
