# MDChat

**LLM-powered, code-free Molecular Dynamics trajectory analysis.**

MDChat lets chemist researchers analyze MD simulation data through natural language conversation. Instead of writing Python scripts, describe what you want to learn from your simulation — MDChat translates your questions into analytical workflows and interprets the results in chemically meaningful language.

```
You> I loaded a 500 ns nanocube simulation. Did the iodide enter the cage?

MDChat> I'll track the guest molecule and check for entry events.
  >> load_trajectory(topology='nanocube.prmtop', trajectory='nanocube',format="TRJ")
  OK load_trajectory (3.2s)
  >> score_simulation()
  OK score_simulation (1.1s)

┌─ MDChat ──────────────────────────────────────────────────┐
│ The iodide entered the cage at frame 2840 (142 ns) and    │
│ remained inside for 38 ns before exiting at frame 3600.   │
│ The simulation scores 0.82/1.0 — a meaningful event with  │
│ strong endpoint-volume correlation (r = 0.87).            │
│                                                           │
│ Generated: output/simulation_score.csv                    │
└───────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone and install

```bash
git clone -b MDchat https://github.com/zonezone/MD_analysis
cd MD_analysis
```

Install with [mamba](https://mamba.readthedocs.io/) (recommended for MDAnalysis/RDKit):

```bash
mamba create -n mdchat python=3.10
conda activate mdchat
mamba install mdanalysis rdkit numpy pandas scipy scikit-learn matplotlib imageio
pip install -e ".[chat]"
```

Or with pip only (if MDAnalysis and RDKit are already available):

```bash
pip install -e ".[all]"
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY for Claude (default), or GEMINI_API_KEY / GOOGLE_API_KEY
# with MDCHAT_PROVIDER=gemini for Google Gemini (requires pip install -e ".[gemini]")
```

### 3. Run

**Terminal:**
```bash
mdchat
```

**VS Code / Cursor:** Open the project folder and press **F5** — the launch configuration starts MDChat in the integrated terminal with `.env` loaded automatically.

---

## How It Works

MDChat uses an LLM (Claude via Anthropic, or Gemini via `google-genai`) as a **reasoning and orchestration** layer, not a code generator. Each analytical capability is wrapped as a **Skill** — a self-describing, validated unit that the LLM can discover, parameterize, and chain via structured tool use.

```
User Question
    │
    ▼
┌────────────────────┐
│  LLM Reasoning     │  Parses intent, selects skills,
│  (Claude / Gemini) │  resolves parameters, chains
└────────┬───────────┘  execution in dependency order
         │ tool_use
         ▼
┌────────────────────┐
│  Skill Registry    │  29 skills registered, each with
│                    │  typed parameters and prerequisites
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Analysis Modules  │  MDAnalysis, RDKit, NumPy,
│  (existing code)   │  scikit-learn — validated pipeline
└────────┬───────────┘
         │ SkillResult
         ▼
┌────────────────────┐
│  LLM Narration     │  Interprets numbers as chemistry,
│                    │  suggests follow-up analyses
└────────────────────┘
```

This design is:
- **Safe** — no arbitrary code execution; only pre-defined skill calls.
- **Reproducible** — every skill call is logged with exact parameters.
- **Auditable** — researchers can inspect exactly what was computed.

---

## Available Skills

| Skill | Description | Requires |
|-------|-------------|----------|
| `load_trajectory` | Load topology + trajectory, optionally align to reference frame | file paths |
| `compute_rmsd` | RMSD over trajectory for a given atom selection | loaded trajectory |
| `find_endpoints` | Identify molecular endpoints via convex hull / graph farness (RDKit) | loaded trajectory |
| `compute_endpoint_distances` | Pairwise endpoint distances across all frames | loaded trajectory |
| `plot_timeseries` | Plot any computed array (RMSD, Rg, volume, ...) as PNG | computed data |
| `plot_endpoint_distances` | Visualize endpoint distance evolution | endpoint distances |
| `score_simulation` | Score simulation quality (0-1) on guest entry, volume, correlations | loaded trajectory |

Use `/skills` in the chat to see live status (ready vs. missing prerequisites).

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/load <topo> <traj>` | Quick-load a trajectory |
| `/status` | Show loaded data and computed results |
| `/skills` | List all skills and their readiness |
| `/reset` | Clear conversation history (keeps loaded data) |
| `/help` | Show help |
| `/quit` | Exit |

---

## Project Structure

```
MD_analysis/
├── src/
│   ├── mdchat/                  # MDChat platform
│   │   ├── skill.py             #   Skill ABC, Parameter, SkillResult
│   │   ├── registry.py          #   Skill discovery and tool schema generation
│   │   ├── context.py           #   Per-session analysis state
│   │   ├── llm.py               #   Claude / Gemini chat engines + factory
│   │   ├── gemini_engine.py     #   Gemini (google-genai) tool loop
│   │   ├── engine_common.py     #   Shared system prompt + skill runner
│   │   ├── cli.py               #   Rich terminal interface
│   │   ├── __main__.py          #   Entry point (mdchat command)
│   │   └── skills/              #   Skill implementations
│   │       ├── trajectory.py    #     load_trajectory
│   │       ├── metrics.py       #     compute_rmsd
│   │       ├── endpoints.py     #     find_endpoints, endpoint_distances
│   │       ├── plotting.py      #     plot_timeseries, plot_endpoints
│   │       └── scoring.py       #     score_simulation
│   │
│   ├── AlignedTrajectory/       # Trajectory alignment
│   ├── TrajectoryMetrics/       # RMSD, RMSF, Rg, PCA, strain
│   ├── ClusteringAnalysis/      # HDBSCAN / KMeans frame clustering
│   ├── EndpointAnalyzer/        # Endpoint finding and distance analysis
│   ├── VolumeAnalyzer/          # Voxel-based volume and cavity detection
│   ├── FrameSelection/          # Frame scoring and simulation quality
│   ├── Plotter/                 # Datashader / matplotlib visualizations
│   ├── task/                    # GSA nanocube-specific analysis
│   ├── TrajectoryIterator/      # Observer-pattern trajectory iteration
│   ├── Aggregator/              # Parallel result merging
│   ├── FrameGatherer/           # Efficient coordinate collection
│   ├── FrameProcessor/          # Frame-level metric computation
│   ├── MetricRegistry/          # Pluggable metric system
│   ├── FileIO/                  # PDB export
│   └── utils/                   # RDKit helpers, guest tracking, monitoring
│
├── scripts/                     # CLI scripts for batch analysis
├── legacy/                      # Deprecated monolithic workflow
├── .vscode/                     # VS Code / Cursor launch, tasks, settings
├── .cursor/rules/               # Cursor AI rules for this project
├── .env.example                 # API key template
├── pyproject.toml               # Python packaging (pip install -e .)
└── requirements.txt             # Flat dependency list (legacy)
```

---

## Adding New Skills

Each skill is a Python class that wraps an existing analysis module. Create a file in `src/mdchat/skills/`, define the class, and register it:

```python
from src.mdchat.skill import Skill, Parameter, ParamType, SkillResult
from src.mdchat.registry import get_default_registry


class ComputeRgSkill(Skill):
    name = "compute_rg"
    description = "Compute radius of gyration over the trajectory."
    category = "metrics"
    parameters = [
        Parameter("selection", ParamType.ATOM_SELECTION,
                  "Atom selection string", required=False, default="protein"),
    ]
    requires = ["universe"]
    produces = ["rg_array"]

    def execute(self, context, **params):
        import numpy as np
        from src.TrajectoryMetrics import TrajectoryMetrics

        u = context.universe
        sel = params.get("selection", "protein")
        tm = TrajectoryMetrics()
        rg = tm.radius_of_gyration(u, sel)

        context.set("rg_array", rg)
        mean_rg = float(np.nanmean(rg))

        return SkillResult(
            success=True,
            data={"rg_array": rg},
            summary=f"Radius of gyration: mean {mean_rg:.2f} A over {len(rg)} frames.",
        )


get_default_registry().register(ComputeRgSkill())
```

Then add `from . import your_module` in `src/mdchat/skills/__init__.py`.

Key conventions:
- Import heavy libraries (`numpy`, `MDAnalysis`) inside `execute()`, not at module level.
- Always set `summary` to a human-readable string the LLM can relay to the user.
- Put generated files (plots, CSVs) in `SkillResult.artifacts`.

---

## Configuration

### Environment variables (`.env`)

| Variable | Description | Default |
|----------|-------------|---------|
| `MDCHAT_PROVIDER` | `anthropic` (Claude) or `gemini` | `anthropic` |
| `ANTHROPIC_API_KEY` | Anthropic API key (Claude) | — |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Google AI Studio key (Gemini) | — |
| `MDCHAT_MODEL` | Model id for the active provider | Claude: `claude-sonnet-4-20250514`; Gemini: `gemini-2.5-flash` |
| `MDCHAT_OUTPUT_DIR` | Output directory for artifacts | system temp dir |

### CLI arguments

```
mdchat [--provider anthropic|gemini] [--api-key KEY] [--model MODEL] [--output-dir DIR]
```

CLI arguments override environment variables.

---

## Analysis Modules Reference

The following modules form the computational backend that MDChat skills wrap. They can also be used directly in Python scripts.

### Core Classes

| Module | Class | Description |
|--------|-------|-------------|
| `src.AlignedTrajectory` | `AlignedTrajectory` | Trajectory alignment to reference frame |
| `src.TrajectoryMetrics` | `TrajectoryMetrics` | RMSD, RMSF, radius of gyration, PCA, strain, contacts |
| `src.ClusteringAnalysis` | `ClusteringAnalysis` | HDBSCAN/KMeans frame clustering, change-point detection |
| `src.FrameSelection` | `FrameSelection` | Frame scoring, selection, simulation quality scoring |
| `src.EndpointAnalyzer` | `EndpointAnalyzer` | Endpoint distances, metrics, volume correlation |
| `src.EndpointAnalyzer` | `EndpointsFinder` | Convex hull + graph farness endpoint detection |
| `src.VolumeAnalyzer` | `VolumeAnalyzer` | Voxel-based volume, cavity detection, 3D visualization |
| `src.Plotter` | `Plotter` | Time series, correlation heatmaps, endpoint plots, GIFs |
| `src.task` | `GSAnalyzer` | Nanocube geometry: face planarity, edge lengths, volume |
| `src.TrajectoryIterator` | `TrajectoryIterator` | Observer-pattern single-pass iteration with Dask/multiprocessing |
| `src.Aggregator` | `ResultsGroup` | Declarative parallel result merging |
| `src.FileIO` | `FileIO` | Frame export as PDB |

### Programmatic Usage

```python
import MDAnalysis as mda
from src.AlignedTrajectory import AlignedTrajectory
from src.TrajectoryMetrics import TrajectoryMetrics
from src.EndpointAnalyzer import EndpointAnalyzer, EndpointsFinder
from src.VolumeAnalyzer import VolumeAnalyzer
from src.Plotter import Plotter

u = mda.Universe("topology.prmtop", "trajectory.nc")
aligned = AlignedTrajectory(u)
u = aligned.get_aligned_universe()

# RMSD
tm = TrajectoryMetrics()
rmsd = tm.compute_rmsd(u, "not water and not name H*")

# Endpoints
ea = EndpointAnalyzer()
ef = EndpointsFinder(step_back_from_terminals=True)
residues = [f"resid {i}" for i in range(1, 7)]
dists = ea.compute_endpoint_distances(u, residues, ef)

# Volume
va = VolumeAnalyzer(u, selection="resname GSA", spacing=1.0)
target_vol, cavity_vol, _, _ = va.compute_frame(0, return_masks=True)
```

### Batch Scripts

Scripts in `scripts/` provide CLI entry points for batch processing:

| Script | Purpose |
|--------|---------|
| `run_trajectory_analysis.py` | Full workflow: metrics, clustering, frame selection, endpoints, volume |
| `run_volume_endpoint_analysis.py` | Focused endpoint + volume correlation analysis |
| `run_volume_endpoint_guest_analysis.py` | Endpoint + volume + guest tracking + simulation scoring |
| `script_in_hpc.py` | Batch processing across multiple systems on SLURM |

Run any script with `--help` for full argument documentation:

```bash
python scripts/run_volume_endpoint_guest_analysis.py --help
```

---

## Dependencies

### Required (installed by `pip install -e .`)

numpy, pandas, scipy, MDAnalysis, scikit-learn, matplotlib, Pillow, imageio

### Optional groups

| Group | Install command | Packages |
|-------|----------------|----------|
| `chat` | `pip install -e ".[chat]"` | anthropic, rich, python-dotenv |
| `gemini` | `pip install -e ".[gemini]"` | google-genai |
| `viz` | `pip install -e ".[viz]"` | datashader, plotly, scikit-image |
| `rdkit` | `pip install -e ".[rdkit]"` | rdkit |
| `all` | `pip install -e ".[all]"` | everything above |

For the best experience, install MDAnalysis and RDKit via conda/mamba first, then install the rest with pip.

---

## License

MIT — feel free to adapt for publications (cite [MDAnalysis](https://www.mdanalysis.org/) and [RDKit](https://www.rdkit.org/)).
