# Research Proposal: LLM-Powered Platform for Code-Free Molecular Dynamics Trajectory Analysis

---

## 1. Title

**MDChat: An LLM-Orchestrated, Skill-Based Platform for Accessible Molecular Dynamics Trajectory Analysis**

---

## 2. Abstract

Molecular dynamics (MD) simulations are indispensable in computational chemistry, yet extracting meaningful insights from trajectory data remains a bottleneck that demands significant programming expertise. We propose **MDChat**, a platform that integrates large language models (LLMs) with a modular *skill* architecture to enable chemist researchers to perform sophisticated MD trajectory analyses through natural language interaction alone. Building on an existing, validated analysis pipeline for supramolecular nanocube systems — encompassing endpoint detection, volume analysis, guest-host tracking, and simulation quality scoring — we will develop an extensible framework where each analytical capability is encapsulated as a composable *skill* that the LLM can discover, parameterize, chain, and execute on behalf of the user. The platform targets a paradigm shift: from "write code to analyze trajectories" to "describe what you want to learn from your simulation."

---

## 3. Problem Statement

### 3.1 The Analysis Bottleneck in Computational Chemistry

MD simulations generate enormous trajectory datasets (tens of thousands of frames, millions of atoms). Extracting chemically meaningful information — conformational transitions, host-guest encapsulation events, correlated structural motions — requires researchers to:

1. **Write substantial code** using libraries such as MDAnalysis, RDKit, NumPy, and scikit-learn.
2. **Understand software engineering patterns** (observer patterns, parallel processing, data aggregation) that are orthogonal to their domain expertise.
3. **Make algorithmic decisions** (clustering method, endpoint-finding heuristics, correlation thresholds) that require both chemical intuition and computational knowledge.
4. **Debug and iterate** through cycles of code modification, re-execution, and result interpretation.

This creates a two-class system: researchers who can code perform deep trajectory analysis, while those who cannot are limited to black-box tools with narrow functionality.

### 3.2 Limitations of Current Approaches

| Approach | Limitation |
|----------|-----------|
| GUI-based tools (VMD, UCSF Chimera) | Predefined analyses only; no custom metrics or workflows |
| Scripting libraries (MDAnalysis, MDTraj) | Require Python proficiency and domain-specific API knowledge |
| Workflow managers (KNIME, Galaxy) | Rigid pipelines; poor support for exploratory, iterative analysis |
| Notebook templates | One-size-fits-all; parameter tuning still requires code editing |

### 3.3 The Opportunity

Recent advances in LLMs have demonstrated their capacity to reason about scientific concepts, generate code, and orchestrate multi-step workflows. By coupling an LLM with a well-structured library of domain-specific analytical *skills*, we can create a system where the LLM acts as an intelligent intermediary — translating a chemist's questions into executable analysis pipelines.

---

## 4. Existing Work: The MD_analysis Pipeline

This proposal builds on a functional, modular Python pipeline developed for analyzing MD trajectories of Gear-Shape Amphiphile (GSA) nanocubes. The current system provides:

### 4.1 Core Analytical Capabilities

| Module | Capability |
|--------|-----------|
| **EndpointAnalyzer** | Identify molecular endpoints via convex hull, graph farness, and ring-aware algorithms (RDKit); compute pairwise distances across frames; correlate with structural metrics |
| **VolumeAnalyzer** | Voxel-based target volume and cavity detection with solvent-excluded surfaces; interactive 3D visualization |
| **GSAnalyzer** | Nanocube-specific geometry: face planarity, edge lengths, volume; automated face clustering via KMeans |
| **TrajectoryIterator** | Single-pass trajectory iteration with an observer pattern; parallel execution via Dask or multiprocessing |
| **ResultsGroup** | Declarative result aggregation for parallel worker output merging |
| **Guest Tracking** | Distance-based and volume-based guest entry/exit detection; residence statistics and timeline visualization |
| **TrajectoryMetrics** | RMSD, RMSF, radius of gyration, PCA, strain analysis, contact distances |
| **ClusteringAnalysis** | HDBSCAN/KMeans frame clustering; change-point detection (ruptures) |
| **FrameSelection** | Multi-criteria frame scoring; simulation quality scoring (0–1 scale) |
| **Plotter** | Time-series plots, correlation heatmaps, endpoint visualizations, animated GIFs |

### 4.2 Architecture Strengths as a Foundation

The pipeline already exhibits properties that make it amenable to LLM orchestration:

- **Modularity**: Each analysis is encapsulated in a self-contained class with a clear API.
- **Observer pattern**: The `TrajectoryIterator` / `FrameObserver` design allows composing multiple analyses into a single trajectory pass.
- **Declarative aggregation**: `ResultsGroup` merges parallel results via named strategies, enabling the LLM to reason about data flow without managing concurrency.
- **Parameterized entry points**: CLI scripts already accept structured arguments (`--endpoint_residues`, `--cube_faces`, `--guest_sel`), providing a natural mapping from natural language to function calls.

---

## 5. Proposed Platform: MDChat

### 5.1 Vision

MDChat will allow a chemist to interact with their MD trajectory data through conversations such as:

> **User**: "I ran a 500 ns simulation of a nanocube with iodide. Did the iodide enter the cage? If so, when, and which faces opened to let it in?"
>
> **MDChat**: *Loads the trajectory, identifies the nanocube and guest selections, runs guest-entry tracking (distance-based), computes face-resolved endpoint distances, correlates endpoint pair expansion with entry events, and returns:*
> "The iodide entered the cage at 142 ns through an opening between faces 2 and 5. The endpoint pair (Res3-EP1, Res8-EP2) expanded by 4.2 Å, correlating with entry (r = 0.87). It remained inside for 38 ns before exiting at 180 ns. Here are the timeline and correlation plots."

### 5.2 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Interface                      │
│          (Chat UI / Jupyter / CLI / Web App)             │
└─────────────────────┬───────────────────────────────────┘
                      │  Natural Language
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   LLM Reasoning Core                     │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Intent       │  │  Skill       │  │  Parameter    │  │
│  │  Parser       │→ │  Selector    │→ │  Resolver     │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                            │             │
│  ┌──────────────┐  ┌──────────────┐        ▼             │
│  │  Result       │← │  Execution   │← ┌───────────────┐ │
│  │  Narrator     │  │  Planner     │  │  Skill DAG    │  │
│  └──────────────┘  └──────────────┘  │  Composer      │  │
│                                       └───────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │  Structured Skill Calls
                      ▼
┌─────────────────────────────────────────────────────────┐
│                    Skill Registry                        │
│                                                          │
│  ┌────────────┐ ┌────────────┐ ┌────────────────────┐   │
│  │  Endpoint   │ │  Volume    │ │  Guest Tracking    │   │
│  │  Analysis   │ │  Analysis  │ │  Skill             │   │
│  └────────────┘ └────────────┘ └────────────────────┘   │
│  ┌────────────┐ ┌────────────┐ ┌────────────────────┐   │
│  │  Clustering │ │  Metrics   │ │  Frame Selection   │   │
│  │  Skill      │ │  Skill     │ │  Skill             │   │
│  └────────────┘ └────────────┘ └────────────────────┘   │
│  ┌────────────┐ ┌────────────┐ ┌────────────────────┐   │
│  │  Plotting   │ │  PCA       │ │  Custom / User-    │   │
│  │  Skill      │ │  Skill     │ │  Defined Skills    │   │
│  └────────────┘ └────────────┘ └────────────────────┘   │
│                                                          │
└─────────────────────┬───────────────────────────────────┘
                      │  Python API Calls
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Computational Backend                      │
│                                                         │
│  MDAnalysis  │  RDKit  │  NumPy/SciPy  │  scikit-learn  │
│  Dask        │  Plotly │  matplotlib   │  HPC/SLURM     │
└─────────────────────────────────────────────────────────┘
```

### 5.3 The Skill Abstraction

Each analytical capability is wrapped as a **Skill** — a self-describing, LLM-callable unit:

```python
class Skill:
    name: str                    # "endpoint_volume_correlation"
    description: str             # Natural language description for LLM
    parameters: ParameterSchema  # Typed parameters with defaults and constraints
    requires: list[str]          # Prerequisite skills (e.g., ["load_trajectory", "find_endpoints"])
    produces: list[str]          # Output artifacts (e.g., ["correlation_df", "correlation_plot"])

    def validate(self, context) -> bool:
        """Check if this skill can run given current analysis state."""

    def execute(self, context, **params) -> SkillResult:
        """Run the analysis and return structured results."""

    def summarize(self, result) -> str:
        """Generate a natural language summary of results for the LLM."""
```

**Key design principles:**

- **Self-describing**: Each skill carries enough metadata for the LLM to understand when and how to use it, without hardcoded prompt engineering per skill.
- **Composable**: Skills declare dependencies and outputs, enabling the LLM to automatically chain them into workflows (directed acyclic graphs).
- **Parameterized with chemical semantics**: Parameters use domain-specific types (`AtomSelection`, `ResidueRange`, `DistanceThreshold`) that the LLM can map from natural language.
- **Extensible**: Researchers can create new skills by wrapping existing analysis functions, without modifying the platform core.

### 5.4 LLM Integration Strategy

Rather than using the LLM to generate arbitrary code (which is fragile and insecure), MDChat uses the LLM as a **reasoning and orchestration** layer:

1. **Intent Parsing**: The LLM interprets the user's question and maps it to one or more analytical intents (e.g., "Did the guest enter?" → guest tracking + temporal analysis).
2. **Skill Selection**: Given the intent, the LLM queries the Skill Registry to find matching skills and resolves dependency chains.
3. **Parameter Resolution**: The LLM extracts parameters from context — user utterances, trajectory metadata, prior results — and fills skill parameter schemas. Ambiguities are resolved by asking the user.
4. **Execution Planning**: The LLM constructs an execution plan (skill DAG), optimizing for single-pass trajectory reading where possible (leveraging the observer pattern).
5. **Result Narration**: After execution, the LLM interprets numerical results, identifies key findings, and presents them in chemically meaningful language with supporting visualizations.

### 5.5 Tool-Use / Function-Calling Protocol

The LLM interacts with skills through a structured function-calling interface (compatible with OpenAI/Anthropic tool-use APIs):

```json
{
  "skill": "guest_tracking",
  "parameters": {
    "topology": "nanocube.prmtop",
    "trajectory": "nanocube.nc",
    "host_selection": "resname GSA",
    "guest_selection": "resname IOD",
    "method": "distance",
    "threshold": "auto"
  }
}
```

This approach is:
- **Safer** than code generation (no arbitrary execution).
- **Reproducible** (skill calls are logged and replayable).
- **Auditable** (researchers can inspect the exact analysis steps).

---

## 6. Research Objectives

### Objective 1: Skill Framework Design and Implementation
Design and implement the `Skill` abstraction layer that wraps existing analytical modules into LLM-callable units. This includes:
- A schema language for skill metadata (parameters, dependencies, outputs).
- An automatic dependency resolver and execution planner.
- A context manager that tracks analysis state across a conversation.
- Migration of all existing modules (EndpointAnalyzer, VolumeAnalyzer, GSAnalyzer, etc.) into skill form.

### Objective 2: LLM Orchestration Engine
Develop the reasoning layer that connects natural language to skill execution:
- Prompt engineering and fine-tuning strategies for MD-domain intent recognition.
- Multi-step planning with intermediate result inspection.
- Error recovery and parameter suggestion when analyses fail.
- Benchmark against expert-written analysis scripts for correctness and completeness.

### Objective 3: Generalization Beyond Nanocubes
Extend the skill library beyond GSA nanocubes to general MD analysis:
- Protein dynamics skills (secondary structure, hydrogen bonding, binding pocket volume).
- Membrane simulation skills (area per lipid, order parameters, thickness).
- Small molecule solvation skills (radial distribution functions, coordination numbers).
- Community-contributed skill templates and a skill marketplace.

### Objective 4: User Study and Validation
Conduct user studies with chemist researchers (non-programmers) to evaluate:
- Task completion rate compared to scripting-based approaches.
- Quality of extracted insights (validated against expert analysis).
- Time-to-insight reduction.
- User satisfaction and trust in LLM-generated analysis.

### Objective 5: Scalability and HPC Integration
Ensure the platform operates on real-world simulation data:
- Integration with SLURM-based HPC clusters for large trajectories.
- Streaming analysis for trajectories too large to fit in memory.
- Leveraging the existing `TrajectoryIterator` observer pattern and `ResultsGroup` aggregation for parallel skill execution.

---

## 7. Methodology

### Phase 1: Skill Abstraction Layer (Months 1–4)

**Task 1.1 — Skill Schema Definition**
Define a declarative schema (YAML/JSON) for skill metadata:
```yaml
name: endpoint_volume_correlation
version: "1.0"
description: >
  Compute Pearson correlation between pairwise endpoint distances
  and nanocube volume over the trajectory. Identifies which structural
  motions drive volume changes.
category: correlation_analysis
parameters:
  endpoint_residues:
    type: list[AtomSelection]
    required: true
    description: Residue selections for endpoint detection
  volume_source:
    type: enum[cube_approximation, voxel_based]
    default: cube_approximation
    description: Method for computing host volume
  correlation_threshold:
    type: float
    default: 0.7
    range: [0.0, 1.0]
    description: Minimum |r| to flag a pair as significant
requires:
  - load_trajectory
  - find_endpoints
  - compute_volume
produces:
  - correlation_dataframe
  - key_endpoint_pairs
  - correlation_plot
```

**Task 1.2 — Skill Registry and Discovery**
Implement a registry that:
- Auto-discovers skills from annotated Python modules.
- Generates LLM-compatible tool descriptions from skill schemas.
- Resolves dependency graphs and suggests execution orders.

**Task 1.3 — Module Migration**
Wrap each existing module as a skill:

| Existing Module | Skill(s) |
|----------------|----------|
| `EndpointsFinder` | `find_endpoints` |
| `EndpointAnalyzer` | `endpoint_distances`, `endpoint_volume_correlation`, `endpoint_variation`, `key_endpoint_pairs` |
| `VolumeAnalyzer` | `compute_voxel_volume`, `detect_cavities`, `visualize_volume_3d` |
| `GSAnalyzer` | `nanocube_metrics`, `auto_detect_faces`, `guest_tracking` |
| `TrajectoryMetrics` | `compute_rmsd`, `compute_rmsf`, `compute_rg`, `run_pca`, `compute_strain` |
| `ClusteringAnalysis` | `cluster_frames`, `detect_change_points` |
| `FrameSelection` | `score_frames`, `select_representative_frames`, `score_simulation` |
| `Plotter` | `plot_timeseries`, `plot_correlation`, `plot_endpoints`, `generate_animation` |
| `AlignedTrajectory` | `align_trajectory` |
| `FileIO` | `export_frames_pdb`, `export_csv` |

### Phase 2: LLM Orchestration Engine (Months 3–8)

**Task 2.1 — Conversation-to-Workflow Mapping**
Develop the prompt architecture that enables the LLM to:
- Parse chemical questions into analytical intents.
- Map intents to skill sequences.
- Handle ambiguity ("analyze my simulation" → ask clarifying questions).
- Maintain conversation context across multi-turn interactions.

**Task 2.2 — Execution Runtime**
Build the execution engine:
- Sandboxed skill execution with resource limits.
- Result caching to avoid redundant computation.
- Progress reporting for long-running analyses.
- Error handling with LLM-generated diagnostic suggestions.

**Task 2.3 — Result Interpretation Layer**
Train/prompt the LLM to interpret analysis results in chemical context:
- Convert statistical outputs (correlation coefficients, p-values) into chemical narratives.
- Highlight surprising or significant findings.
- Suggest follow-up analyses based on results.

### Phase 3: Generalization and Skill Expansion (Months 6–12)

**Task 3.1 — General-Purpose MD Skills**
Develop skills for common MD analyses beyond nanocubes:
- Protein RMSD/RMSF, secondary structure evolution, hydrogen bond analysis.
- Membrane simulations: area per lipid, deuterium order parameters.
- Free energy landscape construction from PCA or metadynamics CVs.

**Task 3.2 — Skill Authoring Toolkit**
Create tools for researchers to define new skills:
- A decorator-based Python API (`@md_skill(name=..., requires=..., produces=...)`).
- Automatic parameter schema inference from type annotations.
- Skill testing and validation framework.
- LLM-assisted skill creation: "I want a skill that computes the angle between three atom groups over time."

**Task 3.3 — Skill Marketplace**
Design a community repository for sharing skills:
- Version control and compatibility tracking.
- Peer review and validation badges.
- Usage analytics and citation tracking.

### Phase 4: User Studies and Deployment (Months 10–14)

**Task 4.1 — User Study Design**
Recruit 20–30 chemist researchers across experience levels:
- Group A: MDChat (natural language interface).
- Group B: Traditional scripting (Jupyter notebooks with documentation).
- Tasks: Analyze a test trajectory for conformational changes, guest encapsulation, and key structural drivers.
- Metrics: Completion rate, time, insight quality, user confidence.

**Task 4.2 — Deployment Infrastructure**
- Web application with trajectory upload and interactive chat.
- JupyterHub integration for users who want hybrid interaction.
- HPC job submission for large trajectories.
- Docker containers for reproducible environments.

---

## 8. Expected Outcomes and Deliverables

| Deliverable | Description | Timeline |
|------------|-------------|----------|
| **Skill Framework** (open-source) | Python package for defining, registering, and executing MD analysis skills | Month 4 |
| **MDChat Engine** | LLM orchestration layer with conversation management and skill execution | Month 8 |
| **Skill Library v1** | 30+ skills covering nanocube, protein, and membrane analysis | Month 10 |
| **Skill Authoring Toolkit** | Decorator API and CLI for creating new skills | Month 10 |
| **Web Application** | Chat-based interface for trajectory analysis | Month 12 |
| **User Study Report** | Quantitative evaluation of accessibility and insight quality | Month 14 |
| **Publications** | 2 peer-reviewed papers (platform architecture; user study) | Months 10, 14 |

---

## 9. Innovation and Significance

### 9.1 Scientific Innovation
- **First LLM-orchestrated MD analysis platform** that uses structured skill execution (not code generation) for safety and reproducibility.
- **Chemical-semantic parameter types** that bridge natural language and computational chemistry concepts.
- **Observer-pattern-aware execution planning** that enables the LLM to compose multi-analysis single-pass workflows, a unique optimization for trajectory data.

### 9.2 Broader Impact
- **Democratization**: Enables non-coding chemists to perform analyses previously requiring months of software development.
- **Reproducibility**: Skill-based execution logs provide complete provenance for every analysis.
- **Accelerated discovery**: Reduces time-to-insight from days/weeks to minutes for standard analyses.
- **Community growth**: The skill marketplace creates a shared, peer-reviewed analysis ecosystem.

### 9.3 Relation to Existing AI-for-Science Efforts
MDChat complements rather than competes with:
- **AI-driven MD** (e.g., machine learning force fields): We analyze trajectories *after* simulation, regardless of the MD engine.
- **LLM code assistants** (e.g., Copilot, Cursor): We provide domain-specific, validated analysis rather than general code completion.
- **Automated workflows** (e.g., AiiDA, Fireworks): We focus on interactive, exploratory analysis rather than high-throughput production pipelines.

---

## 10. Technical Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| LLM misinterprets chemical intent | Wrong analysis executed | Confirmation step before execution; skill-level validation; human-in-the-loop design |
| LLM hallucinated parameter values | Incorrect results | Strict schema validation; parameter bounds from chemical knowledge; no free-form code execution |
| Skill composition explosion | Intractable planning for complex queries | Hierarchical skill grouping; meta-skills that bundle common workflows; execution budget limits |
| Performance on large trajectories | Slow interactive experience | Streaming execution with progress updates; result caching; HPC offloading |
| User trust in AI-generated analysis | Low adoption | Transparent execution logs; side-by-side comparison with manual analysis; uncertainty quantification |

---

## 11. Timeline Summary

```
Month:  1   2   3   4   5   6   7   8   9  10  11  12  13  14
        ├───────────────┤
        Phase 1: Skill Abstraction Layer
                    ├───────────────────────┤
                    Phase 2: LLM Orchestration Engine
                                ├───────────────────────┤
                                Phase 3: Generalization & Skill Expansion
                                                    ├───────────────┤
                                                    Phase 4: User Studies & Deployment
Milestones:
        M1 (Month 2): Skill schema finalized, 5 pilot skills migrated
        M2 (Month 4): Full skill registry, all existing modules migrated
        M3 (Month 6): LLM can execute 3-skill workflows from natural language
        M4 (Month 8): Full orchestration engine with error recovery
        M5 (Month 10): 30+ skills, authoring toolkit, first paper submitted
        M6 (Month 12): Web application deployed, user study begins
        M7 (Month 14): User study complete, second paper submitted
```

---

## 12. Budget Justification (Estimated)

| Category | Item | Cost (USD) |
|----------|------|-----------|
| **Personnel** | 1 Postdoc (14 months) — platform development | $84,000 |
| **Personnel** | 1 Graduate RA (14 months) — skill development and user studies | $42,000 |
| **Compute** | LLM API costs (OpenAI/Anthropic, ~$500/month) | $7,000 |
| **Compute** | HPC allocation for trajectory analysis benchmarks | $5,000 |
| **Compute** | Cloud hosting for web application (AWS/GCP) | $6,000 |
| **Travel** | Conference presentations (2 conferences) | $6,000 |
| **Other** | User study participant compensation | $3,000 |
| | **Total** | **$153,000** |

---

## 13. Conclusion

The MD_analysis pipeline demonstrates that sophisticated trajectory analysis can be decomposed into modular, composable units. MDChat proposes to bridge the gap between these computational capabilities and the chemist's natural way of thinking — through questions, not code. By wrapping analytical modules as LLM-orchestrated skills, we create a platform that is simultaneously more accessible to non-programmers, more reproducible than ad-hoc scripts, and more extensible than monolithic tools. The result is a new paradigm for computational chemistry analysis: one where the researcher's time is spent on scientific reasoning, not software engineering.

---

## References

1. Michaud-Agrawal, N., Denning, E. J., Woolf, T. B., & Beckstein, O. (2011). MDAnalysis: A toolkit for the analysis of molecular dynamics simulations. *J. Comput. Chem.*, 32(10), 2319–2327.
2. Landrum, G. (2006). RDKit: Open-source cheminformatics. https://www.rdkit.org/
3. Brown, T. B., et al. (2020). Language models are few-shot learners. *NeurIPS*, 33, 1877–1901.
4. Schick, T., et al. (2023). Toolformer: Language models can teach themselves to use tools. *NeurIPS*, 36.
5. Bran, A. M., et al. (2024). ChemCrow: Augmenting large-language models with chemistry tools. *Nature Machine Intelligence*, 6, 525–535.
6. Boiko, D. A., et al. (2023). Autonomous chemical research with large language models. *Nature*, 624, 570–578.
7. White, A. D. (2023). The future of chemistry is language. *Nature Reviews Chemistry*, 7, 457–458.

---

*Prepared: March 2026*
*Project Repository: MD_analysis (https://github.com/zonezone/MD_analysis)*
