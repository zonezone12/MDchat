# Changepoint Feature-Group Pipeline

**MD_analysis — developer / user guide**

Detect regime changes in feature time series, optionally tune the Pelt penalty, cluster the resulting segments, and produce cohort summaries/plots.

Two entry paths:

| Path | Input | Feature group(s) | Typical question |
|------|-------|------------------|------------------|
| **GSA features** | Pre-computed `*_gsa_features.csv` | `gsa`, `iodine`, `na_water`, `combined` | Which chemical coordinates change together? |
| **Endpoint sites** | Topology + trajectory → `*_endpoint_features.csv` | `endpoint` | Do ring-centroid / tooth distances recover A→B→C1→C2 deformation motifs? |

Logic lives in `src/ChangepointAnalysis/`. Scripts under `scripts/` are thin argparse wrappers. MDChat exposes the same stages as skills.

---

## Prerequisites

### GSA feature path

1. Feature CSVs from `scripts/compute_gsa_features.py` (or equivalent), named `{traj_id}_gsa_features.csv`.
2. Each CSV must include at least `frame` / `time_ps` (optional but recommended) plus the numeric feature columns listed in `src/ChangepointAnalysis/feature_groups.py`.

### Endpoint-site path

1. Shared topology + one or more trajectories.
2. RDKit available (`pip install -e ".[rdkit]"`) for endpoint / ring detection.
3. Optional: an existing `*_gsa_features.csv` directory if you want an `assembly_rmsd_to_ref` overlay (`--rmsd-from`).

```powershell
# From repo root
python scripts/compute_gsa_features.py --help
python scripts/run_endpoint_changepoint.py --help
```

Repo root must be on `PYTHONPATH` (editable install or run from the repo root).

---

## Pipeline overview

```mermaid
flowchart TD
    feats["*_gsa_features.csv<br/>or *_endpoint_features.csv"] --> detect
    detect["detect: Pelt per feature group"] --> tables["all_breakpoints.csv<br/>all_segment_stats.csv<br/>changepoint_timing_comparison.csv"]
    sweep["optional: penalty sweep"] -.->|"recommended penalty"| detect
    tables --> cluster["cluster segments by group"]
    tables --> summarize["cohort tables + plots"]
    cluster --> clusters["clusters/{group}/..."]
    summarize --> plots["plots/*.png + cohort_*.csv"]
```

| Stage | What it does | Main outputs |
|-------|----------------|--------------|
| **Detect** | Multivariate Pelt (ruptures) per feature group | `all_breakpoints.csv`, `all_segment_stats.csv`, timing comparison |
| **Sweep** (optional) | Grid of penalties → elbow recommendation | `penalty_sweep/` summaries + plots |
| **Cluster** | Hierarchical clustering of segment summaries | `clusters/{group}/`, transitions |
| **Summarize** | Cohort Jaccard/regime tables + timeline plots | `cohort_*.csv`, `plots/` |

### Feature groups

| Group | Meaning | Column list |
|-------|---------|-------------|
| `gsa` | Cage / monomer geometry | `GSA_COLS` |
| `iodine` | IOD guest location and contacts | `IODINE_COLS` |
| `na_water` | Cavity water / Na⁺ environment | `NA_WATER_COLS` |
| `combined` | All numeric non-metadata columns | derived from the CSV |
| `endpoint` | Site-centroid distances (`endpoint_dist_*`) | every column with prefix `endpoint_dist_` |

Constants and helpers: `src/ChangepointAnalysis/feature_groups.py` (`DEFAULT_GROUPS` = the four GSA groups; `ALL_GROUPS` also includes `endpoint`).

Timeline plots fall back to `endpoint_dist_mean/min/max` when the usual GSA panel columns are absent, so summarize works on endpoint-only CSVs.

---

## Source layout

| Path | Role |
|------|------|
| `src/ChangepointAnalysis/feature_groups.py` | Column lists, colors, group helpers, CSV suffix helpers |
| `src/ChangepointAnalysis/detection.py` | `ChangepointConfig`, detection + CSV writers, `discover_feature_csvs(..., suffix=)` |
| `src/ChangepointAnalysis/penalty_sweep.py` | Elbow / plateau sweep |
| `src/ChangepointAnalysis/segment_clustering.py` | Segment clustering |
| `src/ChangepointAnalysis/reporting.py` | Cohort tables, plots, `compare_endpoint_clusters_to_deformation` |
| `src/ChangepointAnalysis/endpoint_features.py` | Ring-site feature extraction → `*_endpoint_features.csv` |
| `src/ChangepointAnalysis/transition_attribution.py` | Attribute cluster transitions → ranked site-pair drivers |
| `src/ChangepointAnalysis/pipeline.py` | `ChangepointPipeline`, `run_endpoint_changepoint` |
| `src/EndpointAnalyzer/endpoints_finder.py` | `find_endpoint_sites` / ring-system grouping |
| `scripts/changepoint_feature_groups.py` | Detect CLI |
| `scripts/sweep_changepoint_penalty.py` | Sweep CLI |
| `scripts/cluster_changepoint_segments.py` | Cluster CLI |
| `scripts/summarize_changepoint_results.py` | Summarize CLI |
| `scripts/run_changepoint_pipeline.py` | End-to-end CLI (GSA features) |
| `scripts/run_endpoint_changepoint.py` | Trajectory → endpoint features → changepoint |
| `src/mdchat/skills/changepoint_pipeline.py` | MDChat skills |
| `tests/test_changepoint_pipeline.py` | Synthetic regression tests |

---

## Quick start — GSA features

One command for detect → cluster → summarize:

```powershell
python scripts/run_changepoint_pipeline.py `
    --features-dir output/gsa_features `
    --output-dir output/changepoints `
    --n-clusters 5
```

With automatic penalty selection (elbow on total breakpoints vs penalty):

```powershell
python scripts/run_changepoint_pipeline.py `
    --features-dir output/gsa_features `
    --output-dir output/changepoints `
    --with-sweep --n-penalties 15 `
    --n-clusters 5
```

Useful flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--penalty` | auto `log(n)` when normalized | Fixed Pelt penalty |
| `--with-sweep` | off | Sweep first, then detect at elbow |
| `--n-clusters` / `--k` | 5 | Segment clusters per group |
| `--groups` | all four | Subset of `gsa iodine na_water combined` |
| `--skip-clustering` | off | Detect (+ optional summarize) only |
| `--skip-summarize` | off | Skip plots / cohort CSVs |
| `--min-size` | 10 | Minimum segment length (frames) |
| `--jump` | 5 | Ruptures subsample step |
| `--tolerance-frames` | 50 | Window for “shared” breakpoints |

---

## Quick start — endpoint sites (A / B / C1 / C2)

Use this path when you want to ask whether **endpoint geometry alone** recovers the paper's escalating deformation motifs (perfect cube → one cation–π open → two open with closed d1 → two open with elongated d1).

### Ring centroids, not atom–atom distances

A cation–π contact is measured from the **ring centroid**, not from individual ring atoms. The extractor therefore:

1. Finds endpoint tips via `EndpointsFinder`.
2. Groups tips into **sites**: fused ring systems (union-find over RDKit rings) become one multi-atom site; non-ring tips stay singleton atom sites.
3. Computes **site-centroid ↔ site-centroid** distances each frame (`EndpointAnalyzerObserver(use_ring_centroids=True)`).

This maps chemically to:

| Paper parameter | Endpoint-site proxy |
|-----------------|---------------------|
| **Raw endpoint atoms** `s3:A` / `s7:A` | Singleton atom endpoint sites (exocyclic / tip carbons) |
| **Paper d1** (open cation–π ≈ **4.5–5.5 Å**) | One-step-in **ring neighbor** of each `s3:A` / `s7:A`, distance `paper_d1_m{i}s3_m{j}s7` / `paper_d1_m{i}s7_m{j}s3` |
| Other openings / elongation | Additional `endpoint_dist_*` site–site columns (ring centroids and other atom sites) |

**Important:** the paper’s d1 is **not** the raw `s3↔s7` endpoint-atom distance. Each endpoint atom is stepped one bond inward onto its bonded ring atom; those ring atoms define d1. Generic `step_back_from_terminals` does not perform this step because `s3`/`s7` are degree-4 carbons.

We do **not** hard-code A/B/C1/C2 labels. Continuous site distances go into the changepoint pipeline; clusters are then compared against the expected RMSD ordering (~1.0 / 1.5 / 1.6–2.5 / 2.7 Å). For motif reading, flag a corrected d1 contact as **open cation–π** when its distance sits in **4.5–5.5 Å** (`closed` &lt; 4.5, `elongated` &gt; 5.5).

```mermaid
flowchart TD
  topo["topology + trajectory"] --> sel["resolve_selections → monomers"]
  sel --> sites["find_endpoint_sites → ring + atom sites"]
  sites --> obs["EndpointAnalyzerObserver use_ring_centroids=True"]
  obs --> csv["*_endpoint_features.csv + endpoint_sites.csv"]
  csv --> pipe["ChangepointPipeline groups=endpoint"]
  pipe --> cmp["endpoint_cluster_deformation_summary.csv"]
```

### CLI

```powershell
python scripts/run_endpoint_changepoint.py `
  --topology traj/BHHpH_ca.prmtop `
  --trajectories "traj/BHHpH_*_mdcrd_v.trj" `
  --gsa-resname MOL `
  --output-dir output/endpoint_changepoints `
  --rmsd-from output/gsa_features `
  --include-site-pairs `
  --with-sweep
```

Nested HPC layout (`$TRAJ_DIR/<run_id>/mdcrd_v`), with trajectory-level parallel by default:

```bash
python scripts/run_endpoint_changepoint.py \
  --topology "$TOPOLOGY" \
  --trajectories "$TRAJ_DIR/*/mdcrd_v" \
  --gsa-resname MOL \
  --output-dir output/endpoint_changepoints \
  --include-site-pairs \
  --traj-jobs -1
```

IDs for bare `mdcrd_v` files become `{parent_folder}_mdcrd_v` (e.g. `109345_mdcrd_v`).

| Flag | Effect |
|------|--------|
| `--no-ring-centroids` | Flat atom endpoints instead of ring-system centroids |
| `--include-site-pairs` | Also emit `endpoint_dist_{i}s{a}_{j}s{b}` (**required** for transition attribution) |
| `--no-paper-d1` | Disable corrected paper-d1 features (enabled by default) |
| `--paper-d1-open-lo` / `--paper-d1-open-hi` | Open cation–π window in Å (default 4.5–5.5) |
| `--rmsd-from DIR` | Overlay mean `assembly_rmsd_to_ref` per cluster from `*_gsa_features.csv` |
| `--step N` | Trajectory frame stride |
| `--traj-jobs N` | Parallel workers across trajectories (default `-1` = all CPUs, capped by traj count) |
| `--n-jobs N` | Parallel workers for frames within one trajectory (default: `1` when traj-jobs>1) |
| `--use-dask` | Use Dask Distributed for the endpoint distance pass |
| `--max-workers-for-io N` | Cap I/O-bound workers (iterator default limit is 16) |
| `--features-dir` | Where to write features (default: `<output-dir>/endpoint_features`) |
| `--with-sweep` | Penalty sweep before final detection |
| `--n-clusters` / `--k` | Segment clusters (default 5) |
| `--skip-transition-attribution` | Skip post-clustering site-pair driver attribution |
| `--transition-top-n` | Keep top-N drivers per directed transition (default 10) |
| `--transition-min-abs-corr` | Drop drivers with \|point-biserial r\| below this threshold |

**Outputs under `--output-dir`:**

| Artifact | Role |
|----------|------|
| `endpoint_features/*_endpoint_features.csv` | Per-frame site-distance features (+ `paper_d1_*` when enabled) |
| `endpoint_features/endpoint_sites.csv` | Site map (monomer, kind, atom ids; s3/s7 rows include `d1_ring_atom_id`) |
| `endpoint_features/paper_d1_atoms.csv` | Traceability: endpoint atom → one-step-in ring neighbor |
| `endpoint_features/endpoint_sites.png` | Visual QC from topology (blue=ring system, orange=atom site); one file per prmtop |
| `all_breakpoints.csv`, `all_segment_stats.csv`, … | Standard changepoint tables (`group=endpoint`) |
| `clusters/` | Segment clusters |
| `endpoint_cluster_deformation_summary.csv` | Clusters ordered by mean endpoint distance (+ optional RMSD) |
| `plots/endpoint_cluster_deformation.png` | Bar/scatter vs deformation proxy |
| `endpoint_transition_events.csv` | Consecutive directed cluster transitions |
| `endpoint_transition_feature_rankings.csv` | All site-pair + paper-d1 features ranked per directed transition |
| `endpoint_transition_top_features.csv` | Top-N drivers with site/atom mapping |
| `paper_d1_transition_top_features.csv` | Top-N corrected d1 drivers only |
| `paper_d1_segment_states.csv` | Per-segment mean d1 + open/closed/elongated pair lists |
| `plots/endpoint_transition_feature_heatmap.png` | Transitions × pairs colored by signed standardized Δ |
| `plots/endpoint_transition_network_attributed.png` | Transition network labeled by top mapped pairs |
| `plots/endpoint_sites_transition_{from}_to_{to}.html` | Interactive 3D NGL view (raw sites + paper-d1 cylinders) |
| `plots/endpoint_sites_transition_{from}_to_{to}_pair_ranks.png` | Ranked-pair bar chart companion |

### Reading results vs A / B / C1 / C2

1. Order clusters by mean endpoint-site distance (the deformation summary CSV already sorts this way).
2. If `--rmsd-from` was set, compare each cluster's mean RMSD to the paper peaks (A≈1.0, B≈1.5, C1≈1.6–2.5, C2≈2.7 Å).
3. With `--include-site-pairs`, inspect which individual site-pair distances jump at breakpoints — a single large jump suggests B; two openings plus an elongated equatorial pair suggest C2.
4. Use **transition attribution** (below) to rank which mapped site pairs drive each directed cluster change.

### Feature column schema (`*_endpoint_features.csv`)

| Columns | Meaning |
|---------|---------|
| `traj_id`, `frame`, `time_ps` | Metadata |
| `endpoint_dist_{i}_{j}_{min,mean,max}` | Aggregates over all site-pairs between monomers *i* and *j* |
| `endpoint_dist_{mean,min,max,std}` | Assembly-wide aggregates |
| `endpoint_dist_{i}s{a}_{j}s{b}` | Optional raw site–site distance (`--include-site-pairs`) |
| `paper_d1_m{i}s3_m{j}s7` / `paper_d1_m{i}s7_m{j}s3` | Corrected directional d1 (ring-neighbor of s3/s7); enabled by default |
| `paper_d1_min`, `paper_d1_n_open`, … | Assembly summaries over all directional d1 contacts |

### Paper d1 (corrected cation–π distance)

For each monomer, sites `s3:A` and `s7:A` are tip/exocyclic carbons. The paper defines **d1** on the ring atom bonded one step inward from each of those tips.

| State | Distance window |
|-------|-----------------|
| `closed` | &lt; 4.5 Å |
| `open` (cation–π) | 4.5–5.5 Å |
| `elongated` | &gt; 5.5 Å |

Outputs: `paper_d1_atoms.csv` (atom map), `paper_d1_*` columns in the features CSV, `paper_d1_segment_states.csv`, and d1 cylinders in the transition NGL HTML (green=open, gray=closed, magenta=elongated).

### Transition driver attribution

After endpoint clustering, the pipeline can attribute each **directed** consecutive-segment transition (`from_cluster → to_cluster`) to the raw site-pair columns **and** corrected paper-d1 columns that change most strongly.

**Requires** `--include-site-pairs`. Without those columns, attribution is skipped with a warning. Clustering itself still uses only the eight aggregate summary columns (+ `log10_n_frames`); attribution deliberately returns to the per-frame site-pair series.

**Scoring (per directed transition type):**

For each site-pair feature:

| Statistic | Definition |
|-----------|------------|
| `mean_source` / `mean_dest` | Mean distance (Å) over source / destination segment frames |
| `mean_delta_angstrom` | `mean_dest − mean_source` (signed) |
| `mean_standardized_delta` | Δ divided by the pooled cohort std of that feature |
| `mean_point_biserial_corr` | Point-biserial correlation of the distance with destination-membership |
| `combined_score` | `0.5 × (abs_std_Δ / max_abs_std_Δ) + 0.5 × \|r\|` within the directed transition |
| `rank` | Descending `combined_score` within `from→to` |
| `n_events` / `n_trajectories` | How often this directed transition was observed |
| `is_descriptive` | `True` when `n_events < 2` (no replication; treat as descriptive) |

Frame joins are **stride-safe**: segments select rows by actual `frame` values, not integer indices.

**Interpretation notes**

- Positive Δ / positive r: the site–site distance **opens** on entering the destination cluster.
- Negative Δ / negative r: the pair **closes**.
- When a transition type has only one observed event (common in single-trajectory tests), rankings remain useful for hypothesis generation but are marked `is_descriptive=True` — do not treat correlations as statistically replicated.
- Prefer reading `endpoint_transition_top_features.csv` together with the attributed network PNG: edge width = transition count; edge labels = top mapped pairs with signed Δ.

**CLI / skill controls**

- `--transition-top-n` / `transition_top_n` (default 10)
- `--transition-min-abs-corr` / `transition_min_abs_corr` (default 0)
- `--skip-transition-attribution` / `skip_transition_attribution`

Python API: `ChangepointPipeline.attribute_endpoint_transitions(...)` or `attribute_endpoint_transitions(...)` from `src.ChangepointAnalysis`.

---

## Stage-by-stage CLIs (GSA features)

Use these when you want to re-run one stage without the full pipeline.

### 1. Detect

```powershell
python scripts/changepoint_feature_groups.py `
    --input-dir output/gsa_features `
    --output-dir output/changepoints `
    --method Pelt --cost-model rbf `
    --min-size 10 --jump 5 --tolerance-frames 50
```

Omit `--penalty` to use the BIC-style `log(n)` default after z-score normalization. Pass `--no-normalize` only if you intentionally want raw-scale features (penalty heuristic then falls back to ruptures’ own default).

**Writes (among others):**

- `{traj_id}_breakpoints.csv` / `{traj_id}_segment_stats.csv`
- `all_breakpoints.csv`
- `all_segment_stats.csv`
- `changepoint_timing_comparison.csv`

### 2. Penalty sweep

```powershell
python scripts/sweep_changepoint_penalty.py `
    --input-dir output/gsa_features `
    --output-dir output/changepoints/penalty_sweep `
    --n-penalties 15 `
    --run-final --final-output-dir output/changepoints
```

- Primary recommendation: **elbow** on cohort total breakpoints vs log-penalty.
- Cross-group Jaccard is diagnostic only (each group is segmented independently).
- `--run-final` re-runs detection **in-process** at the recommended penalty (no subprocess).

**Writes:** `penalty_sweep_summary.csv`, `penalty_elbow.csv`, `penalty_recommendation.txt`, optional plots under `penalty_sweep/plots/`.

### 3. Cluster segments

```powershell
python scripts/cluster_changepoint_segments.py `
    --changepoints-dir output/changepoints `
    --output-dir output/changepoints/clusters `
    --n-clusters 5 --linkage ward
```

Requires `all_segment_stats.csv` (or per-trajectory `*_segment_stats.csv`).

**Writes:** `clusters/{group}/segments_clustered.csv`, dendrogram/PCA/inspection artifacts, `all_segments_clustered.csv`, `transitions/{group}/`.

### 4. Summarize / plot

```powershell
python scripts/summarize_changepoint_results.py `
    --changepoints-dir output/changepoints `
    --features-dir output/gsa_features
```

**Writes:** `cohort_timing_summary.csv`, `cohort_segment_regime_summary.csv`, `breakpoints_per_trajectory.csv`, and plots under `plots/` (Jaccard heatmap, breakpoint histogram, cohort overview, optional timelines).

To summarize an endpoint-only run, point `--features-dir` at the `*_endpoint_features.csv` directory (timeline panels auto-fall back to endpoint aggregates).

---

## Python API

### GSA features end-to-end

```python
from src.ChangepointAnalysis import (
    ChangepointConfig,
    ChangepointPipeline,
    PenaltySweepConfig,
    SegmentClusteringConfig,
)

pipe = ChangepointPipeline(
    "output/gsa_features",
    "output/changepoints",
    detection=ChangepointConfig(penalty=None, min_size=10, jump=5),
    sweep=PenaltySweepConfig(n_penalties=15),  # optional
    clustering=SegmentClusteringConfig(n_clusters=5),
)

artifacts = pipe.run_all(with_sweep=True)
```

Stages can also be called separately; later stages reuse in-memory tables when available, otherwise they read from `output_dir`:

```python
pipe.sweep_penalty()       # sets detection.penalty from elbow
pipe.detect()
pipe.cluster_segments()
pipe.summarize()
```

Pass `features_suffix="_endpoint_features.csv"` when the features directory holds endpoint CSVs:

```python
pipe = ChangepointPipeline(
    "output/endpoint_changepoints/endpoint_features",
    "output/endpoint_changepoints",
    detection=ChangepointConfig(groups=("endpoint",)),
    clustering=SegmentClusteringConfig(groups=("endpoint",)),
    features_suffix="_endpoint_features.csv",
)
```

### Endpoint sites end-to-end

```python
from src.ChangepointAnalysis import run_endpoint_changepoint

artifacts = run_endpoint_changepoint(
    "traj/BHHpH_ca.prmtop",
    ["traj/BHHpH_109345_mdcrd_v.trj"],
    output_dir="output/endpoint_changepoints",
    use_ring_centroids=True,
    include_site_pairs=True,
    rmsd_from="output/gsa_features",
)
```

Lower-level extraction only:

```python
import MDAnalysis as mda
from src.ChangepointAnalysis import (
    EndpointFeatureConfig,
    generate_endpoint_features,
    write_endpoint_features_csv,
)

u = mda.Universe("topo.prmtop", "traj.trj")
features, sites = generate_endpoint_features(
    u,
    EndpointFeatureConfig(use_ring_centroids=True, include_site_pairs=True),
    traj_id="my_traj",
)
write_endpoint_features_csv(features, "output/endpoint_features", "my_traj", sites_df=sites)
```

### Detect only (any suffix)

```python
from pathlib import Path
from src.ChangepointAnalysis import ChangepointConfig, detect_cohort_changepoints
from src.ChangepointAnalysis.detection import discover_feature_csvs

csv_paths = discover_feature_csvs(
    Path("output/gsa_features"),
    suffix="_gsa_features.csv",  # or "_endpoint_features.csv"
)
tables = detect_cohort_changepoints(
    csv_paths,
    ChangepointConfig(penalty=6.2),
    output_dir=Path("output/changepoints"),
)
print(tables.breakpoints.head())
```

### Key dataclasses

```python
ChangepointConfig          # method, cost_model, penalty, groups, min_size, jump, ...
ChangepointTables          # breakpoints, segment_stats, comparison DataFrames
PenaltySweepConfig         # grid size, thresholds, nested ChangepointConfig
PenaltySweepResult         # summary, elbow, recommended_penalty, ...
SegmentClusteringConfig    # n_clusters, linkage, groups, PCA / all-k options
EndpointFeatureConfig      # gsa_resname, use_ring_centroids, include_site_pairs, n_jobs, stride, ...
```

---

## MDChat skills

Register automatically when MDChat loads skills (`/skills` or `/analysis`). Category: `changepoint`.

| Skill | Stage | Needs universe? |
|-------|--------|-----------------|
| `changepoint_feature_groups` | Detect on `*_gsa_features.csv` | No |
| `sweep_changepoint_penalty` | Penalty sweep (+ optional final detect) | No |
| `cluster_changepoint_segments` | Segment clustering | No |
| `summarize_changepoint_results` | Cohort tables + plots | No |
| `run_changepoint_pipeline` | Full GSA-feature chain | No |
| `run_endpoint_changepoint` | Trajectory → endpoint sites → changepoint | No (loads trajectories itself) |

Example asks:

- “Run changepoint detection on `output/gsa_features` into `output/changepoints`.”
- “Sweep the Pelt penalty and re-run detection at the elbow.”
- “Cluster changepoint segments with k=5, then summarize.”
- “Run endpoint-site changepoint on these trajectories and overlay RMSD from `output/gsa_features`.”

---

## Output schema (compatibility)

Downstream scripts (`compare_state_methods.py`, PT analysis helpers, cluster structure tools, etc.) expect these filenames and columns. Do not rename them casually.

### `all_breakpoints.csv`

`traj_id`, `group`, `breakpoint_idx`, `signal_index`, `frame`, `time_ps`, `n_cols_used`

### `all_segment_stats.csv`

Core: `traj_id`, `group`, `segment_id`, `start_frame`, `end_frame`, `start_ps`, `end_ps`, `n_frames`, `n_cols_used`  
Plus per-group `*_mean` / `*_std` columns from `SUMMARY_COLS`.

### `changepoint_timing_comparison.csv`

`traj_id`, `group_a`, `group_b`, `tolerance_frames`, `n_bkps_a`, `n_bkps_b`, `n_shared`, `jaccard`, `mean_timing_offset_frames`, `mean_timing_offset_ps`

Single-group runs (e.g. `endpoint` only) write a **header-only** comparison file — there are no pairwise group comparisons.

### `endpoint_cluster_deformation_summary.csv`

`cluster_label`, `n_segments`, `n_trajectories`, `total_frames`, `endpoint_dist_mean`, `endpoint_dist_std`, and optionally `assembly_rmsd_to_ref_mean` / `_std` when `--rmsd-from` is set. Rows are sorted by mean endpoint distance.

---

## Defaults that matter

| Setting | Typical value | Notes |
|---------|---------------|-------|
| Method / cost | Pelt + `rbf` | Multivariate after z-score |
| Penalty | `log(n)` when normalized | Override with `--penalty` or `--with-sweep` |
| `min_size` | 10 frames | Suppresses very short segments |
| `jump` | 5 | Speed vs resolution trade-off |
| Timing tolerance | 50 frames | Used for Jaccard “shared” matches |
| Clustering | Ward, k=5 | Fixed k by default; `--auto-select-k` available |
| Ring centroids | on for endpoint path | `use_ring_centroids=True`; opt out with `--no-ring-centroids` |

---

## Tests

```powershell
# Full suite (GSA + endpoint)
python -m pytest tests/test_changepoint_pipeline.py -v

# Endpoint-focused only
python -m pytest tests/test_changepoint_pipeline.py -k endpoint -v
```

Coverage includes:

- Synthetic GSA features with a planted regime shift (filenames, schemas, breakpoint near the shift)
- `all_pairs_to_metrics_df` flattening
- Ring-system grouping (biphenyl / naphthalene) and centroid math
- Endpoint-group pipeline on synthetic `*_endpoint_features.csv`

---

## Related docs

- Cohort interpretation notes: `output/changepoints/ANALYSIS_SUMMARY.md`
- Presentation-style writeup: `output/changepoints/CHANGPOINT_PIPELINE_PRESENTATION.md`
- Feature extraction entry point: `scripts/compute_gsa_features.py`
- Endpoint finder / observer: `src/EndpointAnalyzer/`
- Low-level ruptures wrapper: `src/utils/ruptures_utils.py`
