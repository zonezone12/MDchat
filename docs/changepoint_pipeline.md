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
| `endpoint` | Site-centroid distances (`endpoint_dist_*`) | **aggregates only by default** (~49 cols); raw site pairs optional |

Constants and helpers: `src/ChangepointAnalysis/feature_groups.py` (`DEFAULT_GROUPS` = the four GSA groups; `ALL_GROUPS` also includes `endpoint`).

**Endpoint detection vs clustering vs attribution (important):**

| Stage | Features used (default) | Why |
|-------|-------------------------|-----|
| **Detect** | Assembly + per-monomer-pair aggregates (~49 cols); excludes raw `endpoint_dist_{i}s{a}_{j}s{b}` | 960 raw site pairs are highly redundant (~26 PCs for 80% variance) and swamp the signal after z-score |
| **Cluster** | Assembly aggregates + every `endpoint_dist_{i}_{j}_mean` mean/std (~39 dims + `log10_n_frames`) | Clusters must encode *which* monomer pair deformed, not only the global mean |
| **Timeline / attribution** | Raw site-pair columns (when present) ranked by **segment-level η²** (or transition standardized Δ) | Frame-level point-biserial against segment labels is attenuated; η² matches the unit clusters were defined on |

Pass `--include-site-pairs-in-detection` to restore the old “all `endpoint_dist_*`” detection matrix. Timeline plots fall back to `endpoint_dist_mean/min/max` when GSA panel columns are absent, or when `--timeline-top-pairs 0`.

---

## Source layout

| Path | Role |
|------|------|
| `src/ChangepointAnalysis/feature_groups.py` | Column lists, colors, group helpers, CSV suffix helpers |
| `src/ChangepointAnalysis/detection.py` | `ChangepointConfig`, detection + CSV writers, `discover_feature_csvs(..., suffix=)` |
| `src/ChangepointAnalysis/penalty_sweep.py` | Elbow / plateau sweep |
| `src/ChangepointAnalysis/segment_clustering.py` | Segment clustering |
| `src/ChangepointAnalysis/cluster_k_diagnostics.py` | Silhouette-vs-k / switching / geometry comparison across B* cohorts (`group=endpoint` or `gsa`) |
| `src/ChangepointAnalysis/gsa_cohort_run.py` | Split mixed `gsa_features_step1` CSVs by cube; run per-cube detect+cluster |
| `src/ChangepointAnalysis/reporting.py` | Cohort tables, plots, `compare_endpoint_clusters_to_deformation`, `summarize_endpoint_cluster_proxies`, `compare_endpoint_clusters_across_cohorts`, segment-η² timeline panels |
| `src/ChangepointAnalysis/endpoint_features.py` | Ring-site feature extraction → `*_endpoint_features.csv` |
| `src/ChangepointAnalysis/transition_attribution.py` | Attribute cluster transitions → ranked site-pair drivers |
| `src/ChangepointAnalysis/pipeline.py` | `ChangepointPipeline`, `run_endpoint_changepoint` |
| `src/EndpointAnalyzer/endpoints_finder.py` | `find_endpoint_sites` / ring-system grouping |
| `scripts/changepoint_feature_groups.py` | Detect CLI |
| `scripts/sweep_changepoint_penalty.py` | Sweep CLI |
| `scripts/cluster_changepoint_segments.py` | Cluster CLI |
| `scripts/summarize_changepoint_results.py` | Summarize CLI |
| `scripts/compare_changepoint_timing.py` | Post-hoc endpoint vs GSA timing comparison |
| `scripts/compare_endpoint_clusters_across_cohorts.py` | Cross-cohort endpoint cluster vs deformation-proxy tables/plots |
| `scripts/compare_endpoint_cluster_k.py` | Cross-cohort silhouette-vs-k / switching (`--group endpoint` or `gsa`) |
| `scripts/run_gsa_step1_cluster_k.py` | Per-cube GSA changepoint from `gsa_features_step1` + k-diagnostics |
| `scripts/run_changepoint_pipeline.py` | End-to-end CLI (GSA features, one cube per `--features-dir`) |
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

**Mixed `gsa_features_step1` folder (all B\* cubes together):** do **not** point `--features-dir` at that folder as a whole — clustering would mix cubes. Stage per cube and run k-diagnostics with:

```powershell
python scripts/run_gsa_step1_cluster_k.py `
    --features-dir output/gsa_features_step1 `
    --output-root output `
    --out-dir output/gsa_cluster_k_diagnostics `
    --workers 6
```

Writes `output/gsa_changepoints_{CUBE}/` (`--groups gsa` only, skip summarize/PCA) plus the same silhouette / switching tables as the endpoint comparison. BMHpM CSVs in a subdirectory are picked up. Re-run diagnostics only with `--skip-detect`. Existing `output/changepoints` is an older BMMpM-only 500-frame run; do not reuse it for step-1.

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
| `--include-site-pairs` | Also emit `endpoint_dist_{i}s{a}_{j}s{b}` (**required** for transition attribution and η² timeline panels) |
| `--include-site-pairs-in-detection` | Feed raw site-pair columns into Pelt (default **off** = aggregates only, ~49 cols) |
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
| `--timeline-top-pairs` | Top-N site-pair (or pair-mean) features for `timeline_cluster_*.png` by segment η² (default 5; `0` = aggregate panels only) |
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
| `all_breakpoints.csv`, `all_segment_stats.csv`, … | Standard changepoint tables (`group=endpoint`; segment stats include per-monomer-pair aggregates) |
| `clusters/` | Segment clusters (chosen-k under `clusters/endpoint/`; `by_k/` is inspection-only) |
| `endpoint_cluster_deformation_summary.csv` | Clusters ordered by mean endpoint distance (+ optional RMSD) |
| `endpoint_pair_cluster_correlation.csv` | Site-pair (or pair-mean) ranking by segment-level η² vs cluster |
| `plots/endpoint_cluster_deformation.png` | Bar/scatter vs deformation proxy |
| `plots/timeline_cluster_{k}_{traj}.png` | Medoid-segment timelines: `endpoint_dist_mean` + top-N pairs labeled `M0S3-M1S6 (Å)  η²=…` |
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
2. If `--rmsd-from` was set, compare each cluster's mean RMSD to the paper peaks (A≈1.0, B≈1.5, C1≈1.6–2.5, C2≈2.7 Å). When GSA features are unavailable, use the **four deformation proxies** below (endpoint distance, paper d1, site-pair η², Cohen's *d* direction) as RMSD substitutes.
3. With `--include-site-pairs`, inspect which individual site-pair distances jump at breakpoints — a single large jump suggests B; two openings plus an elongated equatorial pair suggest C2.
4. Check `endpoint_pair_cluster_correlation.csv` / `timeline_cluster_*.png` for the site pairs with highest **segment-level η²** (fraction of between-cluster variance). Labels look like `M2S0-M5S0`.
5. Use **transition attribution** (below) to rank which mapped site pairs drive each directed cluster change.

### Cross-cohort cluster proxy comparison (B\* cubes)

After running endpoint changepoint on multiple cube variants (e.g. `output/endpoint_changepoints_BHHpH`, `…_BMMpM`), compare Ward clusters **across cohorts** without re-detecting or re-clustering.

**Why `deformation_rank`?** `cluster_label` integers are arbitrary per cohort (SciPy Ward cut). Cross-cohort alignment uses **`deformation_rank`**: 0 = smallest mean `endpoint_dist_mean` (most closed), increasing to the most open cluster in that cube.

**Four deformation proxies** (used when `assembly_rmsd_to_ref` was not overlaid):

| # | Proxy | Source | Meaning |
|---|-------|--------|---------|
| 1 | Endpoint distance | `endpoint_cluster_deformation_summary.csv` | Assembly-wide cage opening (Å) |
| 2 | Paper d1 | `paper_d1_segment_states.csv` | Cation–π geometry (`paper_d1_min_mean`, `n_open_pairs`, …) |
| 3 | Site-pair η² | Top rows of `endpoint_pair_cluster_correlation.csv` | Which named contacts explain cluster variance |
| 4 | Cohen's *d* direction | `cohens_d_best` on top pairs where `best_cluster` matches | +*d* = more open for that pair; −*d* = more closed vs other clusters |

Clusters with `n_segments < 5` are flagged `low_confidence=True` (e.g. singleton BMHpH clusters) and excluded from rank-averaged cross-cohort tables.

```powershell
python scripts/compare_endpoint_clusters_across_cohorts.py `
  --output-root output `
  --pattern "endpoint_changepoints_B*" `
  --exclude-test `
  --out-dir output/endpoint_cluster_cross_cohort `
  --top-pairs 10
```

Auto-discovers production dirs matching the pattern; skips `*_test` / `*_smoke`. Pass explicit dirs with `--cohort-dirs` to override discovery.

**Writes under `--out-dir`:**

| Artifact | Role |
|----------|------|
| `cluster_proxy_summary.csv` | Long table: `cohort`, `cluster_label`, `deformation_rank`, all four proxies, segment counts |
| `cluster_proxy_by_rank.csv` | Pivoted: rows = `deformation_rank`, columns = cohort × proxy (slide heatmaps) |
| `top_site_pairs_by_cohort.csv` | Top η² pairs per cohort + cross-cohort frequency (`cohort_count`) |
| `pair_best_cluster_matrix.csv` | Which cluster each top pair marks + `deformation_rank_of_best_cluster` |
| `plots/deformation_rank_vs_endpoint_dist.png` | Heatmap: rank × cohort, colored by mean endpoint distance |
| `plots/deformation_rank_vs_paper_d1.png` | Heatmaps for `paper_d1_min_mean` and `n_open_pairs` |
| `plots/top_pairs_eta2_heatmap.png` | Cohort × top-5 pair labels, colored by η² |
| `plots/proxy_consistency_scatter.png` | Per-cohort scatter: endpoint distance vs paper d1 (points labeled by rank) |

**Interpretation workflow (cross-cohort):**

1. For each `deformation_rank`, compare mean endpoint distance across BHH / BMH / BMM and pH / pM variants.
2. Check whether paper d1 shifts consistently with rank (note: many segments are `elongated` in Å; read **relative** rank changes, not absolute 4.5–5.5 Å windows).
3. Inspect `top_site_pairs_by_cohort.csv` for recurring contacts at the same rank (e.g. M2–M3 face pairs).
4. Use `cohens_d_direction` on the cluster's dominant top pair to read open vs closed for that contact.

Python:

```python
from pathlib import Path
from src.ChangepointAnalysis import (
    compare_endpoint_clusters_across_cohorts,
    summarize_endpoint_cluster_proxies,
)

# Single cohort
proxies = summarize_endpoint_cluster_proxies(
    "output/endpoint_changepoints_BMMpM",
    top_pairs=10,
)

# All B* cubes
cohorts = sorted(Path("output").glob("endpoint_changepoints_B*"))
cohorts = [p for p in cohorts if "_test" not in p.name and "_smoke" not in p.name]
written = compare_endpoint_clusters_across_cohorts(
    cohorts,
    "output/endpoint_cluster_cross_cohort",
    top_pairs=10,
)
```

GSA step-1 k-diagnostics (after per-cube `gsa_changepoints_*` dirs exist):

```python
from src.ChangepointAnalysis import (
    compare_cluster_k_diagnostics,
    run_gsa_step1_cohorts,
)

run_gsa_step1_cohorts("output/gsa_features_step1", "output", workers=6)
written = compare_cluster_k_diagnostics(
    list(Path("output").glob("gsa_changepoints_B*")),
    "output/gsa_cluster_k_diagnostics",
    group="gsa",
)
```

Talk-track for cluster definition and η² / Cohen's *d*: [endpoint_cluster_discrimination.md](endpoint_cluster_discrimination.md).

### Why k=5, and why BMMpM’s silhouette rises

All B\* endpoint runs are cut at **fixed k=5** (`--n-clusters`). Silhouette curves under `clusters/endpoint/silhouette_by_k.csv` are inspection-only. BHHpM peaks at k=6; BHHpH / BMHpM have a weak local bump at k=6 after k=2 already wins; BMMpM’s score keeps climbing because paper d1 never enters the 4.5–5.5 Å window and Pelt barely switches.

```powershell
python scripts/compare_endpoint_cluster_k.py `
  --output-root output `
  --pattern "endpoint_changepoints_B*" `
  --exclude-test `
  --out-dir output/endpoint_cluster_k_diagnostics
```

Writes `silhouette_peaks.csv`, `segment_dynamics.csv`, `site_and_d1.csv`, `k_split.csv`, and `plots/silhouette_vs_k.png` (plus switching / d1 / site-kind figures). Full talk-track: [endpoint_cluster_discrimination.md §10](endpoint_cluster_discrimination.md#10-why-every-b-run-has-five-clusters-and-why-bmmpm-differs).

The same tables for **cage-geometry GSA** (no paper-d1) after `run_gsa_step1_cluster_k.py`:

```powershell
python scripts/compare_endpoint_cluster_k.py `
  --group gsa `
  --output-root output `
  --pattern "gsa_changepoints_B*" `
  --out-dir output/gsa_cluster_k_diagnostics
```

```python
from src.ChangepointAnalysis import compare_cluster_k_diagnostics

written = compare_cluster_k_diagnostics(
    list(Path("output").glob("gsa_changepoints_B*")),
    "output/gsa_cluster_k_diagnostics",
    group="gsa",
)
```

See [endpoint_cluster_discrimination.md §11](endpoint_cluster_discrimination.md#11-gsa-cage-geometry-k-diagnostics-gsa_features_step1).

### Feature column schema (`*_endpoint_features.csv`)

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

When methyl (or other exocyclic) tips are absent—common for BHHpM—the s3/s7 site index may collapse to a **multi-atom ring site**. Paper-d1 resolution then falls back to exocyclic C tips on that ring, or to the hull/candidate tooth ring carbon on that site (`endpoint_kind` = `exocyclic_tip` or `ring_site` in `paper_d1_atoms.csv`). For `ring_site`, the tooth ring carbon **is** the d1 atom (no further inward step — that would be ambiguous on a 6-membered ring).

| State | Distance window |
|-------|-----------------|
| `closed` | &lt; 4.5 Å |
| `open` (cation–π) | 4.5–5.5 Å |
| `elongated` | &gt; 5.5 Å |

Outputs: `paper_d1_atoms.csv` (atom map), `paper_d1_*` columns in the features CSV, `paper_d1_segment_states.csv`, and d1 cylinders in the transition NGL HTML (green=open, gray=closed, magenta=elongated).

### Cluster timeline panels (segment-level η²)

After clustering, summarize ranks endpoint features for `plots/timeline_cluster_*.png`:

1. Prefer raw site-pair columns `endpoint_dist_{i}s{a}_{j}s{b}` when present (`--include-site-pairs`); else fall back to `endpoint_dist_{i}_{j}_mean`.
2. For each clustered segment, take the **mean** of each candidate feature over `[start_frame, end_frame]` (raw Å; not per-traj z-scored — absolute levels carry cross-trajectory cluster signal).
3. Score each feature by **η²** (fraction of between-cluster variance), with Kruskal–Wallis ε² as a robustness column.
4. Plot `endpoint_dist_mean` plus the top-N pairs, labeled e.g. `M2S0-M5S0 (Å)  η²=0.81`.

Auditable ranking: `endpoint_pair_cluster_correlation.csv`. Medoid CSVs are taken from `clusters/<group>/cluster_representatives.csv` only (`by_k/` inspection outputs are ignored unless you pass `--cluster-representatives-csv` explicitly).

Talk-track / formulas (cohort-wide cluster definition, η², Cohen’s *d*, why frame-level |r| was attenuated): [endpoint_cluster_discrimination.md](endpoint_cluster_discrimination.md) (§2).

### Transition driver attribution

After endpoint clustering, the pipeline can attribute each **directed** consecutive-segment transition (`from_cluster → to_cluster`) to the raw site-pair columns **and** corrected paper-d1 columns that change most strongly.

**Requires** `--include-site-pairs`. Without those columns, attribution is skipped with a warning.

**Clustering** uses assembly + per-monomer-pair aggregate summaries written into `all_segment_stats.csv` (not the raw site-pair series). Attribution deliberately returns to the per-frame site-pair / paper-d1 series.

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

### 5. Compare endpoint vs GSA timing (post-hoc)

Endpoint runs are `group=endpoint` only, so their `changepoint_timing_comparison.csv` is header-only. After both an **endpoint** changepoint directory and a **GSA** changepoint directory exist, compare them without re-detecting:

```powershell
python scripts/compare_changepoint_timing.py `
    --changepoints-dirs output/endpoint_changepoints_BMMpM output/changepoints `
    --output-dir output/endpoint_vs_gsa_timing_BMMpM `
    --tolerance-frames 50
```

Trajectory IDs are matched after stripping cube prefixes (`BMMpM_109345_mdcrd_v` ↔ `109345_mdcrd_v`). Only trajectories present in both sources are compared.

**Writes:** `changepoint_timing_comparison.csv`, `cohort_timing_summary.csv`, merged `all_breakpoints.csv`, `plots/cohort_jaccard_heatmap.png`.

Python:

```python
from src.ChangepointAnalysis import compare_changepoint_timing

cmp = compare_changepoint_timing(
    "output/endpoint_changepoints_BMMpM",
    "output/changepoints",
    tolerance_frames=50,
    output_dir="output/endpoint_vs_gsa_timing_BMMpM",
)
```

Endpoint-only re-plot (reuse existing features; refresh cluster timelines):

```powershell
python scripts/summarize_changepoint_results.py `
    --changepoints-dir output/endpoint_changepoints_BHHpM `
    --features-dir output/endpoint_changepoints_BHHpM/endpoint_features `
    --features-suffix "_endpoint_features.csv" `
    --skip-tables `
    --skip-individual-timelines `
    --timeline-top-pairs 5
```

| Flag | Meaning |
|------|---------|
| `--features-suffix` | e.g. `_endpoint_features.csv` (auto-detected when omitted) |
| `--timeline-top-pairs` | Top-N endpoint pairs for `timeline_cluster_*.png` by segment η² (default 5; `0` disables) |
| `--skip-cluster-rep-timelines` | Skip medoid timelines |
| `--cluster-representatives-csv` | Explicit medoid CSV(s); default = `clusters/*/cluster_representatives.csv` (ignores `by_k/`) |

Cluster medoid timelines prefer raw site-pair columns when present, ranked by **segment-level η²**, with labels like `M0S3-M1S6 (Å)  η²=0.81`. Ranking is written to `endpoint_pair_cluster_correlation.csv`.

### 6. Compare endpoint clusters across B\* cohorts (post-hoc)

After multiple endpoint runs exist (one output dir per cube), consolidate cluster proxies without re-running the pipeline:

```powershell
python scripts/compare_endpoint_clusters_across_cohorts.py `
  --output-root output `
  --pattern "endpoint_changepoints_B*" `
  --out-dir output/endpoint_cluster_cross_cohort `
  --top-pairs 10
```

See **Cross-cohort cluster proxy comparison** under the endpoint quick start for proxy definitions, output schema, and interpretation.

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
    detection=ChangepointConfig(
        groups=("endpoint",),
        # include_site_pairs_in_detection=False  # default: aggregates only
    ),
    clustering=SegmentClusteringConfig(groups=("endpoint",)),
    features_suffix="_endpoint_features.csv",
)
pipe.detect()
pipe.cluster_segments()
pipe.summarize(cluster_timeline_top_pairs=5)
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

Cross-cohort cluster proxies (reads existing CSVs only):

```python
from src.ChangepointAnalysis import (
    compare_endpoint_clusters_across_cohorts,
    summarize_endpoint_cluster_proxies,
)

proxies = summarize_endpoint_cluster_proxies("output/endpoint_changepoints_BMMpM")
written = compare_endpoint_clusters_across_cohorts(
    [
        "output/endpoint_changepoints_BHHpH",
        "output/endpoint_changepoints_BHHpM",
        "output/endpoint_changepoints_BMHpH",
        "output/endpoint_changepoints_BMHpM",
        "output/endpoint_changepoints_BMMpH",
        "output/endpoint_changepoints_BMMpM",
    ],
    "output/endpoint_cluster_cross_cohort",
)
```

Silhouette / switching / paper-d1 diagnostics (also reads existing CSVs only):

```python
from src.ChangepointAnalysis import (
    compare_endpoint_cluster_k_diagnostics,
    discover_endpoint_k_cohort_dirs,
)

cohorts = discover_endpoint_k_cohort_dirs("output", "endpoint_changepoints_B*")
written = compare_endpoint_cluster_k_diagnostics(
    cohorts,
    "output/endpoint_cluster_k_diagnostics",
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
ChangepointConfig          # method, cost_model, penalty, groups, min_size, jump,
                           # include_site_pairs_in_detection (default False), ...
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
Plus per-group `*_mean` / `*_std` columns from summary bases.

For `group=endpoint`, summary bases are the four assembly aggregates plus every `endpoint_dist_{i}_{j}_mean` present in the features CSV (~38 endpoint columns + metadata). That is what segment clustering consumes (plus `log10_n_frames`).

### `endpoint_pair_cluster_correlation.csv`

Written by summarize when cluster medoid timelines are enabled. Columns include `feature`, `endpoint_label` (e.g. `M2S0-M5S0`), `eta_squared`, `epsilon_squared_kw`, `cohens_d_best`, `best_cluster`, `frame_max_abs_corr`, `rank`. Primary sort key is segment-level η².

### `changepoint_timing_comparison.csv`

`traj_id`, `group_a`, `group_b`, `tolerance_frames`, `n_bkps_a`, `n_bkps_b`, `n_shared`, `jaccard`, `mean_timing_offset_frames`, `mean_timing_offset_ps`

Single-group runs (e.g. `endpoint` only) write a **header-only** comparison file — there are no pairwise group comparisons. Use `scripts/compare_changepoint_timing.py` (or `compare_changepoint_timing`) to compare an endpoint directory against a GSA `output/changepoints` directory after both exist.

### `endpoint_cluster_deformation_summary.csv`

`cluster_label`, `n_segments`, `n_trajectories`, `total_frames`, `endpoint_dist_mean`, `endpoint_dist_std`, and optionally `assembly_rmsd_to_ref_mean` / `_std` when `--rmsd-from` is set. Rows are sorted by mean endpoint distance.

### Cross-cohort outputs (`output/endpoint_cluster_cross_cohort/`)

Produced by `scripts/compare_endpoint_clusters_across_cohorts.py` (or `compare_endpoint_clusters_across_cohorts`).

**`cluster_proxy_summary.csv`** — one row per cohort × cluster. Key columns:

`cohort`, `cluster_label`, `deformation_rank`, `n_segments`, `n_trajectories`, `total_frames`, `endpoint_dist_mean`, `endpoint_dist_std`, `paper_d1_min_mean`, `paper_d1_n_open_mean`, `n_open_pairs`, `n_closed_pairs`, `n_elongated_pairs`, `open_pair_segment_fraction`, `cohort_top_pair_labels`, `dominant_top_pair`, `dominant_top_pair_eta2`, `dominant_top_pair_cohens_d`, `cohens_d_direction`, `n_top_pairs_marking_cluster`, `low_confidence`

**`cluster_proxy_by_rank.csv`** — wide pivot keyed by `deformation_rank`; columns like `{cohort}_endpoint_dist_mean`, `{cohort}_paper_d1_min_mean`, `{cohort}_dominant_top_pair`, … (excludes `low_confidence` clusters).

**`top_site_pairs_by_cohort.csv`** — `cohort`, `rank`, `endpoint_label`, `eta_squared`, `cohens_d_best`, `best_cluster`, `cohens_d_direction`, `cohort_count` (how many cohorts share that pair in the top-N set).

**`pair_best_cluster_matrix.csv`** — `cohort`, `endpoint_label`, `eta_squared`, `best_cluster`, `deformation_rank_of_best_cluster`.

### k-diagnostics outputs (`output/endpoint_cluster_k_diagnostics/`)

Produced by `scripts/compare_endpoint_cluster_k.py` (or `compare_endpoint_cluster_k_diagnostics`). Reads existing `clusters/endpoint/` inspection files plus paper-d1 / site maps; does not re-cluster.

| Artifact | Role |
|----------|------|
| `silhouette_by_k.csv` | Long table: cohort × k silhouette |
| `silhouette_peaks.csv` | Chosen k vs global max, `curve_shape` |
| `k_split.csv` | Ward split k=5 → k=6 (which cluster split, new sizes) |
| `segment_dynamics.csv` | Breakpoints, whole-traj fraction, endpoint-distance span |
| `site_and_d1.csv` | Hull ring/atom counts, s3/s7 kinds, paper-d1 open fraction |
| `k_diagnostics_summary.txt` | Human-readable dump of the tables |
| `plots/silhouette_vs_k.png` | Overlay of silhouette curves; dashed line at chosen k |
| `plots/segment_switching.png` | Segments/traj, whole-traj fraction, never-switch fraction |
| `plots/paper_d1_open.png` | Closest d1 vs 4.5–5.5 Å window; fraction of open segments |
| `plots/hull_site_kinds.png` | Stacked ring vs exocyclic atom hull sites |
| `plots/endpoint_dist_span.png` | Range vs IQR of segment `endpoint_dist_mean` |

### GSA k-diagnostics outputs (`output/gsa_cluster_k_diagnostics/`)

Produced by `scripts/run_gsa_step1_cluster_k.py` (detect+cluster+compare) or `scripts/compare_endpoint_cluster_k.py --group gsa` (compare only). Reads `clusters/gsa/`; no paper-d1.

Same silhouette / switching / k-split tables as the endpoint run, plus `geometry_spread.csv` (Rg / RMSD-to-ref / octahedrality / endpoint-dist spans) and `plots/geometry_span.png`. There is no `site_and_d1.csv`.

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
| Endpoint detection cols | aggregates (~49) | Raw site pairs off unless `--include-site-pairs-in-detection` |
| Endpoint clustering dims | ~39 | Assembly + per-pair means/stds + `log10_n_frames` |
| Timeline top pairs | 5 | Segment-η² ranking; `--timeline-top-pairs 0` disables |
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
- Cross-cohort cluster proxy summary schema (`summarize_endpoint_cluster_proxies`)
- Endpoint k / silhouette diagnostics (`compare_endpoint_cluster_k_diagnostics`)
- GSA k / silhouette diagnostics (`compare_cluster_k_diagnostics`, `group="gsa"`)
- Cube discovery from mixed `gsa_features_step1` (`discover_gsa_feature_csvs_by_cube`)

---

## Related docs

- Endpoint cluster definition (all samples) + ranking (η², Cohen’s *d*, timeline panels): [endpoint_cluster_discrimination.md](endpoint_cluster_discrimination.md)
- Cross-cohort cluster proxy tables/plots (after multiple B\* endpoint runs): `output/endpoint_cluster_cross_cohort/`
- GSA cage-geometry k-diagnostics (after `run_gsa_step1_cluster_k.py`): `output/gsa_cluster_k_diagnostics/`
- Cohort interpretation notes: `output/changepoints/ANALYSIS_SUMMARY.md`
- Presentation-style writeup: `output/changepoints/CHANGPOINT_PIPELINE_PRESENTATION.md`
- Feature extraction entry point: `scripts/compute_gsa_features.py`
- Endpoint finder / observer: `src/EndpointAnalyzer/`
- Low-level ruptures wrapper: `src/utils/ruptures_utils.py`
