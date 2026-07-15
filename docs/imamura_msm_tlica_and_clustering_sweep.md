# Imamura MSM: tlICA, Clustering Sweep, and Related Extensions

**MD_analysis — developer / user guide**  
*Extends the pipeline in [`imamura_msm_presentation.md`](imamura_msm_presentation.md)*

This document describes recent additions to the Imamura nanocube MSM workflow:

1. **Time-lagged ICA (tlICA / TICA)** as an alternative to PCA before clustering  
2. **Clustering hyperparameter sweep** (`scripts/sweep_imamura_clustering.py`)  
3. **Bead-spec persistence** when re-clustering from saved `imamura_features.csv`  
4. **Cluster-based bead importance** tracing (alongside legacy PCA loadings)

---

## Pipeline overview (updated)

```mermaid
flowchart LR
    A[imamura_features.csv] --> B{dim reduction}
    B -->|pca default| C[PCA → 5D]
    B -->|tlica| D[tlICA → 5 TICs]
    C --> E[MiniBatchKMeans microstates]
    D --> E
    E --> F[Ward → macrostates]
    F --> G[Lagged transitions 2 ns]
    G --> H[Artifacts + bead importance]
```

| Step | PCA path (paper default) | tlICA path (new) |
|------|--------------------------|------------------|
| Input | Sorted inter-bead distance matrix per frame | Same |
| Reduction | Instantaneous variance (PCA) | Slow dynamics at lag τ (TICA) |
| Clustering | MiniBatchKMeans + Ward | Same |
| Transitions | Lagged count matrix | Same |
| Coordinates in `frame_states.csv` | `PC1`…`PC5` | `TIC1`…`TIC5` |

---

## Source layout

| Path | Role |
|------|------|
| `src/utils/imamura_msm.py` | Core library: features, tlICA, clustering, transitions, artifacts |
| `scripts/run_imamura_msm.py` | Full MSM CLI (extract or `--features-csv`) |
| `scripts/sweep_imamura_clustering.py` | Grid search over `n_micro` × `n_macro` |
| `scripts/annotate_imamura_bead_pca.py` | Re-cluster + bead-importance maps from saved CSVs |

### Key dataclasses (`imamura_msm.py`)

```python
ImamuraMSMConfig          # hyperparameters (PCA/tlICA, clustering, lag)
ImamuraTICAModel          # tlICA projection (PCA-compatible interface)
ImamuraClusteringResult   # scores, labels, linkage, eigenvalues/timescales
ImamuraClusteringMetrics  # sweep quality scores per grid point
```

`ImamuraClusteringResult` keeps field names `pca_model` / `pca_scores` for backward compatibility; they hold either a sklearn `PCA` object or an `ImamuraTICAModel`, and PC or TIC scores respectively.

---

## tlICA: theory

**Time-lagged independent component analysis (TICA / tlICA)** finds linear combinations of features with **maximal autocorrelation** at lag time τ. In MSM construction this is the standard alternative to PCA: PCA finds directions of maximal *variance*; TICA finds directions of maximal *slow dynamics*.

Given mean-free feature vectors **r**(t), we estimate:

- **C(0)** — instantaneous covariance  
- **C(τ)** — time-lagged covariance (pairs separated by τ, symmetrized)

The generalized eigenvalue problem:

\[
\mathbf{C}(\tau)\,\mathbf{v}_i = \lambda_i\,\mathbf{C}(0)\,\mathbf{v}_i
\]

Eigenvalues λᵢ ≈ 1 correspond to **slow** processes. Implied relaxation timescale:

\[
t_i = -\frac{\tau}{\ln |\lambda_i|}
\]

(τ in ns when lag is specified in ns.)

**References:** Noé & Clementi (2015); PyEMMA / deeptime TICA documentation; Markov model lecture notes ([markovmodel.org TICA](http://docs.markovmodel.org/lecture_tica.html)).

---

## tlICA: code structure

All tlICA logic lives in `src/utils/imamura_msm.py` in a small, layered stack:

```
fit_imamura_tlica()              # public entry point
    ├── StandardScaler           # per-feature z-score (sklearn)
    ├── _tica_covariance_matrices()
    │       └── _tica_trajectory_segments()   # split by traj_id
    ├── scipy.linalg.eigh(Cτ, C₀)  # generalized eigen decomposition
    └── ImamuraTICAModel         # stores components, eigenvalues, timescales
```

### 1. Trajectory-aware segments

Lag pairs must **not** cross trajectory boundaries. `_tica_trajectory_segments` splits the standardized matrix by `traj_id` (from `imamura_features.csv`):

```python
def _tica_trajectory_segments(X, traj_ids):
  # if traj_ids is None → treat all rows as one trajectory
  # else → one segment per unique traj_id (pandas order)
```

### 2. Covariance accumulation

`_tica_covariance_matrices` loops segments:

```text
For each trajectory segment:
  1. center:  x ← x - mean(segment)
  2. C₀  += xᵀ x
  3. x_t   = x[:-lag]
     x_τ   = x[lag:]
     C_τ  += x_tᵀ x_τ
Normalize:
  C₀  /= (n₀ - 1)
  C_τ  /= n_τ
Symmetrize:
  C_τ ← ½(C_τ + C_τᵀ)
```

### 3. Regularization and eigensolve

High-dimensional distance features (e.g. 2571-D endpoint mode) need a small shrinkage on **C(0)**:

```python
reg = regularization * trace(C0) / n_features   # default regularization = 1e-6
C0 += reg * I
eigvals, eigvecs = scipy.linalg.eigh(Ctau, C0)
```

Components are sorted by **descending** eigenvalue (slowest first). Top `n_components` eigenvectors become rows of `components_` (shape `n_components × n_features`), matching sklearn PCA layout.

### 4. `ImamuraTICAModel`

```python
@dataclass
class ImamuraTICAModel:
    components_: np.ndarray           # (n_comp, n_feat)
    eigenvalues_: np.ndarray          # TIC eigenvalues λᵢ
    mean_: np.ndarray                 # StandardScaler mean
    scale_: np.ndarray                # StandardScaler std
    lag_frames: int
    lag_ns: float
    implied_timescales_ns: np.ndarray

    def transform(self, X):
        Xs = (X - self.mean_) / self.scale_
        return Xs @ self.components_.T
```

Projection scores fed to clustering: `scores = X @ components_.T` (same pattern as PCA).

### 5. Integration with clustering

`cluster_imamura_features()` dispatches on `ImamuraMSMConfig.dimensionality_reduction`:

```python
if method == "tlica":
    model, scores, evr = fit_imamura_tlica(
        feature_matrix,
        n_components=cfg.n_pca_components,
        traj_ids=traj_ids,
        lag_frames=tlica_lag_frames,
        lag_ns=lag_ns,
        regularization=cfg.tlica_regularization,
    )
elif method == "pca":
    model, scores, evr = fit_imamura_pca(...)
```

Then **the same** `two_stage_imamura_clustering()` runs on `scores` (MiniBatchKMeans → Ward).

`explained_variance_ratio` for tlICA is a **normalized weight** derived from clipped eigenvalues (used for plotting weights, not literal variance explained).

### 6. Lag conversion

Frame lag for tlICA uses the same helper as MSM transitions:

```python
lag_frames = lag_frames_from_ns(lag_ns, dt_ps, stride)
# lag_ns [ns] → lag_ps → divide by effective dt (ps × stride)
```

For Amber TRJ on BMMpM, **`--dt-ps 2.0`** is required so 2 ns → 1000 frames.

---

## Clustering hyperparameter sweep

`scripts/sweep_imamura_clustering.py` fits **PCA or tlICA once**, then grids:

- `n_micro` — MiniBatchKMeans microclusters (default: 500…2500)  
- `n_macro` — Ward macrostates (default: 12…30)

### Scoring (`evaluate_imamura_clustering`)

Each grid point is ranked by a **composite score** combining:

| Metric | Role |
|--------|------|
| Macro silhouette | Separation in PC/TIC space (subsampled to 8000 rows) |
| Macro occupancy | Fraction of requested macrostates actually populated |
| Micro occupancy | Fraction of requested microclusters occupied |
| Population balance | Penalizes Gini skew across macrostate populations |
| Min macro population | Drops solutions with tiny (<1%) states |

Best row is written to `recommendation.txt`. With `--run-final` (default **on**), `run_imamura_msm.py` is invoked automatically with the winning `n_micro` / `n_macro`.

### Output directories

| `--dim-reduction` | Sweep dir | Tuned MSM dir |
|-------------------|-----------|---------------|
| `pca` (default) | `<features-dir>/cluster_sweep/` | `cluster_tuned/` |
| `tlica` | `<features-dir>/tlica_cluster_sweep/` | `tlica_tuned/` |

### Example commands

**PCA sweep + tuned run (endpoint BMMpM):**

```bash
python scripts/sweep_imamura_clustering.py \
    --features-csv output/imamura_msm/endpoint_BMMpM/imamura_features.csv \
    --topology traj/BMMpM_ca.prmtop \
    --trajectories traj/BMMpM_891249_mdcrd_v.trj \
    --format TRJ \
    --dt-ps 2.0
```

**tlICA sweep + tuned run:**

```bash
python scripts/sweep_imamura_clustering.py \
    --features-csv output/imamura_msm/endpoint_BMMpM/imamura_features.csv \
    --dim-reduction tlica \
    --tlica-lag-ns 2.0 \
    --topology traj/BMMpM_ca.prmtop \
    --trajectories traj/BMMpM_891249_mdcrd_v.trj \
    --format TRJ \
    --dt-ps 2.0
```

**Sweep only:**

```bash
python scripts/sweep_imamura_clustering.py \
    --features-csv ... --dim-reduction tlica --dt-ps 2.0 \
    --no-run-final
```

---

## Bead-spec persistence (`--features-csv`)

When re-clustering from a saved feature table, bead definitions for plots and importance tracing must match the **original extraction**, not CLI defaults.

`run_imamura_msm.py` now:

1. Looks for `bead_spec.json` next to `--features-csv` (or `--bead-spec` override)  
2. Loads it via `load_imamura_bead_spec()`  
3. Skips re-resolving methyl vs endpoint beads from topology defaults  

This prevents a common pitfall: clustering on 2571-D endpoint features while plotting 18 methyl beads.

---

## CLI reference (`run_imamura_msm.py`)

### Dimensionality reduction

| Flag | Default | Description |
|------|---------|-------------|
| `--dim-reduction` | `pca` | `pca` or `tlica` |
| `--n-pca` | 5 | Number of PC or TIC components |
| `--tlica-lag-ns` | same as `--transition-lag-ns` | tlICA covariance lag |
| `--tlica-regularization` | `1e-6` | Shrinkage on C(0) |

### Clustering (unchanged paper defaults overridable)

| Flag | Default |
|------|---------|
| `--n-micro` | 1500 |
| `--n-macro` | 22 |
| `--transition-lag-ns` | 2.0 |
| `--features-csv` | — (skip extraction) |
| `--bead-spec` | sibling of features CSV |

### Example: tlICA MSM from saved features

```bash
python scripts/run_imamura_msm.py \
    --features-csv output/imamura_msm/endpoint_BMMpM/imamura_features.csv \
    --dim-reduction tlica \
    --tlica-lag-ns 2.0 \
    --n-micro 2000 --n-macro 12 \
    --topology traj/BMMpM_ca.prmtop \
    --trajectories traj/BMMpM_891249_mdcrd_v.trj \
    --format TRJ \
    --dt-ps 2.0 \
    --output-dir output/imamura_msm/endpoint_BMMpM/tlica_tuned
```

---

## Output artifacts

### Shared (PCA and tlICA)

```
output/.../
├── bead_spec.json
├── imamura_features.csv
├── pair_identity_trace.npz  # exact rank→physical-pair mapping recorded in-pass
├── pair_index_map.csv       # canonical pair/bead/atom metadata
├── frame_states.csv          # traj_id, frame, features, micro/macro labels, PCs or TICs
├── micro_to_macro.csv
├── micro_pca_centroids.csv
├── pca_microstates.png       # PC1/2 or TIC1/2 scatter (micro centroids)
├── microcluster_ward_dendrogram.png
├── state_transition_matrix.csv
├── state_transition_prob.csv
├── state_transitions.csv
├── transition_matrix.png
├── transition_network.png
├── bead_cluster_importance.csv
├── auto_beads_cluster_top5.png
├── auto_beads_cluster_top5_3d.html
└── pipeline_summary.txt
```

### tlICA-specific

| File | Contents |
|------|----------|
| `tlica_eigenvalues.csv` | `eigenvalue`, `implied_timescale_ns`, `tlica_lag_ns`, `tlica_lag_frames` per TIC |
| `frame_states.csv` columns | `TIC1`…`TIC5` instead of `PC1`…`PC5` |

### Sweep-specific

| File | Contents |
|------|----------|
| `cluster_sweep.csv` | One row per (n_micro, n_macro) with metrics |
| `recommendation.txt` | Best parameters + run command |
| `heatmap_silhouette_macro.png` | Silhouette in PC/TIC space |
| `heatmap_composite_score.png` | Composite ranking |
| `top_candidates.png` | Bar chart of top grid points |

---

## BMMpM endpoint results (reference)

Dataset: `endpoint_BMMpM/imamura_features.csv` — 53,324 frames × 2571 features, 72 endpoint beads.

### PCA sweep (`cluster_sweep/`)

| Setting | Value |
|---------|-------|
| Best | **750 micro / 12 macro** |
| Macro silhouette (PCA) | 0.22 |
| Paper default 1500/22 | silhouette 0.14 (near bottom of grid) |

### tlICA sweep (`tlica_cluster_sweep/`)

| Setting | Value |
|---------|-------|
| Best | **2000 micro / 12 macro** |
| Macro silhouette (TIC) | 0.062 |
| tlICA lag | 2 ns → 1000 frames |
| Leading TIC eigenvalues | 0.73, 0.35, 0.31 |
| TIC1 implied timescale | ~6.3 ns |

**Note:** TIC-space silhouettes are **lower** than PCA silhouettes. That is expected: TICs optimize slow dynamics, not cluster separation. Compare runs by transition-network interpretability and physical bead drivers, not silhouette alone.

---

## Python API (minimal)

```python
from src.utils.imamura_msm import (
    ImamuraMSMConfig,
    cluster_imamura_features,
    fit_imamura_tlica,
    sweep_imamura_clustering,
    load_imamura_features_csv,
    lag_frames_from_ns,
)

# Load saved features
df, feat_names = load_imamura_features_csv("output/.../imamura_features.csv")
X = df[feat_names].to_numpy()
traj_ids = df["traj_id"].to_numpy()

# tlICA only
lag_frames = lag_frames_from_ns(2.0, dt_ps=2.0)
model, scores, weights = fit_imamura_tlica(
    X, n_components=5, traj_ids=traj_ids,
    lag_frames=lag_frames, lag_ns=2.0,
)

# Full clustering
cfg = ImamuraMSMConfig(
    dimensionality_reduction="tlica",
    n_pca_components=5,
    n_microclusters=2000,
    n_macrostates=12,
)
result = cluster_imamura_features(
    X, config=cfg, traj_ids=traj_ids,
    tlica_lag_frames=lag_frames, tlica_lag_ns=2.0,
)

# Sweep
sweep_df = sweep_imamura_clustering(
    X,
    n_micro_values=[750, 1000, 1500, 2000],
    n_macro_values=[10, 12, 15, 18],
    dimensionality_reduction="tlica",
    traj_ids=traj_ids,
    tlica_lag_frames=lag_frames,
    tlica_lag_ns=2.0,
)
```

---

## State-transfer and changepoint comparison

`scripts/compare_state_methods.py` builds matched PCA and tlICA assignments
from the same saved Imamura CV rows, then asks two separate questions:

1. **Are the states reproducible?** PCA, tlICA, and combined-changepoint labels
   are compared with label-invariant ARI, NMI, V-measure, contingency tables,
   and transition-boundary timing.
2. **What changes during a transfer?** For every directed macrostate transfer,
   the script compares pre- and post-transfer `v1_*` / `v4_*` windows and ranks
   CVs by standardized change and consistency across events.

LDA is a **post-hoc supervised interpretation**, not a third state-discovery
method. It projects the already-assigned PCA or tlICA macrostates, reports
central-ring versus endpoint loadings, and uses leave-one-trajectory-out
validation when multiple trajectories are present.

### Required inputs

- An endpoint-style `frame_states.csv` that still contains `traj_id`, `frame`,
  `time_ps`, and all raw `v1_*` / `v4_*` columns. Existing labels and PC/TIC
  columns are discarded before creating the matched variants.
- `output/changepoints/clusters/combined/segments_clustered.csv`.
- `output/changepoints/all_breakpoints.csv`.
- Matching trajectory IDs. If the two pipelines use different names, provide
  an explicit two-column map; the script never guesses trajectory identity.

The reference `output/imamura_msm/BMMpM/frame_states.csv` uses the single ID
`mdcrd_v`, while the changepoint cohort uses IDs such as
`BMMpM_109234_mdcrd_v`. It is suitable for checking the input schema but not
for a cohort comparison without a scientifically valid ID mapping. Use the
multi-trajectory endpoint result for the final analysis.

### Command

```bash
python scripts/compare_state_methods.py \
  --imamura-frame-states output/imamura_endpoint/frame_states.csv \
  --changepoint-segments output/changepoints/clusters/combined/segments_clustered.csv \
  --changepoint-breakpoints output/changepoints/all_breakpoints.csv \
  --output-dir output/state_comparison \
  --n-components 5 \
  --n-micro 2000 \
  --n-macro 12 \
  --tlica-lag-ns 2.0 \
  --min-dwell-rows 5 \
  --window-rows 25 \
  --boundary-tolerance-ps 500
```

Optional explicit ID map:

```csv
imamura_traj_id,changepoint_traj_id
endpoint_run_109234,BMMpM_109234_mdcrd_v
```

Pass `--gsa-features-dir output/gsa_features` to add event-aligned deformation,
pore, gear, interface-contact, guest, and solvent observables. Pass
`--write-aligned-traces` only when the large long-form event/CV table is
needed.

`run_imamura_msm.py` records exact rank-to-pair identities during the original
feature-extraction pass. This adds no second trajectory read and writes:

- `pair_identity_trace.npz` — compact integer arrays mapping every frame and
  `v1_*`/`v4_*` rank to a canonical physical pair.
- `pair_index_map.csv` — canonical pair index to bead and atom metadata.

Recording is enabled by default; use `--no-pair-trace` only when the extra
binary artifact is not needed. When comparison uses the sibling
`frame_states.csv`, both trace files are discovered automatically:

```bash
python scripts/compare_state_methods.py \
  --imamura-frame-states output/imamura_endpoint/frame_states.csv \
  --output-dir output/state_comparison \
  --exact-top-features 10
```

For files stored elsewhere, pass `--pair-identity-trace` and
`--pair-index-map` explicitly. The trace rows are validated against
`traj_id`/`frame`; mismatched extraction and state tables are rejected.

### Principal outputs

Each `pca/` and `tlica/` output folder contains:

- `frame_states.csv` — matched state assignment.
- `lda_loadings.csv`, `lda_state_centroids.csv`, `lda_validation.csv` —
  supervised state interpretation and held-out separability.
- `transition_events.csv` — directed state transfers after short A-B-A
  recrossing suppression.
- `transition_event_deltas.csv` — per-event pre/post CV changes.
- `transition_signatures_top.csv` — strongest and most consistent CV changes
  for each directed state pair, with boundary slopes, peak-change timing, and
  bootstrap confidence intervals.
- `state_agreement.csv`, `state_contingency.csv` — state-label comparison.
- `boundary_matches.csv`, `boundary_agreement_by_trajectory.csv` — temporal
  support from combined changepoints.
- `transition_events_changepoint_support.csv` — marks transfers supported by a
  nearby combined changepoint.
- `exact_transition_pair_trace.csv` — for each event/frame/CV rank, the exact
  current bead pair, distance, monomer assignment, atom indices/IDs/names, and
  pre/post relative time. Built from the recorded trace without rereading raw
  trajectories.
- `exact_transition_pair_summary.csv` — physical-pair rank occupancy and
  pre/post distance change for each directed state transfer.

The root `pca_vs_tlica_agreement.csv` measures how strongly dimensionality
reduction changes the state partition.
`pca_vs_tlica_transition_consensus.csv` maps tlICA states onto PCA states for
display and identifies transfer CVs whose change direction agrees between both
reductions. With GSA inputs,
`pi_proxy_vs_deformation_event_correlations.csv` reports whether endpoint/ring
CV changes co-vary across events with RMSD, deformation, pore, gear, and
interface-contact changes.

### Interpretation limits for the pi system

When endpoint selection includes aromatic atoms, changing `v4_*` distances can
show that the endpoint/pi-system geometry rearranges during deformation.
Likewise, `v1_*` tracks central-benzene centroid separation. These remain
**distance-distribution proxies**:

- they do not contain ring-normal orientation or lateral stacking slip;
- sorted Imamura slots do not preserve one fixed atom-pair identity over time;
  exact tracing resolves the dynamic pair occupying the rank but does not add
  missing orientation information;
- temporal precedence and correlation do not by themselves prove causality.

Therefore, report a result as “endpoint/pi-system distance rearrangement
precedes or accompanies deformation,” not “pi stacking causes deformation,”
unless independent orientation-specific aromatic CVs are added later.

---

## Dependencies

tlICA uses only existing core dependencies — **no PyEMMA / deeptime required**:

- `numpy`, `scipy.linalg.eigh`, `sklearn.preprocessing.StandardScaler`
- `pandas` for trajectory grouping and sweep tables

---

## Related docs

- [`imamura_msm_presentation.md`](imamura_msm_presentation.md) — original pipeline, bead modes, formation detection  
- `scripts/run_imamura_msm.py` module docstring — copy-paste examples  
- `scripts/sweep_imamura_clustering.py` module docstring — sweep examples  
