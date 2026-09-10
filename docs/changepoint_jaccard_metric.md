# Changepoint timing Jaccard

**MD_analysis — notes for talks / writeups**  
Code: `src/ChangepointAnalysis/detection.py` (`compare_breakpoints`)  
Used by: `compare_changepoint_timing`, `detect_trajectory_changepoints`, `penalty_sweep._breakpoint_jaccard_vs_ref`  
Figures: `scripts/plot_jaccard_metric_comparison.py` → `docs/figures/`  
Reads: `output/gsa_changepoints_{CUBE}/all_breakpoints.csv` and `output/endpoint_changepoints_{CUBE}/all_breakpoints.csv`  
Pipeline context: [changepoint_pipeline.md](changepoint_pipeline.md)

Detection and clustering do **not** use this function. It only feeds timing CSVs, Jaccard heatmaps, and the sweep panel `timing_jaccard_vs_prev`.

```powershell
python scripts/plot_jaccard_metric_comparison.py
```

---



## 1. Metric

Greedy 1-to-1 matching within ±50 **original trajectory frames** (sort candidate pairs by distance; each breakpoint used at most once):


| Field      | Definition                                 |
| ---------- | ------------------------------------------ |
| `n_shared` | Number of matched pairs                    |
| Jaccard    | `n_shared / (n_a + n_b − n_shared)`        |
| Offset     | Mean distance of those pairs (NaN if none) |


`compare_breakpoints(A, B)` and `(B, A)` agree on those three values.

The previous formula mixed a fuzzy A→B many-to-one count with an exact-set union, and scored Chamfer offset A→B only. That pulled Jaccard down and made offsets incomparable across pairs with different event rates.

Endpoint detection is stride-5 (1000 signal rows); GSA is every frame (5000). Both `all_breakpoints.csv` files store the original `frame` column, so endpoint vs GSA is compared on that axis, not on `signal_index`.

---



## 2. GSA group pairs

`compare_changepoint_timing` was run on the existing `gsa_changepoints_{CUBE}/all_breakpoints.csv` files (same Pelt breakpoints as before) and overwrote only the timing products: `changepoint_timing_comparison.csv`, `cohort_timing_summary.csv`, `plots/cohort_jaccard_heatmap.png`. Six cubes, **1812** trajectory-pairs (302 traj × 6 pairs), tolerance 50 frames.

The GSA **Pelt sweep was cancelled**. `penalty_sweep/penalty_sweep_summary.csv` and `penalty_sweep_overview.png` for GSA were **not** touched (still 2026-08-31).

Numbers below are 1-to-1. Grey bars in the figures are the old formula on the same breakpoint lists.

### Pooled (302 traj per pair)


| Pair              | Median J (old → 1-to-1) | Mean J | Median n_shared | Median matched offset (frames) |
| ----------------- | ----------------------- | ------ | --------------- | ------------------------------ |
| combined–gsa      | 0.48 → **0.63**         | 0.62   | 20              | 7.8                            |
| combined–iodine   | 0.30 → 0.36             | 0.36   | 13              | 17                             |
| gsa–iodine        | 0.27 → 0.33             | 0.33   | 14              | 22                             |
| combined–na_water | 0.22 → 0.25             | 0.26   | 7               | 21                             |
| gsa–na_water      | 0.21 → 0.23             | 0.24   | 7               | 21                             |
| iodine–na_water   | 0.20 → 0.20             | 0.20   | 6               | 21                             |


Combined still tracks gsa more than iodine or na_water. Matched-pair offsets sit in a narrow band (~8–22 frames). Old Chamfer offsets spanned 20–154 frames depending on which set was larger.

![Median Jaccard by GSA group pair](figures/jaccard_by_pair_old_vs_1to1.png)

### combined–gsa by cube


| Cube  | n   | Median J (old → 1-to-1) |
| ----- | --- | ----------------------- |
| BHHpH | 50  | 0.53 → **0.70**         |
| BHHpM | 50  | 0.50 → **0.69**         |
| BMHpH | 51  | 0.47 → 0.61             |
| BMHpM | 50  | 0.46 → 0.61             |
| BMMpH | 50  | 0.46 → 0.59             |
| BMMpM | 51  | 0.45 → 0.57             |


![Median combined–gsa Jaccard by cube](figures/jaccard_combined_gsa_by_cube.png)

BHHpH `109345_mdcrd_v`, combined vs gsa (22 vs 28 breakpoints): 21 one-to-one matches, Jaccard **0.72**, mean matched offset 6.7 frames.

---



## 3. Endpoint vs GSA (`endpoint_changepoints_B*`)

`endpoint_changepoints_{CUBE}` has only the `endpoint` group, so pairwise Jaccard is vs `gsa_changepoints_{CUBE}` on original `frame` (±50). **1100** trajectory-pairs (275 overlapping traj × 4 pairs). Cube overlap: BHHpH/BHHpM/BMHpM 50, BMHpH 51, BMMpH 44, BMMpM 30.

Endpoint has ~10 breakpoints per trajectory; GSA `gsa` / `combined` have ~20–28. Overlap is therefore low even with the 1-to-1 formula. Combined still ranks first vs endpoint.


| Pair              | n   | Median J (old → 1-to-1) | Mean J | Median n_shared | Median matched offset (frames) |
| ----------------- | --- | ----------------------- | ------ | --------------- | ------------------------------ |
| combined–endpoint | 275 | 0.21 → **0.23**         | 0.24   | 5               | 12.5                           |
| endpoint–gsa      | 275 | 0.15 → **0.18**         | 0.19   | 5               | 14                             |
| endpoint–na_water | 275 | 0.12 → 0.14             | 0.16   | 2               | 24                             |
| endpoint–iodine   | 275 | 0.09 → 0.10             | 0.11   | 3               | 25                             |


![Median Jaccard by pair, endpoint vs GSA](figures/jaccard_endpoint_pairs_old_vs_1to1.png)

### By cube


| Cube  | n   | endpoint–gsa (old → 1-to-1) | combined–endpoint (old → 1-to-1) |
| ----- | --- | --------------------------- | -------------------------------- |
| BHHpH | 50  | 0.23 → **0.29**             | 0.29 → **0.35**                  |
| BHHpM | 50  | 0.15 → 0.18                 | 0.19 → 0.21                      |
| BMHpH | 51  | 0.16 → 0.19                 | 0.23 → 0.24                      |
| BMHpM | 50  | 0.18 → 0.21                 | 0.24 → **0.29**                  |
| BMMpH | 44  | 0.11 → 0.12                 | 0.14 → 0.16                      |
| BMMpM | 30  | 0.10 → 0.11                 | 0.16 → 0.13                      |


BMMpM combined–endpoint is the one cube where 1-to-1 is *lower* than the old formula (many-to-one had inflated `n_shared` on a sparse pair).

![Median endpoint–gsa Jaccard by cube](figures/jaccard_endpoint_gsa_by_cube.png)

![Median combined–endpoint Jaccard by cube](figures/jaccard_combined_endpoint_by_cube.png)

Chamfer offsets on these pairs are not usable. Combined→endpoint median Chamfer is **347** frames vs **15** frames the other way; matched-pair mean is **12.5**. Endpoint→gsa is the reverse (18 vs 375; matched **14**).

![Timing offset, endpoint vs GSA](figures/offset_endpoint_pairs_old_vs_matched.png)

The earlier `output/endpoint_vs_gsa_timing_BMMpM` table (combined–endpoint 0.73, endpoint–gsa 0.67) used `output/changepoints`, a sparse ~500-frame GSA run with ~2–3 breakpoints per group. That is **not** the B* GSA operating set.

---



## 4. Penalty-sweep overview

`penalty_sweep_overview.png` has a **timing Jaccard vs previous grid step** panel (`compare_breakpoints` on consecutive Pelt outputs). Breakpoint *lists* were not stored — `penalty_sweep_by_trajectory.csv` has counts only — so that panel needs a Pelt re-sweep, not a CSV rewrite.


| Family                                           | Overview PNG         | Status                                                                                                                                                                                                                               |
| ------------------------------------------------ | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Endpoint (6 cubes, 1000 frames, `endpoint` only) | Rewritten 2026-09-01 | Re-swept on the saved grids. Elbows unchanged.                                                                                                                                                                                       |
| GSA (6 cubes, 5000 frames, `gsa` + `iodine`)     | Unchanged            | Not re-swept. rbf Pelt on 5000-frame signals is the cost (grid is capped at 2.5× log(n) because the high end slows ~40×). Consecutive steps keep many exact indices, so the old `timing_jaccard_vs_prev` is already close to 1-to-1. |
| `output/changepoints/penalty_sweep`              | Unchanged            | Historic four-group run; not the B operating set.                                                                                                                                                                                    |


§2 is the GSA timing CSV rewrite. Cancelling the GSA sweep did not roll it back.

### Endpoint sweep, after the fix

Mean step-to-step Jaccard rose **+0.03 to +0.05**. More steps sit above the 0.75 plateau threshold. Count-vs-log-penalty **elbows did not move**.


| Cube  | Elbow J (old → new) | Steps with J ≥ 0.75 (old → new) |
| ----- | ------------------- | ------------------------------- |
| BHHpH | 0.77 → 0.81         | 9 → 14                          |
| BHHpM | 0.73 → 0.81         | 9 → 14                          |
| BMHpH | 0.73 → 0.80         | 8 → 11                          |
| BMHpM | 0.78 → 0.78         | 9 → 14                          |
| BMMpH | 0.75 → 0.78         | 9 → 10                          |
| BMMpM | 0.64 → 0.69         | 10 → 11                         |


A GSA re-sweep would be the same kind of lift, not a change in operating penalty ([penalty_sweep_B_cohorts.md](penalty_sweep_B_cohorts.md)).

Cohort overlay: `python scripts/plot_penalty_sweep_cohorts.py` — endpoint curves use the new Jaccard; GSA curves are the previous CSVs.

---



## 5. Old formula vs 1-to-1 (same breakpoint tables)


| Check | GSA pairs | Endpoint vs GSA |
|-------|-----------|-----------------|
| Headline median J | combined–gsa 0.48 → **0.63** | endpoint–gsa 0.15 → **0.18** |
| Spearman old vs 1-to-1 | 0.97 | 0.99 |
| Rows with `n_shared(A,B) ≠ n_shared(B,A)` | 57% | 32% |
| abs J(A,B)−J(B,A) greater than 0.10 | 0.4% | 0.3% |
| Pair rank order | Unchanged on 5/6 cubes (BMMpH swaps 4th/5th among na_water pairs) | Combined first on every cube |


Pair **ranking** was already right. Absolute Jaccard was too low because the exact union counted fuzzy matches as distinct. Offsets were the badly biased number when counts differed.

![Timing offset by GSA group pair](figures/offset_by_pair_old_vs_matched.png)

Do not quote Jaccard or Chamfer offsets from CSVs written before the fix. Regenerating from `all_breakpoints.csv` does not require Pelt:

```powershell
python scripts/plot_jaccard_metric_comparison.py

python scripts/compare_changepoint_timing.py `
    --changepoints-dirs output/gsa_changepoints_BHHpH `
    --output-dir output/gsa_changepoints_BHHpH

python scripts/compare_changepoint_timing.py `
    --changepoints-dirs output/endpoint_changepoints_BHHpH output/gsa_changepoints_BHHpH `
    --output-dir output/endpoint_vs_gsa_timing_BHHpH
```

