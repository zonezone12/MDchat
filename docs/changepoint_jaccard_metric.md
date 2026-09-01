# Changepoint timing Jaccard: metric bug and impact on published tables

**MD_analysis — notes for talks / writeups**  
Code: `src/ChangepointAnalysis/detection.py` (`compare_breakpoints`)  
Used by: `compare_changepoint_timing`, `detect_trajectory_changepoints`, `penalty_sweep._breakpoint_jaccard_vs_ref`  
Pipeline context: [changepoint_pipeline.md](changepoint_pipeline.md) (timing comparison / `cohort_timing_summary.csv`)

These notes record *what* `compare_breakpoints` used to compute, *why* that was not a pairwise Jaccard, and *how much* the **already-written** GSA / endpoint timing CSVs move under the corrected 1-to-1 definition. Detection and clustering do **not** depend on this function; only timing-comparison CSVs, Jaccard heatmaps, and the sweep’s `timing_jaccard_vs_prev` column do.

The function now uses greedy 1-to-1 matching (see §1). Existing `changepoint_timing_comparison.csv` / `cohort_timing_summary.csv` files under `output/` were written with the old formula and are **not** regenerated here. Recompute them from `all_breakpoints.csv` (no re-run of Pelt) after pulling the fix.

---

## 1. What the function used to do (and what it does now)

**Current code** (greedy 1-to-1): `n_shared` = matched pairs within ±tolerance; Jaccard = `n_shared / (n_a + n_b − n_shared)`; offset = mean matched-pair distance. `compare_breakpoints(A, B)` and `(B, A)` agree on those three values.

**Old formula** (still in the published CSVs below):

```python
shared = sum(1 for b in bkps_a if any(abs(b - c) <= tolerance for c in bkps_b))
union_size = len(set(bkps_a) | set(bkps_b))
jaccard = shared / union_size
```

Two independent problems:

| Piece | What it does | Why that is wrong |
|-------|----------------|-------------------|
| **Numerator `n_shared`** | Count of A-points that have *some* B-point within ±50 frames (many-to-one allowed) | Not a set intersection; several A events can claim the same B event |
| **Denominator** | Exact set union `|A ∪ B|` | Does not compose with a fuzzy numerator. Off-by-one-frame matches inflate the union |
| **Offset** | Mean nearest-neighbor distance A→B (Chamfer), including unmatched points | `compare_breakpoints(A, B)` ≠ `compare_breakpoints(B, A)` |

Published group order is alphabetical (`combined` before `gsa` before `iodine` before `na_water`), so the A→B direction is systematic, not random.

### Corrected definition (now in `compare_breakpoints`)

Greedy 1-to-1 matching of breakpoints within ±50 frames (sort candidate pairs by distance, never reuse a point). Then:

- `n_shared` = number of matched pairs
- Jaccard = `n_shared / (n_a + n_b − n_shared)`
- Offset = mean distance of the matched pairs (NaN if none)

When every match is an **exact** frame identity, old Jaccard already equals this. Bias in the CSVs below appears only when matches are within tolerance but not the same index.

---

## 2. Headline

**Pair rankings hold. Absolute Jaccard is too low. Timing offsets are the worse metric.**

Recomputed from `output/gsa_changepoints_{CUBE}/all_breakpoints.csv` for BHHpH, BHHpM, BMHpH, BMHpM, BMMpH, BMMpM: **1812 trajectory-pairs**, tolerance 50 frames.

| Check | Result |
|-------|--------|
| Spearman (published J vs 1-to-1 J) | **0.97** |
| Group-pair rank order | Unchanged on 5/6 cubes; BMMpH swaps 4th/5th among na_water pairs |
| Combined–gsa median J (pooled) | **0.48 → 0.63** (+0.15) |
| Rows with `n_shared(A,B) ≠ n_shared(B,A)` | **57%** |
| Rows with `|J(A,B) − J(B,A)| > 0.10` | **0.4%** (max 0.14) |
| Mean fuzzy-but-not-exact matches per pair | **8.5** |

The mixed fuzzy/exact Jaccard is the **dominant** bias: the exact union is too large, so published J is pulled **down**, not up. Asymmetry of `n_shared` is common but usually 1–2 events; it rarely moves Jaccard by more than 0.05 (12% of rows).

---

## 3. Pooled GSA group pairs (six cubes, 302 traj each)

Median Jaccard and median offset. Published = current `compare_breakpoints`. Corrected = 1-to-1 match as above.

| Pair | Med J published | Med J 1-to-1 | Δ | n_shared pub / 1-to-1 | Rows with n_shared A≠B | Med offset A→B / B→A / sym (frames) |
|------|----------------:|-------------:|--:|----------------------:|-----------------------:|------------------------------------:|
| combined–gsa | 0.48 | 0.63 | +0.15 | 20 / 20 | 52% | 20 / 42 / 31 |
| combined–iodine | 0.30 | 0.36 | +0.07 | 14 / 13 | 64% | 50 / 67 / 61 |
| gsa–iodine | 0.27 | 0.33 | +0.06 | 15 / 14 | 69% | 61 / 62 / 65 |
| combined–na_water | 0.22 | 0.25 | +0.04 | 7 / 7 | 40% | 133 / 54 / 97 |
| gsa–na_water | 0.21 | 0.23 | +0.02 | 8 / 7 | 50% | 149 / 48 / 101 |
| iodine–na_water | 0.20 | 0.20 | ~0 | 8 / 6 | 65% | 154 / 55 / 107 |

Combined still tracks gsa more than iodine or na_water. Do **not** quote published Jaccard as an overlap fraction. Do **not** compare timing offsets across pairs with different event rates: when A is sparser the published offset is optimistic; when A is denser (anything vs na_water) it is 2–3× too large.

---

## 4. combined–gsa by cube

This is the pair the heatmap and writeups treat as “GSA geometry vs the joint signal.” Every cube moves the same way.

| Cube | Med J published | Med J 1-to-1 | Δ |
|------|----------------:|-------------:|--:|
| BHHpH | 0.53 | 0.70 | +0.18 |
| BHHpM | 0.50 | 0.69 | +0.18 |
| BMHpH | 0.47 | 0.61 | +0.14 |
| BMHpM | 0.46 | 0.61 | +0.15 |
| BMMpH | 0.46 | 0.59 | +0.11 |
| BMMpM | 0.45 | 0.57 | +0.11 |

Published `cohort_timing_summary.csv` values match the “published” column (e.g. BHHpH combined–gsa median 0.528).

---

## 5. Worked trajectory: BHHpH `109345_mdcrd_v`, combined vs gsa

22 combined breakpoints, 28 gsa.

- All 22 combined points have a gsa neighbor within 50 frames.
- Swapping arguments still gives `n_shared = 22` — this row does **not** show the n_shared asymmetry.
- Exact intersection 13, exact union 37 → published J = 22/37 = **0.59**.
- 1-to-1 matched 21 (13 exact + 8 within ±50) → J = 21/(22+28−21) = **0.72**.
- Offset: combined→gsa **8.2** frames vs gsa→combined **47.7** frames (symmetric Chamfer 28; matched-pair mean 6.7).

The extra six gsa points have no partner, so A→B Chamfer looks tight and B→A looks late. Cohort “mean timing offset” for this pair is whichever direction alphabetical order picked.

---

## 6. Where argument order actually bites

Worst swaps are unbalanced pairs with the **larger** set as `group_a` (alphabetical: iodine or gsa vs na_water). Several points in the dense set can sit within 50 frames of the same sparse-set event.

| Cube / traj | Pair (A, B) | n_A / n_B | n_shared A→B / B→A / 1-to-1 | J pub / rev / 1-to-1 |
|-------------|-------------|----------:|----------------------------:|---------------------:|
| BMHpH 932487 | iodine, na_water | 25 / 11 | 11 / 6 / 6 | 0.31 / 0.17 / 0.20 |
| BHHpM 178323 | gsa, na_water | 30 / 14 | 13 / 7 / 7 | 0.30 / 0.16 / 0.19 |
| BMMpH 932487 | combined, iodine | 13 / 22 | 9 / 13 / 9 | 0.26 / 0.38 / 0.35 |
| BMHpM 907683 | gsa, na_water | 33 / 12 | 15 / 10 / 10 | 0.34 / 0.23 / 0.29 |

Max Jaccard asymmetry among 1812 GSA rows: **0.14**. Max `n_shared` difference: **6**.

---

## 7. Sparse runs (endpoint vs GSA, legacy `output/changepoints`)

`output/endpoint_vs_gsa_timing_BMMpM` and `output/changepoints` have ~2–3 breakpoints per group. Many-to-one never fires (`n_shared` is fully symmetric). The fuzzy/exact mix still understates Jaccard:

| Pair (BMMpM endpoint vs GSA) | Med J published | Med J 1-to-1 |
|------------------------------|----------------:|-------------:|
| endpoint–gsa (n = 30) | 0.50 | 0.67 |
| combined–endpoint (n = 30) | 0.50 | 0.73 |
| combined–gsa (n = 39) | 0.50 | 0.67 |

The “median Jaccard 0.5” in `endpoint_vs_gsa_timing_BMMpM/cohort_timing_summary.csv` is the same artefact as on the dense GSA tables, just with fewer events.

---

## 8. Penalty sweep

Elbows are computed from **breakpoint counts vs log-penalty**, not from Jaccard — [penalty_sweep_B_cohorts.md](penalty_sweep_B_cohorts.md) operating penalties are untouched.

`timing_jaccard_vs_prev` uses `compare_breakpoints` on consecutive grid steps. Per-penalty breakpoint *lists* were not written (`penalty_sweep_by_trajectory.csv` has counts only), so existing sweep CSVs cannot be recomputed from artifacts; they still contain the old formula. Re-running the sweep would pick up the fix.

Consecutive Pelt steps keep many **exact** indices, which is the case where published Jaccard is already valid. The 0.75 plateau threshold may be slightly conservative if shifted (non-exact) matches are common; it cannot move the elbow. The high-penalty tail still looks “stable” because almost nothing moves — that interpretation in the sweep notes does not depend on the Jaccard formula.

---

## 9. What to trust in tables already on disk

**Trust**

- Pair ranking: combined tracks gsa more than iodine or na_water.
- The statement that cross-group agreement is only moderate (corrected combined–gsa is ~0.6, not ~1).
- Penalty elbows and breakpoint counts.

**Do not quote from old CSVs**

- Published Jaccard as |A ∩ B| / |A ∪ B|.
- `mean_timing_offset_*` compared across pairs with different event rates.
- `compare_breakpoints(A, B)` vs the swapped call as the same number (old formula only).

After this fix, re-running `scripts/compare_changepoint_timing.py` (or `summarize_changepoint_results.py`) on existing `all_breakpoints.csv` directories rewrites the timing CSVs and heatmaps. Detection does not need to be re-run.
