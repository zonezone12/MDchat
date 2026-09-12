# Publication Gap-Closure Plan

**Target:** methods-and-application manuscript on the automated changepoint pipeline applied to GSA nanocubes.
**Companion:** [Manuscript planning brief (artifact)](https://claude.ai/code/artifact/a7594487-ffa7-4190-8ed5-5c7eb65257a0)
**Revised:** G1b decided — **each cube keeps its own elbow**, no forced shared-relative penalty. Automatic, minimal-assumption pipeline is the stated design goal; a matched penalty would itself be a hand-picked human choice. BMMpM reverts from the matched 4.361 test value to its true own elbow, 5.48888, now backed by a genuine full 51-trajectory detection (`output/endpoint_changepoints_g1b_BMMpM`). Headline ratio in §3.3 is the own-elbow number, **1 : 3.18 : 4.70**. **G3 and G4 are now implemented, tested, and run** (real output at `output/endpoint_cluster_chemical_k/`) — reviewed in this revision, including a correction: a prior speculative claim that BMMpM shows no chemical separation at any k was checked and is wrong (see §5, G3).

---

## 0. What the two reference papers actually are

These are **two different papers on two different problems**. Earlier drafts of this plan conflated them.

| | Imamura 2020 | Murata 2026 |
|---|---|---|
| Full cite | Imamura, Yamamoto, Sato, *Chem. Phys. Lett.* **742**, 137135 (2020) | Murata, Kobayashi, Hiraoka, Shimazaki, Tachikawa, *J. Phys. Chem. B* (2026), DOI 10.1021/acs.jpcb.6c01069 |
| Title | Coarse-grained modeling of nanocube self-assembly system and transition network analyses | Metastructure Analysis of Self-Assembled Nanocubes with Different Equatorial Methyl Groups Based on MD Simulations |
| Model | Coarse-grained, 25 beads per monomer, implicit solvent via IBI | All-atom, GAFF/RESP, explicit TIP4P water |
| Sampling | REMD, 12 replicas, 2 µs each | 50 independent 5 ns NVT runs per system |
| Question | How do six dispersed monomers *assemble* into a cube? | How does an *already-assembled* cube deform? |
| States | 22 Ward macrostates from 1500 k-means microstates, 2 ns lag | Four metastructures A, B, C1, C2 from hand-built geometric criteria |
| Relation to us | Reproduced by `src/utils/imamura_msm.py`; a **methodological** precedent for clustering trajectories | **Our trajectory source.** The system, force field, and states we compare against |

**Consequence:** Imamura is background and method precedent, cited in the Introduction. Murata is the paper our results are compared against. Do not describe A/B/C1/C2 as coming from Imamura.

Record both in `docs/references.md`. **D1 is closed.**

---

## 1. Cohort mapping — our six systems vs Murata's four

Verified from the topologies. Nanocube solute (`resname MOL`) atom counts form an exact 18-atom ladder, and 18 atoms = 6 substitution sites x 3 atoms per H-to-CH3 change:

| Our cohort | MOL atoms | Equatorial CH3 | Pole CH3 | Murata system |
|---|---:|---:|---:|---|
| BHHpH | 648 | 0 | 0 | **not in Murata** |
| BHHpM | 666 | 0 | 6 | **3₆** (R1=CH3, R2=R3=H) |
| BMHpH | 666 | 6 | 0 | **not in Murata** |
| BMHpM | 684 | 6 | 6 | **2₆** — confirmed, R2 = H / R3 = CH3 |
| BMMpH | 684 | 12 | 0 | **not in Murata** |
| BMMpM | 702 | 12 | 6 | **1₆** (R1=R2=R3=CH3) |

Naming decodes as `B[eq][eq]p[pole]` (H = hydrogen, M = methyl). Equatorial letters are **HH / MH / MM** (mixed equator is one CH3 on R2 or R3, not the order R2 then R3). Poles bear **R1**. Confirmed occupancy: BHHpH none; BHHpM R1; BMHpH R3; BMHpM R1+R3 (**2₆**); BMMpH R2+R3; BMMpM R1+R2+R3. Canonical site indices: `src/EndpointAnalyzer/gsa_site_map.py`; QC plot `endpoint_sites.png` paints orange atom sites on top of blue rings so R=H para carbons stay visible.

**Two things follow, and both are good for the paper:**

1. **Our `pM` series is Murata's series**, minus one of the two six-methyl isomers. Direct comparison is possible for BHHpM, BMHpM, BMMpM.
2. **Our `pH` series (BHHpH, BMHpH, BMMpH) is not in Murata at all.** Those are pole-hydrogen variants. Murata's ref 19 (Murata et al., *J. Phys. Chem. B* **127**, 328 (2023)) covers pole substitution separately, but not crossed with equatorial substitution. **We have a 3x2 factorial that neither paper has.**

### O1 `[x]` — resolved: BMHpM is 2₆, not 2₆′

**User confirmed:** BMHpM has R2 = H, R3 = CH3, matching Murata's definition of 2₆ exactly ("2₆ has hydrogen atoms at R2 and methyl groups at R3"). Not verifiable independently from the AMBER atom names in `traj/BMHpM_ca.prmtop` (generic sequential `C1`–`C114` labels with no R-group tagging), so this rests on the user's knowledge of the build.

**No numbers in this plan need correcting.** Section 3.3's Murata reference values (A↔B = 8 per ns for the pole-methyl middle term) already used 2₆'s figure, not 2₆′'s — that assumption turned out to be right. Murata's headline finding is precisely that 2₆ and 2₆′ diverge sharply despite equal methyl count (metastructure A at 45.3% vs 19.7%, opposite transition kinetics out of B), so this confirmation matters for G5: our BMHpM is comparable only to the **2₆** column of Murata's Table 1. **2₆′ has no corresponding cohort in our dataset** — we cannot build a contingency table against it, and the G5 population comparison and "other"-resolution tasks below should say so explicitly rather than silently comparing BMHpM against both columns.

---

## 2. Murata's criteria, and why our implementation does not match them

**Still the highest-priority gap.** The G1 re-detection did not touch this.

### What Murata actually defines

| Quantity | Definition | Thresholds |
|---|---|---|
| cation–π opening | Distance between **Py⁺ near the pole** and **Ph** | **open when ≥ 6.5 Å** |
| `d1` | **C2–C3**: ipso-carbons bonding to R2 and R3, across an equatorial edge between two GSAs. Six sites (S6) | normal **4.5–5.5 Å**; elongated **≥ 7.0 Å** |
| `d2` | **CPy–C3**, CPy = carbon of equatorial Py⁺. Six sites | used for the methyl/vdW analysis, no threshold |

Metastructures:

| | opened cation–π | d1 condition | RMSD |
|---|---|---|---|
| **A** | 0 | — | ≈ 1.0 Å |
| **B** | 1 | — | ≈ 1.5 Å |
| **C1** | 2 | no d1 ≥ 7.0 Å | ≈ 1.6–2.5 Å |
| **C2** | 2 | one d1 ≥ 7.0 Å | ≈ 2.7 Å |
| **other** | anything else, including ≥ 3 opened | — | — |

### What our code does

1. **The cation–π distance does not exist in this repository.** No Py⁺–Ph pole-to-equator distance anywhere in `src/` or `scripts/`. Murata's *primary* classification axis is not computed at all.
2. **`paper_d1` uses the 4.5–5.5 Å window with inverted meaning.** `endpoint_features.py:41-42` sets `paper_d1_open_lo = 4.5`, `paper_d1_open_hi = 5.5`, and `paper_d1_n_open` counts values *inside* that window (line 176). In Murata that window is the **normal, compact** state.
3. **Hull `s3`/`s7` d1 was the wrong contact.** Those values ran ~9.4–17 Å because the sites were not R2/R3 (on BMMpM they were R1 and R3). **Remapped** (`output/murata_d1/`): six equatorial R2(i)–R3(j) edges from stored `endpoint_dist_*` columns. BHHpH/BHHpM median ≈ 5.0 Å, 75–77% in 4.5–5.5 Å, 10–14% ≥ 7 Å — Murata's compact window. BMH/BMM still use methyl-carbon sites (~4.3 Å median on BMM), so they are not yet C2–C3. Canonical roles: `src/EndpointAnalyzer/gsa_site_map.py`.
4. **`d2` is not implemented.**

### What this invalidates

The earlier claim that **BMMpH and BMMpM show 0% open cation–π** is an artifact of a mis-specified criterion. BMMpM is Murata's 1₆, their *most stable* system at 78.4% metastructure A, so "zero closed structures" is backwards. Claim withdrawn.

### G0 `[~]` Implement Murata's criteria properly — **BLOCKS G5 and every metastructure claim**

1. Identify, per monomer, the pole Py⁺ ring, the equatorial Py⁺ ring, and the Ph ring forming each of the six cation–π units (Murata Figure 1c).
2. Compute the pole-Py⁺-to-Ph distance per unit per frame; flag open at ≥ 6.5 Å; emit `n_open_cation_pi`.
3. Reimplement `d1` as the C2–C3 ipso-carbon distance across each of the six equatorial edges. **Sanity gate:** the distribution must be bimodal near 4.5–5.5 Å and above 7.0 Å. **`[~]` remap done:** `scripts/remap_murata_d1.py` → `output/murata_d1/`. BHHpH/BHHpM pass the gate (median ≈ 5.0 Å). Remaining: ipso–ipso extract for CH3 cubes (BMH/BMM methyl-carbon proxy is ~0.7 Å short).
4. Implement `d2` (CPy–C3).
5. Emit a per-frame metastructure label A / B / C1 / C2 / other using Murata's rules exactly.
6. Compute RMSD to the initial NVT structure and check peaks near 1.0 / 1.5 / 1.6–2.5 / 2.7 Å. **This is the strongest available validation of the reimplementation.**
7. Rename `paper_d1_n_open` to `d1_n_compact`, or remove it.

**Code:** `src/EndpointAnalyzer/gsa_site_map.py`, `src/ChangepointAnalysis/murata_d1.py` (`scripts/remap_murata_d1.py`), `src/ChangepointAnalysis/endpoint_features.py` (`resolve_paper_d1_atoms`), plus a new cation–π observer. Hull `s3`/`s7` is fallback only for non-GSA monomers.

---

## 3. Results from the matched-penalty re-detection

Source: `output/matched_penalty_recompute_summary.csv`, `output/{endpoint,gsa}_changepoints_matched_*`, `output/endpoint_vs_gsa_timing_matched_*`. Detection only, no re-clustering.

### 3.1 Penalties actually used — final, per G1b

**G1b decision: each cube keeps its own elbow.** No cohort is forced onto another's relative penalty. The rationale is the pipeline's own design goal — automatic detection with minimal human choices — and a shared "matched" relative penalty is itself a hand-picked assumption, the opposite of that goal. G1a's grid-bug fix stands (two cohorts really were computed on the wrong signal length); the separate "matched-relative" test run that forced BMMpM onto BHHpH's 0.631× step is **rejected** and superseded below.

| Cube | Endpoint pen | rel. to log(n) | GSA pen | rel. to log(n) |
|---|---:|---:|---:|---:|
| BHHpH | 4.361 | 0.631 | 5.028 | 0.590 |
| BHHpM | 4.361 | 0.631 | 4.198 | 0.493 |
| BMHpH | 4.361 | 0.631 | 5.028 | 0.590 |
| BMHpM | 3.466 | 0.502 | 5.028 | 0.590 |
| BMMpH | 3.466 | 0.502 | 5.028 | 0.590 |
| BMMpM | **5.489** | **0.795** | 5.028 | 0.590 |

Relative values computed against the true endpoint signal length, log(1000) = 6.908. GSA penalties were already per-cohort own-elbows from the start of G1 and are untouched by this decision.

**BMHpH: `docs/penalty_sweep_B_cohorts.md` §7 item 5 grid bug, fixed in G1a.** Its sweep grid was built from the first CSV length rather than the cohort median. Re-swept on the correct n = 1000 grid (1.38–34.54, 15 steps), the true elbow is 4.361 (0.631×) — not the wrong-grid value of 2.801 (0.405×). This happens to land on the same relative step as BHHpH and BHHpM; that is a coincidence of the data, not a forced match. Confirmed with a full 51-trajectory detection at `output/endpoint_changepoints_g1a_BMHpH` (233 breakpoints, rate 4.569/traj).

**BMMpM: same grid bug, fixed, and its true elbow does not land near the others.** Re-swept on the correct grid, BMMpM's genuine elbow is **5.489 (0.795×)** — even higher than the wrong-grid value of 5.256 (0.761×). The grid bug was not what put BMMpM at a high relative penalty; its count-vs-penalty curve simply kinks later than the other five cohorts. **Confirmed with a full, from-scratch 51-trajectory detection** at `output/endpoint_changepoints_g1b_BMMpM` (penalty 5.48888): **106 breakpoints, rate 2.078/traj, 19/51 trajectories silent (37%)** — exactly matching the earlier sweep-only estimate, so no residual denominator issue. Timing recomputed at `output/endpoint_vs_gsa_timing_g1b_BMMpM`.

Outputs: `output/endpoint_changepoints_g1a_BMHpH/`, `output/endpoint_changepoints_g1b_BMMpM/`, `output/endpoint_vs_gsa_timing_g1b_BMMpM/`, `output/g1a_relative_penalty_correction.csv` (sweep-comparison record only — its own-elbow row for BMMpM is now independently confirmed by the full run above), `output/endpoint_changepoints_{CUBE}/penalty_sweep_median_n/`.

**The rejected matched-relative run** (`output/endpoint_changepoints_g1a_BMMpM`, `output/endpoint_vs_gsa_timing_g1a_BMMpM`, penalty 4.361) still exists on disk as a sensitivity check but is not the reported result. Do not quote its 2.882/traj or 0.111 Jaccard figures as the pipeline's output.

### 3.2 Event rates, published vs matched

Breakpoints per trajectory. Denominator is full cohort size (50 or 51).

| Cube | Endpoint published | Endpoint matched | Δ | GSA cage published | GSA cage matched | Δ |
|---|---:|---:|---:|---:|---:|---:|
| BHHpH | 10.70 | 10.06 | −6% | 31.02 | 49.02 | +58% |
| BHHpM | 6.30 | 9.76 | +55% | 33.84 | 60.80 | +80% |
| BMHpH | 5.98 | 7.51 | +26% | 28.29 | 44.75 | +58% |
| BMHpM | 6.38 | 6.60 | +3% | 27.80 | 43.10 | +55% |
| BMMpH | 2.74 | 3.12 | +14% | 21.80 | 33.98 | +56% |
| BMMpM | 1.63 | 2.08 | +28% | 18.20 | 29.16 | +60% |

The GSA increase of 55–80% confirms the prediction in `docs/penalty_sweep_B_cohorts.md` section 4 that moving from log(n) to the elbow would add roughly 60% more cage events.

**Own-elbow endpoint rates, final (G1b):** BMHpH **4.57** (was 7.51 on the wrong grid), BMMpM **2.078** (was 1.63 published; genuine own-elbow full-cohort run, see §3.1). GSA columns unchanged throughout.

### 3.3 The headline result, own elbow per cube (final)

**Correction to earlier drafts of this plan:** two prior versions of this table used numbers that are now superseded — first a mix of elbow-projected and published rates, then a "matched-relative" test that forced BMMpM onto BHHpH's penalty step. **G1b rejected the matched-relative approach.** Per the stated goal of a minimal-assumption automatic pipeline, each cube keeps its own elbow, full stop. The table below is the final version.

Murata's pole-methyl series against our `pM` cohorts, own elbow throughout:

| Equatorial CH3 | Murata | Murata A↔B per ns | Our cohort | Own-elbow rate |
|---:|---|---:|---|---:|
| 12 | 1₆ | 3 | BMMpM | 2.078 |
| 6 | 2₆ | 8 | BMHpM | 6.60 |
| 0 | 3₆ | 11 | BHHpM | 9.76 |
| ratio vs 1₆ | | **1 : 2.67 : 3.67** | | **1 : 3.18 : 4.70** |

**This overshoots Murata's ratio, and that is the honest, automatic-pipeline answer.** BMMpM's own elbow (5.489, confirmed by a full 51-trajectory detection) sits at a higher relative penalty than the other five cohorts — 0.795× log(n) against 0.50–0.63× elsewhere — because its breakpoint-count-vs-penalty curve genuinely kinks later, not because of any residual grid bug. Forcing it onto a shared relative penalty (the earlier, now-rejected matched run) narrowed the gap to Murata's ratio, but only by imposing a human-chosen alignment the pipeline itself has no principled way to select. The wider gap here is the price of "detect the elbow and use it," with no manual steering per cohort.

The `pH` series, own elbow throughout: **1 : 1.46 : 3.22** (BMMpH 3.12, BMHpH 4.57, BHHpH 10.06) — unaffected by this decision, since BMHpH's own elbow was never in dispute. The equatorial-methyl gradient is weaker at the pole-H end (3.22× top, 1.46× middle) than at the pole-CH3 end (4.70× top, 3.18× middle). That interaction between pole and equatorial substitution is itself a result, not a confound, and it is not something Murata's design (equatorial substitution only) could have shown.

### 3.4 Two controls that strengthen the result

**Cage geometry shows the same trend but much weaker.** GSA cage events per trajectory across the `pM` series: BHHpM 60.8, BMHpM 43.1, BMMpM 29.2, giving **1 : 1.48 : 2.08**. Against Murata's **3.67x**, the own-elbow endpoint coordinate gives **4.70x** (§3.3) while whole-cage geometry gives **2.08x**.

**Endpoint still tracks Murata's methyl dependence more closely than cage geometry, on a relative-error basis, though both now sit on the same automatic footing.** Relative to Murata's 3.67x: endpoint overshoots by 28% (4.70 vs 3.67), cage geometry undershoots by 43% (2.08 vs 3.67). Endpoint is still the closer match, but the comparison is now "smaller error on the same side of no forced alignment" rather than the near-exact 8% agreement reported under the (now-rejected) matched-penalty run. That earlier close match depended on picking BMMpM's penalty to align with BHHpM's; it is not a property of the automatic pipeline itself.

GSA penalties were untouched by G1b (they were already independently own-elbow for each cohort from the start of G1), so the cage ratio of 2.08x stands as reported with no further correction needed.

**The guest is a genuine negative control.** Iodine-group events per trajectory are essentially flat across all six cohorts: 47.2, 58.7, 45.1, 48.8, 49.5, 48.8. The guest does not care how many equatorial methyls the cage has. **The methyl effect is specific to cage deformation, not a global sampling artifact.** This is the cleanest control available and it should be a figure panel.

### 3.5 Cross-family timing, own elbow throughout (final)

Median per-trajectory Jaccard, ±50 frame tolerance, offsets in ps. Every row is now the own-elbow detection, per G1b:

| Cube | endpoint–gsa | offset (ps) | n traj | endpoint–iodine | gsa–iodine | offset (ps) |
|---|---:|---:|---:|---:|---:|---:|
| BHHpH | 0.202 | 12.9 | 50 | 0.168 | 0.516 | 19.6 |
| BHHpM | 0.157 | 13.3 | 50 | 0.140 | 0.557 | 17.0 |
| BMHpH | 0.093 | 11.7 | 51 | 0.125 | 0.471 | 19.0 |
| BMHpM | 0.161 | 11.8 | 50 | 0.099 | 0.470 | 18.9 |
| BMMpH | 0.091 | 9.4 | 44 | 0.058 | 0.430 | 19.9 |
| BMMpM | **0.125** | **15.0** | **32** | 0.049 | 0.404 | 19.7 |

BMMpM's `n traj` column drops to 32 because it is the only cohort with a large silent fraction (19/51, 37%, at its own elbow) — trajectories with zero endpoint breakpoints cannot contribute a Jaccard score against `gsa`, so the pairwise statistic is computed only over the 32 trajectories that have at least one endpoint event. The `iodine` and `gsa`–`iodine` columns are unaffected by G1b, since GSA was already own-elbow throughout.

Three things to take from this:

1. **Endpoint and cage geometry remain only weakly co-timed.** The range is 0.09–0.20 across all six cohorts, superseding the earlier single figure of 0.18.
   **The overlap does not fall monotonically with methylation.** The `pH` series does fall cleanly (0.202, 0.093, 0.091 for 0, 6, 12 equatorial methyls) but the `pM` series does not (0.157, 0.161, 0.125). The honest statement is that the unmethylated cohorts sit highest and the fully methylated sit lowest, without a clean gradient in between.
   **This statistic is penalty-sensitive and denominator-sensitive.** BMHpH's value depends on its own-elbow penalty (0.093 at 4.361, would have been 0.160 on the wrong grid). BMMpM's value is computed over a smaller, non-silent subset of trajectories (32 of 51), not the full cohort — always report the `n traj` alongside the Jaccard for this cohort specifically.
2. **Matched offsets are tight, 9–15 ps**, so the events that *do* correspond are genuinely simultaneous, not loosely associated. BMMpM's 15.0 ps offset is the widest of the six but still well inside a single Pelt tolerance window.
3. **`gsa`–`iodine` at 0.40–0.56 is a legitimate independent pair** — the two column sets are disjoint, unlike `combined` which contains the gsa columns. Cage-geometry and guest events coincide about half the time. **This directly corroborates Murata's stated reason for avoiding Markov modeling**, that "structural deformation and guest encapsulation proceed on comparable time scales." We can now put a number on that claim, which Murata could not.

---

## 4. Positioning: the gap Murata explicitly leaves open

| Murata states | Our pipeline |
|---|---|
| "metastructures are **not claimed to represent thermodynamic metastable states**", they are operational classifications | Segments are found from the data, then clustered; no hand-set criteria |
| "the temporal separation required for Markovian kinetic modeling could not be reliably established; accordingly, hidden Markov models or related kinetic analyses **were not applied**" | Changepoint detection needs no Markov assumption and no lag time. Section 3.5 now *quantifies* the deformation/encapsulation coupling at Jaccard 0.40–0.56 |
| "**other**" reaches **46.1%** for 3₆ and 9.0% for 2₆′ | Clustering assigns every segment; nothing falls outside the scheme |
| "More systematic identification of metastructures... **for example by machine-learning-based clustering approaches, represent an important direction for future studies**" | This is precisely what the pipeline does |

**Thesis:** Murata named unsupervised clustering of nanocube deformation as the needed next step. We built it, applied it to the same trajectories, recovered their structure–property relationship without their criteria, quantified the timescale coupling they could only assert, resolved the conformations their scheme left as "other", and extended the analysis to pole-substituted variants they did not simulate.

---

## 5. Remaining analysis gaps

### G1 `[x]` Matched-penalty re-detection — **done, with two follow-ups**

Executed. Six cohorts re-detected for endpoint and for `gsa` + `iodine`, timing recomputed, summary written to `output/matched_penalty_recompute_summary.csv`. Published tables were left untouched, as intended.

The elbow discrepancy is resolved in favour of the `penalty_recommendation.txt` files; the values in `docs/penalty_sweep_B_cohorts.md` section 3 are stale for BMHpH, BMMpH and BMMpM and **that document still needs correcting**.

**G1a `[x]` Re-run BMHpH and BMMpM on a corrected log(n) reference.** Median-n re-sweep found BMHpH's true elbow at **4.361** (0.631×, up from the wrong-grid 2.801) and BMMpM's true elbow at **5.489** (0.795×, up from the wrong-grid 5.256). Both confirmed with full, from-scratch 51-trajectory detections, not just the sweep search. Clustering was not re-run (G3).

**G1b `[x]` Decided: per-cohort elbow, always.** User's call: the pipeline's stated goal is automatic, minimal-assumption analysis, and a shared relative penalty across cohorts is itself a hand-picked human choice, which defeats that goal. Each of the six cohorts keeps its own elbow with no forced alignment. This reverts the intermediate "matched-relative" test (BMMpM force-aligned to BHHpH's 0.631×, which had produced 1 : 2.29 : 3.39) back to BMMpM's genuine own elbow. Final ratio: **1 : 3.18 : 4.70** — see §3.1 and §3.3. This also closes the "remaining sensitivity" question from the earlier draft: there is no more shared-penalty sensitivity to check, by design decision, not by further computation.

### G2 `[~]` Cohort accounting — **the count mystery is solved**

**Correction:** the earlier claim that BMMpH has 44 and BMMpM has 30 trajectories was wrong. Those were trajectories with **at least one breakpoint**. Verified from `all_segment_stats.csv`:

| Cohort | Trajectories | With zero breakpoints |
|---|---:|---:|
| BHHpH | 50 | 0 |
| BHHpM | 50 | 0 |
| BMHpH | 51 | 0 |
| BMHpM | 50 | 0 |
| BMMpH | 50 | 6 |
| BMMpM | 51 | 21 (41%) at the wrong-grid elbow; **19 (37%) at the true own elbow (G1b, final)** |

**Every cohort is 50 or 51 trajectories**, matching Murata's design exactly. The encapsulation-filtering hypothesis is no longer needed to explain the counts.

**This is itself a result.** At the final own-elbow penalty, 37% of BMMpM replicas show no regime change at all across 5 ns, against 0% for four of the six cohorts. That independently corroborates Murata's 78.4% metastructure A and 0.2% "other" for 1₆. Report it as a per-cohort static fraction, using the 37% figure, not the earlier wrong-grid 41%.

**Guest starting state, verified directly:** checked `n_guest_inside_cavity` / `n_guest_bulk` at frame 0 for every replica in all six cohorts (`output/gsa_features_step1/*_gsa_features.csv`). Result is uniform: **all 24 iodide start in bulk, zero inside the cavity, at t=0, in every cohort with no exceptions.** This matches Murata's stated protocol (extra iodide added specifically to *enhance the chance of* encapsulation during the run) and contradicts a literal reading of "trajectories start from encapsulation states." **Open question O2 below** — resolve before finalizing the G5 comparability claim.

Guest entry *during* the run (`n_guest_inside_cavity > 0` at any frame) does vary sharply by cohort and tracks methylation:

| Cohort | Trajectories with guest entry | Rate |
|---|---:|---:|
| BHHpH | 49/50 | 98% |
| BHHpM | 39/50 | 78% |
| BMHpH | 41/50 | 82% |
| BMHpM | 29/50 | 58% |
| BMMpH | 12/49 | 24% |
| BMMpM | 14/50 | 28% |

This is a looser criterion than Murata's "encapsulation" (any visit, not permanent capture), so it is not directly comparable to their 11/20/27/22-of-50 counts, but the ranking — high-methyl cohorts resist entry, low-methyl cohorts don't — points the same direction as their result.

**`BMHpH_563849_mdcrd_v` resolved — real dissociation event, not an artifact and not encapsulation-driven.** Traced precisely: `assembly_rg` first exceeds 15 Å at **frame 4778 of 5000** (last 4.4% of the trajectory), where `largest_connected_component_size` drops from 6 to 5 — one monomer detaches from the assembly. `n_guest_inside_cavity = 0` at that exact frame, so the break is not coincident with a guest forcing its way through. The guest *did* visit the cavity earlier in the same trajectory (1168/5000 frames, 23%), which had no destabilizing effect until the unrelated dissociation four-fifths of a nanosecond later. Checked `assembly_rg_max > 15 Å` across all ~299 non-`_re_` trajectories in `gsa_features_step1`: **exactly one hit**, this replica. It is genuinely rare, not a systematic issue with the BMHpH cohort or the pole-H design.

**Recommendation:** adopt `assembly_rg_max > 15 Å` (or equivalently, any frame where `largest_connected_component_size < 6`) as the pre-registered QC threshold, applied identically to all six cohorts before clustering. Exclude `BMHpH_563849_mdcrd_v` from the cluster/metastructure comparison under that rule, and report the dissociation itself as a separate one-line finding — a rare, late-trajectory monomer-loss event in the least-stabilized pole-H, low-equatorial-methyl system — rather than a silent drop. Since the threshold catches nothing else across ~299 trajectories, this changes no other result in the plan.

### O2 `[x]` — resolved: apo-optimized cage, no encapsulation-derived geometry

**User confirmed:** our starting structures are a "pure cage form" — an optimized GSA cube built with no guest ever involved — with water and iodide dropped in externally afterward. This is consistent with the frame-0 data above (guest in bulk, zero inside at t=0) and is not a contradiction; the earlier flag was raised because the phrasing was ambiguous, not because the data was wrong.

**This is a methodological point in our favor, worth stating in C1.** Murata's cage geometry traces back to the X-ray structure of the 4₆·(TBM)₂ complex, with the TBM guests removed before their own iodide is added — meaning their starting cavity shape could carry transient memory of a bound aromatic guest. Ours is built as an idealized apo structure with no such history. Comparability to Murata's Table 1 is **not** blocked by any starting-state issue; the guest-entry-rate table above remains useful context (it independently tracks their methylation-dependent encapsulation trend) but is not a required filtering step for G5.

### G3 `[x]` Scan k=2–10 per cohort for chemically meaningful clusters — **implemented, run, and reviewed**

Implemented in `src/ChangepointAnalysis/cluster_k_diagnostics.py` (`scan_cohort_chemical_k`, `assess_chemical_separation`, `scan_chemical_separation_by_k`), wired into `scripts/compare_endpoint_cluster_k.py` via an opt-in `--chemical-scan` flag, covered by `tests/test_cluster_attribution.py` (6/6 passing, including a synthetic replica of the BMHpH outlier-trap scenario), and already run: `output/endpoint_cluster_chemical_k/{chemical_separation_by_k.csv, chemical_k_peaks.csv, chemical_k_summary.txt}`. Verified all numbers below directly against those CSVs.

**Design matches the framing decided earlier:** per cohort, per k in 2–10, report silhouette (geometric) alongside whether the top-ranked site pairs mark ≥2 usable clusters (size ≥ 5) with distinct contacts or mixed open/closed Cohen's *d* signs (`chemical_separation`). No requirement that cohorts agree on a k. `first_chemical_k` is the smallest passing k; `best_chemical_k` is the passing k with the most marked clusters, ties broken toward smaller k.

**Real per-cohort results (published-penalty `by_k/` labels; own-elbow segments not yet re-clustered):**

| Cohort | Silhouette-max k | `chemical_separation` at silhouette-max k | first_chemical_k | best_chemical_k | Chemical at k=5? |
|---|---|---|---|---|---|
| BHHpH | 2 | No | 3 | 5 | Yes |
| BHHpM | 6 | **Yes** | 3 | 8 | Yes |
| BMHpH | 2 (outlier trap) | No — `too_few_usable_clusters` | 5 | 7 | Yes |
| BMHpM | 2 | No | 5 | 7 | Yes |
| BMMpH | 2 | No | 3 | 3 | Yes (window k=3–5 only) |
| BMMpM | 10 (rising) | No | 4 | 6 | Yes (also 6, 9) |

Three things worth stating plainly:

1. **k=5 passes `chemical_separation` in all six cohorts.** The fixed default is not just a comparability convenience; it is independently chemically meaningful everywhere it's used.
2. **Silhouette-max k=2 never passes `chemical_separation`.** That is four of six cubes (BHHpH, BMHpH, BMHpM, BMMpH). BHHpM is the exception: its silhouette peak is k=6, and that cut *does* pass chemically. BMMpM's rising curve peaks at k=10, which fails. Geometric compactness and chemical distinctness are different axes — that is the point of the scan.
3. **BMHpH's notorious k=2/k=3 silhouette spike (0.917 / 0.877, previously explained as an "outlier trap" from the exploded-cage replica) is now automatically excluded** via the `min_cluster_size=5` filter (`too_few_usable_clusters`), rather than needing a manual caveat every time it's cited.

**Correction to the previous draft of this section:** it speculated BMMpM might show "no chemical separation at any k" as a "clean negative case." **That speculation is wrong, checked against the real scan.** BMMpM passes `chemical_separation` at k=4, 5, 6, and 9 (4 of 9 tested), while its silhouette rises essentially monotonically across the whole range (0.298 → 0.412) with no correlation to which k passes chemically. That is a *better* illustration of the core point than a flat "never separates" result would have been: the geometric-compactness score and the chemical-separation flag are tracking genuinely different things, visibly disagreeing at most steps rather than one simply being empty. Do not describe BMMpM as chemically featureless; describe it as chemically intermittent while silhouette climbs steadily. `docs/endpoint_cluster_discrimination.md` §13 already has this corrected version.

**One interpretive caveat worth carrying into the manuscript:** `n_significant_fdr` runs 500–950 out of ~960 candidate columns at almost every k with ≥2 usable clusters. That is expected given the known redundancy among raw site-pair columns (~26 PCs explain 80% of their variance, per `docs/endpoint_cluster_discrimination.md` §4) and should not be read as 900 independent discoveries — report it as "most of a highly correlated feature set clears the FDR bar," not as a large number of separate findings.

**Still open:** the scan used published-penalty `by_k/` labels, not the final own-elbow detections from G1/G1a/G1b. If the paper re-clusters on the final own-elbow segments (see the note below), re-run `scan_chemical_separation_by_k` on those — do not port these numbers over.

**Note:** clustering has *not* been re-run on any of the G1/G1a/G1b detections. All existing cluster artifacts (including the `by_k/` sweep above) belong to the published penalties. Decide whether the paper clusters the final own-elbow segments (`output/endpoint_changepoints_matched_*` for four cohorts, `output/endpoint_changepoints_g1a_BMHpH`, `output/endpoint_changepoints_g1b_BMMpM` — preferable, for consistency with section 3) or keeps clustering on the published tables and reports detection-only results at the own-elbow penalties.

### G4 `[x]` Fix the attribution statistics framing — **implemented, run, and reviewed**

η² still cannot *validate* the clusters — that circularity (clusters built from per-monomer-pair aggregates whose components η² then ranks) is unchanged and correctly still documented. What's now added is a calibrated ranking: a label-shuffle permutation null (default 999 shuffles, cluster sizes preserved) and Benjamini–Hochberg FDR across the ~960 candidate columns.

**Implementation, verified correct:**
- `benjamini_hochberg_qvalues` — standard step-up BH procedure (sort ascending, running minimum from the top rank down, clipped to [0,1]). Matches the textbook definition; unit-tested for monotonicity and NaN handling.
- `permutation_eta_squared_pvalues` — vectorized column-wise η² (`_eta_squared_columns`, checked against the original scalar `_eta_squared` in tests and numerically identical), one label shuffle reused across all candidate columns per permutation (efficient, standard practice), add-one p-value formula `(1 + count) / (n_perm + 1)`. Unit test plants a known signal column and confirms it gets the minimum possible p-value and the smallest q-value among a null background.
- New columns on `endpoint_pair_cluster_correlation.csv`: `eta_squared_p_value`, `eta_squared_q_value`, `n_permutations`, `significant_fdr`.

Ran the full existing test suite after this refactor (`rank_cluster_discriminating_endpoint_features` was split into `collect_endpoint_segment_means` + `score_cluster_discriminating_features` to support G3's per-k reuse) — all 36 pre-existing tests plus the 6 new G3/G4 tests pass, no regressions.

**Not yet done:** this has not been re-run on the published `endpoint_pair_cluster_correlation.csv` files — checked `output/endpoint_changepoints_B*/endpoint_pair_cluster_correlation.csv` directly and none carry the new p-value columns yet. The function signature defaults to `n_permutations=999`, so a plain re-run of `summarize_changepoint_results` per cohort will populate them.

**Code:** `src/ChangepointAnalysis/reporting.py` — `_eta_squared_columns`, `benjamini_hochberg_qvalues`, `permutation_eta_squared_pvalues`, `score_cluster_discriminating_features`, `rank_cluster_discriminating_endpoint_features` (now a thin wrapper). Tests: `tests/test_cluster_attribution.py`.

### G5 `[ ]` The Murata comparison

Depends on G0. G3 and G4 are done and do not block this. Three concrete comparisons:

**1. Population comparison** against Murata Table 1 (percentages, non-encapsulated trajectories). **Cohort mapping resolved (O1):** BMHpM = 2₆, confirmed. **2₆′ has no cohort in our design** — three-way comparison only, not four:

| | 1₆ / BMMpM | **2₆ / BMHpM** | 2₆′ — no cohort | 3₆ / BHHpM |
|---|---:|---:|---:|---|
| A | 78.4 | 45.3 | 19.7 | 7.2 |
| B | 8.7 | 13.8 | 13.3 | 15.8 |
| C1 | 2.3 | 1.7 | 19.1 | 4.7 |
| C2 | 10.4 | 38.6 | 38.9 | 26.3 |
| other | 0.2 | 0.5 | 9.0 | 46.1 |

With G0 done, label every frame by Murata's rules and cross-tabulate against our cluster labels for the three mapped cohorts. The 2₆′ column is retained above only for reference to Murata's own divergence result (§0), not as a target for our G5 comparison.

**2. Transition-frequency comparison.** Section 3.3's final own-elbow ratio (1 : 3.18 : 4.70 vs Murata 1 : 2.67 : 3.67) is the number to quote. No longer pending — G1b is decided.

**3. Resolving "other".** For the frames Murata calls "other" in our three mapped cohorts — 0.5% of 2₆, 46.1% of 3₆ — report what our clusters make of them. The 9.0% figure for 2₆′ is not something we can resolve, since we have no 2₆′ trajectories.

**Code:** new `scripts/compare_to_murata_metastructures.py`

---

## 6. Content gaps

### C1 `[~]` System and Simulations — **largely solved by Murata section 2**

| Parameter | Value |
|---|---|
| Package | Amber20 |
| Charges | RESP at HF/6-31G(d), Gaussian 16 Rev. A.03 |
| Force field | GAFF |
| Water | TIP4P, 17 initially inside the cavity |
| Ions | `ionsjc_tip4pew` for iodide and sodium; iodide as both counterion and guest |
| Constraints | SHAKE on X–H bonds |
| Thermostat | Langevin, 300 K |
| Electrostatics | PME under PBC |
| Equilibration | NPT 20 ps (10,000 x 2 fs), final density 1.00 g/cm³ |
| Production | **NVT, 5 ns, 2 fs timestep, 2,500,000 steps, 50 runs per system** |
| Initial coords (Murata) | from the X-ray structure of 4₆·(TBM)₂, TBM guests removed |
| Initial coords (ours) | **differs — confirmed with the user.** An optimized, apo GSA cube with no guest ever involved in building it; water and iodide dropped in afterward. Not derived from a guest-bound crystal structure. |

**1 frame = 1 ps**, confirmed: Murata classified every 1 ps and our GSA CSVs hold 5000 rows for a 5 ns run. Endpoint at stride 5 is 5 ps per row. The matched timing outputs already report offsets in ps.

**Worth a sentence in Methods or Limitations:** our starting cage carries no shape memory of a bound aromatic guest, unlike Murata's TBM-derived geometry. This is a point of difference, not a problem — if anything it removes a potential confound from their protocol — but it should be stated rather than silently assumed identical. Verified from the frame-0 guest data in G2: all 24 iodide start in bulk with zero inside the cavity, in every cohort, matching this description exactly.

**Still needed from your own records:** whether our trajectories are otherwise Murata's exact production protocol or an independent re-run, and the provenance of the three `pH` cohorts.

### C2 `[ ]` Figure 1 schematic

Pipeline overview: trajectory to two feature families to segmentation to clustering to attribution, with a real segmented time series as the spine.

### C3 `[ ]` Limitations paragraph

No free-energy surface, no rate constants, attribution is correlational, cluster labels are arbitrary integers, cluster count imposed for comparability. Add: our segments are not Markovian states either, for the reason Murata gives and section 3.5 now quantifies.

---

## 7. Release hygiene

- **R1 `[ ]`** Clean the working tree. Note `output/_recompute_matched_penalties.py` is a new untracked script — move it into `scripts/` and commit it, since it produced published numbers.
- **R2 `[ ]`** Tag a release and mint a Zenodo DOI, after G0 lands and the re-cluster decision (published penalties vs final own-elbow segments — G3 note) is made; if re-clustering happens, re-run the G3 chemical-k scan and G4 ranking on the new segments first.
- **R3 `[ ]`** Data availability. Feature CSVs are small enough to deposit in full.
- **R4 `[~]`** Tests for the G0 criteria and the G4 permutation null. G4's tests are done (`tests/test_cluster_attribution.py`, 6/6 passing). G0's cation–π / d2 tests are still needed once that code exists.

---

## 8. Dependency order

```
G0 ──► G5
G1 ─► G1a ─► G1b ✓ done ─► §3.3 final ─► G5
G3 ✓ done, G4 ✓ done ──► Results (attribution)
C1 ──► Methods (done)
G1b ──► R2 ──► R3
```

**Critical path:** G0 (reimplement Murata's criteria) → G5 → abstract resolved. G1b, O1, G3, and G4 are all closed; G0 is the only thing left upstream of G5. The G3 re-cluster decision (published penalties vs final own-elbow segments) is a quality choice for R2, not a blocker for G5 or the draft.

---

## 9. Sequencing

| Stage | Work | Gate |
|---|---|---|
| 1 | ~~Correct `docs/penalty_sweep_B_cohorts.md`~~ done | Docs match code |
| 2 | G0 implementation | d1 bimodal, RMSD peaks reproduce |
| 3 | G5 three-way comparison (G3, G4 done and do not block this) | Contingency table, "other" resolved |
| 4 | Re-cluster decision for R2 (published penalties vs final own-elbow segments); re-run G3/G4 on new segments if so | Quality choice, not a blocker |
| 5 | C2, C3, draft | Full draft |
| 6 | R1–R4, submit | Release tagged |

---

## 10. Numbers to re-verify before quoting

| Quantity | Status | Note |
|---|---|---|
| Endpoint elbows | **resolved, final** | Each cube its own elbow (G1b decision). `docs/penalty_sweep_B_cohorts.md` corrected to match |
| Cohort sizes | **resolved** | All 50–51. Earlier 44/30 were trajectories with ≥1 breakpoint |
| endpoint–gsa Jaccard | **0.09–0.20** | Final, own-elbow throughout. Supersedes the single 0.18. BMMpM's value (0.125) is computed over only 32/51 trajectories — always report `n traj` alongside it. Falls with methylation in `pH` only, not `pM` |
| combined–gsa Jaccard 0.63 | **do not quote** | `combined` contains the gsa columns. Use `gsa`–`iodine` (0.40–0.56) as the independent within-GSA pair |
| Transition-rate ratio | **1 : 3.18 : 4.70 — final** | Own elbow throughout (G1b decided). vs Murata 1 : 2.67 : 3.67 — this overshoots, by design: BMMpM's genuine elbow sits at a higher relative penalty (0.795×) than the rest (0.50–0.63×). Do not quote the intermediate matched-relative figure of 1 : 2.29 : 3.39; that run is rejected |
| GSA-vs-endpoint event ratio | 4.9–13.2x | Not "2–3x". Both cover 5 ns, so comparable, but endpoint cannot resolve sub-5 ps events |
| "0% open cation–π in BMM" | **withdrawn** | Mis-specified criterion; see section 2 |
| Remapped equatorial d1 (BHH*) | **4.97–4.99 Å median** | `output/murata_d1/`. 75–77% in 4.5–5.5 Å, 10–14% ≥ 7 Å. Do **not** quote BMH/BMM compact fractions as C2–C3 (methyl-carbon proxy) |
| Hull `s3`/`s7` d1 ~9–17 Å | **superseded** | Wrong sites; use remapped R2–R3 edges |
| Top site-pair η² 0.81–0.84 | partly circular, now calibrated | Permutation null + BH-FDR implemented and tested (G4). Not yet re-run on the published `endpoint_pair_cluster_correlation.csv` files — none carry the new p/q columns as of this check |
| BMMpM 37% static at true own elbow | **holds, final** | 19/51 have zero breakpoints at 5.489. The earlier 41% (21/51) used the wrong-grid elbow (5.256) and is superseded. Corroborates Murata's 78.4% A for 1₆ |
| k=5 is chemically meaningful | **new, verified** | `chemical_separation`=True at k=5 in all six cohorts (G3, real scan output). Silhouette-max k=2 never passes; BHHpM's k=6 peak does |
| "BMMpM shows no chemical separation at any k" | **withdrawn** | Was speculation, not yet checked. Real scan: passes at k=4,5,6,9 (4/9); silhouette rises monotonically with no correlation to the chemical flag. See G3 |

---

*Repository state: branch `MDchat`. G1 from `output/matched_penalty_recompute_summary.csv`. G1a/G1b from `output/g1a_relative_penalty_correction.csv`, `output/endpoint_changepoints_g1a_BMHpH` (own elbow, 4.361, full 51-traj run), and `output/endpoint_changepoints_g1b_BMMpM` + `output/endpoint_vs_gsa_timing_g1b_BMMpM` (own elbow, 5.48888, full 51-traj run, generated in this session to replace an unverified sweep-only estimate).*
