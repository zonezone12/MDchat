# Publication Gap-Closure Plan

**Target:** methods-and-application manuscript on the automated changepoint pipeline applied to GSA nanocubes.
**Companion:** [Manuscript planning brief (artifact)](https://claude.ai/code/artifact/a7594487-ffa7-4190-8ed5-5c7eb65257a0)
**Revised:** G0 is implemented and G5 Table 1 occupancy is computed. Motif-level π(pole)/π(eq)/d1/d2 histograms reproduce Murata Fig. S11 (`output/murata_motif_rmsd/plots/`). Apo-frame A/B/C1/C2/other % for BMMpM and BMHpM match Table 1 within 0.6 pp in every class; BHHpM (3₆) matches A/B/C1 but splits C2 vs other differently (`output/murata_g5/`; reading drafted in G5 item 3). G1b still stands: each cube keeps its own elbow; §3.3 ratio **1 : 3.11 : 4.60** per 5 ns. The paper clusters the own-elbow segments. G3's automatic k on them is 7, 4, 4, 12, 3, 4 for BHHpH, BHHpM, BMHpH, BMHpM, BMMpH, BMMpM (`output/endpoint_cluster_chemical_k_own_elbow/chemical_k_selected.csv`). The rule: pass the chemical gate, then take the smallest k within one replica-bootstrap SE of the best Murata-label NMI over k=2–15; silhouette is not used. G4 p/q columns are populated on all six own-elbow folders, and the site pairs are re-ranked at those k (`output/endpoint_pair_ranking_selected_k/`).

**Update (Oct 13, afternoon):**
- The G3 k scan was extended to k=15 (re-clustered; the stored k=5 labels came out identical). Max NMI did not settle: BMHpM plateaus at about 0.43 from k=12, and the BHHpH and BHHpM maxima moved to k=15 and k=14 for gains of 0.013 and 0.003, smaller than their replica-bootstrap SEs (0.024 and 0.018). k is now picked by the one-SE rule (user's call; G3).
- Site pairs are re-ranked at the selected k (G4). BMHpH's top pairs mostly mark the single loosened replica `157923`.
- The three belt-broken BHHpM clusters are one deformation in three symmetry-related orientations. Each stretches the cage along a different pair of opposite monomers (G5 item 3).
- R1 closed: `scripts/recompute_own_elbow_g1.py` reproduces the g1a/g1b detections and timing from scratch.

**Update (Oct 13, morning):**
- Cohort definition decided: `_re_` re-runs stay alongside their originals, the truncated runs stay in, and rates are per simulated time (G2, §3.3).
- Own-elbow re-cluster done for all six cubes with the QC rule, which drops only `BMHpH_563849_mdcrd_v`. The summarize step populated G4, and the G3 scan and k selection were re-run. On these segments k=5 fails the chemical gate in BHHpM, BMHpH and BMMpM (G3).
- G5 corrected: old RMSD-only motif CSVs in `output/murata_motif_rmsd/` were being loaded on top of the full ones, and their frames counted as "other". With that fixed, 1₆ and 2₆ "other" is 0.2% and 0.5%, as in Table 1 (G5 item 1).
- The 3₆ C2/other paragraph is drafted (G5 item 3).
- Earlier the same day: the G2 QC rule was implemented, `paper_d1_n_open` became `paper_d1_n_compact`, `docs/references.md` was added, the recompute script moved to `scripts/`, and two §3.5 rows were corrected (BMHpH carried values from the rejected wrong-grid run; BMMpM had been timed against the published-penalty GSA detections).

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
| Relation to us | Reproduced by `src/utils/imamura_msm.py`; a **methodological** precedent for clustering trajectories | **Our comparison reference.** The systems, force field, and states we compare against. Whether our trajectories are Murata's production runs or an independent re-run is open (C1); our starting cage differs from theirs (O2) |

**Consequence:** Imamura is background and method precedent, cited in the Introduction. Murata is the paper our results are compared against. Do not describe A/B/C1/C2 as coming from Imamura.

Both are recorded in `docs/references.md`, together with Murata 2023 (their ref. 19). **D1 is closed.**

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

## 2. Murata's criteria — implemented (G0)

**Closed.** Classification now uses locked intermolecular contacts on the motif CSVs, not hull `s3`/`s7` and not whole-cube RMSD. The visual gate is Murata Fig. S11: d1 stays compact (~4–5 Å) when all pole π are closed and grows an elongated peak (~8–9 Å) as π opens (`output/murata_motif_rmsd/plots/dist_d1_d2_by_open_cation_pi.png`).

### What Murata defines, and how we compute it

| Quantity | Definition | Thresholds | Our implementation |
|---|---|---|---|
| cation–π opening | **π(pole):** pole Py⁺ COM to Ph COM. The sandwich also has **π(equator)** (third monomer's eq Py⁺ → same Ph); opening is pole-side | **open when π(pole) ≥ 6.5 Å** | Six sandwiches locked at frame 0 (Hungarian). `n_open_cation_pi` counts π(pole) ≥ 6.5 Å |
| `d1` | **C2–C3**: R2 and R3 **ipso** carbons across an equatorial edge. Six sites | compact **4.5–5.5 Å**; elongated **≥ 7.0 Å** | Six R2(*i*)–R3(*j*) edges locked at frame 0. Ipso carbon, not the methyl carbon |
| `d2` | **CPy–C3** on the same edge; no threshold | methyl/vdW analysis | CPy = eq Py⁺ carbon **para to the phenylene linker** (Py⁺ C3 in these GSAs; that places CPy at C2, ortho to N). Not para-to-N (C4) |

Metastructures (unchanged):

| | opened cation–π | d1 condition | RMSD (Murata, local motif) |
|---|---|---|---|
| **A** | 0 | — | ≈ 1.0 Å |
| **B** | 1 | — | ≈ 1.5 Å |
| **C1** | 2 | no d1 ≥ 7.0 Å | ≈ 1.6–2.5 Å |
| **C2** | 2 | one d1 ≥ 7.0 Å | ≈ 2.7 Å |
| **other** | anything else, including ≥ 3 opened | — | — |

Murata RMSD is **local heavy atoms of the opening motif**, not Kabsch of the whole cube. We store pooled `rmsd_cation_pi` / `rmsd_equator` and per-unit `rmsd_*_e{k}` plus the distances above.

### What was wrong (withdrawn)

1. Cation–π was missing, then later CPy was taken as para-to-N. **Fixed:** CPy is para to the Py⁺–benzene linker.
2. `paper_d1_n_open` counted the compact 4.5–5.5 Å window as "open." Labels now use elongated d1 ≥ 7.0 Å. The column is now `paper_d1_n_compact` (item 7 below).
3. Hull `s3`/`s7` was not R2/R3. Canonical roles: `src/EndpointAnalyzer/gsa_site_map.py`.
4. Whole-cube `assembly_rmsd_to_ref` is unimodal and does not show Murata's 1.0 / 1.5 / 2.7 Å peaks. Motif RMSD is the right coordinate.

The earlier claim that **BMMpH and BMMpM show 0% open cation–π** is an artifact of a mis-specified criterion. Withdrawn.

### G0 `[x]` Implement Murata's criteria properly

1. `[x]` Pole Py⁺, equatorial Py⁺, Ph per monomer; six Ph sandwiches (pole *i*, Ph *j*, eq Py⁺ *k*).
2. `[x]` π(pole) and π(equator) per sandwich; open = π(pole) ≥ 6.5 Å; `n_open_cation_pi`.
3. `[x]` d1 = C2–C3 ipso on six locked equatorial edges. **Sanity gate passed:** overlay histograms are compact near 4–5 Å with an elongated shoulder ≥ 7 Å that grows when π opens (S11-style split).
4. `[x]` d2 = CPy–C3 (CPy para to the phenylene linker).
5. `[x]` Per-frame A / B / C1 / C2 / other (`label_motif_metastructures`).
6. `[x]` Motif RMSD vs frame 0. Whole-cube RMSD is kept only as a negative control.
7. `[x]` Renamed `paper_d1_n_open` to `paper_d1_n_compact`, keeping the `paper_d1_` prefix of its sibling columns. Older CSVs on disk are read through `rename_legacy_paper_d1_columns`. The per-pair `open_pairs` / `n_open_pairs` columns and `classify_paper_d1_state` still call the 4.5–5.5 Å window "open". Does not affect G5 labels.

**Code:** `src/EndpointAnalyzer/gsa_site_map.py`, `src/ChangepointAnalysis/murata_criteria.py`, `src/ChangepointAnalysis/murata_rmsd.py`. Batch: `scripts/compute_murata_motif_rmsd.py`. Plots: `scripts/plot_murata_motif_rmsd_timeseries.py`. Frames: `output/B*_motif_rmsd/`. Combined plots: `output/murata_motif_rmsd/plots/` (`dist_pole_eq_d1_d2.png`, `dist_d1_d2_by_open_cation_pi.png`, plus `_apo` and per-metric overlays).

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

**BMMpM: same grid bug, fixed, and its true elbow does not land near the others.** Re-swept on the correct grid, BMMpM's genuine elbow is **5.489 (0.795×)** — even higher than the wrong-grid value of 5.256 (0.761×). The grid bug was not what put BMMpM at a high relative penalty; its count-vs-penalty curve simply kinks later than the other five cohorts. **Confirmed with a full, from-scratch 51-trajectory detection** at `output/endpoint_changepoints_g1b_BMMpM` (penalty 5.48888): **106 breakpoints, rate 2.078/traj, 19/51 trajectories silent (37%)** — exactly matching the earlier sweep-only estimate, so no residual denominator issue. Timing recomputed at `output/endpoint_vs_gsa_timing_g1b_BMMpM_gsa_elbow` (§3.5 explains why the earlier `output/endpoint_vs_gsa_timing_g1b_BMMpM` is superseded).

**Root cause of the grid bug, now confirmed:** in both BMHpH and BMMpM the alphabetically first endpoint CSV is the truncated `109345_mdcrd_v` (266 rows in BMHpH, 746 in BMMpM, against a median of 1000; see G2). The wrong-grid elbows match 0.502 × log(266) = 2.80 and 0.795 × log(746) = 5.26.

Outputs: `output/endpoint_changepoints_g1a_BMHpH/`, `output/endpoint_changepoints_g1b_BMMpM/`, `output/endpoint_vs_gsa_timing_g1a_BMHpH/`, `output/endpoint_vs_gsa_timing_g1b_BMMpM_gsa_elbow/`, `output/g1a_relative_penalty_correction.csv` (sweep-comparison record only — its own-elbow row for BMMpM is now independently confirmed by the full run above), `output/endpoint_changepoints_{CUBE}/penalty_sweep_median_n/`.

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

| Equatorial CH3 | Murata | Murata A↔B per ns | Our cohort | Own-elbow events per 5 ns |
|---:|---|---:|---|---:|
| 12 | 1₆ | 3 | BMMpM | 2.125 |
| 6 | 2₆ | 8 | BMHpM | 6.60 |
| 0 | 3₆ | 11 | BHHpM | 9.76 |
| ratio vs 1₆ | | **1 : 2.67 : 3.67** | | **1 : 3.11 : 4.60** |

Rates are per simulated time (decided Oct 13; see the truncation note below). Dividing by trajectory count instead gives 2.078 for BMMpM and a ratio of 1 : 3.18 : 4.70.

**This overshoots Murata's ratio, and that is the honest, automatic-pipeline answer.** BMMpM's own elbow (5.489, confirmed by a full 51-trajectory detection) sits at a higher relative penalty than the other five cohorts — 0.795× log(n) against 0.50–0.63× elsewhere — because its breakpoint-count-vs-penalty curve genuinely kinks later, not because of any residual grid bug. Forcing it onto a shared relative penalty (the earlier, now-rejected matched run) narrowed the gap to Murata's ratio, but only by imposing a human-chosen alignment the pipeline itself has no principled way to select. The wider gap here is the price of "detect the elbow and use it," with no manual steering per cohort.

The `pH` series, own elbow throughout, per 5 ns: **1 : 1.49 : 3.22** (BMMpH 3.12, BMHpH 4.64, BHHpH 10.06) — unaffected by this decision, since BMHpH's own elbow was never in dispute. The equatorial-methyl gradient is weaker at the pole-H end (3.22× top, 1.49× middle) than at the pole-CH3 end (4.60× top, 3.11× middle). That interaction between pole and equatorial substitution is itself a result, not a confound, and it is not something Murata's design (equatorial substitution only) could have shown.

**Truncation: rates are per simulated time (decided Oct 13).** Four trajectories are shorter than 5 ns (G2): one in BMHpH and three in BMMpM. They stay in the cohorts, and rates divide by simulated time instead of by trajectory count. That moves BMMpM from 2.078 to **2.125** events per 5 ns and BMHpH from 4.57 to **4.64**; the other four cubes have only full-length runs and do not change. The `pM` ratio is **1 : 3.11 : 4.60** (1 : 3.18 : 4.70 per trajectory) and the `pH` ratio 1 : 1.49 : 3.22 (1 : 1.46 : 3.22).

### 3.4 Two controls that strengthen the result

**Cage geometry shows the same trend but much weaker.** GSA cage events per 5 ns across the `pM` series give **1 : 1.45 : 2.04** (per trajectory: BHHpM 60.8, BMHpM 43.1, BMMpM 29.2, i.e. 1 : 1.48 : 2.08). Against Murata's **3.67x**, the own-elbow endpoint coordinate gives **4.60x** (§3.3) while whole-cage geometry gives **2.04x**.

**Endpoint still tracks Murata's methyl dependence more closely than cage geometry, on a relative-error basis, though both now sit on the same automatic footing.** Relative to Murata's 3.67x: endpoint overshoots by 25% (4.60 vs 3.67), cage geometry undershoots by 44% (2.04 vs 3.67). Per trajectory these are 28% and 43%. Endpoint is still the closer match, but the comparison is now "smaller error on the same side of no forced alignment" rather than the near-exact 8% agreement reported under the (now-rejected) matched-penalty run. That earlier close match depended on picking BMMpM's penalty to align with BHHpM's; it is not a property of the automatic pipeline itself.

GSA penalties were untouched by G1b (they were already independently own-elbow for each cohort from the start of G1), so the cage ratio (2.04x per 5 ns) needs no further correction.

**The guest is a genuine negative control.** Iodine-group events per trajectory are essentially flat across all six cohorts: 47.2, 58.7, 45.1, 48.8, 49.5, 48.8. The guest does not care how many equatorial methyls the cage has. **The methyl effect is specific to cage deformation, not a global sampling artifact.** This is the cleanest control available and it should be a figure panel.

### 3.5 Cross-family timing, own elbow throughout (final)

Median per-trajectory Jaccard, ±50 frame tolerance, offsets in ps. Every row is now the own-elbow detection, per G1b:

| Cube | endpoint–gsa | offset (ps) | n traj | endpoint–iodine | gsa–iodine | offset (ps) |
|---|---:|---:|---:|---:|---:|---:|
| BHHpH | 0.202 | 12.9 | 50 | 0.168 | 0.516 | 19.6 |
| BHHpM | 0.157 | 13.3 | 50 | 0.140 | 0.557 | 17.0 |
| BMHpH | 0.093 | 11.7 | **50** | **0.068** | 0.471 | 19.0 |
| BMHpM | 0.161 | 11.8 | 50 | 0.099 | 0.470 | 18.9 |
| BMMpH | 0.091 | 9.4 | 44 | 0.058 | 0.430 | 19.9 |
| BMMpM | **0.087** | **13.3** | **32** | **0.047** | 0.404 | 19.7 |

BMMpM's `n traj` column drops to 32 because it is the only cohort with a large silent fraction (19/51, 37%, at its own elbow) — trajectories with zero endpoint breakpoints cannot contribute a Jaccard score against `gsa`, so the pairwise statistic is computed only over the 32 trajectories that have at least one endpoint event (BMHpH has one silent trajectory at its own elbow, hence 50). The `gsa`–`iodine` column does not depend on the endpoint penalty; the endpoint–iodine column does.

**Corrected in this revision (Oct 13).** The BMHpH row had carried the wrong-grid run's `n traj` (51) and endpoint–iodine (0.125); its own-elbow values are 50 and 0.068 (`output/endpoint_vs_gsa_timing_g1a_BMHpH`). The BMMpM row had come from `output/endpoint_vs_gsa_timing_g1b_BMMpM`, which compared the own-elbow endpoint run against the *published* log(n) GSA detections (giving 0.125 at 15.0 ps). Against the GSA own-elbow detections used for the other five cubes it is 0.087 at 13.3 ps (`output/endpoint_vs_gsa_timing_g1b_BMMpM_gsa_elbow`).

Three things to take from this:

1. **Endpoint and cage geometry remain only weakly co-timed.** The range is 0.09–0.20 across all six cohorts, superseding the earlier single figure of 0.18.
   **The overlap does not fall monotonically with methylation.** The `pH` series falls (0.202, 0.093, 0.091 for 0, 6, 12 equatorial methyls); the `pM` series is flat from 0 to 6 and then drops (0.157, 0.161, 0.087). The honest statement is that the fully methylated cohorts sit lowest in both series, without a clean gradient in between.
   **This statistic is penalty-sensitive and denominator-sensitive.** BMHpH's value depends on its own-elbow penalty (0.093 at 4.361, would have been 0.160 on the wrong grid). BMMpM's value is computed over a smaller, non-silent subset of trajectories (32 of 51), not the full cohort — always report the `n traj` alongside the Jaccard for this cohort specifically.
2. **Matched offsets are tight, 9–13 ps**, so the events that *do* correspond are genuinely simultaneous, not loosely associated. The widest (13.3 ps, BHHpM and BMMpM) is still well inside a single Pelt tolerance window.
3. **`gsa`–`iodine` at 0.40–0.56 is a legitimate independent pair** — the two column sets are disjoint, unlike `combined` which contains the gsa columns. Cage-geometry and guest events coincide about half the time. **This directly corroborates Murata's stated reason for avoiding Markov modeling**, that "structural deformation and guest encapsulation proceed on comparable time scales." We can now put a number on that claim, which Murata could not.

---

## 4. Positioning: the gap Murata explicitly leaves open

| Murata states | Our pipeline |
|---|---|
| "metastructures are **not claimed to represent thermodynamic metastable states**", they are operational classifications | Segments are found from the data, then clustered; no hand-set criteria |
| "the temporal separation required for Markovian kinetic modeling could not be reliably established; accordingly, hidden Markov models or related kinetic analyses **were not applied**" | Changepoint detection needs no Markov assumption and no lag time. Section 3.5 now *quantifies* the deformation/encapsulation coupling at Jaccard 0.40–0.56 |
| "**other**" reaches **46.1%** for 3₆ and 9.0% for 2₆′ | Clustering assigns every segment; nothing falls outside the scheme |
| "More systematic identification of metastructures... **for example by machine-learning-based clustering approaches, represent an important direction for future studies**" | This is precisely what the pipeline does |

**Thesis:** Murata named unsupervised clustering of nanocube deformation as the needed next step. We built it, applied it to the same nanocube systems, recovered their structure–property relationship without their criteria, quantified the timescale coupling they could only assert, resolved the conformations their scheme left as "other", and extended the analysis to pole-substituted variants they did not simulate.

---

## 5. Remaining analysis gaps

### G1 `[x]` Matched-penalty re-detection — **done, with two follow-ups**

Executed. Six cohorts re-detected for endpoint and for `gsa` + `iodine`, timing recomputed, summary written to `output/matched_penalty_recompute_summary.csv`. Published tables were left untouched, as intended.

The elbow discrepancy is resolved in favour of the `penalty_recommendation.txt` files, and `docs/penalty_sweep_B_cohorts.md` has since been corrected to match (§9, stage 1).

**G1a `[x]` Re-run BMHpH and BMMpM on a corrected log(n) reference.** Median-n re-sweep found BMHpH's true elbow at **4.361** (0.631×, up from the wrong-grid 2.801) and BMMpM's true elbow at **5.489** (0.795×, up from the wrong-grid 5.256). Both confirmed with full, from-scratch 51-trajectory detections, not just the sweep search. Clustering was not re-run (G3).

**G1b `[x]` Decided: per-cohort elbow, always.** User's call: the pipeline's stated goal is automatic, minimal-assumption analysis, and a shared relative penalty across cohorts is itself a hand-picked human choice, which defeats that goal. Each of the six cohorts keeps its own elbow with no forced alignment. This reverts the intermediate "matched-relative" test (BMMpM force-aligned to BHHpH's 0.631×, which had produced 1 : 2.29 : 3.39) back to BMMpM's genuine own elbow. Final ratio: **1 : 3.11 : 4.60** per 5 ns (1 : 3.18 : 4.70 per trajectory) — see §3.1 and §3.3. This also closes the "remaining sensitivity" question from the earlier draft: there is no more shared-penalty sensitivity to check, by design decision, not by further computation.

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

**Every cohort is 50 or 51 trajectories, and every cohort has exactly 50 unique seeds**, matching Murata's design. The encapsulation-filtering hypothesis is no longer needed to explain the counts. The 51s count one seed twice: BMHpH and BMMpM each hold an original plus an `_re_` re-run (`563849` and `257803`). BMMpH has the re-run `203945_re` but no original, so it is 49 originals plus one re-run. All three re-runs are full length and pass the QC rule below.

**Truncated trajectories (new, Oct 13).** Four runs are shorter than 5000 frames: BMHpH `109345` (1326), BMMpM `109345` (3728), BMMpM `257803` (4469; it has a re-run) and BMMpM `587901` (1460). The other 298 are full length. These caused the G1a grid bug (§3.1) and slightly bias per-trajectory rates (§3.3 truncation note).

**This is itself a result.** At the final own-elbow penalty, 37% of BMMpM replicas show no regime change at all across 5 ns, against 0% for three of the six cohorts (BMHpH 1/51, BMMpH 6/50). That independently corroborates Murata's 78.4% metastructure A and 0.2% "other" for 1₆. Report it as a per-cohort static fraction, using the 37% figure, not the earlier wrong-grid 41%. Two of the 19 silent BMMpM runs are truncated (`109345`, `587901`); leaving out the three truncated BMMpM originals gives 17/48 = 35%, so the conclusion holds.

**Guest starting state, verified directly:** checked `n_guest_inside_cavity` / `n_guest_bulk` at frame 0 for every replica in all six cohorts (`output/gsa_features_step1/*_gsa_features.csv`). Result is uniform: **all 24 iodide start in bulk, zero inside the cavity, at t=0, in every cohort with no exceptions.** This matches Murata's stated protocol (extra iodide added specifically to *enhance the chance of* encapsulation during the run) and contradicts a literal reading of "trajectories start from encapsulation states." **O2 is closed** (apo-optimized cage).

Guest entry *during* the run (`n_guest_inside_cavity > 0` at any frame) does vary sharply by cohort and tracks methylation:

| Cohort | Trajectories with guest entry | Rate |
|---|---:|---:|
| BHHpH | 49/50 | 98% |
| BHHpM | 39/50 | 78% |
| BMHpH | 41/50 | 82% |
| BMHpM | 29/50 | 58% |
| BMMpH | 12/49 | 24% |
| BMMpM | 14/50 | 28% |

Re-verified Oct 13 from `output/gsa_features_by_cube/`. The table counts originals only: the three `_re_` re-runs are left out, which is why BMHpH, BMMpH and BMMpM show 50, 49 and 50. With re-runs included: BMHpH 42/51, BMMpH 12/50, BMMpM 14/51. All 302 trajectories have zero guests inside the cavity at frame 0.

This is a looser criterion than Murata's "encapsulation" (any visit, not permanent capture), so it is not directly comparable to their 11/20/27/22-of-50 counts, but the ranking — high-methyl cohorts resist entry, low-methyl cohorts don't — points the same direction as their result.

**`BMHpH_563849_mdcrd_v` resolved — real dissociation event, not an artifact and not encapsulation-driven.** Traced precisely: `assembly_rg` first exceeds 15 Å at **frame 4778 of 5000** (last 4.4% of the trajectory), where `largest_connected_component_size` drops from 6 to 5 — one monomer detaches from the assembly. It later drops to 4, and Rg peaks at 34.9 Å; the same 123 frames fail on both criteria. `n_guest_inside_cavity = 0` at that exact frame, so the break is not coincident with a guest forcing its way through. The guest *did* visit the cavity earlier in the same trajectory (1168/5000 frames, 23%), which had no destabilizing effect until the unrelated dissociation four-fifths of a nanosecond later. Re-checked with the implemented rule across all 302 trajectories in `output/gsa_features_by_cube/`, re-runs included: **exactly one hit**, this replica. The next-highest Rg anywhere is 12.2 Å (`BMHpH_157923`), and its re-run `BMHpH_563849_re_mdcrd_v` passes. It is genuinely rare, not a systematic issue with the BMHpH cohort or the pole-H design.

**QC rule: adopted and implemented (Oct 13).** A trajectory fails if any frame has `assembly_rg > 15 Å` or `largest_connected_component_size < 6`, applied identically to all six cohorts before clustering. Code: `src/ChangepointAnalysis/assembly_qc.py` (`scan_assembly_qc_dir`, `assembly_qc_failures`, `drop_trajectories`); `SegmentClusteringConfig.exclude_traj_ids` drops failures before clustering. `scripts/cluster_changepoint_segments.py --qc-gsa-dir output/gsa_features_by_cube/{CUBE}` scans one cube and writes `assembly_qc.csv`; `scripts/compare_to_murata_metastructures.py` applies the rule by default (`--no-assembly-qc` turns it off). Tests: `tests/test_assembly_qc.py`. Detection-level rates in §3 still include the replica. The own-elbow re-cluster and the regenerated G5 tables apply the rule; it changes only the BMHpH `pH`-series row. In the published BMHpH k=5 clustering this replica's last two segments (frames 4775–4995, the dissociation window) were the two singleton clusters. Report the dissociation itself as a separate one-line finding — a rare, late-trajectory monomer-loss event in the least-stabilized pole-H, low-equatorial-methyl system — rather than a silent drop.

**Cohort definition: decided (Oct 13).** Re-runs are kept alongside their originals, so BMHpH and BMMpM count seeds `563849` and `257803` twice; the QC rule then removes the dissociated BMHpH `563849` original, leaving BMHpH at 50 clustered trajectories. The truncated runs (BMHpH `109345`, BMMpM `109345`, `257803`, `587901`) stay in, and rates are reported per simulated time (§3.3).

### O2 `[x]` — resolved: apo-optimized cage, no encapsulation-derived geometry

**User confirmed:** our starting structures are a "pure cage form" — an optimized GSA cube built with no guest ever involved — with water and iodide dropped in externally afterward. This is consistent with the frame-0 data above (guest in bulk, zero inside at t=0) and is not a contradiction; the earlier flag was raised because the phrasing was ambiguous, not because the data was wrong.

**This is a methodological point in our favor, worth stating in C1.** Murata's cage geometry traces back to the X-ray structure of the 4₆·(TBM)₂ complex, with the TBM guests removed before their own iodide is added — meaning their starting cavity shape could carry transient memory of a bound aromatic guest. Ours is built as an idealized apo structure with no such history. Comparability to Murata's Table 1 is **not** blocked by any starting-state issue. G5 occupancy uses apo frames (`n_guest_inside_cavity < 1`), matching Murata's non-encapsulated filter. The guest-entry-rate table remains useful context (it independently tracks their methylation-dependent encapsulation trend).

### G3 `[x]` Scan k=2–15 per cohort for chemically meaningful clusters — **implemented, run on the own-elbow segments, and reviewed**

Implemented in `src/ChangepointAnalysis/cluster_k_diagnostics.py` (`scan_cohort_chemical_k`, `assess_chemical_separation`, `scan_chemical_separation_by_k`, `select_k_by_chemical_information`), wired into `scripts/compare_endpoint_cluster_k.py --chemical-scan` and `scripts/select_chemical_k.py`, covered by `tests/test_cluster_attribution.py`. The paper's run is on the own-elbow segments: `output/endpoint_cluster_chemical_k_own_elbow/{chemical_separation_by_k.csv, chemical_k_peaks.csv, chemical_k_murata_nmi_by_k.csv, chemical_k_selected.csv}`. The older `output/endpoint_cluster_chemical_k/` is the published-penalty run and is superseded.

**Gate (not a k picker):** per cohort, per k in 2–15, `chemical_separation` is true when the top-ranked site pairs mark ≥2 usable clusters (size ≥ 5) with distinct contacts or mixed open/closed Cohen's *d* signs. Silhouette is geometric only. `first_chemical_k` = smallest passing k. `best_chemical_k` = passing k with the most marked clusters (ties → smaller k). That last rule inflates with k and is **not** the automatic pick.

**Automatic k (chemical information):** silhouette is never in the argmax.

1. Keep only k with `chemical_separation=True` (kills the k=2 silhouette trap).
2. Among those, find the k with the highest NMI between frame-level apo `cluster_label` and independent Murata A/B/C1/C2/other (`label_motif_metastructures` on motif CSVs; those labels were not used to build the clusters).
3. **One-SE rule (decided Oct 13):** pick the smallest gated k whose NMI is within one standard error of that maximum. The SE is the spread of NMI at the max-NMI k over 500 replica bootstrap resamples (trajectories drawn with replacement, the same draws at every k; `score_murata_nmi_by_k(n_bootstrap=500)`). Without an SE column the picker falls back to the plain maximum.
4. AMI (chance-corrected) is reported; with ~1.3–2.3×10⁵ apo frames it agrees with NMI on every cube. Cramér's V is reported but not used to pick (it peaks at k=2, which fails the chemical gate).

Scripts: `python scripts/cluster_changepoint_segments.py ... --silhouette-k-max 15 --all-k-max 15` per cube (the "Done" note below), `python scripts/compare_endpoint_cluster_k.py --own-elbow --chemical-scan --chemical-k-max 15 --out-dir output/endpoint_cluster_chemical_k_own_elbow`, then `python scripts/select_chemical_k.py --own-elbow --chem-dir output/endpoint_cluster_chemical_k_own_elbow` (k range read from the scan; `--n-bootstrap 500` default). Plot with SE bars and the selected k starred: `output/endpoint_cluster_chemical_k_own_elbow/plots/murata_nmi_vs_k.png`.

**Roles — clustering vs the site-pair gate vs NMI/AMI (do not invert causality).**

The gate does **not** separate segments. Clustering does. Order:

1. Pelt cuts each trajectory into segments (dwells).
2. Clustering groups those segments in endpoint-feature space. That assignment *is* `cluster_label`.
3. Site-pair ranking (η² / Cohen's *d*) takes those labels and asks which named contacts (`M2S2–M3S3`, …) differ between groups. A site pair is one intermolecular endpoint–endpoint distance.
4. `chemical_separation` is a yes/no on that ranking: did the top-10 pairs mark ≥2 clusters of size ≥ 5 with distinct contacts **or** mixed open/closed *d* (`|d| ≥ 0.5`)? If no, discard that k (`too_few_usable_clusters`, `single_marked_cluster`, `no_distinct_contacts_or_signs`). The gate has no rule like "if this pair is open, this is cluster 2."
5. NMI/AMI score surviving partitions against independent Murata A/B/C1/C2/other on apo frames.

Without clustering there are no groups for the gate to inspect. We still cluster even though Murata already has A/B/C1/C2/other: those are **hand geometric criteria on frames** (π(pole) ≥ 6.5 Å, d1 ≥ 7.0 Å). The clusters are **unsupervised segment states** from the changepoint pipeline (aggregates, not Murata's thresholds). The claim is whether those data-driven dwells recover chemically distinct contacts and, secondarily, Murata's table — not to re-implement Murata as a classifier. Using Murata labels or the site-pair gate to *define* groups would make later NMI and η² circular.

**NMI vs AMI.** Both measure shared information between `cluster_label` and Murata A/B/C1/C2/other. They are not the chemical gate and not percent-correct. NMI is shared entropy on `[0, 1]` (`0` = independent, `1` = one label is a function of the other). BHHpH selected NMI **0.21** is weak-to-moderate overlap; BMMpH **0.74** is tight coupling (A-heavy cube). AMI is the same mutual information minus chance agreement at that k — the check that extra clusters are not inflating NMI by chance. Here AMI ≈ NMI to under 10⁻⁴ at every k, and the AMI maximum (`ami_among_chemical_k`) falls at the max-NMI k in every cube, so chance correction changes nothing. What it cannot catch is a finer cut that carries genuinely more, but statistically indistinguishable, label information; that is what the one-SE rule handles. AMI is the diagnostic.

**Per-cohort results on the own-elbow segments, k=2–15 (Oct 13):**

| Cohort | Silhouette-max k | Chem at sil-max? | first_chem | best_chem | Max-NMI k (NMI, SE) | **selected_k (one-SE)** | NMI | Chem at k=5? |
|---|---|---|---|---|---|---|---:|---|
| BHHpH | 2 | No | 3 | 8 | 15 (0.233, 0.024; scan edge) | **7** | 0.214 | Yes |
| BHHpM | 8 | Yes | 4 | 8 | 14 (0.253, 0.018) | **4** | 0.250 | **No** (one marked cluster) |
| BMHpH | 2 (one-replica split) | No | 4 | 4 | 4 (0.496, 0.028) | **4** | 0.496 | **No** (one marked cluster) |
| BMHpM | 12 | Yes | 5 | 10 | 15 (0.431, 0.016; scan edge) | **12** | 0.428 | Yes |
| BMMpH | 7 | Yes | 3 | 3 | 3 (0.744, 0.024) | **3** | 0.744 | Yes |
| BMMpM | 13 | No | 4 | 4 | 4 (0.593, 0.031) | **4** | 0.593 | **No** (only k=4 passes) |

The k ≤ 10 scan had picked 9, 4, 4, 10, 3, 4 by plain max NMI; do not quote those. The published-penalty run selected 4, 3, 7, 5, 3, 9 (max NMI, k ≤ 10) and passed at k=5 in all six cubes. Do not quote it either: those are different segments, and its NMI also counted the stale-CSV "other" frames (G5 item 1).

What to state plainly:

1. **k=5 is no longer chemical in every cube.** On the own-elbow segments it fails the gate in BHHpM, BMHpH and BMMpM, where the top site pairs mark only one cluster. So k=5 is a comparability cut only. G5 now also cross-tabulates at each cube's selected k (`output/murata_g5_own_elbow_selected_k/`).
2. **Silhouette-max k=2 still never passes** (BHHpH, BMHpH). The silhouette peaks of BHHpM (k=8), BMHpM (k=12) and BMMpH (k=7) do pass; only BMHpM's coincides with the selected k. BMMpM's silhouette peaks at k=13 (0.415), which fails the gate.
3. **Two small clusters are single replicas loosening late in the run.** BHHpH k=5 cluster 3 is `907683` from frame 3650 to the end (5 segments). BMHpH k=5 cluster 0 is `157923` from frame 2400 to the end (7 segments); it is also BMHpH's whole k=2 split (silhouette 0.53).
   - After the dissociated `563849`, these two have the largest assembly Rg maxima among all 302 replicas: 12.2 Å (`157923`) and 11.9 Å (`907683`), against a median of 10.9 Å.
   - Mean Rg in the clustered window is about 0.7 Å above the same replica's earlier frames, and the window contains the replica's Rg maximum. In `157923`, monomer 0's endpoints sit 2.5–5.4 Å farther from monomers 1–3 than in any other BMHpH cluster (segment-mean endpoint distances).
   - All six monomers stay connected, so the QC rule (Rg > 15 Å or a component below 6) keeps them. They are loosened but intact cages, not dissociation. Both trajectories end while still expanded, so whether they would recover is unknown.
   - Check: in the published BMHpH clustering the two singleton clusters were `563849`'s last two segments (frames 4775–4995), its dissociation window, which the QC rule now removes.
   - Ward cuts are nested, so both clusters survive unchanged at the selected k: BHHpH k=7 cluster 4 and BMHpH k=4 cluster 0. The BMHpH one dominates that cube's site-pair ranking (G4).
4. **Why the one-SE rule.** The scan was extended to k=15 to see whether BMHpM's NMI peaks. It does not: it falls from k=2 to k=7, then climbs to a plateau of 0.423–0.431 from k=12 to 15. The extension also moved two other maxima by less than the noise. BHHpH's NMI is flat at 0.210–0.222 from k=7 to 14 and reaches 0.233 at k=15. BHHpM's k=4 value (0.250) sits 0.003 below its k=14 maximum. The replica-bootstrap SEs at the six maxima are 0.016–0.031, so plain max NMI was rewarding finer cuts for gains it cannot resolve. The one-SE picks:
   - BHHpH: k=7, the start of the flat stretch.
   - BHHpM: k=4.
   - BMHpM: k=12, the start of the plateau.
   - BMHpH, BMMpH and BMMpM: unchanged, and each is also its max-NMI k.

   Run over k ≤ 10, the same rule gives 7, 4, 4, 10, 3, 4, so only BMHpM's pick depends on the scan range, and at k=15 it sits on the plateau. BHHpH's maximum is still at the new scan edge, but its one-SE pick has not moved between the two ranges. The library NMI matches sklearn's (`tests/test_cluster_attribution.py`). A separate bootstrap script (`output/_explore_nmi_bootstrap.py`, gitignored) reproduced the stored NMI to 10⁻⁶ and gave the same picks. Decided Oct 13 (user's call) over keeping max NMI at k ≤ 10 or at k ≤ 15.

**Correction to an earlier draft of this section:** it speculated BMMpM might show "no chemical separation at any k" as a "clean negative case." That is wrong. On the own-elbow segments BMMpM passes `chemical_separation` at k=4 only, over k=2–15 (on the published segments: k=4, 5, 6, 9, over k=2–10), while its silhouette climbs from 0.280 at k=2 to 0.415 at k=13. The geometric-compactness score and the chemical-separation flag track different things. Do not describe BMMpM as chemically featureless; describe it as having one chemical cut while silhouette climbs steadily. `docs/endpoint_cluster_discrimination.md` §13 has the own-elbow table.

**One interpretive caveat worth carrying into the manuscript:** `n_significant_fdr` runs about 630–960 out of 960 candidate columns at every k, and 830–960 once k ≥ 3. That is expected given the known redundancy among raw site-pair columns (~26 PCs explain 80% of their variance, per `docs/endpoint_cluster_discrimination.md` §4) and should not be read as 900 independent discoveries — report it as "most of a highly correlated feature set clears the FDR bar," not as a large number of separate findings.

**Done (Oct 13): the paper clusters the final own-elbow segments**, for consistency with section 3. The folders are `output/endpoint_changepoints_matched_{BHHpH,BHHpM,BMHpM,BMMpH}`, `output/endpoint_changepoints_g1a_BMHpH` and `output/endpoint_changepoints_g1b_BMMpM` (`OWN_ELBOW_ENDPOINT_DIRS` in `src/ChangepointAnalysis/reporting.py`). They hold no `endpoint_features/` of their own; `resolve_endpoint_features_dir` points them at the published folder's. Steps, logs in `output/own_elbow_recluster_logs/`:
1. `scripts/cluster_changepoint_segments.py --changepoints-dir <own-elbow dir> --groups endpoint --qc-gsa-dir output/gsa_features_by_cube/{CUBE} --silhouette-k-max 15 --all-k-max 15` per cube (Ward, k=5 stored, k=2–15 under `by_k/`). First run with k ≤ 10; re-run to k=15 the same day, and the stored k=5 tables came out identical in all six cubes (Ward is deterministic). `--groups endpoint` used to be rejected by the argument parser; fixed.
2. `scripts/summarize_changepoint_results.py` per cube (populates the G4 columns at k=5).
3. The G3 scan and selection (commands above), with `--own-elbow`.
4. `scripts/rank_endpoint_pairs_at_k.py --own-elbow --k-selected output/endpoint_cluster_chemical_k_own_elbow/chemical_k_selected.csv` (G4 at the selected k).
5. The G5 cross-tabs (G5, "How the numbers were generated").

| Cube | Trajectories clustered | Segments | Silhouette at k=5 | k=5 cluster sizes (segments) |
|---|---:|---:|---:|---|
| BHHpH | 50 | 553 | 0.131 | 116, 107, 201, 5, 124 |
| BHHpM | 50 | 538 | 0.179 | 61, 107, 122, 182, 66 |
| BMHpH | 50 | 276 | 0.228 | 7, 129, 55, 43, 42 |
| BMHpM | 50 | 380 | 0.197 | 131, 45, 60, 68, 76 |
| BMMpH | 50 | 206 | 0.493 | 166, 15, 9, 6, 10 |
| BMMpM | 51 | 157 | 0.341 | 76, 27, 16, 8, 30 |

BMHpH's 50 is 51 minus the QC-excluded `563849` original. `k_diagnostics_summary.txt` in the own-elbow scan folder has blank paper-d1 columns: `paper_d1_segment_states.csv` belongs to the published segmentation and was not regenerated, and it is not used by the gate or the NMI. All cluster artifacts in the published `output/endpoint_changepoints_{CUBE}/` folders still belong to the published penalties.

### G4 `[x]` Fix the attribution statistics framing — **implemented, run, and reviewed**

η² still cannot *validate* the clusters — that circularity (clusters built from per-monomer-pair aggregates whose components η² then ranks) is unchanged and correctly still documented. What's now added is a calibrated ranking: a label-shuffle permutation null (default 999 shuffles, cluster sizes preserved) and Benjamini–Hochberg FDR across the ~960 candidate columns.

**Implementation, verified correct:**
- `benjamini_hochberg_qvalues` — standard step-up BH procedure (sort ascending, running minimum from the top rank down, clipped to [0,1]). Matches the textbook definition; unit-tested for monotonicity and NaN handling.
- `permutation_eta_squared_pvalues` — vectorized column-wise η² (`_eta_squared_columns`, checked against the original scalar `_eta_squared` in tests and numerically identical), one label shuffle reused across all candidate columns per permutation (efficient, standard practice), add-one p-value formula `(1 + count) / (n_perm + 1)`. Unit test plants a known signal column and confirms it gets the minimum possible p-value and the smallest q-value among a null background.
- New columns on `endpoint_pair_cluster_correlation.csv`: `eta_squared_p_value`, `eta_squared_q_value`, `n_permutations`, `significant_fdr`.

Ran the full existing test suite after this refactor (`rank_cluster_discriminating_endpoint_features` was split into `collect_endpoint_segment_means` + `score_cluster_discriminating_features` to support G3's per-k reuse) — all 36 pre-existing tests plus the 6 new G3/G4 tests pass, no regressions.

**Done (Oct 13) on the own-elbow segments.** Each of the six own-elbow folders has `endpoint_pair_cluster_correlation.csv` with the p/q columns (999 permutations, k=5 labels), from `scripts/summarize_changepoint_results.py`. 923–958 of the 960 site-pair columns clear FDR in each cube (see the G3 caveat on redundancy). Top pairs by η²: BHHpH M2S3–M5S3 0.79, BHHpM M0S2–M1S3 0.85, BMHpH M4S5–M5S1 0.84, BMHpM M1S7–M3S2 0.86, BMMpH M1S2–M5S6 0.96, BMMpM M0S6–M4S7 0.97. The published-penalty files were left unpopulated on purpose. In BHHpM, BMHpH and BMMpM the k=5 labels fail the G3 gate, so their k=5 rankings mark only one cluster.

**Done (Oct 13): ranked at each cube's selected k**, with the same 999-shuffle null and BH-FDR (`scripts/rank_endpoint_pairs_at_k.py`). These are the rankings for the attribution Results. The full ranking per cube is `clusters/endpoint/by_k/k_XX/endpoint_pair_cluster_correlation.csv` in its own-elbow folder; summaries are in `output/endpoint_pair_ranking_selected_k/` (`top_pairs.csv`, `fdr_summary.csv`).

| Cube | k | Pairs clearing FDR (of 960) | Top pair (η²) | Clusters marked by the top 10 (segments) |
|---|---:|---:|---|---|
| BHHpH | 7 | 957 | M2S3–M5S3 (0.79) | 97-segment cluster (6), 116 (4) |
| BHHpM | 4 | 955 | M2S3–M5S3 (0.78) | all three belt-broken clusters: 168 (5), 122 (3), 66 (2) |
| BMHpH | 4 | 935 | M2S3–M4S3 (0.76) | **7-segment cluster (6)**, 55 (4) |
| BMHpM | 12 | 960 | M0S2–M4S7 (0.92) | four clusters of 18–28 segments |
| BMMpH | 3 | 848 | M0S6–M3S7 (0.60) | 24-segment cluster (9), 166 (1) |
| BMMpM | 4 | 901 | M2S2–M4S2 (0.84) | 38-segment cluster (8), 27 (2) |

Two cautions for the Results:
- **η² grows with k.** More groups explain more variance (BMHpM's 0.92 at k=12 against BMMpH's 0.60 at k=3), so compare pairs within a cube, not η² values across cubes.
- **BMHpH's ranking is mostly one replica.** Its 7-segment cluster is `157923` from frame 2400 to the end, the loosened cage of G3 item 3. Six of its top 10 pairs mark that cluster, so its population-level contacts are the ones marking the 55-segment cluster. BHHpH's single-replica cluster (`907683`, 5 segments) marks none of its top 10.

**Code:** `src/ChangepointAnalysis/reporting.py` — `_eta_squared_columns`, `benjamini_hochberg_qvalues`, `permutation_eta_squared_pvalues`, `score_cluster_discriminating_features`, `rank_cluster_discriminating_endpoint_features` (now a thin wrapper). Tests: `tests/test_cluster_attribution.py`.

### G5 `[~]` The Murata comparison — Table 1 occupancy final; 3₆ reading drafted

G0 no longer blocks this. Occupancy is from motif frames labeled A/B/C1/C2/other, apo filter (`n_guest_inside_cavity < 1`). **2₆′ has no cohort** — three-way comparison only.

**1. Population comparison `[x]`** against Murata Table 1. Source: `output/murata_g5/murata_table1_occupancy.csv` (the same in the own-elbow folders; occupancy does not depend on clustering), bar plot `output/murata_g5/plots/occupancy_vs_murata_table1.png`, trajectory-bootstrap 95% intervals in `output/murata_g5/label_diagnostics/occupancy_bootstrap.csv`.

| | 1₆ / BMMpM ours | Murata | **2₆ / BMHpM** ours | Murata | 2₆′ — no cohort | 3₆ / BHHpM ours | Murata |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 78.3 | 78.4 | 44.7 | 45.3 | 19.7 | 7.5 | 7.2 |
| B | 8.8 | 8.7 | 13.9 | 13.8 | 13.3 | 15.8 | 15.8 |
| C1 | 2.3 | 2.3 | 1.7 | 1.7 | 19.1 | 5.4 | 4.7 |
| C2 | 10.4 | 10.4 | 39.1 | 38.6 | 38.9 | 40.5 | 26.3 |
| other | 0.2 | 0.2 | 0.5 | 0.5 | 9.0 | 30.7 | 46.1 |

**1₆ and 2₆ match Table 1 within 0.6 pp in every class**, and every Murata value lies inside our 95% interval. 3₆ matches A/B/C1 within 0.7 pp. The residual is C2 vs other (+14.2 pp C2, −15.4 pp other), and both lie outside our intervals (C2 40.5 [34.6, 45.9], other 30.7 [25.2, 36.3]); C2 + other agrees (71.2 vs 72.4). Item 3 has the reading. Quote the apo-frame numbers above, not the `all`/`never` slices, unless a sensitivity paragraph is needed.

**Corrected (Oct 13): stale motif CSVs.** The first version of this table had 1₆ "other" at 2.4% and 2₆ at 2.2%. `discover_motif_rmsd_csvs_by_cube` also loaded the older RMSD-only prototype files in `output/murata_motif_rmsd/{BMMpM,BMHpM,BHHpM}/`, which repeat trajectories already loaded from `output/{CUBE}_motif_rmsd/`. Those files have no opening count, and `attach_murata_metastructure_labels` filled the missing count with −1, which the classifier files as "other". Fixed in code: discovery keeps one file per trajectory and prefers the per-cube folder, and a frame without an opening count stays unlabeled and leaves the denominator (tests in `tests/test_murata_rmsd.py`, `tests/test_murata_criteria.py`). 3₆ barely moved (other 30.8 → 30.7). All G5 tables were regenerated, and the own-elbow G3 NMI was computed after the fix; the published-penalty NMI in `output/endpoint_cluster_chemical_k/` was not.

`pH` series (not in Murata), same labels, apo frames: BHHpH 5.8 / 10.8 / 10.0 / 37.6 / 35.8 (tracks 3₆); BMHpH 43.7 / 11.6 / 3.1 / 39.9 / 1.7 (tracks 2₆; the QC rule drops `563849`); BMMpH **91.2 / 7.2 / 0.4 / 1.2 / 0.0** (more locked in A than 1₆). Pole-H × equatorial-methyl is a result Murata's design cannot show.

**2. Transition-frequency comparison `[x]`.** Section 3.3's own-elbow ratio per 5 ns (1 : 3.11 : 4.60 vs Murata 1 : 2.67 : 3.67) is the number to quote. G1b is decided.

**3. Resolving "other" `[~]`.** The 3₆ C2/other reading is drafted below, and the three belt-broken BHHpM clusters are now named (next paragraph). Still open: 2₆′ "other" (9.0%), which has no cohort.

**What distinguishes the three belt-broken BHHpM clusters (Oct 13).** They are one deformation in three symmetry-related orientations, not three different states. Each stretches the cage along one pair of opposite monomers. Their site pairs change by these amounts relative to the intact-belt cluster (mean over each monomer pair's 64 site pairs; for scale, site pairs within the intact-belt cluster have a mean SD of 1.1 Å):

| Cluster (segments) | Stretched axis | Compressed axis | Pole contacts that open | Pole contacts that stay closed (< 4% open) | Elongated d1 (frames with one elongated) |
|---|---|---|---|---|---|
| 0 (168) | M0–M3, +2.03 Å | M1–M4, −0.98 Å | M0–M1 74%, M2–M0 60%, M5–M3 61%, M3–M4 32% | M1–M2, M4–M5 | M5–M0 70%, M2–M3 25% |
| 1 (122) | M1–M4, +2.09 Å | M2–M5, −1.20 Å | M0–M1 65%, M1–M2 34%, M3–M4 56%, M4–M5 74% | M2–M0, M5–M3 | M0–M4 72%, M3–M1 28% |
| 3 (66) | M2–M5, +2.05 Å | M0–M3, −1.08 Å | M1–M2 54%, M2–M0 60%, M4–M5 75%, M5–M3 52% | M0–M1, M3–M4 | M1–M5 45%, M4–M2 53% |

- Stretching an axis pulls its two monomers away from their pole neighbours. Those four same-pole site-pair sets lengthen by 0.68–1.60 Å, and they are exactly the pole cation–π contacts that open. The elongated d1 is the one in which an axis monomer is the r3 end (each monomer is the r3 end of exactly one d1). The pole pair that involves neither axis monomer shortens by 0.79–1.05 Å and stays closed. Equatorial pairs move by at most 0.68 Å, so the belt break is local, not a gross separation.
- The monomer relabellings that preserve both directed contact graphs (pole and d1; found by brute force over all 720) are the six powers of one rotation along the d1 belt, `(0 4 2 3 1 5)`. That rotation maps the M0–M3 axis to M1–M4, and its fourth power, `(0 1 2)(3 4 5)`, maps M1–M4 to M2–M5. Correlating deformation profiles over all 960 site pairs: as labelled, the three clusters are unrelated (r = −0.03 to −0.12); after the best relabelling they match (r = 0.98 for clusters 0 and 1, 0.93 for 0 and 3, 0.96 for 1 and 3).
- So the contacts that tell them apart are simply the stretched and compressed axes: the opposite-monomer pairs (M0–M3, M1–M4, M2–M5) and the same-pole pairs of the axis monomers. The G4 ranking at k=4 picks exactly these. Seven of its top 10 pairs join opposite monomers (five M2–M5, one M0–M3, one M1–M4) and three join same-pole neighbours; between them they mark all three belt-broken clusters. Ward splits the deformation by orientation because endpoint distances carry monomer identities; Murata's labels count contacts and cannot see orientation. This is also why C2 and "other" spread over the three clusters in similar proportions.
- Sources: `scripts/describe_endpoint_clusters.py --own-elbow --cube BHHpM --k 4`, writing `output/endpoint_cluster_profiles/BHHpM_k04/` (`cluster_motif_profile.csv`, `single_elongated_edge.csv`, `pair_deformation.csv`, `symmetry_profile_correlation.csv`, `subset_pair_ranking.csv`).

**Draft paragraph (3₆ C2 vs other):**

> For 1₆ and 2₆ our frame labels reproduce Murata's Table 1 to within 0.6 percentage points in every class, and for 3₆ they reproduce A, B and C1 to within 0.7 points. The one disagreement is how 3₆'s opened frames divide between C2 and "other": we find 40.5% C2 and 30.7% other, against Murata's 26.3% and 46.1%, while the two together agree (71.2% vs 72.4%). The gap lies outside our trajectory-bootstrap 95% intervals, so it is not sampling noise in our ensemble; it reflects how close 3₆ sits to the boundary between the two classes. A C2 frame has exactly two of the six pole cation–π contacts open (π(pole) ≥ 6.5 Å) and one elongated equatorial d1 (≥ 7.0 Å), and a third opening makes it "other". In 1₆ and 2₆ the third-widest π(pole) of a C2 frame stays closed (median 4.1 Å). In 3₆ it is often half open (median 5.0 Å), and 14% of C2 frames lie within 0.5 Å of the cutoff, against under 1% in 1₆ and 2₆. The number of open contacts in 3₆ accordingly switches between two and three or more 47 times per nanosecond, against 0.5 times in 1₆ and 2₆, and about two-thirds of the three-open episodes last 2 ps or less. Our endpoint clusters, built without reference to Murata's criteria, do not separate the two classes either; they separate frames by whether the equatorial belt is intact. At the automatically selected cut for 3₆ (k = 4), one cluster holds 86% of the frames with no elongated d1 and 99% of the A frames, while C2 and "other" frames fall into the three belt-broken clusters in similar proportions (41.7, 36.2 and 20.5% of C2; 48.5, 30.9 and 18.3% of other). Those three clusters are one deformation in three orientations: in each, the cage is stretched by about 2 Å along a different pair of opposite monomers, and their distance profiles coincide once the monomers are relabelled by the cage's symmetry. What the clustering resolves is orientation, not C2 versus "other". We therefore read 3₆'s C2 and "other" as one family of belt-broken structures in which a third contact opens and closes on a picosecond timescale, and treat their sum as the robust quantity. Because only 3₆ has a large population near the cutoff, a small systematic difference between our π(pole) measurement and Murata's would act almost entirely on this split. Moving our cutoff alone does not reconcile it, however: at 6.0 Å, 3₆ shifts only to 36.3% C2 and 37.7% other, while its B occupancy falls from 15.8 to 12.8% (Murata 15.8%) and 2₆'s A from 44.7 to 43.1% (Murata 45.3%).

**Evidence behind the paragraph** (apo frames unless noted; files under `output/murata_g5/label_diagnostics/` from `scripts/murata_label_diagnostics.py` unless noted):
- Intervals: `occupancy_bootstrap.csv` (2000 trajectory resamples).
- Near-cutoff C2 frames: `c2_near_cutoff.csv`. Third-widest π(pole) in C2 frames: median 5.04 Å in 3₆ (q75 5.70, q90 6.14) vs 4.11 and 4.09 Å in 1₆ and 2₆. Within 0.5 Å below 6.5 Å: 14.4% vs 0.8% and 0.3%. The second-longest d1 is rarely near its cutoff (2.2% in 3₆), so the split is about a third opening, not a second elongated d1.
- What 3₆ "other" is made of (`open_by_elongated.csv`): 27.2 pp from frames with ≥3 openings, 3.5 pp from two openings with ≥2 elongated d1.
- Switching (`open_count_switching.csv`, all frames in time order): 46.9 switches per ns between two and ≥3 open in 3₆, 0.52 in both 1₆ and 2₆. 3₆ has 5,982 ≥3-open episodes, median 2 ps, 65% lasting ≤ 2 ps.
- Cutoff grid (`cutoff_sensitivity.csv`, π 6.0–7.0 Å × d1 6.5–7.5 Å): lowering the π cutoff moves 3₆ toward Murata's split, but no cell fixes it. 3₆'s worst class stays ≥ 9.5 pp off everywhere, and the cells that come closest (π 6.0 Å) put 2₆'s A 2.2 pp off. Summed over all 15 numbers, the deviation from Table 1 only ranges 29.9–32.2 pp for π ≤ 6.5 Å; the error moves between classes rather than shrinking.
- Clusters (`output/murata_g5_own_elbow_selected_k/endpoint_cluster_composition.csv` and `murata_vs_endpoint_cluster.csv`, BHHpM at k=4): intact-belt cluster 2 holds 85.7% of the intact-belt frames; 92.3% of its own frames have no elongated d1; it has 98.8% of A, 65.3% of B and 73.1% of C1 frames, but only 1.6% of C2 and 2.3% of other. Belt-broken clusters 0, 1 and 3 have one elongated d1 in 83–88% of frames, and each mixes two-open (55–62%) and ≥3-open (32–37%) frames.
- How evenly C2 and other spread over the clusters (total-variation distance): 7.4% at k=4 and 8.8% at k=5 (`output/murata_g5_own_elbow/`). The intact-belt cluster is the same from k=4 to k=10; finer cuts separate C2 from other only partly (19–25% at k=8–10; `--k 8`, `--k 10`).
- Not attributable from our data: why Murata's split leans toward "other". That needs Murata's exact π(pole) reference points.

**How the numbers were generated** (do not re-derive by hand). Scripts:

```
python scripts/compare_to_murata_metastructures.py --own-elbow --out-dir output/murata_g5_own_elbow
python scripts/compare_to_murata_metastructures.py --own-elbow --k-selected output/endpoint_cluster_chemical_k_own_elbow/chemical_k_selected.csv --out-dir output/murata_g5_own_elbow_selected_k
python scripts/murata_label_diagnostics.py
```

Two file families are joined, only for the three mapped cubes (BHHpM / BMHpM / BMMpM). Replicas failing the G2 QC rule are dropped first in both scripts.

| Role | Files | What is used |
|---|---|---|
| Murata label, per 1 ps frame | `output/{CUBE}_motif_rmsd/**/*_motif_rmsd.csv`, one file per trajectory | `n_open_cation_pi` (pole π ≥ 6.5 Å) and locked `equator_d1_e*` ≥ 7.0 Å → A/B/C1/C2/other via `label_motif_metastructures`. Apo = `n_guest_inside_cavity < 1` (`guest_filter=frames`) |
| Endpoint cluster, per segment | With `--own-elbow`: the folders in `OWN_ELBOW_ENDPOINT_DIRS` (G3 note). The stored `clusters/endpoint/segments_clustered.csv` is the k=5 cut; `--k` or `--k-selected` read `clusters/endpoint/by_k/k_XX/` instead | `cluster_label`, `start_frame`, `end_frame`, `traj_id` |
| Murata Table 1 % | hardcoded `MURATA_TABLE1_PCT` in `src/ChangepointAnalysis/murata_criteria.py` | Published occupancies. **Not** recomputed from Murata's trajectories |

Join key: cube prefix stripped from `traj_id` (`BHHpM_109345_mdcrd_v` → `109345_mdcrd_v`), then motif `frame` inside the segment interval `[start_frame, end_frame]`. Frames with no matching segment are dropped; these are the 1 ps frames that fall between consecutive stride-5 segments or after the last one, about 1% (BHHpM: 56,484 of 57,024 apo "other" frames joined).

Outputs, in `output/murata_g5_own_elbow/` (k=5) and `output/murata_g5_own_elbow_selected_k/` (k = 4 for BHHpM, 12 for BMHpM, 4 for BMMpM; regenerated Oct 13 after the one-SE rule moved BMHpM from 10 to 12):

- `murata_table1_occupancy.csv` — our % vs Table 1 (item 1; the same in every folder).
- `murata_vs_endpoint_cluster.csv` — row-normalized % of each Murata label in each endpoint cluster, with the k used.
- `murata_other_vs_endpoint_cluster.csv` — among apo frames labeled **other**, the share in each cluster. BHHpM at k=4: 48.5 / 30.9 / 2.3 / 18.3% in clusters 0–3.
- `endpoint_cluster_composition.csv` — each cluster's apo frames by elongated-d1 count and opened-π count, and its share of the cohort's intact-belt frames.
- `assembly_qc.csv` — the G2 QC table for all six cubes.

`output/murata_g5/` holds the same tables against the published-penalty clusters; quote its occupancy, not its cluster tables.

**Cluster integers are arbitrary, per cohort and per k.** BHHpM cluster 1 is not the same state as BMHpM cluster 1, and BHHpM's k=4 cluster 0 splits into k=5 clusters 0 and 1. The script does not match IDs across cubes. Name clusters by content (for example "intact-belt cluster"), not by integer.

**Code:** `scripts/compare_to_murata_metastructures.py` (`label_motif_metastructures` in `src/ChangepointAnalysis/murata_rmsd.py`; Table 1 constants in `src/ChangepointAnalysis/murata_criteria.py`).

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

**Still needed from your own records:** whether our trajectories are otherwise Murata's exact production protocol or an independent re-run (this also decides the wording in §1 and §4), the provenance of the three `pH` cohorts, why the three `_re_` re-runs exist, and why four runs are truncated (G2).

### C2 `[ ]` Figure 1 schematic

Pipeline overview: trajectory to two feature families to segmentation to clustering to attribution, with a real segmented time series as the spine.

### C3 `[ ]` Limitations paragraph

No free-energy surface, no rate constants, attribution is correlational, cluster labels are arbitrary integers, cluster count imposed for comparability. Add: our segments are not Markovian states either, for the reason Murata gives and section 3.5 now quantifies.

---

## 7. Release hygiene

- **R1 `[x]`** Every reported detection has a script.
  - `scripts/recompute_matched_penalties.py` reproduces the `matched_*` runs; its BMHpH and BMMpM outputs are the superseded wrong-grid ones.
  - `scripts/recompute_own_elbow_g1.py` reproduces the final `g1a_BMHpH` / `g1b_BMMpM` detections and their timing runs. With `--sweep` it first redoes the median-length penalty sweep. The original detections' parameters are in their run manifests, `output/711c6c3ecd81/manifest.json` (BMHpH, Sep 10) and `output/7dd823f0fd4a/manifest.json` (BMMpM, Sep 11).
  - Verified Oct 13 with a from-scratch run (`--sweep --output-root output/_g1_check`, a scratch folder deleted after the comparison). All 212 detection and timing files came out byte-for-byte identical: per-trajectory and pooled breakpoints and segment stats, and both timing tables. The 8 sweep tables differ only in the penalty values, by under 10⁻⁶ relative, because the original sweep was given six-digit grid bounds. The recommended penalties, as passed to detection, are the same: 4.36145 and 5.48888.
  - The own-elbow re-cluster, G3, the G4 re-rank and G5 steps are scripted (G3 "Done" note).
- **R2 `[ ]`** Tag a release and mint a Zenodo DOI. The analysis it was waiting on is done, and the quoted k is decided (G3, one-SE rule). No tags exist yet.
- **R3 `[ ]`** Data availability. Feature CSVs are small enough to deposit in full.
- **R4 `[x]`** Tests for the G0 criteria, the G4 permutation null and the G2 QC rule. G4: `tests/test_cluster_attribution.py`. G0: `tests/test_gsa_site_map.py` (CPy para to phenylene linker), `tests/test_murata_criteria.py` (A/B/C1/C2 labels, occupancy vs Table 1 mapping), `tests/test_murata_rmsd.py` (unit columns, distance plots, metastructure labels from elongated d1). G2 QC: `tests/test_assembly_qc.py`. The `paper_d1_n_compact` rename and the legacy-name reader are covered in `tests/test_changepoint_pipeline.py`.

---

## 8. Dependency order

```
G0 ✓ done ──► G5 `[~]` occupancy final; 3₆ C2/other paragraph drafted
G1 ─► G1a ─► G1b ✓ done ─► §3.3 final ─► G5
G2 QC ✓ ─► cohort definition ✓ decided ─► own-elbow re-cluster ✓ done
own-elbow re-cluster ─► G3 ✓ + G4 ✓ re-run ─► quoted k ✓ (one-SE) ─► G4 at that k ✓ + BHHpM belt-broken reading ✓ ─► Results (attribution)
C1 ──► Methods (done; provenance questions open)
own-elbow re-cluster ──► R2 ──► R3
```

**Critical path:** write the attribution Results from the selected-k rankings (G4; treat BMHpH's as mostly one replica), with the BHHpM orientation reading (G5 item 3) for 3₆, then C2 (Figure 1) and C3 (Limitations). G0, G1b, G2, O1, O2, R1, the re-cluster, the k choice, the G3/G4 runs and the BHHpM cluster reading are closed, and the 3₆ C2/other paragraph is drafted.

---

## 9. Sequencing

| Stage | Work | Gate |
|---|---|---|
| 1 | ~~Correct `docs/penalty_sweep_B_cohorts.md`~~ done | Docs match code |
| 2 | ~~G0 implementation~~ done | S11-style d1 vs open π; CPy = para to phenylene linker |
| 3 | G5 three-way comparison | Table 1 occupancy final (1₆/2₆ within 0.6 pp; 3₆ C2 vs other read in G5 item 3). BHHpM belt-broken clusters named (one deformation, three orientations) |
| 4 | ~~Re-cluster on own-elbow segments with the QC rule, re-run G3/G4~~ done | ~~New cluster tables; pick the quoted k~~ done (one-SE rule, k=2–15) |
| 5 | C2, C3, draft | Full draft |
| 6 | R1–R4, submit | Release tagged |

---

## 10. Numbers to re-verify before quoting

| Quantity | Status | Note |
|---|---|---|
| Endpoint elbows | **resolved, final** | Each cube its own elbow (G1b decision). `docs/penalty_sweep_B_cohorts.md` corrected to match |
| Cohort sizes | **resolved, decided** | All 50–51, each with exactly 50 unique seeds; the 51s count one seed twice (`_re_` re-runs), kept as is (decided Oct 13). The QC rule removes BMHpH `563849`, so BMHpH clusters 50. Four runs are truncated and stay in; rates are per simulated time (G2). Earlier 44/30 were trajectories with ≥1 breakpoint |
| Guest-entry denominators | **explained** | The G2 table counts originals only. With the three re-runs: BMHpH 42/51, BMMpH 12/50, BMMpM 14/51 |
| endpoint–gsa Jaccard | **0.09–0.20** | Final, own-elbow throughout. Supersedes the single 0.18. BMMpM is **0.087** (corrected; the earlier 0.125 was timed against published-penalty GSA), computed over only 32/51 trajectories — always report `n traj` alongside it. Lowest in the fully methylated cube of each series; no clean gradient in `pM` |
| combined–gsa Jaccard 0.63 | **do not quote** | `combined` contains the gsa columns. Use `gsa`–`iodine` (0.40–0.56) as the independent within-GSA pair |
| Transition-rate ratio | **1 : 3.11 : 4.60 per 5 ns — final** | Own elbow throughout (G1b decided); per simulated time because four runs are truncated (decided Oct 13; 1 : 3.18 : 4.70 per trajectory). vs Murata 1 : 2.67 : 3.67 — this overshoots, by design: BMMpM's genuine elbow sits at a higher relative penalty (0.795×) than the rest (0.50–0.63×). Do not quote the intermediate matched-relative figure of 1 : 2.29 : 3.39; that run is rejected |
| GSA-vs-endpoint event ratio | 4.9–13.2x | Not "2–3x". Both cover 5 ns, so comparable, but endpoint cannot resolve sub-5 ps events |
| "0% open cation–π in BMM" | **withdrawn** | Mis-specified criterion; see section 2 |
| Motif d1 / π(pole) (all B*) | **S11-style, verified** | `output/murata_motif_rmsd/plots/dist_d1_d2_by_open_cation_pi.png`. Compact ~4–5 Å when π closed; elongated ~8–9 Å grows as π opens. C2–C3 ipso, not methyl-carbon proxy |
| Hull `s3`/`s7` d1 ~9–17 Å | **superseded** | Wrong sites; use locked R2–R3 ipso edges in motif CSVs |
| Table 1 occupancy (apo frames) | **corrected Oct 13, verified** | BMMpM 78.3/8.8/2.3/10.4/0.2 vs 78.4/8.7/2.3/10.4/0.2; BMHpM 44.7/13.9/1.7/39.1/0.5 vs 45.3/13.8/1.7/38.6/0.5; BHHpM 7.5/15.8/5.4/40.5/30.7 vs 7.2/15.8/4.7/26.3/46.1. Quote apo frames. The earlier 2.4% / 2.2% "other" for BMMpM / BMHpM came from stale motif CSVs (G5 item 1). 3₆ residual is C2 vs other |
| 3₆ C2 vs other split | **explained, not reconciled** | Outside our bootstrap 95% intervals; C2 + other agrees (71.2 vs 72.4). Third π(pole) near the cutoff in 14% of 3₆ C2 frames, and switching at 47 per ns; clusters split by belt state, not C2 vs other. No cutoff in the grid fixes it. Draft paragraph in G5 item 3 |
| Remapped equatorial d1 (old hull remap) | **superseded for G5** | `output/murata_d1/` was the hull-column remap. G5 uses motif-CSV ipso d1 (`output/B*_motif_rmsd/`) |
| Top site-pair η² | partly circular, now calibrated | At the selected k: 0.60–0.92 (BHHpH M2S3–M5S3 0.79, BHHpM M2S3–M5S3 0.78, BMHpH M2S3–M4S3 0.76, BMHpM M0S2–M4S7 0.92, BMMpH M0S6–M3S7 0.60, BMMpM M2S2–M4S2 0.84), with 848–960 of 960 pairs clearing FDR (G4). η² grows with k, so do not compare it across cubes. BMHpH's top pairs mostly mark the single replica `157923`. The k=5 values (0.79–0.97) and the published-penalty 0.81–0.84 are superseded |
| BHHpM belt-broken clusters | **resolved Oct 13** | One deformation in three orientations, each stretching a different opposite-monomer axis by about 2 Å. Deformation profiles: r = −0.03 to −0.12 as labelled, 0.93–0.98 after the best symmetry relabelling (G5 item 3) |
| BMMpM 37% static at true own elbow | **holds, final** | 19/51 have zero breakpoints at 5.489. The earlier 41% (21/51) used the wrong-grid elbow (5.256) and is superseded. Leaving out the three truncated originals: 17/48 = 35%. Corroborates Murata's 78.4% A for 1₆ |
| k=5 is chemically meaningful | **revised Oct 13** | On the own-elbow segments k=5 passes the gate only in BHHpH, BMHpM and BMMpH. Silhouette-max k=2 still never passes. Automatic k (chemical gate, then one-SE rule on Murata NMI over k=2–15; `output/endpoint_cluster_chemical_k_own_elbow/chemical_k_selected.csv`): **7, 4, 4, 12, 3, 4** for BHHpH→BMMpM. Superseded: 9, 4, 4, 10, 3, 4 (plain max NMI, k ≤ 10) and the published-segment result (all six pass; 4, 3, 7, 5, 3, 9) |
| "BMMpM shows no chemical separation at any k" | **withdrawn** | Was speculation, not yet checked. Own-elbow scan: passes at k=4 only over k=2–15 (published segments: k=4, 5, 6, 9), while silhouette climbs to 0.415 at k=13. See G3 |

---

*Repository state: branch `MDchat`. G0 from `output/B*_motif_rmsd/` and `output/murata_motif_rmsd/plots/`. G5 occupancy from `output/murata_g5/`. G1 from `output/matched_penalty_recompute_summary.csv`. G1a/G1b from `output/g1a_relative_penalty_correction.csv`, `output/endpoint_changepoints_g1a_BMHpH` (own elbow, 4.361, full 51-traj run), and `output/endpoint_changepoints_g1b_BMMpM` (own elbow, 5.48888, full 51-traj run) + `output/endpoint_vs_gsa_timing_g1b_BMMpM_gsa_elbow` (timed against the GSA own-elbow detections; supersedes `output/endpoint_vs_gsa_timing_g1b_BMMpM`); both regenerate with `scripts/recompute_own_elbow_g1.py`. G2 QC from `src/ChangepointAnalysis/assembly_qc.py` over `output/gsa_features_by_cube/`. Own-elbow re-cluster in the six `OWN_ELBOW_ENDPOINT_DIRS` folders (logs in `output/own_elbow_recluster_logs/`); G3 from `output/endpoint_cluster_chemical_k_own_elbow/` (one-SE selection, k=2–15); G4 at the selected k from `output/endpoint_pair_ranking_selected_k/`; G5 cluster tables from `output/murata_g5_own_elbow/` and `output/murata_g5_own_elbow_selected_k/`; BHHpM belt-broken cluster reading from `output/endpoint_cluster_profiles/BHHpM_k04/`; G5 label diagnostics from `output/murata_g5/label_diagnostics/`.*
