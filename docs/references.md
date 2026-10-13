# References

The papers this project is positioned against. `plan.md` §0 explains how each one is used.

## Murata 2026: the comparison reference

Murata, Kobayashi, Hiraoka, Shimazaki, Tachikawa. "Metastructure Analysis of Self-Assembled Nanocubes with Different Equatorial Methyl Groups Based on MD Simulations." *J. Phys. Chem. B* (2026). DOI: [10.1021/acs.jpcb.6c01069](https://doi.org/10.1021/acs.jpcb.6c01069)

- All-atom GAFF/RESP model in explicit TIP4P water; 50 independent 5 ns NVT runs per system.
- Defines metastructures A, B, C1, C2 and "other" from hand-built geometric criteria: pole cation–π opening (π(pole) ≥ 6.5 Å) and elongated equatorial d1 (≥ 7.0 Å).
- System mapping to our cohorts: 1₆ = BMMpM, 2₆ = BMHpM, 3₆ = BHHpM. 2₆′ has no cohort here.
- Table 1 occupancies are hard-coded as `MURATA_TABLE1_PCT` in `src/ChangepointAnalysis/murata_criteria.py`, not recomputed.

## Imamura 2020: method precedent

Imamura, Yamamoto, Sato. "Coarse-grained modeling of nanocube self-assembly system and transition network analyses." *Chem. Phys. Lett.* **742**, 137135 (2020).

- Coarse-grained model (25 beads per monomer) with implicit solvent via IBI; REMD, 12 replicas × 2 µs.
- Studies how six dispersed monomers assemble into a cube; 22 Ward macrostates from 1500 k-means microstates at a 2 ns lag.
- Cited as background and as a clustering precedent. A/B/C1/C2 do **not** come from this paper.
- Reproduced by `src/utils/imamura_msm.py` (see `docs/imamura_msm_presentation.md`).

## Murata 2023: pole substitution

Murata et al. *J. Phys. Chem. B* **127**, 328 (2023). Reference 19 of Murata 2026. Covers pole substitution on its own, not crossed with equatorial substitution.
