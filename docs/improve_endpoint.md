# Topology-first endpoint detection

The original endpoint finder takes a 2D RDKit depiction, keeps atoms on the convex hull (and a few opposites through the centroid), then optionally caps how many hits a ring may contribute (`ring_max_per_ring=3`). Ring hits are later collapsed to fused systems. That is a reasonable way to pick *visual tips*, but it is a poor chemical model:

- The site count is an artefact of the drawing and of the cap, not of the molecule.
- Internal but chemically important groups (a hexaaryl core, a pyridinium that is not on the hull) are dropped.
- Methyl substitution can turn a ring site into an atom site and shift every later index — the problem that `gsa_site_map.py` already works around for GSA.
- Distances between hull atoms do not encode fragment orientation, aromatic plane normals, hydrogen bonds, cation–π, or π–π stacking.

This note describes a three-layer replacement that **does not impose a fixed number of landmarks**, how to call it, and the tests that lock the behaviour.

The hull pipeline is unchanged and remains the default (`EndpointsFinder(site_source="hull")`). GSA changepoint work still uses the canonical eight-role map in `gsa_site_map.py` unless `site_source="layered"` is set explicitly.

---

## Layer 1 — Molecular topology

`discover_topology_sites(mol)` walks the molecular graph once.

| Unit | How it is found | Site kind |
|------|-----------------|-----------|
| Fused ring systems | RDKit rings merged by shared atoms (union-find) | `fused_ring` |
| Functional groups | SMARTS: pyridinium, ammonium, carboxyl/carboxylate, nitro, cyano, CF₃, methyl, hydroxyl, amine, carbonyl, ether | `functional_group`, or a *tag* on a ring that already contains the match |
| Rigid fragments | Heavy-atom connected components after cutting Lipinski rotatable bonds; leftover atoms not already in a ring or group | `rigid_fragment` |

There is no `k` and no `ring_max_per_ring`. Benzene yields one site; *p*-xylene yields three (ring + two methyls); hexaphenylbenzene yields seven rings.

Each site also stores:

- **labels** — `aromatic`, `phenyl`, `pyridine`, `pyridinium`, `fused`, `methyl`, …
- **formal charge** (sum over the site; pyridinium rings get `charge = +1`)
- **neighbors** — other sites sharing a bond (the molecular connectivity graph of fragments)
- **peripheral** — graph leaf / maximum eccentricity, or on the 2D convex hull of *site centroids* (the hull is a flag, not the source of the catalog)

### Layer 1 results (SMILES)

| Molecule | Hull atoms | Hull sites | Layered sites | Notes |
|----------|-----------:|-----------:|--------------:|-------|
| benzene | 6 | 1 | 1 | phenyl ring |
| toluene | 6 | 1 | **2** | hull misses the methyl |
| *p*-xylene | 6 | 1 | **3** | count grows with substituents |
| biphenyl | 12 | 2 | 2 | one rotatable bond; two phenyls, neighbours of each other |
| naphthalene | 10 | 1 | 1 | one fused 10-atom system |
| pyridine | 6 | 1 | 1 | tagged `pyridine` |
| *N*-methylpyridinium | 6 | 1 | **2** | ring `pyridinium` charge +1, plus methyl |
| nicotinic acid | 7 | 2 | 2 | carboxyl (tagged hydroxyl+carbonyl) + pyridine |
| 4,4′-biphenyldicarboxylic acid | 2 | 2 | **4** | hull keeps only the two carboxyl tips |
| hexaphenylbenzene | 36 | **6** | **7** | hull drops the core ring (`peripheral=0`) |

Every hull atom in these molecules sits inside some layered site (`unmatched = 0`). The layered catalog is therefore a chemically labelled *cover* of the old tips, plus the groups the hull never saw.

Naphthalene is one fused system, not two rings. Biphenyl is two rings joined by a rotatable bond. That is the rigid-fragment / fused-ring distinction the hull never made.

---

## Layer 2 — Three-dimensional geometry

For the whole molecule and for every site, `compute_site_geometry` / `compute_catalog_geometries` return:

- **centroid**
- **principal axes** (SVD/PCA of the gyration tensor; right-handed, largest eigenvalue first)
- **shape descriptors** — Rg, asphericity \(B = \lambda_3 - \tfrac12(\lambda_1+\lambda_2)\), acylindricity \(C = \lambda_2-\lambda_1\), relative shape anisotropy \(\kappa^2\)
- **plane normal and planarity RMSD** for aromatic rings, and for any ≥5-atom site that is planar enough (RMSD < 0.25 Å)

Aromatic systems are therefore represented by **both** a centroid and a plane-normal vector, which Layer 3 needs for cation–π and π–π.

On a regular hexagon of radius 1:

| Quantity | Result |
|----------|--------|
| centroid | (0, 0, 0) |
| plane normal | (0, 0, 1) |
| planarity RMSD | 0 |
| Rg | 1.000 |

A three-point rod along *x* has its first principal axis along ±*x* (`|axis₁,x| > 0.95`).

Live MD coordinates are preferred when an MDAnalysis selection is available (`EndpointAnalyzer.compute_residue_layered_geometry`). The RDKit conformer is only a fallback.

---

## Layer 3 — Dynamic intermolecular interactions

Sites of different residues are scored every frame. Cutoffs (ångström / degree):

| Interaction | Criterion |
|-------------|-----------|
| Contact | heavy-atom pairs closer than **4.5 Å** |
| Hydrogen bond | donor N/O with H, acceptor N/O; D···A ≤ **3.5 Å**, ∠D–H–A ≥ **120°** |
| Cation–π | cation (N⁺ / charged site centroid) to aromatic centroid ≤ **6.5 Å** (Murata open/closed distance) **and** angle between cation–centroid vector and ring normal ≤ **45°** |
| π–π | centroid distance **3.0–5.5 Å**, \(\lvert n\cdot n'\rvert \ge \cos 30^\circ\), in-plane offset ≤ **2.0 Å** |

The same pair also records the continuous distances and angles, so a later analysis can change the cutoff without recomputing geometry.

`contact_run_stats` / `summarize_interface_stability` turn the per-frame flags into **interface stability**: persistence (fraction of frames on), number of on-runs, mean and max run length, and the number of on/off switches.

`LayeredInteractionObserver` discovers the Layer-1 catalog once, then on every frame computes Layer-2 geometries and Layer-3 records. It writes:

- `layered_interaction_frames.csv` — assembly-level counts per frame
- `layered_interface_stability.csv` — per site-pair persistence

### Layer 3 unit results

| Setup | Result |
|-------|--------|
| Two hexagons stacked at *z* = 3.8 Å | π–π **hit**, distance 3.800 Å, angle 0°, offset 0 |
| Same rings at *z* = 8 Å | **no** π–π |
| Cation at (0, 0, 4) above a hexagon | cation–π **hit**, distance 4.000 Å, angle 0° |
| Cation in the ring plane at *x* = 6 Å | **rejected** (angle > 45°) |
| Linear O–H···O (D at 0, H at 0.96, A at 2.8 Å) | 1 hydrogen bond |
| Bent H (angle ~82°) | 0 hydrogen bonds |
| Boolean contact series `[0,1,1,1,0,0,1,1]` | persistence 5/8, 2 runs, mean length 2.5, 3 switches |

Trajectory observer (two six-carbon rings, 6 frames: stacked for 3 frames, then pulled 5 Å apart): π–π is on in the stacked half and off in the separated half, and `contact_n_switches ≥ 1`.

---

## GSA monomers: site count is not eight

The published GSA pipeline maps every monomer to eight named roles (`ph`, `ph_para`, `py_eq`, `r2`, `r3`, `py_pole`, `r1`, `r1_ring`). That map is still the right *labelled* representation for Murata d1 / cation–π. Layered detection answers a different question: what does the graph actually contain?

On the Amber templates in `traj/{CUBE}_ca.prmtop` (monomer 0, topology only):

| Cube | Layered sites | Fused rings | Functional groups | Methyl sites | Pyridinium | Phenyl | Rotatable bonds |
|------|--------------:|------------:|------------------:|-------------:|-----------:|-------:|----------------:|
| BHHpH | 12 | 10 | 2 | 2 | 2 | 8 | 11 |
| BHHpM | 13 | 10 | 3 | 3 | 2 | 8 | 12 |
| BMHpH | 13 | 10 | 3 | 3 | 2 | 8 | 12 |
| BMHpM | 14 | 10 | 4 | 4 | 2 | 8 | 13 |
| BMMpH | 14 | 10 | 4 | 4 | 2 | 8 | 13 |
| BMMpM | 15 | 10 | 5 | 5 | 2 | 8 | 14 |

Ten fused rings are stable across the series (core + aryls, some fused in the Py-linkers). Methyl count is 2 constitutive groups plus R1/R2/R3 when those positions are CH₃, so BHHpH → BMMpM gains three sites and three rotatable bonds. Nothing in this table is forced to eight.

---

## API

```python
from src.EndpointAnalyzer import (
    EndpointsFinder,
    EndpointAnalyzer,
    LayeredInteractionObserver,
    discover_topology_sites,
)

# Layer 1
finder = EndpointsFinder(site_source="layered")
sites = finder.find_layered_sites(mol)          # list[list[int]], variable length
info = finder.find_layered_site_info(mol)       # kind, labels, charge, peripheral, …
catalog = discover_topology_sites(mol)

# Layers 1–2 on the current MD frame
center, mda_groups, info = EndpointAnalyzer.find_residue_layered_sites(
    u, "resname MOL and resid 1"
)
geom = EndpointAnalyzer.compute_residue_layered_geometry(u, "resname MOL and resid 1")

# Layer 3 over a trajectory
from src.TrajectoryIterator import TrajectoryIterator
obs = LayeredInteractionObserver(["resname MOL and resid 1", "resname MOL and resid 2"])
it = TrajectoryIterator(u)
it.subscribe(obs)
it.iterate()
obs.get_frame_summary_df()
obs.get_stability_df()
```

MDChat skills (after `from src.mdchat.skills import endpoints` / a normal `mdchat` session):

| Skill | What it does |
|-------|----------------|
| `find_layered_endpoints` | Layer 1 + 2 for a list of residue selections |
| `compute_layered_interactions` | Layer 3 observer pass; optional CSVs |

`find_endpoints` / `compute_endpoint_distances` still call the hull finder.

`EndpointAnalyzer.find_residue_endpoint_sites` still prefers the GSA canonical map when the residue is a Murata hexaaryl **and** `site_source` is not `"layered"`. Existing changepoint CSVs are therefore unchanged.

---

## Tests

`tests/test_layered_endpoints.py`, plus the existing hull and GSA-map regressions.

```
python -m pytest tests/test_layered_endpoints.py tests/test_gsa_site_map.py \
    tests/test_changepoint_pipeline.py::test_find_endpoint_sites_biphenyl_and_naphthalene -q
```

**32 passed** (26 new layered tests, 5 GSA-map tests, 1 hull biphenyl/naphthalene test). Run on 2026-11-03 against the Amber templates in `traj/`.

| Test | What it locks |
|------|----------------|
| `test_fused_naphthalene_is_one_system` | fused-ring merge |
| `test_biphenyl_has_two_rings_not_fused` | connectivity + phenyl tags |
| `test_toluene_ring_plus_methyl_no_fixed_k` | no landmark quota |
| `test_site_count_grows_with_substituents` | benzene < toluene < xylene |
| `test_hexaaryl_has_seven_rings_not_eight_landmarks` | hexaphenylbenzene = 7, not 6 or 8 |
| `test_pyridine_and_pyridinium_tags` | charge and methyl on Py⁺ |
| `test_carboxyl_is_its_own_site` | functional group vs ring |
| `test_connectivity_neighbors_biphenyl` | site graph |
| `test_peripheral_flag_marks_tips` | carboxyls are peripheral |
| `test_layered_finder_ignores_ring_max_cap` | `ring_max_per_ring` does not apply |
| `test_hull_atoms_map_into_layered_sites` | hull ⊂ layered cover |
| `test_benzene_centroid_and_plane_normal` | Layer 2 aromatic geometry |
| `test_fragment_principal_axis_follows_long_axis` | PCA orientation |
| `test_pi_pi_stacked_hexagons` / `test_pi_pi_too_far_is_not_stacking` | π–π cutoff |
| `test_cation_pi_above_ring` / `test_cation_pi_in_plane_is_rejected` | cation–π angle |
| `test_hydrogen_bond_linear_water_like` | D–H–A criterion |
| `test_contact_run_stats_and_interface_stability` | persistence / switches |
| `test_layered_interaction_observer_two_benzenes` | MD pass: stack then separate |
| `test_gsa_monomer_site_count_is_not_fixed_eight` | BMMpM > BHHpH site count |
| `test_skills_registered` | MDChat skill names |
| `test_find_endpoint_sites_biphenyl_and_naphthalene` | hull default still works |
| GSA map tests | canonical eight-role map untouched |

---

## Files

| Path | Role |
|------|------|
| `src/EndpointAnalyzer/layered_sites.py` | Layers 1–3 and `LayeredInteractionObserver` |
| `src/EndpointAnalyzer/endpoints_finder.py` | `site_source="layered"` and `find_layered_*` |
| `src/EndpointAnalyzer/EndpointAnalyzer.py` | `find_residue_layered_sites`, `compute_residue_layered_geometry` |
| `src/mdchat/skills/endpoints.py` | `find_layered_endpoints`, `compute_layered_interactions` |
| `tests/test_layered_endpoints.py` | tests summarised above |
