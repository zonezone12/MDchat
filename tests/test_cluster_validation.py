"""Synthetic checks for post-hoc endpoint-cluster validation (Tests A–D)."""

import numpy as np

from src.ChangepointAnalysis.cluster_validation import (
    core_split,
    coverage_at_radius,
    dilution_verdict,
    kabsch_rmsd,
    mappings_from_perms,
    pairwise_symmetry_distances,
    refit_ward_labels,
    reorient_rows,
    replica_holdout_folds,
    robust_z,
    sensitivity_family_table,
    symmetry_adjusted_distance,
    tangled_pairs_from_correlations,
    two_population_test,
)
from src.ChangepointAnalysis.metastable_catalog import (
    BELT_SIX_CYCLE,
    INVERSION_PERM,
    adjacency_automorphisms,
    contact_automorphisms,
    family_ids as catalog_family_ids,
    is_proper_permutation,
    perm_cycles,
    rotation_contact_automorphisms,
    site_pair_column_permutation,
)


def _site_specs(n_mon=6, n_sites=2):
    specs = []
    for i in range(n_mon):
        for j in range(i + 1, n_mon):
            for a in range(n_sites):
                for b in range(n_sites):
                    specs.append((f"endpoint_dist_{i}s{a}_{j}s{b}", i, a, j, b))
    return specs


def test_relabelled_copy_has_zero_adjusted_distance():
    specs = _site_specs()
    rng = np.random.default_rng(0)
    x = rng.normal(size=len(specs))
    perm = BELT_SIX_CYCLE
    mapped = site_pair_column_permutation(specs, perm)
    y = np.empty_like(x)
    # y is x after applying perm: y[mapped[k]] = x[k] wait
    # (y ∘ g)[k] = y[mapped[k]] should equal x[k] if y is the relabelled copy of x
    # so y[mapped] = x  → y[mapped[k]] = x[k]
    y[mapped] = x
    mappings = mappings_from_perms(specs, [perm])
    d, _ = symmetry_adjusted_distance(x, y, mappings)
    assert d == 0.0 or d < 1e-12


def test_inversion_maps_opposite_site_pair():
    specs = [
        ("endpoint_dist_0s1_3s2", 0, 1, 3, 2),
        ("endpoint_dist_0s2_3s1", 0, 2, 3, 1),
        ("endpoint_dist_1s0_4s0", 1, 0, 4, 0),
    ]
    mapped = site_pair_column_permutation(specs, INVERSION_PERM)
    # 0s1–3s2 → (3,1,0,2) which is the reverse of 0s2–3s1
    assert int(mapped[0]) == 1
    assert perm_cycles(INVERSION_PERM) == "(0 3)(1 4)(2 5)"


def test_forty_eight_adjacency_and_six_contact_automorphisms():
    adj = adjacency_automorphisms()
    contact = contact_automorphisms()
    assert len(adj) == 48
    assert len(contact) == 6
    assert len({tuple(p) for p in adj}) == 48
    rotations = rotation_contact_automorphisms()
    assert len(rotations) == 3
    assert all(is_proper_permutation(p) for p in rotations)
    assert not is_proper_permutation(INVERSION_PERM)
    assert not is_proper_permutation(BELT_SIX_CYCLE)
    assert perm_cycles(BELT_SIX_CYCLE) == "(0 4 2 3 1 5)"
    assert any(tuple(p) == INVERSION_PERM for p in contact)
    assert any(tuple(p) == BELT_SIX_CYCLE for p in contact)


def test_chiral_object_needs_reflection():
    # A chiral tetrahedron and its mirror image.
    ref = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.4, 1.5, 0.0],
            [0.2, 0.3, 1.8],
        ]
    )
    mirror = ref.copy()
    mirror[:, 0] *= -1.0
    rmsd_rot, rot = kabsch_rmsd(mirror, ref, allow_reflection=False)
    rmsd_ref, rot_i = kabsch_rmsd(mirror, ref, allow_reflection=True)
    assert np.linalg.det(rot) > 0
    assert np.linalg.det(rot_i) < 0
    assert rmsd_ref < 1e-8
    assert rmsd_rot > 0.2


def test_planted_outlier_raises_max_not_p95():
    rng = np.random.default_rng(1)
    specs = _site_specs(n_sites=1)
    X = rng.normal(scale=0.05, size=(40, len(specs)))
    X[-1] = X[-1] + 8.0
    mappings = mappings_from_perms(specs, contact_automorphisms())
    dist, _ = pairwise_symmetry_distances(X, mappings)
    pairs = dist[np.triu_indices(40, k=1)]
    assert pairs.max() > 5.0
    assert np.quantile(pairs, 0.95) < 3.0


def test_two_population_flags_and_ignores_a_single_outlier():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 0.2, size=20)
    b = rng.normal(6.0, 0.2, size=20)
    reps = np.array([f"r{i % 5}" for i in range(40)])
    mixed = two_population_test(np.concatenate([a, b]), reps, min_segments=15)
    assert mixed["flag"] == "two_populations"
    one = np.concatenate([rng.normal(0.0, 0.2, size=24), [8.0]])
    one_reps = np.array([f"r{i % 6}" for i in range(25)])
    outlier = two_population_test(one, one_reps, min_segments=15)
    assert outlier["flag"] in {"single_outlier", "homogeneous"}


def test_coverage_high_on_exchangeable_and_low_on_a_shift():
    rng = np.random.default_rng(3)
    specs = _site_specs(n_sites=1)
    mappings = mappings_from_perms(specs, [(0, 1, 2, 3, 4, 5)])
    train = rng.normal(size=(30, len(specs)))
    test = train[:10] + rng.normal(scale=0.01, size=(10, len(specs)))
    cov = coverage_at_radius(train, test, mappings)
    assert cov >= 0.8
    shifted = test + 4.0
    cov_shift = coverage_at_radius(train, shifted, mappings)
    assert cov_shift <= 0.2


def test_orientation_copies_collapse_after_reorientation():
    specs = _site_specs(n_sites=1)
    rng = np.random.default_rng(4)
    base = rng.normal(size=len(specs))
    perm = BELT_SIX_CYCLE
    mapped = site_pair_column_permutation(specs, perm)
    copy = np.empty_like(base)
    copy[mapped] = base
    noise = rng.normal(scale=0.01, size=(8, len(specs)))
    X = np.vstack([base + noise[:4], copy + noise[4:]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    mappings = mappings_from_perms(specs, contact_automorphisms())
    Y, _ = reorient_rows(X, X[0], mappings)
    # after reorientation the two groups sit on top of each other
    d, _ = pairwise_symmetry_distances(Y, [np.arange(Y.shape[1])])
    within = d[:4, :4].mean()
    between = d[:4, 4:].mean()
    assert between < within * 2 + 0.2
    # two different motifs stay apart
    other = base.copy()
    other[:10] += 5.0
    X2 = np.vstack([base + noise[:4], other + noise[4:]])
    Y2, _ = reorient_rows(X2, X2[0], mappings)
    d2, _ = pairwise_symmetry_distances(Y2, [np.arange(Y2.shape[1])])
    assert d2[:4, 4:].mean() > d2[:4, :4].mean() * 2


def test_family_ids_complete_linkage_keeps_a_weak_pair_apart():
    labels = [0, 1, 2]
    pairs = [(0, 1, 0.96), (1, 2, 0.93), (0, 2, 0.82)]
    single = catalog_family_ids(labels, pairs, threshold=0.90, linkage="single")
    complete = catalog_family_ids(labels, pairs, threshold=0.90, linkage="complete")
    assert single[0] == single[1] == single[2]
    assert complete[0] == complete[1]
    assert complete[2] != complete[0]


def test_sensitivity_table_counts_structures():
    table = sensitivity_family_table(
        [0, 1, 2],
        [(0, 1, 0.96), (1, 2, 0.93), (0, 2, 0.82)],
        thresholds=(0.90,),
        linkages=("single", "complete"),
    )
    single = table[(table["linkage"] == "single") & (table["copy_r"] == 0.90)].iloc[0]
    complete = table[(table["linkage"] == "complete") & (table["copy_r"] == 0.90)].iloc[0]
    assert int(single["n_structures"]) == 1
    assert int(complete["n_structures"]) == 2


def test_tangled_pairs_near_miss_and_chain_weak():
    # Six belt copies join at 0.95 through neighbours; 7–8 is the weak link.
    # Intact 3 meets the belt at 0.91 and stays its own family.
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    belt = [1, 2, 4, 6, 7, 8]
    pairs = []
    for i, a in enumerate(belt):
        for b in belt[i + 1 :]:
            pairs.append((a, b, 0.89 if {a, b} == {7, 8} else 0.97))
    pairs.append((7, 3, 0.91))
    pairs.append((0, 1, 0.10))
    table = tangled_pairs_from_correlations(labels, pairs, threshold=0.95, r_min=0.70)
    reasons = {(int(r.cluster_a), int(r.cluster_b)): r.tangle_reason for r in table.itertuples(index=False)}
    assert reasons[(7, 8)] == "chain_weak"
    assert reasons[(3, 7)] == "near_miss"
    assert (0, 1) not in reasons
    assert (1, 2) not in reasons


def test_dilution_verdict_split_from_distinct():
    assert dilution_verdict(0.75, 0.97, n_core_a=10, n_core_b=10) == "dilution_missed_merge"
    assert (
        dilution_verdict(0.89, 0.97, n_core_a=10, n_core_b=10, tangle_reason="chain_weak")
        == "chain_clears"
    )
    assert dilution_verdict(0.75, 0.76, n_core_a=10, n_core_b=10) == "distinct_deformations"
    assert dilution_verdict(0.97, 0.99, n_core_a=10, n_core_b=10) == "orientation_copies"
    assert dilution_verdict(0.75, 0.97, n_core_a=2, n_core_b=10) == "too_few_core"


def test_core_split_keeps_the_far_part_as_core():
    rng = np.random.default_rng(0)
    near = rng.normal(0.0, 0.1, size=(12, 4))
    far = rng.normal(0.0, 0.1, size=(8, 4)) + np.array([3.0, -3.0, 3.0, -3.0])
    other = rng.normal(0.0, 0.1, size=(10, 4)) - 1.0
    X = np.vstack([near, far, other])
    y = np.array([0] * 20 + [1] * 10)
    parts = core_split(X, y, clusters=[0])
    assert list(parts[12:20]) == ["core"] * 8
    assert list(parts[:12]) == ["weak"] * 12
    assert set(parts[20:]) == {""}
    assert core_split(X[:1], y[:1])[0] == "core"


def test_holdout_folds_hold_each_replica_out():
    reps = np.array(["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3 + ["e"] * 3)
    folds = replica_holdout_folds(reps, n_splits=5, n_repeats=2, random_state=0)
    assert len(folds) == 10
    # every replica is held out at least once per repeat
    for block in (folds[:5], folds[5:]):
        held = set()
        for mask in block:
            held.update(set(reps[~mask].tolist()))
        assert held == set(reps.tolist())


def test_robust_z_flags_a_far_point():
    z = robust_z(np.array([1.0, 1.1, 0.9, 1.05, 10.0]))
    assert z[-1] > 3.5
    assert np.all(np.abs(z[:-1]) < 3.5)


def test_refit_ward_is_deterministic():
    rng = np.random.default_rng(5)
    X = np.vstack(
        [
            rng.normal(0, 0.1, size=(10, 4)),
            rng.normal(3, 0.1, size=(10, 4)),
        ]
    )
    a = refit_ward_labels(X, 2)
    b = refit_ward_labels(X, 2)
    assert np.array_equal(a, b)
    assert len(set(a.tolist())) == 2
