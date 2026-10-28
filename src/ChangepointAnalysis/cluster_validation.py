"""Post-hoc structural checks on endpoint clusters (Tests A–D).

The structural distance is a symmetry-adjusted distance-RMSD over the 960
inter-monomer site pairs. Nothing here re-segments a trajectory. Ward is
refit only inside Test C hold-out folds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd

from .metastable_catalog import (
    COPY_R,
    N_MONOMERS,
    contact_automorphisms,
    family_ids,
    perm_cycles,
    rotation_contact_automorphisms,
    site_pair_column_permutation,
)

POLE_H_FROM_POLE_M = {
    "BHHpH": "BHHpM",
    "BMHpH": "BMHpM",
    "BMMpH": "BMMpM",
}

NEGATIVE_CONTROL_PAIRS: dict[str, list[tuple[int, int]]] = {
    "BHHpH": [(0, 1), (3, 0), (3, 1)],
    "BHHpM": [(0, 1), (0, 3)],
    "BMHpH": [(4, 2), (4, 3), (5, 2), (5, 3)],
    "BMMpH": [(1, 2)],
    "BMMpM": [(1, 2), (1, 3)],
    "BMHpM": [(2, 7)],
}

DEFAULT_FRAME_PAIRWISE_CAP = 400
DEFAULT_FRAME_SAMPLE_CAP = 3000
DEFAULT_FRAME_STRIDE_PS = 50
DEFAULT_ENDPOINT_STRIDE_PS = 5


def kabsch_rmsd(
    mobile: np.ndarray,
    reference: np.ndarray,
    *,
    allow_reflection: bool = False,
) -> tuple[float, np.ndarray]:
    """RMSD and rotation that take *mobile* onto *reference*.

    Both clouds are centred. When ``allow_reflection`` is false the rotation
    has det +1. When true, the best orthogonal matrix (possibly improper) is
    kept.
    """
    p = np.asarray(mobile, dtype=float)
    q = np.asarray(reference, dtype=float)
    if p.shape != q.shape or p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("mobile and reference must be (n, 3) with the same shape")
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    h = pc.T @ qc
    u, _s, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0.0 and not allow_reflection:
        vt = vt.copy()
        vt[-1] *= -1.0
        rot = vt.T @ u.T
    aligned = pc @ rot.T
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - qc) ** 2, axis=1))))
    return rmsd, rot


def mappings_from_perms(
    specs: Sequence[tuple],
    perms: Sequence[Sequence[int]],
    *,
    site_maps: Sequence | None = None,
) -> list[np.ndarray]:
    """Column permutations for each monomer relabelling."""
    maps: list[np.ndarray] = []
    for i, perm in enumerate(perms):
        site_map = None if site_maps is None else site_maps[i]
        mapped = site_pair_column_permutation(specs, perm, site_map=site_map)
        if mapped.size and not (mapped < 0).any():
            maps.append(mapped)
    return maps


def permutation_group(
    kind: str,
    *,
    specs: Sequence[tuple] | None = None,
    site_maps: Mapping[str, Sequence | None] | None = None,
    extra_perms: Sequence[Sequence[int]] | None = None,
) -> tuple[list[tuple[int, ...]], list[np.ndarray]]:
    """Return ``(perms, column mappings)`` for ``rotations``, ``contact`` or an explicit list."""
    kind = str(kind).strip().lower()
    if extra_perms is not None:
        perms = [tuple(int(x) for x in p) for p in extra_perms]
    elif kind == "rotations":
        perms = rotation_contact_automorphisms()
    elif kind in {"contact", "contact_six"}:
        perms = contact_automorphisms()
    else:
        raise ValueError(
            "group must be 'rotations' or 'contact' unless admissible "
            "permutations are supplied via extra_perms"
        )
    maps = mappings_from_perms(specs or [], perms, site_maps=_site_maps_for(perms, site_maps))
    return perms, maps


def _site_maps_for(
    perms: Sequence[Sequence[int]],
    site_maps: Mapping[str, Sequence | None] | None,
) -> list | None:
    if not site_maps:
        return None
    return [site_maps.get(perm_cycles(p)) for p in perms]


def symmetry_adjusted_distance(
    x: np.ndarray,
    y: np.ndarray,
    mappings: Sequence[np.ndarray],
) -> tuple[float, int]:
    """Min distance-RMSD of *x* to a relabelled *y*. Returns ``(distance, mapping_index)``."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if not mappings:
        diff = x - y
        return float(np.sqrt(np.mean(diff * diff))), 0
    best = float("inf")
    best_i = 0
    for i, mapped in enumerate(mappings):
        diff = x - y[mapped]
        d = float(np.sqrt(np.mean(diff * diff)))
        if d < best:
            best = d
            best_i = i
    return best, best_i


def pairwise_symmetry_distances(
    X: np.ndarray,
    mappings: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric min-over-G distance-RMSD matrix and the winning mapping index."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    dist = np.zeros((n, n), dtype=float)
    best = np.zeros((n, n), dtype=int)
    if n < 2:
        return dist, best
    mapped = [X[:, m] if m is not None else X for m in mappings] if mappings else [X]
    stacked = np.stack(mapped, axis=0)
    for i in range(n):
        diff = stacked - X[i]
        rms = np.sqrt(np.mean(diff * diff, axis=2))
        choice = np.argmin(rms, axis=0)
        dist[i] = rms[choice, np.arange(n)]
        best[i] = choice
    dist = np.minimum(dist, dist.T)
    np.fill_diagonal(dist, 0.0)
    return dist, best


def labelled_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Distance-RMSD with fixed monomer labels."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        diff = X[i] - X
        dist[i] = np.sqrt(np.mean(diff * diff, axis=1))
    return dist


def pairwise_percentiles(values: np.ndarray) -> dict[str, float]:
    """Maximum, 95th percentile and median of a 1-d pair list."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"max": float("nan"), "p95": float("nan"), "median": float("nan")}
    return {
        "max": float(np.max(v)),
        "p95": float(np.quantile(v, 0.95)),
        "median": float(np.median(v)),
    }


def upper_triangle(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    if n < 2:
        return np.asarray([], dtype=float)
    return dist[np.triu_indices(n, k=1)]


def robust_z(values: np.ndarray) -> np.ndarray:
    """Median / MAD z-scores. MAD of 0 yields zeros."""
    v = np.asarray(values, dtype=float)
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros(v.shape, dtype=float)
    return 0.6745 * (v - med) / mad


def medoid_index(dist: np.ndarray) -> int:
    """Index with the smallest mean distance to the others."""
    if dist.shape[0] == 0:
        return 0
    if dist.shape[0] == 1:
        return 0
    means = dist.mean(axis=1)
    return int(np.argmin(means))


def cluster_diameter_rows(
    dist: np.ndarray,
    labelled: np.ndarray,
    labels: np.ndarray,
    *,
    keys: pd.DataFrame | None = None,
    unit: str = "segment",
    group_col: str = "cluster_label",
    eccentric_z: float = 3.5,
    eccentric_abs_min: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-group diameter statistics and per-member eccentricity."""
    labels = np.asarray(labels)
    rows = []
    members = []
    pair_cache: dict[int, np.ndarray] = {}
    for lab in sorted(set(int(x) for x in labels if np.isfinite(x))):
        idx = np.where(labels == lab)[0]
        sub = dist[np.ix_(idx, idx)]
        lab_sub = labelled[np.ix_(idx, idx)]
        pairs = upper_triangle(sub)
        lab_pairs = upper_triangle(lab_sub)
        pair_cache[lab] = pairs
        stats = pairwise_percentiles(pairs)
        lab_stats = pairwise_percentiles(lab_pairs)
        max_pair = ("", "")
        if pairs.size and keys is not None:
            ii, jj = np.unravel_index(np.argmax(sub + np.eye(len(idx)) * -1e9), sub.shape)
            a = keys.iloc[int(idx[ii])]
            b = keys.iloc[int(idx[jj])]
            max_pair = (
                f"{a.get('traj_id', '')}:{a.get('segment_id', ii)}",
                f"{b.get('traj_id', '')}:{b.get('segment_id', jj)}",
            )
        ecc = np.array(
            [float(np.median(np.delete(sub[i], i))) if len(idx) > 1 else 0.0 for i in range(len(idx))]
        )
        z = robust_z(ecc)
        flagged = (z > eccentric_z) & (ecc >= eccentric_abs_min)
        for local, global_i in enumerate(idx):
            rec = {
                "unit": unit,
                group_col: lab,
                "member_index": int(global_i),
                "eccentricity": float(ecc[local]),
                "eccentricity_z": float(z[local]),
                "is_eccentric": bool(flagged[local]),
            }
            if keys is not None:
                rec["traj_id"] = keys.iloc[int(global_i)].get("traj_id")
                rec["segment_id"] = keys.iloc[int(global_i)].get("segment_id")
            members.append(rec)
        within_med = float("nan")
        cross_med = float("nan")
        if keys is not None and "traj_id" in keys.columns and len(idx) > 1:
            tids = keys.iloc[idx]["traj_id"].astype(str).to_numpy()
            same = []
            other = []
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    (same if tids[i] == tids[j] else other).append(sub[i, j])
            if same:
                within_med = float(np.median(same))
            if other:
                cross_med = float(np.median(other))
        rows.append(
            {
                "unit": unit,
                group_col: lab,
                "n_members": int(len(idx)),
                "max": stats["max"],
                "p95": stats["p95"],
                "median": stats["median"],
                "labelled_max": lab_stats["max"],
                "labelled_p95": lab_stats["p95"],
                "labelled_median": lab_stats["median"],
                "max_pair_a": max_pair[0],
                "max_pair_b": max_pair[1],
                "n_eccentric": int(flagged.sum()),
                "within_replica_median": within_med,
                "cross_replica_median": cross_med,
                "medoid_index": int(idx[medoid_index(sub)]) if len(idx) else -1,
            }
        )
    # nearest-other separation
    for row in rows:
        lab = int(row[group_col])
        idx = np.where(labels == lab)[0]
        others = [o for o in pair_cache if o != lab]
        sep = float("nan")
        for other in others:
            oidx = np.where(labels == other)[0]
            block = dist[np.ix_(idx, oidx)]
            med = float(np.median(block)) if block.size else float("nan")
            if np.isfinite(med) and (not np.isfinite(sep) or med < sep):
                sep = med
        row["nearest_separation"] = sep
        p95 = row["p95"]
        row["separation_over_p95"] = (
            float(sep / p95) if np.isfinite(sep) and np.isfinite(p95) and p95 > 0 else float("nan")
        )
    return pd.DataFrame(rows), pd.DataFrame(members)


def within_dwell_spread(
    frame_X: np.ndarray,
    segment_index: np.ndarray,
    mappings: Sequence[np.ndarray],
    *,
    max_pairs_per_segment: int = 40,
    rng: np.random.Generator | None = None,
) -> float:
    """Median frame-to-frame symmetry-adjusted distance inside one segment."""
    rng = rng or np.random.default_rng(0)
    spreads = []
    for seg in np.unique(segment_index):
        idx = np.where(segment_index == seg)[0]
        if idx.size < 2:
            continue
        if idx.size > max_pairs_per_segment + 1:
            take = rng.choice(idx, size=max_pairs_per_segment + 1, replace=False)
        else:
            take = idx
        d, _ = pairwise_symmetry_distances(frame_X[take], mappings)
        pairs = upper_triangle(d)
        if pairs.size:
            spreads.append(float(np.median(pairs)))
    return float(np.median(spreads)) if spreads else float("nan")


def overlap_coefficient(a: np.ndarray, b: np.ndarray, *, bins: int = 32) -> float:
    """Histogram overlap coefficient of two 1-d samples."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 1.0
    ha, edges = np.histogram(a, bins=bins, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=True)
    width = np.diff(edges)
    return float(np.sum(np.minimum(ha, hb) * width))


def ashman_d(mu1: float, s1: float, mu2: float, s2: float) -> float:
    """Ashman's D for two Gaussian components. D ≥ 2 is well separated."""
    var = s1 * s1 + s2 * s2
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return float(np.sqrt(2.0) * abs(mu1 - mu2) / np.sqrt(var))


def two_population_test(
    values: np.ndarray,
    replica_ids: np.ndarray | None = None,
    *,
    min_segments: int = 15,
    delta_bic: float = 10.0,
    min_ashman: float = 2.0,
    min_minor: float = 0.15,
    min_minor_replicas: int = 3,
) -> dict[str, float | str | bool]:
    """Compare 1- vs 2-component GMMs on a 1-d or 2-d sample of segment values."""
    X = np.asarray(values, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    finite = np.isfinite(X).all(axis=1)
    X = X[finite]
    reps = None if replica_ids is None else np.asarray(replica_ids)[finite]
    n = X.shape[0]
    out: dict[str, float | str | bool] = {
        "n": int(n),
        "flag": "untested",
        "delta_bic": float("nan"),
        "ashman_d": float("nan"),
        "minor_frac": float("nan"),
        "minor_replicas": 0,
    }
    if n < min_segments:
        return out
    from sklearn.mixture import GaussianMixture

    x_fit = X.copy()
    sd = x_fit.std(axis=0)
    if not np.any(sd > 1e-12):
        out["flag"] = "homogeneous"
        return out
    sd = np.where(sd > 0, sd, 1.0)
    x_fit = (x_fit - x_fit.mean(axis=0)) / sd
    g1 = GaussianMixture(n_components=1, covariance_type="full", random_state=0)
    g2 = GaussianMixture(n_components=2, covariance_type="full", random_state=0)
    try:
        g1.fit(x_fit)
        g2.fit(x_fit)
    except ValueError:
        out["flag"] = "homogeneous"
        return out
    dbic = float(g1.bic(x_fit) - g2.bic(x_fit))
    means = g2.means_.ravel() if x_fit.shape[1] == 1 else g2.means_[:, 0]
    if x_fit.shape[1] == 1:
        cov = g2.covariances_.reshape(-1)
        s1, s2 = float(np.sqrt(max(cov[0], 1e-12))), float(np.sqrt(max(cov[1], 1e-12)))
        d_ash = ashman_d(float(means[0]), s1, float(means[1]), s2)
    else:
        pred = g2.predict(x_fit)
        d_ash = float("nan")
        if (pred == 0).sum() > 1 and (pred == 1).sum() > 1:
            d_ash = ashman_d(
                float(X[pred == 0, 0].mean()),
                float(X[pred == 0, 0].std() or 1e-12),
                float(X[pred == 1, 0].mean()),
                float(X[pred == 1, 0].std() or 1e-12),
            )
    pred = g2.predict(x_fit)
    counts = np.bincount(pred, minlength=2)
    minor = int(np.argmin(counts))
    minor_frac = float(counts[minor] / n)
    minor_reps = 0
    if reps is not None:
        minor_reps = int(pd.Series(reps[pred == minor]).nunique())
    out.update(
        {
            "delta_bic": dbic,
            "ashman_d": d_ash,
            "minor_frac": minor_frac,
            "minor_replicas": minor_reps,
        }
    )
    well = (
        dbic >= delta_bic
        and (not np.isfinite(d_ash) or d_ash >= min_ashman)
        and minor_frac >= min_minor
        and minor_reps >= min_minor_replicas
    )
    single = counts[minor] <= 1 or minor_reps <= 1
    if well:
        out["flag"] = "two_populations"
    elif dbic >= delta_bic and single:
        out["flag"] = "single_outlier"
    else:
        out["flag"] = "homogeneous"
    return out


def variance_split(
    values: np.ndarray,
    segment_ids: np.ndarray,
    replica_ids: np.ndarray,
) -> dict[str, float]:
    """Fraction of variance within segments, between segments, and between replicas."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    v = v[ok]
    segs = np.asarray(segment_ids)[ok]
    reps = np.asarray(replica_ids)[ok]
    tot = float(np.var(v, ddof=0)) if v.size else float("nan")
    if not np.isfinite(tot) or tot <= 0:
        return {"within_segment": float("nan"), "between_segment": float("nan"), "between_replica": float("nan")}
    seg_means = pd.Series(v).groupby(segs).transform("mean").to_numpy()
    within = float(np.var(v - seg_means, ddof=0) / tot)
    between_seg = 1.0 - within
    rep_means = pd.Series(v).groupby(reps).transform("mean").to_numpy()
    between_rep = float(np.var(rep_means, ddof=0) / tot)
    return {
        "within_segment": within,
        "between_segment": between_seg,
        "between_replica": between_rep,
    }


def deformation_deltas(
    X: np.ndarray,
    y: np.ndarray,
    clusters: Sequence[int] | None = None,
) -> dict[int, np.ndarray]:
    """Per-cluster mean change from the complementary segments."""
    y = np.asarray(y)
    X = np.asarray(X, dtype=float)
    if clusters is None:
        clusters = sorted(set(int(c) for c in y))
    clusters = [int(c) for c in clusters]
    ref = ~np.isin(y, clusters)
    base = X[ref].mean(axis=0) if ref.any() else X.mean(axis=0)
    return {c: X[y == c].mean(axis=0) - base for c in clusters if (y == c).any()}


def best_profile_r(
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    mappings: Sequence[np.ndarray],
) -> tuple[float, int]:
    """Pearson r of two deformation profiles after the best relabelling."""
    if not mappings:
        r = float(np.corrcoef(delta_a, delta_b)[0, 1])
        return r, 0
    best_r = float("-inf")
    best_i = 0
    for i, mapped in enumerate(mappings):
        r = float(np.corrcoef(delta_a, delta_b[mapped])[0, 1])
        if np.isfinite(r) and r > best_r:
            best_r = r
            best_i = i
    return (best_r if np.isfinite(best_r) else float("nan")), best_i


def profile_correlation_table(
    X: np.ndarray,
    y: np.ndarray,
    mappings: Sequence[np.ndarray],
    perms: Sequence[Sequence[int]] | None = None,
) -> pd.DataFrame:
    """Pairwise as-labelled and best-relabelled deformation-profile r."""
    deltas = deformation_deltas(X, y)
    labels = list(deltas)
    rows = []
    identity = 0
    for i, mapped in enumerate(mappings):
        if np.array_equal(mapped, np.arange(mapped.size)):
            identity = i
            break
    for a, b in combinations(labels, 2):
        r_id = float(np.corrcoef(deltas[a], deltas[b])[0, 1])
        r_best, idx = best_profile_r(deltas[a], deltas[b], mappings)
        cycles = perm_cycles(perms[idx]) if perms and idx < len(perms) else ""
        rows.append(
            {
                "cluster_a": a,
                "cluster_b": b,
                "n_automorphisms": len(mappings),
                "r_as_labelled": r_id,
                "best_relabelling": cycles,
                "r_best_relabelling": r_best,
            }
        )
    _ = identity
    return pd.DataFrame(rows)


def hungarian_match(contingency: np.ndarray) -> np.ndarray:
    """Map columns of a contingency table onto rows (assignment of pred → true)."""
    from scipy.optimize import linear_sum_assignment

    cost = -np.asarray(contingency, dtype=float)
    n = max(cost.shape)
    square = np.zeros((n, n), dtype=float)
    square[: cost.shape[0], : cost.shape[1]] = cost
    _r, c = linear_sum_assignment(square)
    mapping = np.full(cost.shape[1], -1, dtype=int)
    for pred, true in zip(c, _r):
        if pred < cost.shape[1] and true < cost.shape[0]:
            mapping[pred] = true
    return mapping


def refit_ward_labels(X: np.ndarray, k: int) -> np.ndarray:
    """Standardize *X* and cut Ward at *k*."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=float)
    if X.shape[0] < 2:
        return np.zeros(X.shape[0], dtype=int)
    scaled = StandardScaler().fit_transform(X)
    scaled = np.nan_to_num(scaled, nan=0.0)
    z = linkage(scaled, method="ward")
    k_eff = max(1, min(int(k), X.shape[0]))
    return (fcluster(z, t=k_eff, criterion="maxclust") - 1).astype(int)


def nearest_centroid_labels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Assign test rows to the nearest training-cluster centroid."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    tr = np.nan_to_num(scaler.transform(X_train), nan=0.0)
    te = np.nan_to_num(scaler.transform(X_test), nan=0.0)
    labels = np.array(sorted(set(int(c) for c in y_train)))
    cents = np.stack([tr[y_train == c].mean(axis=0) for c in labels])
    d2 = ((te[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
    return labels[np.argmin(d2, axis=1)]


def replica_holdout_folds(
    replica_ids: Sequence,
    *,
    n_splits: int = 5,
    n_repeats: int = 20,
    random_state: int = 0,
) -> list[np.ndarray]:
    """Repeated replica partitions. Each item is a boolean train mask over segments."""
    reps = np.asarray(replica_ids)
    uniq = np.array(sorted(set(reps.tolist())))
    rng = np.random.default_rng(random_state)
    folds: list[np.ndarray] = []
    n_splits = max(2, min(int(n_splits), max(2, uniq.size)))
    for _ in range(int(n_repeats)):
        order = rng.permutation(uniq.size)
        chunks = np.array_split(order, n_splits)
        for chunk in chunks:
            held = set(uniq[chunk].tolist())
            folds.append(~np.isin(reps, list(held)))
    return folds


def coverage_at_radius(
    train_X: np.ndarray,
    test_X: np.ndarray,
    mappings: Sequence[np.ndarray],
    *,
    q: float = 0.95,
    train_dist: np.ndarray | None = None,
    test_to_train: np.ndarray | None = None,
) -> float:
    """Share of test vectors inside the training 95th-percentile radius of the medoid.

    Pass *train_dist* and *test_to_train* (rows of the precomputed distance
    matrix) to avoid recomputing pairwise RMSDs.
    """
    if train_X is not None and train_X.shape[0] == 0:
        return float("nan")
    n_train = train_dist.shape[0] if train_dist is not None else int(train_X.shape[0])
    n_test = test_to_train.shape[0] if test_to_train is not None else int(test_X.shape[0])
    if n_train == 0 or n_test == 0:
        return float("nan")
    if train_dist is None:
        train_dist, _ = pairwise_symmetry_distances(train_X, mappings)
    med = medoid_index(train_dist)
    radius = float(np.quantile(train_dist[med], q)) if n_train > 1 else 0.0
    if test_to_train is not None:
        hits = int(np.sum(test_to_train[:, med] <= radius))
        return float(hits / n_test)
    hits = 0
    for row in test_X:
        d, _ = symmetry_adjusted_distance(train_X[med], row, mappings)
        if d <= radius:
            hits += 1
    return float(hits / n_test)


def energy_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Energy distance between two samples of vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")

    def _mean_dist(x, y):
        d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(axis=2)
        return float(np.mean(np.sqrt(d2)))

    return 2.0 * _mean_dist(a, b) - _mean_dist(a, a) - _mean_dist(b, b)


def replica_block_energy_p(
    X: np.ndarray,
    labels: np.ndarray,
    replica_ids: np.ndarray,
    *,
    n_perm: int = 199,
    random_state: int = 0,
) -> tuple[float, float]:
    """Energy distance between two labelled groups with replica-block permutations."""
    labs = np.asarray(labels)
    uniq = [int(x) for x in sorted(set(labs.tolist()))]
    if len(uniq) != 2:
        return float("nan"), float("nan")
    a, b = uniq
    obs = energy_distance(X[labs == a], X[labs == b])
    reps = np.asarray(replica_ids)
    uniq_reps = np.array(sorted(set(reps.tolist())))
    # majority label per replica
    majority = {}
    for r in uniq_reps:
        sub = labs[reps == r]
        if sub.size == 0:
            continue
        vals, counts = np.unique(sub, return_counts=True)
        majority[r] = int(vals[np.argmax(counts)])
    rng = np.random.default_rng(random_state)
    count = 0
    keys = list(majority)
    for _ in range(int(n_perm)):
        shuffled = rng.permutation([majority[k] for k in keys])
        new = labs.copy()
        for r, lab in zip(keys, shuffled):
            new[reps == r] = lab
        if len(set(new.tolist())) < 2:
            continue
        d = energy_distance(X[new == a], X[new == b])
        if d >= obs:
            count += 1
    p = (1 + count) / (n_perm + 1)
    return float(obs), float(p)


def balanced_accuracy_grouped(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 0,
) -> float:
    """Nearest-centroid balanced accuracy with group-aware folds."""
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import GroupKFold

    y = np.asarray(y)
    if len(set(y.tolist())) < 2 or X.shape[0] < 4:
        return float("nan")
    n_splits = max(2, min(int(n_splits), int(pd.Series(groups).nunique()), X.shape[0]))
    try:
        cv = GroupKFold(n_splits=n_splits)
        scores = []
        for tr, te in cv.split(X, y, groups):
            pred = nearest_centroid_labels(X[tr], y[tr], X[te])
            scores.append(float(balanced_accuracy_score(y[te], pred)))
        return float(np.mean(scores)) if scores else float("nan")
    except ValueError:
        return float("nan")


def label_permutation_null(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_perm: int = 99,
    random_state: int = 0,
) -> float:
    """Mean grouped balanced accuracy after shuffling *y*."""
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)
    scores = []
    for _ in range(int(n_perm)):
        shuffled = rng.permutation(y)
        scores.append(balanced_accuracy_grouped(X, shuffled, groups, random_state=random_state))
    return float(np.nanmean(scores)) if scores else float("nan")


def merge_ratio(family_p95: float, member_p95: float, *, max_ratio: float = 1.2) -> dict[str, float | bool]:
    ratio = (
        float(family_p95 / member_p95)
        if np.isfinite(family_p95) and np.isfinite(member_p95) and member_p95 > 0
        else float("nan")
    )
    return {"p95_ratio": ratio, "merge_supported": bool(np.isfinite(ratio) and ratio <= max_ratio)}


def sensitivity_family_table(
    labels: Sequence[int],
    pairs: Sequence[tuple[int, int, float]],
    *,
    thresholds: Sequence[float] = (0.80, 0.85, 0.90, 0.95),
    linkages: Sequence[str] = ("single", "complete"),
) -> pd.DataFrame:
    """Structure counts under copy-r and linkage variants."""
    rows = []
    for linkage in linkages:
        for thr in thresholds:
            fam = family_ids(labels, pairs, threshold=float(thr), linkage=str(linkage))
            groups: dict[int, list[int]] = {}
            for lab, head in fam.items():
                groups.setdefault(head, []).append(lab)
            rows.append(
                {
                    "linkage": linkage,
                    "copy_r": float(thr),
                    "n_structures": len(groups),
                    "families": ";".join(
                        " ".join(str(x) for x in sorted(m))
                        for _, m in sorted(groups.items())
                    ),
                }
            )
    return pd.DataFrame(rows)


def tangled_pairs_from_correlations(
    labels: Sequence[int],
    pairs: Sequence[tuple[int, int, float]],
    *,
    threshold: float = COPY_R,
    r_min: float = 0.70,
    linkage: str = "single",
) -> pd.DataFrame:
    """Pairs whose merge status is not settled by the catalog rule alone.

    ``near_miss``: different families, but best-relabelled r is at least
    ``r_min`` (a possible missed merge, or two deformations that still look
    related). ``chain_weak``: same family, but the direct r is below
    ``threshold`` (joined only through neighbours).
    """
    families = family_ids(labels, pairs, threshold=float(threshold), linkage=str(linkage))
    rows = []
    seen: set[tuple[int, int]] = set()
    for a, b, score in pairs:
        left, right = int(a), int(b)
        if left == right:
            continue
        if left > right:
            left, right = right, left
        key = (left, right)
        if key in seen:
            continue
        seen.add(key)
        r = float(score)
        same = families[left] == families[right]
        if same and r < float(threshold):
            reason = "chain_weak"
        elif (not same) and r >= float(r_min):
            reason = "near_miss"
        else:
            continue
        rows.append(
            {
                "cluster_a": left,
                "cluster_b": right,
                "family_a": int(families[left]),
                "family_b": int(families[right]),
                "r_whole": r,
                "tangle_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def core_split(
    X: np.ndarray,
    y: np.ndarray,
    clusters: Sequence[int] | None = None,
) -> np.ndarray:
    """Label-free two-way split of each cluster into a core and a weak part.

    The segments of each cluster are cut in two by Ward on segment means
    standardized over all rows of *X* (the same scaling for every cluster).
    The part whose mean lies farther from the grand mean, in standardized
    units, is ``"core"``; the other is ``"weak"``. Rows outside *clusters*
    get ``""``. A cluster with fewer than two segments is all core.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd <= 0)] = 1.0
    Z = np.nan_to_num((X - mu) / sd, nan=0.0)
    if clusters is None:
        clusters = sorted(set(int(c) for c in y))
    out = np.full(y.shape[0], "", dtype=object)
    for c in clusters:
        idx = np.flatnonzero(y == c)
        if idx.size < 2:
            out[idx] = "core"
            continue
        sub = fcluster(linkage(Z[idx], method="ward"), t=2, criterion="maxclust")
        norms = {lab: float(np.linalg.norm(Z[idx[sub == lab]].mean(axis=0))) for lab in set(sub)}
        core = max(norms, key=norms.get)
        out[idx] = np.where(sub == core, "core", "weak")
    return out


def dilution_verdict(
    r_whole: float,
    r_core: float | None,
    *,
    copy_r: float = COPY_R,
    n_core_a: int = 0,
    n_core_b: int = 0,
    min_segments: int = 5,
    tangle_reason: str | None = None,
) -> str:
    """Read a tangled pair after the core / weak split.

    ``dilution_missed_merge``: different families; weak segments were what
    kept the pair below the catalog line. ``chain_clears``: already one
    family by single linkage; the weak direct link meets the line once
    weak segments are dropped. ``distinct_deformations``: the cores still
    sit below the line. ``orientation_copies``: the whole clusters already
    meet the line. ``too_few_core``: at least one side has too few core
    segments to trust the core r.
    """
    whole = float(r_whole) if r_whole is not None else float("nan")
    core = float(r_core) if r_core is not None else float("nan")
    if int(n_core_a) < int(min_segments) or int(n_core_b) < int(min_segments) or not np.isfinite(core):
        if np.isfinite(whole) and whole >= float(copy_r):
            return "orientation_copies"
        return "too_few_core"
    if np.isfinite(whole) and whole >= float(copy_r):
        return "orientation_copies" if core >= float(copy_r) else "merge_depends_on_weak"
    if core >= float(copy_r):
        if str(tangle_reason or "") == "chain_weak":
            return "chain_clears"
        return "dilution_missed_merge"
    return "distinct_deformations"


def sample_frame_indices(
    n_frames: int,
    *,
    stride: int,
    cap: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Take every *stride*-th index, then downsample to *cap*."""
    idx = np.arange(0, n_frames, max(int(stride), 1))
    if idx.size <= cap:
        return idx
    rng = rng or np.random.default_rng(0)
    return np.sort(rng.choice(idx, size=int(cap), replace=False))


def reorient_rows(
    X: np.ndarray,
    template: np.ndarray,
    mappings: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Put each row of *X* into the template orientation. Returns ``(Y, best_g)``."""
    X = np.asarray(X, dtype=float)
    out = np.empty_like(X)
    choice = np.zeros(X.shape[0], dtype=int)
    for i, row in enumerate(X):
        _d, g = symmetry_adjusted_distance(template, row, mappings)
        out[i] = row[mappings[g]] if mappings else row
        choice[i] = g
    return out, choice


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    if not np.isfinite(pooled) or pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def sign_agreement(train_d: Sequence[float], test_d: Sequence[float]) -> float:
    pairs = [
        (float(a), float(b))
        for a, b in zip(train_d, test_d)
        if np.isfinite(a) and np.isfinite(b) and a != 0
    ]
    if not pairs:
        return float("nan")
    return float(np.mean([np.sign(a) == np.sign(b) for a, b in pairs]))


def wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wasserstein_distance

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


def standardized_median_shift(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    pooled = np.concatenate([a, b])
    sd = float(np.std(pooled))
    if sd == 0:
        return 0.0
    return float((np.median(b) - np.median(a)) / sd)


__all__ = [
    "COPY_R",
    "N_MONOMERS",
    "NEGATIVE_CONTROL_PAIRS",
    "POLE_H_FROM_POLE_M",
    "ashman_d",
    "balanced_accuracy_grouped",
    "best_profile_r",
    "cluster_diameter_rows",
    "cohens_d",
    "coverage_at_radius",
    "deformation_deltas",
    "core_split",
    "dilution_verdict",
    "energy_distance",
    "hungarian_match",
    "kabsch_rmsd",
    "labelled_pairwise_distances",
    "label_permutation_null",
    "mappings_from_perms",
    "medoid_index",
    "merge_ratio",
    "nearest_centroid_labels",
    "overlap_coefficient",
    "pairwise_symmetry_distances",
    "permutation_group",
    "profile_correlation_table",
    "refit_ward_labels",
    "reorient_rows",
    "replica_block_energy_p",
    "replica_holdout_folds",
    "robust_z",
    "sample_frame_indices",
    "sensitivity_family_table",
    "sign_agreement",
    "standardized_median_shift",
    "symmetry_adjusted_distance",
    "tangled_pairs_from_correlations",
    "two_population_test",
    "variance_split",
    "wasserstein_1d",
    "within_dwell_spread",
]
