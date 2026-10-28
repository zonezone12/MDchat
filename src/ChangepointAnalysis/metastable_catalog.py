"""Name endpoint clusters and merge orientation copies of one deformation.

Ward clustering keeps monomer labels, so one belt break appears as several
clusters, one per symmetry-related orientation. Two clusters are treated as
copies when the Pearson correlation of their site-pair deformation profiles,
after the best contact-graph automorphism, is at least ``COPY_R``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations, permutations

import numpy as np

COPY_R = 0.95
_CONTACT_PAIR_RE = re.compile(r"_e\d+_M(\d+)-M(\d+)$")
COMPACT_A_PCT = 90.0
MOSTLY_COMPACT_A_PCT = 50.0
INTACT_BELT_ELONGATED_PCT = 15.0
CLEAN_EDGE_PCT = 50.0
OPEN_POLE_PCT = 40.0
N_MONOMERS = 6

# Directed pole sandwiches and equatorial d1 edges of the six-monomer cage.
DEFAULT_POLE_EDGES: frozenset[tuple[int, int]] = frozenset(
    {(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)}
)
DEFAULT_D1_EDGES: frozenset[tuple[int, int]] = frozenset(
    {(0, 4), (1, 5), (2, 3), (3, 1), (4, 2), (5, 0)}
)
CUBE_ADJACENCY: frozenset[frozenset[int]] = frozenset(
    frozenset(pair) for pair in (DEFAULT_POLE_EDGES | DEFAULT_D1_EDGES)
)
# Octahedron vertices matching the contact graph (opposite = i ↔ i+3).
OCTAHEDRON_COORDS = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=float,
)
INVERSION_PERM: tuple[int, ...] = (3, 4, 5, 0, 1, 2)
BELT_SIX_CYCLE: tuple[int, ...] = (4, 5, 3, 1, 2, 0)  # (0 4 2 3 1 5)


def family_ids(
    labels: Sequence[int],
    pairs: Sequence[tuple[int, int, float]],
    threshold: float = COPY_R,
    *,
    linkage: str = "single",
) -> dict[int, int]:
    """Map each cluster to the smallest label in its orientation family.

    An edge joins two clusters when their best-relabelled profile correlation
    is at least ``threshold``. ``linkage='single'`` takes connected components.
    ``linkage='complete'`` merges two groups only when every cross pair meets
    the threshold (the farthest pair in the family stays above it).
    """
    method = str(linkage).strip().lower()
    if method not in {"single", "complete"}:
        raise ValueError("linkage must be 'single' or 'complete'")

    ids = [int(label) for label in labels]
    if method == "single":
        parent = {label: label for label in ids}

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        for a, b, score in pairs:
            if float(score) < threshold:
                continue
            a, b = int(a), int(b)
            if a not in parent or b not in parent:
                continue
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            keep, drop = (ra, rb) if ra < rb else (rb, ra)
            parent[drop] = keep
        return {label: find(label) for label in parent}

    r_of: dict[tuple[int, int], float] = {}
    for a, b, score in pairs:
        a, b = int(a), int(b)
        r_of[(a, b)] = float(score)
        r_of[(b, a)] = float(score)
    groups: dict[int, set[int]] = {label: {label} for label in ids}
    while True:
        best: tuple[float, int, int] | None = None
        keys = list(groups)
        for a, b in combinations(keys, 2):
            cross = [
                r_of.get((x, y), float("-inf"))
                for x in groups[a]
                for y in groups[b]
            ]
            min_r = min(cross) if cross else float("-inf")
            if min_r >= threshold and (best is None or min_r > best[0]):
                best = (min_r, a, b)
        if best is None:
            break
        _, a, b = best
        keep, drop = (a, b) if a < b else (b, a)
        groups[keep] |= groups[drop]
        del groups[drop]
    out: dict[int, int] = {}
    for members in groups.values():
        head = min(members)
        for label in members:
            out[label] = head
    return out


def perm_cycles(perm: Sequence[int]) -> str:
    """Cycle notation of a monomer permutation, or ``identity``."""
    seen: set[int] = set()
    parts = []
    for start in range(len(perm)):
        if start in seen or int(perm[start]) == start:
            continue
        cycle = [start]
        seen.add(start)
        nxt = int(perm[start])
        while nxt != start:
            cycle.append(nxt)
            seen.add(nxt)
            nxt = int(perm[nxt])
        parts.append("(" + " ".join(str(m) for m in cycle) + ")")
    return "".join(parts) or "identity"


def contact_automorphisms(
    *edge_maps: Mapping[str, tuple],
) -> list[tuple[int, ...]]:
    """Monomer permutations that map each directed contact set onto itself.

    With no arguments, uses :data:`DEFAULT_POLE_EDGES` and
    :data:`DEFAULT_D1_EDGES`. Each positional argument is a motif-column map
    ``{col: (edge, mon_a, mon_b)}`` as in ``describe_endpoint_clusters``.
    """
    if edge_maps:
        edge_sets = [{(int(a), int(b)) for _, a, b in edges.values()} for edges in edge_maps]
        monomers = sorted({m for edges in edge_sets for pair in edges for m in pair})
    else:
        edge_sets = [set(DEFAULT_POLE_EDGES), set(DEFAULT_D1_EDGES)]
        monomers = list(range(N_MONOMERS))
    found = []
    for image in permutations(monomers):
        perm = dict(zip(monomers, image))
        if all({(perm[a], perm[b]) for a, b in edges} == edges for edges in edge_sets):
            found.append(tuple(perm[m] for m in monomers))
    return found


def adjacency_automorphisms(n: int = N_MONOMERS) -> list[tuple[int, ...]]:
    """The 48 face-adjacency automorphisms of the six-monomer cage."""
    if int(n) != N_MONOMERS:
        raise ValueError("adjacency_automorphisms is defined for six monomers")
    found = []
    for image in permutations(range(N_MONOMERS)):
        mapped = {frozenset((image[a], image[b])) for a, b in CUBE_ADJACENCY}
        if mapped == CUBE_ADJACENCY:
            found.append(tuple(image))
    return found


def is_proper_permutation(perm: Sequence[int]) -> bool:
    """True when *perm* is realized by a rotation of the octahedron (det +1)."""
    p = np.asarray(perm, dtype=int)
    coords = OCTAHEDRON_COORDS
    target = coords[p]
    h = coords.T @ target
    u, _s, vt = np.linalg.svd(h)
    rot = vt.T @ u.T
    return float(np.linalg.det(rot)) >= 0.0


def rotation_contact_automorphisms(
    *edge_maps: Mapping[str, tuple],
) -> list[tuple[int, ...]]:
    """Contact-graph automorphisms that are proper rotations."""
    return [perm for perm in contact_automorphisms(*edge_maps) if is_proper_permutation(perm)]


def _map_site(
    site_map: Sequence[int] | Mapping | np.ndarray | None,
    monomer: int,
    site: int,
) -> int:
    if site_map is None:
        return int(site)
    if isinstance(site_map, Mapping):
        if monomer in site_map:
            mapped = site_map[monomer]
            if isinstance(mapped, Mapping):
                return int(mapped.get(site, site))
            return int(np.asarray(mapped)[site])
        if site in site_map:
            return int(site_map[site])
        return int(site)
    arr = np.asarray(site_map)
    if arr.ndim == 1:
        return int(arr[site])
    return int(arr[monomer, site])


def site_pair_column_permutation(
    specs: Sequence[tuple],
    perm: Sequence[int],
    *,
    site_map: Sequence[int] | Mapping | np.ndarray | None = None,
) -> np.ndarray:
    """Column indices that apply *perm* (and optional *site_map*) to *specs*.

    ``specs`` are ``(name, mon_i, site_a, mon_j, site_b)``. Index ``k`` of the
    result is the column of the spec that *perm* sends onto spec ``k``, so
    ``y[mapped]`` is ``y`` after the relabelling. Missing images are ``-1``.
    """
    col_of: dict[tuple[int, int, int, int], int] = {}
    parsed: list[tuple[int, int, int, int]] = []
    for col, spec in enumerate(specs):
        if len(spec) >= 5:
            _name, i, a, j, b = spec[:5]
        else:
            raise ValueError("site-pair specs must be (name, i, a, j, b)")
        i, a, j, b = int(i), int(a), int(j), int(b)
        parsed.append((i, a, j, b))
        col_of[(i, a, j, b)] = col
        col_of[(j, b, i, a)] = col
    p = [int(x) for x in perm]
    mapped = np.full(len(parsed), -1, dtype=int)
    for k, (i, a, j, b) in enumerate(parsed):
        ia = _map_site(site_map, i, a)
        jb = _map_site(site_map, j, b)
        mapped[k] = col_of.get((p[i], ia, p[j], jb), -1)
    return mapped


def _ranked(row: Mapping, prefix: str) -> list[tuple[str, float]]:
    cols = [str(col) for col in row if str(col).startswith(prefix)]
    scored = []
    for col in cols:
        rest = col[len(prefix):]
        monomers = rest.split("_", 1)[1] if "_" in rest else rest
        scored.append((monomers.replace("-", "–"), float(row[col])))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def structure_name(row: Mapping, *, single_replica: bool = False) -> str:
    """Short chemical name from a motif-profile row."""
    elongated = float(row["pct_any_d1_elongated"])
    pct_a = float(row["pct_A"])
    edges = _ranked(row, "pct_d1_elongated_")
    poles = _ranked(row, "pct_pi_open_")
    edge, edge_pct = edges[0] if edges else ("", float("nan"))
    second, second_pct = edges[1] if len(edges) > 1 else ("", 0.0)
    pole, pole_pct = poles[0] if poles else ("", float("nan"))
    share = edge_pct / elongated if elongated else 0.0
    if single_replica:
        return f"loosened single replica, edge {edge}"
    if elongated < INTACT_BELT_ELONGATED_PCT and pct_a >= COMPACT_A_PCT:
        return "compact cage"
    if elongated < INTACT_BELT_ELONGATED_PCT and pct_a >= MOSTLY_COMPACT_A_PCT:
        return "mostly compact"
    if elongated < INTACT_BELT_ELONGATED_PCT:
        if pole_pct >= OPEN_POLE_PCT:
            return f"intact belt, {pole} often open"
        return "intact belt"
    split = second_pct >= 30.0 and second_pct >= 0.7 * edge_pct
    clean = edge_pct >= CLEAN_EDGE_PCT or (share >= 0.80 and edge_pct >= 40.0)
    axis = elongated >= 80.0 and edge_pct >= 40.0 and split
    if clean or axis:
        kind = "belt broken" if elongated >= 70.0 or edge_pct >= CLEAN_EDGE_PCT else "partly belt broken"
        where = f"{edge} or {second}" if split else edge
        return f"{kind} along {where}"
    return f"mixed belt, largest edge {edge}"


def belt_label(edges: Sequence[str]) -> str:
    """Frame label from the elongated equatorial edges, independent of cluster k.

    ``edges`` is the monomer pairs (``M0-M4``) elongated on that frame, in a
    stable order. Zero edges is an intact belt. Two edges are named. Three or
    more stay one bin so rare combinations do not become their own states.
    """
    named = [str(edge) for edge in edges]
    if not named:
        return "intact belt"
    if len(named) == 1:
        return f"edge {named[0]}"
    if len(named) == 2:
        return f"edges {named[0]}+{named[1]}"
    return f"{len(named)} elongated edges"


def belt_kind(belt: str) -> str:
    """Coarse bin of a :func:`belt_label`: ``intact``, ``one_edge`` or ``two_plus_edges``."""
    if belt == "intact belt":
        return "intact"
    return "one_edge" if belt.startswith("edge ") else "two_plus_edges"


def contact_pairs(columns: Iterable[str], prefix: str) -> set[frozenset]:
    """Monomer pairs named by motif columns such as ``pct_pi_open_e0_M0-M1``.

    With ``prefix="pct_pi_open_"`` these are the same-pole pairs; with
    ``prefix="pct_d1_elongated_"`` the equatorial belt pairs.
    """
    pairs = set()
    for col in columns:
        name = str(col)
        if name.startswith(prefix):
            match = _CONTACT_PAIR_RE.search(name)
            if match:
                pairs.add(frozenset((int(match.group(1)), int(match.group(2)))))
    return pairs


def pair_relation(i: int, j: int, pole: set[frozenset], belt: set[frozenset]) -> str:
    """Relation of two monomers: ``same pole``, ``equatorial`` or ``opposite``.

    Same names and rule as ``describe_endpoint_clusters.py``: a pair that is
    neither a pole contact nor a belt edge is opposite.
    """
    pair = frozenset((int(i), int(j)))
    if len(pair) == 1:
        return "same monomer"
    if pair in pole:
        return "same pole"
    return "equatorial" if pair in belt else "opposite"


def opening_label(n_open: int) -> str:
    """Pole cation–π count. Three or more openings are one bin."""
    n = int(n_open)
    if n >= 3:
        return "3+ poles open"
    if n < 0:
        return "unlabeled"
    if n == 1:
        return "1 pole open"
    return f"{n} poles open"


def dwell_stats(
    traj: np.ndarray,
    frame: np.ndarray,
    label: np.ndarray,
    *,
    min_persist: int = 20,
) -> tuple[float, float]:
    """Median run length (ps) and the percent of frames that sit in runs of at least ``min_persist``.

    A run breaks when the trajectory changes, the label changes, or the frame
    index jumps by more than 1. Frames are 1 ps apart.
    """
    traj = np.asarray(traj)
    frame = np.asarray(frame, dtype=float)
    label = np.asarray(label)
    n = len(label)
    if n == 0:
        return float("nan"), float("nan")
    order = np.lexsort((frame, traj))
    traj = traj[order]
    frame = frame[order]
    label = label[order]
    new_run = np.empty(n, dtype=bool)
    new_run[0] = True
    if n > 1:
        new_run[1:] = (
            (traj[1:] != traj[:-1])
            | (label[1:] != label[:-1])
            | ((frame[1:] - frame[:-1]) != 1)
        )
    lengths = np.bincount(np.cumsum(new_run))
    lengths = lengths[lengths > 0]
    persistent = float(lengths[lengths >= min_persist].sum()) / float(lengths.sum()) * 100.0
    return float(np.median(lengths)), persistent
