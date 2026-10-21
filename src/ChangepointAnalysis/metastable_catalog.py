"""Name endpoint clusters and merge orientation copies of one deformation.

Ward clustering keeps monomer labels, so one belt break appears as several
clusters, one per symmetry-related orientation. Two clusters are treated as
copies when the Pearson correlation of their site-pair deformation profiles,
after the best contact-graph automorphism, is at least ``COPY_R``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

import numpy as np

COPY_R = 0.90
_CONTACT_PAIR_RE = re.compile(r"_e\d+_M(\d+)-M(\d+)$")
COMPACT_A_PCT = 90.0
MOSTLY_COMPACT_A_PCT = 50.0
INTACT_BELT_ELONGATED_PCT = 15.0
CLEAN_EDGE_PCT = 50.0
OPEN_POLE_PCT = 40.0


def family_ids(
    labels: Sequence[int],
    pairs: Sequence[tuple[int, int, float]],
    threshold: float = COPY_R,
) -> dict[int, int]:
    """Map each cluster to the smallest label in its orientation family.

    An edge joins two clusters when their best-relabelled profile correlation
    is at least ``threshold``. Families are the connected components
    (single linkage).
    """
    parent = {int(label): int(label) for label in labels}

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
