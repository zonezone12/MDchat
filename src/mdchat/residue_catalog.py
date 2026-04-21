"""
Build a residue-group inventory for MDChat after trajectory load.

Each row is one (segid, resname) group with a suggested MDAnalysis selection string.
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import Any


def _norm_segid(raw: str) -> str | None:
    s = str(raw).strip()
    if not s or s in ("0", "SYSTEM"):
        return None
    return s


def _is_contiguous(sorted_ids: list[int]) -> bool:
    if len(sorted_ids) <= 1:
        return True
    for a, b in zip(sorted_ids, sorted_ids[1:]):
        if b != a + 1:
            return False
    return True


def build_residue_catalog(universe: Any) -> list[dict[str, Any]]:
    """Return JSON-friendly rows: segid, resname, n_residues, resid labels, selection."""
    groups: dict[tuple[str | None, str], list[int]] = defaultdict(list)
    for r in universe.residues:
        seg_key = _norm_segid(getattr(r, "segid", "") or "")
        resname = str(r.resname).strip()
        try:
            rid = int(r.resid)
        except (TypeError, ValueError):
            rid = int(str(r.resid).strip())
        groups[(seg_key, resname)].append(rid)

    rows: list[dict[str, Any]] = []
    for (seg_key, resname) in sorted(groups.keys(), key=lambda k: (k[0] or "", k[1])):
        ids = sorted(set(groups[(seg_key, resname)]))
        n = len(ids)
        rmin, rmax = ids[0], ids[-1]
        contiguous = _is_contiguous(ids)

        sel_parts: list[str] = []
        if seg_key is not None:
            sel_parts.append(f"segid {seg_key}")
        sel_parts.append(f"resname {resname}")
        if n == 1:
            sel_parts.append(f"resid {rmin}")
        elif contiguous and n == (rmax - rmin + 1):
            if rmin == rmax:
                sel_parts.append(f"resid {rmin}")
            else:
                sel_parts.append(f"resid {rmin}-{rmax}")
        # else: many non-contiguous — keep group-level resname (+ segid) only

        selection = " and ".join(sel_parts)
        rows.append(
            {
                "segid": seg_key or "",
                "resname": resname,
                "n_residues": n,
                "resid_min": rmin,
                "resid_max": rmax,
                "contiguous": contiguous,
                "selection": selection,
            }
        )
    return rows


def write_residue_catalog_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """Write catalog TSV-friendly CSV for the user session folder."""
    if not rows:
        return
    fieldnames = [
        "segid",
        "resname",
        "n_residues",
        "resid_min",
        "resid_max",
        "contiguous",
        "selection",
    ]
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def format_catalog_for_prompt(
    rows: list[dict[str, Any]],
    max_lines: int = 28,
) -> str:
    """Compact multi-line block for the system prompt / status."""
    if not rows:
        return ""
    lines: list[str] = []
    header = (
        f"{'segid':<8} {'resname':<10} {'n':>5}  "
        f"{'resid':<12}  selection"
    )
    lines.append(header)
    lines.append("-" * len(header))
    shown = 0
    for r in rows:
        if shown >= max_lines:
            lines.append(f"... ({len(rows) - max_lines} more groups; see residue_catalog.csv)")
            break
        seg = (r.get("segid") or "—")[:7]
        rn = str(r.get("resname", ""))[:9]
        n = int(r.get("n_residues", 0))
        rmin, rmax = r.get("resid_min"), r.get("resid_max")
        if rmin == rmax:
            rl = str(rmin)
        else:
            rl = f"{rmin}-{rmax}"
        sel = str(r.get("selection", ""))
        if len(sel) > 52:
            sel = sel[:49] + "..."
        lines.append(f"{seg:<8} {rn:<10} {n:5d}  {rl:<12}  {sel}")
        shown += 1
    return "\n".join(lines)


__all__ = [
    "build_residue_catalog",
    "format_catalog_for_prompt",
    "write_residue_catalog_csv",
]
