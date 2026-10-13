"""Does a regenerated CSV match the reference copy?

``status`` per file:

* ``identical``: same bytes, ignoring leading and trailing whitespace;
* ``equal_within_tolerance``: same columns and shape, text cells equal, numbers
  within ``atol + rtol * |reference|``;
* ``differs``: anything else;
* ``missing``: one of the two files does not exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def compare_csv(
    reference: Path | str,
    candidate: Path | str,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> dict:
    """Compare two CSV files; ``differing_columns`` names the columns out of tolerance."""
    ref, cand = Path(reference), Path(candidate)
    out = {"status": "missing", "max_abs_diff": np.nan, "max_rel_diff": np.nan, "differing_columns": ""}
    if not ref.is_file() or not cand.is_file():
        return out
    if ref.read_bytes().strip() == cand.read_bytes().strip():
        return {**out, "status": "identical", "max_abs_diff": 0.0, "max_rel_diff": 0.0}
    a, b = pd.read_csv(ref), pd.read_csv(cand)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return {**out, "status": "differs", "differing_columns": f"shape {a.shape} vs {b.shape}"}
    num = [
        c
        for c in a.columns
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])
    ]
    differing = [
        c for c in a.columns if c not in num and not a[c].astype(str).equals(b[c].astype(str))
    ]
    x = a[num].to_numpy(dtype=float)
    y = b[num].to_numpy(dtype=float)
    diff = np.abs(x - y)
    diff[(x == y) | (np.isnan(x) & np.isnan(y))] = 0.0
    diff[np.isnan(diff)] = np.inf
    scale = np.where(np.isfinite(x), np.abs(x), 0.0)
    rel = np.divide(diff, scale, out=np.where(diff > 0, np.inf, 0.0), where=scale > 0)
    over = diff > atol + rtol * scale
    differing += [c for c, bad in zip(num, over.any(axis=0)) if bad]
    return {
        "status": "differs" if differing else "equal_within_tolerance",
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "max_rel_diff": float(rel.max()) if rel.size else 0.0,
        "differing_columns": ",".join(differing),
    }


def compare_csv_trees(
    reference_root: Path | str,
    candidate_root: Path | str,
    rel_paths: Iterable[Path | str],
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> pd.DataFrame:
    """:func:`compare_csv` for each relative path under the two roots."""
    rows = []
    for rel in rel_paths:
        rel = Path(rel)
        result = compare_csv(Path(reference_root) / rel, Path(candidate_root) / rel, rtol=rtol, atol=atol)
        rows.append({"file": rel.as_posix(), **result})
    return pd.DataFrame(rows, columns=["file", "status", "max_abs_diff", "max_rel_diff", "differing_columns"])
