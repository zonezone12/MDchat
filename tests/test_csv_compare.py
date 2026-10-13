"""Reproduction checks: identical, within tolerance, differs, missing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.utils.csv_compare import compare_csv, compare_csv_trees


def _write(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def test_compare_csv_statuses(tmp_path: Path) -> None:
    ref = _write(tmp_path / "ref.csv", pd.DataFrame({"id": ["a", "b"], "penalty": [4.36145, 5.48888]}))
    same = _write(tmp_path / "same.csv", pd.DataFrame({"id": ["a", "b"], "penalty": [4.36145, 5.48888]}))
    near = _write(tmp_path / "near.csv", pd.DataFrame({"id": ["a", "b"], "penalty": [4.361452, 5.48888]}))
    far = _write(tmp_path / "far.csv", pd.DataFrame({"id": ["a", "b"], "penalty": [4.4, 5.48888]}))
    text = _write(tmp_path / "text.csv", pd.DataFrame({"id": ["a", "c"], "penalty": [4.36145, 5.48888]}))
    short = _write(tmp_path / "short.csv", pd.DataFrame({"id": ["a"], "penalty": [4.36145]}))

    assert compare_csv(ref, same)["status"] == "identical"
    within = compare_csv(ref, near)
    assert within["status"] == "equal_within_tolerance"
    assert within["max_rel_diff"] == pytest.approx(2e-6 / 4.36145, rel=1e-3)
    assert compare_csv(ref, near, rtol=1e-7)["status"] == "differs"
    off = compare_csv(ref, far)
    assert off["status"] == "differs"
    assert off["differing_columns"] == "penalty"
    assert compare_csv(ref, text)["differing_columns"] == "id"
    assert compare_csv(ref, short)["status"] == "differs"
    assert compare_csv(ref, tmp_path / "absent.csv")["status"] == "missing"


def test_compare_csv_trees_lists_each_file(tmp_path: Path) -> None:
    frame = pd.DataFrame({"x": [1.0, float("nan")]})
    _write(tmp_path / "ref" / "d" / "a.csv", frame)
    _write(tmp_path / "new" / "d" / "a.csv", frame)
    _write(tmp_path / "ref" / "d" / "b.csv", frame)
    _write(tmp_path / "new" / "d" / "b.csv", pd.DataFrame({"x": [1.0, 2.0]}))
    out = compare_csv_trees(tmp_path / "ref", tmp_path / "new", ["d/a.csv", "d/b.csv", "d/c.csv"])
    assert list(out["file"]) == ["d/a.csv", "d/b.csv", "d/c.csv"]
    assert list(out["status"]) == ["identical", "differs", "missing"]
