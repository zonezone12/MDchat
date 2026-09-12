"""HPC nested trajectory globs (``$TRAJ_DIR/<run>/mdcrd_v``)."""

from __future__ import annotations

from pathlib import Path

from src.ChangepointAnalysis.traj_paths import (
    expand_trajectories,
    trajectory_output_id,
    unique_trajectory_ids,
)


def test_trajectory_output_id_nested_mdcrd_v(tmp_path: Path) -> None:
    run = tmp_path / "109345"
    run.mkdir()
    traj = run / "mdcrd_v"
    traj.write_bytes(b"x")
    assert trajectory_output_id(traj) == "109345_mdcrd_v"


def test_expand_trajectories_nested_mdcrd_v(tmp_path: Path) -> None:
    for run_id in ("109345", "891249"):
        run = tmp_path / run_id
        run.mkdir()
        (run / "mdcrd_v").write_bytes(b"x")
    paths = expand_trajectories([str(tmp_path / "*" / "mdcrd_v")])
    ids = unique_trajectory_ids(sorted(paths, key=lambda p: p.parent.name))
    assert ids == ["109345_mdcrd_v", "891249_mdcrd_v"]


def test_unique_trajectory_ids_disambiguates() -> None:
    paths = [Path("a/mdcrd_v"), Path("a/mdcrd_v")]
    assert unique_trajectory_ids(paths) == ["a_mdcrd_v", "a_mdcrd_v_1"]
