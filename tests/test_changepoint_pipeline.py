"""Regression tests for the ChangepointAnalysis package."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ChangepointAnalysis import (
    GSA_COLS,
    ChangepointConfig,
    ChangepointPipeline,
    SegmentClusteringConfig,
    detect_cohort_changepoints,
)
from src.ChangepointAnalysis.feature_groups import (
    IODINE_COLS,
    NA_WATER_COLS,
    SUMMARY_COLS,
)
from src.utils.metastable_states import DEFAULT_GSA_SIGNAL_COLUMNS

BREAKPOINT_COLS = {
    "traj_id",
    "group",
    "breakpoint_idx",
    "signal_index",
    "frame",
    "time_ps",
    "n_cols_used",
}

SEGMENT_CORE_COLS = {
    "traj_id",
    "group",
    "segment_id",
    "start_frame",
    "end_frame",
    "start_ps",
    "end_ps",
    "n_frames",
    "n_cols_used",
}

COMPARISON_COLS = {
    "traj_id",
    "group_a",
    "group_b",
    "tolerance_frames",
    "n_bkps_a",
    "n_bkps_b",
    "n_shared",
    "jaccard",
    "mean_timing_offset_frames",
    "mean_timing_offset_ps",
}

PLANTED_SHIFT = 100
N_FRAMES = 200


def _make_synthetic_features(traj_id: str, n_frames: int = N_FRAMES, seed: int = 0) -> pd.DataFrame:
    """Build a features CSV with a planted regime shift at PLANTED_SHIFT."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames, dtype=float)
    regime = (t >= PLANTED_SHIFT).astype(float)

    rows: dict[str, np.ndarray] = {
        "traj_id": np.full(n_frames, traj_id),
        "frame": np.arange(n_frames, dtype=int),
        "time_ps": t * 10.0,
    }

    # GSA geometry: clear step at the planted frame
    for col in GSA_COLS:
        base = rng.normal(0.0, 0.05, size=n_frames)
        step = 2.0 + 0.1 * hash(col) % 5
        rows[col] = base + regime * step

    # Iodine guest: correlated step
    for col in IODINE_COLS:
        base = rng.normal(0.0, 0.05, size=n_frames)
        step = 1.5 + 0.05 * (hash(col) % 7)
        rows[col] = base + regime * step

    # Na/water: correlated step
    for col in NA_WATER_COLS:
        base = rng.normal(0.0, 0.05, size=n_frames)
        step = 1.0 + 0.05 * (hash(col) % 3)
        rows[col] = base + regime * step

    return pd.DataFrame(rows)


@pytest.fixture()
def features_dir(tmp_path: Path) -> Path:
    feat_dir = tmp_path / "gsa_features"
    feat_dir.mkdir()
    for i, traj in enumerate(("synth_A", "synth_B")):
        df = _make_synthetic_features(traj, seed=10 + i)
        df.to_csv(feat_dir / f"{traj}_gsa_features.csv", index=False)
    return feat_dir


def test_default_signal_columns_covered_by_feature_groups() -> None:
    """DEFAULT_GSA_SIGNAL_COLUMNS must appear in the union of group column lists."""
    covered = set(GSA_COLS) | set(IODINE_COLS) | set(NA_WATER_COLS)
    missing = set(DEFAULT_GSA_SIGNAL_COLUMNS) - covered
    assert not missing, f"Uncovered DEFAULT_GSA_SIGNAL_COLUMNS: {missing}"
    # Structural geometry columns (excluding solvent) live in GSA_COLS.
    structural = set(DEFAULT_GSA_SIGNAL_COLUMNS) - {"cavity_water_count"}
    assert structural <= set(GSA_COLS)


def test_summary_cols_cover_all_groups() -> None:
    assert set(SUMMARY_COLS.keys()) >= {"gsa", "iodine", "na_water", "combined", "endpoint"}


def test_detect_cohort_output_schemas(features_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "changepoints"
    config = ChangepointConfig(
        penalty=3.0,
        min_size=10,
        jump=5,
        tolerance_frames=20,
        groups=("gsa", "iodine", "na_water", "combined"),
    )
    csv_paths = sorted(features_dir.glob("*_gsa_features.csv"))
    tables = detect_cohort_changepoints(csv_paths, config, output_dir=out)

    assert (out / "all_breakpoints.csv").exists()
    assert (out / "all_segment_stats.csv").exists()
    assert (out / "changepoint_timing_comparison.csv").exists()
    assert (out / "synth_A_breakpoints.csv").exists()
    assert (out / "synth_B_segment_stats.csv").exists()

    bkp = pd.read_csv(out / "all_breakpoints.csv")
    seg = pd.read_csv(out / "all_segment_stats.csv")
    cmp = pd.read_csv(out / "changepoint_timing_comparison.csv")

    assert BREAKPOINT_COLS <= set(bkp.columns)
    assert SEGMENT_CORE_COLS <= set(seg.columns)
    assert COMPARISON_COLS <= set(cmp.columns)

    # At least one breakpoint near the planted shift for the gsa group
    gsa_bkps = bkp[bkp["group"] == "gsa"]
    assert not gsa_bkps.empty
    nearest = (gsa_bkps["signal_index"] - PLANTED_SHIFT).abs().min()
    assert nearest <= 25, f"No gsa breakpoint near planted shift; nearest={nearest}"

    assert len(tables.breakpoints) == len(bkp)
    assert len(tables.segment_stats) == len(seg)


def test_pipeline_run_all_filenames(features_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "pipeline_out"
    pipe = ChangepointPipeline(
        features_dir,
        out,
        detection=ChangepointConfig(penalty=3.0, min_size=10, jump=5),
        clustering=SegmentClusteringConfig(
            n_clusters=2,
            save_all_k=False,
            skip_pca=True,
            silhouette_k_max=3,
        ),
    )
    artifacts = pipe.run_all(with_sweep=False)

    required_names = {
        "all_breakpoints.csv",
        "all_segment_stats.csv",
        "changepoint_timing_comparison.csv",
    }
    for name in required_names:
        assert (out / name).exists(), f"missing {name}"
        assert name in artifacts or any(Path(p).name == name for p in artifacts.values())

    assert (out / "cohort_timing_summary.csv").exists()
    assert (out / "breakpoints_per_trajectory.csv").exists()
    assert (out / "plots" / "cohort_jaccard_heatmap.png").exists()
    assert (out / "clusters" / "all_segments_clustered.csv").exists()


def test_pipeline_discovers_feature_csvs(features_dir: Path, tmp_path: Path) -> None:
    pipe = ChangepointPipeline(features_dir, tmp_path / "out")
    found = pipe.discover_feature_csvs()
    assert len(found) == 2
    assert all(p.name.endswith("_gsa_features.csv") for p in found)


# ── Endpoint-site feature group ─────────────────────────────────────────────


def _make_synthetic_endpoint_features(
    traj_id: str, n_frames: int = N_FRAMES, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames, dtype=float)
    regime = (t >= PLANTED_SHIFT).astype(float)
    rows: dict[str, np.ndarray] = {
        "traj_id": np.full(n_frames, traj_id),
        "frame": np.arange(n_frames, dtype=int),
        "time_ps": t * 10.0,
        "endpoint_dist_mean": rng.normal(5.0, 0.05, n_frames) + regime * 2.0,
        "endpoint_dist_min": rng.normal(4.5, 0.05, n_frames) + regime * 1.5,
        "endpoint_dist_max": rng.normal(6.0, 0.05, n_frames) + regime * 2.5,
        "endpoint_dist_std": rng.normal(0.3, 0.02, n_frames) + regime * 0.4,
    }
    for i in range(6):
        for j in range(i + 1, 6):
            base = rng.normal(5.0, 0.05, n_frames)
            rows[f"endpoint_dist_{i}_{j}_mean"] = base + regime * 1.8
            rows[f"endpoint_dist_{i}_{j}_min"] = base - 0.3 + regime * 1.5
            rows[f"endpoint_dist_{i}_{j}_max"] = base + 0.3 + regime * 2.2
    return pd.DataFrame(rows)


@pytest.fixture()
def endpoint_features_dir(tmp_path: Path) -> Path:
    feat_dir = tmp_path / "endpoint_features"
    feat_dir.mkdir()
    for i, traj in enumerate(("ep_A", "ep_B")):
        df = _make_synthetic_endpoint_features(traj, seed=20 + i)
        df.to_csv(feat_dir / f"{traj}_endpoint_features.csv", index=False)
    return feat_dir


def test_all_pairs_to_metrics_df_basic() -> None:
    from src.ChangepointAnalysis.endpoint_features import all_pairs_to_metrics_df

    n_frames, n_a, n_b = 5, 2, 3
    pair = np.zeros((n_frames, n_a, n_b))
    for f in range(n_frames):
        pair[f] = f + np.arange(n_a)[:, None] + 0.1 * np.arange(n_b)[None, :]
    all_pairs = {(0, 1): pair, (1, 0): np.transpose(pair, (0, 2, 1))}
    df = all_pairs_to_metrics_df(
        all_pairs, n_res=2, n_frames=n_frames, include_site_pairs=True
    )
    assert len(df) == n_frames
    assert "endpoint_dist_0_1_mean" in df.columns
    assert "endpoint_dist_mean" in df.columns
    assert "endpoint_dist_0s0_1s0" in df.columns
    expected_mean = float(np.mean(pair[0]))
    assert abs(df.loc[0, "endpoint_dist_0_1_mean"] - expected_mean) < 1e-9


def test_find_endpoint_sites_biphenyl_and_naphthalene() -> None:
    pytest.importorskip("rdkit")
    from rdkit import Chem

    from src.EndpointAnalyzer.endpoints_finder import EndpointsFinder

    finder = EndpointsFinder(
        ring_min_gap_deg=None,
        ring_max_per_ring=10,
        step_back_from_terminals=True,
        extend_to_ring_atoms=False,
    )

    biphenyl = Chem.MolFromSmiles("c1ccccc1-c2ccccc2")
    assert biphenyl is not None
    sites = finder.find_endpoint_sites(biphenyl)
    ring_sites = [s for s in sites if len(s) > 1]
    assert any(len(s) >= 6 for s in ring_sites)
    assert sites == sorted(sites, key=lambda s: (min(s), len(s), s))

    naph = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    assert naph is not None
    systems = EndpointsFinder._ring_systems(naph)
    assert len(systems) == 1
    assert len(systems[0]) == 10

    info = finder.find_endpoint_site_info(biphenyl)
    assert all("kind" in row and "n_atoms" in row for row in info)


def test_site_centroid_math() -> None:
    """Unweighted mean of a regular hexagon equals its geometric center."""
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    coords = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(6)])
    centroid = coords.mean(axis=0)
    assert np.allclose(centroid, [0.0, 0.0, 0.0], atol=1e-12)


def test_endpoint_group_pipeline(endpoint_features_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "endpoint_cp"
    pipe = ChangepointPipeline(
        endpoint_features_dir,
        out,
        detection=ChangepointConfig(
            groups=("endpoint",),
            penalty=3.0,
            min_size=10,
            jump=5,
        ),
        clustering=SegmentClusteringConfig(
            groups=("endpoint",),
            n_clusters=2,
            save_all_k=False,
            skip_pca=True,
            silhouette_k_max=3,
        ),
        features_suffix="_endpoint_features.csv",
    )
    artifacts = pipe.run_all(with_sweep=False)

    assert (out / "all_breakpoints.csv").exists()
    assert (out / "all_segment_stats.csv").exists()
    bkp = pd.read_csv(out / "all_breakpoints.csv")
    assert set(bkp["group"].unique()) == {"endpoint"}
    nearest = (bkp["signal_index"] - PLANTED_SHIFT).abs().min()
    assert nearest <= 25, f"No endpoint breakpoint near planted shift; nearest={nearest}"
    assert (out / "clusters" / "all_segments_clustered.csv").exists()
    assert "all_breakpoints.csv" in artifacts or (out / "all_breakpoints.csv").exists()


def test_resolve_group_columns_endpoint(endpoint_features_dir: Path) -> None:
    from src.ChangepointAnalysis.feature_groups import resolve_group_columns

    df = pd.read_csv(next(endpoint_features_dir.glob("*_endpoint_features.csv")))
    cols = resolve_group_columns(df, "endpoint")
    assert cols
    assert all(c.startswith("endpoint_dist_") for c in cols)
    assert "traj_id" not in cols
