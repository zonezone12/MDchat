"""Regression tests for the ChangepointAnalysis package."""

from __future__ import annotations

import os

# Headless CI / broken Tk installs: force a non-interactive backend before pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

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


# ── Transition attribution ──────────────────────────────────────────────────


def test_parse_site_pair_feature_and_metadata() -> None:
    from src.ChangepointAnalysis.transition_attribution import (
        parse_site_pair_feature,
        resolve_site_pair_metadata,
    )

    assert parse_site_pair_feature("endpoint_dist_0s0_1s4") == (0, 0, 1, 4)
    assert parse_site_pair_feature("endpoint_dist_0_1_mean") is None
    assert parse_site_pair_feature("endpoint_dist_mean") is None

    sites = pd.DataFrame(
        [
            {"monomer": 0, "site_index": 0, "kind": "ring", "atom_ids": "10|11|12", "n_atoms": 3},
            {"monomer": 1, "site_index": 4, "kind": "atom", "atom_ids": "99", "n_atoms": 1},
        ]
    )
    meta = resolve_site_pair_metadata("endpoint_dist_0s0_1s4", sites)
    assert meta["monomer_i"] == 0
    assert meta["site_i"] == 0
    assert meta["kind_i"] == "ring"
    assert meta["atom_ids_i"] == "10|11|12"
    assert meta["monomer_j"] == 1
    assert meta["site_j"] == 4
    assert meta["kind_j"] == "atom"
    assert "M0:ring0" in meta["pair_label"]
    assert "M1:atom4" in meta["pair_label"]


def test_build_transition_events_and_stride_safe_scoring(tmp_path: Path) -> None:
    from src.ChangepointAnalysis.transition_attribution import (
        EndpointTransitionAttributionConfig,
        attribute_endpoint_transitions,
        build_transition_events,
        score_transition_features,
        top_features_per_transition,
    )

    # Strided frames: 0, 10, 20, … so integer row index ≠ frame value.
    frames = np.arange(0, 200, 10, dtype=int)
    n = len(frames)
    # Planted driver opens after frame 90; decoy stays flat; noise has no step.
    rng = np.random.default_rng(0)
    driver = np.where(frames < 100, 5.0, 8.0).astype(float)
    decoy = np.full(n, 5.0)
    noise = 5.0 + 0.05 * rng.normal(size=n)

    feat_dir = tmp_path / "endpoint_features"
    feat_dir.mkdir()
    for traj in ("tA", "tB"):
        pd.DataFrame(
            {
                "traj_id": traj,
                "frame": frames,
                "time_ps": frames.astype(float),
                "endpoint_dist_mean": 0.5 * (driver + decoy),
                "endpoint_dist_0s0_1s0": driver + (0.01 if traj == "tB" else 0.0),
                "endpoint_dist_0s1_1s1": decoy,
                "endpoint_dist_2s0_3s0": noise,
            }
        ).to_csv(feat_dir / f"{traj}_endpoint_features.csv", index=False)

    sites = pd.DataFrame(
        [
            {"traj_id": "tA", "monomer": 0, "site_index": 0, "kind": "ring", "atom_ids": "1|2", "n_atoms": 2},
            {"traj_id": "tA", "monomer": 0, "site_index": 1, "kind": "atom", "atom_ids": "3", "n_atoms": 1},
            {"traj_id": "tA", "monomer": 1, "site_index": 0, "kind": "ring", "atom_ids": "4|5", "n_atoms": 2},
            {"traj_id": "tA", "monomer": 1, "site_index": 1, "kind": "atom", "atom_ids": "6", "n_atoms": 1},
            {"traj_id": "tA", "monomer": 2, "site_index": 0, "kind": "ring", "atom_ids": "7", "n_atoms": 1},
            {"traj_id": "tA", "monomer": 3, "site_index": 0, "kind": "ring", "atom_ids": "8", "n_atoms": 1},
        ]
    )
    sites.to_csv(feat_dir / "endpoint_sites.csv", index=False)

    # Two trajectories with the same 0→1 transition (aggregation / n_events=2).
    segments = pd.DataFrame(
        [
            {
                "traj_id": "tA",
                "group": "endpoint",
                "segment_id": 0,
                "cluster_label": 0,
                "start_frame": 0,
                "end_frame": 90,
                "start_ps": 0.0,
                "end_ps": 90.0,
                "n_frames": 10,
            },
            {
                "traj_id": "tA",
                "group": "endpoint",
                "segment_id": 1,
                "cluster_label": 1,
                "start_frame": 100,
                "end_frame": 190,
                "start_ps": 100.0,
                "end_ps": 190.0,
                "n_frames": 10,
            },
            {
                "traj_id": "tB",
                "group": "endpoint",
                "segment_id": 0,
                "cluster_label": 0,
                "start_frame": 0,
                "end_frame": 90,
                "start_ps": 0.0,
                "end_ps": 90.0,
                "n_frames": 10,
            },
            {
                "traj_id": "tB",
                "group": "endpoint",
                "segment_id": 1,
                "cluster_label": 1,
                "start_frame": 100,
                "end_frame": 190,
                "start_ps": 100.0,
                "end_ps": 190.0,
                "n_frames": 10,
            },
        ]
    )

    events = build_transition_events(segments, group="endpoint")
    assert len(events) == 2
    assert set(events["from_cluster"]) == {0}
    assert set(events["to_cluster"]) == {1}
    assert int(events.iloc[0]["boundary_frame"]) == 100

    event_df, ranking = score_transition_features(events, feat_dir)
    assert not ranking.empty
    top = top_features_per_transition(ranking, top_n=3)
    best = top.sort_values("rank").iloc[0]
    assert best["feature"] == "endpoint_dist_0s0_1s0"
    assert best["mean_delta_angstrom"] > 0
    assert best["mean_point_biserial_corr"] > 0
    assert int(best["n_events"]) == 2
    assert int(best["n_trajectories"]) == 2
    assert bool(best["is_descriptive"]) is False

    out = tmp_path / "cp_out"
    out.mkdir()
    (out / "clusters" / "endpoint").mkdir(parents=True)
    segments.to_csv(out / "clusters" / "endpoint" / "segments_clustered.csv", index=False)

    arts = attribute_endpoint_transitions(
        out,
        feat_dir,
        config=EndpointTransitionAttributionConfig(top_n=5),
        clustered_df=segments,
        sites_df=sites,
        output_dir=out,
        plot_dir=out / "plots",
    )
    assert (out / "endpoint_transition_events.csv").exists()
    assert (out / "endpoint_transition_feature_rankings.csv").exists()
    assert (out / "endpoint_transition_top_features.csv").exists()
    assert (out / "plots" / "endpoint_transition_feature_heatmap.png").exists()
    assert (out / "plots" / "endpoint_transition_network_attributed.png").exists()
    site_png = out / "plots" / "endpoint_sites_transition_0_to_1.png"
    site_html = out / "plots" / "endpoint_sites_transition_0_to_1.html"
    assert site_png.exists() or site_html.exists()
    assert "endpoint_transition_top_features.csv" in arts

    top_csv = pd.read_csv(out / "endpoint_transition_top_features.csv")
    assert "pair_label" in top_csv.columns
    assert "atom_ids_i" in top_csv.columns
    assert top_csv.iloc[0]["feature"] == "endpoint_dist_0s0_1s0"


def test_attribution_skips_without_site_pairs(tmp_path: Path) -> None:
    from src.ChangepointAnalysis.pipeline import ChangepointPipeline
    from src.ChangepointAnalysis.transition_attribution import attribute_endpoint_transitions

    feat_dir = tmp_path / "endpoint_features"
    feat_dir.mkdir()
    # Aggregate columns only — no site-pair series.
    pd.DataFrame(
        {
            "traj_id": ["x"] * 5,
            "frame": list(range(5)),
            "time_ps": list(range(5)),
            "endpoint_dist_mean": [1.0, 1.1, 1.2, 1.3, 1.4],
            "endpoint_dist_0_1_mean": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    ).to_csv(feat_dir / "x_endpoint_features.csv", index=False)

    out = tmp_path / "out"
    out.mkdir()
    segments = pd.DataFrame(
        [
            {
                "traj_id": "x",
                "group": "endpoint",
                "segment_id": 0,
                "cluster_label": 0,
                "start_frame": 0,
                "end_frame": 1,
                "n_frames": 2,
            },
            {
                "traj_id": "x",
                "group": "endpoint",
                "segment_id": 1,
                "cluster_label": 1,
                "start_frame": 2,
                "end_frame": 4,
                "n_frames": 3,
            },
        ]
    )
    (out / "clusters" / "endpoint").mkdir(parents=True)
    segments.to_csv(out / "clusters" / "endpoint" / "segments_clustered.csv", index=False)

    pipe = ChangepointPipeline(
        feat_dir,
        out,
        features_suffix="_endpoint_features.csv",
    )
    pipe._clustered = segments
    skipped = pipe.attribute_endpoint_transitions()
    assert skipped == {}

    # Direct call with no site-pair columns still writes events but empty rankings.
    arts = attribute_endpoint_transitions(out, feat_dir, clustered_df=segments)
    assert (out / "endpoint_transition_events.csv").exists()
    rankings = pd.read_csv(out / "endpoint_transition_feature_rankings.csv")
    assert rankings.empty
    assert "endpoint_transition_feature_heatmap.png" not in arts


def test_attribution_handles_no_transitions(tmp_path: Path) -> None:
    from src.ChangepointAnalysis.transition_attribution import build_transition_events

    segments = pd.DataFrame(
        [
            {
                "traj_id": "only",
                "group": "endpoint",
                "segment_id": 0,
                "cluster_label": 0,
                "start_frame": 0,
                "end_frame": 10,
                "n_frames": 11,
            },
            {
                "traj_id": "only",
                "group": "endpoint",
                "segment_id": 1,
                "cluster_label": 0,  # same cluster → not a directed transition
                "start_frame": 11,
                "end_frame": 20,
                "n_frames": 10,
            },
        ]
    )
    events = build_transition_events(segments)
    assert events.empty


def test_classify_paper_d1_state() -> None:
    from src.ChangepointAnalysis.endpoint_features import classify_paper_d1_state

    assert classify_paper_d1_state(4.0) == "closed"
    assert classify_paper_d1_state(4.5) == "open"
    assert classify_paper_d1_state(5.0) == "open"
    assert classify_paper_d1_state(5.5) == "open"
    assert classify_paper_d1_state(6.0) == "elongated"


def test_paper_d1_column_naming_and_parser() -> None:
    from src.ChangepointAnalysis.endpoint_features import (
        list_paper_d1_columns,
        paper_d1_column_name,
    )
    from src.ChangepointAnalysis.transition_attribution import parse_paper_d1_feature

    name = paper_d1_column_name(0, 3, 1, 7)
    assert name == "paper_d1_m0s3_m1s7"
    assert parse_paper_d1_feature(name) == (0, 3, 1, 7)
    assert parse_paper_d1_feature("endpoint_dist_0s3_1s7") is None
    cols = list_paper_d1_columns(
        [name, "paper_d1_min", "paper_d1_n_open", "endpoint_dist_0s3_1s7"]
    )
    assert cols == [name]


def test_resolve_paper_d1_atoms_bmm_topology() -> None:
    """Live topology check: s3/s7 map to unique bonded ring neighbors."""
    pytest.importorskip("rdkit")
    import MDAnalysis as mda

    from src.EndpointAnalyzer.EndpointAnalyzer import EndpointAnalyzer
    from src.EndpointAnalyzer import EndpointsFinder
    from src.ChangepointAnalysis.endpoint_features import resolve_paper_d1_atoms
    from src.utils.gsa_selections import resolve_selections

    topo = Path("traj/BMMpM_ca.prmtop")
    traj = Path("traj/BMMpM_891249_mdcrd_v.trj")
    if not topo.exists() or not traj.exists():
        pytest.skip("BMMpM topology/trajectory not present")

    u = mda.Universe(str(topo), str(traj))
    u.trajectory[0]
    sels = resolve_selections(u, gsa_resname="MOL", n_monomers=6, auto_tooth=False)
    finder = EndpointsFinder()
    stored = []
    for mon_sel in sels.monomer_selections:
        _, sites = EndpointAnalyzer.find_residue_endpoint_sites(u, mon_sel, finder)
        stored.append(sites)

    d1 = resolve_paper_d1_atoms(
        u, sels.monomer_selections, stored, traj_id="topo", s3_site=3, s7_site=7
    )
    assert len(d1) == 12  # 6 monomers × {s3,s7}
    # Known mapping for monomer 0 from earlier inspection.
    m0 = d1[(d1["monomer"] == 0) & (d1["endpoint_site"] == 3)].iloc[0]
    assert int(m0["endpoint_atom_id"]) == 70
    assert int(m0["d1_ring_atom_id"]) == 65
    m0b = d1[(d1["monomer"] == 0) & (d1["endpoint_site"] == 7)].iloc[0]
    assert int(m0b["endpoint_atom_id"]) == 112
    assert int(m0b["d1_ring_atom_id"]) == 103


def test_compute_paper_d1_distances_synthetic() -> None:
    """Synthetic two-point geometry: one open and one elongated contact."""
    import types

    from src.ChangepointAnalysis.endpoint_features import (
        classify_paper_d1_state,
        compute_paper_d1_distances,
        paper_d1_column_name,
    )

    class _Atom:
        def __init__(self, idx: int, aid: int, pos):
            self.index = idx
            self.id = aid
            self.position = np.asarray(pos, dtype=float)

    class _Atoms:
        def __init__(self, atoms):
            self._atoms = atoms

        def __iter__(self):
            return iter(self._atoms)

        def __getitem__(self, idx):
            return self._atoms[idx]

    class _Traj:
        def __init__(self):
            self.n = 1

        def __len__(self):
            return 1

        def __getitem__(self, i):
            return None

    # Two monomers, each with s3 and s7 ring atoms.
    # m0s3 at origin, m1s7 at 5.0 Å (open), m0s7 at origin+y, m1s3 far.
    atoms = _Atoms(
        [
            _Atom(0, 10, [0.0, 0.0, 0.0]),  # m0 s3 ring
            _Atom(1, 11, [0.0, 1.0, 0.0]),  # m0 s7 ring
            _Atom(2, 20, [5.0, 0.0, 0.0]),  # m1 s7 ring → open vs m0s3
            _Atom(3, 21, [20.0, 1.0, 0.0]),  # m1 s3 ring → elongated vs m0s7
        ]
    )
    universe = types.SimpleNamespace(atoms=atoms, trajectory=_Traj())
    d1_atoms = pd.DataFrame(
        [
            {"monomer": 0, "endpoint_site": 3, "d1_ring_atom_id": 10},
            {"monomer": 0, "endpoint_site": 7, "d1_ring_atom_id": 11},
            {"monomer": 1, "endpoint_site": 3, "d1_ring_atom_id": 21},
            {"monomer": 1, "endpoint_site": 7, "d1_ring_atom_id": 20},
        ]
    )
    out = compute_paper_d1_distances(universe, d1_atoms, frame_indices=[0])
    c_open = paper_d1_column_name(0, 3, 1, 7)
    c_long = paper_d1_column_name(0, 7, 1, 3)
    assert abs(out.loc[0, c_open] - 5.0) < 1e-6
    assert classify_paper_d1_state(out.loc[0, c_open]) == "open"
    assert out.loc[0, c_long] > 5.5
    assert out.loc[0, "paper_d1_n_open"] >= 1
