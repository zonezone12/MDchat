"""Tests for Imamura/changepoint state and transition comparison."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.imamura_msm import (
    ImamuraBeadSpec,
    ImamuraPairTrace,
    build_imamura_feature_vector_with_trace,
    load_imamura_pair_trace,
    write_imamura_pair_trace,
)
from src.utils.state_comparison import (
    align_state_labels,
    build_imamura_variants,
    compare_boundaries,
    exact_ranked_pair,
    expand_changepoint_segments,
    extract_transition_events,
    fit_posthoc_lda,
    hungarian_state_mapping,
    state_agreement_metrics,
    state_contingency,
    suppress_short_recrossings,
    trace_exact_transition_pairs,
    trace_recorded_transition_pairs,
    transition_window_statistics,
)


def _synthetic_frame_table(n_trajectories: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows: list[dict] = []
    for trajectory in range(n_trajectories):
        for frame in range(60):
            state = 0 if frame < 20 else 1 if frame < 40 else 2
            rows.append(
                {
                    "traj_id": f"traj_{trajectory}",
                    "frame": frame * 10,
                    "time_ps": float(frame * 10),
                    "v1_01": state * 4.0 + rng.normal(scale=0.2),
                    "v1_02": state * -2.0 + rng.normal(scale=0.2),
                    "v4_001": state * 3.0 + rng.normal(scale=0.2),
                    "v4_002": np.sin(frame / 5) + rng.normal(scale=0.1),
                    "macro_label": state,
                }
            )
    return pd.DataFrame(rows)


def test_build_matched_pca_and_tlica_variants() -> None:
    table = _synthetic_frame_table()
    variants = build_imamura_variants(
        table,
        n_components=3,
        n_microclusters=9,
        n_macrostates=3,
        tlica_lag_ns=0.02,
        random_state=2,
    )

    assert set(variants) == {"pca", "tlica"}
    assert len(variants["pca"]) == len(table)
    assert {"micro_label", "macro_label", "PC1"} <= set(variants["pca"])
    assert {"micro_label", "macro_label", "TIC1"} <= set(variants["tlica"])
    assert variants["pca"][["traj_id", "frame"]].equals(
        variants["tlica"][["traj_id", "frame"]]
    )


def test_posthoc_lda_uses_grouped_validation_and_feature_blocks() -> None:
    table = _synthetic_frame_table()
    result = fit_posthoc_lda(table, max_components=5)

    assert {"LD1", "LD2"} <= set(result.frame_scores)
    assert set(result.loadings["feature_block"]) == {"central_ring", "endpoint"}
    assert result.validation.iloc[0]["validation_kind"] == (
        "leave_one_trajectory_out"
    )
    assert result.validation.iloc[0]["balanced_accuracy"] > 0.95


def test_posthoc_lda_marks_single_trajectory_as_in_sample() -> None:
    result = fit_posthoc_lda(_synthetic_frame_table(1))
    assert result.validation.iloc[0]["validation_kind"] == "in_sample_only"


def test_short_aba_recrossing_is_suppressed() -> None:
    labels = [0, 0, 0, 1, 0, 0, 2, 2]
    smoothed = suppress_short_recrossings(labels, min_dwell_rows=2)
    assert smoothed.tolist() == [0, 0, 0, 0, 0, 0, 2, 2]


def test_extract_transition_events_never_crosses_trajectories() -> None:
    table = pd.DataFrame(
        {
            "traj_id": ["a"] * 4 + ["b"] * 4,
            "frame": [0, 1, 2, 3] * 2,
            "time_ps": [0.0, 1.0, 2.0, 3.0] * 2,
            "macro_label": [0, 0, 1, 1, 2, 2, 0, 0],
        }
    )
    events = extract_transition_events(table)
    assert len(events) == 2
    assert set(zip(events["from_state"], events["to_state"])) == {(0, 1), (2, 0)}


def test_transition_windows_recover_known_endpoint_increase() -> None:
    table = pd.DataFrame(
        {
            "traj_id": ["a"] * 12 + ["b"] * 12,
            "frame": list(range(12)) * 2,
            "time_ps": [float(value) for value in range(12)] * 2,
            "macro_label": ([0] * 6 + [1] * 6) * 2,
            "v1_01": ([0.0] * 6 + [1.0] * 6) * 2,
            "v4_001": ([2.0] * 6 + [8.0] * 6) * 2,
        }
    )
    events = extract_transition_events(table)
    deltas, signatures, traces = transition_window_statistics(
        table,
        events,
        window_rows=3,
        bootstrap_samples=50,
        random_state=1,
    )

    endpoint = signatures[signatures["feature"] == "v4_001"].iloc[0]
    assert endpoint["n_events"] == 2
    assert endpoint["mean_delta"] == pytest.approx(6.0)
    assert endpoint["consistent_direction_fraction"] == pytest.approx(1.0)
    assert len(deltas) == 4
    assert not traces.empty


def test_expand_inclusive_changepoint_segments_and_align_exact_frames() -> None:
    segments = pd.DataFrame(
        [
            {
                "traj_id": "a",
                "group": "combined",
                "segment_id": 0,
                "start_frame": 0,
                "end_frame": 20,
                "n_frames": 3,
                "cluster_label": 4,
            },
            {
                "traj_id": "a",
                "group": "combined",
                "segment_id": 1,
                "start_frame": 30,
                "end_frame": 40,
                "n_frames": 2,
                "cluster_label": 7,
            },
        ]
    )
    expanded = expand_changepoint_segments(segments)
    assert expanded["frame"].tolist() == [0, 10, 20, 30, 40]

    imamura = pd.DataFrame(
        {
            "traj_id": ["a"] * 6,
            "frame": [0, 10, 20, 30, 40, 50],
            "time_ps": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            "macro_label": [0, 0, 0, 1, 1, 1],
        }
    )
    aligned = align_state_labels(imamura, expanded)
    assert len(aligned) == 5
    assert aligned["changepoint_label"].tolist() == [4, 4, 4, 7, 7]


def test_label_permutation_has_perfect_agreement_and_mapping() -> None:
    aligned = pd.DataFrame(
        {
            "traj_id": ["a"] * 6,
            "frame": range(6),
            "time_ps": np.arange(6, dtype=float),
            "macro_label": [0, 0, 1, 1, 2, 2],
            "changepoint_segment_id": range(6),
            "changepoint_label": [9, 9, 4, 4, 7, 7],
        }
    )
    metrics = state_agreement_metrics(aligned).iloc[0]
    assert metrics["adjusted_rand_index"] == pytest.approx(1.0)
    contingency = state_contingency(aligned)
    mapping = hungarian_state_mapping(contingency)
    assert mapping["overlap_frames"].sum() == 6


def test_align_reports_trajectory_id_mismatch() -> None:
    imamura = pd.DataFrame(
        {
            "traj_id": ["mdcrd_v"],
            "frame": [0],
            "time_ps": [0.0],
            "macro_label": [0],
        }
    )
    changepoint = pd.DataFrame(
        {
            "traj_id": ["BMMpM_1_mdcrd_v"],
            "frame": [0],
            "changepoint_segment_id": [0],
            "changepoint_label": [0],
        }
    )
    with pytest.raises(ValueError, match="No exact trajectory/frame overlap"):
        align_state_labels(imamura, changepoint)


def test_boundary_matching_is_one_to_one_with_tolerance() -> None:
    imamura = pd.DataFrame(
        {
            "traj_id": ["a", "a"],
            "frame": [100, 200],
            "time_ps": [100.0, 200.0],
            "from_state": [0, 1],
            "to_state": [1, 2],
        }
    )
    changepoints = pd.DataFrame(
        {
            "traj_id": ["a", "a"],
            "group": ["combined", "combined"],
            "frame": [110, 500],
            "time_ps": [110.0, 500.0],
        }
    )
    matches, summary = compare_boundaries(
        imamura, changepoints, tolerance_ps=25.0
    )
    assert len(matches) == 1
    assert summary.iloc[0]["precision"] == pytest.approx(0.5)
    assert summary.iloc[0]["recall"] == pytest.approx(0.5)


def test_exact_ranked_pair_tracks_dynamic_pair_identity() -> None:
    type4 = np.zeros((2, 3), dtype=float)
    first_type1 = np.array([[0.0, 0, 0], [1.0, 0, 0], [4.0, 0, 0]])
    second_type1 = np.array([[0.0, 0, 0], [5.0, 0, 0], [4.0, 0, 0]])

    first = exact_ranked_pair(first_type1, type4, "v1_01")
    second = exact_ranked_pair(second_type1, type4, "v1_01")

    assert first[2:4] == (0, 2)
    assert first[4] == pytest.approx(4.0)
    assert second[2:4] == (0, 1)
    assert second[4] == pytest.approx(5.0)


def test_feature_extraction_records_compact_pair_indices() -> None:
    type1 = np.array([[0.0, 0, 0], [1.0, 0, 0], [4.0, 0, 0]])
    type4 = np.array([[0.0, 1, 0], [0.0, 3, 0]])
    vector, v1_pairs, v4_pairs = build_imamura_feature_vector_with_trace(
        type1,
        type4,
        n_type1=3,
        n_type4=2,
    )

    assert vector.tolist() == pytest.approx([4.0, 3.0, 1.0, 2.0])
    # Canonical order: (0,1)=0, (0,2)=1, (1,2)=2.
    assert v1_pairs.tolist() == [1, 2, 0]
    assert v1_pairs.dtype == np.uint16
    assert v4_pairs.tolist() == [0]


def test_pair_trace_roundtrip_and_recorded_transition_join(tmp_path: Path) -> None:
    trace = ImamuraPairTrace(
        traj_ids=np.asarray(["a", "a", "a"]),
        frames=np.asarray([0, 1, 2], dtype=np.int64),
        v1_pair_indices=np.asarray([[1], [0], [0]], dtype=np.uint16),
        v4_pair_indices=np.asarray([[0], [0], [0]], dtype=np.uint16),
    )
    spec = ImamuraBeadSpec(
        type1_ring_groups=[(0,), (1,), (2,)],
        type4_atom_ids=(3, 4),
    )
    written = write_imamura_pair_trace(
        trace,
        tmp_path,
        spec,
        n_type1=3,
        n_type4=2,
    )
    loaded = load_imamura_pair_trace(written["pair_identity_trace"])
    pair_map = pd.read_csv(written["pair_index_map"])
    assert np.array_equal(loaded.v1_pair_indices, trace.v1_pair_indices)

    frame_table = pd.DataFrame(
        {
            "traj_id": ["a"] * 3,
            "frame": [0, 1, 2],
            "time_ps": [0.0, 1.0, 2.0],
            "v1_01": [4.0, 5.0, 6.0],
            "macro_label": [0, 1, 1],
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_id": 0,
                "traj_id": "a",
                "from_state": 0,
                "to_state": 1,
                "frame": 1,
                "time_ps": 1.0,
            }
        ]
    )
    signatures = pd.DataFrame(
        [
            {
                "from_state": 0,
                "to_state": 1,
                "feature": "v1_01",
                "abs_standardized_delta": 1.0,
            }
        ]
    )
    detailed, summary = trace_recorded_transition_pairs(
        frame_table,
        events,
        signatures,
        loaded,
        pair_map,
        window_rows=1,
        top_features=1,
    )

    assert detailed["pair_index"].tolist() == [1, 0, 0]
    assert detailed["distance"].tolist() == pytest.approx([4.0, 5.0, 6.0])
    assert summary["pre_rank_occupancies"].sum() == 1
    assert summary["post_rank_occupancies"].sum() == 2


def test_exact_transition_trace_reproduces_saved_cv_and_pair_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import MDAnalysis as mda

    frames = [
        np.array(
            [[0.0, 0, 0], [1.0, 0, 0], [4.0, 0, 0], [0, 1, 0], [0, 2, 0]]
        ),
        np.array(
            [[0.0, 0, 0], [5.0, 0, 0], [4.0, 0, 0], [0, 1, 0], [0, 2, 0]]
        ),
        np.array(
            [[0.0, 0, 0], [6.0, 0, 0], [4.0, 0, 0], [0, 1, 0], [0, 2, 0]]
        ),
    ]

    class FakeAtomGroup:
        def __init__(self, universe: "FakeUniverse", indices: list[int]) -> None:
            self.universe = universe
            self.indices = np.asarray(indices, dtype=int)

        @property
        def positions(self) -> np.ndarray:
            return frames[self.universe.current][self.indices]

        @property
        def ids(self) -> np.ndarray:
            return self.indices + 1

        @property
        def names(self) -> np.ndarray:
            return np.asarray([f"A{index}" for index in self.indices])

        @property
        def resids(self) -> np.ndarray:
            return self.indices // 1 + 1

        @property
        def resnames(self) -> np.ndarray:
            return np.asarray(["MOL"] * len(self.indices))

    class FakeAtoms:
        def __init__(self, universe: "FakeUniverse") -> None:
            self.universe = universe

        def __getitem__(self, indices: list[int]) -> FakeAtomGroup:
            return FakeAtomGroup(self.universe, list(indices))

    class FakeTrajectory:
        def __init__(self, universe: "FakeUniverse") -> None:
            self.universe = universe
            self.ts = type("TS", (), {"time": 0.0})()

        def __len__(self) -> int:
            return len(frames)

        def __getitem__(self, frame: int) -> "FakeTrajectory":
            self.universe.current = frame
            self.ts.time = float(frame)
            return self

    class FakeUniverse:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            self.current = 0
            self.atoms = FakeAtoms(self)
            self.trajectory = FakeTrajectory(self)

    monkeypatch.setattr(mda, "Universe", FakeUniverse)
    frame_table = pd.DataFrame(
        {
            "traj_id": ["a"] * 3,
            "frame": [0, 1, 2],
            "time_ps": [0.0, 1.0, 2.0],
            "v1_01": [4.0, 5.0, 6.0],
            "macro_label": [0, 1, 1],
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_id": 0,
                "traj_id": "a",
                "from_state": 0,
                "to_state": 1,
                "frame": 1,
                "time_ps": 1.0,
            }
        ]
    )
    signatures = pd.DataFrame(
        [
            {
                "from_state": 0,
                "to_state": 1,
                "feature": "v1_01",
                "abs_standardized_delta": 1.0,
            }
        ]
    )
    spec = ImamuraBeadSpec(
        type1_ring_groups=[(0,), (1,), (2,)],
        type4_atom_ids=(3, 4),
    )
    trace, summary = trace_exact_transition_pairs(
        frame_table,
        events,
        signatures,
        topology="topology",
        trajectory_paths={"a": "trajectory"},
        bead_spec=spec,
        window_rows=1,
        top_features=1,
    )

    assert trace["distance_residual"].abs().max() == pytest.approx(0.0)
    assert list(zip(trace["bead_i"], trace["bead_j"])) == [
        (0, 2),
        (0, 1),
        (0, 1),
    ]
    assert summary["n_rank_occupancies"].sum() == 3


def test_comparison_cli_writes_end_to_end_artifacts(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame_states.csv"
    segment_path = tmp_path / "segments_clustered.csv"
    breakpoint_path = tmp_path / "breakpoints.csv"
    output_dir = tmp_path / "comparison"
    _synthetic_frame_table().to_csv(frame_path, index=False)

    segments: list[dict] = []
    breakpoints: list[dict] = []
    for trajectory in range(3):
        traj_id = f"traj_{trajectory}"
        for segment_id, (start, stop, label) in enumerate(
            [(0, 190, 0), (200, 390, 1), (400, 590, 2)]
        ):
            segments.append(
                {
                    "traj_id": traj_id,
                    "group": "combined",
                    "segment_id": segment_id,
                    "start_frame": start,
                    "end_frame": stop,
                    "n_frames": 20,
                    "cluster_label": label,
                }
            )
        for frame in (200, 400):
            breakpoints.append(
                {
                    "traj_id": traj_id,
                    "group": "combined",
                    "frame": frame,
                    "time_ps": float(frame),
                }
            )
    pd.DataFrame(segments).to_csv(segment_path, index=False)
    pd.DataFrame(breakpoints).to_csv(breakpoint_path, index=False)

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "compare_state_methods.py"),
            "--imamura-frame-states",
            str(frame_path),
            "--changepoint-segments",
            str(segment_path),
            "--changepoint-breakpoints",
            str(breakpoint_path),
            "--output-dir",
            str(output_dir),
            "--n-components",
            "3",
            "--n-micro",
            "9",
            "--n-macro",
            "3",
            "--tlica-lag-ns",
            "0.02",
            "--min-dwell-rows",
            "2",
            "--window-rows",
            "3",
            "--bootstrap-samples",
            "5",
            "--max-timeline-plots",
            "0",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "pca" / "transition_signatures_top.csv").is_file()
    assert (output_dir / "tlica" / "boundary_matches.csv").is_file()
    assert (output_dir / "pca_vs_tlica_transition_consensus.csv").is_file()

