"""G3/G4 tests: permutation η² null, BH-FDR, chemical k-scan."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ChangepointAnalysis.cluster_k_diagnostics import (
    assess_chemical_separation,
    scan_chemical_separation_by_k,
    summarize_chemical_k_by_cohort,
)
from src.ChangepointAnalysis.reporting import (
    _eta_squared,
    _eta_squared_columns,
    benjamini_hochberg_qvalues,
    permutation_eta_squared_pvalues,
    score_cluster_discriminating_features,
)


def test_eta_squared_columns_matches_scalar() -> None:
    rng = np.random.default_rng(0)
    y = np.array([0] * 20 + [1] * 20)
    X = rng.normal(size=(40, 5))
    X[:, 0] += y * 3.0
    X[5, 2] = np.nan
    vec = _eta_squared_columns(X, y)
    for j in range(5):
        assert vec[j] == pytest.approx(_eta_squared(X[:, j], y, [0, 1]), nan_ok=True)


def test_benjamini_hochberg_qvalues_monotone_and_nan() -> None:
    p = np.array([0.001, 0.01, 0.04, 0.20, 0.80, np.nan])
    q = benjamini_hochberg_qvalues(p)
    assert np.isnan(q[-1])
    order = np.argsort(p[:-1])
    assert np.all(np.diff(q[:-1][order]) >= -1e-12)
    assert q[0] < q[4]
    assert q[0] == pytest.approx(0.001 * 5 / 1)


def test_permutation_null_marks_planted_column() -> None:
    rng = np.random.default_rng(1)
    y = np.array([0] * 25 + [1] * 25 + [2] * 25)
    X = rng.normal(size=(75, 6))
    X[:, 0] += y.astype(float) * 4.0
    p, q, eta = permutation_eta_squared_pvalues(
        X, y, n_permutations=199, random_state=0
    )
    assert eta[0] > 0.7
    assert p[0] == pytest.approx(1.0 / 200.0)
    assert q[0] <= 0.05
    assert bool(q[0] < np.nanmin(q[1:]))
    assert np.nanmin(p[1:]) > p[0]


def test_score_adds_fdr_columns() -> None:
    rng = np.random.default_rng(2)
    y = np.array([0] * 12 + [1] * 12)
    X = rng.normal(size=(24, 3))
    X[:, 0] += y * 5.0
    specs = [
        ("endpoint_dist_0s0_1s0", 0, 0, 1, 0),
        ("endpoint_dist_2s0_3s0", 2, 0, 3, 0),
        ("endpoint_dist_4s0_5s0", 4, 0, 5, 0),
    ]
    ranking = score_cluster_discriminating_features(
        X,
        y,
        specs,
        "site_pair",
        n_permutations=99,
        fdr_alpha=0.05,
        random_state=0,
    )
    assert "eta_squared_p_value" in ranking.columns
    assert "eta_squared_q_value" in ranking.columns
    assert "significant_fdr" in ranking.columns
    top = ranking.iloc[0]
    assert top["endpoint_label"] == "M0S0-M1S0"
    assert bool(top["significant_fdr"])


def test_assess_chemical_separation_distinct_contacts_and_signs() -> None:
    ranking = pd.DataFrame(
        {
            "endpoint_label": ["M0S0-M1S0", "M2S0-M3S0"],
            "eta_squared": [0.70, 0.60],
            "eta_squared_q_value": [0.001, 0.002],
            "cohens_d_best": [1.8, -1.5],
            "best_cluster": [0, 1],
        }
    )
    sizes = {0: 20, 1: 20, 2: 20}
    out = assess_chemical_separation(ranking, sizes)
    assert out["chemical_separation"] is True
    assert out["chemical_separation_fdr"] is True
    assert out["n_marked_clusters"] == 2
    assert out["mixed_cohens_d_signs"] is True
    assert out["used_fdr"] is True

    same = ranking.copy()
    same["best_cluster"] = [0, 0]
    same["endpoint_label"] = ["M0S0-M1S0", "M0S0-M1S0"]
    same["cohens_d_best"] = [1.8, 1.6]
    out_same = assess_chemical_separation(same, sizes)
    assert out_same["chemical_separation"] is False
    assert out_same["fail_reason"] == "single_marked_cluster"

    out_tiny = assess_chemical_separation(ranking, {0: 1, 1: 1, 2: 40})
    assert out_tiny["chemical_separation"] is False
    assert out_tiny["fail_reason"] == "too_few_usable_clusters"

    none = ranking.copy()
    none["eta_squared_q_value"] = [0.4, 0.5]
    out_none = assess_chemical_separation(none, sizes)
    assert out_none["chemical_separation"] is True
    assert out_none["chemical_separation_fdr"] is False
    assert int(out_none["n_significant_fdr"]) == 0


def _write_chemical_scan_cohort(root: Path) -> Path:
    """Tiny published-style cohort with by_k labels and site-pair CSVs."""
    cohort = root / "endpoint_changepoints_BHHpM"
    cdir = cohort / "clusters" / "endpoint"
    feat = cohort / "endpoint_features"
    for k in (2, 3):
        (cdir / "by_k" / f"k_{k:02d}").mkdir(parents=True, exist_ok=True)
    feat.mkdir(parents=True, exist_ok=True)

    n_traj = 3
    segs_per = 4
    frames_per_seg = 10
    rows = []
    k2_labels = [0] * 11 + [1]  # outlier trap at k=2
    k3_labels = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    k2_rows = []
    k3_rows = []
    idx = 0
    for t in range(n_traj):
        tid = f"{100000 + t}_mdcrd_v"
        frames = []
        for s in range(segs_per):
            start = s * frames_per_seg
            end = start + frames_per_seg - 1
            lab5 = k3_labels[idx]  # chosen k=3 in this toy
            rows.append(
                {
                    "traj_id": tid,
                    "group": "endpoint",
                    "segment_id": s,
                    "start_frame": start,
                    "end_frame": end,
                    "n_frames": frames_per_seg,
                    "cluster_label": lab5,
                }
            )
            k2_rows.append(
                {
                    "traj_id": tid,
                    "segment_id": s,
                    "start_frame": start,
                    "end_frame": end,
                    "n_frames": frames_per_seg,
                    "cluster_label": k2_labels[idx],
                }
            )
            k3_rows.append(
                {
                    "traj_id": tid,
                    "segment_id": s,
                    "start_frame": start,
                    "end_frame": end,
                    "n_frames": frames_per_seg,
                    "cluster_label": k3_labels[idx],
                }
            )
            cl = k3_labels[idx]
            for f in range(start, end + 1):
                pair_a = 8.0 if cl == 0 else 5.0
                pair_b = 8.0 if cl == 1 else 5.0
                frames.append(
                    {
                        "frame": f,
                        "time_ps": float(f),
                        "endpoint_dist_0s0_1s0": pair_a,
                        "endpoint_dist_2s0_3s0": pair_b,
                    }
                )
            idx += 1
        pd.DataFrame(frames).to_csv(feat / f"{tid}_endpoint_features.csv", index=False)

    pd.DataFrame(rows).to_csv(cdir / "segments_clustered.csv", index=False)
    pd.DataFrame(k2_rows).to_csv(
        cdir / "by_k" / "k_02" / "segments_clustered.csv", index=False
    )
    pd.DataFrame(k3_rows).to_csv(
        cdir / "by_k" / "k_03" / "segments_clustered.csv", index=False
    )
    (cdir / "cluster_summary.txt").write_text(
        "Selection mode: fixed k=3\n", encoding="utf-8"
    )
    pd.DataFrame(
        {"k": [2, 3], "silhouette": [0.40, 0.22], "n_clusters_effective": [2, 3]}
    ).to_csv(cdir / "silhouette_by_k.csv", index=False)
    return cohort


def test_chemical_k_scan_first_passing_k(tmp_path: Path) -> None:
    cohort = _write_chemical_scan_cohort(tmp_path)
    out = tmp_path / "chem"
    written = scan_chemical_separation_by_k(
        [cohort],
        out,
        k_min=2,
        k_max=3,
        n_permutations=49,
        min_cluster_size=3,
        random_state=0,
    )
    assert (out / "chemical_separation_by_k.csv").exists()
    assert (out / "chemical_k_peaks.csv").exists()
    assert "chemical_k_summary.txt" in written

    long = pd.read_csv(out / "chemical_separation_by_k.csv")
    peaks = pd.read_csv(out / "chemical_k_peaks.csv")
    by_k = long.set_index("k")
    assert bool(by_k.loc[2, "chemical_separation"]) is False
    assert bool(by_k.loc[3, "chemical_separation"]) is True
    assert int(peaks.iloc[0]["first_chemical_k"]) == 3
    assert int(peaks.iloc[0]["best_chemical_k"]) == 3
    summary = summarize_chemical_k_by_cohort(long)
    assert bool(summary.iloc[0]["any_chemical_k"])
