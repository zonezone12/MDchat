"""State alignment and transition-path analysis for Imamura/changepoint results.

The utilities in this module operate on saved CSV tables.  They intentionally
avoid reading trajectories again: Imamura ``v1_*``/``v4_*`` collective
variables and optional GSA observables are aligned by trajectory and frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.utils.imamura_msm import (
    ImamuraBeadSpec,
    ImamuraMSMConfig,
    bead_positions_from_spec,
    cluster_imamura_features,
    imamura_projection_column_names,
)


META_COLUMNS = {"traj_id", "frame", "time_ps", "micro_label", "macro_label"}


@dataclass
class LDAInterpretation:
    """Post-hoc supervised projection of already-discovered macrostates."""

    frame_scores: pd.DataFrame
    loadings: pd.DataFrame
    centroids: pd.DataFrame
    validation: pd.DataFrame


def imamura_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return ordered Imamura distance-feature columns."""
    columns = [
        column
        for column in df.columns
        if column.startswith("v1_") or column.startswith("v4_")
    ]
    if not columns:
        raise ValueError("No Imamura feature columns matching v1_* or v4_*")
    return columns


def validate_frame_table(df: pd.DataFrame) -> None:
    """Validate metadata and uniqueness required for framewise comparisons."""
    required = {"traj_id", "frame", "time_ps"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Frame table missing columns: {sorted(missing)}")
    if df.duplicated(["traj_id", "frame"]).any():
        raise ValueError("Frame table contains duplicate (traj_id, frame) rows")


def infer_sample_time_ps(df: pd.DataFrame) -> float:
    """Infer median time spacing between saved rows, without crossing trajectories."""
    diffs: list[np.ndarray] = []
    for _, group in df.groupby("traj_id", sort=False):
        values = np.sort(group["time_ps"].to_numpy(dtype=float))
        positive = np.diff(values)
        positive = positive[np.isfinite(positive) & (positive > 0)]
        if positive.size:
            diffs.append(positive)
    if not diffs:
        raise ValueError("Cannot infer positive time spacing from time_ps")
    return float(np.median(np.concatenate(diffs)))


def _base_frame_table(df: pd.DataFrame) -> pd.DataFrame:
    """Drop labels/projections from a prior run while retaining raw CVs."""
    projection_prefixes = ("PC", "TIC", "LD")
    drop = [
        column
        for column in df.columns
        if column in {"micro_label", "macro_label"}
        or column.startswith(projection_prefixes)
    ]
    return df.drop(columns=drop, errors="ignore").copy()


def build_imamura_variants(
    frame_table: pd.DataFrame,
    *,
    n_components: int = 5,
    n_microclusters: int = 1500,
    n_macrostates: int = 22,
    tlica_lag_ns: float = 2.0,
    tlica_regularization: float = 1e-6,
    random_state: int = 0,
) -> Dict[str, pd.DataFrame]:
    """Create matched PCA and tlICA state tables from the same frame/CV rows."""
    validate_frame_table(frame_table)
    base = _base_frame_table(frame_table)
    features = imamura_feature_columns(base)
    matrix = base[features].to_numpy(dtype=np.float64)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    traj_ids = base["traj_id"].astype(str).to_numpy()

    sample_time_ps = infer_sample_time_ps(base)
    lag_rows = max(1, int(round(float(tlica_lag_ns) * 1000.0 / sample_time_ps)))
    lengths = base.groupby("traj_id", sort=False).size()
    if (lengths <= lag_rows).all():
        raise ValueError(
            f"tlICA lag ({lag_rows} rows) is not shorter than any trajectory"
        )

    variants: Dict[str, pd.DataFrame] = {}
    for method in ("pca", "tlica"):
        config = ImamuraMSMConfig(
            dimensionality_reduction=method,
            n_pca_components=n_components,
            n_microclusters=n_microclusters,
            n_macrostates=n_macrostates,
            tlica_lag_ns=tlica_lag_ns,
            tlica_regularization=tlica_regularization,
            kmeans_random_state=random_state,
        )
        result = cluster_imamura_features(
            matrix,
            config=config,
            traj_ids=traj_ids,
            tlica_lag_frames=lag_rows,
            tlica_lag_ns=tlica_lag_ns,
        )
        labeled = base.copy()
        labeled["micro_label"] = result.micro_labels
        labeled["macro_label"] = result.macro_labels
        names = imamura_projection_column_names(
            method, result.pca_scores.shape[1]
        )
        for index, name in enumerate(names):
            labeled[name] = result.pca_scores[:, index]
        variants[method] = labeled
    return variants


def fit_posthoc_lda(
    frame_table: pd.DataFrame,
    *,
    label_col: str = "macro_label",
    max_components: int = 5,
) -> LDAInterpretation:
    """Fit LDA for state interpretation and trajectory-grouped validation."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler

    validate_frame_table(frame_table)
    if label_col not in frame_table:
        raise ValueError(f"Frame table missing label column {label_col!r}")

    features = imamura_feature_columns(frame_table)
    X = frame_table[features].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = frame_table[label_col].to_numpy(dtype=int)
    groups = frame_table["traj_id"].astype(str).to_numpy()
    classes = np.unique(y)
    if classes.size < 2:
        raise ValueError("LDA requires at least two macrostate classes")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n_components = min(max_components, classes.size - 1, Xs.shape[1])
    lda = LinearDiscriminantAnalysis(solver="svd", n_components=n_components)
    scores = lda.fit_transform(Xs, y)
    ld_columns = [f"LD{i + 1}" for i in range(scores.shape[1])]

    frame_scores = frame_table[["traj_id", "frame", "time_ps", label_col]].copy()
    for index, column in enumerate(ld_columns):
        frame_scores[column] = scores[:, index]

    scaling = np.asarray(lda.scalings_)[:, : scores.shape[1]]
    loading_rows: list[dict] = []
    for feature_index, feature in enumerate(features):
        block = "central_ring" if feature.startswith("v1_") else "endpoint"
        for component_index, component in enumerate(ld_columns):
            value = float(scaling[feature_index, component_index])
            loading_rows.append(
                {
                    "feature": feature,
                    "feature_block": block,
                    "component": component,
                    "loading": value,
                    "abs_loading": abs(value),
                }
            )
    loadings = pd.DataFrame(loading_rows).sort_values(
        ["component", "abs_loading"], ascending=[True, False]
    )

    centroid_rows: list[dict] = []
    for state in classes:
        mask = y == state
        row = {"macro_label": int(state), "n_frames": int(mask.sum())}
        row.update(
            {column: float(scores[mask, i].mean()) for i, column in enumerate(ld_columns)}
        )
        centroid_rows.append(row)
    centroids = pd.DataFrame(centroid_rows)

    validation_rows: list[dict] = []
    unique_groups = np.unique(groups)
    if unique_groups.size >= 2:
        truth: list[int] = []
        predictions: list[int] = []
        logo = LeaveOneGroupOut()
        for train, test in logo.split(X, y, groups):
            train_classes = np.unique(y[train])
            eligible = np.isin(y[test], train_classes)
            if train_classes.size < 2 or not eligible.any():
                continue
            fold_scaler = StandardScaler()
            X_train = fold_scaler.fit_transform(X[train])
            X_test = fold_scaler.transform(X[test][eligible])
            fold_lda = LinearDiscriminantAnalysis(solver="svd")
            fold_lda.fit(X_train, y[train])
            predicted = fold_lda.predict(X_test)
            fold_truth = y[test][eligible]
            truth.extend(fold_truth.tolist())
            predictions.extend(predicted.tolist())
            validation_rows.append(
                {
                    "scope": f"held_out:{groups[test][0]}",
                    "n_rows": int(eligible.sum()),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(fold_truth, predicted)
                    ),
                    "macro_f1": float(
                        f1_score(fold_truth, predicted, average="macro", zero_division=0)
                    ),
                    "validation_kind": "leave_one_trajectory_out",
                }
            )
        if truth:
            validation_rows.insert(
                0,
                {
                    "scope": "overall",
                    "n_rows": len(truth),
                    "balanced_accuracy": float(
                        balanced_accuracy_score(truth, predictions)
                    ),
                    "macro_f1": float(
                        f1_score(truth, predictions, average="macro", zero_division=0)
                    ),
                    "validation_kind": "leave_one_trajectory_out",
                },
            )
    if not validation_rows:
        predicted = lda.predict(Xs)
        validation_rows.append(
            {
                "scope": "overall",
                "n_rows": len(y),
                "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
                "macro_f1": float(
                    f1_score(y, predicted, average="macro", zero_division=0)
                ),
                "validation_kind": "in_sample_only",
            }
        )

    return LDAInterpretation(
        frame_scores=frame_scores,
        loadings=loadings.reset_index(drop=True),
        centroids=centroids,
        validation=pd.DataFrame(validation_rows),
    )


def _run_bounds(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Return half-open ``(start, stop, label)`` runs."""
    if labels.size == 0:
        return []
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    stops = np.r_[starts[1:], labels.size]
    return [
        (int(start), int(stop), int(labels[start]))
        for start, stop in zip(starts, stops)
    ]


def suppress_short_recrossings(
    labels: Sequence[int], *, min_dwell_rows: int = 1
) -> np.ndarray:
    """Collapse short A-B-A recrossings while preserving genuine A-B-C transfers."""
    values = np.asarray(labels, dtype=int).copy()
    if min_dwell_rows <= 1:
        return values
    changed = True
    while changed:
        changed = False
        runs = _run_bounds(values)
        for index in range(1, len(runs) - 1):
            start, stop, _ = runs[index]
            previous = runs[index - 1][2]
            following = runs[index + 1][2]
            if stop - start < min_dwell_rows and previous == following:
                values[start:stop] = previous
                changed = True
                break
    return values


def extract_transition_events(
    frame_table: pd.DataFrame,
    *,
    label_col: str = "macro_label",
    min_dwell_rows: int = 1,
) -> pd.DataFrame:
    """Enumerate direct macrostate changes after optional recrossing suppression."""
    validate_frame_table(frame_table)
    rows: list[dict] = []
    event_id = 0
    for traj_id, group in frame_table.groupby("traj_id", sort=False):
        ordered = group.sort_values("frame")
        labels = suppress_short_recrossings(
            ordered[label_col].to_numpy(dtype=int),
            min_dwell_rows=min_dwell_rows,
        )
        runs = _run_bounds(labels)
        for left, right in zip(runs[:-1], runs[1:]):
            _, left_stop, from_state = left
            right_start, right_stop, to_state = right
            boundary = right_start
            row = ordered.iloc[boundary]
            rows.append(
                {
                    "event_id": event_id,
                    "traj_id": str(traj_id),
                    "from_state": from_state,
                    "to_state": to_state,
                    "frame": int(row["frame"]),
                    "time_ps": float(row["time_ps"]),
                    "from_dwell_rows": int(left_stop - left[0]),
                    "to_dwell_rows": int(right_stop - right_start),
                }
            )
            event_id += 1
    return pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "traj_id",
            "from_state",
            "to_state",
            "frame",
            "time_ps",
            "from_dwell_rows",
            "to_dwell_rows",
        ],
    )


def transition_window_statistics(
    frame_table: pd.DataFrame,
    events: pd.DataFrame,
    *,
    feature_columns: Optional[Sequence[str]] = None,
    window_rows: int = 25,
    bootstrap_samples: int = 500,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate event-level and directed-transition CV changes.

    Returns ``(event_deltas, aggregate_signatures, aligned_traces)``.
    """
    validate_frame_table(frame_table)
    features = list(feature_columns or imamura_feature_columns(frame_table))
    missing = set(features) - set(frame_table.columns)
    if missing:
        raise ValueError(f"Transition features missing from frame table: {sorted(missing)}")
    if window_rows < 1:
        raise ValueError("window_rows must be at least 1")

    global_scale = (
        frame_table[features]
        .replace([np.inf, -np.inf], np.nan)
        .std(ddof=0)
        .replace(0.0, np.nan)
    )
    event_rows: list[dict] = []
    trace_rows: list[dict] = []

    grouped = {
        str(traj_id): group.sort_values("frame").reset_index(drop=True)
        for traj_id, group in frame_table.groupby("traj_id", sort=False)
    }
    for event in events.itertuples(index=False):
        group = grouped.get(str(event.traj_id))
        if group is None:
            continue
        positions = np.flatnonzero(group["frame"].to_numpy(dtype=int) == int(event.frame))
        if not positions.size:
            continue
        boundary = int(positions[0])
        pre = group.iloc[max(0, boundary - window_rows) : boundary]
        post = group.iloc[boundary : min(len(group), boundary + window_rows)]
        if pre.empty or post.empty:
            continue
        pre_mean = pre[features].mean(numeric_only=True)
        post_mean = post[features].mean(numeric_only=True)
        delta = post_mean - pre_mean
        boundary_dt_ps = float(post.iloc[0]["time_ps"]) - float(
            pre.iloc[-1]["time_ps"]
        )
        if not np.isfinite(boundary_dt_ps) or boundary_dt_ps <= 0:
            boundary_dt_ps = 1.0
        for feature in features:
            scale = float(global_scale.get(feature, np.nan))
            raw_delta = float(delta[feature])
            boundary_jump = float(post.iloc[0][feature] - pre.iloc[-1][feature])
            window_values = pd.concat([pre[feature], post[feature]]).to_numpy(
                dtype=float
            )
            adjacent_changes = np.diff(window_values)
            if adjacent_changes.size and np.isfinite(adjacent_changes).any():
                peak_index = int(np.nanargmax(np.abs(adjacent_changes)))
                peak_abs_change = float(abs(adjacent_changes[peak_index]))
                # A difference at index i ends at row i+1; the first post row is 0.
                peak_relative_row = int(peak_index + 1 - len(pre))
            else:
                peak_abs_change = np.nan
                peak_relative_row = 0
            event_rows.append(
                {
                    "event_id": int(event.event_id),
                    "traj_id": str(event.traj_id),
                    "from_state": int(event.from_state),
                    "to_state": int(event.to_state),
                    "frame": int(event.frame),
                    "time_ps": float(event.time_ps),
                    "feature": feature,
                    "feature_block": (
                        "central_ring"
                        if feature.startswith("v1_")
                        else "endpoint"
                        if feature.startswith("v4_")
                        else "gsa"
                    ),
                    "pre_mean": float(pre_mean[feature]),
                    "post_mean": float(post_mean[feature]),
                    "delta": raw_delta,
                    "standardized_delta": raw_delta / scale
                    if np.isfinite(scale) and scale > 0
                    else np.nan,
                    "boundary_jump": boundary_jump,
                    "boundary_slope_per_ps": boundary_jump / boundary_dt_ps,
                    "peak_abs_adjacent_change": peak_abs_change,
                    "peak_relative_row": peak_relative_row,
                }
            )
        trace = group.iloc[
            max(0, boundary - window_rows) : min(len(group), boundary + window_rows + 1)
        ]
        for position, (_, row) in enumerate(trace.iterrows()):
            relative = position - min(window_rows, boundary)
            for feature in features:
                trace_rows.append(
                    {
                        "event_id": int(event.event_id),
                        "traj_id": str(event.traj_id),
                        "from_state": int(event.from_state),
                        "to_state": int(event.to_state),
                        "relative_row": int(relative),
                        "feature": feature,
                        "value": float(row[feature]),
                    }
                )

    event_deltas = pd.DataFrame(event_rows)
    traces = pd.DataFrame(trace_rows)
    if event_deltas.empty:
        aggregate_columns = [
            "from_state",
            "to_state",
            "feature",
            "feature_block",
            "n_events",
            "mean_delta",
            "mean_standardized_delta",
            "consistent_direction_fraction",
            "mean_boundary_slope_per_ps",
            "mean_peak_abs_adjacent_change",
            "median_peak_relative_row",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
        ]
        return event_deltas, pd.DataFrame(columns=aggregate_columns), traces

    rng = np.random.default_rng(random_state)
    aggregate_rows: list[dict] = []
    keys = ["from_state", "to_state", "feature", "feature_block"]
    for key, group in event_deltas.groupby(keys, sort=True):
        values = group["standardized_delta"].dropna().to_numpy(dtype=float)
        if values.size:
            direction = np.sign(float(values.mean()))
            consistency = float(np.mean(np.sign(values) == direction))
            if bootstrap_samples > 0:
                means = np.array(
                    [
                        rng.choice(values, size=values.size, replace=True).mean()
                        for _ in range(bootstrap_samples)
                    ]
                )
                ci_low, ci_high = np.quantile(means, [0.025, 0.975])
            else:
                ci_low = ci_high = np.nan
        else:
            consistency = ci_low = ci_high = np.nan
        aggregate_rows.append(
            {
                "from_state": int(key[0]),
                "to_state": int(key[1]),
                "feature": key[2],
                "feature_block": key[3],
                "n_events": int(group["event_id"].nunique()),
                "mean_delta": float(group["delta"].mean()),
                "mean_standardized_delta": float(
                    group["standardized_delta"].mean()
                ),
                "consistent_direction_fraction": consistency,
                "mean_boundary_slope_per_ps": float(
                    group["boundary_slope_per_ps"].mean()
                ),
                "mean_peak_abs_adjacent_change": float(
                    group["peak_abs_adjacent_change"].mean()
                ),
                "median_peak_relative_row": float(
                    group["peak_relative_row"].median()
                ),
                "bootstrap_ci_low": float(ci_low),
                "bootstrap_ci_high": float(ci_high),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate["abs_standardized_delta"] = aggregate[
        "mean_standardized_delta"
    ].abs()
    aggregate = aggregate.sort_values(
        ["from_state", "to_state", "abs_standardized_delta"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    return event_deltas, aggregate, traces


def expand_changepoint_segments(segments: pd.DataFrame) -> pd.DataFrame:
    """Expand inclusive changepoint segments onto their original sampled frames."""
    required = {
        "traj_id",
        "start_frame",
        "end_frame",
        "n_frames",
        "cluster_label",
    }
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"Changepoint segments missing columns: {sorted(missing)}")
    if "group" in segments and (segments["group"] != "combined").any():
        segments = segments[segments["group"] == "combined"].copy()

    rows: list[dict] = []
    for segment in segments.itertuples(index=False):
        count = max(1, int(segment.n_frames))
        start = int(segment.start_frame)
        stop = int(segment.end_frame)
        frames = np.rint(np.linspace(start, stop, count)).astype(int)
        if np.unique(frames).size != frames.size:
            raise ValueError(
                f"Cannot reconstruct unique sampled frames for "
                f"{segment.traj_id} segment {segment.segment_id}"
            )
        for frame in frames:
            rows.append(
                {
                    "traj_id": str(segment.traj_id),
                    "frame": int(frame),
                    "changepoint_segment_id": int(segment.segment_id),
                    "changepoint_label": int(segment.cluster_label),
                }
            )
    expanded = pd.DataFrame(rows)
    if not expanded.empty and expanded.duplicated(["traj_id", "frame"]).any():
        raise ValueError("Expanded changepoint segments overlap on sampled frames")
    return expanded


def align_state_labels(
    imamura_frames: pd.DataFrame, changepoint_frames: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join Imamura and changepoint labels on exact trajectory/frame keys."""
    required = {"traj_id", "frame", "macro_label"}
    missing = required - set(imamura_frames.columns)
    if missing:
        raise ValueError(f"Imamura frame table missing columns: {sorted(missing)}")
    aligned = imamura_frames[
        ["traj_id", "frame", "time_ps", "macro_label"]
    ].merge(changepoint_frames, on=["traj_id", "frame"], how="inner")
    if aligned.empty:
        imamura_ids = sorted(imamura_frames["traj_id"].astype(str).unique())
        changepoint_ids = sorted(changepoint_frames["traj_id"].astype(str).unique())
        raise ValueError(
            "No exact trajectory/frame overlap between Imamura and changepoint "
            f"tables. Imamura traj_id examples={imamura_ids[:5]}, "
            f"changepoint examples={changepoint_ids[:5]}"
        )
    return aligned


def state_agreement_metrics(aligned: pd.DataFrame) -> pd.DataFrame:
    """Compute label-permutation-invariant state agreement metrics."""
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        v_measure_score,
    )

    imamura = aligned["macro_label"].to_numpy(dtype=int)
    changepoint = aligned["changepoint_label"].to_numpy(dtype=int)
    return pd.DataFrame(
        [
            {
                "n_aligned_frames": len(aligned),
                "n_trajectories": aligned["traj_id"].nunique(),
                "adjusted_rand_index": adjusted_rand_score(imamura, changepoint),
                "normalized_mutual_info": normalized_mutual_info_score(
                    imamura, changepoint
                ),
                "v_measure": v_measure_score(imamura, changepoint),
            }
        ]
    )


def state_contingency(aligned: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate Imamura and changepoint state occupancy."""
    table = pd.crosstab(
        aligned["macro_label"],
        aligned["changepoint_label"],
        rownames=["macro_label"],
        colnames=["changepoint_label"],
    )
    return table


def hungarian_state_mapping(contingency: pd.DataFrame) -> pd.DataFrame:
    """Map state labels for display only by maximizing contingency overlap."""
    from scipy.optimize import linear_sum_assignment

    if contingency.empty:
        return pd.DataFrame(
            columns=["macro_label", "changepoint_label", "overlap_frames"]
        )
    rows, columns = linear_sum_assignment(-contingency.to_numpy())
    return pd.DataFrame(
        {
            "macro_label": [int(contingency.index[row]) for row in rows],
            "changepoint_label": [
                int(contingency.columns[column]) for column in columns
            ],
            "overlap_frames": [
                int(contingency.iat[row, column])
                for row, column in zip(rows, columns)
            ],
        }
    )


def extract_state_boundaries(
    frame_table: pd.DataFrame, *, label_col: str = "macro_label"
) -> pd.DataFrame:
    """Extract rows where a framewise state assignment changes."""
    rows: list[dict] = []
    for traj_id, group in frame_table.groupby("traj_id", sort=False):
        ordered = group.sort_values("frame")
        labels = ordered[label_col].to_numpy(dtype=int)
        indices = np.flatnonzero(labels[1:] != labels[:-1]) + 1
        for index in indices:
            row = ordered.iloc[int(index)]
            rows.append(
                {
                    "traj_id": str(traj_id),
                    "frame": int(row["frame"]),
                    "time_ps": float(row["time_ps"]),
                    "from_state": int(labels[index - 1]),
                    "to_state": int(labels[index]),
                }
            )
    return pd.DataFrame(rows)


def compare_boundaries(
    imamura_boundaries: pd.DataFrame,
    changepoint_breakpoints: pd.DataFrame,
    *,
    tolerance_ps: float = 500.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Greedily match boundaries one-to-one within a physical time tolerance."""
    breakpoints = changepoint_breakpoints.copy()
    if "group" in breakpoints:
        breakpoints = breakpoints[breakpoints["group"] == "combined"]
    matches: list[dict] = []
    summary: list[dict] = []
    trajectory_ids = sorted(
        set(imamura_boundaries.get("traj_id", pd.Series(dtype=str)).astype(str))
        | set(breakpoints.get("traj_id", pd.Series(dtype=str)).astype(str))
    )
    for traj_id in trajectory_ids:
        imamura = imamura_boundaries[
            imamura_boundaries["traj_id"].astype(str) == traj_id
        ].sort_values("time_ps")
        cp = breakpoints[
            breakpoints["traj_id"].astype(str) == traj_id
        ].sort_values("time_ps")
        available = set(range(len(cp)))
        matched = 0
        for imamura_row in imamura.itertuples(index=False):
            candidates = [
                (
                    index,
                    abs(float(cp.iloc[index]["time_ps"]) - float(imamura_row.time_ps)),
                )
                for index in available
            ]
            if not candidates:
                continue
            index, offset = min(candidates, key=lambda item: item[1])
            if offset <= tolerance_ps:
                cp_row = cp.iloc[index]
                available.remove(index)
                matched += 1
                matches.append(
                    {
                        "traj_id": traj_id,
                        "imamura_frame": int(imamura_row.frame),
                        "imamura_time_ps": float(imamura_row.time_ps),
                        "from_state": int(imamura_row.from_state),
                        "to_state": int(imamura_row.to_state),
                        "changepoint_frame": int(cp_row["frame"]),
                        "changepoint_time_ps": float(cp_row["time_ps"]),
                        "offset_ps": float(cp_row["time_ps"])
                        - float(imamura_row.time_ps),
                    }
                )
        n_imamura = len(imamura)
        n_cp = len(cp)
        precision = matched / n_imamura if n_imamura else np.nan
        recall = matched / n_cp if n_cp else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision)
            and np.isfinite(recall)
            and precision + recall > 0
            else np.nan
        )
        summary.append(
            {
                "traj_id": traj_id,
                "n_imamura_boundaries": n_imamura,
                "n_changepoints": n_cp,
                "n_matched": matched,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_abs_offset_ps": float(
                    np.mean(
                        [
                            abs(row["offset_ps"])
                            for row in matches
                            if row["traj_id"] == traj_id
                        ]
                    )
                )
                if matched
                else np.nan,
            }
        )
    return pd.DataFrame(matches), pd.DataFrame(summary)


def apply_trajectory_id_map(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    source_col: str = "imamura_traj_id",
    target_col: str = "changepoint_traj_id",
) -> pd.DataFrame:
    """Apply an explicit trajectory-ID map; unmapped IDs are left unchanged."""
    required = {source_col, target_col}
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Trajectory-ID map missing columns: {sorted(missing)}")
    if mapping[source_col].duplicated().any():
        raise ValueError(f"Trajectory-ID map has duplicate {source_col} values")
    lookup = dict(
        zip(mapping[source_col].astype(str), mapping[target_col].astype(str))
    )
    result = df.copy()
    result["traj_id"] = result["traj_id"].astype(str).map(
        lambda value: lookup.get(value, value)
    )
    return result


def trace_recorded_transition_pairs(
    frame_table: pd.DataFrame,
    events: pd.DataFrame,
    signatures: pd.DataFrame,
    pair_trace: Any,
    pair_index_map: pd.DataFrame,
    *,
    window_rows: int = 25,
    top_features: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join transition CV ranks to pair identities recorded during extraction."""
    validate_frame_table(frame_table)
    ordered = frame_table.reset_index(drop=True)
    if len(pair_trace.frames) != len(ordered):
        raise ValueError("Pair trace length does not match frame table")
    if not np.array_equal(
        pair_trace.traj_ids.astype(str),
        ordered["traj_id"].astype(str).to_numpy(),
    ) or not np.array_equal(
        pair_trace.frames, ordered["frame"].to_numpy(dtype=np.int64)
    ):
        raise ValueError(
            "Pair trace trajectory/frame rows do not align with frame table"
        )
    required_map = {"feature_block", "pair_index", "bead_i", "bead_j"}
    missing_map = required_map - set(pair_index_map.columns)
    if missing_map:
        raise ValueError(f"Pair index map missing columns: {sorted(missing_map)}")

    selected = _selected_features_by_transition(
        signatures, top_features=top_features
    )
    map_lookup = {
        (str(row["feature_block"]), int(row["pair_index"])): row.to_dict()
        for _, row in pair_index_map.iterrows()
    }
    row_lookup = {
        (str(row.traj_id), int(row.frame)): index
        for index, row in enumerate(
            ordered[["traj_id", "frame"]].itertuples(index=False)
        )
    }
    detail_rows: list[dict[str, Any]] = []
    for event in events.itertuples(index=False):
        features = selected.get((int(event.from_state), int(event.to_state)), [])
        boundary = row_lookup.get((str(event.traj_id), int(event.frame)))
        if boundary is None or not features:
            continue
        traj_id = str(event.traj_id)
        trajectory_rows = np.flatnonzero(
            ordered["traj_id"].astype(str).to_numpy() == traj_id
        )
        local_positions = np.flatnonzero(trajectory_rows == boundary)
        if not local_positions.size:
            continue
        local_boundary = int(local_positions[0])
        selected_rows = trajectory_rows[
            max(0, local_boundary - window_rows) :
            min(len(trajectory_rows), local_boundary + window_rows + 1)
        ]
        for row_index in selected_rows:
            row = ordered.iloc[int(row_index)]
            relative_row = int(
                np.flatnonzero(trajectory_rows == row_index)[0] - local_boundary
            )
            for feature in features:
                block = "v1" if feature.startswith("v1_") else "v4"
                rank = int(feature.split("_", 1)[1]) - 1
                identities = (
                    pair_trace.v1_pair_indices
                    if block == "v1"
                    else pair_trace.v4_pair_indices
                )
                if rank < 0 or rank >= identities.shape[1]:
                    raise ValueError(
                        f"{feature!r} is outside recorded {block} trace width"
                    )
                pair_index = int(identities[int(row_index), rank])
                metadata = map_lookup.get((block, pair_index))
                if metadata is None:
                    raise ValueError(
                        f"No pair map row for {block} pair_index={pair_index}"
                    )
                detail_rows.append(
                    {
                        "event_id": int(event.event_id),
                        "traj_id": traj_id,
                        "from_state": int(event.from_state),
                        "to_state": int(event.to_state),
                        "event_frame": int(event.frame),
                        "event_time_ps": float(event.time_ps),
                        "frame": int(row["frame"]),
                        "frame_time_ps": float(row["time_ps"]),
                        "relative_row": relative_row,
                        "phase": "pre" if relative_row < 0 else "post",
                        "feature": feature,
                        "feature_rank": rank + 1,
                        "distance": float(row[feature]),
                        **metadata,
                    }
                )
    detailed = pd.DataFrame(detail_rows)
    if detailed.empty:
        return detailed, pd.DataFrame()

    metadata_columns = [
        column
        for column in pair_index_map.columns
        if column not in {"feature_block", "pair_index"}
    ]
    pair_keys = [
        "from_state",
        "to_state",
        "feature",
        "feature_block",
        "pair_index",
        *metadata_columns,
    ]
    # Preserve order while removing duplicate key names such as bead_i/bead_j.
    pair_keys = list(dict.fromkeys(pair_keys))
    phase_stats = (
        detailed.groupby(pair_keys + ["phase"], dropna=False)
        .agg(
            mean_distance=("distance", "mean"),
            rank_occupancies=("distance", "size"),
        )
        .unstack("phase")
    )
    phase_stats.columns = [
        f"{phase}_{metric}" for metric, phase in phase_stats.columns
    ]
    summary = phase_stats.reset_index()
    event_counts = (
        detailed.groupby(pair_keys, dropna=False)["event_id"]
        .nunique()
        .rename("n_events")
        .reset_index()
    )
    summary = summary.merge(event_counts, on=pair_keys, how="left")
    for phase in ("pre", "post"):
        column = f"{phase}_rank_occupancies"
        if column not in summary:
            summary[column] = 0
        else:
            summary[column] = summary[column].fillna(0)
    denominators = (
        detailed.groupby(["from_state", "to_state", "feature", "phase"])
        .size()
        .unstack("phase", fill_value=0)
        .reset_index()
        .rename(
            columns={
                "pre": "n_pre_feature_observations",
                "post": "n_post_feature_observations",
            }
        )
    )
    summary = summary.merge(
        denominators, on=["from_state", "to_state", "feature"], how="left"
    )
    summary["pre_rank_occupancy_fraction"] = (
        summary["pre_rank_occupancies"]
        / summary["n_pre_feature_observations"].replace(0, np.nan)
    )
    summary["post_rank_occupancy_fraction"] = (
        summary["post_rank_occupancies"]
        / summary["n_post_feature_observations"].replace(0, np.nan)
    )
    if {"pre_mean_distance", "post_mean_distance"} <= set(summary):
        summary["post_minus_pre_distance"] = (
            summary["post_mean_distance"] - summary["pre_mean_distance"]
        )
    return detailed, summary.sort_values(
        ["from_state", "to_state", "feature", "post_rank_occupancy_fraction"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def exact_ranked_pair(
    type1_positions: np.ndarray,
    type4_positions: np.ndarray,
    feature: str,
) -> tuple[str, int, int, int, float]:
    """Return the exact bead pair occupying a sorted CV slot in one frame.

    Returns ``(bead_kind, zero_based_rank, bead_i, bead_j, distance)``.
    """
    from itertools import combinations

    if feature.startswith("v1_"):
        bead_kind = "type1"
        positions = np.asarray(type1_positions, dtype=np.float64)
    elif feature.startswith("v4_"):
        bead_kind = "type4"
        positions = np.asarray(type4_positions, dtype=np.float64)
    else:
        raise ValueError(f"Not an Imamura distance feature: {feature!r}")
    try:
        rank = int(feature.split("_", 1)[1]) - 1
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid Imamura feature name: {feature!r}") from exc

    pairs = [
        (float(np.linalg.norm(positions[i] - positions[j])), int(i), int(j))
        for i, j in combinations(range(len(positions)), 2)
    ]
    pairs.sort(key=lambda item: item[0], reverse=True)
    if rank < 0 or rank >= len(pairs):
        raise ValueError(
            f"{feature!r} requests rank {rank + 1}, but {bead_kind} has "
            f"only {len(pairs)} pair distances"
        )
    distance, bead_i, bead_j = pairs[rank]
    return bead_kind, rank, bead_i, bead_j, distance


def _bead_atom_indices(
    bead_spec: ImamuraBeadSpec,
    bead_kind: str,
    bead_index: int,
    universe: Any,
) -> tuple[int, ...]:
    if bead_spec.uses_ring_centroids:
        if bead_kind == "type1":
            rings = bead_spec.type1_ring_groups or []
            return (
                tuple(int(value) for value in rings[bead_index])
                if bead_index < len(rings)
                else ()
            )
        type4 = bead_spec.type4_atom_ids or ()
        return (int(type4[bead_index]),) if bead_index < len(type4) else ()

    selection = (
        bead_spec.type1_selection
        if bead_kind == "type1"
        else bead_spec.type4_selection
    )
    if not selection:
        return ()
    atoms = universe.select_atoms(selection)
    if bead_index >= len(atoms):
        return ()
    return (int(atoms[bead_index].ix),)


def _bead_metadata(
    universe: Any,
    bead_spec: ImamuraBeadSpec,
    bead_kind: str,
    bead_index: int,
) -> dict[str, Any]:
    indices = _bead_atom_indices(bead_spec, bead_kind, bead_index, universe)
    if not indices:
        return {
            "atom_indices": "",
            "atom_ids": "",
            "atom_names": "",
            "resids": "",
            "resnames": "",
            "monomer_index": np.nan,
        }
    atoms = universe.atoms[list(indices)]
    resids = [int(value) for value in atoms.resids]
    monomer_index: float = np.nan
    if bead_spec.uses_ring_centroids:
        if bead_kind == "type1":
            monomer_index = float(bead_index)
        else:
            ring_groups = bead_spec.type1_ring_groups or []
            atom_resids = set(resids)
            for index, ring in enumerate(ring_groups):
                ring_resids = set(
                    int(value) for value in universe.atoms[list(ring)].resids
                )
                if atom_resids & ring_resids:
                    monomer_index = float(index)
                    break
    return {
        "atom_indices": ";".join(str(value) for value in indices),
        "atom_ids": ";".join(str(int(value)) for value in atoms.ids),
        "atom_names": ";".join(str(value) for value in atoms.names),
        "resids": ";".join(str(value) for value in resids),
        "resnames": ";".join(str(value) for value in atoms.resnames),
        "monomer_index": monomer_index,
    }


def _selected_features_by_transition(
    signatures: pd.DataFrame,
    *,
    top_features: int,
) -> dict[tuple[int, int], list[str]]:
    required = {
        "from_state",
        "to_state",
        "feature",
        "abs_standardized_delta",
    }
    missing = required - set(signatures.columns)
    if missing:
        raise ValueError(f"Transition signatures missing columns: {sorted(missing)}")
    selected: dict[tuple[int, int], list[str]] = {}
    ordered = signatures.sort_values(
        ["from_state", "to_state", "abs_standardized_delta"],
        ascending=[True, True, False],
    )
    for key, group in ordered.groupby(["from_state", "to_state"], sort=False):
        features = [
            str(value)
            for value in group["feature"].head(max(1, top_features)).tolist()
            if str(value).startswith(("v1_", "v4_"))
        ]
        selected[(int(key[0]), int(key[1]))] = features
    return selected


def trace_exact_transition_pairs(
    frame_table: pd.DataFrame,
    events: pd.DataFrame,
    signatures: pd.DataFrame,
    *,
    topology: Path | str,
    trajectory_paths: Mapping[str, Path | str],
    bead_spec: ImamuraBeadSpec,
    trajectory_format: Optional[str] = None,
    window_rows: int = 25,
    top_features: int = 10,
    distance_tolerance: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trace the physical bead pair occupying each selected sorted CV rank.

    The trajectory is visited only at rows in each event window. The detailed
    output identifies the dynamic rank occupant per frame; the summary reports
    how often each physical pair occupied that rank before and after a directed
    state transfer.
    """
    import MDAnalysis as mda

    validate_frame_table(frame_table)
    if window_rows < 1:
        raise ValueError("window_rows must be at least 1")
    selected = _selected_features_by_transition(
        signatures, top_features=top_features
    )
    trajectory_lookup = {
        str(key): Path(value) for key, value in trajectory_paths.items()
    }
    missing_trajectories = sorted(
        set(events["traj_id"].astype(str)) - set(trajectory_lookup)
    )
    if missing_trajectories:
        raise ValueError(
            "No trajectory path for exact pair tracing: "
            f"{missing_trajectories[:10]}"
        )

    detailed_rows: list[dict[str, Any]] = []
    for traj_id, trajectory_events in events.groupby("traj_id", sort=False):
        traj_id = str(traj_id)
        trajectory_path = trajectory_lookup[traj_id]
        kwargs = {"format": trajectory_format} if trajectory_format else {}
        universe = mda.Universe(str(topology), str(trajectory_path), **kwargs)
        ordered = (
            frame_table[frame_table["traj_id"].astype(str) == traj_id]
            .sort_values("frame")
            .reset_index(drop=True)
        )
        frame_to_position = {
            int(frame): index
            for index, frame in enumerate(ordered["frame"].to_numpy(dtype=int))
        }
        metadata_cache: dict[tuple[str, int], dict[str, Any]] = {}
        geometry_cache: dict[
            int, tuple[np.ndarray, np.ndarray, float]
        ] = {}

        for event in trajectory_events.itertuples(index=False):
            features = selected.get(
                (int(event.from_state), int(event.to_state)), []
            )
            if not features:
                continue
            boundary_position = frame_to_position.get(int(event.frame))
            if boundary_position is None:
                continue
            start = max(0, boundary_position - window_rows)
            stop = min(len(ordered), boundary_position + window_rows + 1)
            window = ordered.iloc[start:stop]
            for row_position, row in window.iterrows():
                frame = int(row["frame"])
                relative_row = int(row_position - boundary_position)
                if frame < 0 or frame >= len(universe.trajectory):
                    raise ValueError(
                        f"Frame {frame} for {traj_id} is outside trajectory "
                        f"range 0..{len(universe.trajectory) - 1}"
                    )
                if frame not in geometry_cache:
                    universe.trajectory[frame]
                    type1, type4 = bead_positions_from_spec(universe, bead_spec)
                    geometry_cache[frame] = (
                        np.asarray(type1, dtype=np.float64).copy(),
                        np.asarray(type4, dtype=np.float64).copy(),
                        float(universe.trajectory.ts.time),
                    )
                type1, type4, trajectory_time_ps = geometry_cache[frame]
                for feature in features:
                    bead_kind, rank, bead_i, bead_j, distance = exact_ranked_pair(
                        type1, type4, feature
                    )
                    key_i = (bead_kind, bead_i)
                    key_j = (bead_kind, bead_j)
                    if key_i not in metadata_cache:
                        metadata_cache[key_i] = _bead_metadata(
                            universe, bead_spec, bead_kind, bead_i
                        )
                    if key_j not in metadata_cache:
                        metadata_cache[key_j] = _bead_metadata(
                            universe, bead_spec, bead_kind, bead_j
                        )
                    meta_i = metadata_cache[key_i]
                    meta_j = metadata_cache[key_j]
                    detailed_rows.append(
                        {
                            "event_id": int(event.event_id),
                            "traj_id": traj_id,
                            "from_state": int(event.from_state),
                            "to_state": int(event.to_state),
                            "event_frame": int(event.frame),
                            "event_time_ps": float(event.time_ps),
                            "frame": frame,
                            "frame_time_ps": float(row["time_ps"]),
                            "trajectory_time_ps": trajectory_time_ps,
                            "relative_row": relative_row,
                            "phase": "pre" if relative_row < 0 else "post",
                            "feature": feature,
                            "feature_rank": rank + 1,
                            "bead_kind": bead_kind,
                            "bead_i": bead_i,
                            "bead_j": bead_j,
                            "distance": distance,
                            "saved_feature_distance": float(row[feature]),
                            "distance_residual": distance - float(row[feature]),
                            **{
                                f"bead_i_{name}": value
                                for name, value in meta_i.items()
                            },
                            **{
                                f"bead_j_{name}": value
                                for name, value in meta_j.items()
                            },
                        }
                    )

    detailed = pd.DataFrame(detailed_rows)
    if detailed.empty:
        return detailed, pd.DataFrame()
    max_residual = float(detailed["distance_residual"].abs().max())
    if max_residual > distance_tolerance:
        worst = detailed.loc[detailed["distance_residual"].abs().idxmax()]
        raise ValueError(
            "Exact pair tracing does not reproduce the saved sorted-distance "
            f"CVs (max residual={max_residual:.6g} Å, "
            f"tolerance={distance_tolerance:.6g} Å; traj_id={worst['traj_id']}, "
            f"frame={int(worst['frame'])}, feature={worst['feature']}). "
            "Check that topology, trajectories, bead_spec.json, and frame "
            "indices come from the same Imamura extraction."
        )

    pair_keys = [
        "from_state",
        "to_state",
        "feature",
        "bead_kind",
        "bead_i",
        "bead_j",
        "bead_i_monomer_index",
        "bead_j_monomer_index",
        "bead_i_atom_indices",
        "bead_j_atom_indices",
        "bead_i_atom_ids",
        "bead_j_atom_ids",
        "bead_i_atom_names",
        "bead_j_atom_names",
        "bead_i_resids",
        "bead_j_resids",
        "bead_i_resnames",
        "bead_j_resnames",
    ]
    phase_means = (
        detailed.groupby(pair_keys + ["phase"], dropna=False)["distance"]
        .mean()
        .unstack("phase")
        .reset_index()
        .rename(columns={"pre": "pre_mean_distance", "post": "post_mean_distance"})
    )
    counts = (
        detailed.groupby(pair_keys, dropna=False)
        .agg(
            n_rank_occupancies=("distance", "size"),
            n_events=("event_id", "nunique"),
        )
        .reset_index()
    )
    phase_counts = (
        detailed.groupby(pair_keys + ["phase"], dropna=False)
        .size()
        .unstack("phase", fill_value=0)
        .reset_index()
        .rename(
            columns={
                "pre": "pre_rank_occupancies",
                "post": "post_rank_occupancies",
            }
        )
    )
    denominators = (
        detailed.groupby(["from_state", "to_state", "feature", "phase"])
        .size()
        .unstack("phase", fill_value=0)
        .reset_index()
        .rename(
            columns={
                "pre": "n_pre_feature_observations",
                "post": "n_post_feature_observations",
            }
        )
    )
    summary = (
        counts.merge(phase_counts, on=pair_keys, how="left")
        .merge(phase_means, on=pair_keys, how="left")
        .merge(
            denominators,
            on=["from_state", "to_state", "feature"],
            how="left",
        )
    )
    summary["n_feature_observations"] = (
        summary["n_pre_feature_observations"]
        + summary["n_post_feature_observations"]
    )
    summary["rank_occupancy_fraction"] = (
        summary["n_rank_occupancies"] / summary["n_feature_observations"]
    )
    summary["pre_rank_occupancy_fraction"] = (
        summary["pre_rank_occupancies"]
        / summary["n_pre_feature_observations"].replace(0, np.nan)
    )
    summary["post_rank_occupancy_fraction"] = (
        summary["post_rank_occupancies"]
        / summary["n_post_feature_observations"].replace(0, np.nan)
    )
    if {"pre_mean_distance", "post_mean_distance"} <= set(summary.columns):
        summary["post_minus_pre_distance"] = (
            summary["post_mean_distance"] - summary["pre_mean_distance"]
        )
    summary = summary.sort_values(
        ["from_state", "to_state", "feature", "rank_occupancy_fraction"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)
    return detailed, summary

