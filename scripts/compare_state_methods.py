"""Compare PCA/tlICA Imamura states with combined changepoint states.

This script starts from an endpoint-style Imamura ``frame_states.csv`` (the
table must retain its raw ``v1_*``/``v4_*`` columns), creates matched PCA and
tlICA assignments, fits post-hoc LDA interpretations, and summarizes the CV
changes around every directed macrostate transfer.

Example
-------
python scripts/compare_state_methods.py \
    --imamura-frame-states output/imamura_endpoint/frame_states.csv \
    --output-dir output/state_comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.imamura_msm import load_imamura_pair_trace
from src.utils.state_comparison import (
    align_state_labels,
    apply_trajectory_id_map,
    build_imamura_variants,
    compare_boundaries,
    expand_changepoint_segments,
    extract_transition_events,
    fit_posthoc_lda,
    hungarian_state_mapping,
    imamura_feature_columns,
    state_agreement_metrics,
    state_contingency,
    trace_recorded_transition_pairs,
    transition_window_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create matched PCA/tlICA Imamura states, interpret them with LDA, "
            "and compare state transfers with combined changepoints."
        )
    )
    parser.add_argument("--imamura-frame-states", required=True, type=Path)
    parser.add_argument(
        "--changepoint-segments",
        type=Path,
        default=Path("output/changepoints/clusters/combined/segments_clustered.csv"),
    )
    parser.add_argument(
        "--changepoint-breakpoints",
        type=Path,
        default=Path("output/changepoints/all_breakpoints.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output/state_comparison")
    )
    parser.add_argument(
        "--trajectory-id-map",
        type=Path,
        default=None,
        help=(
            "Optional CSV with imamura_traj_id,changepoint_traj_id. "
            "IDs are never matched heuristically."
        ),
    )
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--n-micro", type=int, default=1500)
    parser.add_argument("--n-macro", type=int, default=22)
    parser.add_argument("--tlica-lag-ns", type=float, default=2.0)
    parser.add_argument("--tlica-regularization", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--min-dwell-rows",
        type=int,
        default=5,
        help="Collapse only short A-B-A recrossings below this row count",
    )
    parser.add_argument(
        "--window-rows",
        type=int,
        default=25,
        help="Rows before and after each state transfer used for CV deltas",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--boundary-tolerance-ps", type=float, default=500.0)
    parser.add_argument(
        "--top-features",
        type=int,
        default=20,
        help="Number of strongest CVs retained per directed transition",
    )
    parser.add_argument(
        "--write-aligned-traces",
        action="store_true",
        help="Write the potentially large event-aligned long-form CV table",
    )
    parser.add_argument(
        "--gsa-features-dir",
        type=Path,
        default=None,
        help="Optional directory of *_gsa_features.csv files for deformation profiles",
    )
    parser.add_argument("--max-timeline-plots", type=int, default=12)
    trace = parser.add_argument_group("recorded pair identities")
    trace.add_argument(
        "--pair-identity-trace",
        type=Path,
        default=None,
        help=(
            "pair_identity_trace.npz written by run_imamura_msm.py "
            "(default: sibling of --imamura-frame-states)"
        ),
    )
    trace.add_argument(
        "--pair-index-map",
        type=Path,
        default=None,
        help=(
            "pair_index_map.csv written by run_imamura_msm.py "
            "(default: sibling of --imamura-frame-states)"
        ),
    )
    trace.add_argument(
        "--exact-top-features",
        type=int,
        default=10,
        help="Top transition CV ranks joined to recorded physical pairs",
    )
    return parser.parse_args()


def _write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _write_indexed_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return path


def _top_transition_features(
    signatures: pd.DataFrame, top_n: int
) -> pd.DataFrame:
    if signatures.empty:
        return signatures
    return (
        signatures.sort_values(
            ["from_state", "to_state", "abs_standardized_delta"],
            ascending=[True, True, False],
        )
        .groupby(["from_state", "to_state"], group_keys=False)
        .head(max(1, top_n))
        .reset_index(drop=True)
    )


def _plot_contingency(table: pd.DataFrame, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(table.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=90)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_xlabel("Combined changepoint state")
    ax.set_ylabel("Imamura macrostate")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="Aligned sampled frames")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_lda(scores: pd.DataFrame, output_path: Path, title: str) -> None:
    ld_columns = [column for column in scores if column.startswith("LD")]
    if not ld_columns:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    x = scores[ld_columns[0]]
    y = scores[ld_columns[1]] if len(ld_columns) > 1 else np.zeros(len(scores))
    scatter = ax.scatter(
        x,
        y,
        c=scores["macro_label"],
        cmap="tab20",
        s=3,
        alpha=0.35,
        rasterized=True,
    )
    ax.set_xlabel(ld_columns[0])
    ax.set_ylabel(ld_columns[1] if len(ld_columns) > 1 else "0")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="Macrostate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_transition_signature(
    signatures: pd.DataFrame, output_path: Path, title: str
) -> None:
    if signatures.empty:
        return
    counts = (
        signatures[["from_state", "to_state", "n_events"]]
        .drop_duplicates()
        .sort_values("n_events", ascending=False)
    )
    dominant = counts.iloc[0]
    subset = signatures[
        (signatures["from_state"] == dominant["from_state"])
        & (signatures["to_state"] == dominant["to_state"])
    ].nlargest(20, "abs_standardized_delta")
    subset = subset.sort_values("mean_standardized_delta")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.25 * len(subset))))
    colors = [
        "tab:blue" if block == "central_ring" else "tab:orange"
        for block in subset["feature_block"]
    ]
    ax.barh(subset["feature"], subset["mean_standardized_delta"], color=colors)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean post − pre change (global SD)")
    ax.set_title(
        f"{title}: {int(dominant['from_state'])}→{int(dominant['to_state'])} "
        f"(n={int(dominant['n_events'])})"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_timelines(
    aligned: pd.DataFrame, output_dir: Path, method: str, maximum: int
) -> list[Path]:
    paths: list[Path] = []
    for traj_id, group in list(aligned.groupby("traj_id", sort=True))[:maximum]:
        ordered = group.sort_values("time_ps")
        fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
        axes[0].step(
            ordered["time_ps"], ordered["macro_label"], where="post", linewidth=1
        )
        axes[0].set_ylabel(f"{method.upper()} state")
        axes[1].step(
            ordered["time_ps"],
            ordered["changepoint_label"],
            where="post",
            linewidth=1,
            color="tab:orange",
        )
        axes[1].set_ylabel("CP state")
        axes[1].set_xlabel("Time (ps)")
        fig.suptitle(str(traj_id))
        fig.tight_layout()
        safe_id = "".join(
            char if char.isalnum() or char in "-_" else "_" for char in str(traj_id)
        )
        path = output_dir / f"timeline_{safe_id}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _load_gsa_features(directory: Path) -> pd.DataFrame:
    paths = sorted(directory.glob("*_gsa_features.csv"))
    if not paths:
        paths = sorted(directory.rglob("*_gsa_features.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_gsa_features.csv files under {directory}")
    parts = [pd.read_csv(path) for path in paths]
    result = pd.concat(parts, ignore_index=True)
    required = {"traj_id", "frame", "time_ps"}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"GSA feature tables missing columns: {sorted(missing)}")
    return result


def _event_delta_correlations(
    imamura_deltas: pd.DataFrame,
    gsa_deltas: pd.DataFrame,
    *,
    max_imamura_features: int = 10,
) -> pd.DataFrame:
    """Correlate endpoint/ring CV changes with deformation changes across events."""
    columns = [
        "from_state",
        "to_state",
        "imamura_feature",
        "gsa_feature",
        "n_events",
        "pearson_r",
        "abs_pearson_r",
    ]
    if imamura_deltas.empty or gsa_deltas.empty:
        return pd.DataFrame(columns=columns)
    deformation_terms = (
        "deformation",
        "rmsd",
        "pore",
        "gear",
        "contact",
        "interface",
        "octahedral",
        "gyration",
        "assembly_rg",
        "asphericity",
        "acylindricity",
    )
    gsa_deltas = gsa_deltas[
        gsa_deltas["feature"].str.lower().map(
            lambda name: any(term in name for term in deformation_terms)
        )
    ]
    rows: list[dict] = []
    for (from_state, to_state), imamura_group in imamura_deltas.groupby(
        ["from_state", "to_state"]
    ):
        gsa_group = gsa_deltas[
            (gsa_deltas["from_state"] == from_state)
            & (gsa_deltas["to_state"] == to_state)
        ]
        if gsa_group.empty:
            continue
        ranked = (
            imamura_group.groupby("feature")["standardized_delta"]
            .mean()
            .abs()
            .nlargest(max_imamura_features)
            .index
        )
        left = imamura_group[imamura_group["feature"].isin(ranked)].pivot_table(
            index="event_id", columns="feature", values="standardized_delta"
        )
        right = gsa_group.pivot_table(
            index="event_id", columns="feature", values="standardized_delta"
        )
        common = left.index.intersection(right.index)
        if len(common) < 3:
            continue
        left = left.loc[common]
        right = right.loc[common]
        for imamura_feature in left.columns:
            for gsa_feature in right.columns:
                pair = pd.concat(
                    [left[imamura_feature], right[gsa_feature]], axis=1
                ).dropna()
                if len(pair) < 3:
                    continue
                if pair.iloc[:, 0].std() == 0 or pair.iloc[:, 1].std() == 0:
                    continue
                correlation = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
                rows.append(
                    {
                        "from_state": int(from_state),
                        "to_state": int(to_state),
                        "imamura_feature": imamura_feature,
                        "gsa_feature": gsa_feature,
                        "n_events": len(pair),
                        "pearson_r": correlation,
                        "abs_pearson_r": abs(correlation),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["from_state", "to_state", "abs_pearson_r"],
        ascending=[True, True, False],
    )


def _analyze_gsa_windows(
    method_dir: Path,
    variant: pd.DataFrame,
    events: pd.DataFrame,
    imamura_deltas: pd.DataFrame,
    gsa_features: pd.DataFrame,
    *,
    window_rows: int,
    bootstrap_samples: int,
    random_state: int,
    top_features: int,
) -> list[Path]:
    merged = gsa_features.merge(
        variant[["traj_id", "frame", "macro_label"]],
        on=["traj_id", "frame"],
        how="inner",
    )
    if merged.empty:
        return []
    numeric = [
        column
        for column in merged.select_dtypes(include=[np.number]).columns
        if column not in {"frame", "time_ps", "macro_label"}
    ]
    if not numeric:
        return []
    # GSA is usually more coarsely sampled. Match each transition to the nearest
    # saved GSA row in the same trajectory before extracting row windows.
    mapped_rows: list[dict] = []
    for event in events.itertuples(index=False):
        group = merged[merged["traj_id"].astype(str) == str(event.traj_id)]
        if group.empty:
            continue
        nearest = group.iloc[
            np.abs(group["time_ps"].to_numpy(dtype=float) - float(event.time_ps)).argmin()
        ]
        row = event._asdict()
        row["frame"] = int(nearest["frame"])
        row["time_ps"] = float(nearest["time_ps"])
        mapped_rows.append(row)
    mapped = pd.DataFrame(mapped_rows)
    if mapped.empty:
        return []
    deltas, signatures, _ = transition_window_statistics(
        merged,
        mapped,
        feature_columns=numeric,
        window_rows=window_rows,
        bootstrap_samples=bootstrap_samples,
        random_state=random_state,
    )
    correlations = _event_delta_correlations(imamura_deltas, deltas)
    return [
        _write_csv(deltas, method_dir / "gsa_event_deltas.csv"),
        _write_csv(
            _top_transition_features(signatures, top_features),
            method_dir / "gsa_transition_signatures_top.csv",
        ),
        _write_csv(
            correlations,
            method_dir / "pi_proxy_vs_deformation_event_correlations.csv",
        ),
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_table = pd.read_csv(args.imamura_frame_states)
    mapping: Optional[pd.DataFrame] = None
    if args.trajectory_id_map is not None:
        mapping = pd.read_csv(args.trajectory_id_map)
        frame_table = apply_trajectory_id_map(frame_table, mapping)

    variants = build_imamura_variants(
        frame_table,
        n_components=args.n_components,
        n_microclusters=args.n_micro,
        n_macrostates=args.n_macro,
        tlica_lag_ns=args.tlica_lag_ns,
        tlica_regularization=args.tlica_regularization,
        random_state=args.random_state,
    )

    segments = pd.read_csv(args.changepoint_segments)
    breakpoints = pd.read_csv(args.changepoint_breakpoints)
    changepoint_frames = expand_changepoint_segments(segments)
    gsa_features: Optional[pd.DataFrame] = None
    if args.gsa_features_dir is not None:
        gsa_features = _load_gsa_features(args.gsa_features_dir)
    pair_trace_path = args.pair_identity_trace or args.imamura_frame_states.with_name(
        "pair_identity_trace.npz"
    )
    pair_map_path = args.pair_index_map or args.imamura_frame_states.with_name(
        "pair_index_map.csv"
    )
    pair_trace = None
    pair_index_map: Optional[pd.DataFrame] = None
    pair_trace_requested = (
        args.pair_identity_trace is not None or args.pair_index_map is not None
    )
    if pair_trace_path.is_file() and pair_map_path.is_file():
        pair_trace = load_imamura_pair_trace(pair_trace_path)
        pair_index_map = pd.read_csv(pair_map_path)
        if mapping is not None:
            lookup = dict(
                zip(
                    mapping["imamura_traj_id"].astype(str),
                    mapping["changepoint_traj_id"].astype(str),
                )
            )
            pair_trace.traj_ids = np.asarray(
                [lookup.get(value, value) for value in pair_trace.traj_ids.astype(str)]
            )
    elif pair_trace_requested or pair_trace_path.is_file() or pair_map_path.is_file():
        raise FileNotFoundError(
            "Recorded pair tracing requires both files: "
            f"{pair_trace_path} and {pair_map_path}"
        )

    manifest: dict[str, object] = {
        "inputs": {
            "imamura_frame_states": str(args.imamura_frame_states),
            "changepoint_segments": str(args.changepoint_segments),
            "changepoint_breakpoints": str(args.changepoint_breakpoints),
            "trajectory_id_map": (
                str(args.trajectory_id_map)
                if args.trajectory_id_map is not None
                else None
            ),
            "pair_identity_trace": (
                str(pair_trace_path) if pair_trace is not None else None
            ),
            "pair_index_map": (
                str(pair_map_path) if pair_index_map is not None else None
            ),
        },
        "parameters": {
            "n_components": args.n_components,
            "n_micro": args.n_micro,
            "n_macro": args.n_macro,
            "tlica_lag_ns": args.tlica_lag_ns,
            "min_dwell_rows": args.min_dwell_rows,
            "window_rows": args.window_rows,
            "boundary_tolerance_ps": args.boundary_tolerance_ps,
            "exact_top_features": args.exact_top_features,
        },
        "methods": {},
        "interpretation_limit": (
            "v1/v4 distances involving the pi system are geometric proxies; "
            "they do not establish pi-stacking orientation or causality."
        ),
    }

    method_outputs: dict[str, dict[str, object]] = {}
    for method, variant in variants.items():
        method_dir = args.output_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        written.append(_write_csv(variant, method_dir / "frame_states.csv"))

        lda = fit_posthoc_lda(variant, max_components=args.n_components)
        written.extend(
            [
                _write_csv(lda.frame_scores, method_dir / "lda_frame_scores.csv"),
                _write_csv(lda.loadings, method_dir / "lda_loadings.csv"),
                _write_csv(lda.centroids, method_dir / "lda_state_centroids.csv"),
                _write_csv(lda.validation, method_dir / "lda_validation.csv"),
            ]
        )
        _plot_lda(
            lda.frame_scores,
            method_dir / "lda_states.png",
            f"{method.upper()} macrostates — post-hoc LDA",
        )

        events = extract_transition_events(
            variant, min_dwell_rows=args.min_dwell_rows
        )
        event_deltas, signatures, traces = transition_window_statistics(
            variant,
            events,
            feature_columns=imamura_feature_columns(variant),
            window_rows=args.window_rows,
            bootstrap_samples=args.bootstrap_samples,
            random_state=args.random_state,
        )
        top_signatures = _top_transition_features(signatures, args.top_features)
        written.extend(
            [
                _write_csv(events, method_dir / "transition_events.csv"),
                _write_csv(event_deltas, method_dir / "transition_event_deltas.csv"),
                _write_csv(
                    top_signatures, method_dir / "transition_signatures_top.csv"
                ),
            ]
        )
        if args.write_aligned_traces:
            written.append(
                _write_csv(traces, method_dir / "transition_aligned_traces.csv")
            )
        _plot_transition_signature(
            signatures,
            method_dir / "dominant_transition_signature.png",
            method.upper(),
        )
        if pair_trace is not None and pair_index_map is not None:
            exact_trace, exact_summary = trace_recorded_transition_pairs(
                variant,
                events,
                signatures,
                pair_trace,
                pair_index_map,
                window_rows=args.window_rows,
                top_features=args.exact_top_features,
            )
            written.extend(
                [
                    _write_csv(
                        exact_trace,
                        method_dir / "exact_transition_pair_trace.csv",
                    ),
                    _write_csv(
                        exact_summary,
                        method_dir / "exact_transition_pair_summary.csv",
                    ),
                ]
            )

        aligned = align_state_labels(variant, changepoint_frames)
        agreement = state_agreement_metrics(aligned)
        contingency = state_contingency(aligned)
        state_map = hungarian_state_mapping(contingency)
        matches, boundary_summary = compare_boundaries(
            events,
            breakpoints,
            tolerance_ps=args.boundary_tolerance_ps,
        )
        supported_keys = {
            (str(row.traj_id), int(row.imamura_frame))
            for row in matches.itertuples(index=False)
        }
        events_supported = events.copy()
        events_supported["changepoint_supported"] = [
            (str(row.traj_id), int(row.frame)) in supported_keys
            for row in events.itertuples(index=False)
        ]
        written.extend(
            [
                _write_csv(aligned, method_dir / "aligned_frame_labels.csv"),
                _write_csv(agreement, method_dir / "state_agreement.csv"),
                _write_indexed_csv(contingency, method_dir / "state_contingency.csv"),
                _write_csv(state_map, method_dir / "state_mapping_display_only.csv"),
                _write_csv(matches, method_dir / "boundary_matches.csv"),
                _write_csv(
                    boundary_summary, method_dir / "boundary_agreement_by_trajectory.csv"
                ),
                _write_csv(
                    events_supported,
                    method_dir / "transition_events_changepoint_support.csv",
                ),
            ]
        )
        _plot_contingency(
            contingency,
            method_dir / "state_contingency.png",
            f"{method.upper()} vs combined changepoint states",
        )
        written.extend(
            _plot_timelines(
                aligned,
                method_dir,
                method,
                args.max_timeline_plots,
            )
        )
        if gsa_features is not None:
            written.extend(
                _analyze_gsa_windows(
                    method_dir,
                    variant,
                    events,
                    event_deltas,
                    gsa_features,
                    window_rows=args.window_rows,
                    bootstrap_samples=args.bootstrap_samples,
                    random_state=args.random_state,
                    top_features=args.top_features,
                )
            )

        method_outputs[method] = {
            "variant": variant,
            "events": events,
            "signatures": signatures,
            "written": written,
        }
        manifest["methods"][method] = {
            "n_frames": len(variant),
            "n_trajectories": int(variant["traj_id"].nunique()),
            "n_macrostates": int(variant["macro_label"].nunique()),
            "n_transfer_events": len(events),
            "artifacts": [str(path) for path in written],
        }

    pca = method_outputs["pca"]["variant"]
    tlica = method_outputs["tlica"]["variant"]
    reduction_aligned = pca[
        ["traj_id", "frame", "time_ps", "macro_label"]
    ].merge(
        tlica[["traj_id", "frame", "macro_label"]].rename(
            columns={"macro_label": "changepoint_label"}
        ),
        on=["traj_id", "frame"],
        how="inner",
    )
    reduction_metrics = state_agreement_metrics(reduction_aligned)
    _write_csv(reduction_metrics, args.output_dir / "pca_vs_tlica_agreement.csv")
    reduction_contingency = state_contingency(reduction_aligned)
    reduction_mapping = hungarian_state_mapping(reduction_contingency)
    _write_indexed_csv(
        reduction_contingency, args.output_dir / "pca_vs_tlica_contingency.csv"
    )
    _write_csv(
        reduction_mapping, args.output_dir / "pca_vs_tlica_state_mapping.csv"
    )

    tlica_to_pca = {
        int(row.changepoint_label): int(row.macro_label)
        for row in reduction_mapping.itertuples(index=False)
    }
    pca_signatures = method_outputs["pca"]["signatures"].copy()
    tlica_signatures = method_outputs["tlica"]["signatures"].copy()
    tlica_signatures["from_state"] = tlica_signatures["from_state"].map(tlica_to_pca)
    tlica_signatures["to_state"] = tlica_signatures["to_state"].map(tlica_to_pca)
    tlica_signatures = tlica_signatures.dropna(subset=["from_state", "to_state"])
    tlica_signatures[["from_state", "to_state"]] = tlica_signatures[
        ["from_state", "to_state"]
    ].astype(int)
    consensus = pca_signatures.merge(
        tlica_signatures,
        on=["from_state", "to_state", "feature", "feature_block"],
        suffixes=("_pca", "_tlica"),
        how="inner",
    )
    if not consensus.empty:
        consensus["direction_matches"] = (
            consensus["mean_standardized_delta_pca"]
            * consensus["mean_standardized_delta_tlica"]
            >= 0
        )
        consensus["mean_abs_standardized_delta"] = 0.5 * (
            consensus["mean_standardized_delta_pca"].abs()
            + consensus["mean_standardized_delta_tlica"].abs()
        )
        consensus = consensus.sort_values(
            ["direction_matches", "mean_abs_standardized_delta"],
            ascending=[False, False],
        )
    _write_csv(
        consensus, args.output_dir / "pca_vs_tlica_transition_consensus.csv"
    )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote state comparison artifacts to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

