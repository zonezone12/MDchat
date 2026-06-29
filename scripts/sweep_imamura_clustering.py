"""
Sweep Imamura MSM clustering hyperparameters (MiniBatchKMeans + Ward).

Fits PCA or tlICA once, then grids over ``n_micro`` (MiniBatchKMeans) and
``n_macro`` (Ward merge) on an existing ``imamura_features.csv``. Ranks each
combination by macrostate silhouette, occupancy, and population balance.

Example
-------
python scripts/sweep_imamura_clustering.py \\
    --features-csv output/imamura_msm/endpoint_BMMpM/imamura_features.csv \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectories traj/BMMpM_891249_mdcrd_v.trj \\
    --format TRJ \\
    --dt-ps 2.0

tlICA sweep + tuned MSM run:
python scripts/sweep_imamura_clustering.py \\
    --features-csv output/imamura_msm/endpoint_BMMpM/imamura_features.csv \\
    --dim-reduction tlica \\
    --tlica-lag-ns 2.0 \\
    --topology traj/BMMpM_ca.prmtop \\
    --trajectories traj/BMMpM_891249_mdcrd_v.trj \\
    --format TRJ \\
    --dt-ps 2.0

# Sweep only (skip auto MSM run):
python scripts/sweep_imamura_clustering.py \\
    --features-csv output/imamura_msm/endpoint_BMMpM/imamura_features.csv \\
    --dim-reduction tlica --dt-ps 2.0 \\
    --no-run-final
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.imamura_msm import (
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_TLICA_REGULARIZATION,
    DEFAULT_TRANSITION_LAG_NS,
    lag_frames_from_ns,
    load_imamura_features_csv,
    sweep_imamura_clustering,
)
from src.utils.run_log import RunContext, log_event, step


def _default_micro_grid() -> list[int]:
    return [1500, 1750, 2000,2250]


def _default_macro_grid() -> list[int]:
    return [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def _parse_int_list(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    return sorted({int(v) for v in values})


def _projection_space_label(dimensionality_reduction: str) -> str:
    return "TIC" if dimensionality_reduction == "tlica" else "PCA"


def _default_sweep_dir(features_csv: Path, dimensionality_reduction: str) -> Path:
    if dimensionality_reduction == "tlica":
        return features_csv.parent / "tlica_cluster_sweep"
    return features_csv.parent / "cluster_sweep"


def _default_final_dir(features_csv: Path, dimensionality_reduction: str) -> Path:
    if dimensionality_reduction == "tlica":
        return features_csv.parent / "tlica_tuned"
    return features_csv.parent / "cluster_tuned"


def _plot_sweep_heatmap(
    sweep_df: pd.DataFrame,
    output_dir: Path,
    *,
    value_col: str,
    title: str,
    cmap: str = "viridis",
    higher_better: bool = True,
) -> Path:
    pivot = sweep_df.pivot(index="n_micro", columns="n_macro", values=value_col)
    pivot = pivot.sort_index(ascending=False)

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * pivot.shape[1] + 3), 5))
    data = pivot.to_numpy(dtype=float)
    vmin, vmax = np.nanmin(data), np.nanmax(data)
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("n_macro (Ward)")
    ax.set_ylabel("n_micro (MiniBatchKMeans)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = output_dir / f"heatmap_{value_col}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _plot_top_candidates(sweep_df: pd.DataFrame, output_dir: Path, top_n: int = 12) -> Path:
    top = sweep_df.head(top_n).iloc[::-1]
    labels = [f"m{int(r.n_micro)}/M{int(r.n_macro)}" for r in top.itertuples()]
    scores = top["composite_score"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top) + 1)))
    ax.barh(range(len(top)), scores, color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Composite score")
    ax.set_title(f"Top {len(top)} clustering parameter sets")
    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(
            scores[i] + 0.01 * (scores.max() or 1),
            i,
            f"sil={row['silhouette_macro']:.3f}",
            va="center",
            fontsize=8,
        )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = output_dir / "top_candidates.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _pick_recommended(
    sweep_df: pd.DataFrame,
    *,
    min_macro_pop: float,
) -> tuple[pd.Series, str]:
    viable = sweep_df[
        (sweep_df["n_macro_occupied"] >= 2)
        & (sweep_df["min_macro_population_frac"] >= min_macro_pop)
        & sweep_df["silhouette_macro"].notna()
    ]
    if viable.empty:
        return sweep_df.iloc[0], (
            "No grid point met min_macro_pop; using highest composite_score overall."
        )
    return viable.iloc[0], (
        f"Selected among {len(viable)} viable points "
        f"(min_macro_pop>={min_macro_pop:.3g})."
    )


def _write_recommendation(
    sweep_df: pd.DataFrame,
    output_dir: Path,
    *,
    min_macro_pop: float,
    dimensionality_reduction: str,
    tlica_lag_ns: float | None,
    tlica_lag_frames: int | None,
) -> Path:
    best, note = _pick_recommended(sweep_df, min_macro_pop=min_macro_pop)
    space = _projection_space_label(dimensionality_reduction)

    lines = [
        "Imamura clustering sweep recommendation",
        "======================================",
        note,
        "",
        f"Dimensionality reduction: {dimensionality_reduction}",
        f"Recommended n_micro: {int(best['n_micro'])}",
        f"Recommended n_macro: {int(best['n_macro'])}",
        f"Composite score: {best['composite_score']:.4f}",
        f"Macro silhouette ({space} space): {best['silhouette_macro']:.4f}",
        f"Davies-Bouldin (lower better): {best['davies_bouldin_macro']:.4f}",
        f"Calinski-Harabasz (higher better): {best['calinski_harabasz_macro']:.1f}",
        f"Micro occupied / requested: {int(best['n_micro_occupied'])}/{int(best['n_micro'])}",
        f"Macro occupied / requested: {int(best['n_macro_occupied'])}/{int(best['n_macro'])}",
        f"Min macro population fraction: {best['min_macro_population_frac']:.4f}",
        f"Macro population Gini: {best['macro_population_gini']:.4f}",
        f"Projection weight sum: {best['pca_variance_sum']:.4f}",
        f"Ward cut height: {best['ward_cut_height']:.4f}",
    ]
    if dimensionality_reduction == "tlica":
        lines.extend([
            f"tlICA lag: {tlica_lag_ns:g} ns → {tlica_lag_frames} frames",
        ])
    lines.extend([
        "",
        "Run full MSM with:",
        (
            f"  python scripts/run_imamura_msm.py --features-csv <path> "
            f"--dim-reduction {dimensionality_reduction} "
            f"--n-micro {int(best['n_micro'])} --n-macro {int(best['n_macro'])} ..."
        ),
    ])
    out = output_dir / "recommendation.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _bead_spec_for_features(
    features_csv: Path,
    bead_spec: str | None,
) -> Path | None:
    if bead_spec:
        return Path(bead_spec)
    sibling = features_csv.parent / "bead_spec.json"
    return sibling if sibling.is_file() else None


def _invoke_run_imamura_msm(
    *,
    features_csv: Path,
    output_dir: Path,
    n_micro: int,
    n_macro: int,
    n_pca: int,
    dimensionality_reduction: str,
    topology: str,
    trajectories: list[str],
    traj_format: str | None,
    dt_ps: float | None,
    transition_lag_ns: float | None,
    tlica_lag_ns: float | None,
    tlica_regularization: float | None,
    bead_spec: Path | None,
) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_imamura_msm.py"),
        "--features-csv",
        str(features_csv),
        "--output-dir",
        str(output_dir),
        "--topology",
        topology,
        "--trajectories",
        *trajectories,
        "--dim-reduction",
        dimensionality_reduction,
        "--n-micro",
        str(n_micro),
        "--n-macro",
        str(n_macro),
        "--n-pca",
        str(n_pca),
    ]
    if bead_spec is not None:
        cmd.extend(["--bead-spec", str(bead_spec)])
    if traj_format:
        cmd.extend(["--format", traj_format])
    if dt_ps is not None:
        cmd.extend(["--dt-ps", str(dt_ps)])
    if transition_lag_ns is not None:
        cmd.extend(["--transition-lag-ns", str(transition_lag_ns)])
    if dimensionality_reduction == "tlica" and tlica_lag_ns is not None:
        cmd.extend(["--tlica-lag-ns", str(tlica_lag_ns)])
    if dimensionality_reduction == "tlica" and tlica_regularization is not None:
        cmd.extend(["--tlica-regularization", str(tlica_regularization)])
    log_event(
        "info",
        f"Running run_imamura_msm.py {dimensionality_reduction} "
        f"n_micro={n_micro} n_macro={n_macro}",
        component="sweep_imamura_clustering",
    )
    return subprocess.call(cmd)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep Imamura MiniBatchKMeans + Ward parameters on saved features."
        ),
    )
    p.add_argument(
        "--features-csv",
        required=True,
        help="Path to imamura_features.csv from a prior extraction run",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Sweep output directory (default: cluster_sweep or tlica_cluster_sweep)",
    )
    p.add_argument(
        "--dim-reduction",
        choices=["pca", "tlica"],
        default="pca",
        help="Fit PCA (default) or tlICA once before clustering grid",
    )
    p.add_argument(
        "--n-pca",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help=f"PCA/tlICA components before clustering (default: {DEFAULT_PCA_COMPONENTS})",
    )
    p.add_argument(
        "--tlica-lag-ns",
        type=float,
        default=None,
        help="tlICA lag in ns (default: --transition-lag-ns or 2 ns)",
    )
    p.add_argument(
        "--tlica-regularization",
        type=float,
        default=None,
        help="tlICA covariance shrinkage (default: 1e-6 × trace(C0)/d)",
    )
    p.add_argument(
        "--n-micro",
        nargs="+",
        default=None,
        help="MiniBatchKMeans cluster counts to try (default: paper-centered grid)",
    )
    p.add_argument(
        "--n-macro",
        nargs="+",
        default=None,
        help="Ward macrostate counts to try (default: paper-centered grid)",
    )
    p.add_argument(
        "--min-macro-pop",
        type=float,
        default=0.01,
        help="Min macrostate population fraction for recommendation (default: 0.01)",
    )
    p.add_argument(
        "--silhouette-max-rows",
        type=int,
        default=8000,
        help="Subsample rows for silhouette scoring (default: 8000)",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip heatmap / bar plots",
    )
    p.add_argument(
        "--run-final",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After sweep, run run_imamura_msm.py with recommended n_micro/n_macro "
        "(default: on)",
    )
    p.add_argument(
        "--final-output-dir",
        default=None,
        help="Output dir for tuned MSM (default: cluster_tuned or tlica_tuned)",
    )
    final = p.add_argument_group("tuned MSM run (used when --run-final)")
    final.add_argument(
        "--topology",
        default=None,
        help="Topology for run_imamura_msm (required with --run-final)",
    )
    final.add_argument(
        "--trajectories",
        nargs="+",
        default=None,
        help="Trajectory path(s) for run_imamura_msm (required with --run-final)",
    )
    final.add_argument("--format", default=None, help="Trajectory format (e.g. TRJ)")
    final.add_argument(
        "--dt-ps",
        type=float,
        default=None,
        help="Frame spacing in ps (required for tlICA lag framing)",
    )
    final.add_argument(
        "--transition-lag-ns",
        type=float,
        default=None,
        help="Transition lag in ns (default: run_imamura_msm default 2 ns)",
    )
    final.add_argument(
        "--bead-spec",
        default=None,
        help="bead_spec.json for run_imamura_msm (default: sibling of --features-csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    features_csv = Path(args.features_csv).resolve()
    if not features_csv.is_file():
        print(f"Features file not found: {features_csv}", file=sys.stderr)
        sys.exit(1)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _default_sweep_dir(features_csv, args.dim_reduction)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    n_micro_vals = _parse_int_list(args.n_micro, _default_micro_grid())
    n_macro_vals = _parse_int_list(args.n_macro, _default_macro_grid())

    tlica_lag_ns = (
        float(args.tlica_lag_ns)
        if args.tlica_lag_ns is not None
        else float(args.transition_lag_ns or DEFAULT_TRANSITION_LAG_NS)
    )
    tlica_regularization = (
        args.tlica_regularization
        if args.tlica_regularization is not None
        else DEFAULT_TLICA_REGULARIZATION
    )

    if args.dim_reduction == "tlica" and args.dt_ps is None:
        print(
            "Error: --dt-ps is required for tlICA sweeps (to convert lag ns → frames).",
            file=sys.stderr,
        )
        sys.exit(2)

    tlica_lag_frames = (
        lag_frames_from_ns(tlica_lag_ns, args.dt_ps)
        if args.dim_reduction == "tlica" and args.dt_ps is not None
        else 1
    )

    if args.run_final and (not args.topology or not args.trajectories):
        print(
            "Error: --topology and --trajectories are required when --run-final "
            "(use --no-run-final for sweep-only).",
            file=sys.stderr,
        )
        sys.exit(2)

    bead_spec_path = _bead_spec_for_features(features_csv, args.bead_spec)
    space = _projection_space_label(args.dim_reduction)

    with RunContext.from_namespace(args, name="sweep_imamura_clustering") as run:
        print(f"Run log: {run.run_dir}")
        log_event(
            "start",
            f"{args.dim_reduction} sweep {len(n_micro_vals)}×{len(n_macro_vals)} "
            f"on {features_csv.name}",
        )

        with step("load_features"):
            feat_df, feat_names = load_imamura_features_csv(features_csv)
            X = feat_df[feat_names].to_numpy()
            traj_ids = (
                feat_df["traj_id"].to_numpy()
                if "traj_id" in feat_df.columns
                else None
            )
            print(f"  {X.shape[0]} frames × {X.shape[1]} features")

        with step("clustering_sweep"):
            if args.dim_reduction == "tlica":
                print(
                    f"  tlICA lag: {tlica_lag_ns} ns → {tlica_lag_frames} frames "
                    f"(dt={args.dt_ps} ps)"
                )
            sweep_df = sweep_imamura_clustering(
                X,
                n_micro_values=n_micro_vals,
                n_macro_values=n_macro_vals,
                n_pca_components=args.n_pca,
                dimensionality_reduction=args.dim_reduction,
                traj_ids=traj_ids,
                tlica_lag_frames=tlica_lag_frames,
                tlica_lag_ns=tlica_lag_ns,
                tlica_regularization=tlica_regularization,
                silhouette_max_rows=args.silhouette_max_rows,
            )
            sweep_path = out_dir / "cluster_sweep.csv"
            sweep_df.to_csv(sweep_path, index=False)
            print(f"  Wrote {sweep_path}")

        with step("summarize"):
            rec_path = _write_recommendation(
                sweep_df,
                out_dir,
                min_macro_pop=args.min_macro_pop,
                dimensionality_reduction=args.dim_reduction,
                tlica_lag_ns=tlica_lag_ns if args.dim_reduction == "tlica" else None,
                tlica_lag_frames=tlica_lag_frames if args.dim_reduction == "tlica" else None,
            )
            print(f"  Wrote {rec_path}")
            pick, _ = _pick_recommended(sweep_df, min_macro_pop=args.min_macro_pop)
            print(
                f"  Recommended: n_micro={int(pick['n_micro'])}, "
                f"n_macro={int(pick['n_macro'])}, "
                f"silhouette={pick['silhouette_macro']:.3f}, "
                f"composite={pick['composite_score']:.3f}"
            )

        if not args.no_plots:
            with step("plots"):
                plots = [
                    _plot_sweep_heatmap(
                        sweep_df,
                        out_dir,
                        value_col="silhouette_macro",
                        title=f"Macro silhouette ({space} space)",
                    ),
                    _plot_sweep_heatmap(
                        sweep_df,
                        out_dir,
                        value_col="composite_score",
                        title="Composite ranking score",
                    ),
                    _plot_top_candidates(sweep_df, out_dir),
                ]
                for p in plots:
                    print(f"  Wrote {p}")

        if args.run_final:
            pick, note = _pick_recommended(sweep_df, min_macro_pop=args.min_macro_pop)
            final_dir = (
                Path(args.final_output_dir)
                if args.final_output_dir
                else _default_final_dir(features_csv, args.dim_reduction)
            )
            with step("run_final_msm"):
                print(f"  {note}")
                if bead_spec_path:
                    print(f"  bead_spec: {bead_spec_path}")
                rc = _invoke_run_imamura_msm(
                    features_csv=features_csv,
                    output_dir=final_dir,
                    n_micro=int(pick["n_micro"]),
                    n_macro=int(pick["n_macro"]),
                    n_pca=args.n_pca,
                    dimensionality_reduction=args.dim_reduction,
                    topology=args.topology,
                    trajectories=args.trajectories,
                    traj_format=args.format,
                    dt_ps=args.dt_ps,
                    transition_lag_ns=args.transition_lag_ns,
                    tlica_lag_ns=tlica_lag_ns,
                    tlica_regularization=tlica_regularization,
                    bead_spec=bead_spec_path,
                )
                if rc != 0:
                    sys.exit(rc)
                print(f"  Tuned MSM artifacts → {final_dir}")

        log_event("done", f"sweep complete → {out_dir}")


if __name__ == "__main__":
    main()
