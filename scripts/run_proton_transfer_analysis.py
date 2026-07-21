"""
NH3···HNO3(+H2O) proton-transfer analysis: pairwise distances, N–H–O CV,
PIMD delocalization, feature–CV correlation, and ruptures changepoints.

Compares m1n0 (no water) vs m1n1 (one water) under CLMD and PIMD.

Example
-------
python scripts/run_proton_transfer_analysis.py

python scripts/run_proton_transfer_analysis.py \\
    --traj-root traj/NH4NO3 \\
    --output-dir output/nh4no3_pt \\
    --method Pelt --cost-model rbf --penalty 3.0
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ProtonTransfer import (
    build_feature_table,
    build_segment_stats,
    correlate_features_vs_cv,
    count_delta_sign_flips,
    detect_pt_atoms,
    load_ham_dat,
    load_pimd_xyz,
    segment_feature_matrix,
    zscore_matrix,
)
from src.utils.cluster_inspection import plot_pca_clusters
from src.utils.metastable_states import (
    Segment,
    assign_cluster_labels,
    cluster_metastable_states,
    format_cluster_summary,
    plot_dendrogram,
    prepare_feature_clustering,
)
from src.utils.ruptures_utils import ChangePointResult, detect_changepoints

# Metadata / CV columns that are not pairwise-distance features
_NON_FEATURE_COLS = frozenset(
    {
        "frame",
        "step",
        "n_beads",
        "n_amine",
        "h_acid",
        "o_acid",
        "d_NH",
        "d_OH",
        "delta",
        "proton_rg",
        "delta_bead_std",
        "delta_bead_mean",
        "hamiltonian",
        "temperature",
        "potential",
        "dkinetic",
        "ebath_cent",
    }
)

# Default systems under traj/NH4NO3
DEFAULT_JOBS: List[Dict[str, Any]] = [
    {
        "system": "m1n0",
        "method": "CLMD",
        "n_physical": 9,
        "relpath": "2-m1n0/1-CLMD/Result/coor.xyz",
        "ham_relpath": "2-m1n0/1-CLMD/Result/ham.dat",
    },
    {
        "system": "m1n0",
        "method": "PIMD",
        "n_physical": 9,
        "relpath": "2-m1n0/2-PIMD/Result/coor.xyz",
        "ham_relpath": "2-m1n0/2-PIMD/Result/ham.dat",
    },
    {
        "system": "m1n1",
        "method": "CLMD",
        "n_physical": 12,
        "relpath": "3-m1n1/1-CLMD/Result/coor.xyz",
        "ham_relpath": "3-m1n1/1-CLMD/Result/ham.dat",
    },
    {
        "system": "m1n1",
        "method": "PIMD",
        "n_physical": 12,
        "relpath": "3-m1n1/2-PIMD/Result/coor.xyz",
        "ham_relpath": "3-m1n1/2-PIMD/Result/ham.dat",
    },
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NH4NO3 proton-transfer pairwise-distance + changepoint analysis"
    )
    p.add_argument(
        "--traj-root",
        type=Path,
        default=ROOT / "traj" / "NH4NO3",
        help="Root folder containing 2-m1n0 and 3-m1n1",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder (default: output/<timestamp>_nh4no3_pt)",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Cap frames per traj")
    p.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help="Subset of systems, e.g. m1n0 m1n1",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods, e.g. CLMD PIMD",
    )
    p.add_argument("--cpd-method", default="Pelt", help="ruptures search method")
    p.add_argument("--cost-model", default="rbf", help="ruptures cost model")
    p.add_argument("--penalty", type=float, default=None, help="Pelt penalty")
    p.add_argument("--n-bkps", type=int, default=None, help="Exact number of breakpoints")
    p.add_argument("--min-size", type=int, default=20, help="Minimum segment length")
    p.add_argument("--jump", type=int, default=5, help="ruptures jump")
    p.add_argument(
        "--top-corr",
        type=int,
        default=20,
        help="Number of top correlations to plot",
    )
    p.add_argument(
        "--near-zero",
        type=float,
        default=0.05,
        help="|delta| cutoff (Angstrom) for shared-proton region",
    )
    p.add_argument(
        "--cluster-only",
        action="store_true",
        help="Skip XYZ reload; cluster from existing features_*/changepoints_* CSVs",
    )
    p.add_argument(
        "--no-cluster",
        action="store_true",
        help="Skip segment clustering step",
    )
    p.add_argument(
        "--cluster-signal",
        default="pairwise_distances",
        choices=("pairwise_distances", "delta"),
        help="Which CPD segments to cluster",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        default=4,
        help="Number of hierarchical clusters (Ward)",
    )
    p.add_argument(
        "--auto-select-k",
        action="store_true",
        help="Choose k by silhouette instead of --n-clusters",
    )
    p.add_argument(
        "--k-max",
        type=int,
        default=12,
        help="Max k when --auto-select-k is set",
    )
    p.add_argument(
        "--linkage",
        default="ward",
        help="Hierarchical linkage method",
    )
    return p.parse_args()


def _job_tag(system: str, method: str) -> str:
    return f"{system}_{method}"


def _feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in _NON_FEATURE_COLS and "-" in c]


def _changepoint_table(
    result: ChangePointResult,
    *,
    signal_name: str,
    tag: str,
) -> pd.DataFrame:
    rows = []
    for i, (start, end) in enumerate(result.segment_ranges):
        rows.append(
            {
                "tag": tag,
                "signal": signal_name,
                "segment": i,
                "start_frame": int(start),
                "end_frame": int(end),
                "n_frames": int(end - start),
                "mean": float(result.segment_means[i]) if result.segment_means else np.nan,
                "std": float(result.segment_stds[i]) if result.segment_stds else np.nan,
            }
        )
    seg_df = pd.DataFrame(rows)
    bkp_df = pd.DataFrame(
        {
            "tag": tag,
            "signal": signal_name,
            "breakpoint": result.breakpoints,
        }
    )
    return seg_df, bkp_df


def plot_delta_with_changepoints(
    delta: np.ndarray,
    steps: np.ndarray,
    result: ChangePointResult,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    x = steps if np.all(steps >= 0) else np.arange(len(delta))
    ax.plot(x, delta, lw=0.7, color="steelblue", alpha=0.85, label="delta")
    ax.axhline(0.0, color="gray", ls=":", lw=1.0)
    for bkp in result.breakpoints:
        ax.axvline(x[bkp] if bkp < len(x) else bkp, color="crimson", ls="--", lw=1.0, alpha=0.75)
    ax.set_xlabel("step" if np.all(steps >= 0) else "frame")
    ax.set_ylabel(r"$\delta = d(\mathrm{N{-}H}) - d(\mathrm{O{-}H})$ [Å]")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_top_correlations(
    corr_df: pd.DataFrame,
    title: str,
    out_path: Path,
    top_n: int = 20,
) -> None:
    if corr_df.empty:
        return
    sub = corr_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(sub))))
    colors = ["#c0392b" if r < 0 else "#2980b9" for r in sub["pearson_r"]]
    ax.barh(sub["feature"], sub["pearson_r"], color=colors)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Pearson r vs delta")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_delta_distributions(
    series: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, arr in series.items():
        ax.hist(arr, bins=60, density=True, alpha=0.45, label=label)
    ax.axvline(0.0, color="black", ls="--", lw=1.0)
    ax.set_xlabel(r"$\delta$ [Å]")
    ax.set_ylabel("density")
    ax.set_title("Proton-transfer CV distributions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_one_job(
    job: Dict[str, Any],
    traj_root: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    tag = _job_tag(job["system"], job["method"])
    xyz_path = traj_root / job["relpath"]
    ham_path = traj_root / job["ham_relpath"]
    print(f"\n=== {tag} ===")
    print(f"  loading {xyz_path}")

    traj = load_pimd_xyz(
        xyz_path,
        n_physical=job["n_physical"],
        max_frames=args.max_frames,
    )
    assert traj.n_atoms == job["n_physical"], (
        f"{tag}: expected {job['n_physical']} atoms, got {traj.n_atoms}"
    )
    if job["method"] == "CLMD":
        assert traj.n_beads == 1, f"{tag}: CLMD should have 1 bead, got {traj.n_beads}"
    else:
        assert traj.n_beads > 1, f"{tag}: PIMD should have >1 beads, got {traj.n_beads}"

    pt = detect_pt_atoms(traj.species, traj.centroid[0])
    print(
        f"  atoms: n_amine={pt.n_amine} ({traj.species[pt.n_amine]}), "
        f"h_acid={pt.h_acid} ({traj.species[pt.h_acid]}), "
        f"o_acid={pt.o_acid} ({traj.species[pt.o_acid]}), "
        f"n_beads={traj.n_beads}, n_frames={traj.n_frames}"
    )
    if pt.water_o is not None:
        print(f"  water: O={pt.water_o}, H={pt.water_h}")

    ham = load_ham_dat(ham_path) if ham_path.is_file() else None
    table = build_feature_table(traj, pt, ham=ham)

    # Sanity: frame-0 proton on acid side (delta > 0) and d_OH ~ 1 A
    d0 = float(table["delta"].iloc[0])
    doh0 = float(table["d_OH"].iloc[0])
    print(f"  frame0: delta={d0:.3f} A, d_OH={doh0:.3f} A")
    if d0 <= 0:
        print("  WARNING: frame-0 delta <= 0 (expected proton on acid)")
    if not (0.7 < doh0 < 1.4):
        print(f"  WARNING: frame-0 d_OH={doh0:.3f} outside 0.7–1.4 A")

    feat_path = out_dir / f"features_{tag}.csv"
    table.to_csv(feat_path, index=False)
    print(f"  wrote {feat_path.name}")

    feat_cols = _feature_columns(table)
    feat_df = table[feat_cols]
    delta = table["delta"].to_numpy(dtype=np.float64)

    corr = correlate_features_vs_cv(feat_df, delta)
    corr_path = out_dir / f"corr_{tag}.csv"
    corr.to_csv(corr_path, index=False)
    plot_top_correlations(
        corr,
        title=f"Top feature–delta correlations ({tag})",
        out_path=out_dir / f"corr_{tag}.png",
        top_n=args.top_corr,
    )
    print(f"  wrote {corr_path.name}")

    # --- Changepoints on 1-D delta ---
    cpd_kwargs = dict(
        method=args.cpd_method,
        cost_model=args.cost_model,
        penalty=args.penalty,
        n_bkps=args.n_bkps,
        min_size=args.min_size,
        jump=args.jump,
    )
    # Default heuristic penalty for Pelt when neither penalty nor n_bkps given
    if cpd_kwargs["penalty"] is None and cpd_kwargs["n_bkps"] is None:
        cpd_kwargs["penalty"] = float(np.log(max(len(delta), 2)))

    delta_cp = detect_changepoints(delta, **cpd_kwargs)
    plot_delta_with_changepoints(
        delta,
        table["step"].to_numpy(),
        delta_cp,
        title=f"delta CV + changepoints ({tag})",
        out_path=out_dir / f"delta_cp_{tag}.png",
    )

    # --- Multivariate changepoints on z-scored pairwise distances ---
    X = zscore_matrix(feat_df.to_numpy(dtype=np.float64))
    multi_kwargs = dict(cpd_kwargs)
    if multi_kwargs["penalty"] is None and multi_kwargs["n_bkps"] is None:
        multi_kwargs["penalty"] = float(np.log(max(X.shape[0], 2)))
    # Prefer l2 for high-D pairwise features (rbf can be slow / memory-heavy)
    if multi_kwargs["cost_model"] == "rbf" and X.shape[1] > 20:
        multi_kwargs["cost_model"] = "l2"
    multi_cp = detect_changepoints(X, **multi_kwargs)

    seg_delta, bkp_delta = _changepoint_table(delta_cp, signal_name="delta", tag=tag)
    seg_multi, bkp_multi = _changepoint_table(
        multi_cp, signal_name="pairwise_distances", tag=tag
    )
    seg_df = pd.concat([seg_delta, seg_multi], ignore_index=True)
    bkp_df = pd.concat([bkp_delta, bkp_multi], ignore_index=True)
    seg_path = out_dir / f"changepoints_{tag}.csv"
    bkp_path = out_dir / f"breakpoints_{tag}.csv"
    seg_df.to_csv(seg_path, index=False)
    bkp_df.to_csv(bkp_path, index=False)
    print(
        f"  changepoints: delta={delta_cp.n_breakpoints}, "
        f"pairwise={multi_cp.n_breakpoints}"
    )
    print(f"  wrote {seg_path.name}, {bkp_path.name}")

    stats = count_delta_sign_flips(delta, near_zero=args.near_zero)
    stats.update(
        {
            "tag": tag,
            "system": job["system"],
            "method": job["method"],
            "n_atoms": traj.n_atoms,
            "n_beads": traj.n_beads,
            "n_amine": pt.n_amine,
            "h_acid": pt.h_acid,
            "o_acid": pt.o_acid,
            "water_o": pt.water_o if pt.water_o is not None else -1,
            "frame0_delta": d0,
            "frame0_d_OH": doh0,
            "proton_rg_mean": float(table["proton_rg"].mean()),
            "proton_rg_max": float(table["proton_rg"].max()),
            "delta_bead_std_mean": float(table["delta_bead_std"].mean()),
            "n_cp_delta": delta_cp.n_breakpoints,
            "n_cp_pairwise": multi_cp.n_breakpoints,
            "cpd_penalty_delta": float(delta_cp.penalty)
            if delta_cp.penalty is not None
            else np.nan,
            "cpd_cost_pairwise": multi_kwargs["cost_model"],
            "top_corr_feature": str(corr.iloc[0]["feature"]) if not corr.empty else "",
            "top_corr_pearson": float(corr.iloc[0]["pearson_r"]) if not corr.empty else np.nan,
            "features_csv": str(feat_path.name),
            "corr_csv": str(corr_path.name),
            "changepoints_csv": str(seg_path.name),
        }
    )

    artifacts = [
        feat_path.name,
        corr_path.name,
        f"corr_{tag}.png",
        f"delta_cp_{tag}.png",
        seg_path.name,
        bkp_path.name,
    ]

    # Water-related correlations (m1n1)
    if pt.water_o is not None and not corr.empty:
        water_ids = {pt.water_o, *pt.water_h}
        water_hits: List[Dict[str, Any]] = []
        for rank, (_, row) in enumerate(corr.iterrows(), start=1):
            feat = str(row["feature"])
            atom_ids: List[int] = []
            for part in feat.split("-"):
                digits = "".join(ch for ch in part if ch.isdigit())
                if digits:
                    atom_ids.append(int(digits))
            if any(a in water_ids for a in atom_ids):
                water_hits.append(
                    {
                        "feature": feat,
                        "pearson_r": float(row["pearson_r"]),
                        "abs_pearson": float(row["abs_pearson"]),
                        "overall_rank": rank,
                    }
                )
        water_df = pd.DataFrame(water_hits)
        if not water_df.empty:
            water_df = water_df.sort_values(
                "abs_pearson", ascending=False
            ).reset_index(drop=True)
            water_df["water_rank"] = np.arange(1, len(water_df) + 1)
            water_path = out_dir / f"corr_water_{tag}.csv"
            water_df.to_csv(water_path, index=False)
            artifacts.append(water_path.name)
            stats["n_water_corr_features"] = int(len(water_df))
            stats["top_water_feature"] = str(water_df.iloc[0]["feature"])
            stats["top_water_pearson"] = float(water_df.iloc[0]["pearson_r"])
            print(
                f"  water: top={stats['top_water_feature']} "
                f"r={stats['top_water_pearson']:.3f} "
                f"(n={stats['n_water_corr_features']})"
            )

    return {
        "stats": stats,
        "delta": delta,
        "tag": tag,
        "artifacts": artifacts,
    }


def collect_segment_stats(
    out_dir: Path,
    tags: List[str],
    *,
    signal: str,
) -> pd.DataFrame:
    """Build and merge segment-level feature summaries for all tags."""
    parts: List[pd.DataFrame] = []
    for tag in tags:
        feat_path = out_dir / f"features_{tag}.csv"
        cp_path = out_dir / f"changepoints_{tag}.csv"
        if not feat_path.is_file() or not cp_path.is_file():
            print(f"  SKIP segment stats for {tag}: missing features/changepoints CSV")
            continue
        features = pd.read_csv(feat_path)
        cps = pd.read_csv(cp_path)
        stats = build_segment_stats(features, cps, tag=tag, signal=signal)
        stats_path = out_dir / f"segment_stats_{tag}_{signal}.csv"
        stats.to_csv(stats_path, index=False)
        print(f"  wrote {stats_path.name} ({len(stats)} segments)")
        parts.append(stats)
    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    merged_path = out_dir / f"all_segment_stats_{signal}.csv"
    merged.to_csv(merged_path, index=False)
    print(f"  wrote {merged_path.name} ({len(merged)} segments)")
    return merged


def _segments_from_stats(df: pd.DataFrame) -> List[Segment]:
    segments: List[Segment] = []
    for _, row in df.iterrows():
        segments.append(
            Segment(
                traj_id=str(row["traj_id"]),
                segment_id=int(row["segment_id"]),
                start_frame=int(row["start_frame"]),
                end_frame=int(row["end_frame"]),
                rep_frame=int(row["rep_frame"]),
                rmsd_mean=float(row.get("delta_mean", np.nan)),
                rg_mean=float(row.get("d_NH_mean", np.nan)),
            )
        )
    return segments


def cluster_segment_cohort(
    df: pd.DataFrame,
    cohort_name: str,
    out_dir: Path,
    *,
    feature_mode: str,
    n_clusters: int,
    auto_select_k: bool,
    k_max: int,
    linkage: str,
) -> pd.DataFrame:
    """Cluster one cohort of segments; write dendrogram, PCA, and labeled CSV."""
    if len(df) < 2:
        print(f"  SKIP cluster {cohort_name}: need >= 2 segments, got {len(df)}")
        return pd.DataFrame()

    feature_matrix, feature_names = segment_feature_matrix(df, mode=feature_mode)
    if feature_matrix.shape[0] < 2:
        print(f"  SKIP cluster {cohort_name}: empty feature matrix")
        return pd.DataFrame()

    feat_input = prepare_feature_clustering(feature_matrix)
    clustering = cluster_metastable_states(
        feat_input.distance_matrix,
        scaled_features=feat_input.scaled_features,
        method=linkage,
        n_clusters=None if auto_select_k else n_clusters,
        auto_select_k=auto_select_k,
        k_max=k_max,
    )

    sub = df.reset_index(drop=True).copy()
    segments = _segments_from_stats(sub)
    assign_cluster_labels(segments, clustering)

    leaf_labels = [f"{s.traj_id}:seg{s.segment_id}" for s in segments]
    sub["leaf_label"] = leaf_labels
    sub["cluster_label"] = [s.cluster_label for s in segments]
    sub["cohort"] = cohort_name
    sub["feature_mode"] = feature_mode
    sub["features_used"] = ";".join(feature_names)
    sub["n_clusters"] = clustering.n_clusters
    sub["silhouette"] = clustering.silhouette

    cohort_dir = out_dir / "clusters" / cohort_name
    cohort_dir.mkdir(parents=True, exist_ok=True)
    sub.to_csv(cohort_dir / "segments_clustered.csv", index=False)

    pd.DataFrame(
        feat_input.distance_matrix, index=leaf_labels, columns=leaf_labels
    ).to_csv(cohort_dir / "distance_matrix.csv")

    plot_dendrogram(
        clustering.linkage_matrix,
        leaf_labels,
        cohort_dir / "dendrogram.png",
        title=(
            f"PT segments — {cohort_name} "
            f"(k={clustering.n_clusters}, mode={feature_mode})"
        ),
    )

    (cohort_dir / "cluster_summary.txt").write_text(
        format_cluster_summary(segments, clustering),
        encoding="utf-8",
    )

    # PCA can fail for tiny cohorts; skip gracefully
    try:
        plot_pca_clusters(
            feature_matrix,
            clustering.labels,
            leaf_labels,
            cohort_dir / "pca_clusters.png",
            title=f"PCA — {cohort_name} (k={clustering.n_clusters})",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: PCA plot failed for {cohort_name}: {exc}")

    # Timeline: cluster label vs segment midpoint frame, colored by traj
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        for traj_id, g in sub.groupby("traj_id"):
            mid = 0.5 * (g["start_frame"] + g["end_frame"])
            ax.scatter(
                mid,
                g["cluster_label"],
                s=np.clip(g["n_frames"] / 5.0, 10, 120),
                alpha=0.75,
                label=traj_id,
            )
        ax.set_xlabel("frame (segment midpoint)")
        ax.set_ylabel("cluster label")
        ax.set_title(f"Segment clusters — {cohort_name}")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(cohort_dir / "cluster_timeline.png", dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING: timeline plot failed for {cohort_name}: {exc}")

    print(
        f"  clustered {cohort_name}: {len(sub)} segments → "
        f"{clustering.n_clusters} clusters "
        f"(silhouette={clustering.silhouette:.3f}, mode={feature_mode})"
    )
    return sub


def run_segment_clustering(
    out_dir: Path,
    tags: List[str],
    args: argparse.Namespace,
) -> List[str]:
    """
    Build segment stats from CPD ranges and cluster them.

    Cohorts
    -------
    - per tag (pairwise features)
    - per system m1n0 / m1n1 pooling CLMD+PIMD (pairwise)
    - all tags together using CV-only features (cross-system comparable)
    """
    print("\n=== Segment clustering ===")
    signal = args.cluster_signal
    all_stats = collect_segment_stats(out_dir, tags, signal=signal)
    if all_stats.empty:
        print("  No segment stats available; skipping clustering")
        return []

    artifacts: List[str] = [f"all_segment_stats_{signal}.csv"]
    clustered_parts: List[pd.DataFrame] = []

    # 1) Per-trajectory
    for tag in tags:
        sub = all_stats[all_stats["traj_id"] == tag]
        if sub.empty:
            continue
        part = cluster_segment_cohort(
            sub,
            f"{tag}_pairwise",
            out_dir,
            feature_mode="pairwise",
            n_clusters=args.n_clusters,
            auto_select_k=args.auto_select_k,
            k_max=args.k_max,
            linkage=args.linkage,
        )
        if not part.empty:
            clustered_parts.append(part)
            artifacts.append(f"clusters/{tag}_pairwise/")

    # 2) Per-system (CLMD + PIMD), same atom count / pair labels
    for system in sorted(all_stats["system"].dropna().unique()):
        sub = all_stats[all_stats["system"] == system]
        if len(sub) < 2:
            continue
        part = cluster_segment_cohort(
            sub,
            f"{system}_pairwise",
            out_dir,
            feature_mode="pairwise",
            n_clusters=args.n_clusters,
            auto_select_k=args.auto_select_k,
            k_max=args.k_max,
            linkage=args.linkage,
        )
        if not part.empty:
            clustered_parts.append(part)
            artifacts.append(f"clusters/{system}_pairwise/")

    # 3) Cross-system CV features (all tags)
    part = cluster_segment_cohort(
        all_stats,
        "all_cv",
        out_dir,
        feature_mode="cv",
        n_clusters=args.n_clusters,
        auto_select_k=args.auto_select_k,
        k_max=args.k_max,
        linkage=args.linkage,
    )
    if not part.empty:
        clustered_parts.append(part)
        artifacts.append("clusters/all_cv/")

    if clustered_parts:
        merged = pd.concat(clustered_parts, ignore_index=True)
        merged_path = out_dir / "all_segments_clustered.csv"
        merged.to_csv(merged_path, index=False)
        artifacts.append(merged_path.name)
        print(f"  wrote {merged_path.name}")

        # Compact cluster occupancy table
        occ = (
            merged.groupby(["cohort", "traj_id", "cluster_label"], as_index=False)
            .agg(n_segments=("segment_id", "count"), n_frames=("n_frames", "sum"))
        )
        occ_path = out_dir / "cluster_occupancy.csv"
        occ.to_csv(occ_path, index=False)
        artifacts.append(occ_path.name)

    return artifacts


def write_summary(
    out_dir: Path,
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
    *,
    cluster_artifacts: Optional[List[str]] = None,
) -> Path:
    stats_df = pd.DataFrame([r["stats"] for r in results]) if results else pd.DataFrame()
    if not stats_df.empty:
        stats_path = out_dir / "summary_stats.csv"
        stats_df.to_csv(stats_path, index=False)

        delta_series = {r["tag"]: r["delta"] for r in results if "delta" in r}
        if delta_series:
            plot_delta_distributions(delta_series, out_dir / "delta_distributions.png")

    lines: List[str] = []
    lines.append("# NH4NO3 proton-transfer analysis summary")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Traj root: `{args.traj_root}`")
    lines.append(
        f"- CPD: method=`{args.cpd_method}`, cost=`{args.cost_model}`, "
        f"penalty=`{args.penalty}`, n_bkps=`{args.n_bkps}`, "
        f"min_size=`{args.min_size}`"
    )
    lines.append(f"- Near-zero |delta| cutoff: `{args.near_zero}` Å")
    lines.append(
        f"- Segment clustering: signal=`{args.cluster_signal}`, "
        f"n_clusters=`{args.n_clusters}`, auto_k=`{args.auto_select_k}`"
    )
    lines.append("")

    if not stats_df.empty:
        lines.append("## Per-trajectory statistics")
        lines.append("")
        lines.append(
            "| tag | frames | beads | delta mean±std | frac acid | frac amine | "
            "frac shared | sign flips | proton Rg mean | n_cp (delta) | n_cp (pairwise) | top corr |"
        )
        lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, row in stats_df.iterrows():
            lines.append(
                f"| {row['tag']} | {int(row['n_frames'])} | {int(row['n_beads'])} | "
                f"{row['delta_mean']:.3f}±{row['delta_std']:.3f} | "
                f"{row['frac_on_acid']:.3f} | {row['frac_on_amine']:.3f} | "
                f"{row['frac_shared']:.3f} | {int(row['n_sign_flips'])} | "
                f"{row['proton_rg_mean']:.4f} | {int(row['n_cp_delta'])} | "
                f"{int(row['n_cp_pairwise'])} | "
                f"`{row['top_corr_feature']}` ({row['top_corr_pearson']:.3f}) |"
            )
        lines.append("")
        lines.append("## Interpretation notes")
        lines.append("")
        lines.append(
            "- CV: `delta = d(N–H) − d(O–H)`. Positive ⇒ proton closer to acid O; "
            "negative ⇒ closer to amine N. Sign flips count transfer events."
        )
        lines.append(
            "- PIMD `proton_rg` and `delta_bead_std` quantify quantum delocalization "
            "of the transferring proton (zero for CLMD)."
        )
        lines.append(
            "- Feature–CV correlations rank which pairwise distances track the "
            "transfer CV; in m1n1, water-involving pairs (`corr_water_*.csv`) "
            "indicate water's coupling to the transfer coordinate."
        )
        lines.append(
            "- Changepoints: 1-D on delta and multivariate on z-scored pairwise "
            "distances via `src.utils.ruptures_utils.detect_changepoints`."
        )
        lines.append(
            "- Segment clusters: hierarchical Ward clustering of changepoint "
            "segments using pairwise-distance means (per tag / per system) or "
            "CV-only features (`clusters/all_cv`) for cross-system comparison."
        )
        lines.append("")
        lines.append("## Water coupling (m1n1)")
        lines.append("")
        for _, row in stats_df.iterrows():
            if row["system"] != "m1n1":
                continue
            top_w = row.get("top_water_feature", "")
            if pd.isna(top_w) or not top_w:
                lines.append(f"- `{row['tag']}`: no water correlation table written")
            else:
                lines.append(
                    f"- `{row['tag']}`: top water feature `{top_w}` "
                    f"(Pearson r = {row.get('top_water_pearson', float('nan')):.3f}, "
                    f"n = {int(row.get('n_water_corr_features', 0))})"
                )
        lines.append("")

    occ_path = out_dir / "cluster_occupancy.csv"
    if occ_path.is_file():
        lines.append("## Segment cluster occupancy")
        lines.append("")
        occ = pd.read_csv(occ_path)
        # Prefer the all_cv cohort for a compact cross-system view
        view = occ[occ["cohort"] == "all_cv"] if "all_cv" in set(occ["cohort"]) else occ
        lines.append("| cohort | traj | cluster | n_segments | n_frames |")
        lines.append("|---|---|---:|---:|---:|")
        for _, row in view.iterrows():
            lines.append(
                f"| {row['cohort']} | {row['traj_id']} | {int(row['cluster_label'])} | "
                f"{int(row['n_segments'])} | {int(row['n_frames'])} |"
            )
        lines.append("")

    lines.append("## Artifact manifest")
    lines.append("")
    for r in results:
        lines.append(f"### {r['tag']}")
        for name in r.get("artifacts", []):
            lines.append(f"- `{name}`")
        lines.append("")
    if cluster_artifacts:
        lines.append("### Segment clustering")
        for name in cluster_artifacts:
            lines.append(f"- `{name}`")
        lines.append("")
    lines.append("- `summary_stats.csv`")
    lines.append("- `delta_distributions.png`")
    lines.append("- `summary.md`")
    lines.append("")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def _discover_tags(out_dir: Path) -> List[str]:
    tags = []
    for path in sorted(out_dir.glob("features_*.csv")):
        tags.append(path.stem.replace("features_", "", 1))
    return tags


def main() -> int:
    args = parse_args()
    traj_root = args.traj_root.resolve()
    if args.output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "output" / f"{ts}_nh4no3_pt"
    else:
        out_dir = args.output_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output: {out_dir}")
    results: List[Dict[str, Any]] = []

    if args.cluster_only:
        tags = _discover_tags(out_dir)
        if not tags:
            print(f"No features_*.csv found in {out_dir}", file=sys.stderr)
            return 1
        print(f"Cluster-only mode; tags: {tags}")
        # Rebuild minimal results entries for summary manifest
        for tag in tags:
            results.append({"tag": tag, "artifacts": [], "stats": {"tag": tag}})
        # Prefer existing summary_stats if present
        stats_path = out_dir / "summary_stats.csv"
        if stats_path.is_file():
            stats_df = pd.read_csv(stats_path)
            results = []
            for _, row in stats_df.iterrows():
                results.append(
                    {
                        "tag": row["tag"],
                        "artifacts": [],
                        "stats": row.to_dict(),
                        "delta": None,
                    }
                )
    else:
        jobs = DEFAULT_JOBS
        if args.systems:
            systems = {s.lower() for s in args.systems}
            jobs = [j for j in jobs if j["system"].lower() in systems]
        if args.methods:
            methods = {m.upper() for m in args.methods}
            jobs = [j for j in jobs if j["method"].upper() in methods]

        if not jobs:
            print("No jobs selected.", file=sys.stderr)
            return 1

        for job in jobs:
            xyz = traj_root / job["relpath"]
            if not xyz.is_file():
                print(f"SKIP {job['system']}_{job['method']}: missing {xyz}")
                continue
            results.append(run_one_job(job, traj_root, out_dir, args))

        if not results:
            print("No trajectories processed.", file=sys.stderr)
            return 1

    tags = [r["tag"] for r in results]
    cluster_artifacts: List[str] = []
    if not args.no_cluster:
        cluster_artifacts = run_segment_clustering(out_dir, tags, args)

    # Only plot delta distributions when we have arrays
    results_for_summary = [r for r in results if r.get("delta") is not None]
    if not results_for_summary and (out_dir / "summary_stats.csv").is_file():
        # cluster-only: keep stats in summary without replotting deltas
        stats_df = pd.read_csv(out_dir / "summary_stats.csv")
        results_for_summary = [
            {"tag": row["tag"], "artifacts": [], "stats": row.to_dict()}
            for _, row in stats_df.iterrows()
        ]

    summary_path = write_summary(
        out_dir,
        results_for_summary if results_for_summary else results,
        args,
        cluster_artifacts=cluster_artifacts,
    )
    print(f"\nSummary: {summary_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())