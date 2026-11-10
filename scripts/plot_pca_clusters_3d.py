"""3D PCA of endpoint-segment clusters, one panel per GSA cohort.

Uses the same segment features and scaling as ``pca_clusters.png``.
Colors are the cluster labels at each cube's selected physical k
(one-SE cage-geometry rule), read from ``by_k/k_XX/segments_clustered.csv``.
Cluster numbers are labels within a cube and are not matched across cubes.

Writes a static panel, one rotatable HTML per cube, and a copy of the
PNG next to that cube's existing 2D ``pca_clusters.png``.

Example
-------
python scripts/plot_pca_clusters_3d.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.reporting import (
    OWN_ELBOW_ENDPOINT_DIRS,
    endpoint_changepoints_dir,
)
from src.ChangepointAnalysis.segment_clustering import (
    build_feature_matrix_from_segments_df,
)
from src.utils.cluster_inspection import draw_pca_clusters_3d, project_cluster_pca
from src.utils.run_log import RunContext


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D PCA of endpoint clusters for each cohort.")
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Directory that holds endpoint changepoint runs and the k table",
    )
    p.add_argument(
        "--k-table",
        type=Path,
        default=None,
        help="CSV with cohort and selected_k (default: physical_k_selected.csv)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/pca_clusters_3d"),
    )
    p.add_argument(
        "--si-dir",
        type=Path,
        default=Path("docs/figures/si"),
        help="Where the panel and per-cube figures are copied",
    )
    return p.parse_args()


def _selected_k(path: Path) -> dict[str, int]:
    table = pd.read_csv(path)
    if "cohort" not in table.columns or "selected_k" not in table.columns:
        raise ValueError(f"{path} must have cohort and selected_k columns")
    return {str(c): int(k) for c, k in zip(table["cohort"], table["selected_k"])}


def _labels_at_k(root: pd.DataFrame, k_table: pd.DataFrame) -> np.ndarray:
    """Cluster labels from the selected cut, in the feature-table row order."""
    root_key = root["traj_id"].astype(str) + ":" + root["segment_id"].astype(str)
    k_key = k_table["traj_id"].astype(str) + ":" + k_table["segment_id"].astype(str)
    if root_key.duplicated().any() or k_key.duplicated().any():
        raise ValueError("traj_id + segment_id is not unique")
    if set(root_key) != set(k_key):
        raise ValueError(
            "Selected-k segments do not match the feature table "
            f"({len(set(root_key))} vs {len(set(k_key))} keys)"
        )
    mapping = dict(zip(k_key, k_table["cluster_label"].astype(int)))
    return np.array([mapping[key] for key in root_key], dtype=int)


def _copy(src: Path, dest_dir: Path, name: str | None = None) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / (name or src.name)
    dest.write_bytes(src.read_bytes())
    return dest


def _write_html(
    path: Path,
    *,
    title: str,
    coords: np.ndarray,
    labels: np.ndarray,
    variance: np.ndarray,
    meta: pd.DataFrame,
) -> None:
    import plotly.graph_objects as go

    cmap = plt.cm.tab10
    fig = go.Figure()
    for lab in sorted(set(int(v) for v in labels)):
        mask = labels == lab
        color = to_hex(cmap(int(lab) % 10))
        custom = np.column_stack(
            [
                meta.loc[mask, "traj_id"].astype(str),
                meta.loc[mask, "segment_id"].astype(int),
            ]
        )
        fig.add_trace(
            go.Scatter3d(
                x=coords[mask, 0],
                y=coords[mask, 1],
                z=coords[mask, 2],
                mode="markers",
                name=f"cluster {lab} (n={int(mask.sum())})",
                marker=dict(size=3.2, color=color, opacity=0.8),
                customdata=custom,
                hovertemplate=(
                    f"cluster {lab}<br>%{{customdata[0]}} seg %{{customdata[1]}}"
                    "<br>PC1 %{x:.2f}<br>PC2 %{y:.2f}<br>PC3 %{z:.2f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[float(coords[mask, 0].mean())],
                y=[float(coords[mask, 1].mean())],
                z=[float(coords[mask, 2].mean())],
                mode="markers+text",
                text=[str(lab)],
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
                marker=dict(size=6, color=color, line=dict(color="black", width=1)),
            )
        )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=f"PC1 ({variance[0]:.1%})",
            yaxis_title=f"PC2 ({variance[1]:.1%})",
            zaxis_title=f"PC3 ({variance[2]:.1%})",
            aspectmode="cube",
        ),
        legend=dict(itemsizing="constant", title="cluster"),
        margin=dict(l=0, r=0, t=48, b=0),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="directory")


def _load_cohort(output_root: Path, cube: str, k: int):
    group_dir = endpoint_changepoints_dir(output_root, cube, own_elbow=True) / "clusters" / "endpoint"
    root_csv = group_dir / "segments_clustered.csv"
    k_csv = group_dir / "by_k" / f"k_{k:02d}" / "segments_clustered.csv"
    if not root_csv.is_file():
        raise FileNotFoundError(root_csv)
    if not k_csv.is_file():
        raise FileNotFoundError(k_csv)
    root = pd.read_csv(root_csv)
    labels = _labels_at_k(root, pd.read_csv(k_csv))
    features, names = build_feature_matrix_from_segments_df(root)
    coords, variance = project_cluster_pca(features, 3)
    if coords is None or variance is None or coords.shape[1] < 3:
        raise RuntimeError(f"{cube}: PCA did not return 3 components")
    return group_dir, root.reset_index(drop=True), labels, coords, variance, names


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    k_path = (
        Path(args.k_table)
        if args.k_table is not None
        else output_root / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv"
    )
    selected = _selected_k(k_path)
    missing = [cube for cube in KNOWN_CUBES if cube not in selected]
    if missing:
        raise ValueError(f"{k_path} has no selected_k for {', '.join(missing)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    si_dir = Path(args.si_dir) if str(args.si_dir).strip() else None

    with RunContext.from_namespace(args, name="plot_pca_clusters_3d"):
        loaded = {}
        rows = []
        for cube in KNOWN_CUBES:
            k = selected[cube]
            group_dir, meta, labels, coords, variance, _names = _load_cohort(
                output_root, cube, k
            )
            loaded[cube] = (k, meta, labels, coords, variance)
            title = f"{cube}, k = {k}"
            stem = out_dir / f"{cube}_k{k:02d}"
            fig = plt.figure(figsize=(8.2, 6.2))
            ax = fig.add_subplot(111, projection="3d")
            draw_pca_clusters_3d(
                ax, coords, labels, variance, title=title, marker_size=22
            )
            fig.tight_layout()
            png = stem.with_suffix(".png")
            fig.savefig(png, dpi=200, bbox_inches="tight")
            fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
            plt.close(fig)
            # Drop the placeholder written by the unused call if it landed.
            k_png = group_dir / "by_k" / f"k_{k:02d}" / "pca_clusters_3d.png"
            _copy(png, k_png.parent, k_png.name)
            _write_html(
                stem.with_suffix(".html"),
                title=f"{cube} endpoint segments, k = {k}",
                coords=coords,
                labels=labels,
                variance=variance,
                meta=meta,
            )
            coord_df = meta[["traj_id", "segment_id"]].copy()
            coord_df["cluster_label"] = labels
            coord_df["PC1"] = coords[:, 0]
            coord_df["PC2"] = coords[:, 1]
            coord_df["PC3"] = coords[:, 2]
            coord_df.to_csv(stem.with_suffix(".csv"), index=False)
            rows.append(
                {
                    "cohort": cube,
                    "selected_k": k,
                    "n_segments": int(len(labels)),
                    "n_clusters": int(len(set(labels.tolist()))),
                    "changepoints_dir": OWN_ELBOW_ENDPOINT_DIRS[cube],
                    "pc1": float(variance[0]),
                    "pc2": float(variance[1]),
                    "pc3": float(variance[2]),
                    "pc1_to_pc3": float(variance[:3].sum()),
                }
            )
            print(
                f"{cube}: k={k}, n={len(labels)}, "
                f"PC1–PC3 = {variance[0]:.1%} + {variance[1]:.1%} + {variance[2]:.1%} "
                f"= {variance[:3].sum():.1%}"
            )

        fig = plt.figure(figsize=(14.6, 9.4))
        for i, cube in enumerate(KNOWN_CUBES):
            k, _meta, labels, coords, variance = loaded[cube]
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            draw_pca_clusters_3d(
                ax,
                coords,
                labels,
                variance,
                title=f"{cube}, k = {k}",
                marker_size=9,
                show_legend=False,
            )
        fig.suptitle(
            "Endpoint-segment clusters in PC1–PC3",
            fontsize=13,
        )
        fig.text(
            0.5,
            0.012,
            "Numbers mark cluster centroids. Colors and numbers are labels within each cube, not matched across cubes.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="0.25",
        )
        fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.92, wspace=0.08, hspace=0.12)
        panel = out_dir / "pca_clusters_3d.png"
        fig.savefig(panel, dpi=200, bbox_inches="tight")
        fig.savefig(panel.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)

        summary = pd.DataFrame(rows)
        summary.to_csv(out_dir / "explained_variance.csv", index=False)
        print(f"Panel: {panel}")

        if si_dir is not None:
            print(f"Copied panel to {_copy(panel, si_dir, 'figureS_pca_clusters_3d.png')}")
            print(
                f"Copied panel to {_copy(panel.with_suffix('.pdf'), si_dir, 'figureS_pca_clusters_3d.pdf')}"
            )
            for cube in KNOWN_CUBES:
                k = selected[cube]
                stem = out_dir / f"{cube}_k{k:02d}"
                for suffix in (".png", ".pdf"):
                    src = stem.with_suffix(suffix)
                    dest_name = f"figureS_pca_clusters_3d_{cube}{suffix}"
                    print(f"Copied figure to {_copy(src, si_dir, dest_name)}")


if __name__ == "__main__":
    main()
