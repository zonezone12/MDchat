"""Figure 4 and the supporting contact-signature figures for the other cubes.

Figure 4 is BMHpM. The same two panels are written for every other GSA
cube under ``docs/figures/si/``.

(a) Top site pairs by η² at the selected k. Bars are η²; color is the
    signed Cohen's d of the marked cluster (positive: longer distance
    there than in the other segments).
(b) Symmetry-aware correlations (Pearson r of deformation profiles after
    the best contact-graph relabeling). Each orientation family
    (``n_orientations`` > 1) is ordered around the equatorial belt and
    outlined. Bold entries meet the single-linkage merge line (``COPY_R``).

Reads the physical-k catalog. Does not recluster.

Example
-------
python scripts/plot_contact_signatures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.metastable_catalog import COPY_R, DEFAULT_D1_EDGES
from src.utils.run_log import RunContext

MAIN_CUBE = "BMHpM"
FAMILY_EDGE = "#d94801"
# A single-edge belt is used as the tick only when it covers this share
# of the cluster. Family members keep their edge below the cutoff: cluster 7
# of BMHpM is only half that edge and is still that orientation.
EDGE_TICK_PCT = 30.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot contact signatures of the selected states.")
    p.add_argument(
        "--catalog-dir",
        type=Path,
        default=Path("output/metastable_structures_physical_k"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/contact_signatures_figure"),
    )
    p.add_argument("--manuscript-dir", type=Path, default=Path("docs/figures"))
    p.add_argument(
        "--si-dir",
        type=Path,
        default=Path("docs/figures/si"),
        help="Where the other five cubes are copied (default: docs/figures/si)",
    )
    return p.parse_args()


def equatorial_cycle() -> tuple[str, ...]:
    """Belt edges in cycle order, as ``M0–M4`` labels.

    ``DEFAULT_D1_EDGES`` is one directed cycle. The same order is drawn
    on the hexagons in Figure 3.
    """
    nxt = {int(a): int(b) for a, b in DEFAULT_D1_EDGES}
    node = 0
    labels: list[str] = []
    for _ in range(len(nxt)):
        following = nxt[node]
        labels.append(f"M{node}–M{following}")
        node = following
    if node != 0:
        raise ValueError("Equatorial edges are not a single cycle through M0")
    return tuple(labels)


def _edge_index(edge: str, cycle: tuple[str, ...]) -> int:
    if edge in cycle:
        return cycle.index(edge)
    left, right = edge.split("–")
    flipped = f"{right}–{left}"
    if flipped in cycle:
        return cycle.index(flipped)
    raise ValueError(f"Edge {edge} is not on the equatorial cycle")


def dominant_edge(segment_belt: pd.DataFrame, cube: str, cluster: int) -> tuple[str | None, float]:
    """Single elongated edge with the most frames, and its share of the cluster."""
    sub = segment_belt[
        (segment_belt["cohort"] == cube) & (segment_belt["cluster_label"] == int(cluster))
    ]
    single = sub[sub["belt"].astype(str).str.startswith("edge ")]
    if single.empty:
        return None, 0.0
    top = single.sort_values("n_frames", ascending=False).iloc[0]
    edge = str(top["belt"]).removeprefix("edge ").replace("-", "–")
    return edge, float(top["pct_of_cluster"])


def _state_tag(structure: str) -> str:
    if structure.startswith("compact cage"):
        return "compact cage"
    if structure.startswith("mostly compact"):
        return "mostly compact"
    if structure.startswith("intact"):
        return "intact belt"
    if structure.startswith("loosened"):
        return "one replica"
    if structure.startswith("mixed belt"):
        return "mixed belt"
    if structure.startswith(("belt broken", "one belt break", "opened", "partly")):
        return "belt break"
    return structure


def panel_a_table(markers: pd.DataFrame, cube: str) -> pd.DataFrame:
    """Top site pairs of one cube, ranked by η²."""
    part = markers[markers["cohort"] == cube].copy()
    if part.empty:
        raise ValueError(f"{cube} has no rows in structure_markers.csv")
    part = part.sort_values(["eta_squared", "rank"], ascending=[False, True]).reset_index(drop=True)
    part["site_pair"] = part["endpoint_label"].astype(str).str.replace("-", "–", n=1, regex=False)
    part["display_rank"] = np.arange(1, len(part) + 1)
    return part


def _cluster_row(
    rec,
    belts: pd.DataFrame,
    cube: str,
    cycle: tuple[str, ...],
    *,
    in_family: bool,
    block: float,
) -> dict:
    edge, edge_pct = dominant_edge(belts, cube, int(rec.cluster_label))
    show_edge = edge is not None and (in_family or edge_pct >= EDGE_TICK_PCT)
    tag = edge if show_edge else _state_tag(str(rec.structure))
    return {
        "cohort": cube,
        "cluster_label": int(rec.cluster_label),
        "family_id": int(rec.family_id),
        "in_family": in_family,
        "block": block,
        "edge": edge or "",
        "edge_pct": edge_pct,
        "tick": f"{int(rec.cluster_label)}\n{tag}",
        "pct_of_cube": float(rec.pct_of_cube),
        "structure": rec.structure,
        "cycle_index": _edge_index(edge, cycle) if edge else np.nan,
        "n_orientations": int(rec.n_orientations),
    }


def panel_b_table(
    clusters: pd.DataFrame,
    symmetry: pd.DataFrame,
    segment_belt: pd.DataFrame,
    cube: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clusters in display order, and their symmetry-aware correlation matrix.

    Orientation families (more than one cluster) come first, largest family
    first, each ordered around the equatorial belt so the outline is one
    block. The remaining clusters follow by share of apo frames.
    """
    part = clusters[clusters["cohort"] == cube].copy()
    if part.empty:
        raise ValueError(f"{cube} is missing from clusters_ranked.csv")
    cycle = equatorial_cycle()
    belts = segment_belt[segment_belt["cohort"] == cube]
    rows: list[dict] = []
    multi = (
        part[part["n_orientations"] > 1]
        .groupby("family_id", sort=False)["pct_of_cube"]
        .sum()
        .sort_values(ascending=False)
    )
    for block, family_id in enumerate(multi.index):
        members = [
            _cluster_row(rec, belts, cube, cycle, in_family=True, block=float(block))
            for rec in part[part["family_id"] == family_id].itertuples(index=False)
        ]
        members.sort(
            key=lambda row: (
                row["cycle_index"] if np.isfinite(row["cycle_index"]) else 99,
                row["cluster_label"],
            )
        )
        rows.extend(members)

    singles = part[part["n_orientations"] <= 1].sort_values("pct_of_cube", ascending=False)
    for rec in singles.itertuples(index=False):
        rows.append(_cluster_row(rec, belts, cube, cycle, in_family=False, block=np.nan))
    order = pd.DataFrame(rows)

    labels = order["cluster_label"].tolist()
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.eye(len(labels), dtype=float)
    pairs = symmetry[symmetry["cohort"] == cube]
    seen = 0
    for rec in pairs.itertuples(index=False):
        a = int(rec.cluster_a)
        b = int(rec.cluster_b)
        if a not in index or b not in index:
            continue
        matrix[index[a], index[b]] = float(rec.r_best_relabelling)
        matrix[index[b], index[a]] = float(rec.r_best_relabelling)
        seen += 1
    expected = len(labels) * (len(labels) - 1) // 2
    if seen != expected:
        raise ValueError(f"Expected {expected} {cube} pairs, found {seen}")
    corr = pd.DataFrame(matrix, index=labels, columns=labels)
    corr.index.name = "cluster_label"
    return order, corr


def _box_note(order: pd.DataFrame) -> str:
    blocks = order.loc[order["in_family"], "block"]
    n_blocks = int(blocks.nunique()) if not blocks.empty else 0
    rule = f"Bold: r ≥ {COPY_R:.2f}."
    if n_blocks == 0:
        return f"No orientation family. {rule}"
    if n_blocks == 1:
        n = int(order["in_family"].sum())
        if n == 6:
            return f"Orange box: six belt-break orientations. {rule}"
        return f"Orange box: {n}-orientation family. {rule}"
    return f"Orange boxes: orientation families. {rule}"


def _signed(value: float, digits: int) -> str:
    return f"{value:+.{digits}f}".replace("-", "−")


def _ink(rgba: tuple[float, float, float, float]) -> str:
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "white" if luminance < 0.55 else "0.12"


def plot_figure(pairs: pd.DataFrame, order: pd.DataFrame, corr: pd.DataFrame, out: Path) -> Path:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    cube = str(pairs["cohort"].iloc[0])
    k = int(pairs["k"].iloc[0])
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(13.4, 6.5),
        gridspec_kw={"width_ratios": [1.12, 1.0], "wspace": 0.38},
    )

    y = np.arange(len(pairs))
    eta = pairs["eta_squared"].to_numpy(dtype=float)
    effect = pairs["cohens_d_best"].to_numpy(dtype=float)
    d_lim = max(2.0, float(np.ceil(np.max(np.abs(effect)) / 2.0) * 2.0))
    d_norm = Normalize(vmin=-d_lim, vmax=d_lim)
    d_cmap = plt.get_cmap("RdBu_r")
    ax_a.barh(
        y,
        eta,
        height=0.72,
        color=[d_cmap(d_norm(d)) for d in effect],
        edgecolor="0.15",
        linewidth=0.35,
        zorder=2,
    )
    # Numbers sit in a column past η² = 1 so the bar scale stays 0–1.
    eta_col = 1.04
    d_col = 1.18
    for yi, eta_i, d_i in zip(y, eta, effect):
        ax_a.text(eta_col, yi, f"{eta_i:.2f}", va="center", ha="left", fontsize=7.5, color="0.15", zorder=3)
        ax_a.text(
            d_col,
            yi,
            _signed(float(d_i), 1),
            va="center",
            ha="left",
            fontsize=7.5,
            color="0.15",
            zorder=3,
        )
    labels = [
        f"{int(rec.display_rank)}   {rec.site_pair}\n{rec.pair_relation}, cluster {int(rec.best_cluster)}"
        for rec in pairs.itertuples(index=False)
    ]
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels, fontsize=8)
    ax_a.set_ylim(len(pairs) - 0.55, -1.05)
    ax_a.set_xlim(0, 1.38)
    ax_a.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_a.set_xlabel(r"$\eta^{2}$")
    ax_a.grid(axis="x", alpha=0.25, zorder=0)
    ax_a.text(eta_col, -0.72, r"$\eta^{2}$", ha="left", va="center", fontsize=7.5, color="0.35")
    ax_a.text(d_col, -0.72, r"$d$", ha="left", va="center", fontsize=7.5, color="0.35")
    cbar_a = fig.colorbar(
        plt.cm.ScalarMappable(norm=d_norm, cmap=d_cmap),
        ax=ax_a,
        fraction=0.046,
        pad=0.03,
    )
    cbar_a.set_label("Cohen's d\n(+ longer)")
    cbar_a.set_ticks(np.linspace(-d_lim, d_lim, 5))
    ax_a.text(-0.22, 1.04, "(a)", transform=ax_a.transAxes, fontweight="bold", va="bottom", fontsize=12)
    ax_a.text(
        0.42,
        1.04,
        f"{cube}, k = {k}",
        transform=ax_a.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    labels_b = order["cluster_label"].tolist()
    values = corr.loc[labels_b, labels_b].to_numpy(dtype=float)
    r_cmap = plt.get_cmap("YlOrRd")
    r_norm = Normalize(vmin=0.0, vmax=1.0)
    ax_b.imshow(values, cmap=r_cmap, norm=r_norm, origin="upper", interpolation="nearest")
    n = len(labels_b)
    ax_b.set_xticks(np.arange(n))
    ax_b.set_yticks(np.arange(n))
    tick_size = 8.5 if n <= 4 else 7.5
    ax_b.set_xticklabels(order["tick"].tolist(), fontsize=tick_size, rotation=40, ha="right")
    ax_b.set_yticklabels(order["tick"].tolist(), fontsize=tick_size)
    ax_b.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax_b.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax_b.grid(which="minor", color="white", linewidth=0.7)
    ax_b.tick_params(which="minor", length=0)
    for i in range(n):
        for j in range(n):
            value = float(values[i, j])
            if i == j:
                text, weight = "1", "regular"
            else:
                text = f"{value:.2f}".replace("-", "−")
                weight = "bold" if value >= COPY_R else "regular"
            ax_b.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9.0 if n <= 4 else 7.2 if n <= 6 else 6.4,
                color=_ink(r_cmap(r_norm(value))),
                fontweight=weight,
            )
    positioned = order.reset_index(drop=True)
    family_rows = positioned[positioned["block"].notna()]
    for _, grp in family_rows.groupby("block", sort=True):
        idxs = grp.index.to_numpy()
        if int(idxs.max() - idxs.min() + 1) != len(idxs):
            raise ValueError(f"{cube} orientation family is not a contiguous block")
        side = float(len(idxs))
        origin = float(idxs.min()) - 0.5
        ax_b.add_patch(
            Rectangle(
                (origin, origin),
                side,
                side,
                fill=False,
                edgecolor=FAMILY_EDGE,
                linewidth=1.8,
                zorder=4,
            )
        )
    cbar_b = fig.colorbar(
        plt.cm.ScalarMappable(norm=r_norm, cmap=r_cmap),
        ax=ax_b,
        fraction=0.046,
        pad=0.03,
    )
    cbar_b.set_label("symmetry-aware r")
    ax_b.set_xlabel(_box_note(order), fontsize=8.5, labelpad=8)
    ax_b.text(-0.16, 1.04, "(b)", transform=ax_b.transAxes, fontweight="bold", va="bottom", fontsize=12)
    ax_b.text(
        0.46,
        1.04,
        f"{cube} clusters",
        transform=ax_b.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def _copy_manuscript(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    try:
        dest.write_bytes(src.read_bytes())
    except OSError:
        dest = dest_dir / f"{src.stem}_cube{src.suffix}"
        dest.write_bytes(src.read_bytes())
    return dest


def _write_cube(
    cube: str,
    *,
    markers: pd.DataFrame,
    clusters: pd.DataFrame,
    symmetry: pd.DataFrame,
    segment_belt: pd.DataFrame,
    out_dir: Path,
    stem: str,
) -> Path:
    pairs = panel_a_table(markers, cube)
    order, corr = panel_b_table(clusters, symmetry, segment_belt, cube)
    pairs.to_csv(out_dir / f"{stem}_top_site_pairs.csv", index=False)
    order.to_csv(out_dir / f"{stem}_cluster_order.csv", index=False)
    corr.to_csv(out_dir / f"{stem}_symmetry_r.csv")
    fig_path = plot_figure(pairs, order, corr, out_dir / f"{stem}.png")
    outlined = order.loc[order["in_family"], ["block", "cluster_label", "edge", "structure"]]
    print(f"{cube}: {fig_path.name}")
    if outlined.empty:
        print("  no orientation family")
    else:
        print(outlined.to_string(index=False))
    return fig_path


def main() -> None:
    args = parse_args()
    catalog = Path(args.catalog_dir)
    out_dir = Path(args.out_dir)
    si_out = out_dir / "si"
    out_dir.mkdir(parents=True, exist_ok=True)
    si_out.mkdir(parents=True, exist_ok=True)
    with RunContext.from_namespace(args, name="plot_contact_signatures"):
        markers = pd.read_csv(catalog / "structure_markers.csv")
        clusters = pd.read_csv(catalog / "clusters_ranked.csv")
        symmetry = pd.read_csv(catalog / "all_cluster_symmetry.csv")
        segment_belt = pd.read_csv(catalog / "min_share_0" / "segment_belt.csv")
        fig_path = _write_cube(
            MAIN_CUBE,
            markers=markers,
            clusters=clusters,
            symmetry=symmetry,
            segment_belt=segment_belt,
            out_dir=out_dir,
            stem="figure4_contact_signatures",
        )
        # Keep the filenames the main-text figure already used.
        panel_a_table(markers, MAIN_CUBE).to_csv(out_dir / "top_site_pairs.csv", index=False)
        order, corr = panel_b_table(clusters, symmetry, segment_belt, MAIN_CUBE)
        order.to_csv(out_dir / "bmhpm_cluster_order.csv", index=False)
        corr.to_csv(out_dir / "bmhpm_symmetry_r.csv")
        manuscript_dir = str(args.manuscript_dir).strip()
        if manuscript_dir:
            for src in (fig_path, fig_path.with_suffix(".pdf")):
                print(f"Copied figure to {_copy_manuscript(src, Path(manuscript_dir))}")
        si_dir = str(args.si_dir).strip()
        for cube in KNOWN_CUBES:
            if cube == MAIN_CUBE:
                continue
            si_path = _write_cube(
                cube,
                markers=markers,
                clusters=clusters,
                symmetry=symmetry,
                segment_belt=segment_belt,
                out_dir=si_out,
                stem=f"figureS_contact_signatures_{cube}",
            )
            if si_dir:
                for src in (si_path, si_path.with_suffix(".pdf")):
                    print(f"Copied figure to {_copy_manuscript(src, Path(si_dir))}")


if __name__ == "__main__":
    main()
