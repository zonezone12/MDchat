"""Generate a comprehensive summary.md for output/nh4no3_pt."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output" / "nh4no3_pt"


def main() -> None:
    d = OUT
    stats = pd.read_csv(d / "summary_stats.csv")
    occ = (
        pd.read_csv(d / "cluster_occupancy.csv")
        if (d / "cluster_occupancy.csv").is_file()
        else None
    )
    all_cv = (
        pd.read_csv(d / "clusters/all_cv/segments_clustered.csv")
        if (d / "clusters/all_cv/segments_clustered.csv").is_file()
        else None
    )

    water_bits = []
    for tag in ["m1n1_CLMD", "m1n1_PIMD"]:
        wp = d / f"corr_water_{tag}.csv"
        if wp.is_file():
            water_bits.append((tag, pd.read_csv(wp).head(5)))

    corr_bits = []
    for tag in stats["tag"]:
        cp = d / f"corr_{tag}.csv"
        if cp.is_file():
            corr_bits.append((tag, pd.read_csv(cp).head(8)))

    cluster_lines = []
    if all_cv is not None and not all_cv.empty:
        for cl, sub in all_cv.groupby("cluster_label"):
            cluster_lines.append(
                {
                    "cluster": int(cl),
                    "n_seg": len(sub),
                    "n_frames": int(sub["n_frames"].sum()),
                    "delta_mean": float(sub["delta_mean"].mean()),
                    "frac_amine": float(sub["frac_on_amine"].mean())
                    if "frac_on_amine" in sub
                    else float("nan"),
                    "frac_shared": float(sub["frac_shared"].mean())
                    if "frac_shared" in sub
                    else float("nan"),
                    "proton_rg": float(sub["proton_rg_mean"].mean())
                    if "proton_rg_mean" in sub
                    else float("nan"),
                    "trajs": ", ".join(sorted(sub["traj_id"].unique())),
                }
            )

    sil_lines = []
    clusters_root = d / "clusters"
    if clusters_root.is_dir():
        for cohort_dir in sorted(clusters_root.glob("*")):
            csv = cohort_dir / "segments_clustered.csv"
            if not csv.is_file():
                continue
            df = pd.read_csv(csv)
            k = (
                int(df["n_clusters"].iloc[0])
                if "n_clusters" in df.columns
                else int(df["cluster_label"].nunique())
            )
            sil = (
                float(df["silhouette"].iloc[0])
                if "silhouette" in df.columns
                else float("nan")
            )
            sil_lines.append((cohort_dir.name, len(df), k, sil))

    artifacts = [
        str(p.relative_to(d)).replace("\\", "/")
        for p in sorted(d.rglob("*"))
        if p.is_file()
    ]

    lines: list[str] = []
    lines.append("# NH4NO3 proton-transfer analysis summary")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Output folder: `{d.resolve()}`")
    lines.append(
        "- Systems: **m1n0** (NH₃···HNO₃, no water) vs **m1n1** (NH₃···HNO₃ + H₂O)"
    )
    lines.append("- Methods: **CLMD** (classical) vs **PIMD** (60-bead ring polymer)")
    lines.append(
        "- Collective variable: `δ = d(N–H) − d(O–H)` (Å). "
        "Positive ⇒ proton on acid O; negative ⇒ on amine N."
    )
    lines.append(
        "- Atom indices (0-based, frame-0 geometry): "
        "m1n0 → N=0, H=8, O=6; m1n1 → N=0, H=7, O=5, water O=8 / H=10,11"
    )
    lines.append(
        "- Changepoints: Pelt (`rbf` on δ; `l2` on z-scored pairwise distances), "
        "`min_size=20`"
    )
    lines.append(
        "- Segment clustering: Ward hierarchical on CPD `pairwise_distances` "
        "segments; auto-k by silhouette"
    )
    lines.append("")
    lines.append("## Key findings")
    lines.append("")
    lines.append(
        "1. **Water enables proton transfer.** m1n0 never visits the amine-bound "
        "side of δ (0 sign flips). m1n1 shows transfer events "
        "(8 CLMD / 18 PIMD sign flips)."
    )
    lines.append(
        "2. **Nuclear quantum effects amplify transfer.** With water present, "
        "PIMD roughly doubles δ sign flips vs CLMD (8 → 18) and adds "
        "~0.15 Å acid-proton ring-polymer radius of gyration."
    )
    lines.append(
        "3. **Water couples geometrically to the CV.** Among water-involving pairs, "
        "nitrate O6–water O8 distance correlates most strongly with δ "
        "(Pearson r ≈ 0.30 CLMD, 0.38 PIMD)."
    )
    lines.append(
        "4. **Segment clusters recover the transfer-active state.** Cross-system "
        "CV clustering (`clusters/all_cv`) isolates a rare transfer-active cluster "
        "(δ̄ ≈ −0.13) occupied only by m1n1 short segments; the majority cluster "
        "is acid-bound (δ̄ ≈ +0.66)."
    )
    lines.append("")
    lines.append("## Per-trajectory statistics")
    lines.append("")
    lines.append(
        "| tag | frames | beads | δ mean±std | frac acid | frac amine | "
        "frac shared | sign flips | proton Rg mean | n_cp (δ) | "
        "n_cp (pairwise) | top corr |"
    )
    lines.append(
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    for _, row in stats.iterrows():
        lines.append(
            f"| `{row['tag']}` | {int(row['n_frames'])} | {int(row['n_beads'])} | "
            f"{row['delta_mean']:.3f}±{row['delta_std']:.3f} | "
            f"{row['frac_on_acid']:.3f} | {row['frac_on_amine']:.3f} | "
            f"{row['frac_shared']:.3f} | {int(row['n_sign_flips'])} | "
            f"{row['proton_rg_mean']:.4f} | {int(row['n_cp_delta'])} | "
            f"{int(row['n_cp_pairwise'])} | "
            f"`{row['top_corr_feature']}` ({row['top_corr_pearson']:.3f}) |"
        )
    lines.append("")
    lines.append("Shared-proton region uses `|δ| < 0.05` Å.")
    lines.append("")
    lines.append("## Water–CV correlations (m1n1)")
    lines.append("")
    for tag, w in water_bits:
        lines.append(f"### `{tag}` top water-involving features vs δ")
        lines.append("")
        lines.append("| rank | feature | Pearson r | overall rank |")
        lines.append("|---:|---|---:|---:|")
        for i, r in w.iterrows():
            overall = (
                int(r["overall_rank"])
                if "overall_rank" in r and pd.notna(r["overall_rank"])
                else ""
            )
            rank = int(r["water_rank"]) if "water_rank" in r else i + 1
            lines.append(
                f"| {rank} | `{r['feature']}` | {r['pearson_r']:.3f} | {overall} |"
            )
        lines.append("")
    lines.append(
        "Interpretation: O6 is a nitrate oxygen; O8 is the water oxygen. "
        "Modulation of the nitrate–water O···O contact tracks the proton-transfer "
        "coordinate, more strongly under PIMD."
    )
    lines.append("")
    lines.append("## Top pairwise features vs δ (all systems)")
    lines.append("")
    for tag, c in corr_bits:
        lines.append(f"### `{tag}`")
        lines.append("")
        lines.append("| feature | Pearson r | Spearman r |")
        lines.append("|---|---:|---:|")
        for _, r in c.iterrows():
            lines.append(
                f"| `{r['feature']}` | {r['pearson_r']:.3f} | {r['spearman_r']:.3f} |"
            )
        lines.append("")
    lines.append(
        "As expected, the donor/acceptor contacts defining δ (`N*–H*`, `O*–H*`) "
        "dominate. Secondary contacts (including water in m1n1) rank below these."
    )
    lines.append("")
    lines.append("## Changepoint segments → clusters")
    lines.append("")
    lines.append(
        "Per-frame pairwise distances were segmented with multivariate Pelt; "
        "each segment was summarized (feature means + δ stats + `log10(n_frames)`) "
        "and clustered with Ward linkage."
    )
    lines.append("")
    lines.append("### Cohort silhouette scores")
    lines.append("")
    lines.append("| cohort | n_segments | k | silhouette |")
    lines.append("|---|---:|---:|---:|")
    for name, nseg, k, sil in sil_lines:
        lines.append(f"| `{name}` | {nseg} | {k} | {sil:.3f} |")
    lines.append("")

    if cluster_lines:
        lines.append("### Cross-system CV clusters (`clusters/all_cv`)")
        lines.append("")
        lines.append(
            "| cluster | role | n_segments | n_frames | δ̄ | frac amine | "
            "frac shared | proton Rḡ | trajectories |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
        for cl in cluster_lines:
            role = "transfer-active" if cl["delta_mean"] < 0 else "acid-bound"
            lines.append(
                f"| {cl['cluster']} | {role} | {cl['n_seg']} | {cl['n_frames']} | "
                f"{cl['delta_mean']:.3f} | {cl['frac_amine']:.3f} | "
                f"{cl['frac_shared']:.3f} | {cl['proton_rg']:.4f} | {cl['trajs']} |"
            )
        lines.append("")
        lines.append(
            "Only **m1n1** populates the transfer-active cluster; "
            "m1n0 remains entirely acid-bound."
        )
        lines.append("")

    if occ is not None:
        lines.append("### Cluster occupancy (all_cv)")
        lines.append("")
        view = (
            occ[occ["cohort"] == "all_cv"]
            if "all_cv" in set(occ["cohort"])
            else occ
        )
        lines.append("| traj | cluster | n_segments | n_frames |")
        lines.append("|---|---:|---:|---:|")
        for _, row in view.iterrows():
            lines.append(
                f"| `{row['traj_id']}` | {int(row['cluster_label'])} | "
                f"{int(row['n_segments'])} | {int(row['n_frames'])} |"
            )
        lines.append("")
        m1 = occ[occ["cohort"] == "m1n1_pairwise"]
        if not m1.empty:
            lines.append("### m1n1 pairwise-geometry clusters")
            lines.append("")
            lines.append("| traj | cluster | n_segments | n_frames |")
            lines.append("|---|---:|---:|---:|")
            for _, row in m1.iterrows():
                lines.append(
                    f"| `{row['traj_id']}` | {int(row['cluster_label'])} | "
                    f"{int(row['n_segments'])} | {int(row['n_frames'])} |"
                )
            lines.append("")
            lines.append(
                "PIMD visits two pairwise-geometry regimes; CLMD stays in one. "
                "This is consistent with quantum delocalization opening additional "
                "transfer-coupled geometries."
            )
            lines.append("")

    movies = sorted(d.glob("movie_*.gif"))
    if movies:
        lines.append("## Movies: water approach induces transfer")
        lines.append("")
        lines.append(
            "Animations of the deepest transfer event in the water system. The "
            "transferring proton is gold; the water molecule is cyan; the dotted "
            "cyan line is the amine-N to water-O approach distance. Right panels "
            "show `δ` and `d(N···Ow)` with a moving cursor."
        )
        lines.append("")
        for gif in movies:
            still = gif.with_name(gif.stem + "_peak.png")
            lines.append(f"- `{gif.name}`")
            if still.is_file():
                lines.append(f"  - peak still: `{still.name}`")
                lines.append("")
                lines.append(f"![{gif.stem}]({still.name})")
                lines.append("")
        lines.append(
            "At the transfer minimum the water oxygen sits ~2.7-2.8 Å from the "
            "amine nitrogen (H-bond contact), stabilizing the nascent NH4+ and "
            "coinciding with `δ < 0`. Water approach and transfer are concurrent "
            "(near-zero lag), and the effect is stronger under PIMD."
        )
        lines.append("")
        lines.append(
            "Regenerate: `python scripts/make_pt_movie.py --system m1n1 "
            "--method PIMD` (add `--center <frame>` to target a specific event)."
        )
        lines.append("")

    field_movies = sorted(d.glob("field_movie_*.gif"))
    if field_movies:
        lines.append("## Field movies: PIMD bead delocalization")
        lines.append("")
        lines.append(
            "Ring-polymer field view: every atom's 60 PIMD beads are drawn as a "
            "translucent point cloud wrapped in a convex-hull surface, so the "
            "occupied space reflects each atom's quantum delocalization. The "
            "transferring proton (gold) smears between donor O and acceptor N "
            "during transfer; the bottom-right panel tracks its ring-polymer "
            "radius of gyration."
        )
        lines.append("")
        for gif in field_movies:
            still = gif.with_name(gif.stem + "_peak.png")
            lines.append(f"- `{gif.name}`")
            if still.is_file():
                lines.append(f"  - peak still: `{still.name}`")
                lines.append("")
                lines.append(f"![{gif.stem}]({still.name})")
                lines.append("")
        lines.append(
            "Regenerate: `python scripts/make_pt_field_movie.py --system m1n1 "
            "--method PIMD --hull-atoms light` "
            "(`--hull-atoms proton|light|all` selects which atoms get a hull field)."
        )
        lines.append("")

    lines.append("## Methods notes")
    lines.append("")
    lines.append(
        "- **Features:** all unique pairwise interatomic distances on the "
        "bead-averaged (centroid) geometry each frame."
    )
    lines.append(
        "- **CV:** `δ = d(N–H) − d(O–H)` with fixed donor/acceptor/H from "
        "frame-0 bonding detection."
    )
    lines.append(
        "- **PIMD delocalization:** acid-proton ring-polymer Rg and per-bead "
        "δ standard deviation."
    )
    lines.append(
        "- **Changepoints:** `src.utils.ruptures_utils.detect_changepoints` (Pelt). "
        "Default penalty ≈ `log(n_frames)` yields many fine segments "
        "(~230/traj on pairwise); raise `--penalty` for coarser regimes."
    )
    lines.append(
        "- **Clustering:** `src.utils.metastable_states.cluster_metastable_states` "
        "(Ward + silhouette auto-k)."
    )
    lines.append("")
    lines.append("## How to regenerate")
    lines.append("")
    lines.append("```bash")
    lines.append("# Full analysis + clustering")
    lines.append(
        "python scripts/run_proton_transfer_analysis.py "
        "--output-dir output/nh4no3_pt"
    )
    lines.append("")
    lines.append("# Clustering only (reuse existing features/changepoints CSVs)")
    lines.append(
        "python scripts/run_proton_transfer_analysis.py --cluster-only "
        "--output-dir output/nh4no3_pt --auto-select-k --k-max 10"
    )
    lines.append("")
    lines.append("# Coarser changepoints (recommended for fewer, longer segments)")
    lines.append(
        "python scripts/run_proton_transfer_analysis.py --penalty 50 "
        "--output-dir output/nh4no3_pt_p50"
    )
    lines.append("```")
    lines.append("")
    lines.append("## Artifact manifest")
    lines.append("")
    lines.append("### Core per-trajectory outputs")
    lines.append("")
    for tag in stats["tag"]:
        lines.append(f"- `{tag}`:")
        for name in [
            f"features_{tag}.csv",
            f"corr_{tag}.csv",
            f"corr_{tag}.png",
            f"delta_cp_{tag}.png",
            f"changepoints_{tag}.csv",
            f"breakpoints_{tag}.csv",
        ]:
            if (d / name).is_file():
                lines.append(f"  - `{name}`")
        wname = f"corr_water_{tag}.csv"
        if (d / wname).is_file():
            lines.append(f"  - `{wname}`")
        sname = f"segment_stats_{tag}_pairwise_distances.csv"
        if (d / sname).is_file():
            lines.append(f"  - `{sname}`")
    lines.append("")
    lines.append("### Cohort / summary outputs")
    lines.append("")
    for name in [
        "summary_stats.csv",
        "delta_distributions.png",
        "all_segment_stats_pairwise_distances.csv",
        "all_segments_clustered.csv",
        "cluster_occupancy.csv",
        "summary.md",
    ]:
        if (d / name).is_file() or name == "summary.md":
            lines.append(f"- `{name}`")
    for gif in sorted(d.glob("movie_*.gif")) + sorted(d.glob("field_movie_*.gif")):
        lines.append(f"- `{gif.name}`")
        still = gif.with_name(gif.stem + "_peak.png")
        if still.is_file():
            lines.append(f"- `{still.name}`")
    lines.append("")
    lines.append("### Cluster directories")
    lines.append("")
    if clusters_root.is_dir():
        for cohort_dir in sorted(clusters_root.glob("*")):
            if not cohort_dir.is_dir():
                continue
            files = sorted(p.name for p in cohort_dir.iterdir() if p.is_file())
            lines.append(
                f"- `clusters/{cohort_dir.name}/`: "
                + ", ".join(f"`{f}`" for f in files)
            )
    lines.append("")
    lines.append(f"_Total files under this folder: {len(artifacts)}_")
    lines.append("")

    text = "\n".join(lines)
    out_path = d / "summary.md"
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path} ({len(text)} chars, {len(lines)} lines)")


if __name__ == "__main__":
    main()
