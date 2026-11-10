#!/usr/bin/env python3
"""List cluster handoffs worth opening, from the clustering and its site pairs.

A handoff is a pair of neighboring segments whose cluster labels differ.
It is listed when the site pairs matched to those clusters differ, the two
clusters are not symmetry copies of one deformation, and that ordered pair
occurs at most three times in the cohort. Same-label neighbors are splits
of one dwell and are not listed. Every cluster that owns a ranked site pair
still gets one typical segment and one long segment, so the matched states
can be opened even when their transitions are common.

Nothing here assumes a belt, a single elongated edge, or a Murata label.
The inputs are the selected-k segment paths and the site-pair ranking
(``structure_markers.csv``). Another assembly uses the same files from its
own clustering.

Example
-------
python scripts/recommend_transition_inspection.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir
from src.utils.run_log import RunContext

# One-SE k. Same cut as scripts/transition_frequencies.py.
SELECTED_K = {
    "BHHpH": 4,
    "BHHpM": 4,
    "BMHpH": 6,
    "BMHpM": 10,
    "BMMpH": 3,
    "BMMpM": 4,
}
COHORTS = ("BMMpM", "BMHpM", "BHHpM", "BMMpH", "BMHpH", "BHHpH")
STRIDE_PS = 5


def _paths(output_root: Path, cube: str) -> tuple[pd.DataFrame, Path]:
    k = SELECTED_K[cube]
    changepoints = endpoint_changepoints_dir(output_root, cube, own_elbow=True)
    path = (
        changepoints
        / "clusters"
        / "endpoint"
        / "by_k"
        / f"k_{k:02d}"
        / "trajectory_state_paths.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path), path


def _catalog(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = output_root / "metastable_structures_physical_k"
    markers = pd.read_csv(catalog / "structure_markers.csv")
    clusters = pd.read_csv(catalog / "clusters_ranked.csv")
    return markers, clusters


def _signatures(markers: pd.DataFrame, clusters: pd.DataFrame, cube: str) -> pd.DataFrame:
    """One row per cluster: its structure, family, and matched site pairs."""
    part = clusters[clusters["cohort"] == cube].copy()
    mine = markers[(markers["cohort"] == cube) & (markers["k"] == SELECTED_K[cube])]
    grouped = (
        mine.sort_values("rank")
        .groupby("best_cluster")["endpoint_label"]
        .apply(lambda labels: "; ".join(labels.astype(str)))
    )
    part["site_pairs"] = part["cluster_label"].map(grouped).fillna("")
    part["n_site_pairs"] = part["cluster_label"].map(mine.groupby("best_cluster").size()).fillna(0).astype(int)
    return part.set_index("cluster_label")


def _same_deformation(index: pd.DataFrame, src: int, dst: int) -> bool:
    """True when both clusters are symmetry copies of one site-pair pattern."""
    left = index.loc[src]
    right = index.loc[dst]
    if int(left.n_site_pairs) == 0 or int(right.n_site_pairs) == 0:
        return False
    family_src = left.family_id
    family_dst = right.family_id
    if pd.isna(family_src) or pd.isna(family_dst):
        return False
    return int(family_src) == int(family_dst) and int(left.n_orientations) > 1 and int(right.n_orientations) > 1


def _duration_ps(start: int, end: int) -> int:
    return int(end) - int(start) + STRIDE_PS


def recommend(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Handoffs, marked clusters, example segments, and the runs that contain a handoff."""
    markers, clusters = _catalog(output_root)
    handoff_rows: list[dict] = []
    cluster_rows: list[dict] = []
    segment_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    for cube in COHORTS:
        paths, source = _paths(output_root, cube)
        index = _signatures(markers, clusters, cube)
        source_rel = str(source.relative_to(output_root))
        paths = paths.sort_values(["traj_id", "path_position"])
        following = paths.groupby("traj_id", sort=False).shift(-1)
        nxt = pd.to_numeric(paths["next_cluster"], errors="coerce")
        changed = nxt.notna() & (paths["cluster_label"].to_numpy() != nxt.to_numpy())
        handoffs = paths.loc[changed].copy()
        handoffs["to_segment_id"] = following.loc[changed, "segment_id"].to_numpy()
        handoffs["to_start_frame"] = following.loc[changed, "start_frame"].to_numpy()
        handoffs["to_end_frame"] = following.loc[changed, "end_frame"].to_numpy()
        handoffs["to_cluster"] = nxt.loc[changed].astype(int).to_numpy()
        pair_count = handoffs.groupby(["cluster_label", "to_cluster"]).size().rename("pair_count")
        handoffs = handoffs.join(pair_count, on=["cluster_label", "to_cluster"])
        kept: list[dict] = []
        for rec in handoffs.itertuples(index=False):
            src = int(rec.cluster_label)
            dst = int(rec.to_cluster)
            if src not in index.index or dst not in index.index:
                continue
            n_src = int(index.loc[src, "n_site_pairs"])
            n_dst = int(index.loc[dst, "n_site_pairs"])
            if n_src == 0 and n_dst == 0:
                continue
            if _same_deformation(index, src, dst):
                continue
            if int(rec.pair_count) > 3:
                continue
            once = int(rec.pair_count) == 1
            reasons = []
            if n_src and n_dst:
                reasons.append("site-pair change")
            elif n_dst:
                reasons.append("enters marked cluster")
            else:
                reasons.append("leaves marked cluster")
            if once:
                reasons.append("singleton")
            elif int(rec.pair_count) <= 3:
                reasons.append("rare")
            score = 100 if n_src and n_dst else 60
            if once:
                score += 30
            elif int(rec.pair_count) <= 3:
                score += 10
            kept.append(
                {
                    "priority": score,
                    "reasons": "; ".join(reasons),
                    "cohort": cube,
                    "k": SELECTED_K[cube],
                    "traj_id": rec.traj_id,
                    "from_segment_id": int(rec.segment_id),
                    "to_segment_id": int(rec.to_segment_id),
                    "from_start_frame": int(rec.start_frame),
                    "from_end_frame": int(rec.end_frame),
                    "to_start_frame": int(rec.to_start_frame),
                    "to_end_frame": int(rec.to_end_frame),
                    "from_cluster": src,
                    "to_cluster": dst,
                    "from_structure": str(index.loc[src, "structure"]),
                    "to_structure": str(index.loc[dst, "structure"]),
                    "from_site_pairs": str(index.loc[src, "site_pairs"]),
                    "to_site_pairs": str(index.loc[dst, "site_pairs"]),
                    "pair_count": int(rec.pair_count),
                    "state_sequence": rec.state_sequence,
                    "source_csv": source_rel,
                }
            )
        handoff_rows.extend(kept)
        marked = index[index["n_site_pairs"] > 0]
        for label, rec in marked.iterrows():
            segs = paths[paths["cluster_label"].astype(int) == int(label)].copy()
            if segs.empty:
                continue
            segs["duration_ps"] = segs["end_frame"] - segs["start_frame"] + STRIDE_PS
            longest = segs.loc[segs["duration_ps"].idxmax()]
            median = float(segs["duration_ps"].median())
            typical = segs.iloc[(segs["duration_ps"] - median).abs().argmin()]
            cluster_rows.append(
                {
                    "cohort": cube,
                    "k": SELECTED_K[cube],
                    "cluster": int(label),
                    "structure": rec.structure,
                    "n_site_pairs": int(rec.n_site_pairs),
                    "site_pairs": rec.site_pairs,
                    "n_segments": int(len(segs)),
                    "n_trajectories": int(segs["traj_id"].nunique()),
                    "median_segment_ns": round(median / 1000.0, 2),
                    "exemplar_traj_id": typical.traj_id,
                    "exemplar_segment_id": int(typical.segment_id),
                    "exemplar_start_frame": int(typical.start_frame),
                    "exemplar_end_frame": int(typical.end_frame),
                    "source_csv": source_rel,
                }
            )
            for role, chosen in (("typical segment", typical), ("long segment", longest)):
                segment_rows.append(
                    {
                        "cohort": cube,
                        "k": SELECTED_K[cube],
                        "role": role,
                        "cluster": int(label),
                        "structure": rec.structure,
                        "site_pairs": rec.site_pairs,
                        "traj_id": chosen.traj_id,
                        "segment_id": int(chosen.segment_id),
                        "start_frame": int(chosen.start_frame),
                        "end_frame": int(chosen.end_frame),
                        "duration_ps": int(chosen.duration_ps),
                        "state_sequence": chosen.state_sequence,
                        "source_csv": source_rel,
                    }
                )
        listed = {row["traj_id"] for row in kept}
        for label, rec in marked.iterrows():
            segs = paths[paths["cluster_label"].astype(int) == int(label)]
            if segs.empty:
                continue
            duration = segs["end_frame"] - segs["start_frame"]
            typical = segs.iloc[(duration - duration.median()).abs().argmin()]
            if typical.traj_id in listed:
                continue
            listed.add(typical.traj_id)
            trajectory_rows.append(
                {
                    "priority": 40,
                    "n_handoffs": 0,
                    "reasons": "matched site-pair state",
                    "cohort": cube,
                    "k": SELECTED_K[cube],
                    "traj_id": typical.traj_id,
                    "state_sequence": typical.state_sequence,
                    "source_csv": source_rel,
                }
            )
        by_traj: dict[str, list[dict]] = {}
        for row in kept:
            by_traj.setdefault(row["traj_id"], []).append(row)
        for traj, events in by_traj.items():
            sequence = events[0]["state_sequence"]
            trajectory_rows.append(
                {
                    "priority": max(event["priority"] for event in events),
                    "n_handoffs": len(events),
                    "reasons": "; ".join(dict.fromkeys(event["reasons"] for event in events)),
                    "cohort": cube,
                    "k": SELECTED_K[cube],
                    "traj_id": traj,
                    "state_sequence": sequence,
                    "source_csv": source_rel,
                }
            )
    handoffs_out = pd.DataFrame(handoff_rows)
    if not handoffs_out.empty:
        handoffs_out = handoffs_out.sort_values(
            ["priority", "pair_count", "cohort", "traj_id", "from_start_frame"],
            ascending=[False, True, True, True, True],
        ).reset_index(drop=True)
    clusters_out = pd.DataFrame(cluster_rows)
    segments_out = pd.DataFrame(segment_rows)
    trajectories_out = pd.DataFrame(trajectory_rows)
    if not trajectories_out.empty:
        trajectories_out = trajectories_out.sort_values(
            ["priority", "n_handoffs", "cohort", "traj_id"],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
    return handoffs_out, clusters_out, segments_out, trajectories_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "transition_inspection",
    )
    args = parser.parse_args()
    with RunContext.from_namespace(args, name="recommend_transition_inspection"):
        handoffs, clusters, segments, trajectories = recommend(args.output_root)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        written = {
            "recommended_transitions.csv": handoffs,
            "recommended_clusters.csv": clusters,
            "recommended_segments.csv": segments,
            "recommended_trajectories.csv": trajectories,
        }
        for name, table in written.items():
            path = args.out_dir / name
            table.to_csv(path, index=False)
            print(f"wrote {path} ({len(table)} rows)")
        if handoffs.empty:
            return
        counts = handoffs.groupby(["cohort", "reasons"], sort=False).size().rename("n").reset_index()
        summary = args.out_dir / "recommended_transitions_by_reason.csv"
        counts.to_csv(summary, index=False)
        print(f"wrote {summary}")
        preview = handoffs[
            ["priority", "reasons", "cohort", "traj_id", "from_structure", "to_structure", "pair_count"]
        ].head(12)
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
