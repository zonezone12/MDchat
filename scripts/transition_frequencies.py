"""Murata-style transition frequencies for metastructure labels and our states.

Counts label changes between snapshots 1 ps apart, then scales the counts so
they sum to 1,000, the same unit as Murata's transition diagrams (events per
nanosecond of trajectory; one panel sums to 1,000). The runs start compact
and empty and last 5 ns, so the scaled counts are relaxation counts in that
unit, not equilibrium rates. Self-counts are stays. Off-diagonal counts are
label changes.

The three cohorts published with nanocubes 1₆, 2₆ and 3₆ are BMMpM, BMHpM
and BHHpM. Cluster labels are the selected Ward cut. Merged structures follow
``output/metastable_structures_physical_k/structures_summary.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.ChangepointAnalysis.cluster_k_diagnostics import load_segments_at_k
from src.ChangepointAnalysis.murata_criteria import (
    COHORT_TO_MURATA_SYSTEM,
    METASTRUCTURES,
    classify_murata_metastructure_columns,
)
from src.ChangepointAnalysis.reporting import endpoint_changepoints_dir

# Selected k (one-SE rule). See physical_k_selected.csv.
SELECTED_K = {
    "BHHpH": 4,
    "BHHpM": 4,
    "BMHpH": 6,
    "BMHpM": 10,
    "BMMpH": 3,
    "BMMpM": 4,
}

# Integers read from the published diagrams. Panel (a) hides the "other"
# self-count; 3 is the value that makes the panel sum to 1,000. Panel (b)
# sums to 998 and panel (c) to 1,001 as read, so one label on each is off by
# one or two. 2₆′ has no cohort in this repository.
PUBLISHED = {
    "1₆": {
        ("A", "A"): 742,
        ("B", "B"): 103,
        ("C1", "C1"): 28,
        ("C2", "C2"): 108,
        ("other", "other"): 3,
        ("A", "B"): 3,
        ("B", "A"): 3,
        ("B", "C1"): 3,
        ("C1", "B"): 3,
        ("C1", "C2"): 1,
        ("C2", "C1"): 1,
        ("C2", "other"): 1,
        ("other", "C2"): 1,
    },
    "2₆": {
        ("A", "A"): 480,
        ("B", "B"): 145,
        ("C1", "C1"): 14,
        ("C2", "C2"): 321,
        ("other", "other"): 4,
        ("A", "B"): 8,
        ("B", "A"): 8,
        ("B", "C1"): 3,
        ("C1", "B"): 3,
        ("C1", "C2"): 1,
        ("C2", "C1"): 1,
        ("B", "C2"): 4,
        ("C2", "B"): 4,
        ("C2", "other"): 1,
        ("other", "C2"): 1,
    },
    "3₆": {
        ("A", "A"): 99,
        ("B", "B"): 156,
        ("C1", "C1"): 53,
        ("C2", "C2"): 241,
        ("other", "other"): 369,
        ("A", "B"): 11,
        ("B", "A"): 11,
        ("B", "C1"): 11,
        ("C1", "B"): 11,
        ("C1", "C2"): 3,
        ("C2", "C1"): 3,
        ("B", "C2"): 6,
        ("C2", "B"): 6,
        ("B", "other"): 5,
        ("other", "B"): 5,
        ("C2", "other"): 5,
        ("other", "C2"): 5,
    },
}

ORDER_MURATA = list(METASTRUCTURES)


def _usecols(name: str) -> bool:
    return name in {
        "traj_id",
        "frame",
        "n_open_cation_pi",
        "n_guest_inside_cavity",
    } or str(name).startswith("equator_d1_e")


def _count_steps(labels: np.ndarray, frames: np.ndarray) -> dict[tuple[str, str], int]:
    if len(labels) < 2:
        return {}
    src = labels[:-1]
    dst = labels[1:]
    keep = np.diff(frames.astype(np.int64)) == 1
    keep &= pd.notna(src) & pd.notna(dst)
    src = src[keep].astype(str)
    dst = dst[keep].astype(str)
    if src.size == 0:
        return {}
    pairs = pd.Series(list(zip(src.tolist(), dst.tolist())))
    return {key: int(val) for key, val in pairs.value_counts().items()}


def _add_counts(total: dict, part: dict) -> None:
    for key, val in part.items():
        total[key] = total.get(key, 0) + int(val)


def _per_thousand(counts: dict) -> dict[tuple[str, str], float]:
    n = sum(counts.values())
    if n == 0:
        return {}
    return {key: 1000.0 * val / n for key, val in counts.items()}


def _load_structures(path: Path) -> dict[str, dict[int, str]]:
    """Map cluster id to the merged structure. Duplicate names stay separate."""
    df = pd.read_csv(path)
    out: dict[str, dict[int, str]] = {}
    for cube, sub in df.groupby("cohort", sort=False):
        names = sub["structure"].astype(str)
        duplicated = set(names[names.duplicated(keep=False)])
        mapping: dict[int, str] = {}
        for row in sub.itertuples(index=False):
            name = str(row.structure)
            if name in duplicated:
                name = f"{name} [{row.member_clusters}]"
            for token in str(row.member_clusters).split():
                mapping[int(token)] = name
        out[str(cube)] = mapping
    return out


def _load_segments(root: Path, cube: str, k: int) -> pd.DataFrame:
    """Selected-k labels from the own-elbow segmentation used in the catalog."""
    changepoints = endpoint_changepoints_dir(root, cube, own_elbow=True)
    df = load_segments_at_k(changepoints, k, group="endpoint")
    if "group" in df.columns:
        df = df[df["group"].astype(str) == "endpoint"]
    keep = ["traj_id", "segment_id", "start_frame", "end_frame", "cluster_label"]
    df = df[keep].copy()
    df["traj_id"] = df["traj_id"].astype(str)
    return df


def _paint_clusters(
    frames: np.ndarray,
    segs: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    """Label 1 ps frames from inclusive segment bounds.

    A gap of at most one 5 ps sample (next start - previous end <= 5) is
    filled with the previous cluster, so the changepoint is one 1 ps step.
    Wider holes stay unlabeled and are not crossed.
    """
    labels = np.full(len(frames), -1, dtype=int)
    index = {int(f): i for i, f in enumerate(frames)}
    rows = segs.sort_values(["start_frame", "segment_id"])
    bounds = [
        (int(r.start_frame), int(r.end_frame), int(r.cluster_label))
        for r in rows.itertuples(index=False)
    ]
    n_small_gaps = 0
    n_wide_gaps = 0
    wide_ps = 0
    for i, (start, end, lab) in enumerate(bounds):
        hi = end
        if i + 1 < len(bounds):
            nxt = bounds[i + 1][0]
            gap = nxt - end
            if 0 < gap <= 5:
                hi = nxt - 1
                n_small_gaps += 1
            elif gap > 5:
                n_wide_gaps += 1
                wide_ps += gap
        for frame in range(start, hi + 1):
            j = index.get(frame)
            if j is not None:
                labels[j] = lab
    covered = int(np.count_nonzero(labels >= 0))
    info = {
        "n_segments": len(bounds),
        "frames_covered": covered,
        "frames_total": int(len(frames)),
        "small_gaps_filled": n_small_gaps,
        "wide_gaps": n_wide_gaps,
        "wide_gap_ps": int(wide_ps),
    }
    return labels, info


def _long_rows(
    cube: str,
    kind: str,
    guest: str,
    counts: dict,
    *,
    system: str = "",
) -> list[dict]:
    rates = _per_thousand(counts)
    n_steps = sum(counts.values())
    rows = []
    keys = set(counts) | set(rates)
    for src, dst in sorted(keys):
        rows.append(
            {
                "cohort": cube,
                "murata_system": system,
                "label_kind": kind,
                "guest_filter": guest,
                "from_state": src,
                "to_state": dst,
                "n_steps": n_steps,
                "count": counts.get((src, dst), 0),
                "per_ns": rates.get((src, dst), 0.0),
                "is_change": src != dst,
            }
        )
    return rows


def _matrix_l1(rates: dict, published: dict) -> float:
    keys = set(published) | {k for k, v in rates.items() if v >= 0.5}
    total = 0.0
    for key in keys:
        total += abs(rates.get(key, 0.0) - published.get(key, 0.0))
    return total


def analyze(root: Path, out_dir: Path) -> None:
    structures = _load_structures(
        root / "metastable_structures_physical_k" / "structures_summary.csv"
    )
    detail_rows: list[dict] = []
    compare_rows: list[dict] = []
    compare_summary: list[dict] = []
    coverage_rows: list[dict] = []

    cubes = ["BMMpM", "BMHpM", "BHHpM", "BMMpH", "BMHpH", "BHHpH"]
    for cube in cubes:
        k = SELECTED_K[cube]
        system = COHORT_TO_MURATA_SYSTEM.get(cube, "")
        segs = _load_segments(root, cube, k)
        struct_of = structures.get(cube, {})
        motif_dir = root / f"{cube}_motif_rmsd" / cube
        paths = sorted(motif_dir.glob("*_motif_rmsd.csv"))
        murata_all: dict = {}
        murata_apo: dict = {}
        cluster_all: dict = {}
        cluster_apo: dict = {}
        struct_all: dict = {}
        struct_apo: dict = {}
        within_struct: dict[str, dict] = {}
        n_runs = 0
        n_runs_no_guest_col = 0
        covered = 0
        total_frames = 0
        gap_info = {
            "n_segments": 0,
            "small_gaps_filled": 0,
            "wide_gaps": 0,
            "wide_gap_ps": 0,
        }
        labeled_trajs = set(segs["traj_id"].unique())

        for path in paths:
            df = pd.read_csv(path, usecols=_usecols)
            if df.empty:
                continue
            df = df.sort_values("frame")
            frames = df["frame"].to_numpy(dtype=np.int64)
            d1_cols = [c for c in df.columns if str(c).startswith("equator_d1_e")]
            n_open = pd.to_numeric(df["n_open_cation_pi"], errors="coerce").to_numpy()
            n_elong = (
                df[d1_cols].apply(pd.to_numeric, errors="coerce").to_numpy() >= 7.0
            ).sum(axis=1)
            valid = np.isfinite(n_open)
            labels = np.full(len(df), None, dtype=object)
            labels[valid] = classify_murata_metastructure_columns(
                np.nan_to_num(n_open, nan=0).astype(int),
                n_elong.astype(int),
            )[valid]
            guest = pd.to_numeric(df["n_guest_inside_cavity"], errors="coerce").to_numpy()
            apo = np.isfinite(guest) & (guest < 1)
            if not np.isfinite(guest).any():
                n_runs_no_guest_col += 1

            tid = str(df["traj_id"].iloc[0])
            n_runs += 1
            total_frames += len(df)
            _add_counts(murata_all, _count_steps(labels, frames))
            apo_labels = labels.copy()
            apo_labels[~apo] = None
            _add_counts(murata_apo, _count_steps(apo_labels, frames))

            sub = segs[segs["traj_id"] == tid]
            if sub.empty:
                continue
            painted, info = _paint_clusters(frames, sub)
            covered += info["frames_covered"]
            for key in gap_info:
                gap_info[key] += info[key]
            clabels = np.array(
                [str(int(v)) if v >= 0 else None for v in painted], dtype=object
            )
            slabels = np.array(
                [
                    struct_of.get(int(v), f"cluster {int(v)}") if v >= 0 else None
                    for v in painted
                ],
                dtype=object,
            )
            _add_counts(cluster_all, _count_steps(clabels, frames))
            _add_counts(struct_all, _count_steps(slabels, frames))
            if len(frames) >= 2:
                step = np.diff(frames) == 1
                held = step & (painted[:-1] >= 0) & (painted[:-1] == painted[1:])
                src_m = labels[:-1][held]
                dst_m = labels[1:][held]
                st_m = slabels[:-1][held]
                ok = pd.notna(src_m) & pd.notna(dst_m)
                for structure, src, dst in zip(
                    st_m[ok].tolist(), src_m[ok].tolist(), dst_m[ok].tolist()
                ):
                    bucket = within_struct.setdefault(str(structure), {})
                    key = (str(src), str(dst))
                    bucket[key] = bucket.get(key, 0) + 1
            clabels_apo = clabels.copy()
            slabels_apo = slabels.copy()
            clabels_apo[~apo] = None
            slabels_apo[~apo] = None
            _add_counts(cluster_apo, _count_steps(clabels_apo, frames))
            _add_counts(struct_apo, _count_steps(slabels_apo, frames))

        motif_ids = set()
        for p in paths:
            stem = p.name[: -len("_motif_rmsd.csv")]
            motif_ids.add(stem)
        missing_motif = sorted(labeled_trajs - motif_ids)

        coverage_rows.append(
            {
                "cohort": cube,
                "murata_system": system,
                "k": k,
                "n_motif_runs": n_runs,
                "n_clustered_traj_missing_motif": len(missing_motif),
                "missing_motif": " ".join(missing_motif),
                "frames": total_frames,
                "frames_with_cluster": covered,
                "cluster_coverage": covered / total_frames if total_frames else np.nan,
                **gap_info,
                "n_runs_without_guest": n_runs_no_guest_col,
            }
        )
        for kind, guest, counts in (
            ("murata", "all", murata_all),
            ("murata", "apo_frames", murata_apo),
            ("cluster", "all", cluster_all),
            ("cluster", "apo_frames", cluster_apo),
            ("structure", "all", struct_all),
            ("structure", "apo_frames", struct_apo),
        ):
            detail_rows.extend(
                _long_rows(cube, kind, guest, counts, system=system)
            )
        for structure, counts in within_struct.items():
            detail_rows.extend(
                _long_rows(
                    cube,
                    "murata_inside_structure",
                    structure,
                    counts,
                    system=system,
                )
            )

        if system in PUBLISHED:
            for guest, counts in (("all", murata_all), ("apo_frames", murata_apo)):
                rates = _per_thousand(counts)
                published = PUBLISHED[system]
                l1 = _matrix_l1(rates, published)
                our_changes = sum(v for (a, b), v in rates.items() if a != b)
                pub_changes = sum(v for (a, b), v in published.items() if a != b)
                compare_summary.append(
                    {
                        "cohort": cube,
                        "murata_system": system,
                        "guest_filter": guest,
                        "l1_vs_published": l1,
                        "our_changes_per_ns": our_changes,
                        "published_changes_per_ns": pub_changes,
                        "n_steps": sum(counts.values()),
                    }
                )
                for src in ORDER_MURATA:
                    for dst in ORDER_MURATA:
                        key = (src, dst)
                        if key not in published and rates.get(key, 0.0) < 0.05:
                            continue
                        compare_rows.append(
                            {
                                "cohort": cube,
                                "murata_system": system,
                                "guest_filter": guest,
                                "from_state": src,
                                "to_state": dst,
                                "ours_per_ns": rates.get(key, 0.0),
                                "published_per_ns": published.get(key, 0.0),
                                "delta": rates.get(key, 0.0) - published.get(key, 0.0),
                            }
                        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(detail_rows).to_csv(out_dir / "transition_frequencies.csv", index=False)
    pd.DataFrame(compare_summary).to_csv(
        out_dir / "murata_diagram_summary.csv", index=False
    )
    pd.DataFrame(compare_rows).to_csv(
        out_dir / "murata_diagram_comparison.csv", index=False
    )
    pd.DataFrame(coverage_rows).to_csv(out_dir / "coverage.csv", index=False)
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/transition_frequencies"),
    )
    args = parser.parse_args()
    analyze(args.output_root, args.out_dir)


if __name__ == "__main__":
    main()
