#!/usr/bin/env python3
"""Inspect clusters whose merge status is not settled by the catalog rule.

A cluster can hold segments that are only weakly deformed next to strongly
deformed ones. Weak segments pull a cluster's deformation profile toward
the grand mean. This script asks, for every pair that looks tangled,
whether that dilution is what kept the pair apart (or weakly chained), or
whether the strongly deformed parts are already different deformations.

A pair of one cube is tangled when, under the catalog merge (single
linkage, r ≥ ``COPY_R``):

- ``near_miss``: different families, but best-relabelled r ≥ ``--r-min``
- ``chain_weak``: same family, but the direct r is below ``COPY_R``

Those rules use only the all-cluster symmetry table. They do not depend on
which clusters were contrasted for site-pair ranking.

The split is label-free. Each cluster is cut in two by Ward on its
standardized site-pair segment means (``core_split``); the part farther from
the grand mean is the core, the other the weak part. No structural
threshold enters. Profiles are the catalog's: the mean of the 960 site-pair
segment means of a group minus the mean over every clustered segment of
the cube, compared by Pearson r after the best of the six contact
automorphisms.

If two structures were one deformation diluted by weak segments, their
cores would meet ``COPY_R``. If they stay below it, the deformations
differ. ``r_best_subparts`` is the best r over all four core / weak
combinations, a conservative check against a split that happened to cut
the wrong way. Each weak part is also matched to its nearest other
cluster, to show which state the weak segments resemble.

``--belt-crosscheck`` adds, after the fact only, how the core / weak parts
line up with the majority-belt labels of section 5 (d1 ≥ 7.0 Å). It does
not change any r or verdict.

Writes, under ``--out-dir``:

  tangled_inspect.csv                      one row per tangled pair, with a verdict
  {CUBE}_kXX_merge_dilution_parts.csv      core / weak counts per cluster
  {CUBE}_kXX_merge_dilution_pairs.csv      whole / core / weak r for those pairs
  {CUBE}_kXX_core_vs_belt.csv              with --belt-crosscheck only

Pass ``--clusters`` to compare an explicit list instead of auto-detecting
tangled pairs (the original BMHpH check).

Example
-------
python scripts/check_merge_dilution.py
python scripts/check_merge_dilution.py --cube BMHpH
python scripts/check_merge_dilution.py --cube BMHpH --k 6 --clusters 2 3 4 5 --belt-crosscheck
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations, product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from src.ChangepointAnalysis.assembly_qc import assembly_qc_failures, drop_trajectories, scan_assembly_qc_dir
from src.ChangepointAnalysis.cluster_k_diagnostics import load_segments_at_k, resolve_endpoint_features_dir
from src.ChangepointAnalysis.cluster_validation import (
    best_profile_r,
    core_split,
    dilution_verdict,
    mappings_from_perms,
    profile_correlation_table,
    tangled_pairs_from_correlations,
)
from src.ChangepointAnalysis.gsa_cohort_run import KNOWN_CUBES
from src.ChangepointAnalysis.metastable_catalog import (
    COPY_R,
    contact_automorphisms,
    family_ids,
    perm_cycles,
)
from src.ChangepointAnalysis.reporting import collect_endpoint_segment_means, endpoint_changepoints_dir

PARTS = {"c": "core", "w": "weak"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--qc-gsa-root", type=Path, default=ROOT / "output" / "gsa_features_by_cube")
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument("--cube", nargs="+", default=None)
    p.add_argument("--k", type=int, default=None, help="Ward cut; default is the selected k of each cube")
    p.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=None,
        help="Compare this list instead of auto-detecting tangled pairs",
    )
    p.add_argument("--copy-r", type=float, default=COPY_R)
    p.add_argument("--r-min", type=float, default=0.70, help="Cross-family pairs at or above this r are near-misses")
    p.add_argument("--min-segments", type=int, default=5, help="Minimum segments in a part to trust its r")
    p.add_argument(
        "--belt-crosscheck",
        action="store_true",
        help="Also tabulate core / weak against the majority-belt labels (post hoc; needs motif frames)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_validation_physical_k" / "merge_dilution",
    )
    return p.parse_args()


def segment_belt_modes(output_root: Path, qc_root: Path, cube: str, segs: pd.DataFrame) -> pd.Series:
    """Majority belt of each segment (index = row of ``segs`` after reset_index)."""
    import categorize_cluster_mixtures as ccm

    frames = ccm.map_frames_to_segment_clusters(ccm.load_apo_frames(output_root, qc_root, cube), segs, cube=cube)
    frames = frames.dropna(subset=["cluster_label", "n_open_cation_pi"]).copy()
    frames["cluster_label"] = frames["cluster_label"].astype(int)
    frames["frame"] = pd.to_numeric(frames["frame"], errors="coerce")
    frames["belt"] = ccm._frame_belts(frames, ccm.unit_series_columns(frames, "equator_d1_e"))
    frames["opening"] = frames["n_open_cation_pi"].map(lambda n: ccm.opening_label(int(n)))
    modes = ccm._segment_modes(frames, segs, cube)
    return pd.Series(dict(zip(modes["segment_key"], modes["belt_mode"])))


def _pair_r(delta_a: np.ndarray, delta_b: np.ndarray, mappings, perms) -> tuple[float, float, str]:
    r, idx = best_profile_r(delta_a, delta_b, mappings)
    r_id = float(np.corrcoef(delta_a, delta_b)[0, 1])
    return r_id, r, perm_cycles(perms[idx])


def _belt_crosscheck(
    cube: str,
    k: int,
    segs: pd.DataFrame,
    keys: pd.DataFrame,
    y: np.ndarray,
    part: np.ndarray,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Post-hoc table of core / weak against majority belt. Not used by any verdict."""
    segs = segs.copy()
    segs["belt_mode"] = segment_belt_modes(args.output_root, args.qc_gsa_root, cube, segs)
    on = ["traj_id", "segment_id"]
    joined = keys[on].astype(str).merge(segs[on + ["belt_mode"]].astype({c: str for c in on}), on=on, how="left")
    belt = joined["belt_mode"].map(
        lambda m: "no apo frames" if pd.isna(m) else ("intact belt" if m == "intact belt" else "broken belt")
    )
    tab = pd.DataFrame({"cluster_label": np.asarray(y, dtype=int), "part": part, "belt": belt.to_numpy()})
    tab = tab[tab["part"] != ""]
    out = tab.groupby(["cluster_label", "part", "belt"]).size().rename("n_segments").reset_index()
    out.insert(0, "k", k)
    out.insert(0, "cohort", cube)
    return out


def inspect_cube(
    cube: str,
    k: int,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cp_dir = endpoint_changepoints_dir(args.output_root, cube, own_elbow=True)
    segs = load_segments_at_k(cp_dir, k, group="endpoint")
    gsa_dir = args.qc_gsa_root / cube
    if gsa_dir.is_dir():
        failed = assembly_qc_failures(scan_assembly_qc_dir(gsa_dir))
        if failed:
            segs = drop_trajectories(segs, failed)
    segs = segs.reset_index(drop=True)
    X, y, keys, specs, kind, _ = collect_endpoint_segment_means(
        cp_dir, resolve_endpoint_features_dir(cp_dir), group="endpoint", clustered_df=segs, include_frame_corr=False
    )
    if kind != "site_pair" or X.size == 0:
        raise FileNotFoundError(f"{cube}: no site-pair segment means")

    perms = contact_automorphisms()
    mappings = mappings_from_perms(specs, perms)
    symmetry = profile_correlation_table(X, y, mappings, perms)
    corr_pairs = [
        (int(row.cluster_a), int(row.cluster_b), float(row.r_best_relabelling))
        for row in symmetry.itertuples(index=False)
    ]
    labels = sorted({int(c) for c in y})
    families = family_ids(labels, corr_pairs, threshold=args.copy_r)
    r_of = {(min(a, b), max(a, b)): r for a, b, r in corr_pairs}
    listed = [int(c) for c in args.clusters] if args.clusters else None
    if listed:
        inspect = pd.DataFrame(
            [
                {
                    "cluster_a": a,
                    "cluster_b": b,
                    "family_a": int(families[a]),
                    "family_b": int(families[b]),
                    "r_whole": r_of[(min(a, b), max(a, b))],
                    "tangle_reason": "listed",
                }
                for a, b in combinations(listed, 2)
            ]
        )
        involved = set(listed)
    else:
        inspect = tangled_pairs_from_correlations(
            labels, corr_pairs, threshold=args.copy_r, r_min=args.r_min
        )
        involved = (
            set(inspect["cluster_a"].astype(int)).union(set(inspect["cluster_b"].astype(int)))
            if not inspect.empty
            else set()
        )

    split_clusters = sorted(involved or set(labels))
    part = core_split(X, y, split_clusters)
    keys = keys.reset_index(drop=True)
    on = ["traj_id", "segment_id"]
    n_frames = keys[on].astype(str).merge(
        segs[on + ["n_frames"]].astype({c: str for c in on}), on=on, how="left"
    )["n_frames"]
    n_frames = pd.to_numeric(n_frames, errors="coerce").fillna(0).to_numpy()

    base = X.mean(axis=0)
    whole = {str(c): X[y == c].mean(axis=0) - base for c in labels}
    groups: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    part_rows = []
    for c in split_clusters:
        total = max(n_frames[y == c].sum(), 1)
        for tag, name in PARTS.items():
            mask = (y == c) & (part == name)
            key = f"{c}{tag}"
            counts[key] = int(mask.sum())
            if mask.any():
                groups[key] = X[mask].mean(axis=0) - base
            part_rows.append(
                {
                    "cohort": cube,
                    "k": k,
                    "cluster_label": int(c),
                    "part": name,
                    "n_segments": int(mask.sum()),
                    "n_frames": int(n_frames[mask].sum()),
                    "pct_of_cluster_frames": 100.0 * n_frames[mask].sum() / total,
                }
            )
    parts = pd.DataFrame(part_rows)

    pair_rows = []
    verdict_rows = []
    for rec in inspect.to_dict(orient="records") if not inspect.empty else []:
        a = int(rec["cluster_a"])
        b = int(rec["cluster_b"])
        r_whole = float(rec["r_whole"])
        detail = {
            "cohort": cube,
            "k": k,
            "cluster_a": a,
            "cluster_b": b,
            "tangle_reason": rec["tangle_reason"],
            "family_a": int(rec["family_a"]),
            "family_b": int(rec["family_b"]),
            "r_whole": r_whole,
        }
        for compared, table, key_a, key_b, field in (
            ("whole clusters", whole, str(a), str(b), None),
            ("cores", groups, f"{a}c", f"{b}c", "r_core"),
            ("weak parts", groups, f"{a}w", f"{b}w", "r_weak"),
        ):
            if key_a not in table or key_b not in table:
                if field:
                    detail[field] = float("nan")
                continue
            r_id, r, cycles = _pair_r(table[key_a], table[key_b], mappings, perms)
            pair_rows.append(
                {
                    "cohort": cube,
                    "k": k,
                    "compared": compared,
                    "a": key_a,
                    "b": key_b,
                    "r_as_labelled": r_id,
                    "r_best_relabelling": r,
                    "best_relabelling": cycles,
                    "tangle_reason": rec["tangle_reason"],
                }
            )
            if field:
                detail[field] = r

        # Best over all four core / weak combinations, each part with enough segments.
        sub = [
            (best_profile_r(groups[f"{a}{ta}"], groups[f"{b}{tb}"], mappings)[0], f"{a}{ta}-{b}{tb}")
            for ta, tb in product(PARTS, repeat=2)
            if counts.get(f"{a}{ta}", 0) >= args.min_segments and counts.get(f"{b}{tb}", 0) >= args.min_segments
        ]
        detail["r_best_subparts"], detail["best_subparts"] = max(sub) if sub else (float("nan"), "")

        # Which other cluster does each weak part look like?
        for lab, side in ((a, "a"), (b, "b")):
            key = f"{lab}w"
            if key not in groups:
                detail[f"weak_{side}_nearest"], detail[f"r_weak_{side}_nearest"] = "", float("nan")
                continue
            r_near, other = max(
                (best_profile_r(groups[key], whole[str(o)], mappings)[0], o) for o in labels if o != lab
            )
            detail[f"weak_{side}_nearest"], detail[f"r_weak_{side}_nearest"] = int(other), r_near

        n_ca = counts.get(f"{a}c", 0)
        n_cb = counts.get(f"{b}c", 0)
        detail.update(
            {
                "n_core_a": n_ca,
                "n_core_b": n_cb,
                "n_weak_a": counts.get(f"{a}w", 0),
                "n_weak_b": counts.get(f"{b}w", 0),
                "verdict": dilution_verdict(
                    r_whole,
                    detail.get("r_core"),
                    copy_r=args.copy_r,
                    n_core_a=n_ca,
                    n_core_b=n_cb,
                    min_segments=args.min_segments,
                    tangle_reason=rec["tangle_reason"],
                ),
            }
        )
        verdict_rows.append(detail)

    belt = _belt_crosscheck(cube, k, segs, keys, y, part, args) if args.belt_crosscheck else pd.DataFrame()
    return parts, pd.DataFrame(pair_rows), pd.DataFrame(verdict_rows), belt


def _cubes(args: argparse.Namespace) -> list[tuple[str, int]]:
    selected = pd.read_csv(args.k_selected)
    wanted = set(args.cube) if args.cube else set(KNOWN_CUBES)
    out = []
    for rec in selected.itertuples(index=False):
        cube = str(rec.cohort)
        if cube not in wanted:
            continue
        k = int(args.k) if args.k is not None else int(rec.selected_k)
        out.append((cube, k))
    if args.cube and not out:
        if args.k is None:
            raise SystemExit("--k is required when the cube is not in --k-selected")
        out = [(str(c), int(args.k)) for c in args.cube]
    return out


def main() -> int:
    args = parse_args()
    if args.clusters and (not args.cube or len(args.cube) != 1):
        print("--clusters requires a single --cube", flush=True)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_parts = []
    all_pairs = []
    all_verdicts = []
    for cube, k in _cubes(args):
        print(f"{cube} k={k}: loading segments", flush=True)
        parts, pairs, verdicts, belt = inspect_cube(cube, k, args)
        stem = f"{cube}_k{k:02d}"
        parts.to_csv(args.out_dir / f"{stem}_merge_dilution_parts.csv", index=False)
        all_parts.append(parts)
        if not pairs.empty:
            pairs.to_csv(args.out_dir / f"{stem}_merge_dilution_pairs.csv", index=False)
            all_pairs.append(pairs)
        if not belt.empty:
            belt.to_csv(args.out_dir / f"{stem}_core_vs_belt.csv", index=False)
        if not verdicts.empty:
            all_verdicts.append(verdicts)
        n_t = 0 if verdicts.empty else len(verdicts)
        print(f"{cube} k={k}: {n_t} tangled pairs", flush=True)
        if not verdicts.empty:
            show = [
                "cluster_a",
                "cluster_b",
                "tangle_reason",
                "r_whole",
                "r_core",
                "r_best_subparts",
                "n_core_a",
                "n_core_b",
                "verdict",
            ]
            print(verdicts[show].round(3).to_string(index=False), flush=True)

    if all_verdicts:
        stacked = pd.concat(all_verdicts, ignore_index=True)
        path = args.out_dir / "tangled_inspect.csv"
        stacked.to_csv(path, index=False)
        print(f"wrote {path}", flush=True)
    else:
        print("no tangled pairs", flush=True)
    if all_parts:
        pd.concat(all_parts, ignore_index=True).to_csv(args.out_dir / "merge_dilution_parts.csv", index=False)
    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_csv(args.out_dir / "merge_dilution_pairs.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
