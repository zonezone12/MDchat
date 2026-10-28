#!/usr/bin/env python3
"""Rank each cube's endpoint clusters by apo-frame occupancy.

Reads the motif profiles written by ``describe_endpoint_clusters.py`` at the
label-free k, names each cluster, and merges clusters that are one deformation
in different orientations. The merge correlations are computed here over
*every* cluster of the cube, from the 960 site-pair segment means, so they
do not depend on which clusters ``describe_endpoint_clusters.py --clusters``
contrasted. Two clusters join when the best-relabelled profile r is at least
``COPY_R`` (0.95), single linkage.

Each cube is ranked on its own. Nothing is pooled across cube types.

Writes, under ``--out-dir``:

  clusters_ranked.csv       every cluster, with ``rank_in_cube``
  families_ranked.csv       orientation copies collapsed, ranked inside the cube
  by_cube/{CUBE}.csv        that cube's structures only

Example
-------
python scripts/rank_metastable_structures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.ChangepointAnalysis.assembly_qc import (
    assembly_qc_failures,
    drop_trajectories,
    scan_assembly_qc_dir,
)
from src.ChangepointAnalysis.cluster_k_diagnostics import (
    load_segments_at_k,
    resolve_endpoint_features_dir,
)
from src.ChangepointAnalysis.cluster_validation import mappings_from_perms, profile_correlation_table
from src.ChangepointAnalysis.metastable_catalog import (
    COPY_R,
    contact_automorphisms,
    family_ids,
    structure_name,
)
from src.ChangepointAnalysis.reporting import collect_endpoint_segment_means, endpoint_changepoints_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank metastable endpoint structures at the label-free k.")
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument(
        "--k-selected",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv",
    )
    p.add_argument(
        "--profiles",
        type=Path,
        default=ROOT / "output" / "endpoint_cluster_profiles",
    )
    p.add_argument(
        "--replica-audit",
        type=Path,
        default=ROOT / "output" / "cluster_replica_audit_physical_k" / "single_replica_clusters.csv",
    )
    p.add_argument("--copy-r", type=float, default=COPY_R)
    p.add_argument(
        "--own-elbow",
        action="store_true",
        default=True,
        help="Read site-pair means from the own-elbow detections (default)",
    )
    p.add_argument(
        "--qc-gsa-root",
        type=Path,
        default=ROOT / "output" / "gsa_features_by_cube",
    )
    p.add_argument("--no-assembly-qc", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output" / "metastable_structures_physical_k",
    )
    return p.parse_args()


def profile_dirs(profiles: Path, cube: str, k: int) -> list[Path]:
    stem = f"{cube}_k{k:02d}"
    if not profiles.is_dir():
        return []
    return sorted(
        path
        for path in profiles.iterdir()
        if path.is_dir() and (path.name == stem or path.name.startswith(stem + "_"))
    )


def _read_motif(dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for path in dirs:
        csv = path / "cluster_motif_profile.csv"
        if csv.is_file():
            frames.append(pd.read_csv(csv))
    if not frames:
        raise FileNotFoundError("no cluster_motif_profile.csv")
    frames.sort(key=lambda frame: (-len(frame), len(frame.columns)))
    return frames[0]


def all_cluster_symmetry(
    output_root: Path,
    cube: str,
    k: int,
    *,
    own_elbow: bool,
    qc_gsa_root: Path,
    no_assembly_qc: bool,
) -> pd.DataFrame:
    """Best-relabelled profile r for every pair of clusters of this cube."""
    cp_dir = endpoint_changepoints_dir(output_root, cube, own_elbow=own_elbow)
    segs = load_segments_at_k(cp_dir, k, group="endpoint")
    if segs.empty:
        raise FileNotFoundError(f"{cube}: no endpoint clustering at k={k} in {cp_dir}")
    if not no_assembly_qc:
        gsa_dir = qc_gsa_root / cube
        if gsa_dir.is_dir():
            failed = assembly_qc_failures(scan_assembly_qc_dir(gsa_dir))
            if failed:
                segs = drop_trajectories(segs, failed)
    X, y, _keys, specs, kind, _ = collect_endpoint_segment_means(
        cp_dir,
        resolve_endpoint_features_dir(cp_dir),
        group="endpoint",
        clustered_df=segs,
        include_frame_corr=False,
    )
    if kind != "site_pair" or X.size == 0:
        raise FileNotFoundError(f"{cube}: no site-pair segment means")
    perms = contact_automorphisms()
    mappings = mappings_from_perms(specs, perms)
    return profile_correlation_table(X, y, mappings, perms)


def _replica_keys(path: Path) -> set[tuple[str, int, int]]:
    if not path.is_file():
        return set()
    audit = pd.read_csv(path)
    return {
        (str(row.cohort), int(row.k), int(row.cluster_label))
        for row in audit.itertuples(index=False)
    }


def _is_belt_break(name: str) -> bool:
    return name.startswith("belt broken") or name.startswith("partly belt broken")


def _family_name(members: pd.DataFrame) -> str:
    names = [str(name) for name in members["structure"]]
    if len(members) == 1:
        return names[0]
    if all(_is_belt_break(name) for name in names):
        return f"one belt break, {len(members)} orientations"
    elongated = float(
        (members["pct_any_d1_elongated"] * members["n_apo_frames"]).sum() / members["n_apo_frames"].sum()
    )
    if elongated >= 20.0:
        return f"opened cage, {len(members)} orientations"
    return " + ".join(dict.fromkeys(names))


def rank_cube(
    cube: str,
    k: int,
    motif: pd.DataFrame,
    symmetry: pd.DataFrame,
    replicas: set[tuple[str, int, int]],
    copy_r: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = motif.copy()
    work["cluster_label"] = work["cluster_label"].astype(int)
    total = float(work["n_apo_frames"].sum())
    work["pct_of_cube"] = 100.0 * work["n_apo_frames"] / total
    labels = work["cluster_label"].tolist()
    pairs = [
        (int(row.cluster_a), int(row.cluster_b), float(row.r_best_relabelling))
        for row in symmetry.itertuples(index=False)
    ]
    families = family_ids(labels, pairs, threshold=copy_r)
    r_of = {}
    for a, b, score in pairs:
        r_of[(a, b)] = score
        r_of[(b, a)] = score

    rows = []
    for record in work.to_dict(orient="records"):
        label = int(record["cluster_label"])
        replica = (cube, k, label) in replicas or (
            int(record["n_traj"]) == 1 and float(record["top_traj_share_of_segments"]) >= 0.5
        )
        family = families[label]
        mates = [other for other, head in families.items() if head == family and other != label]
        mate_r = [r_of[(label, other)] for other in mates if (label, other) in r_of]
        record.update(
            {
                "cohort": cube,
                "k": k,
                "single_replica": replica,
                "structure": structure_name(record, single_replica=replica),
                "family_id": family,
                "n_orientations": 1 + len(mates),
                "min_r_in_family": min(mate_r) if mate_r else float("nan"),
            }
        )
        rows.append(record)
    clusters = pd.DataFrame(rows).sort_values("pct_of_cube", ascending=False)
    clusters.insert(0, "rank_in_cube", range(1, len(clusters) + 1))

    family_rows = []
    for family, members in clusters.groupby("family_id", sort=False):
        members = members.sort_values("pct_of_cube", ascending=False)
        family_rows.append(
            {
                "cohort": cube,
                "k": int(k),
                "family_id": int(family),
                "member_clusters": " ".join(str(int(c)) for c in members["cluster_label"]),
                "n_orientations": int(len(members)),
                "n_apo_frames": int(members["n_apo_frames"].sum()),
                "pct_of_cube": float(members["pct_of_cube"].sum()),
                "n_segments": int(members["n_segments"].sum()),
                "n_traj_max": int(members["n_traj"].max()),
                "structure": _family_name(members),
                "min_r_in_family": float(members["min_r_in_family"].min(skipna=True)),
                "single_replica": bool(members["single_replica"].all() and len(members) == 1),
                "pct_A": float(
                    (members["pct_A"] * members["n_apo_frames"]).sum() / members["n_apo_frames"].sum()
                ),
                "pct_C2": float(
                    (members["pct_C2"] * members["n_apo_frames"]).sum() / members["n_apo_frames"].sum()
                ),
                "pct_other": float(
                    (members["pct_other"] * members["n_apo_frames"]).sum() / members["n_apo_frames"].sum()
                ),
                "pct_any_d1_elongated": float(
                    (members["pct_any_d1_elongated"] * members["n_apo_frames"]).sum()
                    / members["n_apo_frames"].sum()
                ),
            }
        )
    families_df = pd.DataFrame(family_rows).sort_values("pct_of_cube", ascending=False)
    families_df.insert(0, "rank_in_cube", range(1, len(families_df) + 1))
    return clusters, families_df


def main() -> int:
    args = parse_args()
    selected = pd.read_csv(args.k_selected)
    replicas = _replica_keys(args.replica_audit)
    cluster_parts = []
    family_parts = []
    symmetry_parts = []
    for row in selected.itertuples(index=False):
        cube = str(row.cohort)
        k = int(row.selected_k)
        dirs = profile_dirs(args.profiles, cube, k)
        if not dirs:
            print(f"{cube}: no profile folder at k={k}", flush=True)
            return 1
        symmetry = all_cluster_symmetry(
            args.output_root,
            cube,
            k,
            own_elbow=args.own_elbow,
            qc_gsa_root=args.qc_gsa_root,
            no_assembly_qc=args.no_assembly_qc,
        )
        symmetry_parts.append(symmetry.assign(cohort=cube, k=k))
        clusters, families = rank_cube(cube, k, _read_motif(dirs), symmetry, replicas, args.copy_r)
        cluster_parts.append(clusters)
        family_parts.append(families)
        print(
            f"{cube} k={k}: {len(clusters)} clusters, {len(families)} structures "
            f"after merging r>={args.copy_r:.2f}",
            flush=True,
        )

    clusters = pd.concat(cluster_parts, ignore_index=True)
    families = pd.concat(family_parts, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    by_cube = args.out_dir / "by_cube"
    by_cube.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(args.out_dir / "clusters_ranked.csv", index=False)
    families.to_csv(args.out_dir / "families_ranked.csv", index=False)
    if symmetry_parts:
        pd.concat(symmetry_parts, ignore_index=True).to_csv(
            args.out_dir / "all_cluster_symmetry.csv", index=False
        )
    pooled = args.out_dir / "top10_structures.csv"
    if pooled.is_file():
        pooled.unlink()
    show = ["rank_in_cube", "pct_of_cube", "n_orientations", "member_clusters", "structure"]
    for cube, part in families.groupby("cohort", sort=False):
        part[show].round(1).to_csv(by_cube / f"{cube}.csv", index=False)
        print(f"\n{cube}", flush=True)
        print(part[show].round(1).to_string(index=False), flush=True)
    print(f"\nWrote {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
