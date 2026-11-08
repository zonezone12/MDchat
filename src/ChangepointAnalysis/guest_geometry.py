"""Cage geometry with iodide in the cavity versus the same replica empty.

Two questions, kept separate:

* Association. Within one replica, the mean of a geometry column on frames with
  ``n_guest_inside_cavity >= 1`` minus the mean on the other frames. A late
  entry can make this a time trend rather than a guest effect.
* At entry. For each sustained visit (guest inside for at least 20 ps), the
  mean over the first 100 ps inside minus the mean over the 100 ps just before
  entry. That contrast is local in time.

A third column residualizes a linear time trend inside the replica
(``geometry ~ time + occupied``) and keeps the occupancy coefficient.

The dissociated BMHpH ``563849`` run is dropped. Tests are Wilcoxon signed-rank
on one number per replica, so a replica with several entries contributes once.

Example
-------
python scripts/compare_guest_cage_geometry.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .gsa_cohort_run import KNOWN_CUBES
from .reporting import OWN_ELBOW_ENDPOINT_DIRS, endpoint_changepoints_dir

GEOM_COLS = [
    "assembly_rg",
    "octahedrality_score",
    "assembly_relative_shape_anisotropy",
    "total_inter_monomer_contacts",
    "n_broken_interfaces",
    "n_active_interfaces",
    "endpoint_dist_mean",
    "cavity_inner_atom_distance_mean",
    "pore_opening_distance_1",
    "pore_opening_distance_2",
    "pore_opening_distance_3",
]
MOTIF_BASE = ["n_open_cation_pi"]
D1_PREFIX = "equator_d1_"
MIN_FRAMES = 20
LOCAL_WINDOW = 100
RG_MAX = 15.0
# Compact / intact cluster at the selected k (docs/metastable_structures.md §7).
INTACT_CLUSTER = {
    "BHHpH": 2,
    "BHHpM": 2,
    "BMHpH": 1,
    "BMHpM": 0,
    "BMMpH": 0,
    "BMMpM": 0,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output-root", type=Path, default=ROOT / "output")
    p.add_argument("--gsa-features-root", type=Path, default=ROOT / "output" / "gsa_features_by_cube")
    p.add_argument("--dwells", type=Path, default=ROOT / "output" / "guest_event_cotiming" / "guest_dwells.csv")
    p.add_argument("--k-selected", type=Path, default=ROOT / "output" / "endpoint_cluster_physical_k_own_elbow" / "physical_k_selected.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "guest_cage_geometry")
    p.add_argument("--cubes", nargs="*", default=list(KNOWN_CUBES))
    p.add_argument("--min-dwell", type=int, default=20)
    return p.parse_args()


def _strip(traj_id: str, cube: str) -> str:
    prefix = f"{cube}_"
    return traj_id[len(prefix):] if traj_id.startswith(prefix) else traj_id


def _wilcoxon(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5 or np.allclose(x, 0):
        return float("nan")
    return float(wilcoxon(x, zero_method="wilcox", alternative="two-sided", method="auto").pvalue)


def _summarize(df: pd.DataFrame, value: str) -> pd.DataFrame:
    rows = []
    for (cube, feature), g in df.groupby(["cube", "feature"], sort=False):
        v = g[value].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        rows.append({
            "cube": cube,
            "feature": feature,
            "n_replica": int(v.size),
            "median_diff": float(np.median(v)) if v.size else np.nan,
            "q25": float(np.percentile(v, 25)) if v.size else np.nan,
            "q75": float(np.percentile(v, 75)) if v.size else np.nan,
            "frac_positive": float(np.mean(v > 0)) if v.size else np.nan,
            "p_wilcoxon": _wilcoxon(v),
        })
    return pd.DataFrame(rows)


def _means(y: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    sel = y[mask]
    sel = sel[np.isfinite(sel)]
    if sel.size < MIN_FRAMES:
        return np.nan, np.nan, int(sel.size)
    return float(sel.mean()), float(sel.std(ddof=0)), int(sel.size)


def _lstsq_occupancy(y: np.ndarray, t: np.ndarray, occ: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(t)
    if int(occ[ok].sum()) < MIN_FRAMES or int((~occ[ok]).sum()) < MIN_FRAMES:
        return np.nan
    tt = t[ok]
    tt = (tt - tt.mean()) / 1000.0
    x = np.column_stack([np.ones(ok.sum()), tt, occ[ok].astype(float)])
    coef, *_ = np.linalg.lstsq(x, y[ok], rcond=None)
    return float(coef[2])


def _sustained_mask(frame: np.ndarray, intervals: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(frame.shape, dtype=bool)
    for start, end in intervals:
        mask |= (frame >= start) & (frame < end)
    return mask


def _feature_matrix(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {c: df[c].to_numpy(dtype=float) for c in GEOM_COLS if c in df.columns}
    pores = [out[c] for c in ("pore_opening_distance_1", "pore_opening_distance_2", "pore_opening_distance_3") if c in out]
    if len(pores) == 3:
        out["pore_opening_mean"] = np.nanmean(np.column_stack(pores), axis=1)
    return out


def _d1_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith(D1_PREFIX)]


def compare_guest_cage_geometry(
    *,
    output_root: Path | str = "output",
    gsa_features_root: Path | str = "output/gsa_features_by_cube",
    dwells: Path | str = "output/guest_event_cotiming/guest_dwells.csv",
    k_selected: Path | str = "output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv",
    out_dir: Path | str = "output/guest_cage_geometry",
    cubes: Optional[Sequence[str]] = None,
    min_dwell: int = 20,
) -> dict[str, Path]:
    class _A:
        pass
    args = _A()
    args.output_root = Path(output_root)
    args.gsa_features_root = Path(gsa_features_root)
    args.dwells = Path(dwells)
    args.k_selected = Path(k_selected)
    args.out_dir = Path(out_dir)
    args.cubes = list(cubes) if cubes is not None else list(KNOWN_CUBES)
    args.min_dwell = int(min_dwell)
    return _run_geometry(args)


def _run_geometry(args) -> dict[str, Path]:
    dwells = pd.read_csv(args.dwells)
    dwells = dwells[(dwells["min_dwell"] == args.min_dwell) & (dwells["state"]) & (dwells["event"] == "entry")]
    k_table = pd.read_csv(args.k_selected).set_index("cohort")["selected_k"].astype(int).to_dict()

    assoc_rows: list[dict] = []
    local_rows: list[dict] = []
    time_rows: list[dict] = []
    occupancy_rows: list[dict] = []
    cluster_rows: list[dict] = []
    cluster_within: list[pd.DataFrame] = []

    for cube in args.cubes:
        feat_dir = args.gsa_features_root / cube
        paths = sorted(feat_dir.glob("*_gsa_features.csv"))
        cube_dwells = dwells[dwells["cube"] == cube]
        print(f"geometry {cube}: {len(paths)} trajectories", flush=True)
        n_frames = n_occ = 0
        n_inside_at_0 = 0
        for path in paths:
            usecols = ["traj_id", "frame", "n_guest_inside_cavity", "largest_connected_component_size", *GEOM_COLS]
            header = pd.read_csv(path, nrows=0).columns
            df = pd.read_csv(path, usecols=[c for c in usecols if c in header])
            tid = _strip(str(df["traj_id"].iloc[0]), cube)
            frame = df["frame"].to_numpy()
            guest = df["n_guest_inside_cavity"].to_numpy(dtype=float)
            raw = guest >= 1
            n_frames += int(frame.size)
            n_occ += int(raw.sum())
            at0 = guest[frame == 0]
            n_inside_at_0 += int(np.nansum(at0 >= 1)) if at0.size else 0
            rg = df["assembly_rg"].to_numpy(dtype=float)
            lcc = df["largest_connected_component_size"].to_numpy(dtype=float)
            if np.nanmax(rg) > RG_MAX or np.nanmin(lcc) < 6:
                continue
            intervals = [
                (int(r.start), int(r.start) + int(r.length))
                for r in cube_dwells[cube_dwells["traj_id"] == tid].itertuples(index=False)
            ]
            sustained = _sustained_mask(frame, intervals)
            feats = _feature_matrix(df)
            for name, y in feats.items():
                for kind, mask in (("raw", raw), ("sustained", sustained)):
                    occ_mean, _, n_in = _means(y, mask)
                    apo_mean, apo_std, n_out = _means(y, ~mask)
                    if not np.isfinite(occ_mean) or not np.isfinite(apo_mean):
                        continue
                    assoc_rows.append({
                        "cube": cube,
                        "traj_id": tid,
                        "feature": name,
                        "occupancy": kind,
                        "diff": occ_mean - apo_mean,
                        "apo_mean": apo_mean,
                        "apo_std": apo_std,
                        "n_occupied": n_in,
                        "n_apo": n_out,
                    })
                coef = _lstsq_occupancy(y, frame.astype(float), raw)
                if np.isfinite(coef):
                    time_rows.append({"cube": cube, "traj_id": tid, "feature": name, "occupancy_coef": coef})
            for start, end in intervals:
                pre = (frame >= start - LOCAL_WINDOW) & (frame < start) & ~sustained
                dur = (frame >= start) & (frame < min(end, start + LOCAL_WINDOW)) & sustained
                if int(pre.sum()) < MIN_FRAMES or int(dur.sum()) < MIN_FRAMES:
                    continue
                for name, y in feats.items():
                    pre_m = float(np.nanmean(y[pre]))
                    dur_m = float(np.nanmean(y[dur]))
                    if np.isfinite(pre_m) and np.isfinite(dur_m):
                        local_rows.append({
                            "cube": cube,
                            "traj_id": tid,
                            "entry_frame": start,
                            "feature": name,
                            "diff": dur_m - pre_m,
                        })

        occupancy_rows.append({
            "cube": cube,
            "n_frames": n_frames,
            "n_occupied_raw": n_occ,
            "frac_occupied_raw": n_occ / n_frames if n_frames else np.nan,
            "n_traj_guest_inside_at_frame_0": n_inside_at_0,
        })

        # Motif coordinates on the same two contrasts. Own n_guest column.
        motif_dir = args.output_root / f"{cube}_motif_rmsd"
        motif_paths = sorted(motif_dir.rglob("*_motif_rmsd.csv")) if motif_dir.is_dir() else []
        seen: set[str] = set()
        for path in motif_paths:
            header = list(pd.read_csv(path, nrows=0).columns)
            d1_cols = _d1_columns(header)
            if "n_guest_inside_cavity" not in header or "n_open_cation_pi" not in header or not d1_cols:
                continue
            tid = _strip(path.name[: -len("_motif_rmsd.csv")], cube)
            if tid in seen:
                continue
            seen.add(tid)
            m = pd.read_csv(path, usecols=["frame", "n_guest_inside_cavity", "n_open_cation_pi", *d1_cols])
            frame = m["frame"].to_numpy()
            raw = m["n_guest_inside_cavity"].to_numpy(dtype=float) >= 1
            intervals = [
                (int(r.start), int(r.start) + int(r.length))
                for r in cube_dwells[cube_dwells["traj_id"] == tid].itertuples(index=False)
            ]
            sustained = _sustained_mask(frame, intervals)
            d1 = m[d1_cols].to_numpy(dtype=float)
            feats = {
                "n_open_cation_pi": m["n_open_cation_pi"].to_numpy(dtype=float),
                "equator_d1_mean": np.nanmean(d1, axis=1),
                "frac_d1_elongated": np.nanmean(d1 >= 7.0, axis=1),
            }
            for name, y in feats.items():
                for kind, mask in (("raw", raw), ("sustained", sustained)):
                    occ_mean, _, n_in = _means(y, mask)
                    apo_mean, apo_std, n_out = _means(y, ~mask)
                    if not np.isfinite(occ_mean) or not np.isfinite(apo_mean):
                        continue
                    assoc_rows.append({
                        "cube": cube,
                        "traj_id": tid,
                        "feature": name,
                        "occupancy": kind,
                        "diff": occ_mean - apo_mean,
                        "apo_mean": apo_mean,
                        "apo_std": apo_std,
                        "n_occupied": n_in,
                        "n_apo": n_out,
                    })
                coef = _lstsq_occupancy(y, frame.astype(float), raw)
                if np.isfinite(coef):
                    time_rows.append({"cube": cube, "traj_id": tid, "feature": name, "occupancy_coef": coef})
            for start, end in intervals:
                pre = (frame >= start - LOCAL_WINDOW) & (frame < start) & ~sustained
                dur = (frame >= start) & (frame < min(end, start + LOCAL_WINDOW)) & sustained
                if int(pre.sum()) < MIN_FRAMES or int(dur.sum()) < MIN_FRAMES:
                    continue
                for name, y in feats.items():
                    pre_m = float(np.nanmean(y[pre]))
                    dur_m = float(np.nanmean(y[dur]))
                    if np.isfinite(pre_m) and np.isfinite(dur_m):
                        local_rows.append({
                            "cube": cube,
                            "traj_id": tid,
                            "entry_frame": start,
                            "feature": name,
                            "diff": dur_m - pre_m,
                        })

        # Cluster frame shares at the selected k, raw occupancy, QC already applied per file above
        # is not replayed here: re-read guest and assign frames to segments.
        k = int(k_table[cube])
        seg_path = (
            endpoint_changepoints_dir(args.output_root, cube, own_elbow=True)
            / "clusters" / "endpoint" / "by_k" / f"k_{k:02d}" / "segments_clustered.csv"
        )
        if not seg_path.is_file():
            continue
        segs = pd.read_csv(seg_path, usecols=["traj_id", "start_frame", "end_frame", "cluster_label"])
        intact = INTACT_CLUSTER[cube]
        within_rows: list[dict] = []
        for path in paths:
            header = pd.read_csv(path, nrows=0).columns
            df = pd.read_csv(path, usecols=[c for c in ("traj_id", "frame", "n_guest_inside_cavity", "assembly_rg", "largest_connected_component_size") if c in header])
            tid = _strip(str(df["traj_id"].iloc[0]), cube)
            if float(df["assembly_rg"].max()) > RG_MAX or float(df["largest_connected_component_size"].min()) < 6:
                continue
            sub = segs[segs["traj_id"] == tid]
            if sub.empty:
                continue
            frame = df["frame"].to_numpy()
            raw = df["n_guest_inside_cavity"].to_numpy(dtype=float) >= 1
            label = np.full(frame.shape, -1, dtype=int)
            for rec in sub.itertuples(index=False):
                label[(frame >= rec.start_frame) & (frame < rec.end_frame)] = int(rec.cluster_label)
            labeled = label >= 0
            def _intact_share(mask: np.ndarray) -> float:
                if int(mask.sum()) < MIN_FRAMES:
                    return np.nan
                return float(np.mean(label[mask] == intact))

            mid = float(np.median(frame))
            within_rows.append({
                "cube": cube,
                "traj_id": tid,
                "intact_cluster": intact,
                "n_apo": int((labeled & ~raw).sum()),
                "n_occupied": int((labeled & raw).sum()),
                "apo_share": _intact_share(labeled & ~raw),
                "occupied_share": _intact_share(labeled & raw),
                "apo_early_share": _intact_share(labeled & ~raw & (frame < mid)),
                "apo_late_share": _intact_share(labeled & ~raw & (frame >= mid)),
            })
            for state, mask in (("apo", labeled & ~raw), ("occupied", labeled & raw)):
                if int(mask.sum()) == 0:
                    continue
                labs, counts = np.unique(label[mask], return_counts=True)
                total = int(counts.sum())
                for lab, cnt in zip(labs, counts):
                    cluster_rows.append({
                        "cube": cube,
                        "k": k,
                        "traj_id": tid,
                        "state": state,
                        "cluster": int(lab),
                        "n_frames": int(cnt),
                        "share": int(cnt) / total,
                        "intact_cluster": intact,
                    })
        if within_rows:
            wdf = pd.DataFrame(within_rows)
            wdf["occupied_minus_apo"] = wdf["occupied_share"] - wdf["apo_share"]
            wdf["apo_late_minus_early"] = wdf["apo_late_share"] - wdf["apo_early_share"]
            cluster_within.append(wdf)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    assoc = pd.DataFrame(assoc_rows)
    local = pd.DataFrame(local_rows)
    timed = pd.DataFrame(time_rows)
    occ = pd.DataFrame(occupancy_rows)
    clusters = pd.DataFrame(cluster_rows)

    # One local contrast per replica: mean over that replica's entries.
    local_rep = (
        local.groupby(["cube", "traj_id", "feature"], as_index=False)["diff"].mean()
        if len(local) else local
    )
    summaries = {
        "association_raw": _summarize(assoc[assoc["occupancy"] == "raw"], "diff") if len(assoc) else pd.DataFrame(),
        "association_sustained": _summarize(assoc[assoc["occupancy"] == "sustained"], "diff") if len(assoc) else pd.DataFrame(),
        "at_entry_100ps": _summarize(local_rep, "diff") if len(local_rep) else pd.DataFrame(),
        "time_residual": _summarize(timed, "occupancy_coef") if len(timed) else pd.DataFrame(),
    }
    # Apo fluctuation, so a difference can be read against the replica's own spread.
    if len(assoc):
        scale = (
            assoc[assoc["occupancy"] == "raw"]
            .groupby(["cube", "feature"], as_index=False)
            .agg(median_apo_std=("apo_std", "median"), median_apo=("apo_mean", "median"))
        )
        for key in ("association_raw", "association_sustained", "time_residual"):
            summaries[key] = summaries[key].merge(scale, on=["cube", "feature"], how="left")

    occ.to_csv(args.out_dir / "occupancy_raw.csv", index=False)
    assoc.to_csv(args.out_dir / "within_replica_diffs.csv", index=False)
    local.to_csv(args.out_dir / "entry_window_diffs.csv", index=False)
    timed.to_csv(args.out_dir / "time_residual_coefs.csv", index=False)
    for name, table in summaries.items():
        table.to_csv(args.out_dir / f"summary_{name}.csv", index=False)

    if len(clusters):
        totals = clusters.groupby(["cube", "k", "state", "cluster"], as_index=False)["n_frames"].sum()
        totals["share"] = totals["n_frames"] / totals.groupby(["cube", "state"])["n_frames"].transform("sum")
        wide = totals.pivot_table(index=["cube", "k", "cluster"], columns="state", values="share").reset_index()
        wide["occupied_minus_apo"] = wide.get("occupied", np.nan) - wide.get("apo", np.nan)
        wide.to_csv(args.out_dir / "cluster_share_by_occupancy.csv", index=False)
        intact_rows = []
        for cube, g in wide.groupby("cube"):
            row = g[g["cluster"] == INTACT_CLUSTER[cube]]
            if row.empty:
                continue
            r = row.iloc[0]
            apo = totals[(totals["cube"] == cube) & (totals["state"] == "apo")]
            occs = totals[(totals["cube"] == cube) & (totals["state"] == "occupied")]
            def _tvd(a: pd.DataFrame, b: pd.DataFrame) -> float:
                aa = a.set_index("cluster")["share"]
                bb = b.set_index("cluster")["share"]
                idx = sorted(set(aa.index) | set(bb.index))
                return 0.5 * float(np.abs([aa.get(i, 0) - bb.get(i, 0) for i in idx]).sum())
            intact_rows.append({
                "cube": cube,
                "intact_cluster": INTACT_CLUSTER[cube],
                "apo_share": float(r["apo"]) if "apo" in r and np.isfinite(r["apo"]) else np.nan,
                "occupied_share": float(r["occupied"]) if "occupied" in r and np.isfinite(r["occupied"]) else np.nan,
                "tvd": _tvd(apo, occs) if len(occs) else np.nan,
            })
        pd.DataFrame(intact_rows).to_csv(args.out_dir / "intact_cluster_by_occupancy.csv", index=False)

    if cluster_within:
        within = pd.concat(cluster_within, ignore_index=True)
        within.to_csv(args.out_dir / "intact_cluster_within_replica.csv", index=False)
        both = within[(within["n_apo"] >= MIN_FRAMES) & (within["n_occupied"] >= MIN_FRAMES)]
        wsum = _summarize(both.rename(columns={"occupied_minus_apo": "diff"}).assign(feature="intact_share"), "diff")
        late = both[np.isfinite(both["apo_late_minus_early"])]
        late_sum = _summarize(
            late.rename(columns={"apo_late_minus_early": "diff"}).assign(feature="apo_late_minus_early"),
            "diff",
        )
        pd.concat([wsum, late_sum], ignore_index=True).to_csv(
            args.out_dir / "summary_intact_within_replica.csv", index=False
        )

    focus = [
        "assembly_rg",
        "octahedrality_score",
        "total_inter_monomer_contacts",
        "n_broken_interfaces",
        "endpoint_dist_mean",
        "cavity_inner_atom_distance_mean",
        "pore_opening_mean",
        "n_open_cation_pi",
        "equator_d1_mean",
        "frac_d1_elongated",
    ]
    print("\noccupancy", flush=True)
    print(occ.round(3).to_string(index=False), flush=True)
    for name, table in summaries.items():
        show = table[table["feature"].isin(focus)].copy() if len(table) else table
        print(f"\n{name}", flush=True)
        if len(show):
            print(show.round(4).to_string(index=False), flush=True)
    if (args.out_dir / "summary_intact_within_replica.csv").is_file():
        print("\nintact cluster, within replica", flush=True)
        print(pd.read_csv(args.out_dir / "summary_intact_within_replica.csv").round(3).to_string(index=False), flush=True)
    written = {p.name: p for p in args.out_dir.glob("*.csv")}
    return written


def main() -> int:
    written = _run_geometry(parse_args())
    if not written:
        return 1
    print(f"wrote {next(iter(written.values())).parent}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
