"""Cross-cohort diagnostics for segment k (silhouette, switching, geometry).

Reads existing ``clusters/{group}/`` inspection artifacts. Does not re-detect
or re-cluster. For ``group="endpoint"`` also reads paper-d1 / site maps.

Used to answer: why every B* run is cut at k=5, whether silhouette prefers
another k, and how switching / geometry spread differ across cubes.
"""

from __future__ import annotations

import fnmatch
import re
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.utils.run_log import log_event

from .reporting import cohort_name_from_changepoints_dir

_CHOSEN_K_RE = re.compile(r"fixed k\s*=\s*(\d+)", re.IGNORECASE)

_GEOMETRY_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("assembly_rg", ("assembly_rg_mean", "assembly_rg")),
    (
        "assembly_rmsd_to_ref",
        ("assembly_rmsd_to_ref_mean", "assembly_rmsd_to_ref"),
    ),
    (
        "octahedrality_score",
        ("octahedrality_score_mean", "octahedrality_score"),
    ),
    ("endpoint_dist", ("endpoint_dist_mean_mean", "endpoint_dist_mean")),
    (
        "n_guest_inside_cavity",
        ("n_guest_inside_cavity_mean", "n_guest_inside_cavity"),
    ),
    (
        "guest_radial_distance_mean",
        ("guest_radial_distance_mean_mean", "guest_radial_distance_mean"),
    ),
    ("cavity_ion_count", ("cavity_ion_count_mean", "cavity_ion_count")),
    ("cavity_water_count", ("cavity_water_count_mean", "cavity_water_count")),
)


def discover_cluster_k_cohort_dirs(
    output_root: Path | str,
    pattern: str,
    *,
    group: str = "endpoint",
    exclude_test: bool = True,
) -> list[Path]:
    """Changepoint dirs that have clustered segments for *group*."""
    output_root = Path(output_root)
    if not output_root.is_dir():
        return []

    dirs: list[Path] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        if not fnmatch.fnmatch(child.name, pattern):
            continue
        if exclude_test and ("_test" in child.name or "_smoke" in child.name):
            continue
        clustered = child / "clusters" / group / "segments_clustered.csv"
        if clustered.exists():
            dirs.append(child)
    return dirs


def discover_endpoint_k_cohort_dirs(
    output_root: Path | str,
    pattern: str = "endpoint_changepoints_B*",
    *,
    exclude_test: bool = True,
) -> list[Path]:
    """Changepoint dirs that have clustered endpoint segments."""
    return discover_cluster_k_cohort_dirs(
        output_root,
        pattern,
        group="endpoint",
        exclude_test=exclude_test,
    )


def _cluster_dir(changepoints_dir: Path, group: str) -> Path:
    return Path(changepoints_dir) / "clusters" / group


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def chosen_k_from_cohort(
    changepoints_dir: Path | str,
    *,
    group: str = "endpoint",
) -> int:
    """k used for the published ``segments_clustered.csv`` labels."""
    cdir = _cluster_dir(Path(changepoints_dir), group)
    comp = _read_csv_if_exists(cdir / "by_k" / "k_comparison.csv")
    if not comp.empty and "is_chosen_k" in comp.columns:
        picked = comp[comp["is_chosen_k"].astype(str).str.lower().isin(("true", "1"))]
        if not picked.empty and "k" in picked.columns:
            return int(picked["k"].iloc[0])

    summary = cdir / "cluster_summary.txt"
    if summary.exists():
        text = summary.read_text(encoding="utf-8")
        m = _CHOSEN_K_RE.search(text)
        if m:
            return int(m.group(1))

    segs = _read_csv_if_exists(cdir / "segments_clustered.csv")
    if not segs.empty and "cluster_label" in segs.columns:
        return int(segs["cluster_label"].nunique())
    return 5


def _load_silhouette(changepoints_dir: Path, group: str) -> pd.DataFrame:
    cdir = _cluster_dir(changepoints_dir, group)
    sil = _read_csv_if_exists(cdir / "silhouette_by_k.csv")
    if sil.empty:
        sil = _read_csv_if_exists(cdir / "by_k" / "silhouette_by_k.csv")
    if sil.empty or "k" not in sil.columns or "silhouette" not in sil.columns:
        return pd.DataFrame()
    sil = sil.copy()
    sil["k"] = sil["k"].astype(int)
    sil["silhouette"] = pd.to_numeric(sil["silhouette"], errors="coerce")
    return sil.sort_values("k").reset_index(drop=True)


def _cluster_size_map(df: pd.DataFrame) -> dict[int, int]:
    if df.empty or "cluster_label" not in df.columns:
        return {}
    return {
        int(k): int(v)
        for k, v in df["cluster_label"].value_counts().sort_index().items()
    }


def _sizes_at_k(changepoints_dir: Path, k: int, group: str) -> dict[int, int]:
    cdir = _cluster_dir(changepoints_dir, group)
    chosen = chosen_k_from_cohort(changepoints_dir, group=group)
    if k == chosen:
        return _cluster_size_map(_read_csv_if_exists(cdir / "segments_clustered.csv"))
    by_k = cdir / "by_k" / f"k_{k:02d}" / "segments_clustered.csv"
    return _cluster_size_map(_read_csv_if_exists(by_k))


def _describe_ward_split(
    sizes_lo: dict[int, int],
    sizes_hi: dict[int, int],
) -> str:
    if not sizes_lo or not sizes_hi:
        return ""
    lo = Counter(sizes_lo.values())
    hi = Counter(sizes_hi.values())
    lost: list[int] = []
    gained: list[int] = []
    for size, n in lo.items():
        diff = n - hi.get(size, 0)
        if diff > 0:
            lost.extend([int(size)] * diff)
    for size, n in hi.items():
        diff = n - lo.get(size, 0)
        if diff > 0:
            gained.extend([int(size)] * diff)
    if len(lost) == 1 and len(gained) == 2 and sum(gained) == lost[0]:
        a, b = sorted(gained, reverse=True)
        return f"splits a {lost[0]}-segment cluster → {a}+{b}"
    if lost or gained:
        return f"size multiset {sorted(sizes_lo.values())} → {sorted(sizes_hi.values())}"
    return "no size change"


def _sil_at(sil: pd.DataFrame, k: int) -> float:
    sub = sil.loc[sil["k"] == k, "silhouette"]
    if sub.empty:
        return float("nan")
    return float(sub.iloc[0])


def _classify_silhouette_curve(sil: pd.DataFrame, chosen_k: int) -> str:
    if sil.empty:
        return "missing"
    valid = sil.dropna(subset=["silhouette"])
    if valid.empty:
        return "missing"
    max_k = int(valid.loc[valid["silhouette"].idxmax(), "k"])
    sils = valid.sort_values("k")["silhouette"].to_numpy(dtype=float)
    diffs = np.diff(sils)
    # Allow tiny wiggles (e.g. BMMpM k=6→7) if the curve still climbs to k_max.
    increasing = (
        len(diffs) >= 2
        and bool(np.all(diffs > -0.01))
        and bool(sils[-1] > sils[0] + 0.02)
    )
    k2 = _sil_at(valid, 2)
    k5 = _sil_at(valid, 5)
    k6 = _sil_at(valid, 6)
    k7 = _sil_at(valid, 7)
    local_k6 = (
        np.isfinite(k6)
        and (not np.isfinite(k5) or k6 > k5)
        and (not np.isfinite(k7) or k6 >= k7 - 1e-6)
    )
    if increasing and max_k == int(valid["k"].max()):
        return "increasing"
    if max_k == 6:
        return "global_peak_k6"
    if np.isfinite(k2) and max_k == 2 and local_k6:
        return "k2_global_local_k6"
    if max_k == 2:
        return "k2_global"
    if max_k == chosen_k:
        return "peak_at_chosen_k"
    return f"global_peak_k{max_k}"


def collect_silhouette_by_k(
    cohort_dirs: Sequence[Path | str],
    *,
    group: str = "endpoint",
) -> pd.DataFrame:
    """Long table: one row per cohort × k."""
    rows: list[dict] = []
    for d in cohort_dirs:
        d = Path(d)
        cohort = cohort_name_from_changepoints_dir(d)
        sil = _load_silhouette(d, group)
        if sil.empty:
            continue
        chosen = chosen_k_from_cohort(d, group=group)
        finite = sil["silhouette"].to_numpy(dtype=float)
        max_idx = int(np.nanargmax(finite)) if np.isfinite(finite).any() else -1
        max_k = int(sil.iloc[max_idx]["k"]) if max_idx >= 0 else None
        for _, row in sil.iterrows():
            k = int(row["k"])
            rows.append(
                {
                    "cohort": cohort,
                    "group": group,
                    "k": k,
                    "silhouette": float(row["silhouette"]),
                    "n_clusters_effective": int(
                        row.get("n_clusters_effective", k)
                    )
                    if pd.notna(row.get("n_clusters_effective", k))
                    else k,
                    "is_chosen_k": k == chosen,
                    "is_global_max": max_k is not None and k == max_k,
                }
            )
    return pd.DataFrame(rows)


def collect_silhouette_peaks(
    cohort_dirs: Sequence[Path | str],
    *,
    group: str = "endpoint",
) -> pd.DataFrame:
    """One row per cohort: chosen k vs silhouette peak and curve shape."""
    rows: list[dict] = []
    for d in cohort_dirs:
        d = Path(d)
        cohort = cohort_name_from_changepoints_dir(d)
        sil = _load_silhouette(d, group)
        chosen = chosen_k_from_cohort(d, group=group)
        sizes = _sizes_at_k(d, chosen, group)
        if sil.empty:
            rows.append(
                {
                    "cohort": cohort,
                    "group": group,
                    "chosen_k": chosen,
                    "curve_shape": "missing",
                    "n_segments": sum(sizes.values()) if sizes else 0,
                    "cluster_sizes_chosen": ",".join(
                        str(sizes[k]) for k in sorted(sizes)
                    ),
                }
            )
            continue
        finite = sil.dropna(subset=["silhouette"])
        max_row = finite.loc[finite["silhouette"].idxmax()]
        rows.append(
            {
                "cohort": cohort,
                "group": group,
                "n_segments": sum(sizes.values()) if sizes else 0,
                "chosen_k": chosen,
                "silhouette_at_chosen": _sil_at(sil, chosen),
                "global_max_k": int(max_row["k"]),
                "global_max_silhouette": float(max_row["silhouette"]),
                "silhouette_k2": _sil_at(sil, 2),
                "silhouette_k5": _sil_at(sil, 5),
                "silhouette_k6": _sil_at(sil, 6),
                "curve_shape": _classify_silhouette_curve(sil, chosen),
                "cluster_sizes_chosen": ",".join(
                    str(sizes[k]) for k in sorted(sizes)
                ),
            }
        )
    return pd.DataFrame(rows)


def collect_k_splits(
    cohort_dirs: Sequence[Path | str],
    *,
    k_from: int = 5,
    k_to: int = 6,
    group: str = "endpoint",
) -> pd.DataFrame:
    """Ward split from *k_from* to *k_to* (exactly one cluster splits)."""
    rows: list[dict] = []
    for d in cohort_dirs:
        d = Path(d)
        lo = _sizes_at_k(d, k_from, group)
        hi = _sizes_at_k(d, k_to, group)
        rows.append(
            {
                "cohort": cohort_name_from_changepoints_dir(d),
                "group": group,
                "k_from": k_from,
                "k_to": k_to,
                "sizes_from": ",".join(str(lo[k]) for k in sorted(lo)),
                "sizes_to": ",".join(str(hi[k]) for k in sorted(hi)),
                "min_from": min(lo.values()) if lo else np.nan,
                "max_from": max(lo.values()) if lo else np.nan,
                "min_to": min(hi.values()) if hi else np.nan,
                "max_to": max(hi.values()) if hi else np.nan,
                "split_description": _describe_ward_split(lo, hi),
            }
        )
    return pd.DataFrame(rows)


def _first_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _span_stats(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "range": float("nan"),
            "iqr": float("nan"),
        }
    return {
        "mean": float(s.mean()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
        "range": float(s.max() - s.min()),
        "iqr": float(s.quantile(0.75) - s.quantile(0.25)),
    }


def collect_segment_dynamics(
    cohort_dirs: Sequence[Path | str],
    *,
    group: str = "endpoint",
) -> pd.DataFrame:
    """Pelt switching, segment lengths, and geometry-distance spread."""
    rows: list[dict] = []
    for d in cohort_dirs:
        d = Path(d)
        segs = _read_csv_if_exists(_cluster_dir(d, group) / "segments_clustered.csv")
        if segs.empty:
            continue
        n_traj = int(segs["traj_id"].nunique()) if "traj_id" in segs.columns else 0
        n_seg = len(segs)
        per = (
            segs.groupby("traj_id").size()
            if "traj_id" in segs.columns
            else pd.Series(dtype=int)
        )
        n_whole = int((per == 1).sum()) if not per.empty else 0
        n_one_cluster = 0
        if "traj_id" in segs.columns and "cluster_label" in segs.columns:
            n_one_cluster = int((segs.groupby("traj_id")["cluster_label"].nunique() == 1).sum())
        n_frames = segs["n_frames"] if "n_frames" in segs.columns else pd.Series(dtype=float)
        ed_col = _first_col(segs, ("endpoint_dist_mean_mean", "endpoint_dist_mean"))
        ed = segs[ed_col] if ed_col else pd.Series(dtype=float)
        n_ed_mean_cols = sum(
            1
            for c in segs.columns
            if str(c).startswith("endpoint_dist") and str(c).endswith("_mean")
        )
        row: dict = {
            "cohort": cohort_name_from_changepoints_dir(d),
            "group": group,
            "n_traj": n_traj,
            "n_seg": n_seg,
            "segs_per_traj": (n_seg / n_traj) if n_traj else np.nan,
            "n_breakpoints": int(max(n_seg - n_traj, 0)),
            "whole_traj_segments": n_whole,
            "whole_traj_frac": (n_whole / n_traj) if n_traj else np.nan,
            "one_cluster_traj": n_one_cluster,
            "one_cluster_frac": (n_one_cluster / n_traj) if n_traj else np.nan,
            "med_n_frames": float(n_frames.median()) if len(n_frames) else np.nan,
            "mean_n_frames": float(n_frames.mean()) if len(n_frames) else np.nan,
            "p90_n_frames": float(n_frames.quantile(0.9)) if len(n_frames) else np.nan,
            "max_segs_per_traj": int(per.max()) if not per.empty else 0,
            "endpoint_dist_mean": float(ed.mean()) if len(ed) else np.nan,
            "endpoint_dist_std": float(ed.std()) if len(ed) else np.nan,
            "endpoint_dist_range": float(ed.max() - ed.min()) if len(ed) else np.nan,
            "endpoint_dist_iqr": (
                float(ed.quantile(0.75) - ed.quantile(0.25)) if len(ed) else np.nan
            ),
            "n_endpoint_mean_cols": n_ed_mean_cols,
        }
        for key, cands in _GEOMETRY_SPECS:
            col = _first_col(segs, cands)
            stats = _span_stats(segs[col]) if col else _span_stats(pd.Series(dtype=float))
            if key == "endpoint_dist":
                continue
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
            row[f"{key}_range"] = stats["range"]
            row[f"{key}_iqr"] = stats["iqr"]
        rows.append(row)
    return pd.DataFrame(rows)


def collect_geometry_spread(
    cohort_dirs: Sequence[Path | str],
    *,
    group: str = "gsa",
) -> pd.DataFrame:
    """Segment-mean geometry spans (Rg, RMSD, octahedrality, endpoint dist)."""
    rows: list[dict] = []
    for d in cohort_dirs:
        d = Path(d)
        segs = _read_csv_if_exists(_cluster_dir(d, group) / "segments_clustered.csv")
        row: dict = {
            "cohort": cohort_name_from_changepoints_dir(d),
            "group": group,
            "n_segments": len(segs),
        }
        for key, cands in _GEOMETRY_SPECS:
            col = _first_col(segs, cands) if not segs.empty else None
            stats = _span_stats(segs[col]) if col else _span_stats(pd.Series(dtype=float))
            row[f"{key}_col"] = col or ""
            row[f"{key}_mean"] = stats["mean"]
            row[f"{key}_std"] = stats["std"]
            row[f"{key}_range"] = stats["range"]
            row[f"{key}_iqr"] = stats["iqr"]
        rows.append(row)
    return pd.DataFrame(rows)


def collect_site_and_d1(cohort_dirs: Sequence[Path | str]) -> pd.DataFrame:
    """Hull site kinds and paper-d1 open/elongated occupancy."""
    rows: list[dict] = []
    for d in cohort_dirs:
        d = Path(d)
        feat = d / "endpoint_features"
        sites = _read_csv_if_exists(feat / "endpoint_sites.csv")
        d1_atoms = _read_csv_if_exists(feat / "paper_d1_atoms.csv")
        d1_segs = _read_csv_if_exists(d / "paper_d1_segment_states.csv")

        n_sites = len(sites)
        n_ring = int((sites["kind"] == "ring").sum()) if "kind" in sites.columns else 0
        n_atom = int((sites["kind"] == "atom").sum()) if "kind" in sites.columns else 0

        s3s7_ring = s3s7_atom = 0
        eq_frac = float("nan")
        if not d1_atoms.empty:
            kind_col = "endpoint_kind" if "endpoint_kind" in d1_atoms.columns else None
            if kind_col:
                # One representative trajectory if present (kinds are topology-level).
                sample = d1_atoms
                if "traj_id" in d1_atoms.columns:
                    sample = d1_atoms[d1_atoms["traj_id"] == d1_atoms["traj_id"].iloc[0]]
                s3s7_ring = int(sample[kind_col].astype(str).str.contains("ring").sum())
                s3s7_atom = int((sample[kind_col].astype(str) == "atom").sum())
            if {"endpoint_atom_id", "d1_ring_atom_id"} <= set(d1_atoms.columns):
                eq_frac = float(
                    (d1_atoms["endpoint_atom_id"] == d1_atoms["d1_ring_atom_id"]).mean()
                )

        d1_min = frac_open = n_open_mean = float("nan")
        if not d1_segs.empty:
            if "paper_d1_min_mean" in d1_segs.columns:
                d1_min = float(d1_segs["paper_d1_min_mean"].mean())
            if "n_open_pairs" in d1_segs.columns:
                n_open_mean = float(d1_segs["n_open_pairs"].mean())
                frac_open = float((d1_segs["n_open_pairs"] > 0).mean())
            elif "paper_d1_n_open_mean" in d1_segs.columns:
                n_open_mean = float(d1_segs["paper_d1_n_open_mean"].mean())
                frac_open = float((d1_segs["paper_d1_n_open_mean"] > 0).mean())

        rows.append(
            {
                "cohort": cohort_name_from_changepoints_dir(d),
                "n_hull_sites": n_sites,
                "n_ring_sites": n_ring,
                "n_atom_sites": n_atom,
                "atom_site_frac": (n_atom / n_sites) if n_sites else np.nan,
                "s3s7_ring": s3s7_ring,
                "s3s7_atom": s3s7_atom,
                "s3s7_endpoint_eq_d1_frac": eq_frac,
                "paper_d1_min_mean": d1_min,
                "n_open_pairs_mean": n_open_mean,
                "frac_seg_any_open": frac_open,
            }
        )
    return pd.DataFrame(rows)


def _format_txt_summary(
    peaks: pd.DataFrame,
    splits: pd.DataFrame,
    dynamics: pd.DataFrame,
    sites: pd.DataFrame,
    *,
    group: str = "endpoint",
    geometry: Optional[pd.DataFrame] = None,
) -> str:
    label = "Endpoint" if group == "endpoint" else f"{group.upper()} cage"
    lines = [
        f"{label} cluster k diagnostics",
        "================================",
        "",
        "k=5 is the pipeline default (--n-clusters), not silhouette-selected.",
        f"Silhouette curves are inspection-only (clusters/{group}/silhouette_by_k.csv).",
        "",
        "Silhouette peaks",
        "----------------",
    ]
    if not peaks.empty:
        show = peaks.copy()
        for col in (
            "silhouette_at_chosen",
            "global_max_silhouette",
            "silhouette_k2",
            "silhouette_k5",
            "silhouette_k6",
        ):
            if col in show.columns:
                show[col] = show[col].map(
                    lambda x: f"{x:.3f}" if pd.notna(x) else ""
                )
        lines.append(show.to_string(index=False))
    lines.extend(["", "Ward split k=5 → k=6", "---------------------"])
    if not splits.empty:
        cols = [c for c in ("cohort", "sizes_from", "sizes_to", "split_description") if c in splits.columns]
        lines.append(splits[cols].to_string(index=False))
    lines.extend(["", "Segment dynamics", "----------------"])
    if not dynamics.empty:
        cols = [
            c
            for c in (
                "cohort",
                "n_traj",
                "n_seg",
                "segs_per_traj",
                "n_breakpoints",
                "whole_traj_frac",
                "one_cluster_frac",
                "mean_n_frames",
                "endpoint_dist_range",
                "assembly_rg_range",
                "assembly_rmsd_to_ref_range",
            )
            if c in dynamics.columns
        ]
        lines.append(dynamics[cols].round(3).to_string(index=False))
    if geometry is not None and not geometry.empty:
        lines.extend(["", "Geometry spread (segment means)", "--------------------------------"])
        gcols = [
            c
            for c in (
                "cohort",
                "assembly_rg_range",
                "assembly_rmsd_to_ref_range",
                "octahedrality_score_range",
                "endpoint_dist_range",
                "n_guest_inside_cavity_range",
                "cavity_ion_count_range",
                "cavity_water_count_range",
            )
            if c in geometry.columns
        ]
        lines.append(geometry[gcols].round(3).to_string(index=False))
    if group == "endpoint":
        lines.extend(["", "Paper d1 / site kinds", "---------------------"])
        if not sites.empty:
            cols = [
                c
                for c in (
                    "cohort",
                    "atom_site_frac",
                    "s3s7_ring",
                    "s3s7_atom",
                    "s3s7_endpoint_eq_d1_frac",
                    "paper_d1_min_mean",
                    "frac_seg_any_open",
                )
                if c in sites.columns
            ]
            lines.append(sites[cols].round(3).to_string(index=False))
        lines.extend(
            [
                "",
                "Talk track",
                "----------",
                "- All B* cohorts have five clusters because Ward is cut at fixed k=5.",
                "- BHHpM: global silhouette max at k=6 (balanced split of one cluster).",
                "- BHHpH / BMHpM: global max at k=2; k=6 is a weak local bump (tiny peel).",
                "- BMMpM: silhouette rises with k — few switches, elongated d1, nested packing.",
                "- Do not auto-select k by silhouette for these cubes (k=2 is too coarse;",
                "  BMHpH k=2 is an outlier trap).",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Talk track",
                "----------",
                f"- All B* {group} runs are cut at fixed k=5 (pipeline default).",
            ]
        )
        if not peaks.empty and "curve_shape" in peaks.columns:
            k2 = peaks[
                peaks["curve_shape"].astype(str).str.startswith("k2_")
            ]
            if not k2.empty:
                lines.append(
                    "- k=2 is the global silhouette max for: "
                    + ", ".join(k2["cohort"].astype(str))
                    + "."
                )
            other = peaks[
                ~peaks["curve_shape"].astype(str).str.startswith("k2_")
            ]
            for _, row in other.iterrows():
                lines.append(
                    f"- {row['cohort']}: {row.get('curve_shape', '')} "
                    f"(global max k={row.get('global_max_k', '')})."
                )
        if not dynamics.empty and "segs_per_traj" in dynamics.columns:
            lo = dynamics.loc[dynamics["segs_per_traj"].idxmin()]
            hi = dynamics.loc[dynamics["segs_per_traj"].idxmax()]
            lines.append(
                f"- Most Pelt switching: {hi['cohort']} "
                f"({float(hi['segs_per_traj']):.1f} segs/traj); "
                f"least: {lo['cohort']} "
                f"({float(lo['segs_per_traj']):.1f})."
            )
        if geometry is not None and not geometry.empty and "assembly_rg_iqr" in geometry.columns:
            lines.append(
                "- Prefer IQR over range for geometry span when one cube has "
                "exploded-cage outliers (range is then an outlier trap)."
            )
        lines.append(
            "- Do not auto-select k by silhouette (k=2 is typically too coarse)."
        )
    return "\n".join(lines) + "\n"


def _write_k_diagnostic_plots(
    sil_long: pd.DataFrame,
    peaks: pd.DataFrame,
    dynamics: pd.DataFrame,
    sites: pd.DataFrame,
    plot_dir: Path,
    written: dict[str, Path],
    *,
    chosen_k: int = 5,
    group: str = "endpoint",
    geometry: Optional[pd.DataFrame] = None,
) -> None:
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if "tk" in backend or backend == "macosx":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    cohorts = (
        list(sil_long["cohort"].unique())
        if not sil_long.empty
        else (list(dynamics["cohort"].unique()) if not dynamics.empty else [])
    )
    cmap = plt.get_cmap("tab10")
    colors = {c: cmap(i % 10) for i, c in enumerate(cohorts)}
    group_title = "Endpoint" if group == "endpoint" else group.upper()

    if not sil_long.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4.4))
        for cohort, sub in sil_long.groupby("cohort"):
            sub = sub.sort_values("k")
            ax.plot(
                sub["k"],
                sub["silhouette"],
                "o-",
                lw=1.6,
                ms=5,
                color=colors.get(cohort),
                label=cohort,
            )
        ax.axvline(chosen_k, color="0.35", ls="--", lw=1.1, label=f"chosen k={chosen_k}")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Silhouette score")
        ax.set_title(f"{group_title} segment silhouette vs k")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8, ncol=2)
        fig.tight_layout()
        p = plot_dir / "silhouette_vs_k.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written["plots/silhouette_vs_k.png"] = p

    if not dynamics.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
        x = np.arange(len(dynamics))
        labels = dynamics["cohort"].tolist()
        bar_c = [colors.get(c, "C0") for c in labels]
        axes[0].bar(x, dynamics["segs_per_traj"], color=bar_c, edgecolor="k", lw=0.4)
        axes[0].set_title("Segments per trajectory")
        axes[0].set_ylabel("mean segments / traj")
        axes[1].bar(x, dynamics["whole_traj_frac"], color=bar_c, edgecolor="k", lw=0.4)
        axes[1].set_title("Whole-trajectory segments")
        axes[1].set_ylabel("fraction of trajs")
        axes[2].bar(x, dynamics["one_cluster_frac"], color=bar_c, edgecolor="k", lw=0.4)
        axes[2].set_title("Trajs that never change cluster")
        axes[2].set_ylabel("fraction of trajs")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"Pelt switching ({group} group)", y=1.02)
        fig.tight_layout()
        p = plot_dir / "segment_switching.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written["plots/segment_switching.png"] = p

        if "endpoint_dist_range" in dynamics.columns and dynamics["endpoint_dist_range"].notna().any():
            fig, ax = plt.subplots(figsize=(7.0, 4.0))
            w = 0.38
            ax.bar(
                x - w / 2,
                dynamics["endpoint_dist_range"],
                w,
                color=bar_c,
                edgecolor="k",
                lw=0.4,
                label="max − min",
            )
            if "endpoint_dist_iqr" in dynamics.columns:
                ax.bar(
                    x + w / 2,
                    dynamics["endpoint_dist_iqr"],
                    w,
                    color=bar_c,
                    alpha=0.45,
                    edgecolor="k",
                    lw=0.4,
                    label="IQR",
                )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("endpoint_dist_mean span (Å)")
            ax.set_title("Assembly endpoint-distance span (range vs IQR)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            p = plot_dir / "endpoint_dist_span.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written["plots/endpoint_dist_span.png"] = p

    geo = geometry if geometry is not None else pd.DataFrame()
    if not geo.empty:
        span_cols = [
            ("assembly_rg_range", "Rg range (Å)"),
            ("assembly_rmsd_to_ref_range", "RMSD-to-ref range (Å)"),
            ("octahedrality_score_range", "Octahedrality range"),
            ("n_guest_inside_cavity_range", "Guests inside cavity"),
            ("guest_radial_distance_mean_range", "Guest radial dist (Å)"),
            ("cavity_ion_count_range", "Cavity Na⁺ count"),
            ("cavity_water_count_range", "Cavity water count"),
        ]
        present = [
            (c, lab)
            for c, lab in span_cols
            if c in geo.columns and geo[c].notna().any()
        ][:3]
        if present:
            fig, axes = plt.subplots(1, len(present), figsize=(4.1 * len(present), 4.0), squeeze=False)
            x = np.arange(len(geo))
            labels = geo["cohort"].tolist()
            bar_c = [colors.get(c, "C0") for c in labels]
            for ax, (col, ylab) in zip(axes[0], present):
                ax.bar(x, geo[col], color=bar_c, edgecolor="k", lw=0.4)
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_ylabel(ylab)
                ax.grid(axis="y", alpha=0.25)
            fig.suptitle(f"{group_title} geometry span across segments", y=1.02)
            fig.tight_layout()
            p = plot_dir / "geometry_span.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written["plots/geometry_span.png"] = p

    if group == "endpoint" and not sites.empty and "paper_d1_min_mean" in sites.columns:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
        x = np.arange(len(sites))
        labels = sites["cohort"].tolist()
        bar_c = [colors.get(c, "C0") for c in labels]
        axes[0].bar(x, sites["paper_d1_min_mean"], color=bar_c, edgecolor="k", lw=0.4)
        axes[0].axhspan(4.5, 5.5, color="0.75", alpha=0.35, label="open cation–π window")
        axes[0].set_ylabel("mean paper_d1_min (Å)")
        axes[0].set_title("Closest paper-d1 contact")
        axes[0].legend(fontsize=8)
        axes[1].bar(x, sites["frac_seg_any_open"], color=bar_c, edgecolor="k", lw=0.4)
        axes[1].set_ylabel("fraction of segments")
        axes[1].set_title("Segments with ≥1 open d1 pair")
        for ax in axes:
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle("Paper d1 occupancy (BMM never enters 4.5–5.5 Å)", y=1.02)
        fig.tight_layout()
        p = plot_dir / "paper_d1_open.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written["plots/paper_d1_open.png"] = p

        if "n_ring_sites" in sites.columns and "n_atom_sites" in sites.columns:
            fig, ax = plt.subplots(figsize=(6.8, 4.0))
            ax.bar(x, sites["n_ring_sites"], color="#4c78a8", edgecolor="k", lw=0.4, label="ring")
            ax.bar(
                x,
                sites["n_atom_sites"],
                bottom=sites["n_ring_sites"],
                color="#f58518",
                edgecolor="k",
                lw=0.4,
                label="atom (exocyclic)",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("endpoint hull sites (all trajs)")
            ax.set_title("Endpoint-site kinds")
            ax.legend(loc="best")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            p = plot_dir / "hull_site_kinds.png"
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written["plots/hull_site_kinds.png"] = p

    if not peaks.empty and "curve_shape" in peaks.columns:
        fig, ax = plt.subplots(figsize=(7.2, 3.6 + 0.25 * max(len(peaks), 1)))
        ax.axis("off")
        lines = ["Silhouette curve shape (inspection, does not select k)", ""]
        for _, row in peaks.iterrows():
            lines.append(
                f"{row['cohort']}: {row.get('curve_shape', '')}  "
                f"(chosen k={int(row.get('chosen_k', chosen_k))}, "
                f"global max k={row.get('global_max_k', '')})"
            )
        ax.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=9)
        fig.tight_layout()
        p = plot_dir / "curve_shape_legend.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written["plots/curve_shape_legend.png"] = p


def compare_cluster_k_diagnostics(
    cohort_dirs: Sequence[Path | str],
    out_dir: Path | str,
    *,
    group: str = "endpoint",
    k_from: int = 5,
    k_to: int = 6,
    include_site_d1: Optional[bool] = None,
) -> dict[str, Path]:
    """Write silhouette / switching / geometry tables and plots for B* cohorts."""
    out_dir = Path(out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    dirs = [Path(d) for d in cohort_dirs]
    if not dirs:
        log_event(
            "warning",
            "No cohort directories for k diagnostics",
            component="cluster_k_diagnostics",
        )
        return written

    if include_site_d1 is None:
        include_site_d1 = group == "endpoint"

    sil_long = collect_silhouette_by_k(dirs, group=group)
    peaks = collect_silhouette_peaks(dirs, group=group)
    splits = collect_k_splits(dirs, k_from=k_from, k_to=k_to, group=group)
    dynamics = collect_segment_dynamics(dirs, group=group)
    geometry = collect_geometry_spread(dirs, group=group)
    sites = collect_site_and_d1(dirs) if include_site_d1 else pd.DataFrame()

    tables = {
        "silhouette_by_k.csv": sil_long,
        "silhouette_peaks.csv": peaks,
        "k_split.csv": splits,
        "segment_dynamics.csv": dynamics,
        "geometry_spread.csv": geometry,
    }
    if include_site_d1:
        tables["site_and_d1.csv"] = sites
    for name, df in tables.items():
        path = out_dir / name
        df.to_csv(path, index=False)
        written[name] = path

    summary_path = out_dir / "k_diagnostics_summary.txt"
    summary_path.write_text(
        _format_txt_summary(
            peaks, splits, dynamics, sites, group=group, geometry=geometry
        ),
        encoding="utf-8",
    )
    written["k_diagnostics_summary.txt"] = summary_path

    chosen_vals = peaks["chosen_k"].dropna().astype(int) if not peaks.empty else pd.Series(dtype=int)
    chosen_k = int(chosen_vals.mode().iloc[0]) if len(chosen_vals) else 5
    _write_k_diagnostic_plots(
        sil_long,
        peaks,
        dynamics,
        sites,
        plot_dir,
        written,
        chosen_k=chosen_k,
        group=group,
        geometry=geometry,
    )

    log_event(
        "info",
        f"Wrote {group} cluster k diagnostics for {len(dirs)} cohort(s)",
        component="cluster_k_diagnostics",
    )
    return written


def compare_endpoint_cluster_k_diagnostics(
    cohort_dirs: Sequence[Path | str],
    out_dir: Path | str,
    *,
    k_from: int = 5,
    k_to: int = 6,
) -> dict[str, Path]:
    """Write silhouette / switching / d1 tables and plots for B* endpoint cohorts."""
    return compare_cluster_k_diagnostics(
        cohort_dirs,
        out_dir,
        group="endpoint",
        k_from=k_from,
        k_to=k_to,
        include_site_d1=True,
    )
