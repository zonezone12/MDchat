"""Map existing hull-index endpoint pair distances onto Murata d1 (C2–C3).

The stored ``endpoint_dist_{i}s{a}_{j}s{b}`` columns used convex-hull site
order. This module matches those sites to canonical R2/R3 (C2/C3 ipso or the
stored substituent atom) and keeps the **six equatorial** R2(i)–R3(j) contacts
— one interlocking cube edge each — instead of all 30 monomer-pair directions.

Distances are the already-computed site centroids: para-carbon = C2/C3 when
R=H; methyl carbon (not ipso) when R=CH3. A fresh ipso–ipso pass is still
required for the CH3 cubes if you need Murata's exact C2–C3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from .endpoint_features import paper_d1_column_name
from .transition_attribution import parse_site_pair_feature

D1_COMPACT_LO = 4.5
D1_COMPACT_HI = 5.5
D1_ELONGATED_LO = 7.0

CUBES = ("BHHpH", "BHHpM", "BMHpH", "BMHpM", "BMMpH", "BMMpM")


def site_pair_column(mon_i: int, site_a: int, mon_j: int, site_b: int) -> str:
    return f"endpoint_dist_{int(mon_i)}s{int(site_a)}_{int(mon_j)}s{int(site_b)}"


def classify_murata_d1(distance: float) -> str:
    """Murata d1 window: compact 4.5–5.5 Å, elongated ≥ 7.0 Å."""
    if not np.isfinite(distance):
        return "unknown"
    if D1_COMPACT_LO <= distance <= D1_COMPACT_HI:
        return "compact"
    if distance >= D1_ELONGATED_LO:
        return "elongated"
    if distance < D1_COMPACT_LO:
        return "short"
    return "intermediate"


def _parse_atom_ids(raw: Any) -> set[int]:
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return set()
    text = str(raw).replace("|", " ").replace(",", " ").strip()
    if not text or text.lower() == "nan":
        return set()
    out: set[int] = set()
    for tok in text.split():
        try:
            out.add(int(float(tok)))
        except ValueError:
            continue
    return out


def _six_ring_containing(mol: Any, atom_idx: int) -> set[int]:
    rings = mol.GetRingInfo().AtomRings()
    for ring in rings:
        if int(atom_idx) in ring and len(ring) == 6:
            return {int(a) for a in ring}
    return {int(atom_idx)}


def canonical_role_atoms_mda(
    universe: Any,
    monomer_selections: Sequence[str],
) -> pd.DataFrame:
    """MDA atom ids for each canonical role on each monomer (topology only)."""
    from src.EndpointAnalyzer.gsa_site_map import CANONICAL_ROLES, classify_gsa_monomer

    rows: list[dict[str, Any]] = []
    for mon_idx, mon_sel in enumerate(monomer_selections):
        sel = universe.select_atoms(mon_sel)
        if len(sel) == 0:
            raise ValueError(f"Empty monomer selection {mon_sel}")
        mol = sel.convert_to("RDKIT")
        if mol is None:
            raise ValueError(f"RDKit conversion failed for {mon_sel}")
        gsa = classify_gsa_monomer(mol)
        id_of = {i: int(sel[i].id) for i in range(len(sel))}
        for role in CANONICAL_ROLES:
            site = gsa.by_role(role)
            rdkit_atoms = [int(a) for a in site.rdkit_atoms]
            ipso = int(site.ipso_rdkit) if site.ipso_rdkit is not None else None
            subst = int(site.subst_rdkit) if site.subst_rdkit is not None else None
            cpy = int(site.cpy_rdkit) if site.cpy_rdkit is not None else None
            ring_rdkit = set(rdkit_atoms)
            if ipso is not None:
                ring_rdkit |= _six_ring_containing(mol, ipso)
            mda_ids = {id_of[a] for a in ring_rdkit if a in id_of}
            rows.append(
                {
                    "monomer": mon_idx,
                    "role": role,
                    "kind": site.kind,
                    "methyl": bool(site.methyl),
                    "ipso_mda_id": id_of[ipso] if ipso is not None and ipso in id_of else None,
                    "subst_mda_id": (
                        id_of[subst] if subst is not None and subst in id_of else None
                    ),
                    "cpy_mda_id": id_of[cpy] if cpy is not None and cpy in id_of else None,
                    "atom_ids": " ".join(str(a) for a in sorted(mda_ids)),
                }
            )
    return pd.DataFrame(rows)


def match_hull_sites_to_roles(
    sites_df: pd.DataFrame,
    role_atoms_df: pd.DataFrame,
    *,
    required_roles: Sequence[str] = ("r2", "r3"),
) -> pd.DataFrame:
    """Assign each canonical role the hull ``site_index`` with best atom overlap.

    Prefer an atom site that contains the substituent or ipso carbon over a
    ring-centroid site of the same aryl (so R=H d1 is C2–C3, not ring COM).
    """
    sites = sites_df.copy()
    if "traj_id" in sites.columns and len(sites):
        first = sites["traj_id"].iloc[0]
        sites = sites[sites["traj_id"] == first]
    rows: list[dict[str, Any]] = []
    monomers = sorted(set(int(m) for m in role_atoms_df["monomer"]))
    for mon in monomers:
        hull = sites[sites["monomer"].astype(int) == mon]
        roles = role_atoms_df[role_atoms_df["monomer"].astype(int) == mon]
        used: set[int] = set()
        for _, rrow in roles.iterrows():
            role = str(rrow["role"])
            want = _parse_atom_ids(rrow["atom_ids"])
            ipso = rrow.get("ipso_mda_id")
            subst = rrow.get("subst_mda_id")
            ipso_i = int(ipso) if pd.notna(ipso) else None
            subst_i = int(subst) if pd.notna(subst) else None
            best: Optional[tuple[float, int, pd.Series]] = None
            for _, srow in hull.iterrows():
                si = int(srow["site_index"])
                if si in used:
                    continue
                have = _parse_atom_ids(srow.get("atom_ids"))
                kind = str(srow.get("kind", ""))
                role_kind = str(rrow.get("kind", ""))
                score = 0.0
                if subst_i is not None and subst_i in have:
                    score += 100.0
                if ipso_i is not None and ipso_i in have:
                    score += 90.0
                if want:
                    frac = len(want & have) / max(len(want), 1)
                    score += 10.0 * frac
                    # Ring roles (Ph, Py+) match whole-ring hull sites, not ipso atoms.
                    if role_kind == "ring":
                        score += 100.0 * frac
                # Prefer the singleton C2/C3 (or methyl) over the aryl ring centroid.
                if kind == "atom" and (
                    (subst_i is not None and subst_i in have)
                    or (ipso_i is not None and ipso_i in have)
                ):
                    score += 50.0
                if best is None or score > best[0]:
                    best = (score, si, srow)
            min_ok = 80.0 if str(rrow.get("kind", "")) == "ring" else 90.0
            if best is None or best[0] < min_ok:
                if role not in required_roles:
                    continue
                raise ValueError(
                    f"Could not match monomer {mon} role {role} to a hull site "
                    f"(best score {None if best is None else best[0]})"
                )
            used.add(best[1])
            srow = best[2]
            rows.append(
                {
                    "monomer": mon,
                    "role": role,
                    "site_index": best[1],
                    "hull_kind": str(srow.get("kind", "")),
                    "methyl": bool(rrow.get("methyl", False)),
                    "ipso_mda_id": ipso_i,
                    "subst_mda_id": subst_i,
                    "hull_atom_ids": srow.get("atom_ids", ""),
                    "match_score": best[0],
                    "proxy": (
                        "ipso_or_para"
                        if (subst_i is not None and ipso_i is not None and subst_i == ipso_i)
                        else (
                            "methyl_carbon"
                            if str(srow.get("kind", "")) == "atom"
                            else "ring_centroid"
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def role_site_index(matched: pd.DataFrame, monomer: int, role: str) -> int:
    hit = matched[
        (matched["monomer"].astype(int) == int(monomer))
        & (matched["role"].astype(str) == str(role))
    ]
    if hit.empty:
        raise KeyError(f"No hull site for monomer {monomer} role {role}")
    return int(hit.iloc[0]["site_index"])


def r2_r3_candidate_columns(matched: pd.DataFrame) -> list[tuple[str, int, int]]:
    """All directed R2(i)→R3(j) site-pair columns, i≠j."""
    monomers = sorted(set(int(m) for m in matched["monomer"]))
    out: list[tuple[str, int, int]] = []
    for i in monomers:
        si = role_site_index(matched, i, "r2")
        for j in monomers:
            if i == j:
                continue
            sj = role_site_index(matched, j, "r3")
            out.append((site_pair_column(i, si, j, sj), i, j))
    return out


def assign_equatorial_r2_r3_edges(
    cost: np.ndarray,
) -> list[tuple[int, int]]:
    """Hungarian assignment: each monomer's R2 pairs with one unique neighbor R3.

    ``cost[i, j]`` = typical distance from monomer *i* R2 to monomer *j* R3.
    Diagonal is ignored (set to +inf).
    """
    from scipy.optimize import linear_sum_assignment

    mat = np.asarray(cost, dtype=float).copy()
    n = mat.shape[0]
    if mat.shape != (n, n):
        raise ValueError(f"cost must be square, got {mat.shape}")
    np.fill_diagonal(mat, np.inf)
    if not np.isfinite(mat).any():
        raise ValueError("No finite off-diagonal R2–R3 distances")
    finite_max = float(np.nanmax(mat[np.isfinite(mat)]))
    mat = np.where(np.isfinite(mat), mat, finite_max + 1e6)
    rows, cols = linear_sum_assignment(mat)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if i != j]


def _median_cost_matrix(
    df: pd.DataFrame,
    candidates: Sequence[tuple[str, int, int]],
    n_monomers: int,
) -> np.ndarray:
    cost = np.full((n_monomers, n_monomers), np.inf, dtype=float)
    for col, i, j in candidates:
        if col not in df.columns:
            alt = None
            parsed = parse_site_pair_feature(col)
            if parsed is not None:
                mi, sa, mj, sb = parsed
                alt = site_pair_column(mj, sb, mi, sa)
            if alt is None or alt not in df.columns:
                continue
            col = alt
        cost[i, j] = float(np.nanmedian(pd.to_numeric(df[col], errors="coerce")))
    return cost


def d1_edge_columns(
    matched: pd.DataFrame,
    edges: Sequence[tuple[int, int]],
) -> list[tuple[str, int, int, str]]:
    """``(column, mon_r2, mon_r3, label)`` for the six equatorial contacts."""
    rows: list[tuple[str, int, int, str]] = []
    for k, (i, j) in enumerate(edges):
        si = role_site_index(matched, i, "r2")
        sj = role_site_index(matched, j, "r3")
        col = site_pair_column(i, si, j, sj)
        label = f"d1_e{k}_m{i}r2_m{j}r3"
        rows.append((col, i, j, label))
    return rows


def remap_d1_frame_table(
    features_df: pd.DataFrame,
    edge_cols: Sequence[tuple[str, int, int, str]],
) -> pd.DataFrame:
    """Copy the six equatorial pair columns and add Murata occupancy counts."""
    meta = [c for c in ("traj_id", "frame", "time_ps") if c in features_df.columns]
    out = features_df.loc[:, meta].copy() if meta else pd.DataFrame(index=features_df.index)
    dists = np.full((len(features_df), len(edge_cols)), np.nan, dtype=float)
    for k, (col, i, j, label) in enumerate(edge_cols):
        series = None
        if col in features_df.columns:
            series = features_df[col]
        else:
            parsed = parse_site_pair_feature(col)
            if parsed is not None:
                mi, sa, mj, sb = parsed
                alt = site_pair_column(mj, sb, mi, sa)
                if alt in features_df.columns:
                    series = features_df[alt]
                    col = alt
        if series is None:
            raise KeyError(f"Missing site-pair column {col}")
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        dists[:, k] = vals
        out[label] = vals
        out[f"{label}_src"] = col
        out[paper_d1_column_name(i, "r2", j, "r3")] = vals
    out["murata_d1_min"] = np.nanmin(dists, axis=1)
    out["murata_d1_max"] = np.nanmax(dists, axis=1)
    compact = (dists >= D1_COMPACT_LO) & (dists <= D1_COMPACT_HI)
    elongated = dists >= D1_ELONGATED_LO
    out["murata_d1_n_compact"] = np.sum(compact, axis=1).astype(int)
    out["murata_d1_n_elongated"] = np.sum(elongated, axis=1).astype(int)
    out["murata_d1_n_short"] = np.sum(dists < D1_COMPACT_LO, axis=1).astype(int)
    out["murata_d1_n_intermediate"] = np.sum(
        (dists > D1_COMPACT_HI) & (dists < D1_ELONGATED_LO), axis=1
    ).astype(int)
    return out


def _cohort_from_dir(path: Path) -> str:
    name = path.name
    prefix = "endpoint_changepoints_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def remap_endpoint_cohort_d1(
    changepoints_dir: Path | str,
    topology: Path | str,
    output_dir: Path | str,
    *,
    assignment_csv: Optional[Path | str] = None,
) -> dict[str, Path]:
    """Match hull sites → R2/R3, lock six equatorial edges, rewrite d1 tables."""
    import MDAnalysis as mda

    from src.utils.gsa_selections import resolve_selections

    changepoints_dir = Path(changepoints_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = changepoints_dir / "endpoint_features"
    sites_path = feat_dir / "endpoint_sites.csv"
    if not sites_path.is_file():
        raise FileNotFoundError(sites_path)

    u = mda.Universe(str(topology))
    sels = resolve_selections(u, gsa_resname="MOL", n_monomers=6, auto_tooth=False)
    # Stored hull sites from the CSV (not a fresh canonical find).
    sites_df = pd.read_csv(sites_path)
    role_atoms = canonical_role_atoms_mda(u, sels.monomer_selections)
    matched = match_hull_sites_to_roles(sites_df, role_atoms)
    matched.insert(0, "cohort", _cohort_from_dir(changepoints_dir))
    matched_path = output_dir / "role_site_map.csv"
    matched.to_csv(matched_path, index=False)

    csvs = sorted(feat_dir.glob("*_endpoint_features.csv"))
    if not csvs:
        raise FileNotFoundError(f"No feature CSVs in {feat_dir}")
    assign_path = Path(assignment_csv) if assignment_csv else csvs[0]
    candidates = r2_r3_candidate_columns(matched)
    cand_cols = set()
    for col, _, _ in candidates:
        cand_cols.add(col)
        parsed = parse_site_pair_feature(col)
        if parsed is not None:
            mi, sa, mj, sb = parsed
            cand_cols.add(site_pair_column(mj, sb, mi, sa))
    usecols = {"traj_id", "frame", "time_ps"} | cand_cols
    header = pd.read_csv(assign_path, nrows=0)
    usecols = [c for c in usecols if c in header.columns]
    sample = pd.read_csv(assign_path, usecols=usecols)
    n_mon = int(matched["monomer"].max()) + 1
    cost = _median_cost_matrix(sample, candidates, n_mon)
    edges = assign_equatorial_r2_r3_edges(cost)
    edge_cols = d1_edge_columns(matched, edges)
    edge_rows = []
    for k, (col, i, j, label) in enumerate(edge_cols):
        edge_rows.append(
            {
                "cohort": _cohort_from_dir(changepoints_dir),
                "edge": k,
                "monomer_r2": i,
                "monomer_r3": j,
                "site_r2": role_site_index(matched, i, "r2"),
                "site_r3": role_site_index(matched, j, "r3"),
                "feature": col,
                "label": label,
                "median_A": float(cost[i, j]),
                "r2_proxy": str(
                    matched.loc[
                        (matched["monomer"] == i) & (matched["role"] == "r2"), "proxy"
                    ].iloc[0]
                ),
                "r3_proxy": str(
                    matched.loc[
                        (matched["monomer"] == j) & (matched["role"] == "r3"), "proxy"
                    ].iloc[0]
                ),
            }
        )
    edges_df = pd.DataFrame(edge_rows)
    edges_path = output_dir / "equatorial_edges.csv"
    edges_df.to_csv(edges_path, index=False)

    needed: set[str] = {"traj_id", "frame", "time_ps"}
    for col, _, _, _ in edge_cols:
        needed.add(col)
        parsed = parse_site_pair_feature(col)
        if parsed is not None:
            mi, sa, mj, sb = parsed
            needed.add(site_pair_column(mj, sb, mi, sa))
    frames: list[pd.DataFrame] = []
    for csv_path in csvs:
        hdr = pd.read_csv(csv_path, nrows=0)
        cols = [c for c in needed if c in hdr.columns]
        raw = pd.read_csv(csv_path, usecols=cols)
        if "traj_id" not in raw.columns:
            raw.insert(0, "traj_id", csv_path.name.replace("_endpoint_features.csv", ""))
        frames.append(remap_d1_frame_table(raw, edge_cols))
    all_frames = pd.concat(frames, ignore_index=True)
    frames_path = output_dir / "murata_d1_frames.csv"
    all_frames.to_csv(frames_path, index=False)

    edge_labels = [lab for _, _, _, lab in edge_cols]
    stacked = all_frames[edge_labels].to_numpy(dtype=float).ravel()
    stacked = stacked[np.isfinite(stacked)]
    summary = {
        "cohort": _cohort_from_dir(changepoints_dir),
        "n_trajectories": int(all_frames["traj_id"].nunique()) if "traj_id" in all_frames else 1,
        "n_frames": int(len(all_frames)),
        "n_d1_values": int(stacked.size),
        "d1_median": float(np.median(stacked)) if stacked.size else np.nan,
        "d1_p05": float(np.percentile(stacked, 5)) if stacked.size else np.nan,
        "d1_p95": float(np.percentile(stacked, 95)) if stacked.size else np.nan,
        "frac_compact_4p5_5p5": float(np.mean(
            (stacked >= D1_COMPACT_LO) & (stacked <= D1_COMPACT_HI)
        )) if stacked.size else np.nan,
        "frac_elongated_ge7": float(np.mean(stacked >= D1_ELONGATED_LO)) if stacked.size else np.nan,
        "frac_short_lt4p5": float(np.mean(stacked < D1_COMPACT_LO)) if stacked.size else np.nan,
        "frac_intermediate_5p5_7": float(np.mean(
            (stacked > D1_COMPACT_HI) & (stacked < D1_ELONGATED_LO)
        )) if stacked.size else np.nan,
        "mean_n_compact_per_frame": float(all_frames["murata_d1_n_compact"].mean()),
        "mean_n_elongated_per_frame": float(all_frames["murata_d1_n_elongated"].mean()),
        "frac_frames_any_compact": float((all_frames["murata_d1_n_compact"] > 0).mean()),
        "frac_frames_any_elongated": float((all_frames["murata_d1_n_elongated"] > 0).mean()),
        "d1_min_mean": float(all_frames["murata_d1_min"].mean()),
    }
    summary_path = output_dir / "murata_d1_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    return {
        "role_site_map": matched_path,
        "equatorial_edges": edges_path,
        "murata_d1_frames": frames_path,
        "murata_d1_summary": summary_path,
    }


def plot_murata_d1_overview(
    summaries: pd.DataFrame,
    frames_by_cohort: dict[str, pd.DataFrame],
    output_path: Path | str,
) -> Path:
    """Histogram of equatorial d1 plus compact/elongated occupancy by cube."""
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cohorts = list(frames_by_cohort)
    n = len(cohorts)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    bins = np.linspace(3.0, 16.0, 53)
    for cohort, df in frames_by_cohort.items():
        labs = [
            c
            for c in df.columns
            if str(c).startswith("d1_e") and str(c).endswith("r3")
        ]
        vals = df[labs].to_numpy(dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        axes[0].hist(vals, bins=bins, histtype="step", density=True, label=cohort, lw=1.4)
    axes[0].axvspan(D1_COMPACT_LO, D1_COMPACT_HI, color="0.85", zorder=0, label="compact 4.5–5.5")
    axes[0].axvline(D1_ELONGATED_LO, color="crimson", ls="--", lw=1, label="elongated ≥ 7.0")
    axes[0].set_xlabel("equatorial d1 (Å)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Murata d1 from remapped R2–R3 site pairs")
    axes[0].legend(fontsize=8, ncol=2)

    x = np.arange(n)
    compact = [float(summaries.loc[summaries["cohort"] == c, "frac_compact_4p5_5p5"].iloc[0]) for c in cohorts]
    elong = [float(summaries.loc[summaries["cohort"] == c, "frac_elongated_ge7"].iloc[0]) for c in cohorts]
    axes[1].bar(x - 0.2, compact, 0.4, label="fraction of d1 in 4.5–5.5 Å", color="steelblue")
    axes[1].bar(x + 0.2, elong, 0.4, label="fraction of d1 ≥ 7.0 Å", color="darkorange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cohorts, rotation=20)
    axes[1].set_ylabel("fraction of equatorial d1 values")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
