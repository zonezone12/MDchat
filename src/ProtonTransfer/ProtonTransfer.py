"""
Proton-transfer analysis for small NH3···HNO3(+H2O) systems.

Loads Extended-XYZ trajectories (CLMD or PIMD ring-polymer stacks), builds
all-pairwise-distance features, the N–H–O collective variable
``delta = d(N-H) - d(O-H)``, and PIMD bead-delocalization metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# Bonding cutoffs (Angstrom) used only for frame-0 atom role detection.
_OH_BOND_CUTOFF = 1.25
_NH_BOND_CUTOFF = 1.25
_NO_BOND_CUTOFF = 1.60


@dataclass
class TrajectoryData:
    """Parsed Extended-XYZ trajectory (CLMD or PIMD)."""

    species: List[str]
    steps: np.ndarray  # (n_frames,)
    centroid: np.ndarray  # (n_frames, n_atoms, 3)
    beads: np.ndarray  # (n_frames, n_beads, n_atoms, 3)
    n_atoms: int
    n_beads: int
    n_frames: int
    path: Path


@dataclass
class PTAtomIndices:
    """Fixed atom indices for the N–H–O proton-transfer CV."""

    n_amine: int
    h_acid: int
    o_acid: int
    water_o: Optional[int] = None
    water_h: Tuple[int, ...] = ()


@dataclass
class DelocalizationResult:
    """Per-frame ring-polymer delocalization of the acid proton."""

    proton_rg: np.ndarray  # (n_frames,)
    delta_bead_std: np.ndarray  # (n_frames,)
    delta_bead_mean: np.ndarray  # (n_frames,)


def _parse_step_from_comment(comment: str) -> int:
    """Extract ``step=K`` from an Extended-XYZ comment line."""
    for token in comment.replace("=", " ").split():
        if token.isdigit():
            # Prefer explicit step= value: look for preceding "step"
            pass
    if "step=" in comment:
        after = comment.split("step=", 1)[1]
        num = ""
        for ch in after:
            if ch.isdigit() or (ch == "-" and not num):
                num += ch
            elif num:
                break
        if num:
            return int(num)
    return -1


def infer_n_physical_atoms(species_block: Sequence[str]) -> int:
    """
    Infer number of physical atoms from a single-frame species list.

    For PIMD, atoms are stacked as [bead0_all, bead1_all, ...]. Physical atom
    count is the unique period of the species pattern (9 for m1n0, 12 for m1n1).
    """
    n = len(species_block)
    # Known physical sizes for this project; fall back to full length (CLMD).
    for candidate in (9, 12):
        if n % candidate == 0 and n // candidate >= 1:
            period = species_block[:candidate]
            if all(
                list(species_block[i * candidate : (i + 1) * candidate]) == list(period)
                for i in range(n // candidate)
            ):
                return candidate
    return n


def load_pimd_xyz(
    path: PathLike,
    *,
    n_physical: Optional[int] = None,
    n_beads: Optional[int] = None,
    max_frames: Optional[int] = None,
) -> TrajectoryData:
    """
    Load an Extended-XYZ trajectory that may stack PIMD ring-polymer beads.

    Parameters
    ----------
    path :
        Path to ``coor.xyz``.
    n_physical :
        Number of physical atoms. Inferred from the first frame if omitted.
    n_beads :
        Number of beads. Inferred as ``n_lines_atoms / n_physical`` if omitted.
        Use ``1`` for classical (CLMD) trajectories.
    max_frames :
        Optional cap on frames to load (for debugging).

    Returns
    -------
    TrajectoryData
        Centroid coordinates are bead-averaged. For CLMD, ``n_beads=1`` and
        ``beads`` has shape ``(n_frames, 1, n_atoms, 3)``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    species: List[str] = []
    steps: List[int] = []
    frame_coords: List[np.ndarray] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        while True:
            if max_frames is not None and len(frame_coords) >= max_frames:
                break
            line = fh.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                n_lines = int(line)
            except ValueError as exc:
                raise ValueError(
                    f"Expected atom count at start of frame in {path}, got: {line!r}"
                ) from exc

            comment = fh.readline()
            if not comment:
                break
            step = _parse_step_from_comment(comment)

            symbols: List[str] = []
            coords = np.empty((n_lines, 3), dtype=np.float64)
            for i in range(n_lines):
                atom_line = fh.readline()
                if not atom_line:
                    raise ValueError(
                        f"Truncated frame at step={step} in {path} "
                        f"(expected {n_lines} atoms, got {i})"
                    )
                parts = atom_line.split()
                if len(parts) < 4:
                    raise ValueError(
                        f"Malformed atom line in {path}: {atom_line!r}"
                    )
                symbols.append(parts[0])
                coords[i, 0] = float(parts[1])
                coords[i, 1] = float(parts[2])
                coords[i, 2] = float(parts[3])

            if not species:
                species = symbols
            elif symbols != species:
                raise ValueError(
                    f"Species order changed at step={step} in {path}"
                )

            steps.append(step)
            frame_coords.append(coords)

    if not frame_coords:
        raise ValueError(f"No frames found in {path}")

    n_lines = frame_coords[0].shape[0]
    if n_physical is None:
        n_physical = infer_n_physical_atoms(species)
    if n_lines % n_physical != 0:
        raise ValueError(
            f"Atom lines ({n_lines}) not divisible by n_physical={n_physical} "
            f"in {path}"
        )
    inferred_beads = n_lines // n_physical
    if n_beads is None:
        n_beads = inferred_beads
    if n_beads != inferred_beads:
        raise ValueError(
            f"n_beads={n_beads} but file has {inferred_beads} beads "
            f"({n_lines}/{n_physical}) in {path}"
        )

    n_frames = len(frame_coords)
    beads = np.empty((n_frames, n_beads, n_physical, 3), dtype=np.float64)
    for fi, coords in enumerate(frame_coords):
        # Layout: [bead0_atoms..., bead1_atoms..., ...]
        beads[fi] = coords.reshape(n_beads, n_physical, 3)

    centroid = beads.mean(axis=1)
    physical_species = species[:n_physical]

    return TrajectoryData(
        species=physical_species,
        steps=np.asarray(steps, dtype=np.int64),
        centroid=centroid,
        beads=beads,
        n_atoms=n_physical,
        n_beads=n_beads,
        n_frames=n_frames,
        path=path,
    )


def detect_pt_atoms(
    species: Sequence[str],
    coords0: np.ndarray,
    *,
    oh_cutoff: float = _OH_BOND_CUTOFF,
    nh_cutoff: float = _NH_BOND_CUTOFF,
    no_cutoff: float = _NO_BOND_CUTOFF,
) -> PTAtomIndices:
    """
    Detect amine N, acid proton H, and acid donor O from frame-0 geometry.

    Logic
    -----
    1. Identify nitrogen atoms and oxygen atoms.
    2. Nitrate N = the N bonded to >= 2 oxygens within ``no_cutoff``.
    3. Acid H = the H bonded (``oh_cutoff``) to a nitrate oxygen (not water O).
    4. Acid O = that oxygen.
    5. Amine N = the other N (bonded to H's).
    6. Optionally locate water O (O not bonded to nitrate N) and its H's.
    """
    species = list(species)
    coords0 = np.asarray(coords0, dtype=np.float64)
    if coords0.ndim != 2 or coords0.shape[1] != 3:
        raise ValueError("coords0 must have shape (n_atoms, 3)")
    if len(species) != coords0.shape[0]:
        raise ValueError("species length must match coords0")

    n_idx = [i for i, s in enumerate(species) if s.upper() == "N"]
    o_idx = [i for i, s in enumerate(species) if s.upper() == "O"]
    h_idx = [i for i, s in enumerate(species) if s.upper() == "H"]

    if len(n_idx) < 2:
        raise ValueError(f"Expected >= 2 N atoms, found {len(n_idx)}")
    if not o_idx or not h_idx:
        raise ValueError("Missing O or H atoms for PT detection")

    # Nitrate N: most O neighbours within no_cutoff
    nitrate_n = None
    nitrate_oxys: List[int] = []
    best_count = -1
    for ni in n_idx:
        bonded_o = [
            oi
            for oi in o_idx
            if float(np.linalg.norm(coords0[ni] - coords0[oi])) <= no_cutoff
        ]
        if len(bonded_o) > best_count:
            best_count = len(bonded_o)
            nitrate_n = ni
            nitrate_oxys = bonded_o

    if nitrate_n is None or best_count < 2:
        raise ValueError(
            f"Could not identify nitrate N (best O-count={best_count})"
        )

    amine_candidates = [ni for ni in n_idx if ni != nitrate_n]
    if not amine_candidates:
        raise ValueError("Could not identify amine N")
    # Prefer N with most bonded H's
    n_amine = max(
        amine_candidates,
        key=lambda ni: sum(
            1
            for hi in h_idx
            if float(np.linalg.norm(coords0[ni] - coords0[hi])) <= nh_cutoff
        ),
    )

    # Acid H: H bonded to a nitrate oxygen
    acid_candidates: List[Tuple[float, int, int]] = []
    for oi in nitrate_oxys:
        for hi in h_idx:
            d = float(np.linalg.norm(coords0[oi] - coords0[hi]))
            if d <= oh_cutoff:
                acid_candidates.append((d, hi, oi))
    if not acid_candidates:
        # Fallback: any H nearest to any nitrate O
        for oi in nitrate_oxys:
            for hi in h_idx:
                d = float(np.linalg.norm(coords0[oi] - coords0[hi]))
                acid_candidates.append((d, hi, oi))
        acid_candidates.sort(key=lambda t: t[0])
        if not acid_candidates or acid_candidates[0][0] > 1.8:
            raise ValueError("Could not identify acid proton near nitrate O")

    acid_candidates.sort(key=lambda t: t[0])
    _, h_acid, o_acid = acid_candidates[0]

    # Water oxygen: O not bonded to nitrate N
    water_o: Optional[int] = None
    water_h: Tuple[int, ...] = ()
    for oi in o_idx:
        if oi in nitrate_oxys:
            continue
        d_to_nitrate = float(np.linalg.norm(coords0[nitrate_n] - coords0[oi]))
        if d_to_nitrate > no_cutoff:
            water_o = oi
            wh = [
                hi
                for hi in h_idx
                if hi != h_acid
                and float(np.linalg.norm(coords0[oi] - coords0[hi])) <= oh_cutoff
            ]
            water_h = tuple(wh)
            break

    return PTAtomIndices(
        n_amine=n_amine,
        h_acid=h_acid,
        o_acid=o_acid,
        water_o=water_o,
        water_h=water_h,
    )


def _pair_label(species: Sequence[str], i: int, j: int) -> str:
    return f"{species[i]}{i}-{species[j]}{j}"


def pairwise_distance_features(
    centroid: np.ndarray,
    species: Sequence[str],
) -> pd.DataFrame:
    """
    All unique pairwise distances for every frame.

    Parameters
    ----------
    centroid : (n_frames, n_atoms, 3)
    species : length n_atoms

    Returns
    -------
    DataFrame
        Columns labeled like ``N0-H8``, one row per frame.
    """
    centroid = np.asarray(centroid, dtype=np.float64)
    if centroid.ndim != 3:
        raise ValueError("centroid must have shape (n_frames, n_atoms, 3)")
    _, n_atoms, _ = centroid.shape
    if len(species) != n_atoms:
        raise ValueError("species length must match n_atoms")

    pairs = np.asarray(list(combinations(range(n_atoms), 2)), dtype=np.int64)
    labels = [_pair_label(species, int(i), int(j)) for i, j in pairs]
    # Vectorized: (n_frames, n_pairs, 3) diffs → distances
    diffs = centroid[:, pairs[:, 0], :] - centroid[:, pairs[:, 1], :]
    data = np.linalg.norm(diffs, axis=2)
    return pd.DataFrame(data, columns=labels)


def proton_transfer_cv(
    centroid: np.ndarray,
    n_amine: int,
    h_acid: int,
    o_acid: int,
) -> pd.DataFrame:
    """
    Compute ``delta = d(N-H) - d(O-H)`` plus the two distances.

    Positive delta => proton closer to acid O (donor); negative => closer to amine N.
    """
    centroid = np.asarray(centroid, dtype=np.float64)
    n_pos = centroid[:, n_amine, :]
    h_pos = centroid[:, h_acid, :]
    o_pos = centroid[:, o_acid, :]
    d_nh = np.linalg.norm(n_pos - h_pos, axis=1)
    d_oh = np.linalg.norm(o_pos - h_pos, axis=1)
    delta = d_nh - d_oh
    return pd.DataFrame(
        {
            "d_NH": d_nh.astype(np.float64),
            "d_OH": d_oh.astype(np.float64),
            "delta": delta.astype(np.float64),
        }
    )


def proton_delocalization(
    beads: np.ndarray,
    n_amine: int,
    h_acid: int,
    o_acid: int,
) -> DelocalizationResult:
    """
    Ring-polymer delocalization of the acid proton.

    For each frame:
    - ``proton_rg``: radius of gyration of the H bead cloud.
    - ``delta_bead_std`` / ``delta_bead_mean``: std/mean of per-bead delta CVs.

    For CLMD (``n_beads=1``), Rg and std are zero.
    """
    beads = np.asarray(beads, dtype=np.float64)
    if beads.ndim != 4:
        raise ValueError("beads must have shape (n_frames, n_beads, n_atoms, 3)")
    n_frames, n_beads, _, _ = beads.shape

    h_beads = beads[:, :, h_acid, :]  # (F, B, 3)
    com = h_beads.mean(axis=1, keepdims=True)
    sq = ((h_beads - com) ** 2).sum(axis=2)  # (F, B)
    proton_rg = np.sqrt(sq.mean(axis=1))

    n_beads_pos = beads[:, :, n_amine, :]
    o_beads_pos = beads[:, :, o_acid, :]
    d_nh = np.linalg.norm(n_beads_pos - h_beads, axis=2)  # (F, B)
    d_oh = np.linalg.norm(o_beads_pos - h_beads, axis=2)
    delta_beads = d_nh - d_oh
    if n_beads == 1:
        delta_std = np.zeros(n_frames, dtype=np.float64)
        delta_mean = delta_beads[:, 0]
        proton_rg = np.zeros(n_frames, dtype=np.float64)
    else:
        delta_std = delta_beads.std(axis=1)
        delta_mean = delta_beads.mean(axis=1)

    return DelocalizationResult(
        proton_rg=proton_rg.astype(np.float64),
        delta_bead_std=delta_std.astype(np.float64),
        delta_bead_mean=delta_mean.astype(np.float64),
    )


def load_ham_dat(path: PathLike) -> pd.DataFrame:
    """
    Load ``ham.dat`` energy/temperature log.

    Columns: step, hamiltonian, temperature, potential, dkinetic, ebath_cent
    """
    path = Path(path)
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            try:
                rows.append([float(x) for x in parts[:6]])
            except ValueError:
                continue
    if not rows:
        return pd.DataFrame(
            columns=[
                "step",
                "hamiltonian",
                "temperature",
                "potential",
                "dkinetic",
                "ebath_cent",
            ]
        )
    arr = np.asarray(rows, dtype=np.float64)
    cols = [
        "step",
        "hamiltonian",
        "temperature",
        "potential",
        "dkinetic",
        "ebath_cent",
    ][: arr.shape[1]]
    df = pd.DataFrame(arr, columns=cols)
    df["step"] = df["step"].astype(np.int64)
    return df


def correlate_features_vs_cv(
    features: pd.DataFrame,
    delta: np.ndarray,
    *,
    exclude_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Pearson and Spearman correlation of each feature column vs the CV delta.

    Returns a DataFrame sorted by descending |Pearson|.
    """
    from scipy import stats

    exclude = set(exclude_cols or ())
    delta = np.asarray(delta, dtype=np.float64)
    rows: List[Dict[str, float]] = []
    for col in features.columns:
        if col in exclude:
            continue
        x = features[col].to_numpy(dtype=np.float64)
        if not np.isfinite(x).all() or np.std(x) < 1e-12:
            continue
        pearson_r, pearson_p = stats.pearsonr(x, delta)
        spearman_r, spearman_p = stats.spearmanr(x, delta)
        rows.append(
            {
                "feature": col,
                "pearson_r": float(pearson_r),
                "pearson_p": float(pearson_p),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
                "abs_pearson": abs(float(pearson_r)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("abs_pearson", ascending=False).reset_index(drop=True)


def count_delta_sign_flips(delta: np.ndarray, *, near_zero: float = 0.05) -> Dict[str, float]:
    """Summarize proton-transfer activity from the delta time series."""
    delta = np.asarray(delta, dtype=np.float64)
    sign = np.sign(delta)
    # Treat near-zero as shared / transition region
    shared = np.abs(delta) < near_zero
    flips = int(np.sum(sign[1:] * sign[:-1] < 0))
    return {
        "n_frames": int(len(delta)),
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta)),
        "delta_min": float(np.min(delta)),
        "delta_max": float(np.max(delta)),
        "frac_on_acid": float(np.mean(delta > near_zero)),
        "frac_on_amine": float(np.mean(delta < -near_zero)),
        "frac_shared": float(np.mean(shared)),
        "n_sign_flips": flips,
        "near_zero_cutoff": float(near_zero),
    }


def zscore_matrix(X: np.ndarray) -> np.ndarray:
    """Column-wise z-score; constant columns become 0."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def build_feature_table(
    traj: TrajectoryData,
    pt: PTAtomIndices,
    *,
    ham: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Assemble the per-frame feature CSV content for one trajectory."""
    feat = pairwise_distance_features(traj.centroid, traj.species)
    cv = proton_transfer_cv(traj.centroid, pt.n_amine, pt.h_acid, pt.o_acid)
    deloc = proton_delocalization(traj.beads, pt.n_amine, pt.h_acid, pt.o_acid)

    out = pd.DataFrame(
        {
            "frame": np.arange(traj.n_frames, dtype=np.int64),
            "step": traj.steps,
            "n_beads": traj.n_beads,
            "n_amine": pt.n_amine,
            "h_acid": pt.h_acid,
            "o_acid": pt.o_acid,
        }
    )
    out = pd.concat([out, cv, feat], axis=1)
    out["proton_rg"] = deloc.proton_rg
    out["delta_bead_std"] = deloc.delta_bead_std
    out["delta_bead_mean"] = deloc.delta_bead_mean

    if ham is not None and not ham.empty and "step" in ham.columns:
        ham_cols = [c for c in ham.columns if c != "step"]
        merged = out.merge(ham[["step"] + ham_cols], on="step", how="left")
        out = merged

    return out


_META_FEATURE_COLS = frozenset(
    {
        "frame",
        "step",
        "n_beads",
        "n_amine",
        "h_acid",
        "o_acid",
        "hamiltonian",
        "temperature",
        "potential",
        "dkinetic",
        "ebath_cent",
    }
)

# Scalar columns summarized per segment (mean + std).
_CV_SUMMARY_COLS = (
    "delta",
    "d_NH",
    "d_OH",
    "proton_rg",
    "delta_bead_std",
    "delta_bead_mean",
)


def pairwise_feature_columns(features_df: pd.DataFrame) -> List[str]:
    """Column names that look like pairwise distance labels (``N0-H8``)."""
    return [
        c
        for c in features_df.columns
        if c not in _META_FEATURE_COLS
        and c not in _CV_SUMMARY_COLS
        and "-" in str(c)
    ]


def build_segment_stats(
    features_df: pd.DataFrame,
    changepoints_df: pd.DataFrame,
    *,
    tag: str,
    signal: str = "pairwise_distances",
    include_pairwise_std: bool = False,
) -> pd.DataFrame:
    """
    Aggregate per-frame features into one row per changepoint segment.

    Parameters
    ----------
    features_df :
        Per-frame table from ``build_feature_table`` / ``features_*.csv``.
    changepoints_df :
        Segment table with columns ``signal``, ``segment``, ``start_frame``,
        ``end_frame``, ``n_frames`` (and optionally ``mean``/``std`` of the
        CPD signal).
    tag :
        Trajectory label, e.g. ``m1n1_PIMD``.
    signal :
        Which CPD signal's segments to use (``pairwise_distances`` or ``delta``).
    include_pairwise_std :
        If True, also store per-segment std of each pairwise distance
        (doubles feature dimensionality).

    Returns
    -------
    DataFrame
        One row per segment with mean (and optional std) feature columns,
        plus CV summaries and ``log10_n_frames``.
    """
    segs = changepoints_df[changepoints_df["signal"] == signal].copy()
    if segs.empty:
        raise ValueError(f"No segments for signal={signal!r} in tag={tag}")

    pair_cols = pairwise_feature_columns(features_df)
    cv_cols = [c for c in _CV_SUMMARY_COLS if c in features_df.columns]

    rows: List[Dict[str, float]] = []
    for _, seg in segs.iterrows():
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        # end_frame is exclusive in ruptures_utils segment_ranges
        block = features_df.iloc[start:end]
        if block.empty:
            continue
        n_frames = int(end - start)
        row: Dict[str, Any] = {
            "traj_id": tag,
            "system": tag.split("_")[0] if "_" in tag else tag,
            "method": tag.split("_", 1)[1] if "_" in tag else "",
            "group": signal,
            "segment_id": int(seg["segment"]),
            "start_frame": start,
            "end_frame": end,
            "n_frames": n_frames,
            "rep_frame": (start + end) // 2,
            "log10_n_frames": float(np.log10(max(n_frames, 1))),
        }
        for col in cv_cols:
            vals = block[col].to_numpy(dtype=np.float64)
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_std"] = float(np.nanstd(vals))
        for col in pair_cols:
            vals = block[col].to_numpy(dtype=np.float64)
            row[f"{col}_mean"] = float(np.nanmean(vals))
            if include_pairwise_std:
                row[f"{col}_std"] = float(np.nanstd(vals))
        # Fraction of frames with delta near zero / on amine (transfer activity)
        if "delta" in block.columns:
            delta = block["delta"].to_numpy(dtype=np.float64)
            row["frac_on_acid"] = float(np.mean(delta > 0.05))
            row["frac_on_amine"] = float(np.mean(delta < -0.05))
            row["frac_shared"] = float(np.mean(np.abs(delta) < 0.05))
        rows.append(row)

    return pd.DataFrame(rows)


def segment_feature_matrix(
    segment_stats: pd.DataFrame,
    *,
    mode: str = "pairwise",
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a numeric matrix for hierarchical clustering of segments.

    Modes
    -----
    pairwise :
        All ``*_mean`` pairwise-distance columns + CV means + log10_n_frames.
    cv :
        Only transfer CV summaries (comparable across m1n0/m1n1).
    """
    if segment_stats.empty:
        return np.empty((0, 0)), []

    if mode == "cv":
        preferred = [
            "delta_mean",
            "delta_std",
            "d_NH_mean",
            "d_OH_mean",
            "proton_rg_mean",
            "delta_bead_std_mean",
            "frac_on_acid",
            "frac_on_amine",
            "frac_shared",
            "log10_n_frames",
        ]
        cols = [c for c in preferred if c in segment_stats.columns]
    elif mode == "pairwise":
        pair_means = sorted(
            c
            for c in segment_stats.columns
            if c.endswith("_mean")
            and "-" in c
            and not c.startswith("delta")
        )
        extra = [
            c
            for c in (
                "delta_mean",
                "delta_std",
                "d_NH_mean",
                "d_OH_mean",
                "proton_rg_mean",
                "log10_n_frames",
            )
            if c in segment_stats.columns
        ]
        cols = pair_means + [c for c in extra if c not in pair_means]
    else:
        raise ValueError(f"Unknown feature mode: {mode!r}")

    if not cols:
        raise ValueError(f"No feature columns available for mode={mode!r}")

    mat = segment_stats[cols].to_numpy(dtype=np.float64)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        finite = col[np.isfinite(col)]
        fill = float(np.median(finite)) if len(finite) else 0.0
        bad = ~np.isfinite(col)
        if bad.any():
            mat[bad, j] = fill
    return mat, cols


__all__ = [
    "TrajectoryData",
    "PTAtomIndices",
    "DelocalizationResult",
    "load_pimd_xyz",
    "infer_n_physical_atoms",
    "detect_pt_atoms",
    "pairwise_distance_features",
    "proton_transfer_cv",
    "proton_delocalization",
    "load_ham_dat",
    "correlate_features_vs_cv",
    "count_delta_sign_flips",
    "zscore_matrix",
    "build_feature_table",
    "pairwise_feature_columns",
    "build_segment_stats",
    "segment_feature_matrix",
]
