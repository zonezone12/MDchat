"""Pre-clustering assembly QC: flag replicas in which the nanocube comes apart.

A trajectory fails when any frame has ``assembly_rg`` above ``rg_max`` or a
``largest_connected_component_size`` below ``min_component_size``. The
thresholds are fixed before clustering and applied identically to every
cohort. Failing replicas are dropped from segment clustering and from the
metastructure comparison, not from changepoint detection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .detection import canonicalize_traj_id
from .feature_groups import traj_id_from_features_stem
from .gsa_cohort_run import cube_from_features_name

ASSEMBLY_QC_RG_MAX = 15.0
ASSEMBLY_QC_MIN_COMPONENT_SIZE = 6

ASSEMBLY_QC_COLUMNS = ("assembly_rg", "largest_connected_component_size")

_RESULT_COLUMNS = (
    "traj_id",
    "cube",
    "n_frames",
    "assembly_rg_max",
    "min_component_size",
    "n_frames_rg_fail",
    "n_frames_component_fail",
    "first_fail_frame",
    "qc_pass",
)


def scan_assembly_qc(
    csv_paths: Iterable[Path | str],
    *,
    rg_max: float = ASSEMBLY_QC_RG_MAX,
    min_component_size: int = ASSEMBLY_QC_MIN_COMPONENT_SIZE,
) -> pd.DataFrame:
    """One row per ``*_gsa_features.csv`` with Rg / component extremes and ``qc_pass``."""
    rows: list[dict] = []
    for path in sorted(Path(p) for p in csv_paths):
        header = pd.read_csv(path, nrows=0).columns
        missing = [c for c in ASSEMBLY_QC_COLUMNS if c not in header]
        if missing:
            raise KeyError(f"{path.name} lacks assembly QC column(s): {missing}")
        usecols = [c for c in ("traj_id", "frame", *ASSEMBLY_QC_COLUMNS) if c in header]
        df = pd.read_csv(path, usecols=usecols)

        if "traj_id" in df.columns and len(df):
            traj_id = str(df["traj_id"].iloc[0])
        else:
            traj_id = traj_id_from_features_stem(path.stem)
        frames = df["frame"].to_numpy() if "frame" in df.columns else np.arange(len(df))
        rg = pd.to_numeric(df["assembly_rg"], errors="coerce").to_numpy(dtype=float)
        lcc = pd.to_numeric(
            df["largest_connected_component_size"], errors="coerce"
        ).to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            rg_fail = rg > rg_max
            lcc_fail = lcc < min_component_size
        fail = rg_fail | lcc_fail

        rows.append(
            {
                "traj_id": traj_id,
                "cube": cube_from_features_name(path.name) or "",
                "n_frames": int(len(df)),
                "assembly_rg_max": float(np.nanmax(rg)) if np.isfinite(rg).any() else np.nan,
                "min_component_size": (
                    float(np.nanmin(lcc)) if np.isfinite(lcc).any() else np.nan
                ),
                "n_frames_rg_fail": int(rg_fail.sum()),
                "n_frames_component_fail": int(lcc_fail.sum()),
                "first_fail_frame": int(frames[fail][0]) if fail.any() else pd.NA,
                "qc_pass": not bool(fail.any()),
            }
        )

    out = pd.DataFrame(rows, columns=list(_RESULT_COLUMNS))
    out["first_fail_frame"] = out["first_fail_frame"].astype("Int64")
    out["qc_pass"] = out["qc_pass"].astype(bool)
    return out


def scan_assembly_qc_dir(
    gsa_dir: Path | str,
    *,
    rg_max: float = ASSEMBLY_QC_RG_MAX,
    min_component_size: int = ASSEMBLY_QC_MIN_COMPONENT_SIZE,
    suffix: str = "_gsa_features.csv",
) -> pd.DataFrame:
    """:func:`scan_assembly_qc` over one cube's feature folder.

    Seeds repeat across cubes and exclusions match on the seed, so a folder
    holding more than one cube is rejected.
    """
    paths = sorted(Path(gsa_dir).glob(f"*{suffix}"))
    if not paths:
        raise FileNotFoundError(f"No *{suffix} under {gsa_dir}")
    qc = scan_assembly_qc(paths, rg_max=rg_max, min_component_size=min_component_size)
    cubes = sorted(set(qc["cube"]) - {""})
    if len(cubes) > 1:
        raise ValueError(f"{gsa_dir} mixes cubes {cubes}; use a single cube's folder")
    return qc


def assembly_qc_failures(qc: pd.DataFrame) -> tuple[str, ...]:
    """Trajectory IDs that failed :func:`scan_assembly_qc`."""
    if qc.empty:
        return ()
    return tuple(qc.loc[~qc["qc_pass"], "traj_id"].astype(str))


def drop_trajectories(df: pd.DataFrame, traj_ids: Sequence[str]) -> pd.DataFrame:
    """Drop rows of a single-cohort table whose ``traj_id`` is in ``traj_ids``.

    IDs are compared with the cube prefix stripped, so ``BMHpH_563849_mdcrd_v``
    matches ``563849_mdcrd_v``. Seeds repeat across cubes, so do not pass a
    table that mixes cohorts.
    """
    drop = {canonicalize_traj_id(t) for t in traj_ids}
    if not drop or df.empty:
        return df
    keep = ~df["traj_id"].astype(str).map(canonicalize_traj_id).isin(drop)
    return df.loc[keep].reset_index(drop=True)
