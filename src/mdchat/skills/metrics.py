"""Trajectory metrics skills — RMSD, Rg, RMSF, PCA, contacts, strain."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext

_SAVE_CSV_PARAM = Parameter(
    "save_csv", ParamType.BOOLEAN,
    "Write a CSV of the computed data to the session output directory.",
    required=False, default=True,
)

_PLATEAU_DETECT_PARAM = Parameter(
    "detect_plateau", ParamType.BOOLEAN,
    "If True, also run sliding-window plateau detection on this time series "
    "(steady-state region). Writes plateau_detection.csv when save_csv is True.",
    required=False, default=False,
)
_PLATEAU_WINDOW_PARAM = Parameter(
    "plateau_window", ParamType.INTEGER,
    "Plateau: sliding window length (frames) for local standard deviation.",
    required=False, default=50, min_value=2,
)
_PLATEAU_REL_PARAM = Parameter(
    "rel_std_threshold", ParamType.FLOAT,
    "Plateau: tolerance as a fraction of the global std of the series (e.g. 0.12).",
    required=False, default=0.12, min_value=1e-6, max_value=1.0,
)
_PLATEAU_ABS_PARAM = Parameter(
    "abs_std_max", ParamType.FLOAT,
    "Plateau: optional absolute cap on window std (Å for RMSD and Rg). "
    "Effective threshold is max(abs_std_max, rel_std_threshold * global_std).",
    required=False, default=None, min_value=0.0,
)


def _resolve_selection(context: "AnalysisContext", params: dict, key: str = "selection") -> str:
    """Return the user-supplied selection or fall back to context.main_selection."""
    sel = params.get(key)
    if sel is not None:
        return sel
    return context.main_selection


def _maybe_save_csv(
    context: "AnalysisContext",
    params: dict,
    filename: str,
    columns: Dict[str, list],
) -> Dict[str, str]:
    """If save_csv is truthy, write *columns* to a CSV and return artifacts dict."""
    if not params.get("save_csv", True):
        return {}
    import pandas as pd
    path = os.path.join(context.output_dir, filename)
    pd.DataFrame(columns).to_csv(path, index=False)
    return {filename: path}


def _run_observer_pass(context: "AnalysisContext", **params) -> SkillResult:
    from .trajectory_observer_pass import RunTrajectoryObserverPassSkill

    return RunTrajectoryObserverPassSkill().execute(context, **params)


# ---------------------------------------------------------------------------
# RMSD
# ---------------------------------------------------------------------------

class ComputeRMSDSkill(Skill):
    name = "compute_rmsd"
    description = (
        "Compute the Root Mean Square Deviation (RMSD) of selected atoms over "
        "the trajectory relative to a reference frame. RMSD measures how much "
        "the structure deviates from the reference — increasing RMSD indicates "
        "conformational change. Optional detect_plateau finds a steady-state tail "
        "without a separate skill call; use detect_motion_plateau alone to retune "
        "thresholds on cached RMSD."
    )
    category = "metrics"
    parameters = [
        Parameter("selection", ParamType.ATOM_SELECTION,
                  "MDAnalysis atom selection string (e.g., 'protein', 'resname GSA', "
                  "'resname MOF', 'all'). If omitted, uses the session main "
                  "selection (auto-detected from system composition).",
                  required=False, default=None),
        Parameter("ref_frame", ParamType.INTEGER,
                  "Reference frame index for RMSD calculation (0-based).",
                  required=False, default=0, min_value=0),
        _SAVE_CSV_PARAM,
        _PLATEAU_DETECT_PARAM,
        _PLATEAU_WINDOW_PARAM,
        _PLATEAU_REL_PARAM,
        _PLATEAU_ABS_PARAM,
    ]
    requires = ["universe"]
    produces = [
        "rmsd_array", "rmsd_summary",
        "plateau_detection", "plateau_start_frame", "steady_state_representative_frame",
    ]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        from ..plateau_helpers import apply_plateau_to_metric_compute

        u = context.universe
        sel_str = _resolve_selection(context, params)
        ref_frame = params.get("ref_frame", 0)

        pass_res = _run_observer_pass(
            context,
            include_rmsd=True,
            rmsd_selection=sel_str,
            rmsd_ref_frame=ref_frame,
            save_csv=False,
        )
        if not pass_res.success:
            return pass_res
        rmsd = context.get("rmsd_array")

        context.set("rmsd_array", rmsd)

        mean_rmsd = float(np.nanmean(rmsd))
        max_rmsd = float(np.nanmax(rmsd))
        final_rmsd = float(rmsd[-1]) if len(rmsd) > 0 else 0.0

        summary_text = (
            f"RMSD computed for selection '{sel_str}' ({len(rmsd)} frames). "
            f"Mean: {mean_rmsd:.2f} A, Max: {max_rmsd:.2f} A, "
            f"Final: {final_rmsd:.2f} A. "
        )
        if max_rmsd > 5.0:
            summary_text += "Large deviations detected — significant conformational change."
        elif max_rmsd < 2.0:
            summary_text += "Structure remains relatively stable throughout."
        else:
            summary_text += "Moderate conformational changes observed."

        context.set("rmsd_summary", summary_text)

        artifacts = _maybe_save_csv(context, params, "rmsd.csv", {
            "frame": list(range(len(rmsd))),
            "rmsd": rmsd.tolist(),
        })

        plateau_addon, plateau_data, plateau_art = apply_plateau_to_metric_compute(
            context,
            params,
            universe=u,
            y=rmsd,
            metric="rmsd",
            selection=sel_str,
            compute_label="RMSD",
            save_csv=bool(params.get("save_csv", True)),
            csv_filename="plateau_detection.csv",
        )
        summary_text = summary_text + plateau_addon
        artifacts.update(plateau_art)

        data: Dict = {
            "rmsd_array": rmsd,
            "rmsd_summary": summary_text,
        }
        data.update(plateau_data)

        return SkillResult(
            success=True,
            data=data,
            artifacts=artifacts,
            summary=summary_text,
        )


# ---------------------------------------------------------------------------
# Radius of Gyration
# ---------------------------------------------------------------------------

class ComputeRgSkill(Skill):
    name = "compute_rg"
    description = (
        "Compute the radius of gyration (Rg) over the trajectory. "
        "Rg measures the compactness of the structure — decreasing Rg "
        "indicates compaction, increasing Rg indicates expansion. "
        "Works for any molecular system (proteins, cages, MOFs, etc.). "
        "Optional detect_plateau finds a steady-state tail on Rg; use "
        "detect_motion_plateau alone to retune on cached Rg."
    )
    category = "metrics"
    parameters = [
        Parameter("selection", ParamType.ATOM_SELECTION,
                  "MDAnalysis atom selection (e.g., 'protein', 'resname GSA', "
                  "'resname MOF', 'all'). If omitted, uses the session main "
                  "selection.",
                  required=False, default=None),
        _SAVE_CSV_PARAM,
        _PLATEAU_DETECT_PARAM,
        _PLATEAU_WINDOW_PARAM,
        _PLATEAU_REL_PARAM,
        _PLATEAU_ABS_PARAM,
    ]
    requires = ["universe"]
    produces = [
        "rg_array",
        "plateau_detection", "plateau_start_frame", "steady_state_representative_frame",
    ]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        from ..plateau_helpers import apply_plateau_to_metric_compute

        u = context.universe
        sel_str = _resolve_selection(context, params)

        pass_res = _run_observer_pass(
            context,
            include_rg=True,
            rg_selection=sel_str,
            save_csv=False,
        )
        if not pass_res.success:
            return pass_res
        rg = context.get("rg_array")

        context.set("rg_array", rg)

        mean_rg = float(np.nanmean(rg))
        std_rg = float(np.nanstd(rg))
        min_rg = float(np.nanmin(rg))
        max_rg = float(np.nanmax(rg))

        summary = (
            f"Rg computed for '{sel_str}' ({len(rg)} frames). "
            f"Mean: {mean_rg:.2f} A, Std: {std_rg:.2f} A, "
            f"Range: {min_rg:.2f}–{max_rg:.2f} A. "
        )
        rel_fluct = std_rg / mean_rg if mean_rg > 0 else 0
        if rel_fluct < 0.02:
            summary += "Very stable compactness — no significant size changes."
        elif rel_fluct < 0.05:
            summary += "Modest fluctuations in molecular size."
        else:
            summary += "Significant size changes — possible (un)folding or large-scale motion."

        artifacts = _maybe_save_csv(context, params, "rg.csv", {
            "frame": list(range(len(rg))),
            "rg": rg.tolist(),
        })

        plateau_addon, plateau_data, plateau_art = apply_plateau_to_metric_compute(
            context,
            params,
            universe=u,
            y=rg,
            metric="rg",
            selection=sel_str,
            compute_label="Rg",
            save_csv=bool(params.get("save_csv", True)),
            csv_filename="plateau_detection.csv",
        )
        summary = summary + plateau_addon
        artifacts.update(plateau_art)

        data: Dict = {"rg_array": rg}
        data.update(plateau_data)

        return SkillResult(
            success=True,
            data=data,
            artifacts=artifacts,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# RMSF (per-atom fluctuation)
# ---------------------------------------------------------------------------

class ComputeRMSFSkill(Skill):
    name = "compute_rmsf"
    description = (
        "Compute per-atom Root Mean Square Fluctuation (RMSF) over the "
        "trajectory. RMSF reveals which atoms or residues are most flexible "
        "versus rigid. Works for any system — for proteins use 'name CA' or "
        "'backbone'; for MOFs/cages use the appropriate atom selection."
    )
    category = "metrics"
    parameters = [
        Parameter("selection", ParamType.ATOM_SELECTION,
                  "MDAnalysis atom selection (e.g., 'name CA' for protein "
                  "per-residue, 'backbone', 'resname GSA and name C*', 'all'). "
                  "If omitted, uses the session main selection.",
                  required=False, default=None),
        _SAVE_CSV_PARAM,
    ]
    requires = ["universe"]
    produces = ["rmsf_array", "rmsf_atom_info"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        sel_str = _resolve_selection(context, params)

        pass_res = _run_observer_pass(
            context,
            include_rmsf=True,
            rmsf_selection=sel_str,
            save_csv=False,
        )
        if not pass_res.success:
            return pass_res
        rmsf = context.get("rmsf_array")

        context.set("rmsf_array", rmsf)

        atom_info = context.get("rmsf_atom_info")
        context.set("rmsf_atom_info", atom_info)

        mean_rmsf = float(np.nanmean(rmsf))
        max_rmsf = float(np.nanmax(rmsf))
        max_idx = int(np.nanargmax(rmsf))
        max_atom = atom_info[max_idx] if max_idx < len(atom_info) else {}

        top_n = min(5, len(rmsf))
        top_idx = np.argsort(rmsf)[::-1][:top_n]
        top_lines = []
        for idx in top_idx:
            ai = atom_info[idx]
            top_lines.append(
                f"  {ai['resname']} {ai['resid']}: {rmsf[idx]:.2f} A"
            )

        summary = (
            f"RMSF computed for '{sel_str}' ({len(rmsf)} atoms). "
            f"Mean: {mean_rmsf:.2f} A, Max: {max_rmsf:.2f} A "
            f"({max_atom.get('resname', '?')} {max_atom.get('resid', '?')}).\n"
            f"Top {top_n} most flexible:\n" + "\n".join(top_lines)
        )

        artifacts = _maybe_save_csv(context, params, "rmsf.csv", {
            "resname": [a["resname"] for a in atom_info],
            "resid": [a["resid"] for a in atom_info],
            "atom_name": [a["name"] for a in atom_info],
            "rmsf": rmsf.tolist(),
        })

        return SkillResult(
            success=True,
            data={"rmsf_array": rmsf, "rmsf_atom_info": atom_info},
            artifacts=artifacts,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# PCA on fluctuations
# ---------------------------------------------------------------------------

class ComputePCASkill(Skill):
    name = "compute_pca"
    description = (
        "Perform Principal Component Analysis (PCA) on atomic fluctuations "
        "to identify the dominant modes of motion. Returns PC scores per "
        "frame and the variance explained by each component. "
        "Works for any molecular system — use an appropriate atom selection "
        "to reduce dimensionality."
    )
    category = "metrics"
    parameters = [
        Parameter("selection", ParamType.ATOM_SELECTION,
                  "MDAnalysis atom selection for PCA (e.g., 'name CA' for "
                  "proteins, 'resname GSA and not name H*' for cages, 'all'). "
                  "If omitted, uses the session main selection.",
                  required=False, default=None),
        Parameter("n_components", ParamType.INTEGER,
                  "Number of principal components to compute.",
                  required=False, default=5, min_value=1, max_value=50),
        _SAVE_CSV_PARAM,
    ]
    requires = ["universe"]
    produces = ["pca_scores", "pca_variance_explained"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        sel_str = _resolve_selection(context, params)
        n_comp = params.get("n_components", 5)

        pass_res = _run_observer_pass(
            context,
            include_pca=True,
            pca_selection=sel_str,
            pca_n_components=n_comp,
            save_csv=False,
        )
        if not pass_res.success:
            return pass_res
        pcs = context.get("pca_scores")
        var_explained = context.get("pca_variance_explained")
        context.set("pca_scores", pcs)
        context.set("pca_variance_explained", var_explained)

        total_var = float(np.sum(var_explained)) * 100
        pc_lines = []
        for i, v in enumerate(var_explained):
            pc_lines.append(f"  PC{i+1}: {v*100:.1f}%")

        summary = (
            f"PCA computed for '{sel_str}' ({n_comp} components, "
            f"{pcs.shape[0]} frames).\n"
            f"Total variance explained: {total_var:.1f}%\n"
            + "\n".join(pc_lines)
        )

        cols = {"frame": list(range(pcs.shape[0]))}
        for i in range(pcs.shape[1]):
            cols[f"PC{i+1}"] = pcs[:, i].tolist()
        cols["variance_explained"] = [None] * pcs.shape[0]
        for i, v in enumerate(var_explained):
            cols["variance_explained"][i] = float(v)
        artifacts = _maybe_save_csv(context, params, "pca.csv", cols)

        return SkillResult(
            success=True,
            data={
                "pca_scores": pcs,
                "pca_variance_explained": var_explained,
            },
            artifacts=artifacts,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Contact distances
# ---------------------------------------------------------------------------

class ComputeContactsSkill(Skill):
    name = "compute_contacts"
    description = (
        "Compute the minimum distance between two atom selections over all "
        "trajectory frames. Useful for tracking ligand–protein contacts, "
        "guest–host distances, ion–cage proximity, or any pairwise interaction."
    )
    category = "metrics"
    parameters = [
        Parameter("selection_a", ParamType.ATOM_SELECTION,
                  "First atom selection (e.g., 'resname ATP', 'resid 14', "
                  "'name I', 'resname LIG')."),
        Parameter("selection_b", ParamType.ATOM_SELECTION,
                  "Second atom selection (e.g., 'resname MG', 'protein', "
                  "'resname GSA', 'resname MOF')."),
        Parameter("label", ParamType.STRING,
                  "Label for this contact pair (used as context key suffix).",
                  required=False, default="contact"),
        _SAVE_CSV_PARAM,
    ]
    requires = ["universe"]
    produces = ["contact_distances"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        sel_a = params["selection_a"]
        sel_b = params["selection_b"]
        label = params.get("label", "contact")

        pass_res = _run_observer_pass(
            context,
            include_contacts=True,
            contact_selection_a=sel_a,
            contact_selection_b=sel_b,
            contact_label=label,
            save_csv=False,
        )
        if not pass_res.success:
            return pass_res
        dists = context.get("contact_distances")

        key = f"contact_distances_{label}"
        context.set(key, dists)
        context.set("contact_distances", dists)

        mean_d = float(np.nanmean(dists))
        std_d = float(np.nanstd(dists))
        min_d = float(np.nanmin(dists))
        max_d = float(np.nanmax(dists))

        summary = (
            f"Min-distance '{sel_a}' ↔ '{sel_b}' ({len(dists)} frames): "
            f"mean={mean_d:.2f} A, std={std_d:.2f} A, "
            f"range={min_d:.2f}–{max_d:.2f} A. "
        )
        if min_d < 3.5:
            summary += "Close contact detected (< 3.5 A) — possible direct coordination."
        elif mean_d < 6.0:
            summary += "Persistent proximity — likely a stable interaction."
        else:
            summary += "Relatively distant — transient or no direct contact."

        artifacts = _maybe_save_csv(
            context, params, f"contacts_{label}.csv", {
                "frame": list(range(len(dists))),
                "min_distance": dists.tolist(),
            },
        )

        return SkillResult(
            success=True,
            data={"contact_distances": dists},
            artifacts=artifacts,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Strain proxy
# ---------------------------------------------------------------------------

class ComputeStrainSkill(Skill):
    name = "compute_strain"
    description = (
        "Compute a local affine strain proxy over the trajectory. "
        "This metric captures local deformation intensity — spikes "
        "indicate sudden structural rearrangements. Works for any "
        "molecular system (proteins, cages, frameworks, etc.)."
    )
    category = "metrics"
    parameters = [
        Parameter("selection", ParamType.ATOM_SELECTION,
                  "MDAnalysis atom selection for strain calculation "
                  "(e.g., 'name CA' for proteins, 'resname GSA', 'all'). "
                  "If omitted, uses the session main selection.",
                  required=False, default=None),
        Parameter("window", ParamType.INTEGER,
                  "Sliding window size (frames).",
                  required=False, default=10, min_value=2),
        Parameter("lag", ParamType.INTEGER,
                  "Frame lag for deformation comparison.",
                  required=False, default=1, min_value=1),
        _SAVE_CSV_PARAM,
    ]
    requires = ["universe"]
    produces = ["strain_array"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        from src.TrajectoryMetrics import TrajectoryMetrics

        u = context.universe
        sel_str = _resolve_selection(context, params)
        window = params.get("window", 10)
        lag = params.get("lag", 1)

        tm = TrajectoryMetrics()
        strain = tm.local_affine_strain_proxy(u, sel_str, window=window, lag=lag)

        context.set("strain_array", strain)

        valid = strain[~np.isnan(strain)]
        if len(valid) == 0:
            return SkillResult(
                success=True,
                data={"strain_array": strain},
                summary="Strain computed but all values are NaN (trajectory too short for window).",
            )

        mean_s = float(np.nanmean(valid))
        max_s = float(np.nanmax(valid))
        max_idx = int(np.nanargmax(strain))

        summary = (
            f"Strain proxy computed for '{sel_str}' (window={window}, lag={lag}). "
            f"Mean: {mean_s:.4f}, Max: {max_s:.4f} at frame {max_idx}. "
        )
        if max_s > 0.1:
            summary += "Major structural rearrangement detected."
        elif max_s > 0.01:
            summary += "Moderate deformation events present."
        else:
            summary += "Low strain — structurally quiescent trajectory."

        artifacts = _maybe_save_csv(context, params, "strain.csv", {
            "frame": list(range(len(strain))),
            "strain": strain.tolist(),
        })

        return SkillResult(
            success=True,
            data={"strain_array": strain},
            artifacts=artifacts,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

_registry = get_default_registry()
_registry.register(ComputeRMSDSkill())
_registry.register(ComputeRgSkill())
_registry.register(ComputeRMSFSkill())
_registry.register(ComputePCASkill())
_registry.register(ComputeContactsSkill())
_registry.register(ComputeStrainSkill())
