"""
Optional single-pass trajectory analysis: register observers on TrajectoryIterator
and iterate once (metrics + endpoint distances + GSA nanocube metrics).

Supports parallel iteration (multiprocessing, fork on Linux, or Dask when use_dask=True).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


def _resolve_sel(context: "AnalysisContext", explicit: Optional[str]) -> str:
    if explicit is not None and str(explicit).strip():
        return explicit
    return context.main_selection


def _slice_iter_count(n_total: int, start: Optional[int], stop: Optional[int], step: Optional[int]) -> int:
    s = slice(start, stop, step)
    return len(range(*s.indices(n_total)))


class RunTrajectoryObserverPassSkill(Skill):
    name = "run_trajectory_observer_pass"
    description = (
        "OPTIONAL efficiency path: run one TrajectoryIterator pass with multiple "
        "FrameObservers — standard metrics (RMSD, Rg, contacts), pairwise endpoint "
        "distances, and/or GSA nanocube geometry (GSAnalyzerObserver) — so the "
        "trajectory is read once. Use n_jobs>1 or n_jobs=-1 with optional use_dask "
        "for parallel frame batches (same merge rules as GSAnalyzer.gsa_nanocube_metrics). "
        "Individual compute_* / gsa_nanocube_metrics skills remain for one-off use."
    )
    category = "trajectory"
    parameters = [
        Parameter(
            "include_rmsd",
            ParamType.BOOLEAN,
            "Include RMSD vs ref_frame for rmsd_selection.",
            required=False,
            default=False,
        ),
        Parameter(
            "rmsd_selection",
            ParamType.ATOM_SELECTION,
            "Atom selection for RMSD. If omitted, uses session main_selection.",
            required=False,
            default=None,
        ),
        Parameter(
            "rmsd_ref_frame",
            ParamType.INTEGER,
            "Reference frame index (0-based) for RMSD.",
            required=False,
            default=0,
            min_value=0,
        ),
        Parameter(
            "include_rg",
            ParamType.BOOLEAN,
            "Include radius of gyration for rg_selection.",
            required=False,
            default=False,
        ),
        Parameter(
            "rg_selection",
            ParamType.ATOM_SELECTION,
            "Atom selection for Rg. If omitted, uses session main_selection.",
            required=False,
            default=None,
        ),
        Parameter(
            "include_contacts",
            ParamType.BOOLEAN,
            "Include minimum distance between two selections each frame.",
            required=False,
            default=False,
        ),
        Parameter(
            "contact_selection_a",
            ParamType.ATOM_SELECTION,
            "First selection for contacts (required if include_contacts).",
            required=False,
            default=None,
        ),
        Parameter(
            "contact_selection_b",
            ParamType.ATOM_SELECTION,
            "Second selection for contacts (required if include_contacts).",
            required=False,
            default=None,
        ),
        Parameter(
            "contact_label",
            ParamType.STRING,
            "Suffix for context keys (contact_distances_<label>).",
            required=False,
            default="contact",
        ),
        Parameter(
            "endpoint_residue_selections",
            ParamType.ARRAY,
            "Optional list of residue selections for pairwise endpoint distances "
            "(same as compute_endpoint_distances). Empty = skip endpoints.",
            required=False,
            items_type=ParamType.STRING,
            default=None,
        ),
        Parameter(
            "include_gsa",
            ParamType.BOOLEAN,
            "Include GSA nanocube metrics via GSAnalyzerObserver (faces, volume, guest).",
            required=False,
            default=False,
        ),
        Parameter(
            "gsa_face_selections",
            ParamType.ARRAY,
            "Cube face selection strings when include_gsa is true "
            "(e.g. six resid lists). Required if include_gsa.",
            required=False,
            items_type=ParamType.STRING,
            default=None,
        ),
        Parameter(
            "gsa_corner_selections",
            ParamType.ARRAY,
            "Optional corner selections for GSAnalyzerObserver.",
            required=False,
            items_type=ParamType.STRING,
            default=None,
        ),
        Parameter(
            "gsa_guest_selection",
            ParamType.ATOM_SELECTION,
            "Optional guest atom selection for GSA tracking.",
            required=False,
            default=None,
        ),
        Parameter(
            "gsa_guest_tracking_method",
            ParamType.STRING,
            "Guest inside/outside: 'distance' or 'volume'.",
            required=False,
            default="distance",
        ),
        Parameter(
            "gsa_guest_distance_threshold",
            ParamType.FLOAT,
            "Optional distance threshold (Å) for guest distance method; "
            "auto if omitted.",
            required=False,
            default=None,
            min_value=0.0,
        ),
        Parameter(
            "n_jobs",
            ParamType.INTEGER,
            "1 = sequential iterator. >1 or -1 = parallel frame batches "
            "(requires picklable observers).",
            required=False,
            default=1,
        ),
        Parameter(
            "use_dask",
            ParamType.BOOLEAN,
            "If true and n_jobs!=1, use Dask Distributed for parallel batches; "
            "otherwise multiprocessing (or fork on Linux).",
            required=False,
            default=False,
        ),
        Parameter(
            "parallel_max_workers_for_io",
            ParamType.INTEGER,
            "Cap parallel workers for I/O-bound trajectories (passed to "
            "TrajectoryIterator.iterate as max_workers_for_io).",
            required=False,
            default=None,
            min_value=1,
        ),
        Parameter(
            "start_frame",
            ParamType.INTEGER,
            "First trajectory index (Python slice start; default: 0).",
            required=False,
            default=None,
            min_value=0,
        ),
        Parameter(
            "stop_frame",
            ParamType.INTEGER,
            "Stop index (exclusive; default: end of trajectory).",
            required=False,
            default=None,
            min_value=0,
        ),
        Parameter(
            "step",
            ParamType.INTEGER,
            "Frame stride (default: 1).",
            required=False,
            default=None,
            min_value=1,
        ),
        Parameter(
            "save_csv",
            ParamType.BOOLEAN,
            "Write CSVs for each computed series into the session output directory.",
            required=False,
            default=True,
        ),
    ]
    requires = ["universe"]
    produces = [
        "observer_pass_results",
        "rmsd_array",
        "rg_array",
        "contact_distances",
        "endpoint_distances",
        "residue_selections",
        "cube_metrics_df",
        "volume_array",
    ]

    def execute(self, context: "AnalysisContext", **params) -> SkillResult:
        import numpy as np
        import pandas as pd
        from src.EndpointAnalyzer import EndpointAnalyzerObserver
        from src.task import GSAnalyzerObserver
        from src.TrajectoryIterator import TrajectoryIterator
        from src.TrajectoryMetrics import MetricPassSpec, StackedMetricsObserver

        u = context.universe
        n_total = len(u.trajectory)

        include_rmsd = bool(params.get("include_rmsd", False))
        include_rg = bool(params.get("include_rg", False))
        include_contacts = bool(params.get("include_contacts", False))
        include_gsa = bool(params.get("include_gsa", False))
        ep_list: Optional[List[str]] = params.get("endpoint_residue_selections")
        if ep_list is None:
            ep_list = []

        gsa_faces: List[str] = list(params.get("gsa_face_selections") or [])

        if include_gsa and len(gsa_faces) < 1:
            return SkillResult(
                success=False,
                error="include_gsa requires a non-empty gsa_face_selections list.",
                summary="Provide gsa_face_selections for GSA nanocube metrics.",
            )

        if not (
            include_rmsd
            or include_rg
            or include_contacts
            or len(ep_list) >= 2
            or include_gsa
        ):
            return SkillResult(
                success=False,
                error="Nothing to compute: enable at least one of include_rmsd, "
                "include_rg, include_contacts, include_gsa, or provide at least two "
                "endpoint_residue_selections for pairwise distances.",
                summary="Observer pass needs at least one analysis block.",
            )

        if include_contacts:
            ca = params.get("contact_selection_a")
            cb = params.get("contact_selection_b")
            if not ca or not cb:
                return SkillResult(
                    success=False,
                    error="include_contacts requires contact_selection_a and "
                    "contact_selection_b.",
                    summary="Contact selections missing.",
                )

        start = params.get("start_frame")
        stop = params.get("stop_frame")
        step = params.get("step")
        n_iter = _slice_iter_count(n_total, start, stop, step)
        if n_iter < 1:
            return SkillResult(
                success=False,
                error="Invalid frame slice: zero frames would be visited.",
                summary="Adjust start_frame / stop_frame / step.",
            )

        specs: List[MetricPassSpec] = []
        if include_rmsd:
            sel = _resolve_sel(context, params.get("rmsd_selection"))
            rf = int(params.get("rmsd_ref_frame", 0))
            specs.append(
                MetricPassSpec(
                    result_key="rmsd",
                    kind="rmsd",
                    selection=sel,
                    ref_frame=rf,
                )
            )
        if include_rg:
            sel = _resolve_sel(context, params.get("rg_selection"))
            specs.append(
                MetricPassSpec(result_key="rg", kind="rg", selection=sel)
            )
        contact_label = params.get("contact_label", "contact") or "contact"
        if include_contacts:
            specs.append(
                MetricPassSpec(
                    result_key=f"contacts_{contact_label}",
                    kind="contacts",
                    selection=params["contact_selection_a"],
                    selection_b=params["contact_selection_b"],
                )
            )

        raw_nj = params.get("n_jobs")
        n_jobs = 1 if raw_nj is None else int(raw_nj)
        use_dask = bool(params.get("use_dask", False))
        use_parallel = n_jobs != 1
        iterator = TrajectoryIterator(u, use_dask=(use_dask and use_parallel))

        metrics_observer: Optional[StackedMetricsObserver] = None
        if specs:
            metrics_observer = StackedMetricsObserver(specs, n_frames=n_iter)
            iterator.subscribe(metrics_observer)

        endpoint_obs: Optional[EndpointAnalyzerObserver] = None
        if len(ep_list) >= 2:
            endpoint_obs = EndpointAnalyzerObserver(
                ep_list,
                n_frame_rows=n_iter,
            )
            iterator.subscribe(endpoint_obs)

        gsa_obs: Optional[GSAnalyzerObserver] = None
        save_csv = bool(params.get("save_csv", True))
        if include_gsa:
            gsa_prefix = (
                os.path.join(context.output_dir, "gsa_nanocube_observer_pass")
                if save_csv
                else None
            )
            corner = params.get("gsa_corner_selections")
            gsa_obs = GSAnalyzerObserver(
                face_sel_list=gsa_faces,
                corner_sel_list=list(corner) if corner else None,
                guest_sel=params.get("gsa_guest_selection"),
                out_prefix=gsa_prefix,
                guest_tracking_method=params.get(
                    "gsa_guest_tracking_method", "distance"
                ),
                guest_distance_threshold=params.get("gsa_guest_distance_threshold"),
            )
            iterator.subscribe(gsa_obs)

        sl = slice(start, stop, step)
        iter_kw: Dict[str, Any] = {
            "start": sl.start,
            "stop": sl.stop,
            "step": sl.step,
            "n_jobs": n_jobs,
        }
        mw = params.get("parallel_max_workers_for_io")
        if mw is not None:
            iter_kw["max_workers_for_io"] = int(mw)
        iterator.iterate(**iter_kw)

        pass_results: Dict[str, Any] = {
            "n_frames_iterated": n_iter,
            "metrics": {},
            "n_jobs": n_jobs,
            "use_dask": use_dask and use_parallel,
        }

        summary_parts: List[str] = [
            f"Iterator finished ({n_iter} frame(s) visited, n_jobs={n_jobs})."
        ]

        artifacts: Dict[str, str] = {}

        if metrics_observer is not None:
            for spec in specs:
                arr = metrics_observer.results.get(spec.result_key)
                if arr is None:
                    continue
                pass_results["metrics"][spec.result_key] = arr
                if spec.kind == "rmsd":
                    context.set("rmsd_array", arr)
                    mean_r = float(np.nanmean(arr))
                    summary_parts.append(
                        f"RMSD: mean={mean_r:.2f} Å over {len(arr)} frames."
                    )
                    if save_csv:
                        p = os.path.join(context.output_dir, "observer_pass_rmsd.csv")
                        pd.DataFrame(
                            {"frame_index": np.arange(len(arr)), "rmsd": arr}
                        ).to_csv(p, index=False)
                        artifacts["observer_pass_rmsd.csv"] = p
                elif spec.kind == "rg":
                    context.set("rg_array", arr)
                    mean_g = float(np.nanmean(arr))
                    summary_parts.append(
                        f"Rg: mean={mean_g:.2f} Å over {len(arr)} frames."
                    )
                    if save_csv:
                        p = os.path.join(context.output_dir, "observer_pass_rg.csv")
                        pd.DataFrame(
                            {"frame_index": np.arange(len(arr)), "rg": arr}
                        ).to_csv(p, index=False)
                        artifacts["observer_pass_rg.csv"] = p
                elif spec.kind == "contacts":
                    key = f"contact_distances_{contact_label}"
                    context.set(key, arr)
                    context.set("contact_distances", arr)
                    mean_d = float(np.nanmean(arr))
                    summary_parts.append(
                        f"Contacts ({contact_label}): mean min-distance={mean_d:.2f} Å."
                    )
                    if save_csv:
                        p = os.path.join(
                            context.output_dir,
                            f"observer_pass_contacts_{contact_label}.csv",
                        )
                        pd.DataFrame(
                            {
                                "frame_index": np.arange(len(arr)),
                                "min_distance": arr,
                            }
                        ).to_csv(p, index=False)
                        artifacts[f"observer_pass_contacts_{contact_label}.csv"] = p

        if endpoint_obs is not None:
            ep_dict = {
                "all_pairs": endpoint_obs.all_pairs,
                "n_residues": endpoint_obs.n_res,
                "n_frames": n_iter,
            }
            context.set("endpoint_distances", ep_dict)
            context.set("residue_selections", ep_list)
            pass_results["endpoint_distances"] = ep_dict
            n_pair_keys = len(ep_dict.get("all_pairs", {}))
            summary_parts.append(
                f"Endpoint distances: {n_pair_keys} pair tensors for "
                f"{ep_dict['n_residues']} residues."
            )

        if gsa_obs is not None:
            df = gsa_obs.get_metrics_df()
            pass_results["gsa_nanocube_metrics_shape"] = getattr(
                df, "shape", (0, 0)
            )
            context.set("cube_metrics_df", df)
            if df is not None and len(df) > 0 and "volume" in df.columns:
                vol = df["volume"].values
                context.set("volume_array", vol)
                context.set("volume", vol)
            if gsa_prefix:
                gsa_csv_path = f"{gsa_prefix}_gsa_nanocube.csv"
                if os.path.isfile(gsa_csv_path):
                    artifacts["gsa_nanocube_csv"] = gsa_csv_path
            summary_parts.append(
                f"GSA nanocube metrics: {len(df)} rows."
            )

        context.set("observer_pass_results", pass_results)

        return SkillResult(
            success=True,
            data={"observer_pass_results": pass_results},
            artifacts=artifacts,
            summary=" ".join(summary_parts),
        )


_registry = get_default_registry()
_registry.register(RunTrajectoryObserverPassSkill())
