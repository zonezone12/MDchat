"""Endpoint finding and distance analysis skills."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class FindEndpointsSkill(Skill):
    name = "find_endpoints"
    description = (
        "Identify molecular endpoints for specified residues using convex hull "
        "and graph farness algorithms (RDKit). Endpoints are the extremal atoms "
        "of each residue, useful for measuring pore openings and structural "
        "deformations in nanocubes and similar cages."
    )
    category = "endpoints"
    parameters = [
        Parameter("residue_selections", ParamType.ARRAY,
                  "List of MDAnalysis residue selection strings (e.g., "
                  "['resname GSA and resid 1', 'resname GSA and resid 2']). "
                  "Each selection should pick one residue.",
                  items_type=ParamType.STRING),
    ]
    requires = ["universe"]
    produces = ["endpoint_indices", "endpoint_info", "residue_selections"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        from src.EndpointAnalyzer import EndpointAnalyzer, EndpointsFinder

        u = context.universe
        residue_selections = params["residue_selections"]

        ef = EndpointsFinder()
        ea = EndpointAnalyzer()

        endpoint_info = []
        all_indices = []
        for sel_str in residue_selections:
            try:
                center, ep_indices = ea.find_residue_endpoints(u, sel_str, ef)
                endpoint_info.append({
                    "selection": sel_str,
                    "center": center.tolist() if isinstance(center, np.ndarray) else center,
                    "endpoint_atom_ids": ep_indices,
                    "n_endpoints": len(ep_indices),
                })
                all_indices.append(ep_indices)
            except Exception as exc:
                endpoint_info.append({
                    "selection": sel_str,
                    "error": str(exc),
                    "endpoint_atom_ids": [],
                    "n_endpoints": 0,
                })
                all_indices.append([])

        context.set("endpoint_indices", all_indices)
        context.set("endpoint_info", endpoint_info)
        context.set("residue_selections", residue_selections)

        found = sum(1 for info in endpoint_info if info["n_endpoints"] > 0)
        total_eps = sum(info["n_endpoints"] for info in endpoint_info)

        summary = (
            f"Found endpoints for {found}/{len(residue_selections)} residues "
            f"({total_eps} endpoint atoms total). "
        )
        for info in endpoint_info:
            if info["n_endpoints"] > 0:
                summary += (
                    f"\n  {info['selection']}: {info['n_endpoints']} endpoints "
                    f"(atom IDs: {info['endpoint_atom_ids'][:6]}"
                    f"{'...' if info['n_endpoints'] > 6 else ''})"
                )

        return SkillResult(
            success=True,
            data={
                "endpoint_indices": all_indices,
                "endpoint_info": endpoint_info,
                "residue_selections": residue_selections,
            },
            summary=summary,
        )


class ComputeEndpointDistancesSkill(Skill):
    name = "compute_endpoint_distances"
    description = (
        "Compute pairwise distances between endpoints of different residues "
        "across all trajectory frames. This reveals how pore openings change "
        "over time in cage-like molecules."
    )
    category = "endpoints"
    parameters = [
        Parameter("residue_selections", ParamType.ARRAY,
                  "List of MDAnalysis residue selection strings. "
                  "If omitted, uses selections from a previous find_endpoints call.",
                  required=False, items_type=ParamType.STRING),
    ]
    requires = ["universe"]
    produces = ["endpoint_distances"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        from src.EndpointAnalyzer import EndpointAnalyzer

        u = context.universe
        sel_list = params.get("residue_selections") or context.get("residue_selections")
        if not sel_list:
            return SkillResult(
                success=False,
                error="No residue selections provided and none found in context. "
                      "Run find_endpoints first or provide residue_selections.",
                summary="Cannot compute endpoint distances without residue selections.",
            )

        ea = EndpointAnalyzer()
        dists = ea.compute_endpoint_distances(u, sel_list)

        context.set("endpoint_distances", dists)

        n_pairs = len(dists.get("all_pairs", {}))
        n_frames = dists.get("n_frames", 0)

        return SkillResult(
            success=True,
            data={"endpoint_distances": dists},
            summary=(
                f"Computed endpoint distances for {n_pairs} residue pairs "
                f"across {n_frames} frames."
            ),
        )


class EndpointVolumeCorrelationSkill(Skill):
    name = "endpoint_volume_correlation"
    description = (
        "Compute the Pearson correlation between each pairwise endpoint "
        "distance and the molecular volume. Identifies which residue pairs "
        "drive expansion or shrinkage. Requires endpoint_distances and "
        "volume_array in context."
    )
    category = "endpoints"
    parameters = [
        Parameter("top_n", ParamType.INTEGER,
                  "Number of top-correlated pairs to highlight in the summary.",
                  required=False, default=5, min_value=1),
        Parameter("save_csv", ParamType.BOOLEAN,
                  "Save correlation table as CSV.",
                  required=False, default=True),
    ]
    requires = ["endpoint_distances", "volume_array"]
    produces = ["correlation_df"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import os
        from src.EndpointAnalyzer import EndpointAnalyzer

        dists = context.get("endpoint_distances")
        volume = context.get("volume_array")
        sel_list = context.get("residue_selections", [])
        top_n = params.get("top_n", 5)
        save_csv = params.get("save_csv", True)

        corr_df = EndpointAnalyzer.compute_endpoint_volume_correlation(
            dists, volume, sel_list,
        )
        context.set("correlation_df", corr_df)

        artifacts = {}
        if save_csv and corr_df is not None and len(corr_df) > 0:
            csv_path = os.path.join(
                context.output_dir, "endpoint_volume_correlation.csv"
            )
            corr_df.to_csv(csv_path, index=False)
            artifacts["correlation_csv"] = csv_path

        if corr_df is None or len(corr_df) == 0:
            return SkillResult(
                success=True,
                data={"correlation_df": corr_df},
                summary="No endpoint–volume correlations could be computed.",
            )

        sorted_df = corr_df.reindex(
            corr_df["correlation"].abs().sort_values(ascending=False).index
        )
        top = sorted_df.head(top_n)

        lines = [
            f"Endpoint–volume correlation computed ({len(corr_df)} pairs)."
        ]
        lines.append(f"Top {min(top_n, len(top))} correlated pairs:")
        for _, row in top.iterrows():
            lines.append(
                f"  {row.get('residue_i','?')} ↔ {row.get('residue_j','?')}: "
                f"r = {row['correlation']:.3f} (p = {row.get('p_value', 0):.2e})"
            )

        return SkillResult(
            success=True,
            data={"correlation_df": corr_df},
            artifacts=artifacts,
            summary="\n".join(lines),
        )


_registry = get_default_registry()
_registry.register(FindEndpointsSkill())
_registry.register(ComputeEndpointDistancesSkill())
_registry.register(EndpointVolumeCorrelationSkill())
