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


get_default_registry().register(FindEndpointsSkill())
get_default_registry().register(ComputeEndpointDistancesSkill())
