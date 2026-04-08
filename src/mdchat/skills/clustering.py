"""Frame clustering and change-point detection skill."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..skill import Parameter, ParamType, Skill, SkillResult
from ..registry import get_default_registry

if TYPE_CHECKING:
    from ..context import AnalysisContext


class ClusterFramesSkill(Skill):
    name = "cluster_frames"
    description = (
        "Cluster trajectory frames based on PCA scores (or other embeddings) "
        "using HDBSCAN or KMeans. Also detects change points in the first "
        "principal component. Identifies structurally distinct conformational "
        "states and transition times."
    )
    category = "clustering"
    parameters = [
        Parameter("n_change_points", ParamType.INTEGER,
                  "Number of change points to detect.",
                  required=False, default=5, min_value=1, max_value=50),
        Parameter("cpd_model", ParamType.STRING,
                  "Change-point detection model.",
                  required=False, default="rbf",
                  enum_values=["rbf", "l2", "l1", "normal"]),
    ]
    requires = ["pca_scores"]
    produces = ["cluster_labels", "medoid_indices", "change_point_indices"]

    def execute(self, context: AnalysisContext, **params) -> SkillResult:
        import numpy as np
        from src.ClusteringAnalysis import ClusteringAnalysis

        pcs = context.get("pca_scores")
        n_bkps = params.get("n_change_points", 5)
        cpd_model = params.get("cpd_model", "rbf")

        ca = ClusteringAnalysis()

        labels, medoids = ca.cluster_frames(pcs)
        context.set("cluster_labels", labels)
        context.set("medoid_indices", medoids)

        cpd_idx = ca.change_points(pcs[:, 0], model=cpd_model, n_bkps=n_bkps)
        context.set("change_point_indices", cpd_idx)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_pct = (np.sum(labels == -1) / len(labels) * 100) if len(labels) > 0 else 0

        summary = (
            f"Clustering: {n_clusters} clusters identified from "
            f"{len(labels)} frames"
        )
        if noise_pct > 0:
            summary += f" ({noise_pct:.1f}% noise/outliers)"
        summary += ".\n"

        for cl in sorted(set(labels)):
            if cl == -1:
                continue
            count = int(np.sum(labels == cl))
            summary += f"  Cluster {cl}: {count} frames\n"

        summary += f"  Medoid frames: {medoids.tolist()}\n"
        summary += f"  Change points at frames: {cpd_idx}"

        return SkillResult(
            success=True,
            data={
                "cluster_labels": labels,
                "medoid_indices": medoids,
                "change_point_indices": cpd_idx,
            },
            summary=summary,
        )


get_default_registry().register(ClusterFramesSkill())
