"""MDChat skills for the metastable-structure half of the changepoint pipeline.

These wrap library stages in ``src.ChangepointAnalysis``. They do not require a
loaded universe except ``compute_gsa_features`` and ``compute_murata_motif_rmsd``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..registry import get_default_registry
from ..skill import Parameter, ParamType, Skill, SkillResult

if TYPE_CHECKING:
    from ..context import AnalysisContext


def _paths(written: dict) -> dict[str, str]:
    return {str(k): str(v) for k, v in written.items()}


def _fail(stage: str, exc: BaseException) -> SkillResult:
    return SkillResult(success=False, summary=f"{stage} failed: {exc}", error=str(exc))


def _cubes(params: dict) -> Optional[List[str]]:
    cubes = params.get("cubes")
    if not cubes:
        return None
    return list(cubes)


class ComputeGSAFeaturesSkill(Skill):
    name = "compute_gsa_features"
    description = (
        "Extract per-frame GSA nanocube features (geometry, contacts, guest) "
        "from one or more trajectories and write *_gsa_features.csv. Loads "
        "trajectories itself."
    )
    category = "changepoint"
    parameters = [
        Parameter("topology", ParamType.FILE_PATH, "Shared topology file.", required=True),
        Parameter(
            "trajectories", ParamType.ARRAY, "Trajectory path(s) or globs.",
            required=True, items_type=ParamType.STRING,
        ),
        Parameter("output_dir", ParamType.FILE_PATH, "Directory for feature CSVs.", required=False, default="output/gsa_features"),
        Parameter("gsa_resname", ParamType.STRING, "GSA residue name.", required=False, default="MOL"),
        Parameter("n_monomers", ParamType.INTEGER, "Expected monomer count.", required=False, default=6, min_value=2),
        Parameter("guest_selection", ParamType.ATOM_SELECTION, "Guest selection.", required=False, default="resname IOD"),
        Parameter("stride", ParamType.INTEGER, "Frame stride.", required=False, default=1, min_value=1),
        Parameter("traj_format", ParamType.STRING, "MDAnalysis trajectory format.", required=False, default="TRJ"),
        Parameter("n_jobs", ParamType.INTEGER, "Parallel jobs per trajectory.", required=False, default=1, min_value=1),
    ]
    requires: List[str] = []
    produces = ["gsa_features"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from pathlib import Path

        import MDAnalysis as mda

        from src.utils.gsa_feature_observer import compute_gsa_features
        from src.utils.gsa_selections import GSAFeatureSelections
        from src.ChangepointAnalysis.traj_paths import expand_trajectories, unique_trajectory_ids

        top = str(params["topology"])
        out_dir = Path(params.get("output_dir") or "output/gsa_features")
        out_dir.mkdir(parents=True, exist_ok=True)
        trajs = expand_trajectories(list(params["trajectories"]))
        if not trajs:
            return _fail("compute_gsa_features", FileNotFoundError("No trajectory files matched"))
        ids = unique_trajectory_ids(trajs)
        selections = GSAFeatureSelections(guest_sel=params.get("guest_selection") or "resname IOD")
        artifacts = {}
        n_frames = 0
        try:
            for path, traj_id in zip(trajs, ids):
                u = mda.Universe(top, str(path), format=params.get("traj_format") or "TRJ")
                df = compute_gsa_features(
                    u,
                    selections=selections,
                    gsa_resname=params.get("gsa_resname") or "MOL",
                    n_monomers=int(params.get("n_monomers", 6)),
                    traj_id=traj_id,
                    out_prefix=str(out_dir / traj_id),
                    step=int(params.get("stride", 1)),
                    n_jobs=int(params.get("n_jobs", 1)),
                )
                csv = out_dir / f"{traj_id}_gsa_features.csv"
                artifacts[traj_id] = str(csv)
                n_frames += len(df)
        except Exception as exc:
            return _fail("compute_gsa_features", exc)
        context.set("gsa_features", artifacts)
        return SkillResult(
            success=True,
            data={"gsa_features": artifacts, "n_trajectories": len(trajs), "n_frames": n_frames},
            artifacts=artifacts,
            summary=f"GSA features for {len(trajs)} trajectory(ies), {n_frames} frames → {out_dir}",
        )


class CompareEndpointClusterKSkill(Skill):
    name = "compare_endpoint_cluster_k"
    description = (
        "Silhouette-vs-k, switching, and optional chemical-scan diagnostics on "
        "existing clusters/endpoint (or clusters/gsa). Does not re-cluster."
    )
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Root with changepoint dirs.", required=False, default="output"),
        Parameter("group", ParamType.STRING, "Feature group.", required=False, default="endpoint"),
        Parameter("pattern", ParamType.STRING, "Glob for cohort dirs.", required=False, default=None),
        Parameter("out_dir", ParamType.FILE_PATH, "Diagnostics output directory.", required=False, default=None),
        Parameter("own_elbow", ParamType.BOOLEAN, "Use own-elbow endpoint dirs.", required=False, default=False),
        Parameter("chemical_scan", ParamType.BOOLEAN, "Also run the G3 chemical scan.", required=False, default=False),
        Parameter("chemical_k_max", ParamType.INTEGER, "Max k for chemical scan.", required=False, default=15, min_value=2),
        Parameter("max_top_replica_share", ParamType.FLOAT, "Replica-share cap for the chemical gate.", required=False, default=None, min_value=0.0, max_value=1.0),
        Parameter("k_from", ParamType.INTEGER, "Lower k for Ward-split table.", required=False, default=5),
        Parameter("k_to", ParamType.INTEGER, "Upper k for Ward-split table.", required=False, default=6),
    ]
    requires: List[str] = []
    produces = ["cluster_k_diagnostics"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from pathlib import Path

        from src.ChangepointAnalysis import (
            OWN_ELBOW_ENDPOINT_DIRS,
            compare_cluster_k_diagnostics,
            compare_endpoint_cluster_k_diagnostics,
            discover_cluster_k_cohort_dirs,
            discover_endpoint_k_cohort_dirs,
            endpoint_changepoints_dir,
        )

        root = Path(params.get("output_root") or "output")
        group = str(params.get("group") or "endpoint")
        pattern = params.get("pattern") or (
            "gsa_changepoints_B*" if group == "gsa" else "endpoint_changepoints_B*"
        )
        out_dir = Path(params.get("out_dir") or f"output/{group}_cluster_k_diagnostics")
        if params.get("own_elbow"):
            dirs = [
                d for cube in OWN_ELBOW_ENDPOINT_DIRS
                for d in [endpoint_changepoints_dir(root, cube, own_elbow=True)]
                if (d / "clusters" / "endpoint" / "segments_clustered.csv").exists()
            ]
        elif group == "endpoint":
            dirs = discover_endpoint_k_cohort_dirs(root, pattern, exclude_test=True)
        else:
            dirs = discover_cluster_k_cohort_dirs(root, pattern, group=group, exclude_test=True)
        if not dirs:
            return _fail("compare_endpoint_cluster_k", FileNotFoundError("No cohort directories found"))
        share = params.get("max_top_replica_share")
        try:
            if group == "endpoint":
                written = compare_endpoint_cluster_k_diagnostics(
                    dirs, out_dir,
                    k_from=int(params.get("k_from", 5)),
                    k_to=int(params.get("k_to", 6)),
                    include_chemical_scan=bool(params.get("chemical_scan")),
                    chemical_k_max=int(params.get("chemical_k_max", 15)),
                    max_top_replica_share=float(share) if share is not None else None,
                )
            else:
                written = compare_cluster_k_diagnostics(
                    dirs, out_dir, group=group,
                    k_from=int(params.get("k_from", 5)),
                    k_to=int(params.get("k_to", 6)),
                )
        except Exception as exc:
            return _fail("compare_endpoint_cluster_k", exc)
        artifacts = _paths(written)
        context.set("cluster_k_diagnostics", artifacts)
        return SkillResult(success=True, data={"n_cohorts": len(dirs)}, artifacts=artifacts,
                           summary=f"k-diagnostics for {len(dirs)} cohort(s) → {out_dir}")


class RunGSAStep1ClusterKSkill(Skill):
    name = "run_gsa_step1_cluster_k"
    description = (
        "Per-cube GSA changepoint from mixed gsa_features_step1 CSVs, then "
        "silhouette / switching k-diagnostics."
    )
    category = "changepoint"
    parameters = [
        Parameter("features_dir", ParamType.FILE_PATH, "Mixed *_gsa_features.csv root.", required=False, default="output/gsa_features_step1"),
        Parameter("output_root", ParamType.FILE_PATH, "Where gsa_changepoints_* are written.", required=False, default="output"),
        Parameter("out_dir", ParamType.FILE_PATH, "k-diagnostics directory.", required=False, default=None),
        Parameter("groups", ParamType.ARRAY, "Feature groups.", required=False, default=["gsa"], items_type=ParamType.STRING),
        Parameter("n_clusters", ParamType.INTEGER, "Ward k.", required=False, default=5, min_value=2),
        Parameter("workers", ParamType.INTEGER, "Parallel cube pipelines.", required=False, default=6),
        Parameter("skip_detect", ParamType.BOOLEAN, "Diagnostics only.", required=False, default=False),
        Parameter("force", ParamType.BOOLEAN, "Re-run detect even if clustered.", required=False, default=False),
    ]
    requires: List[str] = []
    produces = ["gsa_step1_cluster_k"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.gsa_cohort_run import run_gsa_step1_cluster_k

        try:
            written = run_gsa_step1_cluster_k(
                features_dir=params.get("features_dir") or "output/gsa_features_step1",
                output_root=params.get("output_root") or "output",
                out_dir=params.get("out_dir"),
                groups=list(params.get("groups") or ["gsa"]),
                n_clusters=int(params.get("n_clusters", 5)),
                workers=int(params.get("workers", 6)),
                skip_detect=bool(params.get("skip_detect")),
                force=bool(params.get("force")),
            )
        except Exception as exc:
            return _fail("run_gsa_step1_cluster_k", exc)
        artifacts = _paths(written)
        context.set("gsa_step1_cluster_k", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"GSA step-1 cluster-k → {len(written)} artifacts")


class SelectPhysicalKSkill(Skill):
    name = "select_physical_k"
    description = (
        "Pick k per cube: chemical gate, then held-out cage-geometry out-of-bag R² "
        "with the one-SE rule. This is the k used in the metastable-structures note."
    )
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("chem_dir", ParamType.FILE_PATH, "Directory with chemical_separation_by_k.csv.", required=False, default="output/endpoint_cluster_chemical_k_own_elbow_replica_aware"),
        Parameter("gsa_features_root", ParamType.FILE_PATH, "Per-cube GSA CSVs.", required=False, default="output/gsa_features_by_cube"),
        Parameter("out_dir", ParamType.FILE_PATH, "Output directory.", required=False, default="output/endpoint_cluster_physical_k_own_elbow"),
        Parameter("own_elbow", ParamType.BOOLEAN, "Score own-elbow clusters.", required=False, default=True),
        Parameter("n_bootstrap", ParamType.INTEGER, "Replica resamples.", required=False, default=500, min_value=0),
    ]
    requires: List[str] = []
    produces = ["physical_k_selected"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.cluster_k_diagnostics import write_physical_k_selection

        try:
            written = write_physical_k_selection(
                output_root=params.get("output_root") or "output",
                chem_dir=params.get("chem_dir") or "output/endpoint_cluster_chemical_k_own_elbow_replica_aware",
                gsa_features_root=params.get("gsa_features_root") or "output/gsa_features_by_cube",
                out_dir=params.get("out_dir") or "output/endpoint_cluster_physical_k_own_elbow",
                cubes=_cubes(params),
                own_elbow=bool(params.get("own_elbow", True)),
                n_bootstrap=int(params.get("n_bootstrap", 500)),
            )
        except Exception as exc:
            return _fail("select_physical_k", exc)
        artifacts = _paths(written)
        context.set("physical_k_selected", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Physical k → {artifacts.get('physical_k_selected', '')}")


class SelectChemicalKSkill(Skill):
    name = "select_chemical_k"
    description = (
        "Optional evaluation: pick k by Murata-label NMI among chemically valid "
        "cuts. Not the k used in the metastable-structures note."
    )
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Root with motif RMSD CSVs.", required=False, default="output"),
        Parameter("chem_dir", ParamType.FILE_PATH, "chemical_separation_by_k.csv directory.", required=False, default="output/endpoint_cluster_chemical_k"),
        Parameter("out_dir", ParamType.FILE_PATH, "Default: chem_dir.", required=False, default=None),
        Parameter("own_elbow", ParamType.BOOLEAN, "Score own-elbow clusters.", required=False, default=False),
        Parameter("n_bootstrap", ParamType.INTEGER, "Replica resamples for NMI SE.", required=False, default=500, min_value=0),
    ]
    requires: List[str] = []
    produces = ["chemical_k_selected"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.cluster_k_diagnostics import write_chemical_k_selection

        try:
            written = write_chemical_k_selection(
                output_root=params.get("output_root") or "output",
                chem_dir=params.get("chem_dir") or "output/endpoint_cluster_chemical_k",
                out_dir=params.get("out_dir"),
                cubes=_cubes(params),
                own_elbow=bool(params.get("own_elbow")),
                n_bootstrap=int(params.get("n_bootstrap", 500)),
            )
        except Exception as exc:
            return _fail("select_chemical_k", exc)
        artifacts = _paths(written)
        context.set("chemical_k_selected", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Chemical (NMI) k → {len(written)} artifacts")


class RankEndpointPairsAtKSkill(Skill):
    name = "rank_endpoint_pairs_at_k"
    description = "Rank endpoint site pairs by segment-level η² at each cube's selected Ward cut."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("out_dir", ParamType.FILE_PATH, "Ranking output.", required=False, default="output/endpoint_pair_ranking_physical_k"),
        Parameter("own_elbow", ParamType.BOOLEAN, "Read own-elbow clusters.", required=False, default=True),
        Parameter("k_selected", ParamType.FILE_PATH, "CSV with selected_k per cohort.", required=False, default=None),
        Parameter("k", ParamType.INTEGER, "Fixed k (overrides k_selected).", required=False, default=None, min_value=2),
        Parameter("n_permutations", ParamType.INTEGER, "Label-shuffle permutations.", required=False, default=999, min_value=0),
        Parameter("top_n", ParamType.INTEGER, "Top pairs kept per cube.", required=False, default=10, min_value=1),
    ]
    requires: List[str] = []
    produces = ["endpoint_pair_ranking"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.reporting import rank_endpoint_pairs_at_k

        try:
            written = rank_endpoint_pairs_at_k(
                output_root=params.get("output_root") or "output",
                out_dir=params.get("out_dir") or "output/endpoint_pair_ranking_physical_k",
                cubes=_cubes(params),
                own_elbow=bool(params.get("own_elbow", True)),
                k=params.get("k"),
                k_selected=params.get("k_selected"),
                n_permutations=int(params.get("n_permutations", 999)),
                top_n=int(params.get("top_n", 10)),
            )
        except Exception as exc:
            return _fail("rank_endpoint_pairs_at_k", exc)
        artifacts = _paths(written)
        context.set("endpoint_pair_ranking", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Site-pair ranking → {artifacts.get('top_pairs', '')}")


class DescribeEndpointClustersSkill(Skill):
    name = "describe_endpoint_clusters"
    description = "Per-cluster motif profile and the site pairs that separate chosen clusters of one cube."
    category = "changepoint"
    parameters = [
        Parameter("cube", ParamType.STRING, "Cube name (e.g. BHHpH).", required=True),
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("own_elbow", ParamType.BOOLEAN, "Read own-elbow clusters.", required=False, default=True),
        Parameter("k", ParamType.INTEGER, "Ward cut.", required=False, default=None, min_value=2),
        Parameter("clusters", ParamType.ARRAY, "Clusters to contrast.", required=False, default=None, items_type=ParamType.INTEGER),
        Parameter("out_dir", ParamType.FILE_PATH, "Profile folder.", required=False, default=None),
        Parameter("n_permutations", ParamType.INTEGER, "Permutations for η² p-values.", required=False, default=999, min_value=0),
    ]
    requires: List[str] = []
    produces = ["endpoint_cluster_profile"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.catalog_runs import describe_endpoint_clusters

        clusters = params.get("clusters")
        try:
            written = describe_endpoint_clusters(
                output_root=params.get("output_root") or "output",
                cube=str(params["cube"]),
                own_elbow=bool(params.get("own_elbow", True)),
                k=params.get("k"),
                clusters=[int(c) for c in clusters] if clusters else None,
                out_dir=params.get("out_dir"),
                n_permutations=int(params.get("n_permutations", 999)),
            )
        except Exception as exc:
            return _fail("describe_endpoint_clusters", exc)
        artifacts = _paths(written)
        context.set("endpoint_cluster_profile", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Described {params['cube']} → {len(written)} files")


class RankMetastableStructuresSkill(Skill):
    name = "rank_metastable_structures"
    description = "Merge orientation copies (r ≥ 0.95) and rank each cube's structures by apo-frame share."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("k_selected", ParamType.FILE_PATH, "physical_k_selected.csv.", required=False, default="output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv"),
        Parameter("profiles", ParamType.FILE_PATH, "describe_endpoint_clusters output.", required=False, default="output/endpoint_cluster_profiles"),
        Parameter("replica_audit", ParamType.FILE_PATH, "single_replica_clusters.csv.", required=False, default="output/cluster_replica_audit_physical_k/single_replica_clusters.csv"),
        Parameter("out_dir", ParamType.FILE_PATH, "Catalog directory.", required=False, default="output/metastable_structures_physical_k"),
        Parameter("copy_r", ParamType.FLOAT, "Orientation-merge correlation.", required=False, default=0.95),
    ]
    requires: List[str] = []
    produces = ["metastable_structures"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.catalog_runs import rank_metastable_structures

        try:
            written = rank_metastable_structures(
                output_root=params.get("output_root") or "output",
                k_selected=params.get("k_selected") or "output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv",
                profiles=params.get("profiles") or "output/endpoint_cluster_profiles",
                replica_audit=params.get("replica_audit") or "output/cluster_replica_audit_physical_k/single_replica_clusters.csv",
                copy_r=float(params.get("copy_r", 0.95)),
                out_dir=params.get("out_dir") or "output/metastable_structures_physical_k",
            )
        except Exception as exc:
            return _fail("rank_metastable_structures", exc)
        artifacts = _paths(written)
        context.set("metastable_structures", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Ranked structures → {artifacts.get('families_ranked', '')}")


class CategorizeClusterMixturesSkill(Skill):
    name = "categorize_cluster_mixtures"
    description = "Map apo frames onto Murata belt / pole-count majority inside each endpoint cluster."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Root with motif RMSD CSVs.", required=False, default="output"),
        Parameter("k_selected", ParamType.FILE_PATH, "physical_k_selected.csv.", required=False, default="output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv"),
        Parameter("out_dir", ParamType.FILE_PATH, "Output directory.", required=False, default="output/metastable_structures_physical_k"),
        Parameter("min_share", ParamType.FLOAT, "Reporting floor (%) for a belt inside a cluster.", required=False, default=5.0),
        Parameter("qc_gsa_root", ParamType.FILE_PATH, "Per-cube GSA CSVs for assembly QC.", required=False, default="output/gsa_features_by_cube"),
    ]
    requires: List[str] = []
    produces = ["cluster_mixtures"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.catalog_runs import categorize_cluster_mixtures

        try:
            written = categorize_cluster_mixtures(
                output_root=params.get("output_root") or "output",
                k_selected=params.get("k_selected") or "output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv",
                qc_gsa_root=params.get("qc_gsa_root") or "output/gsa_features_by_cube",
                min_share=float(params.get("min_share", 5.0)),
                out_dir=params.get("out_dir") or "output/metastable_structures_physical_k",
            )
        except Exception as exc:
            return _fail("categorize_cluster_mixtures", exc)
        artifacts = _paths(written)
        context.set("cluster_mixtures", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Belt majority tables → {len(written)} files")


class SummarizeMetastableStructuresSkill(Skill):
    name = "summarize_metastable_structures"
    description = "Section-7 tables: one row per structure plus site-pair markers mapped onto families."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("catalog_dir", ParamType.FILE_PATH, "Folder with families_ranked.csv.", required=False, default="output/metastable_structures_physical_k"),
        Parameter("segment_belt", ParamType.FILE_PATH, "segment_belt.csv with min-share 0.", required=False, default=None),
        Parameter("top_pairs", ParamType.FILE_PATH, "top_pairs.csv from pair ranking.", required=False, default="output/endpoint_pair_ranking_physical_k/top_pairs.csv"),
        Parameter("out_dir", ParamType.FILE_PATH, "Default: catalog_dir.", required=False, default=None),
    ]
    requires: List[str] = []
    produces = ["structures_summary"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.catalog_runs import summarize_metastable_structures

        try:
            written = summarize_metastable_structures(
                output_root=params.get("output_root") or "output",
                catalog_dir=params.get("catalog_dir") or "output/metastable_structures_physical_k",
                segment_belt=params.get("segment_belt"),
                top_pairs=params.get("top_pairs") or "output/endpoint_pair_ranking_physical_k/top_pairs.csv",
                out_dir=params.get("out_dir"),
            )
        except Exception as exc:
            return _fail("summarize_metastable_structures", exc)
        artifacts = _paths(written)
        context.set("structures_summary", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Structure summary → {artifacts.get('structures_summary', '')}")


class ComputeMurataMotifRmsdSkill(Skill):
    name = "compute_murata_motif_rmsd"
    description = "Per-frame Murata motif RMSD, π(pole)/π(eq), d1 and d2 for one cube's trajectories."
    category = "changepoint"
    parameters = [
        Parameter("topology", ParamType.FILE_PATH, "Shared topology.", required=True),
        Parameter("trajectories", ParamType.ARRAY, "Trajectory path(s) or globs.", required=True, items_type=ParamType.STRING),
        Parameter("output_dir", ParamType.FILE_PATH, "Output directory.", required=False, default="output/murata_motif_rmsd"),
        Parameter("gsa_dir", ParamType.FILE_PATH, "Optional GSA CSVs for apo filter.", required=False, default="output/gsa_features_step1"),
        Parameter("traj_jobs", ParamType.INTEGER, "Parallel workers (-1 = all CPUs).", required=False, default=-1),
        Parameter("step", ParamType.INTEGER, "Frame stride.", required=False, default=1, min_value=1),
        Parameter("cohort", ParamType.STRING, "Cube label (default: from topology).", required=False, default=None),
    ]
    requires: List[str] = []
    produces = ["motif_rmsd"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from pathlib import Path

        from src.ChangepointAnalysis.murata_rmsd import write_trajectory_motif_rmsd
        from src.ChangepointAnalysis.pipeline import _resolve_traj_workers
        from src.ChangepointAnalysis.traj_paths import expand_trajectories, unique_trajectory_ids

        topo = Path(params["topology"])
        name = topo.name
        cube = params.get("cohort")
        if not cube:
            for suffix in ("_ca.prmtop", ".prmtop", ".parm7", ".top"):
                if name.endswith(suffix):
                    cube = name[: -len(suffix)]
                    break
            cube = cube or topo.stem
        trajs = expand_trajectories(list(params["trajectories"]))
        if not trajs:
            return _fail("compute_murata_motif_rmsd", FileNotFoundError("No trajectories matched"))
        out_dir = Path(params.get("output_dir") or "output/murata_motif_rmsd")
        cube_dir = out_dir / cube
        cube_dir.mkdir(parents=True, exist_ok=True)
        gsa_dir = Path(params.get("gsa_dir") or "output/gsa_features_step1")
        ids = unique_trajectory_ids(trajs)
        jobs = []
        for traj, traj_id in zip(trajs, ids):
            hits = list(gsa_dir.rglob(f"{traj_id}_gsa_features.csv")) if gsa_dir.is_dir() else []
            jobs.append((
                str(topo), str(traj), traj_id, cube,
                str(cube_dir / f"{traj_id}_motif_rmsd.csv"),
                str(hits[0]) if hits else None,
                None, None, int(params.get("step", 1)), 0, 6, None,
            ))
        n_workers = _resolve_traj_workers(int(params.get("traj_jobs", -1)), len(jobs))
        artifacts = {}
        try:
            if n_workers == 1 or len(jobs) == 1:
                results = [write_trajectory_motif_rmsd(*job) for job in jobs]
            else:
                results = []
                with ProcessPoolExecutor(max_workers=n_workers) as pool:
                    futs = [pool.submit(write_trajectory_motif_rmsd, *job) for job in jobs]
                    for fut in as_completed(futs):
                        results.append(fut.result())
        except Exception as exc:
            return _fail("compute_murata_motif_rmsd", exc)
        for res in results:
            artifacts[str(res.get("path", "csv"))] = str(res["path"])
        context.set("motif_rmsd", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Motif RMSD for {len(results)} replica(s) → {cube_dir}")


class CompareToMurataMetastructuresSkill(Skill):
    name = "compare_to_murata_metastructures"
    description = "Occupancy vs Murata Table 1 and Murata label × endpoint cluster tables."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Root with motif RMSD CSVs.", required=False, default="output"),
        Parameter("out_dir", ParamType.FILE_PATH, "G5 output.", required=False, default="output/murata_g5_own_elbow_physical_k"),
        Parameter("own_elbow", ParamType.BOOLEAN, "Join own-elbow clusters.", required=False, default=True),
        Parameter("k_selected", ParamType.FILE_PATH, "selected_k CSV.", required=False, default=None),
        Parameter("qc_gsa_root", ParamType.FILE_PATH, "GSA QC root.", required=False, default="output/gsa_features_by_cube"),
    ]
    requires: List[str] = []
    produces = ["murata_g5"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.catalog_runs import compare_to_murata_metastructures

        try:
            written = compare_to_murata_metastructures(
                output_root=params.get("output_root") or "output",
                out_dir=params.get("out_dir") or "output/murata_g5_own_elbow_physical_k",
                cubes=_cubes(params),
                qc_gsa_root=params.get("qc_gsa_root") or "output/gsa_features_by_cube",
                own_elbow=bool(params.get("own_elbow", True)),
                k_selected=params.get("k_selected"),
            )
        except Exception as exc:
            return _fail("compare_to_murata_metastructures", exc)
        artifacts = _paths(written)
        context.set("murata_g5", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Murata G5 → {len(written)} files")


class AuditClusterReplicasSkill(Skill):
    name = "audit_cluster_replicas"
    description = "Flag endpoint clusters carried by a single replica and report that replica's Rg window."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("out_dir", ParamType.FILE_PATH, "Audit output.", required=False, default="output/cluster_replica_audit_physical_k"),
        Parameter("own_elbow", ParamType.BOOLEAN, "Audit own-elbow clusters.", required=False, default=True),
        Parameter("k_selected", ParamType.FILE_PATH, "selected_k CSV.", required=False, default=None),
        Parameter("min_top_share", ParamType.FLOAT, "Flag when one replica holds this share.", required=False, default=0.5, min_value=0.0, max_value=1.0),
        Parameter("gsa_root", ParamType.FILE_PATH, "Per-cube GSA CSVs.", required=False, default="output/gsa_features_by_cube"),
    ]
    requires: List[str] = []
    produces = ["cluster_replica_audit"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.cluster_replicas import audit_cluster_replicas

        try:
            written = audit_cluster_replicas(
                output_root=params.get("output_root") or "output",
                out_dir=params.get("out_dir") or "output/cluster_replica_audit_physical_k",
                cubes=_cubes(params),
                own_elbow=bool(params.get("own_elbow", True)),
                k_selected=params.get("k_selected"),
                gsa_root=params.get("gsa_root") or "output/gsa_features_by_cube",
                min_top_share=float(params.get("min_top_share", 0.5)),
            )
        except Exception as exc:
            return _fail("audit_cluster_replicas", exc)
        artifacts = _paths(written)
        context.set("cluster_replica_audit", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Replica audit → {len(written)} files")


class DetectReplicaOutliersSkill(Skill):
    name = "detect_replica_outliers"
    description = "Flag dissociated runs (assembly QC) and replicas isolated by clustering across k."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("gsa_root", ParamType.FILE_PATH, "Per-cube GSA CSVs.", required=False, default="output/gsa_features_by_cube"),
        Parameter("k_selected", ParamType.FILE_PATH, "physical_k_selected.csv.", required=False, default="output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv"),
        Parameter("out_dir", ParamType.FILE_PATH, "Outlier tables.", required=False, default="output/replica_outliers"),
        Parameter("min_top_share", ParamType.FLOAT, "Isolation threshold.", required=False, default=0.5),
    ]
    requires: List[str] = []
    produces = ["replica_outliers"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.cluster_replicas import write_replica_outliers

        try:
            written = write_replica_outliers(
                output_root=params.get("output_root") or "output",
                gsa_root=params.get("gsa_root") or "output/gsa_features_by_cube",
                k_selected=params.get("k_selected") or "output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv",
                cubes=_cubes(params),
                min_top_share=float(params.get("min_top_share", 0.5)),
                out_dir=params.get("out_dir") or "output/replica_outliers",
            )
        except Exception as exc:
            return _fail("detect_replica_outliers", exc)
        artifacts = _paths(written)
        context.set("replica_outliers", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Replica outliers → {artifacts.get('replica_outliers', '')}")


class ValidateEndpointClustersSkill(Skill):
    name = "validate_endpoint_clusters"
    description = "Structural checks A–D on endpoint clusters at the selected k (does not re-cluster)."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("k_selected", ParamType.FILE_PATH, "physical_k_selected.csv.", required=False, default="output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv"),
        Parameter("out_dir", ParamType.FILE_PATH, "Validation output.", required=False, default="output/endpoint_cluster_validation_physical_k"),
        Parameter("own_elbow", ParamType.BOOLEAN, "Use own-elbow clusters.", required=False, default=True),
        Parameter("apo_only", ParamType.BOOLEAN, "Restrict tests to apo frames.", required=False, default=False),
        Parameter("group", ParamType.STRING, "Relabelling group: admissible, rotations, or contact.", required=False, default="admissible"),
    ]
    requires: List[str] = []
    produces = ["endpoint_cluster_validation"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.endpoint_cluster_checks import validate_endpoint_clusters

        try:
            written = validate_endpoint_clusters(
                output_root=params.get("output_root") or "output",
                k_selected=params.get("k_selected") or "output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv",
                own_elbow=bool(params.get("own_elbow", True)),
                apo_only=bool(params.get("apo_only")),
                out_dir=params.get("out_dir") or "output/endpoint_cluster_validation_physical_k",
                cubes=_cubes(params),
                group=str(params.get("group") or "admissible"),
            )
        except Exception as exc:
            return _fail("validate_endpoint_clusters", exc)
        artifacts = _paths(written)
        context.set("endpoint_cluster_validation", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Cluster validation → {len(written)} files")


class CheckMergeDilutionSkill(Skill):
    name = "check_merge_dilution"
    description = "Inspect tangled orientation-merge pairs: core vs weak deformation-profile r."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("k_selected", ParamType.FILE_PATH, "physical_k_selected.csv.", required=False, default="output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv"),
        Parameter("out_dir", ParamType.FILE_PATH, "Dilution output.", required=False, default="output/endpoint_cluster_validation_physical_k/merge_dilution"),
        Parameter("cubes", ParamType.ARRAY, "Cubes to inspect.", required=False, default=None, items_type=ParamType.STRING),
        Parameter("k", ParamType.INTEGER, "Ward cut override.", required=False, default=None),
        Parameter("clusters", ParamType.ARRAY, "Explicit cluster list (requires one cube).", required=False, default=None, items_type=ParamType.INTEGER),
        Parameter("belt_crosscheck", ParamType.BOOLEAN, "Also tabulate core/weak vs majority belt.", required=False, default=False),
    ]
    requires: List[str] = []
    produces = ["merge_dilution"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.merge_dilution import check_merge_dilution

        clusters = params.get("clusters")
        try:
            written = check_merge_dilution(
                output_root=params.get("output_root") or "output",
                k_selected=params.get("k_selected") or "output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv",
                out_dir=params.get("out_dir") or "output/endpoint_cluster_validation_physical_k/merge_dilution",
                cubes=_cubes(params),
                k=params.get("k"),
                clusters=[int(c) for c in clusters] if clusters else None,
                belt_crosscheck=bool(params.get("belt_crosscheck")),
            )
        except Exception as exc:
            return _fail("check_merge_dilution", exc)
        artifacts = _paths(written)
        context.set("merge_dilution", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Merge dilution → {len(written)} files")


class BootstrapCoreMatchSkill(Skill):
    name = "bootstrap_core_match"
    description = "Replica bootstrap of core-to-core deformation-profile r for listed clusters of one cube."
    category = "changepoint"
    parameters = [
        Parameter("cube", ParamType.STRING, "Cube name.", required=True),
        Parameter("k", ParamType.INTEGER, "Ward cut.", required=True, min_value=2),
        Parameter("clusters", ParamType.ARRAY, "Cluster labels whose cores are compared.", required=True, items_type=ParamType.INTEGER),
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("out_dir", ParamType.FILE_PATH, "Bootstrap output.", required=False, default="output/endpoint_cluster_validation_physical_k/merge_dilution/core_bootstrap"),
        Parameter("n_boot", ParamType.INTEGER, "Replica draws.", required=False, default=200, min_value=1),
    ]
    requires: List[str] = []
    produces = ["core_bootstrap"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.core_bootstrap import bootstrap_core_match

        try:
            written = bootstrap_core_match(
                cube=str(params["cube"]),
                k=int(params["k"]),
                clusters=[int(c) for c in params["clusters"]],
                output_root=params.get("output_root") or "output",
                out_dir=params.get("out_dir") or "output/endpoint_cluster_validation_physical_k/merge_dilution/core_bootstrap",
                n_boot=int(params.get("n_boot", 200)),
            )
        except Exception as exc:
            return _fail("bootstrap_core_match", exc)
        artifacts = _paths(written)
        context.set("core_bootstrap", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Core bootstrap for {params['cube']} k={params['k']}")


class CheckCageSymmetrySkill(Skill):
    name = "check_cage_symmetry"
    description = "Test which monomer relabellings are physically admissible (Test D1) on a compact reference cage."
    category = "changepoint"
    parameters = [
        Parameter("cube", ParamType.STRING, "Cube name.", required=True),
        Parameter("topology", ParamType.FILE_PATH, "Topology (optional for pole-H cubes inferred from pole-M).", required=False, default=None),
        Parameter("trajectory", ParamType.FILE_PATH, "Local compact-cage trajectory.", required=False, default=None),
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("out_dir", ParamType.FILE_PATH, "Symmetry output.", required=False, default=None),
        Parameter("k_selected", ParamType.FILE_PATH, "physical_k_selected.csv.", required=False, default=None),
    ]
    requires: List[str] = []
    produces = ["cage_symmetry"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.cage_symmetry import check_cage_symmetry

        try:
            written = check_cage_symmetry(
                cube=str(params["cube"]),
                topology=params.get("topology"),
                trajectory=params.get("trajectory"),
                output_root=params.get("output_root") or "output",
                out_dir=params.get("out_dir"),
                k_selected=params.get("k_selected"),
            )
        except Exception as exc:
            return _fail("check_cage_symmetry", exc)
        artifacts = _paths(written)
        context.set("cage_symmetry", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Cage symmetry for {params['cube']}")


class GuestEventCotimingSkill(Skill):
    name = "guest_event_cotiming"
    description = "Debounced iodide entry/exit versus cage and endpoint breakpoints."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("gsa_features_root", ParamType.FILE_PATH, "Per-cube GSA CSVs.", required=False, default="output/gsa_features_by_cube"),
        Parameter("out_dir", ParamType.FILE_PATH, "Co-timing output.", required=False, default="output/guest_event_cotiming"),
        Parameter("min_dwell", ParamType.ARRAY, "Dwell thresholds in frames (ps).", required=False, default=[20, 50], items_type=ParamType.INTEGER),
        Parameter("tolerance", ParamType.INTEGER, "Breakpoint match window (frames).", required=False, default=50),
    ]
    requires: List[str] = []
    produces = ["guest_event_cotiming"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.guest_cotiming import guest_event_cotiming

        dwells = params.get("min_dwell") or [20, 50]
        try:
            written = guest_event_cotiming(
                output_root=params.get("output_root") or "output",
                gsa_features_root=params.get("gsa_features_root") or "output/gsa_features_by_cube",
                out_dir=params.get("out_dir") or "output/guest_event_cotiming",
                cubes=_cubes(params),
                min_dwell=[int(x) for x in dwells],
                tolerance=int(params.get("tolerance", 50)),
            )
        except Exception as exc:
            return _fail("guest_event_cotiming", exc)
        artifacts = _paths(written)
        context.set("guest_event_cotiming", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Guest co-timing → {artifacts.get('guest_event_cotiming', '')}")


class CompareGuestCageGeometrySkill(Skill):
    name = "compare_guest_cage_geometry"
    description = "Within-replica cage geometry with iodide in the cavity, and the 100 ps window at entry."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("gsa_features_root", ParamType.FILE_PATH, "Per-cube GSA CSVs.", required=False, default="output/gsa_features_by_cube"),
        Parameter("dwells", ParamType.FILE_PATH, "guest_dwells.csv.", required=False, default="output/guest_event_cotiming/guest_dwells.csv"),
        Parameter("k_selected", ParamType.FILE_PATH, "physical_k_selected.csv.", required=False, default="output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv"),
        Parameter("out_dir", ParamType.FILE_PATH, "Geometry output.", required=False, default="output/guest_cage_geometry"),
        Parameter("min_dwell", ParamType.INTEGER, "Sustained-visit length (ps).", required=False, default=20),
    ]
    requires: List[str] = []
    produces = ["guest_cage_geometry"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.guest_geometry import compare_guest_cage_geometry

        try:
            written = compare_guest_cage_geometry(
                output_root=params.get("output_root") or "output",
                gsa_features_root=params.get("gsa_features_root") or "output/gsa_features_by_cube",
                dwells=params.get("dwells") or "output/guest_event_cotiming/guest_dwells.csv",
                k_selected=params.get("k_selected") or "output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv",
                out_dir=params.get("out_dir") or "output/guest_cage_geometry",
                cubes=_cubes(params),
                min_dwell=int(params.get("min_dwell", 20)),
            )
        except Exception as exc:
            return _fail("compare_guest_cage_geometry", exc)
        artifacts = _paths(written)
        context.set("guest_cage_geometry", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Guest–cage geometry → {len(written)} files")


class GuestEntryGeometryResponseSkill(Skill):
    name = "guest_entry_geometry_response"
    description = "Cage coordinates aligned to a sustained iodide entry, with a kink against random times."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("gsa_features_root", ParamType.FILE_PATH, "Per-cube GSA CSVs.", required=False, default="output/gsa_features_by_cube"),
        Parameter("out_dir", ParamType.FILE_PATH, "Aligned-trace output.", required=False, default="output/guest_cage_geometry"),
        Parameter("min_dwell", ParamType.INTEGER, "Sustained-visit length (ps).", required=False, default=20),
        Parameter("n_shift", ParamType.INTEGER, "Random-time null draws.", required=False, default=200),
    ]
    requires: List[str] = []
    produces = ["guest_entry_response"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.guest_entry_response import guest_entry_geometry_response

        try:
            written = guest_entry_geometry_response(
                output_root=params.get("output_root") or "output",
                gsa_features_root=params.get("gsa_features_root") or "output/gsa_features_by_cube",
                out_dir=params.get("out_dir") or "output/guest_cage_geometry",
                cubes=_cubes(params),
                min_dwell=int(params.get("min_dwell", 20)),
                n_shift=int(params.get("n_shift", 200)),
            )
        except Exception as exc:
            return _fail("guest_entry_geometry_response", exc)
        artifacts = _paths(written)
        context.set("guest_entry_response", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Entry-aligned geometry → {len(written)} files")


class ListAnnotationReplicasSkill(Skill):
    name = "list_annotation_replicas"
    description = "Replicas whose endpoint state leaves the tight family once, with priority cuts per cube."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint root.", required=False, default="output"),
        Parameter("catalog_dir", ParamType.FILE_PATH, "Folder with structures_summary.csv.", required=False, default="output/metastable_structures_physical_k"),
        Parameter("k_selected", ParamType.FILE_PATH, "physical_k_selected.csv.", required=False, default="output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv"),
        Parameter("out_dir", ParamType.FILE_PATH, "Annotation output.", required=False, default="output/annotation_replicas"),
    ]
    requires: List[str] = []
    produces = ["annotation_replicas"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.catalog_runs import list_annotation_replicas

        try:
            written = list_annotation_replicas(
                output_root=params.get("output_root") or "output",
                catalog_dir=params.get("catalog_dir") or "output/metastable_structures_physical_k",
                k_selected=params.get("k_selected") or "output/endpoint_cluster_physical_k_own_elbow/physical_k_selected.csv",
                out_dir=params.get("out_dir") or "output/annotation_replicas",
            )
        except Exception as exc:
            return _fail("list_annotation_replicas", exc)
        artifacts = _paths(written)
        context.set("annotation_replicas", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Annotation replicas → {artifacts.get('replicas', '')}")


class CompareEndpointClustersAcrossCohortsSkill(Skill):
    name = "compare_endpoint_clusters_across_cohorts"
    description = "Cross-cohort endpoint cluster vs deformation-proxy tables and plots."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Root with endpoint_changepoints_* dirs.", required=False, default="output"),
        Parameter("pattern", ParamType.STRING, "Glob for cohort dirs.", required=False, default="endpoint_changepoints_B*"),
        Parameter("out_dir", ParamType.FILE_PATH, "Cross-cohort output.", required=False, default="output/endpoint_cluster_cross_cohort"),
        Parameter("top_pairs", ParamType.INTEGER, "Top η² pairs per cohort.", required=False, default=10, min_value=1),
    ]
    requires: List[str] = []
    produces = ["endpoint_cluster_cross_cohort"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from pathlib import Path
        import fnmatch

        from src.ChangepointAnalysis import compare_endpoint_clusters_across_cohorts

        root = Path(params.get("output_root") or "output")
        pattern = str(params.get("pattern") or "endpoint_changepoints_B*")
        dirs = []
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if child.is_dir() and fnmatch.fnmatch(child.name, pattern):
                    if (child / "endpoint_cluster_deformation_summary.csv").exists():
                        dirs.append(child)
        if not dirs:
            return _fail("compare_endpoint_clusters_across_cohorts", FileNotFoundError("No cohort directories"))
        try:
            written = compare_endpoint_clusters_across_cohorts(
                dirs, Path(params.get("out_dir") or "output/endpoint_cluster_cross_cohort"),
                top_pairs=int(params.get("top_pairs", 10)),
            )
        except Exception as exc:
            return _fail("compare_endpoint_clusters_across_cohorts", exc)
        artifacts = _paths(written)
        context.set("endpoint_cluster_cross_cohort", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Cross-cohort proxies for {len(dirs)} cohort(s)")


class RemapMurataD1Skill(Skill):
    name = "remap_murata_d1"
    description = "Map stored hull-index site pairs onto six equatorial Murata d1 contacts (no traj replay)."
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Root with endpoint_changepoints_*.", required=False, default="output"),
        Parameter("out_dir", ParamType.FILE_PATH, "Remap output.", required=False, default="output/murata_d1"),
        Parameter("traj_dir", ParamType.FILE_PATH, "Topologies (CUBE_ca.prmtop).", required=False, default="traj"),
    ]
    requires: List[str] = []
    produces = ["murata_d1"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.murata_d1 import remap_murata_d1_cohorts

        try:
            written = remap_murata_d1_cohorts(
                output_root=params.get("output_root") or "output",
                out_dir=params.get("out_dir") or "output/murata_d1",
                traj_dir=params.get("traj_dir") or "traj",
                cubes=_cubes(params),
            )
        except Exception as exc:
            return _fail("remap_murata_d1", exc)
        artifacts = _paths(written)
        context.set("murata_d1", artifacts)
        return SkillResult(success=True, artifacts=artifacts, summary=f"Murata d1 remap → {artifacts.get('murata_d1_summary', '')}")


class RunMetastablePipelineSkill(Skill):
    name = "run_metastable_pipeline"
    description = (
        "Full metastable-structure pipeline on existing own-elbow segments: "
        "chemical scan (replica cap 0.5, k through 15), physical k, pair ranking, "
        "orientation merge, belt categorization, structure summary, replica audit, "
        "and guest/annotation stages when their inputs exist. Does not re-detect "
        "changepoints or refit Ward."
    )
    category = "changepoint"
    parameters = [
        Parameter("output_root", ParamType.FILE_PATH, "Changepoint / motif RMSD root.", required=False, default="output"),
        Parameter("chem_dir", ParamType.FILE_PATH, "Chemical-scan output.", required=False, default="output/endpoint_cluster_chemical_k_own_elbow_replica_aware"),
        Parameter("physical_k_dir", ParamType.FILE_PATH, "Physical-k output.", required=False, default="output/endpoint_cluster_physical_k_own_elbow"),
        Parameter("skip_guest", ParamType.BOOLEAN, "Skip iodide co-timing stages.", required=False, default=False),
        Parameter("skip_annotation", ParamType.BOOLEAN, "Skip annotation-replica list.", required=False, default=False),
        Parameter("k_max", ParamType.INTEGER, "Chemical-scan upper k.", required=False, default=15, min_value=2),
        Parameter("max_top_replica_share", ParamType.FLOAT, "Chemical-gate replica cap.", required=False, default=0.5),
    ]
    requires: List[str] = []
    produces = ["metastable_pipeline"]

    def execute(self, context: "AnalysisContext", **params: Any) -> SkillResult:
        from src.ChangepointAnalysis.metastable_pipeline import run_metastable_pipeline

        try:
            written = run_metastable_pipeline(
                output_root=params.get("output_root") or "output",
                chem_dir=params.get("chem_dir") or "output/endpoint_cluster_chemical_k_own_elbow_replica_aware",
                physical_k_dir=params.get("physical_k_dir") or "output/endpoint_cluster_physical_k_own_elbow",
                cubes=_cubes(params),
                k_max=int(params.get("k_max", 15)),
                max_top_replica_share=float(params.get("max_top_replica_share", 0.5)),
                skip_guest=bool(params.get("skip_guest")),
                skip_annotation=bool(params.get("skip_annotation")),
            )
        except Exception as exc:
            return _fail("run_metastable_pipeline", exc)
        artifacts = _paths(written)
        context.set("metastable_pipeline", artifacts)
        return SkillResult(
            success=True,
            artifacts=artifacts,
            summary=f"Metastable pipeline wrote {len(written)} artifact(s)",
        )


_registry = get_default_registry()
for _skill in (
    ComputeGSAFeaturesSkill(),
    CompareEndpointClusterKSkill(),
    RunGSAStep1ClusterKSkill(),
    SelectPhysicalKSkill(),
    SelectChemicalKSkill(),
    RankEndpointPairsAtKSkill(),
    DescribeEndpointClustersSkill(),
    RankMetastableStructuresSkill(),
    CategorizeClusterMixturesSkill(),
    SummarizeMetastableStructuresSkill(),
    ComputeMurataMotifRmsdSkill(),
    CompareToMurataMetastructuresSkill(),
    AuditClusterReplicasSkill(),
    DetectReplicaOutliersSkill(),
    ValidateEndpointClustersSkill(),
    CheckMergeDilutionSkill(),
    BootstrapCoreMatchSkill(),
    CheckCageSymmetrySkill(),
    GuestEventCotimingSkill(),
    CompareGuestCageGeometrySkill(),
    GuestEntryGeometryResponseSkill(),
    ListAnnotationReplicasSkill(),
    CompareEndpointClustersAcrossCohortsSkill(),
    RemapMurataD1Skill(),
    RunMetastablePipelineSkill(),
):
    _registry.register(_skill)
