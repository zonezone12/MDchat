"""MDChat skills for the metastable / changepoint pipeline are registered."""

from src.mdchat.registry import get_default_registry
from src.mdchat.skills import changepoint_pipeline, metastable_pipeline  # noqa: F401


EXPECTED = [
    "scan_endpoint_cluster_chemical_k",
    "compute_gsa_features",
    "compare_endpoint_cluster_k",
    "run_gsa_step1_cluster_k",
    "select_physical_k",
    "select_chemical_k",
    "rank_endpoint_pairs_at_k",
    "describe_endpoint_clusters",
    "rank_metastable_structures",
    "categorize_cluster_mixtures",
    "summarize_metastable_structures",
    "compute_murata_motif_rmsd",
    "compare_to_murata_metastructures",
    "audit_cluster_replicas",
    "detect_replica_outliers",
    "validate_endpoint_clusters",
    "check_merge_dilution",
    "bootstrap_core_match",
    "check_cage_symmetry",
    "guest_event_cotiming",
    "compare_guest_cage_geometry",
    "guest_entry_geometry_response",
    "list_annotation_replicas",
    "compare_endpoint_clusters_across_cohorts",
    "remap_murata_d1",
    "run_metastable_pipeline",
]


def test_metastable_skills_registered():
    names = {s.name for s in get_default_registry().list_skills()}
    missing = [n for n in EXPECTED if n not in names]
    assert not missing, f"unregistered skills: {missing}"


def test_chemical_scan_exposes_replica_cap():
    skill = get_default_registry().get("scan_endpoint_cluster_chemical_k")
    param_names = {p.name for p in skill.parameters}
    assert "max_top_replica_share" in param_names
    assert "own_elbow" in param_names


def test_run_metastable_pipeline_cli_exists():
    from pathlib import Path

    text = Path("scripts/run_metastable_pipeline.py").read_text(encoding="utf-8")
    assert "--trajectories" in text
    assert "run_cube_from_trajectories(" in text
    assert "run_metastable_pipeline(" in text


def test_run_cube_from_trajectories_exported():
    from src.ChangepointAnalysis import run_cube_from_trajectories, run_metastable_pipeline

    assert callable(run_cube_from_trajectories)
    assert callable(run_metastable_pipeline)


def test_orchestrator_does_not_redetect():
    import inspect

    from src.ChangepointAnalysis.metastable_pipeline import run_metastable_pipeline

    source = inspect.getsource(run_metastable_pipeline)
    assert "detect_cohort_changepoints(" not in source
    assert "cluster_all_groups(" not in source
    assert "run_endpoint_changepoint(" not in source
    assert "scan_chemical_separation_by_k(" in source
    assert "write_physical_k_selection(" in source
