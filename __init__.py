"""
MD_analysis Package
===================

A package for molecular dynamics trajectory analysis including:
- Trajectory deformation analysis
- Volume analysis
- Endpoint finding
- Clustering and frame selection
"""

__version__ = "1.0.0"

# Make key classes available at package level for convenience
try:
    from trajectory_deformation_workflow import (
        AlignedTrajectory,
        TrajectoryMetrics,
        ClusteringAnalysis,
        FrameSelection,
        FileIO,
        EndpointAnalyzer,
        Plotter,
        HAS_ENDPOINTS_FINDER,
        HAS_VOLUME_ANALYZER,
    )
    from gs_analyzer import GSAnalyzer
except ImportError:
    # Fallback for when imported from outside
    try:
        from MD_analysis.trajectory_deformation_workflow import (
            AlignedTrajectory,
            TrajectoryMetrics,
            ClusteringAnalysis,
            FrameSelection,
            FileIO,
            EndpointAnalyzer,
            Plotter,
            HAS_ENDPOINTS_FINDER,
            HAS_VOLUME_ANALYZER,
        )
        from MD_analysis.gs_analyzer import GSAnalyzer
    except ImportError:
        pass

try:
    from volume_analyser import VolumeAnalyzer
except ImportError:
    try:
        from MD_analysis.volume_analyser import VolumeAnalyzer
    except ImportError:
        VolumeAnalyzer = None

try:
    from endpoints_finder import EndpointsFinder
except ImportError:
    try:
        from MD_analysis.endpoints_finder import EndpointsFinder
    except ImportError:
        EndpointsFinder = None

__all__ = [
    "GSAnalyzer",
    "AlignedTrajectory",
    "TrajectoryMetrics",
    "ClusteringAnalysis",
    "FrameSelection",
    "FileIO",
    "EndpointAnalyzer",
    "Plotter",
    "VolumeAnalyzer",
    "EndpointsFinder",
    "HAS_ENDPOINTS_FINDER",
    "HAS_VOLUME_ANALYZER",
]

