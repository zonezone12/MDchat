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

from src.AlignedTrajectory import AlignedTrajectory
from src.TrajectoryMetrics import TrajectoryMetrics
from src.ClusteringAnalysis import ClusteringAnalysis
from src.FrameSelection import FrameSelection
from src.FileIO import FileIO
from src.EndpointAnalyzer import EndpointAnalyzer
from src.Plotter import Plotter
from src.VolumeAnalyzer import VolumeAnalyzer
from src.EndpointAnalyzer import EndpointsFinder
from src.task import GSAnalyzer

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
]
