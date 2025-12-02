from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import MDAnalysis as mda
import numpy as np
import os
from MD_analysis.trajectory_deformation_workflow import (
    AlignedTrajectory,
    TrajectoryMetrics,
    ClusteringAnalysis,
    FrameSelection,
    FileIO,
    GSAnalyzer,
    EndpointAnalyzer,
    Plotter,
)
try:
    from src.EndpointAnalyzer import EndpointsFinder as ef
except ImportError:
    # Fallback to old location for backward compatibility
    try:
        from MD_analysis.endpoints_finder import EndpointsFinder as ef
    except ImportError:
        from endpoints_finder import EndpointsFinder as ef
from MD_analysis.volume_analyser import VolumeAnalyzer

prmtop_list=[i for i in os.listdir() if i.endswith('.prmtop')]
for prmtop in prmtop_list:
    crd=prmtop.replace('_ca.prmtop','.bak')+'/109345/mdcrd_v'
    u=mda.Universe(prmtop, crd,format="TRJ")
    Plotter().plot_residue_endpoints(u, 'resid 1', prmtop.replace('_ca.prmtop','_endpoints.png'))
    VA=VolumeAnalyzer(u, selection='not water and not name I and not name Na+')
    VA.make_volume_pipeline_gif(frame_index=0,gif_path=prmtop.replace('_ca.prmtop','.gif'),atom_stride=5,fps=10)
    