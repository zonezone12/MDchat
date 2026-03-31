import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import MDAnalysis as mda
import numpy as np
import os

from src.AlignedTrajectory import AlignedTrajectory
from src.TrajectoryMetrics import TrajectoryMetrics
from src.ClusteringAnalysis import ClusteringAnalysis
from src.FrameSelection import FrameSelection
from src.FileIO import FileIO
from src.EndpointAnalyzer import EndpointAnalyzer, EndpointsFinder as ef
from src.Plotter import Plotter
from src.task import GSAnalyzer
from src.VolumeAnalyzer import VolumeAnalyzer

prmtop_list=[i for i in os.listdir() if i.endswith('.prmtop')]
for prmtop in prmtop_list:
    crd=prmtop.replace('_ca.prmtop','.bak')+'/109345/mdcrd_v'
    u=mda.Universe(prmtop, crd,format="TRJ")
    Plotter().plot_residue_endpoints(u, 'resid 1', prmtop.replace('_ca.prmtop','_endpoints.png'))
    VA=VolumeAnalyzer(u, selection='not water and not name I and not name Na+')
    VA.make_volume_pipeline_gif(frame_index=0,gif_path=prmtop.replace('_ca.prmtop','.gif'),atom_stride=5,fps=10)
