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
    HAS_ENDPOINTS_FINDER,
    HAS_VOLUME_ANALYZER
)
from MD_analysis.endpoints_finder import EndpointsFinder as ef
from MD_analysis.volume_analyser import VolumeAnalyzer

prmtop_list=[i for i in os.listdir() if i.endswith('.prmtop')]
for prmtop in prmtop_list:
    crd=prmtop.replace('_ca.prmtop','.bak')+'/109345/mdcrd_v'
    u=mda.Universe(prmtop, crd,format="TRJ")
    sel=u.select_atoms('not water and not name I and not name Na+')
    resid1=sel.residues[0]
    mol=resid1.atoms.convert_to('RDKIT')
    AllChem.Compute2DCoords(mol)
    efep=ef().find_endpoints(mol)
    mtest=Draw.MolToImage(mol, size=(600, 400), highlightAtoms=efep, highlightColor=(1, 0, 0))  # Red highlight
    mtest.save(prmtop.replace('_ca.prmtop','.png'),format='png')
    VA=VolumeAnalyzer(u, selection='not water and not name I and not name Na+')
    VA.make_volume_pipeline_gif(frame_index=0,gif_path=prmtop.replace('_ca.prmtop','.gif'),atom_stride=5,fps=10)
