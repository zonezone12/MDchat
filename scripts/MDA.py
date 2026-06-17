"""Dev/scratch script for interactive MD analysis exploration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MDAnalysis.analysis import align, rms
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from typing import List
from src.EndpointAnalyzer import EndpointAnalyzer as ea
from src.EndpointAnalyzer import EndpointsFinder as ef
from src.VolumeAnalyzer import VolumeAnalyzer as va
from rdkit.Chem import Draw
from MDAnalysis.analysis import gnm
import matplotlib.pyplot as plt

import MDAnalysis as mda
prmtop=r'traj\BMMpM_ca.prmtop'
crd=r'traj\BMMpM_891249_mdcrd_v.trj'
first_u= mda.Universe(prmtop, crd,format="TRJ")


sel=first_u.select_atoms('resid 6')
mol = sel.convert_to("RDKIT")
from src.utils import RingCenterCalculator
calc = RingCenterCalculator()
rings = calc.find_all_rings(mol)
print(rings)
substructures = calc.find_and_map_substructures(mol, sel)
print(substructures)
sub_centers = calc.calculate_substructure_center(sel.positions, [sub.rdkit_indices for sub in substructures])
print(sub_centers)


from src.Plotter import Plotter
Plotter().plot_residue_endpoints(
    first_u, 'resid 1', 'BMMpM_endpoints',
    highlight_center_benzene=True,
    highlight_methyl_endpoints=True,
)

from src.utils.guest_in import guest_entering
# For lightweight distance calculations, sequential processing is typically fastest
# Dask overhead (Universe recreation, pickling, task submission) outweighs benefits
# Option 1: Sequential (fastest for lightweight work) - ~10 minutes
frame = guest_entering(first_u, 'not water and not name I and not name Na+', 'resname IOD', return_stats=True, n_jobs=-1,use_dask=True)

# Option 2: Multiprocessing (less overhead than Dask for local computations)
# frame = guest_entering(first_u, 'not water and not name I and not name Na+', 'resname IOD', return_stats=True, n_jobs=4, use_dask=False)

# Option 3: Dask (only recommended for heavy computations or distributed clusters)
# from dask.distributed import Client
# client = Client(n_workers=4, threads_per_worker=1)
# frame = guest_entering(first_u, 'not water and not name I and not name Na+', 'resname IOD', return_stats=True, n_jobs=4, dask_client=client)
print(f"Guest entered at frame {frame}")

last_u=mda.Universe(prmtop, crd,format="TRJ")
last_u.trajectory[-1]
first_u.trajectory[0]
first_sel=first_u.select_atoms('not water and not name I and not name Na+')

from MDAnalysis.coordinates.XYZ import XYZWriter
writer = XYZWriter("BMMpM.xyz")
writer.write(first_sel)
writer.close()
import pore_mapper as pm

# Read in host from xyz file.
host = pm.Host.init_from_xyz_file(path='BMMpM.xyz')
host = host.with_centroid([0., 0., 0.])

# Define calculator object.
calculator = pm.Inflater(bead_sigma=1.0, centroid=host.get_centroid())

# Run calculator on host object, analysing output.
final_result = calculator.get_inflated_blob(host=host)

# Analysis.
windows = final_result.pore.get_windows()
print(f'windows: {windows}, pore_volume: {final_result.pore.get_volume()}')

last_sel=last_u.select_atoms('not water and not name I and not name Na+')
unaligned_rmsd = rms.rmsd(first_sel.positions, last_sel.positions, superposition=False)
print(f"Unaligned RMSD: {unaligned_rmsd:.2f}")
rep=ea.find_residue_endpoints(first_sel, 'resid 1',)
rep_sel=first_u.select_atoms('resid 1')
lrep_sel=last_u.select_atoms('resid 1')
unal_rmsd_resid1=rms.rmsd(rep_sel[rep[1]].positions, lrep_sel[rep[1]].positions, superposition=False)
print(f"Unaligned RMSD of residue 1: {unal_rmsd_resid1:.2f}")

aligner = align.AlignTraj(first_u, last_u, select='not water and not name I and not name Na+', in_memory=True).run()
aligned_rmsd = rms.rmsd(first_sel.positions, last_sel.positions, superposition=False)
print(f"Aligned RMSD: {aligned_rmsd:.2f}")
rep_sel=first_u.select_atoms('resid 1')
lrep_sel=last_u.select_atoms('resid 1')
al_rmsd_resid1=rms.rmsd(rep_sel[rep[1]].positions, lrep_sel[rep[1]].positions, superposition=False)
print(f"Aligned RMSD of residue 1: {al_rmsd_resid1:.2f}")

VA=va(first_u, selection='not water and not name I and not name Na+')
fig=VA.plot_interactive_3d()
fig.show()

VA.make_mark_occupancy_gif(frame_index=0,gif_path="occupancy_build_frame0.gif",atom_stride=5,voxel_stride=2,)

VA.make_volume_pipeline_gif(frame_index=0,gif_path="BMMpM_volume.gif",atom_stride=10,voxel_stride=10,fps=5)

target_vol, cavity_vol, inside, cavities=VA.compute_frame(0,return_masks=True)
cmv=AllChem.ComputeMolVolume(first_sel.convert_to('RDKIT'))
from rdkit.Chem import rdMolDescriptors, rdchem
from rdkit.Chem import rdDistGeom
ps = rdDistGeom.ETKDGv3()
ps.randomSeed = 0xa100f

mol = Chem.AddHs(Chem.MolFromSmiles('CCC')) #exp 74 A^3
rdDistGeom.EmbedMolecule(mol,ps)
from MDAnalysis.converters.RDKit import RDKitReader
u_from_rdkit = mda.Universe(mol, reader=RDKitReader)
VA=va(u_from_rdkit, spacing=0.1,probe_radius=0.5,selection='all')
target_vol, cavity_vol, inside, cavities=VA.compute_frame(0,return_masks=True)


mol = first_sel.convert_to('RDKIT')
pt = rdchem.GetPeriodicTable()

radii = [pt.GetRcovalent(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
dclv=rdMolDescriptors.DoubleCubicLatticeVolume(mol, radii,isProtein=False)
vdw_volume = dclv.GetVDWVolume()
polar_volume = dclv.GetPolarVolume()
volume=dclv.GetVolume()
print(vdw_volume,polar_volume,volume)




mtest=Draw.MolToImage(ef().to_2d_coords(rep_sel.convert_to('RDKIT'))[0], size=(600, 400), highlightAtoms=rep[1], highlightColor=(1, 0, 0))  # Red highlight
mtest

first_sel.residues[0].atoms[rep[1]].ids

u = mda.Universe(prmtop, crd,format="TRJ")

nma1 = gnm.closeContactGNMAnalysis(u,select='not water and not name I and not name Na+',cutoff=7.0)
nma1.run(backend='serial')

plt.hist(nma1.results['eigenvalues'])
plt.xlabel('Eigenvalue')
plt.ylabel('Frequency')

ax = plt.plot(nma1.results['times'], nma1.results['eigenvalues'])
plt.xlabel('Time (ps)')
plt.ylabel('Eigenvalue')
plt.show()


sel=u.select_atoms('resid 1')
mol=sel.convert_to('RDKIT')
mol.GetAtomWithIdx(2).GetIntProp('_MDAnalysis_index')
ep=ef(step_back_from_terminals=True).find_endpoints(mol)


mtest=Draw.MolToImage(ef().to_2d_coords(mol)[0], size=(600, 400), highlightAtoms=rep, highlightColor=(1, 0, 0))  # Red highlight
mtest.show()

testsmi=Chem.MolFromSmiles('C1=CC(=CC=C1C2=CC=C(C=C2)C(=O)O)C(=O)O')
testsmi=Chem.AddHs(testsmi)
m2d,xy=ef().to_2d_coords(testsmi)
testm = Draw.MolToImage(m2d, size=(1200, 800), highlightAtoms=ef.find_endpoints(testsmi), highlightColor=(1, 0, 0))  # Red highlight
testm.show()

testm.save('4-(4-carboxyphenyl)benzoicacid_find_endpoints.png')
