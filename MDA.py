import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from typing import List
from trajectory_deformation_workflow import EndpointAnalyzer as ea
from endpoints_finder import EndpointsFinder as ef
from rdkit.Chem import Draw
from MDAnalysis.analysis import gnm
import matplotlib.pyplot as plt

prmtop=r'C:\Users\zonezone\Desktop\YCU_research\BMMpM_ca.prmtop'
crd=r'C:\Users\zonezone\Desktop\YCU_research\BMMpM_mdcrd_v'
first_u= mda.Universe(prmtop, crd,format="TRJ")

last_u=mda.Universe(prmtop, crd,format="TRJ")
last_u.trajectory[-1]
first_u.trajectory[0]
first_sel=first_u.select_atoms('not water and not name I and not name Na+')
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

from volume_analyser import VolumeAnalyzer
VA=VolumeAnalyzer(first_u, selection='not water and not name I and not name Na+')
fig=VA.plot_interactive_3d()
fig.show()

VA.make_mark_occupancy_gif(frame_index=0,gif_path="occupancy_build_frame0.gif",atom_stride=5,voxel_stride=2,)

VA.make_volume_pipeline_gif(frame_index=0,gif_path="nanocube_volume_pipeline.gif",atom_stride=5,fps=)

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
VA=VolumeAnalyzer(u_from_rdkit, spacing=0.1,probe_radius=0.5,selection='all')
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

testsmi=Chem.MolFromSmiles('C1=CC=C(C2=CC=C(C3=C(C4=CC=CC=C4)C(C4=CC=C(C5=C[N+](C)=CC=C5)C=C4)=C(C4=CC=CC=C4)C(C4=CC=C(C5=C[N+](C)=CC=C5)C=C4)=C3C3=CC=CC=C3)C=C2)C=C1')
testsmi=Chem.AddHs(testsmi)

testsmi=Chem.MolFromSmiles('CCCCCCCCCC')
testsmi=Chem.AddHs(testsmi)


def To_2d_coords(mol:Chem.Mol):
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(mol.GetNumAtoms())])
    return mol, xy

from rdkit.Chem import Draw
m2d, xy = To_2d_coords(mol)
tm,txy = To_2d_coords(testsmi)

from scipy.spatial import ConvexHull
def get_endpointsId_from_2d_coords(xy):
    hull=ConvexHull(xy)
    return hull.vertices.tolist()

from rdkit.Chem.rdmolops import GetDistanceMatrix
def graph_farness(mol):
    D = GetDistanceMatrix(mol)
    return D.max(axis=1).astype(float)

    
def center_endpoints_oppsite_ends(xy, endpointsId):
    center = xy.mean(axis=0)
    endpoints = xy[endpointsId]
    vectors = center - endpoints
    opp = []
    for v in vectors:
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            opp.append(None)
            continue
        candidates = []
        for i, xy_i in enumerate(xy):
            opp_xy = xy_i - center
            norm_opp = np.linalg.norm(opp_xy)
            if norm_opp == 0:
                continue
            dot = np.dot(v, opp_xy)
            if dot > 0:  # other side of center
                cos = dot / (norm_v * norm_opp)
                angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
                if angle<15:
                    candidates.append((angle, norm_opp, i))  # angle, dist, id
        if candidates:
            candidates.sort(key=lambda x: (-x[1]))  # max dist
            best_i = candidates[0][2]
            opp.append(best_i)
        else:
            opp.append(None)
    return opp

def find_endpoints(mol: Chem.Mol):
    _, xy = To_2d_coords(mol)
    endpoints = get_endpointsId_from_2d_coords(xy)
    opp_ids = center_endpoints_oppsite_ends(xy, endpoints)
    # Filter None and extend
    opp_ids = [id for id in opp_ids if id is not None]
    endpoints.extend(opp_ids)
    endpoints = list(set(endpoints))
    return endpoints

mimage = Draw.MolToImage(m2d, size=(600, 400), highlightAtoms=center_endpoints_oppsite_ends(xy,get_endpointsId_from_2d_coords(xy)), highlightColor=(1, 0, 0))  # Red highlight
image = Draw.MolToImage(m2d, size=(600, 400), highlightAtoms=get_endpointsId_from_2d_coords(xy), highlightColor=(1, 0, 0))  # Red highlight
test=Draw.MolToImage(tm, size=(600, 400), highlightAtoms=get_endpointsId_from_2d_coords(txy), highlightColor=(1, 0, 0))  # Red highlight
mtest=Draw.MolToImage(tm, size=(600, 400), highlightAtoms=center_endpoints_oppsite_ends(txy,get_endpointsId_from_2d_coords(txy)), highlightColor=(1, 0, 0))  # Red highlight
mtest=Draw.MolToImage(tm, size=(600, 400), highlightAtoms=find_endpoints(tm), highlightColor=(1, 0, 0))  # Red highlight

from endpoints_finder import EndpointsFinder
ef=EndpointsFinder()

mimage = Draw.MolToImage(m2d, size=(1200, 800), highlightAtoms=ef._get_endpoints_id_from_2d_coords(xy), highlightColor=(1, 0, 0))  # Red highlight
mimage.save('endpoints_id_from_2d_coords.png')
mimage = Draw.MolToImage(m2d, size=(1200, 800), highlightAtoms=ef._center_endpoints_opposite_ends(xy, ef._get_endpoints_id_from_2d_coords(xy)), highlightColor=(1, 0, 0))  # Red highlight
mimage.save('center_endpoints_opposite_ends.png')
mimage = Draw.MolToImage(m2d, size=(1200, 800), highlightAtoms=ef.find_endpoints(mol), highlightColor=(1, 0, 0))  # Red highlight
mimage.save('find_endpoints.png')


endpoints=ef._get_endpoints_id_from_2d_coords(xy)
f = ef._graph_farness(m2d)
f = (f - f.min()) / (np.ptp(f) + 1e-12)
# re-rank hull tips by farness and keep top 80–100% (gentle pruning)
ranked = sorted(endpoints, key=lambda i: 0.2 * f[i] + (1 - 0.2) * 1.0, reverse=True)
endpoints = set(ranked)  # no strong pruning by default
mimage = Draw.MolToImage(m2d, size=(1200, 800), highlightAtoms=endpoints, highlightColor=(1, 0, 0))  # Red highlight
mimage.save('graph_farness.png')



endpoints=ef._get_endpoints_id_from_2d_coords(xy)
endpoints = set(ef._ring_spacing_filter(m2d, xy, list(endpoints)))
mimage = Draw.MolToImage(m2d, size=(1200, 800), highlightAtoms=endpoints, highlightColor=(1, 0, 0))  # Red highlight
mimage.save('ring_spacing_filter.png')
endpoints=ef._get_endpoints_id_from_2d_coords(xy)
endpoints = set(ef._step_back_from_terminals(m2d, list(endpoints)))
mimage = Draw.MolToImage(m2d, size=(1200, 800), highlightAtoms=endpoints, highlightColor=(1, 0, 0))  # Red highlight
mimage.save('step_back_from_terminals.png')

endpoints = set(ef._extend_to_ring_atoms(m2d, list(endpoints)))
mimage = Draw.MolToImage(m2d, size=(1200, 800), highlightAtoms=endpoints, highlightColor=(1, 0, 0))  # Red highlight
mimage.save('extend_to_ring_atoms.png')

ef=EndpointsFinder(step_back_from_terminals=True,extend_to_ring_atoms=False)
testsmi=Chem.MolFromSmiles('C1=CC(=CC=C1C2=CC=C(C=C2)C(=O)O)C(=O)O')
testsmi=Chem.AddHs(testsmi)
m2d,xy=To_2d_coords(testsmi)
testm = Draw.MolToImage(m2d, size=(1200, 800), highlightAtoms=ef.find_endpoints(testsmi), highlightColor=(1, 0, 0))  # Red highlight
testm.show()

testm.save('4-(4-carboxyphenyl)benzoicacid_find_endpoints.png')

