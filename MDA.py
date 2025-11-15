import MDAnalysis as mda
from MDAnalysis.analysis import align, rms
import nglview as nv
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
last_u = mda.Universe(prmtop, crd,format="TRJ")
last_u.trajectory[-1]
first_u=mda.Universe(prmtop, crd,format="TRJ")
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

mtest=Draw.MolToImage(ef().to_2d_coords(rep_sel.convert_to('RDKIT'))[0], size=(600, 400), highlightAtoms=rep[1], highlightColor=(1, 0, 0))  # Red highlight
mtest.show()

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
    m = Chem.Mol(mol)
    AllChem.Compute2DCoords(m)
    conf = m.GetConformer()
    xy = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y] for i in range(m.GetNumAtoms())])
    return m, xy

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

    
def center_endpoints_oppsite_ends(xy, endpointsId: List[int]):
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


w = nv.show_mdanalysis(sel)
t = nv.MDAnalysisTrajectory(u)
u
from trajectory_deformation_workflow import (
    faces_from_atomname_blocks,
    gsa_nanocube_metrics,
)

faces = faces_from_atomname_blocks(
    u,
    gsa_reslabel="MOL",   # the resname of the GSA, can be changed to the actual resname
    n_faces=6
)

df = gsa_nanocube_metrics(
    u,
    face_sel_list=faces,
    guest_sel=None,
    out_prefix="gsa_amber_demo"
)


center,atoms,endpoints=endpoints_finder(u, sel_str='resid 1')
al=nv.show_mdanalysis(sel)
al.add_representation("licorice", selection='not 1', color='grey',radiusScale=3.0)
al.add_representation("licorice", selection='@{}'.format(",".join(str(i) for i in endpoints[:4])), color='purple',radiusScale=3.0)

def residue_planar_rms(atoms):
    P = atoms.positions
    if len(P) < 3:
        return np.nan
    P0 = P - P.mean(axis=0)
    U, S, Vt = np.linalg.svd(P0, full_matrices=False)
    # normal is Vt[-1]; distance of points to plane through mean with this normal
    face_normal_vector = Vt[-1]
    dist = np.abs(P0 @ face_normal_vector)
    planar_rms = float(np.sqrt((dist**2).mean()))
    return planar_rms,face_normal_vector

planar_rms,face_normal_vector=residue_planar_rms(u.select_atoms('resid 1'))
print(planar_rms,face_normal_vector)

#ref = mda.Universe(r'C:\Users\zonezone\Desktop\YCU_research\BMMpM_ca.prmtop', r'C:\Users\zonezone\Desktop\YCU_research\mdcrd_v',format="TRJ")
#u.trajectory[-1]
#ref.trajectory[0]
#
#ca = u.select_atoms('not water')
#ref_ca = ref.select_atoms('not water')
#
#unaligned_rmsd = rms.rmsd(ca.positions, ref_ca.positions, superposition=False)
#print(f"Unaligned RMSD: {unaligned_rmsd:.2f}")
#aligner = align.AlignTraj(u, ref, select='not water', in_memory=True).run()
#
#If you don’t have enough memory to do that, write the trajectory out to a file and reload it into MDAnalysis (uncomment the cell below).
# aligner = align.AlignTraj(u, ref, select='not water',
#                           filename='aligned_to_first_frame.dcd').run()
# u = mda.Universe(PSF, 'aligned_to_first_frame.dcd')


#Creating an average structure
average = align.AverageStructure(u, u, select='not water',
                                 ref_frame=0).run()
ref = average.results.universe

