import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdDistGeom
import MDAnalysis as mda

def get_3d_coordinates_from_smiles(smiles: str) -> tuple[np.ndarray, list[str]]:
    # Parse SMILES and generate 3D coordinates
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates using ETKDGv3
    ps = rdDistGeom.ETKDGv3()
    ps.randomSeed = 0xa100f
    status = rdDistGeom.EmbedMolecule(mol, ps)
    if status != 0:
        # If ETKDGv3 fails, try basic embedding
        status = rdDistGeom.EmbedMolecule(mol)
        if status != 0:
            raise RuntimeError(f"Failed to generate 3D coordinates for SMILES: {smiles}")
    
    # Optimize geometry
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except:
        # If MMFF fails, try UFF
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except:
            pass  # Continue without optimization
    
    # Extract coordinates and elements
    conf = mol.GetConformer()
    n_atoms_mol = mol.GetNumAtoms()
    coords = np.zeros((n_atoms_mol, 3), dtype=float)
    elements_list = []
    
    for i in range(n_atoms_mol):
        pos = conf.GetAtomPosition(i)
        coords[i] = [pos.x, pos.y, pos.z]
        atom = mol.GetAtomWithIdx(i)
        elements_list.append(atom.GetSymbol())
    return mol, coords, elements_list


def smiles_to_universe(smiles: str) -> mda.Universe:
    mol, coords, elements_list = get_3d_coordinates_from_smiles(smiles)
    u = mda.Universe(mol)
    return u