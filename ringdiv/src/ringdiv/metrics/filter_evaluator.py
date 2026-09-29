
### Code adapted from https://github.com/molecularsets/moses

import os
import pandas as pd
from rdkit import Chem
import numpy as np
from ..utils.eval_utils import mapper, get_mol

_base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'statics', 'filter_moses')
_mcf = pd.read_csv(os.path.join(_base_dir, 'mcf.csv'))
_pains = pd.read_csv(os.path.join(_base_dir, 'wehi_pains.csv'), names=['smarts', 'names'])
combined_df = pd.concat([_mcf, _pains], ignore_index=True)
_filters = [Chem.MolFromSmarts(x)for x in combined_df['smarts'].values]

def mol_passes_filters(mol,
                       allowed=None,
                       isomericSmiles=False):
    """
    Checks if mol
    * passes MCF and PAINS filters,
    * has only allowed atoms
    * is not charged
    """
    allowed = allowed or {'C', 'N', 'S', 'O', 'F', 'Cl', 'Br', 'H'}
    mol = get_mol(mol)
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 0 and any(
            len(x) >= 8 for x in ring_info.AtomRings()
    ):
        return False
    h_mol = Chem.AddHs(mol)
    if any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
        return False
    if any(atom.GetSymbol() not in allowed for atom in mol.GetAtoms()):
        return False
    if any(h_mol.HasSubstructMatch(smarts) for smarts in _filters):
        return False
    smiles = Chem.MolToSmiles(mol, isomericSmiles=isomericSmiles)
    if smiles is None or len(smiles) == 0:
        return False
    if Chem.MolFromSmiles(smiles) is None:
        return False
    return True


def fraction_passes_filters(gen, n_jobs=1):
    """
    Computes the fraction of molecules that pass filters:
    * MCF
    * PAINS
    * Only allowed atoms ('C','N','S','O','F','Cl','Br','H')
    * No charges
    """
    passes = mapper(n_jobs)(mol_passes_filters, gen)
    return np.mean(passes)