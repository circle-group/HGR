import networkx as nx
from rdkit import Chem
from rdkit.Chem import MolToInchiKey
from rdkit.Chem.SaltRemover import SaltRemover
import numpy as np
import logging
logger = logging.getLogger(__name__)

def is_equal_mol(mol1, mol2) -> bool:
    """
    Check if two molecules are equal, ignoring explicit hydrogens like [nH].
    Accepts either SMILES strings or RDKit Mol objects.
    判断两个分子是否相等，支持忽略像 [nH] 这样的显式氢。
    可接受 SMILES 字符串或 RDKit Mol 对象作为输入。
    """

    # 1. Convert SMILES strings to RDKit Mol if needed
    if isinstance(mol1, str):
        mol1 = Chem.MolFromSmiles(mol1)
    if isinstance(mol2, str):
        mol2 = Chem.MolFromSmiles(mol2)
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES or molecule object.")
        # 无效的 SMILES 或 RDKit 分子对象

    # 2. First, compare by InChIKey (quick and canonical)
    # 2. 通过 InChIKey 快速比对（标准化结构）
    if MolToInchiKey(mol1) == MolToInchiKey(mol2):
        return True

    # 3. If different, compare normalized SMILES ignoring [nH]
    # 3. 若 InChIKey 不一致，尝试比较去除显式氢后的 SMILES
    # s1 = normalize_smiles_ignore_H(Chem.MolToSmiles(mol1, canonical=True))
    # s2 = normalize_smiles_ignore_H(Chem.MolToSmiles(mol2, canonical=True))


    #   若 InChIKey 不一致, 比较原始分子与解码分子是否同构（采用子结构匹配，并考虑立体化学）
    return mol1.HasSubstructMatch(mol2, useChirality=True) and mol2.HasSubstructMatch(mol1, useChirality=True)  # Return True if normalized SMILES match 忽略氢后相等则返回 True


def mols_to_nx(mols):
    nx_graphs = []
    for mol in mols:
        G = nx.Graph()

        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(),
                       label=atom.GetSymbol())
            #    atomic_num=atom.GetAtomicNum(),
            #    formal_charge=atom.GetFormalCharge(),
            #    chiral_tag=atom.GetChiralTag(),
            #    hybridization=atom.GetHybridization(),
            #    num_explicit_hs=atom.GetNumExplicitHs(),
            #    is_aromatic=atom.GetIsAromatic())

        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(),
                       bond.GetEndAtomIdx(),
                       label=int(bond.GetBondTypeAsDouble()))
            #    bond_type=bond.GetBondType())

        nx_graphs.append(G)
    return nx_graphs



# Initialize salt remover
salt_remover = SaltRemover()

def remove_salt_stereo(smiles_or_mol):

    try:
        mol = Chem.MolFromSmiles(smiles_or_mol) if isinstance(smiles_or_mol, str) else smiles_or_mol
        if mol is None:
            return np.nan

        # -------- 1. Strip salts --------
        mol = salt_remover.StripMol(mol, dontRemoveEverything=True)

        # -------- 2. Keep largest fragment at Mol level --------
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
        mol = max(frags, key=lambda m: m.GetNumAtoms())

        # -------- 3. 输出 canonical SMILES（无立体信息） --------
        final_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception as e:
        logger.warning(f"remove_salt_stereo failed: {e}")
        final_smiles = np.nan

    return final_smiles




