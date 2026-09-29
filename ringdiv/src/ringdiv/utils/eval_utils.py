# utils/eval_utils.py

import networkx as nx
from rdkit import Chem
from multiprocessing import Pool

def get_mol(smiles_or_mol):
    '''
    Loads SMILES/molecule into RDKit's object
    '''
    if isinstance(smiles_or_mol, str):
        if len(smiles_or_mol) == 0:
            return None
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            return None
        return mol
    return smiles_or_mol



def mol_to_nx_graph(mol: Chem.Mol) -> nx.Graph:
    """
    将 RDKit Mol 转成无权无属性的 nx.Graph。
    只保留拓扑结构（节点和边），不保存原子/键信息。
    """
    G = nx.Graph()

    # 添加节点：编号 0...(N-1)，不附加任何属性
    num_atoms = mol.GetNumAtoms()
    G.add_nodes_from(range(num_atoms))

    # 添加边：只根据键的两个端点，完全无权、无类型
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        G.add_edge(i, j)

    return G


def mols_to_nx(mols):
    """
    将 RDKit Mol 列表转成 nx.Graph 列表。保留原子/键信息。
    """
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

def mapper(n_jobs):
    '''
    Returns function for map call.
    If n_jobs == 1, will use standard map
    If n_jobs > 1, will use multiprocessing pool
    If n_jobs is a pool object, will return its map function
    '''
    if n_jobs == 1:
        def _mapper(*args, **kwargs):
            return list(map(*args, **kwargs))

        return _mapper
    if isinstance(n_jobs, int):
        pool = Pool(n_jobs)

        def _mapper(*args, **kwargs):
            try:
                result = pool.map(*args, **kwargs)
            finally:
                pool.terminate()
            return result

        return _mapper
    return n_jobs.map


def _get_unique_ordered_subset(data, n):
    """
    从 data 中取出前 n 个不重复的元素
    """
    seen = set()
    result = []
    for item in data:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
        if len(result) >= n:
            break
    return result
