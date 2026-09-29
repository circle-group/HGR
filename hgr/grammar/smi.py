# grammar/smi.py

import numpy as np
import networkx as nx
from copy import deepcopy
import logging
import traceback
# supress warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit import Chem
from rdkit.Chem import rdmolops
from .hypergraph import Hypergraph
from .symbol import TSymbol, NTSymbol, BondSymbol
logger = logging.getLogger(__name__)

def mol_to_bipartite(mol, kekulize):
    """
    get a bipartite representation of a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        molecule object

    Returns
    -------
    nx.Graph
        a bipartite graph representing which bond is connected to which atoms.
    """
    try:
        mol = standardize_stereo(mol)
    except KeyError:
        print(Chem.MolToSmiles(mol))
        raise KeyError

    # if kekulize:
    # Chem.Kekulize(mol)

    bipartite_g = nx.Graph()
    for each_atom in mol.GetAtoms():
        bipartite_g.add_node(f"atom_{each_atom.GetIdx()}",
                             atom_attr=atom_attr(each_atom, kekulize))

    for each_bond in mol.GetBonds():
        bond_idx = each_bond.GetIdx()
        bipartite_g.add_node(
            f"bond_{bond_idx}",
            bond_attr=bond_attr(each_bond, kekulize))
        bipartite_g.add_edge(
            f"atom_{each_bond.GetBeginAtomIdx()}",
            f"bond_{bond_idx}")
        bipartite_g.add_edge(
            f"atom_{each_bond.GetEndAtomIdx()}",
            f"bond_{bond_idx}")
    return bipartite_g


def mol_to_hg_old(mol, kekulize, add_Hs):
    """
    get a bipartite representation of a molecule.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        molecule object
    kekulize : bool
        kekulize or not
    add_Hs : bool
        add implicit hydrogens to the molecule or not.

    Returns
    -------
    Hypergraph
    """
    mol_org = mol
    if add_Hs:
        mol = Chem.AddHs(mol)

    # if kekulize:
    # Chem.Kekulize(mol)

    bipartite_g = mol_to_bipartite(mol, kekulize)
    hg = Hypergraph()
    for each_atom in [each_node for each_node in bipartite_g.nodes()
                      if each_node.startswith('atom_')]:
        node_set = set([])
        for each_bond in bipartite_g.adj[each_atom]:
            hg.add_node(each_bond, attr_dict=bipartite_g.nodes[each_bond]['bond_attr'])
            node_set.add(each_bond)
            hg.add_node
        hg.add_edge(node_set, attr_dict=bipartite_g.nodes[each_atom]['atom_attr'])
    return hg


def mol_to_hg(mol, kekulize=False, add_Hs=False):
    """
    将分子对象转换为超图表示：
    原子 → 超边（连接所有相连的键），键 → 节点

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        分子对象
    kekulize : bool
        是否进行Kekul化
    add_Hs : bool
        是否添加隐式氢原子

    Returns
    -------
    Hypergraph
    """


    if add_Hs:
        mol = Chem.AddHs(mol)
    # if kekulize:
    #     Chem.Kekulize(mol)

    # TODO: 下面一段代码原始的代码中没有呀，我是怎么加上的？？？（加不加都无所谓，没有考虑）
    # 标准化立体化信息
    try:
        mol = standardize_stereo(mol)
    except KeyError:
        print(Chem.MolToSmiles(mol))
        raise KeyError

    hg = Hypergraph()

    # 先为每个键添加超图节点
    for bond in mol.GetBonds():
        bond_id = f"bond_{bond.GetIdx()}"
        hg.add_node(bond_id, attr_dict=bond_attr(bond, kekulize))

    # 再为每个原子添加超边（连接该原子相连的所有键）
    for atom in mol.GetAtoms():
        # 得到该原子所连接的键的ID集合
        bond_ids = {f"bond_{bond.GetIdx()}" for bond in atom.GetBonds()}
        hg.add_edge(bond_ids, attr_dict=atom_attr(atom, kekulize))

    return hg


def hg_to_mol_direct(hg):
    """ Convert a Hypergraph into an RDKit Mol object.

    Parameters
    ----------
    hg : Hypergraph

    Returns
    -------
    mol : Chem.RWMol
    """
    rw_mol = Chem.RWMol()
    atom_dict = {}
    bond_set = set() # track processed bonds to avoid duplicates

    # 1) Add atoms
    for each_edge in hg.edges:
        atom_sym = hg.edge_attr(each_edge)['symbol']
        if isinstance(atom_sym, NTSymbol):
            raise ValueError("[NTSymbolError] NTSymbol still in the mol!")
        atom = Chem.Atom(atom_sym.symbol)
        atom.SetNumExplicitHs(atom_sym.num_explicit_Hs)
        atom.SetFormalCharge(atom_sym.formal_charge)
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[atom_sym.chirality])
        atom_idx = rw_mol.AddAtom(atom)
        atom_dict[each_edge] = atom_idx

    # 2) Add bonds
    for each_node in hg.nodes:
        edge_1, edge_2 = hg.adj_edges(each_node)
        if edge_1 + edge_2 in bond_set:  # 已处理过就跳过
            continue

        bond_sym = hg.node_attr(each_node)['symbol']  # BondSymbol 实例
        bt = bond_sym.bond_type  # 键阶 (int)

        if bt <= 3:
            rdkit_order = Chem.rdchem.BondType.values[bt]
        elif bt == 12:
            rdkit_order = Chem.rdchem.BondType.AROMATIC
        else:
            raise ValueError(f"Unsupported bond_type: {bt}")

        # 加键
        rw_mol.AddBond(atom_dict[edge_1], atom_dict[edge_2], order=rdkit_order)
        bond_obj = rw_mol.GetBondBetweenAtoms(atom_dict[edge_1], atom_dict[edge_2])

        # stereo
        bond_obj.SetStereo(Chem.rdchem.BondStereo.values[bond_sym.stereo])

        # is_aromatic
        if bond_sym.is_aromatic:
            bond_obj.SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(atom_dict[edge_1]).SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(atom_dict[edge_2]).SetIsAromatic(True)

        bond_set.update([edge_1 + edge_2, edge_2 + edge_1])

    # Finalize molecule
    rw_mol.UpdatePropertyCache()
    mol = rw_mol.GetMol()

    # Verify basic validity (这段加了修复机制)
    # if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None:
    #     mol = fix_mol(mol)
    #     if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None:
    #         raise RuntimeError(f'Invalid SMILE: {Chem.MolToSmiles(mol)}')

    # Verify basic validity
    if Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None:
        raise RuntimeError(f'Invalid SMILE: {Chem.MolToSmiles(mol)}')

    # 3) Attempt to assign stereochemistry
    not_stereo_mol = deepcopy(mol)
    try:
        mol = set_stereo(mol)
    except:
        # traceback.print_exc()
        raise RuntimeError(f"Failed to set stereo for mol: {Chem.MolToSmiles(mol)}")


    # 4) Re-aromatize if possible
    mol_tmp = deepcopy(mol)
    Chem.SetAromaticity(mol_tmp)
    # 如果重新设定的芳香结构有效，则使用
    if Chem.MolFromSmiles(Chem.MolToSmiles(mol_tmp)) is not None:
        mol = mol_tmp
    # 否则，如果原始 mol 也无效，回退到无立体信息版本
    elif Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is None:
        mol = not_stereo_mol

    mol.UpdatePropertyCache()

    return mol





def hg_to_mol_kekulize(hg):
    """ Convert a Hypergraph into an RDKit Mol object using Kekulé alternating bonds
        for rings that were originally marked aromatic (bt == 12 or bond_sym.is_aromatic == True).
        如果尝试的Kekulé交替导致配价错误，则恢复原始键类型，不将其视为芳香环。

    Parameters
    ----------
    hg : Hypergraph

    Returns
    -------
    mol_final : Chem.Mol（完整sanitization & aromaticity assignment 后的分子）
    """
    # 0) Prepare containers
    rw_mol = Chem.RWMol()
    atom_dict = {}
    seen_bonds = set()  # track processed bonds to avoid duplicates
    aromatic_bonds = set() # 记录“原始来自 bond_sym.bond_type==12 (is_aromatic) 的键”：用 (min_idx, max_idx) 作为键


    # 1) Add atoms (with explicit H, formal charge, chirality)
    for atom_edge in hg.edges:
        atom_sym = hg.edge_attr(atom_edge)['symbol']
        if isinstance(atom_sym, NTSymbol):
            raise ValueError("[NTSymbolError] NTSymbol still in the mol!")

        atom = Chem.Atom(atom_sym.symbol)
        atom.SetNumExplicitHs(atom_sym.num_explicit_Hs)
        atom.SetFormalCharge(atom_sym.formal_charge)
        atom.SetChiralTag(Chem.rdchem.ChiralType.values[atom_sym.chirality])

        atom_idx = rw_mol.AddAtom(atom)
        atom_dict[atom_edge] = atom_idx

    # 2) Add bonds (并记录哪些边是上游标记为“芳香”的)
    for bond_node in hg.nodes:
        e1, e2 = hg.adj_edges(bond_node)
        if e1 + e2 in seen_bonds:
            continue
        seen_bonds.update([e1 + e2, e2 + e1])
        bond_sym = hg.node_attr(bond_node)['symbol']  # BondSymbol instance
        bt = bond_sym.bond_type  # 键阶 (bt <= 3: 单/双/三键；bt == 12: 芳香标记)
        a1, a2 = atom_dict[e1], atom_dict[e2]
        bond_key = (min(a1, a2), max(a1, a2))

        # 决定初始rdkit添加时用的order
        if bt <= 3:
            rdkit_order = Chem.rdchem.BondType.values[bt]
        elif bt == 12:
            # bt == 12 或 bond_sym.is_aromatic == True 视为“打算芳香”，
            rdkit_order = Chem.rdchem.BondType.SINGLE # 暂时以单键加入，后面再做Kekulé交替
            aromatic_bonds.add(bond_key)
        else:
            raise ValueError(f"Unsupported bond_type: {bt}")


        rw_mol.AddBond(a1, a2, order=rdkit_order)
        bond_obj = rw_mol.GetBondBetweenAtoms(a1, a2)

        # stereo
        bond_obj.SetStereo(Chem.rdchem.BondStereo.values[bond_sym.stereo])



    # 3) 将 RWMol 转成 Mol，并做一次“轻量级”全局 sanitize，以保证后续环检测时 valence & 共轭 正常
    mol_tmp = rw_mol.GetMol()
    Chem.SanitizeMol(mol_tmp, sanitizeOps=(Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION))


    # 4) 对每个最小环（SSSR），检查是否包含“原始芳香键”；如果是，则尝试两种Kekulé交替
    # • 仅测试 valence 冲突，后续再全局 sanitize & aromaticity
    for ring in rdmolops.GetSymmSSSR(mol_tmp):
        ring = list(ring)
        n = len(ring)
        if n < 3:
            continue

        # 4.1) 检查该环是否含至少一条“原始芳香键”
        is_intended_aromatic = False
        for i in range(n):
            a1 = ring[i]
            a2 = ring[(i + 1) % n]
            key = (min(a1, a2), max(a1, a2))
            if key in aromatic_bonds:
                is_intended_aromatic = True
                break

        if not is_intended_aromatic:
            continue # 不是“原本想标芳香”的环，跳过

        # 4.2) 记录这条环上所有边的原始键顺序（用于回退）
        original_orders_this_ring = {}
        for i in range(n):
            a1 = ring[i]
            a2 = ring[(i + 1) % n]
            bnd = mol_tmp.GetBondBetweenAtoms(a1, a2)
            if bnd is not None:
                original_orders_this_ring[(min(a1, a2), max(a1, a2))] = bnd.GetBondType()

        # 4.3) 定义两个交替模式：模式1：偶数位置→双键/奇数位置→单键；模式2：偶数位置→单键/奇数位置→双键
        success = False
        for pattern in [0, 1]:  # pattern=0 表示 偶数->DOUBLE，奇数->SINGLE；pattern=1 反过来
            # 在临时副本上测试
            test_mol = Chem.RWMol(mol_tmp)
            could_apply = True
            # 对环上每条边进行赋值
            for i in range(n):
                a1 = ring[i]
                a2 = ring[(i + 1) % n]
                bnd = test_mol.GetBondBetweenAtoms(a1, a2)
                if bnd is None:
                    could_apply = False
                    break
                # 根据pattern选择键类型
                if (i % 2 == 0 and pattern == 0) or (i % 2 == 1 and pattern == 1):
                    new_type = Chem.rdchem.BondType.DOUBLE
                else:
                    new_type = Chem.rdchem.BondType.SINGLE
                bnd.SetBondType(new_type)

            if not could_apply:
                continue

            # 测试是否能 sanitize (仅valence & 共轭) 成功
            try:
                test_mol.UpdatePropertyCache()
                Chem.SanitizeMol(test_mol, sanitizeOps=(Chem.SanitizeFlags.SANITIZE_PROPERTIES | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION))
            except:
                # 模式1/2 不可用，回退到下一模式
                continue

            # 如果执行到这里，说明模式 valid，可接受，把 test_mol 应用到 mol_tmp
            mol_tmp = test_mol.GetMol()
            success = True
            break

        if not success:
            # 两种模式都失败，回退：不将其视为芳香，直接保持原样(好像不会发生)
            for (i1, i2), order in original_orders_this_ring.items():
                bnd = mol_tmp.GetBondBetweenAtoms(i1, i2)
                if bnd is not None:
                    bnd.SetBondType(order)


    # 5) 全局再一次完整 sanitize & aromaticity assignment
    Chem.SanitizeMol(mol_tmp, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL)
    Chem.SetAromaticity(mol_tmp)
    mol_tmp.UpdatePropertyCache()


    # 6) Verify basic validity：MolToSmiles -> MolFromSmiles
    if Chem.MolFromSmiles(Chem.MolToSmiles(mol_tmp)) is None:
        raise RuntimeError(f"Invalid SMILES: {Chem.MolToSmiles(mol_tmp)}")

    # 7) 赋立体信息
    not_stereo_mol = deepcopy(mol_tmp)
    try:
        mol_final = set_stereo(mol_tmp)
    except:
        traceback.print_exc()
        # raise RuntimeError(f"Failed to set stereo for mol: {Chem.MolToSmiles(mol_tmp)}")


    # 8) Re-aromatize if possible
    mol_tmp2 = deepcopy(mol_final)
    Chem.SetAromaticity(mol_tmp2)
    # 如果重新设定的芳香结构有效，则使用
    if Chem.MolFromSmiles(Chem.MolToSmiles(mol_tmp2)) is not None:
        mol_final = mol_tmp2
    # 否则，如果原始 mol 也无效，回退到无立体信息版本
    elif Chem.MolFromSmiles(Chem.MolToSmiles(mol_final)) is None:
        mol_final = not_stereo_mol

    mol_final.UpdatePropertyCache()
    return mol_final


def hg_to_mol(hg):
    """Wrapper: 先尝试 hg_to_mol_direct，若失败，则改用改进版 hg_to_mol_nkekulize。"""
    try:
        return hg_to_mol_direct(hg)
    except RuntimeError as e:
        return hg_to_mol_kekulize(hg)







def atom_attr(atom, kekulize):
    """
    get atom's attributes

    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
    kekulize : bool
        kekulize or not

    Returns
    -------
    atom_attr : dict
        "is_aromatic" : bool
            the atom is aromatic or not.
        "smarts" : str
            SMARTS representation of the atom.
    """
    if kekulize:
        return {'terminal': atom.GetAtomMapNum() != 1,
                'is_in_ring': atom.IsInRing(),
                'visited': False,
                'NT': False,
                'symbol': TSymbol(degree=0,
                                  is_aromatic=False,
                                  symbol=atom.GetSymbol(),
                                  num_explicit_Hs=atom.GetNumExplicitHs(),
                                  formal_charge=atom.GetFormalCharge(),
                                  chirality=atom.GetChiralTag().real
                                  )}
    else:
        return {'terminal': atom.GetAtomMapNum() != 1,
                'is_in_ring': atom.IsInRing(),
                'visited': False,
                'NT': False,
                'symbol': TSymbol(degree=0,
                                  is_aromatic=atom.GetIsAromatic(),
                                  symbol=atom.GetSymbol(),
                                  num_explicit_Hs=atom.GetNumExplicitHs(),
                                  formal_charge=atom.GetFormalCharge(),
                                  chirality=atom.GetChiralTag().real
                                  )}


def bond_attr(bond, kekulize):
    """
    get atom's attributes

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
    kekulize : bool
        kekulize or not

    Returns
    -------
    bond_attr : dict
        "bond_type" : int
        {0: rdkit.Chem.rdchem.BondType.UNSPECIFIED,
         1: rdkit.Chem.rdchem.BondType.SINGLE,
         2: rdkit.Chem.rdchem.BondType.DOUBLE,
         3: rdkit.Chem.rdchem.BondType.TRIPLE,
         4: rdkit.Chem.rdchem.BondType.QUADRUPLE,
         5: rdkit.Chem.rdchem.BondType.QUINTUPLE,
         6: rdkit.Chem.rdchem.BondType.HEXTUPLE,
         7: rdkit.Chem.rdchem.BondType.ONEANDAHALF,
         8: rdkit.Chem.rdchem.BondType.TWOANDAHALF,
         9: rdkit.Chem.rdchem.BondType.THREEANDAHALF,
         10: rdkit.Chem.rdchem.BondType.FOURANDAHALF,
         11: rdkit.Chem.rdchem.BondType.FIVEANDAHALF,
         12: rdkit.Chem.rdchem.BondType.AROMATIC,
         13: rdkit.Chem.rdchem.BondType.IONIC,
         14: rdkit.Chem.rdchem.BondType.HYDROGEN,
         15: rdkit.Chem.rdchem.BondType.THREECENTER,
         16: rdkit.Chem.rdchem.BondType.DATIVEONE,
         17: rdkit.Chem.rdchem.BondType.DATIVE,
         18: rdkit.Chem.rdchem.BondType.DATIVEL,
         19: rdkit.Chem.rdchem.BondType.DATIVER,
         20: rdkit.Chem.rdchem.BondType.OTHER,
         21: rdkit.Chem.rdchem.BondType.ZERO}
    """
    if kekulize:
        is_aromatic = False
        if bond.GetBondType().real == 12:
            bond_type = 1
        else:
            bond_type = bond.GetBondType().real
    else:
        is_aromatic = bond.GetIsAromatic()
        bond_type = 12 if is_aromatic else bond.GetBondType().real

    return {'symbol': BondSymbol(is_aromatic=is_aromatic,
                                 bond_type=bond_type,
                                 stereo=int(bond.GetStereo())),
            'is_in_ring': bond.IsInRing(),
            'visited': False}


def standardize_stereo(mol):
    '''
 0: rdkit.Chem.rdchem.BondDir.NONE,
 1: rdkit.Chem.rdchem.BondDir.BEGINWEDGE,
 2: rdkit.Chem.rdchem.BondDir.BEGINDASH,
 3: rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT,
 4: rdkit.Chem.rdchem.BondDir.ENDUPRIGHT,

    '''
    # recompute stereo (Kerim changes here)
    rdmolops.AssignStereochemistry(mol, force=True, cleanIt=True)

    # mol = Chem.AddHs(mol) # this removes CIPRank !!!
    for each_bond in mol.GetBonds():
        if int(each_bond.GetStereo()) in [2, 3]:  # 2=Z (same side), 3=E

            # Kerim changes here
            stereo_atoms = each_bond.GetStereoAtoms()
            if len(stereo_atoms) < 2:
                logging.info("stereo_atoms < 2")
                continue 
            
            begin_stereo_atom_idx = each_bond.GetBeginAtomIdx()
            end_stereo_atom_idx = each_bond.GetEndAtomIdx()
            atom_idx_1 = each_bond.GetStereoAtoms()[0]
            atom_idx_2 = each_bond.GetStereoAtoms()[1]
            if mol.GetBondBetweenAtoms(atom_idx_1, begin_stereo_atom_idx):
                begin_atom_idx = atom_idx_1
                end_atom_idx = atom_idx_2
            else:
                begin_atom_idx = atom_idx_2
                end_atom_idx = atom_idx_1

            begin_another_atom_idx = None
            assert len(mol.GetAtomWithIdx(begin_stereo_atom_idx).GetNeighbors()) <= 3
            for each_neighbor in mol.GetAtomWithIdx(begin_stereo_atom_idx).GetNeighbors():
                each_neighbor_idx = each_neighbor.GetIdx()
                if each_neighbor_idx not in [end_stereo_atom_idx, begin_atom_idx]:
                    begin_another_atom_idx = each_neighbor_idx

            end_another_atom_idx = None
            assert len(mol.GetAtomWithIdx(end_stereo_atom_idx).GetNeighbors()) <= 3
            for each_neighbor in mol.GetAtomWithIdx(end_stereo_atom_idx).GetNeighbors():
                each_neighbor_idx = each_neighbor.GetIdx()
                if each_neighbor_idx not in [begin_stereo_atom_idx, end_atom_idx]:
                    end_another_atom_idx = each_neighbor_idx

            ''' 
            relationship between begin_atom_idx and end_atom_idx is encoded in GetStereo
            '''
            begin_atom_rank = int(mol.GetAtomWithIdx(begin_atom_idx).GetProp('_CIPRank'))
            end_atom_rank = int(mol.GetAtomWithIdx(end_atom_idx).GetProp('_CIPRank'))
            try:
                begin_another_atom_rank = int(mol.GetAtomWithIdx(begin_another_atom_idx).GetProp('_CIPRank'))
            except:
                begin_another_atom_rank = np.inf
            try:
                end_another_atom_rank = int(mol.GetAtomWithIdx(end_another_atom_idx).GetProp('_CIPRank'))
            except:
                end_another_atom_rank = np.inf
            if begin_atom_rank < begin_another_atom_rank \
                    and end_atom_rank < end_another_atom_rank:
                pass
            elif begin_atom_rank < begin_another_atom_rank \
                    and end_atom_rank > end_another_atom_rank:
                # (begin_atom_idx +) end_another_atom_idx should be in StereoAtoms
                if each_bond.GetStereo() == 2:
                    # set stereo
                    each_bond.SetStereo(Chem.rdchem.BondStereo.values[3])
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 3)
                elif each_bond.GetStereo() == 3:
                    # set stereo
                    each_bond.SetStereo(Chem.rdchem.BondStereo.values[2])
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 4)
                else:
                    raise ValueError
                each_bond.SetStereoAtoms(begin_atom_idx, end_another_atom_idx)
            elif begin_atom_rank > begin_another_atom_rank \
                    and end_atom_rank < end_another_atom_rank:
                # (end_atom_idx +) begin_another_atom_idx should be in StereoAtoms
                if each_bond.GetStereo() == 2:
                    # set stereo
                    each_bond.SetStereo(Chem.rdchem.BondStereo.values[3])
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 0)
                elif each_bond.GetStereo() == 3:
                    # set stereo
                    each_bond.SetStereo(Chem.rdchem.BondStereo.values[2])
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 3)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 0)
                else:
                    raise ValueError
                each_bond.SetStereoAtoms(begin_another_atom_idx, end_atom_idx)
            elif begin_atom_rank > begin_another_atom_rank \
                    and end_atom_rank > end_another_atom_rank:
                # begin_another_atom_idx + end_another_atom_idx should be in StereoAtoms
                if each_bond.GetStereo() == 2:
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 3)
                elif each_bond.GetStereo() == 3:
                    # set bond dir
                    mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, begin_another_atom_idx, begin_stereo_atom_idx, 4)
                    mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 0)
                    mol = safe_set_bond_dir(mol, end_another_atom_idx, end_stereo_atom_idx, 4)
                else:
                    raise ValueError
                each_bond.SetStereoAtoms(begin_another_atom_idx, end_another_atom_idx)
            else:
                raise RuntimeError
    return mol


def set_stereo(mol):
    '''
 0: rdkit.Chem.rdchem.BondDir.NONE,
 1: rdkit.Chem.rdchem.BondDir.BEGINWEDGE,
 2: rdkit.Chem.rdchem.BondDir.BEGINDASH,
 3: rdkit.Chem.rdchem.BondDir.ENDDOWNRIGHT,
 4: rdkit.Chem.rdchem.BondDir.ENDUPRIGHT,
    '''
    _mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    Chem.Kekulize(_mol, True)
    substruct_match = mol.GetSubstructMatch(_mol)
    if not substruct_match:
        ''' mol and _mol are kekulized.
        sometimes, the order of '=' and '-' changes, which causes mol and _mol not matched.
        '''
        Chem.SetAromaticity(mol)
        Chem.SetAromaticity(_mol)
        substruct_match = mol.GetSubstructMatch(_mol)
    try:
        atom_match = {substruct_match[_mol_atom_idx]: _mol_atom_idx for _mol_atom_idx in
                      range(_mol.GetNumAtoms())}  # mol to _mol
    except:
        logger.warning('two molecules obtained from the same data do not match.')
        return mol
        # raise ValueError('two molecules obtained from the same data do not match.')

    for each_bond in mol.GetBonds():
        begin_atom_idx = each_bond.GetBeginAtomIdx()
        end_atom_idx = each_bond.GetEndAtomIdx()
        _bond = _mol.GetBondBetweenAtoms(atom_match[begin_atom_idx], atom_match[end_atom_idx])
        _bond.SetStereo(each_bond.GetStereo())

    mol = _mol
    for each_bond in mol.GetBonds():
        if int(each_bond.GetStereo()) in [2, 3]:  # 2=Z (same side), 3=E
            begin_stereo_atom_idx = each_bond.GetBeginAtomIdx()
            end_stereo_atom_idx = each_bond.GetEndAtomIdx()
            begin_atom_idx_set = set([each_neighbor.GetIdx()
                                      for each_neighbor
                                      in mol.GetAtomWithIdx(begin_stereo_atom_idx).GetNeighbors()
                                      if each_neighbor.GetIdx() != end_stereo_atom_idx])
            end_atom_idx_set = set([each_neighbor.GetIdx()
                                    for each_neighbor
                                    in mol.GetAtomWithIdx(end_stereo_atom_idx).GetNeighbors()
                                    if each_neighbor.GetIdx() != begin_stereo_atom_idx])
            if not begin_atom_idx_set:
                each_bond.SetStereo(Chem.rdchem.BondStereo(0))
                continue
            if not end_atom_idx_set:
                each_bond.SetStereo(Chem.rdchem.BondStereo(0))
                continue
            if len(begin_atom_idx_set) == 1:
                begin_atom_idx = begin_atom_idx_set.pop()
                begin_another_atom_idx = None
            if len(end_atom_idx_set) == 1:
                end_atom_idx = end_atom_idx_set.pop()
                end_another_atom_idx = None
            if len(begin_atom_idx_set) == 2:
                atom_idx_1 = begin_atom_idx_set.pop()
                atom_idx_2 = begin_atom_idx_set.pop()
                if int(mol.GetAtomWithIdx(atom_idx_1).GetProp('_CIPRank')) < int(
                        mol.GetAtomWithIdx(atom_idx_2).GetProp('_CIPRank')):
                    begin_atom_idx = atom_idx_1
                    begin_another_atom_idx = atom_idx_2
                else:
                    begin_atom_idx = atom_idx_2
                    begin_another_atom_idx = atom_idx_1
            if len(end_atom_idx_set) == 2:
                atom_idx_1 = end_atom_idx_set.pop()
                atom_idx_2 = end_atom_idx_set.pop()
                if int(mol.GetAtomWithIdx(atom_idx_1).GetProp('_CIPRank')) < int(
                        mol.GetAtomWithIdx(atom_idx_2).GetProp('_CIPRank')):
                    end_atom_idx = atom_idx_1
                    end_another_atom_idx = atom_idx_2
                else:
                    end_atom_idx = atom_idx_2
                    end_another_atom_idx = atom_idx_1

            if each_bond.GetStereo() == 2:  # same side
                mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 4)
                each_bond.SetStereoAtoms(begin_atom_idx, end_atom_idx)
            elif each_bond.GetStereo() == 3:  # opposite side
                mol = safe_set_bond_dir(mol, begin_atom_idx, begin_stereo_atom_idx, 3)
                mol = safe_set_bond_dir(mol, end_atom_idx, end_stereo_atom_idx, 3)
                each_bond.SetStereoAtoms(begin_atom_idx, end_atom_idx)
            else:
                raise ValueError
    return mol


def safe_set_bond_dir(mol, atom_idx_1, atom_idx_2, bond_dir_val):
    if atom_idx_1 is None or atom_idx_2 is None:
        return mol
    else:
        mol.GetBondBetweenAtoms(atom_idx_1, atom_idx_2).SetBondDir(Chem.rdchem.BondDir.values[bond_dir_val])
        return mol

