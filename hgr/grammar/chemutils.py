import rdkit
import random
import itertools
from rdkit import Chem
from rdkit.Chem import rdFMCS
from collections import deque

MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':5, 'O':2, 'P':5, 'S':6, 'Na': 1 } #, 'Se':4, 'Si':4}


lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)
    return mol

def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol)
    return mol

def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

def sanitize(mol, kekulize=True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol

def valence_check(atom, bt):
    cur_val = sum([bond.GetBondTypeAsDouble() for bond in atom.GetBonds()])
    return cur_val + bt <= MAX_VALENCE[atom.GetSymbol()]

def get_leaves(mol):
    leaf_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetDegree() == 1]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append( set([a1,a2]) )

    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(rings)

    leaf_rings = []
    for r in rings:
        inters = [c for c in clusters if r != c and len(r & c) > 0]
        if len(inters) > 1: continue
        nodes = [i for i in r if mol.GetAtomWithIdx(i).GetDegree() == 2]
        leaf_rings.append( max(nodes) )

    return leaf_atoms + leaf_rings

def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()

def bond_match(mol1, a1, b1, mol2, a2, b2):
    a1,b1 = mol1.GetAtomWithIdx(a1), mol1.GetAtomWithIdx(b1)
    a2,b2 = mol2.GetAtomWithIdx(a2), mol2.GetAtomWithIdx(b2)
    return atom_equal(a1,a2) and atom_equal(b1,b2)

def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap: 
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

#mol must be RWMol object
def get_sub_mol(mol, sub_atoms):
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms: continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx(): #each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()

def find_clusters(mol):
    n_atoms = mol.GetNumAtoms()
    # 处理特殊情况：单原子分子
    if n_atoms == 1:
        return [(0,)], [[0]]

    # 识别线性化学键（非环）
    clusters = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds() if not bond.IsInRing()]

    # 3. 识别环结构
    ssr = [tuple(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(ssr)

    # 4. 生成原子到子结构的映射
    # atom_cls[a] 存储原子 a 参与的所有子结构索引。
    atom_cls = [[] for _ in range(n_atoms)]
    for i, cluster in enumerate(clusters):
        for atom in cluster:
            atom_cls[atom].append(i)

    return clusters, atom_cls

def bfs_select(clusters, atom_cls, start_cls, n_atoms, blocked=[], return_cls=False):
    blocked = set(blocked)
    selected = set()
    selected_atoms = set()
    queue = deque([start_cls]) 

    while len(queue) > 0 and len(selected_atoms) < n_atoms:
        x = queue.popleft()
        selected.add(x)
        selected_atoms.update(clusters[x])
        for a in clusters[x]:
            for y in atom_cls[a]:
                if y in selected or y in blocked: continue
                queue.append(y)

    selected_atoms = [a for cls in selected for a in clusters[cls]]
    selected_atoms = set(selected_atoms)
    if return_cls:
        return selected, selected_atoms
    else:
        return selected_atoms

def random_subgraph(mol, ratio):
    n_atoms = mol.GetNumAtoms()
    clusters, atom_cls = find_clusters(mol)
    start_cls = random.randrange(len(clusters))
    return bfs_select(clusters, atom_cls, start_cls, n_atoms * ratio)

"""
def dual_random_subgraph(mol, n_atoms):
    clusters, atom_cls = find_clusters(mol)
    block_cls = list( range(len(clusters)) )
    random.shuffle(block_cls)

    for k in block_cls:
        blocked = set( [j for a in clusters[k] for j in atom_cls[a]] )
        if len(blocked) == len(clusters): continue
        
        start_cls1 = random.choice( [i for i in range(len(clusters)) if i not in blocked] )
        sg1_cls, sg1_atoms = bfs_select(clusters, atom_cls, start_cls1, n_atoms=1000, blocked=blocked, return_cls=True)
        if len(sg1_atoms) < n_atoms: continue
       
        blocked.update(sg1_cls)
        if len(blocked) == len(clusters): continue

        start_cls2 = random.choice( [i for i in range(len(clusters)) if i not in blocked] )
        sg2_cls, sg2_atoms  = bfs_select(clusters, atom_cls, start_cls2, n_atoms=1000, blocked=blocked, return_cls=True)
        if len(sg2_atoms) < n_atoms: continue

        blocked = set( [j for a in clusters[k] for j in atom_cls[a]] )
        sg1 = bfs_select(clusters, atom_cls, start_cls1, n_atoms, blocked=blocked)
        sg2 = bfs_select(clusters, atom_cls, start_cls2, n_atoms, blocked=blocked)
        return sg1 | sg2
    
    if n_atoms <= 2:
        return None
    else:
        return dual_random_subgraph(mol, int(n_atoms * 0.8))
"""

def dual_random_subgraph(mol, ratio):
    clusters, atom_cls = find_clusters(mol)
    best_size = 0
    best_block_atom = None

    for atom in mol.GetAtoms():
        blocked_cls = set( atom_cls[atom.GetIdx()] )
        blocked_atoms = set( [a for cls in blocked_cls for a in clusters[cls]] )
        if len(blocked_atoms) <= 1: continue

        components = []
        nei_cls = set([cls for a in blocked_atoms for cls in atom_cls[a]]) - blocked_cls
        for start_cls in nei_cls:
            if start_cls in blocked_cls: continue  # blocked_cls is changing
            sg_cls, sg_atoms = bfs_select(clusters, atom_cls, start_cls, n_atoms=1000, blocked=blocked_cls, return_cls=True)
            components.append( (start_cls, sg_cls, sg_atoms) )
            blocked_cls.update(sg_cls)

        if len(components) < 2: continue
        components = sorted(components, key=lambda x:len(x[1]), reverse=True)
        
        if len(components[1][2]) > best_size: # second_component_atoms
            best_size = len(components[1][2])
            best_block_atom = atom.GetIdx()
            best_components = components
    
    if best_block_atom is None:
        return set()

    # recompute with best block atom
    blocked_cls = set( atom_cls[best_block_atom] )
    selected_atoms = set()
    for start_cls, comp_cls, comp_atoms in best_components:
        n_atoms = len(comp_atoms) * ratio
        sg_atoms = bfs_select(clusters, atom_cls, start_cls, n_atoms, blocked=blocked_cls)
        selected_atoms.update(sg_atoms)
        blocked_cls.update(comp_cls)

    return selected_atoms


def enum_subgraph(mol, ratio_list):
    n_atoms = mol.GetNumAtoms()
    clusters, atom_cls = find_clusters(mol)

    selection = []
    for start_cls in range(len(clusters)):
        for ratio in ratio_list:
            x = bfs_select(clusters, atom_cls, start_cls, n_atoms * ratio)
            selection.append(x)
    return selection


def extract_subgraph(smile_or_mol, selected_atoms):
    """ 提取selected_atoms对应的子图

    参数:
        smiles (str): 分子的 SMILES 表示。
        selected_atoms (list or set): 包含原子索引的列表或集合，表示要保留的原子。

    返回:
        tuple: (subgraph, subgraph_mapped, roots)
            - subgraph: 通过 SMILES 转换回的 RDKit Mol 对象（确保分子完整且无断裂）。
            - subgraph_mapped: 带有原子映射信息的子图 (RWMol) 对象。
            - anchors:  selected_atoms 中与外部原子相连的原子索引列表。
    """
    if isinstance(smile_or_mol, str):
        mol = Chem.MolFromSmiles(smile_or_mol)
        Chem.Kekulize(mol)
    elif isinstance(smile_or_mol, Chem.Mol):
        mol = smile_or_mol

    selected_atoms = set(selected_atoms)
    anchors = [] #记录selected_atoms中哪些有外部邻居
    atom_neighbors = {a.GetIdx(): set(nb.GetIdx() for nb in a.GetNeighbors()) for a in mol.GetAtoms()}

    # 判断每个选中原子是否有外部邻居，如果有则视为anchor (or root)节点。
    for idx in selected_atoms:
        atom = mol.GetAtomWithIdx(idx)
        if any(nb not in selected_atoms for nb in atom_neighbors[idx]):
            anchors.append(idx)

    sub_mol = Chem.RWMol(mol)                       # RWMol 允许修改原子结构
    for atom in sub_mol.GetAtoms():
        atom.SetIntProp('org_idx', atom.GetIdx())   # 记录原子原始索引

    # 标记根节点并处理芳香性
    for atom_idx in anchors:
        atom = sub_mol.GetAtomWithIdx(atom_idx)
        atom.SetAtomMapNum(1)
        #  找到该原子所有的 芳香键，然后进一步筛选出两端都属于 selected_atoms 的芳香键。
        aroma_bonds = [b for b in atom.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.AROMATIC and
                       b.GetBeginAtom().GetIdx() in selected_atoms and
                       b.GetEndAtom().GetIdx() in selected_atoms]
        if len(aroma_bonds) == 0:  # 如果这个原子的芳香键全部连接到外部原子，那么就 取消其芳香性 (atom.SetIsAromatic(False))。
            atom.SetIsAromatic(False)

    # 删除不在 sel_atoms 中的原子，逆序删除以防止索引改变带来的问题
    all_idx = set([a.GetIdx() for a in sub_mol.GetAtoms()])
    remove_atoms = all_idx - selected_atoms
    for idx in sorted(remove_atoms, reverse=True): #倒序删除，避免索引变动问题
        sub_mol.RemoveAtom(idx)

    # 更新属性缓存，并 对子图中所有原子进行“完整性芳香性”检测（★ 新增步骤 ★）
    sub_mol.UpdatePropertyCache()  # 更新属性，确保 GetRingInfo 正确

    ring_info = sub_mol.GetRingInfo()
    # 遍历每个原子，如果其标记为芳香，但不出现在任何环中，则清除芳香标记
    for atom in sub_mol.GetAtoms():
        if atom.GetIsAromatic() and not any(atom.GetIdx() in ring for ring in ring_info.AtomRings()):
            atom.SetIsAromatic(False)

    return sub_mol.GetMol()


def extract_subgraph_old(smiles, selected_atoms):
    """ 提取selected_atoms对应的子图

    参数:
        smiles (str): 分子的 SMILES 表示。
        selected_atoms (list or set): 包含原子索引的列表或集合，表示要保留的原子。

    返回:
        tuple: (subgraph, subgraph_mapped, roots)
            - subgraph: 通过 SMILES 转换回的 RDKit Mol 对象（确保分子完整且无断裂）。
            - subgraph_mapped: 带有原子映射信息的子图 (RWMol) 对象。
            - anchors:  selected_atoms 中与外部原子相连的原子索引列表。
    """
    def _build_subgraph(mol, selected_atoms):
        """
        mol：这是一个 RDKit 的 Mol 对象，表示一个分子。
    	selected_atoms：一个包含原子索引的列表或集合，表示要保留的原子。
        """
        selected_atoms = set(selected_atoms)
        anchors = [] #记录selected_atoms中哪些有外部邻居
        # 判断每个选中原子是否有外部邻居，如果有则视为anchor (or root)节点。
        for idx in selected_atoms:
            atom = mol.GetAtomWithIdx(idx)
            if any(nb.GetIdx() not in selected_atoms for nb in atom.GetNeighbors()):
                anchors.append(idx)

        new_mol = Chem.RWMol(mol)                       # RWMol 允许修改原子结构
        for atom in new_mol.GetAtoms():
            atom.SetIntProp('org_idx', atom.GetIdx())   # 记录原子原始索引

        # 标记根节点并处理芳香性
        for atom_idx in anchors:
            atom = new_mol.GetAtomWithIdx(atom_idx)
            atom.SetAtomMapNum(1)
            #  找到该原子所有的 芳香键，然后进一步筛选出两端都属于 selected_atoms 的芳香键。
            aroma_bonds = [b for b in atom.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.AROMATIC and
                           b.GetBeginAtom().GetIdx() in selected_atoms and
                           b.GetEndAtom().GetIdx() in selected_atoms]
            if len(aroma_bonds) == 0:  # 如果这个原子的芳香键全部连接到外部原子，那么就 取消其芳香性 (atom.SetIsAromatic(False))。
                atom.SetIsAromatic(False)

        # 删除不在 sel_atoms 中的原子，逆序删除以防止索引改变带来的问题
        remove_atoms = [a.GetIdx() for a in new_mol.GetAtoms() if a.GetIdx() not in selected_atoms]
        for idx in sorted(remove_atoms, reverse=True): #倒序删除，避免索引变动问题
            new_mol.RemoveAtom(idx)

        # 更新属性缓存，并 对子图中所有原子进行“完整性芳香性”检测（★ 新增步骤 ★）
        new_mol.UpdatePropertyCache()  # 更新属性，确保 GetRingInfo 正确
        ring_info = new_mol.GetRingInfo()
        # 遍历每个原子，如果其标记为芳香，但不出现在任何环中，则清除芳香标记
        for atom in new_mol.GetAtoms():
            if atom.GetIsAromatic() and not any(atom.GetIdx() in ring for ring in ring_info.AtomRings()):
                atom.SetIsAromatic(False)

        return new_mol.GetMol(), anchors

    mol = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(mol)
    subgraph_mapped, roots = _build_subgraph(mol, selected_atoms)
    return subgraph_mapped

    # try:
    #     # try with kekulization
    #     sub_smiles = Chem.MolToSmiles(subgraph_mapped, kekuleSmiles=True)
    #     assert '.' not in sub_smiles
    #     subgraph = Chem.MolFromSmiles(sub_smiles)
    #     # 验证该子图确实存在于原始分子中
    #     original_mol = Chem.MolFromSmiles(smiles)
    #     if subgraph is not None and original_mol.HasSubstructMatch(subgraph):
    #         return subgraph, subgraph_mapped, roots
    # except Exception as e:
    #     # If fails, try without kekulization
    #     print(f"selected atons:{selected_atoms} 使用 kekuleSmiles 模式失败，错误信息:", e)
    #
    # # subgraph_mapped, roots = _build_subgraph(mol, selected_atoms)
    # sub_smiles = Chem.MolToSmiles(subgraph_mapped)
    # assert '.' not in sub_smiles
    # subgraph = Chem.MolFromSmiles(sub_smiles)
    # if subgraph is not None:
    #     return subgraph, subgraph_mapped, roots
    # else:
    #     return None, subgraph_mapped, None



def find_fragments(mol):
    new_mol = Chem.RWMol(mol)
    for atom in new_mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())

    for bond in mol.GetBonds():
        if bond.IsInRing(): continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()

        if a1.IsInRing() and a2.IsInRing():
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

        elif a1.IsInRing() and a2.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(a1))
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetIdx())
            new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())

        elif a2.IsInRing() and a1.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(a2))
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a2.GetIdx())
            new_mol.AddBond(new_idx, a1.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
    
    new_mol = new_mol.GetMol()
    new_smiles = Chem.MolToSmiles(new_mol)

    hopts = []
    for fragment in new_smiles.split('.'):
        fmol = Chem.MolFromSmiles(fragment)
        indices = set([atom.GetAtomMapNum() for atom in fmol.GetAtoms()])
        fmol = get_clique_mol(mol, indices)
        fmol = sanitize(fmol, kekulize=False)
        fsmiles = Chem.MolToSmiles(fmol)
        hopts.append((fsmiles, indices))
    
    return hopts

def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol) 
    #if tmp_mol is not None: new_mol = tmp_mol
    return new_mol

def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
        #if bt == Chem.rdchem.BondType.AROMATIC and not aromatic:
        #    bt = Chem.rdchem.BondType.SINGLE
    return new_mol

def enum_root(smiles, num_decode):
    mol = Chem.MolFromSmiles(smiles)
    roots = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0]
    outputs = []
    for perm_roots in itertools.permutations(roots):
        if len(outputs) >= num_decode: break
        mol = Chem.MolFromSmiles(smiles)
        for i,a in enumerate(perm_roots):
            mol.GetAtomWithIdx(a).SetAtomMapNum(i + 1)
        outputs.append(Chem.MolToSmiles(mol))

    while len(outputs) < num_decode:
        outputs = outputs + outputs
    return outputs[:num_decode]

def unique_rationales(smiles_list):
    visited = set()
    unique = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        root_atoms = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() > 0:
                root_atoms += 1
                atom.SetAtomMapNum(1)

        smiles = Chem.MolToSmiles(mol)
        if smiles not in visited and root_atoms > 0:
            visited.add(smiles)
            unique.append(smiles)

    return unique

def merge_rationales(x, y):
    xmol = Chem.MolFromSmiles(x)
    ymol = Chem.MolFromSmiles(y)

    mcs = rdFMCS.FindMCS([xmol, ymol], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=1)
    if mcs.numAtoms == 0: return []

    mcs = Chem.MolFromSmarts(mcs.smartsString)
    xmatch = xmol.GetSubstructMatches(mcs, uniquify=False)
    ymatch = ymol.GetSubstructMatches(mcs, uniquify=False)
    
    joined = [__merge_molecules(xmol, ymol, mx, my) for mx in xmatch for my in ymatch]
    joined = [Chem.MolToSmiles(new_mol) for new_mol in joined if new_mol]
    return list(set(joined))


def __merge_molecules(xmol, ymol, mx, my):
    new_mol = Chem.RWMol(xmol)
    for i in mx:  # remove atom maps where overlap happens
        atom = new_mol.GetAtomWithIdx(i)
        atom.SetAtomMapNum(0)

    atom_map = {}
    for atom in ymol.GetAtoms():
        idx = atom.GetIdx()
        if idx in my:
            atom_map[idx] = mx[my.index(idx)]
        else:
            atom_map[idx] = new_mol.AddAtom( copy_atom(atom) )

    for bond in ymol.GetBonds():
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        bt = bond.GetBondType()
        a1, a2 = atom_map[a1.GetIdx()], atom_map[a2.GetIdx()]
        if new_mol.GetBondBetweenAtoms(a1, a2) is None:
            new_mol.AddBond(a1, a2, bt)

    new_mol = new_mol.GetMol()
    new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))
    if new_mol and new_mol.HasSubstructMatch(xmol) and new_mol.HasSubstructMatch(ymol):
        return new_mol
    else:
        return None

