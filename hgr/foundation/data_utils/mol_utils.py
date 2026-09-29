
import os
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors

def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False

def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]


def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list



def gen_robust_conformers_old(mol_conf, n_threads, seed = 22, min_required = 3):
    EMBED_POLICIES = [
    {"num_confs": 5, "prune": 0.25, "random_coords": False}, # 第一次：高质量，适度去重
    {"num_confs": 12, "prune": 0.1, "random_coords": True}, # 第二次：降低去重阈值 + 随机坐标
    {"num_confs": 25, "prune": -1.0, "random_coords": True}, # 第三次：关闭去重 + 随机坐标 + 更多尝试
    ]

    def _try_embed(num_confs, prune, random_coords):
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.pruneRmsThresh = prune  # <0 关闭去重
        params.enforceChirality = True            # 显式开启手性约束
        params.useRandomCoords = random_coords
        try:
            params.numThreads = int(n_threads)    # 某些 RDKit 版本可能不支持，忽略即可
        except Exception:
            pass
        return AllChem.EmbedMultipleConfs(mol_conf, numConfs=num_confs, params=params)

    for policy in EMBED_POLICIES:
        _try_embed(**policy)
        if mol_conf.GetNumConformers() >= min_required:
            break
    
    return mol_conf.GetNumConformers()


def gen_robust_conformers(mol_conf, n_threads, seed=22, min_required=3):
    EMBED_POLICIES = [
        {"num_confs": 5, "prune": 0.25, "random_coords": False}, # 第一次：高质量，适度去重
        {"num_confs": 12, "prune": 0.1, "random_coords": True}, # 第二次：降低去重阈值 + 随机坐标
        {"num_confs": 25, "prune": -1.0, "random_coords": True}, # 第三次：关闭去重 + 随机坐标 + 更多尝试
    ]

    Chem.AssignStereochemistry(mol_conf, force=True, cleanIt=True)

    total = 0
    for try_idx, policy in enumerate(EMBED_POLICIES):
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed + 100*try_idx)  # 提升多样性
        params.pruneRmsThresh = policy["prune"]       # <0 关闭去重
        params.enforceChirality = True
        params.useRandomCoords = policy["random_coords"]
        # 对宏环/小环更友好
        params.useSmallRingTorsions = True
        params.useMacrocycleTorsions = True
        params.useExpTorsionAnglePrefs = True
        params.useBasicKnowledge = True
        # 也可考虑 params.maxAttempts = 1000（默认已较大）

        # 关键：把 numThreads 显式传入函数，而不是 params 里
        conf_ids = AllChem.EmbedMultipleConfs(
            mol_conf,
            numConfs=policy["num_confs"],
            params=params,
            # numThreads=int(n_threads)
        )
        total += len(conf_ids)

        # 早停：已经够用就不再尝试
        if mol_conf.GetNumConformers() >= min_required:
            break

    return mol_conf.GetNumConformers()

def _optimize_all_confs(mol_conf, n_threads, n_heavy=None):
    if n_heavy is None:
        n_heavy = sum(1 for a in mol_conf.GetAtoms() if a.GetAtomicNum() > 1)
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol_conf) # 计算可旋转键数

    # 以更“复杂”者为准
    if n_heavy > 80 or n_rot >= 8:
        maxIters = 1000
    elif n_heavy > 40 or n_rot >= 4:
        maxIters = 500
    else:
        maxIters = 300   

    """选择力场并优: MMFF 优化，不可用则回退 UFF """
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_conf):
            res = AllChem.MMFFOptimizeMoleculeConfs(mol_conf, maxIters=maxIters, numThreads=n_threads)
        else:
            res = AllChem.UFFOptimizeMoleculeConfs(mol_conf, maxIters=maxIters, numThreads=n_threads)
    except Exception:
        # 极少数异常再强制回退 UFF
        res = AllChem.UFFOptimizeMoleculeConfs(mol_conf, maxIters=maxIters, numThreads=n_threads)
    return [r[-1] for r in res]