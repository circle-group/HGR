import math
import torch
import numpy as np
from itertools import compress

from rdkit import RDLogger, Chem
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)   # 仅保留 CRITICAL
RDLogger.DisableLog('rdApp.*') 

from sklearn.model_selection import StratifiedKFold
from rdkit.Chem.Scaffolds import MurckoScaffold
# 兼容不同 RDKit 版本的导入
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:
    from rdkit.Chem import rdMolStandardize  # 老版本


def pretrain_random_split(dataset,
                          task_idx=None,
                          null_value=0,
                          frac_train=0.8,
                          frac_valid=0.1,
                          seed=0,
                          smiles_list=None):
    """
    随机划分为 train/valid （两者占比之和需为 1.0）。
    若 task_idx 不为 None，则优先按照该任务的非空标签进行样本过滤。
    若提供 smiles_list（与输入 dataset 对齐），会返回对应切片。
    """
    # --- 校验比例 ---
    if not math.isclose(frac_train + frac_valid, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"frac_train + frac_valid must be 1.0, got {frac_train + frac_valid}")

    # --- 按 task_idx 过滤（若需要）---
    # 说明：这里会同步过滤 smiles_list，确保与 dataset 对齐
    if task_idx is not None:
        # y_all: [num_samples, num_tasks] 或 [num_samples]
        y_all = dataset.data.y
        if y_all.dim() == 1:
            y_task = y_all
        else:
            y_task = y_all[:, task_idx]

        non_null_mask = (y_task != null_value)
        keep_idx = non_null_mask.nonzero(as_tuple=True)[0]  # tensor of valid indices

        dataset = dataset[keep_idx]
        if smiles_list is not None:
            smiles_list = [smiles_list[i] for i in keep_idx.tolist()]

    # --- 随机划分（使用 torch，确保 seed 生效）---
    num_mols = len(dataset)
    if num_mols == 0:
        raise ValueError("Empty dataset after filtering.")

    g = torch.Generator()
    g.manual_seed(int(seed))

    perm = torch.randperm(num_mols, generator=g)
    cut = int(frac_train * num_mols)
    train_idx = perm[:cut]
    valid_idx = perm[cut:]  # 因为 frac_train + frac_valid == 1

    # --- 生成切片 ---
    train_dataset = dataset[train_idx]
    valid_dataset = dataset[valid_idx]

    if smiles_list is None:
        return train_dataset, valid_dataset
    else:
        # 这里的 smiles_list 已经在 task 过滤时同步过（若发生）
        train_smiles = [smiles_list[i] for i in train_idx.tolist()]
        valid_smiles = [smiles_list[i] for i in valid_idx.tolist()]
        return train_dataset, valid_dataset, (train_smiles, valid_smiles)




# splitter function
def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold

    # --- test generate_scaffold ---
    s = 'Cc1cc(Oc2nccc(CCC)c2)ccc1'
    scaffold = generate_scaffold(s)
    assert scaffold == 'c1ccc(Oc2ccccn2)cc1'
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    def _mol_from_smiles_metal_safe(smi: str):
        m = Chem.MolFromSmiles(smi, sanitize=False)
        if m is None:
            return None
        # 断金属键（把金属从配体上断开）
        try:
            disconnector = rdMolStandardize.MetalDisconnector()
            m = disconnector.Disconnect(m)
        except Exception:
            pass
        # 只保留最大有机片段（去掉孤立的金属盐/小碎片）
        try:
            lfc = rdMolStandardize.LargestFragmentChooser()
            m = lfc.choose(m)
        except Exception:
            pass
        # 再做标准清洗
        try:
            Chem.SanitizeMol(m)
        except Exception:
            # 容错清洗：跳过属性/价态检查等
            try:
                Chem.SanitizeMol(
                    m,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
                )
            except Exception:
                return None
        return m
        
    # 先走“正常路径”
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        # 改走“金属安全路径”
        m = _mol_from_smiles_metal_safe(smiles)
        if m is None:
            # 还是不行 -> 放弃
            # print(f"[skip] bad smiles: {smiles}")
            return None

    try:
        # 旧代码中就下面一行
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=include_chirality)
    except Exception as e:
        print(f"Error generating scaffold for {smiles}: {e}")
        return None


def _build_scaffold_buckets(smiles_pairs, include_chirality=True):
    """
    smiles_pairs: List[(orig_idx, smi)]
    return:
      buckets: dict[scaffold_smiles] = sorted(list[orig_idx])
      bad_indices: list[orig_idx] that failed to generate scaffold
    """
    buckets = {}
    bad = []
    for i, smi in smiles_pairs:
        scaf = generate_scaffold(smi, include_chirality=include_chirality)
        # 原始代码没有检查 scaf 是否为 None，发现tox21加入了bad检查后test准确率大概提升了1个点
        # if not scaf:  # None 或空串 -> 跳过
        #     bad.append(i)
        #     continue
        buckets.setdefault(scaf, []).append(i)

    for k in list(buckets.keys()):
        buckets[k].sort()
    return buckets, bad


def _sorted_bucket_lists(buckets):
    """Return scaffold buckets as a list of index-lists, sorted by size desc then min-index asc."""
    return [ v for (k, v) in sorted(buckets.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)]

def scaffold_split(dataset, smiles_pairs, task_idx=None, null_value=0,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   return_smiles=False):
    """
    Adapted from https://github.com/deepchem/deepchem/blob/master/deepchem/splits/splitters.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param return_smiles:
    :return: train, valid, test slices of the input dataset obj. If
    return_smiles = True, also returns ([train_smiles_list],
    [valid_smiles_list], [test_smiles_list])
    """
    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    # 1) 构造 (orig_idx, smi) 列表（若指定 task_idx 则只保留非空样本）
    if task_idx != None:
        # filter based on null values in task_idx # 若指定了 task_idx，构造布尔掩码 non_null，仅保留该任务上非 null的样本。
        # get task array
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        # boolean array that correspond to non null values
        non_null = y_task != null_value
        smiles_pairs = list(compress(enumerate(smiles_pairs), non_null))
    else:
        smiles_pairs = list(enumerate(smiles_pairs))

    # 2) 按骨架分桶（跳过非法骨架）
    buckets, bad_indices = _build_scaffold_buckets(smiles_pairs, include_chirality=True)
    if not buckets:
        raise RuntimeError("No valid scaffolds could be generated; check your SMILES.")

    all_scaffold_sets = _sorted_bucket_lists(buckets)
    

    # 3) 计算配额（以有效样本数为分母）
    n_effective = sum(len(b) for b in all_scaffold_sets)
    print(f"Effective samples: {n_effective}/{len(smiles_pairs)}")
    train_cutoff = frac_train * n_effective
    valid_cutoff = (frac_train + frac_valid) * n_effective

    # 4) 整桶装配
    train_idx, valid_idx, test_idx = [], [], []
    for scaffold_set in all_scaffold_sets:
        if len(train_idx) + len(scaffold_set) > train_cutoff:
            if len(train_idx) + len(valid_idx) + len(scaffold_set) > valid_cutoff:
                test_idx.extend(scaffold_set)
            else:
                valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    # 5) 互斥检查
    st, sv, ss = set(train_idx), set(valid_idx), set(test_idx)
    assert st.isdisjoint(sv) and st.isdisjoint(ss) and sv.isdisjoint(ss)

    # 6) 切分
    train_dataset = dataset[torch.tensor(train_idx)]
    valid_dataset = dataset[torch.tensor(valid_idx)]
    test_dataset  = dataset[torch.tensor(test_idx)]

    if not return_smiles:
        return train_dataset, valid_dataset, test_dataset
    else:
        idx2smi = dict(smiles_pairs)  # 原始索引 -> SMILES
        train_smiles = [idx2smi[i] for i in train_idx]
        valid_smiles = [idx2smi[i] for i in valid_idx]
        test_smiles  = [idx2smi[i] for i in test_idx]
        return train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles)