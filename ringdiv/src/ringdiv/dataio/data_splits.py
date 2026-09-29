# ringdiv.dataio.data_splits.py

"""
只负责：从 ringdiv 的两个文件（property.csv + test_idx.npz）读取 full/train/test SMILES。

注意：
- 我们只读 property.csv 的 SMILES 列，不读其它属性列，避免 109MB 全量读入。
- split 方式：按 test_idx（基于行号）划分，因此它与 “当前 csv 的行顺序” 绑定。
"""
import os
import numpy as np
import pandas as pd

from .download import read_test_idx
from .paths import dataset_raw_dir

def load_full_smiles(dataset_name, data_root=None, smiles_col="SMILES"):
    """
    返回全量 SMILES（按 csv 当前行顺序）。
    """
    raw_dir = dataset_raw_dir(dataset_name, data_root=data_root)
    property_csv_path = os.path.join(raw_dir, f"{dataset_name}_property.csv")

    s = pd.read_csv(property_csv_path, usecols=[smiles_col])[smiles_col]
    return s.astype(str).tolist()


def load_train_test_smiles(dataset_name, data_root=None, smiles_col="SMILES"):
    """
    返回 (train_smiles, test_smiles)
    """
    raw_dir = dataset_raw_dir(dataset_name, data_root=data_root)
    property_csv_path = os.path.join(raw_dir, f"{dataset_name}_property.csv")
    test_idx_npz_path = os.path.join(raw_dir, f"{dataset_name}_test_idx.npz")

    test_idx, meta = read_test_idx(test_idx_npz_path)

    full_smiles = pd.read_csv(property_csv_path, usecols=[smiles_col])[smiles_col].astype(str).tolist()

    # 如果 npz 中有 n_total，就做一致性检查
    if "n_total" in meta and meta["n_total"] != len(full_smiles):
        raise ValueError(
            f"n_total mismatch: npz says {meta['n_total']} but csv has {len(full_smiles)} rows.\n"
            f"- csv: {property_csv_path}\n"
            f"- idx: {test_idx_npz_path}"
        )

    mask = np.zeros(len(full_smiles), dtype=bool)
    mask[test_idx] = True

    test_smiles = [full_smiles[i] for i in np.where(mask)[0]]
    train_smiles = [full_smiles[i] for i in np.where(~mask)[0]]
    return train_smiles, test_smiles
