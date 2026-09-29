# hgr.utils.data_splits.py
"""
HGR 内部统一入口：load_train_test_smiles(config)

- ringdiv/ringdiv300k：调用 ringdiv.dataio 自动下载 + 两文件 split
- qm9/zinc250k/moses/guacamol：沿用你已有的读取方式（不打包进 ringdiv）
- 结果会缓存到 datasets/{data_name}/cache/{data_name}_train_test_smiles.pt，避免反复读 csv/下载
"""

import json
import os
import torch

from hgr.utils.file_utils import PathManager, load_smiles

import logging
logger = logging.getLogger(__name__)


def _cache_path(data_name):
    cache_dir = os.path.join(PathManager.DATA_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{data_name}_train_test_smiles.pt")


def load_train_test_smiles(config):
    data_name = config.data.name.lower()
    cache_path = _cache_path(data_name)

    # 1) 缓存命中：直接返回
    if os.path.exists(cache_path):
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        logger.info(f"Loaded cached SMILES from {cache_path}: {len(data['train'])} train / {len(data['test'])} test")
        return data["train"], data["test"]

    # 2) 构建 train/test
    if data_name in ["qm9", "zinc250k"]:
        full_smile_path = os.path.join(PathManager.DATA_DIR, config.path.raw_data)
        valid_idx_path = os.path.join(PathManager.DATA_DIR, config.path.valid_idx)
        full_smiles_list = load_smiles(full_smile_path)

        with open(valid_idx_path) as f:
            test_idx = json.load(f)
        if data_name == "qm9":
            test_idx = list(map(int, test_idx["valid_idxs"]))

        train_idx = sorted(set(range(len(full_smiles_list))) - set(test_idx))
        test_smiles = [full_smiles_list[i] for i in test_idx]
        train_smiles = [full_smiles_list[i] for i in train_idx]

    elif data_name in ["moses", "guacamol"]:
        # 用 split 好的 train/test 文件
        train_smiles = load_smiles(os.path.join(PathManager.DATA_DIR, config.path.raw_data.train))
        test_smiles = load_smiles(os.path.join(PathManager.DATA_DIR, config.path.raw_data.test))

    elif data_name in ["ringdiv", "ringdiv300k"]:
        # 关键：ringdiv 系列走 ringdiv 包的下载+split
        from ringdiv import ensure_dataset, load_train_test_smiles as ringdiv_split

        # config.data.hf.repo_id 这种访问方式 OK
        repo_id = None # config.data.hf.repo_id TODO: 后面更新
        revision = None # getattr(config.data.hf, "revision", None)

        # 让数据落到 ASSET_ROOT/datasets，避免写回源码仓库。
        ensure_dataset(
            dataset_name = data_name,
            data_root=PathManager.DATA_ROOT,
            repo_id=repo_id,
            revision=revision,
            allow_hf_download=False,  # <- 关键：暂时禁用
        )

        train_smiles, test_smiles = ringdiv_split(
            dataset_name = data_name,
            data_root=PathManager.DATA_ROOT,
            smiles_col="SMILES",
        )

    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    # 3) 写缓存，后续秒开
    torch.save({"train": train_smiles, "test": test_smiles}, cache_path)
    logger.info(f"Cached SMILES to {cache_path}: {len(train_smiles)} train / {len(test_smiles)} test")
    return train_smiles, test_smiles
