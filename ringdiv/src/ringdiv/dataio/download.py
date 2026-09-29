# ringdiv.dataio.download.py


"""
只负责：下载 + 校验 + 返回路径
不负责：如何读 SMILES / 如何做 train/test split（那是 data_splits.py 的职责）
"""

import hashlib
import os

import numpy as np

from .paths import dataset_raw_dir
from .hf import hf_download_files


def _sha256_file(file_path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_test_idx(test_idx_npz_path):
    """
    Read a `*_test_idx.npz` file and return `(test_idx, meta)`.

    - test_idx: np.ndarray[int64]
    - meta: dict with optional fields, e.g. n_total, seed, test_ratio, smiles_col, property_sha256
    
    读取 test_idx.npz 并返回 dict：
    必含：
      - test_idx: np.ndarray[int64]
    可选 meta：
      - n_total, seed, test_ratio, smiles_col, property_sha256
    """
    if not test_idx_npz_path.endswith(".npz"):
        raise ValueError(f"Expected .npz test_idx file, got: {test_idx_npz_path}")

    z = np.load(test_idx_npz_path, allow_pickle=False)
    meta = {}
    for k in z.files:
        v = z[k]
        # np scalar -> python scalar，便于后续比较/打印
        if isinstance(v, np.ndarray) and v.shape == ():
            v = v.item()
        meta[k] = v

    test_idx = np.asarray(meta.pop("test_idx"), dtype=np.int64)

    # 统一字符串字段
    if "smiles_col" in meta:
        meta["smiles_col"] = str(meta["smiles_col"])
    if "property_sha256" in meta:
        meta["property_sha256"] = str(meta["property_sha256"])

    # 统一数值字段
    if "n_total" in meta:
        meta["n_total"] = int(meta["n_total"])
    if "seed" in meta:
        meta["seed"] = int(meta["seed"])
    if "test_ratio" in meta:
        meta["test_ratio"] = float(meta["test_ratio"])

    return test_idx, meta


def _validate_dataset_files(property_csv_path, test_idx_npz_path, check_sha256=True):
    """
    轻量校验：
    1) 若 npz 内有 n_total：检查 test_idx 是否越界
    2) 若 npz 内有 property_sha256 且 check_sha256=True：检查 property_csv 的 sha256 是否匹配
    """
    test_idx, meta = read_test_idx(test_idx_npz_path)

    if "n_total" in meta:
        n_total = meta["n_total"]
        if np.any(test_idx < 0) or np.any(test_idx >= n_total):
            raise ValueError(f"test_idx out of range [0, {n_total}) in {test_idx_npz_path}")

    if check_sha256 and "property_sha256" in meta:
        actual = _sha256_file(property_csv_path)
        expect = meta["property_sha256"]
        if actual != expect:
            raise ValueError(
                "property_sha256 mismatch!\n"
                f"- property_csv: {property_csv_path}\n"
                f"- expect sha256: {expect}\n"
                f"- actual sha256: {actual}\n"
                "This usually means you are using test_idx from a different csv export."
            )


def download_dataset(
    dataset_name,
    data_root=None,
    repo_id=None,    
    revision=None,
    force=False,
    validate=True,
    check_sha256=True,
):
    """
    从 HF 下载两个文件到 datasets/{name}/raw，并（可选）做校验。
    返回 dict：{ "raw_dir": ..., "property_csv": ..., "test_idx": ... }
    """
    dataset_name = dataset_name.lower()
    raw_dir = dataset_raw_dir(dataset_name, data_root=data_root)
    os.makedirs(raw_dir, exist_ok=True)

    property_filename = f"{dataset_name}_property.csv"
    test_idx_filename = f"{dataset_name}_test_idx.npz"
    property_csv_path = os.path.abspath(os.path.join(raw_dir, property_filename))
    test_idx_path = os.path.abspath(os.path.join(raw_dir, test_idx_filename))

    hf_download_files(
        repo_id=repo_id,
        filenames=[property_filename, test_idx_filename],
        local_dir=raw_dir,
        revision=revision,
        force=force,
    )

    if validate:
        _validate_dataset_files(property_csv_path, test_idx_path, check_sha256=check_sha256)

    return {"raw_dir": raw_dir, "property_csv": property_csv_path, "test_idx": test_idx_path}


def ensure_dataset(
    dataset_name,
    data_root=None,
    repo_id=None,
    revision=None, # 这是 Hugging Face Hub 的版本定位参数：可以指定 repo 的 branch/tag/commit（例如 "main"、某个 tag、某个 commit hash）
    validate=True,
    check_sha256=True,
    allow_hf_download=False,   # TODO: 在数据集发布后改成True
):
    """
    若本地两个文件都存在，则直接返回.
    若缺文件：
      - allow_hf_download=True  -> 走 HF 下载
      - allow_hf_download=False -> 直接报错（不联网）
    """
    dataset_name = dataset_name.lower()
    raw_dir = dataset_raw_dir(dataset_name, data_root=data_root)
    property_csv_path = os.path.abspath(os.path.join(raw_dir, f"{dataset_name}_property.csv"))
    test_idx_path = os.path.abspath(os.path.join(raw_dir, f"{dataset_name}_test_idx.npz"))

    if os.path.exists(property_csv_path) and os.path.exists(test_idx_path):
        if validate:
            _validate_dataset_files(property_csv_path, test_idx_path, check_sha256=check_sha256)
        return {"raw_dir": raw_dir, "property_csv": property_csv_path, "test_idx": test_idx_path}


    if not allow_hf_download:
        missing = []
        if not os.path.exists(property_csv_path):
            missing.append(property_csv_path)
        if not os.path.exists(test_idx_path):
            missing.append(test_idx_path)
        raise RuntimeError(
            "Dataset files missing locally and HF download is disabled.\n"
            f"- dataset_name: {dataset_name}\n"
            f"- expected raw dir: {raw_dir}\n"
            f"- missing files:\n  - " + "\n  - ".join(missing) + "\n\n"
            "Fix:\n"
            "1) Put the files under the raw dir above; OR\n"
            "2) Upload to Hugging Face and set allow_hf_download=True.\n"
            "   (Also ensure config.data.hf.repo_id / property_filename / test_idx_filename are correct.)"
        )

    if repo_id is None:
        raise RuntimeError(f"Please provided repo_id to download dataset!")

    # allow_hf_download=True 时才会走这里
    return download_dataset(
        dataset_name=dataset_name,
        data_root=data_root,
        repo_id=repo_id,
        revision=revision,
        force=False,
        validate=validate,
        check_sha256=check_sha256,
    )
