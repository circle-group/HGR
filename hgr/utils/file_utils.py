# utils/file_utils.py

import gzip
import pickle
import os
import re
import logging
import socket
import pandas as pd
import lzma
from pathlib import Path
import csv
from typing import Dict


logger = logging.getLogger(__name__)


_RUNTIME_PATH_ATTRS = (
    "ASSET_ROOT",
    "DATA_ROOT",
    "RESULTS_ROOT",
    "CKPT_ROOT",
    "LOG_ROOT",
    "CONFIG_RESOLVED_ROOT",
    "TMP_ROOT",
    "WANDB_DIR",
)


def sanitize_filename(name: str, replacement: str = "_") -> str:
    """Replace characters that are invalid on common filesystems."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', replacement, name).replace(" ", replacement)


def ensure_parent_dir(path: str) -> str:
    """Create the parent directory for a file path when needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def _expand_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _default_asset_root() -> str:
    repo_root = Path(__file__).resolve().parent.parent.parent
    return str(repo_root.parent / f"{repo_root.name}-runtime")


def get_runtime_paths(asset_root: str = None) -> Dict[str, str]:
    """Derive canonical runtime paths from one asset root."""
    if asset_root is None:
        asset_root = os.environ.get("ASSET_ROOT", _default_asset_root())
    asset_root = _expand_path(asset_root)
    return {
        "ASSET_ROOT": asset_root,
        "DATA_ROOT": os.path.join(asset_root, "datasets"),
        "RESULTS_ROOT": os.path.join(asset_root, "results"),
        "CKPT_ROOT": os.path.join(asset_root, "checkpoints"),
        "LOG_ROOT": os.path.join(asset_root, "logs"),
        "CONFIG_RESOLVED_ROOT": os.path.join(asset_root, "config_resolved"),
        "TMP_ROOT": os.path.join(asset_root, "tmp"),
        "WANDB_DIR": os.path.join(asset_root, "wandb", "runs"),
    }


def resolve_ckpt_path(ckpt_path: str) -> str:
    if os.path.isabs(ckpt_path):
        return ckpt_path
    return os.path.join(get_runtime_paths()["CKPT_ROOT"], ckpt_path)

class PathManagerMeta(type):
    """
    Metaclass to prevent accessing PathManager attributes before initialization.
    """
    def __getattribute__(cls, name):
        # Always allow internal and init access
        allowed = {'__class__', '__name__', '__module__', '__doc__',
                   'init', '_initialized',
                   'PRO_ROOT', 'CFG_ROOT', }
        if name not in allowed and not super().__getattribute__('_initialized'):
            raise RuntimeError("PathManager not initialized. Call PathManager.init(exp_name, data_name) first.")
        return super().__getattribute__(name)

class PathManager(metaclass=PathManagerMeta):
    """
    管理项目路径。
    _ROOT 结尾的路径可以直访问:
        PRO_ROOT, CFG_ROOT, ASSET_ROOT, DATA_ROOT, RESULTS_ROOT, CKPT_ROOT, LOG_ROOT, WANDB_DIR。
    _DIR 结尾的路径，访问前必须调用 PathManager.init(exp_name, data_name):
        DATA_DIR, CKPT_DIR。
        init后，可通过 PathManager.DATA_DIR 等直接访问。
    """
    HOSTNAME = socket.gethostname().split('.')[0]  # 不同设备缓存的数据格式可能不兼容
    PRO_ROOT = Path(__file__).resolve().parent.parent.parent
    CFG_ROOT = os.path.join(PRO_ROOT, 'configs')


    EXP_NAME = None  # 新增：全局实验名
    DATA_NAME = None
    _initialized = False

    @classmethod
    def init(cls, data_name, exp_name=None):
        if cls._initialized:
            assert exp_name == cls.EXP_NAME and data_name == cls.DATA_NAME,\
                f"PathManager already initialized with data_name={cls.DATA_NAME}, exp_name={cls.EXP_NAME}, cannot reset to {exp_name}"
            return
        cls._initialized = True
        cls.DATA_NAME = data_name
        cls.EXP_NAME = exp_name
        suffix = [exp_name] if exp_name else []

        runtime_paths = get_runtime_paths()

        for attr_name in _RUNTIME_PATH_ATTRS:
            setattr(cls, attr_name, runtime_paths[attr_name])

        cls.DATA_DIR = os.path.join(cls.DATA_ROOT, cls.DATA_NAME)
        cls.CKPT_DIR = os.path.join(cls.CKPT_ROOT, cls.DATA_NAME, *suffix)

        managed_paths = [
            cls.ASSET_ROOT,
            cls.DATA_ROOT,
            cls.RESULTS_ROOT,
            cls.CKPT_ROOT,
            cls.LOG_ROOT,
            cls.CONFIG_RESOLVED_ROOT,
            cls.TMP_ROOT,
            cls.WANDB_DIR,
            cls.DATA_DIR,
            cls.CKPT_DIR,
        ]
        for dir_path in managed_paths:
            os.makedirs(dir_path, exist_ok=True)

        logger.info(f"[INFO] Running exp for {data_name}, Host: {cls.HOSTNAME}, Root dir: {cls.PRO_ROOT}, Exp_name: {exp_name}")




class _ModuleRedirectUnpickler(pickle.Unpickler):
    def __init__(self, file_obj, module_map=None):
        super().__init__(file_obj)
        self._module_map = module_map or {}

    def find_class(self, module, name):
        if module in self._module_map:
            module = self._module_map[module]
        elif module.startswith("grammar."):
            module = module.replace("grammar.", "hgr.grammar.", 1)
        return super().find_class(module, name)


def load_pickle(path, verbose=True):
    if verbose:
        print(f"load file from: {path}")
    # with gzip.open(path, 'rb') as f:
    #     return pickle.load(f)
    
    redirected = False
    with gzip.open(path, 'rb') as f:
        try:
            obj = pickle.load(f)
        except ModuleNotFoundError as exc:
            # Backward-compat for legacy pickles saved under the old top-level "grammar" package.
            if "No module named 'grammar'" not in str(exc):
                raise
            f.seek(0)
            obj = _ModuleRedirectUnpickler(f, module_map={"grammar": "hgr.grammar"}).load()
            redirected = True
    if redirected:
        try:
            dump_pickle(path, obj)
            logger.info("Rewrote legacy pickle with updated module paths: %s", path)
        except Exception as dump_exc:
            logger.warning("Failed to rewrite legacy pickle at %s: %s", path, dump_exc)
    return obj


def dump_pickle(path, data):
    """使用gzip压缩方式保存pickle数据到文件(.pklz)"""
    try:
        with gzip.open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"save file to: {path}")
        return
    except Exception as e_gz:
        logger.warning(f"[dump_pickle: gzip 保存失败] path={path}, 错误: {e_gz}")
        # 如果是磁盘空间等写入异常，继续往下尝试 lzma

    # 2.  尝试用 lzma（xz）格式保存
    try:
        # 将后缀改为 .xz 或 .pklxz，方便区分
        base, ext = os.path.splitext(path)
        lzma_path = base + ".pklxz"
        with lzma.open(lzma_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"[dump_pickle] gzip 失败，已改用 lzma 保存到: {lzma_path}")
        return
    except Exception as e_xz:
        logger.error(f"[dump_pickle: lzma 保存也失败] path={lzma_path}, 错误: {e_xz}")
        raise RuntimeError(f"pickle 保存失败：gzip 错误: {e_gz}；lzma 错误: {e_xz}")


def load_smiles(fname, smiles_col='smile'):
    """
    通用 SMILES 读取函数，支持纯文本和 CSV 格式。

    参数：
        fname: 文件路径（支持 .txt, .csv 等）
        smiles_col: CSV 中 SMILES 所在列名（默认 'smiles'）

    返回：
        smile_list: List[str]，提取到的 SMILES 列表
    """
    print(f"[INFO] Reading SMILES from {os.path.abspath(fname)}")
    ext = os.path.splitext(fname)[-1].lower()

    if ext in ['.txt', '.smi', '.smiles']:
        # 每行一个 SMILES
        with open(fname, 'r') as fin:
            smile_list = [line.strip() for line in fin if line.strip()]
        return smile_list

    elif ext == '.csv':
        # Case 1: CSV 格式，有 SMILES 列
        df = pd.read_csv(fname, dtype=str)
        column = next((c for c in (smiles_col, 'smiles', 'SMILES') if c in df.columns), None)
        if column is not None:
            return df[column].dropna().astype(str).tolist()
        
        
        # Case 2. fallback 到无 header 单列
        df = pd.read_csv(fname, dtype=str, header=None)
        if df.shape[1] != 1:
            raise ValueError(f"[ERROR] CSV has {df.shape[1]} columns but no '{smiles_col}' header; expected exactly 1 column.")
        return df.iloc[:, 0].dropna().astype(str).tolist()

    else:
        raise ValueError(f"[ERROR] Unsupported file format: {ext}")


# def stream_smiles_old(fname, smiles_col='smile'):
#     """
#     流式读取SMILES文件，支持.txt和.csv格式，返回一个生成器和数据总数。

#     参数:
#         path (str): 文件路径。
#         smiles_col (str): 当文件是CSV时，指定SMILES所在的列名。

#     返回:
#         tuple: (一个只生成SMILES字符串的生成器, 文件中的SMILES总数)
#     """
#     filepath = Path(fname) # 以面向对象的方式处理路径
#     ext = os.path.splitext(fname)[-1].lower()

#     if not filepath.is_file():
#         logger.error(f"Input file not found: {fname}")
#         return (item for item in []), 0

#     # 1. 快速计算总行数
#     with open(fname, 'rb') as f:
#         total_lines = sum(1 for _ in f)

#     # 2. 定义处理不同文件类型的内部生成器，逻辑分离更清晰
#     def _txt_generator():
#         with filepath.open('r', encoding='utf-8') as f:
#             for line in f:
#                 smile = line.strip()
#                 if smile:
#                     yield smile

#     def _csv_generator():
#         with filepath.open('r', newline='', encoding='utf-8') as f:
#             reader = csv.reader(f)
#             try:
#                 header = next(reader)
#                 col_idx = header.index(smiles_col)
#             except (StopIteration, ValueError):
#                 logger.error(
#                     f"CSV file is empty or column '{smiles_col}' not found in {fname}. Header: {header if 'header' in locals() else 'N/A'}")
#                 return

#             for row in reader:
#                 if len(row) > col_idx:
#                     smile = row[col_idx].strip()
#                     if smile:
#                         yield smile  # 【修改】只生成SMILES字符串

#     # 3. 根据文件类型选择生成器和计算最终数量
#     if ext in ['.txt', '.smi']:
#         return _txt_generator(), total_lines
#     elif ext == '.csv':
#         data_count = total_lines - 1 if total_lines > 0 else 0
#         return _csv_generator(), data_count
#     else:
#         logger.error(f"Unsupported file format for streaming: {ext}")
#         return (item for item in []), 0








def stream_smiles(fname, smiles_col="smile"):
    """
    流式读取 .txt/.smi/.csv 文件，只生成 SMILES 字符串。

    返回：
        (generator, total_count)
        - generator: 迭代返回 SMILES 字符串
        - total_count: 估计的 SMILES 总数
            * .txt/.smi：总行数
            * .csv：
                - 如果有 header 并包含 smiles_col：总行数 - 1
                - 如果无 header 且单列：总行数
    """
    def _empty_gen():
        """返回一个空的生成器。"""
        return iter(())
    
    if not os.path.isfile(fname):
        return _empty_gen(), 0

    ext = os.path.splitext(fname)[-1].lower()
    with open(fname, "rb") as f:
        total_lines = sum(1 for _ in f) # 用二进制模式统计行数。

    # ---------- 处理 .txt/.smi ----------
    if ext in (".txt", ".smi", ".smiles"):
        def gen_txt():
            with open(fname, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s:
                        yield s
        return gen_txt(), total_lines

    # ---------- 处理 .csv ----------
    elif ext == ".csv":
        # 先读一行，判断是 header 模式还是单列模式
        with open(fname, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            first = next(reader, None)

        if first is None:  # 空文件
            return _empty_gen(), 0

        # 情况 A：有 header，且包含 smiles_col
        column = next((c for c in (smiles_col, "smiles", "SMILES") if c in first), None)
        if column is not None:
            col_idx = first.index(column)
            data_count = max(total_lines - 1, 0)

            def gen_csv_header():
                with open(fname, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # 跳过 header
                    for row in reader:
                        if len(row) > col_idx:
                            s = row[col_idx].strip()
                            if s:
                                yield s
            return gen_csv_header(), data_count

        # 情况 B：无 header，必须是单列
        if len(first) != 1:
            return _empty_gen(), 0

        data_count = total_lines

        def gen_csv_single_col():
            with open(fname, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 1:
                        # 一旦发现多列，停止生成（保持和原逻辑一致）
                        return
                    s = row[0].strip()
                    if s:
                        yield s
        return gen_csv_single_col(), data_count
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # ---------- 其他格式 ----------
    return _empty_gen(), 0
