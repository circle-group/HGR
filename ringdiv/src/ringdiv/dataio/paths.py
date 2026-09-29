# ringdiv.dataio.paths.py

"""
统一管理数据集本地落盘位置。

优先级：
1) 环境变量 RINGDIV_DATASETS_DIR
2) 当前工作目录存在 ./datasets 时使用它（适配你在 repo 根目录运行 HGR）
3) fallback 到 ~/.cache/ringdiv/datasets（适配 pip 安装 ringdiv 后独立使用）
"""

import os


def default_datasets_root():
    env = os.getenv("RINGDIV_DATASETS_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))

    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, "datasets")):
        return os.path.abspath(os.path.join(cwd, "datasets"))

    return os.path.abspath(os.path.join(os.path.expanduser("~"), ".cache", "ringdiv", "datasets"))


def dataset_raw_dir(dataset_name, data_root=None):
    """
    返回 datasets/{name}/raw 的绝对路径。
    data_root=None 表示使用 default_datasets_root() 的策略。
    """
    if data_root is None:
        data_root = default_datasets_root()
    data_root = os.path.abspath(os.path.expanduser(data_root))
    return os.path.join(data_root, dataset_name, "raw")
