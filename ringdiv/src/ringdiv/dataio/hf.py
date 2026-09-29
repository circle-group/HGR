# ringdiv.dataio.hf.py
"""
Hugging Face Hub 下载封装：把 repo 中指定文件下载到 local_dir。
"""

import os


def hf_download_files(repo_id, filenames, local_dir, revision=None, force=False):
    """
    参数：
    - repo_id: "username/repo"
    - filenames: list[str]
    - local_dir: 下载落盘目录（我们希望直接落到 datasets/{name}/raw）
    - revision: 可选（main / tag / commit）
    - force: True 则强制重新下载

    返回：下载后的本地文件路径列表（绝对路径）
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        raise RuntimeError("huggingface_hub is required. Please `pip install huggingface_hub`.") from e

    os.makedirs(local_dir, exist_ok=True)

    out_paths = []
    for fn in filenames:
        p = hf_hub_download(
            repo_id=repo_id,
            filename=fn,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 直接落盘成真实文件，方便你管理 datasets/
            revision=revision,
            force_download=force,
        )
        out_paths.append(os.path.abspath(p))
    return out_paths
