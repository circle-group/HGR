# utils/cache.py

PREF_SCHEMA_VERSION = 1
CURV_SCHEMA_VERSION = 1

import os
import hashlib

def sha256_stream_smiles(smiles_list):
    """
    对整个 smiles_list 做流式 sha256。运行效率很高
    """
    h = hashlib.sha256()
    data = smiles_list

    for s in data:
        if s is None:
            s = ""
        # 统一换行作为分隔符，避免 ["ab","c"] 和 ["a","bc"] 之类的拼接歧义
        h.update(s.strip().encode("utf-8"))
        h.update(b"\n")

    # 加入长度信息作为附加保险（非必需，但便宜）
    h.update(str(len(smiles_list)).encode("utf-8"))
    return h.hexdigest()


def resolve_cache_dir(cache_dir=None, env_var: str = "RINGDIV_CACHE_DIR", app_name: str = "ringdiv") -> str:
    """
    cache_dir 默认优先级：
    1) cache_dir 参数
    2) 环境变量 RINGDIV_CACHE_DIR
    3) 默认位置 ~/.cache/ringdiv
    """
    if cache_dir:
        return os.path.abspath(os.path.expanduser(cache_dir))

    env_path = os.environ.get(env_var)
    if env_path:
        return os.path.abspath(os.path.expanduser(env_path))

    home = os.path.expanduser("~")
    return os.path.join(home, ".cache", app_name)