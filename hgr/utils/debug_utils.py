# debug.utils.py

import os
import sys
import time
import wandb
import yaml
import re
import logging
import torch.nn as nn
from typing import Type

from contextlib import contextmanager


# 如果没有显式设置 WANDB_MODE：sweep 运行时默认 online，否则 disabled
if "WANDB_MODE" not in os.environ:
    os.environ["WANDB_MODE"] = "online" if os.environ.get("WANDB_SWEEP_ID") else "disabled"
WANDB_MODE = os.environ["WANDB_MODE"].lower()
# 全局调试标志
_DEBUG_ = os.environ.setdefault("_DEBUG_", 'true').lower() in  ["1", "true", "yes"]

logger = logging.getLogger(__name__)
_SCI_RE = re.compile(r"^[+-]?\d+(\.\d+)?[eE][+-]?\d+$")


class LevelBasedFormatter(logging.Formatter):
    """
    DEBUG/INFO 级别只打印消息本身；WARNING 及以上打印 [LEVEL] [logger] 前缀。
    """
    def format(self, record):
        if record.levelno <= logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "[%(levelname)s-%(name)s] %(message)s"
        return super().format(record)


def setup_logging():
    # 建议在入口函数处调用 setup_logging()
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if _DEBUG_ else logging.INFO)
    # 清空已有 handler，避免重复
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(LevelBasedFormatter())
    root.addHandler(handler)

    logger.info(f"DEBUG={_DEBUG_}, WANDB_MODE={WANDB_MODE}")



@contextmanager
def Timer(name="Running Time", enabled=_DEBUG_):
    """
    默认仅在_DEBUG_模式时才打印。可以在调用时手动传入 enabled=True/False
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        if enabled:
            end = time.perf_counter()
            print(f"⏱ {name}: {end - start:.4f} 秒")



@contextmanager
def suppress_stderr_old():
    """
    一个上下文，用于临时将 stderr 重定向到 os.devnull，
    以屏蔽所有在上下文内打印到 stderr 的内容。
    不拦截 C/C++ 层直接写到文件描述符 2（stderr）的内容：如果底层库（如 RDKit）用 std::cerr 或 fprintf(stderr, …) 输出，这些依然会出现在终端。
    """
    old_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        yield
        # 用 yield 明确分隔“进入”和“退出”阶段。
        # yield 之前的代码相当于 __enter__；yield 之后的 finally 代码相当于 __exit__。
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr


@contextmanager
def suppress_stderr():
    """
    在上下文中把 OS 层面的 stderr(fd=2) 重定向到 /dev/null，
    C/C++ 直接写 stderr 的都不会出现。
    """
    # 打开 /dev/null
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    # 复制当前 stderr(filedescriptor=2)
    old_stderr_fd = os.dup(2)
    try:
        # 将 fd=2 指向 /dev/null
        os.dup2(devnull_fd, 2)
        yield
    finally:
        # 恢复原来的 stderr
        os.dup2(old_stderr_fd, 2)
        # 关闭多余的 fd
        os.close(devnull_fd)
        os.close(old_stderr_fd)


def to_plain_config(x):
    """
    将 dict / EasyDict 递归转换为普通 dict，便于稳定地打印和写入 wandb。
    """
    x = x if isinstance(x, dict) else dict(x.items())

    if set(x) <= {"state", "dictitems"}:
        return to_plain_config(x.get("state") or x.get("dictitems") or {})

    out = {}
    for k, v in x.items():
        if k in ("state", "dictitems"):
            continue

        if isinstance(v, dict) or hasattr(v, "items"):
            out[k] = to_plain_config(v)
        else:
            out[k] = v
    return out


def flatten_config(d, prefix=""):
    """
    将嵌套 dict 展平成点号键，避免 wandb UI 同时出现 nested/flat 两套配置。
    """
    for k, v in d.items():
        kk = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and v:
            yield from flatten_config(v, kk)
        else:
            yield kk, v


def make_wandb_init_config(config):
    """
    统一生成写入 wandb.init(config=...) 的扁平配置。
    """
    return dict(flatten_config(to_plain_config(config)))


def update_wandb_config(config):
    # 当前 wandb 运行对象；未调用 wandb.init() 时通常为 None
    run = wandb.run

    # 是否处于 sweep 运行：
    # - run 不为 None 表示 wandb 已初始化
    # - run.sweep_id 不为 None 表示当前 run 来自某个 sweep（即参数由 sweep 注入/控制）
    is_sweep = bool(run) and getattr(run, "sweep_id", None) is not None

    # 记录基本运行信息，便于排查“为什么没生效/是不是 sweep”等问题
    logger.info(f"[wandb] wandb.run: {run}, sweep_id: {getattr(run, 'sweep_id', None)}")
    def set_by_path(root, path, val):
        """
        将 wandb.config 中形如 "train.lr" 的点号路径键写回到 config。

        设计目标：
        1) 只覆盖“config 中已存在的字段”，避免 sweep 错拼字段名导致意外新增配置项；
        2) 同时兼容 config 是 dict 或 EasyDict（EasyDict 支持属性访问 getattr/setattr）；
        3) 尝试把科学计数法字符串转换成 float，避免类型不一致。

        参数：
        - root: config 根对象（dict 或 EasyDict）
        - path: 形如 "train.lr" 的路径键
        - val : 需要写入的值
        """
        cur = root
        parts = path.split(".")

        # 逐层下钻到倒数第二层
        for p in parts[:-1]:
            cur = cur.get(p) if isinstance(cur, dict) else getattr(cur, p, None)
            if cur is None:
                return # 中途断掉说明 config 并没有这条路径，直接忽略该键

        # 最后一段 key 是父节点里的“叶子字段名”
        last = parts[-1]
        exists = (last in cur) if isinstance(cur, dict) else hasattr(cur, last)  
        if not exists:
            return # 只允许覆盖已有字段

        # sweep 注入的值如果是科学计数法字符串，尽量转成 float
        if isinstance(val, str) and _SCI_RE.match(val):
            try:
                val = float(val)
            except ValueError:
                pass

        # 写回目标字段
        if isinstance(cur, dict):
            cur[last] = val
        else:
            setattr(cur, last, val)

    # ------------------------ 主逻辑开始 ------------------------

    # 1) sweep 模式：wandb.config -> 本地 config
    # 仅在 sweep 时执行，目的：让后续训练逻辑使用“sweep 注入后的真实参数”
    # 且只覆盖 config 已有字段（避免意外新增）
    if is_sweep:
        for k, v in wandb.config.items():
            set_by_path(config, k, v)

    from hgr.utils.ckpt_manifest import resolve_gvae_ckpt_manifest_selection

    # 某些实验会先从连续 alpha 选择一个离散 ckpt，再进入训练。
    # 这一步需要在打印最终配置之前完成，保证日志里显示的是“真实生效”的 GVAE ckpt。
    resolve_gvae_ckpt_manifest_selection(config)

    # 2) 无论是否 sweep：打印当前（可能已被 sweep 覆盖后的）config
    # 目的：日志中只看到“一套最终生效的配置”，方便复现实验
    cfg = to_plain_config(config)
    title = "Sweep config" if is_sweep else "Config"
    logger.info(f"\n{'='*20} {title} {'='*20}")
    logger.info(yaml.dump(cfg, sort_keys=False, default_flow_style=False, allow_unicode=True))
    logger.info(f"{'='*20} End of Config {'='*20}\n")

    # 3) 本地 config -> wandb.config（写回 UI 展示）
    # 仍然只写“扁平键”，避免嵌套+扁平两套重复展示
    if run:
        flat = dict(flatten_config(cfg))

        # sweep 下 wandb.config 中由 sweep 注入的键通常是 locked 的：
        # 强行更新这些 key 会被 wandb 忽略或提示 locked by sweep
        # 因此这里仅补全 sweep 未设置的键（例如 data.name、train.epochs 等）
        if is_sweep:
            flat = {k: v for k, v in flat.items() if k not in wandb.config}

        wandb.config.update(flat, allow_val_change=True) # allow_val_change=True：允许在非 sweep 或某些情况下更新已有值

    # 返回更新后的 config（sweep 下已被覆盖）
    return config





def with_param_info(unit: str = "MB", precision: int = 2):
    """
    类装饰器：为 nn.Module 子类定制 __repr__，
    在 print(model) 时在类名括号内显示参数统计信息。

    参数:
      unit: 参数大小显示的单位 ('B', 'KB', 'MB')
      precision: 参数大小的小数精度，默认 2
    """
    scale_map = {"B": 1, "KB": 1024, "MB": 1024 ** 2}
    scale = scale_map.get(unit.upper(), 1024)
    unit = unit.upper()

    def decorator(cls: Type[nn.Module]):
        orig_repr = cls.__repr__ if "__repr__" in cls.__dict__ else nn.Module.__repr__

        def new_repr(self: nn.Module) -> str:
            # 统计参数
            total, trainable, bytes_total = 0, 0, 0
            for p in self.parameters(recurse=True):
                n = p.numel()
                total += n
                if p.requires_grad:
                    trainable += n
                bytes_total += n * p.element_size()
            size = bytes_total / scale

            # 构造参数信息字符串
            tag = f"[Params total={total}, trainable={trainable}, size={size:.{precision}f} {unit}]"

            # 若类实现了 _extra_param_info，可额外拼接
            if hasattr(self, "_extra_param_info") and callable(self._extra_param_info):  # type: ignore
                extra = self._extra_param_info()
                if extra:
                    tag += f", {extra}"

            # 原始 repr
            base = orig_repr(self)
            lines = base.splitlines()

            # 尝试在第一行括号内插入 tag
            if lines and lines[0].rstrip().endswith("("):
                lines[0] = lines[0][:-1] + f"({tag}"
                return "\n".join(lines)

            # 回退：追加在最后
            return base + "\n  " + tag

        cls.__repr__ = new_repr
        return cls

    return decorator
