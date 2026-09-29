# foundation/models/ema.py


import torch
from collections import OrderedDict
from contextlib import contextmanager

def _unwrap(model):
    """兼容 DDP/EMA 包裹，统一拿到真实模型"""
    return getattr(model, "module", model)

class EMA:
    """
    Exponential Moving Average for model parameters.
    - 仅跟踪可训练参数（requires_grad=True）
    - 支持 DDP: 自动从 model.module 取参数
    - 可选 FP32 影子副本（更稳的数值）
    - 提供上下文管理器，方便临时切换到 EMA 权重做验证/保存
    """

    def __init__(self, model, decay=0.999, use_num_updates=True, fp32_shadow=True):
        """
        Args:
            model: PyTorch 模型
            decay: EMA 衰减因子，越接近 1 越平滑（常用 0.999 ~ 0.9999）
            use_num_updates: 是否使用“步数热身”，前期自动减小 decay
            fp32_shadow: 影子权重用 FP32 维护（推荐 True，更稳）
        """
        if not (0.0 <= decay <= 1.0):
            raise ValueError("decay must be in [0, 1]")
        self.decay = float(decay)
        self.use_num_updates = bool(use_num_updates)
        self.num_updates = 0
        self.fp32_shadow = bool(fp32_shadow)

        m = _unwrap(model)
        # 记录需要跟踪的参数名，及其影子副本
        self.shadow = OrderedDict()
        for name, p in m.named_parameters():
            if p.requires_grad:
                t = p.detach().clone()
                if self.fp32_shadow:
                    t = t.float()  # 影子用 FP32，数值更稳
                self.shadow[name] = t

        # 用于临时切换权重（验证/保存）
        self._backup = None

    @torch.no_grad()
    def _current_decay(self):
        """带步数热身的 decay：前期小，后期趋近 self.decay（timm 风格）"""
        if not self.use_num_updates:
            return self.decay
        self.num_updates += 1
        warm = (1 + self.num_updates) / (10 + self.num_updates)
        return min(self.decay, warm)

    @torch.no_grad()
    def update(self, model):
        """在每次 optimizer.step() 之后调用，更新影子权重"""
        m = _unwrap(model)
        d = self._current_decay()
        one_m = 1.0 - d
        for name, p in m.named_parameters():
            if name not in self.shadow or (not p.requires_grad):
                continue
            s = self.shadow[name]
            # 对齐设备（dtype 若用 fp32_shadow 则保持 fp32，不跟随 p）
            if s.device != p.device:
                self.shadow[name] = s.to(p.device)
                s = self.shadow[name]
            # EMA: s = d*s + (1-d)*p
            if self.fp32_shadow:
                s.sub_(one_m * (s - p.float()))
            else:
                # 影子与参数同 dtype
                if s.dtype != p.dtype:
                    self.shadow[name] = s.to(p.dtype)
                    s = self.shadow[name]
                s.sub_(one_m * (s - p))

    @torch.no_grad()
    def copy_to(self, model):
        """将影子权重拷贝到模型（永久覆盖模型权重）"""
        m = _unwrap(model)
        for name, p in m.named_parameters():
            if name not in self.shadow:
                continue
            s = self.shadow[name]
            # 若影子是 fp32，但模型参数是 bf16/fp16，需要安全 cast
            if self.fp32_shadow and p.dtype != torch.float32:
                p.copy_(s.to(dtype=p.dtype, device=p.device))
            else:
                p.copy_(s.to(device=p.device))

    @torch.no_grad()
    def store(self, model):
        """保存当前模型权重，便于临时换 EMA 后再恢复"""
        m = _unwrap(model)
        self._backup = {n: p.detach().clone() for n, p in m.named_parameters() if n in self.shadow}

    @torch.no_grad()
    def restore(self, model):
        """恢复到 store() 时保存的权重"""
        if self._backup is None:
            return
        m = _unwrap(model)
        for n, p in m.named_parameters():
            if n in self._backup:
                p.copy_(self._backup[n])
        self._backup = None

    @contextmanager
    def average_parameters(self, model):
        """
        上下文管理器：临时把模型切换到 EMA 权重
        用法：
            with ema.average_parameters(model):
                validate(...)
        """
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    # ----- 可选：保存/加载（影子权重） -----
    def state_dict(self):
        return {
            "decay": self.decay,
            "use_num_updates": self.use_num_updates,
            "num_updates": self.num_updates,
            "fp32_shadow": self.fp32_shadow,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state):
        self.decay = float(state.get("decay", self.decay))
        self.use_num_updates = bool(state.get("use_num_updates", self.use_num_updates))
        self.num_updates = int(state.get("num_updates", self.num_updates))
        self.fp32_shadow = bool(state.get("fp32_shadow", self.fp32_shadow))
        # 影子权重尽量对齐已有 key；多余或缺失的 key 自动忽略/保留
        loaded = state.get("shadow", {})
        for k, v in loaded.items():
            self.shadow[k] = v.detach().clone()

class NoOpEMA:
    def update(self, *args, **kwargs): pass
    @contextmanager
    def average_parameters(self, model):  # 兼容 with 语法
        yield
    def state_dict(self): return {}
    def load_state_dict(self, *args, **kwargs): pass