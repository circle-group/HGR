# foundation/models/sigreg_loss.py

"""
This file is adapted from the following repository:
https://github.com/rbalestr-lab/lejepa/blob/main/lejepa/univariate/epps_pulley.py
https://github.com/rbalestr-lab/lejepa/blob/main/lejepa/multivariate/slicing.py
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.nn import all_reduce as functional_all_reduce
from torch.distributed.nn import ReduceOp

# ==========================================
# 1. 基础工具: DDP Reduce
# ==========================================
def all_reduce(x, op="AVG"):
    """
    对 Tensor x 做 DDP all_reduce，并返回规约后的结果。
    - 单卡 / 非分布式：直接返回 x（保持行为一致）
    - 分布式：使用 torch.distributed.nn.all_reduce（functional）进行规约
    op : str 规约算子字符串, AVG/SUM/MAX

    """
    if dist.is_available() and dist.is_initialized():
        op_attr = getattr(ReduceOp, op.upper(), ReduceOp.AVG)
        return functional_all_reduce(x, op_attr)
    else:
        return x


# ==========================================
# 2. 核心统计量: Epps-Pulley Test (心脏)
# ==========================================
class EppsPulley(nn.Module):
    """
    Fast Epps-Pulley two-sample test statistic for univariate distributions.

    This implementation uses numerical integration over the characteristic function
    to compute a goodness-of-fit test statistic. The test compares the empirical
    characteristic function against a standard normal distribution.

    The statistic is computed as:
        T = N * ∫ |φ_empirical(t) - φ_normal(t)|² w(t) dt

    where φ_empirical is the empirical characteristic function, φ_normal is the
    standard normal characteristic function, and w(t) is an integration weight.

    Args:
        t_max (float, optional): Maximum integration point for linear spacing methods.
            Only used for 'trapezoid' and 'simpson' integration. Default: 3.
        n_points (int, optional): Number of integration points. Must be odd for
            'simpson' integration. For 'gauss-hermite', this determines the number
            of positive nodes. Default: 17.
        integration (str, optional): Integration method to use. One of:
            - 'trapezoid': Trapezoidal rule with linear spacing over [0, t_max]
            Default: 'trapezoid'.

    Attributes:
        t (torch.Tensor): Integration points (positive half, including 0).
        weights (torch.Tensor): Precomputed integration weights incorporating
            symmetry and φ(t) = exp(-t²/2).
        phi (torch.Tensor): Precomputed φ(t) = exp(-t²/2) values.
        integration (str): Selected integration method.
        n_points (int): Number of integration points.

    Notes:
        - The implementation exploits symmetry: only t ≥ 0 are computed, and
          contributions from -t are implicitly added via doubled weights.
        - For 'gauss-hermite', nodes and weights are adapted from the standard
          Gauss-Hermite quadrature to integrate against exp(-t²).
        - Supports distributed training via all_reduce operations.

    Example:
        >>> test = EppsPulley(t_max=5.0, n_points=21, integration='simpson')
        >>> samples = torch.randn(1000)  # Standard normal samples
        >>> statistic = test(samples)
        >>> print(f"Test statistic: {statistic.item():.4f}")


    快速 Epps–Pulley 单变量正态性检验统计量（用于 SIGReg 的“单变量核”）。

    目标
    ----
    给定一维样本 x（或多个切片并行的样本），衡量其分布与标准正态 N(0,1) 的差异。
    SIGReg 中通过“随机投影 slicing”把高维 embedding 投影成多个一维切片，
    然后对每个切片使用该统计量并聚合。

    核心思想（特征函数 CF）
    ---------------------
    - 经验特征函数（Empirical Characteristic Function, ECF）：
        φ̂(t) = (1/N) * Σ_j exp(i t x_j)
      其实部和虚部分别为：
        Re = mean(cos(t x))
        Im = mean(sin(t x))

    - 标准正态的特征函数：
        φ_N(t) = exp(-t^2 / 2)
      注意标准正态的 CF 是纯实数（虚部为 0）。

    - 统计量（简化表述）：
        T ≈ N * ∫ |φ̂(t) - φ_N(t)|^2 w(t) dt
      这里通过数值积分近似（梯形法），并利用对称性只算 t>=0。

    形状约定
    --------
    输入 x: [..., N, K]
      - N：样本数（在 SIGReg 场景里通常是 batch size）
      - K：切片数（num_slices），表示并行评估 K 个一维投影
    输出: [..., K]
      - 每个切片一个统计量值（越大表示偏离标准正态越明显）
    """

    def __init__(self, t_max: float = 3, n_points: int = 17):
        """
        参数
        ----
        t_max : float
            线性积分区间上限，只在 trapezoid 方案使用，积分点为 [0, t_max]
        n_points : int
            积分点数量（梯形法无强制奇偶性，但原始实现要求为奇数）
        """
        super().__init__()
        assert n_points % 2 == 1
        self.n_points = n_points

        # -------- (1) 预计算积分点 t（只取正半轴，含 0） --------
        # t: [P]，P=n_points
        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)

        # -------- (2) 梯形法权重 weights_trapz --------
        # 对称性处理：只在 t>=0 上积分，但权重中隐含了对负半轴的“翻倍”
        # weights 初值为 2*dt，相当于 (f(t)+f(-t)) 的合并；端点（0 和 t_max）半权重
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        # -------- (3) 标准正态 CF 的实部 phi(t)=exp(-t^2/2) --------
        # 同时把积分权重融合进去：weights_final = weights_trapz * phi(t)
        # 注意：phi 与 weights 都是 buffer，不参与训练
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())  # [P]
        self.register_buffer("weights", weights * self.phi)                  # [P]

    def forward(self, x):
        """
        输入
        ----
        x: [..., N, K]
          - N: 样本维（要在这个维度上取 mean 得到经验特征函数）
          - K: 并行切片数（每列是一组一维样本）

        输出
        ----
        stats: [..., K]
          - 每个切片对应一个 Epps–Pulley 统计量（标量）
        """
        N = x.size(-2) # N: 样本数

        # world_size：分布式总 rank 数，用于把统计量按“全局样本数”缩放
        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

        # -------- (1) 构造 t x 的广播张量 --------
        # t: [P]，x: [..., N, K]
        # x_t: [..., N, K, P]
        # 这里不需要 view，因为广播规则会自动对齐最后一维
        x_t = x.unsqueeze(-1) * self.t

        # -------- (2) 计算经验特征函数的 cos/sin 部分 --------
        # cos_vals, sin_vals: [..., N, K, P]
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        # -------- (3) 对样本维 N 求平均，得到 ECF 在各 t 点的估计 --------
        # 约定 dim=-3 是 N 维（因为最后两维是 K, P）
        # cos_mean, sin_mean: [..., K, P]
        cos_mean = cos_vals.mean(dim=-3)
        sin_mean = sin_vals.mean(dim=-3)

        # -------- (4) DDP 同步：使统计量基于“全局 batch” --------
        # 这里用 AVG：各 rank 的均值再平均，相当于全局样本的均值（前提各 rank N 相同）
        cos_mean = all_reduce(cos_mean)
        sin_mean = all_reduce(sin_mean)

        # -------- (5) 计算 |φ̂(t) - φ_N(t)|^2 --------
        # 标准正态 CF 的虚部为 0，因此误差为：
        #   (Re - phi)^2 + (Im)^2
        # err: [..., K, P]
        err = (cos_mean - self.phi).square() + sin_mean.square()

        # -------- (6) 数值积分（对 P 维求和）并按样本数缩放 --------
        # (err @ weights): [..., K]，相当于 Σ_p err[..., p] * weights[p]
        # 再乘 N * world_size，把统计量缩放到“全局样本数”尺度
        return (err @ self.weights) * N * world_size


# ==========================================
# 3. 随机投影: Slicing Wrapper (身体)
# ==========================================
class SlicingUnivariateTest(nn.Module):
    """
    Multivariate distribution test using random slicing and univariate test statistics.
    This module extends univariate statistical tests to multivariate data by projecting
    samples onto random 1D directions (slices) and aggregating univariate test statistics
    across all projections. The approach is based on the sliced method for comparing
    high-dimensional distributions.
    The test projects multivariate samples x ∈ ℝᴰ onto random unit vectors:
        x_projected = x @ A
    where A ∈ ℝᴰˣᴷ contains K normalized random direction vectors. A univariate
    test is then applied to each of the K projected samples, and results are aggregated.
    Args:
        univariate_test (torch.nn.Module): A univariate test module that accepts
            (*, N, K) tensors and returns (*, K) test statistics, where N is the
            number of samples and K is the number of slices.
        num_slices (int): Number of random 1D projections (slices) to use. More
            slices increase test power but add computational cost.
        reduction (str, optional): How to aggregate statistics across slices:
            - 'mean': Return the average statistic across all slices
            - 'sum': Return the sum of statistics across all slices
            - None: Return individual statistics for each slice (*, num_slices)
            Default: 'mean'.
        sampler (str, optional): Random sampling method for projection directions:
            - 'gaussian': Sample from standard normal distribution (Gaussian projections)
            Default: 'gaussian'.
        clip_value (float, optional): Minimum threshold for test statistics. Values
            below this threshold are clipped to zero. Useful for reducing noise from
            negligible deviations. Default: None (no clipping).
    Attributes:
        global_step (torch.Tensor): Counter for deterministic random seed generation,
            synchronized across distributed processes to ensure consistent projections.
    Notes:
        - Projection directions are normalized to unit vectors (L2 norm = 1).
        - In distributed training, the random seed is synchronized across all ranks
          using all_reduce to ensure identical projections on all devices.
        - The generator is cached and reused across forward passes for efficiency.
        - The global step counter increments after each forward pass to ensure
          different random projections in successive calls.
    Shape:
        - Input: (*, N, D) where * is any number of batch dimensions, N is the
          number of samples, and D is the feature dimension.
        - Output:
            - Scalar if reduction='mean' or 'sum'
            - (*, num_slices) if reduction=None
    Example:
        >>> from your_module import FastEppsPulley, SlicingUnivariateTest
        >>>
        >>> # Create univariate test
        >>> univariate_test = FastEppsPulley(t_max=5.0, n_points=21)
        >>>
        >>> # Wrap with slicing for multivariate testing
        >>> test = SlicingUnivariateTest(
        ...     univariate_test=univariate_test,
        ...     num_slices=100,
        ...     reduction='mean',
        ...     sampler='gaussian',
        ...     clip_value=0.01
        ... )
        >>>
        >>> # Test multivariate samples
        >>> samples = torch.randn(1000, 50)  # 1000 samples, 50 dimensions
        >>> statistic = test(samples)
        >>> print(f"Test statistic: {statistic.item():.4f}")
        >>>
        >>> # Batch processing
        >>> batch_samples = torch.randn(32, 1000, 50)  # 32 batches
        >>> batch_stats = test(batch_samples)  # Returns scalar (averaged over slices)
    References:
        - Rabin, J., Peyré, G., Delon, J., & Bernot, M. (2012). Wasserstein
          barycenter and its application to texture mixing. In Scale Space and
          Variational Methods in Computer Vision (pp. 435-446).
        - Bonneel, N., Rabin, J., Peyré, G., & Pfister, H. (2015). Sliced and
          Radon Wasserstein barycenters of measures. Journal of Mathematical
          Imaging and Vision, 51(1), 22-45.

    --- 中文版本 ----
    多变量分布的“切片化 (sliced)”检验/正则框架：

    思想
    ----
    将高维样本 x ∈ R^D 投影到多个随机 1D 方向上，把多维问题转成多个一维问题：
        x_proj = x @ A
    其中 A ∈ R^{D×K} 是 K 个单位向量（每列一个方向）。
    然后对每个切片（每列）调用 univariate_test（如 EppsPulley），得到 K 个统计量，
    最后做 mean/sum 聚合得到一个标量 loss（或返回每个切片值）。

    形状约定
    --------
    输入 x: [..., N, D]
      - N：样本数（通常= batch size）
      - D：特征维度（embedding dim）
    输出：
      - reduction="mean" / "sum": 标量
      - reduction=None: [..., K]（每个切片一个统计量）
    """

    def __init__(
        self,
        univariate_test,
        num_slices: int,
        reduction: str = "mean",
        clip_value: float = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.univariate_test = univariate_test
        self.clip_value = clip_value

        # global_step：用于“确定性随机数种子”同步
        # 每次 forward 递增一次，保证不同 step 使用不同投影方向
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

        # generator 缓存：避免每次 forward 都重新创建 Generator
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        """
        获取或创建指定 device 上的随机数生成器，并设置 seed。
        - 使用同一个 Generator 可以减少对象创建开销
        - seed 来自同步后的 global_step，保证所有 rank 使用相同 A
        """
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        """
        参数
        ----
        x: [..., N, D]
          - 最后一维 D 为特征维
          - 倒数第二维 N 为样本维（SIGReg 场景中通常是 batch size）

        返回
        ----
        - reduction="mean": 标量（所有切片的统计量均值）
        - reduction="sum":  标量（所有切片统计量求和）
        - reduction=None:  [..., K]（每个切片一个统计量）
        """
        with torch.no_grad():
            # -------- (1) 同步 global_step，确保所有 GPU 的投影一致 --------
            # 原始实现用 MAX，把所有 rank 的 global_step 对齐到同一个数
            if dist.is_available() and dist.is_initialized():
                global_step_sync = all_reduce(self.global_step.clone(), op="MAX")
            else:
                global_step_sync = self.global_step

            seed = int(global_step_sync.item())

            # -------- (2) 用同步 seed 得到可复用的 Generator --------
            g = self._get_generator(x.device, seed)

            # -------- (3) 采样随机投影矩阵 A ∈ R^{D×K} --------
            # 默认 gaussian：每个元素 ~ N(0,1)
            D = x.size(-1)
            A = torch.randn((D, self.num_slices), device=x.device, generator=g)
            A /= A.norm(p=2, dim=0)  # 列归一化到单位向量（每列一个投影方向）

            # -------- (4) 更新 global_step，保证下一次 forward 方向不同 --------
            self.global_step.add_(1)

        # -------- (5) 投影到 K 个 1D 切片 --------
        # x_proj: [..., N, K]
        x_proj = x @ A

        # -------- (6) 对每个切片调用单变量统计量 --------
        # stats: [..., K]
        stats = self.univariate_test(x_proj)

        # -------- (7) 可选 clip：将“很小的统计量”置 0，减少噪声/无意义偏差 --------
        # 原始实现语义：stats < clip_value -> 0
        if self.clip_value is not None:
            stats = torch.where(stats < self.clip_value, torch.zeros_like(stats), stats)

        # -------- (8) 聚合多个切片统计量 --------
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        else:
            # reduction=None：返回每个切片的统计量，便于调试/分析
            return stats


# ==========================================
# 4. 用户直接调用的最终类: SIGRegLoss
# ==========================================
class SIGRegLoss(nn.Module):
    """
    SIGReg 正则项（Sliced Isotropic Gaussian Regularization）的封装入口。

    使用方式（典型）：
        loss_sig = SIGRegLoss(num_slices=1024)(z)  # z: [B, D]
    然后把 loss_sig 乘一个很小的 λ 加到总 loss 上。

    备注
    ----
    - 不建议默认对 z 做 L2 normalize（会改变投影分布，使“逼近高斯”的目标变形）。
    - 通常建议在 autocast=False / FP32 下计算该 loss，减少 bf16 抖动（取决于训练设置）。
    """

    def __init__(self, num_slices=1024, t_max=3.0, n_points=17):
        super().__init__()
        # (1) 单变量统计量：Epps–Pulley
        epps_pulley = EppsPulley(t_max=t_max, n_points=n_points)

        # (2) slicing 包装：随机投影到多个 1D 切片并聚合
        self.slicing_test = SlicingUnivariateTest(
            univariate_test=epps_pulley,
            num_slices=num_slices,
            reduction="mean"
        )

    def forward(self, z):
        """
        z: [batch size, embedding dim]
        返回: 标量 loss, 表示 z 的分布偏离 N(0, I) 的程度（越大偏离越明显）
        """
        return self.slicing_test(z)
