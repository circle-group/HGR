# foundation/models/attnbias.py

import os, pickle, re
import math
import torch
from torch import nn
import torch.nn.functional as F
from hgr.utils.debug_utils import with_param_info
from hgr.utils.file_utils import load_pickle
from typing import List, Dict, Tuple, Optional
from hgr.foundation.data_utils.mol_defs import NUM_BOND_TYPE, NUM_BOND_DIRECTION

import logging
logger = logging.getLogger(__name__)


class AtomGraphSpatialBias(nn.Module):
    def __init__(self, max_path_distance: int, num_heads: int):
        super().__init__()
        self.max_path_distance = max_path_distance
        self.num_heads = num_heads
        # self.b = nn.Parameter(torch.randn(self.max_path_distance + 1))
        self.b = nn.Embedding(self.max_path_distance + 1, num_heads, padding_idx=0)

        # init_weights 不会动 nn.Embedding，所以这里初始化是安全的
        nn.init.normal_(self.b.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.b.weight[0].zero_()


    def forward(
        self,
        L_seq, #rule_seq_padded: torch.Tensor,         # (B, L_seq)
        atom_pos_padded: torch.Tensor,         # (B, Nmax), pad=-1
        atom_num: torch.Tensor,            # (B,) 表示第 b 个样本里 有效 atom 的数量
        node_paths_length: torch.Tensor,       # (B, Nmax, Nmax), 0=unreachable/pad, 1=self, 2=1-hop ...
        dtype: torch.dtype
    ) -> torch.Tensor:
        
        # B, L_seq = rule_seq_padded.shape
        B = atom_num.shape[0]
        device = atom_pos_padded.device

        out = torch.zeros((B, self.num_heads, L_seq, L_seq), device=device, dtype=dtype)
        atom_num_list = atom_num.tolist()

        for b in range(B):
            n = atom_num_list[b]
            if n <= 0: continue
            pos = atom_pos_padded[b, :n]
            npl = node_paths_length[b, :n, :n]

            # node_paths_length: 0=unreachable/pad, 1=self, 2=1-hop ...
            idx = npl.clamp(min=0, max=self.max_path_distance).long()
            bias = self.b(idx).permute(2, 0, 1).to(dtype=dtype) # (H,n,n)

            out[b, :, pos[:, None], pos[None, :]] = bias.contiguous()

        return out


class AtomGraphEdgeBias(nn.Module):
    """
    Graphormer 风格的 Edge Bias (向量化优化版)。
    
    核心思想：
    1. 预计算 Batch 中每条边的 (bond_type, bond_dir) 直接查表得到 per-head 标量 bias
    2. 利用 edge_ptr 将所有样本的最短路径（Edge ID）统一映射到全局 Edge ID。
    3. 一次性查表并聚合，避免 Python 循环带来的 CPU 瓶颈。

    Complexity: O(B * N^2 * D) 的 GPU 并行计算，远快于 O(B * D) 的 Python 循环。
    """
    def __init__(self, max_path_distance: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.max_path_distance = max_path_distance
        self.bond_type_head = nn.Embedding(NUM_BOND_TYPE, num_heads)
        self.bond_dir_head  = nn.Embedding(NUM_BOND_DIRECTION, num_heads)
        nn.init.normal_(self.bond_type_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bond_dir_head.weight,  mean=0.0, std=0.02)

    def forward(
        self,
        L_seq: int,
        atom_pos_padded: torch.Tensor,    # (B, Nmax)
        atom_num: torch.Tensor,           # (B,)
        edge_paths_tensor: torch.Tensor,  # (B, Nmax, Nmax, Dpath) local eid, pad=-1
        edge_paths_length: torch.Tensor,  # (B, Nmax, Nmax) hops
        edge_attr: torch.Tensor,          # (E_total, 2)
        edge_ptr: torch.Tensor,           # (B+1,)
        dtype: torch.dtype,
    ) -> torch.Tensor:
        device = atom_pos_padded.device
        B, Nmax = atom_pos_padded.shape
        H = self.num_heads
        # Dpath = self.max_path_distance - 1, 最短路径记录的最大跳数

        # 初始化输出张量
        out = torch.zeros((B, H, L_seq, L_seq), device=device, dtype=dtype)
        

        # Step 1. 全 batch 一次性算出每条边的 per-head bias 
        edge_bias_all = (self.bond_type_head(edge_attr[:, 0]) + self.bond_dir_head(edge_attr[:, 1])).to(dtype=dtype) # (E_total, H)

        # Step 2. 将 Local Edge ID edge_paths_tensor 转换为 Global Edge ID (关键向量化步骤) 
        # edge_ptr[:-1] 包含每个图的起始边索引。形状变换: (B,) -> (B, 1, 1, 1) 以便广播到 (B, N, N, D)
        offset = edge_ptr[:-1].view(B, 1, 1, 1).to(edge_paths_tensor.dtype)
        mask = edge_paths_tensor.ne(-1)        # (B, N, N, D) 掩码: 标记哪些位置是有效的路径节点 (原值为 -1 表示 padding)

        # 计算全局索引: Local ID + Offset
        # masked_fill(~mask, 0): 将 padding (-1) 的位置安全地置为 0，防止 F.embedding 越界。
        # 这些位置的值在稍后会被 mask 乘法清零，所以置 0 是安全的。
        global_idx = (edge_paths_tensor + offset).masked_fill(~mask, 0) # (B, Nmax, Nmax, Dpath)


        # Step 3: 查表与聚合 (Hop 维度的计算), 使用 Global ID 从 edge_bias_all 中获取 Embedding
        step = F.embedding(global_idx, edge_bias_all) # (B, N, N, D)->(B, N, N, D, H)

        # 聚合: 对路径上的所有跳 (Hops) 进行求和
        # * mask.unsqueeze(-1): 将无效跳 (padding) 的 Embedding 置为 0
        # sum(dim=3): 消除 Hop 维度 D
        bias_sum = (step * mask.unsqueeze(-1)).sum(dim=3) # (B, N, N, D, H) -> (B, N, N, H)

        # 平均: 除以路径长度
        denom = edge_paths_length.clamp_min(1).to(dtype=dtype).unsqueeze(-1)  # (B,Nmax,Nmax,1)
        bias = (bias_sum / denom).permute(0, 3, 1, 2).contiguous().to(dtype=dtype)            # (B,H,Nmax,Nmax)

        # Step 4: scatter 到 token 空间，仍然需要按图处理 pos 的不规则映射
        atom_num_list = atom_num.tolist()
        for b in range(B):
            n = atom_num_list[b]
            if n <= 0: continue
            pos = atom_pos_padded[b, :n]
            out[b, :, pos[:, None], pos[None, :]] = bias[b, :, :n, :n]

        return out




@with_param_info()
class TreeStructBias(nn.Module):
    """
    共享的树结构偏置：
      - relation embedding: 7 种关系类型(grammar_tree.py中定义的树关系)
      - distance bucket embedding: num_buckets 个距离桶
    输出: (B, 1, L, L)，供所有层共享

    max_distance 可以按你的树最大深度/直径来调,

    grammar tree distance 统计结果：二者1～12占了90%+的pair，值得“精细刻画”
    - Zinc2m: max=32
    - RingDiv: max=37， >32的pair一共304个，比例 0.000002
    """
    def __init__(self,
                 num_rel_types: int = 7,       # 关系类型总数
                 num_dist_buckets: int = 24,   # 距离桶总数
                 max_exact: int = 12,          # 精确编码的最大距离
                 max_distance: int = 32,       # 距离的最大裁剪值
                 num_heads: int = 8,               # 注意力 head 数
                 struct_dim: int = 8):        # rel/dist 的中间嵌入维度 d
        super().__init__()

        assert num_dist_buckets >= max_exact + 2, f"num_dist_buckets({num_dist_buckets}) must be >= max_exact+2({max_exact+2}) so that we have at least 1 bucket for large distances."
        assert max_distance >= max_exact, f"max_distance({max_distance}) must be >= max_exact({max_exact})"

        self.num_rel_types = num_rel_types
        self.num_dist_buckets = num_dist_buckets
        self.max_distance = max_distance
        self.max_exact = max_exact

        # 1) 关系 / 距离 embedding：d 维
        self.rel_emb = nn.Embedding(num_rel_types, struct_dim)       # 关系嵌入层：将 num_rel_types 个 ID 映射到 1 维标量
        self.dist_emb = nn.Embedding(num_dist_buckets, struct_dim)   # 距离桶嵌入层：将 num_dist_buckets 个 ID 映射到 1 维标量
        
        # # # 2) 把 d 维压到 per-head logits：Linear(d -> H)
        # # # 无 bias，初始权重为 0 → 初始时结构偏置全 0，不破坏已有初始化
        self.rel_proj = nn.Linear(struct_dim, num_heads, bias=False)
        self.dist_proj = nn.Linear(struct_dim, num_heads, bias=False)

        #   - embeddings: small random (non-zero) so proj gets gradients
        #   - projections: zero so initial bias is exactly 0 and doesn't disturb attention
        nn.init.normal_(self.rel_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.dist_emb.weight, mean=0.0, std=0.02)
        self.rel_proj._skip_init = True
        self.dist_proj._skip_init = True
        nn.init.zeros_(self.rel_proj.weight)
        nn.init.zeros_(self.dist_proj.weight)

        # === 优化核心：预计算查找表 (LUT) ===
        # 1. 创建 0 到 max_distance 的所有可能距离
        dist_vals = torch.arange(max_distance + 1, dtype=torch.long)
        
        # 2. 调用原有的 _compute_buckets 逻辑（改个名以免混淆）生成映射表
        # 注意：这里我们只算一次，是在 CPU 上算的，慢点没关系
        bucket_lut = self._compute_buckets(dist_vals) 
        
        # 3. 注册为 Buffer
        self.register_buffer("bucket_lut", bucket_lut, persistent=True) # persistent=True 表示它会被保存到 state_dict 中

    def _compute_buckets(self, dist: torch.Tensor) -> torch.Tensor:
        """
        非线性分桶：
          - 小距离 [0, max_exact] 每个距离单独一个桶
          - 大距离 (max_exact, max_distance] 做 log 压缩到剩余桶
        dist: (B,L,L) int/long
        """
        # 1. 裁剪: 将所有距离限制在 [0, max_distance] 范围内
        # dist = dist.clamp(min=0, max=self.max_distance)
        n = self.num_dist_buckets
 

        # 2. 小距离 (精确编码): 距离 0 到 max_exact，每个距离单独占一个桶
        is_small = dist <= self.max_exact
        small_bucket = dist.clamp(max=self.max_exact)  # 桶 ID = 距离值 (0, 1, ..., max_exact)

        # 3. 大距离 (log 编码): 把 [max_exact+1, max_distance] 的 log 区间映射到剩下的桶
        large_dist = dist.clamp(min=self.max_exact + 1, max=self.max_distance).float()

        # 对数压缩逻辑
        log_base = self.max_distance / float(self.max_exact)

        # 计算 log 缩放因子 (n - max_exact - 1) 是剩余桶的数量，减 1 留给 "very far"
        scale = (n - self.max_exact - 1) / math.log(log_base)

        # 应用对数变换，并将结果线性映射到 [max_exact+1, n-2] 的桶 ID 范围
        large_bucket = self.max_exact + 1 + (torch.log(large_dist / self.max_exact) * scale).long()

        # 确保不会超过最大的桶索引 (n-1)，将所有非常大的距离都归入最后的桶
        large_bucket = large_bucket.clamp(max=n - 1)

        # 4. 合并: 使用 torch.where 合并小距离桶和大距离桶
        buckets = torch.where(is_small, small_bucket.long(), large_bucket)
        return buckets

    def forward(self, dist_mat: torch.Tensor, rel_mat: torch.Tensor) -> torch.Tensor:
        """
        dist_mat, rel_mat: (B, L, L)
        return: struct_bias: (B, H, L, L), H=self.nhead
        """
        dist_mat = dist_mat.to(torch.long)
        rel_mat  = rel_mat.to(torch.long)

        # 1. relation bias
        rel_bias = self.rel_emb(rel_mat)          # (B,L,L,1)

        # 2. Distance Bucket Bias (LUT 查表优化版)
        # 查表操作： input 作为 index 去 lut 里取值
        bucket_ids = self.bucket_lut[dist_mat.clamp(0, self.max_distance)]
        dist_bias = self.dist_emb(bucket_ids)    # (B,L,L,1)

        # 3. 合并偏置
        struct_bias = self.rel_proj(rel_bias) + self.dist_proj(dist_bias)            # (B,L,L,H)

        # 5) 调整形状为 (B,H,L,L)，以便直接与 attention logits 相加
        struct_bias = struct_bias.permute(0, 3, 1, 2).contiguous()      # (B,H,L,L)
        # (struct_bias.float().max()-struct_bias.float().min()).item() 现在发现训练过程中struct_bias是变化的了，虽然比较小
        return struct_bias 
