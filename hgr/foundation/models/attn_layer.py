# foundation/models/attn_layer.py

# 之前为FM/models/rope.py

import torch
from torch import nn
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
from hgr.utils.debug_utils import with_param_info

class RotaryEmbedding(nn.Module):
    """
    [Llama-style 优化版]
    使用复数极坐标 (Polar) 预计算旋转角度。
    缓存的是 complex64 张量 (cos + i*sin)，而非分开的 cos/sin。
    """
    def __init__(self, rotary_dim: int, base: float = 10000.0):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.base = base
        # 缓存：[Max_Seq_Len, rotary_dim // 2] 的复数张量
        self.register_buffer("cis", torch.empty(0), persistent=False)

    def get_cis(self, seqlen: int, device):
        """
        获取长度为 seqlen 的复数旋转因子。
        返回 shape: [seqlen, rotary_dim/2]
        """
        # 检查是否需要更新缓存（长度不够 或 设备不匹配）
        if self.cis.device != device or self.cis.shape[0] < seqlen:
            self.cis = self._update_cis(seqlen, device)
            
        return self.cis[:seqlen]

    def _update_cis(self, seqlen: int, device):
        dim = self.rotary_dim
        # 1. 强制使用 FP32 计算频率，防止长序列精度溢出
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        
        # 2. 生成时间步 t
        t = torch.arange(seqlen, device=device, dtype=torch.float32)
        
        # 3. 外积生成频率表: [seqlen, dim/2]
        freqs = torch.outer(t, inv_freq)
        
        # 4. 生成复数 phasor: exp(i * theta) = cos(theta) + i * sin(theta)
        # torch.polar 需要模长(ones)和角度(freqs)
        cis = torch.polar(torch.ones_like(freqs), freqs)
        
        return cis


def apply_rope_to_qk(q, k, cis, rotary_dim: int):
    """
    [Llama-style 优化版]
    q, k: (B, H, L, D)
    cis:  (L, D_rot/2) <--- 复数张量
    """
    if rotary_dim == 0:
        return q, k

    # 1. 切分：只对前 rotary_dim 维进行旋转
    # q_rot: [B, H, L, rotary_dim]
    # q_pass: [B, H, L, D - rotary_dim]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # 2. Reshape 为复数形式
    # [B, H, L, D_rot] -> [B, H, L, D_rot/2, 2] -> [B, H, L, D_rot/2] (Complex)
    # view_as_complex 是零拷贝操作，极快
    q_complex = torch.view_as_complex(q_rot.float().reshape(*q_rot.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k_rot.float().reshape(*k_rot.shape[:-1], -1, 2))

    # 3. 广播并旋转
    # 兼容两种 cis 形状
    if cis.dim() == 2:
        # cis shape: [L, D_rot/2] -> [1, 1, L, D_rot/2] 以匹配 (B, H, L, D_rot/2)
        cis = cis.view(1, 1, cis.size(0), cis.size(1))
    elif cis.dim() == 4:
        # 假设已经是 (B, 1, L, D_rot/2) 或 (B, H, L, D_rot/2)
        pass
    else:
        raise ValueError(f"Unsupported cis shape: {cis.shape}")
    
    # 复数乘法自动实现旋转：(a+bi)(c+di)
    q_out = q_complex * cis
    k_out = k_complex * cis

    # 4. 还原为实数
    # view_as_real: [..., D_rot/2] (Complex) -> [..., D_rot/2, 2] (Real)
    # flatten: -> [..., D_rot]
    q_rot = torch.view_as_real(q_out).flatten(3)
    k_rot = torch.view_as_real(k_out).flatten(3)

    # 5. 恢复原始数据类型 (如 bf16) 并拼接 pass-through 部分
    q = torch.cat([q_rot.type_as(q), q_pass], dim=-1)
    k = torch.cat([k_rot.type_as(k), k_pass], dim=-1)
    
    return q, k


@with_param_info()
class RopeSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout

        # 合并 QKV 投影，使用一个大 Linear 代替三个小 Linear，提升 GPU 利用率
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)


        # # 🚀 新增：head-specific gate（G1 位置）
        # # 输入：norm1(x) ∈ R^{B,L,D}
        # # 输出：gate ∈ (0,1)^{B,L,H}
        self.gate_proj = nn.Linear(d_model, nhead, bias=True) # head-specific gate 
        # self.gate_proj = nn.Linear(d_model, d_model, bias=True)  # element-wise gate

        self.gate_proj._skip_init = True
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)  # sigmoid(2)≈0.88, 训练初期模型行为更接近“普通 attention”



    def forward(self, x: torch.Tensor, cis = None, attn_mask: torch.Tensor = None, attn_bias = None):
        """
        x:          (B, L, D)
        attn_mask:  (B, 1, 1, L) float additive mask, 0=keep, -inf=mask
        cis:        (B, 1, L, D_rot/2) 或 (L, D_rot/2)，Tree-RoPE 相位
        struct_bias:(B, H, L, L)，共享的树结构偏置
        """
        B, L, D = x.shape
        H, Dh = self.nhead, self.head_dim

        # === 优化 1 应用: 一次计算，然后拆分 ===
        # qkv: (B, L, 3*D)
        qkv = self.qkv_proj(x)
        
        # Split: (B, L, D) -> (B, L, H, Dh) -> (B, H, L, Dh)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, L, H, Dh).transpose(1, 2)
        k = k.view(B, L, H, Dh).transpose(1, 2)
        v = v.view(B, L, H, Dh).transpose(1, 2)

        # 2) ROPE
        if cis is not None:
            rotary_dim = cis.shape[-1] * 2 # 从 cis 的 shape 推断 rotary_dim
            q, k = apply_rope_to_qk(q, k, cis, rotary_dim=rotary_dim)


        attn_params = { "dropout_p": self.dropout if self.training else 0.0,
                        "is_causal": False }

        if attn_bias is not None:
            # # 输入的attn_mask已经是float additive mask， 0=keep, -inf=mask
            nhead_bias = attn_bias.size(1)
            nhead_flash = self.nhead - nhead_bias
            
            # # split heads: 前 nhead_flash 走 fast，后 nhead_bias 走带 dense bias
            q_f, q_b = q.split([nhead_flash, nhead_bias], dim=1)
            k_f, k_b = k.split([nhead_flash, nhead_bias], dim=1)
            v_f, v_b = v.split([nhead_flash, nhead_bias], dim=1)

            # with sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
            ctx_f = F.scaled_dot_product_attention(
                q_f.contiguous(), k_f.contiguous(), v_f.contiguous(),
                attn_mask=attn_mask,  # 只有 padding mask，尽量触发 flash
                **attn_params
            )

            ctx_b = F.scaled_dot_product_attention(
                q_b.contiguous(), k_b.contiguous(), v_b.contiguous(),
                attn_mask=attn_bias,   # dense mask，通常只能走 math
                **attn_params
            )

            ctx = torch.cat([ctx_f, ctx_b], dim=1)  # (B,H,L,Dh)
            
        else:
            # bool mask: True=keep, False=mask  
            ctx = F.scaled_dot_product_attention(
                q.contiguous(), k.contiguous(), v.contiguous(),
                attn_mask=attn_mask,
                **attn_params
            )

        # ✅ head-specific gate after SDPA (G1)
        gate_logits = self.gate_proj(x)        # (B, L, H)
        gate = torch.sigmoid(gate_logits)      # (B, L, H)
        gate = gate.permute(0, 2, 1).unsqueeze(-1)  # (B, H, L, 1)
        ctx = ctx * gate                       # (B, H, L, Dh) 对于当前 token，这个 head 的 attention 输出要保留多少（缩放多少）

        # # ✅ elementwise gate after SDPA (G1)
        # gate_logits = self.gate_proj(x)                 # (B, L, D)
        # gate = torch.sigmoid(gate_logits)
        # gate = gate.view(B, L, H, Dh).permute(0, 2, 1, 3)  # (B, H, L, Dh)
        # ctx = ctx * gate                                # (B, H, L, Dh)
        
        # 4) Output
        ctx = ctx.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(ctx)


@with_param_info()
class RopeTransformerEncoderLayer(nn.Module):
    """
    Pre-LN Transformer Encoder Layer
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = RopeSelfAttention(d_model, nhead, dropout=dropout)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cis, attn_mask, attn_bias=None):
        y = self.self_attn(self.norm1(x), 
                            cis=cis,
                            attn_mask=attn_mask, 
                            attn_bias=attn_bias)
        x = x + self.drop1(y)
        y = self.ff(self.norm2(x))
        x = x + self.drop2(y)
        return x

