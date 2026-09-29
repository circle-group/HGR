# 请仔细阅读下面的代码，这个score model是为了学习score-based diffusion generative in latent sapce设计的，请问上面的代码是否合理，是否有可以改进的地方
# 请不用着急，仔细阅读并思考后回答


import torch
import torch.nn as nn
import torch.nn.functional as F
from hgr.diffusion.models.layers import SinusoidalPosEmb
from hgr.diffusion.models.utils import register_model


class GEGLU(nn.Module):
    """
    Gated Linear Unit as described in "GLU Variants Improve Transformer".
    Splits the input in half, applies GELU to the gate half, and multiplies.
    """
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        """
        SELayer 用于 (B, N, channel) 的输入：
          - 先对 N 维度做全局平均池化 → (B, channel)
          - 再通过两层 FC→Sigmoid，得到 (B, channel) 的权重
          - 把权重广播到 (B, 1, channel)，与输入逐通道相乘后返回 (B, N, channel)
        """
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, nf), 其中 nf == channel
        Returns:
            Tensor of shape (B, N, nf)，已按通道加权
        """
        # x.mean(dim=1) 的结果形状 (B, nf)
        y = x.mean(dim=1)             # (B, nf)
        y = self.fc(y).unsqueeze(1)   # (B, 1, nf)
        return x * y                  # 广播乘：(B, N, nf) * (B, 1, nf) → (B, N, nf)




class FilM(nn.Module):
    def __init__(self, nf):
        """
        FiLM 模块，用来对 (B, N, nf) 大小的 h 作条件调制
        Args:
            nf: 特征维度（hidden_dim）
        """
        super().__init__()
        # 两个全连接，用于生成各通道的 β (添加偏移) 和 γ (缩放因子)
        self.yh_add = nn.Linear(nf, nf)  # 生成 β
        self.yh_mul = nn.Linear(nf, nf)  # 生成 γ

        # 初始化：让 γ、β 初始时接近 0，这样一开始调制近似恒等映射
        nn.init.zeros_(self.yh_mul.weight)
        nn.init.zeros_(self.yh_mul.bias)
        nn.init.zeros_(self.yh_add.weight)
        nn.init.zeros_(self.yh_add.bias)

    def forward(self, h, y):
        """
        Args:
            h: Tensor of shape (B, N, nf)
            y: Tensor of shape (B, nf) —— 全局条件或时间 embedding
        Returns:
            new_h: Tensor of shape (B, N, nf)，FiLM 调制后的结果
        """
        # 先把 y 投到 γ, β
        # yh_mul(y) : (B, nf) → unsqueeze→ (B, 1, nf)
        # yh_add(y) : (B, nf) → unsqueeze→ (B, 1, nf)
        gamma = self.yh_mul(y).unsqueeze(1)  # (B, 1, nf)
        beta = self.yh_add(y).unsqueeze(1)  # (B, 1, nf)
        # FiLM 调制： new_h = β + (1 + γ) * h
        new_h = beta + (1.0 + gamma) * h
        return new_h  # (B, N, nf)


class AttentionFiLMBlock(nn.Module):
    """
    改进版 AttentionFiLMBlock，始终保持 h_seq 为 (B, seq_len, hidden_dim)：
      1) Self-Attention
      2) FiLM(t_emb → h_seq)
      3) SELayer(h_seq)
      4) Token-wise Feed-Forward
      5) 更新 t_emb (通过 h_seq.mean(dim=1))
    """
    def __init__(self, hidden_dim, num_heads, dropout, se_reduction=16):
        super(AttentionFiLMBlock, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads  = num_heads

        # 1) Self-Attention 子层
        self.norm1        = nn.LayerNorm(hidden_dim)
        self.attn         = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)

        # 2) FiLM
        self.film = FilM(nf=hidden_dim)

        # 3) SELayer (channel attention)
        self.se = SELayer(channel=hidden_dim, reduction=se_reduction)

        # 4) Token-wise Feed-Forward：对每个 token (size hidden_dim) 做 FFN
        self.norm2   = nn.LayerNorm(hidden_dim)
        self.ff1     = nn.Linear(hidden_dim, hidden_dim * 4, bias=True)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.ff2     = nn.Linear(hidden_dim * 4, hidden_dim, bias=True)
        # 初始化 ff2 为零，让网络从恒等映射开始
        nn.init.zeros_(self.ff2.weight)
        nn.init.zeros_(self.ff2.bias)

        # 5) 更新 time embedding: Pool → Linear → LayerNorm
        self.time_up   = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.time_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_seq, t_emb):
        """
        Args:
            h_seq: Tensor, shape (B, seq_len, hidden_dim)
            t_emb: Tensor, shape (B, hidden_dim)
        Returns:
            h_out_seq: Tensor, shape (B, seq_len, hidden_dim)
            t_emb:     Tensor, shape (B, hidden_dim)  (已更新)
        """
        B, S, H = h_seq.shape
        assert H == self.hidden_dim, f"Expected hidden_dim={self.hidden_dim}, got {H}"

        # ===== 1) Self-Attention Sub-layer =====
        # LayerNorm → MultiheadAttention → Dropout → Residual
        h1 = self.norm1(h_seq)                          # (B, seq_len, hidden_dim)
        attn_out, _ = self.attn(h1, h1, h1)             # (B, seq_len, hidden_dim)
        h_seq = h_seq + self.attn_dropout(attn_out)     # (B, seq_len, hidden_dim)

        # ===== 2) FiLM 调制：用 t_emb (B, hidden_dim) 生成 γ, β 去调制 h_seq =====
        # h_out = β + (1 + γ) * h = h_out = β + (1 + γ) * h 可以认为已经包含了残差
        h_seq = self.film(h_seq, t_emb)                 # (B, seq_len, hidden_dim)

        # ===== 3) SELayer 通道注意力 =====
        h_seq = h_seq + self.se(h_seq)                          # (B, seq_len, hidden_dim)

        # ===== 4) Token-wise Feed-Forward Sub-layer =====
        # 直接对每个 token 进行相同的 FFN 操作
        # (先 LayerNorm 再 Linear→GELU→Dropout→Linear→Residual)
        h2 = self.norm2(h_seq)                          # (B, seq_len, hidden_dim)
        ff  = self.ff1(h2)                              # (B, seq_len, hidden_dim*4)
        ff  = self.act(ff)                              # (B, seq_len, hidden_dim*4)
        ff  = self.dropout(ff)                          # (B, seq_len, hidden_dim*4)
        ff  = self.ff2(ff)                              # (B, seq_len, hidden_dim)
        h_seq = h_seq + ff                              # (B, seq_len, hidden_dim)

        # ===== 5) 更新 t_emb：先把 h_seq 池化到 (B, hidden_dim)，再 Linear→LayerNorm =====
        h_pooled = h_seq.mean(dim=1)                    # (B, hidden_dim)
        new_t    = self.time_up(h_pooled)               # (B, hidden_dim)
        t_emb    = self.time_norm(new_t + t_emb)        # (B, hidden_dim)

        # ===== 6) 输出 h_seq, t_emb =====
        # h_seq 已经是 (B, seq_len, hidden_dim)
        return h_seq, t_emb


@register_model(name='ScoreNet')
class ScoreNet(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        latent_dim  = model_cfg.latent_dim   # e.g. 128

        hidden_dim  = model_cfg.hidden_dim            # 例如 64
        num_heads   = model_cfg.num_heads
        dropout     = model_cfg.dropout
        se_reduction= getattr(model_cfg, 'se_reduction', 16)

        self.seq_len = model_cfg.seq_len    # 将 latent_dim 划分为 seq_len 个 token  (常见取值4，8，16，32
        self.token_dim = hidden_dim // self.seq_len  #每个token的dim，需要为num_heads的整数倍
        assert hidden_dim % self.seq_len == 0, "hidden_dim should be divisible by seq_len"


        # 1) 把 (latent_dim) 投影到 (seq_len * token_dim)
        self.input_proj = nn.Linear(latent_dim, self.seq_len * self.token_dim)

        # 2) 把 token_dim 投影到 hidden_dim
        self.token_proj = nn.Linear(self.token_dim, hidden_dim)


        # 3) time embedding 部分保持不变
        self.time_modules = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            GEGLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 4) 多个 AttentionFiLMBlock，注意这些 block 现在都接受 (B, seq_len, hidden_dim)
        self.blocks = nn.ModuleList([
            AttentionFiLMBlock(hidden_dim, num_heads, dropout, se_reduction)
            for _ in range(model_cfg.num_layers)
        ])

        # 5) 池化  (B, seq_len, hidden_dim) -> (B, hidden_dim)
        self.pool_linear = nn.Linear(self.seq_len * hidden_dim, hidden_dim)

        # 6) 最终的归一化和输出投影
        self.final_norm = nn.LayerNorm(hidden_dim)

        # 用 out_proj (B, hidden_dim) → (B, latent_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z, t):
        """
        z: (B, latent_dim)   — latent vectors
        t: (B,) 或 (B,1)     — time step
        """
        B = z.shape[0]

        h_flat = self.input_proj(z)  # (B, latent_dim) -> (B, seq_len * token_dim)
        h_seq = h_flat.view(B, self.seq_len, self.token_dim) #(B, seq_len, token_dim)
        h_seq = self.token_proj(h_seq)  # (B, seq_len, hidden_dim)

        # time embedding
        t_emb = self.time_modules(t)  # (B, hidden_dim)

        # AttentionFiLMBlock
        for block in self.blocks:
            h_seq, t_emb = block(h_seq, t_emb) #h_seq: (B, seq_len, hidden_dim)

        # 把 (B, seq_len, hidden_dim) 通过Linear汇总回 (B, hidden_dim) ====
        h_pooled = self.pool_linear(h_seq.view(B, -1))

        # 归一化 + 投到 latent_dim ====
        h_norm = self.final_norm(h_pooled)  # (B, hidden_dim)
        return self.out_proj(h_norm)  # (B, latent_dim)
