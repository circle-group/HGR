import torch
import torch.nn as nn
import torch.nn.functional as F
from hgr.diffusion.models.layers import SinusoidalPosEmb, get_act
from hgr.diffusion.models.utils import register_model


class GEGLU(nn.Module):
    """
    Gated Linear Unit as described in "GLU Variants Improve Transformer".
    Splits the input in half, applies GELU to the gate half, and multiplies.
    """

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class ResidualScoreBlock(nn.Module):
    """一个带时间 FiLM（分离 scale & shift）调制的残差 MLP 块，支持零初始化残差分支"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # Layer norm on input features
        self.norm = nn.LayerNorm(dim)
        # FiLM: project time_emb to 2 * dim for scale (gamma) and shift (beta)
        self.time_proj = nn.Linear(hidden_dim, dim * 2)
        # MLP layers
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # zero-init last linear for stable residual
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x, t_emb):
        # 1) FiLM modulation: split gamma & beta
        gamma, beta = self.time_proj(t_emb).chunk(2, dim=-1)
        h = self.norm(x) * (1 + gamma) + beta
        # 2) MLP + dropout
        h = F.silu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        # 3) Residual connection
        return x + h


class SEChannel(nn.Module):
    """简化版 Squeeze-and-Excite，用于给通道做注意力加权"""

    def __init__(self, channel, reduction=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, hidden_dim)
        # 直接对每个样本的通道特征做 excitation
        w = self.net(x)  # (B, hidden_dim)
        return x * w  # broadcast element-wise

@register_model(name='vanillaScoreNet')
class vanillaScoreNet(nn.Module):
    """
    优化后的 latent-space score-based diffusion 网络：
      1) input_proj：latent_dim → hidden_dim
      2) time_mlp：SinusoidalPosEmb + GEGLU
      3) 多个 ResidualScoreBlock
      4) 多层自注意力
      5) SE 通道注意力
      6) out_proj：hidden_dim → latent_dim
    """

    def __init__(self, model_cfg):
        super().__init__()
        # act_fn = get_act(nonlinearity)
        latent_dim, hidden_dim = model_cfg.prod_rule_embed_dim, model_cfg.hidden_dim
        num_layers, num_heads = model_cfg.num_layers, model_cfg.num_heads
        dropout = model_cfg.dropout

        # 1) 输入／输出 投影
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_dim)

        # 2) 时间编码 MLP：Sinusoidal → Linear→GEGLU→Linear
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),  # → (B, hidden_dim)
            nn.Linear(hidden_dim, hidden_dim * 2),  # → (B, 2*hidden_dim)
            GEGLU(),  # → (B, hidden_dim)
            nn.Linear(hidden_dim, hidden_dim),  # → (B, hidden_dim)
        )

        # 3) 堆叠若干残差块
        self.blocks = nn.ModuleList([
            ResidualScoreBlock(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # 4) 自注意力（可扩展为多层）
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # 5) SE 通道注意力
        self.se = SEChannel(hidden_dim, reduction=4)

    def forward(self, z, t):
        """
        z: (B, latent_dim)
        t: (B,) 或 (B,1)  — 归一化的噪声等级
        """
        # 1) 投影到隐藏空间
        h = self.input_proj(z)  # → (B, hidden_dim)
        # 2) time embedding
        t_emb = self.time_mlp(t)  # → (B, hidden_dim)

        # 3) 残差块
        for block in self.blocks:
            h = block(h, t_emb)  # 每步都注入 time embedding

        # 4) 自注意力
        h_unsq = h.unsqueeze(1)  # → (B,1,hidden_dim)
        h_attn, _ = self.attn(h_unsq, h_unsq, h_unsq)
        h_attn = h_attn.squeeze(1)  # → (B,hidden_dim)
        h = self.attn_norm(h + h_attn)  # Residual + Norm

        # 5) 通道注意力
        h = self.se(h)  # → (B, hidden_dim)

        # 6) 投回 latent space
        return self.out_proj(h)  # → (B, latent_dim)
