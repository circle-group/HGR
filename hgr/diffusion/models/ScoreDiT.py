import torch
import torch.nn as nn
import torch.nn.functional as F

from hgr.diffusion.models.layers import SinusoidalPosEmb
from hgr.diffusion.models.utils import register_model


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


def modulate(x, shift, scale):
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTMLP(nn.Module):
    def __init__(self, hidden_dim, ff_mult=4, dropout=0.0):
        super().__init__()
        inner_dim = hidden_dim * ff_mult
        self.fc1 = nn.Linear(hidden_dim, inner_dim * 2)
        self.act = GEGLU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, ff_mult=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = DiTMLP(hidden_dim, ff_mult=ff_mult, dropout=dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6, bias=True),
        )

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, cond):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(cond).chunk(6, dim=-1)
        attn_in = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * attn_out

        mlp_in = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_out = self.mlp(mlp_in)
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim, patch_dim):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2, bias=True),
        )
        self.linear = nn.Linear(hidden_dim, patch_dim, bias=True)

        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, cond):
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


@register_model(name="ScoreDiT")
class ScoreDiT(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        latent_dim = int(model_cfg.latent_dim)
        hidden_dim = int(model_cfg.hidden_dim)
        num_layers = int(model_cfg.num_layers)
        num_heads = int(model_cfg.num_heads)
        seq_len = int(model_cfg.seq_len)
        ff_mult = int(getattr(model_cfg, "ff_mult", 4))
        dropout = float(getattr(model_cfg, "dropout", 0.0))
        time_cond_dim = int(getattr(model_cfg, "time_cond_dim", hidden_dim))
        input_layer_norm = bool(model_cfg.input_layer_norm)
        use_skip_proj = bool(model_cfg.use_skip_proj)

        if latent_dim % seq_len != 0:
            raise ValueError(f"latent_dim={latent_dim} must be divisible by seq_len={seq_len}")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}")

        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.patch_dim = latent_dim // seq_len

        self.input_norm = nn.LayerNorm(latent_dim) if input_layer_norm else nn.Identity()
        self.token_proj = nn.Linear(self.patch_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, time_cond_dim * 2),
            GEGLU(),
            nn.Linear(time_cond_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    ff_mult=ff_mult,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer = FinalLayer(hidden_dim, self.patch_dim)
        self.skip_proj = nn.Linear(latent_dim, latent_dim, bias=False) if use_skip_proj else None

        if self.skip_proj is not None:
            nn.init.zeros_(self.skip_proj.weight)

    def forward(self, z, t):
        batch_size = z.shape[0]
        x = self.input_norm(z).reshape(batch_size, self.seq_len, self.patch_dim)
        x = self.token_proj(x)
        cond = self.time_embed(t)

        for block in self.blocks:
            x = block(x, cond)

        out = self.final_layer(x, cond).reshape(batch_size, self.latent_dim)
        if self.skip_proj is not None:
            out = out + self.skip_proj(z)
        return out
