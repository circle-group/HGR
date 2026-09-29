from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PackedGRUEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        bidirectional: bool,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.model = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.model.flatten_parameters()

    def forward(self, in_seq_emb: torch.Tensor, lengths: torch.Tensor):
        packed = pack_padded_sequence(
            in_seq_emb,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, h_n = self.model(packed)
        out, _ = pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=in_seq_emb.size(1),
        )
        if self.bidirectional:
            out = out.view(out.size(0), out.size(1), 2, self.hidden_dim)
        else:
            out = out.unsqueeze(2)
        return out, h_n


# ---------- RoPE ----------

class RotaryPositionEmbedding(nn.Module):
    """Precomputes and caches RoPE sin/cos tables (no learnable parameters)."""

    def __init__(self, head_dim: int, max_len: int = 512):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)              # (L, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)             # (L, head_dim)
        self.register_buffer("_cos_cached", emb.cos(), persistent=False)
        self.register_buffer("_sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self._cos_cached.size(0):
            self._build_cache(seq_len)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors.

    Args:
        q, k: (B, H, L, head_dim)
        cos, sin: (L, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, L, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


# ---------- Transformer building blocks ----------

class RoPESelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, Dh)
        q, k, v = qkv.unbind(0)
        q, k = _apply_rope(q, k, cos, sin)

        attn_mask = None
        if padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, L, dtype=q.dtype, device=q.device)
            attn_mask.masked_fill_(padding_mask[:, None, None, :], float("-inf"))

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        return self.out_proj(out.transpose(1, 2).reshape(B, L, D))


class RoPETransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with RoPE."""

    def __init__(self, model_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.self_attn = RoPESelfAttention(model_dim, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ff = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.dropout1(self.self_attn(self.norm1(x), cos, sin, padding_mask))
        x = x + self.ff(self.norm2(x))
        return x


# ---------- Attention pooling ----------

class AttentionPooling(nn.Module):
    """Learnable-query attention pooling: compresses a variable-length
    sequence into a single fixed-size vector."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(model_dim))
        self.scale = model_dim ** -0.5
        nn.init.normal_(self.query, std=0.02)

    def forward(self, enc_seq: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        scores = (enc_seq @ self.query) * self.scale       # (B, L)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)   # (B, L, 1)
        return (enc_seq * weights).sum(dim=1)               # (B, D)


# ---------- Transformer encoder ----------

class TransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        ff_mult: int,
        dropout: float,
        max_len: int,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_layers = num_layers
        self.input_proj = nn.Identity() if input_dim == model_dim else nn.Linear(input_dim, model_dim)
        head_dim = model_dim // num_heads
        self.rope = RotaryPositionEmbedding(head_dim, max_len)
        self.layers = nn.ModuleList([
            RoPETransformerEncoderLayer(model_dim, num_heads, model_dim * ff_mult, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(model_dim)
        self.pool = AttentionPooling(model_dim)

    def init_weights(self):
        """Xavier uniform for all linear layers; scaled init for residual output projections."""
        residual_scale = (2 * self.num_layers) ** -0.5
        for layer in self.layers:
            # QKV
            nn.init.xavier_uniform_(layer.self_attn.qkv.weight)
            nn.init.zeros_(layer.self_attn.qkv.bias)
            # attention output projection (residual path)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight, gain=residual_scale)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            # FF layers
            ff_linears = [m for m in layer.ff if isinstance(m, nn.Linear)]
            for i, lin in enumerate(ff_linears):
                if i == len(ff_linears) - 1:
                    nn.init.xavier_uniform_(lin.weight, gain=residual_scale)
                else:
                    nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
        # input projection
        if isinstance(self.input_proj, nn.Linear):
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)

    def forward(self, in_seq_emb: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (enc_seq, pooled) where pooled is the attention-pooled summary."""
        _, seq_len, _ = in_seq_emb.shape
        hidden = self.input_proj(in_seq_emb)
        padding_mask = torch.arange(seq_len, device=hidden.device).unsqueeze(0) >= lengths.unsqueeze(1)
        cos, sin = self.rope(seq_len)
        for layer in self.layers:
            hidden = layer(hidden, padding_mask, cos, sin)
        hidden = self.final_norm(hidden)
        pooled = self.pool(hidden, padding_mask)
        return hidden, pooled
