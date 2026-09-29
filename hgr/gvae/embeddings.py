from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from hgr.grammar.rule_features import build_wl_vocab_and_csr


@dataclass
class StructuredRuleInitConfig:
    mode: str = "id_only"
    wl_iterations: int = 3
    freeze_feature_bank: bool = True


def _build_structured_feature_bank(prod_rule_corpus, rule_init_cfg) -> torch.Tensor:
    rules = prod_rule_corpus.prod_rule_list
    if len(rules) == 0:
        raise ValueError("Cannot build structured rule features from an empty grammar.")

    H = int(getattr(rule_init_cfg, "wl_iterations", 3))
    _, S_list, _ = build_wl_vocab_and_csr(rules, H=H, size_norm=True, add_unk=True, save_path=None)
    # Structured rule init is intentionally WL-only.
    feature_bank = torch.cat([S.to_dense() for S in S_list], dim=1)

    mean = feature_bank.mean(dim=0, keepdim=True)
    std = feature_bank.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    feature_bank = (feature_bank - mean) / std
    pad_row = feature_bank.new_zeros(1, feature_bank.size(1))
    return torch.cat([feature_bank, pad_row], dim=0)


class StructuredRuleFeatureBank(nn.Module):
    def __init__(self, feature_bank: torch.Tensor, freeze: bool = True):
        super().__init__()
        feature_bank = feature_bank.float()
        if freeze:
            self.register_buffer("feature_bank", feature_bank)
        else:
            self.feature_bank = nn.Parameter(feature_bank)

    @property
    def feature_dim(self) -> int:
        return int(self.feature_bank.size(1))

    def lookup(self, rule_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(rule_ids, self.feature_bank)


class HybridRuleEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        padding_idx: int,
        dropout: float,
        feature_bank: StructuredRuleFeatureBank | None = None,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.padding_idx = padding_idx
        self.id_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(dropout)
        self.feature_bank = feature_bank
        if feature_bank is not None:
            feat_dim = feature_bank.feature_dim
            self.feature_proj = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, model_dim),
                nn.GELU(),
                nn.Linear(model_dim, model_dim),
            )
        else:
            self.feature_proj = None

    @property
    def weight(self) -> torch.Tensor:
        return self.id_embedding.weight

    def reset_parameters(self, init_range: float = 0.1):
        nn.init.uniform_(self.id_embedding.weight, -init_range, init_range)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.id_embedding.weight[self.padding_idx].zero_()

    def forward(self, rule_ids: torch.Tensor) -> torch.Tensor:
        emb = self.id_embedding(rule_ids)
        if self.feature_bank is not None and self.feature_proj is not None:
            emb = emb + self.feature_proj(self.feature_bank.lookup(rule_ids))
        return self.dropout(emb)
