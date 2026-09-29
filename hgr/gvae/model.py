from __future__ import annotations

import torch
from torch import nn

from hgr.grammar.symbol import NTSymbol

from .decoder import GRUDecoder
from .embeddings import HybridRuleEmbedding, StructuredRuleFeatureBank, _build_structured_feature_bank
from .encoder import PackedGRUEncoder, TransformerSequenceEncoder


class GrammarSeq2SeqVAE(nn.Module):
    """Grammar VAE: 基于超图文法的 Seq2Seq VAE。

    Encoder (GRU/Transformer) → μ, logσ² → z → GRU Decoder → 产生式规则序列 → 超图展开。
    支持 NT conditioning 和 hybrid_struct 规则嵌入初始化。
    """

    def __init__(self, hrg, cfg):
        super().__init__()
        # ── 基础配置 ──────────────────────────────────────────────────────
        self.prod_rule_corpus = hrg
        self._grammar_device = None
        self.vocab_size = hrg.num_prod_rule + 1   # +1 为 padding token
        self.latent_dim = cfg.latent_dim
        self.max_vol = cfg.max_vol
        self.padding_idx = cfg.padding_idx % self.vocab_size  # cfg=-1 → vocab_size-1
        self.rule_embed_dim = cfg.rule_embed_dim
        self.start_rule_embedding = getattr(cfg, "start_rule_embedding", False)
        self.use_nt_conditioning = bool(getattr(cfg, "use_nt_conditioning", True))
        self.rule_init_cfg = getattr(cfg, "rule_init", None)

        # ── 提取编解码器超参 ───────────────────────────────────────────────
        dropout = float(getattr(cfg, "dropout", 0.0))
        encoder_name = str(getattr(cfg.encoder_params, "name", "GRU"))
        decoder_name = str(getattr(cfg.decoder_params, "name", "GRU"))
        if encoder_name not in {"GRU", "Transformer"} or decoder_name != "GRU":
            raise ValueError(
                "GVAE supports encoder in {'GRU','Transformer'} and decoder='GRU'; "
                f"got encoder={encoder_name!r}, decoder={decoder_name!r}"
            )
        self.encoder_name = encoder_name
        enc_num_layers = int(cfg.encoder_params.num_layers)
        enc_hidden_dim = int(cfg.encoder_params.hidden_dim)
        dec_num_layers = int(cfg.decoder_params.num_layers)
        dec_hidden_dim = int(cfg.decoder_params.hidden_dim)

        # ── 嵌入层 ────────────────────────────────────────────────────────
        # 构建 WL 结构特征银行 (hybrid_struct 模式); 否则为 None (id_only)
        structured_bank = None
        if str(getattr(self.rule_init_cfg, "mode", "id_only")).lower() == "hybrid_struct":
            feat = _build_structured_feature_bank(self.prod_rule_corpus, self.rule_init_cfg)
            structured_bank = StructuredRuleFeatureBank(
                feat, freeze=bool(getattr(self.rule_init_cfg, "freeze_feature_bank", True))
            )
        embed_kwargs = dict(
            vocab_size=self.vocab_size, model_dim=self.rule_embed_dim,
            padding_idx=self.padding_idx, dropout=dropout, feature_bank=structured_bank,
        )
        self.src_embedding = HybridRuleEmbedding(**embed_kwargs)
        self.tgt_embedding = HybridRuleEmbedding(**embed_kwargs)
        self.tgt_dropout = nn.Dropout(dropout)

        # ── 编码器 ────────────────────────────────────────────────────────
        if encoder_name == "GRU":
            self._bidirectional = bool(cfg.encoder_params.bidirectional)
            self.encoder = PackedGRUEncoder(
                input_dim=self.rule_embed_dim, hidden_dim=enc_hidden_dim,
                num_layers=enc_num_layers, bidirectional=self._bidirectional, dropout=dropout,
            )
            enc_out_dim = enc_hidden_dim * (2 if self._bidirectional else 1)
        else:
            # Transformer: RoPE 位置编码 + 注意力池化; head_dim 须为偶数
            num_heads = int(getattr(cfg.encoder_params, "num_heads", 8))
            head_dim = enc_hidden_dim // num_heads
            if enc_hidden_dim % num_heads != 0:
                raise ValueError(f"hidden_dim={enc_hidden_dim} must be divisible by num_heads={num_heads}")
            if head_dim % 2 != 0:
                raise ValueError(f"head_dim={head_dim} must be even for RoPE")
            self.encoder = TransformerSequenceEncoder(
                input_dim=self.rule_embed_dim, model_dim=enc_hidden_dim,
                num_layers=enc_num_layers, num_heads=num_heads,
                ff_mult=int(getattr(cfg.encoder_params, "ff_mult", 2)),
                dropout=dropout, max_len=self.max_vol,
            )
            enc_out_dim = enc_hidden_dim

        # ── 解码器 ────────────────────────────────────────────────────────
        self.decoder = GRUDecoder(
            input_dim=self.rule_embed_dim, hidden_dim=dec_hidden_dim,
            num_layers=dec_num_layers, dropout=dropout,
        )

        # ── VAE 后验参数映射: encoder output → (μ, logσ²) ─────────────────
        if self.start_rule_embedding:
            # 双通道: 隐藏态与起始规则嵌入各贡献 latent_dim/2
            half = self.latent_dim // 2
            self.emb2mean   = nn.Linear(self.rule_embed_dim, half, bias=False)
            self.emb2logvar = nn.Linear(self.rule_embed_dim, half)
            self.hid2mean   = nn.Linear(enc_out_dim, half, bias=False)
            self.hid2logvar = nn.Linear(enc_out_dim, half)
        else:
            self.hid2mean   = nn.Linear(enc_out_dim, self.latent_dim, bias=False)
            self.hid2logvar = nn.Linear(enc_out_dim, self.latent_dim)

        # ── 隐变量 → 解码器初始状态 ───────────────────────────────────────
        # z → 每步解码输入的全局步嵌入 (latent_dim → rule_embed_dim)
        self.latent2tgt_emb = nn.Sequential(nn.Linear(self.latent_dim, self.rule_embed_dim), nn.Tanh())
        # z → GRU 各层初始隐藏态 h_0, 每层独立投影
        self.latent2hidden = nn.ModuleList([
            nn.Sequential(nn.Linear(self.latent_dim, dec_hidden_dim), nn.Tanh())
            for _ in range(dec_num_layers)
        ])
        self.dec2vocab = nn.Linear(dec_hidden_dim, self.vocab_size)

        # ── NT 嵌入层 + rule→NT 查找表 ────────────────────────────────────
        self.num_nt_symbols = len(self.prod_rule_corpus.nt_symbol_list)
        self.pad_nt_idx = self.num_nt_symbols  # padding 索引置于末尾, 嵌入为零向量
        self.nt_embedding = nn.Embedding(
            self.num_nt_symbols + 1, self.rule_embed_dim, padding_idx=self.pad_nt_idx
        )
        self.register_buffer("rule_to_nt_idx", self._build_rule_to_nt_idx(), persistent=False)

        self._init_weights()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _ensure_grammar_device(self):
        target_device = self.device
        if self._grammar_device != target_device:
            self.prod_rule_corpus.to(target_device)
            self._grammar_device = target_device

    def _init_weights(self, init_range: float = 0.1):
        self.src_embedding.reset_parameters(init_range=init_range)
        self.tgt_embedding.reset_parameters(init_range=init_range)
        nn.init.uniform_(self.nt_embedding.weight, -init_range, init_range)
        with torch.no_grad():
            self.nt_embedding.weight[self.pad_nt_idx].zero_()
        if self.encoder_name == "Transformer":
            self.encoder.init_weights()

    def _build_rule_to_nt_idx(self) -> torch.Tensor:
        """构建 rule_id → NT_index 查找表; padding 位置映射到 pad_nt_idx。"""
        rule_to_nt = torch.full((self.vocab_size,), self.pad_nt_idx, dtype=torch.long)
        for prod_rule in self.prod_rule_corpus.prod_rule_list:
            rule_to_nt[prod_rule.rule_idx] = self.prod_rule_corpus._nt_symbol_to_idx[prod_rule.lhs_nt_symbol]
        return rule_to_nt

    def _left_padded_to_right_padded(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """左填充 → 右填充 (PackedGRUEncoder 要求右填充)。"""
        B, L = x.shape
        arange = torch.arange(L, device=x.device).unsqueeze(0)
        src_idx = (arange + (L - lengths).unsqueeze(1)).clamp(0, L - 1)
        out = x.gather(1, src_idx)
        out[arange >= lengths.unsqueeze(1)] = self.padding_idx
        return out

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """规则序列 x: (B, L) 左填充 → 后验参数 (μ, logσ²)。"""
        lengths = x.ne(self.padding_idx).sum(dim=1).clamp_min(1)
        emb = self.src_embedding(self._left_padded_to_right_padded(x, lengths))  # (B, L, D)

        if self.encoder_name == "Transformer":
            _, enc_last = self.encoder(emb, lengths)
        else:
            _, h_n = self.encoder(emb, lengths)  # h_n: (num_layers*num_dirs, B, H)
            num_dirs = 2 if self._bidirectional else 1
            enc_last = h_n[-num_dirs:].transpose(0, 1).reshape(x.size(0), -1)  # (B, num_dirs*H)

        if self.start_rule_embedding:
            first_emb = emb[:, 0]  # (B, D) — 起始规则嵌入
            mu     = torch.cat((self.hid2mean(enc_last),   self.emb2mean(first_emb)),   dim=1)
            logvar = torch.cat((self.hid2logvar(enc_last), self.emb2logvar(first_emb)), dim=1)
        else:
            mu, logvar = self.hid2mean(enc_last), self.hid2logvar(enc_last)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, stochastic: bool = True) -> torch.Tensor:
        """重参数化: stochastic=True 时 z = μ + σ·ε; stochastic=False 时返回 μ。"""
        if not stochastic:
            return mu
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)

    def sample_prior(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device or self.device)

    def _compose_input_step(
        self,
        prev_rule_ids: torch.Tensor | None,
        latent_step: torch.Tensor,
        nt_indices: torch.Tensor,
    ) -> torch.Tensor:
        """组合单步解码输入: prev_rule_emb + latent_step [+ nt_emb]。
        tgt_embedding(prev_rule_ids) │ 上一步选的产生式规则 embedding（第一步为 None 时跳过）        
        latent_step                  │ latent z 投影到 embedding 空间的全局上下文                                                                                                                                                      
        nt_embedding(nt_indices)     │ 当前非终结符的 embedding（可选，由 use_nt_conditioning 控制）
        """
        base = latent_step if prev_rule_ids is None else self.tgt_embedding(prev_rule_ids).unsqueeze(1) + latent_step
        if self.use_nt_conditioning:
            base = base + self.nt_embedding(nt_indices).unsqueeze(1)
        return self.tgt_dropout(base)

    def _teacher_forcing_inputs(self, z: torch.Tensor, out_seq: torch.Tensor) -> torch.Tensor:
        """构建 teacher forcing 输入: 目标序列右移一步 + 隐空间步嵌入 [+ NT 嵌入]。"""
        latent_step = self.latent2tgt_emb(z).unsqueeze(1)                         # (B, 1, D)
        rule_emb = self.tgt_embedding(out_seq)                                     # (B, T, D)
        inp = torch.cat([latent_step, rule_emb[:, :-1] + latent_step], dim=1)     # (B, T, D)
        if self.use_nt_conditioning:
            inp = inp + self.nt_embedding(self.rule_to_nt_idx[out_seq])
        return self.tgt_dropout(inp)

    def decode(self, z=None, out_seq=None, deterministic=True, return_hg_list=False):
        """从 z 解码: teacher forcing 返回 logits; 自回归返回 logits 或 (finished, hg_list)。"""
        if z is None:
            raise ValueError("GVAE decode requires explicit latent z")
        self._ensure_grammar_device()
        z = z.to(self.device)
        batch_size = z.size(0)
        hidden_dict = {"h": torch.stack([proj(z) for proj in self.latent2hidden], dim=0)}

        # ── Teacher forcing ───────────────────────────────────────────────
        if out_seq is not None:
            out_seq = out_seq.to(self.device)
            out, _ = self.decoder(self._teacher_forcing_inputs(z, out_seq), hidden_dict)
            return self.dec2vocab(out)

        # ── 自回归采样 ────────────────────────────────────────────────────
        with torch.no_grad():
            hg           = [None] * batch_size
            nt_sym       = [NTSymbol(0, False, []) for _ in range(batch_size)]
            nt_edge_list = [None] * batch_size
            finished     = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            logits_collect: list[torch.Tensor] = []
            prev_rule_ids = None
            latent_step = self.latent2tgt_emb(z).unsqueeze(1)  # (B, 1, D) 全局上下文, 每步复用

            for _ in range(self.max_vol):
                nt_indices = torch.tensor(
                    [self.prod_rule_corpus._nt_symbol_to_idx[nt_sym[i]] if not finished[i] else self.pad_nt_idx
                     for i in range(batch_size)],
                    dtype=torch.long, device=self.device,
                )
                inp = self._compose_input_step(prev_rule_ids, latent_step, nt_indices)
                dec_h, hidden_dict = self.decoder(inp, hidden_dict)
                vocab_logits = self.dec2vocab(dec_h).squeeze(1)  # (B, vocab_size)
                logits_collect.append(vocab_logits.unsqueeze(1))

                next_ids = torch.full((batch_size,), self.padding_idx, dtype=torch.long, device=self.device)
                for idx in (~finished).nonzero(as_tuple=False).view(-1).tolist():
                    prod_rule = self.prod_rule_corpus.sample(vocab_logits[idx, :-1], nt_sym[idx], deterministic)
                    next_ids[idx] = prod_rule.rule_idx
                    hg[idx], nt_edge_list[idx] = prod_rule.apply_to_graph(hg[idx], nt_edge_list[idx])
                    if nt_edge_list[idx]:
                        nt_sym[idx] = hg[idx].edge_attr(nt_edge_list[idx][-1])["symbol"]
                    else:
                        finished[idx] = True

                prev_rule_ids = next_ids
                if finished.all():
                    break

            final_logits = torch.cat(logits_collect, dim=1)
            if final_logits.size(1) < self.max_vol:
                final_logits = torch.cat([
                    final_logits,
                    final_logits.new_zeros(batch_size, self.max_vol - final_logits.size(1), self.vocab_size),
                ], dim=1)
            return (finished, hg) if return_hg_list else final_logits
