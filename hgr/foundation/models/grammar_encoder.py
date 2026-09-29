# foundation/models/grammar_encoder.py

import os, pickle, re
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from hgr.utils.debug_utils import with_param_info
from hgr.utils.file_utils import load_pickle


from hgr.foundation.data_utils.mol_defs import PAD, BOS, EOS, MASK, OFFSET
from hgr.foundation.models.attn_layer import RopeTransformerEncoderLayer, RotaryEmbedding
from hgr.foundation.models.attnbias import TreeStructBias, AtomGraphSpatialBias, AtomGraphEdgeBias
from hgr.grammar.rule_features import build_wl_vocab_and_csr, build_wl_csr_with_fixed_vocabs, build_rule_features

import logging
logger = logging.getLogger(__name__)

def init_weights(module):
    # 跳过内部有 _skip_init=True 的模块（给需要保留初始化的模块打标记 _skip_init）
    if getattr(module, "_skip_init", False):
        return

    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (torch.nn.LayerNorm, )):
        if module.weight is not None:
            torch.nn.init.ones_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)


def _prepare_Slist_and_vocabs(rule_list, cfg) -> Tuple[List[torch.Tensor], List[Dict[str, int]]]:
    """
    统一入口：
      * Finetune：若 cfg.pretrain_wl_csr_path 有效 → 读取 vocabs → 为当前 rules 构 CSR（新标签→<UNK>）。
      * 否则：构建 vocabs+CSR，并默认落盘到 grammar 同目录（便于复用）。
    """
            
    H = getattr(cfg, "wl_iterations", 4)

    if hasattr(cfg, "pretrain_grammar_path"): # Finetune
        # Finetune时必须读入vocabs，不允许cache 不可用还会尝试用 cfg.pretrain_grammar_path 重建 vocabs
        pretrain_grammar_tag = re.search(rf"grammar_({cfg.type}\d+)", cfg.pretrain_grammar_path).group(1)
        base_dir = os.path.dirname(os.path.abspath(cfg.pretrain_grammar_path))
        wlcsr_path = os.path.join(base_dir, f"wlcsr_H{H}_{pretrain_grammar_tag}.pkl")
        with open(wlcsr_path, "rb") as f:
            obj = pickle.load(f)
        vocabs = obj.get("vocabs", None)
        assert vocabs is not None, "Failed to load vocabs from cache"
        # 用固定 vocabs 为当前 rules 构 CSR
        S_list = build_wl_csr_with_fixed_vocabs(
            rule_list, fixed_vocabs=vocabs, H=H, size_norm=True, ensure_unk=True)
        logger.info(f"[RuleTransformerEncoder] FINETUNE using fixed vocabs (H={H})")
        
    else:
        # PRETRAIN：构建并保存
        base_dir = os.path.dirname(os.path.abspath(cfg.grammar_path))
        wlcsr_path = os.path.join(base_dir, f"wlcsr_H{H}_{cfg.type}{len(rule_list)}.pkl")

        if os.path.exists(wlcsr_path):
            with open(wlcsr_path, "rb") as f:
                obj = pickle.load(f)
            S_list, vocabs = obj.get("S_list", None), obj.get("vocabs", None)
            logger.info(f"[RuleTransformerEncoder] PRETRAIN loaded WL cache from {os.path.abspath(wlcsr_path)}")
        else:
            vocabs, S_list, _ = build_wl_vocab_and_csr(
                rule_list, H=H, size_norm=True, add_unk=True, save_path=wlcsr_path)
            logger.info(f"[RuleTransformerEncoder] PRETRAIN built & saved WL cache to {os.path.abspath(wlcsr_path)}")
    
    return S_list, vocabs





@with_param_info()
class RuleTransformerEncoder(nn.Module):
    """
    Simplified RuleTransformerEncoder (Linear Mode Only).
    仅使用 WLHistogramEmbedder 提取固定特征，并通过 Linear 投影。
    """
    def __init__(self, cfg):
        super().__init__()
        # ---- 基本参数 ----
        self.d_model: int = cfg.emb_dim
        self.d_out: int = cfg.emb_dim
        
        # ---- Grammar & WL-CSR & Bank 初始化 ----
        grammar = load_pickle(cfg.grammar_path, verbose=False)
        S_list, _ = _prepare_Slist_and_vocabs(grammar.prod_rule_list, cfg)
        self._init_special_embedding() 
        self._init_bank(S_list, grammar.prod_rule_list)
        self.num_rules = len(grammar.prod_rule_list)
        logger.info(f"[RuleTransformerEncoder] Loaded grammar: {self.num_rules} rules")
        
        
        # ---- 规则嵌入器 (Fixed Linear Mode) ----
        # 仅使用直方图嵌入器
        # self.embedder = WLHistogramEmbedder(S_list)
        
        self.pre_ln = (nn.Sequential(nn.LayerNorm(self.d_model), nn.Linear(self.d_model, self.d_model))
                       if getattr(cfg, "use_rule_proj", False) else nn.Identity())

        # # ---- 规则 ID 残差 (可选) ----
        # d_id = int(getattr(cfg, "rule_id_dim", 0) or 0) 
        # self.rule_id, self.id_proj = None, None
        # if d_id > 0:
        #     self.rule_id = nn.Embedding(self.num_rules, d_id)
        #     nn.init.normal_(self.rule_id.weight, std=0.02)
        #     self.id_proj = nn.Linear(self.d_model + d_id, self.d_model)

        # ---- Transformer（depth ROPE）----
        nhead, nL, dropout = cfg.nhead, cfg.num_layers, cfg.dropout
        rotary_pct = float(getattr(cfg, "rotary_pct", 1.0))
        rope_base = float(getattr(cfg, "rope_base", 1000.0)) # 树不会很深，因此从10000改成了1000，
        # 如果 base 太小（例如100），位置之间的相位旋转变化率过大，高频分量可能旋转过快，导致模型对微小的深度变化过于敏感，难以泛化。

        head_dim = self.d_model // nhead
        rotary_dim = int(head_dim * rotary_pct)
        rotary_dim = rotary_dim - (rotary_dim % 2)
        self.rope = RotaryEmbedding(rotary_dim=rotary_dim, base=rope_base) # 共享的 RoPE embedding（按树 depth）

        # 共享的树结构偏置（relation + distance bucket）
        assert cfg.nhead % 4 == 0
        self.heads_per_type = cfg.nhead // 4 # 计算每种 Bias 分配到的 Head 数量
        self.bias_scales = nn.Parameter(torch.zeros(nL, 3))  # (nL, 3)


        max_path_distance = getattr(cfg, "max_path_distance", 4)
        self.struct_bias_module = TreeStructBias(num_dist_buckets=24, max_exact=12, max_distance=32,
                                                 num_heads=self.heads_per_type)
        self.spatial_bias_module = AtomGraphSpatialBias(max_path_distance=max_path_distance, num_heads=self.heads_per_type)
        self.edge_bias_module = AtomGraphEdgeBias(max_path_distance=max_path_distance, num_heads=self.heads_per_type)
        
        self.layers = nn.ModuleList([
            RopeTransformerEncoderLayer(
                d_model=self.d_model, nhead=nhead, dim_feedforward=4 * self.d_model,
            ) for _ in range(nL)
        ])
        
        self.final_ln = nn.LayerNorm(self.d_model)
        self.proj = nn.Identity() if self.d_out == self.d_model else nn.Linear(self.d_model, self.d_out)
                
        self.apply(init_weights)

    
    @torch.no_grad()
    def _init_special_embedding(self):
        # 初始化 PAD/BOS/EOS 向量
        self.special = nn.Embedding(OFFSET, self.d_model, padding_idx=PAD)
        self.special._skip_init = True
        self.special.weight.zero_()
        g = torch.Generator(device='cpu'); g.manual_seed(114514)
        for idx in [BOS, EOS, MASK]:
            self.special.weight[idx].normal_(0.0, 0.02, generator=g)

    @torch.no_grad()
    def _init_bank(self, S_list, rule_list): 
        """
        初始化 Linear 模式的 Bank。
        1. 计算所有规则的原始 WL 直方图特征 (Z_raw)
        2. 拼接到 token_bank
        3. 初始化投影层 wl_linear_proj
        """
        # raw bank: [OFFSET+R, V_sum]
        with torch.no_grad():
            # embedder = WLHistogramEmbedder(S_list)
            # Z_raw = embedder.get_rule_embeddings(rule_list)  # [R, V_sum]
            Z_raw = build_rule_features(rule_list, S_list)
        
        V_sum = Z_raw.size(1)
        # special tokens 在 raw 空间全为 0，具体值在 forward 时通过 self.special 覆盖
        sp = torch.zeros((OFFSET, V_sum), dtype=Z_raw.dtype) 
        
        # 注册为 buffer，不参与梯度更新，
        self.register_buffer("token_bank", torch.cat([sp, Z_raw], dim=0), persistent=False)
        
        self.wl_linear_proj = nn.Sequential(
            nn.LayerNorm(V_sum),  
            nn.Linear(V_sum, self.d_model, bias=False)
        )
        logger.info(f"[RuleTransformerEncoder] Init Linear bank: {tuple(self.token_bank.shape)} -> D={self.d_model}")

    def init_rule_embedding(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Linear 模式嵌入逻辑：
        1. 查 raw bank (直方图特征) -> 线性投影
        2. 覆盖 Special Tokens (BOS/EOS/MASK)
        3. 叠加 Rule ID 残差 (如果启用)
        """
        is_special = (seq < OFFSET)

        # 1. 基础特征：Raw特征 -> 投影 -> d_model
        # [B, L, V_sum]
        x_raw = F.embedding(seq, self.token_bank, padding_idx=PAD)
        # [B, L, D]
        x = self.wl_linear_proj(x_raw)
        
        # 2. 覆盖 Special Tokens
        # if is_special.any():
        x[is_special] = self.special(seq[is_special]).to(x.dtype)
            
        return x

    def forward(self, batch, return_atom_rep=True):
        """
        Returns:
            atom_rep: (N_total, D) or None
            mol_rep:  (B, D)
        """
        
        seq_batch = batch.rule_seq_padded       # (B, L_seq), long.
        depths = batch.grammar_depths           # (B, L_seq), long. 每个token在树中的深度, for depth-based rope, 
        atom_num = batch.atom_num               # (B,), long. 每张图的原子数 n_b, for SpatialBias & EdgeBias
        atom_pos_padded = batch.atom_pos_padded # (B, Nmax), long.  每个元素是该原子在 rule_seq 中对应的 token 位置      for SpatialBias & EdgeBias
        dist_mat = batch.grammar_distances      # (B, L_seq, L_seq) for struct-bias, 每个token到其他token的距离
        rel_mat = batch.grammar_relations       # (B, L_seq, L_seq) for struct-bias, 每个token到其他token的关系
        
        node_paths_length = batch.node_paths_length        # (B, Nmax, Nmax),   for SpatialBias
        edge_paths_tensor = batch.edge_paths_tensor        # (B, Nmax, Nmax, D), for EdgeBias
        edge_paths_length = batch.edge_paths_length        # (B, Nmax, Nmax),   for EdgeBias
        edge_attr         = batch.edge_attr                # (E_total, 2),      for EdgeBias
        edge_ptr          = batch.edge_ptr                 # (B+1,),            for EdgeBias

        device = seq_batch.device
        B, L_seq = seq_batch.shape

        # 1. Embedding
        x = self.init_rule_embedding(seq_batch)
        h = self.pre_ln(x)
        
        # # # 2. 计算 Tree-RoPE 相位 cis
        max_depth = int(depths.max().item()) + 1
        cis_table = self.rope.get_cis(max_depth, device=x.device)     # (max_depth, D_rot/2)
        cis_pos = cis_table[depths]                                   # (B, L_seq, D_rot/2)
        cis = cis_pos.unsqueeze(1)                                    # (B,1,L_seq,D_rot/2)

        # 3. 计算结构偏置
        struct_bias = self.struct_bias_module(dist_mat, rel_mat).to(dtype=x.dtype)  # (B,nhead,L_seq,L_seq) 

        spatial_bias = self.spatial_bias_module(L_seq, atom_pos_padded, atom_num, node_paths_length, dtype=x.dtype)  # (B,nhead,L,L)
        edge_bias = self.edge_bias_module(L_seq, atom_pos_padded, atom_num,
            edge_paths_tensor, edge_paths_length, edge_attr, edge_ptr, dtype=x.dtype)  # (B,nhead,L,L)
        
        
        
        # 4. Masking & Transformer (float mask: 0=keep, -inf=mask; bool mask: True=keep, False=mask)
        attn_keep = (seq_batch != PAD)                       # (B, L_seq), True=keep
        bool_mask = attn_keep.view(B, 1, 1, L_seq) # (B,1,1,L) broadcastable
        float_mask = torch.zeros((B, 1, 1, L_seq), device=device, dtype=x.dtype) # 从 attn_mask 转换为 float mask (float additive mask， 0=keep, -inf=mask)
        float_mask.masked_fill_(~bool_mask, float("-inf")) # (B, 1, 1, L_seq), 以便正确广播到 (B, Heads, Query_L, Key_L), 这样明确告诉 SDPA：这是对 Key (最后一维) 进行 Mask
        

        # 5) Transformer
        for li, lyr in enumerate(self.layers):
            scales = torch.tanh(self.bias_scales[li])
            # scales = self.bias_scales[li].to(dtype=x.dtype)

            combined_attn_bias = float_mask + torch.cat([struct_bias * scales[0], 
                                                    spatial_bias * scales[1], 
                                                    edge_bias * scales[2]
                                                    ], dim=1).to(dtype=x.dtype) # (B, 3 * H/4, L, L)

            # no tree bias
            # combined_attn_bias = float_mask + torch.cat([spatial_bias * scales[1],  edge_bias * scales[2]], dim=1)

            # no graphbias
            # combined_attn_bias = float_mask + struct_bias * scales[0] 


            h = lyr(h, cis=cis, attn_mask=bool_mask, attn_bias=combined_attn_bias)
            # h = lyr(h, cis=None, attn_mask=bool_mask)

        h = self.final_ln(h)
        h_all = self.proj(h)  # (B, L_seq, D_out)

        # 6) 分子表示：取每个样本“最后一个有效 token”(batch.seq_length中是已经包含BOS/EOS的长度，因此-2才是最后一个有效token)
        mol_rep = h_all[torch.arange(B, device=device), batch.seq_length -2]     # (B, D_out),

        atom_rep = None
        if return_atom_rep:
            Nmax = atom_pos_padded.size(1)
            mask = torch.arange(Nmax, device=device)[None, :] < atom_num[:, None]        # (B,Nmax)
            pos_flat = atom_pos_padded.masked_select(mask)                               # (N_total,) 将atom_pos_padded中的pad删除，然后拉成向量
            b_idx = torch.arange(B, device=device).repeat_interleave(atom_num)           # (N_total,) 
            atom_rep = h_all[b_idx, pos_flat]                                            # (N_total, D)
            assert atom_rep.size(0) == batch.pos.size(0), f"atom_rep N={atom_rep.size(0)} but pos N={batch.pos.size(0)}"


        return atom_rep, mol_rep



