# foundation/models/finetune_models.py

import os
import torch
import torch.nn as nn
from hgr.utils.debug_utils import with_param_info
from hgr.foundation.models.ema import EMA

from hgr.foundation.models.grammar_encoder import RuleTransformerEncoder
# from hgr.foundation._ablation_models.multimodal_encoder import GraphGrammarEncoder
# from hgr.foundation._ablation_models.graphormer import GraphormerEncoder as RuleTransformerEncoder 
# from hgr.foundation._ablation_models.pureGNN import PureGNNEncoder as RuleTransformerEncoder
# from hgr.foundation._ablation_models.grammarGPS import GrammarAttnEncoder as RuleTransformerEncoder


@with_param_info()
class MLPProbe(nn.Module):
    def __init__(self, emb_dim, num_tasks, dropout=0.5): # dropout=0.5 比较好
        super().__init__()
        h1 = max(emb_dim // 2, 64)
        h2 = max(emb_dim // 4, num_tasks//2)
        self.net = nn.Sequential(
                   nn.LayerNorm(emb_dim),  nn.Dropout(dropout),
                   nn.Linear(emb_dim, h1), nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(h1), 
                   nn.Linear(h1, h2),  nn.GELU(), nn.Dropout(dropout), nn.LayerNorm(h2), 
                   nn.Linear(h2, num_tasks),
                   )
    def forward(self, h):
        return self.net(h)

@with_param_info()
class EncoderProbe(nn.Module):
    def __init__(self, cfg_grammar, cfg_probe):
        super().__init__()
        # self.encoder = GraphGrammarEncoder(cfg_graph, cfg_grammar)
        self.encoder = RuleTransformerEncoder(cfg_grammar)
        self.probe = MLPProbe(cfg_grammar.emb_dim, cfg_probe.num_tasks)
        # self.probe = nn.Linear(cfg_grammar.emb_dim, cfg_probe.num_tasks)
        self.feat_dropout = nn.Dropout(0.1)

        # self.sigreg = SIGRegLoss(num_slices=1024)

    def from_pretrained(self, ckpt):     
        msg = self.encoder.load_state_dict(ckpt['encoder'], strict=True) # 强制要求所有参数完全匹配
        print(msg)

        if 'ema' in ckpt:
            ema_pre = EMA(self.encoder, decay=0.999, use_num_updates=True, fp32_shadow=True)
            ema_pre.load_state_dict(ckpt['ema'])
            ema_pre.copy_to(self.encoder)
            del ema_pre
    
    def train(self, mode=True):
        super().train(mode)
        if getattr(self, "freeze_encoder", False):
            self.encoder.eval()
        return self

    def forward(self, batch, sigreg=False):
        _, graph_out = self.encoder(batch, return_atom_rep=False)
        # sigreg_loss = None
        # if sigreg and not getattr(self, "freeze_encoder", False):
        #     sigreg_loss = self.sigreg(graph_out.float())
        return self.probe(self.feat_dropout(graph_out)), None

        