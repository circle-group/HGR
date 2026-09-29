import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set

from hgr.foundation.models.gnn_layers import GINConv, GCNConv
from hgr.utils.debug_utils import with_param_info



###------------- decoder of node-level pretext task -------------###
@with_param_info()
class GNNDecoders(nn.Module):
    """
    Node-level decoder:
      - gnn_type in {"gin", "gcn", "linear"}
      - 支持多层
    """
    def __init__(self, emb_dim, args):
        super().__init__()
        hidden_dim = emb_dim
        out_dim = args.out_dim
        gnn_type = args.gnn_type
        num_layer = args.num_layer
        self._dec_type = gnn_type
        self.num_layer = num_layer
        self.activation = nn.PReLU()
        self.enc_to_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if gnn_type in {"gin", "gcn"}:
            self.conv = nn.ModuleList()
            # 第一层
            if gnn_type == "gin":
                self.conv.append(GINConv(hidden_dim, out_dim, aggr="add"))
            else:  # gcn
                self.conv.append(GCNConv(hidden_dim, out_dim, aggr="add"))
            # 后续层
            for _ in range(num_layer - 1):
                if gnn_type == "gin":
                    self.conv.append(GINConv(out_dim, out_dim, aggr="add"))
                else:
                    self.conv.append(GCNConv(out_dim, out_dim, aggr="add"))
        elif gnn_type == "linear":
            self.dec = nn.ModuleList()
            # 第一层
            self.dec.append(nn.Linear(hidden_dim, out_dim))
            # 后续层
            for _ in range(num_layer - 1):
                self.dec.append(nn.Linear(out_dim, out_dim))
        else:
            raise NotImplementedError(f"{gnn_type}")

    def forward(self, x, edge_index, edge_attr, mask_node_indices: Optional[torch.Tensor]):
        if self._dec_type == "linear":
            out = self.activation(x)
            out = self.enc_to_dec(out)
            if mask_node_indices is not None:
                out = out.clone()
                out[mask_node_indices] = 0
            for layer in self.dec:
                out = layer(out)
            return out
        else:
            x = self.activation(x)
            x = self.enc_to_dec(x)
            if mask_node_indices is not None:
                x = x.clone()
                x[mask_node_indices] = 0
            out = x
            for layer in range(self.num_layer):
                out = self.conv[layer](out, edge_index, edge_attr)
            return out

@with_param_info()
class SubgraphPredictor(nn.Module):
    def __init__(self, emb_dim,args):
        super().__init__()
        hidden_dim = emb_dim
        out_dim = args.out_dim

        self.pred_layers = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.PReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(1024, out_dim),
        )

    def forward(self, graph_rep):
        # graph_rep = self.pool(node_rep, batch)
        return self.pred_layers(graph_rep)

@with_param_info()
class DescriptorPredictor(nn.Module):
    def __init__(self, emb_dim, args):
        super().__init__()
        hidden_dim = emb_dim
        out_dim = args.out_dim

        self.pred_layers = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.PReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, graph_rep):
        # graph_rep = self.pool(node_rep, batch)
        return self.pred_layers(graph_rep)

@with_param_info()
class Decoder_DistPreds(nn.Module):
    """
    3D-level 预测器：对节点嵌入做 MLP 后两两计算欧氏距离（cdist）。
    """
    def __init__(self, emb_dim,args):
        super().__init__()
        in_dim = emb_dim
        hidden_dim = emb_dim
        out_dim = args.out_dim
        dropout = args.dropout
        self.node_emb = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout if args is not None else 0.0),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout if args is not None else 0.0),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x):
        node_emb = self.node_emb(x)
        return node_emb
        # return torch.cdist(node_emb, node_emb)
        # return pdist_by_graph(node_emb, node2graph)


