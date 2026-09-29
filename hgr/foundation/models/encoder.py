# foundation/models/encoder.py

import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import (
    global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
)
from hgr.foundation.data_utils.mol_defs import NUM_ATOM_TYPE, NUM_CHIRALITY_TAG
from hgr.foundation.models.gnn_layers import GINConv, GCNConv, GATConv, GraphSAGEConv
from hgr.utils.debug_utils import with_param_info





# ----------------------------- Encoders -----------------------------
@with_param_info()
class GNN(nn.Module):
    """
    Args:
        num_layers (int): number of GNN layers (>=2)
        emb_dim (int): hidden size
        JK (str): 'last' | 'concat' | 'max' | 'sum'
        drop_ratio (float): dropout
        gnn_type (str): 'gin' | 'gcn' | 'graphsage' | 'gat'

    Output:
        node representations (N, D or concat dims)
    """
    def __init__(self, gnn_cfg):
        super().__init__()

        self.cfg = cfg = gnn_cfg
        emb_dim = cfg.emb_dim
        
        if cfg.num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.num_layers = cfg.num_layers
        self.drop_ratio = cfg.dropout
        self.JK = getattr(cfg, "JK", "last")

        self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
        self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight)
        nn.init.xavier_uniform_(self.x_embedding2.weight)

        self.gnns = nn.ModuleList()
        gnn_type = getattr(cfg, "gnn_type", "gin")
        for _ in range(cfg.num_layers):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, emb_dim, aggr="add"))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim, emb_dim, heads=2, aggr="add"))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim, aggr="mean"))
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(cfg.num_layers)])

        self.pool = global_mean_pool

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, node2graph = argv
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, node2graph = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(self.num_layers):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_stack = torch.cat([h.unsqueeze(0) for h in h_list], dim=0)
            node_representation = torch.max(h_stack, dim=0)[0]
        elif self.JK == "sum":
            h_stack = torch.cat([h.unsqueeze(0) for h in h_list], dim=0)
            node_representation = torch.sum(h_stack, dim=0)
        else:
            raise ValueError(f"Unknown JK: {self.JK}")

        return node_representation, self.pool(node_representation, node2graph) # TODO：缺失graph-level representation


class GNN_graphpred(nn.Module):
    """
    Graph-level predictor on top of GNN encoder.
    """
    def __init__(self, num_layers, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin"):
        super().__init__()
        if num_layers < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.num_layers = num_layers
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        self.gnn = GNN(num_layers, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=nn.Linear((self.num_layers + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layers + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        self.mult = 2 if graph_pooling[:-1] == "set2set" else 1
        if self.JK == "concat":
            in_dim = self.mult * (self.num_layers + 1) * self.emb_dim
        else:
            in_dim = self.mult * self.emb_dim
        self.graph_pred_linear = nn.Linear(in_dim, self.num_tasks)

    def from_pretrained(self, model_file: str):
        state = torch.load(model_file, map_location="cpu")
        self.gnn.load_state_dict(state, strict=False)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")
        node_representation = self.gnn(x, edge_index, edge_attr)
        return node_representation, self.graph_pred_linear(self.pool(node_representation, batch))
