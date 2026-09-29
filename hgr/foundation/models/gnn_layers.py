# foundation/models/gnn_layers.py

import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn.inits import glorot, zeros
from hgr.foundation.data_utils.mol_defs import NUM_BOND_TYPE, NUM_BOND_DIRECTION


# ----------------------------- Utils -----------------------------
def _add_self_loops_with_attr(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    num_nodes: int,
    self_loop_type_idx: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    为没有属性的 add_self_loops 增加一份与节点同数目的自环边属性（类型=4，方向=0）。
    返回新的 edge_index 与拼接后的 edge_attr。
    """
    # num_nodes = edge_index.max().item() + 1
    # 1) add self-loops to edge_index
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    # 2) build self-loop attributes (shape: [num_nodes, 2])
    # 为这些自环边补一份“边属性”：第 0 列是“键类型”(bond type, default: 4)，第 1 列是“方向”(bond direction)。
    # 注意：先用 edge_attr 的 device/dtype 构造，再在取嵌入前统一 long()
    self_loop_attr = torch.zeros((num_nodes, 2), device=edge_attr.device, dtype=edge_attr.dtype)
    self_loop_attr[:, 0] = self_loop_type_idx  # bond type for self-loop edge (default: 4)

    # 3) cat edge_attr 把原有 edge_attr 与自环的属性拼起来（行拼接）
    edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
    return edge_index, edge_attr


# ----------------------------- Convs -----------------------------
class GINConv(MessagePassing):
    """
    GIN with edge embeddings.
    See: https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim: int, out_dim: int, aggr: str = "add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, out_dim),
        )
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight)
        nn.init.xavier_uniform_(self.edge_embedding2.weight)

    def forward(self, x, edge_index, edge_attr):
        # 已经提前到外面进行，避免每层重复
        # edge_index, edge_attr = _add_self_loops_with_attr(edge_index, edge_attr, num_nodes=x.size(0))
        
        e1 = self.edge_embedding1(edge_attr[:, 0]) # embeddings 索引必须为 Long
        e2 = self.edge_embedding2(edge_attr[:, 1])
        edge_embeddings = e1 + e2
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim: int, out_dim: int, aggr: str = "add", **kwargs):
        kwargs.setdefault('aggr', aggr)
        super().__init__(**kwargs)
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, out_dim)
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    @staticmethod
    def norm(edge_index: torch.Tensor, num_nodes: int, dtype) -> torch.Tensor:
        # 标准写法：使用目标点列作为度的统计（对称归一化）
        row, col = edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)
        deg = degree(col, num_nodes=num_nodes, dtype=dtype)  # 目标点度
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = _add_self_loops_with_attr(edge_index, edge_attr, num_nodes=x.size(0))
        e1 = self.edge_embedding1(edge_attr[:, 0].long())
        e2 = self.edge_embedding2(edge_attr[:, 1].long())
        edge_embeddings = e1 + e2
        norm = self.norm(edge_index, x.size(0), x.dtype)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

    def update(self, aggr_out):
        return self.linear(aggr_out)


class GATConv(MessagePassing):
    """
    Simple multi-head GAT with edge embeddings added to neighbor features.
    注：保留 out_dim 形参以兼容，但输出维度仍为 emb_dim（对 heads 平均后）。
    """
    def __init__(self, emb_dim, out_dim, heads=2, negative_slope=0.2, aggr = "add"):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, heads * emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, heads * emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = _add_self_loops_with_attr(edge_index, edge_attr, num_nodes=x.size(0))
        e1 = self.edge_embedding1(edge_attr[:, 0].long())
        e2 = self.edge_embedding2(edge_attr[:, 1].long())
        edge_embeddings = (e1 + e2)  # [E, H*D]

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)  # [N, H, D]
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_i, x_j, edge_attr, edge_index_i):
        # edge_attr: [E, H*D] -> [E, H, D]
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j = x_j + edge_attr

        # 注意：按目标点 i 分组归一化
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # [E, H]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i)  # group by target i

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        # aggr_out: [N, H, D] -> mean over heads -> [N, D]
        aggr_out = aggr_out.mean(dim=1)
        return aggr_out + self.bias


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim: int, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
        self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = _add_self_loops_with_attr(edge_index, edge_attr, num_nodes=x.size(0))
        e1 = self.edge_embedding1(edge_attr[:, 0].long())
        e2 = self.edge_embedding2(edge_attr[:, 1].long())
        edge_embeddings = e1 + e2

        x = self.linear(x)
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)