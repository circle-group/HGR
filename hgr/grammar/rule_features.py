# grammar/rule_features.py

"""
------------------------------------------------------------
本文件的目标：
   为 grammar 中的每条 production rule 生成“初始化特征向量”。
   目前组合了三类特征：
     1) WL(Weisfeiler-Lehman) 直方图特征：基于 rule.rhs.hg 的离散标签迭代统计
     2) RWPE(Random Walk Positional Encoding) 特征：基于 atom graph 的随机游走回返概率，排序后展平
     3) Explicit(ad-hoc) 显式属性特征：对单原子规则提取原子属性 one-hot + (anchor_num, num_edges)

典型用途：
   - 在模型初始化阶段一次性计算 Z_raw，用作 rule embedding bank 的原始特征。
------------------------------------------------------------
"""

import os, pickle
from collections import Counter
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from hgr.utils.debug_utils import with_param_info
from typing import List, Optional
import networkx as nx

UNK_TOKEN = "<UNK>"  # 所有“不在词表中”的 WL 字符串标签统一映射到该 token
ADHOC_VOCABS = {
    'symbol': ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'B', 'Si'],
    'charge': [-2, -1, 0, 1, 2],
    'hs':     [0, 1, 2, 3, 4]
}


def _stringify_val(x):
    """
    将常见类型稳定转字符串（用于构造可复现的离散标签）。
    为什么要做这个：
      - WL 第 0 轮需要把节点属性映射为离散标签
      - 如果直接 str(x)，对于 None/bool 等类型可能出现不统一表现
      - 统一转字符串可以提升复现性与稳定性
    """
    if x is None: return "NA"
    if isinstance(x, bool): return "T" if x else "F"
    return str(x)





def _wl_labels_per_iter(rule, H):
    """
    对单条 rule 的 RHS 超图执行 H 轮 1-WL(Weisfeiler-Lehman) 标签迭代。

    返回长度 H 的列表，每个元素是“节点→字符串标签”的字典。
    - 第 0 轮：由节点属性 token 归并得到的标签；
    - 第 1..H-1 轮：中心标签 + 有序邻居标签的串接（经典 1-WL 变体）。
      这里将邻居标签排序后拼接，确保**邻接顺序不影响**结果。
    """
    G = rule.rhs.hg
    anchors = getattr(rule.lhs, "_nodes_", tuple())

    def _canonical_label_from_tokens(attr_tokens):
        """ 
        按字典序排序后连接，保证**顺序无关性**。 这样同一组属性的排列顺序不会影响标签字符串，提升复现性。
        """
        return "##".join(sorted(attr_tokens))

    def _node_attr_tokens(v, data, anchors):
        """
        根据节点属性构造稳定的“原子标签 token 列表”。
        - 注意：这里的具体字段仅作示例，可按项目需求扩展/替换，
        但务必维持**相同输入产生相同 token 集合**的性质。
        """
        tokens = []
        sym = data.get("symbol", None)

        node_type = data.get("bipartite", None)
        tokens.append(f"A:NTYPE={_stringify_val(node_type)}")
        if node_type == "node":  # bond-like
            tokens.append(f"A:BOND_TYPE={_stringify_val(getattr(sym,'bond_type',None))}")
            tokens.append(f"A:ANCHOR={_stringify_val(v in anchors)}")
            tokens.append(f"A:BOND_IS_RING={_stringify_val(data.get('is_in_ring',None))}")
            tokens.append(f"A:BOND_AROMATIC={_stringify_val(getattr(sym,'is_aromatic',None))}")
            tokens.append(f"A:STEREO={_stringify_val(getattr(sym,'stereo',None))}")
        elif node_type == "edge":  # atom-like
            tokens.append(f"A:TERMINAL={_stringify_val(data.get('terminal',None))}")
            tokens.append(f"A:SYMBOL={_stringify_val(getattr(sym,'symbol',None))}")
            tokens.append(f"A:ATOM_AROMATIC={_stringify_val(getattr(sym,'is_aromatic',None))}")
            tokens.append(f"A:DEGREE_VAL={_stringify_val(getattr(sym,'degree',None))}")
            tokens.append(f"A:ATOM_IS_RING={_stringify_val(data.get('is_in_ring',False))}") # NOTE: 这个好像都是false，后面检查一下看看能否删掉
        else:
            raise ValueError(f"Unknown node type: {node_type}")

        # 如果所有字段都为 NA，额外附加一个占位，保证不同节点不会完全坍缩
        if all(t.endswith("=NA") for t in tokens): tokens.append("A:PLACEHOLDER=1")
        return tokens

    # 第 0 轮：属性标签
    labels = {v: _canonical_label_from_tokens(_node_attr_tokens(v, data, anchors)) for v, data in G.nodes(data=True)}
    out = [labels.copy()]
    # 后续迭代：中心 + 有序邻居
    for _ in range(1, H):
        new_labels = {}
        for v in G.nodes():
            neigh = sorted(labels[u] for u in G.neighbors(v))
            new_labels[v] = labels[v] + "||" + ",".join(neigh)
        labels = new_labels
        out.append(labels.copy())
    return out



def _csr_from_counts_per_rule(counts_per_rule, vocab, size_norm, unk_id):
    """
    将“每条规则的标签计数 Counter”转换为 CSR 稀疏矩阵。
      - 对于某一轮 WL（固定 t），每条规则会产生若干节点标签
      - 我们把这些标签做成直方图：hist[label] = count(label)
      - 再将所有规则的直方图堆叠为矩阵 S_t ∈ R^{R x V_t}

    重要不变量：
      1) 行聚合：同一行的相同列先相加再写入（无重复列索引）。
      2) 列升序：每行内列索引严格升序（满足 PyTorch CSR 约束）。
      3) UNK 汇聚：未知标签聚合到 unk_id（若指定），否则直接丢弃。

    参数
        counts_per_rule: 长度为 R（规则数）的列表。每条规则（行）的标签计数。每个元素是一个 Counter 字典，记录了该规则中出现了哪些标签及其次数
        vocab:           当前迭代轮次的的词表：label → 列号。
        size_norm:       是否对行进行 L1 归一化（直方图概率），否则使用原始计数。
        unk_id:          UNK 列号；若 <0 表示禁用 UNK（未知直接忽略）。

    返回
        torch.Tensor (CSR) 形状 [R, V_t]
    """
    R, Vt = len(counts_per_rule), len(vocab)
    # indptr, indices, data = [0], [], []

    data = [] # 所有非零元素的值。
    indices = [] # data 中每个元素对应的列号。
    indptr = [0] # 指向每行在 data 和 indices 中的起始位置。
    for ri in range(R):
        cnt, total, row_map = counts_per_rule[ri], sum(counts_per_rule[ri].values()) or 1, {}
        # 1) 将字符串标签映射到列索引，并聚合同列
        for s, c in cnt.items():
            j = vocab.get(s, unk_id if unk_id >= 0 else None) # 查表：将字符串标签 s 转为列索引 j
            if j is None: continue                            # 未知且禁用 UNK：直接跳过
            val = (c / total) if size_norm else float(c)      # 计算归一化后的值
            row_map[j] = row_map.get(j, 0.0) + val           # 聚合：同一列先相加

        # 2) 写入：列升序保证 CSR 合法性
        for j in sorted(row_map.keys()):
            indices.append(j); data.append(row_map[j])
        indptr.append(len(indices))

    indptr_t = torch.tensor(indptr, dtype=torch.int64)
    indices_t = torch.tensor(indices, dtype=torch.int64)
    data_t = torch.tensor(data, dtype=torch.float32)
    return torch.sparse_csr_tensor(indptr_t, indices_t, data_t, size=(R, Vt))

def build_wl_vocab_and_csr(rules, H=4, size_norm=True, add_unk=True, save_path=None):
    """
    运行 WL 算法生成原始字符串标签，并统计出全局的词表。
    扫描规则集合，构建**每轮**的（1）vocab 和（2）CSR。
    - vocab：每轮独立统计；可选加入 UNK 列（便于后续迁移/微调）。
    - CSR：严格满足“行聚合 + 列升序”。

    返回
    ----
    vocabs:  List[Dict[str,int]]，长度 H
    S_list:  List[torch.Tensor(CSR)]，长度 H， 其中第 $t$ 个元素是一个形状为 [R, Vt] 的稀疏矩阵，代表所有规则在第 $t$ 轮 WL 迭代下的特征分布
    unk_ids: List[int]，与 vocabs 对应；若未启用 UNK，则为 -1
    """
    R = len(rules)
    per_rule_iter_counts = [[] for _ in range(R)] # 存储每个规则在每次迭代中的标签计数 # per_rule_iter_counts[i][j] 是一个 Counter 对象，记录了第 i 条规则在第 j 轮 WL 迭代中所有标签的出现次数。
    per_iter_all_labels = [[] for _ in range(H)] # 收集每次迭代中出现过的所有标签

    # 1) 收集 labels，并按（规则×迭代）计数
    for ri, rule in enumerate(rules):
        labs_list = _wl_labels_per_iter(rule, H)
        for t in range(H):
            cnt = Counter(labs_list[t].values())
            per_rule_iter_counts[ri].append(cnt)
            per_iter_all_labels[t].extend(cnt.keys())

    # 2) 构建每轮 vocab（可选加入 UNK）
    vocabs, unk_ids = [], []
    for t in range(H):
        uniq = sorted(set(per_iter_all_labels[t])) # 对当前轮次 t 的所有标签去重并排序
        vocab = {s: i for i, s in enumerate(uniq)} # 建立映射：字符串 -> ID
        if add_unk: 
            vocab[UNK_TOKEN] = len(vocab)
            unk_ids.append(vocab[UNK_TOKEN])
        else: 
            unk_ids.append(-1)
        vocabs.append(vocab)

    # 3) 构建每轮 CSR
    S_list = []
    for t in range(H):
        counts_t = [per_rule_iter_counts[ri][t] for ri in range(R)]
        S_list.append(_csr_from_counts_per_rule(counts_t, vocabs[t], size_norm, unk_ids[t]))
    
    # 4) 可选落盘（full + vocabs-only）
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f: 
            pickle.dump({"H": H, "vocabs": vocabs, "S_list": S_list, "unk_ids": unk_ids}, f)

    return vocabs, S_list, unk_ids


def build_wl_csr_with_fixed_vocabs(rules, fixed_vocabs, H, size_norm=True, ensure_unk=True):
    """
    用**已给定的预训练 vocabs**为“当前规则集”构 CSR。
    - 所有“新出现但不在预训练 vocab 的标签”都会被汇聚到 UNK；
      若 ensure_unk=False 则丢弃（通常不建议）。
    """
    assert len(fixed_vocabs) == H, "fixed_vocabs 与 H 的长度必须一致"
    R = len(rules)
    # 1) 重新扫描标签（按当前 rule 集）并计数
    per_rule_iter_counts = [[] for _ in range(R)]
    for ri, rule in enumerate(rules):
        labs_list = _wl_labels_per_iter(rule, H)
        for t in range(H): 
            per_rule_iter_counts[ri].append(Counter(labs_list[t].values()))

    # 2) 构建 CSR（使用固定 vocab）
    S_list = []
    for t in range(H):
        vocab = fixed_vocabs[t]
        unk_id = vocab[UNK_TOKEN] if ensure_unk and (UNK_TOKEN in vocab) else -1
        counts_t = [per_rule_iter_counts[ri][t] for ri in range(R)]
        S_list.append(_csr_from_counts_per_rule(counts_t, vocab, size_norm, unk_id))
    return S_list





# -------------------------------------------------------
# 计算图的 RWPE
# -------------------------------------------------------

def compute_graph_rwpe(G: nx.Graph, k_steps: int = 12, max_nodes: int = 30) -> torch.Tensor:
    """
    计算图的 RWPE(Random Walk Positional Encoding)，并输出“置换不变”的图级扁平特征。

    核心思想：
      - 先对每个节点 i 计算回返概率向量：
          p_i = [ (P^1)_{ii}, (P^2)_{ii}, ..., (P^k)_{ii} ] ∈ R^k
        其中 P = D^{-1}A 为随机游走转移矩阵
      - 得到所有节点的矩阵 RWPE ∈ R^{n x k}
      - 对节点按 RWPE 字典序排序，获得 Canonical Ordering
      - 截断/补零到 max_nodes 后展平为固定维度 [max_nodes * k_steps]

    输入:
        G: NetworkX 图对象。
        k_steps: 随机游走的步数（也就是我们要看多远的结构特征）。
        max_nodes: 模型接受的最大节点数（用于统一输出维度）。
    输出:
        一个展平的 PyTorch Tensor，形状为 [max_nodes * k_steps]。
    """

    # 1. 获取全图邻接矩阵 (不要在这里切片！)
    A = nx.to_numpy_array(G, dtype=np.float32)
    num_nodes = A.shape[0]
    
    # 特殊情况：空图
    if num_nodes == 0:
        return torch.zeros(max_nodes * k_steps)

    # 2. 在全图上计算 P = D^-1 * A
    D_vec = np.sum(A, axis=1)
    D_inv = np.zeros_like(D_vec)
    mask = D_vec > 0
    D_inv[mask] = 1.0 / D_vec[mask]
    
    P = D_inv[:, None] * A 
    
    # 3. 迭代计算 RWPE (针对所有节点)
    rwpe_list = []
    P_curr = P.copy()
    
    for _ in range(k_steps):
        diag = np.diagonal(P_curr)
        rwpe_list.append(diag)
        P_curr = P_curr @ P
        
    rwpe_matrix = np.stack(rwpe_list, axis=1) # [num_nodes, k]
    
    # 4. 排序 (Sorting) - 这一步保证置换不变性
    # 此时我们还是对 num_nodes 个节点进行排序，首先比较第 1 步 RWPE 值；如果相等，则比较第 2 步的值；如果还相等，比较第 3 步……以此类推。
    # 这样就保证了图的Canonical Ordering（规范顺序）：无论你给节点的初始编号是什么，只要它们的随机游走特征相同，它们就会被排到相同的位置。
    rwpe_rounded = np.round(rwpe_matrix, decimals=6)
    sorted_idx = np.lexsort(rwpe_rounded.T[::-1, :]) 
    sorted_rwpe = rwpe_matrix[sorted_idx] # [num_nodes, k] 已排序
    
    # 5. 截断 或 填充 (Truncate or Pad)
    # 现在的 sorted_rwpe 是完整的、正确的结构特征，我们在这里进行长度规范化
    
    if num_nodes > max_nodes:
        # 截断：只取排序后的前 max_nodes 个
        # 因为已经排过序了，所以无论原图编号如何，这里取到的总是“相同特征”的那一批节点
        final_rwpe = sorted_rwpe[:max_nodes, :] 
    else:
        # 填充：如果节点数不够，在后面补 max_nodes - num_nodes 行，每行 k 列的 0
        final_rwpe = np.pad(sorted_rwpe, ((0,  max_nodes - num_nodes), (0, 0)), 'constant')
        
    # 6. 展平
    flat_rwpe = final_rwpe.flatten() # [max_nodes * k]
    
    return torch.from_numpy(flat_rwpe).float()






def compute_explicit_attrs(rule):
    """
    提取规则的显式属性 。
    返回: Concatenated Tensor [Symbol | Charge | Aromatic | Hs | Meta]
    """
    def _get_one_hot(val, vocab):
        """辅助函数：生成带 Unknown 位的 One-Hot 向量"""
        vec = np.zeros(len(vocab) + 1, dtype=np.float32)
        try:
            idx = vocab.index(val)
            vec[idx] = 1.0
        except ValueError:
            vec[-1] = 1.0  # Unknown/Other
        return vec

    def _get_zeros(vocab):
        """辅助函数：生成全零向量 (用于多原子情况占位)"""
        return np.zeros(len(vocab) + 1, dtype=np.float32)

    rhs = rule.rhs
    num_atom_edges = rhs.num_edges
    
    # === 2. 特征提取 ===
    feature_chunks = []
    
    # 判断是否为单原子规则 (且存在边)
    if num_atom_edges <= 1:
        # 获取原子对象 (假设存储在唯一的 edge 属性 'symbol' 中)
        # 注意: 这里兼容 list 或 iterator 返回的 edges
        sym = rhs.edge_attr(rule.rhs.edges[0])['symbol']

        # 生成特征向量
        feature_chunks.append(_get_one_hot(sym.symbol, ADHOC_VOCABS['symbol']))
        feature_chunks.append(_get_one_hot(sym.formal_charge, ADHOC_VOCABS['charge']))
        feature_chunks.append(np.array([1.0] if sym.is_aromatic else [0.0], dtype=np.float32))
        feature_chunks.append(_get_one_hot(sym.num_explicit_Hs, ADHOC_VOCABS['hs']))
    else:
        # 多原子规则：特征部分全部填 0
        feature_chunks.append(_get_zeros(ADHOC_VOCABS['symbol']))
        feature_chunks.append(_get_zeros(ADHOC_VOCABS['charge']))
        feature_chunks.append(np.zeros(1, dtype=np.float32)) # Aromatic
        feature_chunks.append(_get_zeros(ADHOC_VOCABS['hs']))

    # === 3. 拼接元数据: [Anchor数量, 原子总数] ===
    anchor_num = len(rule.ext_node)  # Anchor 数量
    feature_chunks.append(np.array([anchor_num, num_atom_edges], dtype=np.float32))

    # === 4. 合并并转为 Tensor ===
    final_vec = np.concatenate(feature_chunks)
    return torch.from_numpy(final_vec)



def build_rule_features(rule_list, S_list):
    """
    构建所有规则的初始化特征矩阵 Z_raw。

    输入：
      rule_list: List[rule]
      S_list:    List[CSR]
        来自 build_wl_vocab_and_csr 或 build_wl_csr_with_fixed_vocabs
        第 t 个 CSR 为 [R, Vt]，表示第 t 轮 WL 的直方图特征

    输出：
      Z_raw: torch.Tensor, shape = [R, V_sum + max_nodes*k + d_attr]
        其中：
          - V_sum = Σ Vt
          - RWPE 部分维度 = max_nodes * k_steps（由 compute_graph_rwpe 的默认参数决定）
          - d_attr = 显式属性维度（onehot + meta）
    """

    # 1) WL 直方图：把每轮 CSR densify 并拼接
    chunks = [S.to_dense() for S in S_list] # S_list: List[CSR] each [R, Vt]
    Z_wl = torch.cat(chunks, dim=1)  # [R, ΣV_t]

    # 2) RWPE + explicit attrs：逐规则计算
    rwpe_list, attrs_list = [], [] 
    for r in rule_list:
        rwpe_list.append(compute_graph_rwpe(r.rhs.get_atom_graph()))
        attrs_list.append(compute_explicit_attrs(r))
    Z_rwpe = torch.stack(rwpe_list, dim=0)  # [R, max_nodes * k]
    Z_attr = torch.stack(attrs_list, dim=0)  # [R, d_attr]

    return torch.cat([Z_wl, Z_rwpe, Z_attr], dim=1)




# # -------------------------------------------------------
# # 嵌入器（用组合持有 WLCSRStore）
# # -------------------------------------------------------
# class WLCSRStore(nn.Module):
#     """
#     TODO： 感觉这个类也写的有点冗余，使得代码更复杂了，后面考虑删除
#     仅负责注册/提供 CSR；不含算子或维度设定。"""

#     def __init__(self, S_list: List[torch.Tensor]):
#         super().__init__()
#         assert len(S_list) > 0
#         self.H = len(S_list)
#         self.R = S_list[0].size(0)
#         for t, S in enumerate(S_list):
#             self.register_buffer(f"S_csr_{t}", S, persistent=False)
#         self._S_names = [f"S_csr_{t}" for t in range(self.H)]

#     def S(self, t: int) -> torch.Tensor:
#         return getattr(self, self._S_names[t])

#     @torch.no_grad()
#     def num_rules(self) -> int:
#         return self.R


# class WLHistogramEmbedder(nn.Module):
#     """
#     直接导出每条规则的 initial feature 拼接向量：
#       z_i = concat_t hist^{(t)}(rule_i) ∈ R^{ΣV_t}
#     """
#     def __init__(self, S_list: List[torch.Tensor]):
#         super().__init__()
#         self.store = WLCSRStore(S_list)
#         self.V_sizes = [self.store.S(t).size(1) for t in range(self.store.H)]
#         self.V_sum = int(sum(self.V_sizes))

#     @torch.no_grad()
#     def num_rules(self) -> int:
#         return self.store.num_rules()

#     @torch.no_grad()
#     def get_rule_embeddings(self, rule_list) -> torch.Tensor:
#         chunks = [self.store.S(t).to_dense() for t in range(self.store.H)]
#         Z_wl = torch.cat(chunks, dim=1)  # [R, ΣV_t]

#         rwpe_list, attrs_list = [], [] 
#         for r in rule_list:
#             rwpe_list.append(compute_graph_rwpe(r.rhs.get_atom_graph()))
#             attrs_list.append(compute_explicit_attrs(r))
#         Z_rwpe = torch.stack(rwpe_list, dim=0)  # [R, max_nodes * k]
#         Z_attr = torch.stack(attrs_list, dim=0)  # [R, d_attr]

#         return torch.cat([Z_wl, Z_rwpe, Z_attr], dim=1)






# def csr_index_rows(S_csr, rows):
#     """
#     TODO: 好像没有用到，后面删除
#     兼容性 CSR 行切片：优先使用原生索引；若 PyTorch 版本不支持，则回退到 COO。我现在的版本不支持，因此直接注释掉了
#     返回子矩阵（仍为 CSR 结构）。
#     """
#     # # 新版 PyTorch 直接支持 CSR 行切片, 可以用 
#     return S_csr[rows, :]

#     # # 回退：转为 COO，筛选行，再还原为 CSR
#     # device = S_csr.device
#     # S_coo = S_csr.to_sparse_coo()
#     # r, c = S_coo.indices()
#     # v = S_coo.values()

#     # # 映射：原 row 索引 -> 新 row 索引（0..N-1），其他行丢弃
#     # rows_cpu = rows.detach().cpu()
#     # row_map = {int(rv): i for i, rv in enumerate(rows_cpu.tolist())}


#     # new_r_list = [row_map.get(int(rr), -1) for rr in r.tolist()]
#     # new_r = torch.tensor(new_r_list, dtype=torch.int64, device=device)
#     # valid = new_r.ge(0)
#     # r_sel = new_r[valid]
#     # c_sel = c[valid]
#     # v_sel = v[valid]
#     # N, V = rows.numel(), S_csr.size(1)
#     # sub = torch.sparse_coo_tensor(torch.stack([r_sel, c_sel], 0), v_sel, size=(N, V), device=device).coalesce()
#     # return sub.to_sparse_csr()

# class JLRuleEmbedder(nn.Module):
#     """
#     JL（Johnson–Lindenstrauss）随机投影：
#     - 每轮固定随机投影 W_t ∈ R^{V_t × d_local}（不训练）。
#     - 轮内聚合 → learnable α 融合 → LN+Linear（可选）。
#     """
#     def __init__(self,
#         S_list: List[torch.Tensor],
#         d_local: int = 128,
#         d_out: int = 256,
#         layernorm_each_iter: bool = True,
#         learn_mix: bool = True,
#         seed: int = 42,
#     ):
#         super().__init__()
#         self.store = WLCSRStore(S_list)
#         self.d_local = d_local
#         self.ln_each = layernorm_each_iter

#         g = torch.Generator(device="cpu")
#         g.manual_seed(seed)
#         self.rp = nn.ParameterList()
#         for t in range(self.store.H):
#             Vt = self.store.S(t).size(1)
#             W = torch.randn(Vt, d_local, generator=g).div_(d_local ** 0.5)
#             self.rp.append(nn.Parameter(W, requires_grad=False))

#         self.alpha = nn.Parameter(torch.zeros(self.store.H))
#         self.mix = nn.Sequential(nn.LayerNorm(d_local), nn.Linear(d_local, d_out)) if learn_mix else nn.Identity()

#     def _agg_one(self, S_sub: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
#         h = torch.sparse.mm(S_sub, E)  # [N, V_t] @ [V_t, d] -> [N, d]
#         return F.layer_norm(h, (h.size(-1),)) if self.ln_each else h

#     def forward_with_rule_ids(self, rule_ids: torch.Tensor) -> torch.Tensor:
#         per_iter = []
#         for t in range(self.store.H):
#             S_sub = csr_index_rows(self.store.S(t), rule_ids)
#             h_t = self._agg_one(S_sub, self.rp[t])
#             per_iter.append(h_t)
#         alpha = torch.softmax(self.alpha, dim=0).view(self.store.H, 1, 1)
#         x = (alpha * torch.stack(per_iter, 0)).sum(0)  # [N, d_local]
#         return self.mix(x)  # [N, d_out]

#     @torch.no_grad()
#     def num_rules(self) -> int:
#         return self.store.num_rules()

