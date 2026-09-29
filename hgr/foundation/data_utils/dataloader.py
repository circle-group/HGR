# foundation/data_utils/dataloader.py

import random
import torch
import torch.nn.functional as F
from collections import deque
from torch.nn.utils.rnn import pad_sequence
from hgr.foundation.data_utils.batch import BatchMasking
from hgr.foundation.data_utils.grammar_tree import GrammarTreeProcessor
from hgr.foundation.data_utils.mol_defs import NUM_BOND_TYPE, NUM_BOND_DIRECTION, PAD, MASK


def _select_random_conformer(data, node_num):
    """
    从 {min1pos, min2pos, min3pos} 随机选择一个构象并赋给 data.pos
    """
    idx = random.randint(1, 3)
    if idx == 1:
        pos = data.min1pos[:node_num]
    elif idx == 2:
        pos = data.min2pos[:node_num]
    else:
        pos = data.min3pos[:node_num]
    data.pos = pos
    return data




# ========== 节点/边遮蔽的基类（不处理 3D） ==========
class MaskAtom:
    """
    负责“遮蔽流水线”的统一实现：
      1) 采样若干节点作为 mask（按 mask_rate，至少 1 个）；
      2) 备份被遮蔽节点的原始特征到 data.mask_node_label；
      3) 生成节点监督标签（默认对原子类型 one-hot，可由子类覆写）；
      4) 将被遮蔽节点特征覆盖为 mask token [num_atom_type, 0]（直接写入 data.x）；
      5) 如开启 mask_edge，则对相关边做同样遮蔽，并生成边监督标签（edge_type 与 bond_direction 的 one-hot 拼接）。
    * 注意：不处理 3D 坐标，任何 3D 行为应由具体子类在其 __call__ 中显式完成，以减少阅读跳转。
    """

    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge, modalities=('graph', 'grammar')):
        self.num_atom_type = num_atom_type      # e.g., 119
        self.num_edge_type = num_edge_type      # e.g., 23
        self.mask_rate = mask_rate
        self.mask_edge = mask_edge
        self.modalities = set(modalities)
        # 保持与原始实现一致的标签空间大小
        self.num_chirality_tag = 8
        self.num_bond_direction = 7

    def __call__(self, data, masked_atom_indices=None):
        num_atoms = data.x.size(0)

        # --- 1) 采样被遮蔽节点 ---
        if masked_atom_indices is None:
            sample_size = int(num_atoms * self.mask_rate + 1)  # will sample at least 1 atom
            masked_atom_indices = random.sample(range(num_atoms), sample_size)
        masked_atom_idx = torch.as_tensor(masked_atom_indices, dtype=torch.long)

        # --- 2) 备份原始特征 ---
        data.mask_node_label = data.x[masked_atom_idx]   # (M, feat_dim) 被遮蔽节点的原始特征向量
        data.masked_atom_indices = masked_atom_idx       # (M,)

        # --- 3) 节点监督标签：仅原子类型 one-hot ---
        atom_type = data.mask_node_label[:, 0].long()
        data.node_attr_label = F.one_hot(atom_type, num_classes=self.num_atom_type).float() # 被遮蔽节点的原子类型 One-Hot 编码。

        # --- 4) 覆盖节点特征为 mask token ---
        # self._apply_node_mask_token(data, masked_atom_idx)
        # 这里将data.x覆盖为mask token，同时保留了未遮蔽节点的原始特征。
        x = data.x.clone() #.clone()目的保护原始数据不被污染，必须要有.clone()。
        token = torch.tensor([self.num_atom_type, 0], dtype=x.dtype, device=x.device)
        x[masked_atom_idx] = token
        data.x = x

        # --- 5) 覆盖 atom_pos 为 mask token ---
        if 'grammar' in self.modalities:
            data.rule_seq = data.rule_seq.clone() #.clone()目的保护原始数据不被污染
            mask_token_pos = data.atom_pos[masked_atom_idx].to(torch.long)  # (M,)
        
            # # 保护：不允许 mask BOS(0) / EOS(L-1) 位置, 运行时没有出现异常因此注释掉了这部分代码
            # valid = (mask_token_pos > 0) & (mask_token_pos < (int(data.rule_seq.numel()) - 1))
            # assert valid.all(), "mask_token_pos should be in [1, L-2]"

            data.rule_seq[mask_token_pos] = MASK
        

        # --- 6) 可选：遮蔽相关边并生成监督标签（边）---
        if self.mask_edge:
            self._apply_edge_mask(data, masked_atom_idx)

        return data


    def _apply_edge_mask(self, data, masked_atom_idx):
        """
        将与被遮蔽节点相连的边都替换为 mask token，并构造监督标签。
        PyG 的有向边成对出现；取 [::2] 作为“无向唯一边”的代表用于监督标签。
        """
        connected_edge_indices = []
        ei = data.edge_index.cpu().numpy().T  # (E, 2)
        masked_set = set(masked_atom_idx.cpu().tolist())

        for bond_idx, (u, v) in enumerate(ei):
            if (u in masked_set) or (v in masked_set):
                connected_edge_indices.append(bond_idx)

        if connected_edge_indices:
            undirected_idx = connected_edge_indices[::2]
            labels = [data.edge_attr[i].view(1, -1) for i in undirected_idx]
            data.mask_edge_label = torch.cat(labels, dim=0)  # (E_u, 2)

            # 覆盖所有相关（有向）边
            ea = data.edge_attr.clone()
            mask_token = torch.tensor([self.num_edge_type, 0], dtype=ea.dtype, device=ea.device)
            for i in connected_edge_indices:
                ea[i] = mask_token
            data.edge_attr = ea

            data.connected_edge_indices = torch.tensor(undirected_idx, dtype=torch.long)
        else:
            data.mask_edge_label = torch.empty((0, 2), dtype=torch.int64)
            data.connected_edge_indices = torch.tensor([], dtype=torch.long)

        # 监督标签：edge_type one-hot 与 bond_direction one-hot 的拼接
        if data.mask_edge_label.numel() > 0:
            edge_type = F.one_hot(data.mask_edge_label[:, 0].long(), num_classes=self.num_edge_type).float()
            bond_dir = F.one_hot(data.mask_edge_label[:, 1].long(), num_classes=self.num_bond_direction).float()
            data.edge_attr_label = torch.cat((edge_type, bond_dir), dim=1)
        else:
            data.edge_attr_label = torch.empty((0, self.num_edge_type + self.num_bond_direction)).float()



class Dist3d:
    """
    只负责选择一个 3D 构象写入 data.pos，不做节点/边遮蔽。
    """
    def __init__(self):
        self.num_atom_type = 119
        self.num_edge_type = 23
        self.num_chirality_tag = 8
        self.num_bond_direction = 7

    def __call__(self, data):
        _select_random_conformer(data, data.x.size(0))
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(num_atom_type={self.num_atom_type}, num_edge_type={self.num_edge_type})"


class MaskAtom3dDist(MaskAtom):
    """
    在 MaskAtom 的基础上再写入 3D 构象：
      - 先 super().__call__ 完成节点/边遮蔽；
      - 再随机选构象写入 data.pos。
    这样 3D 行为在 __call__ 中显式可见，阅读直观。
    """
    def __init__(self, num_atom_type, num_edge_type, mask_rate, mask_edge=False, modalities=('graph', 'grammar')):
        super().__init__(num_atom_type, num_edge_type, mask_rate, mask_edge, modalities)

    def __call__(self, data, masked_atom_indices=None):
        # 先完成遮蔽
        data = super().__call__(data, masked_atom_indices=masked_atom_indices)
        # 再显式设置 3D 构象
        _select_random_conformer(data, data.x.size(0))
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(num_atom_type={self.num_atom_type}, num_edge_type={self.num_edge_type}, mask_rate={self.mask_rate}, mask_edge={self.mask_edge})"


# ========== Graphormer DataCollation ==========

def _build_adj_list(edge_index: torch.Tensor, num_nodes: int):
    """adj[u] = [(v, eid), ...], eid 是 edge_index 的列号（图内）"""
    adj = [[] for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for eid, (u, v) in enumerate(zip(src, dst)):
        adj[u].append((v, eid))
    return adj


def shortest_paths_single(data, max_path_distance: int):
    """
    - node_paths_length: [N,N] 节点数（dist_edges + 1），最大为 D，不可达为 0
    - edge_paths_length: [N,N] 记录路径上的边数（即跳数），最大为 D-1, 不可达为 0
    - edge_paths_tensor: [N,N,D] 记录路径上经过的边 ID 序列。初始化为 -1（Padding）
    
    关键：旧版 cutoff=D 实际只探索到边距离 D-1，所以这里 max_edge_depth=D-1。
    """
    N = data.num_nodes
    D = max_path_distance
    max_edge_depth = D - 1 # node_paths_length 最大不超过 D, 所以边数最大就是 D-1

    adj = _build_adj_list(data.edge_index, N)

    node_paths_length = torch.zeros((N, N), dtype=torch.uint8)
    edge_paths_length = torch.zeros((N, N), dtype=torch.uint8)
    edge_paths_tensor = torch.full((N, N, D), -1, dtype=torch.int32) # 默认 -1：padding，无边

    for src in range(N):
        dist = [-1] * N # 记录从 src 到 v 的最短边距离(-1：还没被访问到, 0: 自己到自己， 1: 一步邻居)
        dist[src] = 0

        npl_src = node_paths_length[src]     # 视图 [N]
        epl_src = edge_paths_length[src]     # 视图 [N]
        ept_src = edge_paths_tensor[src]     # 视图 [N, D]

        npl_src[src] = 1 # 从自己到自己路径只包含一个节点

        q = deque([src])
        q_append = q.append  # 微优化：减少属性查找

        while q:
            u = q.popleft()
            du = dist[u] # src→u 的边距离
            if du >= max_edge_depth:
                continue

            ept_src_u = ept_src[u]  # 视图 [D]，src->u 的路径边序列

            for v, eid in adj[u]:
                if dist[v] != -1:
                    continue

                dv = du + 1
                dist[v] = dv

                npl_src[v] = dv + 1
                epl_src[v] = dv

                # 写入 src->v 的路径：复制 src->u 的 du 条边 + 最后一条 eid
                if du > 0:
                    ept_src[v, :du] = ept_src_u[:du]
                ept_src[v, du] = eid

                q_append(v)

    return node_paths_length, edge_paths_length, edge_paths_tensor


def precalculate_paths_padded(datas, max_path_distance: int):
    """
    输出 padded per-graph 版本（推荐用于分子图，N<=80）：
      npl: [B, Nmax, Nmax] uint8
      epl: [B, Nmax, Nmax] uint8
      ept: [B, Nmax, Nmax, D] int32（edge id，padding=-1）
    注意：这里的 edge id 是“图内 local eid”（0..E-1）。
    如果你一定要对齐到 batch.edge_attr 的全局 eid，可以额外返回 edge_offsets。
    """
    B = len(datas)
    D = max_path_distance
    Nmax = max(int(g.num_nodes) for g in datas) if B > 0 else 1

    npl = torch.zeros((B, Nmax, Nmax), dtype=torch.uint8)
    epl = torch.zeros((B, Nmax, Nmax), dtype=torch.uint8)
    ept = torch.full((B, Nmax, Nmax, D), -1, dtype=torch.int32)

    # 如果你需要“全局 eid”，用 offsets 记录每张图在 batch 中 edge 的起始偏移
    # edge_offsets = []
    edge_off = 0
    edge_ptr = [0] # Question: 请问edge_ptr是什么作用？

    for b, g in enumerate(datas):
        n = int(g.num_nodes)
        e = int(g.edge_index.size(1))
        # edge_offsets.append(edge_off)

        npl_g, epl_g, ept_g = shortest_paths_single(g, D)

        # cast + 写入
        npl[b, :n, :n] = npl_g.to(torch.uint8)
        epl[b, :n, :n] = epl_g.to(torch.uint8)
        ept[b, :n, :n, :] = ept_g.to(torch.int32)

        edge_off += e
        edge_ptr.append(edge_off)

    return npl, epl, ept, torch.tensor(edge_ptr, dtype=torch.int32)


def precalculate_grammar_padded(datas):
    """
    对 grammar 在 batch 维度做一次性预分配并写入，避免 dist_list/rel_list 的峰值内存。
    同时把 dtype 改小：
      - grammar_depths: int16
      - grammar_distances: int16
      - grammar_relations: uint8
      - rule_seq_padded: long（embedding 索引一般用 long）
    """
    B = len(datas)
    if B == 0:
        return None, None, None, None, None

    Lmax = max(int(d.rule_seq.numel()) for d in datas)

    rule_seq_padded = torch.full((B, Lmax), PAD, dtype=torch.long)
    grammar_depths = torch.zeros((B, Lmax), dtype=torch.int16)
    grammar_distances = torch.zeros((B, Lmax, Lmax), dtype=torch.int16)
    grammar_relations = torch.full((B, Lmax, Lmax), GrammarTreeProcessor.REL_OTHER, dtype=torch.uint8)

    # 全 batch 先设对角 Self (1)
    idx_all = torch.arange(Lmax)
    grammar_relations[:, idx_all, idx_all] = GrammarTreeProcessor.REL_SELF

    L_seq_list = [int(d.rule_seq.numel()) for d in datas] # 含 BOS/EOS

    for b_id, data in enumerate(datas):
        L_seq = L_seq_list[b_id]       
        N_tokens = int(data.depths.numel())
        assert L_seq == N_tokens + 2, f"rule seq length {L_seq} != depths length {N_tokens} + 2"

        # rule seq
        rule_seq_padded[b_id, :L_seq] = data.rule_seq # 已经包含 BOS/EOS了

        # depths: 对齐到 rule_seq（BOS/EOS depth=0）
        grammar_depths[b_id, 1:L_seq-1] = data.depths.to(torch.int16)

        # dist/rel: 只写入 [1:-1,1:-1]
        dist_np, rel_np = GrammarTreeProcessor.generate_matrices(data.parents, data.depths)
        grammar_distances[b_id, 1:L_seq-1, 1:L_seq-1] = torch.as_tensor(dist_np, dtype=torch.int16)
        grammar_relations[b_id, 1:L_seq-1, 1:L_seq-1] = torch.as_tensor(rel_np, dtype=torch.uint8)

        # 有效长度内的对角再置 Self（更稳）
        idx = torch.arange(L_seq)
        grammar_relations[b_id, idx, idx] = GrammarTreeProcessor.REL_SELF

    L_seq_list = torch.tensor(L_seq_list, dtype=torch.long)
    return L_seq_list, rule_seq_padded, grammar_depths, grammar_distances, grammar_relations


# ========== 通用预训练 DataLoader ==========

def _collate_batch(batches, modalities, max_path_distance=6, tf=None):
    """
    统一的 collate 函数:
      - 可选图级 transform (tf)
      - 如果包含 'grammar' 模态:
          * padding rule_seq -> rule_seq_padded (B, L_max)
          * padding depths -> grammar_depths (B, L_max)
          * parents/depths -> distances/relations -> padding 成 (B, L_max, L_max)
      - 合并为 BatchMasking
    """
    # 1. 可选 transform
    if tf is not None:
        batchs_transformed = [tf(x) for x in batches]
    else:
        batchs_transformed = list(batches)


    if 'grammar' in modalities:
        L_seq_list, rule_seq_padded, grammar_depths, grammar_dist_padded, grammar_rel_padded = \
            precalculate_grammar_padded(batchs_transformed)

        atom_pos_list = [d.atom_pos for d in batchs_transformed]
        atom_pos_padded = pad_sequence(atom_pos_list, batch_first=True, padding_value=-1)
        atom_num = torch.tensor([int(x.numel()) for x in atom_pos_list], dtype=torch.long)

        # 6. 删除单样本上的临时属性，减小 batch 体积
        for data in batchs_transformed:
            for attr in ('rule_seq', 'rule_len', 'parents', 'depths', 'atom_pos'):
                if hasattr(data, attr):
                    delattr(data, attr)

        # 3) 预计算路径长度 for Graphormer-style attn bias
        npl, epl, ept, edge_ptr = precalculate_paths_padded(batchs_transformed, max_path_distance)
        
    # 4) 执行 PyG 的合并
    batch_obj = BatchMasking.from_data_list(batchs_transformed)

    if 'grammar' in modalities:
        # 5) 挂上 grammar 相关的 batch 级别张量
        batch_obj['rule_seq_padded'] = rule_seq_padded       # (B, L_max) torch.long

        batch_obj['grammar_depths'] = grammar_depths.long()  # (B, L_max) torch.int16
        
        batch_obj["atom_num"] = atom_num                    #  (B,) torch.long
        batch_obj['seq_length'] = L_seq_list                 # (B,) torch.long
        batch_obj['atom_pos_padded'] = atom_pos_padded.long()# (B, L_max) torch.long, 已经考虑了seq first token is EOS, 因此设置了offset=1

        batch_obj['grammar_distances'] = grammar_dist_padded # (B, L_max, L_max) torch.int16
        batch_obj['grammar_relations'] = grammar_rel_padded  # (B, L_max, L_max) torch.uint8
        
        batch_obj['node_paths_length'] = npl # [B, Nmax, Nmax] uint8
        batch_obj['edge_paths_length'] = epl # [B, Nmax, Nmax] uint8
        batch_obj['edge_paths_tensor'] = ept # [B, Nmax, Nmax, D] int32
        batch_obj['edge_ptr'] = edge_ptr    # (B+1,) int32 第 b 张图在 batch 里的边区间是 [edge_ptr[b], edge_ptr[b+1])

    return batch_obj


class PretrainDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, pretexts, modalities, max_path_distance=4, batch_size=1, shuffle=True, mask_rate=0.0, mask_edge=0.0, **kwargs):
        """
        Args:
            dataset: 数据集
            pretexts: 预训练任务集合，如 {'node', 'conf', 'maccs'}
             - 根据 pretexts 选择对应 transform（MaskAtom / Dist3d / MaskAtom3dDist / None）；
            modalities: 输入模态集合，如 {'graph', 'grammar'}
            - 根据 modalities 是否包含 'grammar' 决定是否做规则序列 padding；
            
        最终，合并为 BatchMasking，并把 padding 结果附加到 batch 对象上。
        """
        self.pretexts = set(pretexts)
        self.modalities = set(modalities)
        self.max_path_distance = max_path_distance

        # 选择变换器（谁用 3D，谁明确设置构象）
        if 'node' in pretexts and 'conf' in pretexts: 
            self._tf = MaskAtom3dDist(num_atom_type=119, num_edge_type=23, mask_rate=mask_rate, mask_edge=mask_edge, modalities=self.modalities)
            print("[DataLoader] Using MaskAtom3dDist transform.")
        elif 'node' in pretexts: 
            self._tf = MaskAtom(num_atom_type=119, num_edge_type=23, mask_rate=mask_rate, mask_edge=mask_edge, modalities=self.modalities)
            print("[DataLoader] Using MaskAtom transform.")
        elif 'conf' in pretexts:
            self._tf = Dist3d()
            print("[DataLoader] Using Dist3d transform.")
        else:
            self._tf = None # pretexts 中不包含 'node'，不执行 Masking
            print("[DataLoader] No node/conf transform selected.")
            

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=self.collate_fn,
            **kwargs)

    def collate_fn(self, batches):
        return _collate_batch(batches, modalities=self.modalities, max_path_distance=self.max_path_distance, tf=self._tf)



class FinetuneDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, modalities, max_path_distance=4, batch_size=1, shuffle=True, **kwargs):
        """
        Args:
            dataset: 数据集
            modalities: 输入模态集合，如 {'graph', 'grammar'}
            - 根据 modalities 是否包含 'grammar' 决定是否做规则序列 padding；
            
        最终，合并为 BatchMasking，并把 padding 结果附加到 batch 对象上。
        """

        self.modalities = set(modalities)
        self.max_path_distance = max_path_distance

        super().__init__(dataset, batch_size, shuffle,
            collate_fn=self.collate_fn,
            **kwargs)

    def collate_fn(self, batches):
        # finetune 不做 node/conf masking，所以 tf=None
        return _collate_batch(batches, modalities=self.modalities, max_path_distance=self.max_path_distance, tf=None)


if __name__ == "__main__":
    pass
