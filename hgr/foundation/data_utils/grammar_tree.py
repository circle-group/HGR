# FM.data_utils.grammar_tree.py

import numpy as np
import torch
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from hgr.utils.debug_utils import Timer
from collections import deque
# ==========================================
# 核心工具类：包含 拓扑构建(离线) 和 矩阵生成(在线)
# ==========================================
class GrammarTreeProcessor:
    """
    静态工具类，整合了树的构建逻辑（预处理用）和矩阵生成逻辑（DataLoader用）。
    """
    # 定义关系常量 (Relation Constants)
    REL_SELF = 0
    REL_PARENT = 1
    REL_CHILD = 2
    REL_SIBLING = 3
    REL_ANCESTOR = 4
    REL_DESCENDANT = 5
    REL_OTHER = 6

    # 建立映射字典用于打印或调试
    _REL_ID_TO_STR = {
        REL_SELF: 'Self',
        REL_PARENT: 'Parent',
        REL_CHILD: 'Child',
        REL_SIBLING: 'Sibling',
        REL_ANCESTOR: 'Ancestor',
        REL_DESCENDANT: 'Descendant',
        REL_OTHER: 'Other',
    }

    @staticmethod
    def build_grammar_tree(partitions):
        """
        [预处理阶段使用]
        输入: partitions (List[List[int]]) - 必须是 Bottom-Up DFS 顺序
        输出: parents (List[int]), depths (List[int])
        """
        num_nodes = len(partitions)
        
        # 1. 预处理：将集合转换为整数位掩码 (Bitmask) 加速包含判断
        masks = []
        for frag in partitions:
            mask = 0
            for atom_idx in frag:
                mask |= (1 << atom_idx)
            masks.append(mask)
        
        parents = np.full(num_nodes, -1, dtype=np.int32)
        depths = np.zeros(num_nodes, dtype=np.int32)
        
        # 2. 构建父子关系 (利用 Bottom-Up 特性：父节点索引 j > 子节点索引 i)
        # 复杂度优化：一旦找到第一个超集，即为直接父节点，break
        for i in range(num_nodes):
            mask_i = masks[i]
            for j in range(i + 1, num_nodes):
                # (A & B) == A 等价于 A is subset of B
                if (mask_i & masks[j]) == mask_i:
                    parents[i] = j
                    break 
        
        # 3. 计算深度 (利用拓扑顺序，从 Root 到 Leaf 反向遍历动态规划)
        # Root 的 parent 是 -1, depth 保持 0
        for i in range(num_nodes - 1, -1, -1):
            p = parents[i]
            if p != -1:
                depths[i] = depths[p] + 1
                
        return parents.tolist(), depths.tolist()


    @classmethod
    def relation(cls, relation_idx):
        """ 将 relation_idx 转换为对应的 relation_str """
        return cls._REL_ID_TO_STR.get(relation_idx, "Unknown")

    @classmethod
    def generate_matrices(cls, parents, depths):
        """
        输入:
            parents: List[int] 或 np.ndarray，长度 N，根节点 parent 为 -1
            depths:  List[int] 或 np.ndarray，长度 N，对应每个节点的树深度
        输出:
            dist_mat: (N, N) np.ndarray[int32] 最短路径距离，若不连通则为 -1
            rel_mat:  (N, N) np.ndarray[int32] 关系类型
        说明:
            1 面向树结构，构建无向邻接表
            2 对每个节点做一次 BFS 得到其到所有点的最短路径距离
            3 使用 parents 和 depths 通过广播一次性生成关系矩阵
        """
        parents = np.asarray(parents, dtype=np.int32)
        depths = np.asarray(depths, dtype=np.int32)
        N = int(parents.shape[0])

        # 1 构建无向邻接表
        #    节点 i 和 parent[i] 之间加一条无向边
        adj = [[] for _ in range(N)]
        for i, p in enumerate(parents):
            if p != -1:
                adj[i].append(p)
                adj[p].append(i)

        # 2 计算距离矩阵 dist_mat 使用树上的 BFS
        #    不连通的节点之间距离为 -1
        dist_mat = np.full((N, N), -1, dtype=np.int32)

        for src in range(N):
            dist = dist_mat[src]  # 视图, 直接写入 dist_mat[src, :]
            dist[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                du = dist[u]
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        q.append(v)

        # 3 关系矩阵 rel_mat 初始化为 Other
        rel_mat = np.full((N, N), cls.REL_OTHER, dtype=np.int32)  # 默认 Other

        # Self
        np.fill_diagonal(rel_mat, cls.REL_SELF)

        # Parent Child
        valid_mask = parents != -1
        rows = np.arange(N, dtype=np.int32)[valid_mask]  # 子节点索引
        cols = parents[valid_mask]                       # 对应父节点索引

        # i 的 parent 是 j -> rel[i, j] = Parent
        rel_mat[rows, cols] = cls.REL_PARENT
        # j 的 child 是 i -> rel[j, i] = Child
        rel_mat[cols, rows] = cls.REL_CHILD

        # Sibling
        # 同一个父节点且不是根且不是自己
        p_col = parents[:, None]   # 形状 (N, 1)
        p_row = parents[None, :]   # 形状 (1, N)
        # dist_mat != 0 排除自身, parents != -1 排除根
        is_sibling = (p_col == p_row) & (p_col != -1) & (dist_mat != 0) & (dist_mat != -1)
        rel_mat[is_sibling] = cls.REL_SIBLING

        # Ancestor Descendant
        # 在树上, 如果在同一条垂直链, 则
        # dist(i, j) == abs(depth[i] - depth[j])
        d_col = depths[:, None]    # Query i  对应行
        d_row = depths[None, :]    # Key   j  对应列
        depth_diff = d_col - d_row

        # dist_mat > 1 排除自环和直接父子
        # dist_mat != -1 排除不连通的情况
        on_same_chain = (dist_mat == np.abs(depth_diff)) & (dist_mat > 1) & (dist_mat != -1)

        # Query i 在下, Key j 在上  Key 是 i 的 Ancestor
        rel_mat[on_same_chain & (d_col > d_row)] = cls.REL_ANCESTOR
        # Query i 在上, Key j 在下  Key 是 i 的 Descendant
        rel_mat[on_same_chain & (d_col < d_row)] = cls.REL_DESCENDANT

        return dist_mat, rel_mat
    

    @staticmethod
    def build_atom_pos(partitions, offset: int = 1, fill_value: int = -1):
        """
        从 partitions 里解析 atom_pos。

        约定:
        - partitions 是 List[List[int]]，每个 frag 是一组 atom indices
        - 当 len(frag) == 1 时，该位置对应 atom-rule token
        - 解析得到 atom_pos[a] = token_position

        返回:
        - atom_pos: List[int], 长度为 num_atoms (默认 max_atom_idx+1)，未出现的填 fill_value

        参数:
        - offset: 如果你的 rule_seq 最前面会加 BOS, 那么需要设置 offset=1
        """
        if partitions is None or len(partitions) == 0:
            return []

        # 收集所有 atom index，用来决定输出长度
        all_atoms = set()
        for frag in partitions:
            for a in frag:
                all_atoms.add(int(a))
        if len(all_atoms) == 0:
            return []


        num_atoms = max(all_atoms) + 1
        atom_pos = [fill_value] * num_atoms

        # 遍历 partitions，单元素 frag 的位置就是 atom token 的位置
        for pos, frag in enumerate(partitions):
            if len(frag) != 1:
                continue
            a = int(frag[0])
            if a < 0 or a >= num_atoms:
                raise ValueError(f"atom index out of range: a={a}, num_atoms={num_atoms}")

            # 防止同一个 atom 出现多个 singleton
            if atom_pos[a] != fill_value:
                raise ValueError(f"duplicate singleton atom fragment for atom {a}: {atom_pos[a]} vs {pos + offset}")

            atom_pos[a] = pos + offset

        # 可选的强一致性检查: 每个 atom 都应当有一个 atom token
        # 如果你允许“不是每个 atom 都对应 atom-rule”，可以把下面这段删掉或改成 warning
        missing = [i for i, p in enumerate(atom_pos) if p == fill_value]
        if len(missing) > 0:
            raise ValueError(f"missing atom_pos for atoms: {missing[:10]} (total {len(missing)})")

        return atom_pos


# ==========================================
# 流程一：数据预处理脚本(测试脚本，后面删掉)
# ==========================================
def preprocess_dataset(raw_data, output_path):
    """
    执行 "方案 B": 将复杂的 fragments 转换为轻量的 parents/depths 并保存
    """
    print(f"Preprocessing {len(raw_data)} samples...")
    processed_data = []
    
    for item in tqdm(raw_data):
        # 1. 调用工具类构建拓扑
        parents, depths = GrammarTreeProcessor.build_grammar_tree(item['fragments'])
        
        # 2. 存储轻量级结果
        processed_data.append({
            "input_ids": item['rule_sequence'],
            "parents": parents,
            "depths": depths
        })
    
    # 保存为 JSONL
    with open(output_path, 'w') as f:
        for entry in processed_data:
            f.write(json.dumps(entry) + '\n')
    print(f"Saved to {output_path}")

# ==========================================
# 流程二：PyTorch Dataset 定义(测试脚本，后面删掉)
# ==========================================
class MoleculeBertDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"Processed data not found at {jsonl_path}")
            
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 获取预存储的拓扑信息
        input_ids = item['input_ids']
        parents = item['parents']
        depths = item['depths']
        
        # 2. [关键] 实时生成矩阵 (CPU微秒级)
        with Timer("Generate Matrices"):
            dist_mat, rel_mat = GrammarTreeProcessor.generate_matrices(parents, depths)
        print(dist_mat)
        print(rel_mat)
        
        # 3. 转 Tensor
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "depths": torch.tensor(depths, dtype=torch.long),       # 给 RoPE
            "distances": torch.tensor(dist_mat, dtype=torch.long),  # 给 MLP
            "relations": torch.tensor(rel_mat, dtype=torch.long)    # 给 Embedding
        }

# ==========================================
# 测试与使用示例(测试脚本，后面删掉)
# ==========================================
if __name__ == "__main__":
    # --- 1. 模拟原始数据 ---
    raw_examples = [
        {
            # 示例树结构: Leaf -> Root
            "fragments": [[1],[2],[3],[1,2,3],[4],[5],[4,5],[1,2,3,4,5],[6],[1,2,3,4,5,6]],
            "rule_sequence": [101, 102, 103, 201, 104, 105, 202, 203, 106, 301]
        },
        # 可以添加更多样本...
    ]
    
    # --- 2. 运行预处理 (只需运行一次) ---
    jsonl_file = "grammar_data_processed.jsonl"
    preprocess_dataset(raw_examples, jsonl_file)
    
    # --- 3. 模拟训练时的 DataLoader ---
    dataset = MoleculeBertDataset(jsonl_file)
    loader = DataLoader(dataset, batch_size=2, shuffle=True) # 这里需要配合 pad_collate 使用
    
    # 取出一个样本验证
    sample = dataset[0]
    print("\n--- Runtime Generated Tensors ---")
    print("Input IDs Shape:", sample['input_ids'].shape)
    print("Depths Shape:", sample['depths'].shape)
    print("Distances Shape:", sample['distances'].shape)
    print("Relations Shape:", sample['relations'].shape)
    
    # 验证逻辑正确性
    a, b = 9, 0
    distance = sample['distances'][a, b].item()
    relation = sample['relations'][a, b].item()
    print(f"Distance {a}->{b}: {distance}")
    print(f"Relation {a}<-{b}: {relation}({GrammarTreeProcessor.relation(relation)})")

    # # 0: [1] (Depth 3), 9: Root (Depth 0)
    # # Query(0) vs Key(9): 9 是 0 的 Ancestor (4)
    # rel_0_9 = sample['relations'][0, 9].item()
    # print(f"Relation 0->9 (Expected 4/Ancestor): {rel_0_9}")
    
    # # Query(0) vs Key(1): [1] 和 [2], 同父 [1,2,3], Sibling (3)
    # rel_0_1 = sample['relations'][0, 1].item()
    # print(f"Relation 0->1 (Expected 3/Sibling): {rel_0_1}")