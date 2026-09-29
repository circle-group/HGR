# grammar/mol_graph.py

from rdkit import Chem
from hgr.grammar.chemutils import extract_subgraph
from .smi import mol_to_hg
from functools import cached_property 
import networkx as nx
import matplotlib.pyplot as plt
from functools import cached_property
from collections import deque

# -------------------------
# 工具函数：集合交/并操作
# -------------------------
def _intersection_(list1, list2):
    return list(set(list1) & set(list2))
def _union_(list1, list2):
    return list(set(list1) | set(list2))


class MolGraph():
    def __init__(self, mol, is_subgraph=False, mapping_to_input_mol=None):
        if is_subgraph:
            assert mapping_to_input_mol is not None, "Subgraph requires mapping_to_input_mol"
        self.mol = mol
        self.is_subgraph = is_subgraph
        #self.hypergraph = mol_to_hg(mol, kekulize=True, add_Hs=False)
        self.hypergraph = mol_to_hg(mol, kekulize=False, add_Hs=False)
        self.mapping_to_input_mol = mapping_to_input_mol

    @cached_property
    def _smiles(self):
        return Chem.MolToSmiles(self.mol)

    def get_visit_status_edge(self, edge):
        return self.hypergraph.edge_attr(edge)['visited']

    def get_org_idx_in_input(self, idx):
        assert self.is_subgraph
        return self.mapping_to_input_mol.GetAtomWithIdx(idx).GetIntProp('org_idx')

    def get_org_node_in_input(self, node, subgraph):
        # 返回node（bond) 对应的原图中的bond
        assert not self.is_subgraph
        adj_edges = list(subgraph.hypergraph.adj_edges(node))
        assert len(adj_edges) == 2 #输入的node对应bond, adj_edges为其对应的2个atom
        org_edges = []
        for adj_edge_i in adj_edges:
            new_idx = int(adj_edge_i[1:])
            org_idx = subgraph.get_org_idx_in_input(new_idx)
            org_edges.append(f'e{org_idx}')
        #org_node = list(self.hypergraph.nodes_in_edge(org_edges[0]).intersection(self.hypergraph.nodes_in_edge(org_edges[1])))
        org_node = list(set(self.hypergraph.nodes_in_edge(org_edges[0])) &
                        set(self.hypergraph.nodes_in_edge(org_edges[1])) )
        try:
            assert len(org_node) == 1, "Ambiguous mapping"
        except:
            import pdb; pdb.set_trace()
        return org_node[0]


    def __eq__(self, another):
        # return hasattr(another, 'mol') and Chem.CanonSmiles(Chem.MolToSmiles(self.mol)) == Chem.CanonSmiles(Chem.MolToSmiles(another.mol))
        # return hasattr(another, 'mol') and (self._smiles) == (Chem.MolToSmiles(another.mol))
        return self._smiles == another._smiles

    def __hash__(self):
        return hash(self._smiles)

    def draw(self, node_size=300, font_size=18, figsize=(6, 6)):
        self.hypergraph.draw_with_bond(node_size, font_size, figsize)

    def __getstate__(self):
        """rdkit.Chem.Mol 对象默认序列化时会丢弃自定义字段"""
        # 1) 自动抓所有普通属性
        state = self.__dict__.copy()
        # 2) 单独抽取 org_idx 信息，放到一个新字段里
        mapping_intprops = {}
        if self.mapping_to_input_mol is not None:
            for atom in self.mapping_to_input_mol.GetAtoms():
                if atom.HasProp('org_idx'):
                    mapping_intprops[atom.GetIdx()] = atom.GetIntProp('org_idx')
        state['_mapping_intprops'] = mapping_intprops

        return state

    def __setstate__(self, state):
        # 1) 先取出要特殊处理的数据
        mapping_intprops = state.pop('_mapping_intprops', {})

        # 2) 把基础属性都恢复
        self.__dict__.update(state)

        # 3) 再把 org_idx 再写回 Atom
        for atom in (self.mapping_to_input_mol or self.mol).GetAtoms():
            idx = atom.GetIdx()
            if idx in mapping_intprops:
                atom.SetIntProp('org_idx', mapping_intprops[idx])



class SubGraph(MolGraph):
    def __init__(self, mol, mapping_to_input_mol, subfrags):
        super(SubGraph, self).__init__(mol, is_subgraph=True, mapping_to_input_mol=mapping_to_input_mol)
        assert type(subfrags) == list
        '''
        subfrags: list, atom indices of two sub fragments
        bond: bond index of the connected bond
        '''
        self.subfrags = subfrags
        self.weight = len(subfrags)
        self.children = []
        self.parent = None
        self.rule = None
        self.rule_idx = None

    def transform_children_weight(self, transform_map):
        """
        对给定节点的直接子节点按照当前weight（降序排列）进行权重变换。

        参数:
          node: 要进行权重变换的树节点（SubGraph对象），其子节点保存在node.children中
          transform_map: 字典，表示权重变换关系，例如 {0: 1, 1: 2, 2: 0}
                         意味着：排序后排名0的子节点的weight更新为原排名1的weight，
                               排名1的更新为原排名2的weight，
                               排名2的更新为原排名0的weight。
        """
        # 对当前节点的子节点按照当前weight降序排序
        sorted_children = sorted(self.children, key=lambda child: child.weight, reverse=True)
        # 保存原始的weight序列，避免在更新过程中相互干扰
        original_weights = [child.weight for child in sorted_children]
        # 创建新权重列表（初始为原始值）
        new_weights = original_weights.copy()

        # 根据变换映射修改对应位置的权重
        for rank, _ in enumerate(sorted_children):
            if rank in transform_map:
                source_rank = transform_map[rank]
                if source_rank < len(original_weights):
                    new_weights[rank] = original_weights[source_rank]

        # 更新排序后各子节点的weight
        for idx, child in enumerate(sorted_children):
            child.weight = new_weights[idx]

    def reverse_children(self):
        assert self.symmetry_map is not False
        # 对于True的情况表明不需要调整权重
        # if isinstance(self.symmetry_map, dict):
        #     self.transform_children_weight(self.symmetry_map)
        #     for c in self.children:
        #         c.reverse_children()

        if isinstance(self.symmetry_map, dict):
            sorted_children = sorted(self.children, key=lambda child: child.weight, reverse=True)
            # 只有部分的children需要调整
            for cid, c in enumerate(sorted_children):
                if cid in self.rule.flipped_children_idx:
                    c.reverse_children()
            self.transform_children_weight(self.symmetry_map)


    @cached_property
    def symmetry_map(self):
        assert self.rule is not None

        # 遍历所有直接子节点，利用子节点自身的 symmetry_map 属性实现递归检查
        for child in self.children:
            if child.symmetry_map is False:
                return False

        return self.rule.symmetry_map

    def __str__(self):
        return f"SubGraph {self.subfrags}, rule_idx: {self.rule_idx if self.rule_idx is not None else 'N/A'}, {len(self.children)} children:{self.children} "

    def __repr__(self):
        return self.__str__()

    def visualize_tree(self, show=True, save_path=None):

        # 1) 先把 SubGraph 树转成 NetworkX 有向图
        def build_graph(node, G, parent=None):
            nid = id(node)
            # label 显示子片段和 weight、rule_idx
            lbl = f"{node.subfrags}\nw:{node.weight}"
            if hasattr(node, 'rule_idx'):
                lbl += f"\nR:{node.rule_idx}"
            G.add_node(nid, label=lbl)
            if parent is not None:
                G.add_edge(parent, nid)
            for c in node.children:
                build_graph(c, G, nid)
            return G

        G = build_graph(self, nx.DiGraph())

        # 2) 计算每个节点的“叶子数”（leaf count）
        def count_leaves(nid):
            ch = list(G.successors(nid))
            if not ch:
                return 1
            return sum(count_leaves(c) for c in ch)

        leaf_counts = {n: count_leaves(n) for n in G.nodes()}
        root = id(self)
        total_leaves = leaf_counts[root]

        # 3) 树的深度，用来计算垂直间距
        def tree_depth(nid):
            ch = list(G.successors(nid))
            if not ch:
                return 1
            return 1 + max(tree_depth(c) for c in ch)

        depth = tree_depth(root)
        vert_gap = 1.0 / (depth + 1)
        # 从 y=1 开始，向下递减
        top_y = 1.0

        # 4) 递归分配坐标：每个节点占据自己叶子数/父节点叶子数 * 可用宽度
        pos = {}

        def recurse(nid, left, right, y):
            # x 在区间中点
            x = (left + right) / 2
            pos[nid] = (x, y)
            children = list(G.successors(nid))
            if not children:
                return
            # 按子节点比例分区
            cur = left
            for c in children:
                share = (leaf_counts[c] / leaf_counts[nid]) * (right - left)
                recurse(c, cur, cur + share, y - vert_gap)
                cur += share

        # 整个图横向宽度就设为叶子总数
        recurse(root, 0, total_leaves, top_y)

        # 5) 画图
        labels = nx.get_node_attributes(G, 'label')
        plt.figure(figsize=(max(8, total_leaves * 0.6), max(6, depth * 1.2)))
        nx.draw(G, pos, labels=labels,
                node_size=2000, node_color='lightblue',
                font_size=12, font_family='monospace',
                arrows=False)
        plt.margins(0.05, 0.05)
        plt.axis('off')
        if save_path is not None:
            plt.savefig(save_path)
        if show:
            plt.show()



class InputGraph(MolGraph):
    def __init__(self, mol, smiles, init_subgraphs, subgraphs_idx):
        '''
        init_subgraphs: a list of MolGraph
        subgraph_idx: a list of atom idx list for each subgraph
        '''
        super(InputGraph, self).__init__(mol)
        # self.subgraphs = init_subgraphs # SubGraph list
        # self.subgraphs_idx = subgraphs_idx

        # [!! 优化 !!] 使用 deque 实现 O(1) 的 pop(0)
        self.subgraphs = deque(init_subgraphs) # SubGraph list
        self.subgraphs_idx = deque(subgraphs_idx)
        self.subgraph_map = {frozenset(idx_list): subg 
            for idx_list, subg in zip(self.subgraphs_idx, self.subgraphs)}

        self.smiles = smiles
        # self.map_to_set = self.get_map_to_set()
        self.NT_atoms = set()
        self.rule_list = []
        self.rule_idx_list = []
        self.water_level = 0 # 标记“已经构造了多少轮子图收缩”
        self.watershed_ext_nodes = {} # 映射每个 water_level → 当时对应的“外部节点列表”


    def _refresh_subgraph_status(self, subgraph):
        # 更新子图中每个原子的访问状态和 NT 标记
        for idx in range(subgraph.mol.GetNumAtoms()):
            org_idx = subgraph.get_org_idx_in_input(idx)
            org_visit_state = self.hypergraph.edge_attr(f'e{org_idx}')['visited']
            #subgraph.set_visit_status_with_idx(idx, org_visit_state)
            subgraph.hypergraph.edge_attr(f'e{idx}')['visited'] = org_visit_state
            if org_idx in self.NT_atoms:
                #subgraph.set_NT_status_with_idx(idx, True)
                subgraph.hypergraph.edge_attr(f'e{idx}')['NT'] = True
        # 更新子图中节点的访问状态
        for node in subgraph.hypergraph.nodes:
            org_node = self.get_org_node_in_input(node, subgraph)
            subgraph.hypergraph.node_attr(node)['visited'] = self.hypergraph.node_attr(org_node)['visited']


    def update_visit_status(self, visited_list):
        edge_list = [f'e{i}' for i in visited_list]
        for edge in edge_list:
            self.hypergraph.edge_attr(edge)['visited'] = True

        node_list = self.hypergraph.get_minimal_graph(edge_list)
        for node in node_list:
            self.hypergraph.node_attr(node)['visited'] = True

    def update_NT_atoms(self):
        # p_subg_mapped = extract_subgraph(self.smiles, p_star_idx)
        p_subg_mapped = self.subgraphs[0].mapping_to_input_mol # 每次合并的都是第一个子图
        for idx, atom in enumerate(p_subg_mapped.GetAtoms()):
            org_idx = p_subg_mapped.GetAtomWithIdx(idx).GetIntProp('org_idx')
            if atom.GetAtomMapNum() == 1:
                self.NT_atoms.add(org_idx)
            elif org_idx in self.NT_atoms:
                self.NT_atoms.remove(org_idx)

    def update_watershed(self, p_star_idx):
        # p_star_idx 是一组要收缩的边的原始 idx 列表，比如 [3,5,7] → ['e3','e5','e7']
        edge_list = [f'e{i}' for i in p_star_idx]
        # 找出这些边所组成的“最小连通子图”所包含的节点
        node_set = set(self.hypergraph.get_minimal_graph(edge_list))

        # 1) 给每条要收缩的边打上相同的 water_level
        ext_node_list = []
        for edge in edge_list:
            self.hypergraph.edge_attr(edge)['water_level'] = self.water_level
            # 2) 同时遍历该边所连的所有节点
            for _node in self.hypergraph.nodes_in_edge(edge):
                # 如果这个节点 不在 node_list（即属于外部节点），就收集起来
                if _node not in node_set:
                    ext_node_list.append(_node)

        # 3) 给子图内部节点也打同样的 water_level
        for node in node_set:
            self.hypergraph.node_attr(node)['water_level'] = self.water_level

        # 4) 把这一轮的 “外部节点列表” 存到 watershed_ext_nodes
        assert self.water_level not in self.watershed_ext_nodes.keys()
        self.watershed_ext_nodes[self.water_level] = ext_node_list

        # 5) 轮次计数器 +1，准备下一次
        self.water_level += 1

    def update_subgraph_v1_toDelete(self, motif_to_process):
        subg_idx = motif_to_process.subfrags
        parent_motif = motif_to_process.parent
        
        # Update visit_status and NT_atoms and watershed 更新访问状态、NT_atoms 以及 watershed 信息
        self.update_visit_status(subg_idx)
        self.update_NT_atoms()
        self.update_watershed(subg_idx)

        new_subgraphs = []
        new_subgraphs_idx = []
        for i, subg in enumerate(self.subgraphs):
            current_subg_idx = self.subgraphs_idx[i]
            if current_subg_idx == subg_idx:
                continue

            # #overlap, assemble = self.find_overlap(subg_idx, current_subg_idx)
            # if len(_intersection_(subg_idx, current_subg_idx)) >= 1:
            #     new_subgraph_idx = _union_(subg_idx, current_subg_idx)
            #     new_subgraph_mapped = extract_subgraph(self.smiles, new_subgraph_idx)
            #     subfrags = deepcopy(new_subgraph_idx)
            #     new_subgraph = SubGraph(new_subgraph_mapped, mapping_to_input_mol=new_subgraph_mapped, subfrags=subfrags)
            #     self._refresh_subgraph_status(new_subgraph)
            #     new_subgraphs.append(new_subgraph)
            #     new_subgraphs_idx.append(new_subgraph_idx)
            # else:
            #     new_subgraphs.append(subg)
            #     new_subgraphs_idx.append(current_subg_idx)

            if parent_motif is not None and current_subg_idx == parent_motif.subfrags:
                self._refresh_subgraph_status(subg)
                new_subgraphs.append(subg)
                new_subgraphs_idx.append(current_subg_idx)
            else:
                new_subgraphs.append(subg)
                new_subgraphs_idx.append(current_subg_idx)

        self.subgraphs = new_subgraphs
        self.subgraphs_idx = new_subgraphs_idx
        #self.map_to_set = self.get_map_to_set()
    

    def update_subgraph(self, motif_to_process: SubGraph):
        # 1. 获取信息
        subg_idx = motif_to_process.subfrags
        # parent_motif = motif_to_process.parent

        # 2. Update visit_status and NT_atoms and watershed 
        self.update_visit_status(subg_idx)
        self.update_NT_atoms()
        self.update_watershed(subg_idx)


        # 3. 移除子节点 (O(1))
        self.subgraphs.popleft()
        # assert popped_motif == motif_to_process, "Deque order mismatch" # (断言确保我们弹出的是正确的)
        try:
            del self.subgraph_map[frozenset(subg_idx)] # (使用 frozenset(subg_idx) 因为 subg_idx 是 list)
        except KeyError:
            # 理论上不应发生，但如果发生，记录日志
            pass
            # logger.warning(f"Motif {subg_idx} not found in subgraph_map during del.")


        # # 4. 查找并刷新父节点 (O(1))
        # if parent_motif:
        #     parent_frags_fs = frozenset(parent_motif.subfrags)
        #     parent_to_refresh = self.subgraph_map.get(parent_frags_fs)

        #     if parent_to_refresh:            
        #         self._refresh_subgraph_status(parent_to_refresh) # 找到了父节点，刷新它的 hypergraph 状态
        #     else:
        #         # 父节点在map中不存在，意味着它已经被处理/弹出了
        #         # (这在树结构中不应该发生，但即使发生也不是错误)
        #         pass




def build_tree_with_dfs_sorting(mol, clusters):
    # 注意subgraphs和smiles顺序需要一致

    # Step 1：对 clusters 按大小排序，并转换为 frozenset 便于集合比较
    clusters_sorted = sorted(clusters, key=len)  # 原始 clusters 按大小排序
    assert len(clusters_sorted[-1])== mol.GetNumAtoms(), "The largest cluster should match the original molecule size."
    clusters_frozenset = [frozenset(s) for s in clusters_sorted]  # 转为 frozenset 后用于集合包含关系判断
    assert len(clusters_frozenset) == len(set(clusters_frozenset)), "Clusters should be unique."
    # return None, None # 7000it/s (210k/30s)

    # Step 2：构建映射：key 为标准化的 cluster (frozenset)，value 为 SubGraph 对象
    node_dict = {}
    for cid, each_cluster in enumerate(clusters_sorted):
        # 提取cluster对应的子图
        subg_mapped = extract_subgraph(mol, each_cluster)
        # 注释掉下面两行，为3000it/s (100k/30s)
        tree_node = SubGraph(subg_mapped, mapping_to_input_mol=subg_mapped, subfrags=each_cluster)
        # 注释掉下面一行为2500it/s (74k/30s)
        node_dict[clusters_frozenset[cid]] = tree_node
    
    # return None, None # 2500it/s (64k/30s)

    # Step 3：建立父子关系：每个 cluster 寻找其最小的真超集作为父节点
    for i, cluster in enumerate(clusters_frozenset):
        for other_cluster in clusters_frozenset[i+1:]:
            if cluster < other_cluster: # true包含关系
                node_dict[cluster].parent = node_dict[other_cluster]
                node_dict[other_cluster].children.append(node_dict[cluster])
                break
    
    # return None, None # 2000it/s (58k/30s)

    # Step 4: DFS 遍历并收集集合索引
    counter = [0]  # 用列表模拟可变计数器
    dfs_order = []
    def dfs(node):
        node.weight = counter[0]
        counter[0] += 1
        dfs_order.append(node)
        for child in sorted(node.children, key=lambda c: len(c.subfrags), reverse=True):
            #优先合并大大。reverse=True：表示降序排列（默认是升序），也就是大的在前，小的在后。
            dfs(child)

    # 从最大的 cluster 开始 DFS（clusters_frozenset 已按升序排列，最后一个为最大集合，通常为树根）
    dfs(node_dict[clusters_frozenset[-1]])

    # Step 5：根据 DFS 遍历顺序构造排序后的结果
    # 这边逆序是为了merge时的需要
    subgraphs = dfs_order[::-1]
    clusters = [sg.subfrags for sg in subgraphs]


    return  clusters, subgraphs

