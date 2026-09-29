# grammar.hypergraph.py

import os
import json
import hashlib
import numpy as np
from typing import List
from functools import partial
from copy import deepcopy
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
from hgr.grammar.grammar_utils import  _node_match_prod_rule, _edge_match
from networkx.algorithms import bipartite




class Hypergraph(object):
    '''
    A class of a hypergraph.
    Each hyperedge can be ordered. For the ordered case,
    edges adjacent to the hyperedge node are labeled by their orders.

    Attributes
    ----------
    hg : nx.Graph
        a bipartite graph representation of a hypergraph
    edge_idx : int
        total number of hyperedges that exist so far
    '''

    def __init__(self):
        self.hg = nx.Graph()  # 二分图表示超图
        self.edge_idx = 0
        self._nodes_ = set()
        self._edges_ = set()
        # 保存每个超边所连接的节点（列表或集合），保留顺序信息
        self.nodes_in_edge_dict = {}

    @property
    def nodes(self) -> List[str]:
        return sorted(self._nodes_, key=lambda x:int(x[5:])) # 'bond_x'

    @property
    def edges(self) -> List[str]:
        return sorted(self._edges_, key=lambda x:int(x[1:])) #'e1','e21', ...

    @property
    def num_nodes(self):
        return len(self._nodes_)

    @property
    def num_edges(self):
        return len(self._edges_)

    def __eq__(self, another):
        if self.num_nodes != another.num_nodes:
            return False
        if self.num_edges != another.num_edges:
            return False

        subhg_bond_symbol_counter = Counter([self.node_attr(each_node)['symbol'] for each_node in self.nodes])
        each_bond_symbol_counter = Counter([another.node_attr(each_node)['symbol'] for each_node in another.nodes])
        if subhg_bond_symbol_counter != each_bond_symbol_counter:
            return False

        subhg_atom_symbol_counter = Counter([self.edge_attr(each_edge)['symbol'] for each_edge in self.edges])
        each_atom_symbol_counter = Counter([another.edge_attr(each_edge)['symbol'] for each_edge in another.edges])
        if subhg_atom_symbol_counter != each_atom_symbol_counter:
            return False

        gm = GraphMatcher(self.hg,
                          another.hg,
                          partial(_node_match_prod_rule, ignore_order=True),
                          partial(_edge_match, ignore_order=True))
        try:
            # next(gm.isomorphisms_iter())
            return gm.is_isomorphic()
        except StopIteration:
            return False

    def _iso_mapping_iter_(self, another):
        assert self == another
        gm = GraphMatcher(self.hg,
                          another.hg,
                          partial(_node_match_prod_rule, ignore_order=True),
                          partial(_edge_match, ignore_order=True))
        assert gm.is_isomorphic()
        return gm.isomorphisms_iter()

    def find_isomorphism_mapping(self, another, all_mapping=True, stable=False):
        # TODO:被弃用, 但是对理解代码有辅助作用，先留着吧
        mappings = self._iso_mapping_iter_(another)

        if not stable:
            # 不要求稳定排序
            return list(mappings) if all_mapping else next(mappings)

        # 要求稳定排序
        #mappings = list(mappings)
        return sorted(mappings, key=mapping_sort_key) if all_mapping else min(mappings, key=mapping_sort_key)

    def find_priority_iso_mapping(self, another, prioritys=None):
        mappings = self._iso_mapping_iter_(another)
        return  min(mappings, key=lambda m: mapping_sort_key(m, prioritys))


    def find_amended_iso_mapping(self, another, atom_edge_maps=None):
        """
        Find an isomorphism mapping. 寻找同构映射，
        因为在reconstruct时可能会导致atom两侧的连接位点(bond)产生逆序，atom_edge_maps 就描述了这种逆序关系，用其替换来获取真实的连接位点。
        """
        mappings = self._iso_mapping_iter_(another)

        if atom_edge_maps is None:
            return min(mappings, key=mapping_sort_key)
        else:
            # 1. 利用 atom_edge_maps 替换 iso_mappings 中字典的 key
            amended_iso_mappings = [{atom_edge_maps.get(k, k): v for k, v in m.items()}
                                    for m in mappings]

            # 2. 对 amended_iso_mappings 排序，按 tuple(sorted(mapping.items())) 排序并取第一个映射
            # sorted_iso_mappings = sorted(amended_iso_mappings, key=mapping_sort_key)
            # chosen_mapping = sorted_iso_mappings[0]
            chosen_mapping = min(amended_iso_mappings, key=mapping_sort_key)

            # 3. 利用 atom_edge_maps 反向映射将得到的映射 key 还原回来
            # 构造反向映射，只有 atom_edge_maps 中的键才进行反向替换
            reverse_atom_edge_maps = {v: k for k, v in atom_edge_maps.items()}

            # 如果 key 在反向映射中，则替换回原来的 key，否则保持
            amended_mapping = {reverse_atom_edge_maps.get(k, k): v for k, v in chosen_mapping.items()}
            return amended_mapping


    def add_node(self, node: str, attr_dict=None):
        ''' add a node to hypergraph

        Parameters
        ----------
        node : str
            node name
        attr_dict : dict
            dictionary of node attributes
        '''
        attr_dict = attr_dict or {}
        attr_dict.pop("bipartite", None)  # 移除 attr_dict 中可能存在的 "bipartite" 键，避免重复赋值
        self.hg.add_node(node, bipartite='node', **attr_dict)
        self._nodes_.add(node)

    def add_edge(self, node_list: List[str], attr_dict=None, edge_name=None):
        ''' add an edge consisting of nodes `node_list`

        Parameters
        ----------
        node_list : list
            ordered list of nodes that consist the edge
        attr_dict : dict
            dictionary of edge attributes
        '''
        edge = edge_name if edge_name is not None else f"e{self.edge_idx}"
        if edge in self._edges_:
            raise ValueError(f"超边名称 {edge} 已存在！")

        attr_dict = attr_dict or {}
        attr_dict.pop("bipartite", None)  # 移除 attr_dict 中可能存在的 "bipartite" 键，避免重复赋值
        self.hg.add_node(edge, bipartite='edge', **attr_dict)
        self._edges_.add(edge)
        self.nodes_in_edge_dict[edge] = node_list
        if isinstance(node_list, list):
            for node_idx, each_node in enumerate(node_list):
                self.hg.add_edge(edge, each_node, order=node_idx)
                self._nodes_.add(each_node)
        elif isinstance(node_list, set):
            for each_node in node_list:
                self.hg.add_edge(edge, each_node, order=-1)
                self._nodes_.add(each_node)
        else:
            raise ValueError("node_list 必须为 list 或 set")
        self.edge_idx += 1
        return edge

    def remove_node(self, node: str, remove_connected_edges=True):
        ''' remove a node 删除节点，同时根据参数决定是否删除与该节点相连的超边

        Parameters
        ----------
        node : str
            node name
        remove_connected_edges : bool
            if True, remove edges that are adjacent to the node
        '''
        if remove_connected_edges:
            connected_edges = deepcopy(self.adj_edges(node))
            for each_edge in connected_edges:
                self.remove_edge(each_edge)
        self.hg.remove_node(node)
        self._nodes_.discard(node)

    def remove_nodes(self, node_iter, remove_connected_edges=True):
        ''' remove a set of nodes

        Parameters
        ----------
        node_iter : iterator of strings
            nodes to be removed
        remove_connected_edges : bool
            if True, remove edges that are adjacent to the node
        '''
        for each_node in node_iter:
            self.remove_node(each_node, remove_connected_edges)

    def remove_edge(self, edge: str):
        ''' remove an edge

        Parameters
        ----------
        edge : str
            edge to be removed
        '''
        self.hg.remove_node(edge)
        self._edges_.discard(edge)  # 使用 discard 方法删除集合中的元素，确保删除操作不会因元素不存在而引发异常；
        self.nodes_in_edge_dict.pop(edge)

    def remove_edges(self, edge_iter):
        ''' remove a set of edges

        Parameters
        ----------
        edge_iter : iterator of strings
            edges to be removed
        '''
        for each_edge in edge_iter:
            self.remove_edge(each_edge)

    def remove_edges_with_attr(self, edge_attr_dict):
        remove_edge_list = []
        for each_edge in self.edges:
            satisfy = True
            for each_key, each_val in edge_attr_dict.items():
                if not satisfy:
                    break
                try:
                    if self.edge_attr(each_edge)[each_key] != each_val:
                        satisfy = False
                except KeyError:
                    satisfy = False
            if satisfy:
                remove_edge_list.append(each_edge)
        self.remove_edges(remove_edge_list)

    # def remove_subhg(self, subhg):
    #     ''' remove subhypergraph.
    #     all of the hyperedges are removed.
    #     each node of subhg is removed if its degree becomes 0 after removing hyperedges.
    #
    #     Parameters
    #     ----------
    #     subhg : Hypergraph
    #     '''
    #     for each_edge in subhg.edges:
    #         self.remove_edge(each_edge)
    #     for each_node in subhg.nodes:
    #         if self.degree(each_node) == 0:
    #             self.remove_node(each_node)

    def nodes_in_edge(self, edge):
        ''' return an ordered list of nodes in a given edge.
        返回超边中的节点列表（有序或无序）。
        若 nodes_in_edge_dict 中存在，则直接返回，否则从 hg 中读取 order 信息。

        Parameters
        ----------
        edge : str
            edge whose nodes are returned

        Returns
        -------
        list or set
            ordered list or set of nodes that belong to the edge
        '''
        if edge in self.nodes_in_edge_dict:  # if edge.startswith('e'):
            #return self.nodes_in_edge_dict[edge]
            # TODO: 待优化
            return sorted(self.nodes_in_edge_dict[edge], key=lambda x: int(x.split('_')[1]))
        else:
            # 从图中读取该边与节点之间的连接信息
            adj_node_list = self.hg.adj[edge]
            adj_node_order_list = []
            adj_node_name_list = []
            for each_node, data in adj_node_list.items():
                adj_node_order_list.append(data['order'])
                adj_node_name_list.append(each_node)
            # 如果所有 order 都为 -1，则认为超边是无序的，返回集合
            if adj_node_order_list == [-1] * len(adj_node_order_list):
                return set(adj_node_name_list)
            else:
                # 否则，根据 order 排序返回有序列表
                return [adj_node_name_list[each_idx] for each_idx in np.argsort(adj_node_order_list)]

    def adj_edges(self, node):
        ''' return a dict of adjacent hyperedges

        Parameters
        ----------
        node : str

        Returns
        -------
        set
            set of edges that are adjacent to `node`
        '''
        return self.hg.adj[node]

    def adj_nodes(self, node):
        ''' return a set of adjacent nodes

        Parameters
        ----------
        node : str

        Returns
        -------
        set
            set of nodes that are adjacent to `node`
        '''
        node_set = set([])
        for each_adj_edge in self.adj_edges(node):
            node_set.update(set(self.nodes_in_edge(each_adj_edge)))
        node_set.discard(node)
        return node_set

    # def has_edge(self, node_list, ignore_order=False):
    #     for each_edge in self.edges:
    #         if ignore_order:
    #             if set(self.nodes_in_edge(each_edge)) == set(node_list):
    #                 return each_edge
    #         else:
    #             if self.nodes_in_edge(each_edge) == node_list:
    #                 return each_edge
    #     return False

    def degree(self, node):
        return len(self.hg.adj[node])

    def degrees(self):
        return {each_node: self.degree(each_node) for each_node in self.nodes}

    def edge_degree(self, edge):
        return len(self.nodes_in_edge(edge))

    def edge_degrees(self):
        return {each_edge: self.edge_degree(each_edge) for each_edge in self.edges}

    # def is_adj(self, node1, node2):
    #     return node1 in self.adj_nodes(node2)

    def get_minimal_graph(self, edge_list):
        """
        在给定的超边列表中，返回那些出现在 ≥2 条超边里的节点。
        """
        # 1) 扁平化将所有超边中的节点收集到一个序列里，并计数
        node_counts = Counter(
            node
            for edge in edge_list
            for node in self.nodes_in_edge(edge)
        )
        # 2) 只保留出现次数 >1 的节点
        return [node for node, cnt in node_counts.items() if cnt > 1]
        # candidate_node_list = set() # “所有可能的节点”，它们至少属于一条超边
        # for edge in edge_list:
        #     candidate_node_list.update(self.nodes_in_edge(edge))
        # selected_nodes = []
        # # 筛选「连接多条边」的节点
        # for candidate in candidate_node_list:
        #     cnt = 0
        #     for edge in edge_list:
        #         if candidate in self.nodes_in_edge(edge):
        #             cnt += 1
        #     if cnt > 1:
        #         selected_nodes.append(candidate)
        # return selected_nodes

    # def adj_subhg(self, node, ident_node_dict=None):
    #     """ return a subhypergraph consisting of a set of nodes and hyperedges adjacent to `node`.
    #     if an adjacent node has a self-loop hyperedge, it will be also added to the subhypergraph.
    #
    #     Parameters
    #     ----------
    #     node : str
    #     ident_node_dict : dict
    #         dict containing identical nodes. see `get_identical_node_dict` for more details
    #
    #     Returns
    #     -------
    #     subhg : Hypergraph
    #     """
    #     if ident_node_dict is None:
    #         ident_node_dict = self.get_identical_node_dict()
    #     adj_node_set = set(ident_node_dict[node])
    #     adj_edge_set = set([])
    #     for each_node in ident_node_dict[node]:
    #         adj_edge_set.update(set(self.adj_edges(each_node)))
    #     fixed_adj_edge_set = deepcopy(adj_edge_set)
    #     for each_edge in fixed_adj_edge_set:
    #         other_nodes = self.nodes_in_edge(each_edge)
    #         adj_node_set.update(other_nodes)
    #
    #         # if the adjacent node has self-loop edge, it will be appended to adj_edge_list.
    #         for each_node in other_nodes:
    #             for other_edge in set(self.adj_edges(each_node)) - set([each_edge]):
    #                 if len(set(self.nodes_in_edge(other_edge)) \
    #                        - set(self.nodes_in_edge(each_edge))) == 0:
    #                     adj_edge_set.update(set([other_edge]))
    #     subhg = Hypergraph()
    #     for each_node in adj_node_set:
    #         subhg.add_node(each_node, self.node_attr(each_node))
    #     for each_edge in adj_edge_set:
    #         subhg.add_edge(self.nodes_in_edge(each_edge),
    #                        self.edge_attr(each_edge),
    #                        edge_name=each_edge)
    #     subhg.edge_idx = self.edge_idx
    #     return subhg

    # def get_subhg(self, node_list, edge_list, ident_node_dict=None):
    #     """ return a subhypergraph consisting of a set of nodes and hyperedges adjacent to `node`.
    #     if an adjacent node has a self-loop hyperedge, it will be also added to the subhypergraph.
    #
    #     Parameters
    #     ----------
    #     node : str
    #     ident_node_dict : dict
    #         dict containing identical nodes. see `get_identical_node_dict` for more details
    #
    #     Returns
    #     -------
    #     subhg : Hypergraph
    #     """
    #     if ident_node_dict is None:
    #         ident_node_dict = self.get_identical_node_dict()
    #     adj_node_set = set([])
    #     for each_node in node_list:
    #         adj_node_set.update(set(ident_node_dict[each_node]))
    #     adj_edge_set = set(edge_list)
    #
    #     subhg = Hypergraph()
    #     for each_node in adj_node_set:
    #         subhg.add_node(each_node, attr_dict=deepcopy(self.node_attr(each_node)))
    #     for each_edge in adj_edge_set:
    #         subhg.add_edge(self.nodes_in_edge(each_edge), attr_dict=deepcopy(self.edge_attr(each_edge)),
    #                        edge_name=each_edge)
    #     subhg.edge_idx = self.edge_idx
    #     return subhg

    def copy(self):
        return deepcopy(self)

    def node_attr(self, node):
        return self.hg.nodes[node]  # ['attr_dict']

    def edge_attr(self, edge):
        return self.hg.nodes[edge]  # ['attr_dict']

    def set_node_attr(self, node: str, attr_dict: dict):
        self.hg.nodes[node].update(attr_dict)
        # for each_key, each_val in attr_dict.items():
        #     self.hg.nodes[node]['attr_dict'][each_key] = each_val

    def set_edge_attr(self, edge: str, attr_dict: dict):
        self.hg.nodes[edge].update(attr_dict)
        # for each_key, each_val in attr_dict.items():
        #     self.hg.nodes[edge]['attr_dict'][each_key] = each_val

    def get_all_NT_edges(self):
        NT_edges = []
        for edge in self.edges:
            edge_hg = Hypergraph()
            if not self.edge_attr(edge)['terminal']:
                node_list = list(self.nodes_in_edge(edge))
                for node in node_list:
                    edge_hg.add_node(node, deepcopy(self.node_attr(node)))
                edge_hg.add_edge(node_list, attr_dict=deepcopy(self.edge_attr(edge)), edge_name=edge)
                NT_edges.append(edge_hg)
        return NT_edges


    def to_dict(self) -> dict:
        """
        将 Hypergraph 转换为 JSON 兼容的字典格式，确保结构稳定以便计算哈希。
        """

        def graph_to_dict(G):
            return {
                "nodes": sorted(list(G.nodes(data=True))),
                "edges": sorted(list(G.edges(data=True)))
            }

        return {
            "nodes": sorted(list(self._nodes_)),
            "edges": sorted([(edge, sorted(list(self.nodes_in_edge_dict.get(edge, [])))) for edge in self._edges_]),
            "edge_idx": self.edge_idx,
            "num_nodes": len(self._nodes_),
            "num_edges": len(self._edges_),
            "nodes_in_edge_dict": {key: sorted(value) for key, value in self.nodes_in_edge_dict.items()},
            "graph_structure": graph_to_dict(self.hg)
        }

    def __hash__(self):
        """基于结构计算哈希值"""

        def json_serializer(obj):
            """ 递归转换非 JSON 兼容对象 """
            if hasattr(obj, "to_dict") and callable(obj.to_dict):
                return obj.to_dict()
            raise TypeError(f"Type {type(obj)} is not JSON serializable")

        data_str = json.dumps(self.to_dict(), sort_keys=True, default=json_serializer)  # 转换为 JSON
        return int(hashlib.sha256(data_str.encode()).hexdigest(), 16)  # 生成哈希值

    def __repr__(self):
        return f"Hg Atoms (edges): {self.edges} | Bonds (nodes): {self.nodes} "

    def get_atom_graph(self):
        """ 获取原子图(单模投影图), 用于计算rule的RWPE特征 """
        # 获取所有的原子节点（在二分图中 bipartite='edge' 的点）
        atom_nodes = [n for n, d in self.hg.nodes(data=True) if d.get('bipartite') == 'edge']
        # 生成单模投影图
        # 该图中的节点是原子，如果两个原子在原图中连接到同一个 bond 节点，则它们在投影图中相连
        atom_projection_graph = bipartite.projected_graph(self.hg, atom_nodes)
        # 如果你希望保留权重（例如两个原子之间共享了多少个键），可以使用：
        # atom_projection_graph = bipartite.weighted_projected_graph(self.rhs.hg, atom_nodes)
        return atom_projection_graph

    def draw(self, file_path=None, show=False, with_node=False, with_edge_name=True):
        """
        使用 matplotlib 和 networkx 绘制超图，功能与原 graphviz 实现完全等价。

        参数:
            file_path: 若不为 None，则将绘制结果保存为图片文件（png 格式）。
            with_node: 是否绘制超边中包含的普通节点。
            with_edge_name: 是否在超边标签中附加超边名称。

        返回:
            当前的 matplotlib Figure 对象。
        """
        # -- 1) 可在此集中管理默认绘图尺寸、节点大小等 --
        circle_size = 300
        square_size = 900
        font_size = 25
        layout_k = 0.3          # spring_layout 的理想边长
        layout_iterations = 50  # spring_layout 的迭代次数
        figsize = (5, 5)        # 图的尺寸
        margin_size = 0.1       # 边缘留白空间，防止节点挤出图外

        #  -- 2) 构造新的 MultiGraph 用于绘图
        # 创建图形（Figure）与坐标轴（Axes），确保后续全部操作在同一图上进行
        fig, ax = plt.subplots(figsize=figsize)
        G = nx.MultiGraph()

        # 若为起始规则 (lhs则空), 添加一个占位节点
        if len(self.nodes) == 0 and len(self.edges) == 0:
            G.add_node('Starting', type='edge', marker='s', color='lightgray',
                       edgecolor='black', size=square_size, label='s', filled=True)

        # 添加普通节点（来自 self.nodes）
        for node in self.nodes:
            # 如果节点属性中包含 'ext_id'，绘制为黑色小圆点；否则根据 with_node 参数决定是否添加
            if 'ext_id' in self.node_attr(node):
                G.add_node(node, type='node', marker='o', color='black', size=circle_size)
            elif with_node:
                G.add_node(node, type='node', marker='o', color='gray', size=circle_size)

        # 遍历所有超边，添加超边节点及其与普通节点或其他超边之间的连线
        edge_drawn = []  # 用于记录已添加的边，防止重复添加
        for edge in self.edges:
            # 获取超边的 symbol 属性
            try:
                symbol_label = self.edge_attr(edge)['symbol'].symbol
            except KeyError:
                symbol_label = self.hg.nodes[edge]['tmp']

            # 根据超边是否为 terminal 或 tmp，设置不同的标签和样式
            if self.edge_attr(edge).get('terminal', False):
                # terminal 边绘制为未填充的方形
                label = symbol_label if not with_edge_name else f"{symbol_label}, {edge}"
                G.add_node(edge, type='edge', marker='s', color='white',
                           edgecolor='black', size=square_size, label=label, filled=False)
            elif self.edge_attr(edge).get('tmp', False):
                label = 'tmp' if not with_edge_name else f"tmp, {edge}"
                G.add_node(edge, type='edge', marker='s', color='white',
                           edgecolor='black', size=square_size, label=label, filled=False)
            else:
                # 非 terminal 且非 tmp 边绘制为填充的方形
                label = symbol_label if not with_edge_name else f"{symbol_label}, {edge}"
                G.add_node(edge, type='edge', marker='s', color='lightgray',
                           edgecolor='black', size=square_size, label=label, filled=True)

            # 绘制超边与普通节点之间的连线
            if with_node:
                # 直接连超边与其所包含的所有节点
                for node in self.nodes_in_edge(edge):
                    G.add_edge(edge, node)
            else:
                # 仅绘制节点属性中包含 'ext_id' 的普通节点与超边之间的边
                for node in self.nodes_in_edge(edge):
                    key = set([node, edge])
                    if 'ext_id' in self.node_attr(node) and key not in edge_drawn:
                        G.add_edge(edge, node)
                        edge_drawn.append(key)

                # 绘制超边间的连线：统计两个超边间共有的节点，根据节点中 symbol.bond_type 累加“键数”
                for other_edge in self.adj_nodes(edge):
                    key = set([edge, other_edge])
                    if key not in edge_drawn:
                        # 计算两个超边间共享的普通节点集合
                        common_node_set = set(self.nodes_in_edge(edge)).intersection(
                            set(self.nodes_in_edge(other_edge)))
                        num_bond = 0
                        for node in common_node_set:
                            bond_type = self.node_attr(node)['symbol'].bond_type
                            if bond_type in [1, 2, 3]:
                                num_bond += bond_type
                            elif bond_type in [12]:
                                num_bond += 1
                            else:
                                raise NotImplementedError('unsupported bond type')
                        # 对于每个键画一条边
                        for _ in range(num_bond):
                            G.add_edge(edge, other_edge)
                        edge_drawn.append(key)

        pos = nx.spring_layout(G, k=layout_k, iterations=layout_iterations)  # k 越小图越紧凑


        #  -- 4) 绘制普通节点
        node_list = [n for n in G.nodes() if G.nodes[n].get('type') == 'node']
        marker_groups = {}  # 根据 marker 对节点分组绘制（nx.draw_networkx_nodes 不支持一次性设置不同 marker）
        for n in node_list:
            marker = G.nodes[n].get('marker', 'o')
            marker_groups.setdefault(marker, []).append(n)
        for marker, nodelist in marker_groups.items():
            colors = [G.nodes[n].get('color', 'black') for n in nodelist]
            sizes = [G.nodes[n].get('size', 100) for n in nodelist]
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colors,
                                   node_size=sizes, node_shape=marker)

        #  -- 5) 绘制超边节点
        edge_node_list = [n for n in G.nodes() if G.nodes[n].get('type') == 'edge']
        marker_groups_edge = {}
        for n in edge_node_list:
            marker = G.nodes[n].get('marker', 's')
            marker_groups_edge.setdefault(marker, []).append(n)
        for marker, nodelist in marker_groups_edge.items():
            colors = [G.nodes[n].get('color', 'white') for n in nodelist]
            sizes = [G.nodes[n].get('size', 300) for n in nodelist]
            edgecolors = [G.nodes[n].get('edgecolor', 'black') for n in nodelist]
            nx.draw_networkx_nodes(G, pos, nodelist=nodelist, node_color=colors,
                                   node_size=sizes, node_shape=marker, edgecolors=edgecolors)

        # 绘制超边节点的标签（仅显示超边）
        labels = {n: G.nodes[n].get('label', '') for n in edge_node_list}
        nx.draw_networkx_labels(G, pos, labels, font_color='black', font_size=font_size)

        # -- 6) 绘制边（多重边使用弧形） --
        # 先按无序节点对分组
        edge_dict = {}
        for u, v, key in G.edges(keys=True):
            pair = tuple(sorted((u, v)))
            edge_dict.setdefault(pair, []).append((u, v, key))
        # 对每一对节点间的所有边绘制时分别设置不同弧度
        for pair, edges in edge_dict.items():
            n_multi = len(edges)
            for i, (u, v, _) in enumerate(edges):
                # # 如果只有一条边，rad=0.0；多条边则做弧形偏移
                rad = 0.0 if n_multi == 1 else 0.1 * (i - (n_multi - 1) / 2)
                con_style = f"arc3,rad={rad}"
                arrow = patches.FancyArrowPatch(pos[u], pos[v],
                                                connectionstyle=con_style,
                                                arrowstyle='-', color='black')
                ax.add_patch(arrow)

        ax.axis('off')
        ax.margins(margin_size)  # 给一点边缘留白，防止节点挤出图外
        fig.tight_layout()

        # 5) 保存或显示图形
        if file_path is not None:
            fig.savefig(file_path, bbox_inches='tight')
            print(f"Saved to {os.path.abspath(file_path)}")
            if not show:
                plt.close(fig)
        if show or (file_path is None):
            plt.show()

        return fig

    def draw_with_bond(self, node_size=300, font_size=18, figsize=(6, 6), file_path=None, show=True):
        G = self.hg

        # 1) 构造临时画图用的小图 D（同上）
        ex_nodes = [n for n in G.nodes() if n.startswith('e')]
        ex_labels = {e: self.node_attr(e)['symbol'].symbol + e[1:] for e in ex_nodes}

        double_edges = set()
        edge_labels = {}
        half_bonds = {}
        bond_type_map = {} # 记录每条 D 边对应的 bond_type

        for b in G.nodes():
            if not b.startswith("bond_"):
                continue
            bid = b.split("_", 1)[1]
            symbol = self.node_attr(b)['symbol']
            bt = symbol.bond_type  # 1,2,3,12
            nbrs = list(G.neighbors(b))
            if len(nbrs) == 2:
                u, v = nbrs
                pair = tuple(sorted((u, v)))
                double_edges.add(pair)
                edge_labels[pair] = f"b{bid}"
                bond_type_map[pair] = bt
            elif len(nbrs) == 1:
                half_bonds[f"b{bid}"] = nbrs[0]
                bond_type_map[(f"b{bid}", nbrs[0])] = bt

        D = nx.Graph()
        D.add_nodes_from(ex_nodes)
        D.add_edges_from(double_edges)
        for b, u in half_bonds.items():
            D.add_node(b)
            D.add_edge(u, b)

        # 2) 对 D 做 spring 布局：增大 k, iterations, scale
        pos = nx.spring_layout(
            D,
            k=0.7,  # 弹簧“理想长度”系数，越大节点越分散
            iterations=200,
            scale=2.5  # 整体缩放
        )

        # 3) 画图
        fig, ax = plt.subplots(figsize=figsize)

        # 3.1 e* 节点
        nx.draw_networkx_nodes(
            D, pos,
            nodelist=ex_nodes,
            node_color='skyblue',
            node_shape='o',
            node_size=node_size,
            ax=ax
        )
        # 3.2 半键节点（小方块）
        half_nodes = list(half_bonds.keys())
        nx.draw_networkx_nodes(
            D, pos,
            nodelist=half_nodes,
            node_color='lightgreen',
            node_shape='s',
            node_size=node_size,
            ax=ax
        )

        # 4) 按 bond_type 分组绘制边
        # 定义颜色和线型映射
        color_map = {
            1: 'black',  # 单键 - 黑
            2: 'blue',  # 双键 - 蓝
            3: 'green',  # 三键 - 绿
            12: 'red'  # 芳香键 - 红
        }
        style_map = {
            1: 'solid',
            2: 'solid',
            3: 'solid',
            12: 'dashed'
        }

        # 收集不同类型边
        edges_by_type = {}
        for edge in D.edges():
            t = bond_type_map.get(tuple(sorted(edge)), None)
            if t is not None:
                edges_by_type.setdefault(t, []).append(edge)

        # 分类型绘制
        for bt, eds in edges_by_type.items():
            nx.draw_networkx_edges(
                D, pos,
                edgelist=eds,
                style=style_map[bt],
                edge_color=color_map[bt],
                width=2,
                ax=ax
            )

        # 3.3 边
        # nx.draw_networkx_edges(D, pos, ax=ax)

        # 3.4 标签
        nx.draw_networkx_labels(
            D, pos,
            labels=ex_labels,
            font_size=font_size,
            ax=ax
        )
        nx.draw_networkx_labels(
            D, pos,
            labels={b: b for b in half_nodes},
            font_size=font_size-5,
            ax=ax
        )
        nx.draw_networkx_edge_labels(
            D, pos,
            edge_labels=edge_labels,
            font_size=font_size-5,
            ax=ax
        )

        ax.axis('off')
        plt.tight_layout()

        # 5) 保存或显示图形
        if file_path is not None:
            fig.savefig(file_path, bbox_inches='tight')
            print(f"Saved to {os.path.abspath(file_path)}")
            if not show:
                plt.close(fig)
        if show or (file_path is None):
            plt.show()

        return fig





def mapping_sort_key(mapping, priority_keys=None):
    """
    为一个同构映射字典生成排序键：
    1) 只对每对 (key,value) 解析一次数值化三元组
    2) 首先对priority_keys 进行排序，然后对'bond_x'排序，最后对‘ex'进行排序
    3) 一次性排序后直接返回三元组序列，省掉双重调用
    """
    pk = set(priority_keys) if priority_keys else set()

    decorated = []
    for k, v in mapping.items():
        if k.startswith('bond_'):
            priority = 0 if k in pk else 1  # 在优先列表里则 priority=0，否则 priority=1
            decorated.append((priority, int(k[5:]), int(v[5:])))
        elif k.startswith('e'):
            decorated.append((2, int(k[1:]), int(v[1:])))
        else:
            raise ValueError(f"Unsupported key format: {k!r}")

    decorated.sort()
    return tuple(decorated)