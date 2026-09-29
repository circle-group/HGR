# grammar.rule.py

import logging
import networkx as nx
from functools import partial
from collections import Counter
from typing import Dict
from networkx.algorithms.isomorphism import GraphMatcher

from hgr.grammar.hypergraph import Hypergraph
from hgr.grammar.symbol import TSymbol, NTSymbol, BondSymbol
from hgr.grammar.grammar_utils import _node_match_prod_rule, _edge_match #, has_inversion_bond

logger = logging.getLogger(__name__)


def find_min_mapping(iso_mapping, bID2ruleID):


    # 1️⃣ 替换后的映射列表
    iso_mapping = [{k: v for k, v in map.items() if k.startswith("bond_")} for map in iso_mapping]
    replaced_list = [{int(bID2ruleID[k][5:]): int(v[5:]) for k, v in d.items() } for d in iso_mapping]

    #  2️⃣ 为每个 replaced 生成比较元组 (按数字键升序)
    cmp_tuples = [
        tuple(repl[k] for k in sorted(repl))
        for repl in replaced_list
    ]

    # 3️⃣ 找最小
    min_idx = cmp_tuples.index(min(cmp_tuples))
    return  iso_mapping[min_idx]

class ProductionRule(object):
    """ A class of a production rule

    Attributes
    ----------
    lhs : Hypergraph or None
        the left hand side of the production rule.
        if None, the rule is a starting rule.
    rhs : Hypergraph
        the right hand side of the production rule.
    """

    def __init__(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs
        self.used_count = 1

    @property
    def is_start_rule(self) -> bool:
        # 判断当前产生式是否为起始规则。如果左侧超图没有节点，则说明这个规则是用于超图的初始化。
        return self.lhs.num_nodes == 0


    @property
    def is_ending(self) -> bool:
        return len(self.rhs.get_all_NT_edges()) == 0

    @property
    def ext_node(self) -> Dict[int, str]:
        """ return a dict of external nodes
        返回一个字典，记录了产生式规则中左侧超图的外部节点。
        每个外部节点在属性中通常会有一个 ext_id 标记，表示其在规则中相对外部环境的位置。
        在应用产生式规则时，需要将右侧超图中的相应外部节点与原超图中被替换超边的节点进行对应，从而保证整体连接关系正确。
        """
        if self.is_start_rule:
            return {}
        else:
            return {self.lhs.node_attr(node)["ext_id"]: node for node in self.lhs.nodes}

    @property
    def lhs_nt_symbol(self) -> NTSymbol:
        if self.is_start_rule:
            return NTSymbol(degree=0, is_aromatic=False, bond_symbol_list=[])
        else:
            # 对于普通规则，则从左侧超图中的第一个超边获取其属性中的 symbol 字段。
            #return [self.lhs.edge_attr(edge)['symbol'] for edge in self.lhs.edges]
            return self.lhs.edge_attr(self.lhs.edges[0])['symbol']

    def rhs_adj_mat(self, node_edge_list):
        ''' return the adjacency matrix of rhs of the production rule '''
        return nx.adjacency_matrix(self.rhs.hg, node_edge_list)

    def draw(self, file_path=None):
        return self.rhs.draw(file_path)

    def draw_with_bond(self, node_size=300, font_size=18, figsize=(6, 6)):
        self.rhs.draw_with_bond(node_size, font_size, figsize)

    def _check_iso(self, prod_rule, ignore_order=False):
        gm = GraphMatcher(prod_rule.rhs.hg,
                          self.rhs.hg,
                          partial(_node_match_prod_rule, ignore_order=ignore_order),
                          partial(_edge_match, ignore_order=ignore_order))
        try:
            return True, next(gm.isomorphisms_iter())
        except StopIteration:
            return False, {}

    def is_same(self, prod_rule, ignore_order=False):
        """ judge whether this production rule is the same as the input one, `prod_rule`

        Parameters
        ----------
        prod_rule : ProductionRule
            production rule to be compared

        Returns
        -------
        is_same : bool
        isomap : dict
            isomorphism of nodes and hyperedges.
            ex) {'bond_42': 'bond_37', 'bond_2': 'bond_1',
                 'e36': 'e11', 'e16': 'e12', 'e25': 'e18',
                 'bond_40': 'bond_38', 'e26': 'e21', 'bond_41': 'bond_39'}.
            key comes from `prod_rule`, value comes from `self`.
        """
        if self.is_start_rule != prod_rule.is_start_rule:
            return False, {}
        if (not self.is_start_rule) and (prod_rule.lhs.num_nodes != self.lhs.num_nodes):
            return False, {}
        if (self.rhs.num_nodes != prod_rule.rhs.num_nodes) or (self.rhs.num_edges != prod_rule.rhs.num_edges):
            return False, {}

        # 比较 rhs 中节点（通常表示键）的 symbol 分布
        counter_self_nodes = Counter([self.rhs.node_attr(n)['symbol'] for n in self.rhs.nodes])
        counter_other_nodes = Counter([prod_rule.rhs.node_attr(n)['symbol'] for n in prod_rule.rhs.nodes])
        if counter_self_nodes != counter_other_nodes:
            return False, {}

        # 比较 rhs 中边（通常表示原子）的 symbol 分布
        counter_self_edges = Counter([self.rhs.edge_attr(e)['symbol'] for e in self.rhs.edges])
        counter_other_edges = Counter([prod_rule.rhs.edge_attr(e)['symbol'] for e in prod_rule.rhs.edges])
        if counter_self_edges != counter_other_edges:
            return False, {}

        return self._check_iso(prod_rule, ignore_order)

    def _compute_inner_symmetry(self):
        """
        用来标注规则如何进行对称翻转，
        例1: bond_0 - e0 - bond_1
        例2: bond_0 - e0 - bond_2 - e1 - bond_1 # ['bond_0','e0','bond_2', 'e1', 'bond1']
        例3: bond_0 - e0 - bond_1 - e1 - bond_2

        return:
        self._symmetry_map_ = False 表示不对称, True表示对称且内部symbol不需要调整
        self._symmetry_map_ = True 表示对称且内部symbol不需要调整
        self._symmetry_map_ = Dict 表示内部symbol的对称映射规则，

         - 例2中不需要调整，因为e0左侧的bond id小，e0和e1交换位置后（self._symmetry_map_ 表示），e0'右侧的bond id也小，已经实现了e0内部的翻转，故不需要额外调整
         - 例3中self._symmetry_map_ = {0:1,1:0}, 表示需要交换e0和e1
        """

        self._symmetry_map_ = False
        self._flipped_children_idx_ = set()

        # --------------- Step 1. Compute _symmetry_map_  ---------------
        # Case 1 (例1): 只有一个e肯定是对称的
        if self.rhs.num_edges <= 1:
            self._symmetry_map_ = True
            return self._symmetry_map_, self._flipped_children_idx_

        # Case 2: 若存在多个external nodes，但是都连接在同一个e上认为是对称的
        # 计算与 external nodes 相连的边数量
        ext_edges_num = sum(1 for b_list in self.rhs.nodes_in_edge_dict.values() if set(b_list) & self.lhs._nodes_)
        if ext_edges_num <= 1:
            self._symmetry_map_ = True
            return self._symmetry_map_, self._flipped_children_idx_

        # Case 3: external node >=3 我们认为是不对称的
        if len(self.lhs._nodes_) >= 3:
            self._symmetry_map_ = False
            return self._symmetry_map_, self._flipped_children_idx_

        # Case 4: 有两个位于不同edge(atom)上的external nodes
        # 最复杂的情况，可能对称但是需要调整内部edge顺序
        assert len(self.lhs._nodes_) == 2, "Only 2 non terminal symbol is the legal case"


        def extract_e_numbers(route):
            """ 从一条路径中提取所有以 'e' 开头的节点数字。
            例如: ['bond_1', 'e4', 'bond_2', 'e3', 'bond_3'] -> [4, 3] """
            return [int(node[1:]) for node in route if node.startswith('e')]

        def extract_bond_numbers(route):
            return [int(node[5:]) for node in route if node.startswith('bond_')]


        def symmetric_mapping(lst):
            """
            根据列表 lst 构造一个逆序对称映射字典。
            对于 lst = [a, b, c, ...] 返回 {a: last, b: second_last, ...}。
            例如: [4, 3] -> {4: 3, 3: 4}
            """
            n = len(lst)
            return {lst[i]: lst[n - 1 - i] for i in range(n)}

        s, t = self.lhs.nodes
        routes = list(nx.all_simple_paths(self.rhs.hg, s, t))

        # Step1. 检查bond是否可逆
        bond_paths = [extract_bond_numbers(route) for route in routes]
        for bp in bond_paths:
            b_mapping = symmetric_mapping(bp)
            for x, y in b_mapping.items():
                if self.rhs.node_attr(f'bond_{x}')['symbol'] != self.rhs.node_attr(f'bond_{y}')['symbol']:
                    self._symmetry_map_ = False
                    return self._symmetry_map_, self._flipped_children_idx_

        # Step2：检查atom是否可逆
        # 提取每条路径中以 'e' 开头的节点，并转换为数字
        e_paths = [extract_e_numbers(route) for route in routes]
        e_mapping = [symmetric_mapping(e_lst) for e_lst in e_paths] # 构造e_paths的逆序路径

        # Step 2.1: 检查所有映射字典是否存在冲突，若存在冲突说明不对称
        merged_mapping = {}
        for d in e_mapping:
            for key, value in d.items():
                if key in merged_mapping and merged_mapping[key] != value:
                    self._symmetry_map_ = False
                    return self._symmetry_map_, self._flipped_children_idx_
                merged_mapping[key] = value

        # Step 2.2: 检查是否出现了所有的e，若有没出现的e->不对称
        if self.rhs.num_edges != len(merged_mapping):
            # 说明路径中没有出现所有的e, 不对称
            self._symmetry_map_ = False
            return self._symmetry_map_, self._flipped_children_idx_

        # Step 2.3: 进一步检查映射中的所有点对是否相同，相同才能进行对称翻转
        for x, y in merged_mapping.items():
            if self.rhs.edge_attr(f'e{x}')['symbol'] != self.rhs.edge_attr(f'e{y}')['symbol']:
                self._symmetry_map_ = False
                return self._symmetry_map_, self._flipped_children_idx_

        # merged_mapping反应了所有e之间的对应关系
        self._symmetry_map_ = merged_mapping

        # Step 2.4: find flipped_children_idx_ ---------------
        def find_flipped_indices(path):
            """ 给定一个 bond 编号列表 path，计算其与 reverse(path) 相比，相邻元素大小关系符号发生变化的索引列表。 """

            # 计算相邻元素间的符号列表，使用 True 表示 '<'，False 表示 '>'
            orig_signs = [(a < b) for a, b in zip(path, path[1:])]
            rev_path = path[::-1]
            rev_signs = [(a < b) for a, b in zip(rev_path, rev_path[1:])]

            # 找出符号发生变化的位置
            flipped = [i for i, (o, r) in enumerate(zip(orig_signs, rev_signs)) if o != r]
            return flipped

        flipped_children_idx = set()  # 用于表示在将rule翻转后，内部哪些ex也需要跟着翻转
        for rid, route in enumerate(routes):
            b_path = [int(n[5:]) for n in route if n.startswith('bond_')]
            e_ids = find_flipped_indices(b_path)
            for eid in e_ids:
                flipped_children_idx.add(e_paths[rid][eid])


        self._flipped_children_idx_ = flipped_children_idx
        return self._symmetry_map_, self._flipped_children_idx_

    @property
    def symmetry_map(self):
        if not hasattr(self, '_symmetry_map_'):
            self._compute_inner_symmetry()
        return self._symmetry_map_


    @property
    def flipped_children_idx(self):
        if not hasattr(self, '_flipped_children_idx_'):
            self._compute_inner_symmetry()
        return self._flipped_children_idx_


    def apply_to_graph(self, hg, nt_list):
        """ augment `hg` by replacing `edge` with `self.rhs`.

        Parameters
        ----------
        hg : Hypergraph
        edge : str
            `edge` must belong to `hg`

        Returns
        -------
        hg : Hypergraph
            resultant hypergraph
        nt_edge_list : list
            list of non-terminal edges
        """

        if self.is_start_rule:
            hg = Hypergraph()

            node_map_rhs = {}  # node (bond) id in rhs -> node (bond) id in hg, where rhs is augmented.
            for num_idx, each_node in enumerate(self.rhs.nodes):
                new_node = f"bond_{num_idx}"
                hg.add_node(new_node, attr_dict=self.rhs.node_attr(each_node))
                node_map_rhs[each_node] = new_node

            nt_edge_dict = {}
            for each_edge in self.rhs.edges:
                bID2ruleID =  {node_map_rhs[each_node]:each_node for each_node in self.rhs.nodes_in_edge(each_edge)}
                node_list = list(bID2ruleID.keys()) #[node_map_rhs[each_node] for each_node in self.rhs.nodes_in_edge(each_edge)]
                if isinstance(self.rhs.nodes_in_edge(each_edge), set):
                    node_list = set(node_list)
                 # 这里的edge_attr记录了边对应的rhs中的id
                edge_id = hg.add_edge(node_list, attr_dict={**self.rhs.edge_attr(each_edge), **{'bID2ruleID':bID2ruleID}})
                if "nt_idx" in hg.edge_attr(edge_id):  # 咋没这个属性？？？
                    nt_edge_dict[hg.edge_attr(edge_id)["nt_idx"]] = edge_id

            # nt_edge_list = {nt_edge_dict[key]: None for key in range(len(nt_edge_dict))}  # [::-1]
            nt_edge_list = [nt_edge_dict[key] for key in range(len(nt_edge_dict))]
            return hg, nt_edge_list
        else:
            hg_NT_edges = hg.get_all_NT_edges()
            lhs_NT_edges = self.lhs.get_all_NT_edges()
            assert len(lhs_NT_edges) == 1, "There should be exactly one non-terminal edge in lhs."
            assert len(list(lhs_NT_edges)[0]._edges_) == 1, "非终结边 NT 中应只有1个 edge (atom)"

            lhs_edge = lhs_NT_edges[0]
            if lhs_edge not in hg_NT_edges:
                # If hypergraph does not contain a matching non-terminal edge, return original.
                # 如果超图中不含匹配的非终结边，则直接返回原图
                return hg, []

            # 从超图中找到匹配 lhs 的所有非终结边
            matches = [e for e in hg_NT_edges if e == lhs_edge]
            # if it is empty, meaning that there is only actually one NT in hg that matches multiple NTs in the lhs, the rule should be abandoned
            if len(matches) == 0:
                print("[Wrong] matches is empty")
                return hg, []

            edges_name_to_hg = {list(edge_hg.edges)[0]: edge_hg for edge_hg in matches}
            # edges_cand = edges_name_to_hg.keys()

            # 下面这段是我加的：nt_list从后往前遍历时第一个出现在edges_can中的元素，然后从nt_list中删除该元素
            # for selected_edge in reversed(nt_list.keys()):
            #     if selected_edge in edges_cand:
            #         atom_edge_maps = nt_list.pop(selected_edge)
            #         break
            selected_edge = nt_list[-1]
            # atom_edge_maps = nt_list.pop(selected_edge)

            edge_hg = edges_name_to_hg[selected_edge]
            # iso_mapping = edge_hg.find_amended_iso_mapping(lhs_edge, atom_edge_maps) # From edge_hg to lhs_edge
            iso_mapping = edge_hg.find_isomorphism_mapping(lhs_edge, all_mapping=True, stable=False)

            # r若有多个同构的要根据bID2ruleID进行排序
            if len(iso_mapping) <= 1:
                bID2ruleID = {k: v for k, v in iso_mapping[0].items() if k.startswith("bond_")}
            else:
                bID2ruleID = hg.node_attr(selected_edge)["bID2ruleID"]
                bID2ruleID = find_min_mapping(iso_mapping, bID2ruleID)


            nt_order_dict_inv = {i: k for i, (k, _) in enumerate(
                sorted(bID2ruleID.items(), key=lambda item: int(item[1][5:])))}



            # # order of nodes that belong to the non-terminal edge in hg
            # nt_order_dict = {}  # hg_node -> order ("bond_17" : 1)
            # nt_order_dict_inv = {}  # order -> hg_node
            # for each_idx, each_node in enumerate(hg.nodes_in_edge(selected_edge)):
            #     mapped_node_in_lhs = iso_mapping[each_node]
            #     ext_id = lhs_edge.node_attr(mapped_node_in_lhs)['ext_id']
            #     nt_order_dict[each_node] = ext_id
            #     nt_order_dict_inv[ext_id] = each_node




            # Remove the selected non-terminal edge. 从超图中删除选中的非终结边
            hg.remove_edge(selected_edge)  # delete non-terminal

            # -------2.下面这边和MHG-VAE中的代码一样
            # construct a node_map_rhs: rhs -> new hg
            node_map_rhs = {}  # node id in rhs -> node id in hg, where rhs is augmented.
            node_idx = hg.num_nodes
            for each_node in self.rhs.nodes:
                if "ext_id" in self.rhs.node_attr(each_node):
                    node_map_rhs[each_node] = nt_order_dict_inv[self.rhs.node_attr(each_node)["ext_id"]] # exit_id好像没有用到？
                else:
                    node_map_rhs[each_node] = f"bond_{node_idx}"
                    node_idx += 1

            # add nodes to hg
            for each_node in self.rhs.nodes:
                hg.add_node(node_map_rhs[each_node], attr_dict=self.rhs.node_attr(each_node))

            # add hyperedges to hg
            nt_edge_dict = {}
            for each_edge in self.rhs.edges:
                bID2ruleID = {node_map_rhs[each_node]: each_node for each_node in self.rhs.nodes_in_edge(each_edge)}
                node_list_hg = list(bID2ruleID.keys()) #[node_map_rhs[n] for n in self.rhs.nodes_in_edge(each_edge)]
                edge_id = hg.add_edge(node_list_hg, attr_dict={**self.rhs.edge_attr(each_edge),**{'bID2ruleID': bID2ruleID}})  # deepcopy(self.rhs.edge_attr(each_edge)))
                if "nt_idx" in hg.edge_attr(edge_id):
                    nt_edge_dict[hg.edge_attr(edge_id)["nt_idx"]] = edge_id

            # # 反转字典：键值互换  (node id in hg -> node id in rhs)
            # # atom_edge_maps 用于记录新插入的atom edge两边的bond node编号是否存在逆序
            # hg_rhs_node_map = {v: k for k, v in node_map_rhs.items()}
            # for atom_edge in nt_edge_dict.values():
            #     atom_edge_maps = {hg_bond: hg_rhs_node_map[hg_bond] for hg_bond in hg.nodes_in_edge_dict[atom_edge]}
            #     nt_list[atom_edge] = atom_edge_maps if has_inversion_bond(atom_edge_maps, hg) else None
            nt_list = nt_list[:-1] + [nt_edge_dict[key] for key in range(len(nt_edge_dict))]

            return hg, nt_list











