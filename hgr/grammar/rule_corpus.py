# grammar/rule_corpus.py

import os
import torch
from typing import Tuple
import logging
from collections import defaultdict, Counter
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wlhash

from hgr.grammar.hypergraph import Hypergraph
from hgr.grammar.rule import ProductionRule
# from hgr.grammar.symbol import TSymbol, NTSymbol, BondSymbol
from hgr.grammar.grammar_utils import has_inversion_bond, masked_softmax


logger = logging.getLogger(__name__)

class ProductionRuleCorpus(object):

    '''
    A corpus of production rules.
    This class maintains 
        (i) list of unique production rules,
        (ii) list of unique edge symbols (both terminal and non-terminal), and
        (iii) list of unique node symbols.

    Attributes
    ----------
    prod_rule_list : list
        list of unique production rules
    edge_symbol_list : list
        list of unique symbols (including both terminal and non-terminal)
    node_symbol_list : list
        list of node symbols
    nt_symbol_list : list
        _ist of unique lhs symbols
    ext_id_list : list
        list of ext_ids
    lhs_in_prod_rule : array
        a matrix of lhs vs prod_rule (= lhs_in_prod_rule)
    '''

    def __init__(self, preserve_anchor_order=True):
        """
        preserve_anchor_order:
            True  -> 记录 anchor 的顺序；规则匹配包含对称/顺序等完整检查；可完全重构原始图（lossless）。
            False -> 不记录 anchor 顺序；只做 fingerprint + 最少使用次数选择；更快但不保证可逆。
        """
        self.prod_rule_list = []
        self.edge_symbol_list = []
        self.edge_symbol_dict = {}
        self.node_symbol_list = []
        self.node_symbol_dict = {}
        self.nt_symbol_list = []
        self.ext_id_list = []
        self._lhs_in_prod_rule = None
        self.lhs_in_prod_rule_row_list = []
        self.lhs_in_prod_rule_col_list = []
        self.lhs_list = []
  
        self._fp_index = defaultdict(list) # fingerprint -> list of rule indices,方便快速筛选是否有对应的rule
        self._nt_symbol_to_idx = {} # 为 nt_symbol_list 创建一个 O(1) 查找字典 

        if preserve_anchor_order:
            self.find_rule = self.find_rule_strict
        else:
            self.find_rule = self.find_rule_nosym

    def __setstate__(self, state):
        """
        Backward compatibility for pickled corpora created before `_nt_symbol_to_idx`.
        Ensures new attributes exist after unpickling.
        """
        self.__dict__.update(state)
        # Pickles from older runs may miss these newer fields
        if not hasattr(self, '_fp_index') or len(self._fp_index) == 0:
            self._fp_index = defaultdict(list)
            for rule_idx, pr in enumerate(self.prod_rule_list):
                fp = self._fingerprint(pr)
                self._fp_index[fp].append(rule_idx)

        if not hasattr(self, 'find_rule'):
            self.find_rule = self.find_rule_strict
    
        # 1) rebuild dict
        self._nt_symbol_to_idx = {sym: i for i, sym in enumerate(self.nt_symbol_list)}

        # 2) rebuild row/col lists from prod_rule_list
        self.lhs_in_prod_rule_row_list = []
        self.lhs_in_prod_rule_col_list = []
        for rule_idx, pr in enumerate(self.prod_rule_list):
            lhs_idx = self._nt_symbol_to_idx[pr.lhs_nt_symbol]
            self.lhs_in_prod_rule_row_list.append(lhs_idx)
            self.lhs_in_prod_rule_col_list.append(rule_idx)

        # 3) drop cached dense matrix
        self._lhs_in_prod_rule = None



    @property
    def lhs_in_prod_rule(self):
        """
        shape: (num_nt, num_pr)
        lhs_in_prod_rule[lhs_idx, rule_idx] = 1 表示：第 rule_idx 条产生式的 LHS 属于第 lhs_idx 行对应的非终结符。
        """
        if self._lhs_in_prod_rule is None:
            # old_lhs_in_prod_rule = torch.sparse.FloatTensor(
            #         torch.LongTensor(list(zip(self.lhs_in_prod_rule_row_list, self.lhs_in_prod_rule_col_list))).t(),
            #         torch.FloatTensor([1.0]*len(self.lhs_in_prod_rule_col_list)),
            #         torch.Size([len(self.nt_symbol_list), len(self.prod_rule_list)])
            #         ).to_dense()
            row_idx = torch.tensor(self.lhs_in_prod_rule_row_list, dtype=torch.long)
            col_idx = torch.tensor(self.lhs_in_prod_rule_col_list, dtype=torch.long)
            indices = torch.stack([row_idx, col_idx])  # shape: (2, N)
            values = torch.ones(len(col_idx), dtype=torch.float)
            shape = (len(self.nt_symbol_list), len(self.prod_rule_list))
            self._lhs_in_prod_rule = torch.sparse_coo_tensor(indices, values, size=shape).to_dense()
        return self._lhs_in_prod_rule

    def to(self, device):
        self._lhs_in_prod_rule = self.lhs_in_prod_rule.to(device)
        return self

    @property
    def num_prod_rule(self):
        ''' return the number of production rules '''
        return len(self.prod_rule_list)


    @property
    def num_edge_symbol(self):
        return len(self.edge_symbol_list)

    @property
    def num_node_symbol(self):
        return len(self.node_symbol_list)

    @property
    def num_ext_id(self):
        return len(self.ext_id_list)


    def edge_symbol_idx(self, symbol):
        return self.edge_symbol_dict[symbol]

    def node_symbol_idx(self, symbol):
        return self.node_symbol_dict[symbol]

    @staticmethod
    def _fingerprint_fast(prod_rule: ProductionRule) -> Tuple:
        '''
        Compute an immutable fingerprint for a production rule.
        Fingerprint fields:
          - lhs non-terminal symbol (string)
          - rhs node count, rhs edge count
          - multiset of rhs node symbols
          - multiset of rhs edge symbols
        '''
        # LHS symbol as string
        lhs_sym = prod_rule.lhs_nt_symbol
        # is_start_rule = prod_rule.lhs.num_nodes
        rhs = prod_rule.rhs
        # rhs node count and edge count
        n_nodes = rhs.num_nodes
        n_edges = rhs.num_edges

        # multiset of node symbols
        node_counter = Counter(rhs.node_attr(n)['symbol'] for n in rhs.nodes)
        node_syms = frozenset(node_counter.items())
        # multiset of edge symbols
        edge_counter = Counter(rhs.edge_attr(e)['symbol'] for e in rhs.edges)
        edge_syms = frozenset(edge_counter.items())
        return (lhs_sym, n_nodes, n_edges, node_syms, edge_syms)

    
    @staticmethod
    def _fingerprint(prod_rule: ProductionRule) -> Tuple:
        """
        根据 preserve_anchor_order 策略计算指纹。
        """
        lhs_sym = prod_rule.lhs_nt_symbol

        try:
            """
            为 RHS hypergraph 计算一个规范图哈希 (canonical hash)。
            这完美对应 is_same(..., ignore_order=True) 的逻辑。
            """
            H = prod_rule.rhs.hg.copy() # 创建一个副本，以防修改原始图
            # WL-Hash 要求节点属性是字符串。因此我们将 symbol 对象 (如 BondSymbol(...)) 转换为它们的字符串表示。
            for bond_node, data in H.nodes(data=True):
                # _node_match... 只关心 'symbol'
                data['str_symbol'] = str(data.get('symbol', ''))
            
            rhs_hash = wlhash(H, node_attr='str_symbol', edge_attr=None, iterations=4)
            return (lhs_sym, rhs_hash)

        
        except Exception as e:
            logger.error(f"WL-Hash failed for rule, falling back to fast fingerprint: {e}")
            fast_fp = ProductionRuleCorpus._fingerprint_fast(prod_rule) 
            return ("FAST_FALLBACK", fast_fp)
        


    def find_rule_nosym(self, prod_rule: ProductionRule, subg=None, fp=None):
        '''
        Fast lookup: use fingerprint index to fetch candidates, then apply is_same for verification.
        '''
        if fp is None:
            fp = self._fingerprint(prod_rule)
        cand_idxs = self._fp_index.get(fp, [])
        if not cand_idxs:
            return None, None

        # rule_idx_min = min(cand_idxs, key=lambda idx: self.prod_rule_list[idx].used_count)
        # exist_rule = self.prod_rule_list[rule_idx_min]
        # return exist_rule, rule_idx_min

        for rule_idx in sorted(cand_idxs, key=lambda idx: self.prod_rule_list[idx].used_count):
            each_rule = self.prod_rule_list[rule_idx]
            is_same, _ = prod_rule._check_iso(each_rule, ignore_order=True)
            if is_same:
                return each_rule, rule_idx
        return None, None



    def find_rule_strict(self, prod_rule: ProductionRule, subg, fp=None):
        '''
        Fast lookup: use fingerprint index to fetch candidates, then apply is_same for verification.
        '''
        if fp is None:
            fp = self._fingerprint(prod_rule)

        cand_idxs = self._fp_index.get(fp, [])
        if not cand_idxs:
            return None, None

        # 对当前节点的子节点按照当前weights降序排序(weight小的对应小e_x)
        sorted_children = sorted(subg.children, key=lambda c: c.weight, reverse=True)

        for rule_idx in sorted(cand_idxs, key=lambda idx: self.prod_rule_list[idx].used_count):
            each_rule = self.prod_rule_list[rule_idx]
            is_same, _ = prod_rule._check_iso(each_rule, ignore_order=True)
            if not is_same:
                continue

            # 进入下面的循环说明找到相同的了，再进行进一步检查
            sub_symmetry_mapping = []  # 记录需要对称映射（调整顺序）的 e_id
            symmetry_map = True
            if len(sorted_children) >= 1:
                mapping = prod_rule.rhs.find_priority_iso_mapping(each_rule.rhs, prioritys=prod_rule.lhs._nodes_)
                # mapping_list = list(prod_rule.rhs._iso_mapping_iter_(each_rule.rhs))
                # mapping = min(mapping_list, key=lambda m: mapping_sort_key(m, priority_keys=prod_rule.lhs._nodes_))
                # Step1. 判断rhs自身是否产生了逆序对。
                #   无逆序对：不用处理
                #   产生了 -> 判断规则是否可逆
                #       可逆 -> 最后在e_mapping处处理e顺序的不一致
                #       不可逆 -> 新规则
                ext_b_node_mapping = {bn: mapping[bn] for bn in prod_rule.lhs._nodes_}
                need_symmetry_mapping = True if has_inversion_bond(ext_b_node_mapping, prod_rule.lhs) else False
                if need_symmetry_mapping and each_rule.symmetry_map is False:
                    continue
                # Step 2. 判断内部atom edge是否产生了逆序
                # 按照e0, e1, ...的顺序遍历键值对，这样可以和sorted_children的顺序对应上
                for e_id, edge in enumerate(sorted(prod_rule.rhs.nodes_in_edge_dict, key=lambda k: int(k[1:]))):
                    nodes_in_edge = prod_rule.rhs.nodes_in_edge_dict[edge]
                    # assert e_id == int(edge[1:]) # 一定成立
                    # e_x (atom) 只连接到一个bond, 则一定不存在逆序对
                    if len(nodes_in_edge) <= 1:
                        continue
                    # 一个edge(atom)连接到超过两条边就有可能产生顺序问题,
                    # 如果edge(atom)连接到的两条边类型不一致判断是否存在逆序
                    each_e_mapping = {b: mapping[b] for b in nodes_in_edge}
                    if has_inversion_bond(each_e_mapping, prod_rule.rhs):
                        symmetry_map = sorted_children[e_id].symmetry_map  # 存在逆序对则需要进一步判断该e_x是否对称(= True 表示对称且内部symbol不需要调整)
                        if symmetry_map is False:
                            break # False 表示不对称, 需要构造新的规则
                        elif isinstance(symmetry_map, dict): # 这边each_e_mapping必须长度<=2
                            assert len(each_e_mapping) <= 2, "each_e_mapping should be of length <= 2"
                            sub_symmetry_mapping.append(e_id)  #  Dict 表示内部symbol的对称映射规则


            if symmetry_map is False:  # 如果不对称则需要构造新的规则
                continue

            # # we do not care about edge and node names, but care about the order of non-terminal edges.
            # for key, val in isomap.items():  # key : edges & nodes in each_prod_rule.rhs , val : those in prod_rule.rhs
            #     if key.startswith("bond_"):
            #         continue
            #
            #     # TODO: 后面检查一下下面这段代码什么作用
            #     # rewrite `nt_idx` in `prod_rule` for further processing
            #     if "nt_idx" in prod_rule.rhs.edge_attr(val).keys():
            #         if "nt_idx" not in each_rule.rhs.edge_attr(key).keys():
            #             raise ValueError
            #         prod_rule.rhs.set_edge_attr(val, {'nt_idx': each_rule.rhs.edge_attr(key)["nt_idx"]})

            # 虽然存在逆序对，但是都是对称的（或者不需要调整权重）
            for child_id in sub_symmetry_mapping:
                # children会递归调用reverse_children()
                sorted_children[child_id].reverse_children()

            if len(prod_rule.rhs._edges_) >= 2:
                assert len(sorted_children) == len(prod_rule.rhs._edges_)
                # 这边不能用subg.reverse_children()替换（反例：--x--x，--表示bond node, x 表示atom edge)
                # e_mapping用于表示原子的顺序是否需要改变
                e_mapping = {int(k[1:]): int(v[1:]) for k, v in mapping.items() if k.startswith('e')}
                if any(k != v for k, v in e_mapping.items()):  # 如果是恒等映射，则设为 None
                    subg.transform_children_weight(e_mapping)

            return each_rule, rule_idx

        # 如果遍历结束后未找到匹配的规则，则返回None
        return None, None

    def insert_rule(self, prod_rule):
        # 如果遍历结束后未找到匹配的规则，则构造新规则并更新相应列表
        rule_idx = len(self.prod_rule_list)
        prod_rule.rule_idx = rule_idx
        self.prod_rule_list.append(prod_rule)
        self.lhs_list.append(prod_rule.lhs)
        self._update_edge_symbol_list(prod_rule)
        self._update_node_symbol_list(prod_rule)
        self._update_ext_id_list(prod_rule)

        # lhs_idx = self.nt_symbol_list.index(prod_rule.lhs_nt_symbol)
        lhs_idx = self._nt_symbol_to_idx[prod_rule.lhs_nt_symbol]
        self.lhs_in_prod_rule_row_list.append(lhs_idx)
        self.lhs_in_prod_rule_col_list.append(rule_idx)
        self._lhs_in_prod_rule = None

        # update fingerprint index
        fp = self._fingerprint(prod_rule)
        self._fp_index[fp].append(rule_idx)

        return rule_idx

    def append(self, prod_rule: ProductionRule, subg=None, fp=None) -> Tuple[int, ProductionRule]:
        """ return whether the input production rule is new or not, and its production rule id.
        Production rules are regarded as the same if 
            i) there exists a one-to-one mapping of nodes and edges, and
            ii) all the attributes associated with nodes and hyperedges are the same.
        判断并添加新的产生式规则，如果已存在则返回已有的规则索引及规则本身。

        Parameters
        ----------
        prod_rule : ProductionRule

        Returns
        -------
        prod_rule_id : int
            production rule index. if new, a new index will be assigned.
        prod_rule : ProductionRule
        """
        exist_rule, rule_idx = self.find_rule(prod_rule, subg, fp)

        if rule_idx is not None:
            exist_rule.used_count += 1
            # 如果已经存在相同的规则，则返回该规则的索引和规则本身
            return exist_rule, rule_idx
        else:
            rule_idx = self.insert_rule(prod_rule)
            return prod_rule, rule_idx
    

    def get_prod_rules_with_lhs(self, lhs: Hypergraph):
        assert isinstance(lhs, Hypergraph)
        return [self.get_prod_rule(i) for i, _lhs in enumerate(self.lhs_list)
                if lhs.is_same(_lhs, ignore_order=True)]


    def get_prod_rule(self, prod_rule_idx: int) -> ProductionRule:
        return self.prod_rule_list[prod_rule_idx]

    # @ property
    # def _nt_symbol_to_idx(self):
    #     # self.nt_symbol_to_idx[nt_symbol] 等价于 self.nt_symbol_list.index(nt_symbol)，但效率略高
    #     # if not hasattr(self, '_nt_symbol_to_idx'):
    #     #     self._nt_symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.nt_symbol_list)}
    #     return self._nt_symbol_to_idx


    def sample(self, unmasked_logit_array, nt_symbol, deterministic=False):
        ''' sample a production rule whose lhs is `nt_symbol`, followihng `unmasked_logit_array`.

        Parameters
        ----------
        unmasked_logit_array : array-like, length `num_prod_rule`
        nt_symbol : NTSymbol
        '''
        # if not isinstance(unmasked_logit_array, np.ndarray):
        #     unmasked_logit_array = unmasked_logit_array.numpy().astype(np.float64)

        #nt_idx = self.nt_symbol_list.index(nt_symbol)
        nt_idx = self._nt_symbol_to_idx[nt_symbol]
        #prob = masked_softmax(unmasked_logit_array, self.lhs_in_prod_rule[idx].numpy().astype(np.float64))
        prob = masked_softmax(unmasked_logit_array, self.lhs_in_prod_rule[nt_idx])
        idx = torch.argmax(prob).item() if deterministic else torch.multinomial(prob, num_samples=1).item()
        return self.prod_rule_list[idx]
        # if deterministic:
        #     return self.prod_rule_list[torch.argmax(prob).item()]
        # else:
        #     return np.random.choice(self.prod_rule_list, p=prob)



        # if deterministic:
        #     prob = masked_softmax(unmasked_logit_array,
        #                           self.lhs_in_prod_rule[self.nt_symbol_list.index(nt_symbol)].numpy().astype(np.float64))
        #     return self.prod_rule_list[np.argmax(prob)]
        # else:
        #     return np.random.choice(
        #         self.prod_rule_list, 1,
        #         p=masked_softmax(unmasked_logit_array,
        #                          self.lhs_in_prod_rule[self.nt_symbol_list.index(nt_symbol)].numpy().astype(np.float64)))[0]

    def _update_edge_symbol_list(self, prod_rule: ProductionRule):
        ''' update edge symbol list

        Parameters
        ----------
        prod_rule : ProductionRule
        '''
        if prod_rule.lhs_nt_symbol not in self.nt_symbol_list:
            self.nt_symbol_list.append(prod_rule.lhs_nt_symbol)
            nt_idx = len(self.nt_symbol_list) - 1 # 获取新索引 
            self._nt_symbol_to_idx[prod_rule.lhs_nt_symbol] = nt_idx # 更新字典

        for each_edge in prod_rule.rhs.edges:
            sym = prod_rule.rhs.edge_attr(each_edge)['symbol']
            if sym not in self.edge_symbol_dict:
                edge_symbol_idx = len(self.edge_symbol_list)
                self.edge_symbol_list.append(sym)
                self.edge_symbol_dict[sym] = edge_symbol_idx
            else:
                edge_symbol_idx = self.edge_symbol_dict[sym]
            prod_rule.rhs.edge_attr(each_edge)['symbol_idx'] = edge_symbol_idx
        # pass

    def _update_node_symbol_list(self, prod_rule: ProductionRule):
        ''' update node symbol list

        Parameters
        ----------
        prod_rule : ProductionRule
        '''
        for each_node in prod_rule.rhs.nodes:
            sym = prod_rule.rhs.node_attr(each_node)['symbol']
            if sym not in self.node_symbol_dict:
                node_symbol_idx = len(self.node_symbol_list)
                self.node_symbol_list.append(sym)
                self.node_symbol_dict[sym] = node_symbol_idx
            else:
                node_symbol_idx = self.node_symbol_dict[sym]
            prod_rule.rhs.node_attr(each_node)['symbol_idx'] = node_symbol_idx

    def _update_ext_id_list(self, prod_rule: ProductionRule):
        for each_node in prod_rule.rhs.nodes:
            ext_id = prod_rule.rhs.node_attr(each_node).get('ext_id')
            if (ext_id is not None) and (ext_id not in self.ext_id_list):
                self.ext_id_list.append(ext_id)


    def plot_production_rule(self, save_folder=None, file_name=None):
        from PIL import Image
        rule_name = 'Rule' if file_name is None else (file_name[:-4] if file_name.lower().endswith(".png") else file_name)

        for idx, prod_rule in enumerate(self.prod_rule_list):
            lpth = os.path.join(save_folder, f'tmp-{idx}-L.png')
            rpth = os.path.join(save_folder, f'tmp-{idx}-R.png')
            prod_rule.lhs.draw(lpth, with_edge_name=False)
            prod_rule.rhs.draw(rpth, with_edge_name=False)
            # 使用 PIL 拼接左右两张图片
            img1, img2 = Image.open(lpth), Image.open(rpth)
            # 计算拼接后图片的尺寸
            new_width = img1.width + img2.width
            new_height = max(img1.height, img2.height)
            new_img = Image.new("RGB", (new_width, new_height), "white")

            # 将第一张图片粘贴到左侧，第二张粘贴到右侧
            new_img.paste(img1, (0, 0))
            new_img.paste(img2, (img1.width, 0))

            # 显示拼接后的图片
            new_img.save(os.path.join(save_folder, rule_name + f'_{idx}.png'))

            # 关闭图片文件，防止文件占用问题
            img1.close()
            img2.close()

            # 删除临时图片
            os.remove(lpth)
            os.remove(rpth)

        print(f"Production rules visualizations are saved in {save_folder}.")




