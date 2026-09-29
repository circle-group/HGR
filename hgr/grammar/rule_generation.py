# grammar/rule_generation.py

from hgr.grammar.symbol import NTSymbol
from hgr.grammar.hypergraph import Hypergraph
from hgr.grammar.rule import ProductionRule


def generate_rule(input_graph, subg):
    """根据输入图与子图生成产生式规则，并更新 grammar。

    Returns:
        更新后的 grammar 对象。
    """

    def _add_ext_node(hg, ext_nodes):
        """ mark nodes to be external (ordered ids are assigned)

        Parameters
        ----------
        hg : UndirectedHypergraph
        ext_nodes : list of str
            list of external nodes

        Returns
        -------
        hg : Hypergraph
            nodes in `ext_nodes` are marked to be external
        """
        ext_id_exists = ['ext_id' in hg.node_attr(n) for n in ext_nodes]
        # 如果这批节点中有的已经有 ext_id、有的没有，会抛出异常（因为状态不一致）。
        if ext_id_exists and any(ext_id_exists) != all(ext_id_exists):
            raise ValueError

        # 如果都还没分配过，就按顺序给它们编号。这样后续在产生式规则的 lhs 里，我们就能知道这些外部节点的次序。
        if not all(ext_id_exists):
            for ext_id, each_node in enumerate(ext_nodes):
                hg.node_attr(each_node)['ext_id'] = ext_id

        return hg

    def _check_aromatic(hg, node_list):
        return any(hg.node_attr(node)['symbol'].is_aromatic for node in node_list)

    ''' high-level idea:
    find all ext_nodes: for each edge with [:1], find its adj nodes, for those nodes that are not included in the subg, set them as ext_node
    lhs: single NT with ext_nodes
    rhs: all visited edges shrink to NT
    '''
    input_hg = input_graph.hypergraph
    subhg = subg.hypergraph
    ext_node_list = set()  #用来存储所有“外部节点”（即不在子图内、但与子图中节点相连的那些节点）。后续要将这些节点标记为外部节点。
    watershed_mapping = {} # {water_level: set(of external nodes)} 用于后续处理“已访问边”的情况
    
    # 预计算加速
    edge_map = {e: f'e{subg.get_org_idx_in_input(int(e[1:]))}' for e in subhg.edges}
    node_map = {} # # 返回node（bond) 对应的原图中的bond; 对应input_graph.get_org_node_in_input(n, subg)
    for subg_node in subhg.nodes:
        adj_edges = list(subhg.adj_edges(subg_node))
        # 使用 edge_map 的 O(1) 查找
        org_edge_1 = edge_map[adj_edges[0]] 
        org_edge_2 = edge_map[adj_edges[1]]
        org_node = list(set(input_hg.nodes_in_edge(org_edge_1)) & 
                          set(input_hg.nodes_in_edge(org_edge_2)))[0]
        node_map[subg_node] = org_node
    
    all_nodes_subg_set = set(node_map.values())

    ## ------------------------ Construct RHS ------------------------
    # 目标：复制子图结构，同时将未访问边和已访问边分别处理，
    #      对于未访问边：复制未访问节点；对于未终结边，将原图中不在子图内的节点视为外部节点。
    # Several goals to achieve during the loop
    # for rhs use: make a copy of subg_hyper_graph, with all attr replaced with attr from original input_g, connect ext_nodes to the hypergraph
    # for lhs use: get the list of ext_nodes
    rhs = Hypergraph()
    visited_edges = {}
    for edge in subhg.edges:
        # Map all the nodes and edges to the input to get latest status&attribute
        # org_edge = f'e{subg.get_org_idx_in_input(int(edge[1:]))}' # Get the original name of edge
        org_edge = edge_map[edge]  # Get the original name of edge

        if not input_hg.edge_attr(org_edge)['visited']: # input_graph.hypergraph.hg.nodes['bond_0']
            subg_bnodes = subhg.nodes_in_edge(edge)
            # Get all the original names of nodes (bond) for the current subg
            subg_bnodes_org = set(node_map[n] for n in subg_bnodes)
            # for unvisited edge 则将其所连接的原子/键都复制到 RHS”。
            # 同时，如果该边的 terminal=False（代表它不是叶子边），则查找它在原超图（input_hg）中连向子图以外的节点，将这些节点当作 RHS 中的“外部节点”（ext_node）。
            node_list = []  # Nodes to be added to the rhs
            for bn in subg_bnodes:
                org_node = node_map[bn]
                # only need to add unvisited nodes
                if not input_hg.node_attr(org_node)['visited']:
                    rhs.add_node(org_node, input_hg.node_attr(org_node).copy())
                    node_list.append(org_node)

            if not subhg.edge_attr(edge)['terminal']:
                # 对于非终结的未访问边，如果它连接到子图之外的节点，就把那些节点视为外部节点
                # for those unvisited edges with [:1], add ext nodes with original names if any, and update node_list for the use of rhs
                for b_node in input_hg.nodes_in_edge(org_edge):
                    # add ext_node
                    if b_node not in subg_bnodes_org: #如果它连接到子图之外的节点，
                        rhs.add_node(b_node, input_hg.node_attr(b_node).copy())
                        ext_node_list.add(b_node)
                        node_list.append(b_node)
            rhs.add_edge(node_list, attr_dict=input_hg.edge_attr(org_edge).copy())
        else:
            # 已访问边：因为已经在之前的某轮收缩过，我们只记录它的 water_level，
            # 最晚在下面的第二部分处理“已访问边”时，才会把该轮次对应的“外部节点集合”一次性加到 RHS。
            visit_seq = input_hg.edge_attr(org_edge)['water_level'] # 访问顺序
            visited_edges[visit_seq] = edge

    # Process visited edges: use water_level to collect external nodes.
    # 处理已访问边：依据 water_level 收集对应的外部节点，并建立映射。
    for edge_water_level, edge in sorted(visited_edges.items()):
        # each visited edge will have a water_level attribute
        # record the unvisited nodes and ext_nodes
        # edge_water_level = input_hg.edge_attr(org_edge)['water_level']
        nodes_to_add = input_graph.watershed_ext_nodes[edge_water_level].copy() # 取出“当时的外部节点”
        watershed_mapping.setdefault(edge_water_level, set()).update(nodes_to_add)

        for _node in nodes_to_add:
            if not (_node in all_nodes_subg_set):
                ext_node_list.add(_node)

    # Add necessary edges and nodes for the visited edges
    # 针对已访问边，根据每个 water_level 添加一条非终结边到 rhs
    for nt_idx, (water_level, node_set) in enumerate(sorted(watershed_mapping.items())):
        node_list = sorted(node_set, key=lambda x: int(x[5:])) # 元素形式为：bond_x
        bond_symbol_list = []
        for bn in node_list:
            rhs.add_node(bn, input_hg.node_attr(bn).copy())
            bond_symbol_list.append(input_hg.node_attr(bn)['symbol'])
        edge_attr_dict = dict(terminal=False,
                              nt_idx=nt_idx,
                              symbol=NTSymbol(degree=len(node_list),
                                              is_aromatic=_check_aromatic(input_hg, node_list),
                                              bond_symbol_list=bond_symbol_list))
        rhs.add_edge(node_list, attr_dict=edge_attr_dict)

    ext_node_list = sorted(ext_node_list, key=lambda x: int(x[5:])) # 元素形式为：bond_x
    try:
        rhs = _add_ext_node(rhs, ext_node_list)
    except ValueError:
        import pdb;
        pdb.set_trace()

    ## ------------------------ 构造 lhs ------------------------
    # If no ext_nodes in un_visited_edges, then it should be a starting rule
    starting = True if len(ext_node_list) == 0 else False
    lhs = Hypergraph()
    if not starting:
        bond_symbol_list = []
        for each_node in ext_node_list:
            lhs.add_node(each_node, input_hg.node_attr(each_node).copy())
            bond_symbol_list.append(input_hg.node_attr(each_node)['symbol'])
        edge_attr_dict = dict(terminal=False,
                              symbol=NTSymbol(degree=len(ext_node_list),
                                              is_aromatic=_check_aromatic(input_hg, ext_node_list),
                                              bond_symbol_list=bond_symbol_list))
        lhs.add_edge(ext_node_list, attr_dict=edge_attr_dict)
        try:
            lhs = _add_ext_node(lhs, ext_node_list)
        except ValueError:
            import pdb;
            pdb.set_trace()


    rule = ProductionRule(lhs, rhs)

    # InputGraph 的状态更新
    input_graph.update_subgraph(subg)

    return rule
