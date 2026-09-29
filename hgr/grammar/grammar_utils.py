
import torch


def has_inversion_bond(maps, hg) -> bool:
    """
       判断是否存在键映射中的“逆序对”：
       若 maps 中键的顺序与对应值的顺序不一致（即非单调对应），则视为有逆序。
       注意：不一致的bond_symbol在图同构匹配时不会配对，因此只需要对有相同的symbol的bond进行判断

       示例：
       - Eg1: {'bond_1': 'bond_1', 'bond_2': 'bond_0'} → bond_1 < bond_2 但 bond_1 > bond_0 → 有逆序对 → 返回 True
       - Eg2: {'bond_1': 'bond_3', 'bond_2': 'bond_5'} → bond_1 < bond_2 且 bond_3 < bond_5 → 无逆序 → 返回 False
       """

    # 仅一条映射，不可能有逆序
    if len(maps) <= 1:
        return False

    # -------- 1) 先按 symbol 分组 --------
    groups = {} # { symbol_str: [bond_key1, bond_key2, ...], ... }
    for k in maps.keys():
        symbol = hg.node_attr(k)["symbol"]
        groups.setdefault(symbol, []).append(k)

    # -------- 2) 对每个组单独检测逆序 --------
    for symbol, keys in groups.items():
        if len(keys) <= 1:            # 组内不足两条边，跳过
            continue

        # a) 键按数字下标升序（键名统一是 'bond_' + 数字）
        keys_sorted = sorted(keys, key=lambda k: int(k[5:]))

        # b) 提取对应 value 的数字下标
        values = [int(maps[k][5:]) for k in keys_sorted]

        # c) 如果 values 不是严格递增，则存在逆序
        if values != sorted(values):
            return True

    # 全部组都未发现逆序
    return False

def _node_match(node1, node2):
    # if the nodes are hyperedges, `atom_attr` determines the match
    if node1['bipartite'] == 'edge' and node2['bipartite'] == 'edge':
        return node1["attr_dict"]['symbol'] == node2["attr_dict"]['symbol']
    elif node1['bipartite'] == 'node' and node2['bipartite'] == 'node':
        # bond_symbol
        return node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']
    else:
        return False

def _easy_node_match(node1, node2):
    # if the nodes are hyperedges, `atom_attr` determines the match
    if node1['bipartite'] == 'edge' and node2['bipartite'] == 'edge':
        return node1["attr_dict"].get('symbol', None) == node2["attr_dict"].get('symbol', None)
    elif node1['bipartite'] == 'node' and node2['bipartite'] == 'node':
        # bond_symbol
        return node1['attr_dict'].get('ext_id', -1) == node2['attr_dict'].get('ext_id', -1)\
            and node1['attr_dict']['symbol'] == node2['attr_dict']['symbol']
    else:
        return False


def _node_match_prod_rule(node1, node2, ignore_order=False):
    # if the nodes are hyperedges, `atom_attr` determines the match
    if node1['bipartite'] != node2['bipartite']:
        return False

    # node1['bipartite'] == 'edge' and node2['bipartite'] == 'edge'
    if node1['bipartite'] == 'edge':
        return node1['symbol'] == node2['symbol']

    # node1['bipartite'] == 'node' and node2['bipartite'] == 'node'
    if ignore_order:
        return node1['symbol'] == node2['symbol']
        # if node1.get('intact_ring_bonds') and node2.get('intact_ring_bonds'):
        #     # 环中可能会出现单双键交替的结构，导致匹配的同构结构
        #     return True
        # else:
        #     return node1['symbol'] == node2['symbol']
    else:
        return node1['symbol'] == node2['symbol']\
            and node1.get('ext_id', -1) == node2.get('ext_id', -1)



def _edge_match(edge1, edge2, ignore_order=False):
    #return True
    if ignore_order:
        return True
    else:
        return edge1["order"] == edge2["order"]

# def masked_softmax(logit, mask):
#     ''' compute a probability distribution from logit
#
#     Parameters
#     ----------
#     logit : array-like, length D
#         each element indicates how each dimension is likely to be chosen
#         (the larger, the more likely)
#     mask : array-like, length D
#         each element is either 0 or 1.
#         if 0, the dimension is ignored
#         when computing the probability distribution.
#
#     Returns
#     -------
#     prob_dist : array, length D
#         probability distribution computed from logit.
#         if `mask[d] = 0`, `prob_dist[d] = 0`.
#     '''
#     if logit.shape != mask.shape:
#         raise ValueError('logit and mask must have the same shape')
#     c = np.max(logit)
#     exp_logit = np.exp(logit - c) * mask
#     sum_exp_logit = exp_logit @ mask
#     return exp_logit / sum_exp_logit



def masked_softmax(logit: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    '''
    Compute a masked softmax over logits using a binary mask (0 or 1).

    Parameters
    ----------
    logit : Tensor, shape (D,)
        each element indicates how each dimension is likely to be chosen
            (the larger, the more likely)
    mask : Tensor, shape (D,)
        Binary mask. Elements with 0 are excluded from softmax.

    Returns
    -------
    prob_dist : Tensor, shape (D,)
        probability distribution computed from logit.
            if `mask[d] = 0`, `prob_dist[d] = 0`.
    '''
    if logit.shape != mask.shape:
        raise ValueError('logit and mask must have the same shape')

    # Subtract max for numerical stability
    c = torch.max(logit)
    exp_logit = torch.exp(logit - c) * mask  # Zero out masked positions
    sum_exp_logit = torch.sum(exp_logit)
    return exp_logit / sum_exp_logit


