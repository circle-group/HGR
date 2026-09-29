# lifting/complex_lifting.py

import os
import sys
import heapq
import json
import logging
from rdkit import Chem
from collections import defaultdict, OrderedDict

from hgr.grammar.lifting.chem_utils import smi2mol, mol2smi, get_submol
from hgr.utils.debug_utils import Timer


logger = logging.getLogger(__name__)


class _LRUDict(OrderedDict):
    """
    Least-Recently-Used dictionary (based on ``collections.OrderedDict``). Both
    insertion & lookup are amortized O(1). When the size exceeds ``capacity``
    the oldest item (FIFO order) will be evicted automatically.

    LRU 字典：插入 / 查询 O(1)，超容量后淘汰最旧元素
    """
    def __init__(self, capacity=4096):
        super().__init__()
        self.capacity = capacity

    def __getitem__(self, key):
        """Get item and move it to the end to mark it as recently used."""
        val = super().__getitem__(key)
        try:  # 若在读-写嵌套期间被弹出，忽略即可
            self.move_to_end(key, last=True) # 访问后移动到尾部
        except KeyError:
            pass
        return val

    def __setitem__(self, key, val):
        # 更新现有键需先删除再插入，保证顺序 / Update in‑place keeps LRU order
        if key in self:
            del self[key]
        super().__setitem__(key, val) # 更新 -> 尾部
        # 超容量 → 弹出最旧元素 / Evict oldest when over capacity
        if len(self) > self.capacity:
            super().popitem(last=False)

    def pop(self, key, default=None):
        # 与内置 dict.pop 行为一致 / Exact same semantics as dict.pop
        return super().pop(key, default)



class BaseLifter:
    """公共的子图合并与歧义解决逻辑。"""

    # 使用 __slots__ 减少实例内存占用 / Reduce per‑instance memory footprint
    __slots__ = (
        'vocab',  # motif → frequency 词表 / vocabulary
        'kekulize',  # 是否对 SMILES 做 kekulization / kekulize flag
        '_neighbors',  # Cache neighbor indices for each atom / 缓存原子邻居信息
        '_freq_cache',  # 子图频次缓存 / sub‑graph frequency cache
        '_bound_map',  # 子图边界原子缓存 / boundary atom cache
        'lru_size'  # LRU 容量 / LRU capacity
    )

    def __init__(self, vocab_path, lru_size=None):
        """加载 motif 词表，并根据 ``lru_size`` 决定是否启用 LRU 缓存。

        * vocab_path  ——  两或三列 TSV 文件；第一行为 JSON 配置（是否 kekulize）。
        * lru_size    ——  ``None`` 或 0 表示关闭缓存；> 0 时启用 ``_LRUDict``。
        """
        self.lru_size = lru_size
        # logger.info(f"Load motif dict from {vocab_path}")
        with open(vocab_path, 'r', encoding='utf-8') as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]
        config = json.loads(lines[0])
        self.kekulize = config.get('kekulize', False)


        # smi -> frequency
        cols = len(lines[1].split('\t'))
        if cols == 2:
            pairs = (ln.split('\t') for ln in lines[1:])
            self.vocab = {smi: int(freq) for smi, freq in pairs}
        elif cols == 3:
            trip = (ln.split('\t') for ln in lines[1:])
            self.vocab = {smi: int(freq) for _, smi, freq in trip}
        else:
            raise ValueError("Unsupported vocabulary format; expected 2 or 3 columns.")

        #  cache when lru_size is not None
        self._freq_cache = _LRUDict(capacity=lru_size) if self.lru_size else {} # 缓存子图出现的频数
        self._bound_map = _LRUDict(capacity=lru_size) if self.lru_size else {} # 缓存子图对应的边界原子（即子图中连接到外边的原子）

    def _get_subg_freq(self, mol, subg):
        """ 给定 RDKit Mol 与一个原子集合，返回其 motif 词表频次。

        * **流程 / Workflow**
          1. 若 subg (atom_set) 已缓存则直接返回（O(1)）;
          2. 根据 atom_set 构造子分子 -> 生成该子分子 SMILES;
          3. 在 self.vocab 中查找频次，若不存在则按 –len(atom_set) 处罚;
          4. 写入缓存 self.freq_cache 后返回。
        """
        # If atom_set already in cache, return cached frequency
        if subg in self._freq_cache:
            return self._freq_cache[subg]

        # 新子图：用 get_submol -> mol2smi
        smi = mol2smi(get_submol(mol, list(subg), kekulize=self.kekulize))
        freq = self.vocab.get(smi, -len(subg))
        self._freq_cache[subg] = freq
        return freq

    def _get_boundary_atoms(self, subg):
        """
        给定一个子图 subg，返回它的“边界原子集合”：
        也就是在 subg 里，哪些与子图外部（subg 以外）至少连过一条键的原子。

        Given a subgraph `subg` (frozenset of atom indices), return the set of
        boundary atoms: atoms in `subg` that have at least one neighbor outside `subg`.
        """
        if subg not in self._bound_map:
            nbrs = self._neighbors
            self._bound_map[subg] = {a for a in subg if nbrs[a] - subg}
        return self._bound_map[subg]

    def _get_adj_subgraphs(self, subg, atom2subg):
        """
        Quick retrieval of sub‑graphs attached to *subg* through boundary atoms.
        Only boundary atoms are inspected for maximum efficiency.

        快速找出与 subg 外部相连的子图：
        - 只遍历 subg 的边界原子 boundary_atoms
        - 对每个边界原子 a, 枚举 a 的所有邻居 nei_atom
        - 通过 atom_to_subg[nei_atom] 得到其子图
        """
        adj_subgs = set()
        nbrs = self._neighbors
        for a in self._get_boundary_atoms(subg):                             # 遍历所有边界原子 / iterate boundary atoms
            for nei_atom in nbrs[a]:
                subg_of_nei = atom2subg[nei_atom]
                if subg_of_nei is not subg:
                    adj_subgs.add(subg_of_nei)
        return adj_subgs

    def _require_anchor_expansion(self, subg):
        """
        Determine if atoms have >2 external connections from >1 atom.
        判断子图是否具有多于 2 条来自多于 1 个原子的外部连接:
        - boundary_atoms 为 atom_set 的边界原子集合。
        - contributors: 边界原子中实际连到外部的原子数量
        - ext_edges_count: 总的外部连接（键）数量
        如果 ext_edges_count > 2 且 contributors > 1，则返回 True，否则 False。
        """

        nbrs = self._neighbors
        ext_edges_count = 0 # 子图外部键总数 / total external bonds
        contributors = 0
        # 先算 contributors，再判断 ext_edges
        for a in self._get_boundary_atoms(subg):  # 只遍历边界原子集合 boundary_atoms，加速判断。
            outside_cnt = len(nbrs[a] - subg)
            if outside_cnt:
                contributors += 1
                ext_edges_count += outside_cnt
                if ext_edges_count > 2 and contributors > 1:
                    return True
        return False

    def _absorb_anchors(self, mol, subg, atom2subg, allowed_atoms=None):
        """
        If atom_set has external ambiguity, absorb adjacent groups via best-first search (BFS).
        吸收anchor外部相邻子图以解决多源链接歧义，优先吸收全局频次高的 motif。

        subg_partition: 当前分子的子图划分情况
        allowed为允许的吸收域，=None表示全部允许，若 allowed 非 None，则只能吸收 <= allowed。
        """

        # 若无歧义，直接返回
        if not self._require_anchor_expansion(subg):
            return subg

        tried = {subg}
        heap = []  # priority queue of (-frequency, frozenset(atom_indices))

        # 初始扩展：对每个外部邻接子图进行合并候选
        for nei_subg in self._get_adj_subgraphs(subg, atom2subg):
            cand = subg | nei_subg
            if (cand in tried) or (allowed_atoms and not cand <= allowed_atoms):
                continue
            heapq.heappush(heap, (-self._get_subg_freq(mol, cand), cand))
            tried.add(cand)

        # -- best‑first expansion loop / BFS 合并 ------------------------------
        while heap:
            _, cand = heapq.heappop(heap)

            # 若已解决歧义，返回该子图 / ambiguity resolved
            if not self._require_anchor_expansion(cand):
                return cand

            # 未解决 → 继续向外一层 / still ambiguous, expand further
            for nei_subg in self._get_adj_subgraphs(cand, atom2subg):
                new_can = cand | nei_subg
                if (new_can in tried) or (allowed_atoms and not new_can <= allowed_atoms):
                    continue
                tried.add(new_can)
                heapq.heappush(heap, (-self._get_subg_freq(mol, new_can), new_can))

        return None # 未能消除歧义 / failed to resolve



    def _phase_merge(self, mol, merge_seq, curr_parts, atom2subg,
                     allowed_atoms=None, resolve_anchor=True, allow_anchor_ambiguous=False):
        """
        通用的两子图合并框架；参数化是否使用 resolve、是否允许歧义。

        resolve_anchor = True时 allow_anchor_ambiguous不生效
        """
        cand = {}
        visited = set(curr_parts)
        ext_to_merged0 = defaultdict(list) # {_absort_anchors(merged0): merged0} 记录_absorb_anchors得到的子图 来自哪个merged0

        def push_candidate(merged0):
            # 1) 区域约束 / region constraint ---------------------------------
            if allowed_atoms and not merged0 <= allowed_atoms:
                return

            merged = merged0
            if resolve_anchor:
                merged = self._absorb_anchors(mol, merged0, atom2subg, allowed_atoms)
                if merged is None:
                    return
            elif not allow_anchor_ambiguous:
                if self._require_anchor_expansion(merged):
                    return
            ext_to_merged0[merged].append(merged0)
            if merged in visited or merged in cand:
                return
            cand[merged] = self._get_subg_freq(mol, merged)

        # ------------------------------------------------------------------
        # Step‑0: 生成首批候选 / bootstrap first‑level candidates -------------
        subg_idx = {sg: i for i, sg in enumerate(curr_parts)}
        for sgA in curr_parts:
            if allowed_atoms and sgA - allowed_atoms:
                continue
            for sgB in self._get_adj_subgraphs(sgA, atom2subg):
                # ensure each pair considered once (A idx < B idx)
                if (allowed_atoms and sgB - allowed_atoms) or (subg_idx[sgB] <= subg_idx[sgA]):
                    continue
                push_candidate(sgA | sgB)

        # ------------------------------------------------------------------
        # 主循环：每轮选择全局频次最高的一组非重叠合并 / main merge loop ------
        while cand:
            best_freq = max(cand.values()) # Find highest frequency
            # Collect all candidates with max frequency
            best_cands = [c for c, f in cand.items() if f == best_freq]
            # Select non-overlapping merges
            selected_merges = []
            for c in sorted(best_cands, key=lambda x:(len(x), sorted(x))):
                if all(c.isdisjoint(o) for o in selected_merges):
                    selected_merges.append(c)

            # Remove all subgraphs that intersect any selected merge 找出所有与 best_can 有交集的旧子图，将其移除
            removed_subgs = [sg for sg in curr_parts if any(not sg.isdisjoint(cand_) for cand_ in selected_merges)]
            removed_atoms = set()
            for sg in removed_subgs:
                curr_parts.remove(sg)
                removed_atoms |= sg
                self._bound_map.pop(sg, None)

            # Add each selected merge
            for best_cand in selected_merges:
                curr_parts.append(best_cand)
                merge_seq.append(sorted(best_cand))
                visited.add(best_cand)
                for atom_idx in best_cand:
                    atom2subg[atom_idx] = best_cand

            # Only newly formed sub‑graphs can generate *new* candidate merges.
            to_retry = set()
            for c in list(cand):
                # 若某个cand和removed_atoms有交集，则需要更新
                if not c.isdisjoint(removed_atoms):
                    to_retry |= set(ext_to_merged0[c])
                    ext_to_merged0.pop(c, None)
                    cand.pop(c, None)

            # 删除的candidate对应的merged0需要重新合并，
            for merged0 in to_retry:
                # 但是*新*的merged0中不应该有和已经合并的子图有交集，已经合并的子图必须是*新*merged0的子集（这种情况在后面单独考虑）。
                if not (merged0 & removed_atoms):
                    push_candidate(merged0)

            # —— 扫描新增子图的外部邻接生成下一轮候选 / generate next layer ----
            for sgA in selected_merges:
                for sgB in self._get_adj_subgraphs(sgA, atom2subg):
                    if sgB is sgA or (allowed_atoms and sgB - allowed_atoms):
                        continue
                    push_candidate(sgA | sgB)




class MotifLifter(BaseLifter):
    """
    Hierarchical subgraph tokenizer driven by a motif vocabulary.
    基于 motif 词表的分子子图层次化合并。

    MIG / MEG 策略实现。
    * MEG：全局最高频优先，不主动 *resolve*。
    * MIG：改进策略，通过 *anchor resolve* 来吸收多余的anchor。

    Workflow 描述：
      1. Start with each atom as its own subgraph; 将每个原子视为独立子图
      2. Build adjacency: cache atom neighbors; 构建分子中原子的邻居索引
      3. For each adjacent subgraph pair, merge candidate = A ∪ B;
         若存在“外部连接歧义”，通过 _absorb_anchors 扩展子图;
      4. Score each merged candidate by global motif frequency;
      5. Best-first merging: select highest-frequency candidate, update subgraph partition;
      6. Repeat until no further merges.
    """

    def __init__(self, vocab_path, lifting_type='MIG', lru_size=None):
        super().__init__(vocab_path, lru_size)
        self.lifting_type = lifting_type


    def lifting(self, mol):
        """
        Perform hierarchical subgraph merging on a molecule.
        Returns a list of atom-index lists representing merge steps.
        对分子进行层次化子图合并，返回合并顺序中的原子索引列表。
        """
        if isinstance(mol, str):
            mol = smi2mol(mol, self.kekulize)

        # 缓存原子邻居信息(Cache neighbor indices for each atom.)
        self._neighbors = {atom.GetIdx(): {nei.GetIdx() for nei in atom.GetNeighbors()} for atom in mol.GetAtoms()}
        self._freq_cache.clear()  # 缓存“原子组合 frozenset -> (SMILES, 频次)”结果，避免重复计算
        self._bound_map.clear()

        # initialize partitions. 初始化：每个原子作为独立子图
        N = mol.GetNumAtoms()
        cur_parts = [frozenset([i]) for i in range(N)]
        atom2subg = {i: cur_parts[i] for i in range(N)}
        merge_seq = [[i] for i in range(N)]


        # main merging phase -----------------------------
        if self.lifting_type == 'MIG':
            self._phase_merge(mol, merge_seq, cur_parts, atom2subg, resolve_anchor=True)
        elif self.lifting_type == 'MEG':
            self._phase_merge(mol, merge_seq, cur_parts, atom2subg, resolve_anchor=False, allow_anchor_ambiguous=True)
        else:
            raise ValueError('Unknown lifting_type.')

        return merge_seq





class RingLifter(BaseLifter):
    """RSG 策略实现。

    简述流程：
      1) 先在非环区执行合并；
      2) 各 ring‑system 内部（不跨系统）合并；
      3) 针对系统内各个环（ring level merge）；
      4) 将整个 ring‑system 视为整体做一次 anchor resolve；
      5) 剩余全图合并直至收敛。
    """

    def __init__(self, vocab_path, lru_size=None):
        super().__init__(vocab_path, lru_size)



    def _extract_ring_systems(self, mol):
        """Identify rings, ring systems

        * rings        —— list[set[int]]，单个最小环的原子索引集合。
        * ring_systems —— list[set[int]]，由互相共享原子的多个 rings 所组成的并集。
        """
        rings = [set(r) for r in Chem.GetSymmSSSR(mol)]
        n = len(rings)
        parent = list(range(n))

        # 并查集 / Union‑Find helpers --------------------------------------
        def find(i):
            """Path‑compressed root lookup."""
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a,b):
            pa,pb = find(a), find(b)
            if pa != pb:
                parent[pb] = pa

        # Any two rings sharing ≥1 atom belong to the same ring system. 任意两环共享原子则视为同一环系统
        for i in range(n):
            ri = rings[i]
            for j in range(i+1, n):
                if ri & rings[j]:
                    union(i,j)

        clusters = defaultdict(list)
        for idx in range(n):
            clusters[find(idx)].append(idx)

        ring_systems = []
        for sys_id, ring_ids in clusters.items():
            atm = set().union(*(rings[i] for i in ring_ids))
            ring_systems.append(atm)

        return rings, ring_systems

    def _ring_level_merge(self, mol, merge_seq, cur_parts, atom2sub, sys_atoms, rings_in_sys):
        # 对环系统内的子环尝试合并
        best, best_freq = None, None
        for ring_atoms in rings_in_sys:
            involved = {atom2sub[a] for a in ring_atoms}
            cand = frozenset().union(*involved)
            if (cand in cur_parts) or (not cand <= sys_atoms) or self._require_anchor_expansion(cand):
                continue
            f = self._get_subg_freq(mol, cand)
            if best is None or f > best_freq:
                best, best_freq = cand, f
        if best is None:
            return

        for sg in [sg for sg in cur_parts if not sg.isdisjoint(best)]:
            cur_parts.remove(sg)
            self._bound_map.pop(sg, None)

        cur_parts.append(best)
        merge_seq.append(sorted(best))
        for a in best:
            atom2sub[a] = best

    # -------------------------- Entry‑point ---------------------------
    def lifting(self, mol):

        if isinstance(mol, str):
            mol = smi2mol(mol, self.kekulize)

        # 缓存原子邻居信息(Cache neighbor indices for each atom.)
        self._neighbors = {atom.GetIdx(): {nei.GetIdx() for nei in atom.GetNeighbors()} for atom in mol.GetAtoms()}
        self._freq_cache.clear()  # 缓存“原子组合 frozenset -> (SMILES, 频次)”结果，避免重复计算
        self._bound_map.clear()

        # 初始化：每个原子作为独立子图
        N = mol.GetNumAtoms()
        cur_parts = [frozenset([i]) for i in range(N)]   # 当前的子图划分
        atom2subg = {i: cur_parts[i] for i in range(N)}  # 每个原子所属的子图
        merge_seq = [[i] for i in range(N)]              # 合并顺序


        # =========================== RSG 策略特有阶段  ===========================

        # -------- Phase‑0 : detect ring systems ---------------------------
        rings, ring_systems = self._extract_ring_systems(mol)
        ring_atoms = set().union(*ring_systems) if ring_systems else set()


        # ---- Phase-1 : non‑ring region. 非环系统区域 ----
        out_ring_atoms = set(range(N)) - ring_atoms
        if out_ring_atoms:
            self._phase_merge(mol, merge_seq, cur_parts, atom2subg,
                              allowed_atoms=out_ring_atoms, resolve_anchor=False, allow_anchor_ambiguous=False)
            # 上一步是新增的，增加后运行效率大幅度上升（貌似产生了rule num 也减少了）
            self._phase_merge(mol, merge_seq, cur_parts, atom2subg,
                              allowed_atoms=out_ring_atoms, resolve_anchor=True)


        # ---- Phase-2 : 将支链挂入唯一连接的环系统 --------------------
        # 对每个完全位于非环区域的子图 A，检查：
        #   - A 通过边界原子只与 *一个* 环系统 R 相连，
        #   - 且所有外部连接都指向同一个环系统中的同一个原子 x。
        # 满足时：把 A 与 x 所在子图合并，并把 A 的原子加入环系统 R。

        atom2ringsys = {a: idx for idx, sys_atoms in enumerate(ring_systems) for a in sys_atoms}
        out_ring_subgs = [sg for sg in cur_parts if sg.isdisjoint(ring_atoms)] # 只考虑完全位于非环系统的子图
        for sg in out_ring_subgs:
            # 记录sg连接到的环系统原子
            conn_ring_atoms = [nei
                               for a in self._get_boundary_atoms(sg)
                               for nei in self._neighbors[a]
                               if (nei not in sg) and (nei in ring_atoms)]
            unique_ring_atoms = set(conn_ring_atoms)
            if len(unique_ring_atoms) != 1:
                continue

            x = conn_ring_atoms[0]
            sys_idx = atom2ringsys[x]

            # 合并 sg 与 x 所在子图
            x_subg = atom2subg[x]
            merged = sg | x_subg

            # 更新分区
            for sub in (sg, x_subg):
                if sub in cur_parts:
                    cur_parts.remove(sub)
                    self._bound_map.pop(sub, None)
            cur_parts.append(merged)
            merge_seq.append(sorted(merged))
            for aidx in merged:
                atom2subg[aidx] = merged
                atom2ringsys[aidx] = sys_idx #把 sug A 的原子加入环系统 sys_idx。

            # 把 A 的原子加入环系统 R
            ring_systems[sys_idx] |= sg
            ring_atoms |= sg


        # ---- Phase-3 : ring-system 内部合并----
        for sys_atoms in ring_systems:
            rings_in_sys = [r for r in rings if r <= sys_atoms]
            while True:
                # ---- Phase-3a : 各 ring-system 内部不用resolve_anchor进行一轮合并----
                self._phase_merge(mol, merge_seq, cur_parts, atom2subg,
                                  allowed_atoms=sys_atoms, resolve_anchor=False, allow_anchor_ambiguous=False)
                # ---- Phase-3b : ring‑level incremental merge. 对每个系统内的子环尝试合并 ----
                before = len(merge_seq)
                self._ring_level_merge(mol, merge_seq, cur_parts, atom2subg, sys_atoms, rings_in_sys)
                if len(merge_seq) == before: # 若无变化，说明已无法合并，终止循环
                    break

        # ---- Phase-4: 在每个 ring-system 被整体合并之前，尽量多把无歧义结构合并进来。
        self._phase_merge(mol, merge_seq, cur_parts, atom2subg,
                          resolve_anchor=False, allow_anchor_ambiguous=False)

        # ---- Phase-5 : 把整个 ring-system 当整体，再吸收外部连接 -----------------
        merged_systems = []
        merged_idx = []
        # ---- Phase-5a : 把整个 ring-system 当整体
        # (如果直接在这一步中调用_absorb_anchors可能需要宽度遍历整个环系统，复杂度较高。因此先合并，在 Phase-5b 中再调用 _resolve_ambiguity。)
        for sys_atoms in ring_systems:
            subgraphs_in_sys = [sg for sg in cur_parts if sg <= sys_atoms]
            if not subgraphs_in_sys:  # 该系统已被别人吃掉
                continue

            merged = frozenset().union(*subgraphs_in_sys)
            did_union = len(subgraphs_in_sys)>1
            if did_union: # 真的需要把碎片合并
                for sg in subgraphs_in_sys:
                    cur_parts.remove(sg)
                    self._bound_map.pop(sg, None)
                for a in merged:
                    atom2subg[a] = merged
                cur_parts.append(merged)
                merged_systems.append(merged)
                merged_idx.append(did_union)

        # ---- Phase-5b : 对每个 ring-system 整体做一次 _resolve_ambiguity -------
        for merged_sys, _did_union in zip(merged_systems, merged_idx):
            if merged_sys not in cur_parts:  # 前面某个 ring-system 吞进来后，cur_part 里可能已不存在它
                continue
            final = self._absorb_anchors(mol, merged_sys, atom2subg, allowed_atoms=None)

            if final is None or final == merged_sys:
                # 虽然没有进一步扩张，但是 Phase-5a 中 union 过了
                if _did_union:
                    merge_seq.append(sorted(merged_sys))
                continue

            # ——— ring-system 长大了（可能拉进了别的 ring-system / 支链） ———
            for sg in [sg for sg in cur_parts if not sg.isdisjoint(final)]:
                cur_parts.remove(sg)
                self._bound_map.pop(sg, None)

            cur_parts.append(final)
            merge_seq.append(sorted(final))
            for a in final:
                atom2subg[a] = final


        # ----  Phase-6: global convergence. 全图合并
        self._phase_merge(mol, merge_seq, cur_parts, atom2subg, resolve_anchor=True)

        return merge_seq


class Lifter:
    """统一接口：MIG 或 RSG"""
    def __init__(self, vocab_path, lifting_type='MIG', lru_size=None):
        typ = lifting_type.upper()
        if typ in ['MIG', 'MEG']:
            self.impl = MotifLifter(vocab_path, typ, lru_size)
        elif typ == 'RSG':
            self.impl = RingLifter(vocab_path, lru_size)
        else:
            raise ValueError("lifting_type should be one of ['MIG', 'MEG', 'RSG']")

    def lifting(self, mol):
        return self.impl.lifting(mol)
