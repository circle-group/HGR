# foundation/data_utils/mol_dataset.py

import os, glob
import torch
import pandas as pd
import re
import copy, json
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from hgr.utils.file_utils import PathManager, dump_pickle, load_pickle
from hgr.foundation.data_utils.pretrain_mol_process import process_pretrain
from hgr.foundation.data_utils.finetune_mol_process import process_bbbp, process_tox21, process_toxcast, process_sider, process_clintox, process_hiv, process_bace
from hgr.foundation.data_utils.splitters import pretrain_random_split
from hgr.foundation.data_utils.splitters import scaffold_split #, random_split, random_scaffold_split  # 
from hgr.foundation.data_utils.grammar_tree import GrammarTreeProcessor
from hgr.grammar.grammar_generation import generate_grammar_from_smiles


import logging
logger = logging.getLogger(__name__)

import warnings; warnings.filterwarnings('ignore') ## warning 
# 关掉全局噪声日志（防止海量打印拖慢速度）
from rdkit import RDLogger
lg = RDLogger.logger()         # 拿到 RDKit 全局 logger 对象
# 注意：部分 RDKit 版本没有 `level` 属性，这里不保存旧等级，直接设置等级即可
lg.setLevel(RDLogger.CRITICAL) # 把日志等级设为 CRITICAL，只保留最严重的错误信息

# 真实规则 id 将整体 +OFFSET，避免与特殊符号冲突
from hgr.foundation.data_utils.mol_defs import PAD, BOS, EOS, MASK, OFFSET # = 0, 1, 2, 3, 4

# 数据预处理逻辑
DATA_PREPROCESS_MAP = {
    'zinc2m': process_pretrain,
    'zinc2m_test': process_pretrain,
    'zinc_2m_md': process_pretrain,
    'ringdiv': process_pretrain,
    # # 下面7个finetune中使用的
    'bbbp': process_bbbp,
    'tox21': process_tox21,
    'toxcast': process_toxcast,
    'sider': process_sider,
    'clintox': process_clintox,
    'hiv': process_hiv,
    'bace': process_bace,
}


def postprocess_rule(rule_seq_list, partitions):
    """
    将 (rule_seq_list, partitions) 后处理为 PyG Data 列表 rule_data_list。

    输入：
      - rule_seq_list: List[List[int]] 每个样本对应一个“规则序列”（不含 BOS/EOS，且规则 id 仍是原始空间）
      - partitions: List[Any] 每个样本对应 grammar lifting 得到的 partition  信息，用于构建树拓扑与 atom token 位置

    输出：
      - rule_data_list: List[torch_geometric.data.Data]
          每个样本一个 Data，字段包括：
            * rule_seq: LongTensor [L]，已加 BOS/EOS 且规则 id 整体 +OFFSET
            * rule_len: LongTensor [1]，长度 L（含 BOS/EOS）
            * parents: IntTensor [N_tokens]，每个 token 的父节点索引（用于后续生成树距离/关系矩阵）
            * depths:  IntTensor [N_tokens]，每个 token 的深度
            * atom_pos: IntTensor [num_atoms]，atom_idx -> rule_seq 中 atom token 的位置
                       注意：这里的 atom_pos 使用 offset=1（因为 rule_seq[0] 是 BOS）
    """
    num_samples = len(rule_seq_list)
    if num_samples != len(partitions):
        raise AssertionError("rule_seq_list and partitions length mismatch")

    rule_data_list = []
    for idx, (each_seq, each_partition) in tqdm(enumerate(zip(rule_seq_list, partitions)),
                                         total=num_samples, desc="Collating grammar", dynamic_ncols=True):
        # A) 构建 rule_seq（用于模型的 token embedding）
        # 约定：BOS/EOS/MASK/PAD 等特殊符号占用低位 id；因此真实规则 id 要整体 +OFFSET，避免与特殊符号冲突。
        ids = [BOS] + [int(t) + OFFSET for t in each_seq] + [EOS]  

        # B) 由 partition 构建 grammar tree（parents / depths）
        # parents/depths 的长度是 N_tokens（不含 BOS/EOS）, 后面会交由 dataloader 的 collate 函数中动态构建层次关系树
        parents, depths = GrammarTreeProcessor.build_grammar_tree(each_partition)

        # C) 构建 atom_pos（atom_idx -> token position）
        # atom_pos 表示“第 i 个原子对应的 atom token 在 rule_seq 中的位置”
        # 因为 rule_seq[0] 是 BOS，所以 token 的真实位置整体要 offset=1
        # 这样 atom_pos 才能直接用于定位 rule_seq 中被 mask 的 token。
        atom_pos = GrammarTreeProcessor.build_atom_pos(each_partition, offset=1)

        # D) 打包成 PyG Data
        d = Data(
            rule_seq=torch.tensor(ids, dtype=torch.long),
            rule_len=torch.tensor([len(ids)], dtype=torch.long), # 含 BOS/EOS 的长度
            parents=torch.tensor(parents, dtype=torch.int16), # 存储树的父节点
            depths=torch.tensor(depths, dtype=torch.int16), # 存储树的深度
            atom_pos=torch.tensor(atom_pos, dtype=torch.int16), # 存储 rule_seq 中 atom token 的位置
        )

        rule_data_list.append(d)

    return rule_data_list

class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 config,
                 mode, # pretrain 
                 modalities, # graph, grammar
                 pretexts=None, # node, conf, subgraph, descriptors
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.root = root
        self.mode = mode
        self.modalities = modalities
        self.pretexts = pretexts
        self.config = config
        self.data_name = config.data.name.lower()
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        assert self.data_name  in DATA_PREPROCESS_MAP, f"Invalid dataset name: {self.data_name }"
        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)

        if empty:
            return

        # # 1) 加载主图数据
        # Dataset cache stores PyG Data objects, not plain tensor weights.
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        # 2) 加载 SMILES 数据
        smi_path = os.path.join(self.processed_dir, 'smiles.csv')
        if os.path.exists(smi_path):
            # 方法 A: 使用 pandas 读取 (如果你确定是标准 CSV 且 pandas 已导入)
            # self._smiles = pd.read_csv(smi_path, header=None)[0].tolist()
            
            # 方法 B: 使用原生 IO (更快，更轻量，且完全等价于你之前的 SmilesStore)
            with open(smi_path, "r", encoding="utf-8") as f:
                self._smiles = [line.rstrip("\n") for line in f]
        else:
            # 建议加上 else 处理，防止后续 NoneType 错误
            self._smiles = [] 
            logger.warning(f"SMILES file not found at {smi_path}")


        if 'grammar' in self.modalities:
            """
            文件格式说明：
            - grammar_RSG{num_rules}_{data_name}(tag).pklz 存储原始规则数据
            - rules_RSG{num_rules}_{data_name}(tag).pt 存储PyG用的规则数据, 包含 (rule_data, rule_slices, rule_vocab_size)
            - 其中 tag 在pretrain时为exp_time, 在finetune时为pretrain_grammar_tag
            """
            gcfg = self.config.grammar
            force_rebuild = getattr(gcfg, "force_rebuild_rules", False) # 支持“强制重建”：

            # 2. pretrain时未指定grammar_path -> 重新生成grammar
            if self.mode == 'pretrain':
                if getattr(self.config.grammar, "grammar_path", None) is None or force_rebuild:
                    self.process_grammar()
                elif not os.path.exists(self.config.grammar.grammar_path):
                    raise FileNotFoundError(f"Grammar file not found at {self.config.grammar.grammar_path}")
                else:
                    logger.info(f"Loaded pretraingrammar from {self.config.grammar.grammar_path}")

            elif self.mode == 'finetune':  
                if getattr(self.config.grammar, "grammar_path", None) is None:
                    # finetune时需要提供pretrain_grammar_path
                    assert getattr(self.config.grammar, "pretrain_grammar_path", None) is not None, "pretrain_grammar_path is required for finetune mode"
                    # 提取pretrain grammar的type{num_rule}_{data_name}部分
                    lifting_type = self.config.grammar.type
                    grammar_name = os.path.splitext(os.path.basename(self.config.grammar.pretrain_grammar_path))[0] # 提取grammar_name
                    self.pretrain_grammar_tag = re.search(rf"grammar_({lifting_type}[^()]+)(?=\(|$)",  grammar_name).group(1) 
                    # 在processed_dir下查找是否存在对应的finetune grammar
                    pattern =  os.path.join(self.processed_dir, f"grammar_{lifting_type}*_{self.data_name}({self.pretrain_grammar_tag}).pklz")
                    candidates = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
                    if len(candidates) <= 0 or force_rebuild: # 支持“强制重建”：
                        self.process_grammar()
                    else:
                        self.config.grammar.grammar_path = candidates[0]
                else:
                    logger.info(f"Loaded finetune grammar from {self.config.grammar.grammar_path}")

                
            
            # 3. 加载独立存储的“规则序列”数据 (rules_*.pt)，包含 (rule_data, rule_slices, rule_vocab_size)
            if getattr(self.config.grammar, "rules_path", None) is None:
                # 默认collated rules path 必须和grammar path只有前缀不同，否则会报错
                dirname, filename = os.path.split(self.config.grammar.grammar_path)
                _rules_data_path = filename.replace("grammar_", "rules_collated_").replace(".pklz", ".pt")
                self.config.grammar.rules_path = os.path.join(dirname, _rules_data_path)
            self._rules_data, self._rules_slices, self.rule_vocab_size = torch.load(
                self.config.grammar.rules_path, map_location='cpu', weights_only=False)


        
        # 根据mode参数动态绑定get函数, pretrain模式下动态绑定fingerprint
        if mode == 'pretrain':
            self.get = self._get_graph_rule_smi
            assert self._smiles is not None

            fps, fp_names = [], []
            for k in config.tasks.subgraph.names:
                if k not in self.data:
                    raise KeyError(f"Requested fingerprint '{k}' not found in dataset.data")
                fps.append(self.data[k])
                fp_names.append(k)

            fingerprint = torch.cat(fps, dim=1) if len(fps) > 1 else fps[0]
            self.data['fingerprint'] = fingerprint
            self.slices['fingerprint'] = self.slices[fp_names[0]]  # 切片沿用任一源字段的切片（图级向量相同切法）

            # 节省内存：删除原来的 maccs/morgan
            for k in fp_names:
                del self.data[k]
                self.slices.pop(k, None)    

        elif mode == 'finetune':
            self.get = self._get_graph_and_rule
        else:
            raise ValueError(f"Unsupported mode: {mode}. Must be 'pretrain' or 'finetune'")

    @staticmethod
    def _slice_from_inmemory(data_source, slices_source, idx) -> Data:
        """
        从 PyG InMemoryDataset 风格的 (data, slices) 中切出第 idx 个样本。
        - data_source[key] 通常是 Tensor
        - slices_source[key] 通常是 1D LongTensor 指针数组
        """
        out = Data()
        for key, item in data_source.items():
            if key == 'smiles': 
                continue
            s = slices_source[key]
            sl = [slice(None)] * item.dim()
            sl[out.__cat_dim__(key, item)] = slice(s[idx], s[idx + 1])
            out[key] = item[sl]
        return out

    def _get_graph_and_rule(self, idx: int) -> Data:
        # Step 1. slice graph data
        d = self._slice_from_inmemory(self.data, self.slices, idx)

        if 'grammar' in self.modalities:
            # Step 2. slice grammar data
            r = self._slice_from_inmemory(self._rules_data, self._rules_slices, idx)
            d.rule_seq = r.rule_seq # 1D LongTensor, 含 BOS/EOS，已 +OFFSET
            d.rule_len = r.rule_len # 1D LongTensor

            # 1D 拓扑信息，用于后续在 DataLoader 中生成矩阵
            d.parents = r.parents     # (L,)
            d.depths  = r.depths      # (L,)
            d.atom_pos = r.atom_pos   # (L,)

        return d

    def _get_graph_rule_smi(self, idx: int) -> Data:
        d = self._get_graph_and_rule(idx)
        d['smiles'] = self._smiles[idx]
        return d


    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return f'{self.data_name}_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. No download allowed')


    def process(self):
        data_cfg = self.config.data
        num_workers = getattr(data_cfg, "num_workers", 64)
        chunksize = getattr(data_cfg, "chunksize", 4)
        shard_size = getattr(data_cfg, "shard_size", 10000)

        process_fn = DATA_PREPROCESS_MAP[self.data_name]
        print(f"Processing {self.data_name } ({num_workers} workers, {chunksize} chunksize, {shard_size} shard_size)")
        

        # Core processing
        if self.mode == 'finetune':
            data_list, smiles_list = process_fn(self.raw_paths[0])
            # write data_smiles_list in processed paths
            out_smiles_path = os.path.join(self.processed_dir, 'smiles.csv')
            data_smiles_series = pd.Series(smiles_list)
            data_smiles_series.to_csv(out_smiles_path, index=False, header=False)
            logger.info(f"Saved {len(smiles_list)} SMILES to {out_smiles_path}")
        elif self.mode == 'pretrain':
            raw_path = os.path.join(self.raw_dir, f"{self.data_name}_property.csv")
            # 下面的 process_fn中已经写入了out_smiles_path = os.path.join(self.processed_dir, 'smiles.csv')
            data_list = process_fn(raw_path, num_workers=num_workers, chunksize=chunksize, shard_size=shard_size)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        
        if self.mode == 'pretrain':
            ########## descriptors normlization ##########        
            mean = torch.mean(data['descriptors_ss'], dim=0)
            std = torch.std(data['descriptors_ss'], dim=0)
            std = torch.where(std == 0, torch.ones_like(std), std) # 避免除零错误
            data['descriptors_ss'] = (data['descriptors_ss'] - mean) / std
            # slices['descriptors_ss'] = slices['descriptors']

        torch.save((data, slices), self.processed_paths[0])

        # # 单独处理grammar部分
        # self.process_grammar()
    
    
    def process_grammar(self):
        """
        依赖 processed_dir 下已有的 smiles.csv，重新生成：
          - grammar_{type}{num_rule}_{data_name}({exp_time}).pklz
          - rules_{type}{num_rule}_{data_name}({exp_time}).pt
        """
        data_cfg = self.config.data
        gcfg = self.config.grammar
        

        processed_smiles_path = os.path.join(self.processed_dir, "smiles.csv")
        if not os.path.exists(processed_smiles_path):
            raise FileNotFoundError( f"smiles.csv not found at {processed_smiles_path}.  Cannot rebuild rules without smiles." )

        lifting_type = gcfg.type
        vocab_path = gcfg.vocab_path
        if not os.path.isabs(vocab_path):
            vocab_path = os.path.join(PathManager.DATA_DIR, vocab_path)

        pretrain_grammar_path = getattr(gcfg, "pretrain_grammar_path", None)
        if pretrain_grammar_path is not None:
            assert self.mode == 'finetune', "Only support finetune mode to load pretrain grammar"
            grammar = load_pickle(pretrain_grammar_path, verbose=False)
            logger.info(f"Loaded pretrain grammar from {os.path.abspath(pretrain_grammar_path)}")
        else:
            grammar = None
            assert self.mode == 'pretrain', "Only support pretrain mode for generating new grammar"
            logger.info(f"No grammar path provided, generating new grammar")

        # 1) 生成 grammar, rule_seq, partitions
        num_workers = getattr(data_cfg, "grammar_workers", 16)
        print(f"Generating grammar from {processed_smiles_path} ({num_workers} workers)")
        grammar, rule_seq_list, partitions = generate_grammar_from_smiles(
            processed_smiles_path,
            vocab_path,
            lifting_type,
            grammar=grammar,
            preserve_anchor_order=False,
            num_processes=num_workers,
            chunksize = getattr(data_cfg, "grammar_chunksize", 5),
            cache_size = getattr(data_cfg, "cache_size", 50000)
        ) # 上面代码得到的partitions就对应postprocess_rule的输入 

        if self.mode == 'pretrain':
            tag = self.config.exp_time
        elif self.mode == 'finetune':
            tag = self.pretrain_grammar_tag
        suffix = f"{lifting_type}{grammar.num_prod_rule}_{self.data_name}({tag})"

        # suffix = "RSG6583_ringdiv(251012-16:1209)"
        # suffix = 'RSG825_zinc2m(251011-20:4658)'


        grammar_path = os.path.join(self.processed_dir, f"grammar_{suffix}.pklz")
        rule_path = os.path.join(self.processed_dir, f"rules_{suffix}.pklz")
        partition_path = os.path.join(self.processed_dir, f"partitions_{suffix}.pklz")
        dump_pickle(grammar_path, grammar)
        dump_pickle(rule_path, rule_seq_list)
        dump_pickle(partition_path, partitions)
        logger.info(f"Saved grammar to: {os.path.abspath(grammar_path)}")
        logger.info(f"Saved rule_seq_list to: {os.path.abspath(rule_path)}")
        logger.info(f"Saved partitions to: {os.path.abspath(partition_path)}")

        # grammar = load_pickle(grammar_path)
        # rule_seq_list = load_pickle(rule_path)
        # partitions = load_pickle(partition_path)

        self.config.grammar.grammar_path = grammar_path

        rule_data_list = postprocess_rule(rule_seq_list, partitions) #发现单线程反而快奇怪, 因此这边没用多线程

        rule_data, rule_slices = self.collate(rule_data_list)
        rule_vocab_size = int(grammar.num_prod_rule + OFFSET) # EOS/BOS/MASK/PAD 占了 [0,1,2,3]；真实规则从 4 开始

        rules_collated_pt = os.path.join(self.processed_dir, f"rules_collated_{suffix}.pt")
        torch.save((rule_data, rule_slices, rule_vocab_size), rules_collated_pt)
        logger.info(f"Saved collated rules to: {rules_collated_pt} (vocab_size={rule_vocab_size})")

        



class JointMoleculeDataset(MoleculeDataset):
    """把两个 MoleculeDataset 合成为一个 InMemoryDataset 风格对象。"""
    def __init__(self, ds_a, ds_b, root=None):
        # 注意：这里调用 InMemoryDataset 的 init，绕过 MoleculeDataset.__init__ 的加载逻辑
        InMemoryDataset.__init__(self, root=root)

        # 0. 属性复制 (为了让基类的 get 方法能工作)
        self.modalities = getattr(ds_a, "modalities", []) # 确保有这个属性
        self.mode = getattr(ds_a, "mode", None)
        self.pretexts   = getattr(ds_a, "pretexts", None)
        self.data_name = f"{getattr(ds_a, 'data_name', 'A')}+{getattr(ds_b, 'data_name', 'B')}"

        # 1) 合并 graph 部分
        self.data, self.slices = self._concat_inmemory_tensors(ds_a.data, ds_a.slices, ds_b.data, ds_b.slices)

        # 2) 合并 smiles（简化为列表方式访问）
        self._smiles = None
        # 如果两侧都有 SmilesStore 内存列表，可直接拼接
        if ds_a._smiles is not None and ds_b._smiles is not None:
            self._smiles = ds_a._smiles + ds_b._smiles
        else:
            logger.warning("One of the datasets is missing SMILES, joint SMILES disabled.")

        # 3) grammar：规则数据也拼接
        if hasattr(ds_a, "_rules_data") and hasattr(ds_b, "_rules_data"):
            # 新逻辑：直接拼接，并把词表设为两者的最大值（稍后会在上层用 merged vocab 覆盖）
            self._rules_data, self._rules_slices = self._concat_inmemory_tensors(
                ds_a._rules_data, ds_a._rules_slices,
                ds_b._rules_data, ds_b._rules_slices
            )
            self.rule_vocab_size = int(max(getattr(ds_a, "rule_vocab_size", 0), getattr(ds_b, "rule_vocab_size", 0)))


        if not isinstance(self.data, Data):
            self.data = Data(**self.data)
        
        if self.mode == 'pretrain':
            self.get = self._get_graph_rule_smi
        elif self.mode == 'finetune':
            self.get = self._get_graph_and_rule
        else:
            raise ValueError(f"Unsupported mode: {self.mode}. Must be 'pretrain' or 'finetune'")

    # ===== 工具：按 InMemoryDataset 的规则拼接两份 data/slices =====
    @staticmethod
    def _concat_inmemory_tensors(data_a, slices_a, data_b, slices_b):
        """把两份 InMemoryDataset 的 (data, slices) 在样本维度上直接拼接。
        假设两侧 keys 一致，dtype/shape（除样本维）一致。"""
        out_data, out_slices = {}, {}
        keys = list(data_a.keys())
        dummy = Data()
        for key in keys:
            ta, tb = data_a[key], data_b[key]
            sa, sb = slices_a[key], slices_b[key]
            # 关键：按 PyG 的规则确定拼接维度
            cat_dim = dummy.__cat_dim__(key, ta)
            out_data[key] = torch.cat([ta, tb], dim=cat_dim)
            # 切片指针拼接：b 侧整体偏移 a 侧的“末尾指针值”
            offset = sa[-1].item()
            out_slices[key] = torch.cat([sa, sb[1:] + offset], dim=0)  # 跳过 sb[0]==0
        return out_data, out_slices

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass


def load_joint_dataset(config, modalities, pretexts):
    """
    联合模式：
      - config.data.name: 'base+other'
      - config.data.dir: [merge_dir, base_dir, other_dir]
      - manifest（位于 merge_dir）提供：
          files.grammar_merged_pklz  (相对于 merge_dir)
          files.rules_other_rel      (相对于 merge_dir)
          files.rules_base_rel       (相对于 base_dir)  # 由 merge_corpora 写入
    """
    data_name = config.data.name.lower()
    parts = data_name.split('+')
    assert len(parts) == 2, "Joint mode expects data.name like 'base+other'"
    base_name, other_name = parts[0].strip(), parts[1].strip()

    if not isinstance(config.data.dir, (list, tuple)) or len(config.data.dir) < 3:
        raise ValueError("Joint mode requires data.dir as [merge_dir, base_dir, other_dir].")
    merge_dir, base_dir, other_dir = config.data.dir[0], config.data.dir[1], config.data.dir[2]

    manifest_path = os.path.join(merge_dir, "manifest_grammar_merged.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("Manifest not found at %s" % manifest_path)
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    grammar_rel      = manifest["files"]["grammar_merged_pklz"]
    rules_other_rel  = manifest["files"]["rules_other_rel"]
    rules_base_rel   = manifest["files"].get("rules_base_rel", "")

    merged_grammar_path   = os.path.join(merge_dir, grammar_rel)
    merged_rules_other_pt = os.path.join(merge_dir, rules_other_rel)
    if not os.path.exists(merged_grammar_path):
        raise FileNotFoundError("Merged grammar missing: %s" % merged_grammar_path)
    if not os.path.exists(merged_rules_other_pt):
        raise FileNotFoundError("Merged other rules missing: %s" % merged_rules_other_pt)

    if not rules_base_rel:
        raise ValueError("Manifest missing 'rules_base_rel'. Please re-run merge_corpora with --rules_base_rel.")
    base_rules_pt = os.path.join(base_dir, rules_base_rel)
    if not os.path.exists(base_rules_pt):
        raise FileNotFoundError("Base rules (from manifest) not found: %s" % base_rules_pt)

    # base 用 merged grammar + base 原 rules
    base_cfg = copy.deepcopy(config)
    base_cfg.data.name = base_name
    base_cfg.data.dir  = base_dir
    base_cfg.grammar.grammar_path = merged_grammar_path
    base_cfg.grammar.rules_path   = base_rules_pt

    # other 用 merged grammar + merged rules
    other_cfg = copy.deepcopy(config)
    other_cfg.data.name = other_name
    other_cfg.data.dir  = other_dir
    other_cfg.grammar.grammar_path = merged_grammar_path
    other_cfg.grammar.rules_path   = merged_rules_other_pt

    ds_base  = MoleculeDataset(root=base_cfg.data.dir, mode='pretrain',
                               modalities=modalities, pretexts=pretexts, config=base_cfg)
    ds_other = MoleculeDataset(root=other_cfg.data.dir, mode='pretrain',
                               modalities=modalities, pretexts=pretexts, config=other_cfg)
    logger.info('-'*30 + f' [ds_base] ' + '-'*30)
    logger.info(ds_base)
    logger.info(ds_base.data)
    logger.info('-'*30 + f' [ds_other] ' + '-'*30)
    logger.info(ds_other)
    logger.info(ds_other.data)

    # “无缝合并”为一个 InMemoryDataset 风格数据集
    ds_merged = JointMoleculeDataset(ds_base, ds_other, root=merge_dir)
    return ds_merged



# ===== 简洁的加载封装 =====
def load_pretrain_dataset(config, modalities, pretexts, device):
    data_name = config.data.name.lower()

    if '+' not in data_name:
        ds = MoleculeDataset(
            root=getattr(config.data, "dir", None) or PathManager.DATA_DIR,
            mode='pretrain',
            modalities=modalities,
            pretexts=pretexts,
            config=config)
    else:
        ds = load_joint_dataset(config, modalities, pretexts)

    print('-'*30 + f' [{data_name}] ' + '-'*30)
    print(f'modalities: {modalities}')
    print(f'pretexts: {pretexts}')
    print(ds)
    print(ds.data)
    if 'grammar' in modalities:
        print(ds._rules_data)

    # split train/valid
    train_dataset, val_dataset = pretrain_random_split(ds, null_value=0, frac_train=0.9, frac_valid=0.1)
    print(f'train: {train_dataset}, valid: {val_dataset}')


    loader_kwargs =dict(pretexts=pretexts,
                        modalities=modalities,
                        max_path_distance=config.train.max_path_distance,
                        batch_size=config.train.batch_size, 
                        num_workers=config.train.num_workers_loader, 
                        persistent_workers = True if config.train.num_workers_loader > 0 else False,
                        mask_rate=config.train.mask_rate,
                        mask_edge=config.train.mask_edge,
                        pin_memory=(device.type == 'cuda'),
                        )

    from hgr.foundation.data_utils.dataloader import PretrainDataLoader
    train_loader = PretrainDataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader   = PretrainDataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader




def load_finetune_dataset(args):
    """一次性加载数据与 DataLoader；只受 data_seed 与 split 策略影响。"""
    data_root = getattr(args.data, "dir", None) or PathManager.DATA_DIR
    dataset = MoleculeDataset(root=data_root, config=args,
                            mode='finetune',  modalities=args.modalities)

    if args.train.split in ["scaffold", "random_scaffold"]:
        file_path = f"{data_root}/processed/smiles.csv"
        smiles_list = pd.read_csv(file_path, header=None)[0].tolist()
        print(f"Loaded {len(smiles_list)} SMILES from {os.path.abspath(file_path)}")

    if args.train.split == "scaffold":
        train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1 )
    else:
        raise ValueError("Invalid split option.")

    print('-'*30 + f'[dataset {args.data.name}]' + '-'*30)
    print(dataset)
    print('Train dataset example:', train_dataset[0])

    return train_dataset, valid_dataset, test_dataset



if __name__ == '__main__':
    # dataset = MoleculeDataset(root='datasets/dataset_conf/zinc_2m_MD', dataset='zinc_2m_MD', mode='pretrain')
    pass
