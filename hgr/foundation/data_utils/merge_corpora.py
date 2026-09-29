# foundation/data_utils/merge_corpora.py
# -*- coding: utf-8 -*-
"""
功能：将两个 ProductionRuleCorpus（均要求 preserve_anchor_order=False）合并：
  - 以 based_grammar（推荐 ringdiv）为基底，增量并入 other_grammar（推荐 zinc2m）
  - 读取 other 的 partitions，计算并生成包含 parents/depths/atom_pos 的完整规则数据
  - 直接输出 other 的 merged rules（离线重编码；不保存映射表）
  - 导出合并后的 grammar（仅一份，供联合模式两侧共享）
  - manifest 使用「相对路径」，以合并目录为根

使用建议：
python FM/data_utils/merge_corpora.py \
    --base_pklz  "/data2/yh1924/HGR/datasets/ringdiv/processed/grammar_RSG6583_ringdiv(251012-16:1209).pklz" \
    --other_pklz "/data2/yh1924/HGR/datasets/zinc2m/processed/grammar_RSG825_zinc2m(251011-20:4658).pklz" \
    --base_name ringdiv \
    --other_name zinc2m \
    --outdir /data2/yh1924/HGR/datasets/ringdiv_zinc2m

注意：该脚本要求 rules_*.pklz 和 partitions_*.pklz 与 grammar_*.pklz 位于同一目录，且命名规范一致。
"""
import sys, os

import json
import time
import copy
import argparse
import torch
from tqdm import tqdm
import re
from hgr.utils.file_utils import load_pickle, dump_pickle
from hgr.foundation.data_utils.grammar_tree import GrammarTreeProcessor 
from hgr.foundation.data_utils.mol_defs import PAD, BOS, EOS, MASK, OFFSET # OFFSET=4

def ensure_preserve_false(corpus):
    """
    合并逻辑建立在 ignore_order（find_rule_nosym）之上。只有不记录anchor的corpus才能合并。
    """
    fn = getattr(corpus, "find_rule", None)
    if fn is None:
        raise ValueError("Input corpus has no 'find_rule' method; unexpected class state.")
    if fn.__name__ != "find_rule_nosym":
        raise ValueError(
            "Input corpus was created with preserve_anchor_order=True. "
            "Please rebuild it with preserve_anchor_order=False before merging."
        )

def collate_rules_full(rule_dicts_list):
    """
    打包所有规则字段，与 mol_dataset.postprocess_rule 生成的格式一致。
    输入: list of dict, keys: rule_seq, rule_len, parents, depths, atom_pos
    """
    keys = rule_dicts_list[0].keys()
    rule_data = {}
    rule_slices = {}
    
    # 初始化 slices 指针
    for k in keys:
        rule_slices[k] = [0]
    
    # 临时列表用于 cat
    storage = {k: [] for k in keys}
    
    accumulators = {k: 0 for k in keys}
    
    for d in rule_dicts_list:
        for k in keys:
            tensor_item = d[k]
            storage[k].append(tensor_item)
            accumulators[k] += int(tensor_item.numel())
            rule_slices[k].append(accumulators[k])
            
    for k in keys:
        rule_data[k] = torch.cat(storage[k], dim=0)
        rule_slices[k] = torch.tensor(rule_slices[k], dtype=torch.long)
        
    return rule_data, rule_slices

def _try_guess_name_from_pklz(path):
    """
    尝试从 'grammar_{type}{num}_{name}(...).pklz' 中解析 name
    解析失败返回 None
    """
    fname = os.path.basename(path)
    m = re.search(r"grammar_[A-Za-z]+[0-9]+_([^()]+)\(", fname)
    return m.group(1) if m else None

def _load_sidecar_file(grammar_path, prefix_old, prefix_new):
    """
    辅助函数：根据 grammar_X.pklz 的路径推断 rules_X.pklz 或 partitions_X.pklz
    """
    dirname, fname = os.path.split(grammar_path)
    # 简单的字符串替换：grammar_ -> rules_ 或 partitions_
    if not fname.startswith(prefix_old):
        raise ValueError(f"Filename {fname} does not start with {prefix_old}")
    new_fname = fname.replace(prefix_old, prefix_new, 1)
    new_path = os.path.join(dirname, new_fname)
    
    if not os.path.exists(new_path):
        raise FileNotFoundError(f"Could not find sidecar file: {new_path} based on {grammar_path}")
    
    print(f"Loading sidecar: {new_fname}")
    obj = load_pickle(new_path, verbose=False)
    return obj

def merge_two_corpora(args):
    # 1. Load Base Grammar
    # 注意：新的 mol_dataset process_grammar 分开存储了，这里假设只加载 grammar 对象
    grammar_base = load_pickle(args.base_pklz, verbose=True)
    if isinstance(grammar_base, tuple): grammar_base = grammar_base[0] # 兼容旧格式
    print("Loaded base grammar with %d rules" % grammar_base.num_prod_rule)
    
    # 2. Load Other Grammar, Rules, and Partitions
    grammar_other = load_pickle(args.other_pklz, verbose=True)
    if isinstance(grammar_other, tuple): grammar_other = grammar_other[0]
    print("Loaded other grammar with %d rules" % grammar_other.num_prod_rule)
    
    # 自动查找并加载 rules 和 partitions
    rules_other_list = _load_sidecar_file(args.other_pklz, "grammar_", "rules_")
    partitions_other = _load_sidecar_file(args.other_pklz, "grammar_", "partitions_")
    
    assert len(rules_other_list) == len(partitions_other), "Rules and Partitions length mismatch for other dataset"

    ensure_preserve_false(grammar_base)
    ensure_preserve_false(grammar_other)
    
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    base_name, other_name = args.base_name, args.other_name
    base_size_before = grammar_base.num_prod_rule

    # ---------------------------- Step 1: Merge other rules into base ----------------------------
    approx_overlap = 0
    # 这里我们只需要 grammar_other 的 prod_rule_list
    for old_id, rule in tqdm(enumerate(grammar_other.prod_rule_list),
                             total=len(grammar_other.prod_rule_list),
                             desc="Merging grammar"):
        rule_copy = copy.deepcopy(rule)
        _, new_id = grammar_base.append(rule_copy, subg=None, fp=None)
        if new_id < base_size_before:
            approx_overlap += 1
            
    merged_grammar = grammar_base
    
    # ---------------------------- Step 2: Generate other's merged rules & topology ----------------------------
    # PAD, BOS, EOS, OFFSET = 0, 1, 2, 4 (Imported from mol_defs)
    old2new_cache = {}
    remapped_rule_dicts = []
    
    for old_seq, partition in tqdm(zip(rules_other_list, partitions_other), 
                                   total=len(rules_other_list), 
                                   desc="Processing"):
        # 2.1 Remap Rule IDs
        new_seq = []
        for old_id in old_seq:
            if old_id in old2new_cache:
                new_seq.append(old2new_cache[old_id])
                continue
            rule_obj = grammar_other.prod_rule_list[old_id]
            # 再次查找/添加以获取 ID (因为前面只跑了 loop 没存 map，或者为了保险)
            # 注意：merged_grammar 已经包含了所有规则，这里 append 会直接返回 index
            _, nid = merged_grammar.append(copy.deepcopy(rule_obj), subg=None, fp=None)
            old2new_cache[old_id] = int(nid)
            new_seq.append(int(nid))
            
        # 2.2 Construct full tensors (BOS/EOS, OFFSET)
        # 对应 mol_dataset.postprocess_rule 中的逻辑
        ids = [BOS] + [int(x) + OFFSET for x in new_seq] + [EOS]
        
        # 2.3 Build Tree Topology (parents, depths, atom_pos)
        # 这些只依赖 partition 结构，跟 rule ID 的具体数值无关，所以可以直接计算
        parents, depths = GrammarTreeProcessor.build_grammar_tree(partition)
        atom_pos = GrammarTreeProcessor.build_atom_pos(partition, offset=1) # offset=1 due to BOS
        
        # 2.4 Pack into dict
        d = {
            "rule_seq": torch.tensor(ids, dtype=torch.long),
            "rule_len": torch.tensor([len(ids)], dtype=torch.long),
            "parents": torch.tensor(parents, dtype=torch.int16),
            "depths": torch.tensor(depths, dtype=torch.int16),
            "atom_pos": torch.tensor(atom_pos, dtype=torch.int16)
        }
        remapped_rule_dicts.append(d)

    # ---------------------------- Step 3: Collate and Save ----------------------------
    rule_data, rule_slices = collate_rules_full(remapped_rule_dicts)
    rule_vocab_size = int(merged_grammar.num_prod_rule + OFFSET)
    
    # Save grammar
    # 保持兼容性：输出 (grammar, None) 或者是直接 grammar。
    # 原代码是 (merged_grammar, None)，为了保持一致性继续这样写，
    # 但新的 mol_dataset.load_pretrain_grammar 可能需要适配。
    # 既然是 merge，我们假设它是生成的“新”grammar。
    grammar_rel = os.path.join(f"grammar_{args.grammar_type}{merged_grammar.num_prod_rule}_{base_name}+{other_name}.pklz")
    grammar_abs = os.path.join(outdir, grammar_rel)
    os.makedirs(os.path.dirname(grammar_abs), exist_ok=True)
    dump_pickle(grammar_abs, merged_grammar) # 修改：只存 grammar 对象，与新 mol_dataset 对齐

    # Save other's merged rules
    rules_other_rel = os.path.join(f"rules_{other_name}_merged_ids.pt")
    rules_other_abs = os.path.join(outdir, rules_other_rel)
    os.makedirs(os.path.dirname(rules_other_abs), exist_ok=True)
    torch.save((rule_data, rule_slices, rule_vocab_size), rules_other_abs)

    # Calculate base rules path for manifest
    # 假设 base 的 collated rules 就在 base_dir/processed/ 下，且命名符合 generate_grammar_from_smiles 的输出
    # 这里只能基于 base_pklz 的文件名做替换推断
    grammar_base_filename = os.path.basename(args.base_pklz)
    # 尝试找到 base 的 rules_collated_*.pt
    # 规则：grammar_RSG6583_ringdiv(...).pklz -> rules_collated_RSG6583_ringdiv(...).pt
    rules_base_filename = grammar_base_filename.replace('grammar_', 'rules_collated_').replace('.pklz', '.pt')
    rules_base_rel = os.path.join('processed', rules_base_filename)
    
    manifest = {
        "base_name": base_name,
        "other_name": other_name,
        "grammar_type": args.grammar_type,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "preserve_anchor_order": False,
        "base_num_rules_before": int(base_size_before),
        "other_num_rules": int(grammar_other.num_prod_rule),
        "merged_num_rules": int(merged_grammar.num_prod_rule),
        "approx_overlap_from_other_into_base": int(approx_overlap),
        "files": {
            "grammar_merged_pklz": grammar_rel,  # relative to outdir
            "rules_other_rel": rules_other_rel,  # relative to outdir
            "rules_base_rel": rules_base_rel     # relative to base_dir
        },
        "notes": "Merged grammar. 'rules_other_rel' contains fully collated PyG data (seq, parents, depths, atom_pos)."
    }
    
    print(json.dumps(manifest, indent=2))
    with open(os.path.join(outdir, "manifest_grammar_merged.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    print("[OK] merged_num_rules =", merged_grammar.num_prod_rule)
    print("[OK] %s merged rules saved:" % other_name, rules_other_abs)
    print("[OK] manifest saved:", os.path.join(outdir, "manifest_grammar_merged.json"))

def main():
    parser = argparse.ArgumentParser(description="Merge ProductionRuleCorpus (generate full PyG data)")
    parser.add_argument("--base_pklz", required=True, help="Path to base grammar .pklz")
    parser.add_argument("--other_pklz", required=True, help="Path to other grammar .pklz (must have sibling rules_*.pklz and partitions_*.pklz)")
    parser.add_argument("--outdir", required=True, help="Output merged directory")
    parser.add_argument("--grammar_type", default="RSG", help="Grammar type tag in filenames")
    parser.add_argument("--base_name", default=None, help="Base dataset name")
    parser.add_argument("--other_name", default=None, help="Other dataset name")
    
    args = parser.parse_args()
    
    if args.base_name is None:
        args.base_name = _try_guess_name_from_pklz(args.base_pklz) or "base"
    if args.other_name is None:
        args.other_name = _try_guess_name_from_pklz(args.other_pklz) or "other"
        
    merge_two_corpora(args)

if __name__ == "__main__":
    main()