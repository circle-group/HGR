# grammar/reconstruct.py

# 加入下面这几句就能直接运行本文件了
# import os, sys
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.insert(0, project_root)

import logging
import contextlib
from tqdm import tqdm
from rdkit import Chem
from concurrent.futures import ProcessPoolExecutor

from hgr.utils.mol_utils import is_equal_mol
from hgr.grammar.smi import hg_to_mol


logger = logging.getLogger(__name__)

# Global grammar reference for worker processes
global GRAMMAR
GRAMMAR = None

def _init_worker(grammar):
    """
    初始化每个子进程时调用，将 grammar 存储为全局变量，避免在每次调用时重复传递大对象。
    """
    global GRAMMAR
    GRAMMAR = grammar

def _reconstruct_one(args):
    """
    对单个分子进行重构并对比结果。
    返回 (g_id, success_flag, org_smile, reconstructed_smiles)
    """
    g_id, prod_seq, org_smile = args


    reconstruct_hg, nt_edge_list = None, None
    for rule_id in reversed(prod_seq):
        rule = GRAMMAR.prod_rule_list[rule_id]
        reconstruct_hg, nt_edge_list = rule.apply_to_graph(reconstruct_hg, nt_edge_list)
    reconstructed_mol = hg_to_mol(reconstruct_hg)
    success = is_equal_mol(reconstructed_mol, org_smile)
    recon_smiles = None if success else Chem.MolToSmiles(reconstructed_mol, canonical=True)
    return g_id, success, org_smile, recon_smiles

def reconstruct_from_grammar(grammar, prod_rule_seq_list, org_smile_list, n_workers=1):
    """
    根据 n_workers 决定串行或并行重构分子并比对 SMILES。

    参数:
        grammar: ProductionRuleCorpus 对象，包含所有生产规则
        prod_rule_seq_list: 每个分子的规则索引序列列表
        org_smile_list: 对应的原始 SMILES 列表
        n_workers: 并行进程数；<=1 则使用串行模式，方便调试

    返回:
        reconstructed_smiles_list: 重构后 SMILES 的列表
    """
    total = len(prod_rule_seq_list)
    failure_count = 0

    # 准备参数迭代器
    args_iter = ((i, prod_rule_seq_list[i], org_smile_list[i]) for i in range(total))

    # 构造统一的 iterator 和上下文管理器
    if n_workers > 1:
        logger.info(f"Reconstruct from grammar in parallel mode with {n_workers} workers")
        pool_manager = ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker, initargs=(grammar,)
        )
        # 根据工作进程和总任务数调节 chunksize
        chunksize = min(max(1, total // (n_workers * 4)), 64)
        iterator = pool_manager.map(_reconstruct_one, args_iter, chunksize=chunksize)
    else:
        logger.info("Reconstruct from grammar in serial mode")
        _init_worker(grammar)
        pool_manager = contextlib.nullcontext()
        iterator = (_reconstruct_one(args) for args in args_iter)

    # 统一处理结果并显示进度
    with pool_manager, tqdm(total=total, desc="Reconstructing") as pbar:
        for g_id, success, org_smile, recon_smiles in iterator:
            if not success:
                failure_count += 1
                logger.info('='*10 + f" Mol {g_id+1} failed: {org_smile}.{recon_smiles} " + '='*10)
                pbar.set_postfix(failure=f"{failure_count}")
            pbar.update(1)


    failure_rate = failure_count / total if total else 0
    print(f"Failure rate: {failure_rate:.2%} ({failure_count}/{total})")
    return failure_rate
