# grammar.grammar_generation.py

import sys, os
import logging
import contextlib
import multiprocessing
import threading, queue 
from tqdm import tqdm
from multiprocessing import Pool

from hgr.utils.file_utils import stream_smiles
from hgr.utils.mol_utils import remove_salt_stereo
from hgr.grammar.lifting.complex_lifting import Lifter
from hgr.grammar.rule_corpus import ProductionRuleCorpus
from hgr.grammar.rule_generation import generate_rule
from hgr.grammar.chemutils import get_mol
from hgr.grammar.mol_graph import InputGraph, build_tree_with_dfs_sorting


logger = logging.getLogger(__name__)

# 全局变量：每个 worker 只需要加载一次 Lifter
_global_lifter = None


def _init_worker(vocab_path, lifting_type, cache_size):
    """
    Pool 的 initializer，在子进程中只会运行一次，用来加载全局的 Lifter 实例。
    """
    global _global_lifter
    _global_lifter = Lifter(vocab_path, lifting_type, cache_size)


def _process_smile_to_rules_worker(item):
    """
    **核心** 工作函数，它处理单个SMILES并完成整个流程：
    1. SMILES -> Mol 对象
    2. Lifting -> clusters
    3. build_tree_with_dfs_sorting -> new_clusters, init_subgraphs
    4. 构造 InputGraph
    5. 提取生产规则 (Production Rules)

    参数:
        item (tuple): 一个包含 (索引, 原始SMILES) 的元组, e.g., (0, 'CCO').

    返回:
        tuple: 成功时返回 (索引, (提取出的规则列表, 子图列表))。
               失败时返回 (索引, None)。
    """
    idx, raw_smile = item
    
    try:
        # --- 第1部分：源自 grammar_indata_process_parallel 的逻辑 ---
        # 1) SMILES 规范化
        # smile = get_smiles(get_mol(raw_smile))
        smile = remove_salt_stereo(raw_smile)
        mol = get_mol(smile)
        # 2) Lifting
        clusters = _global_lifter.lifting(smile)
        # return (idx, (None, None)) # 7500it/s (210k/30s)
        # 3) 建树
        new_clusters, init_subgraphs = build_tree_with_dfs_sorting(mol, clusters)
        motif_subgs = init_subgraphs.copy() # 拷贝，因为InputGraph中会改变init_subgraphs
        # return (idx, (None, None)) # 1500it/s (39k/30s)
        # 4) 构造 InputGraph
        input_g = InputGraph(mol, smile, init_subgraphs, new_clusters)
        # return (idx, (None, None)) # 1300it/s

        # 5) 为每个子图（motif）生成规则
        # rule_list = [generate_rule(input_g, motif) for motif in input_g.subgraphs]
        rule_fp_list = []
        while input_g.subgraphs:
            # 总是获取并处理 *当前* 列表中的第一个（即按DFS顺序排好序的、最小的）motif
            motif_to_process = input_g.subgraphs[0]
            rule = generate_rule(input_g, motif_to_process)       
            fp = ProductionRuleCorpus._fingerprint(rule)
            rule_fp_list.append((rule, fp))

        # 返回索引、规则和子图。子图是必需的，因为主进程需要它们来更新规则索引
        # 并最终为这条SMILES生成规则序列。
        return (idx, rule_fp_list, motif_subgs) # 60it/s (1.5k/30s)

    except Exception as e:
        logger.error(f"[Worker Error] Failed to process SMILES at index {idx} ('{raw_smile}'): {e}")
        return (idx, None, None)


def dfs_collect_rule_idx(root):
    """ Perform DFS on the subgraph tree to collect rule indices. """
    rule_seq, partition_list = [], []

    def dfs(node):
        rule_seq.append(node.rule_idx)
        partition_list.append(node.subfrags)
        for child in sorted(node.children, key=lambda c: c.weight):
            dfs(child)

    dfs(root)
    return rule_seq[::-1], partition_list[::-1]


""" 
新版本引入了“生产者-消费者”模型（即 _consumer_task 线程 和 有界队列 queue.Queue），旨在解决旧版本在极端情况下的内存溢出（OOM）问题。

举例：
如果 grammar.append 很快 (不是瓶颈), old 版本：效率最高。因为它没有 Queue 和 consumer 线程的额外开销。
如果 grammar.append 是瓶颈,生产速度远远大于消费速度, old版本中结果会无限堆积在主进程的内存里。最终结果导致主进程内存溢出 (OOM)，程序崩溃。
"""


# 消费者线程，运行在单独的线程中，专门处理 grammar.append
def _consumer_task(results_queue, grammar, rule_seq_list, partition_list, pbar):
    """
    Consumer thread target function.
    从队列拉取结果并执行串行的 grammar.append 逻辑。
    """
    while True:
        result = results_queue.get()  # 阻塞等待，直到拿到元素或哨兵
        if result is None: # 收到“结束”信号
            results_queue.task_done()
            break

        idx, rule_fp_list, motif_subgs = result

        try:
            if motif_subgs is None or rule_fp_list is None: 
                logger.warning(f"Failed to gen grammar for SMILES at index {idx}")
            else:
                root = motif_subgs[-1]
                # 这是唯一的串行瓶颈点，现在它在自己的线程中运行
                for (rule, fp), motif in zip(rule_fp_list, motif_subgs):
                    final_rule, rule_idx = grammar.append(rule, motif, fp) # 调用带 fp 的 append
                    motif.rule_idx = rule_idx
                    motif.rule = final_rule
                
                rule_seq_list[idx], partition_list[idx] = dfs_collect_rule_idx(root)

        except Exception as e:
            logger.error(f"[Consumer Error] Failed processing result for index {idx}: {e}")
        finally:
            # 无论成功失败，都更新进度条并标记任务完成
            pbar.set_postfix(rules=grammar.num_prod_rule, refresh=False) 
            pbar.update(1)
            results_queue.task_done()


# **** 核心函数 ****
def generate_grammar_from_smiles(smile_path, vocab_path, lifting_type, grammar=None, preserve_anchor_order=True,
                                 num_processes=64, chunksize=5, cache_size=50000):
    """
    ... (docstring) ...
    """
    # 1. 读取数据
    smiles_generator, N = stream_smiles(smile_path)
    logger.info(f"Loaded {N} SMILES from {smile_path}.")
    

    # 2. 初始化容器
    rule_seq_list, partition_list = [None] * N, [None] * N
    if grammar is None:
        grammar = ProductionRuleCorpus(preserve_anchor_order)
    max_processes = multiprocessing.cpu_count()
    num_processes = min(num_processes, max_processes)

    # 3. 创建进度条
    pbar = tqdm(total=N, desc="▶ Grammar gen")
    if num_processes <= 1:
        # --- 串行模式 (无需线程) ---
        logger.info("Running in sequential mode (num_processes <= 1).")
        _init_worker(vocab_path, lifting_type, cache_size)

        with pbar:
            for _i, smile in enumerate(smiles_generator):
                idx, rule_fp_list, motif_subgs = _process_smile_to_rules_worker((_i, smile))
                if motif_subgs is None:
                    logger.warning(f"Failed to gen grammar for SMILES at index {idx}")
                    pbar.update(1)
                    continue

                root = motif_subgs[-1]
                for (rule, fp), motif in zip(rule_fp_list, motif_subgs):
                    final_rule, rule_idx = grammar.append(rule, motif, fp) # 调用带 fp 的 append
                    motif.rule_idx = rule_idx
                    motif.rule = final_rule
                    
                rule_seq_list[idx], partition_list[idx] = dfs_collect_rule_idx(root)
                pbar.set_postfix(rules=grammar.num_prod_rule)
                pbar.update(1)

    else:
        # --- 并行模式 (使用生产者-消费者线程) ---
        initargs = (vocab_path, lifting_type, cache_size)
        pool_manager = Pool(processes=num_processes, initializer=_init_worker, initargs=initargs)
        
        logger.info(f"Parallel grammar generation with {num_processes} processes (max {max_processes}, chunksize {chunksize})")
        
        indexed_smiles_generator = enumerate(smiles_generator)
        iterator = pool_manager.imap_unordered(_process_smile_to_rules_worker, indexed_smiles_generator, chunksize=chunksize)

        # 4. 创建队列和消费者线程
        # 设置队列最大长度，提供反压 (back-pressure)
        results_queue = queue.Queue(maxsize=num_processes * 300) 
        consumer = threading.Thread(
            target=_consumer_task, 
            args=(results_queue, grammar, rule_seq_list, partition_list, pbar),
            daemon=True # 保证主线程退出时，该线程也会退出
        )
        consumer.start() # 启动消费者

        # 5. 主线程（生产者）循环
        # 主线程现在只做一件事：从 pool 拿结果 (快), 放入 queue (快)
        # 它不再执行 grammar.append()
        try:
            with pool_manager: # pbar 已经被消费者线程接管，这里不再需要
                for result in iterator:
                    # 如果队列满了 (消费者跟不上)，put会阻塞
                    # 这会自动减慢 workers 的速度，防止内存爆炸
                    results_queue.put(result)
            
        except Exception as e:
            logger.error(f"[Main Thread Error] Error during result iteration: {e}")
        finally:
            # 6. 发送“结束”信号 (None) 并等待消费者完成
            results_queue.put(None)
            logger.info("Main thread finished producing. Waiting for consumer...")
            consumer.join() # 等待消费者线程处理完队列中所有剩余任务
            pbar.close()    # 关闭进度条
            logger.info("Consumer thread finished.")


    num_processed = N - rule_seq_list.count(None)
    logger.info(f"Successfully processed {num_processed} / {N} SMILES.")
    logger.info(f"Generated a total of {grammar.num_prod_rule} unique production rules.")

    return grammar, rule_seq_list, partition_list
