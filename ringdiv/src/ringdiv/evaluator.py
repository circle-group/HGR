# evaluator.py

"""
入口函数，用于评估生成分子的质量

NOTE: 设置PYTHONHASHSEED=123可以加速NSPDK的计算（本来计算NSPDK就是最慢的）
"""

import os, logging
logger = logging.getLogger(__name__)

import pickle, math
from rdkit import Chem, rdBase 
from multiprocessing import Pool
from .metrics.KLdiv_evaluator import eval_KLdiv
from .utils.eval_utils import mol_to_nx_graph, _get_unique_ordered_subset
from .utils.cache import sha256_stream_smiles, resolve_cache_dir, PREF_SCHEMA_VERSION, CURV_SCHEMA_VERSION
from .metrics.molsets import (get_mol, mapper, remove_invalid, fraction_unique, 
    novelty,  SNNMetric, FragMetric, 
    ScafMetric, WassersteinMetric, logP, SA, QED, weight)
from fcd_torch import FCD as FCDMetric
from .metrics.filter_evaluator import fraction_passes_filters

def disable_rdkit_log():
    rdBase.DisableLog('rdApp.*')

def enable_rdkit_log():
    rdBase.EnableLog('rdApp.*')

def compute_intermediate_statistics(smiles, n_jobs=1, device='cpu',
                                    batch_size=512, pool=None):
    """
    The function precomputes statistics such as mean and variance for FCD, etc.
    It is useful to compute the statistics for test and scaffold test sets to
        speedup metrics calculation.

    预计算一批“参考分布”的统计量（pref），用于加速后续指标计算。

    典型用途：
    - 对 test set 先 precalc 一次（例如 FCD 的参考统计、SNN/Frag/Scaf/Properties 的参考统计）
    - 之后对每次生成样本 gen 评估时，直接复用 pref，避免重复计算参考分布
    """
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1
    statistics = {}
    mols = mapper(pool)(get_mol, smiles)
    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
    statistics['FCD'] = FCDMetric(**kwargs_fcd).precalc(smiles)
    statistics['SNN'] = SNNMetric(**kwargs).precalc(mols)
    statistics['Frag'] = FragMetric(**kwargs).precalc(mols)
    statistics['Scaf'] = ScafMetric(**kwargs).precalc(mols)

    for name, func in [('logP', logP), ('SA', SA), ('QED', QED), ('weight', weight)]:
        statistics[name] = WassersteinMetric(func, **kwargs).precalc(mols)

    if close_pool:
        pool.terminate()
    return statistics

# 全局缓存：test set 的 precalc 结果（pref）
# 目的：避免每次调用 get_all_metrics 都重新 precalc test set
_ptest_cache = None
_ref_hashkey = None
_curv_test_cache = {}


class _CompatUnpickler(pickle.Unpickler):
    """Backward-compatible unpickler for caches produced under older/newer deps."""

    def find_class(self, module, name):
        # NumPy 2.x pickles may reference `numpy._core.*`, while NumPy 1.x uses `numpy.core.*`.
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pickle_compat(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError as exc:
            if "numpy._core" not in str(exc):
                raise
            f.seek(0)
            obj = _CompatUnpickler(f).load()
            logger.warning("Loaded legacy cache with NumPy module remap: %s", path)
            return obj



def get_all_metrics(gen, test_smiles=None, train_smiles=None, metrics=None, num_eval=None,
                    n_jobs=16, device='cpu', batch_size=512, pool=None, cache_dir=None,
                    use_cache: bool = True):
    """
    评估主函数：对生成样本 gen 计算多种指标。
    参数：
    - metrics (list of str): 指定要计算的指标列表。
        - basic index: 'validity', 'uniqueness', 'novelty', 'VUN'
        - chemical index: 'FCD', 'SNN', 'Frag', 'Scaf', 'logP', 'SA', 'QED', 'weight', 'Filters', ,  
        - topological index: 'NSPDK'
        - higher-order index: ollivier_ricci_curvature ('curvature', 'ollivier', 'orc'), 
                            balanced_forman_curvature ('bforman', 'bfc')
                            forman_curvature ('forman', 'frc')
                            resistance_curvature ('resist', 'rc')
    
    - gen: 生成分子列表（你这里传的是 RDKit Mol 列表，也可能是 SMILES，取决于上游）
    - test: 参考集（通常是 test SMILES）
    - train: 训练集（用于 Novelty/VUN）
    - eval_k: 评估用前 k 个生成样本（如果 gen 更长，会截断. MOSES eval_k=25,000, 其余数据集为10,000)
    - n_jobs/device/batch_size/pool: 并行与设备配置
    - cache_dir: 用于缓存 precalc 结果的目录, 可以极大的计算效率, HGR项目中设置为: os.path.join(PathManager.DATA_DIR, 'cache')

    """

    global _ptest_cache, _ref_hashkey, _curv_test_cache

    # note: 这边只计算了一种curvature
    BASIC_METRICS = ['validity', 'novelty', 'uniqueness', 'fcd']
    ALL_METRICS = BASIC_METRICS + ['scaf', 'snn', 'logp', 'sa', 'qed', 'frag', 'filters', 'NSPDK', 'KLdiv', 'ollivier', 'bforman']
    ALL_CURVATURE = ['ollivier', 'bforman', 'forman', 'resist']
    BASIC_CURVATURE = ['ollivier', 'bforman']

    # curvature的不同别名
    curvature_map = {
        'ollivier_ricci_curvature': {'curvature', 'ollivier', 'orc'}, # default curvature index
        'balanced_forman_curvature': {'bforman', 'bfc'},
        'forman_curvature': {'forman', 'frc'},
        'resistance_curvature': {'resist', 'rc'}
    }


    if metrics is None:
        metrics = BASIC_METRICS
    elif isinstance(metrics, str):
        metrics = [metrics]
    
    if 'all' in metrics:
        metrics += ALL_METRICS
    if 'all_curvature' in metrics:
        metrics += ALL_CURVATURE
    if 'basic' in metrics:
        metrics += BASIC_METRICS
    if 'qm9' in metrics or 'zinc250k' in metrics or 'ringdiv300k' in metrics:
        metrics += BASIC_METRICS + ['NSPDK']
    if 'moses' in metrics:
        metrics += BASIC_METRICS + ['filters', 'snn','scaf']
    if 'guacamol' in metrics:
        metrics += BASIC_METRICS + ['KLdiv']
    

    metrics = {i.lower() for i in metrics} # 全部变成小写方便比较
    eval_k = min(len(gen), num_eval) if num_eval is not None else len(gen)
    if eval_k <= 0:
        return {"No valid molecules": -1}

    test_required = {"fcd", "snn", "frag", "scaf", "logp", "sa", "qed", "weight", "nspdk", "kldiv"}
    curvature_aliases = set().union(*curvature_map.values())
    test_hash = None
    if metrics & (test_required | curvature_aliases):
        if not test_smiles:
            raise ValueError(
                "test_smiles is required when computing metrics that compare against test data "
                "(e.g. fcd, snn, frag, scaf, logp, sa, qed, weight, nspdk, kldiv, curvature)"
            )
        test_hash = sha256_stream_smiles(test_smiles) # 数据test是固定的，因此可以通过缓存加速
        if use_cache:
            cache_dir = resolve_cache_dir(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
        else:
            cache_dir = None


    disable_rdkit_log()
    results = {}
    close_pool = False
    if pool is None:
        if n_jobs != 1:
            pool = Pool(n_jobs)
            close_pool = True
        else:
            pool = 1

    
    # -------------------------------------------------------------------------
    # 1) 基础指标：Validity / Uniqueness / Novelty / VU10k / VUN10k
    # -------------------------------------------------------------------------
    gen_smi = remove_invalid(gen[:eval_k], canonize=True)
    valid_num = len(gen_smi)

    if valid_num <=0:
        return {"Validity": 0, "Uniqueness": 0, "Novelty": 0, "VUN10k": 0, "FCD": float('inf'), 'NSPDK': float('inf'), 'KLdiv': 0}

    results['Validity'] = valid_num / eval_k #raw_num
    # results['Uniqueness'] = fraction_unique(gen_smi, n_jobs=pool)
    unique_smiles_set = set(gen_smi)
    results['Uniqueness'] = len(unique_smiles_set) / valid_num

    gen_rdkit_mols = mapper(pool)(get_mol, gen_smi)
    if train_smiles is not None and 'novelty' in metrics:
        # metrics['Novelty'] = novelty(gen_rdkit_mols, train, n_jobs=pool)
        train_set = set(train_smiles)
        novel_num = len(unique_smiles_set - train_set)
        results['Novelty'] = novel_num / len(unique_smiles_set) # MOSES-style
        results[f'VUN{eval_k}'] = novel_num / eval_k

    ## Guacamol中Uniqueness和Novelty的定义略有不同
    N=10000
    if 'guacamol' in metrics:
        all_valid_smiles = remove_invalid(gen, canonize=True) if len(gen) > eval_k else gen_smi
        results['Uniqueness (Guacamol-style)'] = len(set(all_valid_smiles[:N]))/N
        if train_smiles is not None and 'novelty' in metrics:
            # 先从 all_valid_smiles 中“按顺序取前 N 个 unique”，再减去 train_set，得到 novel 的数量
            results['Novelty (Guacamol-style)'] = len(set(_get_unique_ordered_subset(all_valid_smiles, N)) - train_set) / N
    
    if 'kldiv' in metrics:
        results['KLdiv'] = eval_KLdiv(gen_smi, test_smiles)



    # -------------------------------------------------------------------------
    # 2) 缓存 test set 的 pref（ptest_cache），避免反复 precalc
    # -------------------------------------------------------------------------
    if (_ptest_cache is None or test_hash != _ref_hashkey) and any(k in metrics for k in ["fcd","snn","frag","scaf","logp","sa","qed","weight"]):
        # 数据test是固定的，因此可以通过缓存加速
        # HOSTNAME = socket.gethostname().split('.')[0]  # 不同设备缓存的数据格式可能不兼容(发现好像是兼容的，因此删掉了)
        # ptest_cache_path = os.path.join(PathManager.DATA_DIR, 'cache', f'ptest.pkl')   
        if use_cache and cache_dir is not None:
            fname = f"ptest_v{PREF_SCHEMA_VERSION}_{test_hash}.pkl"
            ptest_cache_path = os.path.join(cache_dir, fname)
            
            if os.path.exists(ptest_cache_path):
                _ptest_cache = _load_pickle_compat(ptest_cache_path)
                logger.info(f"Loaded ptest_cache from {ptest_cache_path}")
            else:
                _ptest_cache = compute_intermediate_statistics(test_smiles, n_jobs=n_jobs, device=device, batch_size=batch_size, pool=pool)
                with open(ptest_cache_path, 'wb') as f:
                    pickle.dump(_ptest_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"Saved ptest_cache to {ptest_cache_path}")
        else:
            _ptest_cache = compute_intermediate_statistics(test_smiles, n_jobs=n_jobs, device=device, batch_size=batch_size, pool=pool)
        _ref_hashkey = test_hash
    # -------------------------------------------------------------------------
    # 3) 分布匹配指标（FCD/SNN/Frag/Scaf/Filters/NSPDK）与性质分布（Wasserstein）
    # -------------------------------------------------------------------------
    if 'fcd' in metrics:
        kwargs_fcd = {'n_jobs': n_jobs, 'device': device, 'batch_size': batch_size}
        results['FCD'] = FCDMetric(**kwargs_fcd)(gen=gen_smi, pref=_ptest_cache['FCD'])
        results['FCD_score'] = math.exp(-0.2 * results['FCD'])


    kwargs = {'n_jobs': pool, 'device': device, 'batch_size': batch_size}
    if 'snn' in metrics:
        results['SNN'] = SNNMetric(**kwargs)(gen=gen_rdkit_mols, pref=_ptest_cache['SNN'])
    if 'frag' in metrics:
        results['Frag'] = FragMetric(**kwargs)(gen=gen_rdkit_mols, pref=_ptest_cache['Frag'])
    if 'scaf' in metrics:
        results['Scaf'] = ScafMetric(**kwargs)(gen=gen_rdkit_mols, pref=_ptest_cache['Scaf'])
    if 'filters' in metrics:
        results['Filters'] = fraction_passes_filters(gen_rdkit_mols, pool)
    if 'nspdk' in metrics:
        # NSPDK 依赖 dill 等，为了避免在 `import ringdiv` 或不需要 NSPDK 的场景下触发一整条依赖链，这里采用延迟导入。
        try:
            from .metrics.nspdk_evaluator import nspdk_eval_fn
        except (ImportError, ModuleNotFoundError) as e:
            raise ImportError(
                "NSPDK was requested, but required dependencies are missing. "
                "Please install: `pip install dill`."
            ) from e
        results['NSPDK'] = nspdk_eval_fn(gen_rdkit_mols, test_smiles, n_jobs=n_jobs, 
                                         cache_dir=cache_dir, use_cache=use_cache) # NSPDK：图核相关指标，通常耗时较高
    

    # Chemical Properties
    for name, func in [('logP', logP), ('SA', SA), ('QED', QED), ('weight', weight)]:
        if name.lower() not in metrics: continue
        results[name] = WassersteinMetric(func, **kwargs)(gen=gen_rdkit_mols, pref=_ptest_cache[name])


    # -------------------------------------------------------------------------
    # 4) 计算 Curvature
    # -------------------------------------------------------------------------
    # 找出所有需要计算的 curvature
    curvatures_to_compute = []
    for measure, aliases in curvature_map.items():
        if aliases & metrics:
            if measure == 'ollivier_ricci_curvature': name = 'K_OR'
            elif measure == 'balanced_forman_curvature': name = 'K_BF'
            elif measure == 'forman_curvature': name = 'K_FR'
            elif measure == 'resistance_curvature': name = 'K_R'
            curvatures_to_compute.append((measure, name))

    if curvatures_to_compute:
        # Curvature 相关依赖（gudhi / POT(ot)）较重，且多数情况下不需要。
        # 因此这里采用 lazy import：只有确实要算 curvature 才导入相关模块，
        # 避免用户在未安装可选依赖时 `import ringdiv` 直接报错。
        try:
            from .curvature.compare import Comparator as CurvatureComparator
        except ImportError as e:
            raise ImportError(
                "Curvature was requested, but optional dependencies are missing. "
                "Please install `gudhi` and `POT` (import name: `ot`), e.g. `pip install gudhi POT`."
            ) from e

        gen_graphs = mapper(pool)(mol_to_nx_graph, gen_rdkit_mols)
        test_graphs = None

        def _curv_cache_key(measure, homology_dims, extended_persistence):
            dims = "-".join(str(d) for d in homology_dims)
            return (
                f"curv_{measure}_v{CURV_SCHEMA_VERSION}_{test_hash}"
                f"_d{dims}_e{int(extended_persistence)}"
            )

        for measure, name in curvatures_to_compute:
            comp = CurvatureComparator(measure=measure, n_jobs=n_jobs)
            cache_key = _curv_cache_key(measure, comp.homology_dims, comp.extended_persistence)
            if cache_key in _curv_test_cache:
                test_pd = _curv_test_cache[cache_key]
            else:
                cache_path = os.path.join(cache_dir, f"{cache_key}.pkl") if (use_cache and cache_dir is not None) else None
                if cache_path is not None and os.path.exists(cache_path):
                    test_pd = _load_pickle_compat(cache_path)
                    _curv_test_cache[cache_key] = test_pd
                    logger.info(f"Loaded test curvature cache from {cache_path}")
                else:
                    if test_graphs is None:
                        test_mols = mapper(pool)(get_mol, test_smiles)
                        test_graphs = mapper(pool)(mol_to_nx_graph, test_mols)

                    test_pd = comp._curvature_filtration(test_graphs)
                    _curv_test_cache[cache_key] = test_pd

                    if cache_path is not None:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(test_pd, f, protocol=pickle.HIGHEST_PROTOCOL)
                        logger.info(f"Saved test curvature cache to {cache_path}")
            
            # curvarure 计算的核心代码
            results[name] = comp.fit_transform(gen_graphs, None, precomputed_diagram2=test_pd)

    # -------------------------------------------------------------------------
    enable_rdkit_log()
    if close_pool:
        pool.close()
        pool.join()
    return results






if __name__ == '__main__':
    # 1. Prepare your data
    gen_smiles = ["CCO", "c1ccccc1"]
    ref_smiles = ["CCO", "CCCl"] # Reference test set

    gen_mols = [Chem.MolFromSmiles(s) for s in gen_smiles if s is not None]

    # 2. Compute metrics
    # You can specify which metrics to compute via the 'metrics' list
    results = get_all_metrics(
        gen=gen_mols, 
        test_smiles=ref_smiles, 
        train_smiles=ref_smiles, # Optional, for Novelty benchmarks
        metrics=['validity', 'unique', 'fcd', 'fragments', 'scaffolds', 'nspdk'],
        use_cache=False
    )

    print(results)
