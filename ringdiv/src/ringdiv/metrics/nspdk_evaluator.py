# nspdk_evaluator.py

import os
import numpy as np
from rdkit import Chem
import logging
from scipy.sparse import save_npz, load_npz
from sklearn.metrics.pairwise import pairwise_kernels
from .eden import vectorize
from ..utils.cache import resolve_cache_dir, sha256_stream_smiles, PREF_SCHEMA_VERSION
from ..utils.eval_utils import mols_to_nx
logger = logging.getLogger(__name__)


# 一次性准备好 ref_vec、X_cache
_ref_vec = None
_X_mean  = None
_ref_hashkey = None

def _prepare_reference(test_smiles, n_jobs=16, cache_dir=None, use_cache=True):
    global _ref_vec, _X_mean, _ref_hashkey
    test_hash = sha256_stream_smiles(test_smiles)
    if _ref_hashkey == test_hash and _ref_vec is not None and _X_mean is not None:
        return
    seed_env = os.environ.get("PYTHONHASHSEED", None)
    if seed_env is None:
        logger.info("Setting 'PYTHONHASHSEED' can accelerate NSPDK computing by avoid re-computing reference vectors.")

    if use_cache:
        cache_dir = resolve_cache_dir(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

        ref_cache_path = os.path.join(cache_dir, f'ref_vec_v{PREF_SCHEMA_VERSION}_hashseed{seed_env}_{test_hash}.npz')
        Xmean_cache_path   = os.path.join(cache_dir, f'ref_kernel_X_mean_v{PREF_SCHEMA_VERSION}_hashseed{seed_env}_{test_hash}.npy')
        # ref_cache_path = os.path.join(PathManager.DATA_DIR, 'cache', f'ref_vec-{PathManager.HOSTNAME}.npz')
        # Xmean_cache_path = os.path.join(PathManager.DATA_DIR, 'cache', f'ref_kernel_X_mean-{PathManager.HOSTNAME}.npy')
    else:
        cache_dir = None
    
    
    

    if use_cache and (seed_env is not None) and (os.path.exists(ref_cache_path) and os.path.exists(Xmean_cache_path)):
        _ref_vec = load_npz(ref_cache_path)
        _X_mean = np.load(Xmean_cache_path).item()
        logger.info(f"Loaded reference vectors from {ref_cache_path}")
        logger.info(f"Loaded reference kernel mean from {Xmean_cache_path}")
    else:
        # 只在首次调用时执行
        logger.info(f"Building reference vectors for NSPDK (use_cache={use_cache})")
        mols = [Chem.MolFromSmiles(smi) for smi in test_smiles]
        mol_ref_list = mols_to_nx(mols)
        # mol_ref_list = [mol_to_nx_graph(mol) for mol in mols]

        _ref_vec = vectorize(mol_ref_list, complexity=4, discrete=True)
        X = pairwise_kernels(_ref_vec, None, metric='linear', n_jobs=n_jobs)
        _X_mean = np.average(X)

        if use_cache and cache_dir is not None:
            save_npz(ref_cache_path, _ref_vec)
            np.save(Xmean_cache_path, np.array(_X_mean)) # np.save 会保存为一维 array，要用 .item() 取出 float
            logger.info(f"Saved reference vectors to {ref_cache_path}")
            logger.info(f"Saved reference kernel mean to {Xmean_cache_path}")
    _ref_hashkey = test_hash
        


def nspdk_eval_fn(mol_pred_list, test_smiles, n_jobs = 16, cache_dir=None, use_cache=True):
    _prepare_reference(test_smiles, n_jobs=n_jobs,cache_dir=cache_dir, use_cache=use_cache) # 确保参考集只准备一次


    # ref_cache_path = os.path.join(PathManager.DATA_DIR, 'cache', 'ref_vec.npz')      # 稀疏矩阵存储
    # X_cache_path = os.path.join(PathManager.DATA_DIR, 'cache', 'ref_kernel_X.npz')   # 压缩后的核矩阵
    #
    # if os.path.exists(ref_cache_path) and os.path.exists(X_cache_path):
    #     ref = load_npz(ref_cache_path)  # 读稀疏 ref 向量: scipy.sparse.csr_matrix
    #     X = np.load(X_cache_path)['X']  # 读核矩阵 X
    # else:
    #     mols = [Chem.MolFromSmiles(smi) for smi in test_smiles]
    #     mol_ref_list = mols_to_nx(mols)
    #
    #     ref = vectorize(mol_ref_list, complexity=4, discrete=True)
    #     X = pairwise_kernels(ref, None, metric='linear', n_jobs=n_jobs)
    #
    #     save_npz(ref_cache_path, ref) # 保存稀疏矩阵
    #     np.savez_compressed(X_cache_path, X=X) # 用压缩 npz 存储密集矩阵

    mol_pred_list = mols_to_nx(mol_pred_list)
    # mol_pred_list = [mol_to_nx_graph(mol) for mol in mol_pred_list]
    mol_pred_list = [G for G in mol_pred_list if not G.number_of_nodes() == 0]
    pred = vectorize(mol_pred_list, complexity=4, discrete=True)

    Y = pairwise_kernels(pred, None, metric='linear', n_jobs=n_jobs)
    Z = pairwise_kernels(_ref_vec, pred, metric='linear', n_jobs=n_jobs)

    return _X_mean + np.average(Y) - 2 * np.average(Z)


# # ### code adapted from https://github.com/idea-iitd/graphgen/blob/master/metrics/mmd.py
# def cal_nspdk_eval_old(mol_ref_list, mol_pred_list):
#     is_hist = False
#     metric = 'nspdk'
#     n_jobs = 20
#
#     if isinstance(mol_ref_list[0], Chem.Mol):
#         mol_ref_list = mols_to_nx(mol_ref_list)
#     else:
#         assert isinstance(mol_ref_list[0], nx.Graph)
#
#
#     graph_pred_list = mols_to_nx(mol_pred_list)
#     graph_pred_list = [G for G in graph_pred_list if not G.number_of_nodes() == 0]
#
#
#     def kernel_compute(X, Y=None, is_hist=False, metric='nspdk', n_jobs=20):
#         X = vectorize(X, complexity=4, discrete=True)
#         if Y is not None:
#             Y = vectorize(Y, complexity=4, discrete=True)
#         return pairwise_kernels(X, Y, metric='linear', n_jobs=n_jobs)
#
#     cache_nspds_path = os.path.join(PathManager.DATA_DIR, 'cache', 'ref_nspdk_stats.npz')
#     if os.path.exists(cache_nspds_path):
#         X = np.load(cache_nspds_path)["X"]
#     else:
#         X = kernel_compute(mol_ref_list, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
#         # np.save(cache_nspds_path, X)
#         np.savez_compressed(cache_nspds_path, X=X) # save memory: eg. QM9 dataset 1.3G -> 1G
#
#     Y = kernel_compute(graph_pred_list, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
#     Z = kernel_compute(mol_ref_list, Y=graph_pred_list, is_hist=is_hist, metric=metric, n_jobs=n_jobs)
#
#     return np.average(X) + np.average(Y) - 2 * np.average(Z)