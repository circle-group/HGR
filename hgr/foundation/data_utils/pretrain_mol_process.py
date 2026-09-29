# pretrain_mol_process.py

import os, sys


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import random, collections, gc
from multiprocessing import get_context

from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from rdkit import DataStructs  # ← 新增：用于把 bitvect 转 numpy

from hgr.foundation.data_utils.mol_defs import ALLOW_FEATURES
from hgr.foundation.data_utils.mol_utils import gen_robust_conformers, _optimize_all_confs
from hgr.utils.mol_utils import remove_salt_stereo


# ---- RDKit 全局静默（父进程）----
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')



# ---- Fast lookup maps (build once) ----
ATOM_NUM_TO_IDX   = {z:i for i, z in enumerate(ALLOW_FEATURES['atomic_num_list'])}
CHIRAL_TO_IDX     = {c:i for i, c in enumerate(ALLOW_FEATURES['chirality_list'])}
BOND_TYPE_TO_IDX  = {b:i for i, b in enumerate(ALLOW_FEATURES['bonds'])}
BOND_DIR_TO_IDX   = {d:i for i, d in enumerate(ALLOW_FEATURES['bond_dirs'])}



from rdkit.Chem import MACCSkeys ## MACCS

## Descriptor
from rdkit.Chem import Descriptors

# FUNCS = {name:func for name, func in Descriptors.descList}
## to exclude large differences between maximum and minimum (Note that different data needs to be taken differently)
DES_DEL = ['Ipc', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI',  'BCUT2D_MRLOW', 'BCUT2D_MWHI', 'BCUT2D_MWLOW',
           'MaxAbsPartialCharge', 'MaxPartialCharge', 'MinAbsPartialCharge', 'MinPartialCharge'] 
DESC_FN_PAIR = [(name, func) for name, func in sorted(Descriptors.descList) if name not in set(DES_DEL)]



def mol2PyG_conf(mol, numThreads=1):
    """
    Convert an RDKit Mol to a PyG Data with:
      - x: [num_atoms, 2]  (atomic_num_idx, chirality_idx)
      - edge_index, edge_attr (bond_type_idx, bond_dir_idx)
      - descriptors, descriptors_ss, maccs
      - esomeprazole: RDKit Mol with H for conformer embedding
      - min1pos, min2pos, min3pos: torch.FloatTensor [num_atoms, 3]
    """
    try:
        mol_prop = Chem.RemoveHs(mol, implicitOnly=True)
        try:      
            Chem.SanitizeMol(mol_prop)  # 完整清洗：会建立芳香性、价态、环信息等
        except Exception:
            # 若完整清洗失败，至少保证环信息和属性缓存可用
            Chem.rdmolops.FastFindRings(mol_prop)          # 初始化 RingInfo
            mol_prop.UpdatePropertyCache(strict=False)      # 补属性缓存
            # 可选：尽量 kekulize，失败就算了
            try:
                Chem.Kekulize(mol_prop, clearAromaticFlags=True)
            except Exception:
                pass

        
        # ----- STEP 1. atoms (num_atom_features = 2: atom type,  chirality tag) ----- #
        atom_features_list = []  
        # 本地变量绑定加速属性访问
        atom_map = ATOM_NUM_TO_IDX
        ch_map   = CHIRAL_TO_IDX
        for atom in mol_prop.GetAtoms():
            atom_feature = [atom_map[atom.GetAtomicNum()], ch_map[atom.GetChiralTag()]]
            atom_features_list.append(atom_feature)
        # x = torch.tensor(np.array(atom_features_list), dtype=torch.long)
        x = np.array(atom_features_list, dtype=np.int64)
        if x.shape[0] < 3: ## invaild feature remove (addition)
            return None, "Invalid_atom_feature"

        # ----- STEP2. bonds ----- #
        num_bond_features = 2   # bond type, bond direction
        if len(mol_prop.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            bt_map, bd_map = BOND_TYPE_TO_IDX, BOND_DIR_TO_IDX
            for bond in mol_prop.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edges_list.extend([(i, j), (j, i)])
                edge_feature = [bt_map[bond.GetBondType()], bd_map[bond.GetBondDir()]]
                edge_features_list.extend([edge_feature, edge_feature])

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            # edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            # edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
            edge_attr = np.array(edge_features_list, dtype=np.int64)
        else:   # mol has no bonds
            # edge_index = torch.empty((2, 0), dtype=torch.long)
            # edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr  = np.empty((0, num_bond_features), dtype=np.int64)

        
        # ----- STEP3. conformer    
        try:
            mol_conf = Chem.AddHs(mol) ## to make conformer
            heavy_idx = [a.GetIdx() for a in mol_conf.GetAtoms() if a.GetAtomicNum() > 1] # 全原子 → 重原子映射
            n_heavy = len(heavy_idx)

            n_confs = gen_robust_conformers(mol_conf, n_threads=numThreads, seed=22, min_required=3)
            if n_confs <= 0:
                return None, "gen conformer failed"
            
            energies = np.asarray(_optimize_all_confs(mol_conf, n_threads=numThreads, n_heavy=n_heavy), dtype=np.float32)
            sortidx = np.argsort(energies)            

            def pick(cid):
                pos_full = np.asarray(mol_conf.GetConformer(int(cid)).GetPositions(),  dtype=np.float32)  # [N_full,3]
                pos_heavy = pos_full[heavy_idx]                                       # [N_heavy,3]
                return pos_heavy 
            
            # save three with the lowest energies
            min1pos = pick(sortidx[0])
            min2pos = pick(sortidx[1] if len(sortidx) > 1 else sortidx[0])
            min3pos = pick(sortidx[2] if len(sortidx) > 2 else sortidx[0])

            if not (min1pos.shape == min2pos.shape == min3pos.shape == (x.shape[0], 3)):
                # print(f"conformer_shape_mismatch: {min1pos.shape}, {min2pos.shape}, {min3pos.shape}, {n}")
                return None, "conformer_shape_mismatch"

        except Exception:
            return None, "conformer failed"


        # ----- STEP4. MACCS (for Subgraph-level Pretext Task) ----- #
        try:
            maccs_func = MACCSkeys.GenMACCSKeys(mol_prop).ToList()[1:]
            maccs = np.asarray(maccs_func, dtype=np.uint8).reshape(1, -1) ## The value of the first bit of MACCS is not required.
        except Exception:
            return None, "maccs failed"

        # ----- STEPX. Morgan/ECFP (bit vector) ----- #
        try:
            MORGAN_NBITS = 1024
            bv = AllChem.GetMorganFingerprintAsBitVect(
                mol_prop,
                radius=2,
                nBits=MORGAN_NBITS,
                useChirality=True,
                useFeatures=False,
            )
            arr = np.zeros((MORGAN_NBITS,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(bv, arr)  # 显式转 numpy
            morgan = arr.reshape(1, -1)               # 与 maccs/desc 保持 [1, D]
        except Exception:
            return None, "morgan failed"


        # ----- STEP5. Descriptors ----- #
        try:
            descriptors = np.asarray([func(mol_prop) for name, func in DESC_FN_PAIR], dtype=np.float32).reshape(1, -1)
        except Exception:
            return None, "descriptors failed"
        

        data = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr,
            "descriptors_ss": descriptors, 
            "maccs": maccs,
            "morgan": morgan,
            "min1pos": min1pos, "min2pos": min2pos, "min3pos": min3pos }

        return data, None

    except Exception:
        return None, "Unknown error in mol2PyG_conf"




def _init_worker(seed_base: int = 1234):
    pid = os.getpid()
    seed = (seed_base + pid) % 2_147_483_647
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # RDKit 日志静默（每个子进程都要设）
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')


def process_single_mol(args):
    """
    处理单个分子的函数,
    始终返回 (idx, smiles, data, err)；成功时 err=None
    """
    idx, smiles = args
    try:
        smiles = remove_salt_stereo(smiles)
        rdkit_mol = AllChem.MolFromSmiles(smiles)
        if rdkit_mol is None:
            return idx, smiles, None, "MolFromSmiles failed"
        # smiles = Chem.MolToSmiles(rdkit_mol, canonical=True)
        # rdkit_mol = AllChem.MolFromSmiles(smiles)

        data, err = mol2PyG_conf(rdkit_mol, numThreads=1) #关闭内部的并行
        if data is not None:
            data['id'] = idx
        return idx, smiles, data, err # 直接返回 Data（CPU tensor），父进程立刻落盘分片，不累积
    except Exception:  
        return idx, smiles, None, "Unknown error in process_single_mol"


def process_pretrain(input_path, num_workers=1, chunksize=128, shard_size=50000):
    ## 1,974,507 samples (maccs and descriptors preprocessed, with conformer not generated excluded)
    # input_csv_path = os.path.join(self.raw_dir, 'smiles.csv')

     # ---------- 路径 ----------
    raw_dir = os.path.dirname(input_path)
    processed_dir = os.path.join(os.path.dirname(raw_dir), "processed")
    shards_dir = os.path.join(processed_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)


    # 顺序写 smiles（与结果一一对应，严格保持输入顺序）
    out_smiles_path = os.path.join(processed_dir, "smiles.csv")
    if os.path.exists(out_smiles_path):
        os.remove(out_smiles_path)

    # ---------- 分片缓冲 ----------
    shard_buf, shard_id, shard_paths, smiles_buf = [], 0, [], []
    def flush_shard():
        nonlocal shard_buf, shard_id, shard_paths, smiles_buf
        if not shard_buf:
            return
        
        shard_path = os.path.join(shards_dir, f"shard_{shard_id:06d}.pt")
        torch.save(shard_buf, shard_path)
        shard_paths.append(shard_path)
        shard_id += 1
        shard_buf.clear()

        with open(out_smiles_path, "a") as f:
            f.write("\n".join(smiles_buf))
            f.write("\n")
        smiles_buf.clear()

        gc.collect()
    
    
    # 准备数据
    input_df = pd.read_csv(input_path)
    if 'smiles' in input_df.columns:
        smiles_list = input_df['smiles'].astype(str).tolist()
    elif 'SMILES' in input_df.columns:
        smiles_list = input_df['SMILES'].astype(str).tolist()
    else:
        raise ValueError(f"Cannot find 'smiles' or 'SMILES' column in {input_path}")

    mol_args = [(i, smiles_list[i]) for i in range(len(smiles_list))]
    total = len(mol_args)
    err_counter = collections.Counter() # 专门用来统计错误类型/错误消息出现次数的计数器。


    num_workers = min(os.cpu_count() or 1, num_workers)
    if num_workers <=1:
        iterator = map(process_single_mol, mol_args)      
    else:
        ctx = get_context("spawn")
        pool = ctx.Pool(processes=num_workers, maxtasksperchild=512, initializer=_init_worker, initargs=(1234,) )
        iterator = pool.imap_unordered(process_single_mol, mol_args, chunksize=chunksize)


    pending = {}        # idx -> (smiles, data_or_None, err_or_None)
    next_flush = 0      # 期望输出的下一个 idx
    def drain():
        # 只要前缀是连续的，就地输出到 shard 缓冲；满了就 flush
        nonlocal next_flush
        while True:
            entry = pending.pop(next_flush, None)
            if entry is None:
                break
            s, d = entry
            if d is not None:
                shard_buf.append(d)
                smiles_buf.append(s)
                if len(shard_buf) >= shard_size:
                    flush_shard()
            next_flush += 1
    
    progress_bar = tqdm(iterator, total=total, desc="Processing mol", unit="mol", dynamic_ncols=True)
    for idx, smiles, data, err in progress_bar:
        if data is None:
            err_counter[err] += 1
            # print(f"Fail mol {idx}: {smiles}, {err}")
            # 更新进度条显示当前累计失败数
            tqdm.write(f"Fail mol {idx}: {smiles}, {err}")  # 仍然打印详细信息
            pbar_fail = sum(err_counter.values())            # 当前累计失败
            progress_bar.set_postfix_str(f"fail={pbar_fail}")

        pending[idx] = (smiles, data) # 存放 NumPy 字典，带 id（int），data=None时说明失败时也存进去
        drain()


    if num_workers > 1:
        pool.close()
        pool.join()
    
    # 收尾：把尾部连续区间刷完
    drain()
    flush_shard()  # 确保最后一个分片写入
                    
    # 统计失败
    fail_cnt = sum(err_counter.values())  # 直接统计err_counter更快
    print(f"[mol pretrain process] fail: {fail_cnt}/{total} ({fail_cnt/total:.2%})")
    if err_counter:
        print("Top error types:", err_counter.most_common(10))

    # —— 将 numpy 字典转换成 PyG Data（在父进程、最终聚合时做）——
    def _np_dict_to_torch_data(d):
        to_long  = lambda a: torch.tensor(a, dtype=torch.long)
        to_float = lambda a: torch.tensor(a, dtype=torch.float32)
        return Data(
            x=to_long(d["x"]),
            edge_index=to_long(d["edge_index"]),
            edge_attr=to_long(d["edge_attr"]),
            descriptors_ss=to_float(d["descriptors_ss"]),
            maccs=to_long(d["maccs"]),
            morgan=to_long(d["morgan"]),
            min1pos=to_float(d["min1pos"]),
            min2pos=to_float(d["min2pos"]),
            min3pos=to_float(d["min3pos"]),
        )

    # -------------------------------------------------------------------------------------------------------- #
    # 2) 顺序读取分片 -> 聚合为 data_list（若数据极大，可增量 collate；这里一次性读）
    data_list = []
    for shard_path in tqdm(shard_paths, desc="Processing shards", total=len(shard_paths)):
        dict_list = torch.load(shard_path, weights_only=False)
        data_list.extend(_np_dict_to_torch_data(d) for d in dict_list)

    # import shutil; shutil.rmtree(shards_dir, ignore_errors=True)

    # smiles_csv_path之前已经写完了
    return data_list
