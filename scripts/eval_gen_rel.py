
"""
入口函数，用于评估生成分子的质量

NOTE: 设置PYTHONHASHSEED=123可以加速NSPDK的计算（本来计算NSPDK就是最慢的）


Example of running:
PYTHONHASHSEED=123 python scripts/eval_gen_rel.py --config configs/ringdiv300k/gvae_rsg.yaml \
--smi_path ""

"""

"""
PYTHONHASHSEED=123 python scripts/eval_gen_rel.py --config configs/qm9/gvae_mig.yaml \
--smi_path ""


PYTHONHASHSEED=123 python scripts/eval_gen_rel.py --config configs/zinc250k/gvae_rsg.yaml \
--smi_path ""

PYTHONHASHSEED=123 python scripts/eval_gen_rel.py --config configs/moses/gvae_rsg.yaml \
--smi_path ""

PYTHONHASHSEED=123 python scripts/eval_gen_rel.py --config configs/guacamol/gvae_rsg.yaml \
--smi_path ""

PYTHONHASHSEED=123 python scripts/eval_gen_rel.py --config configs/ringdiv300k/gvae_rsg.yaml \
--smi_path ""
"""

import os
from hgr.utils.loader import set_env_from_config
set_env_from_config()

import textwrap
import argparse
from rdkit import Chem

# from hgr.diffusion.data_loader import load_train_test_smiles
from hgr.utils.data_splits import load_train_test_smiles
from ringdiv import get_all_metrics
from ringdiv.utils.cache import sha256_stream_smiles
from hgr.utils.loader import load_config, setup_logging, load_device
from hgr.utils.file_utils import load_smiles, PathManager
from hgr.utils.debug_utils import Timer

if __name__ == '__main__':
    # 1. 加载配置与环境
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='configs/moses/gvae_rsg.yaml',
        help='YAML path relative to the repository root (configs/...) or an absolute path',
    )
    parser.add_argument('--smi_path', type=str, default=None, help='Path to the SMILES file')
    parser.add_argument('--workers', type=int, default=24, help='Number of workers for decoding')
    parser.add_argument('--num_eval', type=int, default=None, help='Number of molecules to evaluate')
    args = parser.parse_args()

    config = load_config(args.config)
    PathManager.init(data_name=config.data.name)
    device = load_device()

    gen_smiles_path = args.smi_path
    gen_smiles = load_smiles(gen_smiles_path)
    gen_mols = [Chem.MolFromSmiles(smi) for smi in gen_smiles]#[:500]
    
    train_smiles, test_smiles = load_train_test_smiles(config)
    print(f"[INFO] train smiles: {len(train_smiles)}, test smiles: {len(test_smiles)}")
    print(f"test hash: {sha256_stream_smiles(test_smiles)}")


    # 6. 评估
    with Timer("Evaluate molecules time"):
        # metrics = [config.data.name] #'all' #'NSPDK' #'all' #['NSPDK', 'Curvature'] #'FCD'
        # metrics = ['basic','fcd', 'NSPDK', 'KLdiv','sa', 'qed']
        # metrics = ['bforman', 'ollivier']
        # metrics = [config.data.name, 'bforman', 'ollivier']
        metrics = ['all']
        cache_dir = os.path.join(PathManager.DATA_DIR, 'cache')
        num_eval = args.num_eval if args.num_eval is not None else 10000 if config.data.name != 'moses' else 25000
        gen_rel = get_all_metrics(gen=gen_mols, test_smiles=test_smiles, train_smiles=train_smiles, metrics=metrics, cache_dir=cache_dir, num_eval=num_eval, device=device)
        # kl_rel = eval_KLdiv(gen_smiles, test_smiles)
        
        

    # 7. 结果展示
    table_width = 35
    print("=" * table_width)
    print(f"Evaluation {len(gen_mols)} samples (num_eval={num_eval if num_eval is not None else len(gen_mols)}):")
    for line in textwrap.wrap(gen_smiles_path, width=table_width):
        print(line.center(table_width))
    print("-" * table_width)
    for key, val in gen_rel.items():
        print(f"{key:<12s}: {val:.6f}") # key 左对齐 12 字符宽度，值保留 4 位小数
    print("=" * table_width)
