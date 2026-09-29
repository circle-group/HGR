# main_diff.py

"""
PYTHONHASHSEED=123 python scripts/main_diff.py --config configs/zinc250k/diff_rsg.yaml --mode sample --seed 123

PYTHONHASHSEED=123 python scripts/main_diff.py --config configs/ringdiv300k/diff_rsg.yaml --mode sample --seed 123
"""

import os
import sys

# os.environ["WANDB_MODE"] = "online"
# os.environ["_DEBUG_"] = "False" # 设置是否打印调试信息

from hgr.utils.loader import set_env_from_config
set_env_from_config()

import wandb
import argparse
import traceback
import logging
import textwrap
from datetime import datetime
from hgr.diffusion import trainer, sampler
from hgr.utils.file_utils import PathManager, sanitize_filename
from hgr.utils.loader import load_config, seed_context, init_exp
from hgr.utils.debug_utils import Timer, make_wandb_init_config, update_wandb_config

logger = logging.getLogger(__name__)

def print_results(results: dict, title: str, headers: list, paths: list, col_width: int = 15):
    """
    打印评估结果表格
    :param results: 指标字典
    :param title: 标题
    :param headers: 表头列表，如 ['Metric', 'Diff_rel']
    :param paths: 需要显示的 checkpoint 路径
    :param col_width: 每列宽度
    """
    sep = " | "
    table_width = col_width * len(headers) + len(sep) * len(headers)
    print(f" {title} ".center(table_width, '='))
    for p in paths:
        for line in textwrap.wrap(p, width=table_width, break_long_words=True, break_on_hyphens=False):
            print(line.center(table_width))
    print("=" * table_width)
    print(f"{headers[0]:<{col_width}}{sep}{headers[1]:>{col_width}}{sep}")
    print("-" * table_width)
    for key in results.keys():
        val = f"{float(results[key]):.7f}"
        print(f"{key:<{col_width}}{sep}{val:>{col_width}}{sep}")
    print("=" * table_width)

if __name__ == '__main__':
    # 1) 环境 & 配置
    # setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='configs/zinc250k/diff_rsg.yaml',
        help='YAML path relative to the repository root (configs/...) or an absolute path',
    )
    parser.add_argument('--mode', type=str, default='sample', choices=['train', 'sample'])
    parser.add_argument('--samples', type=int, default=None, help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--skip-eval', action='store_true',
                        help='Sample only, skip metric evaluation (for two-stage GPU-sample + CPU-eval pipelines)')
    args = parser.parse_args()
    config = load_config(args.config)
    env_cfg = init_exp(config)

    if args.mode == 'sample':
        if args.ckpt is not None:
            config.path.scorenet_ckpt_path = args.ckpt
        if args.samples is not None:
            config.eval.num_samples = args.samples
        if args.seed is not None:
            config.seed = args.seed
        if args.skip_eval:
            config.eval.skip_eval = True

    wandb.init(
        entity=env_cfg.wandb.entity,  # team name
        project="HoGen-Diff",  # project name
        name=f"{os.getenv('WANDB_SWEEP', '')}{config.data.name}-{config.exp_time}",  # experiment name
        dir=PathManager.WANDB_DIR,
        config=make_wandb_init_config(config),
        allow_val_change=True,  # 允许在调参时修改配置
    )

    exit_code = 0
    try:
        config = update_wandb_config(config)
        if args.mode == 'train':
            trainer = trainer.Trainer(config)
            trainer.train()
        elif args.mode == 'sample':
            with Timer(f"sampler", enabled=True):
                sampler = sampler.Sampler(config)
                with seed_context(config.seed):
                    gen_rel, gen_smi = sampler.sample(return_samples=True)
            
            # 存储生成的分子
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ckpt_name = sanitize_filename(os.path.basename(config.path.scorenet_ckpt_path).replace('.pt', ''))
            save_filename = f"{ckpt_name}_n{len(gen_smi)}_seed{config.seed}({timestamp}).smi"
            save_parts = [PathManager.RESULTS_ROOT, "samples", "diff", config.data.name]
            if getattr(config, "exp_name", None):
                save_parts.append(config.exp_name)
            save_dir = os.path.join(*save_parts)
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, save_filename)
            with open(save_path, 'w') as f:
                for smi in gen_smi:
                    f.write(smi + '\n')
            logger.info(f"Successfully saved {len(gen_smi)} SMILES to: {os.path.abspath(save_path)}")



            # 打印采样结果（如果跳过了 eval，gen_rel 为空字典）
            if gen_rel:
                print_results(
                    gen_rel,
                    title=f"GVAE + Diff {config.eval.num_samples} samples",
                    headers=['Metric', 'Diff_rel'],
                    paths=[config.path.gvae_ckpt_path, config.path.scorenet_ckpt_path]
                )
            else:
                logger.info("Eval skipped — SMILES saved to %s", save_path)

        print("Finished main_diff.py")
    except KeyboardInterrupt:
        print(">>> Caught SIGINT, finishing wandb...")
        exit_code = 130
    except Exception as e:
        print(">>> An error occurred:")
        traceback.print_exc()  # 打印详细的异常 traceback
        exit_code = 1
    finally:
        # 无论正常结束还是被中断，都要调用 finish 保证把 log、files 写入磁盘
        wandb.finish()
        sys.exit(exit_code)
