import os
import json
import yaml
import wandb
import argparse
from datetime import datetime
import logging
from hgr.utils.loader import load_seed, load_config, init_exp
from hgr.utils.debug_utils import Timer
from hgr.utils.file_utils import dump_pickle, PathManager, load_smiles
from hgr.grammar.reconstruct import reconstruct_from_grammar
from hgr.grammar.grammar_generation import generate_grammar_from_smiles

logger = logging.getLogger(__name__)


def learn(config, args):
    # Purpose: orchestrate the full grammar learning and optional reconstruction workflow
    # 作用：实现整个文法学习和（可选的）重构测试流程
    data_name = config.data.name.lower()
    vol_path = os.path.join(PathManager.DATA_DIR, config.path.vocab_path)
    cache_size = args.cache_size #getattr(config.grammar, 'cache_size', None)
    lifting_type = args.lifting_type # 'MIG;, 'MEG', 'RSG',

    """
    Grammar construction strategies. MIG and RSG are the two paper-reported
    strategies; both build on a shared frequency-guided motif-merging base step
    (the `_phase_merge` function in hgr/grammar/lifting/complex_lifting.py):
    - MIG: Motif-Induced Grammar, base merging + anchor resolution; slower compute, fewer rules
    - RSG: Ring Scaffold Grammar (default), ring-system-aware merging on the same base step; moderate speed

    - MEG: Motif-Extracted Grammar — the bare frequency-merging base step itself
      (`_phase_merge` with resolve_anchor=False, allow_anchor_ambiguous=True), i.e. no
      anchor resolution. Fastest to compute and produces the most rules, but it is
      retained for ablation/reproducibility ONLY and is NOT a paper-reported strategy.

    Recommendation: use MIG for most molecules with simple ring structures, RSG for complex ring systems
    建议对于环结构简单的小分子用MIG，对于环结构复杂的大分子用RSG；MEG 为无 anchor resolution 的基线变体，仅用于消融/复现，论文不作为主结果汇报
    """


    output = args.output or os.path.join(PathManager.DATA_DIR, f"{lifting_type}_{data_name}_rebuilt.pklz")
    output = os.path.abspath(output)
    partitions_output = os.path.join(os.path.dirname(output), "partitions_" + os.path.basename(output))
    for path in (output, partitions_output, args.output_config):
        if path is None:
            continue
        if os.path.exists(path):
            raise FileExistsError(f"Refusing to overwrite {path}; choose --output with a new path")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    if data_name in ('ringdiv', 'ringdiv300k') and not getattr(config.path, 'raw_data', None):
        config.path.raw_data = f"raw/{data_name}_property.csv"

    with Timer("generate grammar"):
        if data_name in ['qm9', 'zinc250k', 'ringdiv', 'ringdiv300k']:
            smile_path = os.path.join(PathManager.DATA_DIR, config.path.raw_data)
            grammar, rule_seq_list, partition_list = generate_grammar_from_smiles(smile_path, vol_path, lifting_type, 
                                    num_processes=args.workers, cache_size=cache_size)
            sequences, partitions = rule_seq_list, partition_list
        elif data_name in ['moses', 'guacamol','test']:
            train_smile_path = os.path.join(PathManager.DATA_DIR, config.path.raw_data.train)
            grammar, rule_seq_list_train, partition_list_train = generate_grammar_from_smiles(train_smile_path, vol_path, lifting_type, 
                                    num_processes=args.workers, cache_size=cache_size)
            test_smile_path = os.path.join(PathManager.DATA_DIR, config.path.raw_data.test)
            logger.info(f"Finish Generating grammar from train set: {train_smile_path}")
            grammar, rule_seq_list_test, partition_list_test = generate_grammar_from_smiles(test_smile_path, vol_path, lifting_type, 
                                    grammar=grammar, # train and test on the same grammar
                                    num_processes=args.workers, cache_size=cache_size)
            sequences = {'train': rule_seq_list_train, 'test': rule_seq_list_test}
            partitions = {'train': partition_list_train, 'test': partition_list_test}

            # 准备给后面的reconstruct用
            smile_path = train_smile_path
            rule_seq_list = rule_seq_list_train
            
        

        else:
            raise ValueError(f"Unsupported generation dataset: {data_name}")

    groups = rule_seq_list if data_name not in ('moses', 'guacamol', 'test') else rule_seq_list_train + rule_seq_list_test
    if not groups or any(seq is None for seq in groups):
        raise RuntimeError("Grammar construction failed for some rows; do not use this artifact for training")

    # test reconstruction of original SMILES using learned grammar
    if args.reconstruct:
        org_smile_list = load_smiles(smile_path)
        failure_rate = reconstruct_from_grammar(grammar, rule_seq_list, org_smile_list, n_workers=args.workers)
        if failure_rate:
            raise RuntimeError(f"Reconstruction failed for {failure_rate:.2%} of input molecules")
        if isinstance(sequences, dict):
            failure_rate = reconstruct_from_grammar(
                grammar, sequences['test'], load_smiles(test_smile_path), n_workers=args.workers)
            if failure_rate:
                raise RuntimeError(f"Test reconstruction failed for {failure_rate:.2%} of molecules")

    dump_pickle(output, (grammar, sequences))
    dump_pickle(partitions_output, partitions)
    logger.info("Grammar saved to %s (%s rules)", output, grammar.num_prod_rule)

    if args.output_config:
        config.path.grammar_path = output
        os.makedirs(os.path.dirname(os.path.abspath(args.output_config)), exist_ok=True)
        with open(args.output_config, 'x', encoding='utf-8') as handle:
            yaml.safe_dump(json.loads(json.dumps(config)), handle, sort_keys=False)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default='configs/moses/gvae_rsg.yaml',
        help='YAML path relative to the repository root (configs/...) or an absolute path',
    )
    parser.add_argument('--workers', type=int, default=164, help='Number of workers for decoding')
    parser.add_argument('--cache_size', type=int, default=None, help='Cache size for lifting grammar')
    parser.add_argument('--lifting_type', type=str, default='RSG', help='Lifting strategy: MIG or RSG (the two paper-reported strategies). MEG is the no-anchor-resolution frequency-merging base variant, retained for ablation/reproducibility only.')
    parser.add_argument('--output-config', help='Write a new training YAML pointing to the generated grammar')
    parser.add_argument('--output', help='Output grammar path; defaults to RSG_<dataset>_rebuilt.pklz under the dataset directory; never overwrites')
    parser.add_argument('--reconstruct', action=argparse.BooleanOptionalAction, default=True, help='Check reconstruction (--no-reconstruct to skip)')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    config = load_config(args.config)
    load_seed(config.seed)
    env_cfg = init_exp(config)

    wandb.init(
        entity=env_cfg.wandb.entity,  # team name
        project="HoGen",  # project name
        name=f'CreatingGrammar-{config.data.name}-{args.lifting_type}-' + datetime.now().strftime('%y%m%d-%H:%M%S'),  # experiment name
        dir=PathManager.WANDB_DIR,
        config=vars(config), # Track hyperparameters and run metadata.
    )



    # --------------- Grammar learning ---------------
    learn(config, args)
