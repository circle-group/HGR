"""Prepare FM data/grammar and emit a training config, without starting training."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-config', required=True)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--base-config', help='Prepared RingDiv YAML for joint preparation')
    parser.add_argument('--other-config', help='Prepared ZINC2m YAML for joint preparation')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = Path(args.output_config).resolve()
    if output.exists():
        raise FileExistsError(output)

    # Keep --help usable without the scientific runtime installed.
    import yaml
    from hgr.utils.loader import load_config, init_exp, load_seed
    from hgr.utils.file_utils import PathManager
    from hgr.foundation.data_utils.mol_dataset import MoleculeDataset, load_joint_dataset

    config = load_config(args.config)
    init_exp(config)
    load_seed(config.seed)
    config.data.num_workers = args.workers
    config.data.grammar_workers = args.workers
    config.grammar.force_rebuild_rules = False
    config.grammar.vocab_path = str(
        Path(__file__).resolve().parents[1] / 'configs/vocab/ringdiv300k_vocab_1000.txt')
    if '+' in config.data.name:
        if not args.base_config or not args.other_config:
            parser.error('Joint preparation requires --base-config and --other-config')
        from hgr.foundation.data_utils.merge_corpora import merge_two_corpora
        base, other = load_config(args.base_config), load_config(args.other_config)
        if (base.data.name, other.data.name) != ('ringdiv', 'zinc2m'):
            raise ValueError('Expected prepared RingDiv base and ZINC2m other configs')
        merge_dir = Path(PathManager.DATA_ROOT) / 'ringdiv_zinc2m_rebuilt'
        if merge_dir.exists() and any(merge_dir.iterdir()):
            raise FileExistsError(f'Refusing to overwrite merge outputs: {merge_dir}')
        merge_two_corpora(argparse.Namespace(
            base_pklz=base.grammar.grammar_path, other_pklz=other.grammar.grammar_path,
            outdir=str(merge_dir), grammar_type=config.grammar.type,
            base_name='ringdiv', other_name='zinc2m'))
        manifest = json.loads((merge_dir / 'manifest_grammar_merged.json').read_text())
        config.data.dir = [str(merge_dir), base.data.dir, other.data.dir]
        config.grammar.grammar_path = str(merge_dir / manifest['files']['grammar_merged_pklz'])
        config.grammar.pop('rules_path', None)
        dataset = load_joint_dataset(config, config.modalities, config.pretexts)
    else:
        if config.data.name not in ('ringdiv', 'zinc2m'):
            raise ValueError('Single-corpus preparation supports RingDiv and ZINC2m')
        config.data.dir = str(Path(PathManager.DATA_ROOT) / config.data.name)
        processed = Path(config.data.dir) / 'processed'
        if processed.exists() and any(processed.iterdir()):
            raise FileExistsError(
                f'{processed} is not empty. Use a fresh ASSET_ROOT for rebuilding; '
                'existing processed data will not be overwritten or silently reused.')
        raw = Path(config.data.dir) / 'raw' / f'{config.data.name}_property.csv'
        if not raw.is_file():
            raise FileNotFoundError(raw)
        config.grammar.grammar_path = None
        config.grammar.pop('rules_path', None)
        dataset = MoleculeDataset(root=config.data.dir, config=config, mode='pretrain',
                                  modalities=config.modalities, pretexts=config.pretexts)
    if len(dataset) == 0:
        raise RuntimeError('Preparation produced an empty dataset')
    # Reload through the same training dataset interface before publishing the YAML.
    if '+' not in config.data.name:
        dataset = MoleculeDataset(root=config.data.dir, config=config, mode='pretrain',
                                  modalities=config.modalities, pretexts=config.pretexts)
    if len(dataset._smiles) != len(dataset):
        raise RuntimeError('Processed SMILES and graph sample counts differ')
    if len(dataset._rules_slices['rule_seq']) != len(dataset) + 1:
        raise RuntimeError('Rule sequences and graph sample counts differ')
    dataset[0]
    dataset[len(dataset) - 1]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as handle:
        yaml.safe_dump(json.loads(json.dumps(config)), handle, sort_keys=False)
    print(f'Prepared {len(dataset)} samples. Training config: {output}')


if __name__ == '__main__':
    main()
