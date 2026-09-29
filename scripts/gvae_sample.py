"""
GVAE prior sampling + evaluation.

Example:

PYTHONHASHSEED=123 python scripts/gvae_sample.py \
  --config configs/ringdiv300k/gvae_rsg.yaml --samples 10000 --seed 42 \
  --ckpt "path.pth"
"""

import os
import sys
import argparse
import logging
from datetime import datetime

PRO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PRO_ROOT not in sys.path:
    sys.path.insert(0, PRO_ROOT)

logger = logging.getLogger(__name__)


def _build_parser():
    p = argparse.ArgumentParser(description="GVAE prior sampling + evaluation")
    p.add_argument(
        "--config", type=str, required=True,
        help="YAML path relative to the repository root (configs/...) or an absolute path",
    )
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=8, help="CPU workers for decoding / metrics")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--skip-eval", action="store_true",
                   help="Skip evaluation, only sample and save SMILES")
    p.add_argument("--metrics", type=str, default=None,
                   help="Comma-separated metric sets (default: config.data.name). "
                        "E.g. 'ringdiv300k,bforman,ollivier'")
    return p



if __name__ == "__main__":
    args = _build_parser().parse_args()

    # Heavy imports deferred so `--help` does not trigger RDKit / ringdiv.
    import textwrap
    import torch
    from rdkit import Chem
    from ringdiv import get_all_metrics
    from hgr.utils.loader import set_env_from_config

    set_env_from_config()
    from hgr.utils.data_splits import load_train_test_smiles
    from hgr.utils.debug_utils import Timer, setup_logging
    from hgr.utils.file_utils import PathManager, ensure_parent_dir, sanitize_filename
    from hgr.utils.loader import init_exp, load_config, load_device, load_seed

    setup_logging()

    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    load_seed(config.seed)
    init_exp(config)
    device = load_device()

    num_samples = args.samples if args.samples is not None else 10000
    grammar_path = os.path.join(PathManager.DATA_DIR, config.path.grammar_path)
    gvae_ckpt_path = args.ckpt if args.ckpt is not None else config.path.gvae_ckpt_path
    if not gvae_ckpt_path:
        raise ValueError("Checkpoint required. Pass --ckpt or set path.gvae_ckpt_path in config.")

    # ── Load decoder ──────────────────────────────────────
    from hgr.gvae.latent2mol import Latent2MolDecoder

    latent_decoder = Latent2MolDecoder(
        grammar_path=grammar_path, gvae_ckpt_path=gvae_ckpt_path,
        device_str=str(device), workers=min(args.workers, os.cpu_count() or 1),
    )
    gvae_cfg = latent_decoder.gvae_cfg

    # ── Sample & decode ───────────────────────────────────
    z_all = torch.randn(int(num_samples * 1.05), gvae_cfg.latent_dim, device=device)

    with Timer("Decode time"):
        gen_mols = latent_decoder(z_all)
    logger.info("Generated %d molecules", len(gen_mols))

    gen_smiles = [Chem.MolToSmiles(mol) for mol in gen_mols]

    # ── Save ──────────────────────────────────────────────
    if args.out_dir is None:
        ckpt_name = sanitize_filename(os.path.basename(gvae_ckpt_path).replace(".pth", ""))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_filename = f"{ckpt_name}_n{len(gen_smiles)}_seed{config.seed}({timestamp}).smi"
        save_parts = [PathManager.RESULTS_ROOT, "samples", "gvae", config.data.name]
        if getattr(config, "exp_name", None):
            save_parts.append(config.exp_name)
        save_dir = os.path.join(*save_parts)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_filename)
    else:
        save_path = args.out_dir
        ensure_parent_dir(save_path)

    with open(save_path, "w") as f:
        for smi in gen_smiles:
            f.write(smi + "\n")
    logger.info("Saved %d SMILES -> %s", len(gen_smiles), os.path.abspath(save_path))

    # ── Evaluate ──────────────────────────────────────────
    if args.skip_eval:
        if hasattr(latent_decoder, "close"):
            latent_decoder.close()
        sys.exit(0)

    try:
        train_smiles, test_smiles = load_train_test_smiles(config)
        with Timer("Evaluate molecules time"):
            cache_dir = os.path.join(PathManager.DATA_DIR, "cache")
            num_eval = 10000 if config.data.name != "moses" else 25000
            gen_rel = get_all_metrics(
                gen_mols,
                test_smiles=test_smiles,
                train_smiles=train_smiles,
                metrics=args.metrics.split(",") if args.metrics else [config.data.name],
                cache_dir=cache_dir,
                n_jobs=int(args.workers),
                num_eval=num_eval,
                device=device,
            )

        table_width = 35
        print("=" * table_width)
        print(f"Evaluate {len(gen_mols)} samples (num_eval={num_eval}):")
        for line in textwrap.wrap(gvae_ckpt_path, width=table_width):
            print(line.center(table_width))
        print("-" * table_width)
        for key, val in gen_rel.items():
            print(f"{key:<12s}: {val:.6f}")
        print("=" * table_width)
    except Exception as exc:
        logger.error("Error evaluating molecules: %s", exc)
    finally:
        if hasattr(latent_decoder, "close"):
            latent_decoder.close()
