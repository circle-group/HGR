# sampler.py

import os
import torch
import logging
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from hgr.gvae.loader import load_gvae_ckpt_cfg
from hgr.utils.loader import load_device, load_seed
from hgr.utils.file_utils import PathManager, load_pickle
from hgr.utils.debug_utils import Timer, _DEBUG_

from hgr.diffusion import solver
from hgr.diffusion import solver_mixture
from hgr.diffusion.loader import load_scorenet_from_ckpt
from hgr.diffusion.data_loader import load_latent_cache_bundle, restore_latents_for_decoding
# from hgr.diffusion.data_loader import load_train_test_smiles
from hgr.utils.data_splits import load_train_test_smiles
from ringdiv import get_all_metrics

logger = logging.getLogger(__name__)


_CKPT_PATH_FIELDS = (
    "gvae_ckpt_path",
    "latent_cache_mode",
    "latent_preprocess",
    "latent_cache_version",
    "latent_preprocess_eps",
)


def _sync_path_fields_from_ckpt(config, loaded_cfg):
    loaded_path_cfg = getattr(loaded_cfg, "path", None)
    if loaded_path_cfg is None:
        return

    for field in _CKPT_PATH_FIELDS:
        loaded_value = getattr(loaded_path_cfg, field, None)
        if loaded_value is None:
            continue
        current_value = getattr(config.path, field, None)
        if current_value != loaded_value:
            logger.info(
                "Overriding config.path.%s from ScoreNet checkpoint: %s -> %s",
                field,
                current_value,
                loaded_value,
            )
        setattr(config.path, field, loaded_value)


def _load_latent_decoder(grammar_path, gvae_ckpt_path, device_str, workers):
    _, _ = load_gvae_ckpt_cfg(gvae_ckpt_path, device=torch.device(device_str))
    from hgr.gvae.latent2mol import Latent2MolDecoder

    return Latent2MolDecoder(
        gvae_ckpt_path=gvae_ckpt_path,
        grammar_path=grammar_path,
        device_str=device_str,
        workers=workers,
    )





class Sampler:
    def __init__(self, config, device=None, decoder_workers=None):
        """
        并行所需参数: self._gvae_ckpt_path, self._grammar_path, self._gvae_cfg, self._device_str
        """
        load_seed(config.seed)
        self.device = load_device() if device is None else device
        self.eval_metric = getattr(config.eval, 'metrics', 'basic')

        # Load ScoreNet from ckpt
        if config.path.scorenet_ckpt_path is not None:
            self.score_net, loaded_cfg = load_scorenet_from_ckpt(config.path.scorenet_ckpt_path, self.device)
            config.scorenet, config.sde = loaded_cfg.scorenet, loaded_cfg.sde
            _sync_path_fields_from_ckpt(config, loaded_cfg)


        # 加载采样函数 and 数据集
        sde_type = config.sde.type.lower()
        if sde_type in ['vpsde']:
            self.sampling_fn = solver.load_pc_sampler(config, self.device)
        elif sde_type in ['ou']:
            self.sampling_fn = solver_mixture.load_pc_sampler(config, self.device)


        # self.sampling_fn = solver.load_pc_sampler(config, self.device)
        self.train_smiles, self.test_smiles = load_train_test_smiles(config)


        # Latent2MolDecoder 内部会加载grammar -> gvae
        cpu_cores = os.cpu_count() or 1
        decoder_workers = min(decoder_workers or config.eval.get('workers', cpu_cores), cpu_cores)
        grammar_path = os.path.join(PathManager.DATA_DIR, config.path.grammar_path)

        logger.info("Sampling with ScoreNet checkpoint: %s", config.path.scorenet_ckpt_path)
        logger.info("Sampling with GVAE checkpoint: %s", config.path.gvae_ckpt_path)
        logger.info("Sampling with grammar: %s", grammar_path)

        self.latent_decoer = _load_latent_decoder(
            gvae_ckpt_path=config.path.gvae_ckpt_path,
            grammar_path=grammar_path,
            device_str=str(self.device),
            workers=decoder_workers,
        )

        config.gvae = self.latent_decoer.gvae_cfg

        # Only preprocess-aware runs need the latent cache bundle at sampling time.
        self.latent_bundle = None
        latent_preprocess = str(getattr(config.path, "latent_preprocess", "none")).lower()
        if latent_preprocess != "none":
            self.latent_bundle, _ = load_latent_cache_bundle(config, self.device, build_if_missing=True)

        self.config = config
        logger.debug(f"Initialized Sampler (device={self.device}, using up to {decoder_workers} workers).")


    def sample(self, score_net=None, return_samples=False):
        score_net = self.score_net if score_net is None else score_net

        sampling_rounds = int(np.ceil(self.config.eval.num_samples / self.config.eval.batch_size))
        all_latents = []
        for _ in tqdm(range(sampling_rounds), desc="Sampling rounds", disable=not _DEBUG_):
            x = self.sampling_fn(score_net)
            all_latents.append(x)
        all_latent = torch.cat(all_latents, dim=0)

        # Inverse preprocess (e.g. de-standardize) before decoding
        if self.latent_bundle is not None:
            all_latent = restore_latents_for_decoding(all_latent, self.latent_bundle, self.config)

        gen_mols = self.latent_decoer(all_latent)#[:self.config.eval.num_samples]


        logger.debug(f"Generated molecules: {len(gen_mols)}, starting evaluation")

        # Skip expensive metric eval when caller only wants SMILES (e.g. two-stage GPU-sample + CPU-eval pipeline)
        if getattr(self.config.eval, 'skip_eval', False):
            results = {}
            logger.info("config.eval.skip_eval=True → skipping metric evaluation, only returning SMILES.")
        else:
            with Timer("Evaluate molecules time"):
                results = get_all_metrics(gen=gen_mols, test_smiles=self.test_smiles, train_smiles=self.train_smiles,
                                    cache_dir=os.path.join(PathManager.DATA_DIR, 'cache'),
                                    num_eval=self.config.eval.num_samples,
                                    metrics=self.eval_metric,  n_jobs=8, device=self.device)
        if return_samples:
            gen_smiles = [Chem.MolToSmiles(mol) for mol in gen_mols]
            return results, gen_smiles
        else:
            return results
