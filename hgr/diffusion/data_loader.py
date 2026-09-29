import os
from copy import deepcopy
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from hgr.utils.file_utils import PathManager, load_pickle, resolve_ckpt_path

import logging

logger = logging.getLogger(__name__)


_LATENT_CACHE_FORMAT_VERSION = 2


def _resolve_gvae_ckpt_path(config):
    gvae_ckpt_path = getattr(config.path, "gvae_ckpt_path", None)
    if not gvae_ckpt_path:
        raise ValueError("config.path.gvae_ckpt_path is empty. Resolve the GVAE manifest before loading latents.")
    return resolve_ckpt_path(gvae_ckpt_path)


def _get_latent_cache_settings(config):
    path_cfg = getattr(config, "path", {})
    mode = str(getattr(path_cfg, "latent_cache_mode", "mu")).lower()
    preprocess = str(getattr(path_cfg, "latent_preprocess", "none")).lower()
    version = str(getattr(path_cfg, "latent_cache_version", "v1"))
    eps = float(getattr(path_cfg, "latent_preprocess_eps", 1e-6))

    if mode not in {"sample", "mu"}:
        raise ValueError(f"Unsupported path.latent_cache_mode: {mode}")
    if preprocess not in {"none", "standardize"}:
        raise ValueError(f"Unsupported path.latent_preprocess: {preprocess}")
    return mode, preprocess, version, eps


def _build_latent_cache_path(config, gvae_ckpt_path):
    mode, preprocess, version, _ = _get_latent_cache_settings(config)
    ckpt_name = os.path.basename(gvae_ckpt_path)
    stem, _ = os.path.splitext(ckpt_name)
    cache_name = f"{stem}-latent-{mode}-{preprocess}-{version}.pt"
    return os.path.join(PathManager.DATA_DIR, "latent", cache_name)


def _load_existing_latent_cache(latent_path):
    loaded = torch.load(latent_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, torch.Tensor):
        return {
            "latents": loaded,
            "metadata": {
                "format_version": 1,
                "latent_cache_mode": "sample",
                "latent_preprocess": "none",
            },
            "preprocess": {},
        }

    if not isinstance(loaded, dict) or "latents" not in loaded:
        raise ValueError(f"Unsupported latent cache payload at {latent_path}")

    loaded.setdefault("metadata", {})
    loaded.setdefault("preprocess", {})
    loaded["metadata"].setdefault("format_version", _LATENT_CACHE_FORMAT_VERSION)
    return loaded


def _validate_latent_cache_bundle(bundle, config, gvae_ckpt_path):
    metadata = bundle.get("metadata", {})
    mode, preprocess, version, _ = _get_latent_cache_settings(config)
    format_version = int(metadata.get("format_version", 1))

    expected = {
        "gvae_ckpt_path": gvae_ckpt_path,
        "latent_cache_mode": mode,
        "latent_preprocess": preprocess,
        "latent_cache_version": version,
    }
    for key, expected_value in expected.items():
        current_value = metadata.get(key)
        if current_value is None:
            if format_version >= _LATENT_CACHE_FORMAT_VERSION:
                raise ValueError(f"Latent cache metadata missing required field: {key}")
            continue
        if current_value != expected_value:
            raise ValueError(
                f"Latent cache metadata mismatch for {key}: expected {expected_value}, found {current_value}"
            )


def _build_latent_cache_bundle(config, device):
    from hgr.gvae.data_loader import get_dataloaders
    from hgr.gvae.loader import load_gvae_ckpt_cfg

    gvae_ckpt_path = _resolve_gvae_ckpt_path(config)
    mode, preprocess, version, eps = _get_latent_cache_settings(config)

    grammar_path = os.path.join(PathManager.DATA_DIR, config.path.grammar_path)
    grammar, prod_rule_seq_list = load_pickle(grammar_path)
    grammar.to(device)

    gvae_state, gvae_cfg = load_gvae_ckpt_cfg(gvae_ckpt_path, device=device)
    from hgr.gvae.model import GrammarSeq2SeqVAE

    mhg_vae = GrammarSeq2SeqVAE(hrg=grammar, cfg=gvae_cfg).to(device)
    mhg_vae.load_state_dict(gvae_state["gvae"])
    mhg_vae.eval()

    latent_cfg = deepcopy(config)
    latent_cfg.gvae = gvae_cfg
    train_loader, _ = get_dataloaders(latent_cfg, prod_rule_seq_list, grammar.num_prod_rule)
    pad_value = latent_cfg.data.padding_idx % mhg_vae.vocab_size
    gvae_bs = int(gvae_cfg.bs)
    needs_pad = hasattr(mhg_vae, "batch_size")

    all_latents = []
    with torch.no_grad():
        for each_batch in tqdm(train_loader, desc="Building latent cache"):
            in_batch = each_batch[0]
            num_pad = gvae_bs - len(in_batch) if needs_pad else 0
            if num_pad:
                pad_shape = (num_pad,) + in_batch.shape[1:]
                in_batch = torch.cat([in_batch, in_batch.new_full(pad_shape, pad_value)], dim=0)
            in_batch = in_batch.to(device)

            mu, logvar = mhg_vae.encode(in_batch)
            latent = mu if mode == "mu" else mhg_vae.reparameterize(mu, logvar)
            if num_pad:
                latent = latent[:-num_pad]
            all_latents.append(latent.cpu())

    latents = torch.cat(all_latents, dim=0)
    preprocess_payload = {}
    if preprocess == "standardize":
        mean = latents.mean(dim=0)
        std = latents.std(dim=0, unbiased=False).clamp_min(eps)
        preprocess_payload = {
            "mean": mean,
            "std": std,
            "eps": eps,
        }

    return {
        "latents": latents,
        "metadata": {
            "format_version": _LATENT_CACHE_FORMAT_VERSION,
            "gvae_ckpt_path": gvae_ckpt_path,
            "latent_cache_mode": mode,
            "latent_preprocess": preprocess,
            "latent_cache_version": version,
            "latent_dim": int(latents.shape[1]),
        },
        "preprocess": preprocess_payload,
    }


def load_latent_cache_bundle(config, device, build_if_missing=True):
    gvae_ckpt_path = _resolve_gvae_ckpt_path(config)
    latent_path = _build_latent_cache_path(config, gvae_ckpt_path)

    if os.path.exists(latent_path):
        bundle = _load_existing_latent_cache(latent_path)
        _validate_latent_cache_bundle(bundle, config, gvae_ckpt_path)
        logger.info(
            "Load latent cache from %s, shape=%s, mode=%s, preprocess=%s",
            latent_path,
            tuple(bundle["latents"].shape),
            bundle["metadata"].get("latent_cache_mode"),
            bundle["metadata"].get("latent_preprocess"),
        )
        return bundle, latent_path

    if not build_if_missing:
        raise FileNotFoundError(f"Latent cache not found: {latent_path}")

    bundle = _build_latent_cache_bundle(config, device)
    os.makedirs(os.path.dirname(latent_path), exist_ok=True)
    torch.save(bundle, latent_path)
    logger.info(
        "Save latent cache to %s, shape=%s, mode=%s, preprocess=%s",
        latent_path,
        tuple(bundle["latents"].shape),
        bundle["metadata"].get("latent_cache_mode"),
        bundle["metadata"].get("latent_preprocess"),
    )
    return bundle, latent_path


def _apply_latent_preprocess(latents, bundle, config, inverse=False):
    _, preprocess, _, _ = _get_latent_cache_settings(config)

    if preprocess == "none":
        return latents

    if preprocess == "standardize":
        mean = bundle["preprocess"].get("mean")
        std = bundle["preprocess"].get("std")
        if mean is None or std is None:
            raise ValueError("Latent cache is missing standardization stats.")
        if inverse:
            mean = mean.to(latents.device)
            std = std.to(latents.device)
            return latents * std + mean
        return (latents - mean) / std

    raise ValueError(f"Unsupported latent preprocess: {preprocess}")


def prepare_latents_for_training(bundle, config):
    return _apply_latent_preprocess(bundle["latents"], bundle, config, inverse=False)


def restore_latents_for_decoding(latents, bundle, config):
    return _apply_latent_preprocess(latents, bundle, config, inverse=True)


def latent_dataloader(config, device):
    bundle, latent_path = load_latent_cache_bundle(config, device, build_if_missing=True)
    train_latents = prepare_latents_for_training(bundle, config)

    config.path.latent_cache_path = latent_path
    config.path.latent_cache_metadata = bundle["metadata"]

    train_ds = TensorDataset(train_latents)
    train_loader = DataLoader(train_ds, batch_size=config.train.bs, shuffle=True, num_workers=8)
    return train_loader
