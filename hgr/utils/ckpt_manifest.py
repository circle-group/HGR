import os
import yaml
import torch
import logging
from functools import lru_cache

from hgr.utils.file_utils import PathManager, resolve_ckpt_path

logger = logging.getLogger(__name__)


def _resolve_manifest_path(manifest_path: str) -> str:
    if os.path.isabs(manifest_path):
        return manifest_path

    candidates = [
        os.path.join(PathManager.CFG_ROOT, manifest_path),
        os.path.join(PathManager.PRO_ROOT, manifest_path),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    return candidates[0]


@lru_cache(maxsize=8)
def load_gvae_ckpt_manifest(manifest_path: str):
    resolved_path = _resolve_manifest_path(manifest_path)
    if not os.path.isfile(resolved_path):
        raise FileNotFoundError(f"GVAE ckpt manifest not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        manifest = yaml.safe_load(f)

    entries = manifest.get("entries", [])
    if not entries:
        raise ValueError(f"Manifest {resolved_path} does not contain any entries.")

    entries = sorted(entries, key=lambda e: (float(e["alpha"]), int(e["frontier_index"])))
    return resolved_path, manifest, entries


@lru_cache(maxsize=128)
def load_gvae_ckpt_latent_dim(ckpt_path: str) -> int:
    resolved_ckpt_path = resolve_ckpt_path(ckpt_path)
    if not os.path.isfile(resolved_ckpt_path):
        raise FileNotFoundError(f"GVAE checkpoint not found: {resolved_ckpt_path}")

    loaded_state = torch.load(resolved_ckpt_path, map_location="cpu", weights_only=False)
    gvae_cfg = getattr(loaded_state.get("config"), "gvae", None)
    latent_dim = getattr(gvae_cfg, "latent_dim", None)
    if latent_dim is None:
        raise ValueError(f"GVAE latent_dim not found in checkpoint config: {resolved_ckpt_path}")
    return int(latent_dim)


def select_nearest_gvae_ckpt_entry(entries, target_alpha: float):
    clipped_alpha = max(0.0, min(1.0, float(target_alpha)))
    entry = min(
        entries,
        key=lambda e: (abs(float(e["alpha"]) - clipped_alpha), int(e["frontier_index"])),
    )
    return clipped_alpha, entry


def resolve_gvae_ckpt_manifest_selection(config):
    path_cfg = getattr(config, "path", None)
    if path_cfg is None:
        return None

    if not getattr(path_cfg, "gvae_ckpt_use_manifest", False):
        return None

    manifest_path = getattr(path_cfg, "gvae_ckpt_manifest_path", None)
    if not manifest_path:
        raise ValueError("path.gvae_ckpt_use_manifest=True but path.gvae_ckpt_manifest_path is empty.")

    requested_alpha = getattr(path_cfg, "gvae_ckpt_alpha", None)
    if requested_alpha is None:
        raise ValueError("path.gvae_ckpt_use_manifest=True but path.gvae_ckpt_alpha is empty.")

    resolved_manifest_path, manifest, entries = load_gvae_ckpt_manifest(str(manifest_path))
    clipped_alpha, entry = select_nearest_gvae_ckpt_entry(entries, float(requested_alpha))

    path_cfg.gvae_ckpt_manifest_path = manifest_path
    path_cfg.gvae_ckpt_manifest_resolved_path = resolved_manifest_path
    path_cfg.gvae_ckpt_alpha_requested = float(requested_alpha)
    path_cfg.gvae_ckpt_alpha = clipped_alpha
    path_cfg.gvae_ckpt_alpha_resolved = float(entry["alpha"])
    path_cfg.gvae_ckpt_id = entry["ckpt_id"]
    path_cfg.gvae_ckpt_frontier_index = int(entry["frontier_index"])
    path_cfg.gvae_ckpt_is_baseline = bool(entry["baseline"])
    path_cfg.gvae_ckpt_source_dir = entry["source_dir"]
    path_cfg.gvae_ckpt_path = entry["path"]
    path_cfg.gvae_ckpt_recon = float(entry["recon"])
    path_cfg.gvae_ckpt_kld = float(entry["kld"])
    path_cfg.gvae_ckpt_tot = float(entry["tot"])
    path_cfg.gvae_ckpt_epoch = int(entry["epoch"])
    path_cfg.gvae_ckpt_stamp = entry["stamp"]
    path_cfg.gvae_ckpt_manifest_size = len(entries)
    path_cfg.gvae_ckpt_resolved_latent_dim = load_gvae_ckpt_latent_dim(entry["path"])

    scorenet_cfg = getattr(config, "scorenet", None)
    if scorenet_cfg is not None:
        scorenet_cfg.latent_dim = path_cfg.gvae_ckpt_resolved_latent_dim

    logger.info(
        "[gvae-manifest] requested_alpha=%.6f clipped_alpha=%.6f resolved_alpha=%.6f ckpt_id=%s path=%s",
        float(requested_alpha),
        clipped_alpha,
        float(entry["alpha"]),
        entry["ckpt_id"],
        entry["path"],
    )
    logger.info(
        "[gvae-manifest] recon=%.3f kld=%.3f tot=%.3f latent_dim=%d frontier_index=%d baseline=%s manifest=%s",
        float(entry["recon"]),
        float(entry["kld"]),
        float(entry["tot"]),
        path_cfg.gvae_ckpt_resolved_latent_dim,
        int(entry["frontier_index"]),
        bool(entry["baseline"]),
        resolved_manifest_path,
    )

    return entry
