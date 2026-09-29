import os
import torch
import logging
from hgr.utils.file_utils import resolve_ckpt_path
logger = logging.getLogger(__name__)


# Canonical GVAE type string is "GVAE". Legacy checkpoints saved under the
# previous package layout embedded type "GVAEV4"; accept both on load.
_GVAE_TYPE_ALIASES = {"GVAE", "GVAEV4"}
_GVAE_TYPE_CANONICAL = "GVAE"


def _normalize_gvae_type(raw):
    """Return canonical GVAE type string, accepting legacy aliases.

    Valid values: ``"GVAE"`` or legacy ``"GVAEV4"`` (both map to ``"GVAE"``).
    Any other value (including ``None``, which is how V1 configs left it)
    raises, so archived V1 checkpoints fail early with a clear message
    rather than deeper down the load path.
    """
    if raw is None:
        raise ValueError(
            "Checkpoint is missing gvae.type; this is a V1 signature. V1 "
            "checkpoints are archived under archive/legacy_gvae_v1/ and are "
            "no longer loadable through the current pipeline."
        )
    s = str(raw).upper()
    if s not in _GVAE_TYPE_ALIASES:
        raise ValueError(
            f"Unsupported GVAE type {raw!r}; V1 checkpoints are archived under "
            "archive/legacy_gvae_v1/."
        )
    return _GVAE_TYPE_CANONICAL


def load_gvae_ckpt_cfg(ckpt_path, device='cpu', expected_type=None):
    """从 checkpoint 加载模型状态并将保存的超参数同步到 config。

    Normalizes ``gvae_cfg.type`` to the canonical value so downstream code can
    compare against a single string. ``expected_type`` (if provided) is
    normalized by the same rule, so callers may still pass legacy aliases.
    """
    original_path = ckpt_path
    ckpt_path = resolve_ckpt_path(ckpt_path)
    if not os.path.isabs(original_path):
        logger.info("Relative checkpoint path resolved to %s", ckpt_path)

    loaded_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    logger.info("Load GVAE checkpoint and config from %s", ckpt_path)

    gvae_cfg = getattr(loaded_state['config'], 'gvae', None)
    if gvae_cfg is None:
        raise ValueError("GVAE config not found in checkpoint")

    # Normalize type (validates + collapses legacy aliases)
    canonical_type = _normalize_gvae_type(getattr(gvae_cfg, "type", None))
    try:
        gvae_cfg.type = canonical_type
    except (AttributeError, TypeError):
        # Config may be an immutable-ish object; fall back to setattr best-effort.
        try:
            setattr(gvae_cfg, "type", canonical_type)
        except Exception:
            logger.debug("Could not write back canonical gvae.type=%s on loaded config", canonical_type)

    if expected_type is not None:
        expected_canonical = _normalize_gvae_type(expected_type)
        if canonical_type != expected_canonical:
            raise AssertionError(
                f"Checkpoint at {ckpt_path}: expected gvae.type={expected_canonical!r}, "
                f"got {canonical_type!r}"
            )
    return loaded_state, gvae_cfg
