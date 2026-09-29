import os
import torch
import logging
from hgr.diffusion.models.utils import _MODELS
from hgr.utils.ema import ExponentialMovingAverage
from hgr.utils.file_utils import resolve_ckpt_path
# Do not delete the following packages
from hgr.diffusion.models.vanillaScoreNet import vanillaScoreNet
from hgr.diffusion.models.ScoreNet import ScoreNet
from hgr.diffusion.models.ScoreDiT import ScoreDiT

logger = logging.getLogger(__name__)

def load_scorenet(model_cfg, device):
    """Create the score model."""
    model_name = model_cfg.type
    score_model = _MODELS[model_name](model_cfg).to(device)
    return score_model

def load_scorenet_from_ckpt(ckpt_path, device):
    """Load a checkpoint for the score model."""
    original_path = ckpt_path
    ckpt_path = resolve_ckpt_path(ckpt_path)
    if not os.path.isabs(original_path):
        logger.info("Relative checkpoint path resolved to %s", ckpt_path)

    assert os.path.exists(ckpt_path), f"No checkpoint found at {ckpt_path}"
    # Diffusion checkpoints store config objects, so PyTorch 2.6+'s
    # weights_only=True default breaks loading unless we opt out.
    loaded_state = torch.load(ckpt_path, map_location=device, weights_only=False)

    loaded_config = loaded_state['config']
    score_net = load_scorenet(loaded_config.scorenet, device)
    score_net.load_state_dict(loaded_state['scorenet'])

    ema = ExponentialMovingAverage(score_net.parameters(), decay=loaded_config.train.ema_decay)
    ema.load_state_dict(loaded_state['ema'])
    ema.copy_to(score_net.parameters())

    logger.info(f"Loaded score model from {ckpt_path}")

    return score_net, loaded_config
