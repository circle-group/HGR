

import torch
import numpy as np
import abc
from tqdm import trange, tqdm
import math
from hgr.diffusion.losses_mixture import load_mix



class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""
  def __init__(self, mix, drift_fn):
    super().__init__()
    self.mix = mix
    self.drift_fn = drift_fn

  @abc.abstractmethod
  def update_fn(self, z, t):
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""
  def __init__(self, mix, drift_fn, snr, scale_eps, n_steps):
    super().__init__()
    self.mix = mix
    self.drift_fn = drift_fn
    self.snr = snr
    self.scale_eps = scale_eps
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, z, t):
    pass



# -------- Solve from time 0 to 1 --------
class EulerMaruyamaPredictor(Predictor):
  def __init__(self, mix, drift_fn):
    super().__init__(mix, drift_fn)

  def update_fn(self, x, t):
    dt = min(1. / self.mix.N, 1 - t[0].item())
    diffusion_x = self.mix.diffusion(t)
    drift_x = self.drift_fn(x, t)

    x_mean = x + drift_x * dt
    x = x_mean + diffusion_x[:, None] * np.sqrt(dt) * torch.randn_like(x)
    return x, x_mean


class NoneCorrector(object):
  """An empty corrector that does nothing."""
  def __init__(self, mix, drift_fn, snr, scale_eps, n_steps):
    super().__init__()

  def update_fn(self, x, t):
    return x, x



class LangevinCorrector(Corrector):
  """A Langevin-like corrector. Only used for Planar dataset.
  """
  def __init__(self, mix, drift_fn, snr, scale_eps, n_steps):
    super().__init__(mix, drift_fn, snr, scale_eps, n_steps)

  # -------- Use un-scaled drift & diffusion --------
  def score_from_drift(self, mix, drift, t):
    """Use the drift as the score.
    """
    diffusion = mix.diffusion(t)
    time_scaled = 1./(diffusion**2)
    return drift * time_scaled[:, None]

  def correct_one_step(self, mix, drift, noise, z, t):
    alpha = torch.ones_like(t)
    grad = self.score_from_drift(mix, drift, t)
    grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
    noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
    step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
    z_mean = z + step_size[:, None] * grad
    z = z_mean + torch.sqrt(step_size * 2)[:, None] * noise * self.scale_eps
    return z, z_mean

  def update_fn(self, x, t):
    for i in range(self.n_steps):
      drift_x, drift_adj = self.drift_fn(x, t)
      noise_x = torch.randn_like(x)
      x, x_mean = self.correct_one_step(self.mix.x, drift_x, noise_x, x, t)

    return x, x_mean


def load_predictor(predictor, mix, drift_fn):
  PREDICTORS = {
    'Euler': EulerMaruyamaPredictor }
  predictor_fn = PREDICTORS[predictor]
  predictor_obj = predictor_fn(mix, drift_fn)
  return predictor_obj


def load_corrector(corrector, mix, drift_fn, snr, scale_eps, n_steps=1):
  CORRECTORS = {
    'None': NoneCorrector,
    'Langevin': LangevinCorrector }
  corrector_fn = CORRECTORS[corrector]
  corrector_obj = corrector_fn(mix, drift_fn, snr, scale_eps, n_steps)
  return corrector_obj


def get_drift_fn(mix, model):
    model.eval()

    def get_drift_from_pred(mix, pred, z, t):
        drift = pred
        bridge = mix.bridge(0)
        if 'BB' in mix.bridge_type:
            drift = bridge.drift_time_scaled(t)[:, None] * (drift - z)
        elif 'OU' in mix.bridge_type:
            var = bridge.variance(t)
            a_t1 = bridge.a_ou(t, torch.ones_like(t))
            gamma = var * a_t1 * bridge.a_over_v(t)
            drift = (bridge.alpha_t(t) * var)[:, None] * z + gamma[:, None] * (drift / a_t1[:, None] - z)
        else:
            raise NotImplementedError(f'Bridge type: {mix.bridge_type} not implemented.')
        return drift

    def drift_fn(x, t):
        pred_x = model(x, t)
        drift_x = get_drift_from_pred(mix, pred_x, x, t)

        return drift_x

    return drift_fn

def load_pc_sampler(config, device='cuda', eps=1e-3):
    mix = load_mix(config.sde)

    bs = config.eval.batch_size
    shape = (bs, config.scorenet.latent_dim)
    cfg_sampler = config.sample


    denoise = config.sample.noise_removal

    def pc_sampler(model, prior_samples=None):
        drift_fn = get_drift_fn(mix, model)
        predictor_obj = load_predictor(cfg_sampler.predictor, mix, drift_fn)
        corrector_obj = load_corrector(cfg_sampler.corrector, mix, drift_fn, cfg_sampler.snr,
                                       cfg_sampler.scale_eps, cfg_sampler.n_steps)

        with torch.no_grad():
            if prior_samples is None:
                x  = torch.randn(*shape, device=device)
            else:
                x = prior_samples

            steps = mix.N
            T = mix.bridge(0).T
            timesteps = torch.linspace(0, T - eps, steps, device=device)

            for i in range(0, steps):
                vect_t = timesteps[i].expand(bs)
                if config.sample.corrector != 'None':
                    x, _ = corrector_obj.update_fn(x, vect_t)
                x, x_mean = predictor_obj.update_fn(x, vect_t)

        return x_mean if denoise else x
    return pc_sampler




