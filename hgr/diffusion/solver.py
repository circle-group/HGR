# diffusion/solver.py

import torch
from tqdm import tqdm
from hgr.diffusion.sde import _SDES
from hgr.diffusion.losses import get_score_fn
from hgr.utils.debug_utils import _DEBUG_


_PREDICTORS = {}
_CORRECTORS = {}

def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        local_name = cls.__name__ if name is None else name
        assert local_name not in _PREDICTORS, ValueError(f'Already registered predictor with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        local_name = cls.__name__ if name is None else name
        assert local_name not in _CORRECTORS, ValueError(f'Already registered corrector with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


@register_predictor(name='Euler')
class EulerMaruyamaPredictor():
  def __init__(self, sde, score_fn,  probability_flow=False):
    # Compute the reverse SDE/ODE
    self.dt = - torch.tensor(sde.dt)
    self.rsde  = sde.reverse(probability_flow)
    self.score_fn = score_fn

  def update_fn(self, x, t):

    score_x = self.score_fn(x, t=t)

    drift_x, diffusion_x = self.rsde.sde(x, score_x, t)
    z_x = torch.randn_like(x)
    x_mean = x + drift_x * self.dt
    x = x_mean + diffusion_x[:, None] * torch.sqrt(-self.dt) * z_x
    return x, x_mean


@register_corrector(name='None')
class NoneCorrector():
    def __init__(self, *args, **kwargs):
        pass

    def update_fn(self, x, t):
        return x, x

@register_corrector(name='Langevin')
class LangevinCorrector():
  def __init__(self, sde, score_fn, snr, n_steps, scale_eps):

    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.scale_eps = scale_eps
    self.n_steps = n_steps

  def update_fn(self, x, t):
    timestep = (t * (self.sde.N - 1) / self.sde.T).long()
    alpha = self.sde.alphas.to(t.device)[timestep]

    for _ in range(self.n_steps):
      score_x = self.score_fn(x, t=t)

      noise = torch.randn_like(x)
      grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
      # grad_norm[grad_norm == 0] = 1.0 #TODO: 判断一下这个补丁对不对
      noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
      step_size = (self.snr * noise_norm / grad_norm) ** 2 * 2 * alpha
      x_mean = x + step_size[:, None] * score_x
      x = x_mean + torch.sqrt(step_size * 2)[:, None] * noise * self.scale_eps

    return x, x_mean



def load_pc_sampler(config, device='cuda', eps=1e-3, disable_tqdm=False):

    snr = config.sample.snr
    denoise = config.sample.noise_removal
    n_steps = config.sample.n_steps
    scale_eps = config.sample.scale_eps
    probability_flow = config.sample.probability_flow
    bs = config.eval.batch_size

    diff_steps = config.sde.num_scales
    latent_sde = _SDES[config.sde.type](config.sde)

    timesteps = torch.linspace(latent_sde.T, eps, diff_steps, device=device)
    x_shape = (bs, config.scorenet.latent_dim)
    predictor = _PREDICTORS[config.sample.predictor]
    corrector = _CORRECTORS[config.sample.corrector]

    def pc_sampler(model):
        score_fn = get_score_fn(latent_sde, model, train=False)
        predictor_obj = predictor(latent_sde, score_fn, probability_flow)
        corrector_obj = corrector(latent_sde, score_fn, snr, n_steps, scale_eps)

        with torch.no_grad():
            x = torch.randn(*x_shape, device=device)

            for i in tqdm(range(0, diff_steps), desc='[PC Sampling]', leave=False, disable=not _DEBUG_):
            # for i in range(0, diff_steps):
                vec_t = timesteps[i].expand(bs)
                if config.sample.corrector != 'None':
                    x, _ = corrector_obj.update_fn(x, vec_t)

                x, x_mean = predictor_obj.update_fn(x, vec_t)
        return x_mean if denoise else x

    return pc_sampler
