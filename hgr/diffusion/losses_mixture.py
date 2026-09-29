from hgr.diffusion.mix import DiffusionMixture
import torch


def load_mix(cfg):
    mix_type = cfg.type
    num_scales = cfg.num_scales
    drift_coeff = cfg.drift_coeff
    sigma_0 = cfg.sigma_0
    sigma_1 = cfg.sigma_1
    mix = DiffusionMixture(bridge=mix_type, drift_coeff=drift_coeff,
                            sigma_0=sigma_0, sigma_1=sigma_1, N=num_scales)
    return mix

def MixtureMatching(config, is_train=True):
    reduce_mean = config.train.reduce_mean
    mixture = load_mix(config.sde)

    eps = config.sde.eps
    loss_type = config.train.loss_type

    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: torch.sum(*args, **kwargs)

    def compute_loss(pred, target, loss_coeff):
        losses = torch.square((pred - target) * loss_coeff[:, None])
        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * 0.5
        return losses

    def get_loss_coeff(sde, t, loss_type):
        if loss_type == 'default':
            loss_coeff = sde.loss_coeff(t)
        elif loss_type == 'const':
            loss_coeff = torch.ones_like(t)
        else:
            raise NotImplementedError(f'Loss type: {loss_type} not implemented.')
        return loss_coeff

    def loss_fn(model, x, prior_samples=None):
        if is_train:
            model.train()
        else:
            model.eval()

        sde = mixture.bridge(x) # 感觉可以把mixture也写成sde的形式，这边可以简化

        if prior_samples is None:
            x0 = sde.prior_sampling(x.shape, x.device)
        else:
            x0 = prior_samples

        bs = x.shape[0]
        t = torch.rand(bs, device=x.device) * (sde.T - eps)

        mean_x, std_x = sde.marginal_prob(x0, t)
        xt = mean_x + std_x[:, None] * torch.randn_like(x)

        pred = model(xt, t)

        loss_coeff = get_loss_coeff(sde, t, loss_type)

        losses = compute_loss(pred, x, loss_coeff)
        return torch.mean(losses)
    return loss_fn


