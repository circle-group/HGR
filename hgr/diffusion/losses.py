import torch
from hgr.diffusion import sde

def get_score_fn(sde_fn, model, train=True, continuous=True):
    if train:
        model.train()
    else:
        model.eval()

    if isinstance(sde_fn, sde.VPSDE):
        def score_fn(x, t):
            if continuous:
                pred = model(x, t)
                std = sde_fn.marginal_prob(torch.zeros_like(x), t)[1]
            else:
                raise NotImplementedError("Discrete SDE not supported")
            score = -pred / std[:, None]
            return score

    else:
        raise NotImplementedError(f"SDE class {sde_fn.__class__.__name__} not supported.")

    return score_fn


def DenoisingScoreMatching(config, is_train, eps=1e-5):

    reduce_mean = config.train.reduce_mean
    latent_sde = sde._SDES[config.sde.type](config.sde)

    def loss_fn(model, x):
        """ Compute the denoising score matching loss.
            model: A score model.
            x: The input batch of data. (mol fingerprint)
        """
        if is_train:
            model.train()
        else:
            model.eval()

        device = x.device
        bs = x.shape[0]
        t = (torch.rand(bs) * (latent_sde.T - eps) + eps).to(device)  # (128,)

        z = torch.randn_like(x) # [bs, channel]
        mean_x, std_x = latent_sde.marginal_prob(x, t) # mean [bs, channel], std [bs,]
        perturbed_x = mean_x + std_x[:, None] * z

        pred = model(perturbed_x, t)
        # TODO: 看一下这边是否需要除以标准差，后面又乘了
        score = - pred / std_x[:, None]

        losses = torch.square(score * std_x[:, None] + z)
        losses = 0.5 * torch.sum(losses, dim=-1)
        losses = losses.mean()
        return losses

    return loss_fn
