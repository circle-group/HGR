import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(torch.nn.Module):
    # sinusoidal positional embeddings
    def __init__(self, dim, max_positions=10000):
        super().__init__()
        # magic number 10000 is from transformers
        self.dim = dim
        self.max_positions = max_positions

    def forward(self, x):
        device = x.device
        half_dim = (self.dim+1) // 2 # Adjust half_dim to handle odd dimensions
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) #(5,)
        emb = x[:, None] * emb[None, :] #(80,1,5)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = emb[:, :self.dim]  # Ensure the final dimension matches self.dim
        return emb


def get_act(nonlinearity):
    """Get actiuvation functions."""
    fn_name = nonlinearity.lower()
    if fn_name == 'elu':
        return nn.ELU()
    elif fn_name == 'relu':
        return nn.ReLU()
    elif fn_name == 'lrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif fn_name == 'swish':
        return nn.SiLU()
    elif fn_name == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError(f'activation function {fn_name} does not exist!')