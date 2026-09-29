from __future__ import annotations

import torch
from torch import nn


class GRUDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.model.flatten_parameters()

    def init_hidden(self, device: torch.device, batch_size: int):
        return {
            "h": torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim,
                device=device,
            )
        }

    def forward(self, tgt_emb_in: torch.Tensor, hidden_dict: dict[str, torch.Tensor]):
        tgt_emb_out, h_n = self.model(tgt_emb_in, hidden_dict["h"])
        return tgt_emb_out, {"h": h_n}

