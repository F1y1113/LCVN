import torch
from torch import nn
from typing import Tuple, Union
from lcvn_ac.utils.nn_utils import get_activation, NoNorm


def flatten_batch(x: torch.Tensor, nonbatch_dims=1) -> Tuple[torch.Tensor, torch.Size]:
    # (b1,b2,..., X) => (B, X)
    if nonbatch_dims > 0:
        batch_dim = x.shape[:-nonbatch_dims]
        x = torch.reshape(x, (-1,) + x.shape[-nonbatch_dims:])
        return x, batch_dim
    else:
        batch_dim = x.shape
        x = torch.reshape(x, (-1,))
        return x, batch_dim


def unflatten_batch(x: torch.Tensor, batch_dim: Union[torch.Size, Tuple]) -> torch.Tensor:
    # (B, X) => (b1,b2,..., X)
    x = torch.reshape(x, batch_dim + x.shape[1:])
    return x


class MLP(nn.Module):

    def __init__(
        self, in_dim: int, out_dim: int, hidden_dim: int, hidden_layers: int, layer_norm: bool, activation: str
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.norm = nn.LayerNorm if layer_norm else NoNorm
        self.activation = get_activation(activation)

        layers = []
        for i in range(self.hidden_layers):
            layers += [
                nn.Linear(self.in_dim if i == 0 else self.hidden_dim, self.hidden_dim),
                self.norm(hidden_dim, eps=1e-3),
                self.activation,
            ]
        if self.out_dim is not None:
            layers += [nn.Linear(self.hidden_dim, self.out_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, bd = flatten_batch(x)
        y = self.model(x)
        y = unflatten_batch(y, bd)
        return y
