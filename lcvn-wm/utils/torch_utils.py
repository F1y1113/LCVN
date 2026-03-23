
from typing import Optional
import torch
from torch.types import _size
import torch.nn as nn


def freeze_model(model: nn.Module) -> None:
    """Freeze the torch model"""
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def bernoulli_tensor(
    size: _size,
    p: float,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
):
    """
    Generate a tensor of the given size,
    where each element is sampled from a Bernoulli distribution with probability `p`.
    """
    return torch.bernoulli(torch.full(size, p, device=device), generator=generator)
