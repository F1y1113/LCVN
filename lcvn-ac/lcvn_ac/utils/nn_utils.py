import torch
from torch import nn
from typing import Tuple, Union


def get_activation(activation: str):
    if activation is None:
        return nn.Identity()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softmax":
        return nn.Softmax(dim=1)
    else:
        raise NotImplementedError


def init_weights(m):
    if type(m) in {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if isinstance(m, nn.GRUCell):
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)


def transpose_collate(batch):
    """transposes batch and time dimension
    (B, T, ...) -> (T, B, ...)"""
    from torch.utils.data._utils.collate import default_collate

    return {k: torch.transpose(v, 0, 1) for k, v in default_collate(batch).items()}


def transpose_collate_wm(batch):
    """transposes batch and time dimension
    (B, T, ...) -> (T, B, ...)"""
    from torch.utils.data._utils.collate import default_collate

    collated_batch = default_collate(batch)
    transposed_batch = {}

    fields = ["reset", "robot_obs", "frame"]
    nested_fields = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "state_info": ["robot_obs", "pre_robot_obs"],
        "actions": ["rel_actions", "pre_actions"],
    }

    for key, value in collated_batch.items():
        if key in nested_fields:
            transposed_batch[key] = {}
            for sub_key in nested_fields[key]:
                transposed_batch[key][sub_key] = torch.transpose(value[sub_key], 0, 1)
        elif key in fields:
            transposed_batch[key] = torch.transpose(value, 0, 1)
        else:
            transposed_batch[key] = value

    return transposed_batch


def transpose_collate_ag(batch):
    """transposes batch and time dimension
    (B, T, ...) -> (T, B, ...)"""
    from torch.utils.data._utils.collate import default_collate

    collated_batch = default_collate(batch)
    transposed_batch = {}

    fields = ["reset", "feature", "zero_feature", "clip_s", "clip_g", "vae_latent", "discrete_action", "rel_actions"]
    nested_fields = {
        "state_info": ["robot_obs"],
        "actions": ["rel_actions"],
    }

    for key, value in collated_batch.items():
        if key in nested_fields:
            transposed_batch[key] = {}
            for sub_key in nested_fields[key]:
                transposed_batch[key][sub_key] = torch.transpose(value[sub_key], 0, 1)
        elif key in fields:
            transposed_batch[key] = torch.transpose(value, 0, 1)
        else:
            transposed_batch[key] = value

    return transposed_batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


class NoNorm(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
