#!/usr/bin/env python3

from typing import Tuple

import torch
import torch.nn as nn


class CodeNetwork(nn.Module):
    def __init__(
        self,
        activation_function: str,
        dropout_vis_fc: float,
        visual_features: int,
        latent_dim_flat: int,
    ):
        super(CodeNetwork, self).__init__()
        self.act_fn = getattr(nn, activation_function)()

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=latent_dim_flat, out_features=1024), self.act_fn, nn.Dropout(dropout_vis_fc)
        )  # shape: [N, 512]
        self.fc2 = nn.Linear(in_features=1024, out_features=visual_features)  # shape: [N, 64]
        self.ln = nn.LayerNorm(visual_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshape = False
        if len(x.shape) == 3:
            s, b, l = x.shape
            x = x.reshape(-1, l)
            reshape = True
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.ln(x)
        if reshape:
            x = x.reshape(s, b, -1)
        return x
