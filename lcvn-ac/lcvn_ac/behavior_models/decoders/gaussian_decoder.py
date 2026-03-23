import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from omegaconf import ListConfig, OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import lcvn_ac
from lcvn_ac.behavior_models.decoders.action_decoder import ActionDecoder
from lcvn_ac.behavior_models.actor_critic.mlp import MLP

logger = logging.getLogger(__name__)


class GaussianDecoder(ActionDecoder):
    def __init__(
        self,
        perceptual_features: int,
        latent_goal_features: int,
        plan_features: int,
        hidden_size: int,
        layers: int,
        layer_norm: bool,
        activation: str,
        out_features: int,
        log_std_min: float,
        log_std_max: float,
        action_scale: Optional[List[float]] = None,
    ):
        super(GaussianDecoder, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.plan_features = plan_features

        # Robust action scaling strategy
        if action_scale is None:
            # Default to robust values covering typical ranges
            # dx, dy, dyaw ~ 0.3 -> set to 0.5
            # dt ~ 0.1-0.27 -> set to 0.5
            self.action_scale_values = [0.5, 0.5, 0.5, 0.5]
            logger.info(f"GaussianDecoder using default robust action scales: {self.action_scale_values}")
        else:
            if isinstance(action_scale, ListConfig):
                action_scale = OmegaConf.to_container(action_scale, resolve=True)
            self.action_scale_values = list(action_scale)
            logger.info(f"GaussianDecoder using provided action scales: {self.action_scale_values}")
        
        # Register as buffer to move with device, but not a learnable parameter
        self.register_buffer('scale', torch.tensor(self.action_scale_values, dtype=torch.float))

        in_features = plan_features + perceptual_features + latent_goal_features
        self.out_features = out_features  # for discrete gripper act

        self.fc = MLP(in_features, None, hidden_size, layers, layer_norm, activation)

        self.actor_mean = MLP(in_features, self.out_features, hidden_size, 0, layer_norm, activation)
        self.actor_std = MLP(in_features, self.out_features, hidden_size, 0, layer_norm, activation)

    def _sample(self, log_stds: Tensor, means: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        action_stds = torch.exp(log_stds)
        a_dist = torch.distributions.Normal(means, action_stds)
        z = a_dist.rsample()

        # Scale actions using the registered buffer
        # self.scale is automatically on the correct device
        actions = torch.tanh(z) * self.scale
        entropy = a_dist.entropy()
        # Log probability with tanh squashing correction (standard SAC formula)
        log_prob = a_dist.log_prob(z) - torch.log(1 - torch.tanh(z).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)  # (B,)
        return actions, entropy, log_prob

    def forward(  # type: ignore
        self,
        latent_current: Tensor,
        latent_goal: Tensor,
        latent_plan: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

        if latent_plan is not None:
            x = torch.cat([latent_current, latent_goal, latent_plan], dim=-1)  # b, s, (plan + visuo-propio + goal)
        else:
            x = torch.cat([latent_current, latent_goal], dim=-1)

        x = self.fc(x)

        means = self.actor_mean(x)
        log_stds = self.actor_std(x)

        log_stds = torch.tanh(log_stds)
        log_stds = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_stds + 1)

        return log_stds, means

    def get_action(
        self,
        latent_current: Tensor, # [size, 2048]
        latent_goal: Tensor, # [size, 32]
        latent_plan: Optional[Tensor] = None, # [size, 1024]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        log_stds, means = self(latent_current, latent_goal, latent_plan)
        actions, entropy, log_prob = self._sample(log_stds, means)
        return actions, entropy, log_prob, means  # [size, 4], [size, 4], [size], [size, 4]
