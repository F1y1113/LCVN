import torch.nn as nn
from typing import Tuple
from torch import Tensor
from lcvn_ac.behavior_models.actor_critic.mlp import MLP


class Critic(nn.Module):
    """
    Critic (Value) Model.
    """

    def __init__(
        self,
        perceptual_features: int,
        latent_goal_features: int,
        plan_features: int,
        hidden_dim: int,
        layers: int,
        layer_norm: bool,
        activation: str,
    ):
        super().__init__()

        # 2048 + 32 + 1024 = 3104
        in_features = perceptual_features + latent_goal_features + plan_features

        self.critic = MLP(in_features, 1, hidden_dim, layers, layer_norm, activation)
        self.critic_target = MLP(in_features, 1, hidden_dim, layers, layer_norm, activation)
        self.critic_target.requires_grad_(False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        state_values = self.critic(x)
        target_values = self.critic_target(x)

        return (state_values, target_values)

    def update_critic_target(self):
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)
