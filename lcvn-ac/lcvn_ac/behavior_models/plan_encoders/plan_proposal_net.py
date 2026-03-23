#!/usr/bin/env python3
import torch
import torch.nn as nn

from lcvn_ac.utils.distributions import Distribution, State


class PlanProposalNetwork(nn.Module):
    def __init__(
        self,
        perceptual_features: int,
        latent_goal_features: int,
        plan_features: int,
        activation_function: str,
        hidden_size: int,
        dist: Distribution,
    ):
        super(PlanProposalNetwork, self).__init__()
        self.perceptual_features = perceptual_features
        self.latent_goal_features = latent_goal_features
        self.plan_features = plan_features
        self.hidden_size = hidden_size
        self.in_features = self.perceptual_features + self.latent_goal_features
        self.act_fn = getattr(nn, activation_function)()
        self.dist = dist
        self.fc_model = nn.Sequential(
            nn.Linear(in_features=self.in_features, out_features=hidden_size),  # shape: [N, 136]
            # nn.BatchNorm1d(hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            # nn.BatchNorm1d(hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            # nn.BatchNorm1d(hidden_size),
            self.act_fn,
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            # nn.BatchNorm1d(hidden_size),
            self.act_fn,
        )
        self.fc_state = self.dist.build_state(self.hidden_size, self.plan_features)

    def forward(self, initial_percep_emb: torch.Tensor, latent_goal: torch.Tensor) -> State:
        x = torch.cat([initial_percep_emb, latent_goal], dim=-1)
        x = self.fc_model(x)
        my_state = self.fc_state(x)
        state = self.dist.forward_dist(my_state)
        return state
