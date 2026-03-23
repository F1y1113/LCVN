#!/usr/bin/env python3

import math
from typing import Tuple

import torch
import torch.nn as nn

from lcvn_ac.utils.distributions import Distribution, State


class PlanRecognitionBiRNNNetwork(nn.Module):
    def __init__(
        self,
        in_features: int,
        plan_features: int,
        action_space: int,
        birnn_dropout_p: float,
        dist: Distribution,
        rnn_type: str,
    ):
        super(PlanRecognitionBiRNNNetwork, self).__init__()
        self.plan_features = plan_features
        self.action_space = action_space
        self.in_features = in_features
        self.dist = dist
        self.birnn_model = eval(rnn_type)(
            input_size=self.in_features,
            hidden_size=512,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=birnn_dropout_p,
        )  # shape: [N, seq_len, feat]
        self.fc_state = self.dist.build_state(1024, self.plan_features)

    def forward(self, perceptual_emb: torch.Tensor) -> Tuple[State, torch.Tensor]:
        perceptual_emb = perceptual_emb.permute(1, 0, 2)
        x, hn = self.birnn_model(perceptual_emb)
        x = x[:, -1]  # we just need only last unit output
        my_state = self.fc_state(x)
        state = self.dist.forward_dist(my_state)
        return state, x


class PlanRecognitionTransformersNetwork(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        encoder_hidden_size: int,
        fc_hidden_size: int,
        plan_features: int,
        in_features: int,
        action_space: int,
        encoder_normalize: bool,
        positional_normalize: bool,
        position_embedding: bool,
        max_position_embeddings: int,
        dropout_p: bool,
        dist: Distribution,
    ):

        super().__init__()
        self.in_features = 128
        self.plan_features = plan_features
        self.action_space = action_space
        self.padding = False
        self.dist = dist
        self.hidden_size = fc_hidden_size
        self.position_embedding = position_embedding
        self.encoder_normalize = encoder_normalize
        self.positional_normalize = positional_normalize
        mod = self.in_features % num_heads
        if mod != 0:
            self.padding = True
            self.pad = num_heads - mod
            self.in_features += self.pad
        if position_embedding:
            self.position_embeddings = nn.Embedding(max_position_embeddings, self.in_features)
        else:
            self.positional_encoder = PositionalEncoding(self.in_features)
        encoder_layer = nn.TransformerEncoderLayer(
            self.in_features, num_heads, dim_feedforward=encoder_hidden_size, dropout=dropout_p, batch_first=True
        )
        encoder_norm = nn.LayerNorm(self.in_features) if encoder_normalize else None
        if self.positional_normalize:
            self.layernorm = nn.LayerNorm(self.in_features)
        self.dropout = nn.Dropout(p=dropout_p)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
        self.fc = nn.Linear(in_features=self.in_features, out_features=fc_hidden_size)
        self.fc_state = self.dist.build_state(fc_hidden_size, self.plan_features)

    def forward(self, perceptual_emb: torch.Tensor) -> Tuple[State, torch.Tensor]:
        seq_len, batch_size = perceptual_emb.shape[0], perceptual_emb.shape[1]
        perceptual_emb = (
            torch.cat([perceptual_emb, torch.zeros((seq_len, batch_size, self.pad)).to(perceptual_emb.device)], dim=-1)
            if self.padding
            else perceptual_emb
        )
        if self.position_embedding:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=perceptual_emb.device).unsqueeze(0)
            position_embeddings = self.position_embeddings(position_ids)
            position_embeddings = position_embeddings.permute(1, 0, 2)
            x = perceptual_emb + position_embeddings
        else:
            # padd the perceptual embeddig
            # x = self.positional_encoder(perceptual_emb.permute(1, 0, 2))  # [s, b, emb]
            x = self.positional_encoder(perceptual_emb)  # [s, b, emb]

        if self.positional_normalize:
            x = self.layernorm(x)
        # Transformer with batch_first=True expects [batch, seq, emb]
        x = x.permute(1, 0, 2)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        x = torch.mean(x, dim=1)  # gather all the sequence info
        my_state = self.fc_state(x)
        state = self.dist.forward_dist(my_state)
        return state, x


class PositionalEncoding(nn.Module):
    """Implementation from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x
