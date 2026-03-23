from typing import Tuple

import torch
import torch.nn as nn


class ProjVisLang(nn.Module):
    def __init__(self, im_dim: int, lang_dim: int, output_dim: int, proj_lang: bool = True):
        super().__init__()
        self.mlp_im = nn.Sequential(
            nn.Linear(in_features=im_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=output_dim),
        )
        self.mlp_lang = None
        if proj_lang:
            self.mlp_lang = nn.Sequential(
                nn.Linear(in_features=lang_dim, out_features=128),
                nn.ReLU(),
                nn.Linear(in_features=128, out_features=output_dim),
            )

    def forward(self, vis_emb: torch.Tensor, lang_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        vis_emb = self.mlp_im(vis_emb)
        if self.mlp_lang is not None:
            lang_emb = self.mlp_lang(lang_emb)
        return vis_emb, lang_emb
