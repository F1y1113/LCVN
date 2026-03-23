"""
LDiT (Language-conditioned DiT) block.
Extends the standard DiTBlock with cross-attention layers for language conditioning.
"""

from typing import Optional, Any
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp
from .dit_blocks import AdaLayerNormZero, Attention


class CrossAttention(nn.Module):
    """
    Cross-attention module for language conditioning.
    Queries come from the main sequence, keys and values from the instruction context (I_clip).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        q = self.q_norm(q)

        B_ctx, N_ctx, _ = context.shape
        kv = (
            self.kv(context)
            .reshape(B_ctx, N_ctx, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        k = self.k_norm(k)

        # pylint: disable-next=not-callable
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LDiTBlock(nn.Module):
    """
    LDiT (Language-conditioned DiT) transformer block with adaptive layer norm
    zero (AdaLN-Zero) conditioning and cross-attention for language conditioning.

    Structure: MHSA -> CA -> FFN
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: Optional[float] = 4.0,
        use_language: bool = True,
        rope: Optional[Any] = None,
        **block_kwargs: dict,
    ):
        super().__init__()

        # Self-attention (MHSA)
        self.norm1 = AdaLayerNormZero(hidden_size)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, rope=rope, **block_kwargs
        )

        # Cross-attention for language conditioning (CA)
        self.use_language = use_language
        if use_language:
            self.norm_ca = AdaLayerNormZero(hidden_size)
            self.ca = CrossAttention(
                dim=hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
            )

        # FFN
        self.use_mlp = mlp_ratio is not None
        if self.use_mlp:
            self.norm2 = AdaLayerNormZero(hidden_size)
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=int(hidden_size * mlp_ratio),
                act_layer=partial(nn.GELU, approximate="tanh"),
            )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module: nn.Module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.attn.apply(_basic_init)
        if self.use_language:
            self.ca.apply(_basic_init)
        if self.use_mlp:
            self.mlp.apply(_basic_init)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        i_clip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional language conditioning.
        Args:
            x: Input tensor of shape (B, N, C).
            c: Conditioning tensor of shape (B, N, C).
            i_clip: CLIP instruction embedding of shape (B, L, C) where L is instruction sequence length.
        """
        # 1. Self-Attention (MHSA)
        x, gate_msa = self.norm1(x, c)
        x = x + gate_msa * self.attn(x)

        # 2. Cross-Attention (CA) — only if I_clip is provided
        if self.use_language and i_clip is not None:
            inst_ctx = i_clip
            if inst_ctx.dim() == 2:
                inst_ctx = inst_ctx.unsqueeze(1)  # (B, 1, C)
            x_norm, gate_ca = self.norm_ca(x, c)
            x = x + gate_ca * self.ca(x_norm, inst_ctx)

        # 3. FFN
        if self.use_mlp:
            x, gate_mlp = self.norm2(x, c)
            x = x + gate_mlp * self.mlp(x)

        return x
