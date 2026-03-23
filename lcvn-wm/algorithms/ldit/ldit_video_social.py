"""
DFoT Video model for Social Navigation
Implements MLP encoding for delta conditions and optional instruction cross-attention.
"""

from typing import Optional, List
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from transformers import get_scheduler

from .ldit_video import DFoTVideo


class IdentityWithMask(nn.Module):
    def forward(self, x, mask=None):
        return x


class MLPConditionEncoder(nn.Module):
    def __init__(
        self,
        linear_hidden_dim: int = 760,
        angular_hidden_dim: int = 760,
        dt_hidden_dim: int = 8,
        output_dim: int = 768,
    ):
        super().__init__()
        self.linear_hidden_dim = linear_hidden_dim
        self.angular_hidden_dim = angular_hidden_dim
        self.dt_hidden_dim = dt_hidden_dim
        self.output_dim = output_dim
        self.linear_encoder = nn.Sequential(
            nn.Linear(2, linear_hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(linear_hidden_dim * 2, linear_hidden_dim),
        )
        self.angular_encoder = nn.Sequential(
            nn.Linear(1, angular_hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(angular_hidden_dim * 2, angular_hidden_dim),
        )
        self.dt_encoder = nn.Sequential(
            nn.Linear(1, dt_hidden_dim),
            nn.SiLU(),
        )
        for m in [self.linear_encoder[0], self.linear_encoder[2], self.angular_encoder[0], self.angular_encoder[2], self.dt_encoder[0]]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        assert linear_hidden_dim + dt_hidden_dim == output_dim

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        delta_linear = conditions[..., :2]
        delta_angular = conditions[..., 2:3]
        dt = conditions[..., 3:4]
        linear_features = self.linear_encoder(delta_linear)
        angular_features = self.angular_encoder(delta_angular)
        combined_spatial = linear_features + angular_features
        dt_features = self.dt_encoder(dt)
        encoded = torch.cat([combined_spatial, dt_features], dim=-1)
        encoded = encoded / torch.sqrt(torch.tensor(self.output_dim, dtype=encoded.dtype))
        return encoded


class LDiTVideoSocial(DFoTVideo):
    def __init__(self, cfg: DictConfig):
        # Read instruction config before super().__init__ creates the backbone
        self.use_language = getattr(cfg, "use_language", False)
        language_layers = getattr(cfg, "language_layers", [7, 14, 21])
        clip_model = getattr(cfg, "clip_model", "openai/clip-vit-base-patch32")

        if self.use_language:
            # Validate language_layers against backbone depth
            depth = getattr(cfg.backbone, "depth", 28)
            invalid_layers = [l for l in language_layers if l >= depth]
            if invalid_layers:
                raise ValueError(
                    f"language_layers {invalid_layers} exceed backbone depth {depth}. "
                    f"Valid range: 0 to {depth - 1}."
                )
            # Inject instruction config into backbone config so DiT3D → DiTBase picks it up
            with OmegaConf.open_dict(cfg):
                cfg.backbone.use_language = True
                cfg.backbone.language_layers = language_layers

        super().__init__(cfg)

        cond_cfg = getattr(cfg, "mlp_encoder", None)
        if cond_cfg is None:
            raise ValueError("mlp_encoder must be provided in algorithm config for social mode")
        self.mlp_encoder = MLPConditionEncoder(
            linear_hidden_dim=cond_cfg.linear_hidden_dim,
            angular_hidden_dim=cond_cfg.angular_hidden_dim,
            dt_hidden_dim=cond_cfg.dt_hidden_dim,
            output_dim=cond_cfg.output_dim,
        )
        self._replace_external_cond_embedding()

        # Initialize instruction encoder if enabled
        if self.use_language:
            from transformers import CLIPTextModel, CLIPTokenizer

            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
            self.clip_encoder = CLIPTextModel.from_pretrained(clip_model)
            self.clip_encoder.eval()
            for param in self.clip_encoder.parameters():
                param.requires_grad = False

            clip_dim = self.clip_encoder.config.hidden_size  # 512
            hidden_size = cond_cfg.output_dim  # matches DiT hidden_size
            self.clip_projection = nn.Sequential(
                nn.Linear(clip_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        print("=" * 80)
        print("DFoT Social Navigation Model Initialized")
        print(f"  Condition encoder: MLP [4D -> {cond_cfg.output_dim}D]")
        print(f"  - Linear delta (dx, dy): MLP -> {cond_cfg.linear_hidden_dim}D")
        print(f"  - Angular delta (dyaw): MLP -> {cond_cfg.angular_hidden_dim}D")
        print("  - Fusion: ADD (linear + angular)")
        print(f"  - DT encoder: MLP -> {cond_cfg.dt_hidden_dim}D")
        print(f"  - Final: CONCAT -> {cond_cfg.output_dim}D")
        if self.use_language:
            print(f"  Instruction encoder: CLIP ({clip_model}) + projection")
            print(f"  - Cross-attention layers: {language_layers}")
        print("=" * 80)

    def _replace_external_cond_embedding(self):
        if hasattr(self.diffusion_model, "_orig_mod"):
            model = self.diffusion_model._orig_mod
        else:
            model = self.diffusion_model
        if hasattr(model, "model"):
            backbone = model.model
        elif hasattr(model, "backbone"):
            backbone = model.backbone
        else:
            print("Warning: Could not find backbone to replace external_cond_embedding")
            return
        if hasattr(backbone, "external_cond_embedding"):
            backbone.external_cond_embedding = IdentityWithMask()
        else:
            print("Warning: backbone has no external_cond_embedding attribute")

    def encode_language(self, instructions: List[str]) -> torch.Tensor:
        """
        Encode instruction strings into feature vectors.
        Args:
            instructions: List of instruction strings, length B.
        Returns:
            Tensor of shape (B, 1, hidden_size) for cross-attention.
        """
        inputs = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            text_features = self.clip_encoder(**inputs).pooler_output  # (B, 512)

        projected = self.clip_projection(text_features)  # (B, hidden_size)
        return projected.unsqueeze(1)  # (B, 1, hidden_size)

    def _get_i_clip(self, rest) -> Optional[torch.Tensor]:
        """
        Override base class to encode instruction strings from batch.
        rest contains [instructions] from on_after_batch_transfer.
        """
        if not self.use_language:
            return None
        if not rest:
            return None
        instructions = rest[0]
        if instructions is None:
            return None
        # Filter out empty instructions
        if isinstance(instructions, (list, tuple)) and all(s == "" for s in instructions):
            return None
        return self.encode_language(instructions)

    def _process_conditions(
        self,
        conditions: Optional[torch.Tensor],
        noise_levels: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if conditions is None:
            return None
        encoded = self.mlp_encoder(conditions)
        return encoded

    def training_step(self, batch, batch_idx, namespace="training"):
        """Training step with instruction support."""
        xs, conditions, masks, gt_videos, *rest = batch
        i_clip = self._get_i_clip(rest)

        noise_levels, masks = self._get_training_noise_levels(xs, masks)
        xs_pred, loss = self.diffusion_model(
            xs,
            self._process_conditions(conditions),
            k=noise_levels,
            i_clip=i_clip,
        )
        loss = self._reweight_loss(loss, masks)

        if batch_idx % self.cfg.logging.loss_freq == 0:
            self.log(
                f"{namespace}/loss",
                loss,
                on_step=namespace == "training",
                on_epoch=namespace != "training",
                sync_dist=True,
            )

        xs, xs_pred = map(self._unnormalize_x, (xs, xs_pred))

        return {
            "loss": loss,
            "xs_pred": xs_pred,
            "xs": xs,
        }

    def configure_optimizers(self):
        transition_params = list(self.diffusion_model.parameters()) + list(
            self.mlp_encoder.parameters()
        )
        if self.use_language:
            transition_params += list(self.clip_projection.parameters())

        optimizer_dynamics = torch.optim.AdamW(
            transition_params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
            betas=self.cfg.optimizer_beta,
        )
        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer_dynamics,
                **self.cfg.lr_scheduler,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return {
            "optimizer": optimizer_dynamics,
            "lr_scheduler": lr_scheduler_config,
        }
