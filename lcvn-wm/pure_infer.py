import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np
from typing import Tuple, Optional, Union
import sys


class LDiTPureInference:
    def __init__(self, dfot_checkpoint_path: str, vae_checkpoint_path: str, device: str = "cuda"):
        self.device = device
        self.model, self.cfg, _, _ = self._load_dfot_model(dfot_checkpoint_path)
        self.vae = self._load_vae(vae_checkpoint_path)
        print("\n" + "=" * 50)
        print("LDiT 4-to-1 Inferencer ready.")
        print("=" * 50 + "\n")

    def _load_dfot_model(self, checkpoint_path: str):
        print(f"Loading model weights: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        config_path = Path(checkpoint_path).parent / ".hydra" / "config.yaml"
        if config_path.exists():
            full_cfg = OmegaConf.load(config_path)
            OmegaConf.resolve(full_cfg)
            algo_cfg = getattr(full_cfg, "algorithm", full_cfg)
            print(f"Hydra config resolved: {config_path}")
        else:
            algo_cfg = checkpoint.get("hyper_parameters", {}).get("algorithm", None)

        from algorithms.ldit.ldit_video_social import LDiTVideoSocial
        model = LDiTVideoSocial(algo_cfg)

        # Weight key alignment
        state_dict = checkpoint.get("state_dict", checkpoint)
        new_state_dict = {}

        # Common prefixes to strip
        prefixes_to_ignore = ["diffusion_model.model.", "algorithm.", "model.", "module."]

        for k, v in state_dict.items():
            new_k = k
            for p in prefixes_to_ignore:
                if new_k.startswith(p):
                    new_k = new_k.replace(p, "", 1)
            new_state_dict[new_k] = v

        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

        model_keys = list(model.state_dict().keys())
        ckpt_keys = list(new_state_dict.keys())

        print(f"\n[debug] Model expected keys (first 5): {model_keys[:5]}")
        print(f"[debug] Checkpoint keys after processing (first 5): {ckpt_keys[:5]}\n")

        if len(missing) < 50:
            print("Weights aligned successfully.")
        else:
            print(f"Warning: {len(missing)} keys still missing.")

        model.eval().to(self.device)
        return model, algo_cfg, None, None

    def _load_vae(self, vae_checkpoint_path: str):
        print(f"Loading VAE decoder: {vae_checkpoint_path}")
        from algorithms.vae.image_vae.model import Encoder, Decoder
        checkpoint = torch.load(vae_checkpoint_path, map_location="cpu", weights_only=False)

        ddconfig = {
            'resolution': 256, 'in_channels': 3, 'out_ch': 3, 'ch': 128,
            'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2, 'z_channels': 4,
            'double_z': True, 'attn_resolutions': [], 'dropout': 0.0
        }

        vae = torch.nn.Module()
        vae.encoder = Encoder(**ddconfig)
        vae.decoder = Decoder(**ddconfig)
        vae.quant_conv = torch.nn.Conv2d(8, 8, 1)
        vae.post_quant_conv = torch.nn.Conv2d(4, 4, 1)

        def decode(z):
            z = vae.post_quant_conv(z)
            return vae.decoder(z)

        vae.decode = decode

        sd = checkpoint.get("state_dict", checkpoint)
        vae.load_state_dict({k: v for k, v in sd.items() if not k.startswith("loss")}, strict=False)
        return vae.eval().to(self.device)

    @torch.no_grad()
    def predict(self, history_latents, actions, return_rgb=True):
        B, T, C, H, W = history_latents.shape
        # Normalize and predict (simplified)
        mean = torch.tensor([[[-0.0789]], [[-0.4896]], [[-0.4124]], [[0.3865]]], device=self.device)
        std = torch.tensor([[[3.79]], [[12.4186]], [[6.9311]], [[5.6708]]], device=self.device)

        latents_norm = (history_latents.to(self.device) - mean) / std
        zero_pad = torch.zeros(B, 1, actions.shape[-1], device=self.device)
        actions_padded = torch.cat([zero_pad, actions.to(self.device)], dim=1)

        pred_seq = self.model._predict_sequence(context=latents_norm, length=T + 1, conditions=actions_padded)[0]
        pred_latent = pred_seq[:, -1] * std + mean

        rgb = None
        if return_rgb:
            rgb = torch.clamp((self.vae.decode(pred_latent) + 1.0) / 2.0, 0, 1)
        return pred_latent, rgb
