"""
DFoT 4-to-1 inference API
Pure inference interface abstracted from test_dfot_corrected.py

Features:
- Input: 4 historical latent frames and 4 action vectors
- Output: predicted 5th latent frame and decoded RGB image
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
import os
# Whitelist OmegaConf containers for PyTorch safe unpickler
try:
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import ContainerMetadata
except Exception:
    ListConfig = None
    DictConfig = None
    ContainerMetadata = None
import numpy as np
from typing import Tuple, Optional, Union, Any, List, Dict, Type
import warnings
import sys
import os as _os

_repo_root = Path(__file__).absolute().parents[3]
_wm_root = _repo_root / "lcvn-wm"
if _wm_root.as_posix() not in sys.path:
    sys.path.insert(0, _wm_root.as_posix())

# --- Rank-aware logging (only log on primary/rank0) ---
def _get_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank()
    except Exception:
        pass
    # Fallbacks for common DDP/cluster launchers
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID", "PMI_RANK"):
        val = _os.environ.get(key, None)
        if val is not None:
            try:
                return int(val)
            except Exception:
                break
    return 0

def _log0(msg: str) -> None:
    # Silent mode: only print when DFOT_VERBOSE=1
    try:
        verbose = str(_os.environ.get("DFOT_VERBOSE", "0")).lower() in ("1", "true", "yes")
    except Exception:
        verbose = False
    if verbose and _get_rank() == 0:
        print(msg)


class LDiTPureInference:
    """
    DFoT 4-to-1 pure inference API class
    
    Usage:
    1. Initialize: inferencer = LDiTPureInference(dfot_checkpoint, vae_checkpoint)
    2. Inference: latent, rgb_image = inferencer.predict(history_latents, actions)
    """
    
    def __init__(
        self,
        dfot_checkpoint_path: str,
        vae_checkpoint_path: str,
        device: str = "cuda",
        override_denoise_steps: Optional[int] = None
    ):
        """
        Initialize the inferencer
        
        Args:
            dfot_checkpoint_path: path to DFoT checkpoint
            vae_checkpoint_path: path to VAE checkpoint  
            device: compute device
        """
        self.device = device
        
        # Load DFoT model (supports overriding sampling/denoising steps)
        self.model, self.cfg, _, _ = self._load_dfot_model(dfot_checkpoint_path, override_denoise_steps)
        
        # Load VAE decoder
        self.vae = self._load_vae(vae_checkpoint_path)
        
        _log0("✓ DFoT 4-to-1 inferencer initialized")
    
    def _load_dfot_model(self, checkpoint_path: str, override_denoise_steps: Optional[int] = None) -> Tuple[object, OmegaConf, torch.Tensor, torch.Tensor]:
        """Load DFoT model"""
        _log0(f"Loading DFoT model: {checkpoint_path}")
        
        # Add allowed globals for safe unpickling (OmegaConf containers)
        try:
            if hasattr(torch, "serialization") and hasattr(torch.serialization, "add_safe_globals"):
                allowlist = []
                if ListConfig is not None:
                    allowlist.append(ListConfig)
                if DictConfig is not None:
                    allowlist.append(DictConfig)
                if ContainerMetadata is not None:
                    allowlist.append(ContainerMetadata)
                # Allow builtins.list (commonly blocked by PyTorch WeightsUnpickler)
                try:
                    allowlist.append(list)
                except Exception:
                    pass
                # Add typing.Any to allowlist to avoid WeightsUnpickler rejection
                try:
                    if Any is not None:
                        allowlist.append(Any)
                except Exception:
                    pass
                # Pre-allow common typing constructs to reduce repeated warnings
                for t in (Tuple, Optional, Union, List, Dict, Type):
                    try:
                        if t is not None:
                            allowlist.append(t)
                    except Exception:
                        pass
                if allowlist:
                    torch.serialization.add_safe_globals(allowlist)
        except Exception as e:
            if _get_rank() == 0:
                warnings.warn(f"Failed to add OmegaConf/typing safe globals (ignored): {e}")

        # Use standard torch.load; avoid verbose weights_only branch
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError("DFoT checkpoint load failed: " + str(e))
        
        # Extract config from checkpoint
        if "hyper_parameters" in checkpoint:
            cfg = checkpoint["hyper_parameters"]
        elif "cfg" in checkpoint:
            cfg = checkpoint["cfg"]
        else:
            # If no config, try to infer config path from checkpoint location
            try:
                checkpoint_path_obj = Path(checkpoint_path)
                # Prefer environment variable DFOT_CFG if provided
                env_cfg = os.environ.get("DFOT_CFG")
                config_path = None
                if env_cfg and Path(env_cfg).exists():
                    config_path = Path(env_cfg)
                    _log0(f"✓ Using config from environment: {config_path}")
                else:
                    # Search upward from checkpoint path for .hydra/config.yaml (up to 5 levels)
                    current = checkpoint_path_obj.parent
                    for _ in range(5):
                        test_config = current / ".hydra" / "config.yaml"
                        if test_config.exists():
                            config_path = test_config
                            break
                        current = current.parent

                    if not (config_path and config_path.exists()):
                        config_path = _repo_root / "lcvn-wm" / "configurations" / "config.yaml"
            except Exception as e:
                config_path = _repo_root / "lcvn-wm" / "configurations" / "config.yaml"
            # The Hydra root config alone won't compose defaults. Build a robust minimal algorithm config
            # by stitching together dataset + algorithm + backbone local YAMLs.
            try:
                wm_cfg_dir = _repo_root / "lcvn-wm" / "configurations"
                dataset_cfg = OmegaConf.load(wm_cfg_dir / "dataset" / "social.yaml")
                algo_base = OmegaConf.load(wm_cfg_dir / "algorithm" / "dfot_video.yaml")
                algo_social = OmegaConf.load(wm_cfg_dir / "algorithm" / "dfot_video_social.yaml")
                backbone_base = OmegaConf.load(wm_cfg_dir / "algorithm" / "backbone" / "dit3d.yaml")
                # Optional XL overrides
                shortcut_path = wm_cfg_dir / "shortcut" / "DiT" / "social_xl.yaml"
                shortcut_cfg = OmegaConf.load(shortcut_path) if shortcut_path.exists() else OmegaConf.create({})
                # Prepare backbone by overriding base with shortcut if provided
                backbone = OmegaConf.to_container(backbone_base, resolve=True)
                if "algorithm" in shortcut_cfg and "backbone" in shortcut_cfg.algorithm:
                    for k, v in shortcut_cfg.algorithm.backbone.items():
                        backbone[k] = v
                # Prepare mlp encoder from social spec, optionally override from shortcut
                mlp_encoder = OmegaConf.to_container(algo_social.get("mlp_encoder", {}), resolve=True)
                if "algorithm" in shortcut_cfg and "mlp_encoder" in shortcut_cfg.algorithm:
                    for k, v in shortcut_cfg.algorithm.mlp_encoder.items():
                        mlp_encoder[k] = v
                # Fill dataset-dependent fields
                ds = dataset_cfg
                algo_final = OmegaConf.to_container(algo_base, resolve=False)
                algo_final["x_shape"] = list(ds.observation_shape)
                algo_final["max_frames"] = int(ds.max_frames)
                algo_final["n_frames"] = int(ds.n_frames)
                algo_final["frame_skip"] = int(ds.frame_skip)
                algo_final["context_frames"] = int(ds.context_length)
                algo_final["latent"] = OmegaConf.to_container(ds.latent, resolve=True)
                algo_final["data_mean"] = OmegaConf.to_container(ds.data_mean, resolve=True)
                algo_final["data_std"] = OmegaConf.to_container(ds.data_std, resolve=True)
                algo_final["external_cond_dim"] = int(ds.external_cond_dim)
                algo_final["external_cond_stack"] = bool(ds.external_cond_stack)
                algo_final["external_cond_processing"] = getattr(ds, "external_cond_processing", None)
                # Ensure required diffusion and logging fields exist (from base)
                algo_final["diffusion"] = OmegaConf.to_container(algo_base.diffusion, resolve=True)
                algo_final["logging"] = OmegaConf.to_container(algo_base.logging, resolve=True)
                algo_final["tasks"] = OmegaConf.to_container(algo_base.tasks, resolve=True)
                algo_final["optimizer_beta"] = list(algo_base.optimizer_beta)
                algo_final["weight_decay"] = float(algo_base.weight_decay)
                algo_final["lr_scheduler"] = OmegaConf.to_container(algo_base.lr_scheduler, resolve=True)
                # Minimal training hparams to satisfy object construction
                algo_final["lr"] = 1e-4
                algo_final["debug"] = False
                algo_final["compile"] = False
                # Sampling scheme defaults
                algo_final["chunk_size"] = 4
                algo_final["noise_level"] = "random_uniform"
                algo_final["fixed_context"] = {"enabled": False, "indices": None, "dropout": 0}
                algo_final["variable_context"] = {"enabled": False, "prob": 0, "dropout": 0}
                # Backbone and encoder
                algo_final["backbone"] = backbone
                algo_final["mlp_encoder"] = mlp_encoder
                # Instruction support toggle for Social XL
                algo_final["use_instruction"] = True
                cfg = OmegaConf.create({"algorithm": algo_final})
            except Exception:
                # Hard fallback with sane defaults for latent 4x32x32 and DiT-XL backbone
                cfg = OmegaConf.create({
                    "algorithm": {
                        "x_shape": [4, 32, 32],
                        "frame_skip": 1,
                        "chunk_size": 4,
                        "n_frames": 5,
                        "context_frames": 4,
                        "latent": {"enable": True, "type": "pre_sample", "num_channels": 4, "downsampling_factor": [1, 1]},
                        "data_mean": [[[0.0]], [[0.0]], [[0.0]], [[0.0]]],
                        "data_std": [[[1.0]], [[1.0]], [[1.0]], [[1.0]]],
                        "external_cond_dim": 4,
                        "external_cond_stack": False,
                        "external_cond_processing": None,
                        "diffusion": {"is_continuous": False, "timesteps": 1000, "sampling_timesteps": 50, "use_causal_mask": False, "clip_noise": 20.0, "loss_weighting": {"cum_snr_decay": 0.9}}, # "sampling_timesteps": 50
                        "noise_level": "random_uniform",
                        "fixed_context": {"enabled": False, "indices": None, "dropout": 0},
                        "variable_context": {"enabled": False, "prob": 0, "dropout": 0},
                        "backbone": {"patch_size": 2, "hidden_size": 1152, "depth": 28, "num_heads": 16, "mlp_ratio": 4.0, "variant": "full", "pos_emb_type": "rope_3d", "use_gradient_checkpointing": False},
                        "mlp_encoder": {"linear_hidden_dim": 1144, "angular_hidden_dim": 1144, "dt_hidden_dim": 8, "output_dim": 1152},
                        "logging": {"metrics": [], "metrics_batch_size": 1, "max_num_videos": 0},
                        "tasks": {"prediction": {"enabled": False}, "interpolation": {"enabled": False}},
                        "optimizer_beta": [0.9, 0.99],
                        "weight_decay": 1e-3,
                        "lr": 1e-4,
                        "debug": False,
                        "compile": False,
                        "lr_scheduler": {"name": "cosine", "num_warmup_steps": 80000, "num_training_steps": 247200},
                        "use_instruction": True,
                    }
                })
        
        try:
            if not hasattr(cfg, "algorithm") and isinstance(cfg, dict) and "algorithm" in cfg:
                cfg = OmegaConf.create(cfg)
            if hasattr(cfg, "algorithm"):
                algo = cfg.algorithm
                if not hasattr(algo, "max_frames"):
                    try:
                        if hasattr(algo, "n_frames"):
                            algo.max_frames = int(algo.n_frames)
                        elif hasattr(algo, "context_frames"):
                            algo.max_frames = int(algo.context_frames) + 1
                        else:
                            algo.max_frames = 5
                    except Exception:
                        algo.max_frames = 5
                if not hasattr(algo, "diffusion"):
                    algo.diffusion = OmegaConf.create({})
                diff = algo.diffusion
                if not hasattr(diff, "beta_schedule"):
                    diff.beta_schedule = "linear"
                if not hasattr(diff, "schedule_fn_kwargs"):
                    diff.schedule_fn_kwargs = {}
                if not hasattr(diff, "objective"):
                    diff.objective = "pred_x0"
                if not hasattr(diff, "ddim_sampling_eta"):
                    diff.ddim_sampling_eta = 0.0
                if not hasattr(diff, "is_continuous"):
                    diff.is_continuous = False
                if not hasattr(diff, "timesteps"):
                    diff.timesteps = 1000
                if not hasattr(diff, "sampling_timesteps"):
                    diff.sampling_timesteps = 50  #"sampling_timesteps": 50
                if not hasattr(diff, "use_causal_mask"):
                    diff.use_causal_mask = False
                if not hasattr(diff, "clip_noise"):
                    diff.clip_noise = 20.0
                if not hasattr(diff, "loss_weighting"):
                    diff.loss_weighting = OmegaConf.create({"cum_snr_decay": 0.9})
                lw = diff.loss_weighting
                if not hasattr(lw, "strategy"):
                    lw.strategy = "uniform"
                if not hasattr(lw, "snr_clip"):
                    lw.snr_clip = 0.5
                if not hasattr(lw, "sigmoid_bias"):
                    lw.sigmoid_bias = 0.0
                if not hasattr(algo, "backbone"):
                    algo.backbone = OmegaConf.create({"name": "dit3d", "patch_size": 2, "hidden_size": 1152, "depth": 28, "num_heads": 16, "mlp_ratio": 4.0, "variant": "full", "pos_emb_type": "rope_3d", "use_gradient_checkpointing": False})
                else:
                    bb = algo.backbone
                    if not hasattr(bb, "name"):
                        bb.name = "dit3d"
                if not hasattr(algo, "mlp_encoder"):
                    algo.mlp_encoder = OmegaConf.create({"linear_hidden_dim": 1144, "angular_hidden_dim": 1144, "dt_hidden_dim": 8, "output_dim": 1152})
                if not hasattr(algo, "compile"):
                    algo.compile = False
                if not hasattr(algo, "optimizer_beta"):
                    algo.optimizer_beta = [0.9, 0.99]
                if not hasattr(algo, "weight_decay"):
                    algo.weight_decay = 1e-3
                if not hasattr(algo, "lr"):
                    algo.lr = 1e-4
                if not hasattr(algo, "lr_scheduler"):
                    algo.lr_scheduler = OmegaConf.create({"name": "cosine", "num_warmup_steps": 80000, "num_training_steps": 247200})
                if not hasattr(algo, "scheduling_matrix"):
                    algo.scheduling_matrix = "full_sequence"
        except Exception:
            pass
        _log0(f"✓ Config loaded")

        # Record sampling/denoising step configuration for logging
        self._denoise_steps = None
        try:
            if hasattr(cfg, 'algorithm'):
                algo = cfg.algorithm
                # Common locations: algorithm.diffusion.diffusion_timestep_respacing or algorithm.sampling.num_steps
                if hasattr(algo, 'diffusion') and hasattr(algo.diffusion, 'diffusion_timestep_respacing'):
                    respacing = str(algo.diffusion.diffusion_timestep_respacing)
                    # e.g., '50' or 'ddim25'; try numeric parsing
                    if respacing.isdigit():
                        self._denoise_steps = int(respacing)
                if self._denoise_steps is None and hasattr(algo, 'sampling') and hasattr(algo.sampling, 'num_steps'):
                    try:
                        self._denoise_steps = int(algo.sampling.num_steps)
                    except Exception:
                        pass
        except Exception:
            pass
        
        # If override steps provided, update config before model creation
        try:
            if override_denoise_steps is not None:
                # Sync common locations: diffusion_timestep_respacing (string) and sampling.num_steps (int)
                if hasattr(cfg, 'algorithm'):
                    algo = cfg.algorithm
                    if hasattr(algo, 'diffusion'):
                        try:
                            # Try numeric and DDIM formats for broader compatibility
                            val_numeric = str(int(override_denoise_steps))
                            val_ddim = f"ddim{int(override_denoise_steps)}"
                            # Prefer DDIM if sampler expects it; otherwise use numeric
                            try:
                                algo.diffusion.diffusion_timestep_respacing = val_ddim
                            except Exception:
                                algo.diffusion.diffusion_timestep_respacing = val_numeric
                        except Exception:
                            pass
                    if hasattr(algo, 'sampling'):
                        try:
                            algo.sampling.num_steps = int(override_denoise_steps)
                        except Exception:
                            pass
                self._denoise_steps = int(override_denoise_steps)
                _log0(f"✓ Override DFoT sampling steps: {self._denoise_steps}")
        except Exception as e:
            _log0(f"⚠️ Override sampling steps failed (ignored): {e}")

        # Dynamically import model class
        from algorithms.ldit.ldit_video_social import LDiTVideoSocial
        
        # Create model
        model = LDiTVideoSocial(cfg.algorithm)
        
        # Load weights
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        # Load into model
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        
        if missing:
            _log0(f"⚠️  Missing keys: {len(missing)}")
        if unexpected:
            _log0(f"⚠️  Unexpected keys: {len(unexpected)}")
        
        model.eval()
        model.to(self.device)

        # Best-effort override of sampling step fields in the model instance (multi-repo compatibility)
        try:
            if self._denoise_steps is not None:
                changed = []
                steps_int = int(self._denoise_steps)
                # Common location 1: model.sampling.num_steps
                if hasattr(model, 'sampling') and hasattr(model.sampling, 'num_steps'):
                    try:
                        model.sampling.num_steps = steps_int
                        changed.append('model.sampling.num_steps')
                    except Exception:
                        pass
                # Common location 2: model.diffusion.diffusion_timestep_respacing
                if hasattr(model, 'diffusion') and hasattr(model.diffusion, 'diffusion_timestep_respacing'):
                    try:
                        val_numeric = str(steps_int)
                        val_ddim = f"ddim{steps_int}"
                        try:
                            model.diffusion.diffusion_timestep_respacing = val_ddim
                        except Exception:
                            model.diffusion.diffusion_timestep_respacing = val_numeric
                        changed.append('model.diffusion.diffusion_timestep_respacing')
                    except Exception:
                        pass
                # Common location 3: sampler object step fields
                for sampler_attr in ('sampler', 'ddim_sampler', 'ddpm_sampler'):
                    sampler_obj = getattr(model, sampler_attr, None)
                    if sampler_obj is not None:
                        for field in ('num_steps', 'n_steps', 'num_timesteps', 'steps'):
                            if hasattr(sampler_obj, field):
                                try:
                                    setattr(sampler_obj, field, steps_int)
                                    changed.append(f'model.{sampler_attr}.{field}')
                                except Exception:
                                    pass
                _log0(f"✓ Forced DFoT internal sampling steps to {steps_int} (fields: {', '.join(changed) if changed else 'no writable fields found'})")
        except Exception as e:
            _log0(f"⚠️ Failed to override model internal sampling steps (ignored): {e}")
        
        _log0(f"✓ Model loaded")
        _log0(f"  Checkpoint: {checkpoint_path}")
        _log0(f"  Device: {self.device}")
        
        # Pure inference does not need metrics; avoid heavy imports
        if hasattr(model, 'metrics'):
            try:
                model.metrics = None
            except Exception:
                pass
        if hasattr(model, 'metrics_registry'):
            try:
                model.metrics_registry = None
            except Exception:
                pass
        
        # Model handles normalization; no need to load data_mean/data_std
        
        return model, cfg, None, None
    
    def _load_vae(self, vae_checkpoint_path: str) -> object:
        """Load VAE decoder"""
        _log0(f"Loading VAE decoder: {vae_checkpoint_path}")
        
        _repo_root = Path(__file__).absolute().parents[3]
        _wm_root = _repo_root / "lcvn-wm"
        if _wm_root.as_posix() not in sys.path:
            sys.path.insert(0, _wm_root.as_posix())
        
        # Import Encoder/Decoder directly to avoid trainer dependencies
        from algorithms.vae.image_vae.model import Encoder, Decoder
        
        # Use standard torch.load; avoid verbose weights_only branch
        try:
            checkpoint = torch.load(vae_checkpoint_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError("VAE checkpoint load failed: " + str(e))
        
        # Get config
        if "cfg" in checkpoint:
            vae_cfg = checkpoint["cfg"]
        elif "hyper_parameters" in checkpoint:
            vae_cfg = checkpoint["hyper_parameters"]
        else:
            vae_cfg = OmegaConf.create({
                'embed_dim': 4,
                'ddconfig': {
                    'resolution': 256,
                    'in_channels': 3,
                    'out_ch': 3,
                    'ch': 128,
                    'ch_mult': [1, 2, 4, 4],
                    'num_res_blocks': 2,
                    'z_channels': 4,
                    'double_z': True,
                }
            })
        
        # Build minimal VAE using Encoder and Decoder
        ddconfig = vae_cfg["ddconfig"] if isinstance(vae_cfg, dict) else vae_cfg.ddconfig
        embed_dim = vae_cfg["embed_dim"] if isinstance(vae_cfg, dict) else vae_cfg.embed_dim
        
        encoder = Encoder(**ddconfig)
        decoder = Decoder(**ddconfig)
        quant_conv = torch.nn.Conv2d(2 * ddconfig["z_channels"], 2 * embed_dim, 1)
        post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        
        class MinimalImageVAE(torch.nn.Module):
            def __init__(self, encoder, decoder, quant_conv, post_quant_conv):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder
                self.quant_conv = quant_conv
                self.post_quant_conv = post_quant_conv

            def decode(self, z: torch.Tensor) -> torch.Tensor:
                z = self.post_quant_conv(z)
                return self.decoder(z)

        vae = MinimalImageVAE(encoder, decoder, quant_conv, post_quant_conv)
        
        # Load weights (filter out loss-related keys)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("loss")}
        
        missing, unexpected = vae.load_state_dict(state_dict, strict=False)
        
        if missing:
            _log0(f"  ⚠️ Missing keys: {len(missing)}")
        if unexpected:
            _log0(f"  ⚠️ Unexpected keys: {len(unexpected)}")
        
        vae.eval()
        vae.to(self.device)
        
        _log0(f"✓ ImageVAE decoder loaded")
        _log0(f"  Checkpoint: {vae_checkpoint_path}")
        
        return vae
    
    @torch.no_grad()
    def predict(
        self,
        history_latents: torch.Tensor,
        actions: torch.Tensor,
        return_rgb: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        4-to-1 prediction interface
        
        Args:
            history_latents: (B, 4, C, H, W) - 4 historical latent frames
            actions: (B, 4, 4) - 4 actions [dx, dy, dyaw, dt]
            return_rgb: whether to return decoded RGB image
            
        Returns:
            predicted_latent: (B, C, H, W) - predicted 5th latent frame
            rgb_image: (B, 3, H*8, W*8) - decoded RGB image (if return_rgb=True)
        """
        # Input validation
        assert len(history_latents.shape) == 5, f"history_latents must be 5D tensor, got: {history_latents.shape}"
        assert history_latents.shape[1] == 4, f"need 4 historical frames, got: {history_latents.shape[1]}"
        assert len(actions.shape) == 3, f"actions must be 3D tensor, got: {actions.shape}"
        assert actions.shape[1:] == (4, 4), f"actions shape must be (B, 4, 4), got: {actions.shape}"
        
        B, T, C, H, W = history_latents.shape
        
        # Move to device
        history_latents = history_latents.to(self.device)
        actions = actions.to(self.device)
        
        # Predict next latent frame
        predicted_latent = self._predict_next_latent(history_latents, actions)
        
        # Decode to RGB image
        rgb_image = None
        if return_rgb:
            rgb_image = self._decode_latents(predicted_latent)
        
        return predicted_latent, rgb_image
    
    @torch.no_grad()
    def _predict_next_latent(self, latents_input, actions_input):
        """
        Predict next latent
        
        Args:
            latents_input: latent sequence of shape (B, T, C, H, W)
            actions_input: action sequence of shape (B, T, action_dim)
            
        Returns:
            predicted_latent: next latent of shape (B, C, H, W)
        """
        # Reshape to expected format (B, n_tokens, C, H, W)
        B, T, C, H, W = latents_input.shape
        latents_reshaped = latents_input.view(B, T, C, H, W)

        # Align with training: normalize tokens first
        if hasattr(self.model, "_normalize_x"):
            latents_norm = self.model._normalize_x(latents_reshaped)
        else:
            latents_norm = latents_reshaped
        
        # Add 0 padding to front of condition sequence (consistent with training)
        # Training: [0_padding, action1, action2, action3, action4] -> 5-frame condition
        # Inference: [0_padding, action1, action2, action3, action4] -> 5-frame condition
        zero_padding = torch.zeros(B, 1, actions_input.shape[-1], 
                                 device=actions_input.device, dtype=actions_input.dtype)
        actions_padded = torch.cat([zero_padding, actions_input], dim=1)  # (B, T+1, action_dim)
        
        # Predict next latent
        # length = T + 1 means total sequence length (T context + 1 new prediction)
        # Model internally handles normalization and unnormalization
        # Emit sampling/denoising step hint without intruding DFoT
        if self._denoise_steps is not None:
            _log0(f"[DFoT] start denoise sampling, configured steps: {self._denoise_steps}")
        else:
            _log0(f"[DFoT] start denoise sampling (steps unknown; not found in cfg)")

        with torch.no_grad():
            predicted_sequence = self.model._predict_sequence(
                context=latents_norm,
                length=T + 1,  # 4-frame context + 1-frame new prediction = 5 frames total
                conditions=actions_padded  # Use condition with 0-padding prepended
            )[0]  # Take the first return value
            
            # Training flow unnormalizes tokens after sampling before decoding
            if hasattr(self.model, "_unnormalize_x"):
                predicted_sequence = self.model._unnormalize_x(predicted_sequence)

            # Take the last frame as the next prediction (5th frame)
            # predicted_sequence: (B, 5, C, H, W) -> select [:, -1] -> (B, C, H, W)
            predicted_latent = predicted_sequence[:, -1]  # (B, C, H, W)

        _log0("[DFoT] denoise sampling finished")
        
        return predicted_latent
    
    @torch.no_grad()
    def _decode_latents(self, latents):
        """
        Decode latents to RGB images
        
        Args:
            latents: (B, C, H, W) or (B, T, C, H, W)
        
        Returns:
            images: (B, 3, H*8, W*8) or (B, T, 3, H*8, W*8)
        """
        original_shape = latents.shape
        
        # If 5D tensor, reshape to 4D
        if len(original_shape) == 5:
            B, T, C, H, W = original_shape
            latents = latents.view(B * T, C, H, W)
        
        # VAE decode
        images = self.vae.decode(latents)
        
        # Restore original shape
        if len(original_shape) == 5:
            images = images.view(B, T, 3, images.shape[-2], images.shape[-1])
        
        # Normalize to [0, 1]
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        
        return images
    
    def predict_from_numpy(
        self,
        history_latents: np.ndarray,
        actions: np.ndarray,
        return_rgb: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convenience interface: predict from numpy arrays
        
        Args:
            history_latents: (B, 4, C, H, W) numpy array
            actions: (B, 4, 4) numpy array
            return_rgb: whether to return RGB image
            
        Returns:
            predicted_latent: (B, C, H, W) numpy array
            rgb_image: (B, 3, H*8, W*8) numpy array (if return_rgb=True)
        """
        # Convert to torch tensor
        history_latents_tensor = torch.from_numpy(history_latents).float()
        actions_tensor = torch.from_numpy(actions).float()
        
        # Predict
        pred_latent, rgb_image = self.predict(
            history_latents_tensor, actions_tensor, return_rgb
        )
        
        # Convert back to numpy
        pred_latent_np = pred_latent.cpu().numpy()
        rgb_image_np = rgb_image.cpu().numpy() if rgb_image is not None else None
        
        return pred_latent_np, rgb_image_np


def create_inferencer(
    dfot_checkpoint_path: Optional[str] = None,
    vae_checkpoint_path: Optional[str] = None,
    device: str = "cuda",
    denoise_steps: Optional[int] = None
) -> LDiTPureInference:
    """
    Convenience helper to create an inferencer
    
    Args:
        dfot_checkpoint_path: DFoT checkpoint path
        vae_checkpoint_path: VAE checkpoint path
        device: compute device
        
    Returns:
        DFoTPureInference instance
    """
    # Strict requirement: checkpoint paths must be provided via env or args
    env_dfot = os.environ.get("DFOT_WEIGHTS_PT") or os.environ.get("DFOT_CKPT")
    env_vae = os.environ.get("DFOT_VAE_CKPT")

    # Prefer explicit args, then environment variables, else raise error
    dfot_ckpt = dfot_checkpoint_path or env_dfot
    vae_ckpt = vae_checkpoint_path or env_vae

    if not dfot_ckpt:
        raise ValueError("DFoT weights path not provided. Set env `DFOT_WEIGHTS_PT` or `DFOT_CKPT`, or pass `dfot_checkpoint_path`.")
    if not vae_ckpt:
        raise ValueError("VAE checkpoint path not provided. Set env `DFOT_VAE_CKPT`, or pass `vae_checkpoint_path`.")

    _log0(f"✓ Using DFoT weights: {dfot_ckpt}")
    _log0(f"✓ Using VAE weights: {vae_ckpt}")
    return LDiTPureInference(dfot_ckpt, vae_ckpt, device, override_denoise_steps=denoise_steps)


# Usage example
if __name__ == "__main__":
    # Create inferencer
    inferencer = create_inferencer()
    
    # Example data (random; replace with real data in usage)
    batch_size = 1
    history_latents = torch.randn(batch_size, 4, 4, 32, 32)
    actions = torch.randn(batch_size, 4, 4)
    
    # Predict
    predicted_latent, rgb_image = inferencer.predict(history_latents, actions)
    
    print(f"history_latents shape: {history_latents.shape}")
    print(f"actions shape: {actions.shape}")
    print(f"predicted_latent shape: {predicted_latent.shape}")
    print(f"RGB image shape: {rgb_image.shape}")
    
    print("✓ 4-to-1 prediction done!")
