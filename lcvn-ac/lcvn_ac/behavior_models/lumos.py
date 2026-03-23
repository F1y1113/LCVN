import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import hydra
from omegaconf import DictConfig
import yaml

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from lcvn_ac.utils.rl_utils import advantage, max_cos, lambda_return, MC_return, action_mse
from lcvn_ac.utils.distributions import State
import torch.distributions as D
import torch.nn as nn
from torch.nn.functional import cross_entropy


from diffusers.models import AutoencoderKL
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights

from lcvn_ac.behavior_models.decoders.action_decoder import ActionDecoder
from lcvn_ac.utils.pure_infer import LDiTPureInference, create_inferencer
from lcvn_ac.utils.trajectory_logger import TrajectoryLogger
from lcvn_ac.utils.trajectory_logger import disable_trajectory_logging

import sys
from pathlib import Path
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)

# Global silent mode: reduce non-essential logs, keep progress bars only
try:
    _silent = str(os.environ.get("LCVN_AC_SILENT", "1")).lower() in ("1", "true", "yes")
    if _silent:
        logger.setLevel(logging.ERROR)
except Exception:
    pass

torch.set_float32_matmul_precision('high')

import torch._dynamo
torch._dynamo.config.suppress_errors = True

@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


class LUMOS(pl.LightningModule):
    def __init__(
        self,
        perceptual_encoder: DictConfig,
        plan_proposal: DictConfig,
        plan_recognition: DictConfig,
        language_goal: DictConfig,
        visual_goal: DictConfig,
        action_decoder: DictConfig,
        critic: DictConfig,
        distribution: DictConfig,
        loss: DictConfig,
        actor_optimizer: DictConfig,
        critic_optimizer: DictConfig,
        seq_len: int,
        name: str,
        use_clip_auxiliary_loss: bool,
        use_bc_loss: bool,
        temperature: float = 1.0,
        replan_freq: int = 30,
        gripper_control: bool = False,
        proj_vis_lang: Optional[DictConfig] = None,
        lcvn_worldmodel: Optional[DictConfig] = None,
        # Configure diffusion sampling steps for NWM and prediction interface
        nwm_diffusion_steps: int = 6,
        predict_diffusion_steps: int = 250,
        dfot_checkpoint_path: Optional[str] = None,
        dfot_vae_checkpoint_path: Optional[str] = None,
        gt_rollout_epochs: int = 0,  # epochs to use GT states before switching to DFoT

    ) -> None:
        super(LUMOS, self).__init__()
        self.name = name
        self.wm = None
        self.seq_len = seq_len
        # Do not use external world_model by default; enable NWM/DFoT explicitly
        self.latent_half = None

        # Lcvn WorldModel configuration and enable flag
        self.lwm_cfg = lcvn_worldmodel
        self.lwm_enabled = bool(lcvn_worldmodel) and bool(getattr(lcvn_worldmodel, "enabled", False))
        # Disable NWM when Lcvn WorldModel (DFoT) is enabled
        self.nwm_enable: bool = not self.lwm_enabled
        self.nwm_checkpoint_path: str = os.environ.get("NWM_CKPT", "")

        # Diffusion step configuration
        self.nwm_diffusion_steps: int = int(nwm_diffusion_steps)
        self.predict_diffusion_steps: int = int(predict_diffusion_steps)
        log_rank_0(f"Configure NWM diffusion steps: nwm_diffusion_steps={self.nwm_diffusion_steps}, predict_diffusion_steps={self.predict_diffusion_steps}")

        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(self.device).eval()

        self.nwm_components_loaded = False # NWM components loaded flag
        NWM_IMPORTS_OK = False             # Module import success flag
        create_diffusion = None            # Placeholder
        CDiT_models = None

        if self.nwm_enable: # Import and load only when enabled
            try:
                nwm_infer_dir = Path(os.environ.get("NWM_INFER_DIR", ""))
                log_rank_0(f"Checking NWM path: {nwm_infer_dir}")

                if nwm_infer_dir.is_dir():
                    nwm_infer_path_str = str(nwm_infer_dir.resolve())
                    # Add NWM path to sys.path if missing
                    if nwm_infer_path_str not in sys.path:
                        sys.path.insert(0, nwm_infer_path_str)
                        log_rank_0(f"Added '{nwm_infer_path_str}' to sys.path.")
                    else:
                        log_rank_0(f"'{nwm_infer_path_str}' already in sys.path.")

                    # Try importing within the try block
                    try:
                        from diffusion import create_diffusion
                        from models import CDiT_models
                        NWM_IMPORTS_OK = True
                        log_rank_0("Imported NWM modules (diffusion, models).")
                    except ImportError as import_e:
                        log_rank_0(f"ERROR: Failed to import NWM components from '{nwm_infer_path_str}': {import_e}")
                        NWM_IMPORTS_OK = False
                else:
                    log_rank_0(f"ERROR: NWM inference directory not found '{nwm_infer_dir}'. Cannot import.")
                    NWM_IMPORTS_OK = False

            except Exception as path_e:
                 log_rank_0(f"ERROR: Failed to set NWM import path: {path_e}")
                 NWM_IMPORTS_OK = False
        else:
             log_rank_0("NWM disabled (nwm_enable=False).")


        # --- Load VAE
        self.vae_latent_dim = None
        self.vae_latent_shape_chw = None
        # Even if NWM fails, VAE is required for conversion when ResNet loads
        try:
            with torch.no_grad(): # Determine latent dimensionality
                # DFoT compatible: 32x32 latent -> 256x256 input
                test_input_size = 256
                dummy_latent = self.vae.encode(torch.zeros(1, 3, test_input_size, test_input_size, device=self.device)).latent_dist.sample()
                self.vae_latent_dim = torch.flatten(dummy_latent, 1).shape[1]
                self.vae_latent_shape_chw = tuple(dummy_latent.shape[1:])
                log_rank_0(f"VAE loaded. Latent dim(flat):{self.vae_latent_dim}, Latent shape(CHW):{self.vae_latent_shape_chw}")
        except Exception as e:
            log_rank_0(f"ERROR loading VAE: {e}")
            self.vae = None

        # --- Load NWM components (CDiT, Diffusion) ---
        self.cdit_model = None
        self.diffusion = None
        self.nwm_config = None
        self.nwm_latent_size = None
        self.nwm_context_size = None
        # Load only when NWM enabled, imports OK, and VAE available
        if self.nwm_enable and NWM_IMPORTS_OK and self.vae is not None:
            try:
                log_rank_0("Initializing NWM Inference Components...")
                # Load NWM configuration
                data_cfg_path = os.environ.get("NWM_DATA_CONFIG", "")
                with open(data_cfg_path, "r") as f:
                    default_config = yaml.safe_load(f)
                model_cfg_path = os.environ.get("NWM_MODEL_CONFIG", "")
                with open(model_cfg_path, "r") as f:
                    user_config = yaml.safe_load(f)
                self.nwm_config = default_config; self.nwm_config.update(user_config)
                self.nwm_latent_size = self.nwm_config['image_size'] // 8
                self.nwm_context_size = self.nwm_config.get('context_size', 2)
                # Validate latent size
                if self.nwm_latent_size != self.vae_latent_shape_chw[1]: logger.warning(...)

                cdit_config_key = self.nwm_config['model']
                # Ensure CDiT_models is not None
                if CDiT_models is None: raise ImportError("CDiT_models was not imported successfully.")
                self.cdit_model = CDiT_models[cdit_config_key](
                    input_size=self.nwm_latent_size,
                    context_size=self.nwm_context_size,
                    use_instruction=self.nwm_config.get("use_instruction", False)
                ).to(self.device).eval()
                # ... (load checkpoint state_dict) ...
                # Use weights_only=True to avoid loading arbitrary code and silence FutureWarning
                # Prefer weights_only=True; broader fallback for older torch and pickling constraints
                try:
                    ckpt_path = os.environ.get("NWM_CKPT", "")
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                except Exception as e:
                    try:
                        import argparse
                        torch.serialization.add_safe_globals([argparse.Namespace])
                        ckpt_path = os.environ.get("NWM_CKPT", "")
                        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                    except Exception:
                        ckpt_path = os.environ.get("NWM_CKPT", "")
                        ckpt = torch.load(ckpt_path, map_location="cpu")
                # Accept both EMA and state_dict formats; strip 'module.' prefixes
                state_dict = ckpt.get("ema", ckpt.get("state_dict", ckpt))
                if isinstance(state_dict, dict):
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                self.cdit_model.load_state_dict(state_dict, strict=True)
                #self.cdit_model = torch.compile(self.cdit_model)

                # Create Diffusion 
                if create_diffusion is None: raise ImportError("create_diffusion was not imported successfully.")
                # Use configurable NWM diffusion steps
                self.diffusion = create_diffusion(str(self.nwm_diffusion_steps))

                self.nwm_components_loaded = True
                log_rank_0("NWM Inference Components Initialized.")
            except Exception as e:
                log_rank_0(f"ERROR initializing NWM components after import: {e}", exc_info=True)
                self.nwm_components_loaded = False

        self.resnet_encoder = None
        self.resnet_preprocess = None
        try:
            # Load weights, build ResNet model and extract encoder
            weights = getattr(models.ResNet50_Weights, "DEFAULT", models.ResNet50_Weights.DEFAULT)
            resnet = models.resnet50(weights=weights)
            modules = list(resnet.children())[:-1]
            self.resnet_encoder = torch.nn.Sequential(*modules).to(self.device).eval()

            # Set preprocess function that matches weights; handle antialias compatibility
            try:
                self.resnet_preprocess = weights.transforms(antialias=True)
            except Exception:
                self.resnet_preprocess = weights.transforms()

            log_rank_0("ResNet-50 encoder loaded for conversion.")
        except Exception as e:
            # If any step above fails (invalid weight name, download failure, library issues, etc.)
            log_rank_0(f"ERROR loading ResNet-50: {e}")
            # self.resnet_preprocess remains None in this case
        
        world_model = None
        self.setup_input_sizes(
            world_model,
            plan_proposal,
            plan_recognition,
            visual_goal,
            action_decoder,
            critic,
            distribution,
        )

        # Ensure perceptual_encoder has required latent_dim_flat; default to 4096 if missing
        try:
            if isinstance(perceptual_encoder, dict):
                perceptual_encoder.setdefault("latent_dim_flat", 4096)
            else:
                # DictConfig or similar
                if not hasattr(perceptual_encoder, "latent_dim_flat") or perceptual_encoder.get("latent_dim_flat", None) is None:
                    perceptual_encoder["latent_dim_flat"] = 4096
            log_rank_0(f"perceptual_encoder.latent_dim_flat: {perceptual_encoder.get('latent_dim_flat', None)}")
        except Exception as e:
            log_rank_0(f"Skipping perceptual_encoder latent_dim_flat defaulting due to: {e}")

        self.perceptual_encoder = hydra.utils.instantiate(perceptual_encoder)
        # plan networks
        self.dist = hydra.utils.instantiate(distribution)
        self.plan_proposal = hydra.utils.instantiate(plan_proposal, dist=self.dist)
        self.plan_recognition = hydra.utils.instantiate(plan_recognition, dist=self.dist)

        # goal encoders
        self.visual_goal = hydra.utils.instantiate(visual_goal)
        self.language_goal = hydra.utils.instantiate(language_goal) if language_goal else None

        # actor and critic
        self.action_decoder: ActionDecoder = hydra.utils.instantiate(action_decoder)
        self.critic = hydra.utils.instantiate(critic)

        self.lambda_gae = loss.lambda_gae
        self.gamma = loss.gamma
        self.rho = loss.rho
        self.eta = loss.eta
        self.eps = torch.finfo(torch.float32).eps
        self.grad_clip = loss.grad_clip
        self.target_update_interval = loss.target_update_interval
        self.clip_auxiliary_loss_beta = loss.clip_auxiliary_loss_beta
        self.kl_beta = loss.kl_beta
        self.kl_balancing_mix = loss.kl_balancing_mix
        self.temperature = temperature
        self.bc_alpha = loss.bc_alpha
        self.actor_alpha = loss.actor_alpha
        self.gt_rollout_epochs = gt_rollout_epochs

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gripper_control = gripper_control
        self.automatic_optimization = False

        # auxiliary losses
        self.use_clip_auxiliary_loss = use_clip_auxiliary_loss
        if use_clip_auxiliary_loss:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.proj_vis_lang = hydra.utils.instantiate(proj_vis_lang)

        self.use_bc_loss = use_bc_loss

        self.save_hyperparameters()

        # for inference
        self.rollout_step_counter = 0
        self.replan_freq = replan_freq
        self.latent_goal = None
        self.plan = None
        self.lang_embeddings = None

        # DFoT pure inference integration (no HTTP, no mode switching)
        # Path priority: init args > lcvn_worldmodel config > environment variables (handled in create_inferencer)
        self.dfot_checkpoint_path = dfot_checkpoint_path or (getattr(self.lwm_cfg, "dfot_checkpoint_path", None) if self.lwm_cfg is not None else None)
        self.dfot_vae_checkpoint_path = dfot_vae_checkpoint_path or (getattr(self.lwm_cfg, "dfot_vae_checkpoint_path", None) if self.lwm_cfg is not None else None)
        self.dfot_inferencer = None
        self._dfot_history_initialized = False
        self._dfot_history_latents = None   # (B, 4, 4, 28, 28)
        self._dfot_history_actions = None   # (B, 4, 4)

        if self.lwm_enabled:
            # DFoT inferencer lazily initialized; in setup() load based on process rank
            # This avoids each DDP process loading the model and causing OOM
            log_rank_0("DFoT pure inference will be initialized in setup() hook.")
            self._dfot_init_pending = True
        else:
            self._dfot_init_pending = False

        # Global trajectory counter for tracking the first trajectory
        self.trajectory_counter = 0
        
        self.move_modules_to_device()
        # Disable TrajectoryLogger by default (prevent unnecessary output)
        try:
            disable_trajectory_logging()
        except Exception:
            pass

    def move_modules_to_device(self):
        if hasattr(self, 'vae') and self.vae is not None:
            self.vae = self.vae.to(self.device)
        if hasattr(self, 'resnet_encoder') and self.resnet_encoder is not None:
            self.resnet_encoder = self.resnet_encoder.to(self.device)
        if hasattr(self, 'action_classifier') and self.action_classifier is not None:
            self.action_classifier = self.action_classifier.to(self.device)
        if hasattr(self, 'cdit_model') and self.cdit_model is not None:
            self.cdit_model = self.cdit_model.to(self.device)
            
    @staticmethod
    def setup_input_sizes(
        world_model,
        plan_proposal,
        plan_recognition,
        visual_goal,
        action_decoder,
        critic,
        distribution,
    ):
        """
        Configure the input feature sizes of the respective parts of the network.

        Args:
            perceptual_encoder: DictConfig for perceptual encoder.
            plan_proposal: DictConfig for plan proposal network.
            plan_recognition: DictConfig for plan recognition network.
            visual_goal: DictConfig for visual goal encoder.
            action_decoder: DictConfig for action decoder network.
            distribution: DictConfig for plan distribution (continuous or discrete).
        """
        # Use 32x32 VAE latent by default (DFoT-compatible)
        latent_size = 4096

        plan_proposal.perceptual_features = 128  # latent_size
        plan_recognition.in_features = 128  # latent_size
        visual_goal.in_features = 128  # latent_size
        action_decoder.perceptual_features = latent_size
        critic.perceptual_features = latent_size

        if distribution.dist == "discrete":
            plan_proposal.plan_features = distribution.class_size * distribution.category_size
            plan_recognition.plan_features = distribution.class_size * distribution.category_size
            action_decoder.plan_features = distribution.class_size * distribution.category_size
            critic.plan_features = distribution.class_size * distribution.category_size
        elif distribution.dist == "continuous":
            plan_proposal.plan_features = distribution.plan_features
            plan_recognition.plan_features = distribution.plan_features
            action_decoder.plan_features = distribution.plan_features
            critic.plan_features = distribution.plan_features

    def configure_optimizers(self):

        self.actor_params = (
            list(self.action_decoder.parameters())
            + list(self.perceptual_encoder.parameters())
            + list(self.visual_goal.parameters())
            + list(self.language_goal.parameters())
            + list(self.plan_proposal.parameters())
            + list(self.plan_recognition.parameters())
        )
        if self.use_clip_auxiliary_loss:
            self.actor_params += list(self.proj_vis_lang.parameters())
            self.actor_params += [self.logit_scale]

        actor_optimizer = hydra.utils.instantiate(self.actor_optimizer, params=self.actor_params)
        critic_optimizer = hydra.utils.instantiate(self.critic_optimizer, params=self.critic.parameters())

        optimizers = [actor_optimizer, critic_optimizer]
        return optimizers
    
    @torch.no_grad()
    def _vae_latent_to_2048_feature(self, vae_latent_batch: Tensor) -> Tensor:
        """Decode VAE latent to image, then encode with ResNet to 2048D features."""
        vae_latent_batch = vae_latent_batch.to(self.device)
        # 1. Reverse scale VAE latent
        vae_decode_input = vae_latent_batch / 0.18215
        # 2. VAE decode -> image [-1, 1]
        image_batch = self.vae.decode(vae_decode_input).sample.clamp(-1, 1)
        # 3. Preprocess image 
        image_batch_0_1 = image_batch * 0.5 + 0.5
        processed_image = self.resnet_preprocess(image_batch_0_1)
        # 4. ResNet encode
        features_2048_raw = self.resnet_encoder(processed_image) # (B, 2048, 1, 1)
        # 5. Flatten
        features_2048_flat = torch.flatten(features_2048_raw, 1) # (B, 2048)

        return features_2048_flat

    @torch.no_grad()
    def _predict_next_state_with_dfot(self,
                                     current_vae_latent: Tensor, # s_k (B, 4, 32, 32)
                                     action_batch: Tensor,       # a_k (B, 6) LUMOS action format
                                     raw_instructions_batch: Optional[List[str]] = None
                                     ) -> Tensor:
        """
        Predict next state using DFoT pure inference (directly call pure_infer).
        I/O:
        - Input current_vae_latent: (B, 4, 32, 32)
        - Input action_batch: (B, 6) LUMOS action format
        - Output next_vae_latent: (B, 4, 32, 32)
        """
        # Strict DFoT usage - no fallbacks when enabled
        assert self.lwm_enabled and self.dfot_inferencer is not None, "Lcvn WorldModel is enabled but inferencer is not available"
        
        B = current_vae_latent.shape[0]
        # Convert LUMOS action to DFoT 4D action (dx, dy, dyaw, dt=0.1)
        dfot_action = torch.zeros((B, 4), device=current_vae_latent.device, dtype=torch.float32)
        dfot_action[:, 0] = action_batch[:, 0]
        dfot_action[:, 1] = action_batch[:, 1]
        dfot_action[:, 2] = action_batch[:, 2]
        dfot_action[:, 3] = 0.1

        # Initialize or update 4-frame history (latent and action), keep 32x32
        if not self._dfot_history_initialized:
            # History latents: repeat current frame 4 times -> (B, 4, 4, 32, 32)
            assert current_vae_latent.shape[-2:] == (32, 32), f"current_vae_latent must be 32x32, got {current_vae_latent.shape}"
            self._dfot_history_latents = current_vae_latent.unsqueeze(1).repeat(1, 4, 1, 1, 1)
            # History actions: first 3 frames are "stop" with dt=0.1; last frame is current action -> (B, 4, 4)
            self._dfot_history_actions = torch.zeros((B, 4, 4), device=current_vae_latent.device, dtype=torch.float32)
            self._dfot_history_actions[:, :3, 3] = 0.1
            self._dfot_history_actions[:, 3] = dfot_action
            self._dfot_history_initialized = True
        else:
            # If batch size changes (e.g., switch from inference B=1 to training B>1), reinitialize history buffers
            if self._dfot_history_latents.shape[0] != B:
                logger.debug(
                    f"DFoT history batch size changed: {self._dfot_history_latents.shape[0]} -> {B}. Reinitializing history buffers.")
                assert current_vae_latent.shape[-2:] == (32, 32), \
                    f"current_vae_latent must be 32x32, got {current_vae_latent.shape}"
                self._dfot_history_latents = current_vae_latent.unsqueeze(1).repeat(1, 4, 1, 1, 1)
                self._dfot_history_actions = torch.zeros((B, 4, 4), device=current_vae_latent.device, dtype=torch.float32)
                self._dfot_history_actions[:, :3, 3] = 0.1
                self._dfot_history_actions[:, 3] = dfot_action
                self._dfot_history_initialized = True
            else:
                # Sliding window update of history
                self._dfot_history_latents = torch.cat([
                    self._dfot_history_latents[:, 1:],
                    current_vae_latent.unsqueeze(1)
                ], dim=1)
                self._dfot_history_actions = torch.cat([
                    self._dfot_history_actions[:, 1:],
                    dfot_action.unsqueeze(1)
                ], dim=1)

        # Use 32x32 history latent directly
        B, T, C, H, W = self._dfot_history_latents.shape
        assert (H, W) == (32, 32), f"DFoT expects 32x32 latents, got {(H, W)}"
        latents_32 = self._dfot_history_latents

        # Call pure inference (return latent only to avoid RGB decode cost)
        pred_latent_32, _ = self.dfot_inferencer.predict(latents_32, self._dfot_history_actions, return_rgb=False)

        # Keep 32x32 output
        return pred_latent_32

    def _ensure_dfot_inferencer(self) -> None:
        """Initialize DFoT pure inferencer on demand to avoid unintended fallback to NWM.
        Only attempt when use_dfot=True and dfot_inferencer is None.
        On failure set use_dfot=False to avoid repeated attempts and log noise.
        """
        try:
            if self.lwm_enabled and self.dfot_inferencer is None:
                device_str = f'cuda:{self.device.index}' if self.device.type == 'cuda' and hasattr(self.device, 'index') else ('cuda' if self.device.type == 'cuda' else 'cpu')
                log_rank_0(f"Initializing DFoT inferencer on {device_str} (JIT)...")
                from lcvn_ac.utils.pure_infer import create_inferencer
                self.dfot_inferencer = create_inferencer(
                    dfot_checkpoint_path=self.dfot_checkpoint_path,
                    vae_checkpoint_path=self.dfot_vae_checkpoint_path,
                    device=device_str,
                    denoise_steps=int(self.predict_diffusion_steps)
                )
                log_rank_0("DFoT pure inference initialized (JIT).")
        except Exception as e:
            logger.error(f"DFoT JIT init failed: {e}", exc_info=True)
            # Disable Lcvn WorldModel to avoid repeated init stalls and log noise
            self.lwm_enabled = False

    def _reset_dfot_history(self) -> None:
        """
        Reset DFoT history buffers to ensure independence across training windows (batches).
        - Set `_dfot_history_initialized` to False
        - Clear `_dfot_history_latents` and `_dfot_history_actions`
        On next `_predict_next_state_with_dfot` call, initialization repeats current frame 4 times.
        """
        self._dfot_history_initialized = False
        self._dfot_history_latents = None
        self._dfot_history_actions = None
        logger.debug("DFoT history reset at batch boundary.")

    def _warm_start_dfot_history(self, vae_latent_seq: Tensor) -> None:
        """Pre-fill DFoT history with first 4 real frames instead of repeating frame 0.
        Without this, DFoT sees a static scene and outputs near-zero variance predictions.

        Args:
            vae_latent_seq: (T, B, 4, 32, 32) - full latent sequence from training batch
        """
        T, B = vae_latent_seq.shape[:2]
        n = min(4, T)
        frames = vae_latent_seq[:n]  # (n, B, 4, 32, 32)
        if n < 4:
            pad = vae_latent_seq[:1].repeat(4 - n, 1, 1, 1, 1)
            frames = torch.cat([pad, frames], dim=0)  # (4, B, 4, 32, 32)
        # Rearrange to (B, 4, 4, 32, 32) as expected by _dfot_history_latents
        self._dfot_history_latents = frames.permute(1, 0, 2, 3, 4).contiguous()
        self._dfot_history_actions = torch.zeros((B, 4, 4), device=vae_latent_seq.device, dtype=torch.float32)
        self._dfot_history_actions[:, :, 3] = 0.1  # dt=0.1 for all history frames
        self._dfot_history_initialized = True
        logger.debug("DFoT history warm-started with first %d real frames.", n)

    @torch.no_grad()
    def _predict_next_state_with_nwm(self,
                                    current_vae_latent: Tensor, # s_k (B, 4, 32, 32)
                                    action_batch: Tensor,       # a_k (B, 6),
                                    raw_instructions_batch: Optional[List[str]] # List[str] len=B or None
                                    ) -> Tensor:
        # log_rank_0(f"--- Entering _predict_next_state_with_nwm ---")

        # --- Prefer DFoT: initialize on demand to avoid fallback to NWM ---
        if self.lwm_enabled:
            if self.dfot_inferencer is None:
                self._ensure_dfot_inferencer()
            if self.dfot_inferencer is not None:
                logger.debug("Using DFoT pure inference path for state prediction.")
                return self._predict_next_state_with_dfot(
                    current_vae_latent, action_batch, raw_instructions_batch
                )
            else:
                logger.warning("DFoT enabled but inferencer not available; returning zeros and skipping NWM.")
                return torch.zeros_like(current_vae_latent)

        # --- Input and component checks ---
        if not getattr(self, 'nwm_components_loaded', False):
            if getattr(self, 'nwm_enable', False):
                logger.error("NWM components not loaded.")
            else:
                logger.debug("NWM disabled; returning zeros fallback.")
            return torch.zeros_like(current_vae_latent)
        batch_size = action_batch.shape[0]
        expected_latent_shape_chw = getattr(self, 'vae_latent_shape_chw', (4, 32, 32))
        nwm_context_size = getattr(self, 'nwm_context_size', 2)
        action_dim = action_batch.shape[-1]

        # --- Log input shapes ---
        # log_rank_0(f"  [Input] Batch Size: {batch_size}")
        # log_rank_0(f"  [Input] current_vae_latent shape: {current_vae_latent.shape}")
        # log_rank_0(f"  [Input] prev_vae_latent shape: {prev_vae_latent.shape}")
        # log_rank_0(f"  [Input] action_batch shape: {action_batch.shape}")

        DEFAULT_NWM_REL_T_VALUE = 0.0703125
        # --- Prepare fixed rel_t (batched) ---
        rel_t_batch = torch.full((batch_size,), DEFAULT_NWM_REL_T_VALUE, device=self.device)

        # --- Prepare context (batched) ---
        #context_batch = torch.stack([prev_vae_latent, current_vae_latent], dim=1) # (B, 2, 4, 28, 28)
        context_batch = current_vae_latent.unsqueeze(1)
        
        # --- Prepare instructions (batched) ---
        use_instr = self.nwm_config.get("use_instruction", False) if hasattr(self, 'nwm_config') and self.nwm_config else False
        instructions_arg = None
        if use_instr and raw_instructions_batch is not None:
            if len(raw_instructions_batch) == batch_size: instructions_arg = raw_instructions_batch
            else: logger.warning(f"Instruction list length ({len(raw_instructions_batch)}) mismatches batch size ({batch_size}). Ignoring instructions.")
        elif use_instr: logger.warning("NWM configured to use instructions, but none provided.")

        # --- Log prepared NWM input ---
        # log_rank_0(f"  [NWM Input] Context (x_cond) shape: {context_batch.shape}")
        # log_rank_0(f"  [NWM Input] Delta (y) shape: {delta_batch_flat.shape}")
        # log_rank_0(f"  [NWM Input] Rel_t shape: {rel_t_batch.shape}")
        # log_rank_0(f"  [NWM Input] Instruction type: {type(instructions_arg)}")

        # --- Execute NWM prediction (batched) ---
        next_vae_latent_batch = torch.zeros_like(current_vae_latent)
        try:
            z_shape = (batch_size, *expected_latent_shape_chw)
            z = torch.randn(z_shape, device=self.device)

            x_cond_batch = context_batch # Shape (B, 1, 4, 32, 32)
            model_kwargs = dict(
                y=action_batch,         # (B, 4)
                x_cond=x_cond_batch,        # (B, 1, 4, 28, 28)
                rel_t=rel_t_batch,          # (B,)
                instruction=instructions_arg # List[str] or None
            )

            next_vae_latent_batch = self.diffusion.p_sample_loop(
                self.cdit_model.forward,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=self.device
            ) # Expected output shape (B, 4, 32, 32)


        except Exception as e:
            logger.error(f"Error during NWM batch diffusion step: {e}", exc_info=True)

        return next_vae_latent_batch

    def nwm_step(self, current_vae_latent: Tensor, action_batch: Tensor, raw_instructions_batch: Optional[List[str]]):
        return self._predict_next_state_with_nwm(current_vae_latent, action_batch, raw_instructions_batch)

    def get_action_trajectory_vae_nwm(self, state_cur, vae_latent,
                                      latent_goal, sampled_plan, yaw, raw_instructions_batch,
                                      gt_acts=None):
        traj_logger = TrajectoryLogger()
        traj_logger.log_model_forward("action_trajectory_start", {
            "state_cur_shape": state_cur.shape,
            "vae_latent_shape": vae_latent.shape,
            "latent_goal_shape": latent_goal.shape,
            "sampled_plan_shape": sampled_plan.shape,
            "seq_len": self.seq_len
        })
        
        val_buffer, target_buffer, rewards, actions, entropy, log_probs_buf, means_buf = [], [], [], [], [], [], []
        predicted_states = []
        B = state_cur.shape[0]
        device = state_cur.device
        for k in range(self.seq_len):
            current_feature = torch.flatten(state_cur, start_dim=1)
            a_hat, ent, log_prob, mean = self.action_decoder.get_action(current_feature, latent_goal, sampled_plan)
            v_hat, v_t = self.get_value(current_feature, latent_goal, sampled_plan)

            rel_actions = torch.zeros((B, 4), device=device, dtype=torch.float32)
            rel_actions[:, 0] = a_hat[:, 0]
            rel_actions[:, 1] = a_hat[:, 1]
            rel_actions[:, 2] = a_hat[:, 2]
            rel_actions[:, 3] = a_hat[:, 3]
            
            # Log action prediction for first trajectory (step 0)
            if k == 0:
                # Log detailed context to forward logger (preserve key-value info)
                traj_logger.log_model_forward("action_step_0", {
                    "current_feature_shape": current_feature.shape,
                    "a_hat_shape": a_hat.shape,
                    "a_hat_values": a_hat[0].detach().cpu().numpy().tolist() if a_hat.shape[0] > 0 else [],
                    "entropy_mean": ent[0].detach().cpu().view(-1).float().mean().item() if ent.shape[0] > 0 else 0,
                    "v_hat_mean": v_hat[0].detach().cpu().view(-1).float().mean().item() if v_hat.shape[0] > 0 else 0,
                    "v_t_mean": v_t[0].detach().cpu().view(-1).float().mean().item() if v_t.shape[0] > 0 else 0,
                    "rel_actions": rel_actions[0].detach().cpu().numpy().tolist() if rel_actions.shape[0] > 0 else []
                })
                # Record actions via specialized interface (predicted vs current step relative actions)
                traj_logger.log_action_prediction(
                    a_hat[0].detach().cpu(),
                    rel_actions[0].detach().cpu()
                )
            
            if gt_acts is not None:
                # GT rollout mode: use real next state instead of DFoT, reward = action cosine similarity
                if k < self.seq_len - 1:
                    state_cur = vae_latent[k + 1]  # (B, 4, 32, 32) real next frame
                    reward = F.cosine_similarity(a_hat, gt_acts[k], dim=-1)  # (B,)
                else:
                    reward = torch.zeros(B, device=device, dtype=torch.float32)
                predicted_states.append(state_cur)
            else:
                # DFoT rollout mode: predict next state with world model
                prev_state = state_cur.clone()
                state_cur = self.nwm_step(state_cur, rel_actions, raw_instructions_batch)
                predicted_states.append(state_cur)

                if k == 0:
                    traj_logger.log_model_forward("nwm_state_step_0", {
                        "prev_state_shape": prev_state.shape,
                        "next_state_shape": state_cur.shape,
                        "state_change_norm": torch.norm(state_cur - prev_state).detach().cpu().item()
                    })
                    traj_logger.log_latent_output(state_cur, stage="NWM_STEP")

                if k < self.seq_len - 1:
                    pred_flat = state_cur.flatten(start_dim=1)
                    gt_flat = vae_latent[k + 1].flatten(start_dim=1)
                    reward = max_cos(pred_flat, gt_flat)
                else:
                    reward = torch.zeros_like(rewards[-1]).to(self.device)
            actions.append(a_hat)
            log_probs_buf.append(log_prob)
            means_buf.append(mean)
            entropy.append(ent)
            val_buffer.append(v_hat.squeeze())
            target_buffer.append(v_t.squeeze())
            rewards.append(reward)


        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs_buf)  # (T, B)
        means = torch.stack(means_buf)           # (T, B, action_dim)
        # Stack predicted states into tensor (T, B, 4, 32, 32)
        try:
            predicted_states_t = torch.stack(predicted_states)
        except Exception:
            predicted_states_t = None
        entropy = torch.mean(torch.stack(entropy).float())
        values = torch.stack(val_buffer)

        return actions, log_probs, means, entropy, values, target_buffer, rewards, predicted_states_t

    def get_action_trajectory(self, c_state, latent_goal, sampled_plan, latents, raw_instructions_batch):
        val_buffer, target_buffer, rewards, actions, entropy = [], [], [], [], []
        for k in range(self.seq_len):
            a_hat, ent, _, _mean = self.action_decoder.get_action(c_state, latent_goal, sampled_plan)
            v_hat, v_t = self.get_value(c_state, latent_goal, sampled_plan)
            # c_state = self.wm_step(c_state, a_hat)
            if k < self.seq_len - 1:
                c_state = latents[k + 1]


            if k < self.seq_len - 1:
                assert self.latent_half is not None, "latent_half not set: this reward path depends on WM dimension; use when WM is available."
                reward = max_cos(c_state[..., : self.latent_half], latents[k + 1][..., : self.latent_half])
            else:
                reward = torch.zeros_like(rewards[-1]).to(self.device)

            actions.append(a_hat)
            entropy.append(ent)
            val_buffer.append(v_hat.squeeze())
            target_buffer.append(v_t.squeeze())
            rewards.append(reward)


        actions = torch.stack(actions)
        entropy = torch.mean(torch.stack(entropy))
        values = torch.stack(val_buffer)

        return actions, entropy, values, target_buffer, rewards

    def forward_train(self, codes: Tensor, latent_goal: Tensor,
        gt_acts: Tensor,
        gt_discrete_acts: Tensor, robot_obs: Tensor, vae_latent: Tensor, raw_instructions,
        use_gt_rollout: bool = False):

        traj_logger = TrajectoryLogger()
        traj_logger.log_model_forward("forward_train_start", {
            "codes_shape": codes.shape,
            "latent_goal_shape": latent_goal.shape,
            "gt_acts_shape": gt_acts.shape,
            "gt_discrete_acts_shape": gt_discrete_acts.shape,
            "robot_obs_shape": robot_obs.shape,
            "vae_latent_shape": vae_latent.shape
        })

        # ------------Plan Proposal------------ #
        pp_state = self.plan_proposal(codes[0], latent_goal)
        pp_dist = self.dist.get_dist(pp_state)

        # ------------Plan Recognition------------ #
        pr_state, seq_feat = self.plan_recognition(codes)
        pr_dist = self.dist.get_dist(pr_state)

        sampled_plan = pr_dist.rsample()  # sample from recognition net
        if self.dist.dist == "discrete":
            sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)
            
        traj_logger.log_model_forward("plan_processing", {
            "pp_state_shape": pp_state.shape if hasattr(pp_state, 'shape') else str(type(pp_state)),
            "pr_state_shape": pr_state.shape if hasattr(pr_state, 'shape') else str(type(pr_state)),
            "seq_feat_shape": seq_feat.shape,
            "sampled_plan_shape": sampled_plan.shape,
            "dist_type": self.dist.dist
        })
        #c_state = latents[0] # [B, 2048]

        # vae_latent [B, 9, 4, 32, 32]
        state_cur = vae_latent[0]

        yaw = robot_obs[0, :, 3]

        actions, log_probs, means, entropy, values, target_buffer, rewards, _predicted_states = self.get_action_trajectory_vae_nwm(
            state_cur, vae_latent, latent_goal, sampled_plan, yaw, raw_instructions_batch=raw_instructions,
            gt_acts=gt_acts if use_gt_rollout else None,
        )

        unnorm_returns = lambda_return(rewards, target_buffer, self.lambda_gae, self.gamma)
        losses = self.loss(actions, gt_acts, values, unnorm_returns, entropy, pp_state, pr_state, log_probs, means)

        return losses, rewards, unnorm_returns, actions, pp_state, pr_state, seq_feat

    def forward_val(self, latents: Tensor, codes: Tensor, latent_goal: Tensor, robot_obs: Tensor,
                    vae_latent: Tensor, raw_instructions):

        # ------------Plan Proposal------------ #
        pp_state = self.plan_proposal(codes[0], latent_goal)
        pp_dist = self.dist.get_dist(pp_state)

        # ------------ Policy network (strictly aligned with training forward) ------------ #
        # In training path, actions are driven by plans sampled from recognition network distribution
        pr_state, seq_feat = self.plan_recognition(codes)
        pr_dist = self.dist.get_dist(pr_state)
        sampled_plan = pr_dist.rsample()
        if self.dist.dist == "discrete":
            sampled_plan = torch.flatten(sampled_plan, start_dim=-2, end_dim=-1)
        state_cur = vae_latent[0]
        yaw = robot_obs[0, :, 3]
        actions, _log_probs, _means, entropy, values, target_buffer, rewards, predicted_states = self.get_action_trajectory_vae_nwm(
            state_cur, vae_latent, latent_goal, sampled_plan, yaw, raw_instructions_batch=raw_instructions
        )
        # Return calculation aligned with training (lambda GAE return) for logging
        unnorm_returns = lambda_return(rewards, target_buffer, self.lambda_gae, self.gamma)

        loss_kl = self.kl_loss(pp_state, pr_state)
        losses = {"loss_kl": loss_kl}
        # Return sampled plans from recognition and proposal networks for external needs (keep keys consistent)
        sampled_plan_pp = self.dist.sample_latent_plan(pp_dist)
        sampled_plan_pr = sampled_plan
        return losses, rewards, unnorm_returns, actions, pp_state, pr_state, seq_feat, sampled_plan_pp, sampled_plan_pr, predicted_states

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        traj_logger = TrajectoryLogger()
        # DFoT history is reset + warm-started per modality inside the loop below
        
        # Start a new trajectory (only first time), match actual API
        if self.trajectory_counter == 0:
            traj_logger.start_new_trajectory({
                "global_step": int(self.global_step),
                "batch_idx": batch_idx
            })
            
        traj_logger.log_model_forward("training_step_start", {
            "trajectory_id": self.trajectory_counter,
            "batch_idx": batch_idx,
            "batch_keys": list(batch.keys()),
            "global_step": self.global_step
        })
        
        critic_loss, actor_loss, entropy_loss, kl_loss, lang_clip_loss, bc_loss = (
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
            torch.tensor(0.0).to(self.device),
        )

        actor_optimizer, critic_optimizer = self.optimizers()

        if self.global_step % self.target_update_interval == 0:
            self.critic.update_critic_target()

        batch_size: Dict[str, int] = {}
        total_bs = 0

        for self.modality_scope, dataset_batch in batch.items():
            # Reset and warm-start DFoT history per modality to avoid cross-window leakage
            # and to avoid static-scene initialization (repeating same frame 4x)
            if self.lwm_enabled and "vae_latent" in dataset_batch:
                self._reset_dfot_history()
                self._warm_start_dfot_history(dataset_batch["vae_latent"])

            #latents = dataset_batch["feature"]
            latents = dataset_batch["vae_latent"]
            assert latents.shape[-2:] == (32, 32), f"Training requires 32x32 latent; got {latents.shape}"
            latents = torch.flatten(latents, start_dim=2)
            codes = self.perceptual_encoder(latents)
            
            traj_logger.log_model_forward("perceptual_encoder", {
                "modality_scope": self.modality_scope,
                "vae_latent_shape": dataset_batch["vae_latent"].shape,
                "latents_flattened_shape": latents.shape,
                "codes_shape": codes.shape,
                "codes_dtype": str(codes.dtype)
            })

            
            raw_instructions_batch = dataset_batch.get("lang_raw", None)

            # Goal encoding: choose language or visual goal based on current modality
            # - Language modality (keys contain "lang"): use language goal encoder; expects input [B, 384]
            # - Visual modality may not have language embeddings (possibly size 0); use visual goal encoder
            if ("lang" in self.modality_scope) and (dataset_batch.get("lang", None) is not None):
                lang_tensor = dataset_batch["lang"]
                # Guard: skip empty language tensors (e.g., [B, 0] or numel()==0)
                if lang_tensor.numel() > 0 and (lang_tensor.shape[-1] > 0):
                    latent_goal = self.language_goal(lang_tensor)
                    goal_type = "language"
                else:
                    latent_goal = self.visual_goal(codes[-1])
                    goal_type = "visual_fallback"
            else:
                latent_goal = self.visual_goal(codes[-1])
                goal_type = "visual"
                
            traj_logger.log_model_forward("goal_encoding", {
                "modality_scope": self.modality_scope,
                "goal_type": goal_type,
                "latent_goal_shape": latent_goal.shape,
                "latent_goal_dtype": str(latent_goal.dtype),
                "codes_last_shape": codes[-1].shape if len(codes) > 0 else "empty"
            })

            use_gt_rollout = (self.gt_rollout_epochs > 0 and self.current_epoch < self.gt_rollout_epochs)

            # During GT rollout, suppress actor RL loss: REINFORCE log_probs drive policy to
            # tanh saturation (log(1-tanh^2+1e-6)->+inf), overwhelming the BC gradient.
            # Restore actor_alpha automatically when switching to DFoT RL.
            if use_gt_rollout:
                if not hasattr(self, '_saved_actor_alpha'):
                    self._saved_actor_alpha = self.actor_alpha
                self.actor_alpha = 0.0
            elif hasattr(self, '_saved_actor_alpha'):
                self.actor_alpha = self._saved_actor_alpha
                del self._saved_actor_alpha

            losses, rewards, returns, ac_actions, pp_state, pr_state, seq_feat = self.forward_train(
                codes,
                latent_goal,
                dataset_batch["rel_actions"],
                dataset_batch["discrete_action"],
                dataset_batch["state_info"]["robot_obs"],
                dataset_batch["vae_latent"],
                raw_instructions=raw_instructions_batch,
                use_gt_rollout=use_gt_rollout,
            )

            if "lang" in self.modality_scope:
                if not torch.any(dataset_batch["use_for_aux_lang_loss"]):
                    batch_size["aux_lang"] = 1
                else:
                    batch_size["aux_lang"] = torch.sum(dataset_batch["use_for_aux_lang_loss"]).detach()  # type:ignore

                if self.use_clip_auxiliary_loss:
                    lang_clip_loss = self.clip_auxiliary_loss(
                        seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"]
                    )

            critic_loss += losses["loss_critic"]
            actor_loss += losses["loss_actor"]
            entropy_loss += losses["loss_entropy"]
            kl_loss += losses["loss_kl"]
            bc_loss += losses["loss_bc"]
            batch_size[self.modality_scope] = dataset_batch["discrete_action"].shape[1]
            total_bs += dataset_batch["discrete_action"].shape[1]

            # Training-stage per-dimension action error monitoring (ignore dt); log per-epoch to avoid extra cost
            try:
                with torch.no_grad():
                    # Shape [T, B, 4], semantics [dx, dy, dyaw, dt]; training path aligned to T-1
                    diffs_train = dataset_batch["rel_actions"][:-1] - ac_actions[:-1]
                    mse_dx_train = (diffs_train[..., 0] ** 2).mean()
                    mse_dy_train = (diffs_train[..., 1] ** 2).mean()
                    mse_dyaw_train = (diffs_train[..., 2] ** 2).mean()
                    mse_wo_dt_train = (diffs_train[..., :3] ** 2).mean()
                    self.log(
                        f"train/action_mse_dx_{self.modality_scope}",
                        mse_dx_train,
                        on_step=False,
                        on_epoch=True,
                        batch_size=batch_size[self.modality_scope],
                    )
                    self.log(
                        f"train/action_mse_dy_{self.modality_scope}",
                        mse_dy_train,
                        on_step=False,
                        on_epoch=True,
                        batch_size=batch_size[self.modality_scope],
                    )
                    self.log(
                        f"train/action_mse_dyaw_{self.modality_scope}",
                        mse_dyaw_train,
                        on_step=False,
                        on_epoch=True,
                        batch_size=batch_size[self.modality_scope],
                    )
                    self.log(
                        f"train/action_mse_wo_dt_{self.modality_scope}",
                        mse_wo_dt_train,
                        on_step=False,
                        on_epoch=True,
                        batch_size=batch_size[self.modality_scope],
                    )
            except Exception:
                pass


        critic_loss = critic_loss / len(batch)
        actor_loss = actor_loss / len(batch)
        entropy_loss = entropy_loss / len(batch)
        kl_loss = kl_loss / len(batch)
        bc_loss = bc_loss / len(batch)
        loss_policy = actor_loss + bc_loss + kl_loss - entropy_loss

        if self.use_clip_auxiliary_loss:
            loss_policy = loss_policy + self.clip_auxiliary_loss_beta * lang_clip_loss
            self.log(
                "train/lang_clip_loss",
                self.clip_auxiliary_loss_beta * lang_clip_loss,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size["aux_lang"],
                sync_dist=True,
            )

        critic_optimizer.zero_grad()
        self.manual_backward(critic_loss, retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        critic_optimizer.step()

        actor_optimizer.zero_grad()
        self.manual_backward(loss_policy)
        torch.nn.utils.clip_grad_norm_(self.actor_params, self.grad_clip)
        actor_optimizer.step()

        self.log("train/critic_loss", critic_loss, on_step=True, on_epoch=False, batch_size=total_bs)
        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=False, batch_size=total_bs)
        self.log("train/actor_loss", actor_loss, on_step=True, on_epoch=False, batch_size=total_bs)
        self.log("train/bc_loss", bc_loss, on_step=True, on_epoch=False, batch_size=total_bs)
        self.log("train/entropy_loss", entropy_loss, on_step=True, on_epoch=False, batch_size=total_bs)
        self.log("train/policy_loss", loss_policy, on_step=True, on_epoch=False, batch_size=total_bs)

        # Finish trajectory and update counter (only first), match actual API
        if self.trajectory_counter == 0:
            traj_logger.finish_trajectory()
        self.trajectory_counter += 1

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        batch_size: Dict[str, int] = {}
        total_bs = 0
        output = {}
        # DFoT history is reset + warm-started per modality inside the loop below
        traj_logger = TrajectoryLogger()
        if self.trajectory_counter == 0:
            traj_logger.start_new_trajectory({
                "stage": "validation",
                "global_step": int(self.global_step),
                "batch_idx": batch_idx
            })

        for self.modality_scope, dataset_batch in batch.items():
            # Reset and warm-start DFoT history per modality
            if self.lwm_enabled and "vae_latent" in dataset_batch:
                self._reset_dfot_history()
                self._warm_start_dfot_history(dataset_batch["vae_latent"])
            latents = dataset_batch["vae_latent"]
            assert latents.shape[-2:] == (32, 32), f"Validation requires 32x32 latent; got {latents.shape}"
            latents = torch.flatten(latents, start_dim=2)
            codes = self.perceptual_encoder(latents)
            if "lang" in self.modality_scope:
                latent_goal = self.language_goal(dataset_batch["lang"])
                raw_instructions_batch = dataset_batch.get("lang_raw", None)
            else:
                latent_goal = self.visual_goal(codes[-1])
                raw_instructions_batch = None

            losses, rewards, returns, ac_actions, pp_state, pr_state, seq_feat, sampled_plan_pp, sampled_plan_pr, predicted_states = (
                self.forward_val(
                    latents,
                    codes,
                    latent_goal,
                    dataset_batch["state_info"]["robot_obs"],
                    dataset_batch["vae_latent"],
                    raw_instructions=raw_instructions_batch,
                )
            )
            # Compatible with different batch formats: support top-level 'rel_actions' or nested 'actions.rel_actions'; fallback to 'discrete_action' if necessary
            rel_actions = dataset_batch.get("rel_actions")
            if rel_actions is None:
                actions_dict = dataset_batch.get("actions")
                if isinstance(actions_dict, dict):
                    rel_actions = actions_dict.get("rel_actions")
            if rel_actions is None:
                rel_actions = dataset_batch.get("discrete_action")
            if rel_actions is None:
                try:
                    # Safe fallback: create zero actions matching predicted actions' shape to avoid crash
                    rel_actions = torch.zeros_like(ac_actions)
                except Exception:
                    raise KeyError("Validation batch missing 'rel_actions' or 'discrete_action' field; cannot compute action-related metrics")

            batch_size[self.modality_scope] = rel_actions.shape[1]
            total_bs += rel_actions.shape[1]

            # Save logic: by default all ranks save to their own subdirs to avoid only rank0 writing
            # To save only from global_zero, set env var `LCVN_AC_SAVE_ONLY_GLOBAL_ZERO=1`
            try:
                import os as _os
                _save_only_global_zero = str(_os.environ.get("LCVN_AC_SAVE_ONLY_GLOBAL_ZERO", "0")).lower() in ("1", "true", "yes")
            except Exception:
                _save_only_global_zero = False

            # Compatible with Lightning's rank retrieval
            rank = getattr(getattr(self, 'trainer', None), 'global_rank', getattr(self, 'global_rank', 0))
            is_global_zero = bool(getattr(getattr(self, 'trainer', None), 'is_global_zero', rank == 0))

            if (not _save_only_global_zero) or (_save_only_global_zero and is_global_zero):
                try:
                    from pathlib import Path as _Path
                    import json as _json
                    import numpy as _np
                    try:
                        from PIL import Image as _Image
                    except Exception:
                        _Image = None
                    # Root save dir configurable via LCVN_AC_VALIDATION_SAVE_ROOT; if unset use previous default absolute path
                    try:
                        import os as _os
                        _save_root_env = _os.environ.get("LCVN_AC_VALIDATION_SAVE_ROOT", None)
                    except Exception:
                        _save_root_env = None
                    save_root_base = _Path(_save_root_env) if _save_root_env else _Path("nwm_prediction/validation")
                    # Separate write dirs by rank to avoid conflicts
                    save_root = save_root_base / (f"rank_{rank}" if not _save_only_global_zero else "rank_0")
                    # Debug output and pre-create root dir for on-site confirmation of save path
                    try:
                        save_root.mkdir(parents=True, exist_ok=True)
                    except Exception as _e_mkdir:
                        pass
                    B = ac_actions.shape[1]
                    T = ac_actions.shape[0]
                    # Incremental saving: if LCVN_AC_VALIDATION_APPEND=1, append offset to existing batch_* dir to avoid overwrite
                    try:
                        import os as _os
                        _append_batches = str(_os.environ.get("LCVN_AC_VALIDATION_APPEND", "0")).lower() in ("1", "true", "yes")
                    except Exception:
                        _append_batches = False
                    _batch_offset = 0
                    if _append_batches:
                        try:
                            existing = []
                            for _p in save_root.glob("batch_*"):
                                try:
                                    if _p.is_dir() and _p.name.startswith("batch_"):
                                        _idx_str = _p.name.split("_", 1)[1]
                                        if _idx_str.isdigit():
                                            existing.append(int(_idx_str))
                                except Exception:
                                    pass
                            if existing:
                                _batch_offset = max(existing) + 1
                        except Exception:
                            _batch_offset = 0
                    # Try to get identifier per sample from idx
                    idx_tensor = dataset_batch.get("idx")
                    for b in range(B):
                        sample_id = None
                        try:
                            if idx_tensor is not None:
                                if hasattr(idx_tensor, 'shape') and len(idx_tensor.shape) > 0:
                                    sample_id = int(idx_tensor[b].item())
                                else:
                                    sample_id = int(idx_tensor)
                        except Exception:
                            pass
                        _batch_dir_name = f"batch_{_batch_offset + batch_idx}" if _batch_offset > 0 else f"batch_{batch_idx}"
                        sample_dir = save_root / _batch_dir_name / self.modality_scope / (f"sample_{sample_id}" if sample_id is not None else f"sample_{b}")
                        sample_dir.mkdir(parents=True, exist_ok=True)

                        # Save predicted and GT action sequences (entire segment for this sample)
                        pred_actions_json = [ac_actions[t, b].detach().cpu().numpy().tolist() for t in range(T)]
                        try:
                            import os as _os
                            with open(sample_dir / "actions_pred.json", "w", encoding="utf-8") as _fp:
                                _fp.write(_json.dumps(pred_actions_json, indent=2))
                                _fp.flush()
                                try:
                                    _os.fsync(_fp.fileno())
                                except Exception:
                                    pass
                        except Exception as _e_json_pred:
                            pass
                        if rel_actions is not None:
                            try:
                                gt_actions_json = [rel_actions[t, b].detach().cpu().numpy().tolist() for t in range(T)]
                                with open(sample_dir / "actions_gt.json", "w", encoding="utf-8") as _fp:
                                    _fp.write(_json.dumps(gt_actions_json, indent=2))
                                    _fp.flush()
                                    try:
                                        _os.fsync(_fp.fileno())
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                        # Save DFoT/NWM predicted latent sequence (tensor and npz) for later analysis
                        try:
                            if predicted_states is not None:
                                latents_pred_tbchw = predicted_states[:, b]  # (T, 4, 32, 32)
                                import torch as _torch, os as _os
                                # torch.save with file handle then flush+fsync to ensure visibility
                                with open(sample_dir / "latents_pred.pt", "wb") as _fp:
                                    _torch.save(latents_pred_tbchw.detach().cpu(), _fp)
                                    _fp.flush()
                                    try:
                                        _os.fsync(_fp.fileno())
                                    except Exception:
                                        pass
                                try:
                                    with open(sample_dir / "latents_pred.npz", "wb") as _fp:
                                        _np.savez_compressed(_fp, latents=latents_pred_tbchw.detach().cpu().numpy())
                                        _fp.flush()
                                        try:
                                            _os.fsync(_fp.fileno())
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Save GT latent sequence (if available in data)
                        try:
                            gt_latents = dataset_batch.get("vae_latent")
                            if gt_latents is not None:
                                latents_gt_tbchw = gt_latents[:, b]  # (T, 4, 32, 32)
                                import torch as _torch, os as _os
                                with open(sample_dir / "latents_gt.pt", "wb") as _fp:
                                    _torch.save(latents_gt_tbchw.detach().cpu(), _fp)
                                    _fp.flush()
                                    try:
                                        _os.fsync(_fp.fileno())
                                    except Exception:
                                        pass
                                try:
                                    with open(sample_dir / "latents_gt.npz", "wb") as _fp:
                                        _np.savez_compressed(_fp, latents=latents_gt_tbchw.detach().cpu().numpy())
                                        _fp.flush()
                                        try:
                                            _os.fsync(_fp.fileno())
                                        except Exception:
                                            pass
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # Save predicted images (if available)
                        if predicted_states is not None and _Image is not None:
                            for t in range(min(T, predicted_states.shape[0])):
                                try:
                                    with torch.no_grad():
                                        latent_t = predicted_states[t, b]  # (4, 32, 32)
                                        image_tensor_decoded = self.vae.decode(latent_t / 0.18215).sample
                                        samples = image_tensor_decoded.squeeze(0)
                                        samples = samples * 0.5 + 0.5
                                        samples = (samples * 255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).cpu()
                                        import os as _os
                                        _img = _Image.fromarray(samples.numpy())
                                        with open(sample_dir / f"pred_{t}.png", "wb") as _fp:
                                            _img.save(_fp, format="PNG")
                                            _fp.flush()
                                            try:
                                                _os.fsync(_fp.fileno())
                                            except Exception:
                                                pass
                                except Exception:
                                    pass

                        # Save GT images (if available)
                        gt_latents = dataset_batch.get("vae_latent")
                        if gt_latents is not None and _Image is not None:
                            # Data dims are [T, B, 4, 32, 32]
                            for t in range(min(T, gt_latents.shape[0])):
                                try:
                                    with torch.no_grad():
                                        gt_latent_t = gt_latents[t, b]
                                        gt_img_decoded = self.vae.decode(gt_latent_t / 0.18215).sample
                                        gt_samples = gt_img_decoded.squeeze(0)
                                        gt_samples = gt_samples * 0.5 + 0.5
                                        gt_samples = (gt_samples * 255).clamp(0, 255).permute(1, 2, 0).to(torch.uint8).cpu()
                                        import os as _os
                                        _img_gt = _Image.fromarray(gt_samples.numpy())
                                        with open(sample_dir / f"gt_{t}.png", "wb") as _fp:
                                            _img_gt.save(_fp, format="PNG")
                                            _fp.flush()
                                            try:
                                                _os.fsync(_fp.fileno())
                                            except Exception:
                                                pass
                                except Exception:
                                    pass

                    # Record one action comparison (logging only, take first sample first step)
                    try:
                        traj_logger.log_action_prediction(
                            ac_actions[0].detach().cpu(),
                            rel_actions[0].detach().cpu() if rel_actions is not None else None
                        )
                    except Exception:
                        pass
                except Exception as _save_error:
                    import traceback
                    logger.error(f"Severe error during validation saving: batch_idx={batch_idx}, rank={rank}, modality={self.modality_scope}: {type(_save_error).__name__}: {_save_error}")
                    traceback.print_exc()

            if "lang" in self.modality_scope and self.use_clip_auxiliary_loss:
                val_pred_clip_loss = self.clip_auxiliary_loss(
                    seq_feat, latent_goal, dataset_batch["use_for_aux_lang_loss"]
                )
                self.log(
                    "val/lang_clip_loss",
                    val_pred_clip_loss,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size[self.modality_scope],
                )

            act_mse = action_mse(rel_actions[:-1], ac_actions[:-1])
            # Additionally record per-dimension MSE (ignore dt) for finer analysis
            try:
                with torch.no_grad():
                    # Shape: [T-1, B, 4], semantics: [dx, dy, dyaw, dt]
                    diffs = rel_actions[:-1] - ac_actions[:-1]
                    mse_dx = (diffs[..., 0] ** 2).mean()
                    mse_dy = (diffs[..., 1] ** 2).mean()
                    mse_dyaw = (diffs[..., 2] ** 2).mean()
                    mse_wo_dt = (diffs[..., :3] ** 2).mean()  # overall MSE for dx, dy, dyaw only
                    # Record per-dimension metrics
                    self.log(
                        f"val/action_mse_dx_{self.modality_scope}",
                        mse_dx,
                        on_step=False,
                        on_epoch=True,
                        batch_size=batch_size[self.modality_scope],
                    )
                    self.log(
                        f"val/action_mse_dy_{self.modality_scope}",
                        mse_dy,
                        on_step=False,
                        on_epoch=True,
                        batch_size=batch_size[self.modality_scope],
                    )
                    self.log(
                        f"val/action_mse_dyaw_{self.modality_scope}",
                        mse_dyaw,
                        on_step=False,
                        on_epoch=True,
                        batch_size=batch_size[self.modality_scope],
                    )
                    self.log(
                        f"val/action_mse_wo_dt_{self.modality_scope}",
                        mse_wo_dt,
                        on_step=False,
                        on_epoch=True,
                        batch_size=batch_size[self.modality_scope],
                    )
            except Exception:
                pass
            metrics = {
                "metric_return-unnorm": returns.mean(),
                "metric_reward-latent": torch.stack(rewards[:-1]).mean(),
                "metric_action-mse": sum(act_mse) / len(act_mse),
            }
            self.log(
                f"val/kl_loss_{self.modality_scope}",
                losses["loss_kl"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
            self.log(
                f"val/return-unnorm_{self.modality_scope}",
                metrics["metric_return-unnorm"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
            self.log(
                f"val/reward-latent_{self.modality_scope}",
                metrics["metric_reward-latent"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
            self.log(
                f"val/action_mse_{self.modality_scope}",
                metrics["metric_action-mse"],
                on_step=False,
                on_epoch=True,
                batch_size=batch_size[self.modality_scope],
            )
            output[f"sampled_plan_pp_{self.modality_scope}"] = sampled_plan_pp
            output[f"sampled_plan_pr_{self.modality_scope}"] = sampled_plan_pr
            output[f"idx_{self.modality_scope}"] = dataset_batch["idx"]
        # Record one validation trajectory end (only first, avoid excessive logging)
        if self.trajectory_counter == 0:
            try:
                traj_logger.finish_trajectory()
            except Exception:
                pass
        return output

    def loss(self, acts, gt_acts, values, returns, entropy, pp_state, pr_state, log_probs=None, means=None) -> Dict[str, Tensor]:

        loss_critic = self.critic_loss(values, returns)
        loss_actor = self.actor_loss(returns, log_probs)
        loss_entropy = self.entropy_loss(entropy)
        loss_kl = self.kl_loss(pp_state, pr_state)

        loss_bc = self.bc_loss(acts, gt_acts, means)

        losses = {
            "loss_critic": loss_critic,
            "loss_actor": loss_actor,
            "loss_entropy": loss_entropy,
            "loss_kl": loss_kl,
            "loss_bc": loss_bc,
        }

        return losses

    def critic_loss(self, values, returns):
        loss_critic = 0.5 * F.mse_loss(values[:-1], returns.detach()[:-1])

        return loss_critic

    def actor_loss(self, returns, log_probs=None):
        if log_probs is not None:
            # REINFORCE: gradient flows through log_probs back to policy parameters.
            # Since DFoT is @torch.no_grad(), returns cannot provide gradients on their own.
            # log_probs: (T, B), returns: (T, B)
            loss_actor = self.actor_alpha * -(log_probs[:-1] * returns[:-1].detach()).mean()
        else:
            # Fallback: no gradient through returns (legacy behavior)
            loss_actor = self.actor_alpha * (-returns[:-1]).mean()
        return loss_actor

    def bc_loss(self, acts, gt_acts, means=None):
        if means is not None:
            # Compute BC loss in pre-tanh space to avoid vanishing gradients from saturation.
            # means: (T, B, D) — raw MLP output before tanh
            # target: arctanh(gt_acts / scale) maps GT actions to pre-tanh space
            scale = self.action_decoder.scale  # (D,) buffer
            gt_normalized = (gt_acts / scale).clamp(-0.9999, 0.9999)
            gt_pretanh = torch.atanh(gt_normalized)  # (T, B, D)
            loss_bc = self.bc_alpha * F.mse_loss(means[:-1], gt_pretanh[:-1])
        else:
            loss_bc = self.bc_alpha * F.mse_loss(acts[:-1], gt_acts[:-1])
        return loss_bc
    

    def entropy_loss(self, entropy: Tensor) -> Tensor:
        return entropy.mean() * self.eta

    def kl_loss(self, pp_state: State, pr_state: State) -> torch.Tensor:
        """
        Compute the KL divergence loss between the distributions of the plan recognition and plan proposal network.
        We use KL balancing similar to "MASTERING ATARI WITH DISCRETE WORLD MODELS" by Hafner et al.
        (https://arxiv.org/pdf/2010.02193.pdf)

        Args:
            pp_state: Namedtuple containing the parameters of the distribution produced by plan proposal network.
            pr_state: Namedtuple containing the parameters of the distribution produced by plan recognition network.

        Returns:
            Scaled KL loss.
        """
        pp_dist = self.dist.get_dist(pp_state)  # prior
        pr_dist = self.dist.get_dist(pr_state)  # posterior
        kl_lhs = D.kl_divergence(self.dist.get_dist(self.dist.detach_state(pr_state)), pp_dist).mean()
        kl_rhs = D.kl_divergence(pr_dist, self.dist.get_dist(self.dist.detach_state(pp_state))).mean()

        alpha = self.kl_balancing_mix
        kl_loss = alpha * kl_lhs + (1 - alpha) * kl_rhs
        kl_loss_scaled = kl_loss * self.kl_beta
        return kl_loss_scaled

    def clip_auxiliary_loss(self, seq_vis_feat, encoded_lang, use_for_aux_loss):
        """
        CLIP style contrastive loss, adapted from 'Learning transferable visual models from natural language
        supervision' by Radford et al.
        We maximize the cosine similarity between the visual features of the sequence i and the corresponding language
        features while, at the same time, minimizing the cosine similarity between the current visual features and other
        language instructions in the same batch.

        Args:
            seq_vis_feat: Visual embedding.
            encoded_lang: Language goal embedding.
            use_for_aux_loss: Mask of which sequences in the batch to consider for auxiliary loss.

        Returns:
            Contrastive loss.
        """
        assert self.use_clip_auxiliary_loss is not None
        skip_batch = False
        if use_for_aux_loss is not None:
            if not torch.any(use_for_aux_loss):
                # Hack for avoiding a crash when using ddp. Loss gets multiplied with 0 at the end of method to
                # effectively skip whole batch. We do a dummy forward pass, to prevent ddp from complaining.
                # see https://github.com/pytorch/pytorch/issues/43259
                skip_batch = True
                seq_vis_feat = seq_vis_feat[0:1]
                encoded_lang = encoded_lang[0:1]
            else:
                seq_vis_feat = seq_vis_feat[use_for_aux_loss]
                encoded_lang = encoded_lang[use_for_aux_loss]
        image_features, lang_features = self.proj_vis_lang(seq_vis_feat, encoded_lang)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = lang_features / lang_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # symmetric loss function
        labels = torch.arange(logits_per_image.shape[0], device=text_features.device)
        loss_i = cross_entropy(logits_per_image, labels)
        loss_t = cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        if skip_batch:
            loss *= 0
        return loss

    def get_value(self, x_c: Tensor, x_g: Tensor, x_p: Tensor) -> Tensor:
        x = torch.cat([x_c, x_g, x_p], dim=-1)
        return self.critic(x)

    def wm_step(self, latent, action):
        next_latent = torch.zeros_like(latent)
        return next_latent # 2048

    def reset(self):
        self.plan = None
        self.latent_goal = None
        self.rollout_step_counter = 0
        self.pre_action = torch.zeros((1, 1, 7)).to(self.device)
        self.pre_action[:, :, -1] = 1.0
        if self.wm is None:
            # Without WM, do not initialize WM state; keep placeholders None and require caller to use DFoT/NWM path
            self.wm_in_state = None
            self.wm_reset = torch.ones((1, 1, 1), dtype=torch.bool)
        else:
            self.wm_in_state = self.wm.rssm_core.init_state(1)
            self.wm_reset = torch.ones((1, 1, 1), dtype=torch.bool)

    @torch.no_grad()
    def step(self, obs, goal):

        if self.rollout_step_counter % self.replan_freq == 0:

            if isinstance(goal, str):
                embedded_lang = torch.from_numpy(self.lang_embeddings[goal]).to(self.device).squeeze(0).float()
                self.latent_goal = self.language_goal(embedded_lang)

            else:
                raise NotImplementedError("Only language goals are supported for now.")

            # Allow passing through WM features only when WM exists; otherwise require upper layer to use DFoT/NWM inference path
            assert self.wm is not None, "world_model is None: step() requires WM features; use DFoT/NWM inference interface"
            c_state, self.wm_in_state = self.obs_to_wm_latent(obs, self.wm_in_state, self.wm_reset, self.pre_action)
            self.plan = self.get_pp_plan(c_state, self.latent_goal)

        else:
            assert self.wm is not None, "world_model is None: step() requires WM features; use DFoT/NWM inference interface"
            c_state, self.wm_in_state = self.obs_to_wm_latent(obs, self.wm_in_state, self.wm_reset, self.pre_action)

        action, _, _, _ = self.action_decoder.get_action(c_state, self.latent_goal, self.plan)

        self.wm_reset = torch.zeros((1, 1, 1), dtype=torch.bool)
        self.pre_action = action.unsqueeze(0)
        self.rollout_step_counter += 1
        return action

    @torch.no_grad()
    def obs_to_wm_latent(self, obs, in_state, reset, action):
        assert self.wm is not None, "world_model is None: obs_to_wm_latent cannot be called."
        action[:, :, -1] = 1 if action[:, :, -1] > 0 else -1
        features, out_state = self.wm.infer_features(
            obs["rgb_obs"]["rgb_static"],
            obs["rgb_obs"]["rgb_gripper"],
            obs["robot_obs"],
            action,
            obs["robot_obs_raw"],
            reset,
            in_state,
            local_act=action,
        )

        return features[0], out_state

    def get_pp_plan(self, latent_obs: dict, latent_goal: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use plan proposal network to sample new plan using a visual goal embedding.

        Args:
            obs: Observation from environment.
            goal: Embedded language instruction.

        Returns:
            sampled_plan: Sampled plan.
            latent_goal: Encoded language goal.
        """
        with torch.no_grad():
            code = self.perceptual_encoder(latent_obs)
            # ------------Plan Proposal------------ #
            pp_state = self.plan_proposal(code, latent_goal)
            pp_dist = self.dist.get_dist(pp_state)
            sampled_plan = self.dist.sample_latent_plan(pp_dist)
        return sampled_plan

    
    @rank_zero_only
    def on_train_epoch_start(self) -> None:
        # Lazy initialize DFoT inferencer (only on rank 0)
        if self._dfot_init_pending and self.dfot_inferencer is None:
            try:
                device_str = f'cuda:{self.device.index}' if self.device.type == 'cuda' else 'cpu'
                log_rank_0(f"Initializing DFoT pure inference on {device_str}...")

                from lcvn_ac.utils.pure_infer import create_inferencer
                self.dfot_inferencer = create_inferencer(
                    dfot_checkpoint_path=self.dfot_checkpoint_path,
                    vae_checkpoint_path=self.dfot_vae_checkpoint_path,
                    device=device_str,
                    denoise_steps=int(self.predict_diffusion_steps)
                )

                log_rank_0("DFoT pure inference initialized successfully.")
                self._dfot_init_pending = False
            except Exception as e:
                log_rank_0(f"ERROR initializing DFoT pure inference: {e}")
                self.use_dfot = False
                self.dfot_inferencer = None
                self._dfot_init_pending = False

        # At training start, clean old trajectory log directory (fail silently)
        try:
            import shutil
            from pathlib import Path
            log_dir = Path("trajectory_logs")
            if log_dir.exists():
                for p in log_dir.glob("*"):
                    try:
                        if p.is_file():
                            p.unlink()
                        else:
                            shutil.rmtree(p)
                    except Exception:
                        pass
        except Exception:
            pass

        logger.info(f"Start training epoch {self.current_epoch}")

    @rank_zero_only
    def on_validation_epoch_start(self) -> None:
        log_rank_0(f"Start validation epoch {self.current_epoch}")

    def on_save_checkpoint(self, checkpoint):
        keys_to_remove = [key for key in checkpoint["state_dict"].keys() if key.startswith("clip.")]
        for key in keys_to_remove:
            del checkpoint["state_dict"][key]

    def load_lang_embeddings(self, embeddings_path):
        """
        This has to be called before inference. Loads the lang embeddings from the dataset.

        Args:
            embeddings_path: Path to <dataset>/validation/embeddings.npy
        """
        embeddings = np.load(embeddings_path, allow_pickle=True).item()
        # we want to get the embedding for full sentence, not just a task name
        self.lang_embeddings = {v["ann"][0]: v["emb"] for k, v in embeddings.items()}

    @torch.no_grad()
    def predict_action_sequence(
        self,
        initial_image: Tensor,
        language_embedding: Tensor,
        max_steps: int,
        stop_threshold: float = 0.01,
        raw_instruction: Optional[str] = None,
        save_trajectory_dir: Optional[str] = None
    ) -> List[Dict[str, Tensor]]:
        self.eval()

        # Inference entry also resets DFoT history to ensure each predict is independent
        if getattr(self, 'use_dfot', False):
            self._reset_dfot_history()

        from diffusion import create_diffusion
        # Use configurable sequence prediction diffusion steps
        self.diffusion = create_diffusion(str(self.predict_diffusion_steps))

        initial_image = initial_image.to(self.device)
        v_state_cur = self.vae.encode(initial_image).latent_dist.sample() * 0.18215
        
        latents = torch.flatten(v_state_cur, start_dim=1)
        initial_code_128 = self.perceptual_encoder(latents)

        language_embedding = language_embedding.to(self.device)
        if language_embedding.dim() == 1:
            language_embedding = language_embedding.unsqueeze(0)
        latent_goal = self.language_goal(language_embedding)

        pp_state = self.plan_proposal(initial_code_128, latent_goal)
        pp_dist = self.dist.get_dist(pp_state)
        sampled_plan = self.dist.sample_latent_plan(pp_dist)

        raw_instructions_batch = [raw_instruction] if raw_instruction and self.nwm_config.get("use_instruction", False) else None

        predicted_steps: List[Dict[str, Tensor]] = []
        actions_list: List[Tensor] = []
        latents_list: List[Tensor] = []

        # Trajectory logger: enable and record prediction entry context
        traj_logger = TrajectoryLogger()
        try:
            traj_logger.enable()
            traj_logger.start_new_trajectory({
                "mode": "predict_action_sequence",
                "raw_instruction": raw_instruction or "",
                "max_steps": int(max_steps),
                "stop_threshold": float(stop_threshold)
            })
            traj_logger.log_model_forward(
                "PREDICT_START",
                {
                    "initial_image_shape": initial_image.shape,
                    "language_embedding_shape": language_embedding.shape,
                    "latent_goal_shape": latent_goal.shape,
                    "sampled_plan_shape": sampled_plan.shape,
                }
            )
        except Exception:
            pass
        stopped = False

        for step_count in range(max_steps):
            current_feature = torch.flatten(v_state_cur, start_dim=1)

            continuous_action, ent, _, _ = self.action_decoder.get_action(current_feature, latent_goal, sampled_plan)

            predicted_steps.append({
                'action': continuous_action,
                'latent': v_state_cur
            })
            try:
                actions_list.append(continuous_action.detach().cpu())
                latents_list.append(v_state_cur.detach().cpu())
                traj_logger.log_action_prediction(continuous_action.detach().cpu())
                traj_logger.log_latent_output(v_state_cur.detach().cpu(), stage=f"STEP_{step_count+1}")
            except Exception:
                pass

            if torch.all(torch.abs(continuous_action) < stop_threshold):
                logger.info(f"✅ Prediction stopped at step {step_count + 1} (all action values below threshold {stop_threshold}).")
                stopped = True
                break

            v_state_cur = self.nwm_step(
                current_vae_latent=v_state_cur,
                action_batch=continuous_action,
                raw_instructions_batch=raw_instructions_batch
            )

        if not stopped:
            # Reduce noise: only notify on rank 0 when detailed logs explicitly enabled
            if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0 and os.environ.get("LCVN_AC_VERBOSE", "0") in ("1", "true", "yes"):
                logger.info(f"⚠️ Prediction reached max steps {max_steps} without triggering stop condition (threshold {stop_threshold}).")

        # Save full trajectory to directory (includes actions, latents, and metadata)
        try:
            from datetime import datetime as _dt
            ts = _dt.now().strftime("%Y%m%d_%H%M%S")
            base_dir = Path(save_trajectory_dir) if save_trajectory_dir else (TrajectoryLogger().log_dir / f"predict_{ts}")
            base_dir.mkdir(parents=True, exist_ok=True)

            import json as _json
            actions_path = base_dir / "actions.json"
            latents_path = base_dir / "latents.pt"
            meta_path = base_dir / "meta.json"

            actions_serializable = [a.tolist() for a in actions_list]
            with actions_path.open("w", encoding="utf-8") as f:
                _json.dump(actions_serializable, f, indent=2)

            try:
                import torch as _torch
                if len(latents_list) > 0:
                    latents_stack = _torch.stack(latents_list)
                    _torch.save(latents_stack, latents_path)
            except Exception:
                pass

            meta = {
                "raw_instruction": raw_instruction or "",
                "steps": len(actions_list),
                "stopped": bool(stopped),
                "stop_threshold": float(stop_threshold),
            }
            with meta_path.open("w", encoding="utf-8") as f:
                _json.dump(meta, f, indent=2)
        except Exception:
            pass

        try:
            traj_logger.finish_trajectory()
        except Exception:
            pass

        return predicted_steps
