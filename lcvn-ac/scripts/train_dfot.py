#!/usr/bin/env python3
"""
Lcvn-AC DFoT training script
Train Lcvn-AC behavior model using DFoT API
"""

try:
    import flash_attn.flash_attn_interface as _fa
    for _old, _new in [("_wrapped_flash_attn_forward", "_flash_attn_forward"),
                       ("_wrapped_flash_attn_backward", "_flash_attn_backward")]:
        if not hasattr(_fa, _old) and hasattr(_fa, _new):
            setattr(_fa, _old, getattr(_fa, _new))
except ImportError:
    pass

import logging
import sys
import os
import time
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.utilities import rank_zero_only

# Add project root to sys.path
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from lcvn_ac.utils.info_utils import print_system_env_info, get_last_checkpoint, setup_logger, setup_callbacks
from lcvn_ac.utils.trajectory_logger import disable_trajectory_logging

logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../config", config_name="train_dfot")
def train_dfot(cfg: DictConfig) -> None:
    """
    Launch DFoT training
    
    Args:
        cfg: Hydra configuration
    """
    # Set up experiment directory
    if cfg.exp_dir is None:
        cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    
    model_dir = Path(cfg.exp_dir) / "model_weights/"
    cfg.callbacks.checkpoint.dirpath = model_dir
    os.makedirs(model_dir, exist_ok=True)

    # Force-disable trajectory logging to avoid local writes
    try:
        disable_trajectory_logging()
    except Exception:
        pass
    
    log_rank_0("🚀 Starting Lcvn-AC DFoT training")
    log_rank_0(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())
    
    # Check Lcvn WorldModel configuration
    _lwm = cfg.behavior_model.get('lcvn_worldmodel')
    if not (_lwm and bool(getattr(_lwm, 'enabled', False))):
        raise ValueError("❌ Lcvn WorldModel not enabled! Please set behavior_model.lcvn_worldmodel.enabled: true")
    
    # DFoT uses pure inference; no HTTP checks or mode switches
    log_rank_0("🧠 DFoT pure inference mode: enabled (no HTTP)")
    
    # Set random seed
    seed_everything(cfg.seed, workers=True)
    
    # Initialize datamodule
    log_rank_0("📊 Initializing datamodule...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    
    # In pure inference mode, DFoT replaces external world_model
    log_rank_0("🌍 DFoT mode: DFoT acts as Lcvn WorldModel and handles environment prediction")
    
    # Check if explicit ckpt_path is provided (precise ckpt selection)
    explicit_ckpt = None
    try:
        if hasattr(cfg, "ckpt_path") and cfg.ckpt_path:
            explicit_ckpt = Path(cfg.ckpt_path)
            if not explicit_ckpt.exists():
                raise FileNotFoundError(f"Specified ckpt does not exist: {explicit_ckpt}")
    except Exception as e:
        log_rank_0(f"⚠️ Explicit ckpt_path error: {e}")
        explicit_ckpt = None

    # Discover available checkpoint
    ag_chk = explicit_ckpt if explicit_ckpt is not None else get_last_checkpoint(model_dir)

    # Create model via Hydra; let Trainer.fit(ckpt_path=...) handle checkpoint restore
    if ag_chk is not None:
        log_rank_0(f"📂 Will resume from checkpoint via Trainer.fit: {ag_chk}")
    else:
        log_rank_0("🆕 No checkpoint found, training from scratch")
    model = hydra.utils.instantiate(cfg.behavior_model)
    
    # Validate Lcvn WorldModel configuration
    if hasattr(model, 'lwm_enabled') and model.lwm_enabled:
        log_rank_0("✅ Lcvn WorldModel (DFoT) enabled")
    else:
        log_rank_0("⚠️ Warning: Lcvn WorldModel not enabled or misconfigured")
    
    # Setup trainer
    log_rank_0("🏃 Setting up trainer...")
    trainer_args = {
        **cfg.trainer, 
        "logger": setup_logger(cfg), 
        "callbacks": setup_callbacks(cfg.callbacks)
    }
    trainer = Trainer(**trainer_args)
    
    # Start training
    log_rank_0("🎯 Starting training...")
    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=ag_chk)
        log_rank_0("🎉 Training completed!")
    except Exception as e:
        log_rank_0(f"❌ Training failed: {e}")
        raise

@rank_zero_only
def log_rank_0(*args, **kwargs):
    """Log only on rank 0"""
    logger.info(*args, **kwargs)

if __name__ == "__main__":
    train_dfot()
