import logging
from pathlib import Path
import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.utilities import rank_zero_only
from lcvn_ac.utils.info_utils import print_system_env_info, get_last_checkpoint, setup_logger, setup_callbacks

import warnings
warnings.filterwarnings("ignore")

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../config", config_name="train_ag")
def train(cfg: DictConfig) -> None:
    """
    This is called to start a training.
    Args:
        cfg: hydra config
    """
    if cfg.exp_dir is None:
        cfg.exp_dir = hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    model_dir = Path(cfg.exp_dir) / "model_weights/"
    cfg.callbacks.checkpoint.dirpath = model_dir
    os.makedirs(model_dir, exist_ok=True)
    log_rank_0(f"Training a Actor Critic with the following config:\n{OmegaConf.to_yaml(cfg)}")
    log_rank_0(print_system_env_info())

    seed_everything(cfg.seed, workers=True)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    ag_chk = get_last_checkpoint(model_dir)

    if ag_chk is not None:
        if "lcvn-ac" in cfg.behavior_model.name:
            from lcvn_ac.behavior_models.lumos import LUMOS

            model = LUMOS.load_from_checkpoint(ag_chk.as_posix())
        else:
            raise NotImplementedError(f"Unknown model: {cfg.behavior_model.name}")
    else:
        model = hydra.utils.instantiate(cfg.behavior_model)

    # --- Enforce 32x32/4096 latent configuration sanity checks ---
    try:
        pe = getattr(model, 'perceptual_encoder', None)
        pe_flat = getattr(pe, 'latent_dim_flat', None)
        log_rank_0(f"PerceptualEncoder latent_dim_flat: {pe_flat}")
        if pe_flat is not None:
            assert pe_flat == 4096, f"Expected perceptual_encoder.latent_dim_flat=4096, got {pe_flat}"

        actor_pf = getattr(getattr(model, 'action_decoder', None), 'perceptual_features', None)
        critic_pf = getattr(getattr(model, 'critic', None), 'perceptual_features', None)
        log_rank_0(f"ActionDecoder perceptual_features: {actor_pf}; Critic perceptual_features: {critic_pf}")
        if actor_pf is not None:
            assert actor_pf == 4096, f"ActionDecoder perceptual_features={actor_pf}, expected 4096"
        if critic_pf is not None:
            assert critic_pf == 4096, f"Critic perceptual_features={critic_pf}, expected 4096"

        vae_shape = getattr(model, 'vae_latent_shape_chw', None)
        if vae_shape is None:
            setattr(model, 'vae_latent_shape_chw', (4, 32, 32))
            log_rank_0("Set model.vae_latent_shape_chw to (4, 32, 32)")
        else:
            if isinstance(vae_shape, (list, tuple)):
                assert tuple(vae_shape) == (4, 32, 32), f"vae_latent_shape_chw={vae_shape}, expected (4, 32, 32)"
            else:
                log_rank_0(f"vae_latent_shape_chw present but non-sequence: {vae_shape}")
    except AssertionError as e:
        logger.error(f"Latent configuration error: {e}")
        raise
    except Exception as e:
        log_rank_0(f"Skipping latent config validation due to: {e}")

    trainer_args = {**cfg.trainer, "logger": setup_logger(cfg), "callbacks": setup_callbacks(cfg.callbacks)}
    trainer = Trainer(**trainer_args)

    trainer.fit(model, datamodule=datamodule, ckpt_path=ag_chk)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    """
    Log the information using the logger at rank 0.
    """
    logger.info(*args, **kwargs)


if __name__ == "__main__":
    train()
