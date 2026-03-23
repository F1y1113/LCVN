import os
import hydra
from pathlib import Path
from typing import Dict, List, Union
import git
import numpy as np
import pytorch_lightning
import torch
import tqdm
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger


def get_git_commit_hash(repo_path: Path) -> str:
    repo = git.Repo(search_parent_directories=True, path=repo_path.parent)
    assert repo, "not a repo"
    changed_files = [item.a_path for item in repo.index.diff(None)]
    if changed_files:
        print("WARNING uncommitted modified files: {}".format(",".join(changed_files)))
    return repo.head.object.hexsha


def info_packages() -> Dict[str, str]:
    return {
        "numpy": np.__version__,
        "pyTorch_version": torch.__version__,
        "pyTorch_debug": str(torch.version.debug),
        "pytorch-lightning": pytorch_lightning.__version__,
        "tqdm": tqdm.__version__,
    }


def get_all_checkpoints(checkpoint_folder: Path) -> List:
    if checkpoint_folder.is_dir():
        checkpoints = sorted(Path(checkpoint_folder).iterdir(), key=lambda chk: chk.stat().st_mtime)
        if len(checkpoints):
            return [chk for chk in checkpoints if chk.suffix == ".ckpt"]
    return []


def get_last_checkpoint(checkpoint_folder: Path) -> Union[Path, None]:
    # return newest checkpoint according to creation time
    checkpoints = get_all_checkpoints(checkpoint_folder)
    if len(checkpoints):
        return checkpoints[-1]
    return None


def info_cuda() -> Dict[str, Union[str, List[str]]]:
    return {
        "GPU": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        # 'nvidia_driver': get_nvidia_driver_version(run_lambda),
        "available": str(torch.cuda.is_available()),
        "version": torch.version.cuda,
    }


def print_system_env_info():
    details = {
        "Packages": info_packages(),
        "CUDA": info_cuda(),
    }
    lines = nice_print(details)
    text = os.linesep.join(lines)
    return text


def nice_print(details: Dict, level: int = 0) -> List:
    lines = []
    LEVEL_OFFSET = "\t"
    KEY_PADDING = 20
    for k in sorted(details):
        key = f"* {k}:" if level == 0 else f"- {k}:"
        if isinstance(details[k], dict):
            lines += [level * LEVEL_OFFSET + key]
            lines += nice_print(details[k], level + 1)
        elif isinstance(details[k], (set, list, tuple)):
            lines += [level * LEVEL_OFFSET + key]
            lines += [(level + 1) * LEVEL_OFFSET + "- " + v for v in details[k]]
        else:
            template = "{:%is} {}" % KEY_PADDING
            key_val = template.format(key, details[k])
            lines += [(level * LEVEL_OFFSET) + key_val]
    return lines


def setup_logger(cfg: DictConfig) -> Logger:
    """
    Set up the logger (tensorboard or wandb) from hydra config.
    Args:
        cfg: Hydra config
    Returns:
        logger
    """
    # Support safe handling of logger: none
    try:
        if not cfg.logger:
            return None
        # Hydra `logger: none` may map to an empty config or file.
        # If `_target_` is missing, do not instantiate and return None.
        if isinstance(cfg.logger, DictConfig) and "_target_" not in cfg.logger:
            return None
    except Exception:
        # Any exception falls back to no logger to avoid training interruption
        return None
    date_time = "_".join(cfg.exp_dir.split("/")[-2:])
    if cfg.comment != "":
        cfg.logger.name = "%s_%s" % (cfg.comment, date_time)
    else:
        cfg.logger.name = date_time
    cfg.logger.id = cfg.logger.name.replace("/", "_")
    logger = hydra.utils.instantiate(cfg.logger)

    return logger


def setup_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    Set up the callbacks form the hydra config.
    """
    # Instantiate each item and filter out None or entries without `_target_`
    instantiated: List[Callback] = []
    for cb in callbacks_cfg.values():
        try:
            if isinstance(cb, DictConfig) and "_target_" not in cb:
                continue
            obj = hydra.utils.instantiate(cb)
            if obj is not None:
                instantiated.append(obj)
        except Exception:
            # If a callback fails to instantiate, ignore to avoid interrupting training
        
            continue
    return instantiated
