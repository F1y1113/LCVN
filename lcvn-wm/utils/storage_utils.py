
import torch
from pathlib import Path


def safe_torch_save(obj, path: Path):
    """
    Safely save a torch object to disk.
    If the path does not exist, create folders as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)
