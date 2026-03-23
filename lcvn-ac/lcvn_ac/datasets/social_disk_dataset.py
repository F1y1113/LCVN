import logging
from typing import Any

from lcvn_ac.datasets.episode_disk_dataset import EpisodeDiskDataset

logger = logging.getLogger(__name__)


class SocialDiskDataset(EpisodeDiskDataset):
    """
    Social dataset now reads preprocessed cache (cached_feats.pkl) and no longer
    depends on .pt metadata/latents.
    - Inject features via DataModule's load_feats
    - Build load_info using language indices (auto_lang_ann.npy)
    - Reuse continuous action and language processing logic
    """

    def __init__(
        self,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        use_cached_data: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            *args,
            skip_frames=skip_frames,
            save_format="npz",  # Force npz to avoid 'pt' in old configs causing errors
            pretrain=pretrain,
            use_cached_data=use_cached_data,
            **kwargs,
        )
        logger.info("Initialized SocialDiskDataset to use cached feats and language indexing")
