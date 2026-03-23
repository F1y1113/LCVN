from datasets.video import (
    SocialAdvancedVideoDataset,
)
from algorithms.vae import ImageVAETrainer, VideoVAETrainer
from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls


class VideoLatentLearningExperiment(BaseLightningExperiment):
    """
    An experiment for training & validating the first stage model (e.g. VAE)
    that learns the latent representation of the data
    """

    compatible_algorithms = dict(
        image_vae=ImageVAETrainer,
        video_vae=VideoVAETrainer,
    )

    compatible_datasets = dict(
        social=SocialAdvancedVideoDataset,
        lcvn=SocialAdvancedVideoDataset,
    )

    data_module_cls = _data_module_cls
