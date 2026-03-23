from datasets.video import (
    SocialAdvancedVideoDataset,
)
from algorithms.ldit import DFoTVideo, DFoTVideoPose
from algorithms.ldit.ldit_video_social import LDiTVideoSocial
from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls


class VideoGenerationExperiment(BaseLightningExperiment):
    """
    A video generation experiment
    """

    compatible_algorithms = dict(
        dfot_video=DFoTVideo,
        dfot_video_pose=DFoTVideoPose,
        sd_video=DFoTVideo,
        sd_video_3d=DFoTVideoPose,
        ldit_video_social=LDiTVideoSocial,
    )

    compatible_datasets = dict(
        social=SocialAdvancedVideoDataset,
        lcvn=SocialAdvancedVideoDataset,
    )

    data_module_cls = _data_module_cls
