from typing import Optional, Union
import warnings
import numpy as np
import torch
from fractions import Fraction
import os

try:
    import av

    av.logging.set_level(av.logging.ERROR)
    if not hasattr(av.video.frame.VideoFrame, "pict_type"):
        av = ImportError(
            """\
Your version of PyAV is too old for the necessary video operations in torchvision.
If you are on Python 3.5, you will have to build from source (the conda-forge
packages are not up-to-date).  See
https://github.com/mikeboers/PyAV#installation for instructions on how to
install PyAV on your system.
"""
        )
except ImportError:
    av = ImportError(
        """\
PyAV is not installed, and is necessary for the video operations in torchvision.
See https://github.com/mikeboers/PyAV#installation for instructions on how to
install PyAV on your system.
"""
    )
from torchvision.io.video import (
    _check_av_available,
    _read_from_stream,
)


def read_video(
    filename: str,
    start_pts: Union[float, Fraction] = 0,
    end_pts: Optional[Union[float, Fraction]] = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> torch.Tensor:
    """
    Adapted from torchvision.io.video.read_video
    Simplified to only read video frames (not audio and additional info)

    Args:
        filename (str): path to the video file
        start_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The start presentation time of the video
        end_pts (int if pts_unit = 'pts', float / Fraction if pts_unit = 'sec', optional):
            The end presentation time
        pts_unit (str, optional): unit in which start_pts and end_pts values will be interpreted,
            either 'pts' or 'sec'. Defaults to 'pts'.
        output_format (str, optional): The format of the output video tensors. Can be either "THWC" (default) or "TCHW".

    Returns:
        vframes (Tensor[T, H, W, C] or Tensor[T, C, H, W]): the `T` video frames
    """
    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(
            f"output_format should be either 'THWC' or 'TCHW', got {output_format}."
        )

    if not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")

    _check_av_available()

    if end_pts is None:
        end_pts = float("inf")

    if end_pts < start_pts:
        raise ValueError(
            f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}"
        )

    video_frames = []

    try:
        with av.open(filename, metadata_errors="ignore") as container:
            if container.streams.video:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    video_frames = _read_from_stream(
                        container,
                        start_pts,
                        end_pts,
                        pts_unit,
                        container.streams.video[0],
                        {"video": 0},
                    )

    except av.AVError:
        pass

    vframes_list = [frame.to_rgb().to_ndarray() for frame in video_frames]

    if vframes_list:
        vframes = torch.as_tensor(np.stack(vframes_list))
    else:
        vframes = torch.empty((0, 1, 1, 3), dtype=torch.uint8)

    if output_format == "TCHW":
        # [T,H,W,C] --> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    return vframes
