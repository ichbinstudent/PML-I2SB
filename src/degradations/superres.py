from typing import Literal
import torch.nn.functional as F
from src.options import Options


def build_superres(opt: Options, sr_filter: Literal["bicubic", "bilinear", "pool"] = "bicubic", image_size: int = 256):
    """
    Creates a function that performs 4x downsampling and then
    4x nearest-neighbor upsampling.
    """
    factor = 4

    if sr_filter == "bicubic":
        downsample_fn = lambda img: F.interpolate(
            img,
            scale_factor=1 / factor,
            mode="bicubic",
            antialias=True,
            align_corners=True,
        )
    elif sr_filter == "pool":
        downsample_fn = lambda img: F.avg_pool2d(img, kernel_size=factor, stride=factor)
    elif sr_filter == "bilinear":
        downsample_fn = lambda img: F.interpolate(
            img,
            scale_factor=1 / factor,
            mode="bilinear",
            antialias=True,
            align_corners=True,
        )
    else:
        raise ValueError("sr_filter must be 'bicubic', 'bilinear', or 'pool'")

    def superres_fn(img):
        img_down = downsample_fn(img)

        img_up = F.interpolate(img_down, scale_factor=factor, mode="nearest")
        return img_up, None

    return superres_fn
