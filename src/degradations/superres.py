import torch
import torch.nn.functional as F
from src.options import Options


def build_superres(opt: Options, sr_filter="bicubic", image_size=256):
    """
    Creates a function that performs 4x downsampling and then
    4x nearest-neighbor upsampling.
    """
    factor = 4

    def _downsample_bicubic(img: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            img,
            scale_factor=1 / factor,
            mode="bicubic",
            antialias=True,
            align_corners=True,
        )

    def _downsample_pool(img: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(img, kernel_size=factor, stride=factor)

    def _downsample_bilinear(img: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            img,
            scale_factor=1 / factor,
            mode="bilinear",
            antialias=True,
            align_corners=True,
        )

    downsample_fns = {
        "bicubic": _downsample_bicubic,
        "pool": _downsample_pool,
        "bilinear": _downsample_bilinear,
    }

    if sr_filter not in downsample_fns:
        raise ValueError("sr_filter must be 'bicubic', 'bilinear', or 'pool'")

    downsample_fn = downsample_fns[sr_filter]

    def superres_fn(img: torch.Tensor) -> torch.Tensor:
        img_down = downsample_fn(img)

        img_up = F.interpolate(img_down, scale_factor=factor, mode="nearest")
        return img_up

    return superres_fn
