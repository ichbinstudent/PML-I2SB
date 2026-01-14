import torch
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Union
from src.options import Options

def build_blur(opt: Options, kernel_type: Union["uni", "gauss"]):
    assert kernel_type in ["uni", "gauss"]

    gaussian_blur = T.GaussianBlur(kernel_size=(5, 5), sigma=(10.0, 10.0))
    mean_kernel = torch.ones((3, 1, 3, 3), device=opt.device) / 9

    def blur(img):
        # img tensor assumed shape [B, C, H, W], value range [-1, 1]
        img = (img + 1) / 2  # scale to [0, 1]

        if kernel_type == "uni":
            img = F.conv2d(img, mean_kernel, padding=1, groups=3)
        elif kernel_type == "gauss":
            img = gaussian_blur(img)

        img = img * 2 - 1  # scale back to [-1, 1]
        return img, None

    return blur
