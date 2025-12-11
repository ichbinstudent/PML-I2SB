import numpy as np
import io
import logging
import torch
from typing import Literal
from src.options import Options


def bbox2mask(img_shape: tuple[int, int], bbox: tuple[int, int, int, int], dtype='uint8') -> np.ndarray:
    """
    Generate mask in ndarray from bbox.
    """

    height, width = img_shape[:2]

    mask = np.zeros((height, width, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[2], bbox[1]:bbox[1] + bbox[3], :] = 1

    return mask


def load_freeform_masks() -> dict[str, np.ndarray]:
    """Loads freeform masks from a compressed npz file.

    Returns:
      A dictionary containing freeform masks categorized by their coverage percentages.
    """

    logging.info("Loading freeform masks...")

    filename = "src/degradations/imagenet_freeform_masks.npz"
    shape = [10000, 256, 256]

    # shape = [10950, 256, 256] # Uncomment this for places2.

    # Load the npz file.
    with open(filename, "rb") as f:
        data = f.read()

    data = dict(np.load(io.BytesIO(data)))

    # Unpack and reshape the masks.
    for key in data:
        data[key] = (
            np.unpackbits(data[key], axis=None)[: np.prod(shape)]
            .reshape(shape)
            .astype(np.uint8)
        )

    # data[key] contains [10000, 256, 256] array i.e. 10000 256x256 masks.
    logging.info("Freeform masks loaded successfully.")
    return data

_freeform_masks = load_freeform_masks()  # Preload masks at module import.

def get_center_mask(image_size: tuple[int, int]) -> torch.Tensor:
    h, w = image_size
    mask = bbox2mask(image_size, (h//4, w//4, h//2, w//2))
    return torch.from_numpy(mask).permute(2,0,1)

def build_inpaint_center(opt: Options):
    center_mask = get_center_mask((opt.image_size, opt.image_size))[None,...] # [1,1,256,256]
    center_mask = center_mask.to(opt.device)

    def inpaint_center(img):
        # img: [-1,1]
        mask = center_mask
        return img * (1. - mask) + mask, mask

    return inpaint_center

def build_inpaint_freeform(opt: Options, mask_type: Literal['10-20% freeform', '20-30% freeform', '30-40% freeform']):
    assert mask_type in ['10-20% freeform', '20-30% freeform', '30-40% freeform']

    freeform_masks = _freeform_masks[mask_type]
    freeform_masks = torch.from_numpy(freeform_masks).to(opt.device)

    def inpaint_freeform(img):
        index = np.random.randint(freeform_masks.shape[0], size=img.shape[0])
        mask = freeform_masks[index]
        return img * (1. - mask) + mask, mask

    return inpaint_freeform

