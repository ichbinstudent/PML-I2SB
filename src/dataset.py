from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from src.degradations import build_degradations
from src.options import Options
from PIL import Image
import os
from pathlib import Path


class SimpleImageDataset(Dataset):
    """
    A simple dataset that loads images from a flat directory structure.
    Supports common image formats: .jpg, .jpeg, .png, .JPEG, .JPG, .PNG
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        # Get all image files
        valid_extensions = {".jpg", ".jpeg", ".png", ".JPEG", ".JPG", ".PNG"}
        self.image_paths = []

        for file in os.listdir(root):
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path) and Path(file).suffix in valid_extensions:
                self.image_paths.append(file_path)

        self.image_paths.sort()
        print(f"Found {len(self.image_paths)} images in {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, 0  # Return 0 as dummy label for compatibility


def get_base_imagenet_dataset(data_dir, image_size, is_train=True):
    """
    Creates the base ImageNet dataset from a folder path.
    Tries ImageFolder first (for standard ImageNet structure),
    falls back to SimpleImageDataset for flat directories.

    Args:
        data_dir (str): Path to the ImageNet 'train' or 'val' directory.
        image_size (int): The target size to resize and crop images to.

    Returns:
        Dataset: The base dataset.
    """

    transformations = [
        T.Resize(image_size),
        T.CenterCrop(image_size),
    ]

    if is_train:
        transformations.append(T.RandomHorizontalFlip())

    transformations.append(T.ToTensor())
    transformations.append(T.Lambda(lambda x: x * 2.0 - 1.0))  # Scale to [-1, 1]

    pre_transform = T.Compose(transformations)

    print(f"Loading base dataset from: {data_dir}")

    # Try ImageFolder first (ImageNet structure with a folder per class)
    try:
        base_dataset = ImageFolder(root=data_dir, transform=pre_transform)
        print(f"Using ImageFolder with {len(base_dataset.classes)} classes")
    except (FileNotFoundError, RuntimeError):
        # Fall back to flat directory structure
        print("ImageFolder failed, using flat directory structure")
        base_dataset = SimpleImageDataset(root=data_dir, transform=pre_transform)

    return base_dataset


class I2SBImageNetWrapper(Dataset):
    """
    A wrapper dataset that takes a base ImageNet dataset (or any image dataset)
    and returns a pair of (X_0, X_1) images for I2SB training.

    X_0 is the clean image.
    X_1 is the degraded image.
    """

    def __init__(self, base_dataset, opt: Options):
        """
        Args:
            base_dataset: The pre-initialized ImageFolder dataset.
            opt: Options object containing degradation configuration.
        """
        self.base_dataset = base_dataset
        self.opt = opt

        # Build the degradation function based on opt.degradation
        self.degradation_fn = build_degradations(opt, opt.degradation)
        print(f"Built degradation: {opt.degradation}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        X_0, label = self.base_dataset[idx]

        X_0_batch = X_0.unsqueeze(0)
        degradation_output = self.degradation_fn(X_0_batch)

        if isinstance(degradation_output, tuple):
            X_1_batch = degradation_output[0]
        else:
            X_1_batch = degradation_output

        X_1 = X_1_batch.squeeze(0)

        return X_0, X_1, label
