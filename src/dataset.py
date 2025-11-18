from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from src.degradations import build_degradations
from src.options import Options


def get_final_transform():
    """
    Returns the final transformation (PIL -> Tensor) to be applied to
    clean images before degradation.
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])


def get_base_imagenet_dataset(data_dir, image_size):
    """
    Creates the base ImageNet dataset from a folder path.

    Args:
        data_dir (str): Path to the ImageNet 'train' or 'val' directory.
        image_size (int): The target size to resize and crop images to.

    Returns:
        torchvision.datasets.ImageFolder: The base dataset.
    """
    
    # These transforms are applied *before* degradation
    # They output a PIL Image
    pre_transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
    ])
    
    print(f"Loading base dataset from: {data_dir}")
    base_dataset = ImageFolder(
        root=data_dir,
        transform=pre_transform
    )
    
    return base_dataset


class I2SBImageNetWrapper(Dataset):
    """
    A wrapper dataset that takes a base ImageNet dataset (or any image dataset)
    and returns a pair of (X_0, X_1) images for I2SB training.
    
    X_0 is the clean image.
    X_1 is the degraded image.
    """
    def __init__(self, base_dataset, opt: Options, final_transform):
        """
        Args:
            base_dataset: The pre-initialized ImageFolder dataset.
            opt: Options object containing degradation configuration.
            final_transform (callable): The final transform (PIL -> Tensor).
        """
        self.base_dataset = base_dataset
        self.opt = opt
        self.final_transform = final_transform
        
        # Build the degradation function based on opt.degradation
        self.degradation_fn = build_degradations(opt, opt.degradation)
        print(f"Built degradation: {opt.degradation}")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 1. Load the clean image (X_0)
        # base_dataset[idx] returns (image_pil, label)
        X_0_pil, _ = self.base_dataset[idx]

        # 2. Convert to tensor in [-1, 1] range
        X_0 = self.final_transform(X_0_pil)
        
        # 3. Add batch dimension for degradation function
        X_0_batch = X_0.unsqueeze(0).to(self.opt.device)
        
        # 4. Apply degradation
        # Inpainting returns (degraded, mask), others return just degraded
        degradation_output = self.degradation_fn(X_0_batch)
        
        if isinstance(degradation_output, tuple):
            # Inpainting case: (degraded_img, mask)
            X_1_batch = degradation_output[0]
        else:
            # Other degradations
            X_1_batch = degradation_output
        
        # 5. Remove batch dimension
        X_1 = X_1_batch.squeeze(0)
        
        # 6. Return the pair (both in [-1, 1] range, X_1 on device, X_0 on CPU)
        return X_0, X_1
