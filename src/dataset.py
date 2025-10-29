from torch.utils.data import Dataset
import src.degradation as D
import torchvision.transforms as T
from torchvision.datasets import ImageFolder


def get_final_transform():
    """
    Returns the final transformation (PIL -> Tensor) to be applied to
    both X_0 (clean) and X_1 (degraded) images.
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
    def __init__(self, base_dataset, task_name, task_config, final_transform):
        """
        Args:
            base_dataset: The pre-initialized ImageFolder dataset.
            task_name (str): The name of the restoration task 
                             (e.g., 'jpeg', 'deblur', 'inpaint_freeform').
            task_config (dict): A dictionary of parameters for the task.
            final_transform (callable): The final transform (PIL -> Tensor).
        """
        self.base_dataset = base_dataset
        self.task_name = task_name
        self.task_config = task_config
        self.final_transform = final_transform
        
        # Pre-load mask paths for inpainting tasks
        if 'inpaint' in self.task_name:
            mask_dir = self.task_config.get('mask_dir')
            if not mask_dir:
                raise ValueError("Inpainting task requires 'mask_dir' in task_config")
            
            self.mask_paths = D.load_mask_paths(mask_dir)
            print(f"Loaded {len(self.mask_paths)} masks for inpainting.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # 1. Load the clean image (X_0)
        # base_dataset[idx] returns (image_pil, label)
        X_0_pil, _ = self.base_dataset[idx]

        # 2. Create the degraded image (X_1) based on the task
        if self.task_name == 'jpeg':
            quality = self.task_config.get('quality', 10)
            X_1_pil = D.apply_jpeg(X_0_pil, quality=quality)
        
        elif self.task_name == 'deblur':
            kernel_type = self.task_config.get('kernel', 'gaussian')
            X_1_pil = D.apply_blur(X_0_pil, kernel_type=kernel_type)
            
        elif self.task_name == 'inpaint_freeform':
            mask_pil = D.load_random_mask(self.mask_paths)
            # Per the paper, fill masked regions with Gaussian noise
            X_1_pil = D.apply_inpainting_mask(X_0_pil, mask_pil) 

        elif self.task_name == 'super_resolution':
            scale = self.task_config.get('scale', 4)
            X_1_pil = D.apply_super_resolution_degradation(X_0_pil, scale=scale)
        
        else:
            raise ValueError(f"Unknown task name: {self.task_name}")

        # 3. Apply final transforms to both images
        X_0 = self.final_transform(X_0_pil)
        X_1 = self.final_transform(X_1_pil)
        
        # 4. Return the pair
        return X_0, X_1
