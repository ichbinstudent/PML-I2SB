import yaml
from typing import Literal, Optional


class Options:
    # Default values
    image_size: int = 256
    stats_image_size: Optional [int] = None
    device: str = "cuda"
    degradation: str = "jpeg-10"

    # Training params
    batch_size: int = 32
    num_workers: int = 8
    learning_rate: float = 1e-4
    epochs: int = 100

    mode : str = "train"  # "train" or "validate"

    # Data paths
    train_data_dir: str = "./imagenet/train"
    val_data_dir: Optional[str] = None
    adm_checkpoint_path: Optional[str] = None # "./checkpoints/256x256_diffusion_uncond.pt"
    checkpoint_path: Optional[str] = None # "./logs/checkpoints/latest_checkpoint.pt
    imagenet_stats_path: Optional[str] = None # Path to ImageNet statistics for FID calculation
    image_names_file: Optional[str] = None  # Path to text file with allowed image names for validation

    dataset_fraction: float = 0.01


    # Diffusion params
    timesteps: int = 1000
    noise_schedule: Literal["linear", "cosine", "quadratic", "const"] = "linear"

    # Optimization
    mixed_precision: str = "no" # "no", "fp16", "bf16"
    gradient_accumulation_steps: int = 1
    use_checkpoint: bool = False # Gradient checkpointing
    use_compile: bool = False # torch.compile
    dist_backend: str = "gloo" # "nccl" or "gloo"

    # EMA
    use_ema: bool = True
    ema_decay: float = 0.999

    # Logging
    log_dir: str = "logs"
    log_interval: int = 100

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Options":
        """Load options from a YAML file."""
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> "Options":
        """Load options from a dictionary."""
        opt = cls()
        for key, value in config.items():
            if hasattr(opt, key):
                setattr(opt, key, value)

        return opt

    def update(self, **kwargs):
        """Update options with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
