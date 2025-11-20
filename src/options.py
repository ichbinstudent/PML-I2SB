import yaml
from typing import Any


class Options:
    """Options class for I2SB configuration."""
    
    # Default values
    image_size: int = 256
    device: str = 'cuda'
    degradation: str = 'jpeg-10'
    
    # Training params
    batch_size: int = 32
    num_workers: int = 8
    learning_rate: float = 1e-4
    epochs: int = 100
    
    # Data paths
    train_data_dir: str = './imagenet/train'
    val_data_dir: str = None
    
    # Diffusion params
    noise_schedule: str = 'symmetric'
    timesteps: int = 1000
    
    # Logging
    log_dir: str = 'logs'
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Options':
        """Load options from a YAML file."""
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_dict(config)
    
    @classmethod
    def from_dict(cls, config: dict) -> 'Options':
        """Load options from a dictionary."""
        opt = cls()
        
        # Update attributes from config
        for key, value in config.items():
            if hasattr(opt, key):
                setattr(opt, key, value)
            else:
                # Store unknown config keys as attributes anyway
                setattr(opt, key, value)
        
        return opt
    
    def update(self, **kwargs):
        """Update options with keyword arguments."""
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
