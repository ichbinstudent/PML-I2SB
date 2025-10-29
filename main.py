import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    get_base_imagenet_dataset, 
    get_final_transform, 
    I2SBImageNetWrapper
)
from src.model import get_model, load_adm_checkpoint
from src.diffusion import DiffusionProcess
from src.trainer import Trainer
from src.utils import set_seed, setup_logging


def main():
    # --- 1. Configuration Step ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the config file.')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup device, logging, and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    setup_logging(config.get('log_dir', 'logs'))
    
    # --- 2. Setup "Everything Else" (from other files) ---
    
    # Setup Dataset (from src/dataset.py)
    final_transform = get_final_transform()

    base_dataset = get_base_imagenet_dataset(
        config['data_dir'], 
        config['image_size']
    )

    train_dataset = I2SBImageNetWrapper(
        base_dataset=base_dataset,
        task_name=config['task'],
        task_config=config['task_params'],
        final_transform=final_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=8
    )

    # Initialize with ADM checkpoint as paper suggests [cite: 249]
    model = get_model(config)
    model.load_state_dict(torch.load(config['adm_checkpoint_path']))
    model.to(device)
    load_adm_checkpoint(model, config)

    diffusion_process = DiffusionProcess(
        schedule_name=config['diffusion_params']['noise_schedule'],
        timesteps=config['diffusion_params']['timesteps']
    )

    trainer = Trainer(
        model=model,
        diffusion=diffusion_process,
        data_loader=train_loader,
        config=config,
        device=device
    )

    # --- 3. Run ---
    print("Starting training...")
    trainer.train(epochs=config['epochs'])
    print("Training complete.")

if __name__ == "__main__":
    main()
