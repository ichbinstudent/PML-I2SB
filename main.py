import argparse
import yaml
import torch
import os
from torch.utils.data import DataLoader, random_split

from src.dataset import (
    get_base_imagenet_dataset, 
    get_final_transform, 
    I2SBImageNetWrapper
)
from src.options import Options
from src.model import get_model, load_adm_checkpoint
from src.diffusion import DiffusionProcess
from src.trainer import Trainer
from src.utils import set_seed, setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the config file.')
    args = parser.parse_args()

    # Load options from YAML
    opt = Options.from_yaml(args.config)
    
    # Setup device, logging, and seeds
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.device = device
    set_seed(42)
    setup_logging(opt.log_dir)
    
    final_transform = get_final_transform()

    # Load train dataset
    train_base_dataset = get_base_imagenet_dataset(
        opt.train_data_dir, 
        opt.image_size,
        is_train=True
    )

    train_dataset = I2SBImageNetWrapper(
        base_dataset=train_base_dataset,
        opt=opt,
        final_transform=final_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    
    # Load validation dataset
    val_loader = None
    if opt.val_data_dir:
        val_base_dataset = get_base_imagenet_dataset(
            opt.val_data_dir,
            opt.image_size,
            is_train=False
        )
        
        val_dataset = I2SBImageNetWrapper(
            base_dataset=val_base_dataset,
            opt=opt,
            final_transform=final_transform
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True
        )

    model = get_model(opt)
    model.load_state_dict(torch.load(opt.adm_checkpoint_path))
    model.to(device)
    load_adm_checkpoint(model, opt)

    diffusion_process = DiffusionProcess(
        schedule_name=opt.noise_schedule,
        timesteps=opt.timesteps
    )

    trainer = Trainer(
        model=model,
        diffusion=diffusion_process,
        data_loader=train_loader,
        opt=opt,
        device=device
    )

    print("Starting training...")
    trainer.train(epochs=opt.epochs)
    print("Training complete.")

if __name__ == "__main__":
    main()
