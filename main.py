import argparse
import yaml
import torch
import os
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from accelerate import Accelerator

from src.dataset import (
    get_base_imagenet_dataset,
    I2SBImageNetWrapper
)
from src.options import Options
from src.model import get_model, load_adm_checkpoint
from src.diffusion import DiffusionProcess
from src.trainer import Trainer
from src.utils import set_seed, setup_logging, get_beta_schedule, load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the config file.')
    args = parser.parse_args()

    # Load options from YAML
    opt = Options.from_yaml(args.config)
    
    # Initialize Accelerator for multi-GPU training
    accelerator = Accelerator()
    device = accelerator.device
    opt.device = str(device)
    set_seed(42)
    setup_logging(opt.log_dir)
    
    # Load train dataset
    train_base_dataset = get_base_imagenet_dataset(
        opt.train_data_dir, 
        opt.image_size,
        is_train=True
    )

    train_dataset = I2SBImageNetWrapper(
        base_dataset=train_base_dataset,
        opt=opt
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if opt.val_data_dir:
        val_base_dataset = get_base_imagenet_dataset(
            opt.val_data_dir,
            opt.image_size,
            is_train=False
        )
        
        val_dataset = I2SBImageNetWrapper(
            base_dataset=val_base_dataset,
            opt=opt
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=torch.cuda.is_available()
        )

    model = get_model(opt)
    optimizer = AdamW(model.parameters(), lr=opt.learning_rate)
    start_epoch = 0

    if opt.checkpoint_path is not None:
        print("Loading checkpoint...")
        start_epoch = load_checkpoint(model, optimizer, opt.checkpoint_path, device)
    if opt.adm_checkpoint_path is not None:
        print("Loading pretrained UNet weights...")
        load_adm_checkpoint(model, opt)

    # Prepare model, optimizer, and dataloaders with Accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    if val_loader is not None:
        val_loader = accelerator.prepare(val_loader)

    beta_schedule = get_beta_schedule(opt.noise_schedule, opt.timesteps)

    diffusion_process = DiffusionProcess(
        beta_schedule=beta_schedule,
    )

    trainer = Trainer(
        model=model,
        diffusion=diffusion_process,
        data_loader=train_loader,
        device=device,
        config=opt,
        optimizer=optimizer,
        val_loader=val_loader,
        accelerator=accelerator
    )

    print("Starting training...")
    trainer.train(epochs=opt.epochs, start_epoch=start_epoch)
    print("Training complete.")

if __name__ == "__main__":
    main()
