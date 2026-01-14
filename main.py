import os

# Fix for virtualized GPUs (e.g. MIG, containerized) where PCI IDs might be shared/confused
# These must be set BEFORE importing torch
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_NVLS_ENABLE"] = "0"
os.environ["NCCL_SHM_DISABLE"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator, DistributedDataParallelKwargs

from src.dataset import (
    get_base_imagenet_dataset,
    get_base_imagenet_val_dataset,
    I2SBImageNetWrapper
)
from src.options import Options
from src.model import get_model, load_adm_checkpoint
from src.diffusion import DiffusionProcess
from src.trainer import Trainer
from src.utils import set_seed, setup_logging, get_beta_schedule, load_checkpoint, load_checkpoint_superres
from src.validator import Validator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the config file.')
    args = parser.parse_args()

    # Load options from YAML
    opt = Options.from_yaml(args.config)
    
    # Setup accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs]
    )

    device = accelerator.device
    opt.device = str(device)
    
    # Setup logging and seeds
    set_seed(42)
    if accelerator.is_main_process:
        setup_logging(opt.log_dir)
    
    if opt.mode == 'train':
        if accelerator.is_main_process:
            print("Mode: Training")
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

        model = get_model(opt)
        optimizer = AdamW(model.parameters(), lr=opt.learning_rate)

        # model.to(device) # Handled by accelerate

        if opt.checkpoint_path is not None:
            if accelerator.is_main_process:
                print("Loading checkpoint...")
            if opt.degradation == 'superres-bicubic':
                print("here")
                load_checkpoint_superres(model, optimizer, opt.checkpoint_path, device)
            else:
                load_checkpoint(model, optimizer, opt.checkpoint_path, device)
        if opt.adm_checkpoint_path is not None:
            if accelerator.is_main_process:
                print("Loading pretrained UNet weights...")
            load_adm_checkpoint(model, opt)

        # Ensure model parameters are contiguous to avoid DDP stride warnings
        for param in model.parameters():
            param.data = param.data.contiguous()

        beta_schedule = get_beta_schedule(opt.noise_schedule, opt.timesteps)
        beta_schedule = torch.concatenate([beta_schedule[:opt.timesteps//2], torch.flip(beta_schedule[:opt.timesteps//2], dims=[0])], dim=0)

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
            accelerator=accelerator
        )

        if accelerator.is_main_process:
            print("Starting training...")
        trainer.train(epochs=opt.epochs)
        if accelerator.is_main_process:
            print("Training complete.")
    
    elif opt.mode == 'validate':
        if accelerator.is_main_process:
            print("Mode: Validation")

        print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("device_count=", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i))
        
        # Load validation dataset
        val_base_dataset = None

        val_base_dataset = get_base_imagenet_val_dataset(
            opt.val_data_dir,
            opt.image_size,
            opt.image_names_file,
            opt.degradation,
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
        val_loader = accelerator.prepare(val_loader)

        # get the trained model for validation
        trained_model = get_model(opt)
        if accelerator.is_main_process:
            print("Loading trained model checkpoint for validation...")

        if opt.adm_checkpoint_path is not None:
            if accelerator.is_main_process:
                print("Loading pretrained UNet weights...")
            load_adm_checkpoint(trained_model, opt)

        if opt.checkpoint_path is not None:
            if accelerator.is_main_process:
                print("Loading checkpoint...")
            if opt.degradation == 'superres-bicubic':
                print("here")
                load_checkpoint_superres(trained_model, None, opt.checkpoint_path, device)
            else:
                load_checkpoint(model, None, opt.checkpoint_path, device)

        beta_schedule = get_beta_schedule(opt.noise_schedule, opt.timesteps)

        diffusion_process = DiffusionProcess(
            beta_schedule=beta_schedule,
        )

        validator = Validator(
            model=trained_model,
            diffusion=diffusion_process,
            data_loader=val_loader,
            opt=opt,
            accelerator=accelerator
        )

        print("Starting validation...")
        validator.validate()
        print("Validation complete.")

if __name__ == "__main__":
    main()