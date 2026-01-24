import argparse
import torch
import os
import socket
from torch.utils.data import DataLoader
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
from src.validator import Validator


def _distributed_cuda_preflight() -> None:
    rank = int(os.environ.get("RANK", "-1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))

    header = (
        f"[preflight host={socket.gethostname()} rank={rank} local_rank={local_rank} "
        f"world_size={world_size}]"
    )
    print(header)
    print(f"{header} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    if not torch.cuda.is_available():
        print(f"{header} torch.cuda.is_available()=False")
        return

    device_count = torch.cuda.device_count()
    print(f"{header} torch.cuda.device_count()={device_count}")

    for i in range(device_count):
        try:
            props = torch.cuda.get_device_properties(i)
            name = getattr(props, "name", "<unknown>")
            total_mem = getattr(props, "total_memory", None)
            total_mem_gb = (total_mem / (1024**3)) if isinstance(total_mem, (int, float)) else None
            uuid = getattr(props, "uuid", None)
            uuid_str = str(uuid) if uuid is not None else "<unavailable>"
            if total_mem_gb is not None:
                print(f"{header} cuda:{i} name={name} mem_gb={total_mem_gb:.1f} uuid={uuid_str}")
            else:
                print(f"{header} cuda:{i} name={name} uuid={uuid_str}")
        except Exception as exc:
            print(f"{header} cuda:{i} get_device_properties failed: {exc}")

    if local_rank >= 0:
        if local_rank >= device_count:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} but only {device_count} CUDA device(s) visible. "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}"
            )
        torch.cuda.set_device(local_rank)
        print(f"{header} torch.cuda.current_device()={torch.cuda.current_device()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the config file.')
    args = parser.parse_args()

    _distributed_cuda_preflight()

    # Load options from YAML
    opt = Options.from_yaml(args.config)
    
    # Initialize Accelerator with gradient accumulation
    accelerator = Accelerator(
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        mixed_precision=opt.mixed_precision if opt.mixed_precision != "no" else None
    )
    
    # Setup device, logging, and seeds
    device = accelerator.device
    opt.device = str(device)
    set_seed(42)
    
    # Only setup logging on main process to avoid duplicate logs/race conditions on file creation
    if accelerator.is_main_process:
        setup_logging(opt.log_dir)
    
    if opt.mode == 'train':
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

        model.to(device)

        if opt.checkpoint_path is not None:
            print("Loading checkpoint...")
            load_checkpoint(model, optimizer, opt.checkpoint_path, device)
        if opt.adm_checkpoint_path is not None:
            print("Loading pretrained UNet weights...")
            load_adm_checkpoint(model, opt)

        for param in model.parameters():
            param.data = param.data.contiguous()

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
            accelerator=accelerator
        )

        print("Starting training...")
        trainer.train(epochs=opt.epochs)
        print("Training complete.")
    
    elif opt.mode == 'validate':
        print("Mode: Validation")
        # Load validation dataset
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


        # get the trained model for validation
        model = get_model(opt)
        load_checkpoint(model, None, opt.checkpoint_path, device)
        trained_model = model
        trained_model.to(device)
        beta_schedule = get_beta_schedule(opt.noise_schedule, opt.timesteps)

        diffusion_process = DiffusionProcess(
            beta_schedule=beta_schedule,
        )

        validator = Validator(
            model=trained_model,
            diffusion=diffusion_process,
            data_loader=val_loader,
            config=opt,
            device=device
        )

        print("Starting validation...")
        validator.validate()
        print("Validation complete.")

if __name__ == "__main__":
    main()
