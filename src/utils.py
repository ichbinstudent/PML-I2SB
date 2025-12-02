import os
import sys
import logging
import random
import numpy as np
import torch
import torchvision.utils as vutils


def setup_logging(log_dir):
    """
    Configures the logging module to output to both console and a file.
    
    Args:
        log_dir (str): The directory where the log file will be saved.
    """
    # Create the log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = os.path.join(log_dir, 'train.log')

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add formatter to handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logging.info("Logging setup complete. Logs will be saved to %s", log_filename)

def set_seed(seed, deterministic=True):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): The seed value.
        deterministic (bool): Whether to use deterministic cuDNN settings.
                              (Can slow down training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    logging.info(f"Set random seed to {seed}")

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, is_best=False):
    """
    Saves a model checkpoint.
    
    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int): The current epoch number.
        checkpoint_dir (str): Directory to save the checkpoint.
        is_best (bool): Whether this is the best performing model so far.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    # Save the latest checkpoint
    latest_filepath = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(state, latest_filepath)
    logging.info(f"Saved latest checkpoint to {latest_filepath}")

    # Save a separate file for the best model
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(state, best_filepath)
        logging.info(f"Saved best model checkpoint to {best_filepath}")

def load_checkpoint(model, optimizer, filepath, device):
    """
    Loads a model checkpoint.
    
    Args:
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer): The optimizer to load state into.
        filepath (str): Path to the checkpoint file.
        device (torch.device): The device to map the loaded tensors to.

    Returns:
        int: The epoch to start training from (epoch + 1).
    """
    if not os.path.exists(filepath):
        logging.warning(f"Checkpoint file not found: {filepath}. Starting from scratch.")
        return 0 # Start from epoch 0

    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    
    logging.info(f"Loaded checkpoint from {filepath}. Resuming from epoch {start_epoch}")
    
    return start_epoch

def unnormalize_to_zero_one(tensor):
    """
    Un-normalizes a tensor from [-1, 1] to [0, 1].
    """
    return (tensor + 1.0) * 0.5

def save_image_grid(X_1, X_pred, X_0, filepath, n_images=8):
    """
    Saves a grid of images: [Degraded, Predicted, Clean].
    
    Args:
        X_1 (torch.Tensor): The batch of degraded images (Input).
        X_pred (torch.Tensor): The batch of model predictions (Output).
        X_0 (torch.Tensor): The batch of clean images (Reference).
        filepath (str): Path to save the PNG image.
        n_images (int): Number of images from the batch to save.
    """
    
    # Take the first n_images from the batch
    X_1 = X_1.cpu()[:n_images]
    X_pred = X_pred.cpu().detach()[:n_images]
    X_0 = X_0.cpu()[:n_images]
    
    # Un-normalize from [-1, 1] to [0, 1]
    X_1 = unnormalize_to_zero_one(X_1)
    X_pred = unnormalize_to_zero_one(X_pred)
    X_0 = unnormalize_to_zero_one(X_0)
    
    # Concatenate along the batch dimension
    # The grid will be [X_1_row, X_pred_row, X_0_row]
    grid = torch.cat([X_1, X_pred, X_0], dim=0)
    
    # Save the image
    # nrow=n_images will create 3 rows (Input, Output, Reference)
    vutils.save_image(grid, filepath, nrow=n_images, padding=2, pad_value=1.0)
    
    logging.debug(f"Saved image sample grid to {filepath}")

def get_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return torch.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=torch.float64)
    elif schedule_name == "const" or schedule_name == "symmetric":
        return torch.ones(num_diffusion_timesteps, dtype=torch.float64) * 1.0
    else:
        raise NotImplementedError(f"Unknown beta schedule: {schedule_name}")

