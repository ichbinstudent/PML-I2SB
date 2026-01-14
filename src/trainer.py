"""
Implements the main training loop for I2SB.
"""

import logging
import os
import torch
from torch.optim import AdamW
from tqdm import tqdm

from src.diffusion import DiffusionProcess
from src.options import Options
import src.utils as utils

class Trainer:
    def __init__(self, model: torch.nn.Module, diffusion: DiffusionProcess, data_loader: torch.utils.data.DataLoader, config: Options, device: torch.device, optimizer, accelerator):
        self.model = model
        self.diffusion = diffusion
        self.data_loader = data_loader
        self.config = config
        self.device = device
        self.accelerator = accelerator

        self.lr = config.learning_rate

        self.optimizer = optimizer # Optimizer is passed from main already

        self.global_step = 0
        self.log_dir = config.log_dir
        self.sample_dir = os.path.join(self.log_dir, 'samples')
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')

        if self.accelerator.is_main_process:
            os.makedirs(self.sample_dir, exist_ok=True)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Prepare everything with accelerator
        self.model, self.optimizer, self.data_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.data_loader
        )
        
        logging.info("Trainer initialized.")

    def train(self, epochs):
        if self.accelerator.is_main_process:
            logging.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            self.model.train()
            
            if self.accelerator.is_main_process:
                pbar = tqdm(self.data_loader, 
                            desc=f"Epoch {epoch + 1}/{epochs}", 
                            leave=True)
            else:
                pbar = self.data_loader
            
            for X_0, X_1, _, mask in pbar:
                # X_0 and X_1 are already on device thanks to prepare()
                
                with self.accelerator.accumulate(self.model):
                    batch_size = X_0.shape[0]
                    t = self.diffusion.sample_timesteps(batch_size).to(self.device)
                    
                    X_t = self.diffusion.sample_xt(X_0, X_1, t)
                    model_output = self.model(X_t, t)

                    # Check if mask is a real spatial mask (for inpainting) or dummy scalar (for other tasks)
                    # Real masks have more than 1 element per sample
                    if mask.numel() > batch_size:
                        mask = mask.unsqueeze(1)  # [B, 1, H, W] for broadcasting over channels
                        model_output = model_output * mask
                        X_0 = X_0 * mask

                    loss = self.diffusion.calculate_loss(model_output, X_0, X_t, t)

                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                

                if self.accelerator.is_main_process:
                    pbar.set_postfix(loss=loss.item())
                
                self.global_step += 1

                if self.global_step % self.config.log_interval == 0:
                    if self.accelerator.is_main_process:
                        logging.info(f"Step {self.global_step}, Loss: {loss.item():.4f}")
                        logging.info(f"Step {self.global_step}: Sampling...")
                    
                    # Wait for all processes before sampling/logging to avoid conflicts? 
                    # Actually sampling is usually done on main process only
                    self.accelerator.wait_for_everyone()
                    
                    if self.accelerator.is_main_process:
                        self.model.eval()
                        with torch.no_grad():
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            
                            n_samples = min(8, batch_size)
                            X_1_sample = X_1[:n_samples]
                            X_0_sample = X_0[:n_samples]
                            
                            # Use unwrapped model for sampling
                            X_pred = self.diffusion.sample_ddpm(unwrapped_model, X_1_sample, self.diffusion.n_steps)
                            
                            sample_path = os.path.join(self.sample_dir, f"sample_{self.global_step}.png")
                            utils.save_image_grid(X_1_sample, X_pred, X_0_sample, sample_path)
                        self.model.train()

            # Save checkpoint
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                # We need to save the unwrapped model
                keep_checkpoint = (epoch + 1) % 10 == 0
                utils.save_checkpoint(unwrapped_model, self.optimizer, epoch, self.checkpoint_dir, keep_checkpoint=keep_checkpoint)

        if self.accelerator.is_main_process:
            logging.info("Training complete.")
