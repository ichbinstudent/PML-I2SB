"""
Implements the main training loop for I2SB.
"""

import logging
import os
from typing import Optional, Tuple
import torch
from torch.optim import AdamW
from tqdm import tqdm

from src.diffusion import DiffusionProcess
from src.options import Options
import src.utils as utils

class Trainer:
    def __init__(self, model: torch.nn.Module, diffusion: DiffusionProcess, data_loader: torch.utils.data.DataLoader, config: Options, device: torch.device, optimizer, val_loader: Optional[torch.utils.data.DataLoader[Tuple[torch.Tensor, torch.Tensor]]]=None):
        self.model = model
        self.diffusion = diffusion
        self.data_loader = data_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        self.lr = config.learning_rate

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)

        self.global_step = 0
        self.log_dir = config.log_dir
        self.sample_dir = os.path.join(self.log_dir, 'samples')
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')

        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logging.info("Trainer initialized.")

    def train(self, epochs: int, start_epoch: int = 0):
        logging.info(f"Starting training for {epochs} epochs...")

        for epoch in range(start_epoch, epochs):
            self.model.train()
            
            pbar = tqdm(self.data_loader, 
                        desc=f"Epoch {epoch + 1}/{epochs}", 
                        leave=True)
            
            for X_0, X_1 in pbar:
                X_0 = X_0.to(self.device)
                X_1 = X_1.to(self.device)
                
                batch_size = X_0.shape[0]
                t = self.diffusion.sample_timesteps(batch_size).to(self.device)
                
                X_t = self.diffusion.sample_xt(X_0, X_1, t)
                model_output = self.model(X_t, t)
                loss = self.diffusion.calculate_loss(model_output, X_0, X_t, t)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

                pbar.set_postfix(loss=loss.item())
                self.global_step += 1

                if self.global_step % self.config.log_interval == 0:
                    logging.info(f"Step {self.global_step}, Loss: {loss.item():.4f}")
                    logging.info(f"Step {self.global_step}: Sampling...")
                    self.model.eval()
                    with torch.no_grad():
                        n_samples = min(8, batch_size)
                        X_1_sample = X_1[:n_samples]
                        X_0_sample = X_0[:n_samples]
                        
                        X_pred = self.diffusion.sample_ddpm(self.model, X_1_sample, self.diffusion.n_steps)
                        
                        sample_path = os.path.join(self.sample_dir, f"sample_{self.global_step}.png")
                        utils.save_image_grid(X_1_sample, X_pred, X_0_sample, sample_path)
                    self.model.train()

            # Run validation at the end of each epoch
            if self.val_loader is not None:
                val_loss = self.validate()
                logging.info(f"Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}")

            utils.save_checkpoint(self.model, self.optimizer, epoch, self.checkpoint_dir)

        logging.info("Training complete.")

    @torch.no_grad()
    def validate(self):
        """Run validation and return average loss."""
        assert self.val_loader is not None, "val_loader must be set to run validation"
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for X_0, X_1 in tqdm(self.val_loader, desc="Validating", leave=False):
            X_0: torch.Tensor = X_0.to(self.device)
            X_1: torch.Tensor = X_1.to(self.device)

            batch_size = X_0.shape[0]
            t = self.diffusion.sample_timesteps(batch_size).to(self.device)

            X_t = self.diffusion.sample_xt(X_0, X_1, t)
            model_output = self.model(X_t, t)
            loss = self.diffusion.calculate_loss(model_output, X_0, X_t, t)

            total_loss += loss.item()
            num_batches += 1

        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
