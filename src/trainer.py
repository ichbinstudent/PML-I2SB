"""
Implements the main training loop for I2SB.
"""

import logging
import os
import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.optim import AdamW
from tqdm import tqdm

from src.diffusion import DiffusionProcess
import src.utils as utils

class Trainer:
    def __init__(self, model: torch.nn.Module, diffusion: DiffusionProcess, data_loader: torch.utils.data.DataLoader, config, device: str):
        """
        Initializes the Trainer class.

        Args:
            model (torch.nn.Module): The U-Net model.
            diffusion (DiffusionProcess): The diffusion process helper.
            data_loader (DataLoader): The training data loader.
            config (dict): The configuration dictionary.
            device (torch.device): The device to train on.
        """
        self.model = model
        self.diffusion = diffusion
        self.data_loader = data_loader
        self.config = config
        self.device = device

        self.lr = config.get('learning_rate', 1.0e-4)
        self.use_fp16 = config.get('model_params', {}).get('use_fp16', False)
        
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.scaler = GradScaler(device=self.device, enabled=self.use_fp16)

        self.global_step = 0
        self.log_dir = config.get('log_dir', 'logs')
        self.sample_dir = os.path.join(self.log_dir, 'samples')
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')

        # Create output directories
        os.makedirs(self.sample_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logging.info("Trainer initialized.")

    def train(self, epochs):
        """
        The main training loop, iterating over epochs and steps.
        
        Args:
            epochs (int): The total number of epochs to train for.
        """
        logging.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            self.model.train()
            
            pbar = tqdm(self.data_loader, 
                        desc=f"Epoch {epoch + 1}/{epochs}", 
                        leave=True)
            
            for X_0, X_1 in pbar:
                # --- Algorithm 1 ---
                X_0 = X_0.to(self.device)
                X_1 = X_1.to(self.device)
                
                # 1. Sample t (Algorithm 1, line 3) 
                batch_size = X_0.shape[0]
                t = self.diffusion.sample_timesteps(batch_size).to(self.device)
                
                with autocast(device_type=self.device,enabled=self.use_fp16):
                    # 2. Sample X_t and get the loss target (Algorithm 1, line 4) 
                    # This step uses Proposition 3.3 for X_t [cite: 188]
                    # and Equation 12 for the target [cite: 201]
                    X_t, target = self.diffusion.sample_xt(X_0, X_1, t)
                    
                    # 3. Predict epsilon from X_t (Algorithm 1, line 5) 
                    model_output = self.model(X_t, t)
                    
                    # 4. Calculate loss (Algorithm 1, line 5) 
                    loss = self.diffusion.calculate_loss(model_output, target)

                # 5. Gradient descent step (Algorithm 1, line 5) 
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # --- End Algorithm 1 ---

                pbar.set_postfix(loss=loss.item())
                self.global_step += 1

                # Logging and sampling
                if self.global_step % self.config.get('log_interval', 100) == 0:
                    logging.info(f"Step {self.global_step}, Loss: {loss.item():.4f}")

                if self.global_step % self.config.get('sample_interval', 1000) == 0:
                    self._save_samples(X_0, X_1, epoch)

            # Save checkpoint at the end of each epoch
            utils.save_checkpoint(
                self.model, self.optimizer, epoch, self.checkpoint_dir
            )

        logging.info("Training complete.")

    def _save_samples(self, X_0, X_1, epoch, n_samples=8):
        """
        Saves a grid of sample images: [Input, Predicted, Reference].
        This requires running the full generation process (Algorithm 2)[cite: 197].
        """
        logging.info("Generating samples...")
        self.model.eval()
        
        # Take a small batch for sampling
        X_0_sample = X_0[:n_samples]
        X_1_sample = X_1[:n_samples]

        with torch.no_grad():
            with autocast(device_type=self.device, enabled=self.use_fp16):
                # Run the full reverse process (Algorithm 2) [cite: 197]
                # We start from the degraded image X_1 (t=N) [cite: 197]
                X_pred = self.diffusion.sample_ddpm(self.model, X_1_sample)

        filepath = os.path.join(
            self.sample_dir, 
            f"epoch_{epoch + 1}_step_{self.global_step}.png"
        )
        
        # Save grid: [Input (X_1), Output (X_pred), Reference (X_0)]
        utils.save_image_grid(X_1_sample, X_pred, X_0_sample, filepath, n_images=n_samples)
        
        logging.info(f"Saved sample grid to {filepath}")
        self.model.train() # Set model back to training mode
