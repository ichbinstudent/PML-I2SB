import logging
import os
import torch
from tqdm import tqdm
from clean_fid import fid
import torchvision.utils
from transformers import AutoImageProcessor, ResNetForImageClassification

from src.diffusion import DiffusionProcess
from src.options import Options


class Validator:
    def __init__(
        self,
        model: torch.nn.Module,
        diffusion: DiffusionProcess,
        data_loader: torch.utils.data.DataLoader,
        config: Options,
        device: torch.device,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data_loader = data_loader
        self.config = config
        self.device = device

        self.log_dir = config.log_dir

        self.real_dir = os.path.join(self.log_dir, "real")
        self.fake_dir = os.path.join(self.log_dir, "fake")

        self.classifier = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(self.device)
        self.classifier.eval()
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

        os.makedirs(self.real_dir, exist_ok=True)
        os.makedirs(self.fake_dir, exist_ok=True)

        logging.info("Validator initialized.")

    @torch.no_grad()
    def validate(self):
        logging.info("Starting validation...")
        self.model.eval()

        batch_idx = 0
        total_correct = 0
        total_samples = 0
        pbar = tqdm(self.data_loader, desc="Validation", leave=True)

        for X_0, X_1, labels in pbar:
            X_0 = X_0.to(self.device)
            X_1 = X_1.to(self.device)
            labels = labels.to(self.device)

            pred = self.diffusion.sample_ddpm(self.model, X_1, self.diffusion.n_steps)

            # denormalize images to [0, 1] for saving
            X_0_denorm = (X_0 + 1) * 0.5
            pred_denorm = (pred + 1) * 0.5

            # save images for FID calculation
            for i in range(X_0.shape[0]):
                real_path = os.path.join(self.real_dir, f"real_{batch_idx}_{i}.png")
                fake_path = os.path.join(self.fake_dir, f"fake_{batch_idx}_{i}.png")
                torchvision.utils.save_image(X_0_denorm[i], real_path)
                torchvision.utils.save_image(pred_denorm[i], fake_path)

            # classifier on predicted images
            pred_preprocessed = self.image_processor(pred_denorm, return_tensors="pt").to(self.device)
            outputs = self.classifier(**pred_preprocessed)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            batch_idx += 1

        fid_score = fid.compute_fid(self.real_dir, self.fake_dir, mode="legacy_pytorch")
        
        ca_score = total_correct / total_samples if total_samples > 0 else 0.0

        logging.info(f"Validation completed. FID Score: {fid_score} Classifier Accuracy: {ca_score}")
        return fid_score, ca_score
