import logging
import os
import torch
from torchvision.utils import save_image
from torchvision import transforms as T
from tqdm import tqdm
from cleanfid import fid
from transformers import AutoImageProcessor, ResNetForImageClassification
from torchvision.transforms import ToPILImage
from accelerate import Accelerator

from src.diffusion import DiffusionProcess
from src.options import Options
from src.utils import unnormalize_to_zero_one


class Validator:
    def __init__(
        self,
        model: torch.nn.Module,
        diffusion: DiffusionProcess,
        data_loader: torch.utils.data.DataLoader,
        config: Options,
        accelerator: Accelerator,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data_loader = data_loader
        self.config = config
        self.accelerator = accelerator
        self.device = self.accelerator.device

        self.log_dir = config.log_dir

        # Directory to save generated fake images
        self.fake_dir = os.path.join(self.log_dir, "fake")
        # Path to precomputed ImageNet statistics for FID calculation
        self.imagenet_stats_path = config.imagenet_stats_path

        # Prepare model, diffusion, and data_loader with accelerator
        self.model, self.diffusion = self.accelerator.prepare(self.model, self.diffusion)
        self.data_loader = self.accelerator.prepare(self.data_loader)

        # Load pretrained classifier for classifier accuracy calculation
        self.classifier = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(self.device)
        self.classifier.eval()
        # Image processor for the classifier
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.classifier_resize = T.Resize((224, 224))

        if self.accelerator.is_main_process:
            # clear existing fake images directory
            if os.path.exists(self.fake_dir):
                logging.info("Clearing existing fake images directory...")
                for file in os.listdir(self.fake_dir):
                    file_path = os.path.join(self.fake_dir, file)
                    if os.path.isfile(file_path) and file.lower().endswith('.png'):
                        os.remove(file_path)
            logging.info("Creating fake images directory...")

            os.makedirs(self.fake_dir, exist_ok=True)
            logging.info("Validator initialized.")

    @torch.no_grad()
    def validate(self):
        if self.accelerator.is_main_process:
            logging.info("Starting validation...")
        self.model.eval()

        batch_idx = 0
        total_correct = 0
        total_samples = 0
        total_batches = len(self.data_loader)
        pbar = tqdm(self.data_loader, desc="Validation", leave=True)

        for X_0, X_1, labels in pbar:
            X_0 = X_0.to(self.device)
            X_1 = X_1.to(self.device)
            labels = labels.to(self.device)

            with self.accelerator.autocast():
                pred = self.diffusion.sample_ddpm(self.model, X_1, self.diffusion.n_steps)

            # Denormalize restored images to [0, 1] for saving and PIL conversion
            pred_denorm = unnormalize_to_zero_one(pred)

            # Save images for FID calculation
            for i in range(X_0.shape[0]):
                fake_path = os.path.join(self.fake_dir, f"fake_{self.accelerator.process_index}_{batch_idx}_{i}.png")
                save_image(pred_denorm[i], fake_path)

            # Classifier on predicted images
            # Convert to PIL images
            to_pil = ToPILImage()
            pil_images = [self.classifier_resize(to_pil(img.cpu())) for img in pred_denorm]

            # Preprocess for classifier
            pred_preprocessed = self.image_processor(pil_images, return_tensors="pt").to(self.device)
            # Get classifier outputs
            outputs = self.classifier(**pred_preprocessed)
            preds = torch.argmax(outputs.logits, dim=-1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            batch_idx += 1
            if self.accelerator.is_main_process:
                logging.info(f"Processed batch {batch_idx} / {total_batches}, Current CA: {total_correct} / {total_samples} = {total_correct / total_samples:.4f}")
        
        # Gather results from all processes
        total_correct = self.accelerator.gather(total_correct).sum().item()
        total_samples = self.accelerator.gather(total_samples).sum().item()

        if self.accelerator.is_main_process:
            logging.info("All batches processed. Computing FID and Classifier Accuracy...")
        
            # Compute the FID score using precomputed ImageNet statistics
            fid_score = fid.compute_fid(self.imagenet_stats_path, self.fake_dir, mode="legacy_pytorch")

            # Compute classifier accuracy
            ca_score = total_correct / total_samples if total_samples > 0 else 0.0

            logging.info(f"Validation completed. FID Score: {fid_score} Classifier Accuracy: {ca_score}")
            return fid_score, ca_score
        else:
            return None, None
