import logging
import os
import torch

from torchvision.utils import save_image
from tqdm import tqdm
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
        opt: Options,
        accelerator: Accelerator,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data_loader = data_loader
        self.config = opt
        self.accelerator = accelerator
        self.device = self.accelerator.device

        self.log_dir = opt.log_dir

        # Directory to save generated fake images
        self.fake_dir = os.path.join(self.log_dir, f"fake_{opt.degradation}")
        # Path to precomputed ImageNet statistics for FID calculation
        self.imagenet_stats_path = opt.imagenet_stats_path

        # Prepare model, diffusion, and data_loader with accelerator
        self.model, self.diffusion = self.accelerator.prepare(self.model, self.diffusion)
        #self.data_loader = self.accelerator.prepare(self.data_loader)

        # Load pretrained classifier for classifier accuracy calculation
        self.classifier = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        self.classifier.eval()
        self.classifier = self.accelerator.prepare(self.classifier)
        # Image processor for the classifier
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")

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

    @torch.inference_mode()
    def validate(self):
        if self.accelerator.is_main_process:
            logging.info("Starting validation...")
        self.model.eval()


        batch_idx = 0
        total_correct = 0
        total_samples = 0
        total_batches = len(self.data_loader)
        if self.accelerator.is_main_process:
            pbar = tqdm(self.data_loader, desc="Validation", leave=True)
        else:
            pbar = self.data_loader

        for _, X_1, labels in pbar:

            with self.accelerator.autocast():
                pred = self.diffusion.sample_ddpm(self.model, X_1, self.diffusion.n_steps)

            # Denormalize restored images to [0, 1] for saving and PIL conversion
            pred_denorm = unnormalize_to_zero_one(pred)

            # Save images for FID calculation
            for i in range(X_1.shape[0]):
                fake_path = os.path.join(
                    self.fake_dir, 
                    f"fake_{self.accelerator.process_index}_{batch_idx}_{i}.png"
                )
                save_image(pred_denorm[i], fake_path)

            # Classifier on predicted images
            # Convert to PIL images
            to_pil = ToPILImage()
            pil_images = [to_pil(img.cpu()) for img in pred_denorm]

            # Preprocess for classifier
            pred_preprocessed = self.image_processor(pil_images, return_tensors="pt")

            # Get classifier outputs
            with self.accelerator.autocast():
                logits = self.classifier(**pred_preprocessed).logits
        
            preds = torch.argmax(logits, dim=-1)

            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)

            batch_idx += 1
            # if self.accelerator.is_main_process:
            logging.info(f"Processed batch {batch_idx} / {total_batches}, Current CA: {total_correct} / {total_samples} = {total_correct / total_samples:.4f}")
            if not self.accelerator.is_main_process:
                print(f"Processed batch {batch_idx} / {total_batches}, Current CA: {total_correct} / {total_samples} = {total_correct / total_samples:.4f}")

        
        # Barrier to ensure all processes are done
        self.accelerator.wait_for_everyone()

        # Gather results from all processes - convert to tensors first!
        total_correct_tensor = torch.tensor(total_correct, device=self.device)
        total_samples_tensor = torch.tensor(total_samples, device=self.device)

        # Gather results from all processes
        total_correct_all = self.accelerator.gather(total_correct_tensor).sum().item()
        total_samples_all = self.accelerator.gather(total_samples_tensor).sum().item()

        if self.accelerator.is_main_process:
            logging.info("All batches processed. Computing Classifier Accuracy...")

            # Compute classifier accuracy
            ca_score = total_correct_all / total_samples_all if total_samples_all > 0 else 0.0

            logging.info(f"Validation completed. Classifier Accuracy: {ca_score}")
            return ca_score
        else:
            return None
