import torch
import logging
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from cleanfid.features import build_feature_extractor
from cleanfid.fid import get_batch_features

from src.options import Options
from src.dataset import get_base_imagenet_val_dataset
from src.utils import unnormalize_to_zero_one


def collect_features(dataset, batch_size, device):

    """Collect features from the dataset using a feature extractor."""

    feature_extractor = build_feature_extractor(
        mode="legacy_pytorch",
        device=device,
        use_dataparallel=True,
    )
    all_features = []
    pbar = tqdm(DataLoader(dataset, batch_size=batch_size), desc="Extracting features", leave=True)
    for batch in pbar:
        images, _ = batch
        images = unnormalize_to_zero_one(images).to(device)
        all_features.append(
            get_batch_features(
                images,
                feature_extractor,
                device=device,
            )
        )
    np_features = np.concatenate(all_features, axis=0)
    mu = np.mean(np_features, axis=0)
    sigma = np.cov(np_features, rowvar=False)
    return mu, sigma

def compute_fid_stats(supperres_set: bool, opt: Options, device: torch.device = "cpu"):

    """Compute FID statistics (mean and covariance)."""

    degradation = ""
    if supperres_set:
        logging.info("Preparing super-resolution validation dataset (50K images)...")
        degradation = "superres"
    val_dataset = get_base_imagenet_val_dataset(
        opt.val_data_dir,
        opt.stats_image_size,
        opt.image_names_file,
        degradation,
    )   
    logging.info("Loaded validation dataset.")

    # Compute FID statistics
    mu, sigma = collect_features(
        dataset=val_dataset,
        batch_size=opt.batch_size,
        device=device
    )
    logging.info("Computed FID statistics.")

    return mu, sigma




    

