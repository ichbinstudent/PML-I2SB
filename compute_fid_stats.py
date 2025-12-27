import argparse
import torch
import logging
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from cleanfid.features import build_feature_extractor
from cleanfid.fid import get_batch_features

from src.options import Options
from src.dataset import get_base_imagenet_dataset
from src.utils import unnormalize_to_zero_one, setup_logging
    



def collect_features(dataset, batch_size, device):
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

def compute_fid_stats(opt: Options, device: torch.device = "cpu"):

    val_dataset = get_base_imagenet_dataset(
        opt.val_data_dir,
        opt.stats_image_size,
        is_train=False
    )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=opt.batch_size,
    #     shuffle=False,
    #     num_workers=opt.num_workers,
    #     pin_memory=torch.cuda.is_available()
    # )
    logging.info("Loaded validation dataset.")

    # Compute FID statistics
    mu, sigma = collect_features(
        dataset=val_dataset,
        batch_size=opt.batch_size,
        device=device
    )
    logging.info("Computed FID statistics.")

    # Save statistics
    dir_path = os.path.dirname(opt.imagenet_stats_path)
    os.makedirs(dir_path, exist_ok=True)
    fn = os.path.join(dir_path, f"fid_imagenet_{opt.image_size}.npz")
    np.savez(fn, mu=mu, sigma=sigma)
    logging.info(f"Saved FID reference statistics to {fn}!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the config file.')
    args = parser.parse_args()
    opt = Options.from_yaml(args.config)

    setup_logging(opt.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Starting FID statistics computation...")

    compute_fid_stats(opt, device)

    logging.info("FID statistics computation completed.")
    

