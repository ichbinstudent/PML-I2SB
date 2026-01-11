import argparse
import os
import logging
import numpy as np
import torch

from src.options import Options
from src.utils import setup_logging
from src.fid_utils import compute_fid_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, 
                        help='Path to the config file.')
    parser.add_argument('--superres', action='store_true',
                        help='Compute FID stats for super-resolution dataset (50K images)')
    args = parser.parse_args()
    opt = Options.from_yaml(args.config)
    supperres_set = args.superres

    setup_logging(opt.log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Starting FID statistics computation...")

    mu, sigma = compute_fid_stats(supperres_set, opt, device)

    # Save statistics
    dir_path = os.path.dirname(opt.imagenet_stats_path)
    os.makedirs(dir_path, exist_ok=True)
    size = "50K" if supperres_set else "10K"
    fn = os.path.join(dir_path, f"fid_imagenet_{size}_{opt.stats_image_size}.npz")
    np.savez(fn, mu=mu, sigma=sigma)
    logging.info(f"Saved FID reference statistics to {fn}!")

    logging.info("FID statistics computation completed.")