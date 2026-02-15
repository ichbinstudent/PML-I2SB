import numpy as np
import argparse
import torch


from cleanfid.fid import frechet_distance

from src.fid_utils import collect_features
from src.dataset import get_base_imagenet_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_dir', type=str, required=True,
                        help='Path to directory containing generated images')
    parser.add_argument('--stats_path', type=str, required=True,
                        help='Path to reference statistics .npz file')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--image_size', type=int, default=75,
                        help='Image size')
    args = parser.parse_args()

    # get dataset of generated images
    generated_dataset = get_base_imagenet_dataset(
        data_dir=args.fake_dir,
        image_size=args.image_size,
        is_train=False
    )

    # copute mean and covariance of generated images
    mu, sigma = collect_features(
        dataset=generated_dataset,
        batch_size=args.batch_size,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # load reference statistics
    stats = np.load(args.stats_path)
    ref_mu = stats['mu']
    ref_sigma = stats['sigma']
    print("Loaded reference statistics.")

    # compute frechet distance
    fid_score = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    print(f"FID score: {fid_score}")
