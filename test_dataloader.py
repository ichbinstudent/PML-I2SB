"""
Test script for the I2SB dataloader.
Tests that the dataset properly loads images and applies degradations.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from src.dataset import (
    get_base_imagenet_dataset,
    I2SBImageNetWrapper
)
from src.options import Options


def denormalize(tensor):
    """Convert tensor from [-1, 1] to [0, 1] for visualization."""
    return (tensor + 1) / 2


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization."""
    # tensor: [C, H, W] -> [H, W, C]
    return denormalize(tensor).permute(1, 2, 0).cpu().numpy()


def test_dataloader(degradation_type='jpeg-10', batch_size=4):
    """
    Test the dataloader with sample images.
    
    Args:
        degradation_type: Type of degradation to test (e.g., 'jpeg-10', 'blur-gauss')
        batch_size: Number of images to load in a batch
    """
    print(f"Testing dataloader with degradation: {degradation_type}")
    
    # Setup Options
    opt = Options()
    opt.image_size = 256
    opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt.degradation = degradation_type
    
    print(f"Using device: {opt.device}")
        
    # Test with train data
    print("\n--- Loading train dataset ---")
    train_base_dataset = get_base_imagenet_dataset(
        data_dir='./dataset/train',
        image_size=opt.image_size,
        is_train=True
    )
    
    train_dataset = I2SBImageNetWrapper(
        base_dataset=train_base_dataset,
        opt=opt
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for debugging
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Test with val data
    print("\n--- Loading validation dataset ---")
    val_base_dataset = get_base_imagenet_dataset(
        data_dir='./dataset/val',
        image_size=opt.image_size,
        is_train=False
    )
    
    val_dataset = I2SBImageNetWrapper(
        base_dataset=val_base_dataset,
        opt=opt
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Get one batch from train loader
    print("\n--- Testing train batch ---")
    X_0_batch, X_1_batch = next(iter(train_loader))
    
    print(f"X_0 (clean) shape: {X_0_batch.shape}")
    print(f"X_0 value range: [{X_0_batch.min():.3f}, {X_0_batch.max():.3f}]")
    print(f"X_0 device: {X_0_batch.device}")
    
    print(f"\nX_1 (degraded) shape: {X_1_batch.shape}")
    print(f"X_1 value range: [{X_1_batch.min():.3f}, {X_1_batch.max():.3f}]")
    print(f"X_1 device: {X_1_batch.device}")
    
    # Visualize first 4 images in the batch
    print("\n--- Creating visualization ---")
    num_display = min(4, batch_size)
    fig, axes = plt.subplots(2, num_display, figsize=(4*num_display, 8))
    
    if num_display == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_display):
        # Clean image
        axes[0, i].imshow(tensor_to_numpy(X_0_batch[i]))
        axes[0, i].set_title(f'Clean Image {i+1}')
        axes[0, i].axis('off')
        
        # Degraded image
        axes[1, i].imshow(tensor_to_numpy(X_1_batch[i]))
        axes[1, i].set_title(f'Degraded Image {i+1}\n({degradation_type})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    output_path = f'test_dataloader_{degradation_type.replace("-", "_")}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.close()
    
    print("\n✓ Dataloader test completed successfully!")
    return train_loader, val_loader


def test_multiple_degradations():
    """Test multiple degradation types."""
    degradation_types = [
        'jpeg-10',
        'blur-gauss',
        'blur-uni',
        'inpaint-center',
        'inpaint-freeform1020',
        'inpaint-freeform2030',
        'inpaint-freeform3040',
        'superres-bicubic'
    ]
    
    print("=" * 80)
    print("Testing multiple degradation types")
    print("=" * 80)
    
    for deg_type in degradation_types:
        print(f"\n{'='*80}")
        try:
            test_dataloader(degradation_type=deg_type, batch_size=4)
            print(f"✓ {deg_type} test passed")
        except Exception as e:
            print(f"✗ {deg_type} test failed: {str(e)}")
        print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the I2SB dataloader')
    parser.add_argument('--degradation', type=str, default='jpeg-10',
                        help='Degradation type to test (e.g., jpeg-10, blur-gauss)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for testing')
    parser.add_argument('--test_all', action='store_true',
                        help='Test all degradation types')
    
    args = parser.parse_args()
    
    if args.test_all:
        test_multiple_degradations()
    else:
        test_dataloader(
            degradation_type=args.degradation,
            batch_size=args.batch_size
        )
