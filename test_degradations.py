import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from src.options import Options
from src.degradations.blur import build_blur
from src.degradations.jpeg import build_jpeg
from src.degradations.superres import build_superres
from src.degradations.inpainting import build_inpaint_center, build_inpaint_freeform

def test_degradations():
    # Set up options
    opt = Options()
    opt.image_size = 256
    opt.device = 'cpu'  # Use CPU for testing

    # Load sample image
    image_path = "sample.jpg"
    image = Image.open(image_path).convert('RGB')

    # Transform to tensor and normalize to [-1, 1] as expected by degradations
    transform = T.Compose([
        T.Resize((opt.image_size, opt.image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    sample_image = transform(image).unsqueeze(0)  # Add batch dimension

    print("Testing degradation functions...")
    print(f"Loaded image: {image_path}")
    print(f"Sample image shape: {sample_image.shape}")
    print(f"Sample image range: [{sample_image.min():.3f}, {sample_image.max():.3f}]")

    # Test different degradation types
    degradation_types = [
        "jpeg-50",
        "blur-uni",
        "blur-gauss",
        "inpaint-center",
        "inpaint-10-20% freeform",
        "superres-bicubic",
        "superres-bilinear"
    ]

    # Save original for comparison
    original_pil = T.ToPILImage()(sample_image.squeeze(0) * 0.5 + 0.5)  # Denormalize to [0, 1]
    original_pil.save("original.png")

    for deg_type in degradation_types:
        try:
            print(f"\nTesting {deg_type}:")
            if deg_type.startswith("jpeg"):
                quality = int(deg_type.split("-")[1])
                degradation_fn = build_jpeg(quality)
            elif deg_type.startswith("blur"):
                kernel_type = deg_type.split("-")[1]
                degradation_fn = build_blur(opt, kernel_type)
            elif deg_type.startswith("inpaint"):
                if deg_type == "inpaint-center":
                    degradation_fn = build_inpaint_center(opt)
                else:
                    mask_type = deg_type.split("-", 1)[1]  # Get everything after "inpaint-"
                    degradation_fn = build_inpaint_freeform(opt, mask_type)
            elif deg_type.startswith("superres"):
                sr_filter = deg_type.split("-")[1]
                degradation_fn = build_superres(opt, sr_filter, image_size=opt.image_size)
            else:
                continue

            result = degradation_fn(sample_image)
            if isinstance(result, tuple):
                degraded_image, mask = result
            else:
                degraded_image = result

            print(f"  Output shape: {degraded_image.shape}")
            print(f"  Output range: [{degraded_image.min():.3f}, {degraded_image.max():.3f}]")
            print(f"  Mean difference: {torch.abs(degraded_image - sample_image).mean():.4f}")

            # Save degraded image
            degraded_pil = T.ToPILImage()(degraded_image.squeeze(0) * 0.5 + 0.5)  # Denormalize
            output_filename = f"degraded_{deg_type.replace('-', '_').replace('%', 'pct').replace(' ', '_')}.png"
            degraded_pil.save(output_filename)
            print(f"  Saved result to: {output_filename}")

        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    test_degradations()