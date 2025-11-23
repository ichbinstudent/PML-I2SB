import os
from PIL import Image, ImageDraw, ImageFont

def create_degradation_grid():
    # List of images and their labels
    images_info = [
        ("original.png", "Original"),
        ("degraded_jpeg_50.png", "JPEG (50)"),
        ("degraded_blur_uni.png", "Uniform Blur"),
        ("degraded_blur_gauss.png", "Gaussian Blur"),
        ("degraded_inpaint_center.png", "Center Inpaint"),
        ("degraded_inpaint_10_20pct_freeform.png", "Freeform Inpaint\n(10-20%)"),
        ("degraded_superres_bicubic.png", "Super-res\n(Bicubic)"),
        ("degraded_superres_bilinear.png", "Super-res\n(Bilinear)")
    ]

    # Load all images
    images = []
    for filename, label in images_info:
        if os.path.exists(filename):
            img = Image.open(filename)
            # Ensure proper RGB conversion
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append((img, label))
        else:
            print(f"Warning: {filename} not found")

    if not images:
        print("No images found!")
        return

    # Grid layout: 2 rows, 4 columns (perfect for 8 images)
    cols = 4
    rows = 2

    # Image dimensions
    img_width, img_height = images[0][0].size

    # Spacing and text
    margin = 20
    text_height = 60  # Space for text below each image
    grid_width = cols * img_width + (cols - 1) * margin
    grid_height = rows * (img_height + text_height) + (rows - 1) * margin

    # Create the grid image with proper color mode
    grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))  # White background
    draw = ImageDraw.Draw(grid_img)

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Place images in grid
    for i, (img, label) in enumerate(images):
        row = i // cols
        col = i % cols

        # Calculate position
        x = col * (img_width + margin)
        y = row * (img_height + text_height + margin)

        # Paste the image
        grid_img.paste(img, (x, y))

        # Add text label below the image
        text_y = y + img_height + 5
        # Handle multi-line labels
        lines = label.split('\n')
        for line_idx, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = x + (img_width - text_width) // 2  # Center the text
            draw.text((text_x, text_y + line_idx * 20), line, fill='black', font=font)

    # Save the grid image
    output_filename = "degradation_comparison_grid.png"
    grid_img.save(output_filename, 'PNG')
    print(f"Grid image saved as: {output_filename}")
    print(f"Grid size: {grid_width}x{grid_height} pixels")

    if not images:
        print("No images found!")
        return

    # Grid layout: 2 rows, 4 columns (perfect for 8 images)
    cols = 4
    rows = 2

    # Image dimensions
    img_width, img_height = images[0][0].size

    # Spacing and text
    margin = 20
    text_height = 60  # Space for text below each image
    grid_width = cols * img_width + (cols - 1) * margin
    grid_height = rows * (img_height + text_height) + (rows - 1) * margin

    # Create the grid image
    grid_img = Image.new('RGB', (grid_width, grid_height), 'white')
    draw = ImageDraw.Draw(grid_img)

    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Place images in grid
    for i, (img, label) in enumerate(images):
        row = i // cols
        col = i % cols

        # Calculate position
        x = col * (img_width + margin)
        y = row * (img_height + text_height + margin)

        # Paste the image
        grid_img.paste(img, (x, y))

        # Add text label below the image
        text_y = y + img_height + 5
        # Handle multi-line labels
        lines = label.split('\n')
        for line_idx, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = x + (img_width - text_width) // 2  # Center the text
            draw.text((text_x, text_y + line_idx * 20), line, fill='black', font=font)

    # Save the grid image
    output_filename = "degradation_comparison_grid.png"
    grid_img.save(output_filename)
    print(f"Grid image saved as: {output_filename}")
    print(f"Grid size: {grid_width}x{grid_height} pixels")

if __name__ == "__main__":
    create_degradation_grid()