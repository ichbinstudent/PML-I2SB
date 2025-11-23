import torch
import logging

# Import your model factory function
from src.model import get_model

# --- Configuration ---
BATCH_SIZE = 4
IMAGE_SIZE = 256
TIMESTEPS = 1000

# Mock config dictionary (only needs parameters for the model)
mock_config = {
    'image_size': IMAGE_SIZE,
    'adm_checkpoint_path': None, # We don't need to load weights for a shape test
    'model_params': {
        'learn_sigma': True,  # This is the default for the ADM 256x256 model
        'use_fp16': False
    }
}

# --- Test Script ---
def test_model_forward_pass():
    """
    Tests that the U-Net can be instantiated and performs a forward pass.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("--- Testing Model Forward Pass ---")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Instantiate the model
    try:
        model = get_model(mock_config)
        model.to(device)
        model.eval() # Set to evaluation mode
    except Exception as e:
        logging.error(f"Failed to instantiate model: {e}")
        logging.error("Make sure you have placed the guided-diffusion files "
                      "in the src/guided_diffusion/ directory.")
        return

    logging.info("Model instantiated successfully.")

    # 2. Create dummy inputs
    # Dummy image batch (X_t)
    dummy_images = torch.randn(
        BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE
    ).to(device)
    
    # Dummy timestep batch (t)
    dummy_timesteps = torch.randint(
        0, TIMESTEPS, (BATCH_SIZE,)
    ).to(device)

    logging.info(f"Input image shape: {list(dummy_images.shape)}")
    logging.info(f"Input timestep shape: {list(dummy_timesteps.shape)}")

    # 3. Perform a forward pass
    try:
        with torch.no_grad(): # No need to track gradients
            output_tensor = model(dummy_images, dummy_timesteps)
    except Exception as e:
        logging.error(f"Model forward pass failed: {e}")
        return

    # 4. Check the output shape
    expected_shape = (BATCH_SIZE, 6, IMAGE_SIZE, IMAGE_SIZE)
    logging.info(f"Output tensor shape: {list(output_tensor.shape)}")
    
    assert output_tensor.shape == expected_shape, \
        f"Output shape is incorrect! " \
        f"Expected {expected_shape}, but got {list(output_tensor.shape)}"

    logging.info("--- Test Passed! ---")
    logging.info("Model forward pass is working correctly.")
    logging.info("Output shape (B, 6, H, W) is correct for learn_sigma=True.")


if __name__ == "__main__":
    test_model_forward_pass()

    