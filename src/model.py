"""
Model definition for I2SB.

This file imports the U-Net factory functions from the original
OpenAI 'guided-diffusion' repository (placed in src/guided_diffusion).
It uses these factories to build the model specified in the I2SB paper:
an ADM U-Net initialized with the unconditional ImageNet 256x256 checkpoint.

"""

import logging
import torch

try:
    from src.guided_diffusion.script_util import (
        model_and_diffusion_defaults,
        create_model
    )
except ImportError:
    logging.error("Could not import guided-diffusion. "
                  "Did you place the OpenAI repo files in src/guided_diffusion?")
    raise

def get_model(config):
    """
    Creates the U-Net model based on the configuration, using the
    OpenAI factory.
    
    Args:
        config (dict): A dictionary, typically loaded from a YAML file.
                       
    Returns:
        torch.nn.Module: The U-Net model.
    """
    image_size = config.get('image_size', 256)
    if image_size != 256:
        logging.warning(
            f"Model config is optimized for 256x256, but image_size is {image_size}."
        )

    # Get the default ADM parameters for ImageNet 256x256
    # These defaults are defined in the 'model_and_diffusion_defaults'
    adm_config = model_and_diffusion_defaults()

    # These values are from the original ADM paper's scripts
    # for the 256x256 unconditional model.
    adm_config.update({
        'image_size': 256,
        'num_channels': 256,
        'num_res_blocks': 2,
        'attention_resolutions': '32,16,8',
        'learn_sigma': True,
        'use_scale_shift_norm': True,
        'dropout': 0.1,
        'class_cond': False, # This is an unconditional model
        'use_fp16': False    # You can set this in your config if you want
    })

    adm_config.update(config.get('model_params', {}))
    
    model = create_model(
        image_size=int(adm_config['image_size']),
        num_channels=int(adm_config['num_channels']),
        num_res_blocks=int(adm_config['num_res_blocks']),
        learn_sigma=bool(adm_config['learn_sigma']),
        class_cond=False, # Force unconditional
        use_checkpoint=bool(adm_config.get('use_checkpoint', False)),
        attention_resolutions=str(adm_config['attention_resolutions']),
        num_heads=int(adm_config.get('num_heads', 4)),
        num_head_channels=int(adm_config.get('num_head_channels', 64)),
        num_heads_upsample=-1,
        use_scale_shift_norm=bool(adm_config['use_scale_shift_norm']),
        dropout=float(adm_config['dropout']),
        resblock_updown=bool(adm_config.get('resblock_updown', True)),
        use_fp16=bool(adm_config['use_fp16']),
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Created ADM U-Net model with {total_params/1e6:.2f}M params.")
    
    return model

def load_adm_checkpoint(model, config):
    """
    Loads the pre-trained ADM checkpoint weights into the model.
    """
    checkpoint_path = config.get('adm_checkpoint_path')
    if not checkpoint_path:
        logging.warning("No 'adm_checkpoint_path' found in config. "
                        "Starting model from scratch.")
        return

    try:
        device = next(model.parameters()).device
        weights = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(weights)
        logging.info(f"Successfully loaded ADM checkpoint from {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error loading ADM checkpoint: {e}")
        logging.error("Ensure the checkpoint path is correct and the model "
                      "parameters match the checkpoint.")
        raise