"""
Model definition for I2SB.

This file imports the U-Net factory functions from the original
OpenAI 'guided-diffusion' repository (placed in src/guided_diffusion).
It uses these factories to build the model specified in the I2SB paper:
an ADM U-Net initialized with the unconditional ImageNet 256x256 checkpoint.

"""

import logging
import torch

from src.options import Options

try:
    from src.guided_diffusion.script_util import (
        model_and_diffusion_defaults,
        create_model
    )
except ImportError:
    logging.error("Could not import guided-diffusion. "
                  "Did you place the OpenAI repo files in src/guided_diffusion?")
    raise

def get_model(config: Options) -> torch.nn.Module:
    """
    Creates the U-Net model based on the configuration, using the
    OpenAI factory.
    
    Args:
        config (dict): A dictionary, typically loaded from a YAML file.
                       
    Returns:
        torch.nn.Module: The U-Net model.
    """

    adm_config = model_and_diffusion_defaults()

    adm_config.update({
        'image_size': config.image_size,
        'num_channels': 128,
        'num_head_channels': 64,
        'num_res_blocks': 3,
        'learn_sigma': True,
        'class_cond': False,
        'resblock_updown': True,
        'use_new_attention_order': True,
    })
    
    model = create_model(
        image_size=int(adm_config['image_size']),
        num_channels=int(adm_config['num_channels']),
        num_res_blocks=int(adm_config['num_res_blocks']),
        learn_sigma=bool(adm_config['learn_sigma']),
        class_cond=bool(adm_config['class_cond']),
        use_checkpoint=bool(adm_config.get('use_checkpoint', False)),
        attention_resolutions=str(adm_config['attention_resolutions']),
        num_heads=int(adm_config.get('num_heads', 4)),
        num_heads_upsample=-1,
        use_scale_shift_norm=bool(adm_config.get('use_scale_shift_norm', False)),
        dropout=float(adm_config['dropout']),
    )
    
    logging.info("Created ADM U-Net model.")
    
    return model

def load_adm_checkpoint(model: torch.nn.Module, config: Options):
    """
    Loads the pre-trained ADM checkpoint weights into the model.
    """
    checkpoint_path = getattr(config, 'adm_checkpoint_path', None)
    if not checkpoint_path:
        logging.warning("No 'adm_checkpoint_path' found in config. Skipping ADM checkpoint loading.")
        return

    try:
        device = next(model.parameters()).device
        weights = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(weights)
        logging.info(f"Successfully loaded ADM checkpoint from {checkpoint_path}")
    except Exception as e:
        logging.error(f"Error loading ADM checkpoint: {e}")
        raise
