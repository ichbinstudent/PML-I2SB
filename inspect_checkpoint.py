#!/usr/bin/env python3
"""Inspect the checkpoint file to see what keys and shapes it contains."""

import torch
import sys
import os

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "./checkpoints/82nd_checkpoint.pth"

print(f"Loading checkpoint from: {checkpoint_path}")
print(f"File size: {os.path.getsize(checkpoint_path) / 1e9:.2f} GB")

try:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print("\n" + "=" * 80)
    print("CHECKPOINT STRUCTURE")
    print("=" * 80)
    print(f"Top-level keys: {list(checkpoint.keys())}")
    
    if isinstance(checkpoint, dict):
        for key in checkpoint.keys():
            value = checkpoint[key]
            if isinstance(value, dict):
                print(f"\n'{key}' is a dictionary with {len(value)} items")
                if len(value) > 0:
                    first_5_keys = list(value.keys())[:5]
                    print(f"  First 5 keys: {first_5_keys}")
            elif isinstance(value, torch.Tensor):
                print(f"'{key}': Tensor with shape {value.shape}, dtype {value.dtype}")
            else:
                print(f"'{key}': {type(value)} = {value}")
    
    # Try to extract model_state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print("\n" + "=" * 80)
        print("MODEL STATE DICT ANALYSIS")
        print("=" * 80)
        print(f"Number of parameters: {len(state_dict)}")
        print(f"\nFirst 30 keys:")
        for i, key in enumerate(list(state_dict.keys())[:30]):
            shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else "N/A"
            print(f"  {i+1}. {key}: {shape}")
        
        if len(state_dict) > 30:
            print(f"  ... and {len(state_dict) - 30} more keys")
    
    # Show other checkpoint info
    if 'epoch' in checkpoint:
        print(f"\nEpoch: {checkpoint['epoch']}")
    if 'optimizer_state_dict' in checkpoint:
        print(f"Optimizer state dict present: True")
        
except Exception as e:
    print(f"ERROR loading checkpoint: {e}")
    import traceback
    traceback.print_exc()
