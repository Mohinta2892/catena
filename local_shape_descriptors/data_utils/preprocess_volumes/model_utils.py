"""
This script should allow us to figure out the input and output shapes based on given downsample factors and kernel
sizes in the config.
We s

"""
from __future__ import annotations
import logging
import sys
import os

# add current directory to path and allow absolute imports
sys.path.insert(0, '../../.')

from config.config import get_cfg_defaults
from models.models import *

# import the same logger
logger = logging.getLogger(__name__)


def calculate_input_output_sizes(model, start_input_size=(256, 256, 256)):

    input_size = start_input_size
    device = torch.device("cuda:0")
    model.to(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory // 2

    # Estimate memory consumption (considering float32)
    required_memory = input_size[0] * input_size[1] * 4
    if required_memory > total_memory:
        break  # Skip this size if it exceeds the GPU memory limit

    return input_sizes, output_sizes


def rename_keys(original_config, key_mapping):
    for new_key, old_key in key_mapping.items():
        if hasattr(original_config, old_key):
            original_config[new_key] = getattr(original_config, old_key)

    return original_config


if __name__ == '__main__':

    # Get the config - duplicated code for now, to enable running independently
    # TODO: merge with trainer.py
    cfg = get_cfg_defaults()

    # adding a copy of global model params to avoid if-else in train based on input data
    if cfg.DATA.FIB:
        key_mapping = {
            'MODEL': 'MODEL_ISO'
        }
    else:
        key_mapping = {
            'MODEL': 'MODEL_ANISO'
        }

    cfg = rename_keys(cfg, key_mapping)

    # # Define your UNet model architecture
    # in_channels = 1  # Example input channels
    # out_channels = 64  # Example output channels
    # down_factors = [2, 2, 2, 2]  # Example downsampling factors
    # kernel_sizes = [3, 3, 3, 3]  # Example kernel sizes
    # start_input_size = (130, 130)  # Example starting input size
    start_input_size =[(196, 196, 196),  (224, 224, 224),  (268, 268, 268), (292, 292, 292), (368, 368, 368)]
    for si in start_input_size:
        model, _ = initialize_model(cfg)

        # Input and output sizes calculation
        input_sizes, output_sizes = calculate_input_output_sizes(model, start_input_size)

        # Print the results
        print(f"Layer {i + 1}: Input Size: {input_size}, Output Size: {output_size}")
