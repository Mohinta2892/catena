"""
Warning  : Hardcoded to test on Cardona lab's Nvidia DGX with Titan XPs.
Only works with 3D, because more memory intensive.

This is a hack right now. Will try to load inside one docker and then spawn multiple models to different cards from there.
Ideal situation would be to orchestrate the spawning of dockers with kubernetes.


The hack involves:
- Read the `config_predict.py` file.
- Based on the model specified by the user now, we will pick a model checkpoint start value and end value
 and then split available gpus for inference on the datasets provided in the `BRAIN_VOL/data_{2d/3d}/test` folder.



"""
from __future__ import annotations
import sys
import os
import torch
import numpy as np
from glob import glob
import subprocess
import multiprocessing

# add current directory to path and allow absolute imports
sys.path.insert(0, '.')
from config.config_predict import *
from engine.predict.predict_all_yacs import predict
from engine.predict.predict_2d_all_yacs import predict_2d
from engine.post.run_waterz import run_waterz
from data_utils.preprocess_volumes.utils import calculate_min_2d_samples


def rename_keys(original_config, key_mapping):
    for new_key, old_key in key_mapping.items():
        if hasattr(original_config, old_key):
            original_config[new_key] = getattr(original_config, old_key)

    return original_config


def get_checkpoint_number(checkpoint):
    "custom sorter for checkpoint"
    return int(checkpoint.split('_')[-1])


# Function to call predicter.py for a subset of samples
def call_predicter(sample, checkpoint, gpu_id, out_filepath):
    # Modify the cfg object to set the cuda device
    cfg.DATA.CUDA_DEVICE = f"cuda:{gpu_id}"
    cfg.TRAIN.CHECKPOINT = checkpoint
    cfg.DATA.SAMPLE = sample
    # overwrite in the loop - otherwise will create zarr within zarr
    cfg.DATA.OUTFILE = out_filepath
    cfg.DATA.OUTFILE = os.path.join(cfg.DATA.OUTFILE, os.path.basename(cfg.DATA.SAMPLE))
    predict(cfg)


if __name__ == '__main__':
    """
    Reads params/args from `config_predict.py`.
    
    """
    cfg = get_cfg_defaults()
    # can be used to override pre-defined settings
    if os.path.exists("./experiment.yaml"):
        cfg.merge_from_file("experiment.yaml")

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
    # do not freeze this because we want to add other options in the predict script
    # cfg.freeze()
    print(cfg)

    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL)
    # data is expected to be here
    if cfg.DATA.DIM_2D:
        data_dir = os.path.join(data_dir, 'data_2d', 'test')
    else:
        data_dir = os.path.join(data_dir, 'data_3d', 'test')

    samples = glob(f"{data_dir}/*.zarr")

    # make the outfile path here - /basepath/modeltype/2d/checkpoint_name
    out_filepath = os.path.join(cfg.DATA.OUTFILE, cfg.TRAIN.MODEL_TYPE,
                                '2d' if cfg.DATA.DIM_2D else '3d',
                                "/".join(cfg.TRAIN.CHECKPOINT.split("/")[-2:]))
    if not os.path.exists(out_filepath):
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

    # # we expect data going in at this point to be sequentially traversed one at a time.
    # # TODO: batch inference could make it faster
    if cfg.TRAIN.BATCH_SIZE > 1:
        cfg.TRAIN.BATCH_SIZE = 1

    device_count = np.arange(torch.cuda.device_count())  # returns 8 in Nvidia-DGX Cardona-lab

    # split them to use only last 4 now for inference
    device_count = device_count[-4:]

    # read all checkpoints
    checkpoints = sorted(glob(f"{os.path.dirname(cfg.TRAIN.CHECKPOINT)}/*model_checkpoint_*[0-9]*"),
                         key=get_checkpoint_number)

    checkpoint_start = int(os.path.basename(cfg.TRAIN.CHECKPOINT).split('_')[-1])
    checkpoint_end = checkpoints[-1]

    split_datasets = False
    if len(samples) > 1:
        split_datasets = True

    if checkpoint_start == checkpoint_end:
        print("User wants to run on the latest checkpoint"
              " but we are going to check if we have multiple files to inference on and then spawn accordingly")
        if not split_datasets:
            # call predicter.py instead, it is equipped to run one or more datasets, but cannot change the gpu
            # requires refactoring and merging with this
            subprocess.run(["python", "./predicter.py"])
    else:
        if split_datasets:
            samples_per_gpu = len(samples) // len(device_count)
        else:
            samples_per_gpu = len(samples)

        checkpoints_per_gpu = len(checkpoints) // len(device_count)
        sample_chunks = [samples[i:i + samples_per_gpu] for i in range(0, len(samples), samples_per_gpu)]
        checkpoint_chunks = [checkpoints[i:i + checkpoints_per_gpu] for i in
                             range(0, len(checkpoints), checkpoints_per_gpu)]

        processes = []

        for i, gpu_id in enumerate(device_count):
            process = multiprocessing.Process(target=call_predicter,
                                              args=(sample_chunks[0][i], checkpoint_chunks[0][i], gpu_id, out_filepath))
            processes.append(process)
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()
