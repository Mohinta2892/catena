from __future__ import annotations
import sys
import os
from glob import glob
from tqdm import tqdm

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

    # make the outfile path here - /basepath/modeltype/2d/checkpoint_name
    data_dir = os.path.join(cfg.DATA.OUTFILE, cfg.TRAIN.MODEL_TYPE,
                            '2d' if cfg.DATA.DIM_2D else '3d',
                            "/".join(cfg.TRAIN.CHECKPOINT.split("/")[-2:]))

    if not os.path.exists(data_dir):
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)

    # run instance segmentation with waterz; read from the out folder
    samples = glob(f"{data_dir}/*.zarr")

    if cfg.DATA.DIM_2D:
        # sample == .zarr
        for sample in samples:
            # .zarr --> volumes/raw/{0}.. volumes/raw/{n}
            num_samples = calculate_min_2d_samples([sample])
            assert num_samples, "Something went wrong.. num_samples cannot be zero," \
                                " it represents num of z-slices in the 2D zarrs."
            cfg.DATA.SAMPLE = sample
            # overwrite in the loop - otherwise will create zarr within zarr
            cfg.DATA.OUTFILE = data_dir
            cfg.DATA.OUTFILE = os.path.join(cfg.DATA.OUTFILE, os.path.basename(cfg.DATA.SAMPLE))

            """ This is a really bad implementation `Friday night blues`
             because the model is reloaded for each sample!!"""
            for n in range(num_samples):
                cfg.DATA.SAMPLE_SLICE = n
                run_waterz(cfg)
    else:
        # we loop through the datasets here and call inference on them.
        # Todo: add daisy support for spawning
        for sample in tqdm(samples):
            cfg.DATA.SAMPLE = sample
            # overwrite in the loop - otherwise will create zarr within zarr
            cfg.DATA.OUTFILE = data_dir
            cfg.DATA.OUTFILE = os.path.join(cfg.DATA.OUTFILE, os.path.basename(cfg.DATA.SAMPLE))

            run_waterz(cfg)
