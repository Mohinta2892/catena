from __future__ import annotations
import sys
import os

# add current directory to path and allow absolute imports
sys.path.insert(0, '.')
from config.config_predict import *
from engine.predict.predict_3d import predict
from engine.predict.predict_2d_all_yacs import predict_2d
from engine.post.run_waterz import run_waterz
from data_utils.preprocess_volumes.utils import calculate_min_2d_samples
from glob import glob


def rename_keys(original_config, key_mapping):
    for new_key, old_key in key_mapping.items():
        if hasattr(original_config, old_key):
            original_config[new_key] = getattr(original_config, old_key)

    return original_config


def check_augmentation_values():
    pass


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

    # TODO: add the logger here and import the same logging file
    # Follow: https://stackoverflow.com/questions/43947206/automatically-delete-old-python-log-files
    # logger.debug(f"data_dir {data_dir}")

    samples = glob(f"{data_dir}/*.zarr")

    assert len(samples), \
        "No data to run prediction on found. Check if data is placed under `{brain_vol}/data_{2/3d}/test`"
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

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

    if cfg.DATA.DIM_2D:
        # sample == .zarr
        for sample in samples:
            # .zarr --> volumes/raw/{0}.. volumes/raw/{n}
            num_samples = calculate_min_2d_samples([sample])
            assert num_samples, "Something went wrong.. num_samples cannot be zero," \
                                " it represents num of z-slices in the 2D zarrs."
            cfg.DATA.SAMPLE = sample
            # overwrite in the loop - otherwise will create zarr within zarr
            cfg.DATA.OUTFILE = out_filepath
            cfg.DATA.OUTFILE = os.path.join(cfg.DATA.OUTFILE, os.path.basename(cfg.DATA.SAMPLE))

            """ This is a really bad implementation `Friday night blues`
             because the model is reloaded for each sample!!"""
            for n in range(num_samples):
                cfg.DATA.SAMPLE_SLICE = n
                predict_2d(cfg)

    else:
        # we loop through the datasets here and call inference on them.
        # Todo: add daisy support for spawning
        for sample in samples:
            cfg.DATA.SAMPLE = sample
            # overwrite in the loop - otherwise will create zarr within zarr
            cfg.DATA.OUTFILE = out_filepath
            cfg.DATA.OUTFILE = os.path.join(cfg.DATA.OUTFILE, os.path.basename(cfg.DATA.SAMPLE))

            predict(cfg)
