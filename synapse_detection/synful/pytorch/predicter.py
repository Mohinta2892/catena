from __future__ import annotations

import glob
import sys
import os
from pathlib import Path
from yacs.config import CfgNode as CN  # default config
import argparse

# add current directory to path and allow absolute imports
sys.path.insert(0, '.')

from config.config_predict import get_cfg_defaults
from engine.predict.predict_3d import predict


def rename_keys(original_config, key_mapping):
    for new_key, old_key in key_mapping.items():
        if hasattr(original_config, old_key):
            original_config[new_key] = getattr(original_config, old_key)

    return original_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser("You can pass an explicit config file to train.")
    parser.add_argument('-c', default=None, help='Pass the config file"!')
    args = parser.parse_args()
    config_file = args.c
    if config_file is not None:
        # parse the args file to become cfg
        cfg = CN()
        # Allow creating new keys recursively.: https://github.com/rbgirshick/yacs/issues/25
        cfg.set_new_allowed(True)
        cfg.merge_from_file(config_file)
    else:
        cfg = get_cfg_defaults()  # can be used to override pre-defined settings

    # TODO test with explicit path for example setups: ssTEM CREMI, FIBSEM: Hemibrain
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

    # changes to YACS Config beyond this point will throw errors
    # cfg.freeze()

    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL)
    # data is expected to be here
    if cfg.DATA.DIM_2D:
        data_dir = os.path.join(data_dir, 'data_2d', 'test')
    else:
        data_dir = os.path.join(data_dir, 'data_3d', 'test')

    # TODO: add the logger here and import the same logging file
    # Follow: https://stackoverflow.com/questions/43947206/automatically-delete-old-python-log-files
    # logger.debug(f"data_dir {data_dir}")
    try:
        samples = glob.glob(f"{data_dir}/*.h*") + glob.glob(f"{data_dir}/*.zarr")
    except Exception as e:
        print(e)

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
    # # TODO: batch inference could make it faster.
    # # with batchnorm we can no longer do this here, we have to initialise the model first with batch size
    if cfg.TRAIN.BATCH_SIZE > 1:
        logging.warning("If you have trained your models with Batch_Size > 1, comment this whole `if` block."
                        "This ensures you can load the model but the inference will still proceed"
                        " with batch_size=1.")
        cfg.TRAIN.BATCH_SIZE = 1

    for sample in samples:
        cfg.DATA.SAMPLE = sample
        # overwrite in the loop - otherwise will create zarr within zarr
        cfg.DATA.OUTFILE = out_filepath
        cfg.DATA.OUTFILE = os.path.join(cfg.DATA.OUTFILE, os.path.basename(cfg.DATA.SAMPLE))

        predict(cfg)
