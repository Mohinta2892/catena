from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

from yacs.config import CfgNode as CN  # default config

# add current directory to path and allow absolute imports
sys.path.insert(0, '.')

from config.config import get_cfg_defaults
from engine.training.mtlsd.train_all_yacs import train_until
from engine.training.mtlsd.train_2d_all_yacs import train_2d_until
from data_utils.preprocess_volumes.utils import create_data


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
        cfg = get_cfg_defaults()
    # can be used to override pre-defined settings
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

    # if enabled changes to YACS Config beyond this point will throw errors
    # cfg.freeze()

    # make a logs folder in the current dir, this is where `trainer.py` resides.
    # Note: these are all runtime warnings and errors messages, not where the model logs are stored!!
    # TODO: test
    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    # Ensure all folders to save outputs and intermediaries are created
    if not os.path.exists(cfg.MODEL.CKPT_FOLDER):
        os.makedirs(cfg.MODEL.CKPT_FOLDER, exist_ok=False)
    if not os.path.exists(cfg.MODEL.LOG_DIR):
        os.makedirs(cfg.MODEL.LOG_DIR, exist_ok=False)
    if not os.path.exists(cfg.MODEL.OUTPUT_DIR):
        os.makedirs(cfg.MODEL.OUTPUT_DIR, exist_ok=False)

    # sanity check: does the user wish to change the path where checkpoints are saved
    print(f"checkpoint dir: {cfg.MODEL.CKPT_FOLDER}\n"
          f"log dir: {cfg.MODEL.LOG_DIR}\n snapshot dir: {cfg.MODEL.OUTPUT_DIR}")

    # tested for FIBSEM isotropic
    if cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"]:
        if cfg.DATA.DIM_2D:
            cfg.MODEL.INTERMEDIATE_SHAPE = (196, 196)  # we set the intermediate shape here
        else:
            cfg.MODEL.INTERMEDIATE_SHAPE = (196, 196, 196)

    if cfg.DATA.DIM_2D:
        # train the lsd model first and then switch to the aff model
        if cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"] and cfg.TRAIN.LSD_EPOCHS is not None:
            train_2d_until(cfg.TRAIN.LSD_EPOCHS, cfg)
            cfg.TRAIN.LSD_EPOCHS = None
            cfg.MODEL.INPUT_SHAPE = (328, 328)  # set it large, so that from output 204^3 can be cropped 196^3

            # We need to pick the latest checkpoint path; overwrites the config
            cfg.TRAIN.CHECKPOINT_AC = f"{cfg.MODEL.CKPT_FOLDER}/model_checkpoint_latest"
            # Warning: `hardcoded`. Save the affinity checkpoints under AFF folder.
            cfg.MODEL.CKPT_FOLDER = f"{str(Path(cfg.MODEL.CKPT_FOLDER).resolve().parents[1])}/AFF/{os.path.basename(cfg.MODEL.CKPT_FOLDER)}"
        train_2d_until(cfg.TRAIN.EPOCHS, cfg)

    else:
        # Log training start
        if cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"] and cfg.TRAIN.LSD_EPOCHS is not None:
            train_until(cfg.TRAIN.LSD_EPOCHS, cfg)
            cfg.TRAIN.LSD_EPOCHS = None
            cfg.MODEL.INPUT_SHAPE = (328, 328, 328)  # tested for FIBSEM isotropic data

            # We need to pick the latest checkpoint path; overwrites the config
            cfg.TRAIN.CHECKPOINT_AC = f"{cfg.MODEL.CKPT_FOLDER}/model_checkpoint_latest"
            # Warning: `hardcoded`. Save the affinity checkpoints under AFF folder.
            cfg.MODEL.CKPT_FOLDER = f"{str(Path(cfg.MODEL.CKPT_FOLDER).resolve().parents[1])}/AFF/{os.path.basename(cfg.MODEL.CKPT_FOLDER)}"
        train_until(cfg.TRAIN.EPOCHS, cfg)
