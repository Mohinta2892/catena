from __future__ import annotations
import sys
import os
from pathlib import Path

# add current directory to path and allow absolute imports
sys.path.insert(0, '.')

from config.config import get_cfg_defaults
from engine.training.train_3d import train_until


def rename_keys(original_config, key_mapping):
    for new_key, old_key in key_mapping.items():
        if hasattr(original_config, old_key):
            original_config[new_key] = getattr(original_config, old_key)

    return original_config


if __name__ == '__main__':
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

    # changes to YACS Config beyond this point will throw errors
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

    # adapt this to Autocontext setups for Synful
    # if cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"]:
    #     if cfg.DATA.DIM_2D:
    #         cfg.MODEL.INTERMEDIATE_SHAPE = (196, 196)  # we set the intermediate shape here
    #     else:
    #         cfg.MODEL.INTERMEDIATE_SHAPE = (196, 196, 196)

    # if cfg.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"] and cfg.TRAIN.LSD_EPOCHS is not None:
    #     train_until(cfg.TRAIN.LSD_EPOCHS, cfg)
    #     cfg.TRAIN.LSD_EPOCHS = None
    #     # We need to pick the latest checkpoint path; overwrites the config
    #     cfg.TRAIN.CHECKPOINT_AC = f"{cfg.MODEL.CKPT_FOLDER}/model_checkpoint_latest"
    #     cfg.MODEL.CKPT_FOLDER = f"{str(Path(cfg.MODEL.CKPT_FOLDER).resolve().parents[1])}/AFF"

    # either single task or multitask now, ac setups will come later
    train_until(cfg)
