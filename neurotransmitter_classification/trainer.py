import argparse
import logging
import os
import sys

from yacs.config import CfgNode as CN

# Add current directory to path to allow absolute imports
sys.path.insert(0, '.')

from config.config import get_cfg_defaults
from engine.train.train_3d import train_until


def main():
    """
    Main function to run the training pipeline.
    Parses arguments, loads configuration, and starts training.
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Train a 3D classification model for neurotransmitters.")
    parser.add_argument(
        "-c",
        "--config_file",
        default=None,
        help="Path to a custom YAML configuration file to override defaults.",
    )
    args = parser.parse_args()

    # --- Load configuration ---
    cfg = get_cfg_defaults()
    if args.config_file is not None:
        logging.info(f"Merging config from {args.config_file}")
        cfg.merge_from_file(args.config_file)
        # We need to update paths that depend on MODEL_TYPE after merge
        cfg.LOGGING.LOG_DIR = f"{cfg.LOGGING.LOG_DIR}/{cfg.TRAIN.MODEL_TYPE}"
        cfg.LOGGING.SNAPSHOT_DIR = f"{cfg.LOGGING.SNAPSHOT_DIR}/{cfg.TRAIN.MODEL_TYPE}"
        cfg.LOGGING.CKPT_DIR = f"{cfg.LOGGING.CKPT_DIR}/{cfg.TRAIN.MODEL_TYPE}"

    # --- Setup output directories ---
    cfg.LOGGING.LOG_DIR = os.path.join(cfg.LOGGING.LOG_DIR, cfg.DATA.BRAIN_VOL,
                                       cfg.LOGGING.RUN_NAME)
    cfg.LOGGING.SNAPSHOT_DIR = os.path.join(cfg.LOGGING.SNAPSHOT_DIR, cfg.DATA.BRAIN_VOL,
                                            cfg.LOGGING.RUN_NAME)
    cfg.LOGGING.CKPT_DIR = os.path.join(cfg.LOGGING.CKPT_DIR, cfg.DATA.BRAIN_VOL,
                                        cfg.LOGGING.RUN_NAME)

    if not os.path.exists(cfg.LOGGING.LOG_DIR):
        os.makedirs(cfg.LOGGING.LOG_DIR, exist_ok=True)
        logging.info(f"Created log directory: {cfg.LOGGING.LOG_DIR}")
    if not os.path.exists(cfg.LOGGING.SNAPSHOT_DIR):
        os.makedirs(cfg.LOGGING.SNAPSHOT_DIR, exist_ok=True)
        logging.info(f"Created snapshot directory: {cfg.LOGGING.SNAPSHOT_DIR}")
    if not os.path.exists(cfg.LOGGING.CKPT_DIR):
        os.makedirs(cfg.LOGGING.CKPT_DIR, exist_ok=True)
        logging.info(f"Created checkpoint directory: {cfg.LOGGING.CKPT_DIR}")

    # --- Start Training ---
    logging.info(f"Starting training for model: {cfg.TRAIN.MODEL_TYPE}")
    logging.info(f"Full configuration:\n{cfg}")

    try:
        train_until(cfg)
    except Exception as e:
        logging.exception(e)
        raise


if __name__ == "__main__":
    main()
