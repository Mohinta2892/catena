import argparse
import os
import logging
import sys
from yacs.config import CfgNode as CN

# Add the parent directory to the path to allow importing from 'scripts'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from engine.predict.predict_3d import predict


def main():
    """
    Main function to parse arguments and launch the prediction script.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description="Run inference for neurotransmitter classification using a configuration file."
    )

    parser.add_argument('-c', '--config_file', type=str,
                        help='Path to the prediction config file (e.g., config_predict.py).')

    args = parser.parse_args()

    # --- Load Config ---
    try:
        # Assumes the config file can be loaded and exposes a 'cfg' object
        from importlib.machinery import SourceFileLoader
        config_module = SourceFileLoader('config_predict', args.config_file).load_module()
        cfg = config_module.cfg
    except Exception as e:
        logging.error(f"Failed to load config file {args.config_file}: {e}")
        return

    cfg.freeze()

    try:
        predict(cfg)
    except Exception as e:
        logging.exception("Prediction failed.")
        raise e


if __name__ == '__main__':
    main()
