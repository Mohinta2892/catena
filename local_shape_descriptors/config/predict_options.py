"""
Preliminary scripts to diversify the usage of YACS-based config files.
Yet to be thoroughly tested. We appreciate any help with them.

"""

import argparse
import math
import numpy as np

try:
    import torch
except Exception as e:
    raise ModuleNotFoundError


def get_args():
    parser = argparse.ArgumentParser(description='ARGPARSE for YACS Configuration')

    # SYSTEM
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use in the experiment')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for doing things')
    parser.add_argument('--cache_size', type=int, default=40, help='Cache size')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode')

    # DATA
    parser.add_argument('--home', type=str, default="/media/samia/DATA/ark", help='Home directory')
    parser.add_argument('--data_dir_path', type=str, default="connexion/data", help='Data directory path')
    parser.add_argument('--brain_vol', type=str, default="SEYMOUR", help='Brain volume dataset')
    parser.add_argument('--train_test_split', type=int, default=1, help='Train test split')
    parser.add_argument('--fib', type=int, default=1, help='FIBSEM isotropic data')
    parser.add_argument('--dim_2d', action='store_true', help='2D data preprocessing')
    parser.add_argument('--outfile', type=str, default="", help='Output file path')
    parser.add_argument('--invert_pred_affs', action='store_true', help='Invert predicted affinities')

    # PREPROCESS
    parser.add_argument('--export_2d_from_3d', action='store_true', help='Export 2D from 3D')
    parser.add_argument('--histogram_match', nargs='+', default=["HEMI", "OCTO"], help='Histogram match datasets')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb for preprocessing')

    # TRAIN
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--neighborhood', nargs='+', type=int, default=[-1, 0, 0, 0, -1, 0, 0, 0, -1],
                        help='Neighborhood')
    parser.add_argument('--neighborhood_2d', nargs='+', type=int, default=[0, -1, -1, 0], help='2D Neighborhood')
    parser.add_argument('--lr_neighborhood', nargs='+', type=int, default=[-1, 0, 0, 0, -1, 0, 0, 0, -1,
                                                                           -3, 0, 0, 0, -3, 0, 0, 0, -3,
                                                                           -9, 0, 0, 0, -9, 0, 0, 0, -9],
                        help='Long Range Neighborhood')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help='Device (cuda or cpu)')
    parser.add_argument('--model_type', type=str, default="MTLSD", help='Model type')
    parser.add_argument('--checkpoint', type=str, default="", help='Model checkpoint path')

    # Model-specific configurations (Add your model-specific arguments here)

    args = parser.parse_args()
    return args


