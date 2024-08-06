"""
Generates a predict config files in the config folder.
Developed to resolve: Issue #21
"""
import argparse

from yacs.config import CfgNode as CN
import sys
import os

# add current directory to path and allow absolute imports
sys.path.insert(0, '.')


def save_cfg(cfg, filename):
    with open(filename, 'w') as f:
        f.write('from yacs.config import CfgNode as CN\n')
        f.write('import math\n')
        f.write('import numpy as np\n\n')
        f.write('# Switch this off when inferring blockwise with `super_predicter_daisy.py`\n')
        f.write('# try:\n')
        f.write('#     import torch\n')
        f.write('# except Exception as e:\n')
        f.write('#     raise ModuleNotFoundError\n\n')
        f.write('_C = CN()\n\n')

        for key, value in cfg.items():
            if isinstance(value, CN):
                f.write(f'_C.{key} = CN()\n')
                for subkey, subvalue in value.items():
                    f.write(f'_C.{key}.{subkey} = {repr(subvalue)}\n')
            else:
                f.write(f'_C.{key} = {repr(value)}\n')

        f.write('\n')
        f.write('def get_cfg_defaults():\n')
        f.write('    """Get a yacs CfgNode object with default values for this project.\n')
        f.write('    Copied from YACs documentation"""\n')
        f.write('    # Return a clone so that the defaults will not be altered\n')
        f.write('    # This is for the "local variable" use pattern\n')
        f.write('    return _C.clone()\n\n')
        f.write('# Alternatively, provide a way to import the defaults as\n')
        f.write('# a global singleton:\n')
        f.write('cfg = _C  # users can `from config import cfg`\n')


def generate_config_predict(cfg_train):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p_cfg', '--predict_cfg', default="./config_predict.py",
    #                     help="Pass the config_predict_template `config_predict.py`. You pass any other too.")
    #
    # args = parser.parse_args()
    predict_config_file = f"{os.getcwd()}/config/config_predict.py"
    if predict_config_file is not None:
        # parse the args file to become cfg
        cfg_predict = CN()
        # Allow creating new keys recursively.: https://github.com/rbgirshick/yacs/issues/25
        cfg_predict.set_new_allowed(True)
        cfg_predict.merge_from_file(predict_config_file)

    else:
        cfg_predict = get_cfg_defaults()

    if cfg_train.DATA.FIB:  # ==1
        cfg_predict.MODEL_ISO.IN_CHANNELS = cfg_train.MODEL_ISO.IN_CHANNELS
        cfg_predict.MODEL_ISO.NUM_FMAPS = cfg_train.MODEL_ISO.NUM_FMAPS
        cfg_predict.MODEL_ISO.NUM_FMAPS_OUT = cfg_train.MODEL_ISO.NUM_FMAPS_OUT
        cfg_predict.MODEL_ISO.FMAP_INC_FACTOR = cfg_train.MODEL_ISO.FMAP_INC_FACTOR
        cfg_predict.MODEL_ISO.DOWNSAMPLE_FACTORS = cfg_train.MODEL_ISO.DOWNSAMPLE_FACTORS
        cfg_predict.MODEL_ISO.DOWNSAMPLE_FACTORS_2D = cfg_train.MODEL_ISO.DOWNSAMPLE_FACTORS_2D
        cfg_predict.MODEL_ISO.KERNEL_SIZE_DOWN = cfg_train.MODEL_ISO.KERNEL_SIZE_DOWN
        cfg_predict.MODEL_ISO.KERNEL_SIZE_DOWN_2D = cfg_train.MODEL_ISO.KERNEL_SIZE_DOWN_2D
        cfg_predict.MODEL_ISO.KERNEL_SIZE_UP = cfg_train.MODEL_ISO.KERNEL_SIZE_UP
        cfg_predict.MODEL_ISO.KERNEL_SIZE_UP_2D = cfg_train.MODEL_ISO.KERNEL_SIZE_UP_2D
        cfg_predict.MODEL_ISO.PAD_CONV = cfg_train.MODEL_ISO.PAD_CONV
        cfg_predict.MODEL_ISO.CONTROL_POINT_SPACING = cfg_train.MODEL_ISO.CONTROL_POINT_SPACING
        cfg_predict.MODEL_ISO.CONTROL_POINT_SPACING_2D = cfg_train.MODEL_ISO.CONTROL_POINT_SPACING_2D
        cfg_predict.MODEL_ISO.JITTER_SIGMA = cfg_train.MODEL_ISO.JITTER_SIGMA
        cfg_predict.MODEL_ISO.JITTER_SIGMA_2D = cfg_train.MODEL_ISO.JITTER_SIGMA_2D
        cfg_predict.MODEL_ISO.ROTATION_INTERVAL = cfg_train.MODEL_ISO.ROTATION_INTERVAL
        cfg_predict.MODEL_ISO.PROB_SLIP = cfg_train.MODEL_ISO.PROB_SLIP
        cfg_predict.MODEL_ISO.PROB_SHIFT = cfg_train.MODEL_ISO.PROB_SHIFT
        cfg_predict.MODEL_ISO.MAX_MISALIGN = cfg_train.MODEL_ISO.MAX_MISALIGN
        cfg_predict.MODEL_ISO.SUBSAMPLE = cfg_train.MODEL_ISO.SUBSAMPLE
        cfg_predict.MODEL_ISO.TRANSPOSE = cfg_train.MODEL_ISO.TRANSPOSE
        cfg_predict.MODEL_ISO.TRANSPOSE_2D = cfg_train.MODEL_ISO.TRANSPOSE_2D
        cfg_predict.MODEL_ISO.INTENSITYAUG_SCALE_MIN = cfg_train.MODEL_ISO.INTENSITYAUG_SCALE_MIN
        cfg_predict.MODEL_ISO.INTENSITYAUG_SCALE_MAX = cfg_train.MODEL_ISO.INTENSITYAUG_SCALE_MAX
        cfg_predict.MODEL_ISO.INTENSITYAUG_SHIFT_MIN = cfg_train.MODEL_ISO.INTENSITYAUG_SHIFT_MIN
        cfg_predict.MODEL_ISO.INTENSITYAUG_SHIFT_MAX = cfg_train.MODEL_ISO.INTENSITYAUG_SHIFT_MAX
        cfg_predict.MODEL_ISO.GROWBOUNDARY_STEPS = cfg_train.MODEL_ISO.GROWBOUNDARY_STEPS
        cfg_predict.MODEL_ISO.LSD_SIGMA = cfg_train.MODEL_ISO.LSD_SIGMA
        cfg_predict.MODEL_ISO.LSD_DOWNSAMPLE = cfg_train.MODEL_ISO.LSD_DOWNSAMPLE
        cfg_predict.MODEL_ISO.INTENSITYSCALESHIFT_SCALE = cfg_train.MODEL_ISO.INTENSITYSCALESHIFT_SCALE
        cfg_predict.MODEL_ISO.INTENSITYSCALESHIFT_SHIFT = cfg_train.MODEL_ISO.INTENSITYSCALESHIFT_SHIFT
        cfg_predict.MODEL_ISO.INPUT_SHAPE = cfg_train.MODEL_ISO.INPUT_SHAPE
        cfg_predict.MODEL_ISO.INPUT_SHAPE_2D = cfg_train.MODEL_ISO.INPUT_SHAPE_2D
        cfg_predict.MODEL_ISO.OUTPUT_SHAPE = cfg_train.MODEL_ISO.OUTPUT_SHAPE
        cfg_predict.MODEL_ISO.OUTPUT_SHAPE_2D = cfg_train.MODEL_ISO.OUTPUT_SHAPE_2D
        cfg_predict.MODEL_ISO.VOXEL_SIZE = cfg_train.MODEL_ISO.VOXEL_SIZE
        cfg_predict.MODEL_ISO.VOXEL_SIZE_2D = cfg_train.MODEL_ISO.VOXEL_SIZE_2D
        cfg_predict.MODEL_ISO.LOG_DIR = cfg_train.MODEL_ISO.LOG_DIR
        cfg_predict.MODEL_ISO.CKPT_FOLDER = cfg_train.MODEL_ISO.CKPT_FOLDER
        cfg_predict.MODEL_ISO.OUTPUT_DIR = cfg_train.MODEL_ISO.OUTPUT_DIR
        cfg_predict.MODEL_ISO.DEFECT_AUGMENT = cfg_train.MODEL_ISO.DEFECT_AUGMENT

    else:
        cfg_predict.MODEL_ANISO.IN_CHANNELS = cfg_train.MODEL_ANISO.IN_CHANNELS
        cfg_predict.MODEL_ANISO.NUM_FMAPS = cfg_train.MODEL_ANISO.NUM_FMAPS
        cfg_predict.MODEL_ANISO.NUM_FMAPS_OUT = cfg_train.MODEL_ANISO.NUM_FMAPS_OUT
        cfg_predict.MODEL_ANISO.FMAP_INC_FACTOR = cfg_train.MODEL_ANISO.FMAP_INC_FACTOR
        cfg_predict.MODEL_ANISO.DOWNSAMPLE_FACTORS = cfg_train.MODEL_ANISO.DOWNSAMPLE_FACTORS
        cfg_predict.MODEL_ANISO.DOWNSAMPLE_FACTORS_2D = cfg_train.MODEL_ANISO.DOWNSAMPLE_FACTORS_2D
        cfg_predict.MODEL_ANISO.KERNEL_SIZE_DOWN = cfg_train.MODEL_ANISO.KERNEL_SIZE_DOWN
        cfg_predict.MODEL_ANISO.KERNEL_SIZE_DOWN_2D = cfg_train.MODEL_ANISO.KERNEL_SIZE_DOWN_2D
        cfg_predict.MODEL_ANISO.KERNEL_SIZE_UP = cfg_train.MODEL_ANISO.KERNEL_SIZE_UP
        cfg_predict.MODEL_ANISO.KERNEL_SIZE_UP_2D = cfg_train.MODEL_ANISO.KERNEL_SIZE_UP_2D
        cfg_predict.MODEL_ANISO.PAD_CONV = cfg_train.MODEL_ANISO.PAD_CONV
        cfg_predict.MODEL_ANISO.CONTROL_POINT_SPACING = cfg_train.MODEL_ANISO.CONTROL_POINT_SPACING
        cfg_predict.MODEL_ANISO.CONTROL_POINT_SPACING_2D = cfg_train.MODEL_ANISO.CONTROL_POINT_SPACING_2D
        cfg_predict.MODEL_ANISO.JITTER_SIGMA = cfg_train.MODEL_ANISO.JITTER_SIGMA
        cfg_predict.MODEL_ANISO.JITTER_SIGMA_2D = cfg_train.MODEL_ANISO.JITTER_SIGMA_2D
        cfg_predict.MODEL_ANISO.ROTATION_INTERVAL = cfg_train.MODEL_ANISO.ROTATION_INTERVAL
        cfg_predict.MODEL_ANISO.PROB_SLIP = cfg_train.MODEL_ANISO.PROB_SLIP
        cfg_predict.MODEL_ANISO.PROB_SHIFT = cfg_train.MODEL_ANISO.PROB_SHIFT
        cfg_predict.MODEL_ANISO.MAX_MISALIGN = cfg_train.MODEL_ANISO.MAX_MISALIGN
        cfg_predict.MODEL_ANISO.SUBSAMPLE = cfg_train.MODEL_ANISO.SUBSAMPLE
        cfg_predict.MODEL_ANISO.TRANSPOSE = cfg_train.MODEL_ANISO.TRANSPOSE
        cfg_predict.MODEL_ANISO.TRANSPOSE_2D = cfg_train.MODEL_ANISO.TRANSPOSE_2D
        cfg_predict.MODEL_ANISO.INTENSITYAUG_SCALE_MIN = cfg_train.MODEL_ANISO.INTENSITYAUG_SCALE_MIN
        cfg_predict.MODEL_ANISO.INTENSITYAUG_SCALE_MAX = cfg_train.MODEL_ANISO.INTENSITYAUG_SCALE_MAX
        cfg_predict.MODEL_ANISO.INTENSITYAUG_SHIFT_MIN = cfg_train.MODEL_ANISO.INTENSITYAUG_SHIFT_MIN
        cfg_predict.MODEL_ANISO.INTENSITYAUG_SHIFT_MAX = cfg_train.MODEL_ANISO.INTENSITYAUG_SHIFT_MAX
        cfg_predict.MODEL_ANISO.GROWBOUNDARY_STEPS = cfg_train.MODEL_ANISO.GROWBOUNDARY_STEPS
        cfg_predict.MODEL_ANISO.LSD_SIGMA = cfg_train.MODEL_ANISO.LSD_SIGMA
        cfg_predict.MODEL_ANISO.LSD_DOWNSAMPLE = cfg_train.MODEL_ANISO.LSD_DOWNSAMPLE
        cfg_predict.MODEL_ANISO.INTENSITYSCALESHIFT_SCALE = cfg_train.MODEL_ANISO.INTENSITYSCALESHIFT_SCALE
        cfg_predict.MODEL_ANISO.INTENSITYSCALESHIFT_SHIFT = cfg_train.MODEL_ANISO.INTENSITYSCALESHIFT_SHIFT
        cfg_predict.MODEL_ANISO.INPUT_SHAPE = cfg_train.MODEL_ANISO.INPUT_SHAPE
        cfg_predict.MODEL_ANISO.INPUT_SHAPE_2D = cfg_train.MODEL_ANISO.INPUT_SHAPE_2D
        cfg_predict.MODEL_ANISO.OUTPUT_SHAPE = cfg_train.MODEL_ANISO.OUTPUT_SHAPE
        cfg_predict.MODEL_ANISO.OUTPUT_SHAPE_2D = cfg_train.MODEL_ANISO.OUTPUT_SHAPE_2D
        cfg_predict.MODEL_ANISO.VOXEL_SIZE = cfg_train.MODEL_ANISO.VOXEL_SIZE
        cfg_predict.MODEL_ANISO.VOXEL_SIZE_2D = cfg_train.MODEL_ANISO.VOXEL_SIZE_2D
        cfg_predict.MODEL_ANISO.LOG_DIR = cfg_train.MODEL_ANISO.LOG_DIR
        cfg_predict.MODEL_ANISO.CKPT_FOLDER = cfg_train.MODEL_ANISO.CKPT_FOLDER
        cfg_predict.MODEL_ANISO.OUTPUT_DIR = cfg_train.MODEL_ANISO.OUTPUT_DIR
        cfg_predict.MODEL_ANISO.DEFECT_AUGMENT = cfg_train.MODEL_ANISO.DEFECT_AUGMENT

    # few other changes
    cfg_predict.TRAIN.CHECKPOINT = '# PUT the CHECKPOINT PATH'  # user must edit this before inference
    cfg_predict.TRAIN.MODEL_TYPE = cfg_train.TRAIN.MODEL_TYPE  # edit the model type with the train one

    cfg_predict.DATA.HOME = cfg_train.DATA.HOME  # options:/home; /local/path/till/connexion
    cfg_predict.DATA.DATA_DIR_PATH = cfg_train.DATA.DATA_DIR_PATH  # where the code resides and data should too
    cfg_predict.DATA.BRAIN_VOL = cfg_train.DATA.BRAIN_VOL  # datasets, options: HEMI;OCTO;SEYMOUR;LUCCHI;CREMI; expand this to load multiple datasets
    cfg_predict.DATA.FIB = cfg_train.DATA.FIB
    cfg_predict.DATA.DIM_2D = cfg_train.DATA.DIM_2D
    cfg_predict.TRAIN.NEIGHBORHOOD = cfg_train.TRAIN.NEIGHBORHOOD
    cfg_predict.TRAIN.NEIGHBORHOOD_2D = cfg_train.TRAIN.NEIGHBORHOOD
    cfg_predict.TRAIN.LR_NEIGHBORHOOD = cfg_train.TRAIN.NEIGHBORHOOD

    # Write the updated configuration to config_predict.py
    filename = f"{os.getcwd()}/config/config_predict_{cfg_train.DATA.BRAIN_VOL.lower()}.py"
    save_cfg(cfg_predict, filename)

    # with open(f'./config_predict_{cfg_train.DATA.BRAIN_VOL.lower()}.py', 'w', encoding="utf-8") as f:
    #     # f.write(cfg_predict.dump())


if __name__ == '__main__':
    generate_config_predict(cfg_train)
