from yacs.config import CfgNode as CN
import math
import numpy as np

try:
    import torch
except Exception as e:
    raise ModuleNotFoundError

_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 1
# Number of workers for doing things, may not be used in this context
_C.SYSTEM.NUM_WORKERS = 10
_C.SYSTEM.CACHE_SIZE = 40
_C.SYSTEM.VERBOSE = True

_C.DATA = CN()
_C.DATA.HOME = "/media/samia/DATA/ark"  # options: /home; /local/path/till/connexion
_C.DATA.DATA_DIR_PATH = "connexion/data"  # where the code resides and data should too
_C.DATA.BRAIN_VOL = "OCTO"  # datasets, options: HEMI;OCTO;SEYMOUR;LUCCHI;CREMI; expand this to load multiple datasets
_C.DATA.TRAIN_TEST_SPLIT = 1  # TODO splits: 1 = all volumes used to train
_C.DATA.FIB = 1  # Means FIBSEM isotropic data
_C.DATA.DIM_2D = False  # TODO: Data preprocessing functionality here
# prediction specific
_C.DATA.OUTFILE = f"{_C.DATA.HOME}/lsd_outputs"
_C.DATA.INVERT_PRED_AFFS = False

# we copy exactly the trainer config, but switch these off
if _C.DATA.DIM_2D:
    _C.PREPROCESS = CN()
    # creates 2D zarrs from 3D zarrs; 3D zarr files must be placed at the right path
    _C.PREPROCESS.EXPORT_2D_FROM_3D = False
    # for this both source and target datasets must exist at the right paths
    _C.PREPROCESS.HISTOGRAM_MATCH = ["HEMI", "OCTO"]
    _C.PREPROCESS.USE_WANDB = False  # we set it here for now

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.NEIGHBORHOOD = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
_C.TRAIN.NEIGHBORHOOD_2D = [[0, -1], [-1, 0]]
# saved as such for easy switching between Short range and Long range neighborhoods
_C.TRAIN.LR_NEIGHBORHOOD = [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [-3, 0, 0], [0, -3, 0], [0, 0, -3], [-9, 0, 0],
                            [0, -9, 0], [0, 0, -9]]
_C.TRAIN.EPOCHS = 400000
_C.TRAIN.SAVE_EVERY = 2000
# if gpu is found trains on 1 gpu else falls back to cpu, can be explicit here like '
_C.TRAIN.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_C.TRAIN.INITIAL_LR = 0.5e-4
_C.TRAIN.LR_BETAS = (0.95, 0.999)
_C.TRAIN.MODEL_TYPE = "MTLSD"  # options: `MTLSD`, `ACLSD`, `ACRLSD`, `LSD`, `AFF`
# prediction specific
_C.TRAIN.CHECKPOINT = f"{_C.DATA.HOME}/lsd_checkpoints/MTLSD_3D/run-aclsd-together/model_checkpoint_300000"

if _C.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"]:
    _C.TRAIN.CHECKPOINT_AC = f"{_C.DATA.HOME}/lsd_checkpoints/LSD_2D/run-test2/model_checkpoint_8000"
    assert _C.TRAIN.CHECKPOINT_AC is not None or _C.TRAIN.CHECKPOINT_AC != "", \
        "Please provide a checkpoint for your auto-context model!"

# Isotropic model and augmentation hyper-params
_C.MODEL_ISO = CN()
_C.MODEL_ISO.IN_CHANNELS = 1
_C.MODEL_ISO.NUM_FMAPS = 12
_C.MODEL_ISO.NUM_FMAPS_OUT = 12
_C.MODEL_ISO.FMAP_INC_FACTOR = 5
_C.MODEL_ISO.DOWNSAMPLE_FACTORS = [[2, 2, 2], [2, 2, 2], [3, 3, 3]]
_C.MODEL_ISO.DOWNSAMPLE_FACTORS_2D = [[2, 2], [2, 2], [3, 3]]  # 2d counter-parts, we will automate 2d to 3d later
_C.MODEL_ISO.KERNEL_SIZE_DOWN = [
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3]]
_C.MODEL_ISO.KERNEL_SIZE_DOWN_2D = [
    [(3,) * 2, (3,) * 2],
    [(3,) * 2, (3,) * 2],
    [(3,) * 2, (3,) * 2],
    [(3,) * 2, (3,) * 2]]
_C.MODEL_ISO.KERNEL_SIZE_UP = [
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3],
    [(3,) * 3, (3,) * 3]
]
_C.MODEL_ISO.KERNEL_SIZE_UP_2D = [
    [(3,) * 2, (3,) * 2],
    [(3,) * 2, (3,) * 2],
    [(3,) * 2, (3,) * 2]
]
_C.MODEL_ISO.PAD_CONV = 'valid'
_C.MODEL_ISO.CONTROL_POINT_SPACING = (40, 40, 40)
_C.MODEL_ISO.CONTROL_POINT_SPACING_2D = (40, 40)
_C.MODEL_ISO.JITTER_SIGMA = [(0, 2, 2), (2, 2, 2)]
_C.MODEL_ISO.JITTER_SIGMA_2D = [(0, 2), (2, 2)]
_C.MODEL_ISO.ROTATION_INTERVAL = [0, math.pi / 2.0]
_C.MODEL_ISO.PROB_SLIP = [0, 0.1]
_C.MODEL_ISO.PROB_SHIFT = [0, 0.1]
_C.MODEL_ISO.MAX_MISALIGN = [0, 1]
_C.MODEL_ISO.SUBSAMPLE = 8
_C.MODEL_ISO.TRANSPOSE = [0, 1, 2]
_C.MODEL_ISO.TRANSPOSE_2D = [0, 1]
_C.MODEL_ISO.INTENSITYAUG_SCALE_MIN = 0.8
_C.MODEL_ISO.INTENSITYAUG_SCALE_MAX = 1.2
_C.MODEL_ISO.INTENSITYAUG_SHIFT_MIN = -0.2
_C.MODEL_ISO.INTENSITYAUG_SHIFT_MAX = 0.2
_C.MODEL_ISO.GROWBOUNDARY_STEPS = 1  # NUMBER OF VOXELS (NOT IN WORLD UNITS) TO GROW
_C.MODEL_ISO.LSD_SIGMA = 80
_C.MODEL_ISO.LSD_DOWNSAMPLE = 2
# PASSED AS A LIST SINCE THERE ARE TWO SCALE/SHIFT AUGS APPLIED WITH DIFFERENT VALUES
_C.MODEL_ISO.INTENSITYSCALESHIFT_SCALE = [2, 0.5]
_C.MODEL_ISO.INTENSITYSCALESHIFT_SHIFT = [-1, 0.5]
_C.MODEL_ISO.INPUT_SHAPE = (196, 196, 196)  # hemi-octo - (196, 196, 196)
_C.MODEL_ISO.INPUT_SHAPE_2D = (196, 196)  # hemi-octo
# input_shape: Coordinate((172, 172, 172))  # hemi- for gan
_C.MODEL_ISO.OUTPUT_SHAPE = (72, 72, 72)  # hemi-octo (72, 72, 72)
_C.MODEL_ISO.OUTPUT_SHAPE_2D = (72, 72)  # hemi-octo
# output_shape: Coordinate((48, 48, 48))  # hemi-for gan
_C.MODEL_ISO.VOXEL_SIZE = (8, 8, 8)  # (8, 8, 8)
_C.MODEL_ISO.VOXEL_SIZE_2D = (12, 12)
_C.MODEL_ISO.GROW_INPUT = (36, 36, 36)
_C.MODEL_ISO.GROW_INPUT_2D = (36, 36)
# This can problematic when you switch models but forget to change the folderpath.
# If checkpoints exist in this path, gp.torch will try to load the model weights by default
# Hence let's create an umbrella checkpoint folder with model name as subfolder. Add suffixes to customise if needed.
_C.MODEL_ISO.LOG_DIR = f"{_C.DATA.HOME}/lsd_logs/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/log-3dmtlsd-hemi-onlyori"
_C.MODEL_ISO.CKPT_FOLDER = f"{_C.DATA.HOME}/lsd_checkpoints/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/checkpoints-3dmtlsd-hemi-onlyori"
_C.MODEL_ISO.OUTPUT_DIR = f"{_C.DATA.HOME}/lsd_snapshots/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/snapshots_3dmtlsd-hemi-onlyori"
# SPECIAL AUGMENTATION CASE: SET THE PATH TO THE DEFECTS FILE HERE.
_C.MODEL_ISO.DEFECT_AUGMENT = ""

# Anisotropic model and augmentation hyper-params
_C.MODEL_ANISO = CN()
_C.MODEL_ANISO.IN_CHANNELS = 1
_C.MODEL_ANISO.NUM_FMAPS = 12
_C.MODEL_ANISO.FMAP_INC_FACTOR = 5
_C.MODEL_ANISO.DOWNSAMPLE_FACTORS = [[1, 2, 2], [2, 2, 2], [3, 3, 3]]
_C.MODEL_ANISO.DOWNSAMPLE_FACTORS_2D = [[1, 2], [2, 2], [3, 3]]
_C.MODEL_ANISO.NUM_FMAPS_OUT = 12
_C.MODEL_ANISO.KERNEL_SIZE_DOWN = [[(1, 3, 3), (1, 3, 3)], [(3,) * 3, (3,) * 3], [(3,) * 3, (3,) * 3],
                                   [(3,) * 3, (3,) * 3]]
_C.MODEL_ANISO.KERNEL_SIZE_DOWN_2D = [[(1, 3), (1, 3)], [(3,) * 2, (3,) * 2], [(3,) * 2, (3,) * 2],
                                      [(3,) * 2, (3,) * 2]]
_C.MODEL_ANISO.KERNEL_SIZE_UP = [[(3,) * 2, (3,) * 2], [(3,) * 2, (3,) * 2], [(3,) * 2, (3,) * 2]]

_C.MODEL_ANISO.PAD_CONV = 'valid'
_C.MODEL_ANISO.CONTROL_POINT_SPACING = (4, 4, 10)  # xyz
_C.MODEL_ANISO.CONTROL_POINT_SPACING_2D = (4, 4)  # xy
_C.MODEL_ANISO.JITTER_SIGMA = (0, 2, 2)  # zyx
_C.MODEL_ANISO.JITTER_SIGMA_2D = (2, 2)  # xy
_C.MODEL_ANISO.ROTATION_INTERVAL = [0, math.pi / 2.0]
_C.MODEL_ANISO.PROB_SLIP = 0.5  # APPLIED ONCE FOR ANI DATASETS: LSD PAPER
_C.MODEL_ANISO.PROB_SHIFT = 0.5
_C.MODEL_ANISO.MAX_MISALIGN = 10
_C.MODEL_ANISO.SUBSAMPLE = 8
_C.MODEL_ANISO.TRANSPOSE = [1, 2]
_C.MODEL_ANISO.TRANSPOSE_2D = [0, 1]
_C.MODEL_ANISO.INTENSITYAUG_SCALE_MIN = 0.8
_C.MODEL_ANISO.INTENSITYAUG_SCALE_MAX = 1.2
_C.MODEL_ANISO.INTENSITYAUG_SHIFT_MIN = -0.2
_C.MODEL_ANISO.INTENSITYAUG_SHIFT_MAX = 0.2
_C.MODEL_ANISO.GROWBOUNDARY_STEPS = 1  # NUMBER OF VOXELS (NOT IN WORLD UNITS) TO GROW
_C.MODEL_ANISO.LSD_SIGMA = 80
_C.MODEL_ANISO.LSD_DOWNSAMPLE = 2
# PASSED AS A LIST SINCE THERE ARE TWO SCALE/SHIFT AUGS APPLIED WITH DIFFERENT VALUES
_C.MODEL_ANISO.INTENSITYSCALESHIFT_SCALE = [2, 0.5]
_C.MODEL_ANISO.INTENSITYSCALESHIFT_SHIFT = [-1, 0.5]
_C.MODEL_ANISO.INPUT_SHAPE = (132, 268, 268)  # cremi: ZYX
_C.MODEL_ANISO.INPUT_SHAPE_2D = (268, 268)  # cremi
_C.MODEL_ANISO.OUTPUT_SHAPE = (72, 144, 144)  # cremi
_C.MODEL_ANISO.OUTPUT_SHAPE_2D = (144, 144)  # cremi
_C.MODEL_ANISO.VOXEL_SIZE = (40, 4, 4)  # cremi
_C.MODEL_ANISO.VOXEL_SIZE_2D = (4, 4)  # cremi
_C.MODEL_ANISO.GROW_INPUT = (36, 36, 36)
_C.MODEL_ANISO.GROW_INPUT_2D = (36, 36)
_C.MODEL_ANISO.LOG_DIR = f"{_C.DATA.HOME}/lsd_logs/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/log-3dmtlsd-cremi-cropped-z30-170"
_C.MODEL_ANISO.CKPT_FOLDER = f"{_C.DATA.HOME}/lsd_checkpoints/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/checkpoints_3dmtlsd-cremi-cropped-z30-170"
_C.MODEL_ANISO.OUTPUT_DIR = f"{_C.DATA.HOME}/lsd_snapshots/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/snapshots_3dmtlsd-cremi-cropped-z30-170"
# SPECIAL AUGMENTATION CASE: SET THE PATH TO THE DEFECTS FILE HERE.
_C.MODEL_ANISO.DEFECT_AUGMENT = ""

# RUN INSTANCE SEGMENTATION - SUPPORTS WATERZ FOR NOW
_C.INS_SEGMENT = CN()

_C.INS_SEGMENT.THRESHOLDS = list(np.arange(0.35, 0.80, 0.05))


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for this project.
    Copied from YACs documentation"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
