from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# SYSTEM
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.CACHE_SIZE = 20

# -----------------------------------------------------------------------------
# DATA
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Base path for the dataset
_C.DATA.HOME = "/cephfs/smohinta"
# Relative path to the specific dataset directory
_C.DATA.DATA_DIR_PATH = "catena/helpers/neurotransmitter"
# Name of the brain volume to use
_C.DATA.BRAIN_VOL = "sylee_neurotrans_cubes_hemi2025"
# Voxel size in nm (Z, Y, X)
_C.DATA.VOXEL_SIZE = (8, 8, 8)
# --- Data Splitting ---
# Set to True to load data from a pre-computed split file.
# If False, all data in DATA_DIR_PATH will be used for training.
_C.DATA.USE_SPLIT_FILE = True
# Path to the pickle file containing the train/test file split.
# Only used if USE_SPLIT_FILE is True.
_C.DATA.SPLIT_FILE = "/cephfs/smohinta/catena/helpers/neurotransmitter/sylee_neurotrans_cubes_hemi2025/data_3d/train/data_split/data_split.pkl"

# -----------------------------------------------------------------------------
# TRAINING
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
# The model architecture to use. Options: 'VGG' or 'RESNET'
_C.TRAIN.MODEL_TYPE = 'RESNET'
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.EPOCHS = 50000
_C.TRAIN.SAVE_EVERY = 5000
_C.TRAIN.SNAPSHOT_EVERY = 5000
_C.TRAIN.DEVICE = "cuda:0"
_C.TRAIN.INITIAL_LR = 1e-4


# -----------------------------------------------------------------------------
# LOGGING
# -----------------------------------------------------------------------------
_C.LOGGING = CN()
_C.LOGGING.LOG_DIR = f"{_C.DATA.HOME}/transmitter_logs"
_C.LOGGING.SNAPSHOT_DIR = f"{_C.DATA.HOME}/transmitter_snapshots"
_C.LOGGING.CKPT_DIR = f"{_C.DATA.HOME}/transmitter_checkpoints"
_C.LOGGING.RUN_NAME = f"run2"


# -----------------------------------------------------------------------------
# VGG MODEL CONFIGURATION
# -----------------------------------------------------------------------------
_C.MODEL_VGG = CN()
# Input patch size in voxels (Z, Y, X)
_C.MODEL_VGG.INPUT_SIZE = (64, 64, 64)
_C.MODEL_VGG.FMAPS = 16
# e.g., [(2,2,2), (2,2,2), (2,2,2)]
_C.MODEL_VGG.DOWNSAMPLE_FACTORS = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
# Multiplier for feature maps at each level
_C.MODEL_VGG.FMAP_INC = [2, 2, 2]
# Number of convolutions per level
_C.MODEL_VGG.N_CONVOLUTIONS = [2, 2, 2]


# -----------------------------------------------------------------------------
# RESNET MODEL CONFIGURATION
# -----------------------------------------------------------------------------
_C.MODEL_RESNET = CN()
# Input patch size in voxels (Z, Y, X)
_C.MODEL_RESNET.INPUT_SIZE = (64, 64, 64)
_C.MODEL_RESNET.INPUT_CHANNELS = 1
_C.MODEL_RESNET.START_CHANNELS = 16


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    return _C.clone()

cfg = _C
