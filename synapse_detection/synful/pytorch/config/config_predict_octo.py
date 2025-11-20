from yacs.config import CfgNode as CN
import math

# Switch this off when inferring blockwise with `super_predicter_daisy.py`
# try:
#     import torch
# except Exception as e:
#     raise ModuleNotFoundError

_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 1
# Number of workers for doing things, may not be used in this context
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.CACHE_SIZE = 4
_C.SYSTEM.VERBOSE = True

_C.DATA = CN()
_C.DATA.HOME = "/media/samia/DATA"  # options: /home; /media/samia/DATA/ark; this must be a mounted directory such that logs are written to local
_C.DATA.DATA_DIR_PATH = "PhD/dataset/"  # where the code resides and data should too; connexion/data
_C.DATA.BRAIN_VOL = "octo_3cubes_zyx"  # datasets, options: HEMI;OCTO;SEYMOUR;LUCCHI;CREMI; expand this to load multiple datasets
_C.DATA.TRAIN_TEST_SPLIT = 1  # TODO splits: 1 = all volumes used to train
_C.DATA.FIB = 1  # Is FIBSEM isotropic data
_C.DATA.DIM_2D = False  # TODO: Data preprocessing functionality here
_C.DATA.WITH_MITO = 1  # `0: NO MITO, 1: MITO ONLY, 2: MITO + LSD + AFF (MTLSDMITO)`
# specify the datasets inside zarr - UNUSED for now!!
_C.DATA.NEURON_LABELS = "volumes/labels/neuron_ids"
_C.DATA.MITO_LABELS = "volumes/labels/mito_ids"
_C.DATA.RAW = "volumes/raw"
_C.DATA.NEURON_LABELS_MASK = "volumes/labels/labels_mask"
_C.DATA.MITO_LABELS_MASK = "volumes/labels/labels_mask_mito"
# _C.DATA.SAMPLE = ''  # add these NULL keys helps to merge with yaml later; on-the-fly created in predicter(s).py
# _C.DATA.SAMPLE_SLICE = ''  # add these NULL keys helps to merge with yaml later; on-the-fly created in predicter(s).py

# we currently run this explicitly via `preprocess_data.py`. Todo: integrate to trainer, maybe.
_C.PREPROCESS = CN()
if _C.DATA.DIM_2D:
    # creates 2D zarrs from 3D zarrs; 3D zarr files must be placed at the right path
    _C.PREPROCESS.EXPORT_2D_FROM_3D = True
    _C.PREPROCESS.SOURCE_DATA_OFFSET = (0, 0)
    _C.PREPROCESS.SOURCE_DATA_RESOLUTION = (8, 8)
    # for this both source and target datasets must exist at the right paths
    _C.PREPROCESS.HISTOGRAM_MATCH = None  # '["HEMI" , "OCTO"]
    _C.PREPROCESS.USE_WANDB = True  # we set it here for now
else:
    _C.PREPROCESS.USE_WANDB = False  # we set it here for now
    _C.PREPROCESS.SOURCE_DATA_OFFSET = (0, 0, 0)
    _C.PREPROCESS.SOURCE_DATA_RESOLUTION = (8, 8, 8)
    _C.PREPROCESS.HISTOGRAM_MATCH = None  # ["HEMI", "OCTO"]

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 1
_C.TRAIN.NEIGHBORHOOD = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
_C.TRAIN.NEIGHBORHOOD_2D = [[0, -1], [-1, 0]]
# saved as such for easy switching between Short range and Long range neighborhoods
_C.TRAIN.LR_NEIGHBORHOOD = [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [-3, 0, 0], [0, -3, 0], [0, 0, -3], [-9, 0, 0],
                            [0, -9, 0], [0, 0, -9]]
_C.TRAIN.EPOCHS = 300000
_C.TRAIN.SAVE_EVERY = 5000
# if gpu is found trains on 1 gpu else falls back to cpu
_C.TRAIN.DEVICE = "cpu"  # options `cuda:0`, `multi_gpu`, `cpu`
_C.TRAIN.INITIAL_LR = 0.5e-4
_C.TRAIN.LR_BETAS = (0.95, 0.999)
_C.TRAIN.MODEL_TYPE = "SynMT1"  # options: `STMASK`, `STVEC`, `SynMT1`, `ACMASK`, `ACVEC`
_C.TRAIN.ON_SPECIFIC_LOCS = True  # options: `False`, `True`

# prediction specific
_C.TRAIN.CHECKPOINT = "/media/samia/DATA/PhD/syn_checkpoints/SynMT1_3D/run_syn_octo_lossbal/model_checkpoint_300000"  # Path to checkpoint
_C.DATA.OUTFILE = f"{_C.DATA.HOME}/catena/synful_outputs"
# # MongoDB
_C.DATA.DB_NAME = 'synful_predictions'  # this is default
_C.DATA.DB_HOST = 'localhost:27017'  # mongo beast
_C.DATA.DROP_DS_MONGOTABLE = False  # preserves both the dataset and the Mongo Collection
if not _C.TRAIN.CHECKPOINT:
    raise Exception("Path to checkpoint is empty!")

if _C.TRAIN.MODEL_TYPE in ["ACLSD", "ACRLSD"]:
    _C.TRAIN.AC_EPOCHS = 300000  # set this train lsd first then, default None
else:
    _C.TRAIN.AC_EPOCHS = None

if _C.TRAIN.AC_EPOCHS is None:
    _C.TRAIN.CHECKPOINT_AC = "/net/ark/scratch/smohinta/connexion/lsd_checkpoints/LSD_2D/run-test2/model_checkpoint_8000"
    assert _C.TRAIN.CHECKPOINT_AC is not None or _C.TRAIN.CHECKPOINT_AC != "", \
        "Please provide a checkpoint for your auto-context model!"

_C.TRAIN.AUGMENT = False  # should we augment

# Isotropic model and augmentation hyper-params
_C.MODEL_ISO = CN()
_C.MODEL_ISO.IN_CHANNELS = 1
_C.MODEL_ISO.LSDS = 6 if _C.DATA.DIM_2D else 10  # num of lsd features
_C.MODEL_ISO.NUM_FMAPS = 12
_C.MODEL_ISO.NUM_FMAPS_OUT = 12
_C.MODEL_ISO.FMAP_INC_FACTOR = 5
# All models except same convolved aclsd--[[2, 2, 2], [2, 2, 2], [3, 3, 3]] if input shape is 208^3|184^3|168^3
# expected downsampling for this is: [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
_C.MODEL_ISO.DOWNSAMPLE_FACTORS = [[2, 2, 2], [2, 2, 2]]  # [3, 3, 3]]
# _C.MODEL_ISO.DOWNSAMPLE_FACTORS = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
# _C.MODEL_ISO.DOWNSAMPLE_FACTORS = [[2, 2, 2], [2,2,2], [2, 2, 2]]
# _C.MODEL_ISO.DOWNSAMPLE_FACTORS = [[2, 2, 2], [2, 2, 2]]
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
_C.MODEL_ISO.BLOB_RADIUS = 200  # 10
_C.MODEL_ISO.BLOB_MODE = "ball"
_C.MODEL_ISO.D_SCALE = 1  # has something to do with intensity adjustment
_C.MODEL_ISO.M_LOSS_SCALE = 1  # 1
_C.MODEL_ISO.D_LOSS_SCALE = 0.0001  # 1
_C.MODEL_ISO.D_BLOB_RADIUS = 150  # 100  # 150
_C.MODEL_ISO.CLIPRANGE = [7e-4, 0.9993]
_C.MODEL_ISO.REJECT_PROBABILITY = 0.95  # 0.7
# PASSED AS A LIST SINCE THERE ARE TWO SCALE/SHIFT AUGS APPLIED WITH DIFFERENT VALUES
_C.MODEL_ISO.INTENSITYSCALESHIFT_SCALE = [2, 0.5]
_C.MODEL_ISO.INTENSITYSCALESHIFT_SHIFT = [-1, 0.5]
# _C.MODEL_ISO.INPUT_SHAPE = (256, 256, 256)  # hemi-octo -  (196, 196, 196) 1800 x 3760 x 3760
_C.MODEL_ISO.INPUT_SHAPE = (64, 64, 64)  # hemi-octo -  (196, 196, 196) 1800 x 3760 x 3760
_C.MODEL_ISO.INPUT_SHAPE_2D = (196, 196)  # hemi-octo
# input_shape: Coordinate((172, 172, 172))  # hemi- for gan
# _C.MODEL_ISO.OUTPUT_SHAPE = (132, 132, 132)  # hemi-octo - in: (196,)^3, out: (72,^3) with valid conv; in:(208,)^3, out:(84,)^3: 8108
# _C.MODEL_ISO.OUTPUT_SHAPE = (132, 132, 132)
# _C.MODEL_ISO.OUTPUT_SHAPE = (216, 216, 216)
_C.MODEL_ISO.OUTPUT_SHAPE = (24, 24, 24)

_C.MODEL_ISO.OUTPUT_SHAPE_2D = (72, 72)  # hemi-octo
# output_shape: Coordinate((48, 48, 48))  # hemi-for gan
_C.MODEL_ISO.VOXEL_SIZE = (8, 8, 8)
_C.MODEL_ISO.VOXEL_SIZE_2D = (8, 8)
# SPECIAL AUGMENTATION CASE: SET THE PATH TO THE DEFECTS FILE HERE.
_C.MODEL_ISO.DEFECT_AUGMENT = ""

# This can problematic when you switch models but forget to change the folderpath.
# If checkpoints exist in this path, gp.torch will try to load the model weights by default
# Hence let's create an umbrella checkpoint folder with model name as subfolder. Add suffixes to customise if needed.
# _C.MODEL_ISO.LOG_DIR = f"{_C.DATA.HOME}/syn_logs/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/{'LSD' if _C.TRAIN.AC_EPOCHS is not None else ''}/run_syn_octo_cube1"
_C.MODEL_ISO.LOG_DIR = f"/media/samia/DATA/PhD/syn_logs/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/{'LSD' if _C.TRAIN.AC_EPOCHS is not None else ''}/run_syn_octo_lossbal"
_C.MODEL_ISO.CKPT_FOLDER = f"/media/samia/DATA/PhD/syn_checkpoints/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/{'LSD' if _C.TRAIN.AC_EPOCHS is not None else ''}/run_syn_octo_lossbal"
# _C.MODEL_ISO.CKPT_FOLDER = f"{_C.DATA.HOME}/syn_checkpoints/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/{'LSD' if _C.TRAIN.AC_EPOCHS is not None else ''}/run_syn_octo_cube1"
_C.MODEL_ISO.OUTPUT_DIR = f"/media/samia/DATA/PhD/syn_snapshots/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/{'LSD' if _C.TRAIN.AC_EPOCHS is not None else ''}/run_syn_octo_lossbal"
# _C.MODEL_ISO.OUTPUT_DIR = f"{_C.DATA.HOME}/syn_snapshots/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/{'LSD' if _C.TRAIN.AC_EPOCHS is not None else ''}/run_syn_octo_cube1"

# Anisotropic model and augmentation hyper-params
_C.MODEL_ANISO = CN()
_C.MODEL_ANISO.IN_CHANNELS = 1
_C.MODEL_ISO.LSDS = 6  # num of lsd features 6 == 2D; 10 == 3D
_C.MODEL_ANISO.NUM_FMAPS = 12
_C.MODEL_ANISO.FMAP_INC_FACTOR = 5
_C.MODEL_ANISO.DOWNSAMPLE_FACTORS = [[1, 3, 3], [1, 3, 3], [2, 2, 2]]
_C.MODEL_ANISO.DOWNSAMPLE_FACTORS_2D = [[1, 2], [2, 2], [3, 3]]
_C.MODEL_ANISO.NUM_FMAPS_OUT = 12
_C.MODEL_ANISO.KERNEL_SIZE_DOWN = [[(1, 3, 3), (1, 3, 3)], [(3,) * 3, (3,) * 3], [(3,) * 3, (3,) * 3],
                                   [(3,) * 3, (3,) * 3]]
_C.MODEL_ANISO.KERNEL_SIZE_DOWN_2D = [[(1, 3), (1, 3)], [(3,) * 2, (3,) * 2], [(3,) * 2, (3,) * 2],
                                      [(3,) * 2, (3,) * 2]]
_C.MODEL_ANISO.KERNEL_SIZE_UP = [[(3,) * 3, (3,) * 3], [(3,) * 3, (3,) * 3], [(3,) * 3, (3,) * 3]]
_C.MODEL_ANISO.KERNEL_SIZE_UP_2D = [[(3,) * 2, (3,) * 2], [(3,) * 2, (3,) * 2], [(3,) * 2, (3,) * 2]]

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
_C.MODEL_ANISO.BLOB_RADIUS = 10
_C.MODEL_ANISO.BLOB_MODE = "ball"
_C.MODEL_ANISO.D_SCALE = 1
_C.MODEL_ANISO.D_BLOB_RADIUS = 100
_C.MODEL_ANISO.CLIPRANGE = [7e-4, 0.9993]
_C.MODEL_ANISO.REJECT_PROBABILITY = 0.95
# PASSED AS A LIST SINCE THERE ARE TWO SCALE/SHIFT AUGS APPLIED WITH DIFFERENT VALUES
_C.MODEL_ANISO.INTENSITYSCALESHIFT_SCALE = [2, 0.5]
_C.MODEL_ANISO.INTENSITYSCALESHIFT_SHIFT = [-1, 0.5]
_C.MODEL_ANISO.INPUT_SHAPE = (96, 430, 430)  # cremi: ZYX
_C.MODEL_ANISO.INPUT_SHAPE_2D = (268, 268)  # cremi
_C.MODEL_ANISO.OUTPUT_SHAPE = (68, 234, 234)  # cremi
_C.MODEL_ANISO.OUTPUT_SHAPE_2D = (144, 144)  # cremi
_C.MODEL_ANISO.VOXEL_SIZE = (40, 4, 4)  # cremi
_C.MODEL_ANISO.VOXEL_SIZE_2D = (4, 4)  # cremi
_C.MODEL_ANISO.LOG_DIR = f"{_C.DATA.HOME}/syn_logs/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/{'LSD' if _C.TRAIN.AC_EPOCHS is not None else ''}/run_1"
_C.MODEL_ANISO.CKPT_FOLDER = f"{_C.DATA.HOME}/syn_checkpoints/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/{'LSD' if _C.TRAIN.AC_EPOCHS is not None else ''}/run_1"
_C.MODEL_ANISO.OUTPUT_DIR = f"{_C.DATA.HOME}/syn_snapshots/{_C.TRAIN.MODEL_TYPE}_{'2D' if _C.DATA.DIM_2D else '3D'}/{'LSD' if _C.TRAIN.AC_EPOCHS is not None else ''}/run_1"
# SPECIAL AUGMENTATION CASE: SET THE PATH TO THE DEFECTS FILE HERE.
_C.MODEL_ANISO.DEFECT_AUGMENT = ""

# Prediction specific
_C.PREDICT = CN()
_C.PREDICT.SYNAPSE_CONTEXT = 40
_C.PREDICT.EXTRACT_TYPE = 'cc'
_C.PREDICT.CC_THRESHOLD = 0.95
_C.PREDICT.LOC_TYPE = 'edt'
_C.PREDICT.SCORE_THRESHOLD = 100
_C.PREDICT.SCORE_TYPE = 'sum'
_C.PREDICT.NMS_RADIUS = None


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for this project.
    Copied from YACs documentation"""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


# Alternatively, provide a way to import the defaults as
# a global singleton:
cfg = _C  # users can `from config import cfg`
