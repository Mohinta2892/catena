from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# PREDICTION SETTINGS
# -----------------------------------------------------------------------------
_C.PREDICT = CN()
# Path to the model checkpoint (.pt or .pth) to use for inference.
_C.PREDICT.CHECKPOINT = "/cephfs/smohinta/transmitter_checkpoints/RESNET/sylee_neurotrans_cubes_hemi2025/run2/model_checkpoint_50000"
# Directory where the output prediction CSV file will be saved.
_C.PREDICT.OUTPUT_DIR = "./predictions/"
# The name of the final output CSV file.
_C.PREDICT.OUTPUT_FILENAME = "predictions.csv"
# Define the known synapse types for the model's output layer.
# This must match the order and number of classes the model was trained on.
_C.PREDICT.SYNAPSE_TYPES = ['gaba', 'acetylcholine', 'glutamate', 'serotonin', 'octopamine', 'dopamine', 'tyramine'] # must match the num of classes the model was trained on

# -----------------------------------------------------------------------------
# DATA SOURCE (Choose ONE method)
# -----------------------------------------------------------------------------
_C.DATA_SOURCE = CN()
# Method to use for sourcing synapse locations.
# Options: 'pkl', 'directory', 'mongo'
_C.DATA_SOURCE.METHOD = 'pkl'

# --- PKL File Source ---
# Path to the .pkl file containing a dictionary with 'train' and 'val' keys.
# The 'val' key will be used for prediction.
_C.DATA_SOURCE.PKL_FILE_PATH = "/cephfs/smohinta/catena/helpers/neurotransmitter/sylee_neurotrans_cubes_hemi2025/data_3d/train/data_split/data_split.pkl"

# --- Directory Source ---
# Path to a directory containing raw HDF5 files for prediction.
# All .h* files in this directory will be processed.
_C.DATA_SOURCE.DIRECTORY_PATH = "/path/to/unlabeled_data/"

# --- MongoDB Source ---
_C.DATA_SOURCE.MONGO = CN()
_C.DATA_SOURCE.MONGO.DB_NAME = "synister_db"
_C.DATA_SOURCE.MONGO.DB_HOST = "mongodb://localhost:27017/"
# The collection within the DB that holds the synapse documents.
_C.DATA_SOURCE.MONGO.COLLECTION = "synapses"
# Optional: A JSON-formatted query string to filter documents.
# Example: '{"brain_region": " medulla"}'
_C.DATA_SOURCE.MONGO.QUERY = '{}'

# -----------------------------------------------------------------------------
# RAW IMAGE DATA (Required for all source methods)
# -----------------------------------------------------------------------------
_C.RAW_DATA = CN()
# Path to the Zarr or N5 container for the raw image data.
_C.RAW_DATA.CONTAINER = "/path/to/raw_data.zarr"
# The dataset name within the container (e.g., 'volumes/raw/s0').
_C.RAW_DATA.DATASET = "volumes/raw/s0"

# -----------------------------------------------------------------------------
# MODEL (Must match the architecture of the loaded checkpoint)
# -----------------------------------------------------------------------------
_C.MODEL = CN()  # Change for consistency
# The model architecture to use. Options: 'VGG' or 'RESNET'
_C.MODEL.TYPE = 'RESNET' # Change for consistency

_C.MODEL.VGG = CN()
_C.MODEL.VGG.INPUT_SIZE = (64, 64, 64)
_C.MODEL.VGG.FMAPS = 16
_C.MODEL.VGG.DOWNSAMPLE_FACTORS = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
_C.MODEL.VGG.FMAP_INC = [2, 2, 2]
_C.MODEL.VGG.N_CONVOLUTIONS = [2, 2, 2]

_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.INPUT_SIZE = (64, 64, 64)
_C.MODEL.RESNET.INPUT_CHANNELS = 1
_C.MODEL.RESNET.START_CHANNELS = 16

_C.SYSTEM = CN()
_C.SYSTEM.NUM_WORKERS = 4
_C.SYSTEM.BATCH_SIZE = 8
_C.SYSTEM.DEVICE = "cuda:0"
_C.SYSTEM.VOXEL_SIZE = (8, 8, 8)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    return _C.clone()


cfg = _C
