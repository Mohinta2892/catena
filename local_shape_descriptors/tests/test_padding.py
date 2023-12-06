import sys
import os
from glob import glob
import numpy as np

# add current directory to path and allow absolute imports
sys.path.insert(0, '../')
from data_utils.preprocess_volumes.pad_input import pad_input
from config.config_predict import get_cfg_defaults
from data_utils.preprocess_volumes.utils import read_zarr, list_keys


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

    data_dir = os.path.join(cfg.DATA.HOME, cfg.DATA.DATA_DIR_PATH, cfg.DATA.BRAIN_VOL)
    # data is expected to be here
    if cfg.DATA.DIM_2D:
        data_dir = os.path.join(data_dir, 'data_2d', 'test')
    else:
        data_dir = os.path.join(data_dir, 'data_3d', 'test')

    # test with loading a zarr
    samples = glob(f"{data_dir}/*.zarr")

    for sample in samples:
        z = read_zarr(sample, mode='a')
        # expected that this zarr has raw array stored in the dataset `volumes/raw`
        datasets = list_keys(z)

        assert "volumes/raw" in datasets or "/volumes/raw" in datasets, \
            "`volumes/raw` does not exist in your input zarr!!"

        input_raw = z["volumes/raw"]
        padded_input_arr = pad_input(input_raw, cfg.MODEL.VOXEL_SIZE)
        # make a backup of the original, then overwrite with the padded version
        # this should happen only if the padding is needed!
        z["volumes/raw_original"] = z["volumes/raw"][:]
        offset = z["volumes/raw"].attrs["offset"]
        resolution = z["volumes/raw"].attrs["resolution"]
        z["volumes/raw_original"].attrs["offset"] = offset
        z["volumes/raw_original"].attrs["resolution"] = resolution
        z["volumes/raw"] = padded_input_arr
        z["volumes/raw"].attrs["offset"] = offset
        z["volumes/raw"].attrs["resolution"] = resolution

    # test with numpy array that is already a multiple of voxel_size
    # input_raw = np.zeros((512, 512, 512))
    # pad_input(input_raw, voxel_size=(8, 8, 8))
