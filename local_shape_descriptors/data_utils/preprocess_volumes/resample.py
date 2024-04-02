import os
import sys
from skimage.transform import rescale, resize
import zarr
import numpy as np
from typing import List, Union, Tuple

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data_utils.preprocess_volumes.utils import *
from data_utils.preprocess_volumes.histogram_match import *


def get_zarr_list(dir, ds_keys="volumes/raw"):
    """
    Reads all zarrs in the specified directory and returns a list of numpy arrays representing the images

    Args:
    dir: The directory that contains the images.

    Returns:
    A dict of numpy arrays representing the images.
    """
    if dir[-1] == '/':
        dir = dir[:-1]
    train_raw_path = dir + '/*.*'

    train_raw_filenames = glob(train_raw_path, recursive=True)
    train_raw_filenames.sort()

    train_raw = {}

    for x in train_raw_filenames:
        zarr_data = read_zarr(x)  # Read the Zarr file once
        train_raw[x] = zarr_data[ds_keys][:]

    return train_raw


def save_zarr(out_path: Union[str, Path], hm_sx: dict, offset: tuple,
              resolution: tuple, ds_keys: str = "volumes/raw", is_2d=False):
    """ We save a different file, to preserve the input as is. Do not want to mistakenly overwrite anything!
    """
    for k in hm_sx.keys():
        with zarr.open(os.path.join(out_path, os.path.basename(k)), "a") as z:
            if is_2d:
                for i in range(len(hm_sx[k])):
                    z[f"{ds_keys}/{i}"] = hm_sx[k][i]
                    z[f"{ds_keys}/{i}"].attrs["offset"] = offset
                    z[f"{ds_keys}/{i}"].attrs["resolution"] = resolution

                print(f"saved slices {i} in {out_path, os.path.basename(k)}")
            else:
                # 3D
                z[f"{ds_keys}"] = hm_sx[k]
                z[f"{ds_keys}"].attrs["offset"] = offset
                z[f"{ds_keys}"].attrs["resolution"] = resolution
                print(f"saved {k} in {out_path, os.path.basename(k)}")


def resample_3d(data_path: Union[Path, str], datasets: List[str], dimensionality: List[str],
                target_voxel_size: Tuple[int] = (8, 8, 8),
                ds_keys="volumes/raw", interp_order=1, cfg=None):
    """
    Only implemented for 3D for now!!
    Remember `resampling` reduces the size/shape of the input volume.Example: a `512^3` array with `8^3` resolution when
    resampled to a target resolution of `16^3` will become `256^3` in shape. So adjust the patch size before training
    LSD networks.

    **Why we implement this:**
    a) easier to train on downsampled images, occupy less GPU ram
    b) explore the coarse to fine robustness/generalisability of networks when following curriculum learning paradigm over total epochs

    interp_order (``int``, optional):

    The order of interpolation. The order has to be in the range 0-5:
    0: Nearest-neighbor
    1: Bi-linear (default)
    2: Bi-quadratic
    3: Bi-cubic
    4: Bi-quartic
    5: Bi-quintic

  """

    target_voxel_size_str = '_'.join(map(str, target_voxel_size))
    # empty dict to store resampled data
    hm_sx = {}
    for brain_vol in datasets:
        # create the output dir
        for dim in dimensionality:
            if dim == "data_3d":
                out_path = os.path.join(data_path, "preprocessed_3d", brain_vol + "_s_vsize_" + target_voxel_size_str)
            else:
                out_path = os.path.join(data_path, "preprocessed", brain_vol + "_s_vsize_" + target_voxel_size_str)

            create_dir(out_path, overwrite=False)

            source_data_ndlist = get_zarr_list(os.path.join(data_path, brain_vol, dim, "train"), ds_keys=ds_keys)
            for k in source_data_ndlist.keys():
                # k == name of file.zarr
                source_data = source_data_ndlist[k]
                source_data_zarr = read_zarr(os.path.join(data_path, brain_vol, dim, "train", k))[ds_keys]
                # assert "resolution" in source_data_zarr.attrs[
                #     "resolution"], "Resolution must be set in the dataset."
                source_voxel_size = source_data_zarr.attrs["resolution"]
                source_voxel_size_dims = len(source_voxel_size)
                scales = np.array(source_voxel_size) / np.array(target_voxel_size)
                scales = np.array(
                    (1,) * (len(source_data.shape) - source_voxel_size_dims) + tuple(scales))
                # Anti-aliasing must be false here otherwise labels will be pixelated!
                resampled_data = rescale(source_data.astype(np.float32), scales, order=interp_order,
                                         anti_aliasing=False).astype(source_data.dtype)
                # resized_data = resize(resampled_data, output_shape=source_data.shape, order=interp_order,
                #                       anti_aliasing=True)
                hm_sx[k] = resampled_data
            save_zarr(out_path=out_path, hm_sx=hm_sx, offset=(0, 0, 0), resolution=target_voxel_size,
                      ds_keys=ds_keys, is_2d=False)


if __name__ == '__main__':
    # Todo: expand to ingest all keys at once!
    resample_3d("/media/samia/DATA/ark/connexion/data", datasets=["HEMI"], dimensionality=['data_3d'],
                target_voxel_size=(12, 12, 30), interp_order=0, ds_keys="volumes/raw")
    resample_3d("/media/samia/DATA/ark/connexion/data", datasets=["HEMI"], dimensionality=['data_3d'],
                target_voxel_size=(12, 12, 30), interp_order=0, ds_keys="volumes/labels/neuron_ids")
    resample_3d("/media/samia/DATA/ark/connexion/data", datasets=["HEMI"], dimensionality=['data_3d'],
                target_voxel_size=(12, 12, 30), interp_order=0, ds_keys="volumes/labels/labels_mask")
