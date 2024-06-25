"""
Clahe volumes in 2D/3D modes with `skimage.exposure.equalize_adapthist`.
Adapted from: https://github.com/google-research/connectomics/blob/main/connectomics/volume/processor/contrast.py

Todo:
 - Expand this to process all files (zarrs/tiff) in a folder.
 - Allow clahe on large volumes via daisy
Author: Samia Mohinta
Affiliation: Cardona lab, Cambridge University, UK
"""

import argparse
import os

import dask.array
from dask import delayed
from dask.distributed import Client, LocalCluster
import skimage
import numpy as np
import tifffile
import zarr
import multiprocessing as mp
import matplotlib.pyplot as plt
from sys import stdout
import sys
from pathlib import Path
from tqdm.dask import TqdmCallback


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from data_utils.preprocess_volumes.utils import read_zarr, list_keys, collect_items, natural_keys


class CLAHE:
    """Applies CLAHE plane-wise."""

    crop_at_borders = False

    def __init__(
            self,
            kernel_size=None,
            clip_limit=0.01,
            clip_min=None,
            clip_max=None,
            invert=False
    ):
        """Constructor.

        Args:
          kernel_size: Forwarded to equalize_adapthist.
          clip_limit: Forwarded to equalize_adapthist.
          clip_min: Minimum value to retain in the input to CLAHE.
          clip_max: Maximum value to retain in the input to CLAHE.
          invert: Whether to invert the CLAHE result.
        """
        super(CLAHE, self).__init__()
        self._kernel_size = kernel_size
        self._clip_limit = clip_limit
        self._invert = invert
        self._clip_max = clip_max
        self._clip_min = clip_min

    def output_type(self, input_type):
        return np.uint8

    def process_plane(self, image2d: np.ndarray) -> np.ndarray:
        """
        Google uses a plane processor to apply clahe plane-by-plane (yx).
        However, for now we pass the whole 3D array to adapt hist, with the downside of loading the entire
        image into memory.

        """
        if len(set(np.unique(image2d))) == 1:
            # Image is all dark/white check??
            return image2d

        if self._clip_min is not None or self._clip_max is not None:
            c_min = self._clip_min if self._clip_min is not None else -np.inf
            c_max = self._clip_max if self._clip_max is not None else np.inf
            image2d = np.clip(image2d, c_min, c_max)

        clahed = skimage.exposure.equalize_adapthist(
            image2d, kernel_size=self._kernel_size, clip_limit=self._clip_limit
        )
        if self._invert:
            clahed = 1.0 - clahed
        return (clahed * 255).astype(np.uint8)


def apply_clahe_to_slice(args):
    """Wrapper function for multiprocessing."""
    slice_idx, slice_data, clahe_params = args
    apply_clahe = CLAHE(**clahe_params)
    return slice_idx, apply_clahe.process_plane(slice_data)


def apply_clahe_to_chunk(chunk, clahe_params):
    """
    Wrapper function to apply CLAHE to a single chunk

    :param chunk: ndarray
    :param clahe_params: parameters for equalize_adapt_hist
    :return: clahed chunk
    """
    apply_clahe = CLAHE(**clahe_params)
    return apply_clahe.process_plane(chunk)


def copy_datasets_from_multiple_sources(in_zarr, out_zarr, datasets_to_copy=None, is_2d=False):
    """
    Overriding `copy_datasets_from_multiple_sources` under histogram match because this one must behave differently.
    Copy specified datasets from multiple source Zarr stores to the destination Zarr store.

    Args:

    Returns:
        None
    """
    # cast a None argument passed to empty list; it's just easier to pass a None through the `config.py`
    datasets_to_copy = [] if datasets_to_copy is None else datasets_to_copy
    source_z = read_zarr(in_zarr)
    source_z_keys = list_keys(source_z)
    with zarr.open(out_zarr, mode="a") as dest_z:

        for dataset_name in datasets_to_copy:
            # Copy the dataset from source to destination as a whole
            if is_2d:
                zarr.copy(source_z[dataset_name], dest_z[dataset_name.split('/')[0]], log=stdout,
                          if_exists='replace')
            else:
                zarr.copy(source_z[dataset_name], dest_z, name=dataset_name,
                          # [f"/{'/'.join(dataset_name.split('/')[:-1])}"],
                          log=stdout, if_exists='replace', dry_run=False)

    print("Datasets copied successfully")


if __name__ == '__main__':
    # read a zarr
    parser = argparse.ArgumentParser("This script applies CLAHE on 3D/2D RAW EM. Credits-Google Connectomics"
                                     "2D is supported through multiprocessing (smaller) and dask (larger volumes). "
                                     "Dask should be used if the volume will not fit in RAM. "
                                     "3D should be used only when the volume can be fit into memory. "
                                     "3D may render `more uniform` histogram intensities."
                                     )
    parser.add_argument('-f', help='Zarr/tiff to apply clahe on')
    parser.add_argument('-of', default=None, help='Zarr/tiff to save clahed EM')
    parser.add_argument('-ds', default='volumes/raw', help='Dataset inside Zarr.')
    parser.add_argument('-ds_copy', nargs='+',
                        default=None,
                        help='Datasets to copy from source zarr to out zarr. These are either labels/mask.')
    parser.add_argument('-k', nargs='+', default="30 300 300",
                        help='Kernel size for CLAHE. Default: (30,300,300) for 3D. Input format: 30 300 300'
                             ' Larger kernel windows within the shape of '
                             '2D images are better for 2D processing. Please adjust kernels if using dask'
                             '(recommended 20 200 200).')
    parser.add_argument('-cl', default=0.01, help='Clip limit')
    parser.add_argument('-cmin', default=None, help='Clip minimum')
    parser.add_argument('-cmax', default=None, help='Clip maximum')
    parser.add_argument('-inv', default=False, help='Invert the clahed image.')
    parser.add_argument('-mp', default=4, help='Use multiprocessing to apply clahe plane wise.'
                                               'Options: -1=to use 4 dask workers'
                                               '>0=to use python threadpool {num} workers.'
                                               'For larger volumes we recommend dask because it will use chunks of'
                                               '(1,256,256).'
                                               'Default: 4 workers.')
    parser.add_argument('--show_hist', default=False,
                        help='Display histogram of intensities of clahed and original volumes')

    args = parser.parse_args()

    if int(args.mp) > 0 and len(args.k) > 2:
        kernel_size = tuple(args.k)[-2:]  # take only last 2dims
    else:
        kernel_size = tuple(args.k)

    clahe_params = {
        'kernel_size': kernel_size,
        'clip_limit': args.cl,
        'clip_min': args.cmin,
        'clip_max': args.cmax,
        'invert': args.inv,
    }

    # make a clahe object
    apply_clahe = CLAHE(**clahe_params)
    # read the tiff from SNEMI3D
    if args.f.endswith(('.tif', '.tiff')):
        flag_file_type = 'tif'
        file_size = os.path.getsize(args.f) / (1024 * 1024)
        print(f"File size tiff: {file_size}MB")
        if file_size > 500:
            file_ = tifffile.imread(args.f, aszarr=True)  # this will probably be a zarr array without a dataset
        else:
            file_ = tifffile.imread(args.f)
    elif args.f.endswith('.zarr'):  # todo: expand to n5
        flag_file_type = 'zarr'
        zfile_ = zarr.open(args.f, mode='r')
        # remember to provide the correct dataset, no checks are performed
        file_ = zfile_[args.ds]
    else:
        raise Exception("File type not supported. Only Zarrs/tiffs allowed!")

    if int(args.mp) > 0:

        # Prepare data for multiprocessing
        # TODO: expand to incorporate any plane
        slices_to_process = [(z, file_[z, :, :], clahe_params) for z in range(file_.shape[0])]

        # Set up multiprocessing pool
        with mp.Pool(int(args.mp)) as pool:
            results = pool.map(apply_clahe_to_slice, slices_to_process)

        # Sort results and stack them into a 3D array
        results.sort(key=lambda x: x[0])  # Ensure results are sorted by slice index
        clahed_image = np.stack([result[1] for result in results], axis=0)

    elif int(args.mp) == -1:  # use dask

        cluster = LocalCluster(n_workers=4)
        client = Client(cluster)
        chunk_size = (1, 256, 256)  # default small

        # Define a Dask array to apply CLAHE chunk-wise
        data = dask.array.from_zarr(args.f, component=args.ds, chunks=chunk_size)
        # chunked clahe
        processed_chunks = data.map_blocks(apply_clahe_to_chunk, clahe_params=clahe_params, dtype=data.dtype)
        # Compute the result
        # Compute the result with progress bar
        with TqdmCallback(desc="Processing slices"):
            clahed_image = processed_chunks.compute()

    else:
        clahed_image = apply_clahe.process_plane(file_)  # this will be a 3D ndarray

    if args.of is None:
        outfile_name = f"{os.path.splitext(args.f)[0]}_clahed"
        if flag_file_type == 'tif':
            outfile_name += ".tiff"
        elif flag_file_type == 'zarr':
            outfile_name += ".zarr"
    else:
        outfile_name = args.of

    # Process TIFF files
    if flag_file_type == 'tif':
        tifffile.imwrite(outfile_name, clahed_image)

    # Process Zarr files
    elif flag_file_type == 'zarr':
        outfile = zarr.open(outfile_name, "a")
        outfile[args.ds] = clahed_image
        # Copy attributes - fix
        for attr in ['resolution', 'offset']:
            outfile[args.ds].attrs[attr] = zfile_[args.ds].attrs[attr]

        # copy labels across to the clahed zarr now??
        copy_datasets_from_multiple_sources(in_zarr=args.f, out_zarr=outfile_name,
                                            datasets_to_copy=tuple(args.ds_copy) if args.ds_copy is not None else None,
                                            is_2d=False)

    if args.show_hist:
        # Histogram plots are true by default
        fig, axs = plt.subplots(1, 2)
        axs[0].hist(file_[...].ravel())  # ellipsis just in case it is a zarr.core.Array()
        axs[0].set_title("original hist")
        axs[1].hist(clahed_image.ravel())
        axs[1].set_title("clahed hist")

        plt.show()
