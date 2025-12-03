import numpy as np
import os.path
import shutil

import tifffile
import imagecodecs
import dask.array
import dask.dataframe
from typing import Union
from pathlib import Path
from typing import List
from skimage.util import invert
import zarr
import skimage.io as io
from glob import glob

import re


def extract_number(filename):
    # Extract the numerical part from the filename
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return None


def png_to_tifs(lists, substr=''):
    temp_tif_path = os.path.join(os.path.dirname(lists[0]), 'temp_tif')
    if not os.path.exists(temp_tif_path):
        os.makedirs(temp_tif_path, exist_ok=True)
    for f in lists:
        read_f = io.imread(f, as_gray=True)
        # this is an extra check for hemibrain labels since the background is
        # represented as np.uint64(-3) `a very large number`
        if np.any(read_f == np.uint64(-3)):
            background_mask = read_f == np.uint64(-3)
            read_f[background_mask] = 0
        # since values range between 0 and 1 after reading as gray, so explicit cast to 0-255 uint8
        read_f = (read_f * 255).astype(np.uint8)
        tifffile.imwrite(os.path.join(temp_tif_path, os.path.basename(f).split('.')[0] + '.tif'), read_f)

    return sorted(glob(f"{temp_tif_path}/*.tif"),
                  key=lambda x: (extract_number(x.split('/')[-1]), extract_number('/'.join(x.split('/')[:-1]))))


def create_labels_mask(labels):
    labels_mask = np.ones_like(labels).astype(np.uint64)
    # assuming background is represented with 0
    if np.any(labels == np.uint64(-3)):
        background_mask = labels == np.uint64(-3)
    else:
        background_mask = labels == 0

    labels_mask[background_mask] = 0

    return labels_mask.astype(np.uint8)


def train_test_splitter(train_test_split, da, flag='train'):
    # TODO: badly implemented, must automatically split in train/val/test based on the float value provided
    if train_test_split is not None:
        # preferring train set to have more data
        slice_z = np.ceil(da.shape[0] * train_test_split)
        if flag == 'train':
            #  train
            da = da[:slice_z, ...]
        elif flag == 'val':
            # val
            da = da[slice_z:, ...]
    return da


def tiffs2zarr(filenames: Union[List[str], Path], zarrurl: str, chunksize: int, datapath: str, gen_hdf: bool = False,
               train_test_split: Union[float, None] = None, flag_split: Union[str, None] = 'train',
               plant_seg: bool = True, resolution: tuple = (8, 8, 8), offset: tuple = (0, 0, 0), substr='fake_B'):
    """Write images from sequence of TIFF files as zarr."""

    def imread(filename):
        # return first image in TIFF file as numpy array
        with open(filename, 'rb') as fh:
            data = fh.read()
        return imagecodecs.tiff_decode(data)

    if gen_hdf:
        if os.path.exists(zarrurl):
            print("The file exists so removing it!")
            shutil.rmtree(zarrurl)

    if filenames[0].endswith('.png'):
        # as of now we keep the temp_tif folder inside the png folder, we can programmatically delete it at some point
        filenames = png_to_tifs(filenames, substr)
    with tifffile.FileSequence(imread, filenames) as tifs:
        print(tifs),
        with tifs.aszarr() as store:
            da = dask.array.from_zarr(store)
            # print(da[0, ...].compute())
            if flag_split is not None and train_test_split is not None:
                da = train_test_splitter(train_test_split, da, flag='train')
            chunks = (chunksize,) + da.shape[1:]
            if plant_seg:
                if 'aff' in datapath:
                    da = invert(da)

            if 'label' in datapath:
                # assume: unique labels won't be a number larger than 2^16 = 65535 as of now
                # da = da.astype(np.uint64)
                da = da.astype(np.uint16) if np.max(da) > 255 else da.astype(np.uint8)
            else:
                # by default all arrays will be np.uint8
                da = da.astype(np.uint8)

            da.rechunk().to_zarr(zarrurl, datapath, overwrite=True)

            if 'label' in datapath:
                labels_mask = create_labels_mask(da)
                # assume datapath passed is `volumes/labels`
                # will transform to `volumes/labels/labels_mask`
                datapath_mask = f"{os.path.dirname(datapath)}/labels_mask"
                labels_mask.rechunk(chunks).to_zarr(zarrurl, datapath_mask, overwrite=True)

    if gen_hdf:
        # in-elegant way of saving both mask and label
        # TODO- tiffs2hdf() should pick all datasets from the zarr and save it to hdf datasets recursively
        tiffs2hdf(zarrurl, os.path.join(os.path.dirname(zarrurl), os.path.basename(zarrurl).split('.')[0] + '.hdf'),
                  chunksize, datapath)
        if 'label' in datapath:
            tiffs2hdf(zarrurl, os.path.join(os.path.dirname(zarrurl), os.path.basename(zarrurl).split('.')[0] + '.hdf'),
                      chunksize, datapath_mask)
        try:
            shutil.rmtree(zarrurl)
        except Exception as e:
            print(e)
    else:
        # setting resolution and offsets with default FIBSEM Octo values in case
        # you only save the zarr and not generate the hdf
        z = zarr.open(zarrurl, mode='a')
        z[datapath].attrs["resolution"] = resolution
        z[datapath].attrs["offset"] = offset
        if 'label' in datapath:
            z[datapath_mask].attrs["resolution"] = resolution
            z[datapath_mask].attrs["offset"] = offset


def tiffs2hdf(zarrurl, hdfurl, chunksize, datapath):
    """Write images from sequence of TIFF files as hdf."""
    # this would give a shape = (num of tiffs, y, x)
    da = dask.array.from_zarr(zarrurl, datapath)
    chunks = (chunksize,) + da.shape[1:]
    da.rechunk(chunks).to_hdf5(hdfurl, datapath)


# TODO- rewrite this to be able to deal with all datasets in hdf and auto-feed to napari
def sanity_check_hdf():
    import h5py
    import napari
    f = h5py.File(
        '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/temp_train_.hdf')
    v = napari.view_image(f['volumes/raw'])
    v.add_labels(f['volumes/labels/neuron_ids'])
    v.add_image(f['volumes/gt_aff'])


# does not matter how ypu name your datasets, you should able to change them in the config file
# raw
# tiffs2zarr(
#     sorted(glob(
#         '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/tifs/trainA/*.tif'),
#            key=lambda x: (extract_number(x.split('/')[-1]), extract_number('/'.join(x.split('/')[:-1])))),
#     '/media/samia/DATA/PhD/codebases/imagetranslation/SynDiff/EM_sample_data/data_val_hemi.zarr', 500,
#     datapath='volumes/raw', gen_hdf=False, train_test_split=0.2, flag_split='val')

# labels
# tiffs2zarr(
#     sorted(glob(
#         '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/tifs/trainA_labels/*.tif')),
#     '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/trainA_val.zarr', 500,
#     datapath='volumes/labels/neuron_ids', gen_hdf=True, train_test_split=0.8, flag_split='val')
# #
# # # gt_affinities
# tiffs2zarr(
#     sorted(glob('/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/trainA_affs/*.tif')),
#     '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/temp_test.zarr', 500,
#     datapath='volumes/gt_aff', gen_hdf=True, train_test_split=0.2, flag_split='val')

# Octo trainB zarr file to be tested with the trained LongRange MTLSD model
# tiffs2zarr(
#     sorted(
#         glob('/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/tifs/trainB/*tif'),
#         key=lambda x: (extract_number(x.split('/')[-1]), extract_number('/'.join(x.split('/')[:-1])))),
#     '/media/samia/DATA/PhD/codebases/imagetranslation/SynDiff/EM_sample_data/data_val_octo.zarr', 500,
#     datapath='volumes/raw', gen_hdf=False, train_test_split=0.2, flag_split='val')

# Al crop of octo - this is not what the gan was trained on, that data is inside trainB
# we are generating this dataset to check if the LR MTLSD model trained on Hemibrain does well.
# tiffs2zarr(
#     sorted(glob('/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/tifs/testALC/*.tif')),
#     '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-octo/dataset/testALC.zarr', 500,
#     datapath='volumes/raw', gen_hdf=True, train_test_split=1, flag_split='train')

# tremont
# tiffs2zarr(
#     sorted(glob('/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/trainB/*tif')),
#     '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/trainB.zarr', 500,
#     datapath='volumes/raw', gen_hdf=True, train_test_split=1, flag_split='train')
#

# hemi to tremont fake - hemi now looks like tremont,
# passing pngs without the sorted key is still fine, since there is one sorting after tif generation
# tiffs2zarr(
#     sorted(glob(
#         '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2tremont_with_pngs_upsample_crop256/test_pb-groundtruth-with-context-x8472-y2892-z9372/images/*fake_B.png')),
#     '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2tremont_with_pngs_upsample_crop256/HemiLikeTremont_pb-groundtruth-with-context-x8472-y2892-z9372.zarr',
#     500,
#     datapath='volumes/raw', gen_hdf=False, train_test_split=1, flag_split='train')
#
# # hemi original labels
# tiffs2zarr(
#     sorted(glob(
#         '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/pngs/misc/pb-groundtruth-with-context-x8472-y2892-z9372/neuron_ids/*.png')),
#     '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2tremont_with_pngs_upsample_crop256/HemiLikeTremont_pb-groundtruth-with-context-x8472-y2892-z9372.zarr',
#     500,
#     datapath='volumes/labels/neuron_ids', gen_hdf=False, train_test_split=1, flag_split='train')

# hemi 256
# tiffs2zarr(
#     sorted(glob(
#         '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/pngs/misc/lh-groundtruth-with-context-x7737-y20781-z12444/raw/*.png'),
#         key=lambda x: (extract_number(x.split('/')[-1]), extract_number('/'.join(x.split('/')[:-1])))),
#     '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/pngs/misc/lh-groundtruth-with-context-x7737-y20781-z12444.zarr',
#     500,
#     datapath='volumes/raw', gen_hdf=False, train_test_split=1, flag_split='train')
#
# tiffs2zarr(
#     sorted(glob(
#         '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/pngs/misc/lh-groundtruth-with-context-x7737-y20781-z12444/neuron_ids/*.png'),
#         key=lambda x: (extract_number(x.split('/')[-1]), extract_number('/'.join(x.split('/')[:-1])))),
#     '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-tremont/dataset/pngs/misc/lh-groundtruth-with-context-x7737-y20781-z12444.zarr',
#     500,
#     datapath='volumes/labels/neuron_ids', gen_hdf=False, train_test_split=1, flag_split='train')


# hemi to seymour
# tiffs2zarr(
#     sorted(glob(
#         '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/pytorch-CycleGAN-and-pix2pix/results/hemi2seymour_with_pngs_upsample_crop256/test_latest/images/*_fake_B.png'),
#         key=lambda x: (extract_number(x.split('/')[-1]), extract_number('/'.join(x.split('/')[:-1])))),
#     '/media/samia/DATA/cyclegan/pytorch-CycleGAN-pix2pix-2D/experiment-hemi-to-seymour/dataset/pngs/misc/lh-groundtruth-with-context-x7737-y20781-z12444.zarr',
#     500,
#     datapath='volumes/labels/neuron_ids', gen_hdf=False, train_test_split=1, flag_split='train')
